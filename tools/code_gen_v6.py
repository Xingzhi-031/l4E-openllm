import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re

import datasets
import pandas as pd
import concurrent.futures
from prompt.prompt import CodegenPrompt3, CodegenPrompt1
from gpt.gpt_reply import GPTReply
from case_generationGpt import CaseGenerator
from tqdm import tqdm  # 用于显示进度条

import threading  # 如果保留手动线程控制，也需导入
from gpt.multi_thinking import MultiThinking
from gpt.slow_thinking import slow_thinking
from concurrent.futures import ThreadPoolExecutor, as_completed, FIRST_COMPLETED, wait

from sanitize.sanitize import sanitize
from tools.data import get_evalperf_data, get_human_eval_plus, get_mbpp_plus
# 项目根目录（适配 Windows 路径）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CodeGenerator:
    def __init__(self,model):
        self.failed_tasks = {}  # 初始化为字典
        self.lock = threading.Lock()  # 添加锁
        enamel_path = os.path.join(BASE_DIR, "dataset", "enamel.csv")
        self.casegentor = CaseGenerator(problems=enamel_path)
        self.results_lock = threading.Lock()
        self.results = {}
        self.multithinking = MultiThinking(model)
        self.slowthinking = slow_thinking(model)

    def code_evaluate_unbcase(self, code, num, casset):
        try:
            passed, code_effs = self.casegentor.evaluate_diffset(code, num, casset)
            return {"passed": passed, "error case": code_effs}
        except Exception as e:
            return {"passed": False, "error case": "Failed to compute references after multiple attempts."}

    def code_evaluate_resources(self, code, num,case,compare_obj):
        try:
            case_state,code_execution_result = self.casegentor.evaluate_resources_process(num, code,case,compare_obj)
            return case_state,code_execution_result
        except Exception as e:
            return {"passed": False, "error case": "Failed to compute references after multiple attempts."}
    # evaluate_groundtruth

    def code_evaluate_groundtruth(self, code, case,entry_point,num):
        try:
            result,reason = self.casegentor.evaluate_groundtruth(code,case,entry_point,num)
            return result,reason
        except Exception as e:
            return {"passed": False, "error case": "Failed to compute references after multiple attempts."}

    def code_case_result_check(self,code_dict):
        """
        由于code的结果是降序排序，所以针对通过数最多的代码进行校验，
        检查是否存在有的测试用例未通过的情况，如果有则返回Flag=True，用于后续的用例评估和代码迭代
        :param code_dict:
        :return:
        """
        flag = True
        first_key = list(code_dict.keys())[0]
        for key,value in code_dict[first_key].items():
            try:
                if not value['pass_result']:
                    flag =True
                    break
                else:
                    flag =False
            except:
                pass
        return flag

    def code_combine(self,code_dict1,code_dict2):
        """
        合并的逻辑是，针对各个算法所转化成的代码进行分别进行比较，例如
        第一种算法所转化成的代码和修正后的代码进行正确率的比较，如果修正后的代码的正确率大于或等于第一种，则使用修正后的代码
        为什么是大于或者等于？
        如果只是等于，那么可能会出现代码修改后效率下降的情况（因为这里仅做正确性的矫正）
        :param code_dict1:
        :param code_dict2:
        :return:
        """
        code_combine_result = {}
        for code_key in code_dict1.keys():
            if code_dict1[code_key]['result']['pass_result'] >=code_dict2[code_key]['result']['pass_result']:
                code_combine_result[code_key] = code_dict1[code_key]
            else:
                code_combine_result[code_key] = code_dict2[code_key]
        return code_combine_result

    def case_update(self,itercase_candidate):
        """
        更新测试用例，具体包括，删掉timeout、大模型判断结果为False的测试用例
        :param itercase_candidate:
        :return:
        """
        itercase_result = []
        for key, value in itercase_candidate.items():
            try:
                if isinstance(value['time'], str) and "timeout" in value['time']:
                    itercase_result.append("")
                elif itercase_candidate[key]['correct_flag']:
                    itercase_result.append(value['case_value'])
                else:
                    itercase_result.append("")
            except:
                itercase_result.append(value['case_value'])

        return itercase_result

    def update_iter_case(self,case_execution_status,correct_ficase):
        tmp_recode = {}
        for key,value in correct_ficase.items():
            try:
                if case_execution_status[key]["failed_reuslt"] == 0:
                    tmp_recode[key] = True
                else:
                    tmp_recode[key] = False
            except Exception as e:
                print(f"update_iter_case",e)
        return tmp_recode


    def code_filter(self,Gptreply, code_dict,first_key):
        """
        首先比较通过率，如果通过率相等加入候选队列，然后将整个候选队列发送给大模型判断
        :param code_dict: Dictionary containing code data
        :return: Selected code string
        """
        final_code_key = first_key
        wating_list = []

        for key, value in code_dict.items():
            if code_dict[key]['result']['pass_result'] > code_dict[final_code_key]['result']['pass_result']:
                final_code_key = key
            elif code_dict[key]['result']['pass_result'] == code_dict[final_code_key]['result']['pass_result']:
                wating_list.append(key)

        if final_code_key not in wating_list:
            wating_list.append(final_code_key)

        # Step 2: If there are ties, select the one with the shortest total_time
        # if len(wating_list) > 1:
        #     tmp_key = wating_list[0]
        #     for key in wating_list[1:]:
        #         if code_dict[key]['result']['total_time'] < code_dict[tmp_key]['result']['total_time']:
        #             tmp_key = key
        # else:
        #     tmp_key = wating_list[0]

        compare_code = []
        if len(wating_list) ==1:
            try:
                return "","",code_dict[wating_list[0]]['new_code']
            except:
                return "","",code_dict[wating_list[0]]['code']
        for fast_key in wating_list:
            try:
                compare_code.append({fast_key:code_dict[fast_key]['new_code']})
            except:
                compare_code.append({fast_key:code_dict[fast_key]['code']})

        while True:

            try:
                fast_code_reply = Gptreply.getreply(CodegenPrompt3.fask_code_choice_system,
                                             json.dumps(compare_code),"")
                code_key_pattern = re.compile(r"```text\n(.*?)```", re.DOTALL)
                fast_code = re.findall(code_key_pattern, fast_code_reply)[0]
                fast_code_key = fast_code.strip()
                fast_code = [d[fast_code_key] for d in compare_code if fast_code_key in d][0]
                break
            except Exception as e:
                pass

        return compare_code,fast_code_reply,fast_code
        # Step 3: Return the best code
        # try:
        #     return code_dict[tmp_key]['new_code']
        # except Exception as e:
        #     # print("code_filter",e)
        #     return code_dict[tmp_key]['code']

    def case_summaries_result(self,GPTReply,correct_case_item,case_item,task_description,code):
        if len(case_item) !=0:
            case_failed_reason = self.case_summaries(GPTReply,correct_case_item,task_description,code)
            return case_failed_reason,case_item
        else:
            return None
    def case_check_iterate(self,GPTReply,case_item,task_description,code):
        """
        使用大模型校验测试用例是否为正确
        :param GPTReply:
        :param case_item:
        :param task_description:
        :param code:
        :return:
        """

        def process_case(key, value, case_item,  task_description):
            """处理单个case的逻辑"""
            if value['failed_reuslt'] != 0:
                if isinstance(value['time'], str) and "timeout" not in value['time']:
                    check_result = GPTReply.getreply(
                        CodegenPrompt3.case_check_agent_system,
                        CodegenPrompt3.case_check_agent_user.format(task_description,value['case_value']),
                        ""
                    )
                    if "NO" in check_result.upper():
                        case_item[key]['correct_flag'] = True
                    else:
                        case_item[key]['correct_flag'] = False

        def process_cases_multithreaded(case_item,  task_description):
            """多线程处理case_item中的所有任务"""
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(process_case, key, value, case_item, task_description)
                    for key, value in case_item.items()
                    if value['failed_reuslt'] != 0 and isinstance(value['time'], str) and "timeout" not in value['time']
                ]
                # 可选：等待所有线程完成
                for future in as_completed(futures):
                    future.result()  # 获取线程执行结果（这里不会返回值）

        # 调用多线程处理
        process_cases_multithreaded(case_item,  task_description)

        correct_case_item ={}
        tmp_judge_dict = {}
        for key, value in case_item.items():
            try:
                if case_item[key]['correct_flag']:
                    tmp_judge_dict[key]=False
                    correct_case_item[key] = value
            except:
                pass

        return tmp_judge_dict,correct_case_item,case_item

    def case_summaries(self,GPTReply,case_dict,task_description,code):
        result = GPTReply.getreply(CodegenPrompt3.case_summarize_system,CodegenPrompt3.case_summarize_user.format(task_description,case_dict),
                                   "")
        return  result



    def algorithm_generation_single(self, Gptreply, task_description):
        def load_algorithm(result,num):
            """
            Extract the algorithm from the result using regular expressions.
            """
            try:
                code_regexp_pattern = re.compile(rf"```algorithm{num}\n(.*?)```", re.DOTALL)
                matches = re.findall(code_regexp_pattern, result)
                return matches[0] if matches else ""
            except Exception as e:
                raise RuntimeError(f"Error in load_algorithm: {str(e)}")

        def process_time_complexity(time_complexity):
            """
            Process each time complexity level in a separate thread.
            """
            while True:
                try:
                    # Generate the reply from Gptreply
                    algorithm_tmp_candidate = Gptreply.getreply(
                        CodegenPrompt1.algorithim_generation,
                        task_description,
                        ""
                    )

                    # Extract algorithm
                    return algorithm_tmp_candidate
                except Exception as e:
                    # Handle retries
                    print(e)
                    continue

        # time_complexity_levels = ['O(1)', 'O(logn)', 'O(n*m)', 'O(n)', 'O(nlogn)']
        time_complexity_levels = "as small as possible"
        algorithm_candidate ={}
        while True:
            try:
                algorithm_tmp_candidate = process_time_complexity(str(time_complexity_levels))
                for i in range(1,6):

                    load_result_tmp = load_algorithm(algorithm_tmp_candidate, i)

                    if load_result_tmp:
                        # Return the valid result
                        algorithm_candidate[str(i)] = task_description + "\n" + load_result_tmp

                break
            except:
                pass

        return algorithm_candidate

    import re
    import concurrent.futures

    def algorithm_generation(self, Gptreply, task_description):
        def load_algorithm(result, num):
            """
            Extract the algorithm from the result using regular expressions.
            """
            try:
                code_regexp_pattern = re.compile(rf"```algorithm{num}\n(.*?)```", re.DOTALL)
                matches = re.findall(code_regexp_pattern, result)
                return matches[0] if matches else ""
            except Exception as e:
                raise RuntimeError(f"Error in load_algorithm: {str(e)}")

        def process_time_complexity():
            """
            Generate the algorithm candidate for a given time complexity level.
            """
            algorithm_candidate = {}
            flag =1
            while True:
                try:
                    # Generate the reply from Gptreply
                    algorithm_tmp_candidate = Gptreply.getreply(
                        CodegenPrompt1.algorithim_generation,
                        task_description,
                        ""
                    )
                    for i in range(1, 6):
                        algo_content = load_algorithm(algorithm_tmp_candidate, i)
                        if algo_content!="":
                            algorithm_candidate[str(i)] = task_description + "\n" + algo_content
                        else:
                            flag=0
                    if flag==1:
                        return algorithm_candidate
                except Exception as e:
                    # Handle retries
                    print(f"Retrying due to error: {e}")
                    continue

        # Initialize the result dictionary

        # Use ThreadPoolExecutor to manage threads
        algorithm_candidate = process_time_complexity()
        return algorithm_candidate

    def code_generation_multi(self, Gptreply, task_description,entry_point):
        def load_algorithm(result, num):
            """
            Extract the algorithm from the result using regular expressions.
            """
            try:
                code_regexp_pattern = re.compile(rf"```python{num}\n(.*?)```", re.DOTALL)
                matches = re.findall(code_regexp_pattern, result)
                code_candidate = matches[0] if matches else ""

                sanitized_solution = sanitize(
                    code_candidate, entrypoint=entry_point
                )
                return sanitized_solution
            except Exception as e:
                raise RuntimeError(f"Error in load_algorithm: {str(e)}")

        def process_time_complexity():
            """
            Generate the algorithm candidate for a given time complexity level.
            """
            algorithm_candidate = {}
            flag =1
            while True:
                try:
                    # Generate the reply from Gptreply
                    code_tmp_candidate = Gptreply.getreply(
                        CodegenPrompt3.special_1_get_code_system,
                        task_description,
                        ""
                    )
                    for i in range(1, 6):
                        algo_content = load_algorithm(code_tmp_candidate, i)
                        if algo_content!="":
                            algorithm_candidate[str(i)] = algo_content
                        else:
                            flag=0
                    if flag==1:
                        return algorithm_candidate
                except Exception as e:
                    # Handle retries
                    print(f"Retrying due to error: {e}")
                    continue

        # Initialize the result dictionary

        # Use ThreadPoolExecutor to manage threads
        algorithm_candidate = process_time_complexity()
        return algorithm_candidate

    def generate_task_description(self, Gptreply, ques):
        task_description_gen = ""
        task_description_check = ""
        max_loop = 5
        for _ in range(max_loop):
            task_description_gen = Gptreply.getreply(CodegenPrompt3.task_description_gen_system,
                                                     CodegenPrompt3.task_description_gen_user.format(ques),
                                                     task_description_gen + task_description_check)
        # task_description_gen = self.multithinking.main_process(CodegenPrompt3.task_description_gen_system,
        #                                          CodegenPrompt3.task_description_gen_user.format(ques),
        #                                          task_description_gen + task_description_check)
            task_description_check = Gptreply.getreply(CodegenPrompt3.task_description_check_one_system,
                                                       CodegenPrompt3.task_description_check_one_user.format(ques,
                                                                                                             task_description_gen),
                                                       "")
            if "yes" in task_description_check.lower():
                break
        return task_description_gen, task_description_check

    def generate_initial_code(self, Gptreply, ques, max_retries=5):
        retries = 0
        while retries < max_retries:
            try:
                inicode = Gptreply.getreply(
                    "Please generate the code(DO not generate example,like print(xxxx)) and return the code format like ```python\n{code}```",
                    ques, ""
                )
                code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                matches = re.findall(code_regexp_pattern, inicode)
                if matches:
                    return matches[0]
            except Exception as e:
                print(f"Attempt {retries + 1} failed with error: {e}")
            retries += 1
        raise RuntimeError("Failed to generate initial code after multiple attempts.")

    # def optimize_algorithm(self, Gptreply, algorithim_description, ques):
    #     algorithim_description_opti = Gptreply.getreply(CodegenPrompt1.algorithim_generation_opti,
    #                                                     algorithim_description, "")
    #     loop_time = 0
    #     max_loop = 5
    #     logically_planning_iteropt_list = {}
    #     algorithim_description_opti_list = {}
    #     while loop_time <= max_loop:
    #         try:
    #             logically_planning_iteropt = Gptreply.getreply(CodegenPrompt3.logically_planning_iteropt_system,
    #                                                            CodegenPrompt3.logically_planning_iteropt_user.format(
    #                                                                algorithim_description_opti), "")
    #             logically_planning_iteropt_list[str(loop_time)] = logically_planning_iteropt
    #             algorithim_description_opti_list[str(loop_time)] = algorithim_description_opti
    #             try:
    #                 if "Yes" in logically_planning_iteropt or "yes" in logically_planning_iteropt:
    #                     break
    #             except Exception as e:
    #                 print(e)
    #                 code_regexp_pattern = re.compile("```json\n(.*?)```", re.DOTALL)
    #                 maches = re.findall(code_regexp_pattern, logically_planning_iteropt)
    #                 logically_planning_iteropt = maches[0]
    #                 if "Yes" in logically_planning_iteropt or "yes" in logically_planning_iteropt:
    #                     break
    #
    #             loop_time += 1
    #             algorithim_description_opti = Gptreply.getreply(CodegenPrompt1.algorithim_generation_opti,
    #                                                             algorithim_description_opti, logically_planning_iteropt)
    #
    #         except Exception as e:
    #             print(f"[+]logically_planning_opt_{ques}:", e)
    #             continue
    #     # print(f"[+]logically_planning_iteropt_{num}:ok\r\n")
    #     # loop_time = 0
    #     # max_loop = 5
    #
    #     algorithim_description_opti = Gptreply.getreply(CodegenPrompt3.algorithim_description_iterchose_system,
    #                                                     str(algorithim_description_opti_list),
    #                                                     "")
    #
    #     algorithim_description_opti_list[str(6)] = algorithim_description_opti
    #     return algorithim_description_opti_list

    def generate_code_from_package(self, Gptreply, package_all, num,task_description,entry_point,add_knowledge):
        while True:
            # CodegenPrompt3.COT_system
            try:
                algorithim_trans_code = Gptreply.getreply(CodegenPrompt3.algorithim_trans_code_system,
                                                          package_all, "This is a knowledge base:"+add_knowledge)
                # algorithim_trans_code = self.slowthinking.slow_code_thinking("generate correct and effective code based on the algorithm and return the code as the format :```python\n<code>```",
                #                                                              package_all)
                # sanitized_solution = sanitize(
                #     algorithim_trans_code, entrypoint=entry_point
                # )

                return algorithim_trans_code
            except Exception as e:
                print(f"[+]algorithim_trans_code_{num} error: {e}")

    def iterate_code_single(self, Gptreply, case_check, code_execution_result, task_description,code_type,input_case):
        """
        code_execution_result是按照通过数从大到小排的
        {
        "1":{
        "code":...}
        }
        case_check是检验为true的case

        :param Gptreply:
        :param case_check:
        :param code_execution_result:
        :return:
        """
        iter_code = {}
        def process_code(key, value, task_description, case_check,input_case):
            """
            Function to process a single code_execution_result entry.
            """
            while True:
                try:
                    # Call GPT reply method
                    code_iteration = Gptreply.getreply(
                        CodegenPrompt3.code_iteration_system,
                        CodegenPrompt3.code_iteration_user.format(task_description, value[code_type], case_check),
                        ""
                    )
                    # 如果用例过长，则只使用输入来进行迭代
                    if code_iteration==False:
                        case_check = f"The is the input case:{input_case}"
                    # Extract Python code from the response
                    tmp_code_afcase_0 = re.findall(r"```python\n(.*?)```", code_iteration, re.DOTALL)[0]
                    return key, tmp_code_afcase_0
                except:
                    pass  # Retry on failure

        iter_code = {}
        for key, value in code_execution_result.items():
            try:
                # Process each entry sequentially
                key, result = process_code(key, value, task_description, case_check, input_case)
                iter_code[key] = result
            except Exception as e:
                print(f"Error processing key {key}: {e}")
        return iter_code

    def iterate_code(self, Gptreply, case_check, code_execution_result, task_description, code_type, input_case,
                     entry_point):
        """
        code_execution_result是按照通过数从大到小排的
        {
        "1":{
        "code":...}
        }
        case_check是检验为true的case

        :param Gptreply:
        :param case_check:
        :param code_execution_result:
        :return:
        """

        def process_code(key, value, task_description, case_check, input_case):
            """
            Function to process a single code_execution_result entry.
            """
            flag = 0
            while True:
                try:
                    # Call GPT reply method
                    code_iteration = Gptreply.getreply(
                        CodegenPrompt3.code_iteration_system,
                        CodegenPrompt3.code_iteration_user.format(task_description, value[code_type], case_check),
                        ""
                    )
                    # 如果用例过长，则只使用输入来进行迭代
                    if code_iteration == False:
                        if flag == 1:
                            case_check = f"The is the input case:{input_case[0]}"
                        flag = 1
                        case_check = f"The is the input case:{input_case}"
                    # Extract Python code from the response

                    sanitized_solution = sanitize(
                        code_iteration, entrypoint=entry_point
                    )

                    # if benchmark_type == "Mercur":
                    #     max_retry = 3
                    #     _ = 0
                    #     flag = 0
                    #     while _ < max_retry:
                    #         _ += 1
                    #         judge_result = Gptreply.getreply(
                    #             "Please determine whether the following code is complete. If it is complete, please return Yes, otherwise please return No",
                    #             sanitized_solution, "")
                    #         if "YES" in judge_result.upper():
                    #             flag = 1
                    #             break
                    #         else:
                    #             formated_result = Gptreply.getreply(
                    #                 "Please extract and format the code and Out in the following format,Don't Output the example test case:```python\n{code}```",
                    #                 code_iteration, "")
                    #             sanitized_solution = sanitize(
                    #                 formated_result, entrypoint=entry_point
                    #             )
                    #     if flag == 0:
                    #         code_regexp_pattern = re.compile(rf"```python\n(.*?)```", re.DOTALL)
                    #         matches = re.findall(code_regexp_pattern, formated_result)
                    #         sanitized_solution = matches[0] if matches else ""

                    tmp_code_afcase_0 = sanitized_solution

                    return key, tmp_code_afcase_0
                except Exception as e:
                    print(f"Retrying key {key} due to error: {e}")
                    continue  # Retry on failure

        iter_code = {}

        # Use ThreadPoolExecutor to manage threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to the thread pool
            futures = {
                executor.submit(process_code, key, value, task_description, case_check, input_case): key
                for key, value in code_execution_result.items()
            }

            # Wait for all tasks to complete and process results
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                    key, result = future.result()
                    iter_code[key] = result
                except Exception as e:
                    print(f"Error processing key {key}: {e}")

        return iter_code

    def codegen_process4(self, model, ques, num, benchmark_type):
        """
        主程序
        :param model:
        :param ques:
        :param num:
        :return:
        """
        while True:
            try:
                def process_task(key, value, task_description):
                    ""
                    while True:
                        try:
                            # 从实践层面优化代码
                            additional_cost_knowledge = Gptreply.getreply(CodegenPrompt3.knowledge_databases_system,
                                                                          str(value), "")

                            tmp_code_result = str(
                                self.generate_code_from_package(Gptreply, value, num, task_description,
                                                                task["entry_point"], additional_cost_knowledge))

                            # additional_cost_analysis = Gptreply.getreply(CodegenPrompt3.additional_cost_analysis_system,
                            #                                                 str(tmp_code_result),"")

                            # opt_code_result = self.slowthinking.slow_opt_code_thinking("",tmp_code_result)
                            # opt_code_result = Gptreply.getreply(CodegenPrompt3.code_iter_practice_system,
                            #                                                 CodegenPrompt3.code_iter_practice_user.format(str(additional_cost_analysis),str(tmp_code_result)),"")
                            sanitized_solution = sanitize(
                                tmp_code_result, entrypoint=task["entry_point"]
                            )

                            # code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                            code_dict_for_resources[key] = sanitized_solution
                            additional_cost_knowledge_result[key] = additional_cost_knowledge
                            after_opt[key] = tmp_code_result
                            # opt_code_result_all[key] = opt_code_result
                            break
                        except Exception as e:
                            print("process_task", e)
                            pass

                new_case_candidate = []
                code_dict_for_resources = {}
                additional_cost_knowledge_result = {}
                # opt_code_result_all={}
                after_opt = {}
                Gptreply = GPTReply(model)
                slug_name = ""
                if benchmark_type == "Mercury":
                    num = ques["slug_name"]
                    slug_name = ques["id"]
                    task = ques
                    ques = task["prompt"] + "\n" + task["pretty_content"][0]
                else:
                    # task = dict(ques.values())
                    num = ques['task_id']
                    task = ques
                    ques = task["prompt"].strip() + "\n"

                # 检查 num 是否在特定范围
                if benchmark_type == "humaneval":
                    if num in [2, 23, 41, 45, 53, 60, 71, 92, 97, 99, 102, 123, 124, 135, 137, 138, 144, 148, 156, 157,
                               159, 160]:
                        return {
                            "code": "",
                            # "result": str(execution_result["passed"]),
                            "task_description_gen": "",
                            "task_description_check": "",
                            "algorithim_description_opti": "",
                            "debug_code_1": "",
                            "debug_code_2": "",
                            "final_code_dict": "",
                            "compare_code": "",
                            "fast_code_reply": ""
                        }

                # 生成任务描述
                task_description_gen, task_description_check = self.generate_task_description(Gptreply, ques)
                print(f"{num}", task_description_gen)

                # 生成用例候选
                case_candidate = self.casegentor.caseGenerator_testchain(model, ques + task_description_gen)
                print("case generation success", num)

                # 生成算法描述候选
                algorithm_description_candidate = self.algorithm_generation(Gptreply, ques + task_description_gen)

                # print("卡顿检查1",num)
                # 使用多线程生成代码
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(process_task, key, value, ques + task_description_gen)
                        for key, value in algorithm_description_candidate.items()
                    ]
                    for future in futures:
                        future.result()  # 等待所有线程完成

                # case_state是生成的测试用例在本次投票机制中通过的情况，其中用例的投票结果分为通过、不通过、超时不通过
                # code_execution_result是代码的通过情况，按照降序排序
                case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num,
                                                                                 case_candidate, 'code')
                first_key = list(code_execution_result.keys())[0]

                # 检查代码执行结果
                tmp_judge_dict = []
                if self.code_case_result_check(code_execution_result):
                    # tmp_judge_dict是判断为True且未通过的用例集合用于但他的值只有False和True，用于后续多轮迭代的用例状态实时更行
                    # correct_ficase是判断为True且未通过的用例集合，包括用例的值item，用于后续多轮迭代
                    # case_iteresult是所有的测试用例，并且前面执行为False的测试用例被加入了大模型判断为True和False的标志
                    tmp_judge_dict, correct_ficase, case_iteresult = self.case_check_iterate(Gptreply, case_state,
                                                                                             ques + task_description_gen,
                                                                                             code_execution_result[
                                                                                                 first_key])
                    new_case_candidate = self.case_update(case_iteresult)
                    # print("卡顿检查2", num)
                    # 基于new_case_candidate在评估一次
                    new_case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num,
                                                                                         new_case_candidate, 'code')
                else:
                    correct_ficase = {}
                itercode_execution_result = {}
                code_type = ""
                corret_flag = 0
                if len(correct_ficase) == 0:
                    # 如果correct_ficase的长度为0，意味着没有未通过且为正确的测试用例
                    final_code_list = code_execution_result
                else:
                    # 首先基于correct_ficase进行迭代测试
                    code_type = 'code'
                    try:
                        # 这里是做了一个判断，因为每次代码的更新迭代都会重新执行一次所有的测试用例，这个时候就会判断如果本来执行错的测试用例执行正确了，就不会再使用该测试用例进行迭代判断
                        #     corret_flag += 1

                        itercode_execution_result = code_execution_result
                        # _+=1
                        # 进行代码的迭代矫正
                        iter_code = self.iterate_code(Gptreply, correct_ficase, itercode_execution_result,
                                                      ques + task_description_gen, code_type,
                                                      list(correct_ficase.keys()), task['entry_point'])
                        # 迭代以后的代码在评估一次
                        itercase_state, itercode_execution_result = self.code_evaluate_resources(iter_code, num,
                                                                                                 new_case_candidate,
                                                                                                 'new_code')

                        # 更新tmp_judge_dict的状态
                    except Exception as e:
                        print(e)
                    # 合并迭代前和迭代后的代码候选
                    final_code_list = self.code_combine(code_dict1=code_execution_result,
                                                        code_dict2=itercode_execution_result)

                # print("卡顿检查3", num)
                # 选择最终的代码final_code
                compare_code, fast_code_reply, final_code = self.code_filter(Gptreply, final_code_list, first_key)
                # execution_result = self.code_evaluate_unbcase(final_code, num, True)


                # 构造返回结果
                code_generesult = {
                    "code": [final_code],
                    "slug_name": slug_name,
                    # "result": str(execution_result["passed"]),
                    "task_description_gen": task_description_gen,
                    "task_description_check": task_description_check,
                    "algorithim_description_opti": algorithm_description_candidate,
                    "debug_code_1": code_execution_result,
                    "debug_code_2": itercode_execution_result,
                    "final_code_dict": final_code_list,
                    "compare_code": compare_code,
                    "fast_code_reply": fast_code_reply,
                    "additional_cost_knowledge_result": additional_cost_knowledge_result,
                    "after_opt": after_opt
                }

                # print(f"[+]{num}final result", json.dumps([final_code]))
                return num, code_generesult
            except Exception as e:
                print(e)

    def codegen_process5(self, model, ques, num,benchmark_type):
        """
        variant1: 无算法到代码，直接从描述到代码,所以生成代码的步骤是直接让他生成code，换了prompt
        :param model:
        :param ques:
        :param num:
        :param benchmark_type:
        :return:
        """
        new_case_candidate = []
        code_dict_for_resources = {}
        Gptreply = GPTReply(model)
        # task = dict(ques.values())
        num = ques['task_id']
        task = ques
        ques = task["prompt"].strip() + "\n"

        # 检查 num 是否在特定范围
        if benchmark_type == "humaneval":
            if num in [2, 23, 41, 45, 53, 60, 71, 92, 97, 99, 102, 123, 124, 135, 137, 138, 144, 148, 156, 157, 159,
                       160]:
                return {
                    "code": "",
                    # "result": str(execution_result["passed"]),
                    "task_description_gen": "",
                    "task_description_check": "",
                    "algorithim_description_opti": "",
                    "debug_code_1": "",
                    "debug_code_2": "",
                    "final_code_dict": "",
                    "compare_code": "",
                    "fast_code_reply": ""
                }

        task_description_gen, task_description_check = self.generate_task_description(Gptreply, ques)
        # print(f"{num}",task_description_gen)

        # 生成用例候选
        case_candidate = self.casegentor.caseGenerator_testchain(model, ques + task_description_gen)
        # print("case generation success",num)

        def process_task(key, value,task_description):
            ""
            while True:
                try:
                    #从实践层面优化代码
                    additional_cost_knowledge = Gptreply.getreply(CodegenPrompt3.knowledge_databases_system,
                                                                 str(task_description), "")

                    tmp_code_result = self.generate_code_from_package(Gptreply, value, num,task_description,task["entry_point"],additional_cost_knowledge)

                    # additional_cost_analysis = Gptreply.getreply(CodegenPrompt3.additional_cost_analysis_system,
                    #                                                 str(tmp_code_result),"")

                    # opt_code_result = self.slowthinking.slow_opt_code_thinking("",tmp_code_result)
                    # opt_code_result = Gptreply.getreply(CodegenPrompt3.code_iter_practice_system,
                    #                                                 CodegenPrompt3.code_iter_practice_user.format(str(additional_cost_analysis),str(tmp_code_result)),"")
                    sanitized_solution = sanitize(
                        tmp_code_result, entrypoint=task["entry_point"]
                    )
                    # code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                    code_dict_for_resources[key] = sanitized_solution
                    break
                except:
                    pass

        process_task("1",ques + task_description_gen,ques + task_description_gen)
        case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num, case_candidate,
                                                                         'code')
        first_key = list(code_execution_result.keys())[0]

        # 检查代码执行结果
        tmp_judge_dict = []
        if self.code_case_result_check(code_execution_result):
            # tmp_judge_dict是判断为True且未通过的用例集合用于但他的值只有False和True，用于后续多轮迭代的用例状态实时更行
            # correct_ficase是判断为True且未通过的用例集合，包括用例的值item，用于后续多轮迭代
            # case_iteresult是所有的测试用例，并且前面执行为False的测试用例被加入了大模型判断为True和False的标志
            tmp_judge_dict, correct_ficase, case_iteresult = self.case_check_iterate(Gptreply, case_state,
                                                                                     ques + task_description_gen,
                                                                                     code_execution_result[first_key])
            new_case_candidate = self.case_update(case_iteresult)
            # print("卡顿检查2", num)
            # 基于new_case_candidate在评估一次
            new_case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num,
                                                                                 new_case_candidate, 'code')
        else:
            correct_ficase = {}
        itercode_execution_result = {}
        code_type = ""
        corret_flag = 0
        if len(correct_ficase) == 0:
            # 如果correct_ficase的长度为0，意味着没有未通过且为正确的测试用例
            final_code_list = code_execution_result
        else:
            # 首先基于correct_ficase进行迭代测试
            code_type = 'code'
            try:
                # 这里是做了一个判断，因为每次代码的更新迭代都会重新执行一次所有的测试用例，这个时候就会判断如果本来执行错的测试用例执行正确了，就不会再使用该测试用例进行迭代判断
                #     corret_flag += 1

                itercode_execution_result = code_execution_result
                # _+=1
                # 进行代码的迭代矫正
                iter_code = self.iterate_code(Gptreply, correct_ficase, itercode_execution_result,
                                              ques + task_description_gen, code_type, list(correct_ficase.keys()),
                                              task['entry_point'])
                # 迭代以后的代码在评估一次
                itercase_state, itercode_execution_result = self.code_evaluate_resources(iter_code, num,
                                                                                         new_case_candidate, 'new_code')

                # 更新tmp_judge_dict的状态
            except Exception as e:
                print(e)
            # 合并迭代前和迭代后的代码候选
            final_code_list = self.code_combine(code_dict1=code_execution_result, code_dict2=itercode_execution_result)

        # print("卡顿检查3", num)
        # 选择最终的代码final_code
        compare_code, fast_code_reply, final_code = self.code_filter(Gptreply, final_code_list, first_key)
        # execution_result = self.code_evaluate_unbcase(final_code, num, True)

        # 构造返回结果
        code_generesult = {

            "code": [final_code],
            # "result": str(execution_result["passed"]),
            "task_description_gen": task_description_gen,
            "task_description_check": task_description_check,
            "algorithim_description_opti": "",
            "debug_code_1": code_execution_result,
            "debug_code_2": itercode_execution_result,
            "final_code_dict": final_code_list,
            "compare_code": compare_code,
            "fast_code_reply": fast_code_reply
        }

        # print(f"[+]{num}final result", json.dumps([final_code]))
        return num, code_generesult


    def codegen_process6(self, model, ques, num,benchmark_type):
        """
        variant2: 没有实践优化，直接从算法到代码
        :param model:
        :param ques:
        :param num:
        :param benchmark_type:
        :return:
        """
        while True:
            try:
                def process_task(key, value, task_description):
                    ""
                    while True:
                        try:
                            # 从实践层面优化代码
                            additional_cost_knowledge =""

                            tmp_code_result = self.generate_code_from_package(Gptreply, value, num, task_description,
                                                                              task["entry_point"],
                                                                              additional_cost_knowledge)

                            # additional_cost_analysis = Gptreply.getreply(CodegenPrompt3.additional_cost_analysis_system,
                            #                                                 str(tmp_code_result),"")

                            # opt_code_result = self.slowthinking.slow_opt_code_thinking("",tmp_code_result)
                            # opt_code_result = Gptreply.getreply(CodegenPrompt3.code_iter_practice_system,
                            #                                                 CodegenPrompt3.code_iter_practice_user.format(str(additional_cost_analysis),str(tmp_code_result)),"")
                            sanitized_solution = sanitize(
                                tmp_code_result, entrypoint=task["entry_point"]
                            )
                            # code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                            code_dict_for_resources[key] = sanitized_solution
                            break
                        except:
                            pass

                new_case_candidate = []
                code_dict_for_resources = {}
                Gptreply = GPTReply(model)
                # task = dict(ques.values())
                num = ques['task_id']
                task = ques
                ques = task["prompt"].strip() + "\n"

                # 检查 num 是否在特定范围
                if benchmark_type == "humaneval":
                    if num in [2, 23, 41, 45, 53, 60, 71, 92, 97, 99, 102, 123, 124, 135, 137, 138, 144, 148, 156, 157,
                               159, 160]:
                        return {
                            "code": "",
                            # "result": str(execution_result["passed"]),
                            "task_description_gen": "",
                            "task_description_check": "",
                            "algorithim_description_opti": "",
                            "debug_code_1": "",
                            "debug_code_2": "",
                            "final_code_dict": "",
                            "compare_code": "",
                            "fast_code_reply": ""
                        }

                # 生成任务描述
                task_description_gen, task_description_check = self.generate_task_description(Gptreply, ques)
                # print(f"{num}",task_description_gen)

                # 生成用例候选
                case_candidate = self.casegentor.caseGenerator_testchain(model, ques + task_description_gen)
                # print("case generation success",num)

                # 生成算法描述候选
                algorithm_description_candidate = self.algorithm_generation(Gptreply, ques + task_description_gen)

                # print("卡顿检查1",num)
                # 使用多线程生成代码
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(process_task, key, value, ques + task_description_gen)
                        for key, value in algorithm_description_candidate.items()
                    ]
                    for future in futures:
                        future.result()  # 等待所有线程完成

                # case_state是生成的测试用例在本次投票机制中通过的情况，其中用例的投票结果分为通过、不通过、超时不通过
                # code_execution_result是代码的通过情况，按照降序排序
                case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num,
                                                                                 case_candidate, 'code')
                first_key = list(code_execution_result.keys())[0]

                # 检查代码执行结果
                tmp_judge_dict = []
                if self.code_case_result_check(code_execution_result):
                    # tmp_judge_dict是判断为True且未通过的用例集合用于但他的值只有False和True，用于后续多轮迭代的用例状态实时更行
                    # correct_ficase是判断为True且未通过的用例集合，包括用例的值item，用于后续多轮迭代
                    # case_iteresult是所有的测试用例，并且前面执行为False的测试用例被加入了大模型判断为True和False的标志
                    tmp_judge_dict, correct_ficase, case_iteresult = self.case_check_iterate(Gptreply, case_state,
                                                                                             ques + task_description_gen,
                                                                                             code_execution_result[
                                                                                                 first_key])
                    new_case_candidate = self.case_update(case_iteresult)
                    # print("卡顿检查2", num)
                    # 基于new_case_candidate在评估一次
                    new_case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num,
                                                                                         new_case_candidate, 'code')
                else:
                    correct_ficase = {}
                itercode_execution_result = {}
                code_type = ""
                corret_flag = 0
                if len(correct_ficase) == 0:
                    # 如果correct_ficase的长度为0，意味着没有未通过且为正确的测试用例
                    final_code_list = code_execution_result
                else:
                    # 首先基于correct_ficase进行迭代测试
                    code_type = 'code'
                    try:
                        # 这里是做了一个判断，因为每次代码的更新迭代都会重新执行一次所有的测试用例，这个时候就会判断如果本来执行错的测试用例执行正确了，就不会再使用该测试用例进行迭代判断
                        #     corret_flag += 1

                        itercode_execution_result = code_execution_result
                        # _+=1
                        # 进行代码的迭代矫正
                        iter_code = self.iterate_code(Gptreply, correct_ficase, itercode_execution_result,
                                                      ques + task_description_gen, code_type,
                                                      list(correct_ficase.keys()), task['entry_point'])
                        # 迭代以后的代码在评估一次
                        itercase_state, itercode_execution_result = self.code_evaluate_resources(iter_code, num,
                                                                                                 new_case_candidate,
                                                                                                 'new_code')

                        # 更新tmp_judge_dict的状态
                    except Exception as e:
                        print(e)
                    # 合并迭代前和迭代后的代码候选
                    final_code_list = self.code_combine(code_dict1=code_execution_result,
                                                        code_dict2=itercode_execution_result)

                # print("卡顿检查3", num)
                # 选择最终的代码final_code
                compare_code, fast_code_reply, final_code = self.code_filter(Gptreply, final_code_list, first_key)
                # execution_result = self.code_evaluate_unbcase(final_code, num, True)

                # 构造返回结果
                code_generesult = {
                    "code": [final_code],
                    # "result": str(execution_result["passed"]),
                    "task_description_gen": task_description_gen,
                    "task_description_check": task_description_check,
                    "algorithim_description_opti": algorithm_description_candidate,
                    "debug_code_1": code_execution_result,
                    "debug_code_2": itercode_execution_result,
                    "final_code_dict": final_code_list,
                    "compare_code": compare_code,
                    "fast_code_reply": fast_code_reply
                }

                # print(f"[+]{num}final result", json.dumps([final_code]))
                return num, code_generesult
            except Exception as e:
                print(e)

    def codegen_process7(self, model, ques, num,benchmark_type):
        """
        variant3: 没有测试用例矫正
        :param model:
        :param ques:
        :param num:
        :param benchmark_type:
        :return:
        """
        while True:
            try:
                def process_task(key, value, task_description):
                    ""
                    while True:
                        try:
                            # 从实践层面优化代码
                            additional_cost_knowledge = Gptreply.getreply(CodegenPrompt3.knowledge_databases_system,
                                                                          str(value), "")

                            tmp_code_result = self.generate_code_from_package(Gptreply, value, num, task_description,
                                                                              task["entry_point"],
                                                                              additional_cost_knowledge)

                            # additional_cost_analysis = Gptreply.getreply(CodegenPrompt3.additional_cost_analysis_system,
                            #                                                 str(tmp_code_result),"")

                            # opt_code_result = self.slowthinking.slow_opt_code_thinking("",tmp_code_result)
                            # opt_code_result = Gptreply.getreply(CodegenPrompt3.code_iter_practice_system,
                            #                                                 CodegenPrompt3.code_iter_practice_user.format(str(additional_cost_analysis),str(tmp_code_result)),"")
                            sanitized_solution = sanitize(
                                tmp_code_result, entrypoint=task["entry_point"]
                            )
                            # code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                            code_dict_for_resources[key] = sanitized_solution
                            break
                        except:
                            pass

                new_case_candidate = []
                code_dict_for_resources = {}
                Gptreply = GPTReply(model)
                # task = dict(ques.values())
                num = ques['task_id']
                task = ques
                ques = task["prompt"].strip() + "\n"

                # 检查 num 是否在特定范围
                if benchmark_type == "humaneval":
                    if num in [2, 23, 41, 45, 53, 60, 71, 92, 97, 99, 102, 123, 124, 135, 137, 138, 144, 148, 156, 157,
                               159, 160]:
                        return {
                            "code": "",
                            # "result": str(execution_result["passed"]),
                            "task_description_gen": "",
                            "task_description_check": "",
                            "algorithim_description_opti": "",
                            "debug_code_1": "",
                            "debug_code_2": "",
                            "final_code_dict": "",
                            "compare_code": "",
                            "fast_code_reply": ""
                        }

                # 生成任务描述
                task_description_gen, task_description_check = self.generate_task_description(Gptreply, ques)
                # print(f"{num}",task_description_gen)

                # 生成用例候选
                # case_candidate = self.casegentor.caseGenerator_testchain(model, ques + task_description_gen)
                # print("case generation success",num)

                # 生成算法描述候选
                algorithm_description_candidate = self.algorithm_generation(Gptreply, ques + task_description_gen)

                # print("卡顿检查1",num)
                # 使用多线程生成代码
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(process_task, key, value, ques + task_description_gen)
                        for key, value in algorithm_description_candidate.items()
                    ]
                    for future in futures:
                        future.result()  # 等待所有线程完成

                # case_state是生成的测试用例在本次投票机制中通过的情况，其中用例的投票结果分为通过、不通过、超时不通过
                # code_execution_result是代码的通过情况，按照降序排序
                # case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num,
                #                                                                  case_candidate, 'code')
                # first_key = list(code_execution_result.keys())[0]

                # 检查代码执行结果
                tmp_judge_dict = []
                # if self.code_case_result_check(code_execution_result):
                #     # tmp_judge_dict是判断为True且未通过的用例集合用于但他的值只有False和True，用于后续多轮迭代的用例状态实时更行
                #     # correct_ficase是判断为True且未通过的用例集合，包括用例的值item，用于后续多轮迭代
                #     # case_iteresult是所有的测试用例，并且前面执行为False的测试用例被加入了大模型判断为True和False的标志
                #     tmp_judge_dict, correct_ficase, case_iteresult = self.case_check_iterate(Gptreply, case_state,
                #                                                                              ques + task_description_gen,
                #                                                                              code_execution_result[
                #                                                                                  first_key])
                #     new_case_candidate = self.case_update(case_iteresult)
                #     # print("卡顿检查2", num)
                #     # 基于new_case_candidate在评估一次
                #     new_case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num,
                #                                                                          new_case_candidate, 'code')
                # else:
                #     correct_ficase = {}
                # itercode_execution_result = {}
                # code_type = ""
                # corret_flag = 0
                # if len(correct_ficase) == 0:
                #     # 如果correct_ficase的长度为0，意味着没有未通过且为正确的测试用例
                #     final_code_list = code_execution_result
                # else:
                #     # 首先基于correct_ficase进行迭代测试
                #     code_type = 'code'
                #     try:
                #         # 这里是做了一个判断，因为每次代码的更新迭代都会重新执行一次所有的测试用例，这个时候就会判断如果本来执行错的测试用例执行正确了，就不会再使用该测试用例进行迭代判断
                #         #     corret_flag += 1
                #
                #         itercode_execution_result = code_execution_result
                #         # _+=1
                #         # 进行代码的迭代矫正
                #         iter_code = self.iterate_code(Gptreply, correct_ficase, itercode_execution_result,
                #                                       ques + task_description_gen, code_type,
                #                                       list(correct_ficase.keys()), task['entry_point'])
                #         # 迭代以后的代码在评估一次
                #         itercase_state, itercode_execution_result = self.code_evaluate_resources(iter_code, num,
                #                                                                                  new_case_candidate,
                #                                                                                  'new_code')
                #
                #         # 更新tmp_judge_dict的状态
                #     except Exception as e:
                #         print(e)
                #     # 合并迭代前和迭代后的代码候选
                #     final_code_list = self.code_combine(code_dict1=code_execution_result,
                #                                         code_dict2=itercode_execution_result)

                # print("卡顿检查3", num)
                def code_filter(Gptreply,compare_code):
                    while True:

                        try:
                            fast_code_reply = Gptreply.getreply(CodegenPrompt3.fask_code_choice_system,
                                                                json.dumps(compare_code), "")
                            code_key_pattern = re.compile(r"```text\n(.*?)```", re.DOTALL)
                            fast_code = re.findall(code_key_pattern, fast_code_reply)[0]
                            fast_code_key = fast_code.strip()
                            fast_code = compare_code[fast_code_key]


                            break
                        except Exception as e:
                            print(e)
                            pass
                    return compare_code,fast_code_reply,fast_code
                # 选择最终的代码final_code
                compare_code, fast_code_reply, final_code = code_filter(Gptreply,code_dict_for_resources)
                # execution_result = self.code_evaluate_unbcase(final_code, num, True)

                # 构造返回结果
                code_generesult = {
                    "code": [final_code],
                    # "result": str(execution_result["passed"]),
                    "task_description_gen": task_description_gen,
                    "task_description_check": task_description_check,
                    "algorithim_description_opti": algorithm_description_candidate,
                    "debug_code_1": code_dict_for_resources,
                    "debug_code_2": "",
                    "final_code_dict": "",
                    "compare_code": compare_code,
                    "fast_code_reply": fast_code_reply
                }

                # print(f"[+]{num}final result", json.dumps([final_code]))
                return num, code_generesult
            except Exception as e:
                print(e)

    def codegen_process8(self, model, ques, num,benchmark_type):
        """
        独特性1，直接生成5个正确且高效的代码然后也不经过什么优化了
        :param model:
        :param ques:
        :param num:
        :param benchmark_type:
        :return:
        """
        """
        主程序
        :param model:
        :param ques:
        :param num:
        :return:
        """
        while True:
            try:
                def process_task(key, value,task_description):
                    ""
                    while True:
                        try:
                            #从实践层面优化代码
                            additional_cost_knowledge = Gptreply.getreply(CodegenPrompt3.knowledge_databases_system,
                                                                         str(value), "")

                            tmp_code_result = self.generate_code_from_package(Gptreply, value, num,task_description,task["entry_point"],additional_cost_knowledge)

                            # additional_cost_analysis = Gptreply.getreply(CodegenPrompt3.additional_cost_analysis_system,
                            #                                                 str(tmp_code_result),"")

                            # opt_code_result = self.slowthinking.slow_opt_code_thinking("",tmp_code_result)
                            # opt_code_result = Gptreply.getreply(CodegenPrompt3.code_iter_practice_system,
                            #                                                 CodegenPrompt3.code_iter_practice_user.format(str(additional_cost_analysis),str(tmp_code_result)),"")
                            sanitized_solution = sanitize(
                                tmp_code_result, entrypoint=task["entry_point"]
                            )
                            # code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                            code_dict_for_resources[key] = sanitized_solution
                            break
                        except:
                            pass

                new_case_candidate = []
                code_dict_for_resources = {}
                Gptreply = GPTReply(model)
                # task = dict(ques.values())
                num = ques['task_id']
                task = ques
                ques = task["prompt"].strip() + "\n"


                # 检查 num 是否在特定范围
                if benchmark_type == "humaneval":
                    if num in [2, 23, 41, 45, 53, 60, 71, 92, 97, 99, 102, 123, 124, 135, 137, 138, 144, 148, 156, 157, 159, 160]:
                        return {
                                "code": "",
                                # "result": str(execution_result["passed"]),
                                "task_description_gen": "",
                                "task_description_check": "",
                                "algorithim_description_opti": "",
                                "debug_code_1": "",
                                "debug_code_2": "",
                                "final_code_dict":"",
                                "compare_code":"",
                                "fast_code_reply":""
                               }

                # 生成任务描述
                task_description_gen, task_description_check = self.generate_task_description(Gptreply, ques)
                # print(f"{num}",task_description_gen)


                # 生成用例候选
                case_candidate = self.casegentor.caseGenerator_testchain(model, ques + task_description_gen)
                # print("case generation success",num)

                # 生成算法描述候选
                code_dict_for_resources = self.code_generation_multi(Gptreply, ques + task_description_gen,entry_point=task['entry_point'])

                # print("卡顿检查1",num)
                # 使用多线程生成代码
                # with ThreadPoolExecutor() as executor:
                #     futures = [
                #         executor.submit(process_task, key, value,ques + task_description_gen)
                #         for key, value in algorithm_description_candidate.items()
                #     ]
                #     for future in futures:
                #         future.result()  # 等待所有线程完成

                # case_state是生成的测试用例在本次投票机制中通过的情况，其中用例的投票结果分为通过、不通过、超时不通过
                # code_execution_result是代码的通过情况，按照降序排序
                case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num, case_candidate,'code')
                first_key = list(code_execution_result.keys())[0]

                # 检查代码执行结果
                tmp_judge_dict = []
                if self.code_case_result_check(code_execution_result):
                    # tmp_judge_dict是判断为True且未通过的用例集合用于但他的值只有False和True，用于后续多轮迭代的用例状态实时更行
                    # correct_ficase是判断为True且未通过的用例集合，包括用例的值item，用于后续多轮迭代
                    # case_iteresult是所有的测试用例，并且前面执行为False的测试用例被加入了大模型判断为True和False的标志
                    tmp_judge_dict,correct_ficase,case_iteresult = self.case_check_iterate(Gptreply, case_state, ques + task_description_gen,code_execution_result[first_key])
                    new_case_candidate = self.case_update(case_iteresult)
                    # print("卡顿检查2", num)
                    # 基于new_case_candidate在评估一次
                    new_case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num,
                                                                                     new_case_candidate, 'code')
                else:
                    correct_ficase = {}
                itercode_execution_result={}
                code_type = ""
                corret_flag = 0
                if len(correct_ficase) ==0 :
                    # 如果correct_ficase的长度为0，意味着没有未通过且为正确的测试用例
                    final_code_list = code_execution_result
                else:
                    # 首先基于correct_ficase进行迭代测试
                    code_type = 'code'
                    try:
                    # 这里是做了一个判断，因为每次代码的更新迭代都会重新执行一次所有的测试用例，这个时候就会判断如果本来执行错的测试用例执行正确了，就不会再使用该测试用例进行迭代判断
                    #     corret_flag += 1

                        itercode_execution_result = code_execution_result
                        # _+=1
                        #进行代码的迭代矫正
                        iter_code = self.iterate_code(Gptreply, correct_ficase, itercode_execution_result ,
                                                            ques + task_description_gen,code_type,list(correct_ficase.keys()),task['entry_point'])
                        # 迭代以后的代码在评估一次
                        itercase_state,itercode_execution_result = self.code_evaluate_resources(iter_code, num,
                                                                                         new_case_candidate,'new_code')

                        #更新tmp_judge_dict的状态
                    except Exception as e:
                        print(e)
                    # 合并迭代前和迭代后的代码候选
                    final_code_list = self.code_combine(code_dict1=code_execution_result,code_dict2=itercode_execution_result)

                # print("卡顿检查3", num)
                # 选择最终的代码final_code
                compare_code,fast_code_reply,final_code = self.code_filter(Gptreply,final_code_list, first_key)
                # execution_result = self.code_evaluate_unbcase(final_code, num, True)

                # 构造返回结果
                code_generesult = {
                    "code": [final_code],
                    # "result": str(execution_result["passed"]),
                    "task_description_gen": task_description_gen,
                    "task_description_check": task_description_check,
                    "algorithim_description_opti": code_dict_for_resources,
                    "debug_code_1": code_execution_result,
                    "debug_code_2": itercode_execution_result,
                    "final_code_dict":final_code_list,
                    "compare_code":compare_code,
                    "fast_code_reply":fast_code_reply
                }

                # print(f"[+]{num}final result", json.dumps([final_code]))
                return num,code_generesult
            except Exception as e:
                print(e)

    def codegen_process9(self, model, ques, num, benchmark_type):
        """
        special2: 无算法到代码，直接从描述到代码,所以生成代码的步骤是直接让他生成code，换了prompt
        :param model:
        :param ques:
        :param num:
        :param benchmark_type:
        :return:
        """
        new_case_candidate = []
        code_dict_for_resources = {}
        Gptreply = GPTReply(model)
        # task = dict(ques.values())
        num = ques['task_id']
        task = ques
        ques = task["prompt"].strip() + "\n"

        # 检查 num 是否在特定范围
        if benchmark_type == "humaneval":
            if num in [2, 23, 41, 45, 53, 60, 71, 92, 97, 99, 102, 123, 124, 135, 137, 138, 144, 148, 156, 157, 159,
                       160]:
                return {
                    "code": "",
                    # "result": str(execution_result["passed"]),
                    "task_description_gen": "",
                    "task_description_check": "",
                    "algorithim_description_opti": "",
                    "debug_code_1": "",
                    "debug_code_2": "",
                    "final_code_dict": "",
                    "compare_code": "",
                    "fast_code_reply": ""
                }

        task_description_gen, task_description_check = self.generate_task_description(Gptreply, ques)
        print(f"{num}",task_description_gen)

        # 生成用例候选
        case_candidate = self.casegentor.caseGenerator_testchain(model, ques + task_description_gen)

        # print("case generation success",num)

        def process_task(key, value, task_description):
            ""
            while True:
                try:
                    # 从实践层面优化代码
                    additional_cost_knowledge = ""
                    tmp_code_result = Gptreply.getreply(CodegenPrompt3.varient_1_get_code_system,
                                                              task_description, "")


                    # additional_cost_analysis = Gptreply.getreply(CodegenPrompt3.additional_cost_analysis_system,
                    #                                                 str(tmp_code_result),"")

                    # opt_code_result = self.slowthinking.slow_opt_code_thinking("",tmp_code_result)
                    # opt_code_result = Gptreply.getreply(CodegenPrompt3.code_iter_practice_system,
                    #                                                 CodegenPrompt3.code_iter_practice_user.format(str(additional_cost_analysis),str(tmp_code_result)),"")
                    sanitized_solution = sanitize(
                        tmp_code_result, entrypoint=task["entry_point"]
                    )
                    # code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                    code_dict_for_resources[key] = sanitized_solution
                    break
                except:
                    pass

        process_task("1", "", ques + task_description_gen)
        case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num, case_candidate,
                                                                         'code')
        first_key = list(code_execution_result.keys())[0]

        # 检查代码执行结果
        tmp_judge_dict = []
        if self.code_case_result_check(code_execution_result):
            # tmp_judge_dict是判断为True且未通过的用例集合用于但他的值只有False和True，用于后续多轮迭代的用例状态实时更行
            # correct_ficase是判断为True且未通过的用例集合，包括用例的值item，用于后续多轮迭代
            # case_iteresult是所有的测试用例，并且前面执行为False的测试用例被加入了大模型判断为True和False的标志
            tmp_judge_dict, correct_ficase, case_iteresult = self.case_check_iterate(Gptreply, case_state,
                                                                                     ques + task_description_gen,
                                                                                     code_execution_result[first_key])
            new_case_candidate = self.case_update(case_iteresult)
            # print("卡顿检查2", num)
            # 基于new_case_candidate在评估一次
            new_case_state, code_execution_result = self.code_evaluate_resources(code_dict_for_resources, num,
                                                                                 new_case_candidate, 'code')
        else:
            correct_ficase = {}
        itercode_execution_result = {}
        code_type = ""
        corret_flag = 0
        if len(correct_ficase) == 0:
            # 如果correct_ficase的长度为0，意味着没有未通过且为正确的测试用例
            final_code_list = code_execution_result
        else:
            # 首先基于correct_ficase进行迭代测试
            code_type = 'code'
            try:
                # 这里是做了一个判断，因为每次代码的更新迭代都会重新执行一次所有的测试用例，这个时候就会判断如果本来执行错的测试用例执行正确了，就不会再使用该测试用例进行迭代判断
                #     corret_flag += 1

                itercode_execution_result = code_execution_result
                # _+=1
                # 进行代码的迭代矫正
                iter_code = self.iterate_code(Gptreply, correct_ficase, itercode_execution_result,
                                              ques + task_description_gen, code_type, list(correct_ficase.keys()),
                                              task['entry_point'])
                # 迭代以后的代码在评估一次
                itercase_state, itercode_execution_result = self.code_evaluate_resources(iter_code, num,
                                                                                         new_case_candidate, 'new_code')

                # 更新tmp_judge_dict的状态
            except Exception as e:
                print(e)
            # 合并迭代前和迭代后的代码候选
            final_code_list = self.code_combine(code_dict1=code_execution_result, code_dict2=itercode_execution_result)

        # print("卡顿检查3", num)
        # 选择最终的代码final_code
        compare_code, fast_code_reply, final_code = self.code_filter(Gptreply, final_code_list, first_key)
        def code_opti(code):
            opt_code_result = Gptreply.getreply(CodegenPrompt3.code_iter_practice_system,
                                                        code,"")
            sanitized_solution = sanitize(
                opt_code_result, entrypoint=task['entry_point']
            )
            return sanitized_solution

        final_code_1 = code_opti(final_code)


        # execution_result = self.code_evaluate_unbcase(final_code, num, True)

        # 构造返回结果
        code_generesult = {
            "code": [final_code_1],
            # "result": str(execution_result["passed"]),
            "task_description_gen": task_description_gen,
            "task_description_check": task_description_check,
            "algorithim_description_opti": "",
            "debug_code_1": code_execution_result,
            "debug_code_2": itercode_execution_result,
            "final_code_dict": final_code_list,
            "compare_code": compare_code,
            "fast_code_reply": fast_code_reply
        }

        # print(f"[+]{num}final result", json.dumps([final_code]))
        return num, code_generesult

    def codegen_process10(self, model, ques, num, benchmark_type):
        """
        COT prompt
        :param model:
        :param ques:
        :param num:
        :param benchmark_type:
        :return:
        """
        # new_case_candidate = []
        # code_dict_for_resources = {}
        # Gptreply = GPTReply(model)
        # # task = dict(ques.values())
        # num = ques['task_id']
        # task = ques
        # ques = task["prompt"].strip() + "\n"

        # 检查 num 是否在特定范围
        if benchmark_type == "humaneval":
            if num in [2, 23, 41, 45, 53, 60, 71, 92, 97, 99, 102, 123, 124, 135, 137, 138, 144, 148, 156, 157, 159,
                       160]:
                return {
                    "code": "",
                    # "result": str(execution_result["passed"]),
                    "task_description_gen": "",
                    "task_description_check": "",
                    "algorithim_description_opti": "",
                    "debug_code_1": "",
                    "debug_code_2": "",
                    "final_code_dict": "",
                    "compare_code": "",
                    "fast_code_reply": ""
                }

        new_case_candidate = []
        code_dict_for_resources = {}
        Gptreply = GPTReply(model)
        slug_name = ""
        if benchmark_type == "Mercury":
            num = ques["slug_name"]
            slug_name = ques["id"]
            task = ques
            ques = task["prompt"] + "\n" + task["pretty_content"][0]
        else:
            # task = dict(ques.values())
            num = ques['task_id']
            task = ques
            ques = task["prompt"].strip() + "\n"

        # 构造返回结果
        final_code_1 = Gptreply.getreply(
            CodegenPrompt3.COT_system,
            ques,
            ""

        )

        final_code_1 = sanitize(
            final_code_1, entrypoint=task["entry_point"]
        )

        code_generesult = {
            "code": [final_code_1],
            # "result": str(execution_result["passed"]),
        }

        # print(f"[+]{num}final result", json.dumps([final_code]))
        return num, code_generesult



    def custom_serializer(self, obj):
        """
        自定义序列化函数，用于处理无法直接序列化的对象。
        """
        return str(obj)

    def mainprocess(self, module, prompt, key, date_time,benchmark_type):
        """
        处理任务的主函数，模拟实际逻辑。
        """
        try:
            # 假设 process 方法会返回结果
            result = self.codegen_process4(module, prompt, key,benchmark_type)  # 你的实际处理函数
            with self.results_lock:
                self.results[key] = result
                self.save_result(module, date_time, key, result)  # 使用键名保存结果
        except Exception as e:
            with self.results_lock:
                self.failed_tasks[key] = str(e)  # 记录失败任务和错误信息
            print(f"任务 {key} 失败: {e}")

    def threading_execution(self, module, date_time, benchmark_type, test_data):
        """
        使用多线程执行任务，捕获异常并保证其他任务继续执行，并设置超时重试机制。
        """
    # 解析 test_data，提取键名和值
        parsed_data = {list(item.keys())[0]: item[list(item.keys())[0]] for item in test_data}

        timeout = 1500  # 每个任务的超时时间
        max_retries = 3  # 每个任务的最大重试次数

        # 初始化任务重试次数
        task_retry_count = {key: 0 for key in parsed_data.keys()}

        # 检查文件，跳过已存在的任务
        filename = f"../cache/self_codegen_{module}_{date_time}_all.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
            # 过滤掉已存在的任务
            tasks_to_process = [key for key in parsed_data.keys() if key not in existing_data.keys()]
            print(f"跳过已存在的任务: {set(parsed_data.keys()) - set(tasks_to_process)}")
            print(f"需要执行的任务: {set(tasks_to_process)}")
        else:
            tasks_to_process = list(parsed_data.keys())

        with tqdm(total=len(tasks_to_process), desc="任务执行中", unit="任务") as pbar:
            with ThreadPoolExecutor(max_workers=1) as executor:
                # 初始提交所有任务
                futures = {executor.submit(self.mainprocess, module, parsed_data[key], key, date_time,benchmark_type): key for key in tasks_to_process}

                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        future.result(timeout=timeout)
                        with self.results_lock:
                            if key in self.results:
                                pbar.update(1)
                    except TimeoutError:
                        with self.results_lock:
                            self.failed_tasks[key] = "任务超时"
                        print(f"任务 {key} 超时")
                        # 重试逻辑
                        if task_retry_count[key] < max_retries:
                            task_retry_count[key] += 1
                            print(f"任务 {key} 重试，次数 {task_retry_count[key]}")
                            future = executor.submit(self.mainprocess, module, parsed_data[key], key, date_time,benchmark_type)
                            futures[future] = key
                        else:
                            self.failed_tasks[key] = "任务超时"
                            pbar.update(1)
                    except Exception as e:
                        with self.results_lock:
                            self.failed_tasks[key] = str(e)
                        print(f"任务 {key} 失败: {e}")
                        pbar.update(1)

    def save_result(self, module, date, key, result):
        """
        将单个任务的结果追加写入 JSON 文件。
        如果键名已存在，则跳过写入。
        """
        filename = f"../cache/self_codegen_{module}_{date}_all.json"
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # 如果文件存在，先读取内容
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        # 检查键名是否已存在
        if key in existing_data:
            print(f"任务 {key} 的键名已存在，跳过写入")
            return

        # 更新现有数据
        existing_data[key] = result

        # 写入更新后的数据
        with open(filename, "w") as f:
            json.dump(existing_data, f, indent=4, default=self.custom_serializer)

        print(f"任务 {key} 的结果已保存到 {filename}")


def get_data(dataset):
    version = "default"
    if dataset == "humaneval":
        dataset_dict = get_human_eval_plus(version=version)
    elif dataset == "mbpp":
        dataset_dict = get_mbpp_plus(version=version)
    elif dataset == "evalperf":
        original_dataset = {**get_human_eval_plus(), **get_mbpp_plus()}
        dataset_dict = {k: original_dataset[k] for k in get_evalperf_data()}
        # assert id_range is None, "id_range not supported for evalperf"
    elif dataset == "Mercury":
        mercury_path = os.path.join(BASE_DIR, "dataset", "Mercury", "eval-00000-of-00001.parquet")
        dataset_dict = datasets.load_dataset("parquet", data_files=mercury_path, split="train")
        return [{example["slug_name"]:dict(example)} for example in dataset_dict]
    else:
        raise ValueError(f"Invalid dataset {dataset}")
    dataset_list = [{key: value} for key, value in dataset_dict.items()]
    return dataset_list

model_name = ["Qwen/Qwen3.5-9B"]
for i in model_name:
    print(f"当前执行为第{i}轮")
    prom_data = get_data("humaneval")[:2]   # 只跑前 2 道题（测试用）
    codegenerator = CodeGenerator(i)
    # codegenerator.threading_execution(i, f"enamel_0125_1", "humaneval", prom_data)  # 批量生成生成
    codegenerator.threading_execution(i, f"qwen35_9b_2", "humaneval", prom_data)  # 批量生成，结果文件名带 qwen35_9b_2