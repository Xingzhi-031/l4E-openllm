import threading
import traceback
import time
from typing import List

from prompt.prompt import caseGen
import re
from gpt.gpt_reply import GPTReply
from execute import *
import subprocess
import tempfile
try:
    import resource
except ImportError:
    resource = None
from concurrent.futures import ThreadPoolExecutor, as_completed
class CaseGenerator:
    TPL_MAKE = '''%s
    %s
    random.seed(%d)
    __input = generate_input(size = %d, lid = %d, cid = %d)
    '''  # (disprompt, generator, seed, size)
    TPL_RUN = '''%s
%s
__t0 = time.time()
__output = %s(*__input)
__t1 = time.time()
    '''  # (disprompt, solution, entry_point)
    TPL_RUN2 = """%s
%s
%s
"""
    TPL_TEST = '''%s
    pass
%s
__accepted = __check(__input, __answer, __output)
'''
    TPL_RUN3 = """%s
%s
%s
    """
    def __init__(self,problems):
        self.lock = threading.Lock()  # 添加锁
        # with open("../cache/0918/self_codegen_deepseek-coder_0914_3_extract_case_turn0_all_modified.json", "r") as f:
        #     content = f.read()
        #     data = json.loads(content)  # 解析 JSON 数据
        # self.test_case = data

        # with open("../cache/backup/self_codegen_deepseek_coder_0923_0_votingcase_turn0_all_failed.json", "r") as f:
        #     content = f.read()
        #     # print(content)
        #     data1 = json.loads(content)  # 解析 JSON 数据
        # self.untrust_test_case = data1
        # self.problems = pd.read_csv(problems)
        # self.evalplus_case = self.generate_case_from_evalpuls()
        # testcase = data[i]['ini_test_case']
        # self.tolerence_sec = tolerence_sec
        # self.timeout_factor = timeout_factor

    def caseGenerator(self, model, ques):
        Gptreply = GPTReply(model)
        task_description_gen = Gptreply.getreply(caseGen.pro_description_case, ques, "")
        # print(task_description_gen)
        edge_case_description = Gptreply.getreply(caseGen.edge_case_description_system,
                                                  caseGen.edge_case_description_user.format(task_description_gen, ques),
                                                  "")
        max_loop = 10
        current_loop = 0
        tmp_case = ""
        while current_loop<max_loop:
            current_loop+=1
            try:
                inputcase_generator = Gptreply.getreply(caseGen.inputcase_generator_system,
                                                        caseGen.inputcase_generator_user.format(task_description_gen,
                                                                                                edge_case_description, ques),
                                                        "")

                code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                matches = re.findall(code_regexp_pattern, inputcase_generator)
                tmp_code = matches[0] if matches else ""
                tmp_case = eval(tmp_code)
                break
            except Exception as e:
                print("case gene failed")
                pass

        return tmp_case
    def caseGenerator2(self, model, ques):
        Gptreply = GPTReply(model)
        # task_description_gen = Gptreply.getreply(caseGen.pro_description_case, ques, "")
        # # print(task_description_gen)
        # edge_case_description = Gptreply.getreply(caseGen.edge_case_description_system,
        #                                           caseGen.edge_case_description_user.format(task_description_gen, ques),
        #                                           "")
        max_loop = 10
        current_loop = 0
        tmp_case = ""
        while current_loop<max_loop:
            current_loop+=1
            try:
                inputcase_generator = Gptreply.getreply(caseGen.inputcase_generator_system1,
                                                        caseGen.inputcase_generator_user1,
                                                        ques)

                code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                matches = re.findall(code_regexp_pattern, inputcase_generator)
                tmp_code = matches[0] if matches else ""
                tmp_case = eval(tmp_code)
                break
            except Exception as e:
                print("case gene failed")
                pass

        return tmp_case
    #  def running_constructor(code):
    #             code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    #             matches = re.findall(code_regexp_pattern, code)
    #             code= matches[0] if matches else ""
    #             with tempfile.NamedTemporaryFile(suffix=".py", delete=True, mode='w') as temp_file:
    #                 temp_file.write(code)
    #                 temp_file.flush()
    #                 temp_file_path = temp_file.name
    #                 # 使用subprocess来执行该代码文件
    #                 try:
    #                     result = subprocess.run(
    #                         ["python", temp_file_path],
    #                         capture_output=True, text=True
    #                     )
    #                     test_result = result.stdout if result.returncode == 0 else result.stderr
    #                     if "Error" in test_result:
    #                         return False, test_result
    #                     else:
    #                         return test_result
    #                 except Exception as e:
    #                     return f"Error executing file: {str(e)}"

    def caseGenerator_testchain(self, model, ques):

        def case_formated(case):
            try:
                code_regexp_pattern = re.compile(r"```text\n(.*?)```", re.DOTALL)
                matches = re.findall(code_regexp_pattern, case)
                inputcase = matches[0].split("\n") if matches else []
                inputcase = [item for item in inputcase if item.strip()]
            except:
                raise RuntimeError()
            return inputcase

        def total_case_formated(case):
            code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
            matches = re.findall(code_regexp_pattern, case)
            tmp_case = matches[0] if matches else ""
            if not tmp_case:
                raise RuntimeError("No valid Python code block found")
            formatted_case = [
                line.strip() for line in tmp_case.splitlines()
                if "assert" in line
            ]
            if not formatted_case:
                raise RuntimeError("No 'assert' statements found in the code block")
            return formatted_case[0]

        Gptreply = GPTReply(model)
        case_num = 20
        while True:
            try:
                designer_agent_1 = Gptreply.getreply(caseGen.designer_agent_system.format(case_num),
                                                     caseGen.designer_agent_user.format(ques), "")
                if not designer_agent_1:
                    if not designer_agent_1 and case_num == 10:
                        case_num = 2
                    else:
                        case_num = 10
                inputcase_1 = case_formated(designer_agent_1)
                break
            except Exception as e:
                print(e)
                pass

        def process_input_case(Gptreply, caseGen, ques, input_item, final_case, lock):
            while True:
                try:
                    case_check = ""
                    inputcase_generator = Gptreply.getreply(
                        caseGen.calculator_agent_nopy_system,
                        caseGen.calculator_agent_nopy_user.format(ques, input_item),
                        case_check
                    )
                    if not inputcase_generator:
                        break
                    tmp_case = total_case_formated(inputcase_generator)
                    with lock:
                        final_case.append(tmp_case)
                    break
                except Exception as e:
                    print("Case generation failed:", e)
                    break

        def generate_test_cases_multithreaded(Gptreply, caseGen, ques, inputcase_1):
            max_retry = 5
            current = 0
            while current<max_retry:
                current+=1
                final_case = []
                lock = threading.Lock()
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(process_input_case, Gptreply, caseGen, ques, input_item, final_case, lock)
                        for input_item in inputcase_1
                    ]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Thread failed with exception: {e}")
                final_case = list(dict.fromkeys(final_case))
                if len(final_case)>=5:
                    break
            return final_case

        return generate_test_cases_multithreaded(Gptreply, caseGen, ques, inputcase_1)

    # while True:
    #     try:
    #         input_constructor = Gptreply.getreply(caseGen.input_constructor_system,
    #                                                 caseGen.input_constructor_user.format(ques),
    #                                                 "")
    #         complexity_case = running_constructor(input_constructor)
    #         try:
    #
    #             tmp_case+=eval(complexity_case)
    #             break
    #         except:
    #             try:
    #                 complexity_case = complexity_case.split("\n")
    #                 tmp_case+= [item for item in complexity_case if item.strip()]
    #                 break
    #             except:
    #                 continue
    #     except:
    #         continue

    def caseGenerator_testchainwithfewshot(self, model, ques,num):
        Gptreply = GPTReply(model)
        case = self.test_case[str(num)]['ini_test_case']
        while True:
            try:
                designer_agent = Gptreply.getreply(caseGen.designer_agent_system.format(case),caseGen.designer_agent_user.format(ques), "")
                code_regexp_pattern = re.compile(r"```text\n(.*?)```", re.DOTALL)
                matches = re.findall(code_regexp_pattern, designer_agent)
                inputcase = matches[0] if matches else ""
                break
            except Exception as e:
                pass

        max_loop = 10
        current_loop = 0
        tmp_case = ""
        while current_loop < max_loop:
            current_loop += 1
            try:
                inputcase_generator = Gptreply.getreply(caseGen.calculator_agent_agent_fewshot_system.format(case),
                                                        caseGen.calculator_agent_nopy_user.format(ques,inputcase),
                                                        "")

                code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                matches = re.findall(code_regexp_pattern, inputcase_generator)
                tmp_case = matches[0] if matches else ""
                # tmp_case = tmp_code
                break
            except Exception as e:
                print("case gene failed")
                pass

        return tmp_case

    def caseGenerator_votingcase(self, model, ques,num):
        Gptreply = GPTReply(model)
        with open("../cache/self_codegen_deepseek-coder_0919_allcodecandidate_turn0_all_failed.json","r") as f:
            data = json.loads(f.read())

        test_case = self.caseGenerator_testchain(model,ques)
        code_list = data[str(num)]['algorithim_trans_tmpcode_list']
        flag = False
        isgenerate_caseagain = False
        codecase_json = True
        tmp_case = ''
        case_check = ""
        max_loop = 5
        current_loop =0
        while current_loop<max_loop:
            result = {}
            for i in code_list.keys():
                result[i] = {}
                passed,reason,code = self.evaluate_case(code_list[i],test_case,flag)
                # print(code)
                result[i] = {
                    'passed': passed,
                    'reason': reason,
                    'code': code
                }
            has_false = any(not item['passed'] for item in result.values())

            if has_false:
                while True:
                    try:
                        isgenerate_caseagain = True
                        case_check = Gptreply.getreply(caseGen.case_checkwithcode_system,
                                                                caseGen.case_checkwithcode_user.format(ques,result),
                                                                "")
                        if not case_check:
                            codecase_json = False
                            break
                        code_regexp_pattern = re.compile(r"```json\n(.*?)```", re.DOTALL)
                        matches = re.findall(code_regexp_pattern, case_check)
                        codecase_json = json.loads(matches[0])
                        break
                    except:
                        pass
            else:
                break
            code_list = codecase_json
            flag = True
            current_loop +=1
            if not codecase_json:
                isgenerate_caseagain = False
                test_case = ""
                break
        if isgenerate_caseagain:
            while True:
                try:
                    case_result = Gptreply.getreply(caseGen.case_extraction_in_code_system,
                                                            case_check,
                                                            "")
                    code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                    matches = re.findall(code_regexp_pattern, case_result)
                    tmp_case = matches[0] if matches else ""
                    break
                except Exception as e:
                    pass
        else:
            tmp_case = test_case
        # print("a")
        return tmp_case

    def caseGenerator_votingcase2(self, model, ques,num):
        '''
        先生成list类型的case，
        然后使用正则表达式分别提取输出和输出，接下来比对结果：

        如果执行结果和大模型生成的output case大于半数不一致，则case有问题
        不一致则筛掉

        edge case：代码中没有添加针对一些边界测试用例的处理，导致结果出错（能正常执行，但是输出有问题）
        在voting的环境下，是不是就会全被筛选掉？

        voting结束以后，再添加一轮测试用例与题目的一致性检验，降低正确测试用例被筛选掉的几率

        :param model:
        :param ques:
        :param num:
        :return:

        '''
        Gptreply = GPTReply(model)
        with open("../cache/self_codegen_deepseek-coder_0919_allcodecandidate_turn0_all_failed.json", "r") as f:
            data = json.loads(f.read())
        gpt_reply_case = self.caseGenerator_testchain(model, ques)
        if gpt_reply_case != "":
            test_case = gpt_reply_case.split('\n')
            code_list = data[str(num)]['algorithim_trans_tmpcode_list']
            flag = False
            result = {}
            final_case = []
            for case in test_case:
                # assert largest_prime_factor(15) == 5
                code_regexp_pattern = re.compile(r"assert(.+)==", re.DOTALL)
                matches = re.findall(code_regexp_pattern,case)
                input_case = f"print({(matches[0] if matches else '')})"
                # 'assert has_close_elements([1.0, 2.0, 3.0], 0.0) == True'
                code_regexp_pattern = re.compile(r"==(.+)", re.DOTALL)
                matches = re.findall(code_regexp_pattern, case)
                output_case = matches[0] if matches else ''
                result[input_case] = {}
                for item in code_list.keys():
                    result[input_case][item] = {}
                    passed, reason, code = self.evaluate_case2(code_list[item], input_case, flag,output_case)
                    # print(code)
                    result[input_case][item] = {
                        'passed': passed,
                        'reason': reason,
                        'code': code
                    }

                total_codes = len(result[input_case])
                failed_codes = sum(1 for item in result[input_case].values() if item['passed'] is False)
                delflag = failed_codes > (total_codes / 2)
                if not delflag:
                    final_case.append(case)
                else:
                    final_case.append("")

            tmp_case = "\n".join(final_case)
        else:
            tmp_case = ""
            result = ""
        return tmp_case,gpt_reply_case,result

    def evaluate(self,code,ini_test_case):
        scope = dict(time=time, input=None, print=None, List=List)
        for i in ini_test_case:
            code_executed = self.TPL_RUN2 % (code, i)
            try:
                unsafe_execute(code_executed,scope)
            except Exception as e:
                # error_message = traceback.format_exc()
                error_msg =  {"The wrong case":i,"Error":str(e)}
                print(error_msg)
                return False, error_msg
        return True,None

    def case_extraction(self, model,ques):
        Gptreply = GPTReply(model)
        while True:
            try:
                inputcase_generator = Gptreply.getreply(caseGen.case_extraction_system,
                                                        ques,
                                                        "")

                code_regexp_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                matches = re.findall(code_regexp_pattern, inputcase_generator)
                tmp_case = matches[0] if matches else ""
                break
            except:
                pass

        return tmp_case

    def evaluate_easy(self,code,num):
        code = self.TPL_RUN2 % (code, self.test_case[str(num)]['ini_test_case'])
        with tempfile.NamedTemporaryFile(suffix=".py", delete=True, mode='w') as temp_file:
            temp_file.write(code)
            temp_file.flush()
            temp_file_path = temp_file.name
        # 使用subprocess来执行该代码文件
            try:
                result = subprocess.run(
                    ["python", temp_file_path],
                    capture_output=True, text=True
                )
                test_result = result.stdout if result.returncode == 0 else result.stderr
                if "Error" in test_result:
                    return False,test_result
                else:
                    return True,None
            except Exception as e:
                return f"Error executing file: {str(e)}"

    def generate_case_from_evalpuls(self):
        with open("../cache/HumanEvalPlusInputs.jsonl","r") as f:
            data = f.readlines()
        all_case = {}
        for line in data:
            try:
                ini_case = json.loads(line)
                test_case = []
                task_id = int(ini_case['task_id'])
                problem = self.problems.iloc[task_id]
                for i in ini_case['inputs']:
                    test_case.append(f"print({problem.entry_point.strip()}{tuple(i)})")
                all_case[ini_case['task_id']] = "\n".join(test_case)
            except:
                print(line)
        return all_case

    def evaluate_gem5_docker(self,code_case,num):
        """
        1.写入文件（文件位置待定）
        2.评估每个代码的效率
        3.使用正则匹配比对得到最小的
        :param code_case:
        :return:
        """
        success_code_list = {}
        try:
            # 确保目录存在，如果不存在则创建
            for key,code_item in code_case.items():
                with open(f"gem5/out/test{key}","w+") as f:
                    f.write(code_item)


                docker_command = (
                    "docker exec -w /gem5 26245b209 "
                    "sh -c \"export M5_PATH=/gem5/configs/example && "
                    f"./build/X86/gem5.opt -d /gem5/out/{num}/test{key} configs/deprecated/example/se.py "
                    "--cpu-type=TimingSimpleCPU "
                    "--mem-size=2GB "
                    f"--cmd=/usr/bin/python3 --options='/gem5/gem5test_code/{num}/script_{key}.py'\""
                )

                # 执行Docker命令
                print(f"Running Docker command for {num}, test {key}...")
                result = subprocess.run(docker_command, shell=True, capture_output=True, text=True)
                # 打印命令的标准输出和标准错误
                print(result.stdout)
                if result.stderr:
                    print(f"Error: {result.stderr}")
                    success_code_list[key] = False
                else:
                    success_code_list[key] = True
            return success_code_list
        except Exception as e:
            print(f"Error in task {num}: {e}")

    def time_extractor(self,file_content):
        # stats_compile = re.compile("simSeconds\s+(.*?)\s+# Number o", re.DOTALL)
        pattern = re.compile(r"simSeconds\s+(.*?)\s+# Number o", re.DOTALL)
        result = re.findall(pattern,file_content)
        return result

    def evaluate_gem5_code_time(self,code_dict,num):
        """
        如果代码是错的，那么他的运行时间是很短的，这个时候可能会筛选到时间很短的，这个筛选的策略还需要进一步确定
        :param code_dict:
        :param num:
        :return:
        """
        code_time_list = {}
        for key,value in code_dict.items():
            if value== True:
                with open(f"/gem5/out/{num}/test{key}","r") as f:
                    time = float(self.time_extractor(f.read()))
                    code_time_list[key] =time
        #返回code_time_list中time最小的数字
        if code_time_list:
            min_key = min(code_time_list, key=code_time_list.get)
            return min_key, code_time_list[min_key]  # Return both the key and the minimum time value
        else:
            print("No valid time entries found.")
            return None, None
    def evaluate_gem5_process(self,num,code_dict):
        """
        使用gem5评估效率，这里的输入应该是代码(dic{string(number):string}类型)
        1.将testcase嵌入到代码中
        2.然后使用gem5跑出所有代码的时间
        3.对比得到时间最短的代码
        code需要以字典的形式存储，后面比对的时候才能取到
        :param num:
        :return:
        """
    #print(has_close_elements([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], 0.5))
        test_case = self.evalplus_case[str(num)]
        code_case ={}
        for key,code_item in code_dict.items():
            code = self.TPL_RUN2 % (code_item, test_case)
            code_case[key] = code
        #写一个多线程跑code_case的代码
        code_gem5run_result = self.evaluate_gem5_docker(code_case,num)
        #提取code_gem5run_result 这个字典中value为True的键值
        code_gem5run_min_efficiency_key,time = self.evaluate_gem5_code_time(code_gem5run_result,num)
        return code_gem5run_min_efficiency_key

    def evaluate_diffset(self,code,num,caseset):
        with self.lock:
            if caseset:
                test_case = self.test_case[str(num)]['ini_test_case']
            else:
                test_case = self.untrust_test_case[str(num)]['modified_test_case']
        try:
            code = self.TPL_RUN2 % (code, test_case)
        except:
            code = self.TPL_RUN2 % (code, "")
        with tempfile.NamedTemporaryFile(suffix=".py", delete=True, mode='w') as temp_file:
            temp_file.write(code)
            temp_file.flush()
            temp_file_path = temp_file.name
        # 使用subprocess来执行该代码文件
            try:
                result = subprocess.run(
                    ["python", temp_file_path],
                    capture_output=True, text=True
                )
                test_result = result.stdout if result.returncode == 0 else result.stderr
                if "Error" in test_result:
                    return False,test_result
                else:
                    return True,None
            except Exception as e:
                return f"Error executing file: {str(e)}"
    def evaluate_case(self,code,testcase,flag):

        if not flag:
            code = self.TPL_RUN2 % (code,testcase)
        else:
            code = self.TPL_RUN2 % (code, "")

        with tempfile.NamedTemporaryFile(suffix=".py", delete=True, mode='w') as temp_file:
            temp_file.write(code)
            temp_file.flush()
            temp_file_path = temp_file.name
        # 使用subprocess来执行该代码文件
            try:
                result = subprocess.run(
                    ["python", temp_file_path],
                    capture_output=True, text=True
                )
                test_result = result.stdout if result.returncode == 0 else result.stderr
                if "Error" in test_result:
                    return False,test_result,code
                else:
                    return True,None,code
            except Exception as e:
                return f"Error executing file: {str(e)}"
#  执行测试用例的时候，如果是错的，再执行一下他的输出（作为对比）
    def evaluate_resources_process(self, num, code_dict, case, compare_obj):
        def code_execution(code):
            with tempfile.NamedTemporaryFile(suffix=".py", delete=True, mode='w') as temp_file:
                temp_file.write(code)
                temp_file.flush()
                temp_file_path = temp_file.name
                try:
                    time_start = time.time()
                    result = subprocess.run(
                        ["python", temp_file_path],
                        capture_output=True,
                        text=True,
                        timeout=1
                    )
                    user_time = time.time() - time_start
                    test_result = f"This is the execution output:{result.stdout}" if result.returncode == 0 else f"This is the execution output:{result.stdout}"+f"This is the error output:{result.stderr}"
                    if "Error" in test_result:
                        return False, f"This is the wrong execution output:{result.stdout}"
                    else:
                        if "print" not in code:
                            user_time = 0
                        return True, user_time
                except subprocess.TimeoutExpired:
                    return False, "timeout"
                except Exception as e:
                    return False, f"Error executing file: {str(e)}"

        def result_statistics(code_execution_result, compare_obj):
            case_state = {}
            for key, value in code_execution_result.items():
                tmp_pass_result = {}
                tmp_pass_flag = 0
                tmp_pass_time = 0
                for case_key, case_result in value.items():
                    if case_key not in case_state:
                        try:
                            if case_key == compare_obj:
                                continue
                            case_state[case_key] = {
                                'failed_reuslt': 0,
                                "case_value": case_result.get("case_item", ""),
                                "time": case_result.get("time", 0)
                            }
                        except Exception as e:
                            print(f"Error processing case_key {case_key}: {e}")
                            pass

                    if case_result.get('pass_result', False):
                        tmp_pass_flag += 1
                        tmp_pass_time += case_result.get('time', 0)
                    else:
                        case_state[case_key]['failed_reuslt'] += 1
                        case_state[case_key]['failed_reason'] = case_result.get('time', "unknown")

                tmp_pass_result['pass_result'] = tmp_pass_flag
                tmp_pass_result['average_time'] = tmp_pass_time / tmp_pass_flag if tmp_pass_flag > 0 else 0
                tmp_pass_result['total_time'] = tmp_pass_time
                code_execution_result[key]['result'] = tmp_pass_result

            return case_state, code_execution_result

        try:
            code_execution_result = {}
            for key, code_item in code_dict.items():
                code_execution_result[key] = {compare_obj: code_item}
                flag = 0
                for case_item in case:
                    code_execution_result[key][str(flag)] = {'case_item': case_item}
                    debug_case_item = re.sub(r"assert\s+(.*?)(==.*)",
                                             lambda m: f"#Output of the case '{case_item}' executing result:\r\nprint({m.group(1)})",
                                             case_item
                                             )
                    result = code_execution(self.TPL_RUN2 % (code_item, debug_case_item,case_item))
                    if isinstance(result, tuple) and len(result) == 2:
                        pass_result, exec_time = result
                    else:
                        pass_result, exec_time = False, "error"
                    code_execution_result[key][str(flag)]['pass_result'] = pass_result
                    code_execution_result[key][str(flag)]['time'] = exec_time
                    flag += 1

            case_state, code_execution_result = result_statistics(code_execution_result, compare_obj)

            #按照通过的解题数进行降序排序
            sorted_data = sorted(
                code_execution_result.items(),
                key=lambda item: float(item[1].get("result", {}).get("pass_result", 0)),
                reverse=True
            )

            return case_state, dict(sorted_data)
        except Exception as e:
            print("Error occurred:")
            print(traceback.format_exc())

    def evaluate_groundtruth(self,code,case,entry_point,num):
        def code_execution(code):
            with tempfile.NamedTemporaryFile(suffix=".py", delete=True, mode='w') as temp_file:
                temp_file.write(code)
                temp_file.flush()
                temp_file_path = temp_file.name
                try:
                    time_start = time.time()
                    result = subprocess.run(
                        ["python", temp_file_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    user_time = time.time() - time_start
                    test_result = f"This is the execution output:{result.stdout}" if result.returncode == 0 else f"This is the execution output:{result.stdout}"+f"This is the error output:{result.stderr}"
                    if "Error" in test_result:
                        return False, f"This is the wrong execution output:{test_result}"
                    else:
                        if "print" not in code:
                            user_time = 0
                        return True, user_time
                except subprocess.TimeoutExpired:
                    return False, "timeout"
                except Exception as e:
                    return False, f"Error executing file: {str(e)}"
        if "HumanEval" in num:
            result,reason = code_execution(self.TPL_RUN3 % (code,case,f"check({entry_point})"))
        elif "Mbpp" in num:
            result,reason = code_execution(self.TPL_RUN3 % (code,case,""))

        return result,reason

    def evaluate_case2(self,code,testcase,flag,outputcase):

        if not flag:
            code = self.TPL_RUN2 % (code,testcase)
        else:
            code = self.TPL_RUN2 % (code, "")
        # 使用subprocess来执行该代码文件
        # output_capture = io.StringIO()
        # sys.stdout = output_capture  # 重定向标准输出到StringIO
        # with self.lock:
        #     try:
        #         exec(code)  # 执行代码
        #         sys.stdout = sys.__stdout__  # 恢复标准输出
        #         test_result = output_capture.getvalue()  # 获取exec执行过程中产生的输出
        #
        #         if "Error" in test_result:
        #             return False, test_result, code
        #         else:
        #             if test_result.strip() in outputcase:  # 去除输出的前后空白符，进行对比
        #                 return True, None, code
        #             else:
        #                 return False, None, code
        #     except Exception as e:
        #         sys.stdout = sys.__stdout__  # 确保在异常时恢复标准输出
        #         return False, f"Error executing file: {str(e)}", code
        with tempfile.NamedTemporaryFile(suffix=".py", delete=True, mode='w') as temp_file:
            temp_file.write(code)
            temp_file.flush()
            temp_file_path = temp_file.name
        # 使用subprocess来执行该代码文件
            try:
                result = subprocess.run(
                    ["python", temp_file_path],
                    capture_output=True, text=True
                )
                test_result = result.stdout if result.returncode == 0 else result.stderr
                if "Error" in test_result:
                    return False, test_result, code
                else:
                    if test_result.strip() in outputcase:  # 去除输出的前后空白符，进行对比
                        return True,test_result, code
                    else:
                        return False, test_result, code
            except Exception as e:
                return False,f"Error executing file: {str(e)}",code


    def evaluate_easy2(self, code,num):
        try:
            # print(code)
            exec(code)
            # print(num)
            return True
        except Exception as e:
            # print(code)
            # error_message = traceback.format_exc()
            return False

def casecode_package(case):
    with open("../cache/backup/self_codegen_deepseek_coder_0923_0_votingcase_turn0_all_failed.json", "r") as f:
        data = json.loads(f.read())

    with open("../cache/0820_1/self_codegen_deepseek-coder_0913_4_turn0_all_sorted_list.json", "r") as f:
        data3 = json.loads(f.read())
    # for i in data.keys():
    #     print(i)
    TPL_RUN2 = """%s
%s
    """
    code_executed={}
    with open("../cache/case_check/self_codegen_0914_25.json","w+") as f:
        for i in range(0,164):
            try:
                if int(i) in [2, 23, 41, 45, 53, 60, 71, 92, 97, 99, 102, 123, 124, 135, 137, 138, 144, 148, 156, 157, 159, 160]:
                    pass
                else:
                    code_executed[str(i)] = {}
                    # print(data3[i][0])
                    tmp_code = data3[i][0]+"\n"
                    testcase = data[str(i)]['modified_test_case']

                    # testcase_all = "\n".join(testcase)
                    # print(testcase_all)
                    code_executed[str(i)]['code'] = TPL_RUN2 % (tmp_code, testcase)
                    # print("This is the code",code_executed)
                    # code_executed[i]['result'] = data3[i]['result']
                # print(code_executed[i]['code'])
            except Exception as e:
                print(i)
                # print(e)

        f.write(json.dumps(code_executed,indent=4))

# file_path = '../dataset/enamel.csv'
#
# casegen = CaseGenerator(file_path)
# casegen.generate_case_from_evalpuls()
