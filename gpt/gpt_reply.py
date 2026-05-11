import os
from openai import OpenAI
class GPTReply:
    def __init__(self,model,client="openai"):
        self.model = model
        self.client = client

    def getreply(self,systemprompt,user1prompt,user2prompt):
        while True:
            try:
                if self.client=="openai":
                    # Qwen：走 OpenRouter，需设置环境变量 OPENROUTER_API_KEY
                    if "Qwen" in self.model:
                        client = OpenAI(
                            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                            base_url="https://openrouter.ai/api/v1"
                        )
                    else:
                        client = OpenAI(
                            api_key = os.getenv("OPENAI_API_KEY", ""),
                            base_url="https://api.openai.com/v1"
                        )
                # client = OpenAI()
                if self.model=="deepseek-coder":
                    self.model = "deepseek/deepseek-chat"
                completion = client.chat.completions.create(
                    model= self.model,
                    messages = [
                        {"role": "system","content":systemprompt+"_"},
                        {"role" : "user", "content": user1prompt+"_"},
                        {"role": "user", "content": user2prompt+"_"}
                    ],
                    temperature = 0
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"API Error: {type(e).__name__}: {e}")
                if "maximum context length is" in str(e):
                    return False
                if "Range of input length should" in str(e) or "Exceeded limit on max byt" in str(e):
                    return False
                pass
