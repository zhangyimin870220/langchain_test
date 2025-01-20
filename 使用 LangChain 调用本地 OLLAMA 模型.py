import os
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 设置OLLAMA模型的API URL和密钥
ollama_api_url = "http://localhost:11434"
ollama_key = "sk-7800dc8fded44016b70814bf80f4c78f"

# 配置OpenAI客户端以使用OLLAMA模型
openai.api_base = ollama_api_url
openai.api_key = ollama_key


# 定义一个函数来调用OLLAMA模型
def call_ollama_model(prompt, functions=None, function_call="auto"):
    response = openai.ChatCompletion.create(
        model="ollama-model-name",  # 替换为您的OLLAMA模型名称
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        function_call=function_call
    )
    return response


# 定义一个简单的函数调用示例
functions = [
    {
        "name": "calculate_average",
        "description": "Calculate the average of a list of numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            },
            "required": ["numbers"]
        }
    }
]

# 创建一个PromptTemplate
template = """You are a helpful assistant. Answer the user's question."""
prompt_template = PromptTemplate(template=template)

# 创建一个LLMChain
llm = OpenAI(client=openai, streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
chain = LLMChain(llm=llm, prompt=prompt_template)

# 示例提示
user_prompt = "Calculate the average of the following numbers: 10, 20, 30."

# 调用OLLAMA模型
response = call_ollama_model(user_prompt, functions=functions)

# 处理响应
if response.choices[0].get("message").get("function_call"):
    function_name = response.choices[0]["message"]["function_call"]["name"]
    function_args = eval(response.choices[0]["message"]["function_call"]["arguments"])

    if function_name == "calculate_average":
        result = sum(function_args["numbers"]) / len(function_args["numbers"])
        print(f"The average is: {result}")
else:
    print(response.choices[0]["message"]["content"])

if __name__ == '__main__':
    call_ollama_model(user_prompt, functions=functions)
