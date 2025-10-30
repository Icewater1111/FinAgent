import os
from dotenv import load_dotenv
load_dotenv(override=True)

DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
print(DeepSeek_API_KEY)

import gradio as gr
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = init_chat_model(model="deepseek-chat", model_provider="deepseek")

system_prompt = ChatPromptTemplate.from_messages([
    ("system", "你叫小黑, 是一名乐于助人的助手。"),
    ("human", "{input}")
])

basic_qa_chain = system_prompt | model | StrOutputParser()

async def chat_response(message, history):
    partial_message = ""

    async for chunk in basic_qa_chain.astream({"input" : message}):
        partial_message += chunk
        yield partial_message

#
# question = "你好, 请你介绍一下自己。"
#
# # result = model.invoke(question)
# result = basic_qa_chain.invoke(question)
#
# print(result)


# from langchain_community.chat_models.tongyi import ChatTongyi
#
# model = ChatTongyi()
#
#
# question = "你好, 请你介绍一下自己。"
#
# result = model.invoke(question)

# print(result)









# import os
# from dotenv import load_dotenv
# load_dotenv(override=True)
#
# DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# print(DeepSeek_API_KEY)

# from openai import OpenAI
#
# client = OpenAI(api_key=DeepSeek_API_KEY, base_url="https://api.deepseek.com")
#
# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content" : "你是乐于助人的助手，请根据用户的问题给出回答"},
#         {"role": "user", "content": "你好, 请你介绍一下自己。"},
#     ],
# )
#
# print(response.choices[0].message.content)