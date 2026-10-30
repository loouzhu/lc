import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool

load_dotenv()

base_url = os.environ["DEEPSEEK_BASE_URL"]
api_key = os.environ["DEEPSEEK_API_KEY"]

model = init_chat_model(
    model="deepseek-flash",
    model_provider="deepseek",
    base_url=base_url,
    api_key=api_key,
)

system_prompt = """
# 身份
你是一个宠物专家，了解如常生活中各式各样的宠物

# 任务
用户会给你一些关于宠物的问题，你根据问题给出回答即可
"""

agent = create_agent(
    model=model,
    system_prompt=system_prompt,
)

for token, metadata in agent.stream(
    {
        "messages": [
            HumanMessage(content="我最喜欢小猫"),
        ],
    },
    version="v2",
    stream_mode="messages",
):
    print(token.text, end="", flush=True)
