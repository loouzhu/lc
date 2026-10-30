import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
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

# response = model.stream("你是谁")
# for chunk in response:
#     print(chunk.content, end="", flush=True)

# agent = create_agent(model="deepseek-chat")
# messages = agent.stream(
#     {
#         "messages": [
#             {"role": "system", "content": "你是一个AI助手"},
#             {"role": "user", "content": "北京今天天气如何？"},
#         ]
#     },
#     stream_mode="messages",
# )

# for token,metadata in messages:
#     if token.content:
#         print(token.content,end='',flush=True)


@tool
def get_weather(location: str) -> str:
    """
    Get the weather in the given location
    Args:  
        location: city name or coordinates
    """
    return f"Current weather in {location} is sunny"


agent = create_agent(model="deepseek-chat", tools=[get_weather])

response = agent.invoke(
    {
        "messages": [
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": "北京今天天气如何？"},
            AIMessage(),
        ],
    }
)

for message in response["messages"]:
    message.pretty_print()
