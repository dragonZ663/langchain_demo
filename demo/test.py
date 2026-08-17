from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="ollama:qwen3.5:9b", tools=[get_weather], system_prompt="你是一个有用的助手"
)

result = agent.invoke(
    input={"messages": [{"role": "user", "content": "南昌的天气怎么样？"}]}
)

print(result["messages"][-1].content_blocks)
