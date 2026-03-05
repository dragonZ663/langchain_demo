from dotenv import load_dotenv

load_dotenv()

import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from schema import AgentResponse


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


tools = [TavilySearch(), add, divide, multiply]
llm = ChatOpenAI(
    base_url=os.environ.get("CLOSEAI_BASE_URL"),
    api_key=os.environ.get("CLOSEAI_API_KEY"),
    model="gpt-3.5-turbo",
)

# llm = ChatOllama(model="gpt-oss:20b")

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-demo!")
    result = agent.invoke(
        # {"messages": [{"role": "user", "content": "计算 6 * 3 + 4 / 2 的结果"}]}
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
                }
            ]
        }
        
    )
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
