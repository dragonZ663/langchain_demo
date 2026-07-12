"""
main文件逻辑的模拟测试
"""

from dotenv import load_dotenv

load_dotenv()
import os

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langchain_tavily import TavilySearch

from schema import AgentResponse, Source

model = init_chat_model(
    "qwen3.5:9b",
    model_provider="openai",
    api_key=os.getenv("OLLAMA_API_KEY"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
)


@tool
def add(a: float, b: float) -> float:
    """
    返回 a 和 b 的加法运算结果

    Args
        a: 第一个加数
        b: 第二个加数
    """
    return a + b


@tool
def multiple(a: float, b: float) -> float:
    """
    返回 a 和 b 的乘法运算结果

    Args
        a: 第一个乘数
        b: 第二个乘数
    """
    return a * b


web_search = TavilySearch(max_results=3, topic="general")

tools = [add, multiple, web_search]


def extract_response(
    response: dict,
) -> AgentResponse:
    """
    安全提取结构化输出，如果 Agent 未自动产生则从最后一条消息回退解析。

    Args:
        response: agent.invoke() 的返回值

    Returns:
        始终返回 AgentResponse 实例
    """
    structured = response.get("structured_response")
    if structured is not None:
        # Agent 正确调用了结构化输出工具
        return structured

    # ===== 回退逻辑: 从最后一条有文本内容的AI消息中提取 =====
    messages = response.get("messages", [])
    # 逆序查找最后一条无tool_calls且有文本的AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            # AIMessage.content 可能是 str 或 list，统一转成 str
            raw = msg.content
            if isinstance(raw, list):
                parts: list[str] = []
                for chunk in raw:
                    if isinstance(chunk, dict):
                        parts.append(str(chunk.get("text", "")))
                    else:
                        parts.append(str(chunk))
                content = "".join(parts)
            else:
                content = raw

            # 尝试从markdown代码块中提取JSON
            import re

            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
            )
            if json_match:
                import json as _json

                try:
                    data = _json.loads(json_match.group(1))
                    return AgentResponse(
                        answer=data.get("answer", content),
                        sources=[Source(**s) for s in data.get("sources", [])],
                    )
                except (ValueError, TypeError):
                    pass
            # 纯文本回退
            return AgentResponse(answer=content, sources=[])

    return AgentResponse(answer="(未能生成答案)", sources=[])


def main():
    system_prompt = """
        你是一个智能助手，能够准确无误的回答用户的问题。
        <instruction>
        1、当问题涉及到数学运算时，优先查看是否有现成的工具可以使用，有的话优先使用工具，否则你自己进行推理计算
        2、当涉及的问题你不知道时，**绝对不要杜撰**，可以尝试查询网络上的知识，实在查不到的话，就说你不知道就行
    """
    agent = create_agent(
        model=model, tools=tools, response_format=ToolStrategy(AgentResponse)
    )
    response = agent.invoke(
        input={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "1 + 3 * 2 = ?"},
            ]
        }
    )

    structured_output = extract_response(response)
    print(structured_output.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
