""" level 1 实现 Agent Loop
1、通过 langchain的 init_chat_model, 生成LLM实例
2、通过 langchain.tools模块的tool装饰器, 快速定义大模型的可用工具
3、通过 Tool.invoke 统一调用方法
"""
import os

from dotenv import load_dotenv

load_dotenv()

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langsmith import traceable

MAX_ITERATIONS = 10


@tool
def get_cur_weather(city: str) -> str:
    """获取指定城市的当前天气数据
    Args:
        city: 城市名称
    """
    return f"{city} 当前的天气为：多云转晴，气温 26度"


@tool
def get_product_price(product: Literal["平板", "手机", "耳机"]) -> float | None:
    """获取产品价格，如果查询的product不在产品目录中，则返回None
    Args:
        product: 产品名称
    """
    prices = {"平板": 1299.99, "手机": 1899.99, "耳机": 299.99}
    return prices.get(product, None)


@tool
def cal_discount(level: Literal["gold", "sliver", "bronze"], price: float) -> float:
    """根据折扣等级level 和 原价 price，计算折扣后的价格
    Args:
        level: 折扣等级，取值范围 Literal["gold", "sliver", "bronze"]
        price: 产品原价
    Returns:
        折扣后的价格
    """
    discounts = {"gold": 0.8, "sliver": 0.9, "bronze": 0.95}

    return discounts.get(level, 1) * price


@traceable(name="agent loop with tool calling")
def run_agent(query: str):
    print("Start execute agent loop")
    # 可用的工具集
    tools = [get_cur_weather, get_product_price, cal_discount]
    # 工具集字典
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(
        f"openai:qwen3.5:9b",
        base_url=os.getenv("OLLAMA_BASE_URL"),
        api_key=os.getenv("OLLAMA_API_KEY"),
    ).bind_tools(tools)

    messages = [
        SystemMessage(content="""
                你是一个购物智能助手，能够回答顾客的一些基本问题。你应该按照如下的规则来回答问题：
                1、在你回答问题时，先检查你能够使用的工具，然后根据问题推理出你的下一步action。
                2、优先使用可用的工具来回答用户的问题，如果可用的工具和问题不相关，则依靠你自己的知识尝试回答，
                但是注意，**不要杜撰**答案，如果你知不知道问题的答案，就直接回答不知道即可
                3、涉及到调用工具时，请**每次只调用单个工具**，不要多个工具同时调用
            """),
        HumanMessage(content=query),
    ]

    for turn in range(1, MAX_ITERATIONS + 1):
        print(f"第 {turn} 轮对话开始")
        ai_message = llm.invoke(messages)

        tools = ai_message.tool_calls

        # 检查是否有工具调用，如果没有，则直接输出答案
        if len(tools) == 0:
            print(f"Agent answer: {ai_message.content}")
            return ai_message.content

        # 只处理第一个工具调用
        tool = tools[0]
        tool_name = tool.get("name")
        tool_args = tool.get("args")
        tool_id = tool.get("id")

        # 获取工具
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(
                f"The tool: {tool_name}, with agrs: {tool_args}, is not found!"
            )

        print(f"Execute tool: {tool_name}, with agrs: {tool_args}")
        observation = tool_to_use.invoke(tool_args)
        print(f"Observation: {observation}")

        # 将ai message 和 tool message 追加到消息列表
        messages.append(ai_message)
        messages.append(ToolMessage(content=observation, tool_call_id=tool_id))

    print(f"超出最大迭代次数: {MAX_ITERATIONS}")
    return None


if __name__ == "__main__":
    print("Hello Agent Loop.")
    # query = "我想买个平板电脑，我想知道在gold的折扣等级下，折扣价是多少？"
    # query = "南京的今天的天气怎么样？"
    # query = "南京后天的天气怎么样？"
    query = "人工智能的发展，会导致很多普通职工失去工作吗？"
    run_agent(query=query)
