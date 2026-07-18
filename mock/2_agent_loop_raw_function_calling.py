""" level 2 实现 Agent Loop:
在 1_agent_loop_langchain_tool_calling.py 的基础上, 不借助langchain, 实现原始的工具调用
1、通过 Ollama.chat, 调用LLM, 返回对应的Response
2、通过编写 function json schema, 把可用方法传给大模型
3、通过 tool_to_use(**kw)语法，原生调用工具
"""
import os

from dotenv import load_dotenv

load_dotenv()

import ollama
from typing import Literal
from langsmith import traceable

MAX_ITERATIONS = 10

# traceable是用来使用 langsmith 跟踪的
@traceable(run_type="tool", name="get_cur_weather")
def get_cur_weather(city: str) -> str:
    """获取指定城市的当前天气数据
    Args:
        city: 城市名称
    """
    return f"{city} 当前的天气为：多云转晴，气温 26度"


@traceable(run_type="tool", name="get_product_price")
def get_product_price(product: Literal["平板", "手机", "耳机"]) -> float | None:
    """获取产品价格，如果查询的product不在产品目录中，则返回None
    Args:
        product: 产品名称，可选值："平板", "手机", "耳机"
    """
    prices = {"平板": 1299.99, "手机": 1899.99, "耳机": 299.99}
    return prices.get(product, None)


@traceable(run_type="tool", name="cal_discount")
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
# Function JSON schema 定义，LLM 就是根据这种格式来理解可用的工具
# langchain 的 @tool 装饰器，其实就是干的这件事
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_cur_weather",
            "description":"获取指定城市的当前天气数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "获取产品价格(float类型)，如果查询的product不在产品目录中，则返回None。可选的产品：['平板', '手机', '耳机']",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "产品名称，可选的产品：['平板', '手机', '耳机']",
                        "enum": ['平板', '手机', '耳机']
                    },
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cal_discount",
            "description": "根据折扣等级level 和 原价 price，计算折扣后的价格。",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "description": "折扣等级",
                        "enum": ['gold', 'sliver', 'bronze']
                    },
                    "price": {
                        "type": "number",
                        "description": "产品折扣前的原价"
                    },
                },
                "required": ["level", "price"]
            }
        }
    }
]

# Ollama 也能够自动生成上述的schemas，前提是函数注释需要满足 Google docstring 格式
tools_with_docstring = [get_cur_weather, get_product_price, cal_discount]

@traceable(run_type="llm", name="Ollama Chat")
def ollama_chat(messages: list[ollama.Message]):
    return ollama.chat(model="qwen3.5:9b", messages=messages, tools=tools)


@traceable(name="agent loop with tool calling")
def run_agent(query: str):
    print("Start execute agent loop")
    
    # Difference：函数没有name，不能使用列表生成式了
    tools_dict = {
        "get_cur_weather": get_cur_weather,
        "get_product_price": get_product_price,
        "cal_discount": cal_discount
    }

    messages = [
        {
            "role": "system",
            "content": """
                你是一个购物智能助手，能够回答顾客的一些基本问题。你应该按照如下的规则来回答问题：
                1、在你回答问题时，先检查你能够使用的工具，然后根据问题推理出你的下一步action。
                2、优先使用可用的工具来回答用户的问题，如果可用的工具和问题不相关，则依靠你自己的知识尝试回答，
                但是注意，**不要杜撰**答案，如果你知不知道问题的答案，就直接回答不知道即可
                3、涉及到调用工具时，请**每次只调用单个工具**，不要多个工具同时调用
            """
        },
        {
            "role": "user", "content": query
        },
    ]

    for turn in range(1, MAX_ITERATIONS + 1):
        print(f"第 {turn} 轮对话开始")
        response = ollama_chat(messages)
        ai_message = response.message

        tools = ai_message.tool_calls

        # 检查是否有工具调用，如果没有，则直接输出答案
        if not tools:
            print(f"Agent answer: {ai_message.content}")
            return ai_message.content

        # 只处理第一个工具调用
        tool = tools[0]
        # Difference，不能再使用get('name'), get('args')获取工具名和参数了
        tool_name = tool.function.name
        tool_args = tool.function.arguments
        print(f"Selected Tool: {tool_name}, with args: {tool_args}")

        # 获取工具
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(
                f"The tool: {tool_name}, with agrs: {tool_args}, is not found!"
            )

        print(f"Execute tool: {tool_name}, with agrs: {tool_args}")
        observation = tool_to_use(**tool_args)
        print(f"Observation: {observation}")

        # 将ai message 和 tool message 追加到消息列表
        messages.append(ai_message.model_dump())
        messages.append(
            {
                "role": "tool",
                "content": str(observation), 
            }
        )

    print(f"超出最大迭代次数: {MAX_ITERATIONS}")
    return None


if __name__ == "__main__":
    print("Hello Agent Loop.")
    query = "我想买个平板电脑，我想知道在gold的折扣等级下，折扣价是多少？"
    # query = "南京的今天的天气怎么样？"
    # query = "南京后天的天气怎么样？"
    # query = "人工智能的发展，会导致很多普通职工失去工作吗？"
    run_agent(query=query)
