""" level 3 实现 Agent Loop:
1、通过 prompt 定义工作模式，即Reasoning(Thought) -> Action -> Observation
2、工具也插入到 prompt中，写明：工具名，签名，主要功能。不再依赖tools传入(假设LLM不具备工具调用能力)
3、从response中提取工具执行信息，然后尝试执行
"""
""" ReAct prompt 实现 agent loop 技巧
1. Prompt 结尾没有引导词 — 你的 prompt 以 - Question: {question}. 结尾，模型不知道该从哪开始生成，需要以 - Thought: 结尾来引导
2. 没有 one-shot 示例 — Qwen 模型在没有具体示例的情况下，很难严格遵循 ReAct 格式
3. Observaton 拼写错误 — prompt 和 stop token 里都是 Observaton（少了个 i），但 scratchpad 里却写的是正确拼写 Observation:，会导致 stop token 无法匹配 scratchpad 中的内容
4. 全部塞在一条 user message — 没有利用 system/user 角色分离
5. Stop token 写法 — 原版 ReAct 用 \nObservation 作为 stop，这里没有换行符前缀

"""
import os

from dotenv import load_dotenv

load_dotenv()

import inspect
import re

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

    return discounts.get(level, 1) * float(price)

tools = {
    "get_cur_weather": get_cur_weather,
    "get_product_price": get_product_price,
    "cal_discount": cal_discount
}

# 获取工具的描述信息
def get_tool_description(tools_dict: dict) -> str:
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_description(tools)
print(tool_descriptions)
tool_names = [key for key in tools.keys()]

@traceable(run_type="llm", name="Ollama Chat")
def ollama_chat(messages: list[ollama.Message]):
    return ollama.chat(
        model="qwen3.5:9b",
        messages=messages,
        options={
            "stop": ["Observation:"],
            "temperature": 0
        }
    )

# system prompt: 规则 + 工具 + 格式 + 示例
system_prompt = f"""
你是一名专业的导购助手，擅长回答用户的相关产品咨询问题。

## 硬性规则
- 优先使用工具的能力回答用户的问题，不能随意假设、猜想答案。
- 如果当前可用的工具以及你自己的知识储备无法回答用户的问题，请直接回答不知道即可。
- 不要随意猜测产品的价格或者打折后的价格，所有产品价格均以工具执行结果为准。

## 可用工具
{tool_descriptions}

## 工作流格式（严格遵守）
你必须严格按照以下格式输出，每次只能输出一个 Thought/Action/ActionArgs 组合：

Question: 用户问题的描述
Thought: 你的推理过程，分析当前步骤应该做什么、用什么工具
Action: 工具名称，只能是 [{", ".join(tool_names)}] 之一
ActionArgs: 工具参数，格式 key=value，多个参数用英文逗号分隔

以上 Thought → Action → ActionArgs 可循环 N 次，每次系统会填充 Observation。
当信息足够回答用户时，输出：
FinalAnswer: 你的最终答案

## 示例
Question: 我想查一下南京的天气
Thought: 用户想查询南京的天气，我需要调用 get_cur_weather 工具。
Action: get_cur_weather
ActionArgs: city=南京
Observation: 南京 当前的天气为：多云转晴，气温 26度
Thought: 已经获取到天气数据，可以给出最终答案。
FinalAnswer: 南京当前的天气为多云转晴，气温 26 度。

---

现在开始处理用户问题。记住：严格遵循格式，不要跳过 Thought 直接输出 Action！
"""

# user prompt: 以 Thought: 结尾，引导模型开始推理
user_prompt = """Question: {question}
Thought:"""


@traceable(name="agent loop by raw react prompt")
def run_agent(query: str):
    print("Start execute agent loop")

    # 修复: messages 在循环外初始化，每轮往里面追加，完整保留对话历史
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(question=query)}
    ]

    for turn in range(1, MAX_ITERATIONS + 1):
        print(f"\n===== 第 {turn} 轮对话 =====")
        print(f"[DEBUG] 当前 messages 数量: {len(messages)}")

        response = ollama_chat(messages)
        ai_message = response.message
        output = ai_message.content
        print(f"[LLM Output]:\n{output}")

        if not output or not output.strip():
            print("[WARNING] LLM 返回了空内容！可能是模型未遵循 ReAct 格式。")
            print(f"[DEBUG] 完整 response 对象: {response}")
            break

        # 把 assistant 的输出追加到对话历史
        messages.append({"role": "assistant", "content": output})

        final_answer_match = re.search(r"FinalAnswer:\s*(.+)", output, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"The Final Answer is:\n{final_answer}")
            return final_answer

        print("Start parsing Action and ActionArgs")
        tool_name_match = re.search(r"Action:\s*(.+)", output)
        tool_args_match = re.search(r"ActionArgs:\s*(.+)", output)

        if not tool_name_match or not tool_args_match:
            print(f"[Error] failed to parse Action/ActionArgs from LLM output")
            print(f"[DEBUG] 模型没有按照格式输出 Action 和 ActionArgs")
            break

        tool_name = tool_name_match.group(1).strip()
        tool_input_raw = tool_args_match.group(1).strip()

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        if tool_name not in tool_names:
            print(f"[Error] Unknown tool: {tool_name}")
            return

        print(f"Selected Tool: {tool_name}, with args: {args}")

        func = tools.get(tool_name)
        observation = func(*args) if func else ""
        print(f"Observation: {observation}")

        # 把 Observation 作为 user message 追加，并用 Thought: 引导下一轮推理
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}\nThought:"
        })

    print(f"超出最大迭代次数: {MAX_ITERATIONS}")
    return None


if __name__ == "__main__":
    print("Hello Agent Loop.")
    # query = "我想买个平板电脑，我想知道在gold的折扣等级下，折扣价是多少？"
    # query = "南京的今天的天气怎么样？"
    # query = "南京后天的天气怎么样？"
    query = "人工智能的发展，会导致很多普通职工失去工作吗？"
    run_agent(query=query)
