# Agent Loop 三层实现分析

三个文件实现的是**同一个模式**（Reasoning → Action → Observation 循环），但使用了**从高级到低级**的不同技术栈，构建了一条清晰的"剥洋葱"路径。

---

## 🟢 Level 1 — LangChain 封装层

**文件：** `1_agent_loop_langchain_tool_calling.py`

| 维度 | 说明 |
|------|------|
| **LLM 调用** | `langchain.chat_models.init_chat_model` + `.bind_tools(tools)` |
| **工具定义** | `@tool` 装饰器 — 一行代码，自动生成 JSON Schema |
| **消息格式** | `SystemMessage` / `HumanMessage` / `ToolMessage` 对象 |
| **工具调用解析** | `ai_message.tool_calls` — LangChain 自动解析好的结构体，直接 `tool.get("name")` |
| **工具执行** | `tool_to_use.invoke(tool_args)` — 统一接口 |

### Agent Loop 核心逻辑

```python
ai_message = llm.invoke(messages)          # LangChain 内部完成 chat → 解析 tool_calls
tools = ai_message.tool_calls              # 结构化数据，开箱即用
observation = tool_to_use.invoke(tool_args) # 统一 .invoke()
messages.append(ai_message)                # 用 Message 对象追加
messages.append(ToolMessage(...))          # LangChain 封装好的消息类型
```

**抽象程度：最高**。开发者无需关心 tool schema 的 JSON 格式、响应解析、消息序列化等细节，LangChain 全包了。

---

## 🟡 Level 2 — 原生 Function Calling 层

**文件：** `2_agent_loop_raw_function_calling.py`

| 维度 | 说明 |
|------|------|
| **LLM 调用** | 直接使用 `ollama.chat(model=..., messages=..., tools=tools)` |
| **工具定义** | **手写** JSON Function Schema（`type: "function"`, `parameters` 等） |
| **消息格式** | 原生 Python `dict` — `{"role": "system", "content": ...}` |
| **工具调用解析** | `tool.function.name` / `tool.function.arguments` — 需要理解 Ollama 响应结构 |
| **工具执行** | 原生 Python `tool_to_use(**tool_args)` |

### Agent Loop 核心逻辑

```python
response = ollama.chat(model=..., messages=messages, tools=tools)  # 原生 SDK
tools = ai_message.tool_calls       # 仍走 function calling 协议，返回结构化数据
tool_name = tool.function.name      # ⚠️ 区别: 不能再用 .get("name")
observation = tool_to_use(**tool_args) # 原生 Python 调用
messages.append(ai_message.model_dump()) # 手动序列化
messages.append({"role": "tool", ...})   # 手动构造消息字典
```

**抽象程度：中等**。去掉了 LangChain 这个中间层，直接面对 Ollama SDK 和 OpenAI 兼容的 function calling 协议。手动写 JSON Schema 的过程揭示了 `@tool` 装饰器的底层原理。

---

## 🔴 Level 3 — 纯 ReAct Prompt 层（最底层）

**文件：** `3_raw_react_prompt.py`

| 维度 | 说明 |
|------|------|
| **LLM 调用** | `ollama.chat(messages=..., options={"stop": ["Observation:"]})` — **不传 `tools` 参数** |
| **工具定义** | 工具的签名和描述直接**拼进 system prompt 文本**中 |
| **消息格式** | 原生 Python `dict` |
| **工具调用解析** | **正则表达式** `re.search(r"Action:\s*(.+)", output)` — 从纯文本中提取 |
| **工具执行** | 原生 Python `func(*args)` |
| **循环控制** | 完全依赖 **prompt 工程**（格式约束 + one-shot 示例 + stop token） |

### Agent Loop 核心逻辑

```python
# ⚠️ 没有 tools 参数！Agent 的所有行为全靠 prompt 驱动
response = ollama.chat(messages=..., options={"stop": ["Observation:"]})
output = response.message.content         # 纯文本，没有 tool_calls 字段
tool_name = re.search(r"Action:\s*(.+)", output)    # 用正则从文本中扒工具名
args = re.search(r"ActionArgs:\s*(.+)", output)     # 用正则从文本中扒参数
observation = func(*args)                 # 原生 Python 调用
messages.append({"role": "assistant", "content": output})
messages.append({"role": "user", "content": f"Observation: {observation}\nThought:"})  # 观察结果也是纯文本填入
```

**抽象程度：最底层**。连 function calling 协议都不用了。LLM 完全不知道自己是 Agent — 所有的"智能体行为"都来自 prompt 中的格式约束 + 正则解析。这就是 **ReAct 论文**最原始的思路。

---

## 三层对比一览

```
┌─────────────────────────────────────────────────────────┐
│ Level 1: LangChain 封装                                  │
│   @tool → bind_tools → .invoke() → ToolMessage          │
│   开发者心智负担：★☆☆☆☆                                  │
├─────────────────────────────────────────────────────────┤
│ Level 2: 原生 Function Calling                           │
│   手写 JSON Schema → ollama.chat(tools=) → 原生调用       │
│   揭示了 Level 1 底层在做什么                              │
│   开发者心智负担：★★★☆☆                                  │
├─────────────────────────────────────────────────────────┤
│ Level 3: 纯 ReAct Prompt                                 │
│   Prompt 文本描述工具 → 正则解析输出 → stop token 控制     │
│   LLM 不知道自己是个 Agent，全靠 prompt 工程              │
│   开发者心智负担：★★★★★                                  │
└─────────────────────────────────────────────────────────┘
```

## 学习路径总结

这条路径非常有教学价值：

- **Level 1 → Level 2**：揭示了 LangChain 的 `@tool` + `bind_tools` 本质上就是生成 JSON Schema 并解析响应。
- **Level 2 → Level 3**：揭示了 function calling 本质上不过是一个更可靠的结构化输出机制，而 ReAct 模式的核心思想（Reasoning → Action → Observation 循环）在 function calling 出现之前就已经存在，仅靠 prompt + 文本解析就能实现。
