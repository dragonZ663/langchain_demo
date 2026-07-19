# Image QA & Self Reflection 工具分析

> 基于 `webwright` 源码和配置文件的讨论总结。

---

## 一、概述

Webwright 项目中，LLM Agent 通过两种工具来理解页面内容和验证执行结果：

| 工具 | 定位 | 使用阶段 | 调用方式 |
|---|---|---|---|
| `image_qa` | 可选辅助 | 探索阶段 | Agent **自主决定**通过 bash 调用 CLI |
| `self_reflection` | 必需裁判 | 最终执行后 | Agent **必须**在声明完成前调用 CLI |

两者都依赖 LLM 的**多模态（图片识别）能力**，但互补作用不同。

---

## 二、Image QA 工具

### 2.1 文件位置

`src/webwright/tools/image_qa.py`

### 2.2 作用

对一张或多张截图提出视觉问题，返回结构化的 JSON 答案。用于探索阶段**验证 UI 状态**，例如"这个筛选条件是否已选中？"。

### 2.3 CLI 参数

```bash
python -m webwright.tools.image_qa \
  --workspace-dir "{{ workspace_dir }}" \
  --image screenshots/explore.png \
  --question "Is the BMW filter chip visibly selected?"
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `--image` | ✅ 是 | 图片路径，可重复使用（`action="append"`） |
| `--question` | ✅ 是 | 要向图片提出的视觉问题 |
| `--workspace-dir` | ❌ 否 | 相对路径的基准目录（默认当前目录） |
| `--model-config` | ❌ 否 | 模型配置文件路径（默认读取快照配置） |
| `--timeout-seconds` | ❌ 否 | HTTP 请求超时（默认 60s） |

### 2.4 返回值格式

```json
{
  "image_path": "screenshots/explore.png",
  "question": "Is the BMW filter chip visibly selected?",
  "answer": "Yes, the BMW filter chip is highlighted in blue.",
  "evidence": ["The chip has a blue background with a checkmark icon"],
  "unknown": false,
  "confidence": 0.95
}
```

### 2.5 触发机制

`image_qa` **不会**被任何 Python 代码程序化地自动调用。它是通过系统提示词（`base.yaml` 等配置文件）告知 LLM Agent 这个工具的存在和用法，由 Agent **自主决定**在适当的时机通过 bash 执行 CLI：

```
base.yaml 等配置文件
  └─ 提示词模板中描述：
       "Use image_qa during exploration to inspect screenshots
        and verify UI state:
        python -m webwright.tools.image_qa ..."
       │
       ▼
  LLM Agent 读取提示词
       │
       ▼
  Agent 自主判断时机（如刚截完图）
       │
       ▼
  Agent 输出 bash 命令：
  python -m webwright.tools.image_qa --image screenshots/... --question "..."
       │
       ▼
  读取 stdout 的 JSON → 决定下一步
```

这种"提示词驱动"的设计让 Agent 保持灵活性，而非在代码里写死调用逻辑。

### 2.6 图片传递方式

和 `self_reflection` 相同——通过 **Base64 编码**后以内联数据 URL 传递（见下文第四章）。

---

## 三、Self Reflection 工具

### 3.1 文件位置

`src/webwright/tools/self_reflection.py`

### 3.2 作用

一个**两阶段截图裁判工具**，用于自动评估 Agent 最终执行结果的正确性。**必须在声明完成之前运行**，且必须输出 `predicted_label: 1`（PASS）。

### 3.3 CLI 参数

```bash
python -m webwright.tools.self_reflection \
  --config {{ workspace_dir }}/self_reflect_config.json \
  --workspace-dir "{{ workspace_dir }}" \
  --output {{ workspace_dir }}/final_runs/run_<id>/self_reflect_result.json
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | **必填** | JSON 配置文件路径（或 `-` 表示从 stdin 读取） |
| `--workspace-dir` | 当前目录 | 相对路径的基准目录 |
| `--output` | stdout | 结果 JSON 的输出路径 |
| `--auto-latest-run` | `final_runs` | 自动发现最新运行截图的根目录 |
| `--max-image-parse-retries` | 3 | 逐图评分最大重试次数 |
| `--image-max-new-tokens` | 1024 | 阶段 1 模型响应最大 token 数 |
| `--final-max-new-tokens` | 8192 | 阶段 2 模型响应最大 token 数 |
| `--model-config` | 自动读取 | 模型配置文件路径 |
| `--timeout-seconds` | 120 | HTTP 请求超时 |

### 3.4 两阶段流程

```
CLI 入口 main()
  │
  ├─ 1. 读取 --config JSON 配置文件
  ├─ 2. 解析图片路径（或自动从 final_runs/ 发现最新截图）
  ├─ 3. 加载操作日志 final_script_log.txt
  └─ 4. 调用两阶段裁判
       │
       ├── 阶段 1：逐图评分（并发）
       │     ┌─────────────────────────────────┐
       │     │ 每张截图 → (system + user+image) │
       │     │ 并行调用 asyncio.gather           │
       │     │ 解析 Score: 1-5 + Reasoning       │
       │     │ 失败自动重试最多 3 次              │
       │     └──────────────┬──────────────────┘
       │                    ▼
       │      输出: [{image_path, Score, Reasoning, ParseFailed}, ...]
       │
       └── 阶段 2：最终汇总裁判
             ┌──────────────────────────────────┐
             │ {image_reasonings} ← 所有 Reasoning 拼接 │
             │ {action_history_log} ← 操作日志     │
             │ 填充到 final_verdict_user_prompt    │
             │ + 附上所有截图 → 单次模型调用        │
             │ 解析 Status: success / failure       │
             └──────────────┬───────────────────┘
                           ▼
              predicted_label: 1 (PASS) / 0 (FAIL) / null (未解析)
```

### 3.5 配置文件结构

`self_reflect_config.json` 由 Agent 在规划阶段编写一次，后续复用：

```json
{
  "image_judge_system_prompt": "你是一个严格的评估者...",
  "image_judge_user_prompt": "任务描述和关键检查点列表...",
  "final_verdict_system_prompt": "你是汇总裁判，以 Status: success/failure 结尾...",
  "final_verdict_user_prompt": "任务描述 + {image_reasonings} + {action_history_log}..."
}
```

也支持通过 `_file` 后缀指向外部文本文件（如 `image_judge_user_prompt_file`），避免内联时花括号转义问题。

### 3.6 自动截图发现

如果配置中没提供 `images` 列表，工具自动扫描 `<workspace-dir>/final_runs/` 下**编号最高**且包含截图的 `run_<id>/screenshots/` 目录，按 `final_execution_<数字>_<描述>.png` 排序。

---

## 四、图片传递机制（两者通用）

无论是 `image_qa` 还是 `self_reflection`，图片都是通过 **Base64 编码 + 数据 URL** 传递给 LLM：

```
图片文件（PNG/JPEG/WebP）
  │
  ├─ read_bytes() → 二进制数据
  ├─ base64.b64encode() → ASCII 字符串
  └─ 包装为：
       {
         "type": "input_image",
         "image_url": "data:image/png;base64,iVBORw0KGgo...",
         "detail": "high"
       }
```

`detail: "high"` 表示高细节模式，对视觉精度要求高但消耗更多 token。

**两种情况传递方式不同：**

| | 每次调用携带图片数 | 并发 |
|---|---|---|
| `image_qa` | 1 张（单图）或多张 | 无 |
| `self_reflection` 阶段 1 | **1 张** | 多图异步并发处理 |
| `self_reflection` 阶段 2 | **所有图片一次性** | 单次调用 |

---

## 五、页面理解的三种手段

Agent 在探索阶段理解网页有三种途径：

| 手段 | 说明 | 多模态依赖 | 使用时机 |
|---|---|---|---|
| **ARIA 树** | `aria_snapshot()` 打印无障碍树 | ❌ 不依赖 | 每次 Playwright 脚本都会输出 |
| **`image_qa`** | 对截图提出视觉问题 | ✅ 依赖 | 可选，Agent 自主判断是否需要 |
| **直接提取** | 打印 URL、标题、可见标签、提取文本 | ❌ 不依赖 | 每次脚本都会输出 |

探索阶段**以 ARIA 树为主**，截图验证为辅；最终验证阶段 **`self_reflection` 强制依赖多模态**。

---

## 六、对 LLM 多模态能力的要求

| 场景 | 是否依赖图片识别 | 说明 |
|---|---|---|
| 探索阶段理解页面 | ❌ **不必须** | ARIA 树 + 文本提取已足够 |
| 探索阶段 UI 状态确认 | ⚠️ 可选 | `image_qa` 是增强手段 |
| 最终执行验证 | ✅ **必须** | `self_reflection` 两阶段都依赖看图 |

所以结论：

- **纯文本模型**（如 GPT-3.5、Claude 3 Haiku）可以在探索阶段工作，但 `self_reflection` 环节会完全失效
- **多模态模型**（如 Claude 4/3.5 Sonnet、GPT-4o）才能完整运行整个流程
- 这也是项目配置文件默认使用 `model_claude.yaml` / `model_openai.yaml` 的原因——它们指向的都是多模态模型

---

## 七、两个工具的对比总结

| 维度 | `image_qa` | `self_reflection` |
|---|---|---|
| **本质** | 视觉问答工具 | 两阶段截图裁判系统 |
| **调用方** | Agent 自主通过 bash 调用 | Agent 在完成检查门前强制调用 |
| **触发方式** | 提示词描述 → Agent 自主决策 | 提示词描述 → Agent 必须执行 |
| **阶段** | 单次调用 | 两阶段（逐图评分 + 汇总裁判） |
| **并发** | 无 | 阶段 1 并发处理所有截图 |
| **输出** | `answer` / `evidence` / `confidence` | `predicted_label: 1/0/null` |
| **失败重试** | 无 | 阶段 1 最多重试 3 次 |
| **多模态依赖** | 是 | 是 |
| **代码中是否有硬编码调用** | ❌ 无 | ❌ 无 |
| **使用场景示例** | "这个 BMW 筛选器图标是否选中？" | "这批截图和操作日志是否满足所有关键检查点？" |
