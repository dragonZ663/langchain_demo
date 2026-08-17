from dotenv import load_dotenv

load_dotenv()
from urllib.error import URLError
from urllib.request import Request, urlopen
import http.client
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = """ 你是一个文学助理
## 能力
- fetch_text_from_url: 将文档文本从URL加载到对话中。
不要猜测行数或位置——要根据保存文件中的工具结果来判断。
"""


@tool
def fetch_text_from_url(url: str) -> str:
    """从url 查询文档内容"""
    req = Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"}
    )

    for attempt in range(3):
        try:
            with urlopen(req, timeout=120) as resp:
                raw = resp.read()
            return raw.decode("utf-8", errors="replace")
        except http.client.IncompleteRead as e:
            if attempt == 2:
                return f"Fetch failed (incomplete): {len(e.partial)} bytes"
        except URLError as e:
            return f"Fetch failed: {e}"

    return "Fetch failed"


model = init_chat_model(
    model="openai:deepseek-v4-flash", 
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.5,
    max_tokens=1024
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = f""" 古腾堡计划托管了F·斯科特·菲茨杰拉德的《了不起的盖茨比》的完整纯文本版本。
URL：https://www.gutenberg.org/files/64317/64317-0.txt

请尽可能回答以下问题：

1) 在完整的古腾堡文件中，有多少行包含子字符串“Gatsby”（统计行数，而非行内出现次数，每行以换行符结尾）。
2) 文件中第一行包含“Daisy”的行号是多少（从1开始计数）。
3) 用两句话概括原文。

请尽力回答问题 (1) 和 (2)。如果您发现无法使用现有工具和推理方法**验证**某个确切答案，请不要随意编造数字：在该字段中使用“null”，并在“how_you_computed_counts”中详细说明您的限制。
如果遇到任何错误，请报告错误内容和错误信息。
"""

agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "great-gatsby-lc"}},
)

print(agent_result["messages"][-1].content_blocks)
