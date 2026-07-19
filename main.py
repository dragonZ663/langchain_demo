import os

from dotenv import load_dotenv

load_dotenv()
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pydantic import SecretStr

print("Initializing components...")

llm = ChatOpenAI(
    model="qwen3.5:9b",
    api_key=SecretStr(os.environ.get("OLLAMA_API_KEY", "")),
    base_url=os.environ.get("OLLAMA_BASE_URL"),
)

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
## Context
    {context}

## Task
    Question: {question}
    Provide a detailed answer:"""
)

vectorStore = PineconeVectorStore(
    embedding=embeddings, index_name=os.environ.get("INDEX_NAME")
)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# LCEL = LangChain Expression Language
# LangChain 表达式语言，是 LangChain 提供的一种声明式、用管道符 | 串联组件的编程方式。
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # 查询向量库
    docs = retriever.invoke(query)

    # 将返回的Chunk内容转为一个字符串
    context = format_docs(docs)

    # 生成完整提示词
    prompt = prompt_template.format_messages(context=context, question=query)

    # 调用大模型
    result_raw = llm.invoke(prompt)

    # 输出结果
    return result_raw.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    # RunnablePassthrough.assign(context=...) 的作用是：
    # "把输入 dict 原封不动地往下传，但在传之前，先把 context 字段的值算出来并塞进去"。
    # 其中 context 的值是通过 itemgetter("question") | retriever | format_docs 这条子链（提取问题 → 检索 → 格式化）计算得到的。
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "Image QA一般什么情况下会被调用？"

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 2: With LCEL")
    # print("=" * 70)
    # print("Why LCEL is better:")
    # print("- More concise and declarative")
    # print("- Built-in streaming: chain.stream()")
    # print("- Built-in async: chain.ainvoke()")
    # print("- Easy to compose with other chains")
    # print("- Better for production use")
    # print("=" * 70)

    # chain_with_lcel = create_retrieval_chain_with_lcel()
    # result_with_lcel = chain_with_lcel.invoke({"question": query})
    # print("\nAnswer:")
    # print(result_with_lcel)
