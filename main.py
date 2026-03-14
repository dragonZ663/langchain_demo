import os

from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

print("Initializing components...")

llm = ChatOpenAI(
    model="qwen/qwen3.5-9b",
    api_key=os.environ.get("LM_STUDIO_API_KEY"),
    base_url=os.environ.get("LM_STUDIO_BASE_URL")
)

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

prompt_template=ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""
)

vectorStore = PineconeVectorStore(embedding=embeddings, index_name=os.environ.get("INDEX_NAME"))
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join([doc.page_content  for doc in docs])

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

if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    # print("=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer:")
    # print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)
