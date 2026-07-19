import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def embed(
    file_path: str,
    *,
    embedding_model_name: str | None = None,
    index_name: str | None = None,
    chunk_size=512,
    chunk_overlap=0,
):
    """ Markdown 文本向量化，分块策略：  
    1、第一层, 根据Markdown的header进行最多三级header分块  
    2、第二层, 在第一层的基础上, 进行文本结构的分块, 同时限制chunk_size 和 chunk_overlap

    Args:
        - file_path 文件路径，支持绝对 或 相对路径
        - embedding_model_name 向量化模型名称，如果未提供，则优先查找环境变量 EMBEDDINGS_MDOEL_NAME，\
        如果环境变量未找到，则使用默认模型：qwen3-embedding:0.6b
        - index_name 向量数据库的名称，如果未提供，则优先查找环境变量 INDEX_NAME，如果环境变量未到找，\
        则使用默认值：COMMON_STORE
        - chunk_size 单个chunk最大字符数，默认 512
        - chunk_overlap 前后chunk之间的重叠数，默认 0
    """
    if not embedding_model_name:
        load_dotenv()
        embedding_model_name = (
            os.getenv("EMBEDDINGS_MDOEL_NAME") or "qwen3-embedding:0.6b"
        )

    if not index_name:
        load_dotenv()
        index_name = os.getenv("INDEX_NAME") or "COMMON_STORE"

    print("Start Markdown Ingesting...")
    # 读取文档 — 用 TextLoader 保留原始 Markdown 语法（# ## ### 等标题符号）
    # 这样 MarkdownHeaderTextSplitter 才能正确识别和分割
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"read *{len(documents)}* documents")

    # 进行文档分块
    print("splitting...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 先进行 markdown 结构的分块
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    all_md_chunks: list[Document] = []
    for doc in documents:
        md_splits = markdown_splitter.split_text(doc.page_content)

        for chunk in md_splits:
            all_md_chunks.append(chunk)
    print(f"Split *{len(all_md_chunks)}* markdown chunks")

    # 然后进行文本层面的进行一步拆分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    final_chunks = text_splitter.split_documents(all_md_chunks)
    print(f"Create *{len(final_chunks)}* final chunks")

    print("Start embedding...")
    # 创建向量化模型
    embeddings = OllamaEmbeddings(model=embedding_model_name)

    # 进行向量化，然后写入向量数据库 pinecone
    PineconeVectorStore.from_documents(final_chunks, embeddings, index_name=index_name)
    print("Finish!")
