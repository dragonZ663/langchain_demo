import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    # 读取文档
    loader = TextLoader(
        "E:/dragon/Udemy-business/langchain_demo/mediumblog1.txt", encoding="utf-8"
    )
    document = loader.load()

    # 进行文档分块
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    # 创建向量化模型
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # 进行向量化，然后写入向量数据库 pinecone
    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ.get("INDEX_NAME")
    )
    print("finish")
