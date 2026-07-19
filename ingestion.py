from embeddings.md_embedding import embed

FILE_PATH = "webwright.md"

if __name__ == "__main__":
    print("Document Ingestion")
    embed(FILE_PATH)
