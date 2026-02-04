import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# ✅ Paths
DATA_DIR = "rag/data"
DB_DIR = "rag/db"

os.makedirs(DB_DIR, exist_ok=True)

def load_documents():
    loader = DirectoryLoader(
        DATA_DIR,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} documents")
    return docs

def ingest():
    documents = load_documents()

    # ✅ Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(docs)} chunks")

    # ✅ Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ Create chroma vectorstore (persist automatically)
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="medisage",
        persist_directory=DB_DIR
    )

    print("✅ Vector DB stored successfully!")


if __name__ == "__main__":
    ingest()
