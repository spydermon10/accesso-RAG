import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# ✅ Folder paths (auto-detect script directory)
pdf_folder = os.path.dirname(os.path.abspath(__file__))
db_folder = os.path.join(pdf_folder, "db")


# 1️⃣ Load and extract text
docs = []
loader = PyMuPDFLoader

import os
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        doc_path = os.path.join(pdf_folder, file)
        docs.extend(loader(doc_path).load())

print(f"Loaded {len(docs)} text chunks from PDFs")

# 2️⃣ Split into chunks for embeddings
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

print(f"Split into {len(chunks)} chunks")

# 3️⃣ Embeddings model for medical text
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 4️⃣ Create Vector DB (Chroma)
db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=db_folder)

db.persist()
print("✅ Vector DB created successfully!")
