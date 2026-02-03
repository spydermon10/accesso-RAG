import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

db_folder = os.path.join(os.path.dirname(__file__), "db")
emb = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(persist_directory=db_folder, embedding_function=emb)

query = "What is the recommended management of dengue fever? Avoid NSAIDs?"
results = db.similarity_search(query, k=4)   # returns Document objects

for i, doc in enumerate(results, 1):
    print(f"--- Hit {i} ---")
    print(doc.page_content[:400].replace("\n", " "))
    print("METADATA:", doc.metadata)
    print()
