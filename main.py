from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# ✅ OpenRouter Key Setup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ Missing OPENROUTER_API_KEY in .env")
else:
    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Vector DB and Embeddings
DB_DIR = "rag/db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_name="medisage"
)
retriever = db.as_retriever(search_kwargs={"k": 4})


# ✅ RAG Prompt (strict context usage)
template = """
You are a Course assistant who solves student query based on. Use ONLY the context provided.
If the answer is not in the context reply:
"I don't know. Please provide more information, as your question seems to be out of my scope."
If you think the question was not related to the context, politely inform the user that you can only answer questions related to the provided context.
Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# ✅ LLM Model
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    temperature=0.0
)

def run_rag(query):
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "I don't know. Please provide more information.", []

    context = "\n\n".join(d.page_content for d in docs)
    formatted_prompt = prompt.format(context=context, question=query)

    response = llm.invoke(formatted_prompt)
    return response.content, docs


@app.get("/")
def home():
    return {"message": "✅ Accesso RAG Backend Running Successfully"}


@app.post("/chat")
async def chat(request: dict):
    question = request.get("question", "")
    answer, docs = run_rag(question)

    # sources = [
    #     {
    #         "source": d.metadata.get("source", "Unknown"),
    #         "page": d.metadata.get("page", "?")
    #     }
    #     for d in docs
    # ]

    # return {"answer": answer, "sources": sources}
    return {"answer": answer}
