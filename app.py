import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
# --- UI Title ---
st.title("📄 Budget PDF Chatbot (RAG System)")

# --- Load Document ---
loader = TextLoader("clean_budget.txt", encoding="utf-8")
documents = loader.load()

# --- Chunking ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = splitter.split_documents(documents)

# --- Embeddings ---
embedding = HuggingFaceEmbeddings()

# --- Vector Store (FAISS - LOCAL) ---
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- LLM (FREE) ---
pipe = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=80,
    temperature=0.2,
    repetition_penalty=1.2
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- Question Input ---
query = st.text_input("Ask a question from the document:")

# --- Answer Logic ---
if query:
    docs = retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer ONLY from the context below.

If answer is not found, say "Not found in context".

Context:
{context}

Question:
{query}

Answer in 2-3 lines:
"""

    result = llm.invoke(prompt)

    st.write("### Answer:")
    st.write(result)