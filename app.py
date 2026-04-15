import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# --- Page config ---
st.set_page_config(page_title="AI Budget Chatbot", layout="wide")
st.title("💬 Budget PDF Chatbot")

# =========================
# 🔥 LOAD VECTOR STORE (CACHE)
# =========================
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("clean_budget.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================
# 🔥 LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    pipe = pipeline(
        "text-generation",
        model="distilgpt2",   # change to flan-t5-base if needed
        max_new_tokens=60,
        temperature=0.2,
        repetition_penalty=1.2
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()

# =========================
# 💬 CHAT UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask something about the budget...")

if prompt:
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking... 🤖"):
        retrieved_docs = retriever.invoke(prompt)

        context = "\n".join([doc.page_content for doc in retrieved_docs])

        full_prompt = f"""
Answer ONLY from the context below.

If answer is not found, say "Not found in context".

Context:
{context}

Question:
{prompt}

Answer in 2-3 lines:
"""

        result = llm.invoke(full_prompt)

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})