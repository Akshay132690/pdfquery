

# 📄 AI Budget PDF Chatbot (RAG System)

An AI-powered chatbot that answers questions from a document using **Retrieval-Augmented Generation (RAG)**.
The system retrieves relevant context from the document and generates accurate responses using a language model.

---

## 🚀 Features

* 💬 ChatGPT-style chatbot interface using **Streamlit**
* 📄 Answers questions based on document content
* 🔍 Semantic search using **FAISS vector database**
* 🧠 Context-aware responses using **HuggingFace Transformers**
* ⚡ Fast performance with caching
* ❌ Reduces hallucinations by grounding answers in context

---

## 🧠 How It Works

1. The document is loaded and split into smaller chunks
2. Each chunk is converted into embeddings using HuggingFace
3. Embeddings are stored in a FAISS vector database
4. User query is converted into embedding
5. Relevant chunks are retrieved using similarity search
6. Retrieved context is passed to an LLM
7. LLM generates an answer based only on that context

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **LLM:** HuggingFace Transformers
* **Embeddings:** sentence-transformers
* **Vector DB:** FAISS
* **Framework:** LangChain

---

## 📂 Project Structure

```bash
pdfquery/
│── app.py
│── clean_budget.txt
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/pdfquery.git
cd pdfquery
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 💡 Example Questions

* What is Green Growth?
* What are the priorities of Budget 2023?
* Explain Infrastructure and Investment
* What is PMGKAY?

---

## ⚠️ Limitations

* Uses a lightweight model (`distilgpt2`) → may produce less accurate answers
* Works best with well-structured text documents
* No multi-document support (can be extended)

---

## 🚀 Future Improvements

* 📂 Upload multiple PDFs
* 🤖 Use better models (Flan-T5, Mistral)
* 📊 Show source references for answers
* 🌐 Deploy online
* 💬 Improve conversational memory

---

## 💼 Use Case

* Document-based chatbots
* Government reports / PDFs Q&A
* Knowledge base assistants
* Research tools

---

## 👨‍💻 Author

**Akshay Ingle**


---

## ⭐ If you like this project

Give it a star ⭐ and feel free to contribute!

---

