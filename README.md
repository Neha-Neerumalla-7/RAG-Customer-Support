# TechVault Customer Support RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) customer support assistant built from scratch using Python, LangChain, FAISS, and Streamlit.

---

## 🏛️ System Architecture

```
User Query ──► [Streamlit UI]
                     │
                     ▼
             [FAISS Vector Store] ◄── [HuggingFace Embeddings (all-MiniLM-L6-v2)]
                     │ (Top-k Chunks)                ▲
                     ▼                               │
          [Grounded System Prompt] ◄── [Document Splitter: 500-char / 50-overlap]
                     │                               ▲
                     ▼                               │
               [Language Model] ◄────── [data/support_kb.txt]
                     │
                     ▼
         [Verified Support Answer + Source Citations]
```

---

## 🚀 Step-by-Step Execution Guide

### 1. Environment Setup
Open a terminal in this workspace and create an isolated virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration (Optional)
Copy the example environment file:
```powershell
copy .env.example .env
```
*(Optional: Provide an `OPENAI_API_KEY` or `GOOGLE_API_KEY` in `.env`. If omitted, the app operates smoothly in educational demo mode).*

### 3. Run Automated Diagnostic Evaluation
Verify retrieval recall across benchmark tests before launching the server:
```powershell
python evaluate_rag.py
```

### 4. Launch Streamlit Application
Start the interactive customer support web interface:
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 📋 7-Step Implementation Summary

1. **Scope Definition & Resource Gathering**: Structured knowledge base with returns, shipping, and warranty policies in `data/support_kb.txt`.
2. **Data Preparation & Embedding Creation**: Hierarchical document chunking (`RecursiveCharacterTextSplitter`, 500 size, 50 overlap) with local CPU embeddings (`all-MiniLM-L6-v2`) in `src/data_preparation.py`.
3. **Vector Database Setup**: Flat geometric indexing and zero-recomputation disk persistence via FAISS (`index.faiss` + `index.pkl`) in `src/vector_store.py`.
4. **RAG System Integration**: Defensive system prompt template with strict hallucination negative constraints and LCEL pipeline in `src/rag_chain.py`.
5. **Chatbot Interface Building**: Reactive frontend with persistent conversation history (`st.session_state`), cached index loading (`@st.cache_resource`), and expandable source chunk citations in `app.py`.
6. **Testing & Parameter Fine-Tuning**: Automated evaluation suite measuring retrieval recall on factual, conditional, and out-of-domain queries in `evaluate_rag.py`.
7. **Deployment Readiness**: 12-factor configuration externalization (`src/config.py`, `.env.example`), clear run protocols, and report documentation.
