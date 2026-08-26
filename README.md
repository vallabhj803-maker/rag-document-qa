# RAG Document Q&A Chatbot — Free & Local Version

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in the content of uploaded PDF documents — **runs entirely on your machine, zero API cost**. Built with **LangChain**, **FAISS**, **sentence-transformers**, **Ollama**, and **Streamlit**.

## What it does

Upload one or more PDFs. The app:
1. Splits documents into overlapping text chunks
2. Generates vector embeddings **locally** using `sentence-transformers` (all-MiniLM-L6-v2, ~80MB, runs on CPU)
3. Stores embeddings in a local **FAISS** vector index for fast similarity search
4. On each question, retrieves the top-k most relevant chunks
5. Passes those chunks + your question to a **local LLM served by Ollama** (`llama3.2:3b`), which answers using **only** the retrieved context — reducing hallucination vs. asking the LLM directly

No OpenAI/Anthropic API key. No per-query cost. No data leaves your machine.

## Architecture

```
PDF Upload → Chunking → Local Embedding (sentence-transformers) → FAISS Vector Store
                                                │
User Question → Embed Question → Similarity Search → Top-k Chunks
                                                │
                          Chunks + Question → Local LLM (Ollama) → Grounded Answer
```

## Tech Stack

- **LangChain** — orchestration for chunking, retrieval, and the QA chain
- **sentence-transformers** — free, local embedding model
- **FAISS** — local vector similarity search
- **Ollama** — runs open-source LLMs (Llama 3.2, Mistral, etc.) locally
- **Streamlit** — web UI

## Setup

### 1. Install Ollama
Download from [ollama.com](https://ollama.com) — available for Mac, Windows, and Linux. It runs as a background service once installed.

### 2. Pull a model
```bash
ollama pull llama3.2:3b
```
This downloads a ~2GB model. It's small enough to run on most modern laptops (8GB+ RAM recommended). For better quality answers on a stronger machine, try `ollama pull llama3.2` (the larger default) or `ollama pull mistral`.

### 3. Install Python dependencies
```bash
git clone <your-repo-url>
cd rag-project
pip install -r requirements_local.txt
```

### 4. Run the app
```bash
streamlit run app_local.py
```

Upload a PDF, click **Build Knowledge Base**, and ask questions. First run will download the embedding model (~80MB) — after that, everything is offline.

## Deploying this for a live demo link

Free local LLMs need a GPU/CPU running somewhere — **Streamlit Community Cloud won't run Ollama** (no persistent background process for the model). Two options if you want a clickable live link for your resume:

- **Easiest for a resume demo:** switch back to `rag_pipeline.py` / `app.py` (the OpenAI-API version in this same repo) just for deployment — OpenAI API costs are a few cents per demo query, and it deploys cleanly on Streamlit Cloud.
- **Fully local + live:** deploy on a small cloud VM (e.g. an AWS/GCP free-tier instance) with Ollama installed, or use a service like [Replicate](https://replicate.com) or [Groq](https://groq.com) (free tier, very fast) as a drop-in swap for `ChatOllama`.

Either way, keep this local version and its README in the repo — it demonstrates you know the offline/free path too, which is a good talking point in interviews.

## What I learned building this

- How vector embeddings enable semantic (meaning-based) search vs. keyword search
- Trade-offs in chunk size/overlap and their effect on retrieval quality
- How RAG reduces LLM hallucination by grounding answers in retrieved context
- Running an LLM entirely locally/offline via Ollama, and the trade-offs vs. hosted APIs (cost vs. speed/quality)
- End-to-end deployment considerations for AI applications, not just notebooks
