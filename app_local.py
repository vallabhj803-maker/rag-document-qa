"""
app.py  (FREE / LOCAL VERSION)
-------------------------------
Streamlit UI for the RAG chatbot — no API key, no cost.
Requires Ollama running locally: https://ollama.com

Setup (one-time):
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull llama3.2:3b
    3. Make sure Ollama is running (it runs as a background service after install)

Run:
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from rag_pipeline_local import (
    load_and_split_pdfs,
    build_vector_store,
    build_qa_chain,
    answer_question,
    OLLAMA_MODEL_NAME,
)

st.set_page_config(page_title="RAG Document Q&A (Local)", page_icon="📄", layout="centered")

st.title("📄 RAG Document Q&A — Free & Local")
st.caption(
    f"Upload PDFs, then ask questions grounded in their content — "
    f"runs fully offline with local embeddings + Ollama ({OLLAMA_MODEL_NAME}). No API key, no cost."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Upload one or more PDFs\n"
        "2. Documents are chunked & embedded **locally** (sentence-transformers)\n"
        "3. Chunks are stored in a FAISS vector index\n"
        "4. Your question retrieves the most relevant chunks\n"
        "5. A **local LLM** (via Ollama) answers using only that retrieved context"
    )
    st.markdown("---")
    st.markdown(
        "**Before running:**\n"
        "- Install [Ollama](https://ollama.com)\n"
        f"- Run `ollama pull {OLLAMA_MODEL_NAME}` in a terminal\n"
        "- Make sure Ollama is running in the background"
    )

# --- Session state ---
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- File upload + indexing ---
uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

if st.button("Build Knowledge Base", disabled=not uploaded_files):
    with st.spinner("Reading, chunking, and embedding your documents locally... (first run downloads the embedding model, ~80MB)"):
        temp_paths = []
        for f in uploaded_files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(f.read())
            tmp.close()
            temp_paths.append(tmp.name)

        try:
            chunks = load_and_split_pdfs(temp_paths)
            vector_store = build_vector_store(chunks)
            st.session_state.qa_chain = build_qa_chain(vector_store)
        except Exception as e:
            st.error(
                f"Something went wrong: {e}\n\n"
                f"Make sure Ollama is installed and running, and that you've pulled "
                f"the model with `ollama pull {OLLAMA_MODEL_NAME}`."
            )
        finally:
            for p in temp_paths:
                os.remove(p)

    if st.session_state.qa_chain:
        st.success(f"Knowledge base built from {len(uploaded_files)} document(s) — ask away below.")

st.markdown("---")

# --- Chat interface ---
if st.session_state.qa_chain is None:
    st.info("Upload PDFs and click **Build Knowledge Base** to get started.")
else:
    question = st.text_input("Ask a question about your documents")
    if st.button("Ask", disabled=not question):
        with st.spinner("Thinking locally..."):
            result = answer_question(st.session_state.qa_chain, question)
        st.session_state.chat_history.append((question, result["answer"], result["sources"]))

    for q, a, sources in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {q}**")
        st.write(a)
        with st.expander("Show retrieved source chunks"):
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Source {i} (page {page}):**")
                st.text(doc.page_content[:400] + "...")
        st.markdown("---")
