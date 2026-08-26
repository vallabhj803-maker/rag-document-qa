"""
rag_pipeline.py  (FREE / LOCAL VERSION)
----------------------------------------
Core RAG (Retrieval-Augmented Generation) logic — runs entirely on your
machine, no API keys or paid usage required:

1. Load PDF documents
2. Split them into chunks
3. Embed chunks locally with sentence-transformers and store in FAISS
4. Retrieve relevant chunks for a query
5. Pass retrieved chunks + query to a local LLM (via Ollama) to generate
   a grounded answer

Requirements to run this file:
- Ollama installed and running locally (https://ollama.com)
- A model pulled, e.g.:  ollama pull llama3.2:3b
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Small, fast, good-quality free embedding model (runs on CPU, ~80MB)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Local LLM served by Ollama. 3B is small enough to run on most laptops.
# Bigger/better options if your machine can handle them: "llama3.2", "mistral"
OLLAMA_MODEL_NAME = "llama3.2:3b"


def load_and_split_pdfs(pdf_paths, chunk_size=500, chunk_overlap=50):
    """
    Load one or more PDF files and split them into overlapping chunks.

    chunk_overlap keeps a bit of context between chunks so an answer
    that spans a chunk boundary doesn't lose meaning.
    """
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(all_docs)
    return chunks


def get_embeddings():
    """
    Local, free embedding model — downloads once (~80MB) then runs
    fully offline on CPU. No API key needed.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(chunks, index_path="faiss_index"):
    """
    Embed the chunks locally and store them in a FAISS index on disk.
    """
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    return vector_store


def load_vector_store(index_path="faiss_index"):
    """Load a previously saved FAISS index from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )


def build_qa_chain(vector_store, k=4):
    """
    Build a RetrievalQA chain using a local Ollama model:
    retrieve top-k relevant chunks, then ask the LLM to answer
    using ONLY that context.
    """
    prompt_template = """Use the following pieces of context to answer the
question at the end. If you don't know the answer based on the context,
just say you don't know — do not make up an answer.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


def answer_question(qa_chain, question):
    """Run a query through the chain and return answer + source chunks."""
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": result["source_documents"],
    }
