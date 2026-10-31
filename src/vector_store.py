"""
Step 3: Vector Database Setup Module
Manages FAISS indexing, disk serialization, and cached index loading.
"""

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


def build_and_save_vector_store(chunks: list, embeddings: Embeddings, store_path: str = "faiss_index") -> FAISS:
    """
    Creates a FAISS vector index from document chunks and saves it locally.
    """
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local(folder_path=store_path)
    return vector_store


def load_local_vector_store(store_path: str, embeddings: Embeddings) -> FAISS:
    """
    Loads an existing FAISS index from disk.
    allow_dangerous_deserialization is set to True because this is our local trusted index.
    """
    return FAISS.load_local(
        folder_path=store_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


def get_or_create_vector_store(chunks: list, embeddings: Embeddings, store_path: str = "faiss_index") -> FAISS:
    """
    Convenience factory: loads the index if it exists on disk; otherwise builds and saves it.
    """
    path = Path(store_path)
    if (path / "index.faiss").exists() and (path / "index.pkl").exists():
        return load_local_vector_store(store_path, embeddings)
    return build_and_save_vector_store(chunks, embeddings, store_path)
