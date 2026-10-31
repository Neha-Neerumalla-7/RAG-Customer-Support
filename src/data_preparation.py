"""
Step 2: Data Preparation & Embedding Creation Module
Handles document loading, hierarchical text chunking, and embedding initialization.
"""

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_and_split_documents(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Loads raw text and splits it into manageable, overlapping chunks.
    Separators are applied hierarchically: double newlines, single newlines, then spaces.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    raw_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(raw_documents)
    return chunks


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initializes a lightweight, local embedding model that runs on CPU without API keys.
    Normalized embeddings ensure cosine similarity matches dot-product ranking.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


if __name__ == "__main__":
    kb_path = Path("data") / "support_kb.txt"
    chunks = load_and_split_documents(str(kb_path))
    print(f"Generated {len(chunks)} chunks from {kb_path.name}.")
    if chunks:
        print(f"Sample chunk [0]:\n---\n{chunks[0].page_content}\n---")
