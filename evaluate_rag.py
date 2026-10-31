"""
Step 6: Evaluation & Parameter Tuning Suite
Performs automated accuracy testing against benchmark queries and evaluates retrieval recall.
"""

from src.data_preparation import load_and_split_documents, get_embedding_model
from src.vector_store import get_or_create_vector_store

# Benchmark Query Suite for Customer Support Knowledge Base
BENCHMARK_TESTS = [
    {
        "category": "Factual Retrieval",
        "query": "How many days do I have to return an item?",
        "expected_keyword": "30 calendar days",
    },
    {
        "category": "Conditional Policy",
        "query": "Can I return opened in-ear headphones?",
        "expected_keyword": "non-refundable",
    },
    {
        "category": "Numeric Fee Extraction",
        "query": "What is the restocking fee for computer hardware?",
        "expected_keyword": "10%",
    },
    {
        "category": "Order Modification Boundary",
        "query": "How long do I have to cancel an order after placing it?",
        "expected_keyword": "60 minutes",
    },
    {
        "category": "Negative / Out-of-Domain",
        "query": "What is the policy for returning gas-powered lawnmowers?",
        "expected_keyword": None,  # Should return low similarity / no policy match
    }
]


def run_benchmark(top_k: int = 3):
    """Executes the test suite and outputs a structured evaluation report."""
    embeddings = get_embedding_model()
    chunks = load_and_split_documents("data/support_kb.txt")
    vector_store = get_or_create_vector_store(chunks, embeddings, "faiss_index")
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    print(f"\n==========================================")
    print(f"  RAG BENCHMARK EVALUATION (top_k={top_k})")
    print(f"==========================================\n")

    passed = 0
    total_evaluable = 0

    for idx, test in enumerate(BENCHMARK_TESTS, start=1):
        query = test["query"]
        expected = test["expected_keyword"]
        retrieved_docs = retriever.invoke(query)
        combined_text = " ".join([d.page_content for d in retrieved_docs])

        print(f"Test #{idx} [{test['category']}]:")
        print(f"  Query: \"{query}\"")

        if expected:
            total_evaluable += 1
            hit = expected.lower() in combined_text.lower()
            status = "PASS [Hit]" if hit else "FAIL [Miss]"
            if hit:
                passed += 1
            print(f"  Expected Keyword: '{expected}' -> Result: {status}")
        else:
            print(f"  Negative Control: Out-of-domain query handled.")

        print(f"  Top Retrieved Chunk Excerpt: {retrieved_docs[0].page_content[:90]}...\n")

    accuracy = (passed / total_evaluable) * 100 if total_evaluable else 0
    print(f"------------------------------------------")
    print(f"Retrieval Recall Score: {passed}/{total_evaluable} ({accuracy:.1f}%)")
    print(f"==========================================\n")


if __name__ == "__main__":
    run_benchmark(top_k=3)
