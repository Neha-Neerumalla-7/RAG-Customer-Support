"""
Step 4: RAG System Integration Module
Configures prompt templates, context formatting, and LCEL chain execution.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs: list) -> str:
    """Concatenates retrieved document contents separated by clean spacing."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_prompt() -> ChatPromptTemplate:
    """
    Constructs the grounded customer support prompt template.
    Enforces hallucination minimization through explicit negative constraints.
    """
    template = (
        "You are a professional customer support assistant for TechVault.\n"
        "Use ONLY the following retrieved policy excerpts to answer the customer's question.\n"
        "If the answer is not contained in the context, politely inform them that you do not have\n"
        "that information in your policy documents. Never guess or fabricate answers.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Customer Question: {question}\n"
        "Support Answer:"
    )
    return ChatPromptTemplate.from_template(template)


def build_rag_chain(retriever, llm):
    """
    Constructs the runnable LCEL RAG chain.
    Flow: User query -> (Parallel: Retrieve Context + Pass Query) -> Prompt -> LLM -> String Output
    """
    prompt = create_rag_prompt()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
