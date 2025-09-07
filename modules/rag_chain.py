from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

def _format_docs(docs: List[Document]) -> str:
    """Formats the retrieved documents into a single string."""
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}]\n{d.page_content}")
    return "\n\n".join(parts)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
    "You are a helpful assistant. Answer the user's question using only the context provided.\n"
    "If the answer is not in the context, say you don't know.\n"
    "Provide a concise, direct answer.\n\n"
    "Context:\n{context}"),
    ("human", "Question: {question}"),
])

def build_chain(retriever, llm):
    """
    Builds the LangChain Expression Language (LCEL) pipeline for the RAG.
    """
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
