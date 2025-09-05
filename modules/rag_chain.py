from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {src})\n{d.page_content}")
    return "\n\n".join(parts)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
    "You are a helpful assistant. Answer the user's question **using only** the context.\n"
    "If the answer is not in the context, say you don't know.\n"
    "Provide a concise, direct answer followed by numbered sources you used."),
    ("human",
    "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:")
])

def build_chain(retriever, llm):
    """
    LCEL pipeline:
    {question} + {retrieved context} -> prompt -> llm -> text
    """
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
