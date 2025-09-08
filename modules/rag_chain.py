from typing import List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def _format_docs(docs: List[Document]) -> str:
    """Formats a list of documents into a single string."""
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"--- Document {i} (Source: {src}) ---\n{d.page_content}")
    return "\n\n".join(parts)

REPHRASE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given the conversation below and a follow-up question, rephrase the follow-up question to be a standalone question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
    "You are an expert assistant for answering questions about documents. "
    "Answer the user's question based **only** on the provided context.\n"
    "If the answer is not in the context, say you don't know.\n"
    "Be concise and clear in your answer."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
])


def build_chain(retriever, llm):
    """
    Builds a conversational RAG chain with proper chaining logic.
    """
    rephrase_question_chain = (
        REPHRASE_QUESTION_PROMPT
        | llm
        | StrOutputParser()
    )
    context_retrieval_chain = rephrase_question_chain | retriever | _format_docs
    conversational_rag_chain = (
        RunnablePassthrough.assign(
            context=context_retrieval_chain
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    return conversational_rag_chain

