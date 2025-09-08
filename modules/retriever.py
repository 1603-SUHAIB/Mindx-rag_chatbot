import os
os.environ["CHROMA_TELEMETRY"] = "false"

from modules.embeddings import get_embeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

def load_retriever(db_path: str = "db", k: int = 3):
    """Load Chroma vector store and return a retriever."""
    embeddings = get_embeddings()
    vs = Chroma(persist_directory=db_path, embedding_function=embeddings)
    count = vs._collection.count()
    if count == 0:
        raise RuntimeError(
            f"No documents found in '{db_path}'. "
            "Run your Day-2 ingestion (process.py) first."
        )
    return vs.as_retriever(search_kwargs={"k": min(k, count)})
