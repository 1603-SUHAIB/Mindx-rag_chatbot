from modules.loader import load_document
from modules.preprocess import split_into_chunks
from modules.embeddings import get_embeddings
from modules.vectorstore import store_embeddings
from langchain_community.vectorstores import Chroma
import os
os.environ["CHROMA_TELEMETRY"] = "false"

if __name__ == "__main__":
    with open("data/sample.pdf", "rb") as f:
        text = load_document(f)

    chunks = split_into_chunks(text)

    embeddings = get_embeddings()

    vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory="db")

    print("✅ Document processed & stored successfully!")
