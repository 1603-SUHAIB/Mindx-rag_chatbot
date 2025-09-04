from langchain_community.vectorstores import Chroma

def store_embeddings(chunks, embeddings, db_path="chroma_store"):
    vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory=db_path)
    vectorstore.persist()
    return vectorstore