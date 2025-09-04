from langchain_community.vectorstores import Chroma
from modules.embeddings import get_embeddings
import os
os.environ["CHROMA_TELEMETRY"] = "false"

embeddings = get_embeddings()
db_path = "db"
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
print("Number of documents in DB:", vectorstore._collection.count())
query = "What is LangChain?"
results = vectorstore.similarity_search(query, k=2)

for i, res in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(res.page_content[:200], "...")
