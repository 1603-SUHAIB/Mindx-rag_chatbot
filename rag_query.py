import sys
import os
os.environ["CHROMA_TELEMETRY"] = "false"

from modules.retriever import load_retriever
from modules.llm import get_llm
from modules.rag_chain import build_chain

def main():
    try:
        retriever = load_retriever(db_path="db", k=3)
    except Exception as e:
        print(f"❌ Failed to load retriever: {e}")
        sys.exit(1)
    llm = get_llm()
    chain = build_chain(retriever, llm)
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("Ask a question about your documents: ").strip()
    if not question:
        print("⚠️ Empty question. Exiting.")
        return
    try:
        answer = chain.invoke(question)
        print("\n🧠 Answer:\n")
        print(answer)
    except Exception as e:
        print(f"❌ Error during RAG answer generation: {e}")

if __name__ == "__main__":
    main()
