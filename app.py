import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from modules.loader import load_document
    from modules.preprocess import split_into_chunks
    from modules.embeddings import get_embeddings
    from modules.llm import get_llm
    from modules.rag_chain import build_chain
    from langchain_community.vectorstores import Chroma
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    st.error(f"Failed to import a required module: {e}. Please ensure all dependencies in requirements.txt are installed.", icon="🚨")
    st.stop()

st.set_page_config(
    page_title="DocuMentor",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #1A1D24; border-right: 1px solid #2D3038; }
    .stSidebar > div:first-child { border-bottom: 1px solid #2D3038; padding-bottom: 1rem; }
    .stButton > button { border-radius: 8px; border: none; padding: 10px 20px; color: #FFFFFF; background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%); transition: all 0.2s ease-in-out; font-weight: 500; width: 100%; }
    .stButton > button:hover { background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%); box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3); }
    [data-testid="stFileUploader"] { border: 1px dashed #4F46E5; border-radius: 8px; padding: 20px; background-color: rgba(79, 70, 229, 0.05); }
    h1 { color: #FFFFFF; font-weight: 700; background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    [data-testid="stChatMessage"] { border-radius: 12px; padding: 1rem; margin-bottom: 16px; }
    [data-testid="stChatInput"] { background-color: #1A1D24; border: 1px solid #2D3038; border-radius: 12px; }
    .stAlert { background-color: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; }
    hr { height: 1px; background: linear-gradient(90deg, transparent, #4F46E5, transparent); border: none; margin: 1.5rem 0; }
    .stSuccess { background-color: rgba(6, 78, 59, 0.3); border: 1px solid #065F46; border-radius: 8px; color: #10B981; }
    .stError { background-color: rgba(127, 29, 29, 0.3); border: 1px solid #7F1D1D; border-radius: 8px; color: #EF4444; }
</style>
""", unsafe_allow_html=True)

if not os.getenv("GOOGLE_API_KEY") or "your-google-api-key" in os.getenv("GOOGLE_API_KEY", ""):
    st.error("🚨 Google API Key Not Found! Please add it to your .env file.", icon="🚨")
    st.stop()

with st.sidebar:
    st.header("⚙️ Controls")

    with st.expander("📁 Upload & Process", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a document", type=["pdf", "txt"], label_visibility="collapsed"
        )
        process_button = st.button("🔍 Analyze Document", use_container_width=True)

    st.markdown("---")
    st.caption("💡 Tip: Upload a PDF or TXT file and start asking DocuMentor questions!")

st.markdown("<h1 style='text-align: center;'>🧠 DocuMentor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9CA3AF;'>Your intelligent assistant for document analysis.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if process_button and uploaded_file:
    progress = st.progress(0, "Initializing...")
    try:
        with st.spinner("🔄 Analyzing your document..."):
            llm = get_llm()
            progress.progress(20, "Loading document...")
            text = load_document(uploaded_file)

            progress.progress(40, "Splitting into chunks...")
            chunks = split_into_chunks(text)

            progress.progress(60, "Creating embeddings...")
            embeddings = get_embeddings()

            progress.progress(80, "Building knowledge chain...")
            vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            st.session_state.rag_chain = build_chain(retriever, llm)

            st.session_state.messages = []
            progress.progress(100, "Completed ✅")
            st.success("Analysis complete! Start chatting below.", icon="✅")
    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}", icon="🚨")
        st.session_state.rag_chain = None

if st.session_state.rag_chain is None:
    st.info("📝 Upload a document and click 'Analyze' to start chatting.")

for message in st.session_state.messages:
    role = message["role"]
    if role == "user":
        with st.chat_message("You", avatar="👤"):
            st.markdown(
                f"<div style='padding:10px; border-radius:12px; background:linear-gradient(135deg,#1E293B,#334155); color:#E5E7EB'>{message['content']}</div>",
                unsafe_allow_html=True
            )
    else:
        with st.chat_message("DocuMentor", avatar="🤖"):
            st.markdown(
                f"<div style='padding:10px; border-radius:12px; background:linear-gradient(135deg,#1A1D24,#1E293B); color:#D1D5DB'>{message['content']}</div>",
                unsafe_allow_html=True
            )

if prompt := st.chat_input("Ask a question about the document...", disabled=not st.session_state.rag_chain):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("You", avatar="👤"):
        st.markdown(
            f"<div style='padding:10px; border-radius:12px; background:linear-gradient(135deg,#1E293B,#334155); color:#E5E7EB'>{prompt}</div>",
            unsafe_allow_html=True
        )

    with st.spinner("🤔 Thinking..."):
        try:
            chat_history_for_model = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history_for_model.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history_for_model.append(AIMessage(content=msg["content"]))

            chain_input = {
                "question": prompt,
                "chat_history": chat_history_for_model
            }

            response = st.session_state.rag_chain.invoke(chain_input)

            with st.chat_message("DocuMentor", avatar="🤖"):
                st.markdown(
                    f"<div style='padding:10px; border-radius:12px; background:linear-gradient(135deg,#1A1D24,#1E293B); color:#D1D5DB'>{response}</div>",
                    unsafe_allow_html=True
                )

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"⚠️ Error while generating the answer: {e}", icon="🚨")
