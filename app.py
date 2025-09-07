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
except ImportError as e:
    st.error(f"Failed to import a required module: {e}. Please ensure all dependencies in requirements.txt are installed.", icon="🚨")
    st.stop()


st.set_page_config(
    page_title="Chat",
    page_icon="",
    layout="centered",
)

# --- Custom CSS for a Professional Dark UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container styling */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1A1D24;
        border-right: 1px solid #2D3038;
    }
    
    .stSidebar > div:first-child {
        border-bottom: 1px solid #2D3038;
        padding-bottom: 1rem;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        color: #FFFFFF;
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        transition: all 0.2s ease-in-out;
        font-weight: 500;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    }
    .stButton > button:active {
        transform: scale(0.97);
        box-shadow: none;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 1px dashed #4F46E5;
        border-radius: 8px;
        padding: 20px;
        background-color: rgba(79, 70, 229, 0.05);
    }
    
    /* Title styling */
    h1 {
        color: #FFFFFF;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 16px;
    }
    
    /* User message styling */
    [data-testid="stChatMessage"][aria-label="You"] {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        border-left: 4px solid #4F46E5;
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"][aria-label="DocuMentor"] {
        background: linear-gradient(135deg, #1A1D24 0%, #1E293B 100%);
        border-left: 4px solid #6366F1;
    }
    
    /* Chat input styling */
    [data-testid="stChatInput"] {
        background-color: #1A1D24;
        border: 1px solid #2D3038;
        border-radius: 12px;
    }
    
    /* Info box styling */
    .stAlert {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 8px;
    }
    
    /* Spinner color */
    .stSpinner > div {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    }
    
    /* Divider styling */
    hr {
        height: 1px;
        background: linear-gradient(90deg, transparent, #4F46E5, transparent);
        border: none;
        margin: 1.5rem 0;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: rgba(6, 78, 59, 0.3);
        border: 1px solid #065F46;
        border-radius: 8px;
        color: #10B981;
    }
    
    /* Error message styling */
    .stError {
        background-color: rgba(127, 29, 29, 0.3);
        border: 1px solid #7F1D1D;
        border-radius: 8px;
        color: #EF4444;
    }
    
    /* Text input focus */
    .stTextInput > div > div > input:focus {
        border-color: #6366F1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }

</style>
""", unsafe_allow_html=True)


if not os.getenv("GOOGLE_API_KEY") or "your-google-api-key" in os.getenv("GOOGLE_API_KEY", ""):
    st.error("🚨 Google API Key Not Found!", icon="🚨")
    st.stop()


with st.sidebar:
    st.header("📁 Upload & Process")
    st.markdown("Upload a document and click 'Analyze' to begin.")
    uploaded_file = st.file_uploader(
        "Choose a document", type=["pdf", "txt"], label_visibility="collapsed"
    )
    process_button = st.button("🔍 Analyze Document")


st.title("Chat")
st.markdown("Your intelligent assistant for document analysis.")
st.markdown("---")


if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []


if process_button and uploaded_file:
    with st.spinner("Analyzing document... This may take a moment."):
        try:
            llm = get_llm()
            text = load_document(uploaded_file)
            chunks = split_into_chunks(text)
            embeddings = get_embeddings()
            
            vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            
            st.session_state.rag_chain = build_chain(retriever, llm)
            st.session_state.messages = []  # Clear chat history
            st.success("Analysis complete! You can now ask questions about your document.", icon="✅")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}", icon="🚨")


if st.session_state.rag_chain is None:
    st.info("📝 Upload a document and click 'Analyze' to start chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document...", disabled=not st.session_state.rag_chain):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.rag_chain.invoke(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {e}", icon="🚨")