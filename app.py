import streamlit as st
import os
import tempfile
from utils.loader import load_document
from utils.splitter import chunk_documents
from utils.vector_store import create_vector_store, get_relevant_context
from utils.ollama_api import query_ollama, get_ollama_models

# Set page config
st.set_page_config(page_title="📚 DocsGyaanGuru", layout="wide")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar config
with st.sidebar:
    st.header("Configuration")
    available_models = get_ollama_models()
    selected_model = st.selectbox("Select Ollama Model", available_models)

    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
    top_k = st.slider("Top K Chunks", 1, 10, 3)

# Upload document
st.title("📚 DocsGyaanGuru")
uploaded_file = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("Processing..."):
        suffix = uploaded_file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(uploaded_file.getvalue())
            path = tmp.name

        try:
            docs = load_document(path, suffix)
            chunks = chunk_documents(docs, chunk_size, chunk_overlap)
            st.session_state.vector_store = create_vector_store(chunks)
            st.session_state.file_processed = True
            st.success(f"{len(chunks)} chunks created.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            os.unlink(path)

# Q&A Section
if st.session_state.file_processed:
    st.header("Ask Your Question")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Generating answer..."):
            context = get_relevant_context(query, st.session_state.vector_store, top_k=top_k)
            answer = query_ollama(query, selected_model, context)
            st.session_state.chat_history.append({"question": query, "answer": answer})

    for i, item in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")
        st.divider()

# Clear history
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

