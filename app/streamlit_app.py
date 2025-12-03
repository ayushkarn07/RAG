import os
from pathlib import Path
import streamlit as st

from config import UPLOAD_DIR
from ingest import ingest_pdf, ingest_url
from qa import generate_answer

# Ensure upload directory exists
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="ChatGPT-Style RAG", layout="wide", page_icon="🤖")

# Load CSS
def load_css():
    with open("app/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_index" not in st.session_state:
    st.session_state.current_index = None

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.title("📁 Data Source")
    st.markdown("Upload PDF or enter a website URL to index.")
    st.markdown("---")

    input_type = st.radio("Choose Input Type", ["PDF Document", "Website URL"])

    if input_type == "PDF Document":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file:
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("Process PDF", type="primary"):
                with st.spinner("Indexing PDF..."):
                    num, idx_or_msg = ingest_pdf(save_path)
                    if num:
                        st.success(f"Indexed {num} document chunks")
                        st.session_state.current_index = idx_or_msg
                    else:
                        st.error(idx_or_msg)

    else:
        website_url = st.text_input("Enter Website URL", placeholder="https://example.com")
        if website_url and st.button("Process Website", type="primary"):
            with st.spinner("Indexing Website..."):
                num, idx_or_msg = ingest_url(website_url)
                if num:
                    st.success(f"Indexed {num} website chunks")
                    st.session_state.current_index = idx_or_msg
                else:
                    st.error(idx_or_msg)

    st.markdown("---")
    st.subheader("Active Index")
    if st.session_state.current_index:
        st.info(f"Using index: `{os.path.basename(st.session_state.current_index)}`")
        if st.button("Clear Index"):
            st.session_state.current_index = None
            st.rerun()
    else:
        st.warning("No index selected.")

# ---------------------- MAIN CHAT AREA ----------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.header("🤖 ChatGPT-Style RAG Assistant")

# Show messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Message your AI assistant…"):
    # Store & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = generate_answer(prompt, index_folder=st.session_state.current_index)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("</div>", unsafe_allow_html=True)
