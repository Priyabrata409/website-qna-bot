import streamlit as st
import time
from ingest import ingest_url
from rag_chain import get_rag_chain

st.set_page_config(page_title="Website Q&A Bot", page_icon="🌐")

st.title("🌐 Website Q&A Bot")
st.write("Enter a URL to scrape and chat with its content.")

# Sidebar for URL input
with st.sidebar:
    st.header("Settings")
    url = st.text_input("Website URL", placeholder="https://example.com")
    process_button = st.button("Process URL")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

# Process URL
if process_button and url:
    with st.spinner("Scraping and indexing content... This may take a moment."):
        try:
            ingest_url(url)
            st.session_state.vector_store_ready = True
            st.success("Ingestion Complete! You can now chat with the content.")
        except Exception as e:
            st.error(f"Error processing URL: {e}")

# Chat interface
if st.session_state.vector_store_ready:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the website..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                rag_chain = get_rag_chain()
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")
else:
    if not url:
        st.info("Please enter a URL in the sidebar to get started.")
