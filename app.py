# app.py
import streamlit as st
from web_search import fetch_web_content
from chunking import chunk_plain_text
from generate_embeddings import generate_embeddings
from search_embeddings import search
from groq import Groq

st.set_page_config(page_title="Web QnA with RAG", layout="wide")
st.title("🌐 Web QnA – Ask questions from the web")

# Secrets
SERP_API_KEY = st.secrets.get("SERPAPI_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not SERP_API_KEY or not GROQ_API_KEY:
    st.error("Please set SERPAPI_API_KEY and GROQ_API_KEY in secrets.")
    st.stop()

# Session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "query_done" not in st.session_state:
    st.session_state.query_done = False

# Step 1: Search & Load
st.subheader("1. Enter your search query")
user_query = st.text_input("What do you want to know about?")
if st.button("Search & Load Content") and user_query:
    with st.spinner(f"Searching for '{user_query}' and scraping pages..."):
        combined_text = fetch_web_content(user_query, SERP_API_KEY, max_pages=3)
        if combined_text:
            chunks = chunk_plain_text(combined_text, chunk_size=500, overlap=50)
            st.session_state.chunks = chunks
            st.session_state.embeddings = generate_embeddings(chunks)
            st.session_state.query_done = True
            st.success(f"Loaded {len(chunks)} chunks from the web.")
        else:
            st.error("No usable content found. Try a different query.")

# Step 2: Ask questions
if st.session_state.query_done and st.session_state.chunks:
    st.subheader("2. Ask a question about the retrieved content")
    question = st.text_input("Your question:")
    if question:
        with st.spinner("Generating answer..."):
            # search returns list of (chunk_text, score)
            top_results = search(question, st.session_state.embeddings, st.session_state.chunks, top_k=3)
            if not top_results:
                st.warning("No relevant chunks found.")
            else:
                top_chunks = [chunk for chunk, _ in top_results]
                context = "\n\n---\n\n".join(top_chunks)

                client = Groq(api_key=GROQ_API_KEY)
                prompt = f"""You are a helpful assistant. Answer based on the context below. If you don't know, say so.

Context:
{context}

Question: {question}

Answer:"""
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=500
                    )
                    answer = response.choices[0].message.content
                    st.markdown("### Answer")
                    st.write(answer)
                    with st.expander("Show retrieved chunks"):
                        for i, (chunk, score) in enumerate(top_results):
                            st.markdown(f"**Chunk {i+1}** (similarity: {score:.3f})")
                            st.write(chunk)
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("Enter a search query above to get started.")