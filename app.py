# app.py
import streamlit as st
from web_search import fetch_web_content
from chunking import chunk_plain_text
from generate_embeddings import generate_embeddings
from search_embeddings import search
from groq import Groq

st.set_page_config(page_title="Web QnA with RAG", layout="wide")
st.title("🌐 Web QnA – Ask questions from the web")

# --- Secrets ---
SERP_API_KEY = st.secrets.get("SERPAPI_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not SERP_API_KEY or not GROQ_API_KEY:
    st.error("Please set SERPAPI_API_KEY and GROQ_API_KEY in secrets.")
    st.stop()

# --- Session state ---
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "query_done" not in st.session_state:
    st.session_state.query_done = False
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []

# --- Step 1: Search & Load ---
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
            # --- IMPROVED MESSAGE ---
            st.error(
                "❌ **No content could be scraped from the search results.**\n\n"
                "This can happen if:\n"
                "- The pages are behind paywalls or require login.\n"
                "- They contain very little text (e.g., forums with minimal content).\n"
                "- The site blocks scraping (robots.txt).\n\n"
                "**Try:**\n"
                "- Using a more specific query (e.g., `\"bloodborne vs elden ring review\"`).\n"
                "- Adding keywords like `article`, `review`, or `comparison`.\n"
                "- Searching for a single topic first (e.g., `bloodborne review`).\n\n"
                "Check the **debug info** below to see which URLs were tried."
            )

    # Show debug info (always show, even if nothing scraped)
    if st.session_state.debug_info:
        with st.expander("🔍 Debug info (what was scraped)"):
            for line in st.session_state.debug_info:
                st.write(line)

# --- Step 2: Ask questions ---
if st.session_state.query_done and st.session_state.chunks:
    st.subheader("2. Ask a question about the retrieved content")
    question = st.text_input("Your question:")
    if question:
        with st.spinner("Generating answer..."):
            top_results = search(question, st.session_state.embeddings, st.session_state.chunks, top_k=3)
            if not top_results:
                st.warning("No relevant chunks found.")
            else:
                top_chunks = [chunk for chunk, _ in top_results]
                context = "\n\n---\n\n".join(top_chunks)

                client = Groq(api_key=GROQ_API_KEY)

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a concise assistant. "
                            "Answer the question directly based ONLY on the provided context. "
                            "Do not add opinions, speculation, or mention that it's subjective. "
                            "If the context does not contain a clear answer, say exactly: "
                            "'The context does not provide enough information to answer.'"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ]

                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=300
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