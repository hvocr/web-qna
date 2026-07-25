from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

def get_top_k_chunks(user_query, chunks, embeddings, k=3):
    if not chunks or embeddings is None or len(chunks) != len(embeddings):
        return []
    model = get_embedding_model()
    chunk_texts = [f"{ch['heading']}\n{ch['content']}" for ch in chunks]
    query_emb = model.encode([user_query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, embeddings)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    return [(chunks[i], similarities[i]) for i in top_indices]