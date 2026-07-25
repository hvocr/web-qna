from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

def get_embeddings(paragraphs):
    model = get_embedding_model()
    embeddings = model.encode(paragraphs, convert_to_numpy=True)
    return embeddings