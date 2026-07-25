# generate_embeddings.py
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

def generate_embeddings(chunks: list) -> list:
    """Return numpy array of embeddings for a list of text strings."""
    model = get_embedding_model()
    return model.encode(chunks, convert_to_numpy=True)