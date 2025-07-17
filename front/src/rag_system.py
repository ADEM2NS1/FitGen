# front/src/rag_system.py

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch
import streamlit as st # Used for st.cache_resource, st.error, st.warning
from src.config import VECTOR_DB_PATH, TEXT_DIR, K, CHUNK_TRUNCATE_TOKENS, KEYWORD_TOPIC_MAP # Import from config

@st.cache_resource
def load_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-base-en-v1.5").to(device)
    if device == "cuda":
        model = model.half() # Use half-precision for speed on GPUs
    return model

@st.cache_resource
def load_all_chunks_and_bm25(text_directory):
    """Loads all chunks and builds the BM25 model."""
    print("Loading all chunks and preparing BM25...")
    all_chunks_data = []
    all_sources_data = []
    for file in os.listdir(text_directory):
        if file.startswith("chunks_") and file.endswith(".txt"):
            topic = file[len("chunks_"):-len(".txt")]
            file_path = os.path.join(text_directory, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    entries = [line.strip() for line in f.read().split("\n\n") if line.strip()]
                    all_chunks_data.extend(entries)
                    all_sources_data.extend([topic] * len(entries))
            except Exception as e:
                st.error(f"Error reading chunk file {file}: {e}")
                continue
    
    if not all_chunks_data:
        st.warning(f"No chunk files found in {text_directory}. Please ensure your data is prepared.")
        return [], None, []

    bm25_model_instance = BM25Okapi([chunk.split() for chunk in all_chunks_data])
    return all_chunks_data, bm25_model_instance, all_sources_data

@st.cache_resource
def load_faiss_index(topic, faiss_index_path, chunk_file_path):
    """Loads and caches a specific FAISS index and its corresponding chunks."""
    try:
        index = faiss.read_index(faiss_index_path)
        with open(chunk_file_path, "r", encoding="utf-8") as f:
            topic_chunks = [line.strip() for line in f.read().split("\n\n") if line.strip()]
        return index, topic_chunks
    except FileNotFoundError:
        st.error(f"Required FAISS index or chunk file not found for topic '{topic}'. Please ensure your '{VECTOR_DB_PATH}' directory is correctly populated.")
        return None, None
    except Exception as e:
        st.error(f"Error loading FAISS index or chunks for topic '{topic}': {e}")
        return None, None

def route_topic(query_text):
    """Routes the query to a specific topic based on keywords."""
    for keyword, topic in KEYWORD_TOPIC_MAP.items(): # Use KEYWORD_TOPIC_MAP from config
        if keyword in query_text.lower():
            return topic
    return "general" # Default topic if no specific keyword matches

def truncate(text, max_tokens=CHUNK_TRUNCATE_TOKENS): # Use CHUNK_TRUNCATE_TOKENS from config
    """Truncates text to a specified number of tokens (words)."""
    return ' '.join(text.split()[:max_tokens])

def retrieve_context(query_text, embedding_model, all_chunks, bm25_model):
    """
    Retrieves relevant context chunks using a hybrid approach (Dense + BM25).
    """
    topic = route_topic(query_text)
    st.info(f"🔁 Query routed to topic: **{topic.capitalize()}**")

    dense_chunks = []
    bm25_chunks_retrieved = [] # Renamed to avoid conflict with outside variable

    faiss_index_path = os.path.join(VECTOR_DB_PATH, f"index_{topic}.faiss")
    chunk_file_path = os.path.join(VECTOR_DB_PATH, f"chunks_{topic}.txt")

    index, topic_chunks = load_faiss_index(topic, faiss_index_path, chunk_file_path)

    if index is not None and topic_chunks is not None:
        try:
            query_vec = embedding_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
            D, I = index.search(query_vec, k=K)
            valid_indices = [i for i in I[0] if i < len(topic_chunks)]
            dense_chunks = [truncate(topic_chunks[i]) for i in valid_indices]
        except Exception as e:
            st.error(f"Error during FAISS search for topic '{topic}': {e}")
    else:
        st.warning(f"FAISS index or chunks not loaded for topic '{topic}'. This may affect retrieval quality.")

    tokenized_query = query_text.lower().split()
    if bm25_model:
        bm25_scores = bm25_model.get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[-K:][::-1]
        valid_bm25_indices = [i for i in top_n_indices if i < len(all_chunks)]
        bm25_chunks_retrieved = [truncate(all_chunks[i]) for i in valid_bm25_indices]
    else:
        st.warning("BM25 model not available for keyword search.")

    # Combine and de-duplicate results, then truncate to K
    all_results = list(dict.fromkeys(dense_chunks + bm25_chunks_retrieved))[:K]
    context = "\n\n".join(all_results)
    
    if not context:
        st.warning("No relevant context found for your query. The answer might be general or limited.")
        
    return context