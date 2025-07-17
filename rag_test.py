import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from rank_bm25 import BM25Okapi
from together import Together
import os

# === Configuration ===
K = 5
CHUNK_TRUNCATE_TOKENS = 100
MAX_TOKENS = 1500
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
VECTOR_DB_PATH = "fitgen_vector_db_topic"
TEXT_DIR = "fitgen_vector_db_topic"

# === Setup ===
st.set_page_config(page_title="FitGen Chat", layout="wide")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
st.title("🏊 FitGen Adaptive RAG Chatbot")

# === API + Models ===
client = Together()
device = "cuda" if torch.has_cuda else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2")  # Let model handle device internally
# Removed .to(device) and .half() due to meta tensor error

# === Load all chunks + BM25 ===
@st.cache_data(show_spinner=False)
def load_chunks_and_bm25():
    all_chunks = []
    all_sources = []
    for file in os.listdir(TEXT_DIR):
        if file.startswith("chunks_") and file.endswith(".txt"):
            topic = file[len("chunks_"):-len(".txt")]
            with open(os.path.join(TEXT_DIR, file), encoding="utf-8") as f:
                entries = [line.strip() for line in f.read().split("\n\n") if line.strip()]
                all_chunks.extend(entries)
                all_sources.extend([topic] * len(entries))
    bm25_model = BM25Okapi([chunk.split() for chunk in all_chunks])
    return all_chunks, all_sources, bm25_model

all_chunks, all_sources, bm25_model = load_chunks_and_bm25()

# === Topic routing ===
keyword_topic_map = {
    "shoulder": "injury", "pain": "injury", "rehab": "injury",
    "freestyle": "freestyle", "crawl": "freestyle", "kick": "freestyle",
    "butterfly": "butterfly", "dolphin": "butterfly",
    "breaststroke": "breaststroke", "frog": "breaststroke",
    "breathing": "breathing", "inhale": "breathing", "exhale": "breathing",
    "warm-up": "warmup", "cooldown": "warmup", "stretch": "warmup",
    "technique": "technique", "form": "technique",
    "endurance": "conditioning", "strength": "conditioning"
}

def route_topic(query):
    for keyword, topic in keyword_topic_map.items():
        if keyword in query.lower():
            return topic
    return "general"

def truncate(text, max_tokens=100):
    return ' '.join(text.split()[:max_tokens])

# === Main Chat Interface ===
st.subheader("💬 Ask a swimming-related question")
query = st.chat_input("💬 Ask your question:")

if query:
    topic = route_topic(query)
    st.info(f"🔁 Routed to topic: {topic}")

    # FAISS retrieval
    dense_chunks = []
    faiss_index_path = f"{VECTOR_DB_PATH}/index_{topic}.faiss"
    chunk_path = f"{VECTOR_DB_PATH}/chunks_{topic}.txt"
    if os.path.exists(faiss_index_path):
        index = faiss.read_index(faiss_index_path)
        faiss_dim = index.d
        with open(chunk_path, "r", encoding="utf-8") as f:
            topic_chunks = [line.strip() for line in f.read().split("\n\n") if line.strip()]
        query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)
        assert query_vec.shape[1] == faiss_dim, f"Embedding dim {query_vec.shape[1]} does not match FAISS index dim {faiss_dim}"  # Ensure correct shape for FAISS
        _, I = index.search(query_vec, k=K)
        dense_chunks = [truncate(topic_chunks[i], CHUNK_TRUNCATE_TOKENS) for i in I[0]]

    # BM25 retrieval
    tokenized_query = query.lower().split()
    bm25_scores = bm25_model.get_scores(tokenized_query)
    top_n = np.argsort(bm25_scores)[-K:][::-1]
    bm25_chunks = [truncate(all_chunks[i], CHUNK_TRUNCATE_TOKENS) for i in top_n]

    # Merge & deduplicate
    context_chunks = list(dict.fromkeys(dense_chunks + bm25_chunks))[:K]
    context = "\n\n".join(context_chunks)

    # Prompt shaping
    if query.lower().startswith("define") or len(query.split()) < 5:
        prompt = f"Answer this directly and concisely:\n{query}"
    else:
        prompt = f"""You are a helpful assistant.

Use the following context to answer the user's question concisely.

Context:
{context}

Question: {query}
Answer:"""

    # Call Together API
    with st.spinner("🧠 Thinking with DeepSeek-V3..."):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.7
            )
            answer = response.choices[0].message.content.strip()
            st.success("✅ Answer:")
            st.markdown(answer)
            # Save conversation turn
            st.session_state.chat_history.append({"user": query, "assistant": answer})
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Show chat history
if st.session_state.chat_history:
    with st.expander("🕓 Chat History"):
        for turn in st.session_state.chat_history:
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**Assistant:** {turn['assistant']}")
