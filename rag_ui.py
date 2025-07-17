import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch
from together import Together
import streamlit as st

# === Configuration ===
K = 5 # Number of top chunks to retrieve
CHUNK_TRUNCATE_TOKENS = 500 # Max tokens for chunks passed to LLM
MAX_TOKENS = 1500 # Max tokens for LLM response
# --- CURRENT MODEL ---
MODEL_NAME = "deepseek-ai/DeepSeek-V3" # Or "serverless-qwen-qwen3-32b-fp8" or "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" depending on your last choice
# ---------------------
VECTOR_DB_PATH = "fitgen_vector_db_topic3" # Directory where FAISS indices and chunk files are stored
TEXT_DIR = "fitgen_vector_db_topic3" # Directory containing chunk text files

# === 🔑 API Key for Together.ai ===
# It's recommended to use st.secrets for deployment, but os.environ works locally.
# Ensure TOGETHER_API_KEY is set in your environment variables.
try:
    # Use os.environ.get() for security. Do NOT hardcode your key here.
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
except Exception as e:
    st.error(f"Error initializing Together.ai client. Make sure TOGETHER_API_KEY is set in your environment variables: {e}")
    st.stop() # Stop the app if API key is not found

# === Caching heavy resources ===
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
        return [], None, [] # Return empty if no chunks are found

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
        return None, None
    except Exception as e:
        st.error(f"Error loading FAISS index or chunks for topic '{topic}': {e}")
        return None, None

# === Load resources ===
embedding_model = load_embedding_model()
all_chunks, bm25_model, all_sources = load_all_chunks_and_bm25(TEXT_DIR)

# === Topic routing ===
keyword_topic_map = {
    "shoulder": "injury", "pain": "injury", "rehab": "injury",
    "freestyle": "freestyle", "crawl": "freestyle", "kick": "freestyle",
    "butterfly": "butterfly", "dolphin": "butterfly",
    "breaststroke": "breaststroke", "frog": "breaststroke",
    "breathing": "breathing", "inhale": "breathing", "exhale": "breathing",
    "warm-up": "warmup", "cooldown": "warmup", "stretch": "warmup",
    "technique": "technique", "form": "technique",
    "endurance": "conditioning", "strength": "conditioning",
    "diet": "nutrition", "food": "nutrition", "eating": "nutrition"
}

def route_topic(query_text):
    """Routes the query to a specific topic based on keywords."""
    for keyword, topic in keyword_topic_map.items():
        if keyword in query_text.lower():
            return topic
    return "general" # Default topic if no specific keyword matches

def truncate(text, max_tokens=CHUNK_TRUNCATE_TOKENS):
    """Truncates text to a specified number of tokens (words)."""
    return ' '.join(text.split()[:max_tokens])

# === Streamlit UI ===
st.set_page_config(page_title="FitGen AI Assistant", layout="wide")

st.title("🏊‍♂️ FitGen AI Assistant: Your Swimming Knowledge Hub")
st.markdown("Ask questions about swimming training, technique, and more!")

# User input
query = st.text_input("🔍 Ask your question:", placeholder="E.g., What are good drills for freestyle technique?")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question.")
    elif not all_chunks or bm25_model is None:
        st.error("RAG system not initialized. Check your chunk files and directory path.")
    else:
        with st.spinner("Processing your request..."):
            topic = route_topic(query)
            st.info(f"🔁 Query routed to topic: **{topic.capitalize()}**")

            dense_chunks = []
            bm25_chunks = []

            # --- FAISS search (topic-routed) ---
            faiss_index_path = os.path.join(VECTOR_DB_PATH, f"index_{topic}.faiss")
            chunk_file_path = os.path.join(VECTOR_DB_PATH, f"chunks_{topic}.txt")

            index, topic_chunks = load_faiss_index(topic, faiss_index_path, chunk_file_path)

            if index and topic_chunks:
                try:
                    query_vec = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                    D, I = index.search(query_vec, k=K)
                    valid_indices = [i for i in I[0] if i < len(topic_chunks)]
                    dense_chunks = [truncate(topic_chunks[i]) for i in valid_indices]
                except Exception as e:
                    st.error(f"Error during FAISS search for topic '{topic}': {e}")
            else:
                st.warning(f"No FAISS index or chunks found for topic '{topic}'. Falling back to general search if available.")

            # --- BM25 search ---
            tokenized_query = query.lower().split()
            if bm25_model:
                bm25_scores = bm25_model.get_scores(tokenized_query)
                top_n_indices = np.argsort(bm25_scores)[-K:][::-1]
                valid_bm25_indices = [i for i in top_n_indices if i < len(all_chunks)]
                bm25_chunks = [truncate(all_chunks[i]) for i in valid_bm25_indices]
            else:
                st.warning("BM25 model not available for keyword search.")

            # --- Combine results ---
            all_results = list(dict.fromkeys(dense_chunks + bm25_chunks))[:K]
            context = "\n\n".join(all_results)
            
            if not context:
                st.warning("No relevant context found for your query. The answer might be general or limited.")
                
            # --- Dynamic prompt ---
            if query.lower().startswith("define") or len(query.split()) < 5:
                prompt_content = f"Answer this directly and concisely:\n{query}"
            else:
                # Updated to explicitly prevent "thinking" output from the LLM with stronger instruction
                prompt_content = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Provide the answer directly and concisely based *only* on the provided context. Do NOT include any preamble, internal thoughts, or conversational filler. State only the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {query}<|eot_of_turn|><|start_header_id|>assistant<|end_header_id|>
"""

            st.subheader("Generated Answer:")
            # Create an empty placeholder for the streamed text
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Modify the API call to enable streaming
                stream_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt_content}],
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                    stream=True # <--- Set stream to True for word-by-word display
                )

                # Iterate over the streamed chunks
                for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.write(full_response + "▌") # Add a blinking cursor for better UX
                
                # After the loop, display the final response without the cursor
                message_placeholder.write(full_response)

            except Exception as e:
                st.error(f"Error generating answer from LLM: {e}")
                if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    st.info("You might have hit your Together.ai quota or rate limit. Please check your Together.ai account.")
                elif "invalid api key" in str(e).lower():
                    st.info("Please verify your TOGETHER_API_KEY environment variable.")


st.markdown("---")
st.markdown(f"Powered by SentenceTransformers (BGE-base-en-v1.5), FAISS, BM25, and Together.ai ({MODEL_NAME})")