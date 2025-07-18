# front/src/config.py

import os

# --- General Configuration ---
K = 5 # Number of top chunks to retrieve
CHUNK_TRUNCATE_TOKENS = 300 # Max tokens for chunks passed to LLM
MAX_TOKENS = 1200 # Max tokens for LLM response

# --- Model Configuration ---
MODEL_NAME = "deepseek-ai/DeepSeek-V3" # Or your preferred LLM model

# --- Data Paths ---
VECTOR_DB_PATH = "fitgen_vector_db_topic3" # Directory where FAISS indices and chunk files are stored
TEXT_DIR = "fitgen_vector_db_topic3" # Directory containing chunk text files

# --- Backend API Configuration ---
BACKEND_URL = "http://127.0.0.1:8000" # URL of your FastAPI backend

# --- API Keys ---
# Retrieve API key from environment variable
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

# === Topic Routing Map ===
KEYWORD_TOPIC_MAP = {
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