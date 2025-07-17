# test.py

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch
from together import Together
import streamlit as st
import requests

# === Configuration ===
K = 5 # Number of top chunks to retrieve
CHUNK_TRUNCATE_TOKENS = 300 # Max tokens for chunks passed to LLM
MAX_TOKENS = 200 # Max tokens for LLM response
# --- CURRENT MODEL ---
MODEL_NAME = "deepseek-ai/DeepSeek-V3" # Or your preferred model
# ---------------------
VECTOR_DB_PATH = "fitgen_vector_db_topic3" # Directory where FAISS indices and chunk files are stored
TEXT_DIR = "fitgen_vector_db_topic3" # Directory containing chunk text files

# === Backend API Configuration ===
BACKEND_URL = "http://127.0.0.1:8000" # URL of your FastAPI backend

# === 🔑 API Key for Together.ai ===
# Make sure TOGETHER_API_KEY is set in your environment variables
try:
    # Retrieve API key from environment variable first
    together_api_key = os.environ.get("TOGETHER_API_KEY")
    
    # Explicitly check if the environment variable was found
    if not together_api_key:
        st.error("TOGETHER_API_KEY environment variable is not set. Please set it to use Together.ai services.")
        st.stop() # Stop execution if key is missing

    # Initialize client with the retrieved API key
    client = Together(api_key=together_api_key)

except Exception as e:
    st.error(f"Error initializing Together.ai client: {e}. Please ensure your TOGETHER_API_KEY is correct.")
    st.stop()

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

# === Backend API Functions ===
def register_user_api(username, password):
    url = f"{BACKEND_URL}/register"
    response = requests.post(url, json={"username": username, "password": password})
    return response.json(), response.status_code

def login_user_api(username, password):
    url = f"{BACKEND_URL}/token"
    # FastAPI's OAuth2PasswordRequestForm expects form-urlencoded data
    response = requests.post(url, data={"username": username, "password": password})
    return response.json(), response.status_code

def get_conversations_api(token):
    url = f"{BACKEND_URL}/conversations"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        st.error(f"API Error fetching conversations: {e}")
        return {"detail": str(e)}, 500

def create_conversation_api(token, title="New Chat"):
    url = f"{BACKEND_URL}/conversations"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(url, json={"title": title}, headers=headers)
        response.raise_for_status()
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        st.error(f"API Error creating conversation: {e}")
        return {"detail": str(e)}, 500

def get_messages_api(conversation_id, token):
    url = f"{BACKEND_URL}/conversations/{conversation_id}/messages"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        st.error(f"API Error fetching messages: {e}")
        return {"detail": str(e)}, 500

def save_message_api(conversation_id, role, content, token):
    url = f"{BACKEND_URL}/conversations/{conversation_id}/messages"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"role": role, "content": content}
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        st.error(f"API Error saving message: {e}")
        return {"detail": str(e)}, 500

def delete_conversation_api(conversation_id, token):
    url = f"{BACKEND_URL}/conversations/{conversation_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        st.error(f"API Error deleting conversation: {e}")
        return {"detail": str(e)}, 500

# === Streamlit UI ===
st.set_page_config(page_title="FitGen AI Assistant", layout="wide")

st.title("🏊‍♂️ FitGen AI Assistant: Your Swimming Knowledge Hub")
st.markdown("Ask questions about swimming training, technique, and more!")

# Initialize session state variables for authentication and chat
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "current_username" not in st.session_state:
    st.session_state.current_username = None
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = [] # Current displayed messages for selected convo
if "confirm_delete_triggered" not in st.session_state: # New state for delete confirmation
    st.session_state.confirm_delete_triggered = False

# --- Login/Registration Sidebar ---
st.sidebar.title("Account")

if not st.session_state.logged_in:
    st.sidebar.subheader("Login / Register")
    login_username = st.sidebar.text_input("Username", key="login_username")
    login_password = st.sidebar.text_input("Password", type="password", key="login_password")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            if login_username and login_password:
                data, status_code = login_user_api(login_username, login_password)
                if status_code == 200:
                    st.session_state.logged_in = True
                    st.session_state.auth_token = data["access_token"]
                    st.session_state.current_username = login_username
                    st.sidebar.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.sidebar.error(f"Login failed: {data.get('detail', 'Unknown error')}")
            else:
                st.sidebar.warning("Please enter both username and password.")
    with col2:
        if st.button("Register", use_container_width=True):
            if login_username and login_password:
                data, status_code = register_user_api(login_username, login_password)
                if status_code == 200:
                    st.sidebar.success("Registration successful! You can now log in.")
                elif status_code == 400 and "Username already registered" in data.get("detail", ""):
                    st.sidebar.warning("Username already exists. Please choose a different one or log in.")
                else:
                    st.sidebar.error(f"Registration failed: {data.get('detail', 'Unknown error')}")
            else:
                st.sidebar.warning("Please enter both username and password for registration.")
    # Message to display when not logged in
    st.info("Please log in or register to start chatting!")
else:
    # --- Logged-in State ---
    st.sidebar.write(f"Logged in as: **{st.session_state.current_username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.auth_token = None
        st.session_state.current_username = None
        st.session_state.conversations = []
        st.session_state.current_conversation_id = None
        st.session_state.messages = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Your Discussions")

    # Fetch conversations if logged in and not already loaded
    if st.session_state.auth_token and not st.session_state.conversations:
        with st.spinner("Loading conversations..."):
            convs, status_code = get_conversations_api(st.session_state.auth_token)
            if status_code == 200:
                st.session_state.conversations = convs
                # Validate current_conversation_id after loading conversations
                if st.session_state.current_conversation_id is not None:
                    # Check if the previously selected conversation ID still exists in the loaded list
                    current_id_exists = any(conv['id'] == st.session_state.current_conversation_id for conv in st.session_state.conversations)
                    if not current_id_exists:
                        # If the old ID is not found, reset to None to select "New Discussion"
                        st.session_state.current_conversation_id = None 
                        st.session_state.messages = [] # Clear messages associated with the stale ID
            else:
                st.sidebar.error("Failed to load conversations. Please try logging out and in again.")
                st.session_state.conversations = [] # Ensure it's an empty list on failure
                st.session_state.current_conversation_id = None # Also reset current ID on failure
                st.session_state.messages = [] # Clear messages on failure

    # Display conversations and allow selection
    conversation_options = {"New Discussion": None}
    # Sort conversations by creation date, newest first
    sorted_conversations = sorted(st.session_state.conversations, key=lambda x: x.get('created_at', ''), reverse=True)
    for conv in sorted_conversations:
        title_display = conv.get('title', 'Untitled Discussion')
        # Truncate long titles for display
        if len(title_display) > 30:
            title_display = title_display[:27] + "..."
        conversation_options[title_display] = conv['id']
    
    # Find current selected index, default to "New Discussion" if no conversation selected
    initial_index = 0
    if st.session_state.current_conversation_id is not None:
        try:
            current_title_for_id = None
            for title, conv_id in conversation_options.items():
                if conv_id == st.session_state.current_conversation_id:
                    current_title_for_id = title
                    break
            
            if current_title_for_id is not None:
                initial_index = list(conversation_options.keys()).index(current_title_for_id)
            else:
                initial_index = 0 # Fallback to "New Discussion" if current ID's title not found
        except Exception: # Catch any potential indexing errors as a fallback
            initial_index = 0 # Fallback if current_conversation_id somehow isn't in current list

    selected_title = st.sidebar.radio(
        "Select an existing discussion or start a new one:",
        list(conversation_options.keys()),
        index=initial_index,
        key="conversation_radio" # Add a key to prevent reruns from resetting radio button
    )
    
    selected_conv_id = conversation_options[selected_title]

    # Only load messages if the conversation ID has changed or if it's a new discussion selected
    # This block triggers a rerun when conversation changes in sidebar
    if selected_conv_id != st.session_state.current_conversation_id:
        st.session_state.current_conversation_id = selected_conv_id
        st.session_state.messages = [] # Clear messages when switching conversations
        # Reset confirm delete state when switching conversations
        st.session_state.confirm_delete_triggered = False 
        if selected_conv_id:
            with st.spinner(f"Loading messages for '{selected_title}'..."):
                msgs, status_code = get_messages_api(selected_conv_id, st.session_state.auth_token)
                if status_code == 200:
                    st.session_state.messages = [{"role": msg['role'], "content": msg['content']} for msg in msgs]
                else:
                    st.error(f"Failed to load messages for conversation ID {selected_conv_id}. Status: {status_code}. Please try logging out and in again.")
                    st.session_state.messages = [] # Reset on error
        st.rerun() # Rerun to update the chat display based on new conversation selection

    st.sidebar.markdown("---") 

    # --- Delete Conversation Button (Custom Confirmation) ---
    # The main delete button itself
    if st.sidebar.button("🗑️ Delete Current Discussion", use_container_width=True, disabled=selected_conv_id is None):
        if selected_conv_id is not None: # Only trigger confirmation if a conversation is actually selected
            st.session_state.confirm_delete_triggered = True
    
    # Conditional display of the actual confirmation dialog
    if st.session_state.confirm_delete_triggered:
        st.warning("Are you sure you want to delete this discussion and all its messages? This cannot be undone.")
        col_del_yes, col_del_cancel = st.columns(2)
        with col_del_yes:
            if st.button("Yes, delete"):
                with st.spinner("Deleting discussion..."):
                    data, status_code = delete_conversation_api(selected_conv_id, st.session_state.auth_token)
                    if status_code == 200:
                        st.success("Discussion deleted successfully!")
                        st.session_state.current_conversation_id = None
                        st.session_state.messages = []
                        st.session_state.conversations = [] # Force reload of conversation list
                    else:
                        st.error(f"Failed to delete discussion: {data.get('detail', 'Unknown error')}")
                st.session_state.confirm_delete_triggered = False # Reset confirmation state
                st.rerun() # Rerun to update UI after deletion
        with col_del_cancel:
            if st.button("Cancel"):
                st.info("Deletion canceled.")
                st.session_state.confirm_delete_triggered = False # Reset confirmation state
                st.rerun() # Rerun to clear the warning and buttons

    st.sidebar.markdown("---") # Another separator


    # --- Main Chat UI ---
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input form at the bottom
    # Chat input is disabled if not logged in
    if query := st.chat_input("Ask your question here...", disabled=not st.session_state.logged_in):
        # If starting a new conversation, create it first
        if st.session_state.current_conversation_id is None:
            with st.spinner("Starting a new discussion..."):
                # Use first 50 chars of the query as the initial title
                initial_title = query[:50] + ("..." if len(query) > 50 else "")
                new_conv_data, status_code = create_conversation_api(st.session_state.auth_token, title=initial_title)
                if status_code == 200:
                    st.session_state.current_conversation_id = new_conv_data['id']
                    # Clear conversations to force a full reload on the next rerun
                    # This ensures the radio button's options are fresh and sorted correctly
                    st.session_state.conversations = [] 
                    st.rerun() # Rerun to update sidebar with new conversation selected
                else:
                    st.error(f"Failed to start new discussion: {new_conv_data.get('detail', 'Unknown error')}")
                    # If conversation creation fails, prevent adding user query to chat history
                    st.stop() # Stop further execution if conversation isn't created

        # If a conversation ID exists (either new or existing), proceed to add messages
        # Add user message to chat history (local session state and backend)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Save user message to backend
        save_message_api(st.session_state.current_conversation_id, "user", query, st.session_state.auth_token)

        # Prepare messages for LLM, including system prompt and context
        llm_messages = []

        # Add the system instruction for the LLM
        system_instruction = """You are a helpful assistant. Provide the answer directly and concisely based *only* on the provided context. Do NOT include any preamble, internal thoughts, or conversational filler. State only the answer."""
        llm_messages.append({"role": "system", "content": system_instruction})

        # Add conversation history for context (up to current user query)
        for msg in st.session_state.messages[:-1]: # Exclude the current user query for now
            if msg["role"] in ["user", "assistant"]:
                llm_messages.append(msg)

        # --- RAG logic: Retrieve context for the current query ---
        with st.spinner("Processing your request..."):
            topic = route_topic(query)
            st.info(f"🔁 Query routed to topic: **{topic.capitalize()}**")

            dense_chunks = []
            bm25_chunks = []

            faiss_index_path = os.path.join(VECTOR_DB_PATH, f"index_{topic}.faiss")
            chunk_file_path = os.path.join(VECTOR_DB_PATH, f"chunks_{topic}.txt")

            index, topic_chunks = load_faiss_index(topic, faiss_index_path, chunk_file_path)

            if index is not None and topic_chunks is not None:
                try:
                    query_vec = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                    D, I = index.search(query_vec, k=K)
                    valid_indices = [i for i in I[0] if i < len(topic_chunks)]
                    dense_chunks = [truncate(topic_chunks[i]) for i in valid_indices]
                except Exception as e:
                    st.error(f"Error during FAISS search for topic '{topic}': {e}")
            else:
                st.warning(f"FAISS index or chunks not loaded for topic '{topic}'. This may affect retrieval quality.")

            tokenized_query = query.lower().split()
            if bm25_model:
                bm25_scores = bm25_model.get_scores(tokenized_query)
                top_n_indices = np.argsort(bm25_scores)[-K:][::-1]
                valid_bm25_indices = [i for i in top_n_indices if i < len(all_chunks)]
                bm25_chunks = [truncate(all_chunks[i]) for i in valid_bm25_indices]
            else:
                st.warning("BM25 model not available for keyword search.")

            # Combine and de-duplicate results, then truncate to K
            all_results = list(dict.fromkeys(dense_chunks + bm25_chunks))[:K]
            context = "\n\n".join(all_results)
            
            if not context:
                st.warning("No relevant context found for your query. The answer might be general or limited.")
                
            # Append the current user query with context to the LLM messages
            user_message_with_context = f"""Context:
{context}

Question: {query}"""

        # --- Generate response from LLM ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # The 'messages' list now contains system instruction + history + current user query with context
            llm_messages.append({"role": "user", "content": user_message_with_context})

            try:
                stream_response = client.chat.completions.create( 
                    model=MODEL_NAME,
                    messages=llm_messages, # Pass the full conversation history
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                    stream=True
                )

                for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.write(full_response + "▌")
                
                message_placeholder.write(full_response)
                
                # Add assistant response to chat history (local session state and backend)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message_api(st.session_state.current_conversation_id, "assistant", full_response, st.session_state.auth_token)


            except Exception as e:
                st.error(f"Error generating answer from LLM: {e}")
                if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    st.info("You might have hit your Together.ai quota or rate limit. Please check your Together.ai account.")
                elif "invalid api key" in str(e).lower():
                    st.info("Please verify your TOGETHER_API_KEY environment variable.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"Powered by SentenceTransformers (BGE-base-en-v1.5), FAISS, BM25, and Together.ai ({MODEL_NAME})")