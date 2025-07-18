# front/app.py

import streamlit as st
from together import Together

# Import modules from our new 'src' package
from src.config import (
    K, CHUNK_TRUNCATE_TOKENS, MAX_TOKENS, MODEL_NAME, 
    VECTOR_DB_PATH, TEXT_DIR, BACKEND_URL, TOGETHER_API_KEY
)
from src.backend_api import (
    register_user_api, login_user_api, get_conversations_api, 
    create_conversation_api, get_messages_api, save_message_api, 
    delete_conversation_api
)
from src.rag_system import (
    load_embedding_model, load_all_chunks_and_bm25, 
    load_faiss_index, retrieve_context # retrieve_context now handles topic routing and truncation
)
from src.persistent_state import (
    set_persistent_login_data, get_persistent_login_data, clear_persistent_login_data
)

# === Initialize Together.ai Client ===
try:
    if not TOGETHER_API_KEY:
        st.error("TOGETHER_API_KEY environment variable is not set. Please set it to use Together.ai services.")
        st.stop()
    client = Together(api_key=TOGETHER_API_KEY)
except Exception as e:
    st.error(f"Error initializing Together.ai client: {e}. Please ensure your TOGETHER_API_KEY is correct.")
    st.stop()

# === Load RAG Resources ===
# These are cached by Streamlit via decorators in rag_system.py
embedding_model = load_embedding_model()
all_chunks, bm25_model, all_sources = load_all_chunks_and_bm25(TEXT_DIR)

# === Streamlit UI Setup ===
st.set_page_config(page_title="FitGen AI Assistant", layout="wide")

st.title("🏊‍♂️ FitGen AI Assistant: Your Swimming Knowledge Hub")
st.markdown("Ask questions about swimming training, technique, and more!")

# === Session State Initialization ===
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
if "confirm_delete_triggered" not in st.session_state: # State for delete confirmation dialog
    st.session_state.confirm_delete_triggered = False

# === Persistent Login Check at App Start ===
# This block runs on every rerun (including full page refreshes)
if not st.session_state.logged_in:
    persisted_username, persisted_token = get_persistent_login_data()
    if persisted_username and persisted_token:
        st.session_state.auth_token = persisted_token
        st.session_state.current_username = persisted_username
        st.session_state.logged_in = True
        st.info(f"Welcome back, {persisted_username}!")
        st.rerun() # Rerun to switch to the logged-in UI immediately

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
                    set_persistent_login_data(login_username, data["access_token"]) # Store token in local storage
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
    st.info("Please log in or register to start chatting!")
else:
    # --- Logged-in State UI ---
    st.sidebar.write(f"Logged in as: **{st.session_state.current_username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.auth_token = None
        st.session_state.current_username = None
        st.session_state.conversations = []
        st.session_state.current_conversation_id = None
        st.session_state.messages = []
        clear_persistent_login_data() # Clear token from local storage
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Your Discussions")

    if st.session_state.auth_token and not st.session_state.conversations:
        with st.spinner("Loading conversations..."):
            convs, status_code = get_conversations_api(st.session_state.auth_token)
            if status_code == 200:
                st.session_state.conversations = convs
                if st.session_state.current_conversation_id is not None:
                    current_id_exists = any(conv['id'] == st.session_state.current_conversation_id for conv in st.session_state.conversations)
                    if not current_id_exists:
                        st.session_state.current_conversation_id = None 
                        st.session_state.messages = []
            else:
                st.sidebar.error("Failed to load conversations. Please try logging out and in again.")
                st.session_state.conversations = []
                st.session_state.current_conversation_id = None
                st.session_state.messages = []

    conversation_options = {"New Discussion": None}
    sorted_conversations = sorted(st.session_state.conversations, key=lambda x: x.get('created_at', ''), reverse=True)
    for conv in sorted_conversations:
        title_display = conv.get('title', 'Untitled Discussion')
        if len(title_display) > 30:
            title_display = title_display[:27] + "..."
        conversation_options[title_display] = conv['id']
    
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
                initial_index = 0
        except Exception:
            initial_index = 0

    selected_title = st.sidebar.radio(
        "Select an existing discussion or start a new one:",
        list(conversation_options.keys()),
        index=initial_index,
        key="conversation_radio"
    )
    
    selected_conv_id = conversation_options[selected_title]

    if selected_conv_id != st.session_state.current_conversation_id:
        st.session_state.current_conversation_id = selected_conv_id
        st.session_state.messages = []
        st.session_state.confirm_delete_triggered = False 
        if selected_conv_id:
            with st.spinner(f"Loading messages for '{selected_title}'..."):
                msgs, status_code = get_messages_api(selected_conv_id, st.session_state.auth_token)
                if status_code == 200:
                    st.session_state.messages = [{"role": msg['role'], "content": msg['content']} for msg in msgs]
                else:
                    st.error(f"Failed to load messages for conversation ID {selected_conv_id}. Status: {status_code}. Please try logging out and in again.")
                    st.session_state.messages = []
        st.rerun()

    st.sidebar.markdown("---") 

    # --- Delete Conversation Button (Custom Confirmation) ---
    if st.sidebar.button("🗑️ Delete Current Discussion", use_container_width=True, disabled=selected_conv_id is None):
        if selected_conv_id is not None:
            st.session_state.confirm_delete_triggered = True
    
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
                        st.session_state.conversations = []
                    else:
                        st.error(f"Failed to delete discussion: {data.get('detail', 'Unknown error')}")
                st.session_state.confirm_delete_triggered = False
                st.rerun()
        with col_del_cancel:
            if st.button("Cancel"):
                st.info("Deletion canceled.")
                st.session_state.confirm_delete_triggered = False
                st.rerun()

    st.sidebar.markdown("---")


    # --- Main Chat UI ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask your question here...", disabled=not st.session_state.logged_in):
        if st.session_state.current_conversation_id is None:
            with st.spinner("Starting a new discussion..."):
                initial_title = query[:50] + ("..." if len(query) > 50 else "")
                new_conv_data, status_code = create_conversation_api(st.session_state.auth_token, title=initial_title)
                if status_code == 200:
                    st.session_state.current_conversation_id = new_conv_data['id']
                    st.session_state.conversations = []
                    st.rerun()
                else:
                    st.error(f"Failed to start new discussion: {new_conv_data.get('detail', 'Unknown error')}")
                    st.stop()

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        save_message_api(st.session_state.current_conversation_id, "user", query, st.session_state.auth_token)

        llm_messages = []
        system_instruction = """You are a helpful assistant. Provide the answer directly and concisely based *only* on the provided context. Do NOT include any preamble, internal thoughts, or conversational filler. State only the answer."""
        llm_messages.append({"role": "system", "content": system_instruction})

        for msg in st.session_state.messages[:-1]:
            if msg["role"] in ["user", "assistant"]:
                llm_messages.append(msg)

        # Retrieve context using the new function from rag_system
        with st.spinner("Processing your request..."):
            context = retrieve_context(query, embedding_model, all_chunks, bm25_model)
            
            user_message_with_context = f"""Context:
{context}

Question: {query}"""

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            llm_messages.append({"role": "user", "content": user_message_with_context})

            try:
                stream_response = client.chat.completions.create( 
                    model=MODEL_NAME,
                    messages=llm_messages,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                    stream=True
                )

                for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.write(full_response + "▌")
                
                message_placeholder.write(full_response)
                
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