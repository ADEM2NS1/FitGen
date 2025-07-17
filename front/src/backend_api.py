# front/src/backend_api.py

import requests
import streamlit as st # Used for st.error in API calls
from src.config import BACKEND_URL # Import BACKEND_URL from config

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