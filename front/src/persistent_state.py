# front/src/persistent_state.py

import streamlit as st # Not strictly necessary here, but common to include
from streamlit_js_eval import streamlit_js_eval # Import the component

# Note: It's important to provide unique 'key' for each streamlit_js_eval call.
# This prevents Streamlit from reusing the same component instance incorrectly
# across different evaluation calls within the same run.

def set_persistent_login_data(username, token):
    """Stores username and token in browser's local storage."""
    # Ensure values are properly escaped for JS string literal
    escaped_username = username.replace("'", "\\'")
    escaped_token = token.replace("'", "\\'")
    streamlit_js_eval(js_expressions=f"localStorage.setItem('fitgen_username', '{escaped_username}');", key="set_username_js")
    streamlit_js_eval(js_expressions=f"localStorage.setItem('fitgen_token', '{escaped_token}');", key="set_token_js")

def get_persistent_login_data():
    """Retrieves username and token from browser's local storage."""
    # wait_for_output and timeout are crucial for getting the value back
    username = streamlit_js_eval(js_expressions="localStorage.getItem('fitgen_username');", key="get_username_js", want_output=True, wait_for_output=True, timeout=5000)
    token = streamlit_js_eval(js_expressions="localStorage.getItem('fitgen_token');", key="get_token_js", want_output=True, wait_for_output=True, timeout=5000)
    return username, token

def clear_persistent_login_data():
    """Removes username and token from browser's local storage."""
    streamlit_js_eval(js_expressions="localStorage.removeItem('fitgen_username');", key="clear_username_js")
    streamlit_js_eval(js_expressions="localStorage.removeItem('fitgen_token');", key="clear_token_js")