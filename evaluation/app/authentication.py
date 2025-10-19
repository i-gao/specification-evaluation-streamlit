import streamlit as st
import json
from typing import Optional, Dict, Any, List


def token_to_configs_json(token: str) -> Optional[List[Dict[str, Any]]]:
    """
    Rather than using the CSV file, directly checks for jsons in the app directory.
    The json will be named f"{token}.json"
    
    Args:
        token: The user's experiment token
        
    Returns:
        List of dictionaries containing the user's configurations, or None if not found
        The list represents the different rounds of the experiment.
    """
    try:
        # Check if the token is a valid json file
        config_path = f"streamlit_logs/test_configs/{token}.json"
        configs = st.session_state.connection.read(config_path)
        return configs
        
    except Exception as e:
        print(f"Error loading configs: {e}")
        return None

def check_token() -> Optional[str]:
    """
    Check if the user has entered a valid token.
    
    Returns:
        Token if valid, None otherwise
    """
    # Check if user has already entered a valid token in session state
    if "valid_token" in st.session_state:
        return st.session_state.valid_token
    
    return None


def require_token():
    """
    Ensure user has entered a valid token before proceeding.
    This should be called at the start of any app that requires a token.
    
    Returns:
        Token if valid
    """
    # Check if already has valid token
    token = check_token()
    if token:
        return token
    
    # Show token input form    
    with st.form("token_form"):
        token_input = st.text_input(
            "Please enter your experiment token to start.", 
            key="token_input",
            help="Your experiment token is a unique alphanumeric string that was in the HIT."
        )
        submit_button = st.form_submit_button("Start Experiment", type="primary")
        
        if submit_button:
            if token_input:
                # Validate the token
                configs = token_to_configs_json(token_input)
                if configs:
                    # Token is valid, store it and proceed
                    st.session_state.valid_token = token_input
                    st.session_state.user_configs = configs
                    st.rerun()
                else:
                    st.error("Invalid token. Please check your token and try again.")
            else:
                st.error("Please enter a token.")
    
    # If no valid token, don't render the rest of the app
    st.stop()


def get_user_configs(token: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get the user's configuration based on their token.
    
    Args:
        token: The user's experiment token
        
    Returns:
        List of user configuration dictionaries, or None if not found
    """
    if token is None:
        return None
    configs = token_to_configs_json(token)
    if configs is None:
        st.error(f"Configuration not found for token: {token}")
        st.stop()
    
    return configs
