# filename: utility.py
import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import asyncio
import logging

SGT = timezone(timedelta(hours=8))

# """
# This file contains the common components used in the Streamlit App.
# This includes the sidebar, the title, the footer, and the password check.
# """

def init_session(config: dict):
    if "downloads" not in st.session_state:
        st.session_state.downloads = []

    if "state" not in st.session_state:
        st.session_state.state = {
            "pending_input_compare": None,
            "pending_input_research": None,
            "waiting_for_user_compare": True,
            "waiting_for_user_research": True,
            "next_agent_research": "research_scope",
            "chat_has_started": False,
            "evaluate": False,
        }

    if "models" not in st.session_state:
        st.session_state.models = []
        
        for model in config['openai_models']:
            st.session_state.models.append({
                "label": f"OpenAI | {model}",
                "provider": "openai",
                "model": model,
            })

        for model in config['anthropic_models']:
            st.session_state.models.append({
                "label": f"Anthropic | {model}",
                "provider": "anthropic",
                "model": model,
            })

        for model in config['gemini_models']:
            st.session_state.models.append({
                "label": f"Gemini | {model}",
                "provider": "gemini",
                "model": model,
            })
            
    if "messages_compare" not in st.session_state:
        st.session_state.messages_compare = []
        
    if "messages_research" not in st.session_state:
        st.session_state.messages_research = []

    if "selected_model_labels" not in st.session_state:
        st.session_state.selected_model_labels = []
        
    if "agents" not in st.session_state:
        st.session_state.agents = {
            "compare": [],
            "evaluator": None,
            "research_scope": None,
        }
    
    if "config" not in st.session_state:
        st.session_state.config = config

def check_password():
    """Returns `True` if the user has the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if os.getenv("PASSWORD") == st.session_state.get("password"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Check if the PASSWORD environment variable is set
    password_env = os.getenv("PASSWORD")
    if password_env is None or password_env == "":
        return True  # Skip password check if not set

    # If the password has already been validated, return True
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )

    # Show error if the password is incorrect
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")

    return False

def load_config():
    load_dotenv()

    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_data_source_id = os.getenv("NOTION_DATA_SOURCE_ID")
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODELS", "").strip()
    anthropic_model = os.getenv("ANTHROPIC_MODELS", "gpt-4.1-mini").strip()
    gemini_model = os.getenv("GEMINI_MODELS", "gpt-4.1-mini").strip()

    return {'openai_api_key': openai_api_key,
            'anthropic_api_key': anthropic_api_key,
            'gemini_api_key': gemini_api_key,
            'openai_models': [m.strip() for m in openai_model.split(",") if m.strip()],
            'anthropic_models': [m.strip() for m in anthropic_model.split(",") if m.strip()],
            'gemini_models': [m.strip() for m in gemini_model.split(",") if m.strip()],
            'notion_api_key': notion_api_key,
            'notion_data_source_id': notion_data_source_id,            
            }

def append_message_compare(role, content):
    st.session_state.messages_compare.append({
        "role": role,
        "content": content,
        # "timestamp": datetime.now(SGT).strftime("%Y-%m-%dT%H-%M-%S"),
    })
    
def append_message_research(role, content):
    st.session_state.messages_research.append({
        "role": role,
        "content": content,
        # "timestamp": datetime.now(SGT).strftime("%Y-%m-%dT%H-%M-%S"),
    })

async def run_agents(user_message, agents):
    tasks = [agent.response(user_message) for agent in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

async def run_agents_research(agent, user_message, system_prompts):
    tasks = [agent.response(msg, prompt) for msg, prompt in zip(user_message, system_prompts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
    
def add_download(file_name, content):
    st.session_state.downloads.append({
        "label": file_name+".md",
        "data": content,
        "file_name": file_name+".md",
        "mime": "text/markdown",
        # "created_at": datetime.now(SGT).strftime("%Y-%m-%dT%H-%M"),
    })
    
def setup_logging():
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )