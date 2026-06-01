import streamlit as st
from utils.utility import append_message_research, add_download, run_agents_research
from utils.research_scope import research_scope
from utils.generic_search import generic_search
from utils.search_planner import search_planner
from utils.researcher import researcher
from utils.summarise import summarise
from utils.store import save_to_notion
import asyncio
import logging
import yaml
from pathlib import Path

PROMPT_PATH = Path(__file__).parent / "research_prompts.yaml"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

# region <--------- Streamlit Page Configuration --------->

st.set_page_config(
    layout="wide",
    page_title="Research Agent",
)

# endregion <--------- Streamlit Page Configuration --------->

st.title("Research Agent")

logger = logging.getLogger(__name__)
config = st.session_state.config

# -----------------------------
# FILES DOWNLOAD SIDEBAR
# -----------------------------

#Display files that can be downloaded
with st.sidebar:
    if len(st.session_state.messages_research) > 2:
        if st.button("Summarise chat into document", use_container_width=True):
            with st.spinner("Summarising chat..."):
                summary, title, topic = summarise(api_key=config['openai_api_key'], model="gpt-5.4-nano", messages=st.session_state.messages_research)
                add_download(title, summary)
                save_to_notion(
                    config["notion_api_key"], 
                    config["notion_data_source_id"],
                    "2026-03-11",
                    title,
                    topic,
                    summary
                )
                
    
    st.subheader("Generated Files")

    if not st.session_state.downloads:
        st.caption("No files generated yet.")
    else:
        for i, file in enumerate(st.session_state.downloads):
            st.download_button(
                label=file["label"],
                data=file["data"],
                file_name=file["file_name"],
                mime=file["mime"],
                key=f"sidebar_download_{i}",
            )

# -----------------------------
# CHAT INTERFACE
# -----------------------------

for msg in st.session_state.messages_research:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], list):
            for content in msg["content"]:
                st.markdown(content)
                st.markdown("---")
        else:
            st.markdown(msg["content"])

if st.session_state.state["waiting_for_user_research"]:
    user_input = st.chat_input("Type your message...")

    if user_input:
        append_message_research("user", user_input)

        st.session_state.state.update({
            "pending_input_research": user_input,
            "waiting_for_user_research": False,
        })
        
        st.rerun()
else:
    if len(st.session_state.messages_research) == 1:
        st.session_state.agents["research_scope"] = research_scope(api_key=config['openai_api_key'], model="gpt-5.4-nano")
        st.session_state.agents["generic_search"] = generic_search(api_key=config['openai_api_key'], model="gpt-5.4-nano")
        st.session_state.agents["search_planner"] = search_planner(api_key=config['openai_api_key'], model="gpt-5.4-nano")
            
    with st.chat_message("assistant"):
        with st.spinner(f"AI is thinking..."):
            next_agent = st.session_state.state["next_agent_research"]
            pending_inputs = st.session_state.state["pending_input_research"]
            if isinstance(next_agent, list):
                system_prompts = [prompts[a] for a in next_agent]
                results = asyncio.run(run_agents_research(researcher(api_key=config['openai_api_key'], model="gpt-5.4-nano"), pending_inputs, system_prompts))
                
                response = {
                    "next_agent": "research_scope",
                    "response_to_user": results,
                    "context_next_agent": "done with research. check with the user what's next",
                }
            else:
                response = st.session_state.agents[next_agent].response(user_message=pending_inputs)
            
            if response:
                logger.info(response)
            
            if response["response_to_user"]:
                append_message_research("assistant", response["response_to_user"])

            if response["next_agent"] == "self":
                st.session_state.state.update({
                    "pending_input_research": None,
                    "waiting_for_user_research": True,
                })
            else:
                st.session_state.state.update({
                    "pending_input_research": response["context_next_agent"],
                    "next_agent_research": response["next_agent"],
                })
                
            st.rerun()