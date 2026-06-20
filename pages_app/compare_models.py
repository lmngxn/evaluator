import streamlit as st
from utils.utility import append_message_compare, add_download, run_agents
from utils.openai import openAIAgent
from utils.anthropic import claudeAgent
from utils.gemini import geminiAgent
from utils.evaluator import evaluatorAgent
import asyncio
import logging


# region <--------- Streamlit Page Configuration --------->

st.set_page_config(
    layout="wide",
    page_title="Multi-model chatbot and evaluator",
)

# endregion <--------- Streamlit Page Configuration --------->

st.title("Multi-model chatbot and evaluator")

logger = logging.getLogger(__name__)
config = st.session_state.config

# -----------------------------
# MODEL SELECTION, MODEL DISPLAY TOGGLE, EVALUATION TOGGLE
# -----------------------------

cols = st.columns([2.5,3,1], border=True)

with cols[0]:
    st.markdown("#### Choose models")
    selected_model_labels = st.multiselect(
        "Select up to 3 models",
        options=[item["label"] for item in st.session_state.models],
        max_selections=3,
        disabled=st.session_state.state["chat_has_started"],
    )

visible_labels_ordered = selected_model_labels
if st.session_state.state["chat_has_started"]:
    with cols[1]:
        st.markdown("#### Display models")
        visible_labels = st.segmented_control(
            "Click to show/hide model outputs",
            options=[label for label in selected_model_labels],
            selection_mode="multi",
            default=[label for label in selected_model_labels],
        )
        #Ensure order is fixed
        visible_labels_ordered = [label for label in selected_model_labels if label in visible_labels]

with cols[2]:
    st.markdown("#### Evaluation On/Off")
    st.session_state.state["evaluate"] = st.toggle(
        "Activate feature", 
        value=st.session_state.state["evaluate"], 
        disabled=st.session_state.state["chat_has_started"]
    )

if st.session_state.state["chat_has_started"]:
    if st.button("Start new chat and choose models again"):
        st.session_state.messages_compare = []
        st.session_state.selected_model_labels = []
        st.session_state.agents["compare"] = []
        st.session_state.agents["evaluator"] = None
        st.session_state.state["chat_has_started"] = False
        st.rerun()

 
# -----------------------------
# FILES DOWNLOAD SIDEBAR
# -----------------------------

#Display files that can be downloaded
with st.sidebar:
   
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

if not selected_model_labels:
    st.warning("Please select at least one model.")
    st.stop()

for msg in st.session_state.messages_compare:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg["content"], dict):
            cols = st.columns(len(visible_labels_ordered), border=True)
            for label, col in zip(visible_labels_ordered, cols):
                if label not in msg["content"]:
                    continue
                with col:
                    logger.info(msg["content"])
                    st.markdown(f"#### {label}")
                    st.markdown(msg["content"][label]["result"])
                    if "evaluation" in msg["content"][label]:
                        with st.expander("Evaluation Result:", expanded=True):
                            st.markdown(msg["content"][label]["evaluation"])
        else:
            st.markdown(msg["content"])

if st.session_state.state["evaluating"]:
    with st.spinner("Evaluating responses..."):
        evaluator = st.session_state.agents["evaluator"]
        last_content = st.session_state.messages_compare[-1]["content"]
        results_to_eval = [
            last_content[label]["result"]
            for label in st.session_state.selected_model_labels
            if label in last_content and "result" in last_content[label]
        ]
        try:
            evaluation_result = asyncio.run(evaluator.response(
                st.session_state.state["pending_prompt_for_eval"], results_to_eval
            ))
            for label, eva_result in zip(st.session_state.selected_model_labels, evaluation_result):
                if eva_result and label in last_content:
                    last_content[label]["evaluation"] = "\n\n" + evaluator.format_response(eva_result)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

        st.session_state.state["evaluating"] = False
        st.rerun()

elif st.session_state.state["waiting_for_user_compare"]:
    user_input = st.chat_input("Type your message...")

    if user_input:
        append_message_compare("user", user_input)

        st.session_state.state.update({
            "pending_input_compare": user_input,
            "waiting_for_user_compare": False,
            "chat_has_started": True,
        })

        st.rerun()

else:
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            if not st.session_state.agents["compare"]:
                st.session_state.selected_model_labels = selected_model_labels
                st.session_state.agents["compare"] = []
                for model in st.session_state.models:
                    if model["label"] in st.session_state.selected_model_labels:
                        if model["provider"] == "openai":
                            st.session_state.agents["compare"].append(openAIAgent(api_key=config['openai_api_key'], model=model["model"]))
                        elif model["provider"] == "anthropic":
                            st.session_state.agents["compare"].append(claudeAgent(api_key=config['anthropic_api_key'], model=model["model"]))
                        elif model["provider"] == "gemini":
                            st.session_state.agents["compare"].append(geminiAgent(api_key=config['gemini_api_key'], model=model["model"]))

            results = asyncio.run(run_agents(st.session_state.state["pending_input_compare"], st.session_state.agents["compare"]))

            assistant_outputs = {}
            for model, result in zip(st.session_state.selected_model_labels, results):
                if isinstance(result, Exception):
                    logger.error(f"Error from {model}: {result}")
                    assistant_outputs[model] = {"result": f"Error: {result}"}
                else:
                    assistant_outputs[model] = {"result": result}

            logger.info(assistant_outputs)
            append_message_compare("assistant", assistant_outputs)

            if st.session_state.state["evaluate"]:
                if not st.session_state.agents["evaluator"]:
                    st.session_state.agents["evaluator"] = evaluatorAgent(
                        openai_api_key=config.get("openai_api_key", ""),
                        anthropic_api_key=config.get("anthropic_api_key", ""),
                        gemini_api_key=config.get("gemini_api_key", ""),
                    )
                st.session_state.state["pending_prompt_for_eval"] = st.session_state.state["pending_input_compare"]
                st.session_state.state["evaluating"] = True

            st.session_state.state.update({
                "pending_input_compare": None,
                "waiting_for_user_compare": True,
            })

            st.rerun()