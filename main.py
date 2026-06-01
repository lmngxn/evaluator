import streamlit as st
import logging
from utils.utility import init_session, check_password, setup_logging, load_config

st.set_page_config(
    page_title="LLM App",
    layout="wide",
)

# Do not continue if valid_password is not True.
if not check_password():
    st.stop()

setup_logging()

logger = logging.getLogger(__name__)
logger.info("App started")

config = load_config()
init_session(config)

pages = {
    "Main": [
        st.Page("pages_app/compare_models.py", title="Compare Models"),
        st.Page("pages_app/research.py", title="Research Agent"),
    ],
    # "Admin": [
    #     st.Page("pages_app/settings.py", title="Settings", icon="⚙️"),
    # ],
}

pg = st.navigation(pages)
pg.run()

    
