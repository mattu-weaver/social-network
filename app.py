"""
Entry point for the Streamlit application.
"""
import streamlit as st
import toml

import utils as ut
from views.DataLoadPage import DataLoadPage
from views.FriendshipPage import FriendshipPage


# Dictionary of application views (pages).

page_map = {
    'Load Data': DataLoadPage(),
    'The Friendship Paradox': FriendshipPage(),

    # Add pages here as required
}


def main():
    """
    Main is called when the script is run.
    """
    log_cfg = toml.load('.streamlit/config.toml')['logger']
    log = ut.configure_logging(log_cfg['format'], log_cfg['level'])
    ut.set_initial_session_state('logger', log)

    st.set_page_config(page_title="GenAI Demo", layout="wide")
    page_name = st.sidebar.selectbox('Select page', options=list(page_map))

    st.session_state['page_name'] = page_name
    st.session_state['current_page'] = page_map[page_name]
    st.session_state['current_page'].render_page()

if __name__ == "__main__":
    main()
