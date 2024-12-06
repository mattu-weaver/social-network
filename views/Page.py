"""
Base class for streamlit pages.
"""
import streamlit as st

class Page():
    """
    Base class for streamlit pages.
    """
    def __init__(self, page_title):
        self.page_title = page_title


    def render_page(self):
        """
        Base render method for each page view.
        """
        st.header(self.page_title)