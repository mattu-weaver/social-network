"""
Renders the DataLoadPage.
"""
import streamlit as st
import pandas as pd
from .Page import Page

class DataLoadPage(Page):
    """
    Class for the data load page.
    """
    def __init__(self):
        """
        Constructor for the DataLoadPage class.
        """
        super().__init__('Upload Data')
        self.uploaded_file = None
        self.df = None
        self.feature_descriptions = {}

    def create_default_feature_descriptions(self):
        """
        Creates a dictionary of feature descriptions.
        """
        if self.df is not None:
            for col in self.df.columns:
                self.feature_descriptions[col] = 'Not set'
        return self.feature_descriptions
    
    
    def render_page(self):
        """
        Renders the DataLoadPage page from which other views may be selected.
        """
        super().render_page()
        self.uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if self.uploaded_file is not None:
            self.df = pd.read_csv(self.uploaded_file)

            self.create_default_feature_descriptions()

            st.write(self.feature_descriptions)

            st.dataframe(self.df.head())
            st.write(f'Number of columns: {len(self.df.columns)}')
            st.write(f'Number of rows: {len(self.df)}')
            st.session_state['logger'].info(f'The following file was uploaded: {self.uploaded_file.name}')

            feature = st.selectbox('Select a feature', self.df.columns)
            description = st.text_area('Provide a description for the selected feature')

            if description:
                st.write(f'Description for {feature}: {description}')
                st.session_state['logger'].info(f'Description for {feature}: {description}')
        else:
            st.write('No file uploaded')
            st.session_state['logger'].info('No file was uploaded')
