import streamlit as st

def render():
    with st.sidebar:
        OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
    return OPENAI_API_KEY