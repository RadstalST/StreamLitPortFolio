import streamlit as st
import constants 
import streamlit.components.v1 as components

def sidebar_footer():
    with st.sidebar:
        st.header("Contact Me")

        st.write("""
        Email: Suchat.tangjarukij@gmail.com
        """)
        components.html(
            constants.linkedin_embed,
            height=600
        )

