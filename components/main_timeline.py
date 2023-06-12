import streamlit as st
from streamlit_timeline import timeline


def render(height=800):
    with open('src/main_timeline.json', "r") as f:
        data = f.read()
    return timeline(data, height=height)