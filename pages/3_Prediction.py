import streamlit as st
from prediction import user_input

st.set_page_config(
    page_title="FNOL Claims Intelligence System - Prediction",
    page_icon=":bar_chart:",
    layout="wide"
)
st.title("📋 Claims Prediction Dashboard")

data = st.session_state.data

user_input(data)