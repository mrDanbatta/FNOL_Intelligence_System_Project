import streamlit as st
import pandas as pd
from visualisation import claim_types, claim_amount_distribution, data_distribution

st.set_page_config(
    page_title="FNOL Claims Intelligence System - Visualizations",
    page_icon=":bar_chart:",
    layout="wide"
)
data = st.session_state.data

st.title("📊 Visualizations")
claim_amount_distribution(data)
claim_types(data)
data_distribution(data)


