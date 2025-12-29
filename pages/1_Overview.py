import streamlit as st
from overview import load_data, overview, claims_analysis, weather_analysis, get_statistics
import pandas as pd

st.set_page_config(
    page_title="FNOL Claims Intelligence System - Overview",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title("🚨 First Notice of Loss (FNOL) Claims Intelligence System")

data = load_data()
stats = get_statistics(data)
overview(stats)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.header("📋 Claims Analysis by Type")
    claims_analysis(stats)
with col2:
    st.header("🌧️ Weather Condition Analysis")
    weather_analysis(stats)
