import streamlit as st
from overview import load_data

st.set_page_config(
    page_title="FNOL Claims Intelligence System",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🚨 FNOL System")

data = load_data()
st.session_state.data = data

page = st.navigation([
    st.Page("pages/1_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/2_Visualizations.py", title="Visualizations", icon="📈"),
    st.Page("pages/3_Prediction.py", title="Prediction", icon="📋"),
    st.Page("pages/4_Retrain.py", title="Model Retraining", icon="🤖"),
])

page.run()