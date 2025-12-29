import pandas as pd
import streamlit as st
from models import retrain_model
from model_uploader import upload_model

st.set_page_config(
    page_title="FNOL Claims Intelligence System - Model Retraining",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title("🤖 Model Retraining Dashboard")

# Use cached data from session state (loaded in app.py)
uploaded_file = st.file_uploader("Upload New Dataset for Retraining", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    st.write(f"Total records: {len(data)}")
    st.write(f"Total columns: {len(data.columns)}")

    st.write("### Column Information")
    st.dataframe(data.dtypes)

    if st.button("Start Retraining"):
        with st.spinner("Retraining the model..."):
            result = retrain_model(data)
        st.success("Model retraining completed successfully!")
        st.write(f"Previous RMSE: {result['old_model_rmse']:.2f}")
        st.write(f"New RMSE: {result['new_model_rmse']:.2f}")
        if result["promoted"]:
            st.balloons()
            st.success("The new model has been promoted to production!")
            # Upload the new model to Hugging Face Hub
            upload_model()
        else:
            st.info("The new model did not outperform the current production model.")