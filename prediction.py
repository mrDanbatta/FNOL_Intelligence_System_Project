import pandas as pd
import numpy as np
from models import load_model, data_pipeline
import streamlit as st



def user_input(df):
    # st.title("🚗 FNOL Claim Amount Prediction")
    st.write("Provide the following details to predict the Ultimate Claim Amount:")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Claim Details")
        with st.form(key='claim_form'):
            col1a, col2a = st.columns(2)
            with col1a:
                claim_type = st.selectbox("Claim Type", df['Claim_Type'].unique())
                estimated_claim_amount = st.number_input("Estimated Claim Amount (£)", min_value=0.0, value=1000.0, step=100.0)
                traffic_condition = st.selectbox("Traffic Condition", df['Traffic_Condition'].unique())
                weather_condition = st.selectbox("Weather Condition", df['Weather_Condition'].unique())
            with col2a:
                driver_age = st.number_input("Driver Age", min_value=16, max_value=100, value=30, step=1)
                license_years = st.number_input("Years with License", min_value=0, max_value=90, value=10, step=1)
                vehicle_type = st.selectbox("Vehicle Type", df['Vehicle_Type'].unique())
                vehicle_year = st.number_input("Vehicle Year", min_value=1980, max_value=2025, value=2015, step=1)
            submit_button = st.form_submit_button(label='Predict Ultimate Claim Amount')
    with col2:
        st.subheader("Input Summary")
        st.write(f"**Claim Type:** {claim_type}")
        st.write(f"**Estimated Claim Amount (£):** {estimated_claim_amount:,.2f}")
        st.write(f"**Traffic Condition:** {traffic_condition}")
        st.write(f"**Weather Condition:** {weather_condition}")
        st.write(f"**Driver Age:** {driver_age}")
        st.write(f"**Years with License:** {license_years}")
        st.write(f"**Vehicle Type:** {vehicle_type}")
        st.write(f"**Vehicle Year:** {vehicle_year}") 

    if submit_button:
        try:
            st.info("Predicting Ultimate Claim Amount...", icon="⏳")
            input_data = pd.DataFrame({
                'Claim_Type': [str(claim_type)],
                'Estimated_Claim_Amount': [estimated_claim_amount],
                'Traffic_Condition': [str(traffic_condition)],
                'Weather_Condition': [str(weather_condition)],
                'Driver_Age': [driver_age],
                'License_Years': [license_years],
                'Vehicle_Type': [str(vehicle_type)],
                'Vehicle_Year': [vehicle_year]
            })

            model, feature_columns = load_model()

            input_data_encoded = data_pipeline(input_data)
            input_data_encoded = input_data_encoded.reindex(columns=feature_columns, fill_value=0)

            prediction = model.predict(input_data_encoded)
            prediction = np.expm1(prediction)  # Reverse log1p transformation

            col1aa, col2aa, col3aa = st.columns(3)
            with col1aa:
                st.metric("Estimated Claim Amount (£)", f"£{estimated_claim_amount:,.2f}")
            with col2aa:
                st.metric("Predicted Ultimate Claim Amount (£)", f"£{prediction[0]:,.2f}",
                            delta=f"£{(prediction[0] - estimated_claim_amount):,.2f} vs Estimate")
            with col3aa:
                st.metric("Variance (£)", f"£{(prediction[0] - estimated_claim_amount):,.2f}")
            
            st.success(f"Predicted Ultimate Claim Amount: £{prediction[0]:,.2f}", icon="✅")
            st.balloons()

            if prediction[0] > estimated_claim_amount:
                st.warning("The predicted ultimate claim amount exceeds the estimated amount. Review the claim details for accuracy.", icon="⚠️")   
            else:
                st.info("The predicted ultimate claim amount is within the estimated range.", icon="ℹ️")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}", icon="❌")
            st.info("Please ensure all inputs and models are valid and try again.", icon="ℹ️")


