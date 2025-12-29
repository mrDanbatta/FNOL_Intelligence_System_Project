import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


@st.cache_data
def load_data():
    """Load and cache the claims data"""
    return pd.read_csv('data/claims_policy_merged_cleaned.csv')

def overview(stats):
    """
    Display an overview of the dataset including basic statistics and visualizations.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to analyze.
    
    Returns:
    None: Displays the overview in a Streamlit app.
    """
    st.title("📊 Customer Claims Overview")

    st.subheader("A comprehensive view of the dataset for insurance claims and key metrics")

    st.write("### 📈 Key Statistics:")

    # create 3 columns to display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lowest Claim Amount", f"£{stats['Min Claim Amount']:.2f}")
    with col2:
        st.metric("Highest Claim Amount", f"£{stats['Max Claim Amount']:.2f}")
    with col3:
        st.metric("Min Driver Age", f"{stats['Min_Driver_Age']} years")
    with col4:
        st.metric("Max Driver Age", f"{stats['Max_Driver_Age']} years")



def claims_analysis(stats):
    st.write("### 🔍 Claims Analysis by Type:")
    claim_types = [key.split(" for ")[1] for key in stats.keys() if "Highest Claim Amount for" in key]
    
    # Create a DataFrame for better table display
    claims_data = []
    for claim in claim_types:
        claims_data.append({
            "Claim Type": claim,
            "Highest Amount": f"£{stats[f'Highest Claim Amount for {claim}']:.2f}",
            "Lowest Amount": f"£{stats[f'Lowest Claim Amount for {claim}']:.2f}",
            "Average Amount": f"£{stats[f'Average Estimated Claim Amount for {claim}']:.2f}",
            "Count of Claims": stats[f'Count of Claims for {claim}']
        })
    
    claims_df = pd.DataFrame(claims_data)
    st.dataframe(claims_df, use_container_width=True, hide_index=True)

def weather_analysis(stats):
    st.write("### 🌦️ Weather Analysis:")
    weather_conditions = [key.split(" in ")[1] for key in stats.keys() if "Average Claim Amount in" in key]
    
    # Create a DataFrame for better table display
    weather_data = []
    for condition in weather_conditions:
        weather_data.append({
            "Weather Condition": condition,
            "Average Claim Amount": f"£{stats[f'Average Claim Amount in {condition}']:.2f}",
            "Median Claim Amount": f"£{stats[f'Median Claim Amount in {condition}']:.2f}",
            "Highest Claim Amount": f"£{stats[f'Highest Claim Amount in {condition}']:.2f}",
            "Lowest Claim Amount": f"£{stats[f'Lowest Claim Amount in {condition}']:.2f}",
            "Total Claims": stats[f'Total Claims in {condition}']
        })
    
    weather_df = pd.DataFrame(weather_data)
    st.dataframe(weather_df, use_container_width=True, hide_index=True)


def get_statistics(df):
    stats = {
        "Min Claim Amount": df['Ultimate_Claim_Amount'].min(),
        "Max Claim Amount": df['Ultimate_Claim_Amount'].max(),
        "Min_Driver_Age": df['Driver_Age'].min(),
        "Max_Driver_Age": df['Driver_Age'].max()
    }
    claims = df['Claim_Type'].unique()
    for claim in claims:
        claim_data = df[df['Claim_Type'] == claim]
        stats[f"Highest Claim Amount for {claim}"] = claim_data['Ultimate_Claim_Amount'].max()
        stats[f"Lowest Claim Amount for {claim}"] = claim_data['Ultimate_Claim_Amount'].min()
        stats[f"Average Estimated Claim Amount for {claim}"] = claim_data['Estimated_Claim_Amount'].mean()
        stats[f"Median Estimated Claim Amount for {claim}"] = claim_data['Estimated_Claim_Amount'].median()
        stats[f"Average Ultimate Claim Amount for {claim}"] = claim_data['Ultimate_Claim_Amount'].mean()
        stats[f"Count of Claims for {claim}"] = len(claim_data)

    weather_conditions = df['Weather_Condition'].unique()
    for condition in weather_conditions:
        condition_data = df[df['Weather_Condition'] == condition]
        stats[f"Average Claim Amount in {condition}"] = condition_data['Ultimate_Claim_Amount'].mean()
        stats[f"Median Claim Amount in {condition}"] = condition_data['Ultimate_Claim_Amount'].median()
        stats[f"Total Claims in {condition}"] = len(condition_data)
        stats[f"Average Estimated Claim Amount in {condition}"] = condition_data['Estimated_Claim_Amount'].mean()
        stats[f"Median Estimated Claim Amount in {condition}"] = condition_data['Estimated_Claim_Amount'].median()
        stats[f"Highest Claim Amount in {condition}"] = condition_data['Ultimate_Claim_Amount'].max()
        stats[f"Lowest Claim Amount in {condition}"] = condition_data['Ultimate_Claim_Amount'].min()


    return stats

