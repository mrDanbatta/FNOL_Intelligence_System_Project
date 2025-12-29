import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def claim_types(df):
    """
    Visualise the distribution of claim types in the dataset.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing claim-related columns.
    
    Returns:
    None: Displays a bar plot of claim type distribution.
    """
    st.subheader("📊 Claim Types Distribution")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.countplot(x='Weather_Condition', data=df, palette='viridis', ax=axes[0, 0])
    axes[0, 0].set_title('Frequency of Weather Conditions claims')
    axes[0, 0].tick_params(axis='x', rotation=45)

    sns.countplot(x='Traffic_Condition', data=df, palette='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Frequency of Traffic Conditions claims')
    axes[0, 1].tick_params(axis='x', rotation=45)

    sns.countplot(x='Claim_Type', data=df, palette='Greens', ax=axes[1, 0])
    axes[1, 0].set_title('Frequency of Claim Types')
    axes[1, 0].tick_params(axis='x', rotation=45)

    sns.countplot(x='Vehicle_Type', data=df, palette='Reds', ax=axes[1, 1])
    axes[1, 1].set_title('Frequency of Vehicle Types')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

def claim_amount_distribution(df):
    """
    Visualise the distribution of claim amounts in the dataset.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing claim-related columns.
    
    Returns:
    None: Displays histograms of claim amount distributions.
    """
    st.subheader("📊 Claim Amount Distribution")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(df['Ultimate_Claim_Amount'], bins=40, kde=True, color='#4472C4', ax=axes[0])
    axes[0].set_title('Distribution of Ultimate Claim Amount', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Claim Amount (£)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)

    sns.histplot(df['Estimated_Claim_Amount'], bins=40, kde=True, color='#FF7F0E', ax=axes[1])
    axes[1].set_title('Distribution of Estimated Claim Amount', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Claim Amount (£)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)


def data_distribution(df):
    """
    Visualise the distribution of key numerical features in the dataset.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing numerical columns.
    
    Returns:
    None: Displays histograms of numerical feature distributions.
    """
    st.subheader("📊 Data Distribution of Key Numerical Features")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.histplot(df['Driver_Age'], bins=30, kde=True, color='#70AD47', ax=axes[0, 0])
    axes[0, 0].set_xlabel('Driver Age', fontsize=11)
    axes[0, 0].set_title('Distribution of Driver Age', fontsize=13, fontweight='bold')

    sns.histplot(df['License_Years'], bins=30, kde=True, color='#5B9BD5', ax=axes[0, 1])
    axes[0, 1].set_xlabel('License Years', fontsize=11)
    axes[0, 1].set_title('Distribution of License Years', fontsize=13, fontweight='bold')

    sns.histplot(df['FNOL_delay'], bins=30, kde=True, color='#C5504F', ax=axes[1, 0])
    axes[1, 0].set_xlabel('FNOL Delay (Days)', fontsize=11)
    axes[1, 0].set_title('Distribution of FNOL Delay (Days)', fontsize=13, fontweight='bold')

    sns.histplot(df['Settlement_Days'], bins=30, kde=True, color='#A5611A', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Settlement Days', fontsize=11)
    axes[1, 1].set_title('Distribution of Settlement Days', fontsize=13, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)