import streamlit as st
import pandas as pd
import os

st.title("IND320 • Data Check v2")

# Load data
data_path = os.path.join("data", "open-meteo-subset.csv") 
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.subheader("metero_dataset")
    st.write(df.head())

    st.subheader("check the data")
    st.write(df.describe())
else:
     st.error("CSV file not found. Please check the 'data' folder.")










