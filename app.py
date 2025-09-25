import streamlit as st
import pandas as pd
import os

# app setup
st.set_page_config(page_title="IND320 • Weather mini-app", layout="wide")
st.title("IND320 • Data Check v2")

# Load data
data_path = os.path.join("data", "open-meteo-subset.csv") 

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Put the file in a 'data' folder.")
    df = pd.read_csv(path)
    # Ensure 'time' is datetime if present
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df

# sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Data table & summary", "Plots (interactive)", "About / Notes"],
    index=0
)

# page 1: Home
if page == "Home":
    st.subheader("Welcome to the Weather Data Mini-App")
    st.write(
        "Use the sidebar to navigate. Page 2 shows the table and summary. "
        "Page 3 will host plots with a dropdown and a month selector slider. "
        "Data is cached for speed."
    )
    st.info("CSV path expected: `data/open-meteo-subset.csv`")

# page 2: Data table & summary
elif page == "Data table & summary":


    try:
        df = load_data(data_path)
        st.subheader("First rows")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Describe")
        st.dataframe(df.describe(include='all').transpose(), use_container_width=True)

    except Exception as e:
        st.error(str(e))

# page 3: Plots (interactive)
elif page == "Plots (interactive)":
    st.subheader("Interactive plots")
    st.write("Placeholder for now. We’ll add:")
    st.markdown(
        "- A selectbox to choose a single column or all columns\n"
        "- A month slider to subset the data\n"
        "- Line charts per column and combined"
    )

# page 4: About / Notes
else:
    st.subheader("About this mini-app")
    st.write(
        "Assignment structure: 4 pages, cached CSV, table + summary, interactive plots, and a README."
    )











