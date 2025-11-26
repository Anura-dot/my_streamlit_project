"""
IND320 Assignment 4 - Streamlit Application
Complete implementation combining Assignment 3 work with Assignment 4 requirements

Structure:
- Section 1: Data Overview (info page)
- Section 2: Exploratory (weather + energy plots)
- Section 3: Advanced (STL, outliers, correlation)
- Section 4: Regional & Predictive (map, forecasting - placeholders)

Global Setting: Year range only (2021-2024)
Each page has own filters (area, group, variables)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Analysis libraries
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram
from scipy.fftpack import dct, idct
from sklearn.neighbors import LocalOutlierFactor

# Data libraries
from pymongo import MongoClient
import requests
from streamlit_plotly_events import plotly_events  

# =================== PAGE CONFIG ===================
st.set_page_config(
    page_title="IND320 Assignment 4",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== CONSTANTS ===================
PRICE_AREA_COORDS = {
    "NO1": (59.91, 10.75),    # Oslo
    "NO2": (58.15, 8.00),     # Kristiansand  
    "NO3": (63.43, 10.39),    # Trondheim
    "NO4": (69.65, 18.96),    # Tromsø
    "NO5": (60.39, 5.32),     # Bergen
}

CITIES = {
    "Oslo": (59.91, 10.75),
    "Kristiansand": (58.15, 8.00),
    "Trondheim": (63.43, 10.39),
    "Tromsø": (69.65, 18.96),
    "Bergen": (60.39, 5.32),
}

HOURLY_VARS = "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,wind_direction_10m"
PRICE_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# =================== MONGODB CONNECTION ===================
@st.cache_resource
def init_mongodb_connection():
    """Initialize MongoDB connection using secrets"""
    try:
        if "mongodb" not in st.secrets:
            return None, "MongoDB secrets not configured"
        
        connection_string = st.secrets["mongodb"]["connection_string"]
        client = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        client.admin.command('ping')
        return client, None
        
    except Exception as e:
        return None, str(e)

# =================== DATA LOADERS ===================

@st.cache_data(ttl=3600, show_spinner="Loading production data...")
def load_mongodb_production(_client):
    """Load production data from MongoDB"""
    if _client is None:
        return pd.DataFrame()
    
    try:
        db_name = st.secrets["mongodb"]["database_name"]
        prod_coll = st.secrets["mongodb"]["production_collection"]
        
        db = _client[db_name]
        collection = db[prod_coll]
        
        data = list(collection.find({}, {"_id": 0}))
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Normalize column names
        df = df.rename(columns={
            "pricearea": "priceArea",
            "productiongroup": "productionGroup",
            "starttime": "startTime",
            "endtime": "endTime",
            "lastupdatedtime": "lastUpdatedTime",
            "quantitykwh": "quantityKwh"
        })
        
        # Convert timestamps
        df['startTime'] = pd.to_datetime(df['startTime'])
        df['month'] = df['startTime'].dt.to_period('M').astype(str)
        df['year'] = df['startTime'].dt.year
        
        return df
        
    except Exception as e:
        st.error(f"Error loading production data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Loading consumption data...")
def load_mongodb_consumption(_client):
    """Load consumption data from MongoDB"""
    if _client is None:
        return pd.DataFrame()
    
    try:
        db_name = st.secrets["mongodb"]["database_name"]
        cons_coll = st.secrets["mongodb"]["consumption_collection"]
        
        db = _client[db_name]
        collection = db[cons_coll]

        # Show progress for large datasets
        total_docs = collection.count_documents({})
        
        if total_docs > 100000:
            st.info(f"Loading {total_docs:,} records... This may take a moment.")
        
        data = list(collection.find({}, {"_id": 0}))
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Normalize column names
        df = df.rename(columns={
            "pricearea": "priceArea",
            "consumptiongroup": "consumptionGroup",
            "starttime": "startTime",
            "endtime": "endTime",
            "lastupdatedtime": "lastUpdatedTime",
            "quantitykwh": "quantityKwh"
        })
        
        # Convert timestamps
        df['startTime'] = pd.to_datetime(df['startTime'])
        df['month'] = df['startTime'].dt.to_period('M').astype(str)
        df['year'] = df['startTime'].dt.year
        
        return df
        
    except Exception as e:
        st.error(f"Error loading consumption data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_openmeteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo API"""
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/era5"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly={HOURLY_VARS}"
            f"&start_date={start}&end_date={end}&timezone=Europe/Oslo"
        )
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        j = r.json()

        hourly = j.get("hourly", {})
        cols = {"time": pd.to_datetime(hourly["time"])}

        for var in HOURLY_VARS.split(","):
            if var in hourly:
                cols[var] = hourly[var]

        df = pd.DataFrame(cols).sort_values("time")
        return df
        
    except Exception as e:
        st.error(f"Weather API error: {e}")
        return pd.DataFrame()


# =================== ANALYSIS FUNCTIONS ===================

def stl_decompose(series: pd.Series, period: int, seasonal: int, trend: int, robust: bool = True):
    """Perform STL decomposition on time series"""
    s = series.astype(float).dropna()
    if len(s) < 2 * period:
        st.warning(f"⚠️ Not enough data for STL (need at least {2*period} points)")
        return pd.DataFrame()
    
    res = STL(s, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()
    return pd.DataFrame({
        "observed": res.observed,
        "trend": res.trend,
        "seasonal": res.seasonal,
        "resid": res.resid
    }, index=s.index)


def make_spectrogram(series: pd.Series, fs: float = 1.0, nperseg: int = 168, noverlap: int = 84):
    """Generate spectrogram from time series"""
    y = series.astype(float).ffill().values
    if len(y) < nperseg:
        st.warning(f"⚠️ Not enough data for spectrogram (need at least {nperseg} points)")
        return None, None, pd.DataFrame()
    
    f, t, Sxx = spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
    return f, t, pd.DataFrame(Sxx, index=f, columns=t)


def dct_lowpass_baseline(series: pd.Series, keep_k: int = 168) -> pd.Series:
    """Smooth a series using the first keep_k low-frequency DCT coefficients"""
    s = series.dropna().astype(float)
    if s.empty:
        return s

    c = dct(s.values, norm="ortho")
    k = min(keep_k, len(c))
    mask = np.zeros_like(c)
    mask[:k] = 1
    c_lp = c * mask
    
    baseline_vals = idct(c_lp, norm="ortho")
    return pd.Series(baseline_vals, index=s.index)


def dct_spc_on_series(series: pd.Series, keep_k: int = 168, sd: float = 3.0):
    """Full DCT-SPC pipeline for outlier detection"""
    s = series.dropna().astype(float)

    if s.empty:
        return pd.DataFrame(), {"n": 0, "n_outliers": 0, "pct_outliers": 0.0, "sigma_satv": 0.0}

    baseline = dct_lowpass_baseline(s, keep_k=keep_k)
    satv = s - baseline
    sigma_satv = float(satv.std(ddof=1))

    upper = baseline + sd * sigma_satv
    lower = baseline - sd * sigma_satv
    outlier_mask = (s > upper) | (s < lower)

    df_spc = pd.DataFrame({
        "time": s.index,
        "value": s.values,
        "baseline": baseline.values,
        "lower": lower.values,
        "upper": upper.values,
        "satv": satv.values,
        "outlier": outlier_mask.values,
    })

    summary = {
        "n": len(df_spc),
        "n_outliers": int(outlier_mask.sum()),
        "pct_outliers": float(100 * outlier_mask.sum() / len(df_spc)),
        "sigma_satv": sigma_satv,
    }

    return df_spc, summary


def lof_anomalies(values: pd.Series, k: int = 20):
    """Detect anomalies using Local Outlier Factor"""
    X = values.astype(float).ffill().to_frame()
    lof = LocalOutlierFactor(n_neighbors=k, contamination="auto")
    yhat = lof.fit_predict(X.values)
    score = np.abs(lof.negative_outlier_factor_)
    return pd.DataFrame({
        "value": values.values, 
        "LOF": score, 
        "flag": (yhat == -1)
    }, index=values.index)


def sliding_window_correlation(weather_series, energy_series, window, lag=0):
    """Compute rolling correlation between weather and energy with optional lag"""
    if lag != 0:
        energy_shifted = energy_series.shift(lag)
    else:
        energy_shifted = energy_series
    
    df = pd.DataFrame({
        'weather': weather_series,
        'energy': energy_shifted
    }).dropna()
    
    if len(df) < window:
        return pd.DataFrame(columns=['time', 'correlation'])
    
    corr = df['weather'].rolling(window).corr(df['energy'])
    
    return pd.DataFrame({
        'time': corr.index, 
        'correlation': corr.values
    }).dropna()

# =================== SESSION STATE INITIALIZATION ===================

if "year_range" not in st.session_state:
    st.session_state.year_range = (2021, 2024)

if "current_page" not in st.session_state:
    st.session_state.current_page = "P1: Home & Overview"

# =================== CUSTOM CSS FOR SIDEBAR ===================

st.markdown("""
    <style>
        /* Sidebar width */
        [data-testid="stSidebar"] {
            min-width: 250px;
            max-width: 250px;
        }
        
        /* Reduce internal padding */
        [data-testid="stSidebar"] > div:first-child {
            padding: 1rem 0.5rem;
        }
        
        /* Compact section headers */
        [data-testid="stSidebar"] h3 {
            font-size: 0.90rem;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        /* Compact buttons */
        [data-testid="stSidebar"] button {
            padding: 0.1rem 0.3rem;
            font-size: 0.80rem;
        }
        
        /* Reduce spacing between elements */
        [data-testid="stSidebar"] .element-container {
            margin-bottom: 0.1rem;
        }
    </style>
""", unsafe_allow_html=True)

# =================== GLOBAL SIDEBAR SETTINGS ===================

st.sidebar.title("🔧 IND320 Assignment 4")
st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Global Settings")

# Temporary year range (doesn't trigger reload)
temp_year_range = st.sidebar.select_slider(
    "Data Years",
    options=[2021, 2022, 2023, 2024],
    value=st.session_state.year_range,
    key="temp_year_range"
)

# Apply button
if st.sidebar.button("Apply Year Range", type="primary"):
    st.session_state.year_range = temp_year_range
    st.rerun()

# Show current active range
st.sidebar.caption(f"📅 Active: {st.session_state.year_range[0]}-{st.session_state.year_range[1]}")

st.sidebar.markdown("---")

# =================== PAGE FUNCTIONS ===================

def page_home():
    """P1: Home & Overview - Information only"""
    st.title("📊 Weather and Energy Data in Norway")
    
    st.markdown("""
    
    This application integrates weather and energy data to demonstrate advanced data engineering concepts.
    
    ---
    
    ### 📋 Data Sources
    
    **1. Weather Data** 🌤️
    - **Source:** Open-Meteo API (ERA5 Archive)
    - **Variables:** Temperature, precipitation, wind speed, wind gusts, wind direction
    - **Coverage:** Hourly data for 5 Norwegian cities (2021-2024)
    - **Cities:** Oslo (NO1), Kristiansand (NO2), Trondheim (NO3), Tromsø (NO4), Bergen (NO5)
    
    **2. Energy Production Data** ⚡
    - **Source:** Elhub API
    - **Dataset:** PRODUCTION_PER_GROUP_MBA_HOUR
    - **Groups:** Hydro, thermal, wind, solar, other
    - **Coverage:** Hourly production by price area (2021-2024)
    
    **3. Energy Consumption Data** 🔌
    - **Source:** Elhub API
    - **Dataset:** CONSUMPTION_PER_GROUP_MBA_HOUR
    - **Groups:** cabin, household, primary, secondary, tertiary            
    - **Coverage:** Hourly consumption by price area (2021-2024)
    
     ### 🔄 Data Pipeline
    
    1. **Data Retrieval:** Elhub API → Raw JSON data
    2. **Storage:** Cassandra database (initial storage)
    3. **Transformation:** Apache Spark (data processing & aggregation)
    4. **Loading:** MongoDB Atlas (final curated data)
    5. **Visualization:** Streamlit application
    
    ---
    
    ### 📊 Available Analyses
    
    **Section 2: Exploratory**
    - Interactive weather plots with city selection
    - Energy production dashboard with pie charts and time series
    
    **Section 3: Advanced**
    - STL decomposition (trend, seasonal, residual components)
    - Spectrogram analysis (frequency domain)
    - Outlier detection using DCT-SPC method
    - Anomaly detection using Local Outlier Factor
    - Weather-Energy correlation analysis with sliding windows
    
    **Section 4: Regional & Predictive**
    - Regional map analysis (GeoJSON price areas)
    - SARIMAX forecasting models
    
    ---
    
    ### 🎯 How to Use This App
    
    1. **Select year range** using the slider in the sidebar (Global Settings)
    2. **Navigate** to different pages using the sidebar menu
    3. **Each page** has its own filters for price area, production group, etc.
    4. **Explore** visualizations and analyses
    
    ---
    
    👈 **Use the navigation menu to explore the app**
    """)


def page_weather_plots():
    """P2: Weather Plots - Interactive visualizations"""
    st.title("🌤️ Weather Data Visualization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        use_city = st.checkbox("Use predefined city", value=True)
        
        if use_city:
            city = st.selectbox("Select city", list(CITIES.keys()))
            lat, lon = CITIES[city]
            st.caption(f"📍 {lat}°N, {lon}°E")
        else:
            lat = st.number_input("Latitude", value=59.91, format="%.2f")
            lon = st.number_input("Longitude", value=10.75, format="%.2f")
    
    year_start, year_end = st.session_state.year_range
    
    try:
        with st.spinner(f"Loading weather data for {year_start}-{year_end}..."):
            dfs = []
            num_years = year_end - year_start + 1
            progress_bar = st.progress(0, text="Loading weather data...")
            
            for i, year in enumerate(range(year_start, year_end + 1)):
                progress_bar.progress(
                    (i + 1) / num_years, 
                    text=f"Loading weather data for {year}... ({i+1}/{num_years})"
                )
                df_year = load_openmeteo(lat, lon, f"{year}-01-01", f"{year}-12-31")
                if not df_year.empty:
                    dfs.append(df_year)
            
            progress_bar.empty()
            
            if not dfs:
                st.error("❌ No weather data could be loaded")
                st.info("💡 Please check your internet connection or try a different location")
                return
            
            df = pd.concat(dfs, ignore_index=True)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            
    except Exception as e:
        st.error("❌ Failed to load weather data")
        st.info("💡 This might be due to network issues or API limitations")
        with st.expander("Technical Details"):
            st.code(f"Error: {e}")
        return
    
    df['month'] = df['time'].dt.to_period('M')
    months = sorted(df['month'].unique())
    month_labels = [str(m) for m in months]
    
    with col2:
        if len(month_labels) > 1:
            start_month, end_month = st.select_slider(
                "Select time range",
                options=month_labels,
                value=(month_labels[0], month_labels[-1])
            )
        else:
            start_month = end_month = month_labels[0]
    
    mask = (df['month'] >= pd.Period(start_month)) & (df['month'] <= pd.Period(end_month))
    df_filtered = df[mask].copy()
    
    st.caption(f"Showing {len(df_filtered):,} hourly records from {start_month} to {end_month}")
    
    num_cols = df_filtered.select_dtypes("number").columns.tolist()
    
    # Add sparklines section
    st.subheader("📊 Variable Overview with Sparklines")
    
    # Create a container for all sparklines
    sparkline_data = []
    
    for col_name in num_cols:
        col_data = df_filtered[col_name].dropna()
        if len(col_data) > 0:
            sparkline_data.append({
                'name': col_name,
                'data': col_data.values,
                'time': df_filtered['time'].values
            })
    
    # Create combined sparkline figure
    fig_spark = make_subplots(
        rows=len(sparkline_data),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[s['name'] for s in sparkline_data]
    )
    
    for i, spark_info in enumerate(sparkline_data, start=1):
        fig_spark.add_trace(
            go.Scatter(
                x=spark_info['time'],
                y=spark_info['data'],
                mode='lines',
                line=dict(color='#e74c3c', width=1),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.1)',
                name=spark_info['name'],
                hovertemplate="<b>%{x|%b %d, %Y, %H:%M}</b><br>" +
                             f"{spark_info['name']}: %{{y:.2f}}<br>" +
                             '<extra></extra>'
            ),
            row=i,
            col=1
        )
    
    # Update layout
    fig_spark.update_layout(
        height=len(sparkline_data) * 80,
        margin=dict(l=20, r=20, t=40, b=40),
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    # Update all y-axes to hide ticks
    fig_spark.update_yaxes(showticklabels=False, showgrid=False)
    
    # Update x-axis for the bottom subplot only
    fig_spark.update_xaxes(
        showgrid=False,
        showticklabels=True,
        row=len(sparkline_data),
        col=1
    )
    
    # Update subplot titles styling
    fig_spark.update_annotations(
        font=dict(size=10),
        xanchor='left',
        x=0
    )
    
    st.plotly_chart(fig_spark, use_container_width=True)
    
    st.divider()
    
    # Original detailed plot section
    var_choice = st.selectbox(
        "Select variable for detailed plot",
        ["All variables (normalized)"] + num_cols,
        index=0
    )
    
    if var_choice == "All variables (normalized)":
        df_norm = df_filtered[['time'] + num_cols].copy()
        
        for col in num_cols:
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()
            if col_max > col_min:
                df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
            else:
                df_norm[col] = 0.5
        
        df_long = df_norm.melt(id_vars='time', var_name='variable', value_name='value')
        
        fig = px.line(
            df_long,
            x='time',
            y='value',
            color='variable',
            title=f"All Weather Variables (Normalized) — {start_month} to {end_month}",
            labels={'value': 'Normalized Value (0-1)', 'time': 'Date'}
        )
        
    else:
        fig = px.line(
            df_filtered,
            x='time',
            y=var_choice,
            title=f"{var_choice} — {start_month} to {end_month}",
            labels={var_choice: var_choice, 'time': 'Date'}
        )
    
    fig.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def page_energy_dashboard():
    """P3: Energy Dashboard - Production visualization"""
    st.title("⚡ Energy Production Dashboard")
    
    client, error = init_mongodb_connection()
    if client is None:
        st.error(f"❌ MongoDB connection error: {error}")
        st.info("""
        **Setup Instructions:**
        1. Create `.streamlit/secrets.toml` file
        2. Add MongoDB connection details
        3. Restart the app
        """)
        return
    
    df = load_mongodb_production(client)
    
    if df.empty:
        st.warning("⚠️ No production data available")
        return
    
    year_start, year_end = st.session_state.year_range
    df = df[(df['year'] >= year_start) & (df['year'] <= year_end)]
    
    left, right = st.columns(2)
    
    with left:
        st.subheader("Production by Price Area")
        
        areas = sorted(df['priceArea'].dropna().unique().tolist())
        area = st.radio("Select price area", areas, index=0 if areas else None)
        
        pie_df = (
            df[df['priceArea'] == area]
            .groupby('productionGroup', as_index=False)['quantityKwh']
            .sum()
            .rename(columns={'productionGroup': 'Group', 'quantityKwh': 'Total (kWh)'})
            .sort_values('Total (kWh)', ascending=False)
        )
        
        fig_pie = px.pie(
            pie_df,
            values='Total (kWh)',
            names='Group',
            title=f"Production Distribution — {area}<br>({year_start}-{year_end})",
            hole=0.2
        )
        fig_pie.update_layout(
            title_x=0,  # Position at left (not 1)
            title_xanchor='left',  # Anchor point for the title alignment
            title_y=0.98,  # Move title down slightly (default is around 1)
            margin=dict(l=20, r=20, t=80, b=20)  # Add left margin to prevent cutoff
        )
        # ✅ outside labels + adjust spacing
        fig_pie.update_traces(
            textposition='outside',  # Labels outside the chart
            textinfo='label+percent',
            textfont=dict(size=12),
            marker=dict(line=dict(color='white', width=2)),
            pull=[0.05, 0.02, 0, 0, 0]  # Slightly separate top slices
        )
        
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="right",
                x=0.5
            ),
            height=500,  # ✅ Increased from 500
            margin=dict(
                t=150,   # ✅ More top margin for title
                l=0,
                r=0,
                b=0   # ✅ More bottom margin for legend
            ),
            title=dict(
                y=0.98,  # ✅ Push title to very top
                x=0.5,
                xanchor='right',
                yanchor='top',
                font=dict(size=16)
            )
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with right:
        st.subheader("Hourly Production by Group & Month")
        
        groups = sorted(df[df['priceArea'] == area]['productionGroup'].dropna().unique().tolist())
        
        selected_groups = st.pills(
            "Select production group(s):",
            options=groups,
            default=[groups[0]] if groups else [] ,
            selection_mode="multi"
        )
        
        months = sorted(df['month'].dropna().unique().tolist())
        if not months:
            st.warning("No monthly data available")
            return
        
        month = st.selectbox("Select month", options=months, index=0)
        
        if selected_groups and month:
            line_df = df[
                (df['priceArea'] == area) &
                (df['productionGroup'].isin(selected_groups)) &
                (df['month'] == month)
            ].copy()
            
            if line_df.empty:
                st.warning(f"No data for selected filters")
            else:
                line_df = line_df.sort_values('startTime')
                
                fig_line = px.line(
                    line_df,
                    x='startTime',
                    y='quantityKwh',
                    color='productionGroup',
                    title=f"Hourly Production — {month} • {area}",
                    labels={'startTime': 'Date & Time', 'quantityKwh': 'Production (kWh)', 'productionGroup': 'Group'}
                )
                
                fig_line.update_layout(hovermode='x unified', height=420)
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Select at least one production group and a month")
    
    with st.expander("📚 Data Source Information"):
        st.markdown(f"""
        **Source:** Elhub API - `PRODUCTION_PER_GROUP_MBA_HOUR`
        
        **Years:** {year_start} - {year_end}
        
        **Processing Pipeline:**
        1. Data retrieved from Elhub API
        2. Stored in Cassandra
        3. Transformed with Apache Spark
        4. Loaded into MongoDB
        5. Visualized in Streamlit
        
        **Total Records:** {len(df):,}
        """)


def page_stl_spectrogram():
    """P4: STL & Spectrogram - Time series decomposition"""
    st.title("📈 Time Series Analysis: STL & Spectrogram")
    
    client, error = init_mongodb_connection()
    if client is None:
        st.error(f"❌ MongoDB connection error: {error}")
        return
    
    df = load_mongodb_production(client)
    
    if df.empty:
        st.warning("⚠️ No data available")
        return
    year_start, year_end = st.session_state.year_range
    df = df[(df['year'] >= year_start) & (df['year'] <= year_end)]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        areas = sorted(df['priceArea'].unique().tolist())
        area = st.selectbox("Price area", areas, index=0)
    
    with col2:
        groups = sorted(df[df['priceArea'] == area]['productionGroup'].unique().tolist())
        group = st.selectbox("Production group", groups, index=0 if groups else None)
    
    with col3:
        agg_choice = st.selectbox("Aggregation", ["hourly", "daily", "weekly"], index=0)
    
    df_filtered = df[(df['priceArea'] == area) & (df['productionGroup'] == group)].copy()
    df_filtered = df_filtered.set_index('startTime').sort_index()
    
    if agg_choice == "daily":
        series = df_filtered['quantityKwh'].resample('D').sum()
    elif agg_choice == "weekly":
        series = df_filtered['quantityKwh'].resample('W').sum()
    else:
        series = df_filtered['quantityKwh']
    
    tab1, tab2 = st.tabs(["📊 STL Decomposition", "🌈 Spectrogram"])
    
    with tab1:
        st.subheader("Seasonal-Trend decomposition using LOESS")
        
        default_period = {"hourly": 24*7, "daily": 7, "weekly": 52}[agg_choice]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            period = st.number_input(
                "Period length",
                min_value=2,
                max_value=365 if agg_choice != "hourly" else 24*30,
                value=int(default_period),
                step=1 if agg_choice != "hourly" else 24
            )
        with col2:
            seasonal = st.number_input("Seasonal smoother (odd)", 7, 501, 13, step=2)
        with col3:
            trend_suggest = default_period + 1 if default_period % 2 == 0 else default_period + 2
            trend = st.number_input("Trend smoother (odd)", 7, 1201, trend_suggest, step=2)
        with col4:
            robust = st.toggle("Robust", value=True)
        
        with st.spinner("Performing STL decomposition..."):
            stl_df = stl_decompose(series, period=period, seasonal=seasonal, trend=trend, robust=robust)
        
        if stl_df.empty:
            st.warning("Could not perform STL decomposition")
            return
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        components = ['observed', 'trend', 'seasonal', 'resid']
        for i, comp in enumerate(components, 1):
            fig.add_trace(
                go.Scatter(x=stl_df.index, y=stl_df[comp], name=comp.capitalize(), line=dict(width=1)),
                row=i, col=1
            )
        
        fig.update_layout(
            height=800,
            title_text=f"STL Decomposition: {area} • {group} ({agg_choice})",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Component Statistics"):
            st.dataframe(stl_df.describe().T, use_container_width=True)
    
    with tab2:
        st.subheader("Frequency Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            nperseg = st.number_input("Window length", 32, 2048, 168, step=8)
        with col2:
            noverlap = st.number_input("Window overlap", 0, 2047, int(nperseg // 2), step=4)
        
        with st.spinner("Computing spectrogram..."):
            f, t, S = make_spectrogram(series, fs=1.0, nperseg=int(nperseg), noverlap=int(noverlap))
        
        if S.empty:
            st.warning("Could not compute spectrogram")
            return
        
        fig = go.Figure(data=go.Heatmap(
            z=10 * np.log10(S.values + 1e-10),
            x=t,
            y=f,
            colorscale='Viridis',
            colorbar=dict(title='Power (dB)')
        ))
        
        fig.update_layout(
            title=f"Spectrogram: {area} • {group} ({agg_choice})",
            xaxis_title='Window index',
            yaxis_title='Frequency (cycles per sample)',
            height=520
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Spectral matrix: {S.shape[0]} frequencies × {S.shape[1]} windows")

        with st.expander("📊 Spectral power statistics"):
            st.dataframe(S.T.describe().T, use_container_width=True)


def page_outliers():
    """P5: Outliers & Anomalies - Detection using SPC and LOF"""
    st.title("🔍 Outlier & Anomaly Detection")
    
    # City selector for coordinates
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("Select city for weather data", list(CITIES.keys()), index=0)
        lat, lon = CITIES[city]
    with col2:
        st.metric("Coordinates", f"{lat}°N, {lon}°E")
    
    year_start, year_end = st.session_state.year_range
    
    with st.spinner("Loading weather data..."):
        dfs = []
        for year in range(year_start, year_end + 1):
            df_year = load_openmeteo(lat, lon, f"{year}-01-01", f"{year}-12-31")
            if not df_year.empty:
                dfs.append(df_year)
        
        if not dfs:
            st.error("No weather data available")
            return
        
        df = pd.concat(dfs, ignore_index=True)
        df = df.set_index('time').sort_index()
    
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    
    if not num_cols:
        st.error("No numeric columns found")
        return
    
    tab1, tab2 = st.tabs(["📉 SPC Outliers", "🎯 LOF Anomalies"])
    
    with tab1:
        st.subheader("Statistical Process Control (DCT-SPC)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            default_idx = num_cols.index("temperature_2m") if "temperature_2m" in num_cols else 0
            var = st.selectbox("Variable", num_cols, index=default_idx)
        
        with col2:
            keep_k = st.slider("DCT coefficients", 24, 24*30, 24*7, step=24)
        
        with col3:
            sd = st.slider("SPC ±σ", 1.0, 5.0, 3.0, step=0.5)
        
        series = df[var].astype(float)
        df_spc, summary = dct_spc_on_series(series, keep_k=int(keep_k), sd=float(sd))
        
        if df_spc.empty:
            st.warning("Could not compute SPC")
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_spc['time'], y=df_spc['value'],
            mode='lines', name=var,
            line=dict(width=1.2, color='royalblue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_spc['time'], y=df_spc['baseline'],
            mode='lines', name='Baseline (DCT)',
            line=dict(width=2, dash='dot', color='black')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_spc['time'], y=df_spc['lower'],
            mode='lines', name=f'SPC lower (±{sd}σ)',
            line=dict(width=1, dash='dash', color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_spc['time'], y=df_spc['upper'],
            mode='lines', name=f'SPC upper (±{sd}σ)',
            line=dict(width=1, dash='dash', color='red')
        ))
        
        outliers = df_spc[df_spc['outlier']]
        if not outliers.empty:
            fig.add_trace(go.Scatter(
                x=outliers['time'], y=outliers['value'],
                mode='markers', name='Outliers',
                marker=dict(color='red', size=8, symbol='x')
            ))
        
        fig.update_layout(
            title=f"DCT-SPC Outlier Detection: {var} ({city})",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Points", f"{summary['n']:,}")
        col2.metric("Outliers", summary['n_outliers'])
        col3.metric("% Outliers", f"{summary['pct_outliers']:.2f}%")
        col4.metric("σ(SATV)", f"{summary['sigma_satv']:.2f}")
    
    with tab2:
        st.subheader("Local Outlier Factor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            var2 = st.selectbox("Variable for LOF", num_cols, index=0, key="lof_var")
        
        with col2:
            k = st.slider("Neighbors (k)", 5, 60, 20)
        
        res = lof_anomalies(df[var2], k=k)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=res.index, y=res['value'], name=var2, line=dict(color='steelblue')),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=res.index, y=res['LOF'], name='LOF score', line=dict(color='orange', width=1)),
            secondary_y=True
        )
        
        anomalies = res[res['flag']]
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies.index, y=anomalies['value'],
                    mode='markers', name='Anomalies',
                    marker=dict(color='red', size=10, symbol='x')
                ),
                secondary_y=False
            )
        
        fig.update_layout(
            title=f"Local Outlier Factor Analysis: {var2} ({city})",
            height=500,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text=var2, secondary_y=False)
        fig.update_yaxes(title_text="LOF Score", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Points", len(res))
        col2.metric("Anomalies", int(res['flag'].sum()))
        col3.metric("% Anomalies", f"{100 * res['flag'].sum() / len(res):.2f}%")
        
        with st.expander("🔝 Top 10 Anomalies"):
            st.dataframe(res[res['flag']].nlargest(10, 'LOF')[['value', 'LOF']])


def page_correlation():
    """P6: Weather-Energy Correlation - Sliding window analysis"""
    st.title("🔗 Weather-Energy Correlation Analysis")
    
    st.markdown("""
    Analyze the relationship between meteorological conditions and energy production/consumption
    using sliding window correlation with configurable lag.
    """)
    
    client, error = init_mongodb_connection()
    if client is None:
        st.error(f"❌ MongoDB connection error: {error}")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weather_var = st.selectbox(
            "Weather variable",
            ["temperature_2m", "wind_speed_10m", "precipitation", "wind_gusts_10m", "wind_direction_10m"]
        )
    
    with col2:
        energy_type = st.radio("Energy type", ["Production", "Consumption"])
    
    with col3:
        area = st.selectbox("Price area", PRICE_AREAS, index=0)
    
    if energy_type == "Production":
        df_energy = load_mongodb_production(client)
        group_col = "productionGroup"
    else:
        df_energy = load_mongodb_consumption(client)
        group_col = "consumptionGroup"
    
    if df_energy.empty:
        st.warning("No energy data available")
        return
    
    groups = sorted(df_energy[df_energy['priceArea'] == area][group_col].unique().tolist())
    group = st.selectbox("Energy group", groups, index=0 if groups else None)
    
    col1, col2 = st.columns(2)
    with col1:
        window = st.slider("Window length (hours)", 24, 720, 168, step=24)
    with col2:
        lag = st.slider("Lag (hours)", -168, 168, 0, step=1,
                       help="Positive = energy lags weather, Negative = weather lags energy")
    
    # ========== GET GLOBAL YEAR RANGE ==========
    year_start, year_end = st.session_state.year_range
    
    with st.spinner("Loading and processing data..."):
        progress = st.progress(0, text="Step 1/4: Loading weather data...")
    
        lat, lon = PRICE_AREA_COORDS[area]
    
    # Step 1: Load weather data
        dfs_weather = []
        num_years = year_end - year_start + 1
        for idx, year in enumerate(range(year_start, year_end + 1)):
            progress.progress(
                (idx + 1) / (num_years * 4),  # First quarter of progress
                text=f"Loading weather data for {year}... ({idx+1}/{num_years})"
        )
            df_year = load_openmeteo(lat, lon, f"{year}-01-01", f"{year}-12-31")
            if not df_year.empty:
                dfs_weather.append(df_year)
    
        if not dfs_weather:
            progress.empty()
            st.error("❌ Could not load weather data")
            return
    
    # Step 2: Process weather data
        progress.progress(0.30, text="Step 2/4: Processing weather data...")
        df_weather = pd.concat(dfs_weather, ignore_index=True)
        df_weather = df_weather.set_index('time')
        weather_series = df_weather[weather_var]
    
    # Step 3: Load energy data
        progress.progress(0.50, text="Step 3/4: Loading energy data...")
        df_energy_filtered = df_energy[
            (df_energy['priceArea'] == area) &
            (df_energy[group_col] == group) &
            (df_energy['year'] >= year_start) &
            (df_energy['year'] <= year_end)
        ].copy()
    
        if df_energy_filtered.empty:
            progress.empty()
            st.error("❌ No matching energy data found")
            st.info(f"💡 Try selecting a different area or energy group")
            return
    
    # Step 4: Calculate correlation
        progress.progress(0.75, text="Step 4/4: Calculating correlation...")
        df_energy_filtered = df_energy_filtered.set_index('startTime').sort_index()
        df_energy_filtered = df_energy_filtered[~df_energy_filtered.index.duplicated(keep='first')]
        energy_series = df_energy_filtered['quantityKwh']
    
        weather_series = weather_series[~weather_series.index.duplicated(keep='first')]
    
        corr_df = sliding_window_correlation(weather_series, energy_series, window, lag)
    
        progress.progress(1.0, text="Complete!")
        progress.empty()
    
    if corr_df.empty:
        st.warning("Could not compute correlation (insufficient data)")
        return
    
    max_corr = corr_df['correlation'].abs().max()
    max_idx = corr_df['correlation'].abs().idxmax()
    max_time = corr_df.loc[max_idx, 'time']
    max_val = corr_df.loc[max_idx, 'correlation']
    
    st.success(f"**Maximum |correlation|: {max_corr:.3f}** (value: {max_val:.3f}) at {max_time}")

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    weather_series_hourly = weather_series.resample('H').mean()
    energy_series_hourly = energy_series.resample('H').mean()

    w_norm = (weather_series_hourly - weather_series_hourly.mean()) / weather_series_hourly.std()
    e_norm = (energy_series_hourly - energy_series_hourly.mean()) / energy_series_hourly.std()

    w_norm = w_norm.dropna()
    e_norm = e_norm.dropna()

    fig1.add_trace(
        go.Scatter(x=w_norm.index, y=w_norm.values, name=f"{weather_var} (normalized)", line=dict(color='blue')),
        secondary_y=False
    )

    fig1.add_trace(
        go.Scatter(x=e_norm.index, y=e_norm.values, name=f"{group} (normalized)", line=dict(color='green')),
        secondary_y=True
    )

    fig1.update_layout(
        title="Time Series Comparison (Normalized)",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=corr_df['time'],
        y=corr_df['correlation'],
        mode='lines',
        name='Correlation',
        line=dict(color='purple', width=2)
    ))
    
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig2.update_layout(
        title=f"Rolling Correlation (window={window}h, lag={lag}h)",
        xaxis_title="Time",
        yaxis_title="Correlation",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("📊 Correlation Summary"):
        st.markdown(f"""
        **Configuration:**
        - Weather: {weather_var}
        - Energy: {energy_type} - {group} ({area})
        - Window: {window} hours
        - Lag: {lag} hours
        - **Years: {year_start}-{year_end}** ← Shows active range
        
        **Results:**
        - Mean correlation: {corr_df['correlation'].mean():.3f}
        - Max correlation: {max_val:.3f} at {max_time}
        - Correlation std: {corr_df['correlation'].std():.3f}
        
        """)

        

def page_regional():
    """P7: Regional Analysis - Map and Snow Drift"""
    st.title("🗺️ Regional Analysis")

    # Initialize session state
    if "selected_coord" not in st.session_state:
        st.session_state.selected_coord = None
    if "selected_price_area" not in st.session_state:
        st.session_state.selected_price_area = None

    # Two main tabs
    tab1, tab2 = st.tabs(["🗺️ Interactive Map", "❄️ Snow Drift Analysis"])

    # ========== TAB 1: INTERACTIVE MAP ==========
    with tab1:
        st.subheader("Norwegian Price Areas Map")

        # Get MongoDB connection
        client, error = init_mongodb_connection()
        if client is None:
            st.error("❌ Unable to connect to the database")
            st.info("💡 Please check your internet connection and try refreshing the page")
            with st.expander("Technical Details"):
                st.code(f"Error: {error}")
            return

        # Selectors for choropleth coloring
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            energy_type = st.radio("Energy type", ["Production", "Consumption"], key="map_energy_type")

        with col2:
            if energy_type == "Production":
                df_energy = load_mongodb_production(client)
                group_col = "productionGroup"
            else:
                df_energy = load_mongodb_consumption(client)
                group_col = "consumptionGroup"

            if df_energy.empty:
                st.warning("⚠️ No energy data available")
                return

            all_groups = sorted(df_energy[group_col].dropna().unique().tolist())
            selected_group = st.selectbox("Energy group", all_groups, key="map_group")

        with col3:
            year_start, year_end = st.session_state.year_range
            max_days = (year_end - year_start + 1) * 365

            time_days = st.slider(
                "Last N days",
                min_value=1,
                max_value=max_days,
                value=30,
                key="map_days",
                help=f"Show mean for the most recent N days (max: {max_days} days for {year_start}-{year_end})"
            )

        with col4:
            map_mode = st.selectbox(
                "Map View",
                ["Overview (Price Areas)", "Detailed (Subplots)"],
                key="map_mode",
                help="Overview: Single map with 5 price areas\nDetailed: Individual subplot for each area"
            )

        st.caption(f"Showing mean {energy_type.lower()} for {selected_group} over last {time_days} days")

        # Get most recent data
        with st.spinner("📊 Calculating statistics..."):
            df_filtered = df_energy[df_energy[group_col] == selected_group].copy()
            df_filtered = df_filtered.sort_values("startTime", ascending=False)

            if not df_filtered.empty:
                latest_date = df_filtered["startTime"].max()
                cutoff_date = latest_date - pd.Timedelta(days=time_days)
                df_period = df_filtered[df_filtered["startTime"] >= cutoff_date]

                area_means = (
                    df_period.groupby("priceArea")["quantityKwh"]
                    .mean()
                    .reindex(PRICE_AREAS, fill_value=0)
                )
                choropleth_values = area_means.tolist()
            else:
                choropleth_values = [0] * 5

        # ========== LOAD GEOJSON FILE ==========
        possible_paths = [
            "data/file.geojson",
            "data/ElSpot_omraade.json"
        ]

        geojson_data = None

        for geojson_path in possible_paths:
            try:
                with open(geojson_path, "r", encoding="utf-8") as f:
                    geojson_data = json.load(f)
                break
            except FileNotFoundError:
                continue
            except json.JSONDecodeError as e:
                st.error("❌ The map data file is corrupted")
                with st.expander("Technical Details"):
                    st.code(f"JSON Error in {geojson_path}: {e}")
                continue

        if geojson_data is not None:
            # Fix property names for NVE file
            if "features" in geojson_data:
                for feature in geojson_data["features"]:
                    props = feature.get("properties", {})
                    area_name = props.get("ElSpotOmr", "")
                    area_name_clean = area_name.replace(" ", "")
                    props["name"] = area_name_clean

            # Highlight selector
            default_idx = 0
            if st.session_state.selected_price_area in PRICE_AREAS:
                default_idx = PRICE_AREAS.index(st.session_state.selected_price_area) + 1

            highlight_area = st.selectbox(
                "Highlight Price Area",
                ["None"] + PRICE_AREAS,
                index=default_idx,
                key="highlight_area_selector"
            )

            if highlight_area == "None":
                st.session_state.selected_price_area = None
            else:
                st.session_state.selected_price_area = highlight_area

            # ========== CREATE MAP BASED ON MODE ==========
            try:
                if map_mode == "Overview (Price Areas)":
                    # Highlight border colors
                    marker_lines = []
                    for area_code in PRICE_AREAS:
                        if st.session_state.selected_price_area == area_code:
                            marker_lines.append("red")
                        else:
                            marker_lines.append("white")

                    fig = go.Figure()
                    
                    # Add choropleth layer
                    fig.add_trace(go.Choroplethmapbox(
                        geojson=geojson_data,
                        locations=PRICE_AREAS,
                        z=choropleth_values,
                        featureidkey="properties.name",
                        colorscale="Viridis",
                        zmin=0,
                        zmax=max(choropleth_values) if max(choropleth_values) > 0 else 1,
                        marker_opacity=0.7,
                        marker_line_width=3,
                        marker_line_color=marker_lines,
                        text=PRICE_AREAS,
                        hovertemplate="<b>%{text}</b><br>Value: %{z:,.0f} kWh<extra></extra>",
                        colorbar=dict(
                            title=f"{energy_type}<br>(kWh)",
                            thickness=15,
                            len=0.7
                        )
                    ))

                    # ========== ADD RED MARKER FOR SELECTED COORDINATE ==========
                    if st.session_state.get("selected_coord"):
                        lat_marker, lon_marker = st.session_state.selected_coord
                        
                        fig.add_trace(go.Scattermapbox(
                            lat=[lat_marker],
                            lon=[lon_marker],
                            mode="markers+text",
                            marker=dict(
                                size=20,
                                color="red",
                                symbol="circle"
                            ),
                            text=["📍"],
                            textposition="top center",
                            textfont=dict(size=24, color="red"),
                            name="Snow Drift Location",
                            hovertemplate=f"<b>Snow Drift Location</b><br>Lat: {lat_marker:.2f}°<br>Lon: {lon_marker:.2f}°<extra></extra>",
                            showlegend=True
                        ))

                    fig.update_layout(
                        mapbox_style="carto-positron",
                        mapbox_zoom=4,
                        mapbox_center={"lat": 65, "lon": 13},
                        height=600,
                        margin={"r": 0, "t": 30, "l": 0, "b": 0},
                        title=dict(
                            text=f"Mean {energy_type}: {selected_group} ({time_days} days)",
                            x=0.5,
                            xanchor="center"
                        )
                    )

                    # Display map without plotly_events (for stability)
                    st.plotly_chart(fig, use_container_width=True, key="overview_map")

                else:
                    # ========== SUBPLOTS VIEW (DETAILED) ==========
                    st.info("🔍 **Detailed View**: Individual map for each price area with zoomed perspective")

                    fig = make_subplots(
                        rows=2, cols=3,
                        subplot_titles=[f"{area}: {val:,.0f} kWh"
                                        for area, val in zip(PRICE_AREAS, choropleth_values)],
                        specs=[
                            [{"type": "mapbox"}, {"type": "mapbox"}, {"type": "mapbox"}],
                            [{"type": "mapbox"}, {"type": "mapbox"}, None]
                        ],
                        horizontal_spacing=0.01,
                        vertical_spacing=0.08
                    )

                    area_centers = {
                        "NO1": {"lat": 59.91, "lon": 10.75, "zoom": 6},
                        "NO2": {"lat": 58.15, "lon": 8.00, "zoom": 6},
                        "NO3": {"lat": 63.43, "lon": 10.39, "zoom": 5.5},
                        "NO4": {"lat": 69.65, "lon": 18.96, "zoom": 5},
                        "NO5": {"lat": 60.39, "lon": 5.32, "zoom": 6},
                    }

                    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

                    for idx, (area, pos) in enumerate(zip(PRICE_AREAS, positions)):
                        row, col = pos

                        area_geojson = {
                            "type": "FeatureCollection",
                            "features": [
                                f for f in geojson_data["features"]
                                if f["properties"].get("name", "").replace(" ", "") == area
                            ]
                        }

                        highlight = (st.session_state.selected_price_area == area)

                        fig.add_trace(
                            go.Choroplethmapbox(
                                geojson=area_geojson,
                                locations=[area],
                                z=[choropleth_values[idx]],
                                featureidkey="properties.name",
                                colorscale="Viridis",
                                zmin=0,
                                zmax=max(choropleth_values) if max(choropleth_values) > 0 else 1,
                                marker_opacity=0.7,
                                marker_line_width=4 if highlight else 2,
                                marker_line_color="red" if highlight else "white",
                                showscale=False,
                                hovertemplate=f"<b>{area}</b><br>Value: {choropleth_values[idx]:,.0f} kWh<extra></extra>",
                            ),
                            row=row, col=col
                        )

                        mapbox_key = f"mapbox{idx+1}" if idx > 0 else "mapbox"
                        fig.update_layout(**{
                            mapbox_key: dict(
                                style="carto-positron",
                                center={"lat": area_centers[area]["lat"],
                                        "lon": area_centers[area]["lon"]},
                                zoom=area_centers[area]["zoom"]
                            )
                        })

                    fig.update_layout(
                        height=800,
                        margin={"r": 20, "t": 50, "l": 20, "b": 20},
                        title=dict(
                            text=f"Detailed View: {energy_type} by Price Area "
                                 f"({selected_group}, {time_days} days)",
                            x=0.5,
                            xanchor="center"
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Data table below map
                with st.expander("📊 View Data Table"):
                    df_display = pd.DataFrame({
                        "Price Area": PRICE_AREAS,
                        f"Mean {energy_type} (kWh)": [f"{v:,.0f}" for v in choropleth_values]
                    })
                    st.dataframe(df_display, use_container_width=True)

            except Exception as e:
                st.error("❌ Unable to create map visualization")
                st.info("💡 This might be due to invalid GeoJSON data or missing map libraries")
                with st.expander("Technical Details"):
                    st.code(f"Error: {e}")

        else:
            # ========== FALLBACK: NO GEOJSON ==========
            st.warning("⚠️ Map data file not found")
            st.info("""
            **To enable the interactive map:**
            1. Download GeoJSON from: https://temakart.nve.no/tema/nettanlegg
            2. Search for "NVE Elspot områder" and "Elspot_omraade"
            3. Select all areas (NO1-NO5) and export to GeoJSON format
            4. Save as `data/file.geojson` in your project directory
            """)

            df_display = pd.DataFrame({
                "Price Area": PRICE_AREAS,
                f"Mean {energy_type} (kWh)": choropleth_values
            })

            fig_bar = px.bar(
                df_display,
                x="Price Area",
                y=f"Mean {energy_type} (kWh)",
                color=f"Mean {energy_type} (kWh)",
                color_continuous_scale="Viridis",
                title=f"Mean {energy_type} by Price Area ({selected_group}, {time_days} days)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ========== COORDINATE SELECTION FOR SNOW DRIFT ==========
        st.markdown("---")
        st.subheader("📍 Select Location for Snow Drift Analysis")

        # Show current selection if exists
        if st.session_state.get("selected_coord"):
            lat_stored, lon_stored = st.session_state.selected_coord
            st.success(f"✅ **Current coordinate:** {lat_stored:.2f}°N, {lon_stored:.2f}°E (marked on map above)")

        col1, col2, col3 = st.columns(3)

        with col1:
            manual_city = st.selectbox(
                "Quick select city",
                ["Custom"] + list(CITIES.keys()),
                key="manual_city"
            )

        if manual_city != "Custom":
            lat, lon = CITIES[manual_city]
            st.info(f"📍 {manual_city}: {lat}°N, {lon}°E")
        else:
            with col2:
                lat = st.number_input("Latitude", value=65.0, format="%.2f", key="manual_lat")
            with col3:
                lon = st.number_input("Longitude", value=13.0, format="%.2f", key="manual_lon")

        # Action buttons
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("📍 Set Coordinates & Mark on Map", type="primary", use_container_width=True):
                st.session_state.selected_coord = (lat, lon)
                st.success(f"✅ Coordinates set: {lat:.2f}°N, {lon:.2f}°E")
                st.info("🔄 Red marker will appear on map above")
                st.rerun()
        
        with col_b:
            if st.button("🗑️ Clear Selection", use_container_width=True):
                st.session_state.selected_coord = None
                st.info("Cleared")
                st.rerun()

    # ========== TAB 2: SNOW DRIFT ANALYSIS ==========
    with tab2:
        st.subheader("❄️ Snow Drift Calculation")

        if not st.session_state.get("selected_coord"):
            st.warning("⚠️ No coordinates selected")
            st.info("👈 Go to **Interactive Map** tab and select coordinates")
            return

        lat, lon = st.session_state.selected_coord
        st.info(f"📍 Analysis location: {lat:.2f}°N, {lon:.2f}°E")

        # Year range selector
        col1, col2 = st.columns(2)

        with col1:
            start_year = st.selectbox(
                "Start year (July 1st)",
                options=[2021, 2022, 2023, 2024],
                index=0,
                key="snow_start_year"
            )

        with col2:
            end_year = st.selectbox(
                "End year (June 30th)",
                options=[2021, 2022, 2023, 2024, 2025],
                index=1,
                key="snow_end_year"
            )

        if end_year <= start_year:
            st.error("❌ End year must be after start year")
            return

        start_date = f"{start_year}-07-01"
        end_date = f"{end_year}-06-30"

        st.caption(f"Analysis period: **{start_date}** to **{end_date}**")

        # Load weather data
        with st.spinner("🌦️ Loading weather data..."):
            try:
                dfs = []
                for year in range(start_year, end_year + 1):
                    y_start = f"{year}-01-01"
                    y_end = f"{year}-12-31"
                    df_year = load_openmeteo(lat, lon, y_start, y_end)
                    if not df_year.empty:
                        dfs.append(df_year)

                if not dfs:
                    st.error("❌ Could not load weather data")
                    return

                df_weather = pd.concat(dfs, ignore_index=True)
                df_weather["time"] = pd.to_datetime(df_weather["time"])
                df_weather = df_weather.sort_values("time")

                df_weather = df_weather[
                    (df_weather["time"] >= start_date) &
                    (df_weather["time"] <= end_date)
                ]

                if df_weather.empty:
                    st.error("❌ No data available for selected period")
                    return

                st.success(f"✅ Loaded {len(df_weather):,} hourly records")

            except Exception as e:
                st.error("❌ Error loading weather data")
                with st.expander("Technical Details"):
                    st.code(f"Error: {e}")
                return

        # ========== SNOW DRIFT CALCULATION ==========
        st.markdown("---")
        st.subheader("📊 Snow Drift Results")

        try:
            from Snow_drift import calculate_snow_drift

            snow_drift_result = calculate_snow_drift(
                wind_speed=df_weather["wind_speed_10m"].values,
                wind_direction=df_weather["wind_direction_10m"].values,
                precipitation=df_weather["precipitation"].values
            )

            st.metric(
                label="Annual Snow Drift",
                value=f"{snow_drift_result:.2f} kg/m²"
            )

        except ImportError:
            df_weather["snow_potential"] = (
                df_weather["precipitation"] *
                (df_weather["wind_speed_10m"] / 10)
            )

            annual_drift = df_weather["snow_potential"].sum()

            st.metric(
                label="Estimated Annual Snow Drift (simplified)",
                value=f"{annual_drift:.2f} units"
            )

            st.caption("⚠️ Using simplified calculation. For accurate results, add `Snow_drift.py` module.")

        # ========== WIND ROSE PLOT ==========
        st.markdown("---")
        st.subheader("🌬️ Wind Rose")

        wind_speed = df_weather["wind_speed_10m"].values
        wind_dir = df_weather["wind_direction_10m"].values

        direction_bins = np.arange(0, 360, 22.5)
        direction_labels = [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
        ]

        wind_dir_binned = np.digitize(wind_dir, direction_bins)

        direction_speeds = []
        for i in range(1, len(direction_bins) + 1):
            mask = wind_dir_binned == i
            if mask.any():
                direction_speeds.append(wind_speed[mask].mean())
            else:
                direction_speeds.append(0)

        fig_rose = go.Figure()

        fig_rose.add_trace(go.Barpolar(
            r=direction_speeds,
            theta=direction_bins,
            width=[22.5] * len(direction_bins),
            marker_color=direction_speeds,
            marker_colorscale="Viridis",
            marker_line_color="white",
            marker_line_width=1,
            name="Wind Speed",
            hovertemplate="<b>%{theta}°</b><br>Mean Speed: %{r:.1f} m/s<extra></extra>"
        ))

        fig_rose.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(direction_speeds) * 1.1] if max(direction_speeds) > 0 else [0, 1],
                    title="Wind Speed (m/s)"
                ),
                angularaxis=dict(
                    tickmode="array",
                    tickvals=direction_bins,
                    ticktext=direction_labels,
                    direction="clockwise",
                    rotation=90
                )
            ),
            title=f"Wind Rose: {start_date} to {end_date}",
            height=550,
            showlegend=False
        )

        st.plotly_chart(fig_rose, use_container_width=True)

        # ========== WIND STATISTICS ==========
        with st.expander("📊 Wind & Precipitation Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Wind Speed", f"{wind_speed.mean():.1f} m/s")
            col2.metric("Max Wind Speed", f"{wind_speed.max():.1f} m/s")
            col3.metric("Mean Precipitation", f"{df_weather['precipitation'].mean():.2f} mm/h")
            col4.metric("Total Precipitation", f"{df_weather['precipitation'].sum():.0f} mm")

            dominant_idx = np.argmax(direction_speeds)
            dominant_dir = direction_labels[dominant_idx]
            st.success(
                f"**Dominant Wind Direction:** {dominant_dir} "
                f"({direction_bins[dominant_idx]}°) with mean speed {direction_speeds[dominant_idx]:.1f} m/s"
            )

        # ========== MONTHLY ANALYSIS ==========
        st.markdown("---")
        st.subheader("📅 Monthly Snow Drift Analysis")

        df_weather["month"] = df_weather["time"].dt.to_period("M")

        monthly_drift = df_weather.groupby("month").agg({
            "precipitation": "sum",
            "wind_speed_10m": "mean"
        }).reset_index()

        monthly_drift["month_str"] = monthly_drift["month"].astype(str)
        monthly_drift["drift_estimate"] = (
            monthly_drift["precipitation"] *
            (monthly_drift["wind_speed_10m"] / 10)
        )

        fig_monthly = go.Figure()

        fig_monthly.add_trace(go.Bar(
            x=monthly_drift["month_str"],
            y=monthly_drift["drift_estimate"],
            name="Snow Drift Estimate",
            marker_color="steelblue",
            hovertemplate="<b>%{x}</b><br>Drift: %{y:.1f} units<extra></extra>"
        ))

        fig_monthly.update_layout(
            title="Monthly Snow Drift Estimate",
            xaxis_title="Month",
            yaxis_title="Drift Estimate (units)",
            height=400,
            hovermode="x unified",
            xaxis=dict(tickangle=-45)
        )

        st.plotly_chart(fig_monthly, use_container_width=True)

        # ========== COMBINED ANNUAL + MONTHLY PLOT ==========
        st.markdown("---")
        st.subheader("📊 Combined View: Monthly & Annual Snow Drift")

        monthly_drift["cumulative"] = monthly_drift["drift_estimate"].cumsum()
        annual_jul_jun = monthly_drift["drift_estimate"].sum()

        fig_combined = make_subplots(specs=[[{"secondary_y": True}]])

        fig_combined.add_trace(
            go.Bar(
                x=monthly_drift["month_str"],
                y=monthly_drift["drift_estimate"],
                name="Monthly Drift",
                marker_color="steelblue",
                hovertemplate="<b>%{x}</b><br>Monthly: %{y:.1f} units<extra></extra>"
            ),
            secondary_y=False
        )

        fig_combined.add_trace(
            go.Scatter(
                x=monthly_drift["month_str"],
                y=monthly_drift["cumulative"],
                name="Cumulative Drift",
                mode="lines+markers",
                line=dict(color="red", width=3),
                marker=dict(size=8),
                hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f} units<extra></extra>"
            ),
            secondary_y=True
        )

        fig_combined.add_hline(
            y=annual_jul_jun,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text=f"Annual Total: {annual_jul_jun:.1f} units",
            annotation_position="top right",
            secondary_y=True
        )

        fig_combined.update_layout(
            title=f"Monthly & Cumulative Snow Drift: {start_date} to {end_date}",
            xaxis_title="Month",
            height=500,
            hovermode="x unified",
            xaxis=dict(tickangle=-45),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig_combined.update_yaxes(title_text="Monthly Drift (units)", secondary_y=False)
        fig_combined.update_yaxes(title_text="Cumulative Drift (units)", secondary_y=True)

        st.plotly_chart(fig_combined, use_container_width=True)

        # ========== DOWNLOAD OPTION ==========
        st.markdown("---")

        monthly_drift_export = monthly_drift[["month_str", "precipitation",
                                              "wind_speed_10m", "drift_estimate"]].copy()
        monthly_drift_export.columns = [
            "Month", "Total Precipitation (mm)", "Mean Wind Speed (m/s)", "Drift Estimate"
        ]

        csv = monthly_drift_export.to_csv(index=False)

        st.download_button(
            label="📥 Download Monthly Data (CSV)",
            data=csv,
            file_name=f"snow_drift_monthly_{lat}_{lon}_{start_date}_to_{end_date}.csv",
            mime="text/csv",
            type="primary"
        )

        st.caption(f"💾 Export includes {len(monthly_drift)} months of data")

def page_forecasting():
    """P8: Forecasting - SARIMAX energy prediction"""
    st.title("🔮 Energy Forecasting with SARIMAX")
    
    # Get MongoDB connection
    client, error = init_mongodb_connection()
    if client is None:
        st.error(f"❌ MongoDB connection error: {error}")
        return
    
    # ========== MODEL CONFIGURATION ==========
    st.subheader("⚙️ Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Energy Selection**")
        energy_type = st.radio("Energy type", ["Production", "Consumption"], key="forecast_energy_type")
        
        # Load data
        if energy_type == "Production":
            df_energy = load_mongodb_production(client)
            group_col = "productionGroup"
        else:
            df_energy = load_mongodb_consumption(client)
            group_col = "consumptionGroup"
        
        if df_energy.empty:
            st.warning("No energy data available")
            return
        
        # Select group and price area
        all_groups = sorted(df_energy[group_col].dropna().unique().tolist())
        selected_group = st.selectbox("Energy group", all_groups, key="forecast_group")
        
        price_area = st.selectbox("Price area", PRICE_AREAS, key="forecast_area")
    
    with col2:
        st.markdown("**SARIMAX Parameters**")
        st.caption("Non-seasonal (p, d, q)")
        p = st.number_input("p (AR order)", 0, 5, 1, key="sarimax_p")
        d = st.number_input("d (differencing)", 0, 2, 1, key="sarimax_d")
        q = st.number_input("q (MA order)", 0, 5, 1, key="sarimax_q")
        
        st.caption("Seasonal (P, D, Q, s)")
        P = st.number_input("P (seasonal AR)", 0, 2, 1, key="sarimax_P")
        D = st.number_input("D (seasonal diff)", 0, 2, 1, key="sarimax_D")
        Q = st.number_input("Q (seasonal MA)", 0, 2, 1, key="sarimax_Q")
        s = st.number_input("s (seasonal period)", 0, 168, 24, key="sarimax_s", 
                           help="24 for daily, 168 for weekly seasonality")
    
    with col3:
        st.markdown("**Training & Forecast**")
        
        # Training period
        train_days = st.slider("Training period (days)", 30, 365, 90, key="train_days")
        
        # Forecast horizon
        forecast_days = st.slider("Forecast horizon (days)", 1, 30, 7, key="forecast_days")
        
        # Exogenous variables
        use_exog = st.checkbox("Include weather exogenous variables", key="use_exog")
        
        if use_exog:
            exog_vars = st.multiselect(
                "Select weather variables",
                ["temperature_2m", "wind_speed_10m", "precipitation"],
                default=["temperature_2m"],
                key="exog_vars"
            )
        else:
            exog_vars = []
    
    # ========== DATA PREPARATION ==========
    st.markdown("---")
    
    if st.button("🚀 Train Model & Generate Forecast", type="primary"):
        
        with st.spinner("📊 Preparing data..."):
            # Filter data
            df_filtered = df_energy[
                (df_energy[group_col] == selected_group) & 
                (df_energy['priceArea'] == price_area)
            ].copy()
            
            if df_filtered.empty:
                st.error("❌ No data available for selected filters")
                return
            
            # Sort by time
            df_filtered = df_filtered.sort_values('startTime')
            
            # Aggregate to daily data
            df_daily = df_filtered.groupby(df_filtered['startTime'].dt.date).agg({
                'quantityKwh': 'sum'
            }).reset_index()
            df_daily.columns = ['date', 'energy']
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily.set_index('date')
            
            # Get most recent data for training
            df_recent = df_daily.tail(train_days + forecast_days)
            
            if len(df_recent) < train_days:
                st.error(f"❌ Insufficient data. Only {len(df_recent)} days available, need {train_days} for training")
                return
            
            # Split train/test
            train_data = df_recent['energy'].iloc[:train_days]
            test_data = df_recent['energy'].iloc[train_days:] if len(df_recent) > train_days else None
            
            st.success(f"✅ Prepared {len(train_data)} days of training data")
        
        # ========== LOAD EXOGENOUS VARIABLES ==========
        exog_train = None
        exog_forecast = None
        
        if use_exog and exog_vars:
            with st.spinner("🌦️ Loading weather data..."):
                try:
                    # Get coordinates for price area (use center)
                    lat, lon = PRICE_AREA_COORDS[price_area]
                    
                    # Load weather data
                    start_date = train_data.index.min().strftime('%Y-%m-%d')
                    end_date = (train_data.index.max() + pd.Timedelta(days=forecast_days)).strftime('%Y-%m-%d')
                    
                    df_weather = load_openmeteo(lat, lon, start_date, end_date)
                    
                    if not df_weather.empty:
                        # Aggregate to daily
                        df_weather['date'] = pd.to_datetime(df_weather['time']).dt.date
                        df_weather_daily = df_weather.groupby('date')[exog_vars].mean().reset_index()
                        df_weather_daily['date'] = pd.to_datetime(df_weather_daily['date'])
                        df_weather_daily = df_weather_daily.set_index('date')
                        
                        # Align with training data
                        exog_train = df_weather_daily.loc[train_data.index, exog_vars]
                        
                        # Forecast period weather (last available values as constant)
                        forecast_dates = pd.date_range(
                            start=train_data.index.max() + pd.Timedelta(days=1),
                            periods=forecast_days,
                            freq='D'
                        )
                        
                        # Use most recent weather as forecast (simplified)
                        last_weather = df_weather_daily[exog_vars].iloc[-1]
                        exog_forecast = pd.DataFrame(
                            [last_weather.values] * forecast_days,
                            index=forecast_dates,
                            columns=exog_vars
                        )
                        
                        st.success(f"✅ Loaded weather data: {', '.join(exog_vars)}")
                    else:
                        st.warning("⚠️ Could not load weather data, proceeding without exogenous variables")
                        use_exog = False
                
                except Exception as e:
                    st.warning(f"⚠️ Weather data error: {e}. Proceeding without exogenous variables")
                    use_exog = False
        
        # ========== TRAIN SARIMAX MODEL ==========
        st.markdown("---")
        st.subheader("🤖 Model Training")
        
        # Progress container
        progress_container = st.empty()
        status_text = st.empty()
        
        with st.spinner("Training SARIMAX model..."):
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                
                # Configure model
                order = (p, d, q)
                seasonal_order = (P, D, Q, s)
                
                status_text.info("⏳ Initializing model... (this may take 1-2 minutes)")
                progress_bar = progress_container.progress(0, text="Initializing SARIMAX...")
                
                # Train model
                model = SARIMAX(
                    train_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog_train if use_exog else None,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                progress_bar.progress(0.3, text="Model initialized. Starting training...")
                
                # Fit with iterations tracking
                results = model.fit(disp=False, maxiter=100)
                
                progress_bar.progress(1.0, text="Training complete!")
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Model trained successfully!")
                
                # Display model summary
                with st.expander("📊 Model Summary"):
                    st.text(str(results.summary()))
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ Model training failed")
                st.info("💡 Try adjusting the parameters or using a longer training period")
                with st.expander("Technical Details"):
                    st.code(f"Error: {e}")
                return
        
        # ========== GENERATE FORECAST ==========
        st.markdown("---")
        st.subheader("📈 Forecast Results")
        
        try:
            # Generate forecast
            forecast = results.forecast(steps=forecast_days, exog=exog_forecast if use_exog else None)
            forecast_conf = results.get_forecast(steps=forecast_days, exog=exog_forecast if use_exog else None)
            forecast_ci = forecast_conf.conf_int()
            
            # Create forecast dates
            forecast_dates = pd.date_range(
                start=train_data.index.max() + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # ========== PLOT FORECAST ==========
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=train_data.index,
                y=train_data.values,
                mode='lines',
                name='Historical',
                line=dict(color='steelblue', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast.values,
                mode='lines',
                name='Forecast',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_ci.iloc[:, 1],
                mode='lines',
                name='Upper CI (95%)',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_ci.iloc[:, 0],
                mode='lines',
                name='Lower CI (95%)',
                fill='tonexty',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(width=0),
                showlegend=False
            ))
            
            # Test data if available
            if test_data is not None and len(test_data) > 0:
                fig.add_trace(go.Scatter(
                    x=test_data.index,
                    y=test_data.values,
                    mode='lines',
                    name='Actual (test)',
                    line=dict(color='green', width=2)
                ))
            
            fig.update_layout(
                title=f"{energy_type} Forecast: {selected_group} in {price_area}",
                xaxis_title="Date",
                yaxis_title="Energy (kWh)",
                height=500,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ========== EVALUATION METRICS ==========
            if test_data is not None and len(test_data) > 0:
                st.markdown("---")
                st.subheader("📊 Model Evaluation")
                
                # Calculate predictions for test period
                test_forecast = forecast[:len(test_data)]
                
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                
                rmse = np.sqrt(mean_squared_error(test_data.values, test_forecast.values))
                mae = mean_absolute_error(test_data.values, test_forecast.values)
                mape = np.mean(np.abs((test_data.values - test_forecast.values) / test_data.values)) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{rmse:,.2f} kWh")
                col2.metric("MAE", f"{mae:,.2f} kWh")
                col3.metric("MAPE", f"{mape:.2f}%")
            
            # ========== RESIDUAL ANALYSIS ==========
            st.markdown("---")
            st.subheader("🔍 Residual Analysis")
            
            residuals = results.resid
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Residual plot
                fig_resid = go.Figure()
                fig_resid.add_trace(go.Scatter(
                    x=train_data.index,
                    y=residuals,
                    mode='lines',
                    name='Residuals',
                    line=dict(color='red')
                ))
                fig_resid.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_resid.update_layout(
                    title="Residuals Over Time",
                    xaxis_title="Date",
                    yaxis_title="Residuals",
                    height=350
                )
                st.plotly_chart(fig_resid, use_container_width=True)
            
            with col2:
                # Residual histogram
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name='Residuals',
                    marker_color='steelblue'
                ))
                fig_hist.update_layout(
                    title="Residual Distribution",
                    xaxis_title="Residuals",
                    yaxis_title="Frequency",
                    height=350
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Residual statistics
            with st.expander("📊 Residual Statistics"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{residuals.mean():.2f}")
                col2.metric("Std Dev", f"{residuals.std():.2f}")
                col3.metric("Min", f"{residuals.min():.2f}")
                col4.metric("Max", f"{residuals.max():.2f}")
            
            # ========== FORECAST TABLE ==========
            st.markdown("---")
            st.subheader("📋 Forecast Data")
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast (kWh)': forecast.values,
                'Lower CI (kWh)': forecast_ci.iloc[:, 0].values,
                'Upper CI (kWh)': forecast_ci.iloc[:, 1].values
            })
            
            if test_data is not None and len(test_data) > 0:
                forecast_df['Actual (kWh)'] = test_data.values[:len(forecast_df)]
                forecast_df['Error (kWh)'] = forecast_df['Actual (kWh)'] - forecast_df['Forecast (kWh)']
            
            st.dataframe(forecast_df.style.format({
                'Forecast (kWh)': '{:,.2f}',
                'Lower CI (kWh)': '{:,.2f}',
                'Upper CI (kWh)': '{:,.2f}',
                'Actual (kWh)': '{:,.2f}',
                'Error (kWh)': '{:,.2f}'
            }), use_container_width=True)
            
            # Download forecast
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast (CSV)",
                data=csv,
                file_name=f"forecast_{price_area}_{selected_group}_{forecast_days}days.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Forecast generation failed: {e}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
    
    # ========== INFORMATION PANEL ==========
    st.markdown("---")
    
    with st.expander("ℹ️ About SARIMAX Forecasting"):
        st.markdown("""
        ### SARIMAX Model
        
        **SARIMAX** = Seasonal AutoRegressive Integrated Moving Average with eXogenous variables
        
        **Parameters:**
        - **p**: Order of autoregression (AR) - how many past values to use
        - **d**: Degree of differencing - makes data stationary
        - **q**: Order of moving average (MA) - how many past errors to use
        - **P, D, Q**: Seasonal equivalents of p, d, q
        - **s**: Seasonal period (24 for daily patterns, 168 for weekly)
  
        
        **Data Requirements:**
        - Minimum 30 days for training recommended
        - More data = better model (90+ days ideal)
        - Data should have clear patterns/seasonality
        """)
    
    with st.expander("📚 Model Interpretation"):
        st.markdown("""
        ### Understanding Results
        
        **Confidence Intervals:**
        - 95% confidence interval shown in shaded area
        - Wider intervals = more uncertainty
        - Forecast uncertainty increases with horizon
        
        **Evaluation Metrics:**
        - **RMSE**: Root Mean Square Error - penalizes large errors
        - **MAE**: Mean Absolute Error - average error magnitude
        - **MAPE**: Mean Absolute Percentage Error - relative error
        
        **Residual Analysis:**
        - Residuals should be random (white noise)
        - No patterns = good model fit
        - Normal distribution preferred
        - Mean close to zero
        
        **Parameter Significance:**
        - Check p-values in model summary (< 0.05 = significant)
        
        """)
# =================== NAVIGATION ===================

st.sidebar.markdown("### 📑 Navigation")

PAGE_STRUCTURE = [
    ("SECTION 1: APP OVERVIEW", [
        ("P1: Home", page_home),
    ]),
    ("SECTION 2: EXPLORATORY ANALYSIS", [
        ("P2: Weather Plots", page_weather_plots),
        ("P3: Energy Dashboard", page_energy_dashboard),
    ]),
    ("SECTION 3: STATISTICAL ANALYSIS", [
        ("P4: STL & Spectrogram", page_stl_spectrogram),
        ("P5: Outliers & Anomalies", page_outliers),
        ("P6: Weather–Energy Correlation", page_correlation),
    ]),
    ("SECTION 4: PREDICTIVE MODELING", [
        ("P7: Regional Analysis", page_regional),
        ("P8: Forecasting", page_forecasting),
    ]),
]

# Build page map
page_map = {}
all_page_names = []
for section_name, pages in PAGE_STRUCTURE:
    for page_name, page_func in pages:
        page_map[page_name] = page_func
        all_page_names.append(page_name)

# Display navigation - each section independently
for section_idx, (section_name, pages) in enumerate(PAGE_STRUCTURE):
    st.sidebar.markdown(f"**{section_name}**")
    
    page_names = [name for name, func in pages]
    
    # Check if current page is in this section
    if st.session_state.current_page in page_names:
        current_index = page_names.index(st.session_state.current_page)
        
        # Show radio for this section with current selection
        choice = st.sidebar.radio(
            "",
            page_names,
            index=current_index,
            label_visibility="collapsed",
            key=f"nav_section_{section_idx}"
        )
        
        # Update if selection changed
        if choice != st.session_state.current_page:
            st.session_state.current_page = choice
            st.rerun()
    else:
        # No selection in this section - show as clickable options
        for page_name in page_names:
            if st.sidebar.button(
                page_name,
                key=f"btn_{page_name}",
                use_container_width=True
            ):
                st.session_state.current_page = page_name
                st.rerun()
    
    st.sidebar.markdown("")

# Run selected page
page_func = page_map.get(st.session_state.current_page)
if page_func is not None:
    try:
        page_func()
    except Exception as e:
        st.error(f"❌ Error loading page: {e}")
        with st.expander("Show error details"):
            st.exception(e)
else:
    # Fallback to home if page not found
    st.session_state.current_page = "P1: Home"
    page_home()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("IND320 Assignment 4 • 2024")