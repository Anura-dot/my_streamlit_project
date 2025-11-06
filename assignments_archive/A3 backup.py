import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram
from sklearn.neighbors import LocalOutlierFactor
import requests


# =================== APP SETUP ===================
st.set_page_config(page_title="IND320 • Data Check v2", layout="wide")

# =================== CONSTANTS ===================
DATA_PATH = Path(r"C:\Users\AnuraArembage\Documents\Copy folder\my_streamlit_project\data")

CITIES = {
    "Oslo": (59.91, 10.75),
    "Kristiansand": (58.15, 8.00),
    "Trondheim": (63.43, 10.39),
    "Tromsø": (69.65, 18.96),
    "Bergen": (60.39, 5.32),
}

HOURLY_VARS = "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,wind_direction_10m"


# =================== MONGODB FUNCTIONS ===================
@st.cache_resource
def init_mongodb_connection():
    """Initialize MongoDB connection using secrets"""
    try:
        connection_string = st.secrets["mongodb"]["connection_string"]
        client = MongoClient(connection_string)
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        return None

@st.cache_data(ttl=3600)
def load_mongodb_data(_client):
    """Load data from MongoDB with caching"""
    try:
        db_name = st.secrets["mongodb"]["database_name"]
        collection_name = st.secrets["mongodb"]["collection_name"]
        
        db = _client[db_name]
        collection = db[collection_name]
        
        data = list(collection.find({}, {"_id": 0}))
        df = pd.DataFrame(data)
        
        if 'startTime' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['startTime']):
                df['startTime'] = pd.to_datetime(df['startTime'])
            df['month'] = df['startTime'].dt.to_period('M').astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading data from MongoDB: {e}")
        return pd.DataFrame()


# =================== WEATHER DATA FUNCTIONS ===================
@st.cache_data(show_spinner=False)
def load_csv_weather(path: Path) -> pd.DataFrame:
    """Load weather data from CSV file"""
    csv_file = path / "open-meteo-subset.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    df = pd.read_csv(csv_file, parse_dates=["time"]).sort_values("time")
    return df

@st.cache_data(ttl=900, show_spinner=False)
def load_openmeteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo API"""
    url = (
        "https://archive-api.open-meteo.com/v1/era5"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={HOURLY_VARS}"
        f"&start_date={start}&end_date={end}&timezone=Europe/Oslo"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()
    
    # Build DataFrame safely
    hourly = j.get("hourly", {})
    cols = {"time": pd.to_datetime(hourly["time"])}
    
    for var in HOURLY_VARS.split(","):
        if var in hourly:
            cols[var] = hourly[var]
    
    df = pd.DataFrame(cols).sort_values("time")
    return df

def get_weather_df(source: str, csv_path: Path, **kw) -> pd.DataFrame:
    """Load weather data from CSV or API based on source selection"""
    if source.startswith("CSV"):
        return load_csv_weather(csv_path)
    else:
        return load_openmeteo(**kw)


# =================== ANALYSIS FUNCTIONS ===================
def stl_decompose(series: pd.Series, period: int, seasonal: int, trend: int, robust: bool = True):
    """Perform STL decomposition on time series"""
    s = series.astype(float).dropna()
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
    f, t, Sxx = spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
    return f, t, pd.DataFrame(Sxx, index=f, columns=t)

def satv_highpass(series: pd.Series, cutoff_hours: int = 24):
    """Apply high-pass filter using Direct Cosine Transform for seasonally adjusted values"""
    x = series.astype(float)
    return x - x.rolling(cutoff_hours, min_periods=max(2, cutoff_hours//3)).mean()

def spc_limits(s: pd.Series, k: float = 3.0, prop: float = 0.01):
    """Calculate Statistical Process Control limits and identify outliers"""
    s = s.dropna()
    if 0 < prop < 0.5:
        ql, qu = s.quantile(prop/2), s.quantile(1 - prop/2)
        s_use = s.clip(lower=ql, upper=qu)
    else:
        s_use = s
    mu, sigma = s_use.mean(), s_use.std(ddof=1)
    UCL, LCL = mu + k*sigma, mu - k*sigma
    out = s[(s > UCL) | (s < LCL)]
    return mu, sigma, LCL, UCL, out

def lof_anomalies(values: pd.Series, k: int = 20):
    """Detect anomalies using Local Outlier Factor"""
    X = values.astype(float).ffill().to_frame()
    lof = LocalOutlierFactor(n_neighbors=k, contamination="auto")
    yhat = lof.fit_predict(X.values)
    score = -lof.negative_outlier_factor_
    return pd.DataFrame({"value": values.values, "LOF": score, "flag": (yhat == -1)}, index=values.index)


# =================== SESSION STATE ===================
if "weather_source" not in st.session_state:
    st.session_state.weather_source = "CSV (local)"
if "price_area" not in st.session_state:
    st.session_state.price_area = "NO1"


# =================== NAVIGATION ===================
page = st.sidebar.radio(
    "Navigate",
    [
        "P 1: Home",
        "P 4: Elhub Data",
        "P new A: STL & Spectrogram",
        "P 2: Data table & summary",
        "P 3: Plots (interactive)",
        "P new B: Outliers & Anomalies"
    ],
    index=0
)


# =================== PAGE LOGIC STARTS HERE ===================

# Home page setup
if page == "P 1: Home":
    st.title("IND320 • Streamlit app")
    st.subheader("Welcome to the App")



# Page 2: Data table & summary
elif page == "P 2: Data table & summary":
    st.subheader("Weather data source")
    
    # Source selector
    st.session_state.weather_source = st.radio(
        "Select source",
        ["CSV (local)", "Open-Meteo API"],
        index=0 if st.session_state.weather_source == "CSV (local)" else 1,
        help="Selector controls the data used by pages A/B and plots."
    )

    # Initialize variables
    lat, lon, start, end = None, None, None, None

    # Show appropriate input fields based on selection
    if st.session_state.weather_source == "Open-Meteo API":
        st.markdown("**API Configuration**")
        
        # Area selector or manual coordinates
        use_area = st.checkbox("Use predefined city", value=True)
        
        if use_area:
            # City selector with coordinates from CITIES dict
            city_options = list(CITIES.keys())
            selected_city = st.selectbox("Select city", city_options, index=0)
            lat, lon = CITIES[selected_city]
            st.caption(f"Coordinates: {lat}°N, {lon}°E")
        else:
            # Manual coordinate input
            c1, c2 = st.columns(2)
            with c1:
                lat = st.number_input("Latitude", value=59.91, format="%.2f")
            with c2:
                lon = st.number_input("Longitude", value=10.75, format="%.2f")
        
        # Year selector
        year = st.selectbox("Select year", [2021, 2020, 2019, 2018], index=0)
        
        # Auto-calculate start and end dates
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        
        st.caption(f"Date range: {start} to {end}")

    # Load data based on selection
    try:
        if st.session_state.weather_source == "Open-Meteo API":
            df = get_weather_df(
                st.session_state.weather_source,
                DATA_PATH,
                lat=lat, lon=lon, start=start, end=end
            ).copy()
        else:
            df = get_weather_df(
                st.session_state.weather_source,
                DATA_PATH
            ).copy()
        
        df = df.set_index("time")
    # Save API parameters to session state for other pages
        if st.session_state.weather_source == "Open-Meteo API":
            st.session_state.api_lat = lat
            st.session_state.api_lon = lon
            st.session_state.api_year = year    
        
    except Exception as e:
        st.error(f"Failed to load weather data: {e}")
        st.stop()

    # Display data table (first month)
    st.subheader("Data table (first month)")
    first_month = df.index.to_period("M").min()
    month_df = df[df.index.to_period("M") == first_month].copy()
    st.caption(f"Showing first month: {first_month}")
    st.dataframe(month_df.reset_index(), use_container_width=True)

    # Sparklines for first month
    num_cols = month_df.select_dtypes("number").columns.tolist()
    if num_cols:
        spark = pd.DataFrame({
            "Column": num_cols,
            "First month": [month_df[c].astype(float).tolist() for c in num_cols],
        })

        st.subheader(f"Sparklines by column — {first_month}")
        st.dataframe(
            spark,
            column_config={
                "Column": "Column",
                "First month": st.column_config.LineChartColumn("First month"),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No numeric columns found to build sparklines for the first month.")

    # Display summary statistics
    st.subheader("Summary statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

# Page 3: Plots (interactive)
elif page == "P 3: Plots (interactive)":
    try:
        src = st.session_state.get("weather_source", "CSV (local)")
        
        # Get parameters based on source
        if src == "Open-Meteo API":
            # Use session state or defaults for API
            lat = st.session_state.get("api_lat", 59.91)
            lon = st.session_state.get("api_lon", 10.75)
            year = st.session_state.get("api_year", 2021)
            start = f"{year}-01-01"
            end = f"{year}-12-31"
            df = get_weather_df(src, DATA_PATH, lat=lat, lon=lon, start=start, end=end)
        else:
            # CSV mode
            df = get_weather_df(src, DATA_PATH)

        # Work on a copy to avoid changing the cached df
        dfx = df.copy()

        # If the index is named 'time' AND there's also a 'time' column, kill the index name to avoid ambiguity
        if isinstance(dfx.index, pd.DatetimeIndex) and dfx.index.name == "time":
            dfx.index.name = None

        # Ensure we have a real 'time' column
        if "time" not in dfx.columns:
            orig_idx_name = dfx.index.name or "index"
            dfx = dfx.reset_index().rename(columns={orig_idx_name: "time"})

        # Coerce to datetime and sort
        dfx["time"] = pd.to_datetime(dfx["time"], errors="coerce")
        dfx = dfx.dropna(subset=["time"]).sort_values("time")

        # Month slider (use Period[M] but show as strings)
        months_period = dfx["time"].dt.to_period("M")
        months_unique = sorted(months_period.unique().tolist())
        if not months_unique:
            st.warning("No month values found.")
            st.stop()

        month_labels = [str(m) for m in months_unique]
        st.subheader("Interactive plots")
        c1, c2 = st.columns([1, 2])

        with c1:
            picked_label = st.select_slider(
                "Select month",
                options=month_labels,
                value=month_labels[0],
                help="Filter the data to a single month."
            )
        picked_month = pd.Period(picked_label)

        # Filter by selected month
        mask = months_period == picked_month
        dff = dfx.loc[mask].copy()

        # Numeric columns 
        num_cols = dff.select_dtypes("number").columns.tolist()
        choices = ["All columns"] + num_cols

        with c2:
            pick = st.selectbox(
                "Choose a column to plot",
                options=choices,
                index=0,
                help="Plot one variable or all numeric variables together."
            )

        st.caption(f"Rows in {picked_month}: {len(dff)}")

        # --- Build Altair chart (vertical x labels, proper titles) ---
        if pick == "All columns":
            if not num_cols:
                st.warning("No numeric columns to plot.")
            else:
                # Min–max normalize so different scales are comparable
                base = dff[["time"] + num_cols].copy()
                
                # Calculate min and max for normalization
                col_min = base[num_cols].min()
                col_max = base[num_cols].max()
                col_range = col_max - col_min
                
                # Only normalize columns that have variation
                norm = base[num_cols].copy()
                for col in num_cols:
                    if col_range[col] > 0:  # Has variation
                        norm[col] = (base[col] - col_min[col]) / col_range[col]
                    else:  # Constant value
                        norm[col] = 0.5  # Set to middle of 0-1 range
                
                long = norm.assign(time=base["time"]).melt("time", var_name="variable", value_name="value")

                chart = (
                    alt.Chart(long)
                    .mark_line()
                    .encode(
                        x=alt.X("time:T", axis=alt.Axis(labelAngle=90, title=None, format="%b %d")),
                        y=alt.Y("value:Q", title="normalized (0–1)", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("variable:N", title=None),
                        tooltip=["time:T", "variable:N", "value:Q"],
                    )
                    .properties(height=320, title=f"All columns (normalized) — {picked_month}")
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
        else:
            base = dff[["time", pick]].rename(columns={pick: "value"})
            chart = (
                alt.Chart(base)
                .mark_line()
                .encode(
                    x=alt.X("time:T", axis=alt.Axis(labelAngle=90, title=None, format="%b %d")),
                    y=alt.Y("value:Q", title=pick),
                    tooltip=["time:T", "value:Q"],
                )
                .properties(height=320, title=f"{pick} — {picked_month}")
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())



# =================== PAGE 4 (Assignment 2 inside Assignment 1) ===================
elif page == "P 4: Elhub Data":

    st.title("P 4: Energy Production Dashboard")

    # 1) Connect to MongoDB
    client = init_mongodb_connection()
    if client is None:
        st.error("❌ Cannot connect to MongoDB. Check Streamlit Secrets and network.")
        st.info(
            "**Cloud Secrets must contain:**\n\n"
            "```\n[mongodb]\n"
            'connection_string = "mongodb+srv://..."\n'
            'database_name = "elhub"\n'
            'collection_name = "production_mbahour"\n'
            "```"
        )
        st.stop()

    # 2) Load data
    with st.spinner("Loading data from MongoDB..."):
        df = load_mongodb_data(client)

    if df.empty:
        st.warning("⚠️ No data available in MongoDB collection.")
        st.stop()

    # 3) Coalesce duplicate semantic columns BEFORE renaming
    if "startTime" in df.columns and "timestamp" in df.columns:
        df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce").combine_first(
            pd.to_datetime(df["timestamp"], errors="coerce")
        )
        df.drop(columns=["timestamp"], inplace=True)

    if "priceArea" in df.columns and "region" in df.columns:
        df["priceArea"] = df["priceArea"].astype(object).combine_first(df["region"])
        df.drop(columns=["region"], inplace=True)

    # 4) Normalize column names (case-insensitive)
    rename_map = {
        "timestamp":"startTime","start_time":"startTime","time":"startTime",
        "pricearea":"priceArea","price_area":"priceArea","region":"priceArea",
        "productiongroup":"productionGroup","production_group":"productionGroup","group":"productionGroup",
        "quantitykwh":"quantityKwh","quantity_kwh":"quantityKwh","kwh":"quantityKwh","quantity":"quantityKwh",
        "energy_production":"quantityKwh"
    }
    renames = {c: rename_map[c.strip().lower()] for c in list(df.columns)
               if c.strip().lower() in rename_map and rename_map[c.strip().lower()] != c}
    if renames:
        df.rename(columns=renames, inplace=True)

    if df.columns.duplicated().any():
        dupes = [c for c, d in zip(df.columns, df.columns.duplicated()) if d]
        df = df.loc[:, ~df.columns.duplicated()]

    # 5) Detect columns and coerce types
    def pick(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    cols = set(df.columns)
    time_col  = pick(cols, ["startTime","timestamp","time"])
    area_col  = pick(cols, ["priceArea","region","price_area","pricearea"])
    group_col = pick(cols, ["productionGroup","group","production_group"])
    value_col = pick(cols, ["quantityKwh","energy_production","quantity","kwh","value"])

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df["month"] = df[time_col].dt.to_period("M").astype(str)
    if value_col:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Optional expanders for data inspection
    if "month" in df.columns:
        month_counts = df["month"].value_counts().sort_index()
        with st.expander("📅 Rows per month"):
            st.dataframe(month_counts.rename("rows").to_frame(), use_container_width=True)

    with st.expander("👀 Data preview", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    # Validate required columns
    missing_roles = [k for k, v in {"time":time_col,"area":area_col,"group":group_col,"value":value_col}.items() if v is None]
    if missing_roles:
        st.error(f"Missing required columns for plotting: {missing_roles}. Check your MongoDB data.")
        st.stop()

    # 6) Two-column layout
    left, right = st.columns(2)

    # LEFT: Pie chart
    with left:
        st.subheader("📊 Production by Price Area")
        areas = sorted(df[area_col].dropna().unique().tolist())
        if areas:
            area = st.radio("Select price area", areas, key="p4_area")
            st.session_state.price_area = area  # persist for other pages

            pie_df = (
                df[df[area_col] == area]
                .groupby(group_col, dropna=False, as_index=False)[value_col]
                .sum()
                .rename(columns={group_col: "Production Group", value_col: "Total (kWh)"})
                .sort_values("Total (kWh)", ascending=False)
            )

            fig_pie = px.pie(
                pie_df, values="Total (kWh)", names="Production Group",
                title=f"Production Distribution — {area}", hole=0.3
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=420)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No price areas found.")

    # RIGHT: Line chart
    with right:
        st.subheader("📈 Hourly Production (select groups + month)")
        groups = sorted(df[group_col].dropna().unique().tolist())
        selected_groups = (
            st.pills("Select production group(s):", options=groups, selection_mode="multi",
                     default=[groups[0]] if groups else [], key="p4_groups")
            if hasattr(st, "pills") else
            st.multiselect("Select production group(s):", options=groups,
                           default=[groups[0]] if groups else [], key="p4_groups_ms")
        )
        months = sorted(df["month"].dropna().unique().tolist()) if "month" in df.columns else []
        month = st.selectbox("Select month", options=months, index=0 if months else None, key="p4_month")

        if selected_groups and month:
            line_df = df[(df[group_col].isin(selected_groups)) & (df["month"] == month)].copy()
            if line_df.empty:
                st.warning(f"No data for {selected_groups} in {month}.")
            else:
                line_df = line_df.sort_values(time_col)
                fig_line = px.line(
                    line_df, x=time_col, y=value_col, color=group_col,
                    title=f"Hourly Production — {month}",
                    labels={time_col: "Date & Time", value_col: "Production (kWh)", group_col: "Group"}
                )
                fig_line.update_layout(hovermode="x unified", height=420)
                st.plotly_chart(fig_line, use_container_width=True)
                st.caption(f"Showing {len(line_df):,} points")
        else:
            st.info("Pick at least one production group and a month.")

    st.markdown("---")
    with st.expander("📚 Data source & pipeline"):
        st.markdown(
            "- **Source:** Elhub API — 2021 hourly production (PRODUCTION_PER_GROUP_MBA_HOUR)\n"
            "- **ETL:** API → Cassandra (staging) → Spark transform → MongoDB Atlas (`elhub.production_mbahour`)\n"
            "- **This page:** reads from MongoDB via Streamlit secrets. No credentials in code."
        )



#=================Page A====================
elif page == "P new A: STL & Spectrogram":
    st.title("Elhub series: STL & Spectrogram")
   
    # Reuse Mongo loader from Page 4
    client = init_mongodb_connection()
    if client is None:
        st.error("Mongo connection needed for this page.")
        st.stop()
    df_e = load_mongodb_data(client)

    # normalize like Page 4 did
    for cand in ["startTime", "timestamp", "time"]:
        if cand in df_e.columns:
            df_e[cand] = pd.to_datetime(df_e[cand], errors="coerce")

    time_col = "startTime" if "startTime" in df_e.columns else "timestamp" if "timestamp" in df_e.columns else "time"
    area_col = "priceArea"
    group_col = "productionGroup"
    value_col = "quantityKwh"

    df_e = (
        df_e.dropna(subset=[time_col, area_col, group_col, value_col])
            .sort_values(time_col)
    )

    # selectors
    areas = sorted(df_e[area_col].unique().tolist())
    default_area = st.session_state.get("price_area", areas[0] if areas else None)

    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.selectbox("Price area", areas, index=areas.index(default_area) if default_area in areas else 0)
    with c2:
        groups = sorted(df_e.loc[df_e[area_col] == area, group_col].unique().tolist())
        group = st.selectbox("Production group", groups, index=0 if groups else None)
    with c3:
        agg_choice = st.selectbox("Aggregate", ["hourly", "daily", "weekly"], index=0)

    # prep series
    dfx = (
        df_e[(df_e[area_col] == area) & (df_e[group_col] == group)]
        [[time_col, value_col]]
        .rename(columns={time_col: "time", value_col: "value"})
        .set_index("time").sort_index()
    )

    if agg_choice == "daily":
        series = dfx["value"].resample("D").sum()
    elif agg_choice == "weekly":
        series = dfx["value"].resample("W").sum()
    else:
        series = dfx["value"]  # hourly

    tab1, tab2 = st.tabs(["STL decomposition", "Spectrogram"])

    # ========== TAB 1 ==========
    with tab1:
        default_period = {"hourly": 24*7, "daily": 7, "weekly": 52}[agg_choice]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            period = st.number_input(
                "Period length",
                min_value=2 if agg_choice != "hourly" else 24,
                max_value=365 if agg_choice != "hourly" else 24*60,
                value=int(default_period),
                step=1 if agg_choice != "hourly" else 24
            )
        with c2:
            seasonal = st.number_input("Seasonal smoother (odd)", 7, 501, 13, step=2)
        with c3:
            # suggest an odd number > period
            trend_suggest = default_period + 1 if default_period % 2 == 0 else default_period + 2
            trend = st.number_input("Trend smoother (odd, > period)", 7, 1201, trend_suggest, step=2)
        with c4:
            robust = st.toggle("Robust", value=True)

        stl_df = stl_decompose(series, period=period, seasonal=seasonal, trend=trend, robust=robust)

        fig_stl = go.Figure()
        components = ['observed', 'trend', 'seasonal', 'resid']
        titles = ['Observed', 'Trend', 'Seasonal', 'Residual']
        for i, (comp, title_) in enumerate(zip(components, titles)):
            fig_stl.add_trace(go.Scatter(
                x=stl_df.index, y=stl_df[comp],
                name=title_, yaxis=f'y{i+1}' if i > 0 else 'y'
            ))
        fig_stl.update_layout(
            title=f"STL Decomposition: {area} • {group} ({agg_choice})",
            height=800, showlegend=True,
            xaxis=dict(domain=[0, 1]),
            yaxis=dict(title='Observed', domain=[0.775, 1.0]),
            yaxis2=dict(title='Trend',    domain=[0.525, 0.75]),
            yaxis3=dict(title='Seasonal', domain=[0.275, 0.5]),
            yaxis4=dict(title='Residual', domain=[0.0, 0.25])
        )
        st.plotly_chart(fig_stl, use_container_width=True)
        st.caption("STL Component Statistics")
        st.dataframe(stl_df.describe().T, use_container_width=True)

    # ========== TAB 2 ==========
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            nper = st.number_input("Window length", 32, 2048, 168, step=8)
        with c2:
            nov = st.number_input("Window overlap", 0, 2047, int(nper // 2), step=4)

        f, t, S = make_spectrogram(series, fs=1.0, nperseg=int(nper), noverlap=int(nov))

        fig_spec = go.Figure(data=go.Heatmap(
            z=10 * np.log10(S.values + 1e-10),
            x=t, y=f, colorscale='Viridis', colorbar=dict(title='Power (dB)')
        ))
        fig_spec.update_layout(
            title=f"Spectrogram: {area} • {group} ({agg_choice})",
            xaxis_title='Window index',
            yaxis_title='Frequency (cycles per sample)',
            height=520
        )
        st.plotly_chart(fig_spec, use_container_width=True)
        st.caption(f"Spectral matrix shape: {S.shape[0]} frequencies × {S.shape[1]} windows")
        with st.expander("📊 Spectral power statistics"):
            st.dataframe(S.describe().T, use_container_width=True)



#New Page “B”: Outliers & Anomalies (SPC + LOF)
# =================== PAGE B: Outliers & Anomalies ===================
elif page == "P new B: Outliers & Anomalies":
    st.title("Weather: Outliers (SPC) & Anomalies (LOF)")
    
    # Get data using Page 2 settings
    src = st.session_state.get("weather_source", "CSV (local)")
    
    # Get parameters based on source
    if src == "Open-Meteo API":
        lat = st.session_state.get("api_lat", 59.91)
        lon = st.session_state.get("api_lon", 10.75)
        year = st.session_state.get("api_year", 2019)
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        dfw = get_weather_df(src, DATA_PATH, lat=lat, lon=lon, start=start, end=end)
    else:
        dfw = get_weather_df(src, DATA_PATH)
    
    # Ensure time is index
    if "time" in dfw.columns:
        dfw = dfw.set_index("time")
    dfw = dfw.sort_index()
    
    # Define numeric columns once for both tabs
    num_cols = [c for c in dfw.columns if np.issubdtype(dfw[c].dtype, np.number)]
    if not num_cols:
        st.error("No numeric columns found for analysis")
        st.stop()

    tab1, tab2 = st.tabs(["Outlier / SPC", "Anomaly / LOF"])

    # ========== TAB 1: SPC CONTROL CHART ==========
    with tab1:
             
        var = st.selectbox("Variable", num_cols, index=0)
        w = st.slider("High-pass window (hours)", 6, 240, 24, step=6)
        k = st.slider("Sigma (k)", 1.0, 5.0, 3.0, step=0.5)
        prop = st.slider("Trim tails (proportion)", 0.0, 0.2, 0.01, step=0.01)
        
        satv = satv_highpass(dfw[var], cutoff_hours=w)
        mu, sigma, LCL, UCL, out = spc_limits(satv, k=k, prop=prop)
        
        # Create SPC control chart
        fig_spc = go.Figure()
        
        # Plot SATV values
        fig_spc.add_trace(go.Scatter(
            x=satv.index,
            y=satv.values,
            mode='lines+markers',
            name='SATV',
            line=dict(color='steelblue', width=1),
            marker=dict(size=3)
        ))
        
        # Add control limits
        fig_spc.add_hline(
            y=mu, 
            line_dash="solid", 
            line_color="green", 
            annotation_text=f"Mean: {mu:.2f}",
            annotation_position="right"
        )
        fig_spc.add_hline(
            y=UCL, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"UCL: {UCL:.2f}",
            annotation_position="right"
        )
        fig_spc.add_hline(
            y=LCL, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"LCL: {LCL:.2f}",
            annotation_position="right"
        )
        
        # Highlight outliers
        if not out.empty:
            fig_spc.add_trace(go.Scatter(
                x=out.index,
                y=out.values,
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=8, symbol='x')
            ))
        
        fig_spc.update_layout(
            title=f"Statistical Process Control: {var}",
            xaxis_title="Time",
            yaxis_title=f"SATV ({var})",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_spc, use_container_width=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean (μ)", f"{mu:.2f}")
        col2.metric("Std Dev (σ)", f"{sigma:.2f}")
        col3.metric("Outliers", len(out))
        col4.metric("% Outliers", f"{100*len(out)/len(satv):.2f}%")

    # ========== TAB 2: LOF ANOMALY DETECTION ==========
    with tab2:
        var2 = st.selectbox(
            "Variable for LOF", 
            num_cols, 
            index=0, 
            key="lofvar2"
        )
        k = st.slider("Neighbors (k)", 5, 60, 20, key="lof_k")
        
        res = lof_anomalies(dfw[var2], k=k)
        
        # Create dual-axis plot: values + LOF scores
        fig_lof = go.Figure()
        
        # Original values
        fig_lof.add_trace(go.Scatter(
            x=res.index,
            y=res['value'],
            name=var2,
            yaxis='y1',
            line=dict(color='steelblue')
        ))
        
        # LOF scores
        fig_lof.add_trace(go.Scatter(
            x=res.index,
            y=res['LOF'],
            name='LOF Score',
            yaxis='y2',
            line=dict(color='orange', width=1)
        ))
        
        # Highlight anomalies
        anomalies = res[res['flag']]
        if not anomalies.empty:
            fig_lof.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies['value'],
                mode='markers',
                name='Anomalies',
                yaxis='y1',
                marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
            ))
        
        # Configure dual y-axes
        fig_lof.update_layout(
            title=f"Local Outlier Factor Analysis: {var2}",
            xaxis=dict(title="Time"),
            yaxis=dict(title=var2, side='left'),
            yaxis2=dict(title='LOF Score', overlaying='y', side='right'),
            height=500,
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99)
        )
        
        st.plotly_chart(fig_lof, use_container_width=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Points", len(res))
        col2.metric("Anomalies", int(res['flag'].sum()))
        col3.metric("% Anomalies", f"{100*res['flag'].sum()/len(res):.2f}%")
        
        # Top anomalies table
        with st.expander("🔍 Top 10 anomalies by LOF score"):
            top_anomalies = res.nlargest(10, 'LOF')[['value', 'LOF', 'flag']]
            st.dataframe(top_anomalies, use_container_width=True)
