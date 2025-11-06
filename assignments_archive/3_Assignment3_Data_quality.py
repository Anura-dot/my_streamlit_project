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
#DATA_PATH = Path(r"C:\Users\AnuraArembage\Documents\Copy folder\my_streamlit_project\data")

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

    hourly = j.get("hourly", {})
    cols = {"time": pd.to_datetime(hourly["time"])}

    for var in HOURLY_VARS.split(","):
        if var in hourly:
            cols[var] = hourly[var]

    df = pd.DataFrame(cols).sort_values("time")
    return df



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
    score = np.abs(lof.negative_outlier_factor_)
    return pd.DataFrame({"value": values.values, "LOF": score, "flag": (yhat == -1)}, index=values.index)


# =================== SESSION STATE ===================
if "api_lat" not in st.session_state:
    st.session_state.api_lat = 59.91  # default Oslo
if "api_lon" not in st.session_state:
    st.session_state.api_lon = 10.75
if "api_year" not in st.session_state:
    st.session_state.api_year = 2021

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
    st.subheader("Weather data (Open-Meteo API)")

    # ===== API configuration selector (this is the main connector) =====
    st.markdown("**API configuration**")

    use_area = st.checkbox("Use predefined city", value=True)

    if use_area:
        city_options = list(CITIES.keys())
        selected_city = st.selectbox("Select city", city_options, index=0)
        lat, lon = CITIES[selected_city]
        st.caption(f"Coordinates: {lat}°N, {lon}°E")
    else:
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", value=st.session_state.api_lat, format="%.2f")
        with c2:
            lon = st.number_input("Longitude", value=st.session_state.api_lon, format="%.2f")

    # Assignment says: chosen year = 2021
    year = 2021
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    st.caption(f"Year fixed to {year}: {start} to {end}")

    # Save settings for other pages (connector)
    st.session_state.api_lat = lat
    st.session_state.api_lon = lon
    st.session_state.api_year = year

    # ===== Load data from API =====
    try:
        df = load_openmeteo(lat=lat, lon=lon, start=start, end=end).copy()
        df = df.set_index("time")
    except Exception as e:
        st.error(f"Failed to load weather data: {e}")
        st.stop()

    # ===== Table for the FIRST MONTH ONLY =====
    st.subheader("Data table (first month)")
    first_month = df.index.to_period("M").min()
    month_df = df[df.index.to_period("M") == first_month].copy()
    st.caption(f"Showing first month: {first_month}")
    st.dataframe(month_df.reset_index(), use_container_width=True)

    # Sparklines for the first month
    num_cols = month_df.select_dtypes("number").columns.tolist()
    if num_cols:
        spark = pd.DataFrame(
            {
                "Column": num_cols,
                "First month": [month_df[c].astype(float).tolist() for c in num_cols],
            }
        )

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



# Page 3: Plots (interactive)
elif page == "P 3: Plots (interactive)":
    try:
        # Read API settings from Page 2 (connector)
        lat = st.session_state.get("api_lat", 59.91)
        lon = st.session_state.get("api_lon", 10.75)
        year = st.session_state.get("api_year", 2021)
        start = f"{year}-01-01"
        end = f"{year}-12-31"

        df = load_openmeteo(lat=lat, lon=lon, start=start, end=end)

        # Work on a copy to avoid changing the cached df
        dfx = df.copy()

        # Ensure we have a real 'time' column
        if "time" not in dfx.columns:
            st.error("Weather data is missing a 'time' column.")
            st.stop()

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

        # ---------- interval select_slider ----------
        with c1:
            start_label, end_label = st.select_slider(
                "Select months",
                options=month_labels,
                value=(month_labels[0], month_labels[0]),  # default: first month only
                help="Select a range of months.",
            )

        start_month = pd.Period(start_label)
        end_month = pd.Period(end_label)

        # Filter by selected month RANGE
        mask = (months_period >= start_month) & (months_period <= end_month)
        dff = dfx.loc[mask].copy()

        # Numeric columns
        num_cols = dff.select_dtypes("number").columns.tolist()
        choices = ["All columns"] + num_cols

        with c2:
            pick = st.selectbox(
                "Choose a column to plot",
                options=choices,
                index=0,
                help="Plot one variable or all numeric variables together.",
            )

        # Caption with range
        if start_month == end_month:
            st.caption(f"Rows in {start_month}: {len(dff)}")
        else:
            st.caption(f"Rows from {start_month} to {end_month}: {len(dff)}")

        # --- Build Altair chart ---
        if pick == "All columns":
            if not num_cols:
                st.warning("No numeric columns to plot.")
            else:
                base = dff[["time"] + num_cols].copy()

                col_min = base[num_cols].min()
                col_max = base[num_cols].max()
                col_range = col_max - col_min

                norm = base[num_cols].copy()
                for col in num_cols:
                    if col_range[col] > 0:
                        norm[col] = (base[col] - col_min[col]) / col_range[col]
                    else:
                        norm[col] = 0.5

                long = norm.assign(time=base["time"]).melt(
                    "time", var_name="variable", value_name="value"
                )

                title_range = (
                    f"{start_month}"
                    if start_month == end_month
                    else f"{start_month} to {end_month}"
                )

                chart = (
                    alt.Chart(long)
                    .mark_line()
                    .encode(
                        x=alt.X("time:T", axis=alt.Axis(labelAngle=90, title=None, format="%b %d")),
                        y=alt.Y("value:Q", title="normalized (0–1)", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("variable:N", title=None),
                        tooltip=["time:T", "variable:N", "value:Q"],
                    )
                    .properties(height=320, title=f"All columns (normalized) — {title_range}")
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
        else:
            base = dff[["time", pick]].rename(columns={pick: "value"})

            title_range = (
                f"{start_month}"
                if start_month == end_month
                else f"{start_month} to {end_month}"
            )

            chart = (
                alt.Chart(base)
                .mark_line()
                .encode(
                    x=alt.X("time:T", axis=alt.Axis(labelAngle=90, title=None, format="%b %d")),
                    y=alt.Y("value:Q", title=pick),
                    tooltip=["time:T", "value:Q"],
                )
                .properties(height=320, title=f"{pick} — {title_range}")
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
        "timestamp": "startTime",
        "start_time": "startTime",
        "time": "startTime",
        "pricearea": "priceArea",
        "price_area": "priceArea",
        "region": "priceArea",
        "productiongroup": "productionGroup",
        "production_group": "productionGroup",
        "group": "productionGroup",
        "quantitykwh": "quantityKwh",
        "quantity_kwh": "quantityKwh",
        "kwh": "quantityKwh",
        "quantity": "quantityKwh",
        "energy_production": "quantityKwh",
    }
    renames = {
        c: rename_map[c.strip().lower()]
        for c in list(df.columns)
        if c.strip().lower() in rename_map and rename_map[c.strip().lower()] != c
    }
    if renames:
        df.rename(columns=renames, inplace=True)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # 5) Detect columns and coerce types
    def pick(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    cols = set(df.columns)
    time_col = pick(cols, ["startTime", "timestamp", "time"])
    area_col = pick(cols, ["priceArea", "region", "price_area", "pricearea"])
    group_col = pick(cols, ["productionGroup", "group", "production_group"])
    value_col = pick(cols, ["quantityKwh", "energy_production", "quantity", "kwh", "value"])

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df["month"] = df[time_col].dt.to_period("M").astype(str)
    if value_col:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Validate required columns
    missing_roles = [
        k for k, v in {"time": time_col, "area": area_col, "group": group_col, "value": value_col}.items() if v is None
    ]
    if missing_roles:
        st.error(f"Missing required columns for plotting: {missing_roles}. Check your MongoDB data.")
        st.stop()

    # 6) Two-column layout
    left, right = st.columns(2)

    # LEFT: Pie chart (price area selector)
    with left:
        st.subheader("Production by price area")
        areas = sorted(df[area_col].dropna().unique().tolist())
        if areas:
            area = st.radio("Select price area", areas, key="p4_area")
            st.session_state.price_area = area  # optional: reuse elsewhere

            pie_df = (
                df[df[area_col] == area]
                .groupby(group_col, dropna=False, as_index=False)[value_col]
                .sum()
                .rename(columns={group_col: "Production group", value_col: "Total (kWh)"})
                .sort_values("Total (kWh)", ascending=False)
            )

            fig_pie = px.pie(
                pie_df,
                values="Total (kWh)",
                names="Production group",
                title=f"Production distribution — {area}",
                hole=0.3,
            )
            # Better labels & hover info
            fig_pie.update_traces(
                textposition="outside",
                # label on first line, percentage with 1 decimal on second
                texttemplate="%{label}<br>%{percent:.1%}",
                hovertemplate=(
                    "Group: %{label}<br>"
                    "Share: %{percent:.2%}<br>"
                    "Total: %{value:,.0f} kWh"
                    "<extra></extra>"
                ),
            )

            # Extra margin so labels aren't cut off
            fig_pie.update_layout(
                height=420,
                margin=dict(t=120, b=40, l=20, r=20),
            )

            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No price areas found.")

    # RIGHT: Line chart (pills for groups + month selector)
    with right:
        st.subheader("Hourly production by group & month")

        groups = sorted(df[group_col].dropna().unique().tolist())
        if hasattr(st, "pills"):
            selected_groups = st.pills(
                "Select production group(s):",
                options=groups,
                selection_mode="multi",
                default=[groups[0]] if groups else [],
                key="p4_groups",
            )
        else:
            selected_groups = st.multiselect(
                "Select production group(s):",
                options=groups,
                default=[groups[0]] if groups else [],
                key="p4_groups_ms",
            )

        months = sorted(df["month"].dropna().unique().tolist()) if "month" in df.columns else []
        month = st.selectbox("Select month", options=months, index=0 if months else None, key="p4_month")

        if selected_groups and month:
            # IMPORTANT: filter by area + group(s) + month
            line_df = df[
                (df[area_col] == area)
                & (df[group_col].isin(selected_groups))
                & (df["month"] == month)
            ].copy()

            if line_df.empty:
                st.warning(f"No data for {selected_groups} in {month} for area {area}.")
            else:
                line_df = line_df.sort_values(time_col)
                fig_line = px.line(
                    line_df,
                    x=time_col,
                    y=value_col,
                    color=group_col,
                    title=f"Hourly production — {month} • {area}",
                    labels={time_col: "Date & time", value_col: "Production (kWh)", group_col: "Group"},
                )
                fig_line.update_layout(hovermode="x unified", height=420)
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Select at least one production group and a month.")

    # === Data source expander below the columns (assignment requirement) ===
    with st.expander("Data source"):
        st.markdown(
            "- **Source:** Elhub API, dataset `PRODUCTION_PER_GROUP_MBA_HOUR` "
            "(hourly energy production per production group and price area, 2021).\n"
            "- **Processing:** Data retrieved from the API, stored in Cassandra, "
            "transformed with Spark, and loaded into MongoDB.\n"
            "- **This page:** Reads the curated data from MongoDB and visualizes "
            "total production by price area (pie chart) and hourly production by "
            "group and month (line chart)."
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



# =================== PAGE B: Outliers & Anomalies (SPC & LOF)===================
elif page == "P new B: Outliers & Anomalies":
    st.title("Weather: Outliers (SPC) & Anomalies (LOF)")
    
    # Get data using Page 2 API settings
    lat = st.session_state.get("api_lat", 59.91)
    lon = st.session_state.get("api_lon", 10.75)
    year = st.session_state.get("api_year", 2021)
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    dfw = load_openmeteo(lat=lat, lon=lon, start=start, end=end)
    
    # Ensure time is index
    if "time" in dfw.columns:
        dfw = dfw.set_index("time")
    dfw = dfw.sort_index()

    # Numeric columns for both tabs
    num_cols = [c for c in dfw.columns if np.issubdtype(dfw[c].dtype, np.number)]
    if not num_cols:
        st.error("No numeric columns found for analysis")
        st.stop()

    # One tabs object for both SPC + LOF
    tab1, tab2 = st.tabs(["Outlier / SPC", "Anomaly / LOF"])

    # ========== TAB 1: SPC CONTROL CHART ==========
    with tab1:
        var = st.selectbox("Variable", num_cols, index=0)
        w = st.slider("High-pass window (hours)", 6, 240, 24, step=6)
        k = st.slider("Sigma (k)", 1.0, 5.0, 3.0, step=0.5)
        prop = st.slider("Trim tails (proportion)", 0.0, 0.2, 0.01, step=0.01)
        
        satv = satv_highpass(dfw[var], cutoff_hours=w)
        mu, sigma, LCL, UCL, out = spc_limits(satv, k=k, prop=prop)
        
        fig_spc = go.Figure()
        fig_spc.add_trace(go.Scatter(
            x=satv.index,
            y=satv.values,
            mode='lines+markers',
            name='SATV',
            line=dict(color='steelblue', width=1),
            marker=dict(size=3)
        ))
        fig_spc.add_hline(
            y=mu, line_dash="solid", line_color="green",
            annotation_text=f"Mean: {mu:.2f}", annotation_position="right"
        )
        fig_spc.add_hline(
            y=UCL, line_dash="dash", line_color="red",
            annotation_text=f"UCL: {UCL:.2f}", annotation_position="right"
        )
        fig_spc.add_hline(
            y=LCL, line_dash="dash", line_color="red",
            annotation_text=f"LCL: {LCL:.2f}", annotation_position="right"
        )

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
        
        res = lof_anomalies(dfw[var2], k=k)   # already returns abs(LOF)

        fig_lof = go.Figure()
        # original values
        fig_lof.add_trace(go.Scatter(
            x=res.index,
            y=res["value"],
            name=var2,
            yaxis="y1",
            line=dict(color="steelblue"),
        ))
        # LOF scores
        fig_lof.add_trace(go.Scatter(
            x=res.index,
            y=res["LOF"],
            name="LOF score",
            yaxis="y2",
            line=dict(color="orange", width=1),
        ))

        anomalies = res[res["flag"]]
        if not anomalies.empty:
            fig_lof.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies["value"],
                mode="markers",
                name="Anomalies",
                yaxis="y1",
                marker=dict(color="red", size=10, symbol="x", line=dict(width=2)),
            ))

        fig_lof.update_layout(
            title=f"Local Outlier Factor Analysis: {var2}",
            xaxis=dict(title="Time"),
            yaxis=dict(title=var2, side="left"),
            yaxis2=dict(title="LOF score", overlaying="y", side="right"),
            height=500,
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_lof, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Points", len(res))
        col2.metric("Anomalies", int(res["flag"].sum()))
        col3.metric("% Anomalies", f"{100 * res['flag'].sum() / len(res):.2f}%")

        with st.expander("🔍 Top 10 anomalies by LOF score"):
            top_anomalies = res[res["flag"]].nlargest(10, "LOF")[["value", "LOF"]]
            st.dataframe(top_anomalies, use_container_width=True)

