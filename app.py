import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go

from mongodb_config import get_mongodb_collection

# App setup
st.set_page_config(page_title="IND320 • Data Check v2", layout="wide")

DATA_PATH = Path("data/open-meteo-subset.csv")

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").set_index("time")
    return df

# MongoDB connection function
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

# Sidebar navigation 
page = st.sidebar.radio(
    "Navigate",
    ["P 1: Home", "P 2: Data table & summary", "P 3: Plots (interactive)", "P 4: Elhub Data"],
    index=0
)

#  numeric columns only
def numeric_cols(df: pd.DataFrame):
    return df.select_dtypes("number").columns.tolist()


# Home page setup
if page == "P 1: Home":
    st.title("IND320 • Streamlit app")
    st.subheader("Welcome to the App")



# Page 2: Data table & summary
elif page == "P 2: Data table & summary":
    try:
        df = load_data(DATA_PATH)  

        # First-month subset (works if 'time' is index or column)
        df2 = df.copy()

        if "time" in df2.columns:
            ts = pd.to_datetime(df2["time"], errors="coerce")
            months = ts.dt.to_period("M")
        else:
            if isinstance(df2.index, pd.DatetimeIndex):
                months = df2.index.to_period("M")
            else:
                idx = pd.to_datetime(df2.index, errors="coerce")
                months = pd.Series(idx, index=df2.index).dt.to_period("M")

        first_month = months.min()
        mask = months == first_month
        month_df = df2.loc[mask].copy()

        # Table (first month) 
        st.subheader("Data table (first month)")
        st.caption(f"Showing first month: {first_month}")
        # show time as a column for readability
        table_show = month_df.reset_index()
        st.dataframe(table_show, width="stretch")

        # Summary 
        st.subheader("Summary statistics")
        st.dataframe(df.describe(include="all").transpose(), width="stretch")

        # Row-wise sparklines for first month
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
                width="stretch",
            )
        else:
            st.info("No numeric columns found to build sparklines for the first month.")

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")



# page 3: Plots (interactive)
elif page == "P 3: Plots (interactive)":
    try:
        df = load_data(DATA_PATH)  

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
                norm = (base[num_cols] - base[num_cols].min()) / (base[num_cols].max() - base[num_cols].min())
                long = norm.assign(time=base["time"]).melt("time", var_name="variable", value_name="value")

                chart = (
                    alt.Chart(long)
                    .mark_line()
                    .encode(
                        x=alt.X("time:T", axis=alt.Axis(labelAngle=90, title=None, format="%b %d")),
                        y=alt.Y("value:Q", title="normalized (0–1)"),
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



# =================== PAGE 4 (Assignment 2 inside Assignment 1) ===================
elif page == "P 4: Elhub Data":

    st.title("P 4: Energy Production Dashboard")
    st.caption("Assignment 2 requirements implemented within Page 4 of Assignment 1")

    # 1) Connect to MongoDB
    try:
        coll = get_mongodb_collection()
    except Exception as e:
        st.error(f"❌ Cannot connect to MongoDB: {e}")
        st.info(
            "**Check `.streamlit/secrets.toml`**:\n\n"
            "```\n[mongodb]\n"
            'connection_string = "mongodb+srv://..."\n'
            'database_name = "elhub"\n'
            'collection_name = "production_mbahour"\n'
            "```"
        )
        st.stop()

    # 2) Load data
    with st.spinner("Loading data from MongoDB..."):
        df = pd.DataFrame(list(coll.find({}, {"_id": 0})))

    # Always show what we actually got
    st.caption(f"Raw rows: {len(df):,}")
    st.caption(f"Raw columns: {sorted(df.columns.tolist())}")

    if df.empty:
        st.warning("⚠️ No data available in MongoDB collection.")
        st.stop()

    # ---------- 3) Coalesce duplicate semantic columns BEFORE renaming ----------
    # If both exist, merge to canonical and drop the twin
    if "startTime" in df.columns and "timestamp" in df.columns:
        st.caption("Coalescing 'startTime' with 'timestamp'")
        df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce").combine_first(
            pd.to_datetime(df["timestamp"], errors="coerce")
        )
        df.drop(columns=["timestamp"], inplace=True)

    if "priceArea" in df.columns and "region" in df.columns:
        st.caption("Coalescing 'priceArea' with 'region'")
        df["priceArea"] = df["priceArea"].astype(object).combine_first(df["region"])
        df.drop(columns=["region"], inplace=True)

    # ---------- 4) Normalize column names (case-insensitive) ----------
    rename_map = {
        # time
        "timestamp": "startTime", "start_time": "startTime", "time": "startTime",
        # price area
        "pricearea": "priceArea", "price_area": "priceArea", "region": "priceArea",
        # group
        "productiongroup": "productionGroup", "production_group": "productionGroup", "group": "productionGroup",
        # value
        "quantitykwh": "quantityKwh", "quantity_kwh": "quantityKwh",
        "kwh": "quantityKwh", "quantity": "quantityKwh",
        "energy_production": "quantityKwh",  # your Atlas field
    }
    renames = {}
    for c in list(df.columns):
        lc = c.strip().lower()
        if lc in rename_map and rename_map[lc] != c:
            renames[c] = rename_map[lc]
    if renames:
        df.rename(columns=renames, inplace=True)
        st.caption(f"Normalized columns: {renames}")

    # If duplicates still remain after rename, keep first occurrence
    if df.columns.duplicated().any():
        dupe_list = [c for c, d in zip(df.columns, df.columns.duplicated()) if d]
        st.caption(f"Dropping duplicate columns: {dupe_list}")
        df = df.loc[:, ~df.columns.duplicated()]

    st.caption(f"Columns after cleanup: {sorted(df.columns.tolist())}")

    # ---------- 5) Detect columns and coerce types ----------
    def pick(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    cols = set(df.columns)
    time_col  = pick(cols, ["startTime", "timestamp", "time"])
    area_col  = pick(cols, ["priceArea", "region", "price_area", "pricearea"])
    group_col = pick(cols, ["productionGroup", "group", "production_group"])
    value_col = pick(cols, ["quantityKwh", "energy_production", "quantity", "kwh", "value"])

    st.caption(f"Detected columns → time: {time_col}, area: {area_col}, group: {group_col}, value: {value_col}")

    missing_roles = [k for k, v in {
        "time": time_col, "area": area_col, "group": group_col, "value": value_col
    }.items() if v is None]

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df["month"] = df[time_col].dt.to_period("M").astype(str)

    if value_col:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # --- show month coverage ---
    if "startTime" in df.columns:
        st.caption(f"Date range: {df['startTime'].min()} → {df['startTime'].max()}")
    if "month" in df.columns:
        month_counts = df["month"].value_counts().sort_index()
        st.caption(f"Months found: {', '.join(month_counts.index.tolist())}")
    with st.expander("📅 Rows per month"):
        st.dataframe(month_counts.rename("rows").to_frame(), use_container_width=True)

    with st.expander("👀 Data preview", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    if missing_roles:
        st.error(f"Missing required roles for plotting: {missing_roles}. "
                 "Fix names in Mongo or extend the mapping above.")
        st.stop()

    st.success(f"✅ Ready. Using → time: {time_col}, area: {area_col}, group: {group_col}, value: {value_col}")

    # ---------- 6) Two-column layout ----------
    left, right = st.columns(2)

    # LEFT: radio + PIE
    with left:
        st.subheader("📊 Production by Price Area")
        areas = sorted(df[area_col].dropna().unique().tolist())
        if not areas:
            st.warning("No price areas found.")
        else:
            area = st.radio("Select price area", areas, key="p4_area")
            pie_df = (
                df[df[area_col] == area]
                .groupby(group_col, dropna=False)[value_col]
                .sum()
                .reset_index()
                .rename(columns={group_col: "Production Group", value_col: "Total (kWh)"})
                .sort_values("Total (kWh)", ascending=False)
            )
            fig_pie = px.pie(
                pie_df,
                values="Total (kWh)",
                names="Production Group",
                title=f"Production Distribution — {area}",
                hole=0.3,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=420)
            st.plotly_chart(fig_pie, use_container_width=True)

    # RIGHT: pills/multiselect + month + LINE
    with right:
        st.subheader("📈 Hourly Production (select groups + month)")
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
            line_df = df[(df[group_col].isin(selected_groups)) & (df["month"] == month)].copy()
            if line_df.empty:
                st.warning(f"No data for {selected_groups} in {month}.")
            else:
                line_df = line_df.sort_values(time_col)
                fig_line = px.line(
                    line_df,
                    x=time_col,
                    y=value_col,
                    color=group_col,
                    title=f"Hourly Production — {month}",
                    labels={time_col: "Date & Time", value_col: "Production (kWh)", group_col: "Group"},
                )
                fig_line.update_layout(hovermode="x unified", height=420)
                st.plotly_chart(fig_line, use_container_width=True)
                st.caption(f"Showing {len(line_df):,} points")
        else:
            st.info("Pick at least one production group and a month.")

    # DOC EXPANDER
    st.markdown("---")
    with st.expander("📚 Data source & pipeline"):
        st.markdown(
            """
- **Source:** Elhub API — 2021 hourly production (PRODUCTION_PER_GROUP_MBA_HOUR)  
- **ETL:** API → Cassandra (staging) → Spark transform → MongoDB Atlas (`elhub.production_mbahour`)  
- **This page:** reads from MongoDB via Streamlit secrets. No credentials in code.
            """
        )
