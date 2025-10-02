import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# App setup
st.set_page_config(page_title="IND320 • Data Check v2", layout="wide")

DATA_PATH = Path("data/open-meteo-subset.csv")

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").set_index("time")
    return df

# Sidebar navigation 
page = st.sidebar.radio(
    "Navigate",
    ["P 1: Home", "P 2: Data table & summary", "P 3: Plots (interactive)", "P 4: "],
    index=0
)

#  numeric columns only
def numeric_cols(df: pd.DataFrame):
    return df.select_dtypes("number").columns.tolist()


# Home page setup
if page == "P 1: Home":
    st.title("IND320 • Streamlit app")
    st.subheader("Welcome to the Weather Data Mini-App")



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



elif page == "P 4: ":
    st.title("P 4 — under construction 🛠️")






