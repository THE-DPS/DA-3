from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "day_cleaned.csv"
METRICS = ROOT / "outputs" / "model_metrics.json"

st.set_page_config(page_title="Bike-Sharing Demand Explorer", layout="wide")
st.title("Bike-Sharing Demand Analysis and Prediction")
st.caption("Historical Capital Bikeshare data from Washington, D.C. (2011–2012). This is not a live feed or a current operational forecast.")
st.markdown("**Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) · DOI: [10.24432/C5W894](https://doi.org/10.24432/C5W894)")

if not DATA.exists():
    st.error("Cleaned data not found. Run `python src/download_data.py` and `python src/prepare_data.py` from the project directory.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=["dteday"])

df = load_data(str(DATA))
start, end = df["dteday"].min().date(), df["dteday"].max().date()
with st.sidebar:
    st.header("Filters")
    dates = st.date_input("Date range", value=(start, end), min_value=start, max_value=end)
    season_codes = sorted(df["season"].dropna().unique().tolist())
    seasons = st.multiselect("Season code", season_codes, default=season_codes)
    weather_codes = sorted(df["weathersit"].dropna().unique().tolist())
    weather = st.multiselect("Weather situation code", weather_codes, default=weather_codes)

if isinstance(dates, (tuple, list)) and len(dates) == 2:
    lo, hi = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
else:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
filtered = df[(df["dteday"] >= lo) & (df["dteday"] <= hi)]
if seasons:
    filtered = filtered[filtered["season"].isin(seasons)]
else:
    filtered = filtered.iloc[0:0]
if weather:
    filtered = filtered[filtered["weathersit"].isin(weather)]
else:
    filtered = filtered.iloc[0:0]

if filtered.empty:
    st.warning("No records match the selected filters. Expand the date range or select additional categories.")
    st.stop()

a, b, c, d = st.columns(4)
a.metric("Total rentals in selection", f"{int(filtered['cnt'].sum()):,}")
b.metric("Mean rentals/day", f"{filtered['cnt'].mean():,.1f}")
c.metric("Median rentals/day", f"{filtered['cnt'].median():,.1f}")
d.metric("Days represented", f"{len(filtered):,}")

left, right = st.columns(2)
with left:
    fig = px.line(filtered.sort_values("dteday"), x="dteday", y="cnt", title="Daily rental demand")
    fig.update_layout(xaxis_title="Date", yaxis_title="Total daily rentals")
    st.plotly_chart(fig, use_container_width=True)
with right:
    by_season = filtered.groupby("season", as_index=False)["cnt"].mean()
    fig = px.bar(by_season, x="season", y="cnt", title="Mean rentals by season code")
    fig.update_layout(xaxis_title="Season code (see source data dictionary)", yaxis_title="Mean daily rentals")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    fig = px.histogram(filtered, x="cnt", nbins=25, title="Distribution of daily rentals")
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.scatter(filtered, x="temp", y="cnt", color="weathersit",
                     hover_data=["dteday"], title="Demand vs normalized temperature")
    fig.update_layout(xaxis_title="Normalized temperature (source scale)", yaxis_title="Daily rentals")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Model evaluation (fixed chronological holdout)")
if METRICS.exists():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    st.write(f"Test period: **{metrics.get('test_start')} to {metrics.get('test_end')}** · "
             f"Train rows: {metrics.get('train_rows')} · Test rows: {metrics.get('test_rows')}")
    st.dataframe(pd.DataFrame(metrics.get("metrics", [])), use_container_width=True)
    st.caption("These metrics are fixed for the holdout set and do not change when descriptive filters change.")
else:
    st.info("Model metrics not found. Run `python src/train_model.py` to generate them.")

st.subheader("Filtered records")
st.dataframe(filtered[["dteday", "season", "weathersit", "temp", "hum", "windspeed", "cnt"]], use_container_width=True)
st.caption("Interpretation is descriptive, not causal. The dataset is historical and geographically limited.")
