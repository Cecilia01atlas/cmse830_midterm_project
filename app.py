import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from ucimlrepo import fetch_ucirepo
import statsmodels.api as sm


# ---------------------------------------------------
# Load data once at start (cache for performance)
# ---------------------------------------------------
@st.cache_data
def load_data():
    el_nino = fetch_ucirepo(id=122)
    X = el_nino.data.features
    Y = el_nino.data.targets
    df = pd.concat([X, Y], axis=1)

    # Fix year and create date column
    df["year"] = df["year"] + 1900
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


df = load_data()

st.title("🌊 El Niño / La Niña Data Explorer")
st.write(
    "Exploring sea surface temperature trends and their relationship with other variables."
)

# ---------------------------------------------------
# 1. Sea Surface Temperature: Daily vs Monthly
# ---------------------------------------------------
st.header("1️⃣ Sea Surface Temperature Over Time")

numeric_cols = df.select_dtypes(include="number").columns

# Daily averages
df_daily = df.groupby("date")[numeric_cols].mean().reset_index()

# Monthly averages
df_monthly = df.set_index("date")[numeric_cols].resample("M").mean().reset_index()

fig_temp = go.Figure()
fig_temp.add_trace(
    go.Scatter(
        x=df_daily["date"],
        y=df_daily["ss_temp"],
        mode="markers",
        marker=dict(size=3, color="royalblue", opacity=0.5),
        name="Daily Avg",
    )
)
fig_temp.add_trace(
    go.Scatter(
        x=df_monthly["date"],
        y=df_monthly["ss_temp"],
        mode="lines",
        line=dict(color="darkorange", width=2),
        name="Monthly Avg",
    )
)
fig_temp.update_layout(
    title="Sea Surface Temperature: Daily vs Monthly Average",
    xaxis_title="Date",
    yaxis_title="Sea Surface Temperature (°C)",
    template="plotly_white",
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig_temp, use_container_width=True)

# ---------------------------------------------------
# 2. Correlation Heatmap
# ---------------------------------------------------
st.header("2️⃣ Correlation Heatmap")

selected_features = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]

# Ensure numeric
subset_df = df[selected_features].apply(pd.to_numeric, errors="coerce").dropna()

correlation_matrix = subset_df.corr().values

fig_corr = ff.create_annotated_heatmap(
    z=correlation_matrix,
    x=selected_features,
    y=selected_features,
    colorscale="Viridis",
    showscale=True,
)
fig_corr.update_layout(
    title="Correlation Heatmap of Climate Variables",
    xaxis_title="Features",
    yaxis_title="Features",
)
st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------
# 3. Scatterplot with regression line
# ---------------------------------------------------
st.header("3️⃣ Air Temperature vs Sea Surface Temperature")

df_plot = df[["air_temp", "ss_temp"]].apply(pd.to_numeric, errors="coerce").dropna()

fig_scatter = px.scatter(
    df_plot,
    x="air_temp",
    y="ss_temp",
    trendline="ols",
    labels={
        "air_temp": "Air Temperature (°C)",
        "ss_temp": "Sea Surface Temperature (°C)",
    },
    title="Scatterplot with Regression Line",
)
st.plotly_chart(fig_scatter, use_container_width=True)
