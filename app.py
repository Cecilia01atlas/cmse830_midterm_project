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
# 2. Regression Heatmap (WLS)
# ---------------------------------------------------
st.header("2️⃣ Weighted Regression Coefficients")

feature_columns = ["zon_winds", "mer_winds", "humidity", "air_temp"]
target_variable = "ss_temp"

year_counts = df["year"].value_counts()
df["weight"] = df["year"].map(lambda y: 1 / year_counts[y])

df_numeric = df[feature_columns + [target_variable, "weight"]].apply(
    pd.to_numeric, errors="coerce"
)

coef_dict = {}
for col in feature_columns:
    X = sm.add_constant(df_numeric[[col]])
    y = df_numeric[target_variable]
    w = df_numeric["weight"]

    mask = X.notnull().all(axis=1) & y.notnull() & np.isfinite(w)
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]
    w_clean = w.loc[mask]

    model = sm.WLS(y_clean, X_clean, weights=w_clean)
    results = model.fit()
    coef_dict[col] = results.params[col]

coef_df = pd.DataFrame.from_dict(coef_dict, orient="index", columns=["Slope"])

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=coef_df.values.T,
        x=coef_df.index,
        y=[target_variable + " (slope)"],
        colorscale="Viridis",
        showscale=True,
    )
)
fig_heatmap.update_layout(
    title="Weighted Regression Coefficients",
    xaxis_title="Features",
    yaxis_title="Target",
)
st.plotly_chart(fig_heatmap, use_container_width=True)

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
