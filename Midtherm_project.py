# pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler

# ---------- IDA ----------
# ------------------------------------------------------

# ---------------------- 1. Getting overview of the dataset: ---------------------------
# fetch dataset
el_nino = fetch_ucirepo(id=122)

# data (as pandas dataframes)
X = el_nino.data.features
Y = el_nino.data.targets
df = pd.concat([X, Y], axis=1)

# print first 5 rows of features
print("\nFirst 5 rows of X:")
print(X.head())

# variable information
print(el_nino.variables)

# Inspect the columns to find the date components
print("colums", df.columns)

# ---- Let's check temporal coverage ----
# Coverage by year
year_counts = df["year"].value_counts().sort_index()
print("\nNumber of records per year:")
print(year_counts)

# Plot records per year
plt.figure(figsize=(12, 4))
year_counts.plot(kind="bar")
plt.title("Number of Observations per Year")
plt.xlabel("Year")
plt.ylabel("Count")
# plt.show()

# Coverage by month
month_counts = df["month"].value_counts().sort_index()
print("\nNumber of records per month:")
print(month_counts)

plt.figure(figsize=(10, 4))
month_counts.plot(kind="bar")
plt.title("Number of Observations per Month")
plt.xlabel("Month")
plt.ylabel("Count")
# plt.show()

# Check for combinations of year-month (good for spotting gaps)
df["year_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
ym_counts = df["year_month"].value_counts().sort_index()

plt.figure(figsize=(15, 4))
ym_counts.plot()
plt.title("Temporal Coverage Over Time (Year-Month)")
plt.xlabel("Year-Month")
plt.ylabel("Number of Records")
plt.xticks(rotation=90)
plt.tight_layout()
# plt.show()

# Count of missing values per column
missing_counts = df.isna().sum()
print("Missing values per column:", missing_counts)

# -------------------------- 2. Data cleaning and preparing -----------------------------

# Fix year by adding 1900 (e.g. 80 -> 1980)
df["year"] = df["year"] + 1900

df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")

# Sort by date
df = df.sort_values("date").reset_index(drop=True)

# Heatmap for missing values
nan_mask = df.isna()
nan_array = nan_mask.astype(int).to_numpy()

plt.figure(figsize=(14, 6))
im = plt.imshow(nan_array.T, interpolation="nearest", aspect="auto", cmap="viridis")

plt.ylabel("Features")
plt.title("Visualizing Missing Values Over Time")

plt.yticks(range(len(df.columns)), df.columns)

n_rows = len(df)
n_ticks = min(10, n_rows)
tick_positions = np.linspace(0, n_rows - 1, n_ticks).astype(int)
tick_labels = df.loc[tick_positions, "date"].dt.strftime("%Y-%m-%d")

plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")
plt.xlabel("Date")

plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# Getting and printing the max length for temporal missingness:
def missing_streak_lengths(series):
    dates = series.index[series.isna()]
    if len(dates) == 0:
        return []
    streaks = []
    current = [dates[0]]
    for d in dates[1:]:
        if (d - current[-1]).days == 1:
            current.append(d)
        else:
            streaks.append(current)
            current = [d]
    streaks.append(current)
    return [len(s) for s in streaks]


df = df.set_index("date")
for col in ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]:
    lengths = missing_streak_lengths(df[col])
    if lengths:
        print(
            f"{col}: avg gap = {np.mean(lengths):.1f} days, max gap = {np.max(lengths)} days, #gaps = {len(lengths)}"
        )
df = df.reset_index()

# ---------- EDA ----------
# 1. Temporal plots:
fig = px.scatter(
    df,
    x="date",
    y="ss_temp",
    title="Sea Surface Temperature Over Time (Raw Scatter)",
    labels={"date": "Date", "ss_temp": "Sea Surface Temperature (°C)"},
    opacity=0.4,
)
fig.show()

# Select numeric columns
numeric_cols = df.select_dtypes(include="number").columns

# Daily averages (date is a column)
df_daily = df.groupby("date")[numeric_cols].mean().reset_index()

# Set date as index for resampling
df.set_index("date", inplace=True)

# Resample to monthly averages
df_monthly = df[numeric_cols].resample("M").mean().reset_index()

# Plot
fig = go.Figure()

# Daily average (scatter for visibility)
fig.add_trace(
    go.Scatter(
        x=df_daily["date"],
        y=df_daily["ss_temp"],
        mode="markers",
        marker=dict(size=4, color="royalblue", opacity=0.4),
        name="Daily Average",
    )
)

# Monthly average (smooth line)
fig.add_trace(
    go.Scatter(
        x=df_monthly["date"],
        y=df_monthly["ss_temp"],
        mode="lines",
        line=dict(color="darkorange", width=2),
        name="Monthly Average",
    )
)

fig.update_layout(
    title="Sea Surface Temperature: Daily vs Monthly Average",
    xaxis_title="Date",
    yaxis_title="Sea Surface Temperature (°C)",
    template="plotly_white",
    legend=dict(x=0.02, y=0.98),
)

fig.show()

# 2. Linear regression:
# Select the specific features you want
selected_features = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]

# Subset the dataframe and ensure all are numeric
subset_df = df[selected_features].apply(pd.to_numeric, errors="coerce")

# Compute correlation matrix
correlation_matrix = subset_df.corr().values

# Create interactive heatmap
fig_heatmap = ff.create_annotated_heatmap(
    z=correlation_matrix,
    x=selected_features,
    y=selected_features,
    colorscale="Viridis",
    showscale=True,
)

fig_heatmap.update_layout(
    title="Correlation Heatmap (zon_winds, mer_winds, humidity, air_temp)",
    xaxis_title="Features",
    yaxis_title="Features",
)

fig_heatmap.show()

# Ensure numeric values for safety
df_plot = df[["air_temp", "ss_temp"]].apply(pd.to_numeric, errors="coerce").dropna()

# Create interactive scatterplot with regression line
fig_scatter = px.scatter(
    df_plot,
    x="air_temp",
    y="ss_temp",
    trendline="ols",  # ordinary least squares regression line
    labels={"air_temp": "Air Temperature", "ss_temp": "Sea Surface Temperature"},
    title="Interactive Scatterplot of Air Temperature vs Sea Surface Temperature",
)

# fig_scatter.show()

year_counts = df["year"].value_counts()

# Assign a weight to each row = 1 / number of rows for that year to balance out yearly imbalance
df["weight"] = df["year"].map(lambda y: 1 / year_counts[y])

# Features and target
target_variable = "ss_temp"
feature_columns = ["zon_winds", "mer_winds", "humidity", "air_temp"]

# Ensure numeric
df_numeric = df[feature_columns + [target_variable, "weight"]].apply(
    pd.to_numeric, errors="coerce"
)

# Calculate slope of linear regression with WLS model from the module statsmodel
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

# DataFrame for heatmap
coef_df = pd.DataFrame.from_dict(coef_dict, orient="index", columns=["Slope"])

# Plot interactive heatmap with weighs according to year
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
    title="Weighted Regression Coefficients (Interactive)",
    xaxis_title="Features",
    yaxis_title="Target",
    yaxis=dict(tickmode="array"),
)

fig_heatmap.show()
