import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    # Fetch dataset
    el_nino = fetch_ucirepo(id=122)
    X = el_nino.data.features
    Y = el_nino.data.targets
    df = pd.concat([X, Y], axis=1)

    # Fix year and create date column (same as your working code)
    df["year"] = df["year"] + 1900
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # --- Merge second dataset safely ---
    enso = pd.read_csv("data_index.csv")
    # Only merge year and month; drop duplicates in enso if needed
    enso = enso.drop_duplicates(subset=["year", "month"])
    df = df.merge(enso, on=["year", "month"], how="left")

    return df, el_nino  # keep el_nino for column info if needed


# Load dataset
df, el_nino = load_data()

# --- Sidebar Menu ---
menu = ["Overview", "Visualization 1", "Visualization 2", "Visualization 3"]
choice = st.sidebar.radio("Menu", menu)

# --- Tab 1: Overview ---
if choice == "Overview":
    st.title("Dataset Overview")

    # 1️⃣ Column names only
    st.subheader("Column Information")
    st.write(pd.DataFrame({"Columns": df.columns}))

    # 2️⃣ First 15 rows
    st.subheader("First 15 Rows of the Dataset")
    st.dataframe(df.head(15))

    # 3️⃣ Summary statistics (exclude year, month, day, date)
    st.subheader("Summary Statistics")
    numeric_df = df.drop(columns=["year", "month", "day", "date"], errors="ignore")
    st.write(numeric_df.describe())

    # 4️⃣ Temporal coverage over time (year-month)
    st.subheader("Temporal Coverage Over Time (Year-Month)")
    df["year_month"] = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    ym_counts = df["year_month"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(15, 4))
    ym_counts.plot(ax=ax)
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Number of Records")
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # 5️⃣ Missing values
    st.subheader("Missing Values per Column")
    missing_counts = df.isna().sum()
    st.bar_chart(missing_counts)

    # 6️⃣ Duplicates
    st.subheader("Duplicates in Dataset")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    # 7️⃣ Outlier detection
    st.subheader("Outlier Detection (Z-score > 3)")
    numeric_cols = [
        "zon_winds",
        "mer_winds",
        "humidity",
        "air_temp",
        "ss_temp",
        "ClimAdjust",
        "ANOM",
    ]
    outlier_dict = {}
    for col in numeric_cols:
        if col in df.columns:
            z_col = (df[col] - df[col].mean()) / df[col].std()
            outliers = (z_col.abs() > 3).sum()
            outlier_dict[col] = outliers
    st.write(pd.DataFrame.from_dict(outlier_dict, orient="index", columns=["Outliers"]))
