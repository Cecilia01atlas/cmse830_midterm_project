import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fetch_ucirepo import fetch_ucirepo  # keep this, it works locally


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

    st.subheader("First 15 Rows of the Dataset")
    st.dataframe(df.head(15))

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Column Information")
    st.write(pd.DataFrame(el_nino.variables))

    st.subheader("Missing Values per Column")
    missing_counts = df.isna().sum()
    st.bar_chart(missing_counts)

    st.subheader("Duplicates in Dataset")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    st.subheader("Temporal Coverage by Year")
    year_counts = df["year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    year_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Records")
    st.pyplot(fig)

    st.subheader("Temporal Coverage by Month")
    month_counts = df["month"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    month_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Records")
    st.pyplot(fig)

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
        if col in df.columns:  # only calculate if column exists after merge
            z_col = (df[col] - df[col].mean()) / df[col].std()
            outliers = (z_col.abs() > 3).sum()
            outlier_dict[col] = outliers
    st.write(pd.DataFrame.from_dict(outlier_dict, orient="index", columns=["Outliers"]))
