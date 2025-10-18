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
menu = [
    "Overview",
    "Missingness",
    "Visualization 1",
    "Visualization 2",
    "Visualization 3",
]
choice = st.sidebar.radio("Menu", menu)

# --- Tab 1: Overview ---
if choice == "Overview":
    st.title("Dataset Overview")

    # 1️⃣ Column information
    st.subheader("Column Information")

    col_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Role": [
                el_nino.variables[col]["role"] if col in el_nino.variables else "-"
                for col in df.columns
            ],
            "Type": df.dtypes.values,
            "Missing Values": df.isna().sum().values,
        }
    )
    st.dataframe(col_info)

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

    # 5️⃣ Duplicates
    st.subheader("Duplicates in Dataset")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    # 6️⃣ Outlier detection
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


# --- Tab 2: Missingness ---
elif choice == "Missingness":
    st.title("Missingness Analysis")

    # --- Missing values per column ---
    st.subheader("Missing Values per Column")
    missing_counts = df.isna().sum().sort_values(ascending=False)
    st.bar_chart(missing_counts)

    # --- Missingness summary table ---
    st.subheader("Missingness Summary Table")
    summary_table = pd.DataFrame(
        {
            "Missing Values": df.isna().sum(),
            "Missing %": (df.isna().mean() * 100).round(2),
        }
    ).sort_values("Missing Values", ascending=False)
    st.dataframe(summary_table)

    # --- Missingness heatmap ---
    st.subheader("Heatmap of Missing Values Over Time")
    nan_mask = df.isna()
    nan_array = nan_mask.astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(nan_array.T, interpolation="nearest", aspect="auto", cmap="viridis")

    ax.set_ylabel("Features")
    ax.set_title("Missing Values Heatmap (1 = Missing, 0 = Present)")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)

    n_rows = len(df)
    n_ticks = min(10, n_rows)
    tick_positions = np.linspace(0, n_rows - 1, n_ticks).astype(int)
    tick_labels = df.loc[tick_positions, "date"].dt.strftime("%Y-%m-%d")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.colorbar(im, ax=ax, label="Missingness")
    plt.tight_layout()
    st.pyplot(fig)

    # --- Humidity Imputation ---
    st.subheader("Humidity Imputation")

    st.markdown("""
    Missing humidity values are imputed using a **linear regression** model with 
    air temperature and sea surface temperature as predictors. 
    Stochastic noise is added to mimic natural variability.
    """)

    if st.button("Run Humidity Imputation"):
        from sklearn.linear_model import LinearRegression

        # Copy original humidity
        df["humidity_original"] = df["humidity"].copy()

        feature_cols = ["air_temp", "ss_temp"]
        target_col = "humidity"

        # Missing fraction per year
        missing_per_year = df.groupby("year")[target_col].apply(
            lambda x: x.isna().mean()
        )
        threshold = 0.5
        years_to_impute = missing_per_year[missing_per_year > threshold].index

        # Training data
        mask_train = df[feature_cols].notna().all(axis=1) & df[target_col].notna()
        X_train = df.loc[mask_train, feature_cols]
        y_train = df.loc[mask_train, target_col]

        model = LinearRegression()
        model.fit(X_train, y_train)
        residual_std = np.std(y_train - model.predict(X_train))

        # Rows to impute
        mask_impute = (
            df[feature_cols].notna().all(axis=1)
            & df[target_col].isna()
            & df["year"].isin(years_to_impute)
        )
        X_missing = df.loc[mask_impute, feature_cols]

        n_simulations = 100
        stochastic_predictions = []
        for _ in range(n_simulations):
            noise = np.random.normal(0, residual_std, size=X_missing.shape[0])
            stochastic_predictions.append(model.predict(X_missing) + noise)
        y_imputed = np.mean(stochastic_predictions, axis=0)

        df.loc[mask_impute, target_col] = y_imputed

        # Plot before vs imputed
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.scatter(
            df.loc[mask_impute, "date"],
            df.loc[mask_impute, target_col],
            color="orange",
            s=20,
            label="Imputed",
        )
        ax.plot(
            df["date"], df["humidity_original"], label="Original Humidity", alpha=0.7
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Humidity (%)")
        ax.set_title("Humidity After Imputation")
        ax.legend()
        st.pyplot(fig)

        st.success("Humidity imputation complete ✅")
        st.write("Remaining missing values per column:")
        st.write(df.isna().sum())
