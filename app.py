import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression


# -----------------------
# Load and cache data
# -----------------------
@st.cache_data
def load_data():
    el_nino = fetch_ucirepo(id=122)
    X = el_nino.data.features
    Y = el_nino.data.targets
    df = pd.concat([X, Y], axis=1)
    df["year"] += 1900
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    # Merge ENSO index
    enso = pd.read_csv("data_index.csv").drop_duplicates(subset=["year", "month"])
    df = df.merge(enso, on=["year", "month"], how="left")
    return df, el_nino


# Load once per session
if "df" not in st.session_state:
    st.session_state["df"], st.session_state["el_nino"] = load_data()

df = st.session_state["df"]
el_nino = st.session_state["el_nino"]

# -----------------------
# Sidebar Menu
# -----------------------
st.sidebar.title("🌊 ENSO Dashboard")
menu = [
    "Overview",
    "Missingness",
    "Temporal Coverage",
    "Correlation study",
    "Summary and Conclusion",
]
choice = st.sidebar.radio("Navigation", menu)

# ========================
# Tab 1: Overview
# ========================
if choice == "Overview":
    st.title("🌊 Dataset Overview")

    # Key metrics panel
    total_rows = len(df)
    total_cols = len(df.columns)
    total_missing = df.isna().sum().sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", total_rows)
    col2.metric("Total Features", total_cols)
    col3.metric("Missing Values", total_missing)

    st.markdown("""
    This tab provides initial exploration: column info, summary stats, duplicates, and outliers.
    """)

    with st.expander("📋 Column Information & Missing"):
        col_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Missing Values": df.isna().sum().values,
            }
        )
        st.dataframe(col_info.astype(str))

    with st.expander("📊 Summary Statistics"):
        numeric_df = df.select_dtypes(include=np.number)
        st.dataframe(numeric_df.describe())

    with st.expander("🔁 Duplicate Records"):
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            st.warning(f"{duplicate_count} duplicate rows found")
            st.dataframe(df[df.duplicated()].head(10))
        else:
            st.success("No duplicate rows found")

    with st.expander("⚠️ Outlier Detection (Z-score > 3)"):
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
                outlier_dict[col] = (z_col.abs() > 3).sum()
        st.dataframe(
            pd.DataFrame.from_dict(outlier_dict, orient="index", columns=["Outliers"])
        )

# ========================
# Tab 2: Missingness & Imputation
# ========================
elif choice == "Missingness":
    st.title("💧 Missingness & Humidity Imputation")

    # Metrics
    total_missing = df.isna().sum().sum()
    missing_humidity = df["humidity"].isna().sum()
    col1, col2 = st.columns(2)
    col1.metric("Total Missing", total_missing)
    col2.metric("Humidity Missing", missing_humidity)

    with st.expander("📋 Missingness Summary"):
        summary_table = pd.DataFrame(
            {
                "Missing Values": df.isna().sum(),
                "Missing %": (df.isna().mean() * 100).round(2),
            }
        ).sort_values("Missing Values", ascending=False)
        st.dataframe(summary_table)

    with st.expander("🗺 Missingness Heatmap"):
        excluded_cols = ["day", "month", "year"]
        heatmap_cols = [c for c in df.columns if c not in excluded_cols]
        nan_mask = df[heatmap_cols].copy()
        if "humidity_original" in df.columns:
            nan_mask["humidity"] = df["humidity_original"]
        nan_array = nan_mask.isna().astype(int).to_numpy()
        fig, ax = plt.subplots(figsize=(20, 10))
        im = ax.imshow(nan_array.T, aspect="auto", cmap="cividis")
        ax.set_yticks(range(len(heatmap_cols)))
        ax.set_yticklabels(heatmap_cols)
        ax.set_xticks(np.linspace(0, len(df) - 1, 15).astype(int))
        ax.set_xticklabels(
            df["date"]
            .dt.strftime("%Y-%m-%d")
            .iloc[np.linspace(0, len(df) - 1, 15).astype(int)],
            rotation=45,
        )
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

    st.subheader("Humidity Imputation")
    st.markdown("""
    Missing humidity values are imputed using **linear regression** on air and SST.
    Stochastic noise ensures variability. Only years with <40% missing humidity are included.
    """)

    if st.button("Run Humidity Imputation"):
        if "humidity_original" not in df.columns:
            df["humidity_original"] = df["humidity"].copy()

        feature_cols = ["air_temp", "ss_temp"]
        target_col = "humidity"
        missing_per_year = df.groupby("year")[target_col].apply(
            lambda x: x.isna().mean()
        )
        threshold = 0.4
        years_allowed = missing_per_year[missing_per_year <= threshold].index

        mask_train = df[feature_cols].notna().all(axis=1) & df[target_col].notna()
        X_train = df.loc[mask_train, feature_cols]
        y_train = df.loc[mask_train, target_col]
        model = LinearRegression().fit(X_train, y_train)
        residual_std = np.std(y_train - model.predict(X_train))

        mask_impute = (
            df[feature_cols].notna().all(axis=1)
            & df[target_col].isna()
            & df["year"].isin(years_allowed)
        )
        X_missing = df.loc[mask_impute, feature_cols]

        # stochastic
        n_sim = 100
        predictions = np.mean(
            [
                model.predict(X_missing)
                + np.random.normal(0, residual_std, X_missing.shape[0])
                for _ in range(n_sim)
            ],
            axis=0,
        )
        df.loc[mask_impute, target_col] = predictions
        st.session_state["df"] = df.copy()

        st.success("Humidity imputation complete ✅")
        st.write(f"Remaining missing humidity: {df['humidity'].isna().sum()}")

        # Plot imputed
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.scatter(
            df.loc[mask_impute, "date"],
            df.loc[mask_impute, target_col],
            color="orange",
            label="Imputed",
        )
        ax.plot(df["date"], df["humidity_original"], label="Original", alpha=0.7)
        ax.set_xlabel("Date")
        ax.set_ylabel("Humidity")
        ax.legend()
        st.pyplot(fig)

# ========================
# Tab 3: Temporal Coverage
# ========================
elif choice == "Temporal Coverage":
    st.header("📅 Temporal Coverage & ENSO")
    st.markdown("Explore variables over time and highlight ENSO events.")

    # Select variable
    numeric_cols = ["humidity", "air_temp", "ss_temp", "zon_winds", "mer_winds"]
    feature = st.selectbox(
        "Variable", numeric_cols, index=numeric_cols.index("ss_temp")
    )

    df_daily = df.groupby("date")[numeric_cols + ["ANOM"]].mean().reset_index()
    # ENSO shading
    el_thresh, la_thresh = 1.0, -1.0
    df_daily["event"] = np.where(
        df_daily["ANOM"] > el_thresh,
        "El Niño",
        np.where(df_daily["ANOM"] < la_thresh, "La Niña", None),
    )

    fig = px.line(
        df_daily,
        x="date",
        y=feature,
        title=f"{feature} Over Time",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ========================
# Tab 4: Correlation Study
# ========================
elif choice == "Correlation study":
    st.header("📊 Correlation and Relationships")
    selected_features = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]
    subset_df = df[selected_features].dropna()

    fig_corr = px.imshow(
        subset_df.corr(),
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    fig_pair = px.scatter_matrix(
        df[selected_features],
        dimensions=selected_features,
        color="air_temp",
        color_continuous_scale="RdBu_r",
        opacity=0.5,
    )
    st.plotly_chart(fig_pair, use_container_width=True)

# ========================
# Tab 5: Summary & Conclusion
# ========================
elif choice == "Summary and Conclusion":
    st.title("📖 Key Insights")
    st.markdown("""
- ENSO impacts SST, air temperature, and humidity.
- Strong correlation between air_temp and SST.
- Seasonal cycles and ENSO events are clear.
- Missing humidity values were imputed using regression with stochastic noise.
""")

# -----------------------
# End of App
# -----------------------
