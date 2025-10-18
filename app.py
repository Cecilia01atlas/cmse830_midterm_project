import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------
# Load data once at start (cache for performance)
# ---------------------------------------------------
@st.cache_data
def load_data():
    el_nino = fetch_ucirepo(id=122)
    X = el_nino.data.features
    Y = el_nino.data.targets
    df = pd.concat([X, Y], axis=1)

    # Fix year and create date
    df["year"] = df["year"] + 1900
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Merge ENSO index dataset
    enso = pd.read_csv("data_index.csv").drop_duplicates(subset=["year", "month"])
    df = df.merge(enso, on=["year", "month"], how="left")

    return df, el_nino


# Load data
df, el_nino = load_data()

# Ensure session_state persistence
if "df" not in st.session_state:
    st.session_state["df"] = df.copy()

df = st.session_state["df"].copy()

# --- Sidebar Menu ---
menu = ["Overview", "Missingness", "Temporal Coverage", "Visualization 2"]
choice = st.sidebar.radio("Menu", menu)

# ----------------------------
# Tab 1: Overview
# ----------------------------
if choice == "Overview":
    st.title("Dataset Overview")

    # Column info
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

    # First 15 rows
    st.subheader("First 15 Rows of the Dataset")
    st.dataframe(df.head(15))

    # Summary stats
    st.subheader("Summary Statistics")
    numeric_df = df.drop(columns=["year", "month", "day", "date"], errors="ignore")
    st.write(numeric_df.describe())

    # Temporal coverage (year-month)
    st.subheader("Temporal Coverage Over Time")
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

    # Duplicates
    st.subheader("Duplicates in Dataset")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    # Outlier detection
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
            outlier_dict[col] = (z_col.abs() > 3).sum()
    st.write(pd.DataFrame.from_dict(outlier_dict, orient="index", columns=["Outliers"]))

# ----------------------------
# Tab 2: Missingness
# ----------------------------
elif choice == "Missingness":
    st.title("Missingness Analysis")

    # Missing values bar chart
    st.subheader("Missing Values per Column")
    missing_counts = df.isna().sum().sort_values(ascending=False)
    st.bar_chart(missing_counts)

    # Summary table
    st.subheader("Missingness Summary Table")
    summary_table = pd.DataFrame(
        {
            "Missing Values": df.isna().sum(),
            "Missing %": (df.isna().mean() * 100).round(2),
        }
    ).sort_values("Missing Values", ascending=False)
    st.dataframe(summary_table)

    # Heatmap of missingness (original humidity)
    st.subheader("Heatmap of Missing Values Over Time")
    nan_mask = df.copy()
    if "humidity_original" in df.columns:
        nan_mask["humidity"] = df["humidity_original"]
    nan_array = nan_mask.isna().astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(24, 12))
    im = ax.imshow(nan_array.T, interpolation="nearest", aspect="auto", cmap="viridis")
    ax.set_ylabel("Features")
    ax.set_title("Missing Values Heatmap (1=Missing, 0=Present)")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    n_rows = len(df)
    n_ticks = min(15, n_rows)
    tick_positions = np.linspace(0, n_rows - 1, n_ticks).astype(int)
    tick_labels = df.loc[tick_positions, "date"].dt.strftime("%Y-%m-%d")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.colorbar(im, ax=ax, label="Missingness")
    plt.tight_layout()
    st.pyplot(fig)

    # Humidity imputation
    st.subheader("Humidity Imputation")
    st.markdown("""
Missing humidity values are imputed using **linear regression** with air temperature and sea surface temperature as predictors.  
Stochastic noise mimics natural variability.
""")
    if st.button("Run Humidity Imputation"):
        df = st.session_state["df"].copy()
        if "humidity_original" not in df.columns:
            df["humidity_original"] = df["humidity"].copy()

        feature_cols = ["air_temp", "ss_temp"]
        target_col = "humidity"

        # Identify years to impute
        missing_per_year = df.groupby("year")[target_col].apply(
            lambda x: x.isna().mean()
        )
        threshold = 0.5
        years_to_impute = missing_per_year[missing_per_year > threshold].index

        # Train model
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

        # Stochastic predictions
        n_sim = 100
        stochastic_preds = [
            model.predict(X_missing)
            + np.random.normal(0, residual_std, X_missing.shape[0])
            for _ in range(n_sim)
        ]
        y_imputed = np.mean(stochastic_preds, axis=0)

        df.loc[mask_impute, target_col] = y_imputed
        st.session_state["df"] = df.copy()

        # Plot
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

# ----------------------------
# Tab 3: Temporal Coverage + ENSO Story
# ----------------------------
elif choice == "Temporal Coverage":
    st.title("📅 Temporal Coverage & ENSO Story")
    st.markdown("""
El Niño and La Niña are opposite phases of the **ENSO** phenomenon.  
They influence **sea surface temperature (SST)**, **air temperature**, and **humidity**.  

Red = El Niño, Blue = La Niña. Explore how each variable responds to ENSO events.
""")

    # Feature dropdown
    numeric_cols = ["humidity", "air_temp", "ss_temp", "zon_winds", "mer_winds"]
    feature = st.selectbox(
        "Select variable to visualize:",
        numeric_cols,
        index=numeric_cols.index("ss_temp"),
    )

    df = st.session_state["df"].copy()
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")

    # Scatter colored by ENSO
    anom_abs = max(abs(df["ANOM"].min()), abs(df["ANOM"].max()))
    fig_scatter = px.scatter(
        df,
        x="date",
        y=feature,
        color="ANOM",
        color_continuous_scale="RdBu_r",
        opacity=0.5,
        labels={feature: feature.replace("_", " ").title(), "ANOM": "ENSO Index"},
        title=f"{feature.replace('_', ' ').title()} Over Time (ENSO-Colored)",
    )
    fig_scatter.update_layout(
        coloraxis=dict(
            cmin=-anom_abs,
            cmax=anom_abs,
            cmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title="ENSO Index"),
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ENSO-shaded line plot
    st.subheader(f"Daily {feature.replace('_', ' ').title()} with ENSO Shading")
    show_shading = st.checkbox("Show ENSO shading", value=True)

    df_daily = df.groupby("date")[numeric_cols + ["ANOM"]].mean().reset_index()
    el_thresh, la_thresh = 1.0, -1.0
    df_daily["event"] = np.where(
        df_daily["ANOM"] > el_thresh,
        "El Niño",
        np.where(df_daily["ANOM"] < la_thresh, "La Niña", None),
    )

    shading_periods = []
    current_event = None
    start_date = None
    for _, row in df_daily.iterrows():
        event = row["event"]
        date = row["date"]
        if event != current_event:
            if current_event is not None:
                shading_periods.append(
                    {"event": current_event, "start": start_date, "end": date}
                )
            current_event = event
            start_date = date
    if current_event is not None:
        shading_periods.append(
            {
                "event": current_event,
                "start": start_date,
                "end": df_daily["date"].iloc[-1],
            }
        )

    fig_line = go.Figure()
    if show_shading:
        for period in shading_periods:
            if period["event"] is None:
                continue
            color = (
                "rgba(255,0,0,0.1)"
                if period["event"] == "El Niño"
                else "rgba(0,0,255,0.1)"
            )
            fig_line.add_vrect(
                x0=period["start"],
                x1=period["end"],
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0,
            )

    fig_line.add_trace(
        go.Scatter(
            x=df_daily["date"],
            y=df_daily[feature],
            mode="lines",
            line=dict(color="royalblue", width=1.5),
            name=f"Daily {feature}",
        )
    )
    fig_line.update_layout(
        title=f"Daily {feature.replace('_', ' ').title()}"
        + (" with ENSO Shading" if show_shading else ""),
        template="plotly_white",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Monthly violin
    st.subheader("Monthly Distribution")
    df["month_cat"] = pd.Categorical(df["month"], categories=range(1, 13), ordered=True)
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    seasonal_colors = [
        "#003366",
        "#3366cc",
        "#6699ff",
        "#99ccff",
        "#ffcc99",
        "#ff6666",
        "#cc0033",
        "#ff6666",
        "#ffcc99",
        "#99ccff",
        "#6699ff",
        "#3366cc",
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        x="month_cat",
        y=feature,
        data=df,
        inner="box",
        cut=0,
        linewidth=1.2,
        palette=seasonal_colors,
        ax=ax,
    )
    ax.set_title(
        f"Distribution of {feature.replace('_', ' ').title()} by Month (All Years)",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel(feature.replace("_", " ").title(), fontsize=12)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    sns.despine()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)

# ----------------------------
# Tab 4: Visualization 2 - Correlation and Binned Analysis
# ----------------------------
elif choice == "Visualization 2":
    st.title("🌐 Correlation and SST-AirTemp Relationships")
    st.markdown("""
Explore **how ENSO events influence variable relationships**:
- Correlation heatmap
- Scatter plots and regression
- Pairwise relationships
- Binned line plots
""")

    # Correlation heatmap
    selected_features = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]
    subset_df = df[selected_features].apply(pd.to_numeric, errors="coerce").dropna()
    corr_matrix = subset_df.corr().values
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    masked_corr = np.where(mask, None, corr_matrix)[::-1]

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=masked_corr,
            x=selected_features,
            y=selected_features[::-1],
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            hoverongaps=False,
        )
    )
    for i, row in enumerate(masked_corr):
        for j, val in enumerate(row):
            if val is not None:
                fig_corr.add_annotation(
                    x=selected_features[j],
                    y=selected_features[::-1][i],
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(color="black"),
                )
    fig_corr.update_layout(title="Correlation Heatmap", xaxis=dict(tickangle=-45))
    st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter with regression
    soft_blue = "#4a7c9b"
    fig_scatter2 = px.scatter(
        df,
        x="air_temp",
        y="ss_temp",
        opacity=0.4,
        trendline="ols",
        trendline_color_override="red",
        labels={
            "air_temp": "Air Temperature (°C)",
            "ss_temp": "Sea Surface Temperature (°C)",
        },
        title="Air Temperature vs Sea Surface Temperature",
    )
    fig_scatter2.update_traces(
        marker=dict(size=5, color=soft_blue, line=dict(width=0.5, color="#2f4f5f"))
    )
    fig_scatter2.update_layout(
        title=dict(font=dict(size=18), x=0.5), plot_bgcolor="white"
    )
    st.plotly_chart(fig_scatter2, use_container_width=True)

    # Scatter matrix
    fig_matrix = px.scatter_matrix(
        df[selected_features],
        dimensions=selected_features,
        color="air_temp",
        labels={col: col.replace("_", " ").title() for col in selected_features},
        title="Pairwise Relationships between SST and Other Features",
        opacity=0.5,
    )
    fig_matrix.update_traces(diagonal_visible=True)
    fig_matrix.update_layout(height=800, width=800)
    st.plotly_chart(fig_matrix, use_container_width=True)

    # Binned line plot
    df["air_temp_bin"] = pd.cut(df["air_temp"], bins=20)
    avg_sst = (
        df.groupby(["air_temp_bin", "month"])["ss_temp"]
        .mean()
        .reset_index(name="avg_ss_temp")
    )
    month_labels_map = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    avg_sst["month_name"] = avg_sst["month"].map(month_labels_map)
    avg_sst["month_name"] = pd.Categorical(
        avg_sst["month_name"], categories=list(month_labels_map.values()), ordered=True
    )
    avg_sst["air_temp_bin_str"] = avg_sst["air_temp_bin"].astype(str)

    fig_line2 = px.line(
        avg_sst,
        x="air_temp_bin_str",
        y="avg_ss_temp",
        markers=True,
        color="month_name",
        labels={
            "air_temp_bin_str": "Air Temperature Bin",
            "avg_ss_temp": "Average SST (°C)",
            "month_name": "Month",
        },
        title="Average SST per Air Temperature Bin Colored by Month",
    )
    fig_line2.update_layout(title_x=0.5, xaxis_tickangle=-45, plot_bgcolor="white")
    st.plotly_chart(fig_line2, use_container_width=True)
