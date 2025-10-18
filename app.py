import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
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

    df["year"] = df["year"] + 1900
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Merge ENSO index
    enso = pd.read_csv("data_index.csv").drop_duplicates(subset=["year", "month"])
    df = df.merge(enso, on=["year", "month"], how="left")

    return df, el_nino


# Load dataset
df, el_nino = load_data()

# Use session state for persistence
if "df" in st.session_state:
    df = st.session_state["df"].copy()
else:
    st.session_state["df"] = df.copy()

# --- Sidebar Menu ---
menu = [
    "Overview",
    "Missingness",
    "Temporal Coverage",
    "Correlation study",
    "Summary and Conclusion",
]
choice = st.sidebar.radio("Menu", menu)

# ================================================
# Tab 1: Overview
# ================================================
if choice == "Overview":
    st.title("🌊 Dataset Overview")
    st.markdown("""
This dataset contains measurements of **ocean-atmosphere variables** over time, including:
- Sea Surface Temperature (SST)
- Air Temperature
- Humidity
- Zonal and Meridional Winds

Before diving into the analysis of sea–air interactions and El Niño/La Niña patterns, 
it's important to get a clear picture of the dataset itself — its structure, size, 
and basic characteristics. This tab provides a quick overview and initial checks.

Below, you can explore:
1. Column information and data types.
2. Summary statistics.
3. Temporal coverage of the dataset.
4. Outliers and duplicates.
""")

    # --- Column info ---
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

    # --- Summary statistics ---
    st.subheader("Summary Statistics")
    numeric_df = df.drop(columns=["year", "month", "day", "date"], errors="ignore")
    st.write(numeric_df.describe())

    # --- Temporal coverage plot ---
    st.subheader("Temporal Coverage Over Time (Year-Month)")
    st.markdown("""
The plot below shows the **number of records collected per year-month**.  
It highlights periods where we have denser observations, which is important for assessing trends and variability over time.
""")
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

    # --- Duplicates ---
    st.subheader("Duplicates in Dataset")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    # --- Outlier detection ---
    st.subheader("Outlier Detection (Z-score > 3)")
    st.markdown("""
Extreme values (Z-score > 3) may indicate **measurement errors or unusual events**.  
Understanding these outliers helps ensure robust analyses.
""")
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


# ================================================
# Tab 2: Missingness
# ================================================
elif choice == "Missingness":
    st.title("💧 Missingness Analysis")
    st.markdown("""
Some variables in the dataset have **missing measurements**, which is common in long-term environmental observations.  
Understanding **which variables and time periods are missing** is critical before performing imputation or trend analyses.
""")

    # --- Missingness table ---
    st.subheader("Missingness Summary Table")
    st.markdown("The table below shows a summary of missing entries for each feature.")

    summary_table = pd.DataFrame(
        {
            "Missing Values": df.isna().sum(),
            "Missing %": (df.isna().mean() * 100).round(2),
        }
    ).sort_values("Missing Values", ascending=False)
    st.dataframe(summary_table)

    # --- Missingness heatmap ---
    st.markdown("""
    ### Missingness Analysis

    Missing data can reveal patterns about how and when measurements were taken, and they influence
    how we handle imputation. Here we explore the structure of missing values in the dataset.
    """)

    ## --- Exclude day, month, and year for missingness heatmap ---
    excluded_cols = ["day", "month", "year"]
    cols_for_heatmap = [col for col in df.columns if col not in excluded_cols]

    nan_mask = df[cols_for_heatmap].copy()

    # Restore original humidity if it exists
    if "humidity_original" in df.columns:
        nan_mask["humidity"] = df["humidity_original"]

    # Create missingness array
    nan_array = nan_mask.isna().astype(int).to_numpy()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(30, 18))

    # Use a high-contrast colormap (0 = white, 1 = dark)
    im = ax.imshow(
        nan_array.T,
        interpolation="nearest",
        aspect="auto",
        cmap="cividis",
    )

    ax.set_ylabel("Features")
    ax.set_title("Missing Values Heatmap (1 = Missing, 0 = Present)")

    # Y-axis: feature names
    ax.set_yticks(range(len(cols_for_heatmap)))
    ax.set_yticklabels(cols_for_heatmap)

    # X-axis: date ticks (up to 15 evenly spaced)
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

    # --- Humidity Imputation ---
    st.subheader("Humidity Imputation")
    st.markdown("""
Missing humidity values are imputed using a **linear regression model** based on air temperature and sea surface temperature.  
Stochastic noise is added to mimic natural variability.
""")
    if st.button("Run Humidity Imputation"):
        # Preserve original humidity
        if "humidity_original" not in df.columns:
            df["humidity_original"] = df["humidity"].copy()

        feature_cols = ["air_temp", "ss_temp"]
        target_col = "humidity"
        mask_train = df[feature_cols].notna().all(axis=1) & df[target_col].notna()
        X_train = df.loc[mask_train, feature_cols]
        y_train = df.loc[mask_train, target_col]

        model = LinearRegression()
        model.fit(X_train, y_train)
        residual_std = np.std(y_train - model.predict(X_train))

        mask_impute = df[feature_cols].notna().all(axis=1) & df[target_col].isna()
        X_missing = df.loc[mask_impute, feature_cols]

        # Stochastic imputation
        n_simulations = 100
        stochastic_predictions = []
        for _ in range(n_simulations):
            noise = np.random.normal(0, residual_std, size=X_missing.shape[0])
            stochastic_predictions.append(model.predict(X_missing) + noise)
        y_imputed = np.mean(stochastic_predictions, axis=0)

        df.loc[mask_impute, target_col] = y_imputed
        st.session_state["df"] = df.copy()

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
        st.write("Remaining missing values per column after imputation:")
        st.write(df.isna().sum())


# ================================================
# Tab 3: Temporal Coverage
# ================================================
elif choice == "Temporal Coverage":
    st.header("🌊 Temporal Coverage & ENSO Influence")
    st.markdown("""
The **Temporal Coverage** tab provides a deeper look at how key climate variables evolve over time.  
This is where we highlight the influence of **ENSO events (El Niño & La Niña)** on variables such as
sea surface temperature, air temperature, humidity, and winds.

- **El Niño** (red shading): Typically leads to **warmer sea surface temperatures** in the central and eastern Pacific.  
- **La Niña** (blue shading): Usually associated with **cooler sea surface temperatures**.  

Use the dropdown below to select which variable you'd like to explore over time.
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

    # ================= Scatter plot colored by ENSO =================
    st.subheader(
        f"📅 {feature.replace('_', ' ').title()} Over Time (ENSO-Colored Scatter)"
    )
    st.markdown("""
    Each point represents a daily observation. The color indicates the **ENSO index (ANOM)**:  
    - 🔴 **Positive values** → El Niño conditions (warmer anomalies)  
    - 🔵 **Negative values** → La Niña conditions (cooler anomalies)  
    - ⚪ **Near zero** → Neutral conditions
    """)

    anom_abs = max(abs(df["ANOM"].min()), abs(df["ANOM"].max()))
    fig_scatter = px.scatter(
        df,
        x="date",
        y=feature,
        color="ANOM",
        color_continuous_scale="RdBu_r",
        opacity=0.6,
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
        ),
        template="plotly_white",
        title_x=0.5,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ================= ENSO-shaded line plot =================
    st.subheader(
        f"📈 Daily {feature.replace('_', ' ').title()} with ENSO Event Shading"
    )
    st.markdown("""
    The line plot below shows the smoothed daily values.  
    Shaded regions highlight **ENSO events** over time, illustrating how they align with 
    changes in the selected variable.
    """)

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
                "rgba(255, 0, 0, 0.15)"
                if period["event"] == "El Niño"
                else "rgba(0, 0, 255, 0.15)"
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
        title_x=0.5,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ================= Monthly violin plot =================
    st.subheader(f"📊 Monthly Distribution of {feature.replace('_', ' ').title()}")
    st.markdown("""
    The violin plot below shows the **seasonal distribution** of the selected variable, aggregated over all years.
    It highlights the **annual cycle** and seasonal patterns that interact with ENSO dynamics.
    """)

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

    fig, ax = plt.subplots(figsize=(16, 7))
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
        fontsize=16,
        pad=15,
    )
    ax.set_xlabel("Month", fontsize=13)
    ax.set_ylabel(feature.replace("_", " ").title(), fontsize=13)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=11)
    sns.despine()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)


# ================================================
# Tab 4: Correlation & Scatter
# ================================================
elif choice == "Correlation study":
    st.header("📊 Correlation and Feature Relationships")
    st.markdown("""
This tab explores relationships between key variables:
- Correlation heatmap shows **how variables co-vary**.
- Scatter plot examines **air temperature vs sea surface temperature**, highlighting the linear relationship.
- Pairwise scatter matrix shows interactions between all selected variables.
- Binned line plots summarize **average SST per air temperature bin** across months.
""")
    selected_features = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]
    subset_df = df[selected_features].apply(pd.to_numeric, errors="coerce").dropna()

    # --- Correlation heatmap ---
    corr_matrix = subset_df.corr().values
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    masked_corr = np.where(mask, None, corr_matrix)[::-1]

    fig = go.Figure(
        data=go.Heatmap(
            z=masked_corr,
            x=selected_features,
            y=selected_features[::-1],
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            hoverongaps=False,
            showscale=True,
        )
    )
    for i, row in enumerate(masked_corr):
        for j, val in enumerate(row):
            if val is not None:
                fig.add_annotation(
                    x=selected_features[j],
                    y=selected_features[::-1][i],
                    text=f"{val:.2f}",
                    showarrow=False,
                )
    fig.update_layout(title="Correlation Heatmap", xaxis=dict(tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)

    # --- Scatter with regression ---
    fig_scatter = px.scatter(
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
    )
    fig_scatter.update_traces(
        marker=dict(size=5, color="#4a7c9b", line=dict(width=0.5, color="#2f4f5f"))
    )
    fig_scatter.update_layout(
        title=dict(text="Air Temperature vs Sea Surface Temperature", x=0.5),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ----------------------------
# Tab 5: ENSO Summary
# ----------------------------
elif choice == "Summary and Conclusion":
    st.title("🌊 ENSO Summary and Insights")
    st.markdown("""
This tab summarizes the **impact of ENSO (El Niño / La Niña) events** on the ocean-atmosphere system:

- **El Niño:** Warmer sea surface temperatures (SST), increased air temperature, sometimes lower humidity.
- **La Niña:** Cooler SST, slightly cooler air, potential increases in humidity.
- **ENSO Index (ANOM):** Used to identify the strength and timing of events.
""")

    df_daily = (
        df.groupby("date")[["ss_temp", "air_temp", "humidity", "ANOM"]]
        .mean()
        .reset_index()
    )
    el_thresh, la_thresh = 1.0, -1.0
    df_daily["event"] = np.where(
        df_daily["ANOM"] > el_thresh,
        "El Niño",
        np.where(df_daily["ANOM"] < la_thresh, "La Niña", None),
    )

    # Aggregate by ENSO event
    event_summary = []
    current_event = None
    start_date = None
    for _, row in df_daily.iterrows():
        event, date = row["event"], row["date"]
        if event != current_event:
            if current_event is not None:
                segment = df_daily[
                    (df_daily["date"] >= start_date) & (df_daily["date"] < date)
                ]
                event_summary.append(
                    {
                        "event": current_event,
                        "start": start_date,
                        "end": date,
                        "avg_sst": segment["ss_temp"].mean(),
                        "avg_air_temp": segment["air_temp"].mean(),
                        "avg_humidity": segment["humidity"].mean(),
                    }
                )
            current_event = event
            start_date = date
    if current_event is not None:
        segment = df_daily[df_daily["date"] >= start_date]
        event_summary.append(
            {
                "event": current_event,
                "start": start_date,
                "end": df_daily["date"].iloc[-1],
                "avg_sst": segment["ss_temp"].mean(),
                "avg_air_temp": segment["air_temp"].mean(),
                "avg_humidity": segment["humidity"].mean(),
            }
        )

    summary_df = pd.DataFrame(event_summary)
    st.subheader("Average Conditions per ENSO Event")
    st.dataframe(
        summary_df[["event", "start", "end", "avg_sst", "avg_air_temp", "avg_humidity"]]
    )

    # Interactive time series with ENSO shading
    fig = go.Figure()
    for period in event_summary:
        if period["event"] is None:
            continue
        color = (
            "rgba(255,0,0,0.1)" if period["event"] == "El Niño" else "rgba(0,0,255,0.1)"
        )
        fig.add_vrect(
            x0=period["start"],
            x1=period["end"],
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line_width=0,
        )

    fig.add_trace(
        go.Scatter(
            x=df_daily["date"],
            y=df_daily["ss_temp"],
            mode="lines",
            name="SST (°C)",
            line=dict(color="royalblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_daily["date"],
            y=df_daily["air_temp"],
            mode="lines",
            name="Air Temp (°C)",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_daily["date"],
            y=df_daily["humidity"],
            mode="lines",
            name="Humidity (%)",
            line=dict(color="green"),
        )
    )

    fig.update_layout(
        title="SST, Air Temp, and Humidity with ENSO Shading",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Narrative conclusion
    st.markdown("""
**Insights:**
- Peaks of **El Niño events** correspond to the warmest SST and elevated air temperature.
- **La Niña events** tend to show cooler SST and slightly cooler air.
- Humidity shows moderate fluctuations, often influenced by ENSO phase.
- This demonstrates the **strong coupling between ocean temperatures and atmospheric conditions**, which affects climate patterns globally.
""")
