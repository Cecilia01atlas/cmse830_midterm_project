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
The goal of this app is to provide interactive illustration to highlight the influence of a influencial climatique events called ENSO (El Niño & La Niña). Two datasets were merged for this purpose. These datasets contain measurements of **ocean-atmosphere variables** over time, including:
- Sea Surface Temperature (SST)
- Air Temperature
- Humidity
- Zonal and Meridional Winds
- ANOM (ENSO index)

Before diving into the analysis of sea-air interactions and El Niño/La Niña patterns, 
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
            "Type": df.dtypes.values,
            "Missing Values": df.isna().sum().values,
        }
    )
    st.dataframe(col_info.astype(str))

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
    st.subheader("🔁 Duplicate Records")

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        st.warning(f"⚠️ There are {duplicate_count} duplicate rows in the dataset.")
        st.dataframe(df[df.duplicated()].head(10))
        st.markdown(
            "Duplicate records may arise from repeated measurements or data logging errors. "
            "These should be examined to avoid skewing the analysis."
        )
    else:
        st.success("✅ No duplicate records found in the dataset.")
        st.markdown(
            "This indicates the dataset is already clean in terms of repeated entries — a good sign!"
        )

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

    st.markdown("""
In this case there are just a few outlier, not more than 1 %. It also makes sense that the air temperature and sea surface temperature have the heighest number of outliers due to extreme events during ENSO events.
""")

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
    st.dataframe(summary_table.astype(str))

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
Missing humidity values can be imputed using a **linear regression model** based on air temperature and sea surface temperature.  
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
        st.write(df.isna().sum().astype(str))

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
Understanding how **atmospheric and oceanic variables interact** is crucial to explaining ENSO.

- 🌡 **Air ↔ Sea Surface Temperature**: warming of the Pacific surface during El Niño affects atmospheric patterns.  
- 💨 **Winds**: zonal (east-west) and meridional (north-south) winds drive currents and influence upwelling.  
- 💧 **Humidity**: links evaporation from oceans with atmospheric moisture content.  

This tab combines **correlation statistics and scatterplots** to explore these links.
""")

    df = st.session_state["df"].copy()
    selected_features = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]
    subset_df = df[selected_features].apply(pd.to_numeric, errors="coerce").dropna()

    # --- Correlation heatmap ---
    st.subheader("🔸 Correlation Heatmap")
    st.markdown(
        "The heatmap quantifies **linear correlations** (Pearson) between all variables."
    )

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
            showscale=True,
        )
    )
    # Annotate values
    for i, row in enumerate(masked_corr):
        for j, val in enumerate(row):
            if val is not None:
                fig_corr.add_annotation(
                    x=selected_features[j],
                    y=selected_features[::-1][i],
                    text=f"{val:.2f}",
                    font=dict(color="black"),
                    showarrow=False,
                )

    fig_corr.update_layout(
        title=dict(text="Correlation Heatmap of Key Variables", x=0.5),
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- Pairwise scatter matrix (moved up + colored) ---
    st.subheader("🔸 Pairwise Relationships")
    st.markdown("""
To get a better understanding of the realationship between different feature, the scatter matrix lets us **inspect all variable pairs simultaneously**.  
Here, points are colored by **air temperature**, which helps reveal ENSO-related structure and seasonal gradients.
""")

    fig_matrix = px.scatter_matrix(
        df[selected_features],
        dimensions=selected_features,
        color="air_temp",
        labels={col: col.replace("_", " ").title() for col in selected_features},
        title="Pairwise Relationships between Key Variables",
        opacity=0.5,
        color_continuous_scale="RdBu_r",
    )
    fig_matrix.update_traces(diagonal_visible=True)
    fig_matrix.update_layout(height=800, width=800)
    st.plotly_chart(fig_matrix, use_container_width=True)

    # --- Scatter with regression (moved down) ---
    st.subheader("🔸 Air Temperature vs Sea Surface Temperature")
    st.markdown("""
As can be noticed with the plots above, there is a strong correlation between air temperature and sea surface temperature. Because the ocean warms the air directly above it, we expect a **tight linear relationship** between SST and air temperature.  
This scatter plot confirms that relationship, with a regression line shown in red.
""")

    fig_scatter = px.scatter(
        df,
        x="air_temp",
        y="ss_temp",
        opacity=0.5,
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

    # --- Binned line plots ---
    st.subheader("🔸 Binned SST by Air Temperature")
    st.markdown("""
    Air temperature values are **grouped into bins** to highlight seasonal trends more clearly.  
    We then plot the **mean SST per air temperature bin across months**, which reduces noise while preserving structure.
    """)

    # Use quantile-based binning to avoid empty bins
    num_bins = 15
    df["air_temp_bin"] = pd.qcut(df["air_temp"], q=num_bins, duplicates="drop")

    # Extract bin midpoints for plotting
    df["air_temp_bin_center"] = df["air_temp_bin"].apply(
        lambda x: x.mid if pd.notnull(x) else np.nan
    )

    # Group by bin center and month
    binned = (
        df.groupby(["air_temp_bin_center", "month"], observed=True)["ss_temp"]
        .mean()
        .reset_index()
        .sort_values(by=["month", "air_temp_bin_center"])
    )

    # Plot with clear binning on x-axis
    fig_binned = px.line(
        binned,
        x="air_temp_bin_center",
        y="ss_temp",
        color="month",
        labels={
            "air_temp_bin_center": "Binned Air Temperature (°C)",
            "ss_temp": "Mean Sea Surface Temperature (°C)",
            "month": "Month",
        },
        title="Binned SST vs Air Temperature by Month",
    )

    fig_binned.update_traces(mode="lines+markers", opacity=0.85)
    fig_binned.update_layout(xaxis=dict(dtick=1))  # optional: force visible ticks

    st.plotly_chart(fig_binned, use_container_width=True)


# ----------------------------
# Tab 5: Summary & Conclusion
# ----------------------------
elif choice == "Summary and Conclusion":
    st.title("📖 Summary and Key Insights")
    st.markdown("""
This final tab presents the **key insights** from the analysis of the El Niño / La Niña dataset:

- **ENSO Influence:** El Niño and La Niña events strongly affect sea surface temperatures (SST) and air temperatures, with humidity showing moderate variations. ENSO periods are clearly visible in temporal visualizations, underscoring the tight coupling between the ocean and atmosphere.

- **Variable Relationships:** Air temperature and SST exhibit a strong positive correlation, as seen in heatmaps, scatterplots, and binned line plots. Wind components are also related to temperature and humidity, indicating broader climate interactions.

- **Seasonal and Temporal Patterns:** Pronounced seasonal cycles are observed across all variables. Violin plots and temporal visualizations highlight fluctuations over months and years, revealing both cyclical and event-driven variability.

- **Data Quality:** Missing values, particularly in humidity, were imputed using regression-based stochastic methods. This approach preserves variability while enabling more robust analysis without introducing systematic bias.
""")

    # --- Narrative conclusion ---
    st.subheader("🔹 Overall Conclusion")
    st.markdown("""
ENSO events (El Niño / La Niña) are major drivers of variability in the ocean–atmosphere system.  
Seasonal cycles and long-term trends are evident, and correlations highlight the interconnected nature of key climate variables.  
Careful handling of data quality, including the imputation of missing humidity values, ensures the robustness of these findings.

To further improve this analysis, future work could incorporate **sea surface temperatures at different depths**, not just at the surface. Subsurface temperature profiles would provide a more complete picture of ocean dynamics and their role in driving atmospheric responses during ENSO events.

Overall, this study highlights the importance of combining **statistical methods, temporal analysis, and visualization** to uncover meaningful patterns in climate datasets.
""")
