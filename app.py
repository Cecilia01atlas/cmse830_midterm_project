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
    enso = pd.read_csv("data/data_index.csv").drop_duplicates(subset=["year", "month"])
    df = df.merge(enso, on=["year", "month"], how="left")

    return df, el_nino


if "df" not in st.session_state:
    st.session_state["df"], st.session_state["el_nino"] = load_data()

# Always work on the session version
df = st.session_state["df"]
el_nino = st.session_state["el_nino"]

st.set_page_config(
    page_title="ENSO Explorer 🌊",
    layout="wide",
)

# --- Sidebar ---
menu = [
    "Overview",
    "Missingness",
    "Temporal Coverage",
    "Correlation study",
    "Summary and Conclusion",
]

with st.sidebar:
    st.title("🌐 ENSO Explorer")
    st.markdown(
        """
Explore the **impact of El Niño & La Niña** on ocean–atmosphere variables  
through interactive visualizations and imputation tools.
"""
    )
    st.markdown("---")
    choice = st.radio("Navigate to:", menu)

# =========================
# Tab 1: Overview
# =========================
if choice == "Overview":
    st.title("🌊 Dataset Overview")
    st.markdown("""
    Welcome to the **ENSO Explorer App**, an interactive platform ..
    """)

    ## Key metrics panel
    total_rows = len(df)
    total_cols = len(df.columns)
    total_missing = df.isna().sum().sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", total_rows)
    col2.metric("Total Features", total_cols)
    col3.metric("Missing Values", total_missing)

    st.markdown("""
    Two datasets were merged for this app. These datasets contain measurements of **ocean-atmosphere variables** over time, including:

    - Sea Surface Temperature (Sea Surface Temperature)
    - Air Temperature
    - Humidity
    - Zonal and Meridional Winds
    - ANOM (ENSO index)

    Most of the data was collected using the TAO (Tropical Atmosphere Ocean) array through mooring measurements.
    Before diving into the analysis of sea-air interactions and El Niño/La Niña patterns, it's important to get a clear picture of the dataset itself — its structure, size, and basic characteristics.
    """)

    # --- Column info ---
    with st.expander("📋 Column Information"):
        # Define descriptions for your key columns
        descriptions = {
            "year": "Year of observation",
            "month": "Month of observation",
            "day": "Day of observation",
            "date": "Datetime object combining year, month, day",
            "latitude": "Latitude of the measurement location",
            "longitude": "Longitude of the measurement location",
            "ss_temp": "Sea Surface Temperature (°C)",
            "air_temp": "Air Temperature (°C)",
            "humidity": "Relative Humidity (%)",
            "zon_winds": "Zonal Wind Component (east-west)",
            "mer_winds": "Meridional Wind Component (north-south)",
            "ClimAdjust": "Climatological adjustment applied to Sea Surface Temperature",
            "ANOM": "ENSO Index (El Niño/La Niña anomaly)",
            # Add other columns if necessary
        }

        # Build dataframe
        col_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Missing Values": df.isna().sum().values,
                "Description": [
                    descriptions.get(col, "") for col in df.columns
                ],  # empty string if no description
            }
        )

        st.dataframe(col_info.astype(str))

    # --- Summary statistics ---
    with st.expander("📈 Summary Statistics"):
        numeric_df = df.drop(columns=["year", "month", "day", "date"], errors="ignore")
        st.write(numeric_df.describe())

    # --- Temporal coverage ---
    with st.expander("🕒 Temporal Coverage Plot"):
        st.markdown("""
        The following graph shows the **number of data sets collected over the years**.
        It highlights periods in which we have denser observations. In particular, it shows that there is an imbalance, as more data was collected later on than in earlier years.
        """)
        df["year_month"] = (
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
        )
        ym_counts = df["year_month"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(15, 4))
        ym_counts.plot(ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Records")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # --- Duplicates ---
    with st.expander("🔁 Duplicate Records"):
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
    with st.expander("🚨 Outlier Detection (Z-score > 3)"):
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
        st.write(
            pd.DataFrame.from_dict(outlier_dict, orient="index", columns=["Outliers"])
        )
        st.markdown("""
        In this case, there are only a few outliers, no more than 1% per column.
        It is also makes sense that air temperature and sea surface temperature have the highest number of outliers due to the conditions during ENSO events.
        """)


# ================================================
# Tab 2: Missingness
# ================================================
elif choice == "Missingness":
    st.title("🚧 Missingness Analysis")
    st.markdown("""
    Some variables in the dataset have **missing measurements**, which is common in long-term environmental observations. 
    Understanding **which variables and time periods are missing** is critical before performing imputation or trend analyses.
    """)

    # --- Missingness table ---
    with st.expander("📋 Missingness Summary Table"):
        summary_table = pd.DataFrame(
            {
                "Missing Values": df.isna().sum(),
                "Missing %": (df.isna().mean() * 100).round(2),
            }
        ).sort_values("Missing Values", ascending=False)
        st.dataframe(summary_table.astype(str))

    # --- Missingess Heatmap ---
    with st.expander("🌡 Missingness Heatmap"):
        st.markdown("""
        Missing data can reveal patterns about how and when measurements were taken, and they influence how we handle imputation.
        """)
        excluded_cols = ["day", "month", "year"]
        cols_for_heatmap = [col for col in df.columns if col not in excluded_cols]
        nan_mask = df[cols_for_heatmap].copy()

        if "humidity_original" in df.columns:
            nan_mask["humidity"] = df["humidity_original"]

        nan_array = nan_mask.isna().astype(int).to_numpy()

        fig, ax = plt.subplots(figsize=(30, 18))
        im = ax.imshow(
            nan_array.T, interpolation="nearest", aspect="auto", cmap="cividis"
        )
        ax.set_title(
            "Missing Values Heatmap (1 = Missing, 0 = Present)", fontsize=28, pad=20
        )
        ax.set_ylabel("Features", fontsize=24)
        ax.set_xlabel("Date", fontsize=24)
        ax.set_yticks(range(len(cols_for_heatmap)))
        ax.set_yticklabels(cols_for_heatmap, fontsize=18)

        n_rows = len(df)
        n_ticks = min(15, n_rows)
        tick_positions = np.linspace(0, n_rows - 1, n_ticks).astype(int)
        tick_labels = df.loc[tick_positions, "date"].dt.strftime("%Y-%m-%d")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=16)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("Missingness", fontsize=20)
        plt.tight_layout()
        st.pyplot(fig)

    # --- Humidity Imputation ---
    st.subheader("Humidity Imputation")
    st.markdown("""
    Missing humidity values can be imputed using a **linear regression model** based on air temperature and sea surface temperature. 
    To avoid unreliable imputations, **years with less than 40 % missing humidity data are excluded** from the imputation process. 
    Additionally, stochastic noise is added to mimic natural variability.
    """)

    if st.button("Run Humidity Imputation"):
        if "humidity_original" not in df.columns:
            df["humidity_original"] = df["humidity"].copy()

        feature_cols = ["air_temp", "ss_temp"]
        target_col = "humidity"

        # --- Calculate missing fraction per year ---
        missing_per_year = df.groupby("year")[target_col].apply(
            lambda x: x.isna().mean()
        )

        # --- Choose years where missing rate is above threshold ---
        threshold = 0.4
        years_allowed = missing_per_year[missing_per_year >= threshold].index

        # --- Training data: only measured where predictors are not missing ---
        mask_train = df[feature_cols].notna().all(axis=1) & df[target_col].notna()
        X_train = df.loc[mask_train, feature_cols]
        y_train = df.loc[mask_train, target_col]

        # --- Fit linear regression ---
        model = LinearRegression()
        model.fit(X_train, y_train)

        # --- Residual standard deviation for stochastic noise ---
        residual_std = np.std(y_train - model.predict(X_train))

        mask_impute = (
            df[feature_cols].notna().all(axis=1)
            & df[target_col].isna()
            & df["year"].isin(years_allowed)
        )
        X_missing = df.loc[mask_impute, feature_cols]

        # --- Stochastic imputation ---
        n_simulations = 100
        stochastic_predictions = []
        for _ in range(n_simulations):
            noise = np.random.normal(0, residual_std, size=X_missing.shape[0])
            stochastic_predictions.append(model.predict(X_missing) + noise)
        y_imputed = np.mean(stochastic_predictions, axis=0)

        # --- Assign imputed values ---
        df.loc[mask_impute, target_col] = y_imputed
        st.session_state["df"] = df.copy()

        # --- Plot before vs imputed ---
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
        ax.set_title(
            f"Humidity After Imputation (Years > {int(threshold * 100)}% Missing Excluded)"
        )
        ax.legend()
        st.pyplot(fig)

        st.session_state["df"] = df.copy()

        # --- Diagnostics ---
        st.success("Humidity imputation complete ✅")
        st.write("Remaining missing values per column after imputation:")
        st.write(df.isna().sum())


# ================================================
# Tab 3: Temporal Coverage
# ================================================
elif choice == "Temporal Coverage":
    st.header("📆 Temporal Coverage & ENSO Influence")
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
- 💨 **Winds**: zonal (east-west) and meridional (north-south) winds drive currents.  
- 💧 **Humidity**: links evaporation from oceans with atmospheric moisture content.  

This tab combines **correlation statistics and scatterplots** to explore these links.
""")

    df = st.session_state["df"].copy()
    selected_features = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]
    subset_df = df[selected_features].apply(pd.to_numeric, errors="coerce").dropna()

    # ================= Correlation heatmap =================
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

    # ================= Pairwise scatterplots =================
    st.subheader("🔸Pairwise Relationships")
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

    # ================= Scatterplot with regression line for air temp vs sea surface temp  =================
    st.subheader("🔸 Air Temperature vs Sea Surface Temperature")
    st.markdown("""
As can be seen from the graphs above, there is a strong correlation between air temperature and sea surface temperature. Since the ocean warms the air directly above it, we expect a **close linear relationship** between Sea Surface Temperature and air temperature.
This scatter plot confirms this relationship with a red regression line.
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

    # ================= Binned line plots =================
    st.subheader("🔸 Binned Sea Surface Temperature by Air Temperature")
    st.markdown("""
    Air temperature values are **grouped into bins** to highlight seasonal trends more clearly.  
    The line plot below shows **Sea Surface Temperature values distributed across air temperature bins**, with each bin as a discrete x-axis category.
    """)

    num_bins = 15
    df["air_temp_bin"] = pd.qcut(df["air_temp"], q=num_bins, duplicates="drop")
    df["air_temp_bin_str"] = df["air_temp_bin"].astype(str)

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
    df["month_name"] = pd.Categorical(
        df["month"].map(dict(zip(range(1, 13), month_labels))),
        categories=month_labels,
        ordered=True,
    )

    binned = (
        df.groupby(["air_temp_bin_str", "month_name"], observed=True)["ss_temp"]
        .mean()
        .reset_index()
    )

    fig_binned = px.line(
        binned,
        x="air_temp_bin_str",
        y="ss_temp",
        color="month_name",
        category_orders={"month_name": month_labels},
        labels={
            "air_temp_bin_str": "Air Temperature Bin",
            "ss_temp": "Sea Surface Temperature (°C)",
            "month_name": "Month",
        },
        title="Sea Surface Temperature vs Air Temperature Bin by Month",
    )

    fig_binned.update_traces(mode="lines+markers", opacity=0.85)
    fig_binned.update_layout(
        xaxis=dict(tickangle=-45),
    )

    st.plotly_chart(fig_binned, use_container_width=True)

# ----------------------------
# Tab 5: Summary & Conclusion
# ----------------------------
elif choice == "Summary and Conclusion":
    st.title("📖 Summary and Key Insights")
    st.markdown("""
This last tab contains a summary of the **key findings** from the analysis of the El Niño/La Niña dataset and a conclusion:
- **ENSO influence:** El Niño and La Niña events have a strong influence on sea surface temperatures (Sea Surface Temperature) and air temperatures, while humidity shows only moderate fluctuations. ENSO periods are clearly visible in temporal visualizations and illustrate the coupling between the ocean and the atmosphere.
- **Variable relationships:** Air temperature and Sea Surface Temperature show a strong positive correlation, as can be seen from heat maps, scatter plots, and binned line plots. Wind components are also related to temperature and humidity, suggesting broader climatic interactions.
- **Seasonal and temporal patterns:** Seasonal cycles can be observed for all variables. Violin plots and temporal visualizations illustrate fluctuations over months and years.

""")

    # --- Narrative conclusion ---
    st.subheader("🔹 Overall Conclusion")
    st.markdown("""
ENSO events (El Niño / La Niña) are important drivers of variability in the ocean-atmosphere system.
Seasonal cycles and long-term trends are evident, and correlations show the connection between different key climate variables.
To further improve this analysis, future work could incorporate **sea surface temperatures at various depths** rather than just at the surface. Temperature profiles below the surface would provide a more complete picture of ocean dynamics and their role in driving atmospheric responses during ENSO events.
                
**Final note**: AI was used, specifically ChatGPT-5, to improve all the visualizations and implement everything into the app with Streamlit.                
""")
