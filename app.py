import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from ucimlrepo import fetch_ucirepo
import statsmodels.api as sm
import seaborn as sns


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

if "df" in st.session_state:
    df = st.session_state["df"].copy()

# --- Sidebar Menu ---
menu = [
    "Overview",
    "Missingness",
    "Temporal Coverage",
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

    fig, ax = plt.subplots(figsize=(24, 12))
    im = ax.imshow(nan_array.T, interpolation="nearest", aspect="auto", cmap="viridis")

    ax.set_ylabel("Features")
    ax.set_title("Missing Values Heatmap (1 = Missing, 0 = Present)")
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

    # --- Humidity Imputation ---
    st.subheader("Humidity Imputation")

    st.markdown("""
    Missing humidity values are imputed using a **linear regression** model with 
    air temperature and sea surface temperature as predictors. 
    Stochastic noise is added to mimic natural variability.
    """)

    if st.button("Run Humidity Imputation"):
        from sklearn.linear_model import LinearRegression

        # If already imputed, use session_state version
        if "df" in st.session_state:
            df = st.session_state["df"].copy()

        # Preserve original humidity for comparison
        if "humidity_original" not in df.columns:
            df["humidity_original"] = df["humidity"].copy()

        feature_cols = ["air_temp", "ss_temp"]
        target_col = "humidity"

        # --- Identify years to impute ---
        missing_per_year = df.groupby("year")[target_col].apply(
            lambda x: x.isna().mean()
        )
        threshold = 0.5
        years_to_impute = missing_per_year[missing_per_year > threshold].index

        # --- Train model ---
        mask_train = df[feature_cols].notna().all(axis=1) & df[target_col].notna()
        X_train = df.loc[mask_train, feature_cols]
        y_train = df.loc[mask_train, target_col]

        model = LinearRegression()
        model.fit(X_train, y_train)
        residual_std = np.std(y_train - model.predict(X_train))

        # --- Rows to impute ---
        mask_impute = (
            df[feature_cols].notna().all(axis=1)
            & df[target_col].isna()
            & df["year"].isin(years_to_impute)
        )
        X_missing = df.loc[mask_impute, feature_cols]

        # --- Stochastic predictions ---
        n_simulations = 100
        stochastic_predictions = []
        for _ in range(n_simulations):
            noise = np.random.normal(0, residual_std, size=X_missing.shape[0])
            stochastic_predictions.append(model.predict(X_missing) + noise)
        y_imputed = np.mean(stochastic_predictions, axis=0)

        # --- Apply imputation ---
        df.loc[mask_impute, target_col] = y_imputed

        # --- Save back to session_state ---
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
        ax.set_title("Humidity After Imputation")
        ax.legend()
        st.pyplot(fig)

        st.success("Humidity imputation complete ✅")
        st.write("Remaining missing values per column after imputation:")
        st.write(df.isna().sum())


# --- Tab 3: Temporal coverage ---
elif choice == "Temporal Coverage":
    st.header("📅 Temporal Coverage")

    # --- Dropdown to select feature ---
    numeric_cols = ["humidity", "air_temp", "ss_temp", "zon_winds", "mer_winds"]
    feature = st.selectbox(
        "Select variable to visualize:",
        options=numeric_cols,
        index=numeric_cols.index("ss_temp") if "ss_temp" in numeric_cols else 0,
    )

    # --- Prepare dates and weights ---
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    year_counts = df["year"].value_counts()
    df["weight"] = df["year"].map(lambda y: 1 / year_counts[y])

    # --- ENSO Scatter Plot ---
    anom_abs = max(abs(df["ANOM"].min()), abs(df["ANOM"].max()))
    fig_scatter = px.scatter(
        df,
        x="date",
        y=feature,
        color="ANOM",
        color_continuous_scale="RdBu_r",
        title=f"{feature.replace('_', ' ').title()} Over Time (ENSO-Colored)",
        labels={
            "date": "Date",
            feature: feature.replace("_", " ").title(),
            "ANOM": "ENSO Index",
        },
        opacity=0.5,
    )
    fig_scatter.update_layout(
        coloraxis=dict(
            cmin=-anom_abs,
            cmax=anom_abs,
            cmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title="ENSO Index (ANOM)"),
        ),
        template="plotly_white",
        margin=dict(l=50, r=20, t=60, b=60),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Daily averages ---
    df_daily = df.groupby("date")[numeric_cols + ["ANOM"]].mean().reset_index()

    # --- ENSO shading toggle ---
    show_shading = st.checkbox("Show ENSO shading on line plot", value=True)

    # --- Identify ENSO event periods ---
    el_nino_thresh = 1.0
    la_nina_thresh = -1.0
    df_daily["event"] = np.where(
        df_daily["ANOM"] > el_nino_thresh,
        "El Nino",
        np.where(df_daily["ANOM"] < la_nina_thresh, "La Nina", None),
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

    # --- ENSO-shaded Line Plot ---
    fig_line = go.Figure()

    if show_shading:
        for period in shading_periods:
            if period["event"] is None:
                continue
            color = (
                "rgba(255, 0, 0, 0.1)"
                if period["event"] == "El Nino"
                else "rgba(0, 0, 255, 0.1)"
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

    fig_line.update_xaxes(
        dtick="M12",
        tickformat="%Y",
        tickangle=-45,
        showgrid=True,
        gridcolor="lightgrey",
    )

    fig_line.update_layout(
        title=dict(
            text=f"Daily {feature.replace('_', ' ').title()}"
            + (" with ENSO Event Shading" if show_shading else ""),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Date",
        yaxis_title=feature.replace("_", " ").title(),
        template="plotly_white",
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        margin=dict(l=50, r=20, t=60, b=60),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Monthly Violin Plot ---
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
    plt.tight_layout()
    st.pyplot(fig)
