import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GridSense",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚡ GridSense")

st.markdown(
    """
    ### Electricity Load Forecasting & Anomaly Detection

    GridSense uses machine learning to analyze electricity
    demand, forecast future demand and identify unusual
    consumption behavior.
    """
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "models"
DATA_PATH = "outputs/gridsense_processed.csv"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    os.path.join(MODEL_DIR, "forecast_model.pkl"),
    os.path.join(MODEL_DIR, "anomaly_model.pkl"),
    os.path.join(MODEL_DIR, "feature_columns.pkl"),
    os.path.join(MODEL_DIR, "anomaly_features.pkl"),
    DATA_PATH
]


missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.error(
        "Some required GridSense files are missing."
    )

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- {file}")

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

forecast_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "forecast_model.pkl"
    )
)


anomaly_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "anomaly_model.pkl"
    )
)


MODEL_FEATURES = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_columns.pkl"
    )
)


ANOMALY_FEATURES = joblib.load(
    os.path.join(
        MODEL_DIR,
        "anomaly_features.pkl"
    )
)


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    "GridSense models and processed dataset loaded successfully."
)


# ============================================================
# BASIC INFORMATION
# ============================================================

st.subheader("📊 Dataset Information")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Observations",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Number of Features",
        f"{len(MODEL_FEATURES):,}"
    )


with col3:

    st.metric(
        "Target Variable",
        "nat_demand"
    )


# ============================================================
# DEMAND STATISTICS
# ============================================================

total_observations = len(df)

average_demand = df["nat_demand"].mean()

maximum_demand = df["nat_demand"].max()

minimum_demand = df["nat_demand"].min()


# ============================================================
# ANOMALY STATISTICS
# ============================================================

if "anomaly" in df.columns:

    anomaly_count = (
        df["anomaly"] == "Anomaly"
    ).sum()

else:

    anomaly_count = 0


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("⚡ Demand Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Observations",
        f"{total_observations:,}"
    )


with col2:

    st.metric(
        "Average Demand",
        f"{average_demand:,.2f}"
    )


with col3:

    st.metric(
        "Peak Demand",
        f"{maximum_demand:,.2f}"
    )


with col4:

    st.metric(
        "Detected Anomalies",
        f"{anomaly_count:,}"
    )


# ============================================================
# ELECTRICITY DEMAND CHART
# ============================================================

st.subheader("📈 Electricity Demand")


fig = px.line(
    df,
    y="nat_demand",
    title="National Electricity Demand"
)


fig.update_layout(
    xaxis_title="Observation",
    yaxis_title="Electricity Demand"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.subheader("🚨 Anomaly Detection")


if "anomaly" in df.columns:

    plot_df = df.copy()

    plot_df["Anomaly Value"] = np.where(
        plot_df["anomaly"] == "Anomaly",
        plot_df["nat_demand"],
        np.nan
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            y=plot_df["nat_demand"],
            mode="lines",
            name="Electricity Demand"
        )
    )


    fig.add_trace(
        go.Scatter(
            y=plot_df["Anomaly Value"],
            mode="markers",
            name="Anomaly"
        )
    )


    fig.update_layout(
        title="Detected Electricity Demand Anomalies",
        xaxis_title="Observation",
        yaxis_title="Electricity Demand"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.warning(
        "Anomaly results are not available in the processed dataset."
    )


# ============================================================
# ANOMALY TABLE
# ============================================================

if "anomaly" in df.columns:

    st.subheader("🔎 Detected Anomalies")


    anomaly_table = df[
        df["anomaly"] == "Anomaly"
    ].copy()


    if len(anomaly_table) > 0:

        columns_to_show = [
            column
            for column in [
                "nat_demand",
                "anomaly_score",
                "lag_1",
                "lag_24",
                "demand_change_1",
                "demand_change_24"
            ]
            if column in anomaly_table.columns
        ]


        st.dataframe(
            anomaly_table[
                columns_to_show
            ],
            use_container_width=True
        )

    else:

        st.info(
            "No anomalies detected."
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🔍 Forecasting Feature Importance"
)


importance_df = pd.DataFrame({

    "Feature":
        MODEL_FEATURES,

    "Importance":
        forecast_model.feature_importances_

})


importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=True
    )
)


fig = px.bar(
    importance_df.tail(15),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Forecasting Features"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# PROCESSED DATA TABLE
# ============================================================

st.subheader(
    "📋 Processed Dataset"
)


st.dataframe(
    df.tail(100),
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "GridSense | Electricity Load Forecasting & Anomaly Detection"
)

