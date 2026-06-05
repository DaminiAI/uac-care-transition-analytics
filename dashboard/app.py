import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import zscore
from prophet import Prophet

# Page Config

st.set_page_config(
    page_title="UAC Care Transition Analytics",
    layout="wide"
)

# Dashboard Title

st.title(
    "UAC Care Transition Efficiency & Placement Outcome Analytics Dashboard"
)

st.markdown("""
This dashboard provides operational analytics for the UAC care pipeline:

- CBP → HHS transfer efficiency
- Sponsor placement effectiveness
- Bottleneck and backlog detection
- Outcome stability monitoring
- Care pipeline performance evaluation
""")

# Load Dataset

df = pd.read_csv("data/raw/uac_data.csv")

# Remove Empty Rows

df = df.dropna(how='all')

# Rename Columns

df.columns = [
    "date",
    "cbp_apprehended",
    "cbp_in_custody",
    "cbp_transferred",
    "hhs_in_care",
    "hhs_discharged"
]

# Convert Date

df['date'] = pd.to_datetime(df['date'])

# Clean HHS Care Column

df['hhs_in_care'] = (
    df['hhs_in_care']
    .astype(str)
    .str.replace(',', '')
)

df['hhs_in_care'] = pd.to_numeric(df['hhs_in_care'])

# Fill Missing Values

df = df.fillna(0)

# KPI Calculations

df['transfer_efficiency'] = np.where(
    df['cbp_in_custody'] == 0,
    0,
    df['cbp_transferred'] /
    df['cbp_in_custody']
)

df['discharge_effectiveness'] = np.where(
    df['hhs_in_care'] == 0,
    0,
    df['hhs_discharged'] /
    df['hhs_in_care']
)

df['pipeline_throughput'] = np.where(
    df['cbp_apprehended'] == 0,
    0,
    df['hhs_discharged'] /
    df['cbp_apprehended']
)

df['backlog_accumulation'] = (
    df['cbp_apprehended'] -
    df['hhs_discharged']
)

df['daily_net_flow'] = (
    df['cbp_transferred'] -
    df['hhs_discharged']
)

# Rolling Metrics

df['rolling_transfer_efficiency'] = (
    df['transfer_efficiency']
    .rolling(window=7)
    .mean()
)

df['rolling_discharge_effectiveness'] = (
    df['discharge_effectiveness']
    .rolling(window=7)
    .mean()
)

df['outcome_stability_score'] = (
    df['discharge_effectiveness']
    .rolling(window=7)
    .std()
)

# Cumulative Backlog

df['cumulative_backlog'] = (
    df['cbp_apprehended'].cumsum()
    -
    df['hhs_discharged'].cumsum()
)

# Additional Features

df['month'] = (
    df['date']
    .dt.to_period('M')
    .astype(str)
)

df['weekday'] = (
    df['date']
    .dt.day_name()
)

# Anomaly Detection

df['transfer_zscore'] = zscore(
    df['transfer_efficiency']
)

anomalies = df[
    abs(df['transfer_zscore']) > 2
]

# Sidebar

st.sidebar.header("Dashboard Filters")

start_date = st.sidebar.date_input(
    "Start Date",
    df['date'].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    df['date'].max()
)

selected_metric = st.sidebar.selectbox(
    "Select KPI Metric",
    [
        "transfer_efficiency",
        "discharge_effectiveness",
        "pipeline_throughput",
        "backlog_accumulation"
    ]
)

# Filter Dataset

filtered_df = df[
    (df['date'] >= pd.to_datetime(start_date)) &
    (df['date'] <= pd.to_datetime(end_date))
]

# Empty Filter Protection

if filtered_df.empty:

    st.warning(
        "No data available for selected date range."
    )

    st.stop()

# KPI Cards

avg_transfer = round(
    filtered_df['transfer_efficiency'].mean(),
    2
)

avg_discharge = round(
    filtered_df['discharge_effectiveness'].mean(),
    2
)

avg_throughput = round(
    filtered_df['pipeline_throughput'].mean(),
    2
)

total_backlog = int(
    filtered_df['backlog_accumulation'].sum()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Avg Transfer Efficiency",
    avg_transfer
)

col2.metric(
    "Avg Discharge Effectiveness",
    avg_discharge
)

col3.metric(
    "Avg Pipeline Throughput",
    avg_throughput
)

col4.metric(
    "Total Backlog",
    total_backlog
)

# Pipeline Stages

st.subheader("UAC Care Pipeline Stages")

st.markdown("""
CBP Apprehension  
↓  
CBP Custody  
↓  
Transfer to HHS Care  
↓  
Medical Screening & Case Management  
↓  
Sponsor Placement / Reunification
""")

# Tabs

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pipeline Monitoring",
    "Operational Analytics",
    "Risk Detection",
    "Forecasting",
    "Executive Summary"
])

# Pipeline Monitoring

with tab1:

    st.subheader("Care Pipeline Flow Over Time")

    fig1 = px.line(
        filtered_df,
        x='date',
        y=[
            'cbp_apprehended',
            'cbp_transferred',
            'hhs_discharged'
        ]
    )

    st.plotly_chart(
        fig1,
        use_container_width=True,
        key="pipeline_flow_chart"
    )

    st.subheader("Selected KPI Trend")

    fig2 = px.line(
        filtered_df,
        x='date',
        y=selected_metric
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
        key="selected_kpi_chart"
    )

    st.subheader("Transfer Efficiency Trend")

    fig3 = px.line(
        filtered_df,
        x='date',
        y='transfer_efficiency'
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
        key="transfer_efficiency_chart"
    )

    st.subheader("Discharge Effectiveness Trend")

    fig4 = px.line(
        filtered_df,
        x='date',
        y='discharge_effectiveness'
    )

    st.plotly_chart(
        fig4,
        use_container_width=True,
        key="discharge_effectiveness_chart"
    )

# Operational Analytics

with tab2:

    st.subheader("7-Day Rolling Transfer Efficiency")

    fig5 = px.line(
        filtered_df,
        x='date',
        y='rolling_transfer_efficiency'
    )

    st.plotly_chart(
        fig5,
        use_container_width=True,
        key="rolling_transfer_chart"
    )

    st.subheader("7-Day Rolling Discharge Effectiveness")

    fig6 = px.line(
        filtered_df,
        x='date',
        y='rolling_discharge_effectiveness'
    )

    st.plotly_chart(
        fig6,
        use_container_width=True,
        key="rolling_discharge_chart"
    )

    st.subheader("Backlog Accumulation")

    fig7 = px.line(
        filtered_df,
        x='date',
        y='backlog_accumulation'
    )

    st.plotly_chart(
        fig7,
        use_container_width=True,
        key="backlog_chart"
    )

    st.subheader("Cumulative Backlog Trend")

    fig8 = px.line(
        filtered_df,
        x='date',
        y='cumulative_backlog'
    )

    st.plotly_chart(
        fig8,
        use_container_width=True,
        key="cumulative_backlog_chart"
    )

    st.subheader("Daily Net Flow")

    fig9 = px.line(
        filtered_df,
        x='date',
        y='daily_net_flow'
    )

    st.plotly_chart(
        fig9,
        use_container_width=True,
        key="daily_net_flow_chart"
    )

# Risk Detection

with tab3:

    st.subheader("Operational Risk Alerts")

    if len(anomalies) > 0:

        st.error(
            f"{len(anomalies)} anomalous operational days detected."
        )

        st.dataframe(
            anomalies[
                [
                    'date',
                    'transfer_efficiency',
                    'transfer_zscore'
                ]
            ].head(20)
        )

    else:

        st.success(
            "No major anomalies detected."
        )

    st.subheader("Outcome Stability Score")

    fig10 = px.line(
        filtered_df,
        x='date',
        y='outcome_stability_score'
    )

    st.plotly_chart(
        fig10,
        use_container_width=True,
        key="stability_chart"
    )

    st.subheader("Weekday Transfer Efficiency Analysis")

    weekday_avg = (
        filtered_df.groupby('weekday')[
            'transfer_efficiency'
        ]
        .mean()
        .reset_index()
    )

    fig11 = px.bar(
        weekday_avg,
        x='weekday',
        y='transfer_efficiency'
    )

    st.plotly_chart(
        fig11,
        use_container_width=True,
        key="weekday_analysis_chart"
    )

# Forecasting

with tab4:

    st.subheader("30-Day Transfer Efficiency Forecast")

    forecast_df = df[
        ['date', 'transfer_efficiency']
    ].copy()

    forecast_df.columns = ['ds', 'y']

    model = Prophet()

    model.fit(forecast_df)

    future = model.make_future_dataframe(
        periods=30
    )

    forecast = model.predict(future)

    forecast_chart = px.line(
        forecast,
        x='ds',
        y='yhat',
        title='Future Transfer Efficiency Prediction'
    )

    st.plotly_chart(
        forecast_chart,
        use_container_width=True,
        key="forecast_chart"
    )

    st.subheader("Forecast Preview")

    st.dataframe(
        forecast[
            ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        ].tail(10)
    )

# Executive Summary

with tab5:

    st.subheader("Executive Summary")

    avg_efficiency = round(
        filtered_df['transfer_efficiency'].mean(),
        2
    )

    avg_discharge_exec = round(
        filtered_df['discharge_effectiveness'].mean(),
        2
    )

    total_backlog_exec = int(
        filtered_df['backlog_accumulation'].sum()
    )

    # Operational Status

    st.subheader("Operational Status Assessment")

    if avg_efficiency > 0.8:

        st.success(
            "The UAC care transition pipeline is operating efficiently with stable transfer movement and effective sponsor placement outcomes."
        )

    elif avg_efficiency > 0.5:

        st.warning(
            "The pipeline is showing moderate operational stress with visible transfer delays and backlog accumulation risk."
        )

    else:

        st.error(
            "The care transition pipeline is experiencing severe operational inefficiencies and elevated bottleneck risk."
        )

    # Best Operational Day

    st.subheader("Best Operational Day")

    best_day = filtered_df.loc[
        filtered_df['transfer_efficiency'].idxmax()
    ]

    st.info(
        f"""
        Date: {best_day['date'].date()}
        
        Highest Transfer Efficiency Recorded: {round(best_day['transfer_efficiency'], 2)}
        """
    )

    # Worst Operational Day

    st.subheader("Highest Bottleneck Day")

    worst_day = filtered_df.loc[
        filtered_df['backlog_accumulation'].idxmax()
    ]

    st.error(
        f"""
        Date: {worst_day['date'].date()}
        
        Highest Backlog Accumulation Recorded: {int(worst_day['backlog_accumulation'])}
        """
    )

    # Risk Classification

    st.subheader("Operational Risk Classification")

    if total_backlog_exec < 10000:

        st.success(
            "Low Operational Risk"
        )

    elif total_backlog_exec < 50000:

        st.warning(
            "Moderate Operational Risk"
        )

    else:

        st.error(
            "High Operational Risk"
        )

    # Policy Recommendations

    st.subheader("Policy Recommendations")

    recommendations = []

    if avg_efficiency < 0.7:

        recommendations.append(
            "- Improve CBP to HHS transfer coordination workflows."
        )

    if avg_discharge_exec < 0.05:

        recommendations.append(
            "- Increase sponsor placement processing capacity."
        )

    if total_backlog_exec > 20000:

        recommendations.append(
            "- Deploy emergency backlog reduction strategies."
        )

    recommendations.append(
        "- Enhance real-time operational monitoring systems."
    )

    recommendations.append(
        "- Strengthen inter-agency coordination mechanisms."
    )

    for rec in recommendations:

        st.markdown(rec)

    # Stakeholder Insights

    st.subheader("Stakeholder Insights")

    st.markdown(f"""
    - Average Transfer Efficiency observed during the selected period was **{avg_transfer}**.
    
    - Average Discharge Effectiveness remained at **{avg_discharge}**.
    
    - Total backlog accumulation reached **{total_backlog}** unresolved cases.
    
    - Operational flow monitoring indicates that transfer and discharge activities require continuous performance stabilization.
    
    - Predictive forecasting suggests future operational fluctuations may continue if bottleneck conditions persist.
    """)

    # Dataset Preview

    st.subheader("Filtered Dataset Preview")

    st.dataframe(filtered_df)

# Download Dataset

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name='filtered_uac_data.csv',
    mime='text/csv'
)