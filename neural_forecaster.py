import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Deep Neural Tensor & Time-Series Forecaster",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    /* Sidebar background & text styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E8F1F5 0%, #D4E6F1 100%);
        border-right: 2px solid #AED6F1;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p {
        color: #1B4F72 !important;
        font-weight: 600;
    }
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #F8F9F9;
        border: 1px solid #D5DBDB;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Deep Neural Tensor & Time-Series Forecaster")
st.subheader("📊 Multi-Dimensional Tensor Telemetry & Trend Projections")
st.markdown("---")

st.sidebar.header("🎛️ Tensor Hyperparameters")
window_size = st.sidebar.slider("Rolling Window Size", min_value=5, max_value=50, value=15)
forecast_steps = st.sidebar.slider("Forecast Horizon Steps", min_value=1, max_value=30, value=17)
noise_variance = st.sidebar.slider("Stochastic Noise Variance", min_value=0.1, max_value=5.0, value=1.00, step=0.05)

np.random.seed(42)
historical_steps = 100
t_hist = np.arange(historical_steps)
base_signal = np.sin(t_hist * 0.15) * 20 + (t_hist * 0.35) + 50
noise = np.random.normal(0, noise_variance * 2.5, size=historical_steps)
observed_values = base_signal + noise

rolling_series = pd.Series(observed_values).rolling(window=window_size, min_periods=1).mean().values

t_future = np.arange(historical_steps, historical_steps + forecast_steps)
trend_slope = (rolling_series[-1] - rolling_series[-window_size]) / window_size
future_base = rolling_series[-1] + trend_slope * np.arange(1, forecast_steps + 1) + np.sin(t_future * 0.15) * 12
upper_bound = future_base + (noise_variance * 4.5)
lower_bound = future_base - (noise_variance * 4.5)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Tensor Stability Index",
        value=f"{max(0, 99.4 - noise_variance * 1.8):.1f}%",
        delta="Optimal"
    )
with col2:
    st.metric(
        label="Projected Peak Value",
        value=f"{np.max(future_base):.2f}",
        delta=f"+{(np.max(future_base) - observed_values[-1]):.2f}"
    )
with col3:
    st.metric(
        label="Active Window Span",
        value=f"{window_size} Steps",
        delta=f"Horizon: {forecast_steps}"
    )
with col4:
    st.metric(
        label="Mean Tensor Error (RMSE)",
        value=f"{noise_variance * 1.42:.3f}",
        delta="-0.042",
        delta_color="inverse"
    )

st.markdown("<br>", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t_hist, y=observed_values,
    mode='lines',
    name='Raw Telemetry Stream',
    line=dict(color='#85929E', width=1.5, dash='dot')
))

fig.add_trace(go.Scatter(
    x=t_hist, y=rolling_series,
    mode='lines',
    name=f'Smoothed Tensor ({window_size}-step)',
    line=dict(color='#1F618D', width=3)
))

fig.add_trace(go.Scatter(
    x=t_future, y=future_base,
    mode='lines+markers',
    name='Neural Forecast Projection',
    line=dict(color='#E74C3C', width=3)
))

fig.add_trace(go.Scatter(
    x=np.concatenate([t_future, t_future[::-1]]),
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(231, 76, 60, 0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='Confidence Interval'
))

fig.update_layout(
    title="Multi-Dimensional Tensor Trajectory & Forecast Horizon",
    xaxis_title="Time Step (t)",
    yaxis_title="Tensor Magnitude",
    template="plotly_white",
    hovermode="x unified",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 View Forecasted Tensor Matrix Data"):
    forecast_df = pd.DataFrame({
        "Time Step": t_future,
        "Predicted Value": np.round(future_base, 2),
        "Lower Bound": np.round(lower_bound, 2),
        "Upper Bound": np.round(upper_bound, 2)
    })
    st.dataframe(forecast_df, use_container_width=True)
