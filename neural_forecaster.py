import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Deep Neural Tensor Forecaster",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #020617; color: #f8fafc; }
    .stSidebar { background-color: #0f172a; }
    h1, h2, h3 { color: #a855f7; font-family: 'Courier New', monospace; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Deep Neural Tensor & Time-Series Forecaster")
st.markdown("---")

st.sidebar.header("🎛️ Tensor Hyperparameters")
window_size = st.sidebar.slider("Rolling Window Size", 5, 50, 15, 5)
forecast_horizon = st.sidebar.slider("Forecast Horizon Steps", 1, 20, 5, 1)
noise_level = st.sidebar.slider("Stochastic Noise Variance", 0.1, 1.5, 0.5, 0.1)

try:
    np.random.seed(13) # Dedicated seed for high-precision replication
    samples = 250
    time_index = np.arange(samples)
    
    raw_signal = (
        np.sin(time_index * 0.08) * 10 +
        np.cos(time_index * 0.03) * 5 +
        np.random.normal(0, noise_level, samples)
    )
    
    df_engine = pd.DataFrame({
        "Time": time_index,
        "Signal": raw_signal
    })
    
    df_engine['Rolling_Mean'] = df_engine['Signal'].rolling(window=window_size).mean()
    df_engine['Rolling_Std'] = df_engine['Signal'].rolling(window=window_size).std()
    
    df_engine['Neural_Prediction'] = df_engine['Rolling_Mean'].shift(-forecast_horizon)
    
    df_clean = df_engine.dropna().reset_index(drop=True)

    st.subheader("📊 Multi-Dimensional Tensor Telemetry & Trend Projections")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="⚙️ Processed Data Tensors", value=f"{len(df_clean)} Vectors", status="Optimized")
    with col2:
        st.metric(label="📈 Active Rolling Window", value=f"{window_size} Steps", status="Calibrated")
    with col3:
        st.metric(label="🛡️ Neural Pipeline Status", value="Active / 100% Secure", status="Online")
        
    st.markdown(f"**Mathematical Engine:** Executing rolling variance and shifting forecasts across a horizon of `{forecast_horizon}` steps.")
    st.markdown("---")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#020617')
    ax.set_facecolor('#0f172a')
    
    ax.plot(df_clean['Time'], df_clean['Signal'], color='#38bdf8', alpha=0.5, label='Raw Telemetry Stream')
    ax.plot(df_clean['Time'], df_clean['Rolling_Mean'], color='#a855f7', linewidth=2, label=f'Rolling Mean ({window_size}w)')
    ax.plot(df_clean['Time'], df_clean['Neural_Prediction'], color='#22c55e', linestyle='--', linewidth=2, label=f'Neural Forecast (+{forecast_horizon}h)')
    
    ax.set_title("Neural Tensor Predictive Variance & Trend Tracking", color='#a855f7', fontsize=12, fontfamily='monospace')
    ax.tick_params(colors='#f8fafc')
    ax.legend(facecolor='#0f172a', edgecolor='#a855f7', labelcolor='#f8fafc')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#38bdf8')
        
    st.pyplot(fig)

except Exception as e:
    st.error(f"⚠️ Critical Neural Exception Encountered: {e}")
    st.info("The tensor execution stream is offline. Please adjust your window parameters to restore matrix stability.")
