import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import pickle
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="AI Trading Bot Dashboard v1.0.0", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
    .warning-card {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
    }
    .success-card {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🤖 AI Trading Bot Dashboard v1.0.0</div>', unsafe_allow_html=True)
st.markdown("**Hybrid ML-DRL Trading System** | Real-time Monitoring & Analysis")

dashboard_mode = st.sidebar.radio(
    "Select Dashboard Mode",
    ["📈 Market Analysis", "🤖 Bot Performance", "📊 Portfolio Overview", "⚙️ System Status"]
)

@st.cache_data
def load_historical_data():
    """Load all historical data files"""
    data_folder = "historical_data"
    data_dict = {}
    
    if not os.path.exists(data_folder):
        return data_dict
    
    asset_files = glob.glob(os.path.join(data_folder, "*_hourly.csv"))
    
    for file_path in asset_files:
        asset_name = os.path.basename(file_path).replace("_hourly.csv", "")
        try:
            df = pd.read_csv(file_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            data_dict[asset_name] = df
        except Exception as e:
            st.sidebar.warning(f"Failed to load {asset_name}: {e}")
    
    return data_dict

@st.cache_data
def load_ml_models():
    """Load ML model information"""
    models_folder = "models"
    model_info = {}
    
    if not os.path.exists(models_folder):
        return model_info
    
    model_files = glob.glob(os.path.join(models_folder, "*.pkl"))
    
    for model_path in model_files:
        asset_name = os.path.basename(model_path).replace("_model.pkl", "")
        try:
            model_info[asset_name] = {
                "path": model_path,
                "size": os.path.getsize(model_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(model_path))
            }
        except Exception as e:
            st.sidebar.warning(f"Model info error for {asset_name}: {e}")
    
    return model_info

def load_performance_logs():
    """Load performance logs if they exist"""
    log_files = glob.glob("performance_logs/*.json")
    logs = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                logs.append(log_data)
        except:
            continue
    
    return logs

def get_trading_signals():
    """Generate current trading signals"""
    try:
        from signal_predictor import TradingSignalPredictor
        predictor = TradingSignalPredictor()
        return predictor.get_all_signals()
    except Exception as e:
        return {"error": str(e)}

# Load data
historical_data = load_historical_data()
ml_models = load_ml_models()
performance_logs = load_performance_logs()

# --- DASHBOARD MODES ---

if dashboard_mode == "📈 Market Analysis":
    st.header("📈 Market Analysis & Trading Signals")
    
    if not historical_data:
        st.stop()
    
    # Asset selection
    selected_assets = st.multiselect(
        "Select Assets to Analyze",
        list(historical_data.keys()),
        default=list(historical_data.keys())[:2]
    )
    
    if not selected_assets:
        st.stop()
    
    # Time range selection
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox("Analysis Period", [7, 30, 90, 180, 365], index=1)
    with col2:
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"])
    
    # Generate signals
    st.subheader("🎯 Current Trading Signals")
    signals = get_trading_signals()
    
    if "error" not in signals:
        signal_cols = st.columns(len(selected_assets))
        for i, asset in enumerate(selected_assets):
            with signal_cols[i]:
                if asset in signals:
                    signal_data = signals[asset]
                    signal_type = signal_data.get('signal', 'HOLD')
                    confidence = signal_data.get('confidence', 0.5)
                    
                    if signal_type == 'BUY':
                        st.success(f"**{asset}**\n🟢 BUY\nConfidence: {confidence:.1%}")
                    elif signal_type == 'SELL':
                        st.error(f"**{asset}**\n🔴 SELL\nConfidence: {confidence:.1%}")
                    else:
                        st.info(f"**{asset}**\n⚪ HOLD\nConfidence: {confidence:.1%}")
    else:
        pass
    
    # Price charts for selected assets
    for asset in selected_assets:
        if asset in historical_data:
            df = historical_data[asset].copy()
            
            # Filter by time range
            if 'timestamp' in df.columns:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                df = df[df['timestamp'] >= cutoff_date]
            
            st.subheader(f"📊 {asset} Price Chart ({days_back} days)")
            
            # Create chart based on type
            fig = go.Figure()
            
            if chart_type == "Candlestick" and all(col in df.columns for col in ['1. open', '2. high', '3. low', '4. close']):
                fig.add_trace(go.Candlestick(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    open=df['1. open'],
                    high=df['2. high'],
                    low=df['3. low'],
                    close=df['4. close'],
                    name=asset
                ))
                price_col = '4. close'
            else:
                # Line chart fallback
                price_col = '4. close' if '4. close' in df.columns else df.columns[-1]
                fig.add_trace(go.Scatter(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    y=df[price_col],
                    mode='lines',
                    name=asset,
                    line=dict(width=2)
                ))
            
            # Add moving averages
            if len(df) >= 20:
                for window in [20, 50]:
                    if len(df) >= window:
                        ma = df[price_col].rolling(window=window).mean()
                        fig.add_trace(go.Scatter(
                            x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                            y=ma,
                            mode='lines',
                            name=f'MA{window}',
                            line=dict(dash='dot', width=1)
                        ))
            
            fig.update_layout(
                title=f'{asset} Price Chart',
                xaxis_title='Time',
                yaxis_title='Price',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            if len(df) > 0:
                with col1:
                    current_price = df[price_col].iloc[-1]
                    st.metric("Current Price", f"{current_price:.4f}")
                with col2:
                    price_change = df[price_col].iloc[-1] - df[price_col].iloc[0]
                    st.metric("Period Change", f"{price_change:.4f}")
                with col3:
                    pct_change = (price_change / df[price_col].iloc[0]) * 100
                    st.metric("% Change", f"{pct_change:.2f}%")
                with col4:
                    volatility = df[price_col].std()
                    st.metric("Volatility", f"{volatility:.4f}")

elif dashboard_mode == "🤖 Bot Performance":
    st.header("🤖 Bot Performance & Trading History")
    
    # Performance metrics
    if performance_logs:
        st.subheader("📊 Recent Performance")
        
        # Convert logs to DataFrame
        df_logs = pd.DataFrame(performance_logs)
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
        df_logs = df_logs.sort_values('timestamp')
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_win_rate = df_logs['win_rate'].astype(float).mean()
            st.metric("Average Win Rate", f"{avg_win_rate:.1f}%")
        with col2:
            avg_reward = df_logs['avg_reward'].astype(float).mean()
            st.metric("Average Reward", f"{avg_reward:.2f}")
        with col3:
            total_episodes = len(df_logs)
            st.metric("Total Episodes", total_episodes)
        with col4:
            best_performance = df_logs['win_rate'].astype(float).max()
            st.metric("Best Win Rate", f"{best_performance:.1f}%")
        
        # Performance over time
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Win Rate Over Time', 'Average Reward Over Time'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=df_logs['timestamp'], y=df_logs['win_rate'].astype(float),
                      mode='lines+markers', name='Win Rate', line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_logs['timestamp'], y=df_logs['avg_reward'].astype(float),
                      mode='lines+markers', name='Avg Reward', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # ML Model Performance
    st.subheader("🧠 ML Model Status")
    
    if ml_models:
        model_data = []
        for asset, info in ml_models.items():
            model_data.append({
                "Asset": asset,
                "File Size": f"{info['size'] / 1024:.1f} KB",
                "Last Updated": info['modified'].strftime("%Y-%m-%d %H:%M"),
                "Status": "✅ Ready"
            })
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)

elif dashboard_mode == "📊 Portfolio Overview":
    st.header("📊 Portfolio Overview & Risk Analysis")
    
    # Simulated portfolio data (in real implementation, this would come from trading logs)
    assets = list(historical_data.keys()) if historical_data else ['AUDUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    
    # Portfolio allocation (placeholder)
    st.subheader("💼 Current Portfolio Allocation")
    
    allocation_data = {
        'Asset': assets,
        'Allocation %': [25, 25, 25, 25],  # Equal weighting for demo
        'Current Position': ['Long', 'Short', 'Long', 'Neutral'],
        'P&L': [150.25, -75.50, 220.75, 0.00]
    }
    
    df_portfolio = pd.DataFrame(allocation_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Portfolio pie chart
        fig_pie = px.pie(df_portfolio, values='Allocation %', names='Asset', 
                        title="Portfolio Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Portfolio summary
        total_pnl = df_portfolio['P&L'].sum()
        winning_positions = (df_portfolio['P&L'] > 0).sum()
        total_positions = len(df_portfolio)
        
        st.metric("Total P&L", f"${total_pnl:.2f}")
        st.metric("Winning Positions", f"{winning_positions}/{total_positions}")
        st.metric("Win Rate", f"{(winning_positions/total_positions)*100:.1f}%")
    
    # Detailed portfolio table
    st.subheader("📋 Detailed Positions")
    st.dataframe(df_portfolio, use_container_width=True)
    
    # Risk metrics
    st.subheader("⚠️ Risk Metrics")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    with risk_col1:
        st.metric("Portfolio Volatility", "12.5%")
    with risk_col2:
        st.metric("Max Drawdown", "-8.2%")
    with risk_col3:
        st.metric("Sharpe Ratio", "1.24")

elif dashboard_mode == "⚙️ System Status":
    st.header("⚙️ System Status & Configuration")
    
    # System health checks
    st.subheader("🏥 System Health")
    
    health_checks = {
        "Historical Data": len(historical_data) > 0,
        "ML Models": len(ml_models) > 0,
        "Python Environment": True,  # Always true if dashboard is running
        "Required Packages": True,   # Always true if dashboard is running
    }
    
    for check, status in health_checks.items():
        if status:
            st.success(f"✅ {check}: OK")
        else:
            st.error(f"❌ {check}: Failed")
    
    # File system status
    st.subheader("📁 File System Status")
    
    folders_to_check = ["historical_data", "models", "performance_logs"]
    
    for folder in folders_to_check:
        if os.path.exists(folder):
            file_count = len(os.listdir(folder))
            folder_size = sum(os.path.getsize(os.path.join(folder, f)) 
                            for f in os.listdir(folder) 
                            if os.path.isfile(os.path.join(folder, f)))
            st.info(f"📁 {folder}: {file_count} files, {folder_size/1024:.1f} KB")
    
    # Configuration
    st.subheader("⚙️ Configuration")
    
    # Show current configuration
    config_info = {
        "Trading Pairs": list(historical_data.keys()) if historical_data else "None loaded",
        "Dashboard Version": "v1.0.0",
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Working Directory": os.getcwd()
    }
    
    for key, value in config_info.items():
        st.text(f"{key}: {value}")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    <p><strong>AI Trading Bot v1.0.0</strong> | Developed by Nyenzo | Powered by Streamlit</p>
    <p>⚠️ <em>For educational purposes only. Trading involves risk of loss.</em></p>
</div>
""", unsafe_allow_html=True)
