import streamlit as st
import pandas as pd
import os
import glob
import plotly.graph_objs as go
import numpy as np

st.set_page_config(page_title="Trading Bot Analysis Dashboard", layout="wide")
st.title("📊 Trading Bot Analysis Dashboard")

# --- Sidebar: Asset Selection ---
st.sidebar.header("Select Asset")

data_folder = "historical_data"
asset_files = glob.glob(os.path.join(data_folder, "*_hourly.csv"))
assets = [os.path.basename(f).replace("_hourly.csv", "") for f in asset_files]

if not assets:
    st.sidebar.warning("No data files found in 'historical_data'. Upload your data to this folder.")
    st.error("No data found for None.")
    st.markdown("---")
    st.markdown("Developed by Peter Nyenzo | Powered by Streamlit")
    # --- Deployment Note ---
    st.sidebar.info("For Streamlit Cloud deployment, ensure the 'historical_data' folder and CSV files are included in your repo.")
    st.stop()

selected_asset = st.sidebar.selectbox("Asset", assets)

# --- Load Data ---
file_path = os.path.join(data_folder, f"{selected_asset}_hourly.csv")
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.subheader(f"Hourly Data for {selected_asset}")
    st.write(f"Showing {len(df)} rows.")
    
    # --- Data Preview ---
    with st.expander("Show raw data"):
        st.dataframe(df, use_container_width=True)
    
    # --- Date Column Detection ---
    date_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            date_col = col
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
    
    # --- Price Column Detection ---
    preferred_price_cols = ["4. close", "close", "price", "close_price"]
    price_col = None
    for col in preferred_price_cols:
        if col in df.columns:
            price_col = col
            break
    if not price_col:
        # fallback: try to find any column with 'close' in its name
        for col in df.columns:
            if "close" in col.lower():
                price_col = col
                break
    if not price_col:
        price_col = df.columns[-1]  # fallback to last column
    
    # --- Main Chart (Enhanced) ---
    st.markdown("### Price Chart")
    fig = go.Figure()
    # Detect OHLC columns for candlestick
    ohlc_cols = [
        ('open', 'high', 'low', 'close'),
        ('1. open', '2. high', '3. low', '4. close'),
        ('Open', 'High', 'Low', 'Close'),
    ]
    found_ohlc = None
    for o, h, l, c in ohlc_cols:
        if all(col in df.columns for col in [o, h, l, c]):
            found_ohlc = (o, h, l, c)
            break
    if found_ohlc and date_col:
        o, h, l, c = found_ohlc
        fig.add_trace(go.Candlestick(
            x=df[date_col],
            open=df[o], high=df[h], low=df[l], close=df[c],
            name='Candlestick',
        ))
        price_for_ma = df[c]
    else:
        fig.add_trace(go.Scatter(
            x=df[date_col] if date_col else df.index,
            y=df[price_col],
            mode='lines',
            name=price_col.title(),
        ))
        price_for_ma = df[price_col]
    # Add moving averages
    for window in [20, 50]:
        if len(price_for_ma) >= window:
            ma = price_for_ma.rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=df[date_col] if date_col else df.index,
                y=ma,
                mode='lines',
                name=f"MA{window}",
                line=dict(dash='dot')
            ))
    fig.update_layout(title=f"{selected_asset} Price Chart", xaxis_title='Date', yaxis_title=price_col.title())
    st.plotly_chart(fig, use_container_width=True)

    # --- Insights Section ---
    st.markdown("### Insights")
    insights = []
    if date_col:
        start_val = df[price_col].iloc[0]
        end_val = df[price_col].iloc[-1]
        change = end_val - start_val
        pct_change = (change / start_val) * 100 if start_val != 0 else 0
        volatility = df[price_col].std()
        high = df[price_col].max()
        low = df[price_col].min()
        trend = "upward 📈" if change > 0 else ("downward 📉" if change < 0 else "sideways ➖")
        insights.append(f"**Price Change:** {change:.2f} ({pct_change:.2f}%) over the period.")
        insights.append(f"**Volatility (Std Dev):** {volatility:.4f}")
        insights.append(f"**Highest Price:** {high:.4f}")
        insights.append(f"**Lowest Price:** {low:.4f}")
        insights.append(f"**Trend:** {trend}")
    else:
        insights.append("Date column not detected; insights limited.")
    for ins in insights:
        st.write(ins)

    # --- Summary Statistics ---
    st.markdown("### Summary Statistics")
    st.write(df.describe())

    # --- Interactive Filtering (Enhanced) ---
    if date_col:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Date Range Filter")
        min_date, max_date = df[date_col].min(), df[date_col].max()
        start_date, end_date = st.sidebar.date_input(
            "Select date range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
        filtered_df = df.loc[mask]
        st.markdown(f"#### Filtered Data: {start_date} to {end_date}")
        # Enhanced filtered chart
        fig_filt = go.Figure()
        if found_ohlc:
            o, h, l, c = found_ohlc
            fig_filt.add_trace(go.Candlestick(
                x=filtered_df[date_col],
                open=filtered_df[o], high=filtered_df[h], low=filtered_df[l], close=filtered_df[c],
                name='Candlestick',
            ))
            price_for_ma_filt = filtered_df[c]
        else:
            fig_filt.add_trace(go.Scatter(
                x=filtered_df[date_col] if date_col else filtered_df.index,
                y=filtered_df[price_col],
                mode='lines',
                name=price_col.title(),
            ))
            price_for_ma_filt = filtered_df[price_col]
        for window in [20, 50]:
            if len(price_for_ma_filt) >= window:
                ma = price_for_ma_filt.rolling(window=window).mean()
                fig_filt.add_trace(go.Scatter(
                    x=filtered_df[date_col] if date_col else filtered_df.index,
                    y=ma,
                    mode='lines',
                    name=f"MA{window}",
                    line=dict(dash='dot')
                ))
        fig_filt.update_layout(title=f"{selected_asset} Filtered Price Chart", xaxis_title='Date', yaxis_title=price_col.title())
        st.plotly_chart(fig_filt, use_container_width=True)
        # Filtered summary
        st.write(filtered_df.describe())
        # Filtered insights
        st.markdown("**Filtered Insights:**")
        if not filtered_df.empty:
            start_val = filtered_df[price_col].iloc[0]
            end_val = filtered_df[price_col].iloc[-1]
            change = end_val - start_val
            pct_change = (change / start_val) * 100 if start_val != 0 else 0
            volatility = filtered_df[price_col].std()
            high = filtered_df[price_col].max()
            low = filtered_df[price_col].min()
            trend = "upward 📈" if change > 0 else ("downward 📉" if change < 0 else "sideways ➖")
            st.write(f"**Price Change:** {change:.2f} ({pct_change:.2f}%)")
            st.write(f"**Volatility (Std Dev):** {volatility:.4f}")
            st.write(f"**Highest Price:** {high:.4f}")
            st.write(f"**Lowest Price:** {low:.4f}")
            st.write(f"**Trend:** {trend}")
        else:
            st.write("No data in selected range.")
else:
    st.error(f"No data found for {selected_asset}.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by Peter Nyenzo | Powered by Streamlit")

# --- Deployment Note ---
# For Streamlit Cloud deployment, make sure the 'historical_data' folder and all required CSV files are included in your GitHub repository.
# Streamlit Cloud will only have access to files that are tracked in the repo at deploy time. 