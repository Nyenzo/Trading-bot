import streamlit as st
import pandas as pd
import os
import glob

st.set_page_config(page_title="Trading Bot Analysis Dashboard", layout="wide")
st.title("📊 Trading Bot Analysis Dashboard")

# --- Sidebar: Asset Selection ---
st.sidebar.header("Select Asset")

data_folder = "historical_data"
asset_files = glob.glob(os.path.join(data_folder, "*_hourly.csv"))
assets = [os.path.basename(f).replace("_hourly.csv", "") for f in asset_files]

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
    price_col = None
    for col in df.columns:
        if col.lower() in ["close", "price", "close_price"]:
            price_col = col
            break
    if not price_col:
        price_col = df.columns[-1]  # fallback to last column
    
    # --- Main Chart ---
    st.plotly_chart({
        "data": [{
            "x": df[date_col] if date_col else df.index,
            "y": df[price_col],
            "type": "scatter",
            "mode": "lines",
            "name": price_col
        }],
        "layout": {"title": f"{selected_asset} {price_col.title()} Over Time"}
    })
    
    # --- Summary Statistics ---
    st.markdown("### Summary Statistics")
    st.write(df.describe())
    
    # --- Interactive Filtering ---
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
        st.line_chart(filtered_df.set_index(date_col)[price_col])
else:
    st.error(f"No data found for {selected_asset}.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by [Your Name] | Powered by Streamlit") 