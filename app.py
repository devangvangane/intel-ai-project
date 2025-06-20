import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import numpy as np
import altair as alt

st.set_page_config(page_title="Inventory Forecasting Dashboard", layout="wide")
st.title("📦 Demand-based Inventory Management System")

# --- Upload CSV File ---
uploaded_file = st.file_uploader("Upload Inventory Dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head(10))

    # --- Encode Categorical Features ---
    cat_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # --- Feature Selection ---
    features = [
        'Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Discount',
        'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing',
        'Category', 'Region', 'Seasonality'
    ]
    X = df[features]
    y = df['Demand Forecast']

    # --- Train Model ---
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X, y)

    df['Predicted Demand'] = model.predict(X)

    # --- Reorder Logic ---
    lead_time = st.slider("Select lead time (days)", min_value=1, max_value=10, value=3)
    safety_stock_pct = st.slider("Select safety stock (%)", min_value=0, max_value=50, value=10)
    safety_factor = 1 + safety_stock_pct / 100

    df['Reorder Point'] = df['Predicted Demand'] * lead_time * safety_factor
    df['Order Needed'] = df['Inventory Level'] < df['Reorder Point']
    df['Recommended Order Qty'] = df.apply(lambda row: max(row['Reorder Point'] - row['Inventory Level'], 0), axis=1)

    # --- Filters ---
    with st.expander("🔍 Filter Results"):
        category = st.selectbox("Category", options=["All"] + list(df['Category'].unique()))
        region = st.selectbox("Region", options=["All"] + list(df['Region'].unique()))

    df_filtered = df.copy()
    if category != "All":
        df_filtered = df_filtered[df_filtered['Category'] == category]
    if region != "All":
        df_filtered = df_filtered[df_filtered['Region'] == region]

    # --- Show Final Forecast Table ---
    st.subheader("📦 Reorder Suggestions")
    st.dataframe(df_filtered[['Product ID', 'Inventory Level', 'Predicted Demand',
                              'Reorder Point', 'Order Needed', 'Recommended Order Qty']])

    # --- Download Button ---
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast as CSV", csv, "inventory_forecast.csv", "text/csv")

    # --- Time Series Plot for Individual Product ---
    st.subheader("📈 Product-wise Demand Trend (Time Series)")

    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    product_id = st.selectbox("Select Product ID for time-series view", df['Product ID'].unique())
    y_axis_option = st.radio("Select metric to visualize:", ['Units Sold', 'Predicted Demand', 'Both'], horizontal=True)

    product_df = df[df['Product ID'] == product_id].sort_values(by='Date')

    if not product_df.empty:
        base = alt.Chart(product_df).encode(x='Date:T')

        if y_axis_option == 'Units Sold':
            chart = base.mark_line(point=True, color='steelblue').encode(
                y=alt.Y('Units Sold:Q', title='Units Sold'),
                tooltip=['Date:T', 'Units Sold:Q']
            )

        elif y_axis_option == 'Predicted Demand':
            chart = base.mark_line(point=True, color='green').encode(
                y=alt.Y('Predicted Demand:Q', title='Predicted Demand'),
                tooltip=['Date:T', 'Predicted Demand:Q']
            )

        else:  # Both
            sold_line = base.mark_line(color='red').encode(
                y='Units Sold:Q',
                tooltip=['Date:T', 'Units Sold:Q']
            )

            sold_points = base.mark_point(color='red').encode(
                y='Units Sold:Q'
            )

            forecast_line = base.mark_line(color='green').encode(
                y='Predicted Demand:Q',
                tooltip=['Date:T', 'Predicted Demand:Q']
            )

            forecast_points = base.mark_point(color='green').encode(
                y='Predicted Demand:Q'
            )

            chart = alt.layer(sold_line, sold_points, forecast_line, forecast_points).resolve_scale(y='independent')

        st.altair_chart(chart.properties(
            title=f"{y_axis_option} Over Time - Product {product_id}",
            width=800,
            height=400
        ), use_container_width=True)

    else:
        st.warning("No data found for selected product.")

else:
    st.info("👆 Upload a CSV file to begin.")