import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.title('Gold Price Prediction')

# Load historical prices
df_historical = pd.read_csv('./datasets/historical_prices.csv')
df_historical['Date'] = pd.to_datetime(df_historical['Date'])

# Load predictions
df_predictions = pd.read_csv('./datasets/predictions.csv')
df_predictions['ds'] = pd.to_datetime(df_predictions['ds'])

# Get the last 6 months of historical data
last_date = df_historical['Date'].max()
start_date = last_date - timedelta(days=180)
df_last_6_months = df_historical[df_historical['Date'] >= start_date]

# Create the plot
fig, ax = plt.subplots(figsize=(14, 7))

# Plot last 6 months of historical prices
ax.plot(df_last_6_months['Date'], df_last_6_months['gold'], label='Historical Gold Price', linewidth=2)

# Plot predictions for the next month (30 days)
next_month = df_predictions[df_predictions['ds'] > last_date][:30]
ax.plot(next_month['ds'], next_month['yhat'], label='Predicted Gold Price', color='red', linewidth=2)

# Customize the plot
ax.set_xlabel('Date')
ax.set_ylabel('Gold Price')
ax.set_title('Last 6 Months of Historical Gold Prices and 1-Month Prediction')
ax.legend()
ax.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot in Streamlit
st.pyplot(fig)

# Display data tables
st.subheader('Last 6 Months of Historical Prices')
st.write(df_last_6_months)

st.subheader('Predictions for Next Month')
st.write(next_month)