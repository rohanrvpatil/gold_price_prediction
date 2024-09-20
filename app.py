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

# Get the last 30 days of historical data
last_date = df_historical['Date'].max()
start_date = last_date - timedelta(days=29)
df_last_30_days = df_historical[df_historical['Date'] >= start_date]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot last 30 days of historical prices
ax.plot(df_last_30_days['Date'], df_last_30_days['gold'], label='Historical Gold Price', linewidth=2)

# Plot predictions for the next 5 days
next_5_days = df_predictions[df_predictions['ds'] > last_date][:5]
ax.plot(next_5_days['ds'], next_5_days['yhat'], label='Predicted Gold Price', color='red', linewidth=2)

# Customize the plot
ax.set_xlabel('Date')
ax.set_ylabel('Gold Price')
ax.set_title('Last 30 Days of Historical Gold Prices and 5-Day Prediction')
ax.legend()
ax.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot in Streamlit
st.pyplot(fig)

# Display data tables
# st.subheader('Last 30 Days of Historical Prices')
# st.write(df_last_30_days)

# st.subheader('Predictions for Next 5 Days')
# st.write(next_5_days)