import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import joblib

st.title('Gold Price Prediction')

# Load historical prices
df_historical = pd.read_csv('./datasets/historical_prices.csv')
df_historical['Date'] = pd.to_datetime(df_historical['Date'])

# Load predictions
df_predictions = pd.read_csv('./datasets/predictions.csv')
df_predictions['ds'] = pd.to_datetime(df_predictions['ds'])

# Load the scaler used during training
scaler = joblib.load('scaler.pkl')

# Inverse transform the scaled values
numeric_columns = ['fear_and_greed', 'gold', 'crude_oil', 'platinum', 'usd_index']
df_historical[numeric_columns] = scaler.inverse_transform(df_historical[numeric_columns])
df_predictions['yhat'] = scaler.inverse_transform(df_predictions[['yhat']].values.reshape(-1, 1)).flatten()
# Get the last 6 months of historical data
last_date = df_historical['Date'].max()
start_date = last_date - timedelta(days=180)
df_last_6_months = df_historical[df_historical['Date'] >= start_date]

# Create the plot
fig, ax = plt.subplots(figsize=(20, 10))  # Increased figure size

# Plot last 6 months of historical prices
ax.plot(df_last_6_months['Date'], df_last_6_months['gold'], label='Historical Gold Price', linewidth=3)

# Plot all predictions
ax.plot(df_predictions['ds'], df_predictions['yhat'], label='Predicted Gold Price', color='red', linewidth=3)

# Customize the plot
ax.set_xlabel('Date', fontsize=16)  # Increased font size
ax.set_ylabel('Gold Price', fontsize=16)  # Increased font size
ax.set_title('Last 6 Months of Historical Gold Prices and All Predictions', fontsize=20)  # Increased font size
ax.legend(fontsize=14)  # Increased font size
ax.grid(True)

# Increase font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

# Display data tables
st.subheader('Last 6 Months of Historical Prices')
st.write(df_last_6_months)

st.subheader('All Predictions')
st.write(df_predictions)