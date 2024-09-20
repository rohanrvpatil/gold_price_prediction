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

# Inverse transform the 'yhat' predictions
# Create a DataFrame with the same structure as the original data for inverse transformation
df_temp = pd.DataFrame(columns=numeric_columns)
df_temp['gold'] = df_predictions['yhat']
df_temp = df_temp.fillna(0)  # Fill other columns with zeros

# Inverse transform and extract the 'gold' column
df_predictions['yhat'] = scaler.inverse_transform(df_temp)[:, numeric_columns.index('gold')]



# Get the last 6 months of historical data
last_date = df_historical['Date'].max()
start_date = last_date - timedelta(days=180)
df_last_6_months = df_historical[df_historical['Date'] >= start_date]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size

# Plot last 6 months of historical prices
ax.plot(df_last_6_months['Date'], df_last_6_months['gold'], label='Historical Gold Price', linewidth=3)

# Plot all predictions
ax.plot(df_predictions['ds'], df_predictions['yhat'], label='Predicted Gold Price', color='red', linewidth=3)

# Customize the plot
ax.set_xlabel('Date')
ax.set_ylabel('Gold Price ($)')
ax.set_title('Last 6 Months of Historical Gold Prices and All Predictions')
ax.legend()
ax.grid(True)

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