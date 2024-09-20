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

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot historical prices
ax.plot(df_historical['Date'], df_historical['gold'], label='Historical Gold Price')

# Plot predictions for the next 5 days
last_date = df_historical['Date'].max()
next_5_days = df_predictions[df_predictions['ds'] > last_date][:5]
ax.plot(next_5_days['ds'], next_5_days['yhat'], label='Predicted Gold Price', color='red')

# Customize the plot
ax.set_xlabel('Date')
ax.set_ylabel('Gold Price')
ax.set_title('Historical Gold Prices and 5-Day Prediction')
ax.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot in Streamlit
st.pyplot(fig)

# Display data tables
# st.subheader('Historical Prices')
# st.write(df_historical)

# st.subheader('Predictions for Next 5 Days')
# st.write(next_5_days)