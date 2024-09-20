import requests
from datetime import datetime
import pandas as pd
import yfinance as yf
from prophet import Prophet
import hsfs
import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar



PARAMETERS = ['fear_and_greed', 'crude_oil', 'usd_index', 'platinum']


def fetch_new_data():
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'no-cache',
        'origin': 'https://edition.cnn.com',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'referer': 'https://edition.cnn.com/',
        'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    }

    response = requests.get('https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2021-01-01', headers=headers)
    fear_and_greed_data = response.json()
    
    fear_and_greed_list = []

    for entry in fear_and_greed_data['fear_and_greed_historical']['data']:
        timestamp = entry['x'] / 1000  # Convert milliseconds to seconds
        date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
        fear_and_greed_list.append({'Date': date, 'fear_and_greed': entry['y']})

    df_fear_and_greed = pd.DataFrame(fear_and_greed_list)

    tickers = {
        "GC=F": "gold",
        "CL=F": "crude_oil",
        "PL=F": "platinum",
        "DX-Y.NYB": "usd_index"
    }

    historical_prices=df_fear_and_greed.copy()

    for ticker, column_name in tickers.items():
        ticker_data = yf.download(ticker, start="2021-01-01")
        ticker_data.reset_index(inplace=True)
        ticker_data = ticker_data[['Date', 'Open']]
        ticker_data.columns = ['Date', column_name]

        historical_prices['Date'] = pd.to_datetime(historical_prices['Date'])
        ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])

        historical_prices = pd.merge(historical_prices, ticker_data, on='Date', how='left')
    
    historical_prices = historical_prices.iloc[:-1]  # Remove the last row
    historical_prices.to_csv('./datasets/historical_prices.csv', index=False)
   



def train_model():
    df = pd.read_csv('./datasets/historical_prices.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)  # Updated line

    df = df.rename(columns={'Date': 'ds', 'gold': 'y'})

    model = Prophet()
    for parameter in PARAMETERS:
        model.add_regressor(parameter)

    model.fit(df)
    joblib.dump(model, 'prophet_model.pkl')


def make_predictions():
    model = joblib.load('prophet_model.pkl')
    df = pd.read_csv('./datasets/historical_prices.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True) 
    df = df.rename(columns={'Date': 'ds', 'gold': 'y'})

    # Generate future dates for the next 5 trading days
    last_date = df['ds'].max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)
    
    # Remove USA holidays
    us_holidays = USFederalHolidayCalendar()
    holidays = us_holidays.holidays(start=future_dates.min(), end=future_dates.max())
    future_dates = future_dates[~future_dates.isin(holidays)]

    future_df = pd.DataFrame({'ds': future_dates})

    # Add other required columns with the last known values
    for col in df.columns:
        if col not in ['ds', 'y']:
            future_df[col] = df[col].iloc[-1]

    # Combine historical data with future dates
    forecast_df = pd.concat([df, future_df], ignore_index=True)

    # Make predictions
    forecast = model.predict(forecast_df)
    
    # Select only the future predictions
    prediction = forecast[forecast['ds'].isin(future_dates)][['ds', 'yhat']]
    
    # Export predictions to CSV
    prediction.to_csv('./datasets/predictions.csv', index=False)


def main():
    fetch_new_data()
    train_model()
    make_predictions()

    # connection = hsfs.connection()
    # fs = connection.get_feature_store(name='gold_price_prediction_featurestore')
    # #fg = fs.get_feature_group('default', version=1)
    
    # # Upload historical prices
    # df_historical = pd.read_csv('./datasets/historical_prices.csv')
    # fs.save(df_historical, "historical_prices", primary_key=["Date"], description="Historical prices data")

    # # Upload predictions
    # df_predictions = pd.read_csv('./datasets/predictions.csv')
    # fs.save(df_predictions, "predictions", primary_key=["ds"], description="Predictions for the next 5 days")

if __name__ == "__main__":
    main()