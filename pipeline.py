import requests
from datetime import datetime
import pandas as pd
import yfinance as yf
from prophet import Prophet
import hsfs


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
        historical_prices = pd.merge(historical_prices, ticker_data, on='Date', how='left')
    
    historical_prices.to_csv('./datasets/historical_prices.csv', index=False)
   



def train_model():
    df = pd.read_csv('./datasets/historical_prices.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.rename(columns={'Date': 'ds', 'gold': 'y'})

    model = Prophet()
    for parameter in PARAMETERS:
        model.add_regressor(parameter)

    model.fit(df)
    model.save('prophet_model.pkl')


def make_predictions():
    model = Prophet.load('prophet_model.pkl')
    df = pd.read_csv('./datasets/historical_prices.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.rename(columns={'Date': 'ds', 'gold': 'y'})

    forecast = model.predict(df)
    prediction = forecast[['ds', 'yhat']]
    prediction.to_csv('./datasets/predictions.csv', index=False)


def main():
    fetch_new_data()
    train_model()
    make_predictions()

    connection = hsfs.connection()
    fs = connection.get_feature_store(name='gold_price_prediction_featurestore')
    #fg = fs.get_feature_group('default', version=1)
    
    # Upload historical prices
    df_historical = pd.read_csv('./datasets/historical_prices.csv')
    fs.save(df_historical, "historical_prices", primary_key=["Date"], description="Historical prices data")

    # Upload predictions
    df_predictions = pd.read_csv('./datasets/predictions.csv')
    fs.save(df_predictions, "predictions", primary_key=["ds"], description="Predictions for the next 5 days")

if __name__ == "__main__":
    main()