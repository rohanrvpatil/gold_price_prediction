# Gold Price Prediction

**Aim:** To predict the price of gold using historical data and other factors like fear and greed index <br>
**Model used:** Facebook Prophet <br>
**Pipeline running time:** runs daily at midnight UTC time <br>
**Cron job:** Github Actions <br>
**Regressors used:** Fear and Greed Index, Crude Oil Prices, USD Index, Platinum Prices <br>


## Implementation steps:

1. **Imports**: Imported necessary libraries for data fetching, processing, and modeling.
2. **Parameters**: Defined parameters and start date for data fetching.
3. **fetch_new_data()**: Fetched and processed fear and greed index and financial data, saved to CSV.
4. **train_model()**: Trained a Prophet model on the historical data, saved the model.
5. **make_predictions()**: Loaded the model, made future predictions, and saved them to CSV.
6. **main()**: Orchestrated the pipeline by calling data fetching, model training, and prediction functions.


## Steps to update pipeline.py in the future:

1) Make changes to the pipeline.py file
2) Run "python pipeline.py". All the .csv files will be updated
3) Push the changes to the repository
4) The changes will be reflected in the website instantly


## Data resources:

Gold price : https://finance.yahoo.com/quote/GC%3DF?p=GC%3DF <br>
Platinum price : https://finance.yahoo.com/quote/PL%3DF?p=PL%3DF <br>
USD index : https://finance.yahoo.com/quote/DX-Y.NYB?p=DX-Y.NYB <br>
Crude oil price : https://finance.yahoo.com/quote/CL%3DF?p=CL%3DF <br>
Fear and greed index: https://edition.cnn.com/markets/fear-and-greed <br>
