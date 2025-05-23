import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
import logging

# API Credentials
API_KEY = os.environ.get('APCA_API_KEY_ID')
API_SECRET = os.environ.get('APCA_API_SECRET_KEY')
trading_client = TradingClient(API_KEY, API_SECRET)
stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# File paths
DATA_PATH = "data/Out_of_Sample_data.csv"

# Logging Setup
logging.basicConfig(
    filename='logs/trading_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_active_stocks():
    """Fetch tradable stocks and filter out cryptocurrency symbols."""
    assets = trading_client.get_all_assets()
    return [asset.symbol for asset in assets if asset.tradable and '/' not in asset.symbol]

def fetch_historical_data(symbols, start_date, end_date, timeframe=TimeFrame.Day, batch_size=200):
    """Fetch historical data for multiple symbols in batches."""
    all_bars = pd.DataFrame()
    
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i + batch_size]
        request_params = StockBarsRequest(
            symbol_or_symbols=batch_symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            limit=1000  # Max per request
        )
        bars = stock_client.get_stock_bars(request_params).df
        all_bars = pd.concat([all_bars, bars])
    
    return all_bars

def save_historical_data():
    """Fetch and save a large set of historical stock data."""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    active_stocks = get_active_stocks()
    #end_date = datetime.today().strftime('%Y-%m-%d')
    #start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 years of data
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    print(f"Fetching historical data from {start_date} to {end_date}...")
    data = fetch_historical_data(active_stocks, start_date, end_date)
    data.to_csv(DATA_PATH)
    print("Historical data saved successfully!")
    logging.info("Historical data collected and saved.")

def update_historical_data():
    """Fetch and append new stock data to keep the dataset up to date."""
    if not os.path.exists(DATA_PATH):
        print("No historical data found. Collecting initial data...")
        save_historical_data()
        return
    
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    last_date = df.index.max().strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"Updating historical data from {last_date} to {end_date}...")
    active_stocks = get_active_stocks()
    new_data = fetch_historical_data(active_stocks, last_date, end_date)
    
    df = pd.concat([df, new_data]).drop_duplicates()
    df.to_csv(DATA_PATH)
    print("Historical data updated successfully!")
    logging.info("Historical data updated with latest market data.")

def main():
    """Main execution function."""
    update_historical_data()
    print("Stock trading system is ready.")

if __name__ == "__main__":
    main()