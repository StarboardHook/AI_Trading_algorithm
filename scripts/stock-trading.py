
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import alpaca
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream

from alpaca.data.requests import (
    CorporateActionsRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import (
    AssetExchange,
    AssetStatus,
    OrderClass,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)
API_KEY = ('AKWUHST4J3CGTKD3TPA9')
API_SECRET = ('qVjQoaphj3Q8qUoEfGNuXvq7dHjts2qMljhiNipt')
paper = True

data_api_url = None

stock_historical_data_client = StockHistoricalDataClient(API_KEY,API_SECRET, url_override = data_api_url)
now = datetime.now(ZoneInfo("America/New_York"))
req = StockBarsRequest(
    symbol_or_symbols=['TSLA,AAPL'],
    timeframe=TimeFrame.Day,
    start = now - timedelta(days=20),
    #end_date=None      #specify end datetime, default=now
    limit=100
)

df = stock_historical_data_client.get_stock_bars(req).df
print(df)