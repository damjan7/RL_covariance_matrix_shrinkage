# tests

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

# For time stamps
import datetime as dt


tickers = ['GOOG','AMZN','MSFT','AAPL', 'FB']


df = web.DataReader('GOOG', 'yahoo', start='2019-09-10', end='2019-10-09')


print(df)