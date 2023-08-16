# silence warnings
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pyfolio as pf
import numpy as np
import matplotlib.pyplot as plt

aapl = yf.Ticker("AAPL")
history = aapl.history('max')
history.index = history.index.tz_convert('utc')
returns = history.Close.pct_change()


# Create returns tear sheet
data = pf.create_returns_tear_sheet(returns, live_start_date='2020-1-1')

