# regressing to see which way obvious factor stocks load for sanity checks and 
# accuracy. 

import yfinance as yf
import statsmodels.api as sm

tickers = ["ACCO", "AAPL", "BRK-B", "MSFT", "V", "PLUG", "KO", "ORCL", "SPY"]
prices = yf.download(tickers, start='2016-01-01', end='2026-01-01', interval='1d')
prices = prices['Close']
returns = prices.pct_change()
returns_pct = returns * 100
returns_pct.describe()
returns_pct = returns_pct.dropna()







# Small Cap (ACCO) vs Big Cap 