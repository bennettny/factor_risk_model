# regressing to see which way obvious factor stocks load for sanity checks and 
# accuracy. 
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import numpy as np

tickers = ["ACCO", "AAPL", "BRK-B", "MSFT", "V", "PLUG", "KO", "ORCL", "SPY"]
prices = yf.download(tickers, start='2016-01-01', end='2026-01-01', interval='1d')
prices = prices['Close']
returns = prices.pct_change()
returns_pct = returns * 100
returns_pct.describe()
returns_pct = returns_pct.dropna()
fffive = pd.read_csv('famafrench.csv',skiprows=35)
fffive['Date'] = pd.to_datetime(fffive['Date'],format='%Y%m%d')
data = pd.merge(returns_pct,fffive, left_index=True,right_on='Date')
data = data.set_index('Date')
data[tickers] = data[tickers].sub(data['RF'],axis='index') #returns above RF

dt = {}
dt_res = {}

for t in tickers:
    model = sm.OLS(data[t], exog=sm.add_constant(data[['Mkt-RF','SMB','HML','RMW','CMA']]))
    model_fit = model.fit(cov_type='HC3')
    dt[t] = model_fit.params
    dt_res[t] = model_fit.resid.var()

betas = pd.DataFrame(dt).T
betas = betas.drop(columns='const')
variances = pd.Series(dt_res)

cov_matrix = data[['Mkt-RF','SMB','HML','RMW','CMA']].cov()
corr_matrix = data[['Mkt-RF','SMB','HML','RMW','CMA']].corr()
diagonal_variance_matrix = np.diag(variances)
weights = pd.Series(1/8, index=tickers) #equal weighted
weights['SPY'] = 0 #sanity check

exposure = weights @ betas 
factor_variance = exposure @ cov_matrix @ exposure
specific_risk = weights @ diagonal_variance_matrix @ weights
total_variance = factor_variance + specific_risk
annual_vol = np.sqrt(factor_variance+specific_risk)*np.sqrt(252)
factor_share = factor_variance / total_variance
individual = exposure * (cov_matrix @ exposure)
individual_share = individual / total_variance

daily_port_returns = data[tickers] @ weights
std_port_returns = daily_port_returns.std() * np.sqrt(252)

print(annual_vol)
print(std_port_returns)








