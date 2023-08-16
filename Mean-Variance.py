import pandas as pd
import yfinance as yf
import riskfolio as rp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

start ="2021-01-01"
end ="2023-04-27"

#assets = ["LOMA.BA", "PAMP.BA", "TXAR.BA", "AGRO.BA", "MOLI.BA", "ALUA.BA"]
assets  = ["LOMA", "PAM", "TX", "AGRO.BA", "VIST"]
assets.sort()

# Downloading data
data = yf.download(assets, start = start, end = end)
data = data.loc[:, ('Adj Close', slice(None))]
data.columns = assets

Y = data[assets].pct_change().dropna()

#assets = ["Renta_Capital", "Cobertura"]
#data = pd.read_excel(r'C:\Users\rumtl\PycharmProjects\Riskfolio\Rendimientos.xlsx', index_col='Date')
Y = data[assets].pct_change().dropna()
print(Y.tail())

# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio

# Select method and estimate input parameters:

method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:

model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk free rate
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

print(w.T)

# Plotting the composition of the portfolio

ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)

plt.show()


# Calculating the portfolio that maximizes Return/CVaR ratio.
rm = 'CVaR' # Risk measure

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

ax = rp.plot_pie(w=w, title='Sharpe Mean CVaR', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)

plt.show()

rp.jupyter_report(Y, w, bins=50, t_factor=365, days_per_year=365)
plt.savefig("MV_report_local.jpg")