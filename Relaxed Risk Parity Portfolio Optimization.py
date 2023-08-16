import pandas as pd
import yfinance as yf
import riskfolio as rp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format


start ="2021-01-01"
end ="2023-04-27"

#assets = ["LOMA.BA", "PAMP.BA", "TXAR.BA", "AGRO.BA", "MOLI.BA", "ALUA.BA"]
assets  = ["LOMA", "PAM", "TX", "AGRO.BA", "VIST"]
assets.sort()

# Downloading data
data = yf.download(assets, start = start, end = end)
data = data.loc[:,('Adj Close', slice(None))]
data.columns = assets


# Calculating returns

#assets = ["Renta_Capital", "Cobertura"]
#data = pd.read_excel(r'C:\Users\rumtl\PycharmProjects\Riskfolio\Rendimientos.xlsx', index_col='Date')
Y = data[assets].pct_change().dropna()
print(Y.tail())


# Calculating the vanilla risk parity portfolio.
# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio

# Select method and estimate input parameters:

method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:

model = 'Classic' # Could be Classic (historical) or FM (Factor Model)
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk free rate
b = None # Risk contribution constraints vector

w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)

print(w_rp.T)

# Calculating the relaxed risk parity portfolio version A.
b = None # Risk contribution constraints vector
version = 'A' # Could be A, B or C
l = 1 # Penalty term, only valid for C version

# Setting the return constraint
port.lowerret = 0.00056488 * 1.5

w_rrp_a = port.rrp_optimization(model=model, version=version, l=l, b=b, hist=hist)

print(w_rrp_a.T)

# Plotting portfolio composition
ax = rp.plot_pie(w=w_rrp_a, title='Relaxed Risk Parity A', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)
plt.show()

# Calculating the relaxed risk parity portfolio version B.
version = 'B' # Could be A, B or C

w_rrp_b = port.rrp_optimization(model=model, version=version, l=l, b=b, hist=hist)

print(w_rrp_b.T)


fig, ax = plt.subplots(figsize=(10,6))

# Plotting the risk composition of the portfolio
ax = rp.plot_risk_con(w_rp, cov=port.cov, returns=port.returns, rm=rm, rf=0, alpha=0.01,
                      color="tab:blue", height=6, width=10, ax=ax)
plt.show()

# Plotting equal risk contribution line
a1 = rp.Sharpe_Risk(w_rp, cov=port.cov, returns=port.returns, rm=rm, rf=0, alpha=0.01)
ax.axhline(y=a1/len(assets) * 252**0.5, color='r', linestyle='-')

# Plotting portfolio composition
ax = rp.plot_pie(w=w_rrp_b, title='Relaxed Risk Parity B', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)
plt.show()

# Plotting Risk Composition
fig, ax = plt.subplots(figsize=(10,6))

# Plotting the risk composition of the portfolio
ax = rp.plot_risk_con(w_rrp_b, cov=port.cov, returns=port.returns, rm=rm, rf=0, alpha=0.01,
                      color="tab:blue", height=6, width=10, ax=ax)
plt.show()

# Plotting equal risk contribution line
ax.axhline(y=a1/len(assets) * 252**0.5, color='r', linestyle='-')

plt.show()


# Calculating the relaxed risk parity portfolio version C.

version = 'C' # Could be A, B or C
w_rrp_c = port.rrp_optimization(model=model, version=version, l=l, b=b, hist=hist)
print(w_rrp_c.T)

# Plotting Risk Composition

fig, ax = plt.subplots(figsize=(10,6))

# Plotting the risk composition of the portfolio
ax = rp.plot_risk_con(w_rrp_c, cov=port.cov, returns=port.returns, rm=rm, rf=0, alpha=0.01,
                      color="tab:blue", height=6, width=10, ax=ax)
plt.show()

# Plotting equal risk contribution line
ax.axhline(y=a1/len(assets) * 252**0.5, color='r', linestyle='-')

plt.show()


# Reportes
#rp.excel_report(ret, w, name='report')
rp.jupyter_report(Y, w_rrp_a, bins=50, t_factor=365, days_per_year=365)
plt.savefig("Relaxed_Risk_parity_report_A.jpg")

rp.jupyter_report(Y, w_rrp_b, bins=50, t_factor=365, days_per_year=365)
plt.savefig("Relaxed_Risk_parity_report_B.jpg")



