import pandas as pd
import yfinance as yf
import riskfolio as rp
import matplotlib.pyplot as plt
from riskfolio import Sharpe_Risk

import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format

start ="2021-01-01"
end = "2023-04-27"
#assets = ["LOMA.BA", "PAMP.BA", "TXAR.BA", "AGRO.BA", "MOLI.BA", "ALUA.BA"]
assets  = ["LOMA", "PAM", "TX", "AGRO.BA", "VIST"]
assets.sort()

# Downloading data
data = yf.download(assets, start = start, end = end)
data = data.loc[:,('Adj Close', slice(None))]
data.columns = assets
print(data)
Y = data[assets].pct_change().dropna()
# Calculating returns

# Plotting Assets Clusters

ax = rp.plot_dendrogram(returns=Y,
                        codependence='pearson',
                        linkage='single',
                        k=None,
                        max_k=10,
                        leaf_order=True,
                        ax=None)
plt.show()
# Calculating the HRP portfolio

# Building the portfolio object
port = rp.HCPortfolio(returns=Y)

# Estimate optimal portfolio:

model='HRP' # Could be HRP or HERC
codependence = 'pearson' # Correlation matrix used to group assets in clusters
rm = 'MV' # Risk measure used, this time will be variance
rf = 0 # Risk free rate
linkage = 'single' # Linkage method used to build clusters
max_k = 10 # Max number of clusters used in two difference gap statistic, only for HERC model
leaf_order = True # Consider optimal order of leafs in dendrogram

w = port.optimization(model=model,
                      codependence=codependence,
                      rm=rm,
                      rf=rf,
                      linkage=linkage,
                      max_k=max_k,
                      leaf_order=leaf_order)

print(w.T)

# Plotting portfolio composition

ax = rp.plot_pie(w=w,
                 title='HRP Naive Risk Parity',
                 others=0.05,
                 nrow=25,
                 cmap="tab20",
                 height=8,
                 width=10,
                 ax=None)
plt.show()

# Plotting the risk contribution per asset

mu = Y.mean()
cov = Y.cov() # Covariance matrix
returns = Y # Returns of the assets

ax = rp.plot_risk_con(w=w,
                      cov=cov,
                      returns=returns,
                      rm=rm,
                      rf=0,
                      alpha=0.05,
                      color="tab:blue",
                      height=6,
                      width=10,
                      t_factor=252,
                      ax=None)

plt.show()

# Reportes
#rp.excel_report(ret, w, name='report')
rp.jupyter_report(Y, w, bins=50, t_factor=365, days_per_year=365)
plt.savefig("HRP_report_local.jpg")
