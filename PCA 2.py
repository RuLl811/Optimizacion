from sklearn.decomposition import PCA
from pypfopt import risk_models, expected_returns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

start_date = '2020-01-01'
end_date = '2023-04-30'

tickers = ['GOOG', 'AMZN', 'TSLA', 'NFLX', 'AAPL', 'MSFT', 'NOK', 'BABA', 'NVDA', 'AMD', 'SNAP']

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close'].dropna()

monthly_prices = data.asfreq(freq='BM').ffill()

covariance_matrix_annualized = risk_models.sample_cov(monthly_prices, frequency = 12)  # Se ponene los precios

print(covariance_matrix_annualized)

# Analisis de Componentes principales

pca = PCA()
pca.fit(covariance_matrix_annualized)
pca_columns = [f'PC{i+1}' for i in range(pca.components_.T.shape[0])]
loadings = pd.DataFrame(pca.components_.T, columns=pca_columns, index=covariance_matrix_annualized.columns) # Matriz Z de PCA
print(loadings.applymap(lambda x: f"{x:.4f}"))

# Grafico
'''

fig = plt.figure(constrained_layout=True, figsize=(12, 8), dpi=100)
spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[2, 2],  height_ratios=[1])

# Varianza
ax1 = fig.add_subplot(spec[0, 0])
ax1.bar(loadings.columns, pca.explained_variance_, width=0.9, color='tab:blue', label="Varianza Explicada")
plt.grid(color='gray', linestyle='-.', linewidth=0.5, alpha=0.8)

for i, v in enumerate(pca.explained_variance_):
    ax1.annotate(f"{v:.4f}", xy=(i, v), xytext=(-7, 7), color='blue', fontweight='bold', textcoords='offset points', fontsize=12)

plt.xticks(fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.ylim(0, np.max(pca.explained_variance_) *1.3)
ax1.set_title('Varianza Explicada por Componente Principal', fontsize=14, fontweight='bold', wrap=True)

ax12 = ax1.twinx()
ax12.plot(loadings.columns, pca.explained_variance_.cumsum(), color='darkred', label="Varianza acumulada", marker='.', lw=1)
for i, v in enumerate(pca.explained_variance_.cumsum()):
    ax12.annotate(f"{v:.4f}", xy=(i, v), xytext=(-35, 7), color='red', textcoords='offset points', fontsize=12)

plt.grid(color='tab:red', lw=0.8, alpha=0.2, linestyle='-')
plt.xticks(fontsize=14)
plt.ylim(0, np.max(pca.explained_variance_.cumsum()) * 1.1)
plt.yticks(color='red')
plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.001, 0.95))

# Porporcion
ax2 = fig.add_subplot(spec[0, 1])
ax2.bar(loadings.columns, pca.explained_variance_ratio_, width=0.9, color='tab:gray', label="Proporcion de la Varianza Explicada")
plt.xticks(fontsize=14)
plt.grid(color='gray', linestyle='-.', linewidth=0.5, alpha=0.8)
for i, v in enumerate(pca.explained_variance_ratio_):
    ax2.annotate(f"{v:.4f}", xy=(i, v), xytext=(-7, 7), color='k', fontweight='bold', textcoords='offset points', fontsize=12)
plt.legend(fontsize=12, loc='upper left')
plt.ylim(0, np.max(pca.explained_variance_ratio_ * 1.3))
ax2.set_title('Proporcion de la Varianza \n en las Componentes Principales', fontsize=14, fontweight='bold', wrap=True)

ax22 = ax2.twinx()
ax22.plot(loadings.columns, pca.explained_variance_ratio_.cumsum(), color='darkred', label="Prop Acum de la Var", marker='.', lw=1)
for i, v in enumerate(pca.explained_variance_ratio_.cumsum()):
    ax22.annotate(f"{v:.4f}", xy=(i, v), xytext=(-35, 7), color='red', textcoords='offset points', fontsize=12)

plt.grid(color='tab:red', lw=0.8, alpha=0.2, linestyle='-')
plt.xticks(fontsize=14)
plt.ylim(0, np.max(pca.explained_variance_ratio_.cumsum() * 1.1))
plt.yticks(color='red')
plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.001, 0.95))
plt.tight_layout()
plt.show()
'''

# Modelo con todo el Mercado





# HPCA: https://gmarti.gitlab.io/qfin/2020/07/05/hierarchical-pca-avellaneda-paper.html