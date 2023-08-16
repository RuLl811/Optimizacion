# Importo librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from pypfopt import expected_returns
from pypfopt import risk_models

# Defino algunas funciones para emprolijar los datos
def format_df(df):
    df.columns = [ '_'.join(x) for x in df.columns ]
    df = df.loc[:, df.columns.str.endswith('_Close')]
    df.columns = df.columns.str.replace(r'_Close$', '')
    return df

tickers = ['GOOG', 'AMZN', 'TSLA', 'NFLX', 'MELI', 'ORCL']

start_date = '2015-01-01'
end_date = dt.datetime.now().date()

data = yf.download(
    tickers = tickers,
    interval = '1d',
    start = start_date,
    end = end_date
    )
data = data.loc[:, ('Adj Close', slice(None))]
data.columns = tickers

# Grafico las series de precios
''' 
fig, ax = plt.subplots(1,1, figsize = (12,7), tight_layout = True, dpi = 100)

plot_assets = tickers  #['AMZN','MELI','GOOG'] # esto era por si quieria ver menos assets de todos los que habiabajado

ax.set_title(f'Precios Diarios \n{start_date} - {end_date}', weight = 'bold', fontsize = 16)

ax.plot(data.loc[:, plot_assets], lw = 0.8, label = plot_assets)

# Si quisiera personalizar aun mas los graficos, algunos ejemplos de como hacerlo:
ax.plot(asset_data.loc[:,'AMZN'], lw = 0.5, color = 'darkorange', label = 'Amazon')
#ax.plot(asset_data.loc[:,'TSLA'], lw = 0.5, color = 'blue', label = 'Tesla')
ax.plot(asset_data.loc[:,'GOOG'], lw = 0.5, color = 'darkgreen', label = 'Google')
ax.plot(asset_data.loc[:,'MELI'], lw = 0.5, color = 'blue', label = 'Mercado Libre')


ax.semilogy()
ax.set_ylabel('Log Precio [USD]', fontsize=16)
ax.set_xlabel('Fecha [Días]', fontsize=16)
ax.grid()
ax.legend(ncol=1, fontsize=14)
plt.show()
'''

# Busco precios mensuales:
# Muestreo los precios diarios quedandome con el del ultimo dia habil de cada mes ("BM": "Business Month")

monthly_prices = data.asfreq(freq='BM').ffill()


#Vector de medias y matriz de covarianzas

# Calculo el vector de medias y matriz de covarianzas anualizadas.
# Como pase datos menusales, frequency = 12; si hubiera pasado datos diariosn, frequency = 252. Los resultados no tienen por que coincidir, ya que los datos diarios son mas ruidosos.
mean_returns_annualized = expected_returns.mean_historical_return(monthly_prices,frequency=12)
covariance_matrix_annualized = risk_models.sample_cov(monthly_prices, frequency = 12)

#print(covariance_matrix_annualized)

# Las volatilidades individuales son la varianza de cada activo y ellas se encuentran en la diagonal de la matriz de covarianzas. Las tomo de ahi directamente en vez de calcularlas una por una:
volatility_annualized = np.sqrt(np.diag(covariance_matrix_annualized))
volatility_annualized = pd.DataFrame(volatility_annualized, index = covariance_matrix_annualized.columns)
#print(volatility_annualized)

# Visualizacion de activos en el plano riesgo-rendimiento
'''
fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 100)

for i in range(len(tickers)):
    ax.plot(volatility_annualized.to_numpy()[i], mean_returns_annualized.to_numpy()[i],
               markersize = 10, marker="o", markeredgecolor= 'red', markerfacecolor= 'red')

    ax.text(x =volatility_annualized.to_numpy()[i] + 0.01, y = mean_returns_annualized[i] - 0.02, s = mean_returns_annualized.index[i],
             fontdict = dict(color = 'blue', alpha = 1, size = 14),
            bbox = dict(facecolor = 'orange', alpha = 0.25))



ax.set_xlim(0, 1.1 * np.max(volatility_annualized.to_numpy()))
ax.set_ylim(0, 1.1 * np.max(mean_returns_annualized.to_numpy()))


ax.set_ylabel('Rendimiento Esperado ($\mu$)', fontsize=16)
ax.set_xlabel('Volatilidad ($\sigma$)', fontsize=16)
ax.set_title('Espacio de Parámetros', weight='bold', fontsize=16)
ax.grid()

plt.show()
'''
# Portfolios

# Defino una funcion para obtener el riesgo y rendimiento deu n portafolio:
# Calculamos volatilidad y retorno de un portafolio
def portfolio_metrics(weights, mean_returns, cov_matrix):
    ret = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, std

# Defino una funcion para generar portafolios aleatorios, dado un vector de medias y una matriz de covarianzas
def random_portfolios(num_port, mean_returns, cov_matrix):
    metrics = np.zeros((2, num_port))
    weights_matrix = []

    for i in range(num_port):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_matrix.append(weights)
        port_mu, port_std = portfolio_metrics(weights, mean_returns, cov_matrix) # Invoco la primera funcion para calcular el retorno y el riesgo de los portafolios aleatorios.
        metrics[0,i] = port_mu
        metrics[1,i] = port_std
    return metrics, weights_matrix

# Simmulo portafolios aleatorios y almaceno sus metricas de desempenio y sus composiciones (por separeado)

# Número de portafolios
np.random.seed(123)
num_sim_portfolios = int(1e4) # Simulo 10000 portafolios

# Simulacion
metrics, weights_matrix = random_portfolios(num_port = num_sim_portfolios,
                                            mean_returns = mean_returns_annualized,
                                            cov_matrix = covariance_matrix_annualized)

# Optimizacion de portafolios

# Importo librerias
import cvxpy as cp

# Renombro las variables por comodidad:
mu = mean_returns_annualized
Sigma = covariance_matrix_annualized
n = len(mu)

# Calculo del portafolio de minima varianza global:

# Creo el vector de weights como variable de decision
w = cp.Variable(n)

# Creo el retorno del portafolio, pasando el vector de weights (variable de decision) y el vector de medias (dato)
ret = w.T @ mu   #mu.T @ w  # mu.T w    ==      w.T mu    (por ser escalar)

# Creo el riesgo del portafolio como una forma cuadratica, pasando el vector de weights (variable de decision) y la matriz de covarianzas (dato)
risk = cp.quad_form(w, Sigma)   # wT Sigma w

# Especifico las restricciones como una lista de condiciones booleanas
constraints = [cp.sum(w) == 1,  # Fully invested
               w >= 0]          # Long-Only

# Creo una instancia del porblema de optimizacion, pasando el objetivo, y las restricciones:
prob = cp.Problem(cp.Minimize(risk), constraints) # Primero es el objetivo, despues la funcion objetivo y luego las restricciones

# Resuelvo el problema
print(prob.solve()) # Esto me arroja el objetivo de varianza
print('\n')


# Convierto el modelo en una funcion que toma como parametros un vector de retornos esperados, una matriz de covarianzas, y un rendimiento minimo admisible (ya no estoy en el caso de minima varianza global):

def optimize_portfolio(mean_returns_vec, covariance_matrix, base_portfolio_return):

    mu = mean_returns_vec
    Sigma = covariance_matrix
    n = len(mu)

    w = cp.Variable(n)

    ret = w.T @ mu   #mu.T @ w  # mu.T w    ==      w.T mu    (por ser escalar)
    risk = cp.quad_form(w, Sigma)   # wT Sigma w


    # Si pido minima varianza global, entonces es el problemade antes
    if base_portfolio_return == 'min_global_variance':
        constraints = [cp.sum(w) == 1,
                       w>=0]

    # por elcontrario, si pido un rendimiento minimo admisible, lo agrego como restriccion
    else:
        constraints = [ret >= base_portfolio_return,
                       cp.sum(w) == 1,
                       w>=0]

    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()

    # Una vez que resuelvo el problema, guardo las variables relevantes en un diccionario por comodidad.
    return {"portfolio_risk": risk.value,
            "portfolio_return":ret.value,
            "weights": pd.DataFrame(w.value, index = mean_returns_vec.index).apply(lambda r: np.where(r <= 1e-06, 0, r))} # el np.where es como un if, si r es menor o igual a 1e-06, entonces pongo 0, sino pongo r

# Pruebo la funcion con el portafolio de minima varianza global
min_global_variance_solution = optimize_portfolio(mean_returns_annualized, covariance_matrix_annualized, base_portfolio_return = 'min_global_variance')
print("Optimizacion de portafolio con min varianza global:")
print(f'Rendimiento esperado: {min_global_variance_solution["portfolio_return"]:.4f}')
print(f'Riesgo esperado: {min_global_variance_solution["portfolio_risk"]:.4f}')
# Darle formato de 4 decimales a los numeros de weights
min_global_variance_solution["weights"] = min_global_variance_solution["weights"].applymap(lambda r: "{:.4f}".format(r))
print(f'Los weights son: {min_global_variance_solution["weights"]}')
print('\n')

# Ejemplo con otro rendimiento esperado
rendimiento_minimo_exigido = 0.2   # quizas no para todo rendimiento minimo encuentre una solucion para el vector de media y matriz de covarianza que hayamos pasado

optimal_portfolio = optimize_portfolio(mean_returns_annualized, covariance_matrix_annualized, base_portfolio_return = rendimiento_minimo_exigido)

print('\n')
print("Optimizacion con rendimiento esperado:")
print(f'Desvio estandar esperado: {np.sqrt(optimal_portfolio["portfolio_risk"]):.4f}')  # La raiz cuadrada del riesgo esperado, es el desvio estandar esperado
print(f'Retorno esperado: {optimal_portfolio["portfolio_return"]:.4f}')
print(optimal_portfolio["weights"])

# Frontera eficiente

# Preparamos los datos para trazar una frontera eficiente

# Paso con el que voy a incrementar el rendimiento minimo exigido
step = 0.005
# Punto de comienzo de rendimiento para comenzar a trazar: el rendimiento del portafoliode minima varianza global, NO CERO!
min_return = min_global_variance_solution['portfolio_return']
# Punto para finalizar la traza: el rendimiento mas alto disponible en el mercado
max_return = np.max(mean_returns_annualized)

# Cantidad de puntos usados para trazar la frontera (sera la cantidad de portafolios optimos a resolver)
num_efficient_points = 60  # Es la cantidad de puntos del scatterplot

# Creo un vector de rendimientos exigidos para barrer la frontera
mu_range = np.linspace(start=min_return, stop=max_return, num=num_efficient_points, endpoint=True)

# Creo estructuras vacias para ir guardando los renimientos minimos exigidos y las volatilidades optimizadas
efficient_risk = np.zeros(num_efficient_points)
efficient_return = np.zeros(num_efficient_points)

# Calculo de la frontera eficiente:
# Voy barriendo los rendimientosm inimos, resolviendo el probelma de optimizacion, y guardando las coordenadas optimas:
for i in range(num_efficient_points):
    solution = optimize_portfolio(mean_returns_annualized, covariance_matrix_annualized, base_portfolio_return = mu_range[i]) # a diferencia delo anterior recorre los rendimientos minimos y maximos
    efficient_risk[i] = np.sqrt(solution["portfolio_risk"])
    efficient_return[i] = solution["portfolio_return"]


# Visualizacion de la frontera eficiente sobre el conjunto de lso portafolios simulados, y los otros activos individuales:
'''
fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 100)

ax.plot(metrics[1,:], metrics[0,:], 'o', markersize = 2,  alpha = 0.1, color = 'lightgray')

ax.plot(volatility_annualized.to_numpy(), mean_returns_annualized.to_numpy(), "o",markersize = 8, markeredgecolor= 'darkred', markerfacecolor= 'red', label = 'Activos Individuales')

for i in range(len(tickers)):

    ax.text(x = volatility_annualized.to_numpy()[i] + 0.01, y = mean_returns_annualized[i] - 0.01, s = mean_returns_annualized.index[i],
            fontdict = dict(color = 'blue', alpha = 1, size = 14),
            bbox = dict(facecolor = 'orange', alpha = 0.25))


ax.set_xlim(0.95 * np.min(min_global_variance_solution["portfolio_risk"]), 1.1 * np.max(volatility_annualized.to_numpy()))
ax.set_ylim(0.8 * np.min(mean_returns_annualized), 1.1 * np.max(mean_returns_annualized))

ax.plot(efficient_risk, efficient_return, 'o-', markersize = 1, color = 'blue', lw = 0.5, label = 'Frontera Eficiente')

ax.plot(np.sqrt(min_global_variance_solution["portfolio_risk"]), min_global_variance_solution['portfolio_return'], 's', markersize = 8, color = 'darkgreen', label = 'Portafolio de Mínima Varianza Global')

ax.legend(loc = 'upper left')
ax.set_ylabel('Rendimiento Esperado ($\mu$)', fontsize = 16)
ax.set_xlabel('Volatilidad ($\sigma$)', fontsize = 16)
ax.set_title('Frontera de Portafolios Eficientes', weight = 'bold', fontsize = 16)
ax.grid()
plt.show()
'''
# Optimizacion con ratio de sharpe

# Busco el portafolio de maximo ratio de Sharpe

# Importo librerias:
import scipy.optimize as sco
from scipy import stats

# Defino funciones auxiliares:
## Calculo de rendimiento, volatilidad, y ratio de Sharpe
rf, mu, covmat = 0.01, mu, Sigma

def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(np.array(mu) * weights)
    vol = np.sqrt(np.dot(weights.T,np.dot(np.array(covmat),weights)))
    sr = (ret-rf)/vol
    return np.array([ret,vol,sr])

# Voy a minimizar el menos-ratio de sharpe, esto lo hago porque scipy no tiene maximizacion...
def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1

# Restriccion fully invested: suma de ponderaciones = 1 (booleano)
def check_sum(weights):
    return np.sum(weights) - 1

# Creo las restricciones del modelo:
## Una tupla con diccionarios, donde cada diccionario es una restriccion:
## - primera componente: si es de igualdad (eq) o de desigualdad (ineq)
## - segunda compnente: una funcion que representa la restricccion. Debe ser una funcion que devuelve valores booleanos.
cons = ({'type':'eq','fun':check_sum})

# Creo cotas para los weights, en este caso,entre 0 y 1 para cada weight
bounds = ((0, 1),) * len(mu)

# Puedo comenzar sugiriendo una solucion , para asistir el optimizador. En ese caso, sugeri que comience de un portafolio de ponderaciones uniformes ("equally weighted")
init_guess = [1/len(mu)] * len(mu)

# Resuelvo el problema
opt_results = sco.minimize(neg_sharpe, init_guess,
                           method='SLSQP', bounds=bounds, constraints=cons)

sharpe_weights = pd.DataFrame(opt_results['x'].T, index = mu.index).apply(lambda r: np.where(r<=1e-6,0,r))

sharpe_return, sharpe_risk = portfolio_metrics(opt_results['x'].T, mu, Sigma)

max_sr = -opt_results['fun'] # Revierto el negativo que puse en la optimizacion
print('\n')
print("Optimizacion con Ratio de Sharpe:")
print('\n')
print(f"Rendimiento: {sharpe_return}")
print(f"Riesgo: {sharpe_risk}")
print("Ratio de Sharpe:", max_sr)
print(sharpe_weights)

# Grafico la frontera eficiente, losp ortafolios aleatorios, los activos individuales, el protafolio de minima varianza global, y el portafolio de maximoratio de Sharpe:

## Ademas, cada punto del plano, que es un portafolio, sera coloreado de acuerdo a su propio ratio de sharpe (excepto el de MVG y los activos individuales)
'''
fig, ax = plt.subplots(1,1, figsize = (12,7), dpi = 100)

sharpe = ax.scatter(metrics[1,:], metrics[0,:], s = 5, c=metrics[0,:]/metrics[1,:], cmap ='RdYlBu',  alpha = 0.5) #, color = 'lightgray'  'o', markersize = 2,
cbar = plt.colorbar(sharpe, pad = 0.02)
cbar.set_label('Sharpe Ratio')

ax.plot(volatility_annualized.to_numpy(), mean_returns_annualized.to_numpy(), "o",markersize = 8, markeredgecolor= 'darkred', markerfacecolor= 'red', label = 'Activos Individuales')

for i in range(len(tickers)):

    ax.text(x = volatility_annualized.to_numpy()[i] + 0.01, y = mean_returns_annualized[i] - 0.01, s = mean_returns_annualized.index[i],
            fontdict = dict(color = 'blue', alpha = 1, size = 14),
            bbox = dict(facecolor = 'orange', alpha = 0.25))


ax.set_xlim(0.95 * np.min(min_global_variance_solution["portfolio_risk"]), 1.1 * np.max(volatility_annualized.to_numpy()))
ax.set_ylim(0.8 * np.min(mean_returns_annualized), 1.1 * np.max(mean_returns_annualized))

ax.plot(efficient_risk, efficient_return, 'o-', markersize = 1.5, color = 'blue', lw = 1, label = 'Frontera Eficiente')

ax.plot(np.sqrt(min_global_variance_solution["portfolio_risk"]), min_global_variance_solution['portfolio_return'], 's', markersize = 8, color = 'darkgreen', label = 'Portafolio de Mínima Varianza Global')

ax.plot(sharpe_risk, sharpe_return, 's', markersize = 8, color = 'red', label = f'Portafolio de Maximo Ratio de Sharpe: {max_sr:.4f}')

ax.legend(loc = 'upper left')
ax.set_ylabel('Rendimiento Esperado ($\mu$)', fontsize=16)
ax.set_xlabel('Volatilidad ($\sigma$)', fontsize=16)
ax.set_title('Frontera de Portafolios Eficientes', weight='bold', fontsize=16)
ax.grid()
plt.show()
'''

#############################################################################################################


# Repetir analisis con componentes del SP 500

# Defino 2 funciones auximilares de forma un poco mas prolija para: 1) calcular rendimientos; y 2) calcular media y covarianzas.
def return_fn(prices, resample_freq=None, type="log", since=None):
    if since is None:
        prices_ = prices
    else:
        prices_ = prices.loc[since:]

    if resample_freq is not None:
        prices_ = prices_.asfreq(resample_freq).ffill()
        #(resample_freq).last()

    if type == 'log':
        returns = prices_.apply(np.log).diff()
    else:
        returns = prices_.pct_change()

    returns.dropna(inplace = True)
    return returns

def mean_var_estimates(returns, freq = None):
    if freq is not None:
        mean_estimate = np.expm1(returns.mean()*freq)
        cov_estimate = returns.cov()*freq
    else:
        mean_estimate = np.expm1(returns.mean())
        cov_estimate = returns.cov()
    return mean_estimate, cov_estimate

# Bajo componentes del SP 500
np.random.seed(123)

# definimos el universo de activos para trabajar
#tickers = ['GOOG','AMZN','TSLA','NFLX','MELI']

import ssl
import urllib.request

# Deshabilitar verificación de certificado SSL
ssl._create_default_https_context = ssl._create_unverified_context

# leemos las tablas disponibles
tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
# seleccionamos la primera
spx_components = tables[0]
# fijamos el index
spx_components = spx_components.set_index('Symbol')
# reemplazamos para poder descargar los instrumentos desde Yahoo! Finance
spx_components['YF Tickers'] = spx_components.index.str.replace('.', '-', regex = False)
# componemos una lista, ordenados alfabéticamente, de forma ascendente
tickers_list = spx_components[['YF Tickers', 'GICS Sector']]

# selección aleatoria sin repetición: un activo por sector, para tener una muestra balanceada
tickers_sample = tickers_list.groupby('GICS Sector').sample(n = 1, random_state = 1)

# Me quedo con una lista de esos tickers
tickers_sample = tickers_sample['YF Tickers'].to_list()

# Descargo la serie de precios de esos tickers:
prices = yf.download(
    tickers = tickers_sample,
    interval = '1d',
    start = '2015-01-01',
    end = '2023-04-30').loc[:, ('Adj Close', slice(None))]

# Obtener solo los nombres de las acciones
nombres_acciones = prices.columns.get_level_values(1)
prices.columns = nombres_acciones

# Visualizamos la evolución de precios
'''
plt.figure(figsize=(12, 7), dpi = 100)

for i in prices.columns.values:
    plt.plot(prices.index, np.log(prices[i]), lw=0.7, alpha=0.8,label=i)

plt.legend(loc='upper left', fontsize=12)
plt.grid()
plt.ylabel('Precios logarítmicos')
plt.show()
'''
# Calculo los retornos
returns = return_fn(prices, resample_freq='BM', type = 'discrete')
#returns.plot()
#plt.show()
# Cambiamos la frecuencia a mensual
monthly_prices = prices.asfreq('BM').ffill()
monthly_returns = monthly_prices.pct_change()

# Estimamos vector de medias y volatilidades anualizadas
df_sample = pd.DataFrame()
df_sample['Mu'] = mean_var_estimates(returns = monthly_returns, freq = 12)[0]
df_sample['Sigma'] = np.sqrt(np.diag(mean_var_estimates(returns = monthly_returns, freq = 12)[1]))
cov_matrix = mean_var_estimates(returns = monthly_returns, freq = 12)[1]

# Visualizacion de estas componentes del SP en el plano riesgo-retorno
'''
fig, ax = plt.subplots(figsize = (12,7), dpi = 100, tight_layout = True)

#df_sample.plot(x='Sigma', y = 'Mu', kind = 'scatter', ax = ax)

plt.plot(df_sample.Sigma, df_sample.Mu, "o", markersize=8, markeredgecolor='darkred', markerfacecolor= 'red')

plt.title('Mean-Variance Analysis')
plt.ylabel('Retorno esperado')
plt.xlabel('Volatilidad')
plt.grid()

for k, v in df_sample.iterrows():
    ax.annotate(k, v[::-1], fontsize=12)
'''

#############################################################################################################

# Modelos Factoriales

# Adopto como indice de mercado al SP 500 y bajo su cotizacion (ETF: SPY) como proy de portafoliode mercado:

start_date = '2015-01-01'   # Tiene que coincidir con el periodo de mis datos
end_date = '2023-04-30' # dt.datetime.now()

SPY = yf.download(
    tickers = 'SPY',
    interval = '1d',
    start = start_date,
    end = end_date,
    group_by = 'ticker',
    auto_adjust = True,
    prepost = True,
    threads = True,
    )


SPY = SPY['Close'].copy()
SPY.head()


monthly_SPY = SPY.asfreq('BM').ffill()
monthly_ret_SPY = monthly_SPY.pct_change().dropna()
monthly_ret_SPY.name = "SPY"

# Descargo mi activo libre de riesgo: BIL

start_date = '2015-01-01'   # Tiene que coincidir con el periodo de mis datos
end_date = '2023-04-30'  # dt.datetime.now()

BIL = yf.download(
    tickers = 'BIL',
    interval = '1d',
    start = start_date,
    end = end_date,
    group_by = 'ticker',
    auto_adjust = True,
    prepost = True,
    threads = True,
    )


BIL = BIL['Close'].copy()
monthly_BIL = BIL.asfreq('BM').ffill()
monthly_ret_BIL = monthly_BIL.pct_change().dropna()
monthly_ret_BIL.name = "BIL"

total = pd.concat((monthly_returns, monthly_ret_SPY, monthly_ret_BIL), axis =1).dropna()

# Calculamos los betas con su formula cerrada (
def beta_function(asset, factor):
    cov_mat = pd.concat((asset, factor), axis = 1).cov()
    beta = cov_mat.iloc[0, 1]/(cov_mat.iloc[1, 1])
    return beta

# Ejemplo de beta para un activo en particular
asset_name = 'IVZ'
beta_particular = beta_function(total[asset_name], total.SPY)
print(f"Beta de {asset_name} contra el {total.SPY.name} es: {beta_particular}.")

# Varianzas y covarianzas contra el mercado:
asset_name = 'PPL'
print(pd.concat((total[asset_name], total["SPY"]), axis = 1).cov())

# la regresion es estadisticamente significativa?

from statsmodels import regression
import statsmodels.api as sm

asset_name = 'PPL'
y = total[asset_name]-total.BIL
X = sm.add_constant(total.SPY - total.BIL)
CAPM_regression = regression.linear_model.OLS(y, X).fit()
CAPM_beta = CAPM_regression.params.iloc[1]

print(CAPM_regression.summary())

# Cp de Mallow
'''
El valor del estadístico Cp de Mallow se utiliza para evaluar la capacidad predictiva del modelo.
Un valor de Cp cercano a p indica un buen ajuste del modelo, mientras que un valor de Cp mucho mayor que p
indica un sobreajuste del modelo.

Dado que el coeficiente Cp de Mallow es menor que la cantidad de variables predictoras en el modelo
(en este caso, 1), esto indica un buen ajuste del modelo. Un valor de Cp cercano o inferior al número de variables 
predictoras sugiere que el modelo es capaz de capturar la variabilidad de los datos y hacer predicciones razonables.
'''
N = len(y)  # Cant de observaciones
p = len(X.columns)      # Cant de covariables (intercept + UNA covariable)
RSS = CAPM_regression.ssr
s2 = (1/(N-p-1)) * RSS

#Computing
Cp = (1/N) * (RSS + 2 * p * s2)
print(Cp)

# Rolling beta

total['beta'] = total[asset_name].rolling(12).cov(total['SPY']) / total['SPY'].rolling(12).var()
'''
fig, ax = plt.subplots(figsize = (15, 5), dpi = 100)
data_line = ax.plot(total.index, total.beta, lw = 0.7, label=f'Beta de {asset_name}', marker='o')
mean_line = ax.plot(total.index, [np.mean(total.beta)]*len(total.beta), label='Avg', linestyle='--')

# Make a legend
legend = ax.legend(loc='upper right')
plt.title("Rolling beta")
plt.grid()
plt.show()
'''
forecast = total.BIL + total.beta*(total.SPY - total.BIL) # CAPM equation
'''
fig, ax = plt.subplots(figsize = (15,6), dpi = 100)

forecast.plot(lw = 0.7, color='blue', label ="CAPM Predicted Return")
total[asset_name].plot(color='darkgreen', lw = 0.7, label = 'True Return')
#R.plot(color='Y')
plt.legend()

plt.ylabel('Retorno mensual')
plt.grid()
plt.show()
'''

# Security Market Line

risk_free_rate = np.mean(total.BIL)

# We have two coordinates that we use to map the SML: (0, risk-free rate) and (1, market return)
'''
eqn_of_the_line = lambda x : ( (np.mean(total.SPY)-risk_free_rate) / 1.0) * x + risk_free_rate
xrange = np.linspace(0., 2.5, num=2)

plt.figure(figsize=(16,5))
plt.plot(xrange, [eqn_of_the_line(x) for x in xrange], color='red', linestyle='--', linewidth=2, label = 'Security Market Line' )

plt.plot([1], [np.mean(total.SPY)], marker='o', color='navy', markersize=10)
plt.annotate('Market', xy=(1, np.mean(total.SPY)), xytext=(0.9, np.mean(total.SPY)+0.00004))

#monthly_returns.columns = map(lambda x: x.symbol, monthly_returns.columns)

betas = [
    regression.linear_model.OLS(total[asset]-risk_free_rate,
                                sm.add_constant(total.SPY-risk_free_rate)).fit().params[1]
    for asset in monthly_returns.columns]

for asset, beta in zip(monthly_returns, betas):
    plt.plot([beta], [np.mean(total[asset])], marker='o', color='g', markersize=10)
    plt.annotate(
        asset,
        xy=(beta, np.mean(total[asset])),
        xytext=(beta + 0.015, np.mean(total[asset]) + 0.000025)
    )

plt.title("Security Market Line")
plt.xlabel("Beta")
plt.ylabel("Retorno esperado")
plt.legend()
plt.grid()

plt.show()
'''

# Modelos Factores: 3 y 5 de Fama French

# Importo librerias a utilizar
from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web

# Estos son los datasets disponibles, asociados a los distintos mdoelos y a distitnas frecuencias
#print(get_available_datasets())

# Descargo la serie de factores diarios del modelo de 5 factores, desde el comienzo del 2015.

factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench',start="2015")

factors = factors[0]

# Como los datos son diarios, debo acumularlos para muestrear mensualmente (no puedo simplemente muestrear retornos en otroa frecuencia, como si puedo hacerlo con precios!)
factors_monthly = factors.add(100).div(100).cumprod().resample('M').last().pct_change()
factors_monthly.columns = ['Market', 'SMB', 'HML', 'RMW', 'CMA', 'rf']
factors_monthly = factors_monthly.dropna()

# Como vamos a predecir los retornos mediante factores, removemos el ultimo retorno observado, pero dejamos la ultima observacion de factores

print(monthly_returns[:-1].tail())
print(factors_monthly.tail())

# Creo un dataframe nuevo con el valor de los retornos de los activos completo excepto el ultimo registro (que vamos a predecir con factores)
monthly_returns_factores = total[:-1]
print(monthly_returns_factores)

# Combino los dtafarames de facotres hasta el ultimo y el de retornos hasta el anteultimo
monthly_returns_factores = pd.concat([monthly_returns_factores, factors_monthly], axis = 1)

# Visualizo el retorno acumulado por factor
'''
fig, ax = plt.subplots(figsize = (15,5), dpi = 100)
((factors_monthly + 1).cumprod() - 1).plot(subplots=False,
                                                title='Cumulative Returns for Fundamental Factors',
                                                lw = 0.7, label = True, ax = ax)
plt.grid()
plt.show()
'''

# Estimacion de sensibilidades
# Creo la matriz de disenio (paral a regresion lineal
monthly_returns_factores = monthly_returns_factores.dropna()
X = sm.add_constant(monthly_returns_factores[['Market', 'SMB', 'HML', 'RMW', 'CMA']])

# Calculo la regresion y su reporte
# es fundamental que X e y tengan el mismo indice
asset_name = 'PPL'
y = monthly_returns_factores[asset_name].dropna()
results = sm.OLS(y, X).fit()
print(results.summary())

# Volvemos a definir la función para calcular las sensibilidades (betas)
# factor exposure
Betas = pd.DataFrame(index=total[monthly_returns.columns].columns, dtype=np.float32) # Voy a guardar las betas de cada activo para cada factor
epsilon = pd.DataFrame(index=total.index[:-1], dtype=np.float32) # voy a guardar el retorno idiosincratico en cada instante (los residuos de la regresion)

assets = monthly_returns.columns

for i in assets:
    y = monthly_returns_factores.loc[:,i]
    y_inlier = y[np.abs(y - y.mean())<=(3*y.std())]
    x_inlier = X[np.abs(y - y.mean())<=(3*y.std())]
    result = sm.OLS(y_inlier, x_inlier).fit()

    Betas.loc[i, "MKT_beta"] = result.params[1]
    Betas.loc[i, "SMB_beta"] = result.params[2]
    Betas.loc[i, "HML_beta"] = result.params[3]
    Betas.loc[i, "RMW_beta"] = result.params[4]
    Betas.loc[i, "CMA_beta"] = result.params[5]

    epsilon.loc[:, i] = y - (X.iloc[:, 0] * result.params[0] +
                            X.iloc[:, 1] * result.params[1] +
                            X.iloc[:, 2] * result.params[2] +
                            X.iloc[:, 3] * result.params[3] +
                            X.iloc[:, 4] * result.params[4] +
                            X.iloc[:, 5] * result.params[5])

Retornos = total[monthly_returns.columns][:-1]
Factores = monthly_returns_factores[['Market', 'SMB', 'HML','RMW', 'CMA']]

w = np.ones([1,Retornos.shape[1]])/Retornos.shape[1]

def compute_common_factor_variance(factors, factor_exposures, w):
    B = np.asarray(factor_exposures)
    F = np.asarray(factors)
    Cov_Factores = np.asarray(factors.cov())

    return w @ B @ Cov_Factores @ B.T @ w.T

common_factor_variance = compute_common_factor_variance(Factores, Betas, w)[0][0]
print(f"Common Factor Variance: {common_factor_variance:.6f}")

def compute_specific_variance(epsilon, w):

    D = np.diag(np.asarray(epsilon.var())) * epsilon.shape[0] / (epsilon.shape[0]-1)

    return w.dot(D).dot(w.T)

specific_variance = compute_specific_variance(epsilon, w)[0][0]
print(f"Specific Variance: {specific_variance:.6f}")

common_factor_pct = common_factor_variance / (common_factor_variance + specific_variance) * 100.0
print(f"Percentage of Portfolio Variance Due to Common Factor Risk: {common_factor_pct:.4f}%")
print()
specific_factor_pct = specific_variance / (common_factor_variance + specific_variance) * 100.0
print(f"Percentage of Portfolio Variance Due to Specific Risk: {specific_factor_pct:.4f}%")



