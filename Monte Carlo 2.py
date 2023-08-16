import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import random as rd
from sklearn.model_selection import train_test_split

colorlist = [
    '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff', '#ff00ff', '#ff8000',
    '#8000ff', '#008000', '#000080', '#800000', '#008080', '#808000', '#800080',
    '#808080', '#ff8080', '#80ff80', '#8080ff', '#ffff80', '#80ffff', '#ff80ff'
]


def monte_carlo(data, test_size=0.5, simulation=1000, **kwargs):
    df, test = train_test_split(data, test_size=test_size, shuffle=False, **kwargs) # se divide el conjunto de datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split
    forecast_horizon = len(test)
    df = df[['Close']]
    returns = np.log(df['Close'].iloc[1:] / df['Close'].shift(1).iloc[1:])
    drift = returns.mean() - returns.var() / 2 # Se calcula el "drift" (deriva) de los retornos utilizando la media y la varianza.
    simulations = {}

    # we use geometric brownian motion to compute the next pric
    for counter in range(simulation):
        simulations[counter] = [df['Close'].iloc[0]]

        for i in range(len(df) + forecast_horizon - 1):
            # generate a pseudo random number using standard normal distribution
            sde = drift + returns.std() * rd.gauss(0, 1)
            temp = simulations[counter][-1] * np.exp(sde)
            simulations[counter].append(temp.item())

    std = float('inf')
    pick = 0
    for counter in range(simulation):
        temp = np.std(np.subtract(simulations[counter][:len(df)], df['Close']))
        if temp < std:
            std = temp
            pick = counter

    return forecast_horizon, simulations, pick   # La función monte_carlo devuelve el horizonte de pronóstico (forecast_horizon), todas las predicciones generadas (simulations) y la predicción seleccionada con menor desviación estándar (pick).


def plot(df, forecast_horizon, simulations, pick, ticker):
    ax = plt.figure(figsize=(10, 5)).add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i in range(len(simulations)):
        if i != pick:
            ax.plot(df.index[:len(df) - forecast_horizon], simulations[i][:len(df) - forecast_horizon], alpha=0.05)

    ax.plot(df.index[:len(df) - forecast_horizon], simulations[pick][:len(df) - forecast_horizon],
            c='#5398d9', linewidth=5, label='Best Fitted')

    df['Close'].iloc[:len(df) - forecast_horizon].plot(c='#d75b66', linewidth=5, label='Actual')

    plt.title(f'Monte Carlo Simulation\nTicker: {ticker}')
    plt.legend(loc=0)
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.show()

    ax = plt.figure(figsize=(10, 5)).add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(simulations[pick], label='Best Fitted', c='#edd170')
    plt.plot(df['Close'].tolist(), label='Actual', c='#02231c')
    plt.axvline(len(df) - forecast_horizon, linestyle=':', c='k')

    plt.text(len(df) - forecast_horizon - 50, max(max(df['Close']), max(simulations[pick])), 'Training',
             horizontalalignment='center', verticalalignment='center')
    plt.text(len(df) - forecast_horizon + 50, max(max(df['Close']), max(simulations[pick])), 'Testing',
             horizontalalignment='center', verticalalignment='center')

    plt.title(f'Training versus Testing\nTicker: {ticker}\n')
    plt.legend(loc=0)
    plt.ylabel('Price')
    plt.xlabel('T+Days')
    plt.show()

#La función test realiza un análisis adicional de las predicciones generadas por diferentes números de simulaciones.
# Toma los datos financieros (df), el símbolo del ticker (ticker), el número de simulaciones inicial (simu_start),
# el número de simulaciones final (simu_end) y el incremento de simulaciones (simu_delta).
def test(df, ticker, simu_start=100, simu_end=1000, simu_delta=100, **kwargs):
    table = pd.DataFrame()
    table['Simulations'] = np.arange(simu_start, simu_end + simu_delta, simu_delta)
    table.set_index('Simulations', inplace=True)
    table['Prediction'] = 0

    for i in np.arange(simu_start, simu_end + 1, simu_delta):
        print(i)

        forecast_horizon, simulations, pick = monte_carlo(df, simulation=i, **kwargs)
        actual_return = np.sign(df['Close'].iloc[len(df) - forecast_horizon] - df['Close'].iloc[-1])
        best_fitted_return = np.sign(simulations[pick][len(df) - forecast_horizon] - simulations[pick][-1])
        table.at[i, 'Prediction'] = np.where(actual_return == best_fitted_return, 1, -1)

    ax = plt.figure(figsize=(10, 5)).add_subplot(111)
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_visible(False)

    plt.barh(np.arange(1, len(table) * 2 + 1, 2), table['Prediction'],
             color=colorlist[0::int(len(colorlist) / len(table))])

    plt.xticks([-1, 1], ['Failure', 'Success'])
    plt.yticks(np.arange(1, len(table) * 2 + 1, 2), table.index)
    plt.xlabel('Prediction Accuracy')
    plt.ylabel('Times of Simulation')
    plt.title(f"Prediction accuracy doesn't depend on the numbers of simulation.\nTicker: {ticker}\n")
    plt.show()


def main():
    stdate = '2016-01-15'
    eddate = '2019-01-15'
    ticker = 'AAPL'

    df = yf.download(ticker, start=stdate, end=eddate)
    df.index = pd.to_datetime(df.index)

    forecast_horizon, simulations, pick = monte_carlo(df)
    plot(df, forecast_horizon, simulations, pick, ticker)
    test(df, ticker)


if __name__ == '__main__':
    main()
