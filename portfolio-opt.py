from more_itertools import sample
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt
"""
# Quais são as features que eu quero nesse programa?
# Função para previsão dos retornos
# Olhar a relação risco-retorno de diversas formas diferentes
# Partindo de X e Y (Risco e Retorno) 
# Criar um dataframe com bullet / simulações 
# Criar as carteiras otimizadas para métricas de min_metrica_de_risco e max_metrica_de_risco_retorno
"""

YEAR = 10
DAYS = 365 * YEAR
stocks = ['EEM', 'SPY', 'XLB', 'XLK','XLY']
spy = ['SPY']

def price_to_returns(df_price):
    return df_price.pct_change().dropna()

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=DAYS)
df = pd.concat([pdr.get_data_yahoo(stocks,
                                   start=startDate,
                                   end=endDate)['Close']], axis=1)
dfr = price_to_returns(df)
dfr = dfr.set_index(pd.to_datetime(dfr.index))
dfr = dfr[:-1]

def simple_expected_returns(df):
    returns = []
    for column in df.columns:
        returns.append((1 + np.mean(df[column])) ** (252) - 1)
    return returns

def sample_covariance_matrix(df):
    return df.cov()

def ewma_weights(num, Lambda):
    lst_ewma_weights = []
    for index in range(0, num):
        lst_ewma_weights.append((1-Lambda) * (Lambda ** index))
    lst_ewma_weights = [ewma_weight / sum(lst_ewma_weights) for ewma_weight in lst_ewma_weights]
    return lst_ewma_weights

def ewma_volatility_rolling(df, 
                            Lambda = 0.94, 
                            rolling = 60, 
                            freq = 252):
    
    dfc = df.copy().reset_index(drop=True)
    n_asset = len(dfc.columns)
    for column in dfc.columns:
        dfc[f'U^2 {column}'] = dfc[column] ** 2
    lst_ewma_weights = ewma_weights(rolling, Lambda)
    lst_ewma_weights = lst_ewma_weights[::-1]
    for column in dfc.columns[0:n_asset]:
        for row in range(rolling, len(dfc)):
            dfc.loc[row, f'EWMA Vol. {column}'] = (np.sqrt((np.transpose(np.array(lst_ewma_weights))
                                                @ np.array(dfc[f'U^2 {column}'].iloc[row-rolling:row])) * freq))
    return dfc[dfc.columns[n_asset * 2:]].dropna().set_index(dfr.index[rolling:])

# Acho que estou fazendo invertido, procurar saber a multiplicação matricial via Excel

def ewma_covariance_matrix(df, Lambda = 0.94):
    dfc = df.copy()
    lst_ewma_weights = ewma_weights(len(dfc), Lambda)
    ret_ewma = dfc.mul(lst_ewma_weights, axis= 0 )
    return np.transpose(ret_ewma) @ dfc

ewma_covariance_matrix(dfr).corr()
dfr.corr()