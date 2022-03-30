import pandas as pd
import numpy as np
import pandas_datareader as pdr
import datetime as dt

# Recebe-se uma série de retornos e uma lista de pesos
"""
# ! Base de testes
YEAR = 10
DAYS = 365 * YEAR
stocks = ['EEM', 'SPY', 'XLB', 'XLK','XLY']
spy = ['SPY']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=DAYS)
df = pd.concat([pdr.get_data_yahoo(stocks,
                                   start=startDate,
                                   end=endDate)['Close']], axis=1)
dfr = price_to_returns(df)
dfr = dfr.set_index(pd.to_datetime(dfr.index))
w = [0.2, 0.2, 0.2, 0.2, 0.2]
"""

def price_to_returns(df_price):
    return df_price.pct_change().dropna()

def fixed_portfolio(df, weights, name_portfolio = 'Portfolio'):
    # Getting a copy of a dataframe, reseting your index for a safe loc iteration 
    dfc = df.copy().reset_index(drop=True)
    num_assets = len(dfc.columns)
    # Iteration of w_i * r_i vectors
    for row in range(0, len(dfc)):
        dfc.loc[row, name_portfolio] = np.transpose(weights) @ np.array(dfc[dfc.columns[0:num_assets]].iloc[row])
    return dfc.set_index(df.index)

def dinamyc_portfolio(df, weights, name_portfolio = 'Portfolio'):
    # Getting a copy of a dataframe, reseting your index for a safe loc iteration 
    dfc = df.copy().reset_index(drop=True)
    num_assets = len(dfc.columns)
    lst_weights = []
    # Iteration for adding new columns of the first weights vector
    for num in range(1, num_assets + 1):
        lst_weights.append(f'w_{num}')
    for num, weights_name in enumerate(lst_weights):
        dfc.loc[0, weights_name] = weights[num]
    # Iteration for calculate the dynamism of the weights vector including the returns and the column of portfolio performance
    for row in range(1, len(dfc) - 1):
        for num, weights_name in enumerate(lst_weights):
            dfc.loc[row, weights_name] = ((dfc.loc[row - 1, weights_name] * (dfc.loc[row - 1, dfc.columns[num]] + 1)) 
                                        / (np.transpose((np.array(dfc[dfc.columns[0:num_assets]].iloc[row - 1]) + 1)) 
                                        @ np.array(dfc[dfc.columns[num_assets:num_assets * 2]].iloc[row - 1])))
            dfc.loc[row - 1, name_portfolio] = (np.transpose(np.array(dfc[dfc.columns[num_assets:num_assets * 2]].iloc[row - 1]))
                                            @ np.array(dfc[dfc.columns[0:num_assets]].iloc[row - 1]))
    return dfc.set_index(df.index).dropna()

def weekly_rebalancing(df):
    dfc = df.copy()
    dfc = dfc.set_index(pd.to_datetime(dfc.index))
    dfc['DateAux'] = dfc.index
    dfc['DateAux'] = dfc['DateAux'].dt.week.astype(str) + '| ' + dfc['DateAux'].dt.year.astype(str)
    return dfc

def monthly_rebalancing(df):
    dfc = df.copy()
    dfc = dfc.set_index(pd.to_datetime(dfc.index))
    dfc['DateAux'] = dfc.index
    dfc['DateAux'] = dfc['DateAux'].dt.month.astype(str) + '| ' + dfc['DateAux'].dt.year.astype(str)
    return dfc

def quarterly_rebalancing(df):
    dfc = df.copy()
    dfc = dfc.set_index(pd.to_datetime(dfc.index))
    dfc['DateAux'] = dfc.index
    dfc['DateAux'] = dfc['DateAux'].dt.quarter.astype(str) + '| ' + dfc['DateAux'].dt.year.astype(str)
    return dfc

def semiannually_rebalancing(df):
    dfc = df.copy()
    dfc = dfc.set_index(pd.to_datetime(dfc.index))
    dfc['DateAux'] = dfc.index
    dfc['DateAux'] = np.where(dfc['DateAux'].dt.quarter.gt(2), 2, 1).astype(str) + '| ' + dfc['DateAux'].dt.year.astype(str)
    return dfc

def annually_rebalancing(df):
    dfc = df.copy()
    dfc = dfc.set_index(pd.to_datetime(dfc.index))
    dfc['DateAux'] = dfc.index
    dfc['DateAux'] = dfc['DateAux'].dt.year.astype(str)
    return dfc

def rebalancing_period(df, period = 'W'):
    if period == 'W':
        lst_period = weekly_rebalancing(df)
    elif period == 'M':
        lst_period = monthly_rebalancing(df)
    elif period == 'Q':
        lst_period = quarterly_rebalancing(df)
    elif period == 'S':
        lst_period = semiannually_rebalancing(df)
    elif period == 'Y':
        lst_period = annually_rebalancing(df)
    else:
        lst_period = "Please choice between W (weekly), M (monthly), Q (quarterly), S (semiannually) and Y (annually)."
    return lst_period

def portfolio_rebalancing(df, weights, period, name_portfolio = 'Portfolio'):
    dfc = rebalancing_period(df.copy(), period)
    lst_period = dfc['DateAux'].unique()
    lst_dataframe = []
    for month_year in lst_period:
        lst_dataframe.append(dfc[dfc['DateAux'] == month_year])
    dff = pd.DataFrame()
    for dataframe in lst_dataframe:
        dff = dff.append(dinamyc_portfolio(dataframe[dataframe.columns[:-1]], weights, name_portfolio), ignore_index=True)
    return dff