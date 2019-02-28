#!/usr/bin/env python3
"""
Load and convert data to the format that can be used with MLP network.

We're using a free daily historical exchange rates from
https://www.investing.com/currencies/usd-eur
"""
import sys
import pandas
import numpy as np
import conf

def get_raw_xy(data):
    """
    Get the data from the right columns, convert if necessary,reverse to
    get the data from oldest to newest entries.
    """
    # Pandas way to reverse the dataset
    data=data.iloc[::-1]
    dates=data['Date']
    values=data['Price'].values.astype('float32')
    return values, dates

def get_vpo(values):
    """
    Reformat data as a supervising learning problem
    that can be used used in a neural network.

    values - exchange rate for a given day

    For example:
    values=[0.82, 0.83, 0.84, 0.85]

    Meaning:
    0.82 # data for day 1
    0.83 # 2
    0.84 # 3
    0.85 # 4
    ...

    Out neural network needs two set of values for training to solve
    a prediction problem.

    Y - is a value that we want to predict
    X - is a value that we base our prediction on

    We  assume that the exchange rate for a given day (Y)
    is corelated with an exchange rate for the previous day (X).

    X         Y (those are values that we want to predict)
    nan       0.82  # X=no data for previous day Y=data for day 1
    0.82      0.83  # X=data for day 1           Y=data for day 2
    0.83      0.84  # X=2                        Y=3
    0.84      0.85  # X=4                        Y=5
    ...

    Here we want to generate X as a list of values that we use
    for prediction for each day.
    """
    yy=[np.nan]*len(values)
    for i, v in enumerate(values):
        if i+1>len(values)-1:
            break
        yy[i+1]=values[i]
    return yy

def get_data(f='data/ec.csv', logme=lambda x:x):
    """
    Get exchange rate data and prepare it for use in a neural network.
    We have to skip the first day where we have no past data, so we have no base
    for a prediction.
    """
    d=pandas.read_csv(f)
    values, years = get_raw_xy(d)
    logme('Raw values')
    rv=list(zip(years, values))
    logme(rv[:10])
    past_values=get_vpo(values)
    logme("Previous year's values X/ this year's value Y")
    logme("For every value from X expect value from Y")
    dpo=list(zip(['nan']+list(years), past_values, years, values))
    logme(dpo[:10])
    logme("X/Y values (wihout empty ones)")
    years=years[1:]
    past_values=past_values[1:]
    values=list(values[1:])
    dpo=list(zip(past_values, values))
    logme(dpo[:10])
    logme('X')
    logme(past_values[:10])
    logme('Y')
    logme(values[:10])
    return years, past_values, values

if __name__ == "__main__":
    from pprint import pprint
    get_data(logme=pprint)
