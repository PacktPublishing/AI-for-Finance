"""
Prepare stock prices data for use in an LSTM network.

Our goal here is to predict a closing price of a share/stock
on a given day of a company based on the metrics from previous day.

We're working with historical stock prices data in .CSV format
for Apple (APPL) (but the code is general and you can use it
with other stocks).

Each row is a set of metrics for a given company for a given day.

We're using the exact same method that we've used in
Section 2.

The main difference is that we're using multiple
metrics/values (instead of just one in Section 2)
to predict closing price.
"""

import pandas
import numpy as np
from matplotlib import pyplot

def get_raw_xy(data):
    """
    Return only metrics/values that we will base
    our predictions on first, then a list of closing prices.

    Both metrics/values are from the same day.
    """
    # Removing data column and adj close.
    # Assuming that our data don't have any
    # dividents, so close column is the same as
    # adj close.
    data=data.drop(columns=['Date','Adj Close'])
    values=data.values
    # Each column number match a specific metric:
    # Open=0,High=1,Low=2,Close=3,Volume=4
    return values[:, [0,1,2,3,4]], values[:, 3]

def train_test_split(X, Y, trs_len=0.80):
    """
    Split both X and Y into train and test set.

    trs_len - how much data should we use for training?
              by default it's 0.80 meaning 80%, the remining
              20% of the data will be used for testing.
    """
    lx=len(X)
    trs=int(lx*trs_len)
    train_x, train_y = X[:trs], Y[:trs]
    test_x, test_y = X[trs:], Y[trs:]
    return train_x, train_y, test_x, test_y

def get_vpo(values):
    """
    This is the minimalistic version of
    get_vpo function from prep.py from Section 2.

    Day 1 A1,A2,*A3
    Day 2 B1,B2,*B3
    Day 3 C1,C2,*C3

    We want to predict values with *,so

    if we want to train our network to predict Day1
    we don't have any data from previous day, so we can't
    do that, but for we base prediction of m23 from the
    metrics from prev data: m11,m12,m13 and we can do
    the same for day 2:

    X: A1,A2,A3 Y: B3
    X: B1,B2,B3 Y: C3

    What about data from Day 3? Well, we don't have any
    data from Day4 to use for prediction using metrics
    from Day3.

    So this is how we're constructing our data:
    X:No data   Y:A3 <- We discard this first value,
    X:A1,A2,A3  Y:B3    since we don't have any X data from Day 0
    X:B1,B2,B3  Y:C3
    X:C1,C2,C3  Y:No data <- We need to discard this as well,
                                since there's no data for Y from Day 4
    """
    shifted_y=list(values)
    shifted_y.pop(0)
    shifted_y.append(None)
    return shifted_y

def get_data(f='data/stock_prices.csv'):
    """
    Load stock prices data from .CSV file,
    convert to the correct form.
    """
    d=pandas.read_csv(f)
    x,y=get_raw_xy(d)
    yy=get_vpo(y)
    return x[:-1], yy[:-1]

def prep_data(train_x, train_y):
    """
    Split data and return in the exact format
    we need it for our LSTM to learn.
    """
    train_x, train_y, test_x, test_y = train_test_split(train_x, train_y)

    # We need one more "dimension" (we need to put our values
    # into on more list) to our x values
    train_x=np.array(np.expand_dims(train_x, axis=1))
    train_y=np.array(train_y)
    test_x=np.array(np.expand_dims(test_x, axis=1))
    test_y=np.array(test_y)

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    # Here we want to first show how the "shifting" works
    # the plot a part of our train and test data.
    d=pandas.read_csv('data/stock_prices.csv')
    x, y = get_raw_xy(d)
    yy=get_vpo(y)
    print('Data before conversion.')
    for i in range(5):
        print('X[%d]/Y[%d]' % (i, i), x[i], y[i])
    print('Shitfted')
    for i in range(5):
        print('X[%d]/YY[%d]' % (i, i), x[i], yy[i])
    train_x, train_y, test_x, test_y=train_test_split(x, yy)
    # Prepare data for plotting.
    p_tx=list(train_x[:, 3])+[None]*(len(test_x))
    p_ttx=([None]*(len(train_x)-1))+list(train_x[:, 3])[-1:]+list(test_x[:, 3])
    # Plot closing prices for each day.
    pyplot.plot(p_tx, label='train_x')
    pyplot.plot(p_ttx, label='test_x')
    pyplot.legend()
    pyplot.show()
    x,y=get_data()
    print('Data before preparation')
    print(x[0],y[0])
    print('Data after preparation')
    train_x, train_y, test_x, test_y=prep_data(x, y)
    print(train_x[0],train_y[0])
