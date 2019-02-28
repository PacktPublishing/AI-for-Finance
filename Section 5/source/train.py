#!/usr/bin/env python3
"""
Train and test our prediction model.

Note:
Correct batch-size for LSTM is one
that you can use to divde a number of
samples for both training and testing set by.

Note: Make sure to install matplolib-2.2.3 if you want
to generate the charts with plot option, you can do that
by simply running:
conda install matplotlib=2.2.3
"""
# Configure to get the same
# results every time.
import conf

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

import math
import os
import sys

from prep import get_data, prep_data
from matplotlib import pyplot

import numpy as np

def get_lstm(batches, input_shape):
    """
    Define and return stateful LSTM.

    Stateful simply means that on every epoch we're not
    starting from scratch, but we're using "remembered"
    sequences from previous epochs which in practice
    means that we should learn "better" and faster.

    input_shape = (number of past data to look for, number of metrics)

    When stateful is True we need to provide batch_input_shape.
    """
    model = Sequential()
    model.add(LSTM(60, input_shape=input_shape, stateful=True, batch_input_shape=(batches, input_shape[0], input_shape[1])))
    model.add(Dense(1))
    return model

confs={'default': dict(model=get_lstm)}

def train_model(name, train_x, train_y, epochs, batches, test_x, test_y):
    """
    Get model if it exists, train if needed.
    """
    mparams=confs[name]
    model=mparams['model'](batches, (train_x.shape[1], train_x.shape[2]))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mape'])
    # Since the stateful LSTM is learning dependencies between data points we want
    # keep the data in the same order on each epoch, thus we don't want to shuffle it.
    history=model.fit(train_x, train_y, verbose=2, epochs=epochs, batch_size=batches, validation_data=(test_x, test_y), shuffle=False)
    return model, name, mparams, history

def get_params(script='train.py'):
    """
    Get command line parameters.
    """
    xa=''
    if script == 'train.py':
        xa='ploth'
    try:
        name, epochs, batches=sys.argv[1:4]
    except ValueError:
        print('Usage: %s model_name epochs batch_size %s' % (script, xa))
        exit(1)
    try:
        plot=sys.argv[4]
    except IndexError:
        plot=False

    return name, int(epochs), int(batches), plot

if __name__ == '__main__':
    X,Y=get_data()
    train_x,train_y,test_x,test_y=prep_data(X,Y)
    # Getting our command line parameters
    name, epochs, batches, plot=get_params()
    # Do the training
    model, name, mp, history=train_model(name, train_x, train_y, epochs, batches, test_x, test_y)
    # Save models and the training history for later use
    mname='models/model-%s-%d-%d' % (name, epochs, batches)
    model.save(mname+'.h5')
    title='%s (epochs=%d, batch_size=%d)' % (name, epochs, batches)
    # Test our model on both data that has been seen
    # (training data set) and unseen (test data set)
    print('Scores for %s' % title)
    # Notice that we need to specify batch_size in evaluate when we're
    # using LSTM.
    train_score = model.evaluate(train_x, train_y, verbose=0, batch_size=batches)
    trscore='RMSE: $%s MAPE: %.0f%%' % ("{:,.0f}".format(math.sqrt(train_score[0])), train_score[2])
    print('Train Score: %s' % trscore)
    test_score = model.evaluate(test_x, test_y, verbose=0, batch_size=batches)
    tscore='RMSE: $%s MAPE: %.0f%%' % ("{:,.0f}".format(math.sqrt(test_score[0])), test_score[2])
    print('Test Score: %s' % tscore)
    # Plot history
    if plot:
        pyplot.plot([ math.sqrt(l) for l in history.history['loss'] ], label='train RMSE ($)')
        pyplot.plot([ math.sqrt(l) for l in history.history['val_loss'] ], label='test RMSE ($)')
        pyplot.legend()
        pyplot.show()

        pyplot.plot(history.history['mean_absolute_percentage_error'], label='train mape')
        pyplot.plot(history.history['val_mean_absolute_percentage_error'], label='test mape')
        pyplot.legend()
        pyplot.show()
