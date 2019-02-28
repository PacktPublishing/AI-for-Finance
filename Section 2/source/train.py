#!/usr/bin/env python3
"""
Train and test our prediction model.

Note:
Make sure you've installed a specific version of
matplotlib if you want to display data charts
with:
conda install matplotlib=2.2.3

The current version of matplotlib installed
by default using conda crashes.
"""
# Configure to get the same
# results every time.
import conf

from keras.models import Sequential, load_model
from keras.layers import Dense

import math
import os
import sys
import pandas
import pickle

from plot import prep_tt_for_plot, plotme, plothist
from prep import get_data
from tools import get_params, train_test_split
from baseline import get_baseline

def get_mlp(inputs=1):
    """
    Create and return a simple Multilayer Perceptron model
    for regression/prediction problem.

    inputs - number of values we're using for prediction,
             here we're using=1
    """
    model = Sequential()
    # input_dim is our input layer
    # A number of neurons here is equal
    # to the number of values we're using for
    # prediction, which is just 1
    #
    # Next is our hidden layer, number of neurons
    # should be between number of neurons in
    # input and output layers, here we have 1 again
    #
    # Activation function is one of key component
    # in the way our network will learn.
    # Rectified linear unit (ReLU) or Rectifier
    # activation function is a great default choice
    # for most problems.
    model.add(Dense(1, input_dim=inputs, activation='relu'))
    # Last layer is our output layer
    # Number of neurons here is the same
    # as number of output values
    # Here we just want to get one prediction,
    # so just one neuron here
    model.add(Dense(1))
    return model

confs={'default': dict(model=get_mlp, inputs=1)}

def train_model(name, train_x, train_y, epochs, batches):
    """
    Get model if it exists, train if needed.
    """
    mparams=confs[name]
    model=mparams['model'](mparams['inputs'])
    # Loss is our loss or cost function - mean_squared_error
    # is a good choice assuming we don't have a lot of "outliers"
    # in our dataset.
    # Adam optimizer works great for most problems.
    #
    # Metrics are loss metrics that we want to have available for each epoch,
    # so we can review how are we doing at each training stage.
    # mse is mean_squared_error, mpe is mean_absolute_percentage_error
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mape'])
    # Here we're starting our training
    history=model.fit(train_x, train_y, verbose=2, epochs=epochs, batch_size=batches)
    return model, name, mparams, history

if __name__ == '__main__':
    # Getting data formatted as a supervised problem
    years, past_values, values=get_data()
    X, Y = past_values, values
    # Split data into two parts: one for training, one for testing
    # Test part won't be seen by a model during training so it will
    # give us some idea how our model performs on a unseen data.
    train_x, train_y, test_x, test_y = train_test_split(X, Y)
    # Getting our command line parameters
    name, epochs, batches, plot=get_params()
    # Do the training
    model, name, mp, history=train_model(name, train_x, train_y, epochs, batches)
    # Save models and the training history for later use
    mname='models/model-%s-%d-%d' % (name, epochs, batches)
    model.save(mname+'.h5')
    with open(mname+'-history.pickle', 'wb') as ms:
        pickle.dump(history.history, ms)
    print()
    print('Model and its history saved in %s*' % mname)
    title='%s (epochs=%d, batch_size=%d)' % (name, epochs, batches)
    # Test our model on both data that has been seen
    # (training data set) and unseen (test data set)
    print('Scores for %s' % title)
    train_score = model.evaluate(train_x, train_y, verbose=0)
    trscore='RMSE: %s MAPE: %.0f%%' % ("{:,.0f}".format(math.sqrt(train_score[0])), train_score[2])
    print('Train Score: %s' % trscore)
    test_score = model.evaluate(test_x, test_y, verbose=0)
    tscore='RMSE: %s MAPE: %.0f%%' % ("{:,.0f}".format(math.sqrt(test_score[0])), test_score[2])
    print('Test Score: %s' % tscore)
    # Get our baseline prediction
    # Our baseline is simply a mse of the difference between values from X and Y
    baseline=get_baseline()
    if plot == 'plot':
        train_plot, test_plot=prep_tt_for_plot(model, years, train_x, train_y, test_x, test_y)
        plotme(values, years, train_x=train_plot, test_x=test_plot, baseline=baseline[1], bttscore=baseline[3], trscore=trscore, tscore=tscore, title=title)
    elif plot == 'ploth':
        plothist(history)
