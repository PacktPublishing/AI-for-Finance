#!/usr/bin/env python3
"""
Predict a number of passangers carried
in the next year based on a trained model.
"""
import conf

from keras.models import Sequential, load_model
from keras.layers import Dense

from prep import get_data
from plot import plotpred
from tools import get_params
from train import confs
import numpy as np

import os

if __name__ == '__main__':
    name, epochs, batches, _=get_params(script='predict.py')
    model,_=confs[name]
    mname='models/model-%s-%d-%d.h5' % (name, epochs, batches)
    # Loading the model.
    if os.path.exists(mname):
        model=load_model(mname)
        print('Model loaded!')
    else:
        print("Can't find %s model, train it first using 'train.py %s %d %d'" % (mname, name, epochs, batches))
    years, _, values=get_data()
    # We're using the last value of our dataset
    # as a base for prediction.
    values, years=list(values), list(years)
    predict_on_value=values[-1:]
    predict_for_year=years[-1]+'next'
    # This is where magic happens,
    # we get the predicted value
    x=model.predict(np.array(predict_on_value))
    # Calcularing the error,
    # About 3% error in our best models
    error=x[0][0]*0.02
    print('Prediction from %s based on %s' % (years[-1], "{:,.0f}".format(predict_on_value[0])))
    print('Prediction for %s is %s +/- %s' % (predict_for_year, "{:,.0f}".format(x[0][0]), "{:,.0f}".format(error)))
    # Plot both values and predicted value
    plotpred(values[-30:], years[-30:], x, error, "Prediction for model %s, Epochs=%d, batch-size=%d" % (name, epochs, batches))
