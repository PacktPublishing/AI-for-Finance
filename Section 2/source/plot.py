#!/usr/bin/env python3
import matplotlib as mpl
from matplotlib import pyplot as pp
import pandas
from prep import get_raw_xy, get_vpo
import sys
from pprint import pprint
import numpy as np

sizes=[[15,8, 10], [20, 10, 20]]
sidx=1

def setup_plot(sidx=sidx, yfrom=1973, yto=2020, step=4, xls=sizes[sidx][2]):
    pp.rcParams['figure.figsize'] = sizes[sidx][:2]
    mpl.rc('xtick', labelsize=xls)
    mpl.rc('ytick', labelsize=sizes[sidx][2])
    pp.style.use('dark_background')
    ticks_range=list(range(yfrom, yto, step))
    ax=pp.gca()
    ax.set_xticks(ticks_range)
    ax.tick_params(grid_alpha=0.5)
    pp.ylabel('Exchange rate', fontsize=sizes[sidx][2])
    pp.xlabel('Day', fontsize=sizes[sidx][2])
    pp.grid()
    return pp, ax

def plotme(values, years, prep_values=[], train_x=np.array([None]), test_x=np.array([None]), baseline=[], trscore='', tscore='', title='', bttscore=''):
  pp, ax=setup_plot(yfrom=1, yto=len(years), step=365)
  pp.plot(years, values, label='Raw values', color='red', linewidth=2)
  if prep_values != []:
    pp.plot(years, prep_values, label='Prepared values', color="white")
  if train_x.all() != None:
    pp.plot(years, train_x, label='Training prediction'.ljust(25)+'%s' % trscore, color='green', linewidth=2)
  if test_x.all() != None:
    pp.plot(years, test_x, label='Test prediction  '.ljust(27)+'%s' % tscore, color='blue', linewidth=2)
  if baseline != []:
    pp.plot(years, baseline, label='Baseline Training/Test prediction'.ljust(35)+'%s' % bttscore, color='yellow', linewidth=2)
  ax.legend(loc='best', fontsize=sizes[sidx][2])
  if title:
      pp.title(title)
  pp.show()

def plothist(history, sidx=sidx):
    pp.rcParams['figure.figsize'] = sizes[sidx][:2]
    mpl.rc('xtick', labelsize=sizes[sidx][2])
    mpl.rc('ytick', labelsize=sizes[sidx][2])
    pp.ylabel('Mean absolute percentage error', fontsize=sizes[sidx][2])
    pp.xlabel('Epoch', fontsize=sizes[sidx][2])
    pp.grid()
    h=history.history
    m='mean_absolute_percentage_error'
    pp.plot(range(1, len(h[m])+1), h[m])
    ax=pp.gca()
    ax.set_yticks(range(1, 100, 9))
    ax.set_xticks(range(1, len(h[m])+1), 1)
    pp.title('Learning curve')
    pp.show()

def plotpred(values, years, predicted, error, title=''):
    pp, ax=setup_plot(yfrom=1, yto=len(values), step=4, xls=12)
    years+=[years[-1]+'next']
    v=values+[ np.nan ]
    p=[ np.nan for i in values ]
    p+=[predicted[0][0]]
    p[-2]=v[-2]
    pv=predicted[0][0]
    pp.plot(years, v, label='Raw values', color='red', linewidth=1)
    pp.plot(years, p, label='Prediction %s +/- %s' % ("{:,.0f}".format(pv), "{:,.0f}".format(error)), color='green', linewidth=4, linestyle=":")
    ax.legend(loc='best', fontsize=sizes[sidx][2])
    pp.title(title)
    pp.show()

def prep_tt_for_plot(model, years, train_x, train_y, test_x, test_y):
    trainPredict = model.predict(train_x)
    testPredict = model.predict(test_x)

    trainPredictPlot = np.empty((len(years), 1))
    trainPredictPlot[:] = np.nan

    trainPredictPlot[:len(trainPredict)] = trainPredict
    # Remove the gap in graph by duplicating the last value
    # at the end of our training data set
    trainPredictPlot[len(trainPredict)] = trainPredict[-1]

    testPredictPlot = np.empty((len(years), 1))
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict):] = testPredict

    return trainPredictPlot, testPredictPlot

if __name__ == '__main__':
    d=pandas.read_csv(sys.argv[1])
    values, years = get_raw_xy(d)
    p_values = get_vpo(values)
    print('Year\tCurrent\tFuture')
    pprint(list(zip(zip(years, values), p_values)))
    plotme(values, years, prep_values=p_values)
