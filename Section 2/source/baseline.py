from sklearn.metrics import mean_squared_error
import sys
from tools import train_test_split
from plot import plotme
import math
from prep import get_data
import numpy as np

def get_baseline(all=True):
	years, past_values, values=get_data()
	train_x, train_y, test_x, test_y =train_test_split(past_values, values)

	pred=train_x
	train_score = mean_squared_error(train_y, pred)
	print('Baseline Training Score: RMSE: %s' % '{:,.0f}'.format(math.sqrt(train_score)))

	pred=test_x
	test_score = mean_squared_error(test_y, pred)
	print('Baseline Test Score: RMSE: %s' % '{:,.0f}'.format(math.sqrt(test_score)))

	bttscore='RMSE: %s/%s' % ('{:,.0f}'.format(math.sqrt(train_score)), '{:,.0f}'.format(math.sqrt(test_score)))

	if all:
		plot_y=[i for i in train_y] + [x for x in test_y]
		plot_pred=[i for i in train_x] + [x for x in test_x]
	else:
		plot_y=[None for i in train_y] + [x for x in test_y]
		plot_pred=[None for i in train_x] + [x for x in test_x]
	return np.array(plot_y), np.array(plot_pred), np.array(years), bttscore

if __name__ == '__main__':
	py, pp, y, _=get_baseline(all=True)
	plotme(years=y, values=py, baseline=pp)
