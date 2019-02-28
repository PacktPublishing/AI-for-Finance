import numpy as np
from pprint import pprint
import sys

def get_params(script='train.py'):
    xa=''
    if script == 'train.py':
        xa='[plot|ploth]'
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

def train_test_split(rawx, xpo):
    train_size=int(len(rawx)*0.80)
    test_size=int(len(rawx)*0.20)
    print("Train size:", train_size, "Test size:", test_size, len(rawx))
    train_x, train_y = np.array(rawx[:train_size]), np.array(xpo[:train_size])
    test_x, test_y = np.array(rawx[train_size:]), np.array(xpo[train_size:])
    return train_x, train_y, test_x, test_y

def logme(msg):
    pprint(msg)
