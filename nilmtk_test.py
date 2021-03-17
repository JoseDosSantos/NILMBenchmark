import shutil
import contextlib
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)

import nilmtk
from nilmtk import DataSet
from nilmtk.utils import print_dict
from nilmtk.api import API

from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85
from nilmtk_contrib.disaggregate import DAE, Seq2Point, Seq2Seq, RNN, WindowGRU, AFHMM, DSC, AFHMM_SAC, fhmm_exact


# Helper function to print without warnings
def print_nowarn(f):
    with warnings.catch_warnings(record=True):
        x = print(f())
        print(f())


# Returns formatted percentage value for labeling (see pie chart)
def get_pct(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct, absolute)


# Assigns each element in list unique number (e.g. duplicates are labeled dup 1, dup 2, dup 3)
def get_numbered_list(l):
    result = []
    for fname in l:
        orig = fname
        i=1
        while fname + ' ' + str(i) in result:
            i += 1
        result.append(fname  + ' ' + str(i))
    return result


# Helper function to run use as context for running functions without output
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout




dred = DataSet('../data/DRED.h5')
pass


dred.set_window() # makes sure you reset any window you might have set before and can use whole timeframe
experiment1 = {
  'power': {'mains': ['apparent','active'],'appliance': ['apparent','active']},
  'sample_rate': 60,
  'artificial_aggregate': True,
  'chunksize': None,
  'DROP_ALL_NANS': True,
  'appliances': ['fridge', 'cooker', 'electric heating element'], # use only 3 appliances
  'display_predictions': True,  # automatically visualize predictions vs ground truth
  'methods': {
      'Mean': Mean({}),
      #'CO': CO({})
      "AFHMM" : AFHMM({})
      #'FHMMExact' : FHMMExact({}),
      #'Seq2Point' : Seq2Point({'n_epochs':30,'batch_size':1024})
  },
  'train': {
    'datasets': {
        'DRED': {
            'path': '../data/DRED.h5', # choose training set / file path
            'buildings': {
                1: {
                    'start_time': '2015-07-20', # set time window for training
                    'end_time': '2015-08-20'}
                }
            }
        }
    },
  'test': {
    'datasets': {
        'DRED': {
            'path': '../data/DRED.h5', # choose test set / file path
            'buildings': {
                1: {
                    'start_time': '2015-09-20', # set time window for predictions
                    'end_time': '2015-09-21'
                    }
                }
            }
        },
        'metrics':['mae', 'rmse'] # fit metrics
    }
}

if __name__ == '__main__':

    results_experiment1 = API(experiment1)