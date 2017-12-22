# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:49:36 2017

@author: maoli
"""

import sys
#import cPickle
import numpy as np
import pandas as pd
#from nlp_utils import clean_text, pos_tag_text
sys.path.append("../")
from param_config import config


###############
## Load Data ##
###############
print("Load data...")

dfTrain = pd.read_csv(config.original_train_data_path).fillna("")
dfTest = pd.read_csv(config.original_test_data_path).fillna("")
# number of train/test samples
num_train, num_test = dfTrain.shape[0], dfTest.shape[0]

print("Done.")

