# -*- coding:utf-8 -*-
###########################################################################################
# STUDY: Predicting Self-Esteem
# Leuphana Univerisity Lueneburg - Institute of Information Systems
###########################################################################################
##########
# import modules
##########
from helpers import *
import numpy as np
import pandas as pd
import pystan
import pickle
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.model_selection import StratifiedKFold
import scipy
##########
# load data
##########
all_data = pd.read_csv('data/data.csv') 
##########
# manipulate data
##########
all_data = all_data.sort(['ID', 'date'], ascending=[0, 1])

data = create_data(all_data)

# change all not possible values over 10 and add real ID´s 
real_id = []
unique_id = pd.DataFrame(all_data["ID"].unique())
for i in range(0, len(data)):
    real_id.extend(unique_id.iloc[data['id'].iloc[i]-1])
    for k in range(2, 8):
        if data.iloc[i][k] > 10:
            data.iloc[i][k] = 10
data['real_IDs'] = real_id
d = data.astype(float)
data = d.round(0)

data_all = drop_na_and_scale(data)

##########
# prepare for data analysis
##########
inds = []
ids = data['id']
k_fold = StratifiedKFold(n_splits=10, random_state=0)
for train_indices, test_indices in k_fold.split(data, ids):    
    inds.append((train_indices, test_indices))
    
# save data
with open('dat_prepared.pickle', 'wb') as f:  
    pickle.dump([data, inds], f)