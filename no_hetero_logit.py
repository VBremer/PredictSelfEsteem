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
# load prepared data
##########
with open('data/dat_prepared.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    data, inds = pickle.load(f)

##########
# CV no hetero logit
##########
rmse, mae, dic, waic, rmse_mean, mae_mean = [], [], [], [], [], []
for i in range(10):
    data_train = data.iloc[list(inds[i][0])]
    test_set = data.iloc[list(inds[i][1])]

    x = data_train[['mood', 'worrying', 'sleep', 'enjoyed activities',
    'social contact']]
    x_pred = test_set[['mood', 'worrying', 'sleep', 'enjoyed activities',
    'social contact']]
    Y = test_set['self-esteem']
    y_mean = np.repeat(np.round(np.mean(data_train['self-esteem'])),len(Y))
        
    regress_dat_no_hetero = {'C': 10, 'y': data_train["self-esteem"].astype(int),
        'N': len(data_train), 'K': 5,
        'x': x,
        'alpha': np.repeat(1, 9),
        'x_pred': x_pred,
        'N2': len(test_set)}
    
    fitlog1 = pystan.stan(file='models/myOrderedLogitModel_no_hetero_pred.stan', data=regress_dat_no_hetero, iter=50, warmup=10, thin=2, chains=2, n_jobs=1)   
    
    # calculate rmse/mae
    res1 = fitlog1.extract(permuted=True)
    y1 = res1['y_pred']
    y2 = np.mean(y1, axis=0)
    y = np.round(y2)    
    rmse.append(sqrt(mean_squared_error(Y, y)))
    mae.append(mean_absolute_error(Y, y))
    mae_mean.append(mean_absolute_error(Y, y_mean))
    rmse_mean.append(sqrt(mean_squared_error(Y, y_mean)))
    
    # calculate DIC
    deviance = pd.DataFrame(res1['dev'])
    # posterior mean of deviance
    Dbar = np.mean(deviance)
    # effective number of parameters 
    pD = np.var(deviance)
    # Deviance Information Criterion / DIC = Dbar + pD
    DIC = Dbar + .5 * pD
    dic.append(DIC)
    
    # calculate WAIC
    log_lik = pd.DataFrame(res1['logLik'])
    lppd = sum(np.log(np.mean(np.exp(log_lik), axis=0)))
    p_waic = np.matrix.sum(colVars(np.matrix(log_lik)))
    wai_c = -2*lppd + 2*p_waic
    waic.append(wai_c)
        
# save res
with open('Nohetero_stereo.pickle', 'wb') as f:  
    pickle.dump([rmse, mae, rmse_mean, mae_mean, dic, waic, res1, data, inds], f)