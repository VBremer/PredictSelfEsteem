# -*- coding:utf-8 -*-
###########################################################################################
# STUDY: Predicting Self-Esteem
# Leuphana Univerisity Lueneburg - Institute of Information Systems
###########################################################################################
import pandas as pd
import math
import numpy as np
def create_data(all_data):
    """Manipulate and create first dataset."""
    list_var = ['dummy', 'sleep', 'mood', 'worrying', 'self-esteem',
                'enjoyed activities', 'social contact', 'pleasant activities']
    cols = ('id', 'day', 'mood', 'worrying', 'pleasant activities', 'sleep',
            'enjoyed activities', 'self-esteem', 'social contact')
    data = pd.DataFrame(columns=cols)
    cols2 = ('mood', 'worrying', 'pleasant activities', 'sleep', 'enjoyed activities',
             'self-esteem', 'social contact')
    vars = pd.DataFrame(columns=cols2)
    for i in cols2:
        vars = vars.set_value(1, i, 1)
    k = 1
    id = 1
    day = 1
    matching_var = [s for s in list_var if all_data['question'].iloc[0] in s]
    data = data.set_value(k, matching_var, all_data['rating'].iloc[0])
    data = data.set_value(1, 'id', 1)
    data = data.set_value(1, 'day', 1)
    vars = vars.set_value(1, matching_var, 1)
    for i in range(1, len(all_data)):
        if (all_data['ID'].iloc[i - 1] == all_data['ID'].iloc[i] and
           all_data['day'].iloc[i - 1] == all_data['day'].iloc[i]):
            data = data.set_value(k, 'id', id)
            data = data.set_value(k, 'day', day)
            matching_var = [s for s in list_var if all_data['question'].iloc[i] in s]
            if math.isnan(data[matching_var].iloc[k - 1]):
                data = data.set_value(
                    k, matching_var, all_data['rating'].iloc[i])
            else:
                vars[matching_var] += 1
                data = data.set_value(
                    k, matching_var, all_data['rating'].iloc[i] + data[matching_var].iloc[k - 1])
        else:
            for variable in cols2:
                data = data.set_value(
                    k, variable, data[variable].iloc[k - 1] / vars[variable][1])
                vars[variable] = 1
            matching_var = [s for s in list_var if all_data['question'].iloc[i] in s]
            k = k + 1
            day = day + 1
            data = data.set_value(k, 'day', day)
            if all_data['ID'].iloc[i - 1] != all_data['ID'].iloc[i]:
                day = 1
                id = id + 1
                data = data.set_value(k, 'id', id)
            else:
                data = data.set_value(k, 'id', id)
            data = data.set_value(k, matching_var, all_data['rating'].iloc[i])
    return(data)

def drop_na_and_scale(data):
    """Delete all missing values and scale the data."""
    # change when amount of variables to analyze differs
    c = np.arange(0, 9)
    # 7 stands for self esteem, change to 2 for mood i.e.
    x = 7
    # 4 stands for pleasant activities to drop
    dropping = 4
    char = list(data.columns.values)[x]
    data_final = data.dropna().copy()
    proc = data_final.drop(data_final.columns[[0, 1, dropping, x, 9]], axis=1)
    #proc = data_final.drop(data_final.columns[[0, 1, x, 9]], axis=1)
    proc = (proc - proc.mean()) / proc.std()
    proc = proc.astype(float)
    data_final = data_final[[0,1,9,x]].join(proc)
    new = data_final[[char]].astype(float)
    new = np.round(new, 0)
    data_final = data_final.drop(data_final[[char]], axis=1).join(new)
    return(data_final)
    
def manipulate_ID(data):
    """Manipulate ID´s from 1-N."""
    data_final = data.copy()
    dat_check = data.copy()
    id = 1
    for i in range(1, len(data)):
        if dat_check.iloc[i-1, 0] == dat_check.iloc[i, 0]:
            data_final.iloc[i, 0] = id
        else:
            id = id + 1
            data_final.iloc[i, 0] = id
    data_final.iloc[0,0] = 1
    return(data_final)
    
def colVars(a):
    """Function to calculate posterior variances from simulation."""
    diff = np.matrix(a) - np.repeat(np.mean(a, axis=0), len(a), axis=0)
    vars = vars = np.mean(np.square(diff), axis=0)*len(a)/(len(a)-1)
    return(vars)