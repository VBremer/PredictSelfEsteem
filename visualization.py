# -*- coding:utf-8 -*-
###########################################################################################
# STUDY: Predicting Self-Esteem
# Leuphana Univerisity Lueneburg - Institute of Information Systems
###########################################################################################
##########
# import modules
##########
import numpy as np
import pandas as pd
import pystan
import math
import pickle
import statsmodels
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from scipy import stats
##########
# load data
##########
# all_data = pd.read_csv('/media/bremer/Share HDD/Data/ECompared/data_EMA.csv') ### Linux HOME
#all_data = pd.read_csv('/media/bremer/eedd7f3f-9bf2-4fbc-8212-d8907912b2fa/EComparedData/161024/data_EMA.csv') ### Linux new data (include new path)
#all_data = pd.read_csv('/media/bremer/eedd7f3f-9bf2-4fbc-8212-d8907912b2fa/EComparedData/161024/data_EMA.csv') ### Linux new data (include new path)
#all_data = pd.read_csv('/Users/Bremer/Documents/EComparedData/161024/data_EMA.csv') ### MAC

# load res
with open('/Users/Bremer/Documents/GIT/Hub/PredictSelfEsteem/data/res_example_Nohetero_stereo.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    [rmse, mae, rmse_mean, mae_mean, dic, waic, res3, data, inds] = pickle.load(f)

##########
# Visualization of predictors
##########
data['id'] = data['id'].astype(int)
mood, worr, enjo, sleep, social, self = [], [], [], [], [], []
for i in range(max(data['id'])):
    mood.append(data['mood'][data['id'] == i])
    worr.append(data['worrying'][data['id'] == i])
    enjo.append(data['enjoyed activities'][data['id'] == i])
    sleep.append(data['sleep'][data['id'] == i])
    social.append(data['social contact'][data['id'] == i])
    self.append(data['self-esteem'][data['id'] == i])
    
var_mood, var_worr, var_enjo, var_sleep, var_social, var_self = [], [], [], [], [], [] 
for i in range(1,len(mood)):
    var_mood.append(np.var(mood[i]))
    var_worr.append(np.var(worr[i]))
    var_enjo.append(np.var(enjo[i]))
    var_sleep.append(np.var(sleep[i]))
    var_social.append(np.var(social[i]))
    var_self.append(np.var(self[i]))

var_mood = pd.DataFrame(var_mood)
var_worr = pd.DataFrame(var_worr)
var_enjo = pd.DataFrame(var_enjo)
var_sleep = pd.DataFrame(var_sleep)
var_social = pd.DataFrame(var_social)
var_self = pd.DataFrame(var_self)

# variance plot 1
sns.set(color_codes=True)
sns.set(style="darkgrid", palette="muted")
fig, axs = plt.subplots(ncols=3, nrows=2)
sns.distplot(var_mood, kde=False, ax=axs[0,0])
sns.distplot(var_worr, kde=False, ax=axs[0,1])
sns.distplot(var_enjo, kde=False, ax=axs[0,2])
sns.distplot(var_sleep, kde=False, ax=axs[1,0])
sns.distplot(var_social, kde=False, ax=axs[1,1])
sns.distplot(var_self, kde=False, ax=axs[1,2])
axs[0,0].set_xlabel('Mood', fontsize=20)
axs[0,1].set_xlabel('Worry', fontsize=20)
axs[0,2].set_xlabel('Enjoyed Activities', fontsize=20)
axs[1,0].set_xlabel('Sleep', fontsize=20)
axs[1,1].set_xlabel('Social Contact', fontsize=20)
axs[1,2].set_xlabel('Self-Esteem', fontsize=20)
axs[0,0].set_ylabel('Amount Clients', fontsize=20)
axs[0,1].set_ylabel('Amount Clients', fontsize=20)
axs[0,2].set_ylabel('Amount Clients', fontsize=20)
axs[1,0].set_ylabel('Amount Clients', fontsize=20)
axs[1,1].set_ylabel('Amount Clients', fontsize=20)
axs[1,2].set_ylabel('Amount Clients', fontsize=20)
sns.plt.show()

# total values plot
sns.set(color_codes=True)
sns.set(style="darkgrid")
fig, axs = plt.subplots(ncols=3, nrows=2)
sns.countplot(data['mood'], ax=axs[0,0], color="#B5C9EB")
sns.countplot(data['worrying'], ax=axs[0,1], color="#B5C9EB")
sns.countplot(data['enjoyed activities'], ax=axs[0,2], color="#B5C9EB")
sns.countplot(data['sleep'], ax=axs[1,0], color="#B5C9EB")
sns.countplot(data['social contact'], ax=axs[1,1], color="#B5C9EB")
sns.countplot(data['self-esteem'], ax=axs[1,2], color="#B5C9EB")
axs[0,0].set_xlabel('Mood', fontsize=20)
axs[0,1].set_xlabel('Worry', fontsize=20)
axs[0,2].set_xlabel('Enjoyed Activities', fontsize=20)
axs[1,0].set_xlabel('Sleep', fontsize=20)
axs[1,1].set_xlabel('Social Contact', fontsize=20)
axs[1,2].set_xlabel('Self-Esteem', fontsize=20)
axs[0,0].set_ylabel('Amount of occurences', fontsize=20)
axs[0,1].set_ylabel('Amount of occurences', fontsize=20)
axs[0,2].set_ylabel('Amount of occurences', fontsize=20)
axs[1,0].set_ylabel('Amount of occurences', fontsize=20)
axs[1,1].set_ylabel('Amount of occurences', fontsize=20)
axs[1,2].set_ylabel('Amount of occurences', fontsize=20)
axs[0,0].set_xlim(-1,9.5)
axs[0,1].set_xlim(-1,9.5)
axs[0,2].set_xlim(-1,9.5)
axs[1,0].set_xlim(-1,9.5)
axs[1,1].set_xlim(-1,9.5)
axs[1,2].set_xlim(-1,9.5)
sns.plt.show()

# cumulative distribution plot
sns.set(color_codes=True)
sns.set(style="darkgrid", palette='muted')
fig, axs = plt.subplots(ncols=3, nrows=2)
sns.distplot(data['mood'], ax=axs[0,0], hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
sns.distplot(data['worrying'], ax=axs[0,1], hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
sns.distplot(data['enjoyed activities'], ax=axs[0,2], hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
sns.distplot(data['sleep'], ax=axs[1,0], hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
sns.distplot(data['social contact'], ax=axs[1,1], hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
sns.distplot(data['self-esteem'], ax=axs[1,2], hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
axs[0,0].set_xlabel('Mood', fontsize=18)
axs[0,1].set_xlabel('Worry', fontsize=18)
axs[0,2].set_xlabel('Enjoyed Activities', fontsize=18)
axs[1,0].set_xlabel('Sleep', fontsize=18)
axs[1,1].set_xlabel('Social Contact', fontsize=18)
axs[1,2].set_xlabel('Self-Esteem', fontsize=18)
axs[0,0].set_ylabel('CDF', fontsize=18)
axs[0,1].set_ylabel('CDF', fontsize=18)
axs[0,2].set_ylabel('CDF', fontsize=18)
axs[1,0].set_ylabel('CDF', fontsize=18)
axs[1,1].set_ylabel('CDF', fontsize=18)
axs[1,2].set_ylabel('CDF', fontsize=18)
axs[0,0].set_xlim(0,10)
axs[0,1].set_xlim(0,10)
axs[0,2].set_xlim(0,10)
axs[1,0].set_xlim(0,10)
axs[1,1].set_xlim(0,10)
axs[1,2].set_xlim(0,10)
sns.plt.show()

##########
# Visualization of Predictions and True Values
##########
# assume last fold as test set 
test_set = data.iloc[list(inds[9][1])]

# prepare plot
true_val = pd.DataFrame(test_set['self-esteem'])
true_val = true_val.reset_index(drop=True)
y1 = res3['y_pred']
y2 = np.mean(y1, axis=0)
preds = np.round(y2) 
preds = pd.DataFrame(preds)
preds = preds.reset_index(drop=True)
vals_stereo = preds.join(true_val.rename(columns={0:'true'}))
vals_stereo.columns = ['preds', 'true']
vals_stereo = vals_stereo.sort_values(['true'], ascending=[True])
vals_stereo['num'] = range(0,172)

# plot
sns.set(color_codes=True)
sns.set(style="darkgrid", palette="muted")
ax = sns.regplot("num", "preds", data=vals_stereo, fit_reg=False, marker='x', color='#990000')
plt.plot("num", "true", data=vals_stereo, color='black')
ax.set_xlabel('Number of prediction', fontsize=26)
ax.set_ylabel('Self-Esteem value', fontsize=26)
ax.set_xlim(-1,175)
ax.set_ylim(0.8,10.2)
sns.plt.show()

##########
# Parameter for each user - calculation
##########
with open('/Users/Bremer/ownCloud/1_RESEARCH/Mental Health/Self-Esteem/Python/Results/ecompared/prior_res/all_res/hetero_res_stereo_ALL.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    data, res = pickle.load(f)
    
beta = res3['beta']
#perc = np.percentile(beta, [2.5, 50, 97.5], axis=0) also works just like this: just to be sure, I have coded the following:

list_mood = [[] for x in range(len(beta[1]))]
list_rum = [[] for x in range(len(beta[1]))]
list_act = [[] for x in range(len(beta[1]))]
list_sleep = [[] for x in range(len(beta[1]))]
list_social = [[] for x in range(len(beta[1]))]

for k in range(len(beta)):
    for i in range(len(beta[1])):
        list_mood[i].append(beta[k][i][0])
        list_rum[i].append(beta[k][i][1])
        list_sleep[i].append(beta[k][i][2])
        list_act[i].append(beta[k][i][3])
        list_social[i].append(beta[k][i][4])

index = list(range(130))
columns = ['2.5', '50', '97.5']
perc_mood = pd.DataFrame(index=index, columns=columns)
perc_rum = pd.DataFrame(index=index, columns=columns)
perc_act = pd.DataFrame(index=index, columns=columns)
perc_sleep = pd.DataFrame(index=index, columns=columns)
perc_social = pd.DataFrame(index=index, columns=columns)

for i in range(len(beta[1])):
    perc_mood.iloc[i] = np.percentile(list_mood[i], [2.5, 50, 97.5], axis=0)
    perc_rum.iloc[i] = np.percentile(list_rum[i], [2.5, 50, 97.5], axis=0)
    perc_sleep.iloc[i] = np.percentile(list_sleep[i], [2.5, 50, 97.5], axis=0)
    perc_act.iloc[i] = np.percentile(list_act[i], [2.5, 50, 97.5], axis=0)
    perc_social.iloc[i] = np.percentile(list_social[i], [2.5, 50, 97.5], axis=0)

# amount of users that are in accordance with results
m = 0
r = 0
a = 0
sl = 0
so = 0
for i in range(len(beta[1])):
    if perc_mood['2.5'].iloc[i] >= 0 and perc_mood['97.5'].iloc[i] >= 0:
        m = m + 1
    if perc_rum['2.5'].iloc[i] <= 0 and perc_rum['97.5'].iloc[i] <= 0:
        r = r + 1
    if perc_act['2.5'].iloc[i] >= 0 and perc_act['97.5'].iloc[i] >= 0:
        a = a + 1
    if perc_sleep['2.5'].iloc[i] >= 0 and perc_sleep['97.5'].iloc[i] >= 0:
        sl = sl + 1
    if perc_social['2.5'].iloc[i] >= 0 and perc_social['97.5'].iloc[i] >= 0:
        so = so + 1

# see and boxplot for the users
perc_mood = perc_mood.sort(columns=['50'], axis=0)
perc_rum = perc_rum.sort(columns=['50'], axis=0)
perc_act = perc_act.sort(columns=['50'], axis=0)
perc_sleep = perc_sleep.sort(columns=['50'], axis=0)
perc_social = perc_social.sort(columns=['50'], axis=0)

fig, ax = plt.subplots(ncols=1, nrows=5)

for i in range(len(beta[1])):
    if perc_mood['2.5'].iloc[i] < 0:
        a = 1
    else:
        a = 1
    ax[0].plot( [i,i], [perc_mood['2.5'].iloc[i], perc_mood['97.5'].iloc[i]], 'k-', alpha = a) 
    ax[0].plot( [i], [perc_mood['50'].iloc[i]], 'ko', alpha = a) 
    
    if perc_rum['2.5'].iloc[i] > 0:
        a = 1
    else:
        a = 1
    ax[1].plot( [i,i], [perc_rum['2.5'].iloc[i], perc_rum['97.5'].iloc[i]], 'k-', alpha = a) 
    ax[1].plot( [i], [perc_rum['50'].iloc[i]], 'ko', alpha = a)
    
    if perc_act['2.5'].iloc[i] < 0:
        a = 1
    else:
        a = 1
    ax[2].plot( [i,i], [perc_act['2.5'].iloc[i], perc_act['97.5'].iloc[i]], 'k-', alpha = a) 
    ax[2].plot( [i], [perc_act['50'].iloc[i]], 'ko', alpha = a) 
    
    if perc_sleep['2.5'].iloc[i] < 0:
        a = 1
    else:
        a = 1
    ax[3].plot( [i,i], [perc_sleep['2.5'].iloc[i], perc_sleep['97.5'].iloc[i]], 'k-', alpha = a) 
    ax[3].plot( [i], [perc_sleep['50'].iloc[i]], 'ko', alpha = a)
    
    if perc_social['2.5'].iloc[i] < 0:
        a = 1
    else:
        a = 1
    
    ax[4].plot( [i,i], [perc_social['2.5'].iloc[i], perc_social['97.5'].iloc[i]], 'k-', alpha = a) 
    ax[4].plot( [i], [perc_social['50'].iloc[i]], 'ko', alpha = a) 
ax[0].axhline(0, c='r', lw=0.8)
ax[1].axhline(0, c='r', lw=0.8)
ax[2].axhline(0, c='r', lw=0.8)
ax[3].axhline(0, c='r', lw=0.8)
ax[4].axhline(0, c='r', lw=0.8)
ax[0].set_xlim(-1,131)
ax[1].set_xlim(-1,131)
ax[2].set_xlim(-1,131)
ax[3].set_xlim(-1,131)
ax[4].set_xlim(-1,131)
ax[0].set_ylabel('Mood', fontsize=18)
ax[1].set_ylabel('Worry', fontsize=18)
ax[2].set_ylabel('Enjoyed Act.', fontsize=18)
ax[3].set_ylabel('Sleep', fontsize=18)
ax[4].set_ylabel('Social C.', fontsize=18)
ax[4].set_xlabel('Clients', fontsize=18)
plt.show()




