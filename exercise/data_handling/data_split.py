#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:13:29 2018

@author: uppala
"""

import pandas as pd
import os 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

## Reading the csv file from the computer
userhome = os.path.expanduser('~')
## CSV File location
csvfile= userhome + r'/Documents/intelliperf/data/mg2/PSrad.exe.codeblocks.fused.any.any.any.slope.labelled .csv'

data_df = pd.read_csv(csvfile,delimiter = ';',names = ["module_sub_routine","id","hardware_Counter","time","event"])

## Data Processing 

data_df = data_df.drop(['module_sub_routine','id'],axis=1)

# Getting the unique name from the dataset
counter_name = data_df['hardware_Counter']
counter_name = counter_name.unique()


counter_name_per_inst = ['time']
counter_name_normal = ['time']

for str in counter_name:
    if "_per_ins" in str:
        counter_name_per_inst.append(str)
    elif "LABEL" == str:
        counter_name_per_inst.append(str)
        counter_name_normal.append(str)
    else:
        counter_name_normal.append(str)
  
df_per_ins = pd.DataFrame(columns = counter_name_per_inst)

df_rest = pd.DataFrame(columns = counter_name_normal)


for counterName in counter_name_normal:
    df_normal = data_df[data_df['hardware_Counter'] == counterName]
    df_rest[counterName] = df_normal['event'].values
    
for counterName in counter_name_per_inst:
    temp_df = data_df[data_df['hardware_Counter'] == counterName]
    df_per_ins[counterName] = temp_df['event'].values

df_rest['time'] = data_df['time'].unique()
df_per_ins['time'] = data_df['time'].unique()


df_split = np.split(df_rest,[46],axis=1)
df_features = df_split[0]
df_labels = df_split[1]

# Splitting the data
df_features_train, df_features_test, df_labels_train, df_labels_test = train_test_split(df_features,df_labels,test_size = 0.20, random_state = 42 ) 

# Create a Random Fporest with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Train the model on training data
rf.fit(df_features_train,np.ravel(df_labels_train,order = 'C'));

#Testing the model
prediction = rf.predict(df_features_test)
##ax1 = plt.subplot(211)
# Calculating the Error
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
error = abs(prediction - df_labels_test['LABEL'])
error=error.sort_values(ascending= False)

time = set({})
bar_plot_dict = {}
for index,val in error.items():
    if val != 0:
        time.add(index)
        bar_plot_dict[index] = val


print('Mean absolute Error: (normal hardware counter)', np.mean(error))

df_split_ins = np.split(df_per_ins,[45],axis=1)
df_features_ins = df_split_ins[0]
df_labels_ins = df_split_ins[1]

# Splitting the data
df_features_ins_train, df_features_ins_test, df_labels_ins_train, df_labels_ins_test = train_test_split(df_features_ins,df_labels_ins,test_size = 0.20, random_state = 42 ) 

# Create a Random Fporest with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Train the model on training data
rf.fit(df_features_ins_train,np.ravel(df_labels_ins_train,order = 'C'));

#Testing the model
prediction_ins = rf.predict(df_features_ins_test)

# Calculating the Error
error_ins = abs(prediction_ins - df_labels_ins_test['LABEL'])
error_ins=error_ins.sort_values(ascending= False)

bar_ins_plot_dict = {}
for index,val in error_ins.items():
    if val != 0:
        time.add(index)
        bar_ins_plot_dict[index] = val


for time_val in time:
    if time_val not in bar_ins_plot_dict.keys():
        bar_ins_plot_dict[time_val] = 0


for time_val in time:
    if time_val not in bar_plot_dict.keys():
        bar_plot_dict[time_val] = 0

time = sorted(time)

error_bar_val = []
error_ins_bar_val = []

for val in time:
    error_bar_val.append(bar_plot_dict[val])
    
for val in time:
    error_ins_bar_val.append(bar_ins_plot_dict[val])

N = len(time)
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(ind, error_bar_val, width, color='r')

rects2 = ax.bar(ind + width, error_ins_bar_val, width, color='y')
ax.set_ylabel('Error')
ax.set_title('Error from Random Forest Classifier for different hardware counters')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(time)

ax.legend((rects1[0], rects2[0]), ('Hardware Counters', 'Hardware Counter with ins'))
plt.savefig('visualization.png')
plt.show()  


print('Mean absolute Error: (per_ins hardware counter)', np.mean(error_ins))