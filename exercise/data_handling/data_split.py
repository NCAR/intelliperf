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

## Reading the csv file from the computer

userhome = os.path.expanduser('~')
## CSV File location
csvfile= userhome + r'/Documents/intelliperf/data/PSrad.exe.codeblocks.fused.any.any.any.slope.labelled.csv'

data_df = pd.read_csv(csvfile,encoding = 'utf-8')

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

# Calculating the Error
error = abs(prediction - df_labels_test['LABEL'])