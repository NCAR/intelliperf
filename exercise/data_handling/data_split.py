#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:13:29 2018

@author: uppala
"""

import pandas as pd
import os 
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

## Reading the csv file from the computer
userhome = os.path.expanduser('~')

# Font family  
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
pdf = PdfPages('RandomForestClassifierDataPlot2.pdf')
TITLE_SIZE = 20
SUBTITLE_SIZE = 16
TEXT_SIZE = 14
LABEL_SIZE = 16
LINEWIDTH = 3

#n_trees = [10,100,500,750,1000,1500,2000]
n_trees = [10,100,500]

## CSV File location
## __file__   os.path change the absolute path
csvfile= userhome + r'/Documents/intelliperf/data/mg2/PSrad.exe.codeblocks.fused.any.any.any.slope.labelled .csv'

data_df = pd.read_csv(csvfile,delimiter = ';',names = ["module_sub_routine","id","hardware_Counter","time","event"])

## Data Processing 
data_df = data_df.drop(['module_sub_routine','id'],axis=1)

# Getting the unique name from the dataset
counter_name = data_df['hardware_Counter']
counter_name = counter_name.unique()
counter_name_per_inst = []
counter_name_normal = []

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


df_split = np.split(df_rest,[45],axis=1)
df_features = df_split[0]
df_labels = df_split[1]


print df_features.shape

''' Random Forest on _ins counters '''

df_split_ins = np.split(df_per_ins,[44],axis=1)
df_features_ins = df_split_ins[0]
df_labels_ins = df_split_ins[1]

rmse_idx = 0

for idx in range(len(n_trees)):
    
    ##################### Variables
    hardware_name = []
    importance_abs_dict = {}
    importance_ins_dict = {}
    
    rmse_abs_error = [0] * len(n_trees)
    rmse_ins_error = [0] * len(n_trees)
    error_abs_arr = [0] * 1000
    error_ins_arr = [0] * 1000
    
    mse_abs_arr = [0]* 1000
    mse_ins_arr = [0] * 1000
    
    print "***********************"
    print "at tree number "
    print idx
    plt.figure()
    
    rect_bot = [0.1,0.65,0.8,0.1]
    rect_top = [0.1,0.85,0.8,0.1]
    rect_bot_right = [0.6,0.11,0.35,0.3]
    rect_bot_left = [0.1,0.075,0.35,0.5]
    plt.figure(1,figsize= (9,9))
    
    ax = plt.axes(rect_top)
    bx = plt.axes(rect_bot)
    cx = plt.axes(rect_bot_left)
    dx = plt.axes(rect_bot_right)
    size = 0
    for rs in range(1,10):
        
        print "at split "
        print rs
        # Splitting the data
        df_features_train, df_features_test, df_labels_train, df_labels_test = train_test_split(df_features,df_labels,test_size = 0.20,random_state=rs) 

        # Create a Random Fporest with 1000 decision trees
        rf = RandomForestRegressor(n_estimators=n_trees[idx], random_state=rs)
    
        # Train the model on training data
        rf.fit(df_features_train,np.ravel(df_labels_train,order = 'C'));
    
        
        importance_normal = list(zip(rf.feature_importances_,df_features_train.columns))
        importance_normal.sort(reverse= True)
        
        for item in importance_normal:
            if item[1] in importance_abs_dict.keys():
                importance_abs_dict[item[1]] = importance_abs_dict[item[1]] + item[0]
            else:
                importance_abs_dict[item[1]] = item[0]
        
                
        #Testing the model
        prediction_test = rf.predict(df_features_test)
        
        #error_test = abs(prediction_test - df_labels_test['LABEL'])
        mse_abs = mean_squared_error(df_labels_test['LABEL'], prediction_test)
        
        mse_abs_arr[rs-1] = mse_abs
        
        prediction = rf.predict(df_features)
        
        
        # Calculating the Error
        error = abs(prediction - df_labels['LABEL'])
        
        
        for index,val in error.items():
            error_abs_arr[index] = error_abs_arr[index] + val
        
        # Splitting the data
        df_features_ins_train, df_features_ins_test, df_labels_ins_train, df_labels_ins_test = train_test_split(df_features_ins,df_labels_ins,test_size = 0.20, random_state=rs ) 

    
        # Create a Random Fporest with 1000 decision trees
        rf2 = RandomForestRegressor(n_estimators=n_trees[idx], random_state=rs)

        # Train the model on training data
        rf2.fit(df_features_ins_train,np.ravel(df_labels_ins_train,order = 'C'));

        importance_ins = list(zip(rf2.feature_importances_,df_features_ins_train.columns))
        importance_ins.sort(reverse= True)
        
        for item in importance_ins:
            temp = item[1].replace("_per_ins","")
            if temp in importance_ins_dict.keys():
                importance_ins_dict[temp] = importance_ins_dict[temp] + item[0]
            else:
                importance_ins_dict[temp] = item[0]
        
        #print importance_normal
        prediction_ins_test = rf2.predict(df_features_ins_test)
        
        #error_test = abs(prediction_test - df_labels_test['LABEL'])
        mse_ins = mean_squared_error(df_labels_ins_test['LABEL'], prediction_ins_test)
        mse_ins_arr[rs-1] = mse_ins 
    
        #Testing the model
        prediction_ins = rf2.predict(df_features_ins)
    
        # Calculating the Error
        error_ins = abs(prediction_ins - df_labels_ins['LABEL'])
        
        for index,val in error_ins.items():
            error_ins_arr[index] = error_ins_arr[index] + val
        size = rs
    
    
    w, h = 3, 15;
    
    Matrix = [[0 for x in range(w)] for y in range(h)]
    
    labelr = []
    for x in range(1,16):
        labelr.append(x)
    
    labelc = ['Hardware Counter Name ', 'Importance of RF1','Importance of RF2']
    
    sorted_dict = sorted(importance_abs_dict)
    
    for x in importance_abs_dict:
        print x,importance_abs_dict[x]
        
    name_counter= []    
    print '------------------------------'
    for key, value in sorted(importance_abs_dict.iteritems(), key=lambda (k,v): (v,k),reverse=True):
        if key != 'PAPI_TOT_INS':    
            name_counter.append(key)
    cnt = 0
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            if j == 0 :
                Matrix[i][j] = name_counter[cnt]
            if j == 1 :
                Matrix[i][j] = round(importance_abs_dict[name_counter[cnt]]/size,4)
            if j == 2 :
                Matrix[i][j] = round(importance_ins_dict[name_counter[cnt]]/size,4)
        cnt = cnt+1
    
    print '                                      '
    print 'Ins values'
    xList = range(0,1000)
    print len(xList)
    print len(error_abs_arr)
    ax.bar(xList,error_abs_arr)
    ax.set_title('Random Forest classifier with number of Trees={}'.format(n_trees[idx]))
    ax.autoscale(tight= True)
    ax.set_ylabel('Error')
    ax.set_xlabel('Time')
    bx.bar(xList,error_ins_arr)
    bx.autoscale(tight= True)
    bx.set_ylabel('Error')
    rmse_abs_error[rmse_idx] = np.sqrt(np.mean(mse_abs_arr))
    rmse_ins_error[rmse_idx] = np.sqrt(np.mean(mse_ins_arr))
    lightgrn = (0.5, 0.8, 0.5)
    table = cx.table(cellText = Matrix,
              colLabels=labelc,
              colColours=[lightgrn] * 16,
              cellLoc='center',
              colWidths=[0.4 for x in labelc],    
              loc='center')
    table.set_fontsize(25)
    '''
    table_props = table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell._text.set_fontsize(50)
        cell._text.set_color('red')
    '''    
    cx.axis('off')
    b1 = dx.bar([1], [rmse_abs_error[rmse_idx]], width=0.4,
            label="Bar 1", align="center")

    b2 = dx.bar([2], [rmse_ins_error[rmse_idx]], color="red", width=0.4,
            label="Bar 2", align="center")
    dx.legend()
    dx.autoscale(tight= True)
    dx.set_ylabel('Error')
    dx.set_xlabel('Type of hardware counter')
    dx.set_xlim([0,3])
    val = max(rmse_abs_error[rmse_idx],rmse_ins_error[rmse_idx])
    dx.set_ylim([0,val*1.5])
    rmse_idx = rmse_idx + 1;
    pdf.savefig()
    
    
plt.plot(n_trees,rmse_abs_error,color = 'g')
plt.plot(n_trees,rmse_ins_error, color = 'orange')
plt.xlabel('Trees')
plt.ylabel('RMSE')
plt.title('RMSE plot for different number of trees')
pdf.savefig()   
plt.close()
print rmse_abs_error
print rmse_ins_error    
pdf.close()
