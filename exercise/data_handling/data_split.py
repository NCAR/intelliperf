#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:13:29 2018

@author: uppala
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages('RandomForestClassifierDataPlot1.pdf')


############################################## Data Preprocessing ##############################################
# Relative path to the Data file
csvfile= '../../data/mg2/PSrad.exe.codeblocks.fused.any.any.any.slope.labelled .csv'

#Converting CSV data to dataframe
data_df = pd.read_csv(csvfile,delimiter = ';',names = ["module_sub_routine","id","hardware_Counter","time","event"])
data_df = data_df.drop(['module_sub_routine','id'],axis=1)

# Getting the unique name from the dataset
counter_name = data_df['hardware_Counter']
counter_name = counter_name.unique()
counter_name_per_inst = []
counter_name_normal = []

# Splitting the hardware counters to absolute counters and Per_Ins counter
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

# Splitting the abs_data into features and labes
df_split = np.split(df_rest,[45],axis=1)
df_features = df_split[0]
df_labels = df_split[1]

# Splitting the ins_data into features and labes
df_split_ins = np.split(df_per_ins,[44],axis=1)
df_features_ins = df_split_ins[0]
df_labels_ins = df_split_ins[1]

############################################## Building RandomForest Classifier ##############################################

n_trees = [10,20,40,60]

rmse_idx = 0
rmse_abs_error = [0] * len(n_trees)
rmse_ins_error = [0] * len(n_trees)
sd_abs_error = [0] * len(n_trees)
sd_ins_error = [0] * len(n_trees)

# Looping with different number of trees    
for idx in range(len(n_trees)):
    
    ##################### Variables
    hardware_name = []
    importance_abs_dict = {}
    importance_ins_dict = {}
    
    error_abs_arr = [0] * 1000
    error_ins_arr = [0] * 1000
    
    mse_abs_arr = [0]* 1000
    mse_ins_arr = [0] * 1000
    
    plt.figure()
    
    # Boundaries for plots
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
    for rs in range(1,101):
        
        # Training and Testing data from original data in the ration of 70:30
        df_features_train, df_features_test, df_labels_train, df_labels_test = train_test_split(df_features,df_labels,test_size = 0.30,random_state=rs) 

        # Create a Random Fporest with different number of decision trees
        rf = RandomForestRegressor(n_estimators=n_trees[idx], random_state=rs)
    
        # Train the model on training data
        rf.fit(df_features_train,np.ravel(df_labels_train,order = 'C'));
        importance_normal = list(zip(rf.feature_importances_,df_features_train.columns))
        importance_normal.sort(reverse= True)
        
        # Adding all the imporatance values for each random seed
        for item in importance_normal:
            if item[1] in importance_abs_dict.keys():
                importance_abs_dict[item[1]] = importance_abs_dict[item[1]] + item[0]
            else:
                importance_abs_dict[item[1]] = item[0]
        
        #Testing the model with only training data
        prediction_test = rf.predict(df_features_test)
        
        mse_abs = mean_squared_error(df_labels_test['LABEL'], prediction_test)
        mse_abs_arr[rs-1] = mse_abs
        
        #Testing model with all the data
        prediction = rf.predict(df_features)
        # Calculating the Error
        error = abs(prediction - df_labels['LABEL'])
        for index,val in error.items():
            error_abs_arr[index] = error_abs_arr[index] + val
        
        ###### Reapting the above process for the PER_INS values
        
        # Splitting the data for PER_INS counters
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
    
    
############################################## Plotting the results ##############################################
    # For plotting table with imporatance values 
    w, h = 3, 15;
    Matrix = [[0 for x in range(w)] for y in range(h)]
    labelr = []
    for x in range(1,16):
        labelr.append(x)
    
    labelc = ['Hardware Counter Name ', 'Importance of RF1','Importance of RF2']
    
    sorted_dict = sorted(importance_abs_dict)
        
    name_counter= []    
    for key, value in sorted(importance_abs_dict.iteritems(), key=lambda (k,v): (v,k),reverse=True):
        # PAPI_TOT_INS counter present in only abs counters but not in per_ins values
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
    
    xList = range(0,1000)
    # Plotting error values for the ABS hardware counter 
    ax.bar(xList,error_abs_arr)
    ax.set_title('Random Forest classifier with number of Trees={}'.format(n_trees[idx]))
    ax.autoscale(tight= True)
    ax.set_ylabel('Error')
    ax.set_xlabel('Time')
    
    # Plotting error values for the Per_Ins hardware counter
    bx.bar(xList,error_ins_arr)
    bx.autoscale(tight= True)
    bx.set_ylabel('Error')
    rmse_abs_error[rmse_idx] = np.sqrt(np.mean(mse_abs_arr))
    rmse_ins_error[rmse_idx] = np.sqrt(np.mean(mse_ins_arr))
    lightgrn = (0.5, 0.8, 0.5)
    # Creating table for the imporatance values 
    table = cx.table(cellText = Matrix,
              colLabels=labelc,
              colColours=[lightgrn] * 16,
              cellLoc='center',
              colWidths=[0.4 for x in labelc],    
              loc='center')
    table.set_fontsize(25)
    cx.axis('off')
    rmse_values= [0,rmse_abs_error[rmse_idx],rmse_ins_error[rmse_idx]]
    values = ['','Abs','PER_INS']
    ypos = np.arange(len(values))
    dx.autoscale(tight= True)
    dx.set_ylabel('Error')
    dx.set_xlabel('Type of hardware counter')
    dx.set_xlim([0,3])
    val = max(rmse_abs_error[rmse_idx],rmse_ins_error[rmse_idx])
    dx.set_ylim([0,val*1.5])
    dx.set_title("RMSE value for ABS and PER_INS")
    dx.bar(values,rmse_values)
    rmse_idx = rmse_idx + 1;
    pdf.savefig()
    plt.close()

# Plotting bar graph for RMSE values for different values of the number of trees 
N = len(n_trees)
fig1,ex = plt.subplots()
ind = np.arange(N)
width = 0.35
n_trees_Arr = np.asarray(n_trees)  
p1 = ex.bar(ind,rmse_abs_error, width, color= 'r')
p2 = ex.bar(ind+width,rmse_ins_error,width,color='g')
ex.set_title('RMSE plot for different number of trees')
ex.legend((p1[0],p2[0]),('ABS','PER_INS'))
ex.set_xticks(ind+width/2)
ex.set_xticklabels(n_trees_Arr)
ex.autoscale_view()
pdf.savefig()   
plt.close()
pdf.close()