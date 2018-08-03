#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:42:18 2018

@author: uppala
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


pdf = PdfPages('DataPlotForRF.pdf')

############################################## Data Preprocessing ##############################################
# Relative path to the Data file

csvfileScalar= '../../data/WACCM_imp_sol_scaler.slope.labelled.csv'
csvfileVector = '../../data/WACCM_imp_sol_vector.slope.labelled.csv'

#Converting CSV data to Scalar dataframe
scalarDF = pd.read_csv(csvfileScalar,delimiter = ';',names = ["module_sub_routine","id","hardware_Counter","time","event"])
scalarDF = scalarDF.drop(['module_sub_routine','id','time'],axis=1)

#Converting CSV data to Vector dataframe
vectorDF = pd.read_csv(csvfileVector,delimiter = ';',names = ["module_sub_routine","id","hardware_Counter","time","event"])
vectorDF = vectorDF.drop(['module_sub_routine','id','time'],axis=1)

counter_name = scalarDF['hardware_Counter']
counter_name = counter_name.unique()
counterName = []

for str in counter_name:
    if "_per_ins" in str:
        counterName.append(str)
    elif "LABEL" == str:
        counterName.append(str)

counterName.remove('PAPI_L3_TCR_per_ins')

scalarDF_ins = pd.DataFrame(columns = counterName)

for tempStr in counterName:
    temp_df = scalarDF[scalarDF['hardware_Counter'] == tempStr]
    scalarDF_ins[tempStr] = temp_df['event'].values

scalarDF_ins = scalarDF_ins[~scalarDF_ins.isin([np.nan,np.inf,-np.inf]).any(1)]

vectorDF_ins = pd.DataFrame(columns = counterName)

for tempStr in counterName:
    temp_df = vectorDF[vectorDF['hardware_Counter'] == tempStr]
    vectorDF_ins[tempStr] = temp_df['event'].values

vectorDF_ins = vectorDF_ins[~vectorDF_ins.isin([np.nan,np.inf,-np.inf]).any(1)]

for str in counterName:
    total = scalarDF_ins[str].sum()
    if total == 0:
       counterName.remove(str)
print len(counterName)       
for str in counterName:
    total = vectorDF_ins[str].sum()
    if total == 0:
       counterName.remove(str)

df_per_ins = pd.DataFrame(columns = counterName)

resultDF = scalarDF.append(vectorDF)

for tempStr in counterName:
    temp_df = resultDF[resultDF['hardware_Counter'] == tempStr]
    df_per_ins[tempStr] = temp_df['event'].values

df_per_ins = df_per_ins[~df_per_ins.isin([np.nan,np.inf,-np.inf]).any(1)]

print df_per_ins.shape

df_split_ins = np.split(df_per_ins,[35],axis=1)
df_features_ins = df_split_ins[0]
df_labels_ins = df_split_ins[1]

print df_features_ins.shape
print df_labels_ins.shape
n_trees = [1,3,5,7,10,15,17,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,200]
#n_trees = [1,3,5,7,10,15,17,20,25,30,35,40,45,50]
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
    error_ins_arr = [0] * 2000
    mse_ins_arr = [0] * 100
    plt.figure()
    
    # Boundaries for plots
    rect_top = [0.1,0.65,0.8,0.2]
    rect_bot_left = [0.1,0.075,0.35,0.5]
    plt.figure(1,figsize= (9,9))
    
    ax = plt.axes(rect_top)
    cx = plt.axes(rect_bot_left)
    size = 0
    for rs in range(1,101):
        ###### Reapting the above process for the PER_INS values
        
        # Splitting the data for PER_INS counters
        df_features_ins_train, df_features_ins_test, df_labels_ins_train, df_labels_ins_test = train_test_split(df_features_ins,df_labels_ins,test_size = 0.30, random_state=rs ) 

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
        
        ## testing with 20% of data
        prediction_ins_test = rf2.predict(df_features_ins_test)
        
        #error_test = abs(prediction_test - df_labels_test['LABEL'])
        mse_ins = mean_squared_error(df_labels_ins_test['LABEL'], prediction_ins_test)
        mse_ins_arr[rs-1] = mse_ins 
    
        #Testing the model with both the train and test 
        prediction_ins = rf2.predict(df_features_ins)
        
        # Calculating the Error
        error_ins = abs(prediction_ins - df_labels_ins['LABEL'])
        print len(error_ins)
        for index,val in error_ins.items():
            error_ins_arr[index] = error_ins_arr[index] + val
            
        size = rs
    
############################################## Plotting the results ##############################################
    # For plotting table with imporatance values 
    w, h = 2, 15;
    Matrix = [[0 for x in range(w)] for y in range(h)]
    labelr = []
    for x in range(1,16):
        labelr.append(x)
    
    labelc = ['Hardware Counter Name ', 'Importance of INS Counter']
    
    sorted_dict = sorted(importance_abs_dict)
        
    name_counter= []    
    for key, value in sorted(importance_ins_dict.iteritems(), key=lambda (k,v): (v,k),reverse=True):
        # PAPI_TOT_INS counter present in only abs counters but not in per_ins values
        if key != 'PAPI_TOT_INS':    
            name_counter.append(key)
    cnt = 0
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            if j == 0 :
                Matrix[i][j] = name_counter[cnt]
            if j == 1 :
                Matrix[i][j] = round(importance_ins_dict[name_counter[cnt]]/size,4)
        cnt = cnt+1
    
    xList = range(0,2000)
    # Plotting error values for the INS hardware counter 
    ax.bar(xList,error_ins_arr)
    ax.set_title('Random Forest classifier with number of Trees={}'.format(n_trees[idx]))
    ax.set_ylabel('Error')
    ax.set_xlabel('Time')    
    val = max(error_ins_arr)
    ax.set_ylim([0,val*1.5])
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
    values = ['PER_INS']
    ypos = np.arange(len(values))
    rmse_idx = rmse_idx + 1;
    pdf.savefig()
    plt.close()

# Plotting bar graph for RMSE values for different values of the number of trees 
N = len(n_trees)
fig1,ex = plt.subplots()
ind = np.arange(N)
n_trees_Arr = np.asarray(n_trees)  
ex.plot(rmse_ins_error,'-o',ms=10,lw=2,alpha =0.7,mfc = 'orange')
ex.set_xticks(ind)
ex.set_xlabel('Number of trees')
ex.set_xticklabels(n_trees_Arr)
ex.grid()
pdf.savefig()   
plt.close()
pdf.close()