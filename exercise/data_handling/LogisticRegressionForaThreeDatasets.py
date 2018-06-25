#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:48:54 2018

@author: uppala
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split


pdf = PdfPages('RFPlotWithLogisticRegression.pdf')

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

# Relative path to the Data file
csvfile= '../../data/mg2/PSrad.exe.codeblocks.fused.any.any.any.slope.labelled .csv'

#Converting CSV data to dataframe
dataDF = pd.read_csv(csvfile,delimiter = ';',names = ["module_sub_routine","id","hardware_Counter","time","event"])
dataDF = dataDF.drop(['module_sub_routine','id','time'],axis=1)

counter_name = vectorDF['hardware_Counter']
counter_name = counter_name.unique()
counterName = []

for str in counter_name:
    if "_per_ins" in str:
        counterName.append(str)
    elif "LABEL" == str:
        counterName.append(str)

resultDF = scalarDF.append(vectorDF)

resultDF = resultDF.append(dataDF)

print resultDF.shape
df_per_ins = pd.DataFrame(columns = counterName)

for tempStr in counterName:
    temp_df = resultDF[resultDF['hardware_Counter'] == tempStr]
    df_per_ins[tempStr] = temp_df['event'].values
 
df_per_ins = df_per_ins[~df_per_ins.isin([np.nan,np.inf,-np.inf]).any(1)]

df_split_ins = np.split(df_per_ins,[43],axis=1)
df_features_ins = df_split_ins[0]
df_labels_ins = df_split_ins[1]


C_Val = [0.01,0.1,0.5,1,2,5,7,10,50,100,1000,10000, 10000000000000, 50000000000000000]

rmse_idx = 0
rmse_abs_error = [0] * len(C_Val)
rmse_ins_error = [0] * len(C_Val)


for idx in range(len(C_Val)):
    
    error_ins_arr = [0] * 3000
    mse_ins_arr = [0] * 3000
    fig,ax = plt.subplots()
    
    plt.figure(1,figsize= (9,9))
    size = 0
    
    
    for rs in range(1,101):
        model = LogisticRegression(penalty = 'l2',C=C_Val[idx])

        df_features_ins_train, df_features_ins_test, df_labels_ins_train, df_labels_ins_test = train_test_split(df_features_ins,df_labels_ins,test_size = 0.05, random_state=rs) 

        model.fit(df_features_ins_train,df_labels_ins_train)

        prediction_ins_test = model.predict(df_features_ins_test)
        
        #error_test = abs(prediction_test - df_labels_test['LABEL'])
        mse_ins = mean_squared_error(df_labels_ins_test['LABEL'], prediction_ins_test)
        mse_ins_arr[rs-1] = mse_ins 

        prediction_ins = model.predict(df_features_ins)
        
        # Calculating the Error
        error_ins = abs(prediction_ins - df_labels_ins['LABEL'])
        
        
        for index,val in error_ins.items():
            error_ins_arr[index] = error_ins_arr[index] + val
        
        size=rs
    
    xList = range(0,3000)
    rmse_ins_error[rmse_idx] = np.sqrt(np.mean(mse_ins_arr))
    # Plotting error values for the INS hardware counter 
    ax.bar(xList,error_ins_arr)
    ax.set_title('Logistic Classifier with value={}'.format(C_Val[idx]))
    ax.set_ylabel('Error')
    ax.set_xlabel('Time')    
    val = max(error_ins_arr)
    rmse_idx = rmse_idx + 1
    ax.set_ylim([0,val*1.5])
    pdf.savefig()
    plt.close()
    
# Plotting bar graph for RMSE values for different values of the number of trees 
N = len(C_Val)
fig1,ex = plt.subplots()
ind = np.arange(N)
width = 0.35
n_trees_Arr = np.asarray(C_Val)  
ex.bar(ind,rmse_ins_error,width,color='g')
ex.set_title('RMSE plot for different number of trees')
ex.set_xticks(ind)
ex.set_xlabel('Number of trees')
ex.set_ylabel('Error')
val = max(rmse_ins_error)
ex.set_ylim([0,val*1.5])
ex.set_xticklabels(n_trees_Arr)
ex.autoscale_view()
pdf.savefig()   
plt.close()
pdf.close()