#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:08:58 2018

@author: uppala
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error

def load_data_from_csv(CSV_PATH):
    return pd.read_csv(CSV_PATH,delimiter = ';',
                       names = ["module_sub_routine","id","hardware_Counter","time","event"])

def drop_columns(dataFrame,list_col):
    return dataFrame.drop(list_col,axis=1)

def get_CounterNames(dataFrame):
    counter_name = dataFrame['hardware_Counter']
    return counter_name.unique()

def remove_ABS_Counter(counterNames):
    counterName = []
    for str in counterNames:
        if "_per_ins" in str:
            counterName.append(str)
        elif "LABEL" == str:
            counterName.append(str)
    return counterName

def append_DataFrames(dataFrameA,dataFrameB):
    return dataFrameA.append(dataFrameB)


def rearrange(dataFrame,counterNameList):
    df_per_ins = pd.DataFrame(columns = counterNameList)
    for tempStr in counterNameList:
        temp_df = dataFrame[dataFrame['hardware_Counter'] == tempStr]
        df_per_ins[tempStr] = temp_df['event'].values
    df_per_ins = df_per_ins[~df_per_ins.isin([np.nan,np.inf,-np.inf]).any(1)]
    return df_per_ins

def modifyArr(errorArr):
    cnt =100
    errorArr = [x/cnt for x in errorArr]
    return errorArr

def splitData(df_per_ins):
    df_split = np.split(df_per_ins,[43],axis=1)  
    return df_split[0],df_split[1]
def splitTrainAndTest(df_features,df_labels,rs):
    return train_test_split(df_features,df_labels,test_size = 0.20, random_state=rs) 


def creatingModelPlot(trainDF,testDF,n_trees):
    rmse_idx = 0
    rmse_testArr = [0] * len(n_trees)
    rmse_clubbArr = [0] * len(n_trees)
    
    for idx in range(len(n_trees)):
        mse_testArr = [0] * 100
        mse_clubbArr = [0] * 100
        
        for rs in range(1, 11):
            
            featuresTrainDF, labelTrainDF = splitData(trainDF)
    
            featuresTrainDF, featuresTestDF,labelTrainDF, labelTestDF = splitTrainAndTest(featuresTrainDF,labelTrainDF,rs)
            
            
            rf = RandomForestRegressor(n_estimators=n_trees[idx],random_state=rs)
            
        
            rf.fit(featuresTrainDF,np.ravel(labelTrainDF,order = 'C'))
            
            
            predictionTest = rf.predict(featuresTestDF)
            
            errorTest = mean_squared_error(predictionTest,labelTestDF)
            mse_testArr[rs-1] = errorTest
            
            featuresDevDF, labelDevDF = splitData(testDF)
            
            prediction = rf.predict(featuresDevDF)
            error = mean_squared_error(prediction,labelDevDF)
            mse_clubbArr[rs-1] = error
        rmse_testArr[rmse_idx] = np.mean(mse_testArr)
        rmse_clubbArr[rmse_idx] = np.mean(mse_clubbArr)
        rmse_idx = rmse_idx+1
        
    return rmse_testArr,rmse_clubbArr


def main():
    # array of columns to delete
    arr = ['module_sub_routine','id','time']
    plt.figure()
    # Reading the data from csv file 
    scalerPath = '../../data/WACCM_imp_sol_scaler.slope.labelled.csv'
    scalarDF = load_data_from_csv(scalerPath)
    scalarDF = drop_columns(scalarDF,arr)
    
    vectorPath = '../../data/WACCM_imp_sol_vector.slope.labelled.csv'
    vectorDF = load_data_from_csv(vectorPath)
    vectorDF = drop_columns(vectorDF,arr)
    
    psradPath = '../../data/mg2/PSrad.exe.codeblocks.fused.any.any.any.slope.labelled .csv'
    psradDF = load_data_from_csv(psradPath)
    psradDF = drop_columns(psradDF,arr)
    
    wetdepaPath = '../../data/wetdepa_driver_v0.labelled.csv'
    wetdepaDF = load_data_from_csv(wetdepaPath)
    wetdepaDF = drop_columns(wetdepaDF,arr)
    
    clubbPath = '../../data/clubb.labelled.csv'
    clubbDF = load_data_from_csv(clubbPath)
    clubbDF = drop_columns(clubbDF,arr)

    
    counter_name = get_CounterNames(vectorDF)
    counterNameList = remove_ABS_Counter(counter_name)
    
    scalarDF = rearrange(scalarDF,counterNameList)
    vectorDF = rearrange(vectorDF,counterNameList)
    psradDF = rearrange(psradDF,counterNameList)
    wetdepaDF = rearrange(wetdepaDF,counterNameList)
    clubbDF = rearrange(clubbDF,counterNameList)
    #n_trees= [1,10,20]
    n_trees = [1,3,5,7,10,15,17,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

    trainDataArr = [psradDF,scalarDF,vectorDF,wetdepaDF]
    
    resultDF = pd.DataFrame(columns = counterNameList)
    
    rmseTest = [0] * len(trainDataArr)
    rmseClubb = [0] * len(trainDataArr)
    
    for i in range(len(trainDataArr)):
        resultDF = append_DataFrames(resultDF,trainDataArr[i])
        rmseTest[i] , rmseClubb[i]= creatingModelPlot(resultDF,clubbDF,n_trees)
    
    pdf = PdfPages('RFClassifierClubDataPlot.pdf')     
    
    
    N = len(n_trees)
    
    fig,ax1 = plt.subplots()
    
    ind = np.arange(N)
    
    n_trees_Arr = np.asarray(n_trees)
    
    for i in range(len(trainDataArr)):
        ax1.plot(rmseClubb[i],'-o',ms=10,lw=2,alpha =0.7)
    
    ax1.legend(['psradDF','scalarDF','vectorDF','wetdepaDF'],loc = 'upper right')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(n_trees_Arr)
    ax1.grid()
    ax1.title.set_text('Clubb Data')
    pdf.savefig()
    plt.close()
    fig,ax2 = plt.subplots()
    
    for i in range(len(trainDataArr)):
        ax2.plot(rmseTest[i],'-o',ms=10,lw=2,alpha =0.7)
    
    ax2.legend(['psradDF','scalarDF','vectorDF','wetdepaDF'],loc = 'upper right')
    ax2.set_xticks(ind)
    ax2.set_xlabel('Number of trees')
    ax2.set_xticklabels(n_trees_Arr)
    ax2.grid()    
    ax2.title.set_text('Test Data')    
    
    pdf.savefig()
    plt.close()
    pdf.close()

if __name__ == "__main__":
    main()
