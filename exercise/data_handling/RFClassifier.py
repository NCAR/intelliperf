#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:01:39 2018

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

def modifyArr(errorArr):
    cnt =100
    errorArr = [x/cnt for x in errorArr]
    return errorArr


def rearrange(dataFrame,counterNameList):
    df_per_ins = pd.DataFrame(columns = counterNameList)
    for tempStr in counterNameList:
        temp_df = dataFrame[dataFrame['hardware_Counter'] == tempStr]
        df_per_ins[tempStr] = temp_df['event'].values
    df_per_ins = df_per_ins[~df_per_ins.isin([np.nan,np.inf,-np.inf]).any(1)]
    return df_per_ins

def splitData(df_per_ins):
     return np.split(df_per_ins,[43],axis=1)    

def splitTrainAndTest(df_features,df_labels,rs):
    return train_test_split(df_features,df_labels,test_size = 0.30, random_state=rs) 

def creating_plotting_model(n_trees,dfFeatures,dfLabels):
    
    pdf = PdfPages('ClubbDataMergePlot.pdf')

    rmse_idx = 0
    rmse_ins_error = [0] * len(n_trees)
    
    # Looping with different number of trees    
    for idx in range(len(n_trees)):
        
        ########## Variables #############
        importance_ins_dict = {}
        cnt = dfFeatures.shape[0]
        cnt = cnt +3
        error_ins_arr = [0] * 4400
        mse_ins_arr = [0] * 100
        plt.figure()
        print cnt
        # Boundaries for plots
        rect_top = [0.1,0.65,0.8,0.2]
        rect_bot_left = [0.1,0.075,0.35,0.5]
        plt.figure(1,figsize= (9,9))
        
        ax = plt.axes(rect_top)
        cx = plt.axes(rect_bot_left)
        size = 0
        print n_trees[idx]
        for rs in range(1,101):
            df_feature_train, df_features_test, df_labels_train, df_labels_test = train_test_split(dfFeatures,dfLabels,test_size = 0.30, random_state=rs )
            rf = RandomForestRegressor(n_estimators=n_trees[idx], random_state=rs)
            print df_feature_train.shape
            print df_labels_train.shape
            rf.fit(df_feature_train,np.ravel(df_labels_train,order = 'C'))
            
            importance_ins = list(zip(rf.feature_importances_,df_feature_train.columns))
            importance_ins.sort(reverse= True)
            
            
            for item in importance_ins:
                temp = item[1].replace("_per_ins","")
                if temp in importance_ins_dict.keys():
                    importance_ins_dict[temp] = importance_ins_dict[temp] + item[0]
                else:
                    importance_ins_dict[temp] = item[0]
            ## testing with 20% of data
            prediction_ins_test = rf.predict(df_features_test)
        
            #error_test = abs(prediction_test - df_labels_test['LABEL'])
            mse_ins = mean_squared_error(df_labels_test['LABEL'], prediction_ins_test)
            mse_ins_arr[rs-1] = mse_ins
            
            #Testing the model with both the train and test 
            prediction_ins = rf.predict(dfFeatures)
        
            # Calculating the Error
            error_ins = abs(prediction_ins - dfLabels['LABEL'])
            print('Length of the %d',len(error_ins))
            for index,val in error_ins.items():
                error_ins_arr[index] = error_ins_arr[index] + val
                
            size = rs
        
############################################## Plotting the results ##############################################
        print 'asdasd'
        # For plotting table with imporatance values 
        w, h = 2, 15;
        Matrix = [[0 for x in range(w)] for y in range(h)]
        labelr = []
        for x in range(1,16):
            labelr.append(x)
            
        labelc = ['Hardware Counter Name ', 'Importance of INS Counter']
            
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
    
        xList = range(0,4400)
        # Plotting error values for the INS hardware counter
        error_ins_arr = modifyArr(error_ins_arr)
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

        
          
def main():
    # Reading the data from csv file 
    scalerPath = '../../data/WACCM_imp_sol_scaler.slope.labelled.csv'
    scalarDF = load_data_from_csv(scalerPath)
    
    vectorPath = '../../data/WACCM_imp_sol_vector.slope.labelled.csv'
    vectorDF = load_data_from_csv(vectorPath)
    
    psradPath = '../../data/mg2/PSrad.exe.codeblocks.fused.any.any.any.slope.labelled .csv'
    psradDF = load_data_from_csv(psradPath)
    
    wetdepaPath = '../../data/wetdepa_driver_v0.labelled.csv'
    wetdepaDF = load_data_from_csv(wetdepaPath)
    
    clubbPath = '../../data/clubb.labelled.csv'
    clubbDF = load_data_from_csv(clubbPath)
    
    # array of columns to delete
    arr = ['module_sub_routine','id','time']
    
    # Droping the columns 
    scalarDF = drop_columns(scalarDF,arr)
    vectorDF = drop_columns(vectorDF,arr)
    psradDF = drop_columns(psradDF,arr)
    wetdepaDF = drop_columns(wetdepaDF,arr)
    clubbDF = drop_columns(clubbDF,arr)
    clubbDF
    counter_name = get_CounterNames(vectorDF)
    counterNameList = remove_ABS_Counter(counter_name)
    
    # adding all the dataframes
    resultDF = append_DataFrames(scalarDF,vectorDF)
    resultDF = append_DataFrames(resultDF, psradDF)
    resultDF = append_DataFrames(resultDF, wetdepaDF)
    resultDF = append_DataFrames(resultDF,clubbDF)
    
    df_per_ins = rearrange(resultDF,counterNameList)
    dfSplit = splitData(df_per_ins)
    dfFeatures = dfSplit[0]
    dfLabels = dfSplit[1]
    #n_trees = [1,3,5]   
    n_trees = [1,3,5,7,10,15,17,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,200]

    
    creating_plotting_model(n_trees,dfFeatures,dfLabels)
    
    print "out"
    
    return 0
          
if __name__ == "__main__":
    main()
