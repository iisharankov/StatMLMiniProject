#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:41:01 2021

@author: linasundqvist
"""
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)


import constants as const
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

import crossvalidation as cv

import warnings

df_train = pd.read_csv(const.TRAIN_DATASET)
df_test = pd.read_csv(const.TEST_DATASET)



#%%
#First approach: DIVIDE DATA INTO TEST AND VALIDATION

# 1) divide the train data into test and train of approximately equal size

def split_test_train(n,dataframe,output,x = None):
    np.random.seed(1)
    train = np.random.choice(df_train.index,n//2)
    movie_train = dataframe.iloc[train]
    movie_valid = dataframe.iloc[~train]
    y_train = movie_train[output].values.ravel()
    x_train = movie_train.loc[:, df_train.columns != output]
    y_test = movie_valid[output].values.ravel()
    x_test = movie_valid.loc[:, df_train.columns != output]

    return y_train, x_train, y_test, x_test

n = 1039
output = 'Lead'
y_train, x_train, y_test, x_test = split_test_train(n,df_train,output)

#%%
def ROC(method,r, x_train, y_train, x_test, y_test):
    
    warnings.filterwarnings('ignore', 'Variables are collinear', )
    model = method.fit(x_train, y_train)
    probability = model.predict_proba(x_test)
    
    female_index = list(model.classes_).index('Female')

    predict = np.where(probability[:,female_index] > r,'Female','Male')
    
    
    FF = np.sum( (predict == 'Female') & (y_test == 'Male'))
    TF = np.sum( (predict == 'Female') & (y_test == 'Female'))

    males = np.sum (y_test == 'Male')
    females = np.sum (y_test == 'Female')
    
    error = np.mean(predict != y_test)
    return FF/males,TF/females,error
    

QDA = skl_da.QuadraticDiscriminantAnalysis()
LDA = skl_da.LinearDiscriminantAnalysis()
rs = 0.01*(np.arange(-1,100)+1)

false_female_QDA = np.zeros(len(rs))
true_female_LDA = np.zeros(len(rs))
true_female_QDA = np.zeros(len(rs))
false_female_LDA = np.zeros(len(rs))
error = np.zeros(len(rs))
i =0
 

for r in rs:
    false_female_QDA[i], true_female_QDA[i], error[i] = ROC(QDA,r, x_train, y_train, x_test, y_test)
    false_female_LDA[i], true_female_LDA[i], error[i] = ROC(LDA,r, x_train, y_train, x_test, y_test)
    i+=1

plt.plot(false_female_QDA,true_female_QDA, label ='QDA')
plt.plot(false_female_LDA,true_female_LDA, label= 'LDA')

#for point in [5,20,50,90,99]:
    #plt.text(false_female[point],true_female[point], f"r={rs[point]:.2f}")
plt.xlabel('False female')
plt.ylabel('True female')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.grid()
plt.title('ROC curves for LDA and QDA')
plt.legend()

ind = np.argmin(error)
print(rs[ind],error[ind])

#%%
# LDA with confusion table
LDA = skl_da.LinearDiscriminantAnalysis()
model = LDA.fit(x_train,y_train)
y_predict_LDA = model.predict(x_test)

error = np.mean(y_predict_LDA != y_test)
print('\nLDA error: ',error)
print(pd.crosstab(y_predict_LDA,y_test))

#%%
# QDA with confusion table
QDA = skl_da.QuadraticDiscriminantAnalysis()    
model = QDA.fit(x_train,y_train)
y_predict_QDA= model.predict(x_test)

error = np.mean(y_predict_QDA != y_test)

print('\n QDA error: ', error)
print(pd.crosstab(y_predict_QDA,y_test))

#%%
#------ CROSS VALIDATION ------------

#define all inputs to use

all_inputs = ['Number words female', 'Total words', 'Number of words lead',
       'Difference in words lead and co-lead', 'Number of male actors', 'Year',
       'Number of female actors', 'Number words male', 'Gross',
       'Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead']


correlation = df_train.corr().round(2)

#remove number of words lead

optimal_inputs = ['Number words female', 'Total words',
 'Difference in words lead and co-lead', 'Number of male actors',
 'Number of female actors', 'Number words male', 'Gross', 'Mean Age Male',
 'Mean Age Female' ,'Age Lead', 'Age Co-Lead']

#%%
# LDA with cross validation
def lda_crossvalidation(inputs, df_train,n):
    lda_enew,enew_vec= cv.n_crossvalidation(n,LDA,df_train,inputs,'Lead')
    print('\n Estimated test error, LDA:', lda_enew)

    #learn model with all data 
    x_train = df_train[inputs]
    y_train = df_train['Lead']

    learned_lda = LDA.fit(x_train, y_train)
    accuracy = learned_lda.score(x_train,y_train)
    print(accuracy)
    
    return lda_enew,enew_vec


lda_crossvalidation(optimal_inputs, df_train,1039)
#%%
# QDA with cross validation
def qda_crossvalidation(inputs, df_train,n):
    output = 'Lead'
    y_train, x_train, y_test, x_test = split_test_train(1039,df_train,output)

    QDA = skl_da.QuadraticDiscriminantAnalysis()
    warnings.filterwarnings('ignore', 'Variables are collinear', )
    qda_enew, enew_vec = cv.n_crossvalidation(n,QDA,df_train,inputs,'Lead')
    print('\n Estimated test error, QDA: ',qda_enew)
    learned_qda = QDA.fit(x_train, y_train)

    accuracy = learned_qda.score(x_train,y_train)
    print('\nQDA, accuracy on training data: ',accuracy)
    return qda_enew, enew_vec


#%%
optimal_inputs = ['Number words female', 'Total words',
 'Difference in words lead and co-lead', 'Number of male actors',
 'Number of female actors', 'Number words male', 'Gross', 'Mean Age Male',
 'Mean Age Female' ,'Age Lead', 'Age Co-Lead']

df_train = pd.read_csv(const.TRAIN_DATASET)
qda_enew, enew_vec=qda_crossvalidation(optimal_inputs, df_train, 10)
plt.boxplot(enew_vec)
plt.show()