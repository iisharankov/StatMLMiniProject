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
from sklearn.ensemble import BaggingClassifier

import crossvalidation as cv

import warnings

df_train = pd.read_csv(const.TRAIN_DATASET)
df_test = pd.read_csv(const.TEST_DATASET)


#%%
#First approach: DIVIDE DATA INTO TEST AND VALIDATION

# 1) divide the train data into test and train of approximately equal size

n = df_train.size

np.random.seed(1)
train = np.random.choice(df_train.index,2*n//3)
movie_train = df_train.iloc[train]
movie_valid = df_train.iloc[~train]



y_train = movie_train['Lead'].values.ravel()
x_train = movie_train.loc[:, df_train.columns != 'Lead']

y_test = movie_valid['Lead'].values.ravel()
x_test = movie_valid.loc[:, df_train.columns != 'Lead']

#%%
# LDA with crosstable
LDA = skl_da.LinearDiscriminantAnalysis()
model = LDA.fit(x_train,y_train)
y_predict_LDA = model.predict(x_test)

error = np.mean(y_predict_LDA != y_test)
print('\nLDA error: ',error)
print(pd.crosstab(y_predict_LDA,y_test))

#%%
# QDA with crosstable
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

inputs = ['Number words female','Total words',
       'Difference in words lead and co-lead', 'Number of male actors', 'Year',
       'Number of female actors','Number words male', 'Gross',
       'Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead']

#%%
# LDA with cross validation

lda_enew = cv.n_crossvalidation(1039,LDA,df_train,inputs,'Lead')
print('\n Estimated test error, LDA:', lda_enew)

#learn model with all data 
x_train = df_train[inputs]
y_train = df_train['Lead']

learned_lda = LDA.fit(x_train, y_train)
#%%
# QDA with cross validation
QDA = skl_da.QuadraticDiscriminantAnalysis()
warnings.filterwarnings('ignore', 'Variables are collinear', )
qda_enew = cv.n_crossvalidation(1039,QDA,df_train,inputs,'Lead')
print('\n Estimated test error, QDA: ',qda_enew)
learned_lda = QDA.fit(x_train, y_train)
