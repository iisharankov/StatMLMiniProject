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

import crossvalidation as cv

df_train = pd.read_csv(const.TRAIN_DATASET)
df_final = pd.read_csv(const.TEST_DATASET)

# 1) divide the train data into test and train of approximately equal size

n = df_train.size

np.random.seed(1)
train = np.random.choice(df_train.index,n//2)
movie_train = df_train.iloc[train]
movie_test = df_train.iloc[~train]



y_train = movie_train['Lead'].values.ravel()
x_train = movie_train.loc[:, df_train.columns != 'Lead']

y_test = movie_train['Lead'].values.ravel()
x_test = movie_test.loc[:, df_train.columns != 'Lead']

#%%
# Basic LDA Stuff

LDA = skl_da.LinearDiscriminantAnalysis()
model = LDA.fit(x_train,y_train)
y_predict_LDA = model.predict(x_test)

error = np.mean(y_predict_LDA != y_test)
print('\nLDA error: ',error)
print(pd.crosstab(y_predict_LDA,y_test))
#%%
# LDA with cross validation
#define all inputs to use
inputs = df_final.keys()
Enew = cv.n_crossvalidation(10,LDA,df_train,inputs,'Lead')
print(Enew)

#%%
QDA = skl_da.QuadraticDiscriminantAnalysis()
model = QDA.fit(x_train,y_test)
y_predict_QDA= model.predict(x_test)

error = np.mean(y_predict_QDA != y_test)
print('\n QDA error: ', error)
print(pd.crosstab(y_predict_QDA,y_test))

