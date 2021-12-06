#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

import constants as const
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.discriminant_analysis as skl_da
import crossvalidation as cv
import itertools
import warnings

#%%
def optimal_inputs_discrimanant_analysis(inputs):
    QDA = skl_da.QuadraticDiscriminantAnalysis()
    LDA = skl_da.LinearDiscriminantAnalysis()

    #loop through all combinations and find the best one 
 
    for L in range(0, len(inputs)+1):
        for subs in itertools.combinations(inputs, L):
            subset = np.asarray(subs)
            if len(subset) > 1:
                qda_enew,evec = cv.n_crossvalidation(10,QDA,df_train,subset,'Lead')
                lda_enew,evec = cv.n_crossvalidation(10,LDA,df_train,subset,'Lead')
            
                if lda_enew < 0.1:
                    print('\n Estimated test error for input ',subset,', LDA:', lda_enew)
                if qda_enew < 0.10:
                    print('\n Estimated test error for input ',subset, ', QDA:', qda_enew)
                
    
#%%

df_train = pd.read_csv(const.TRAIN_DATASET)
df_test = pd.read_csv(const.TEST_DATASET)

correlation = df_train.corr().round(2)

all_inputs = ['Number words female', 'Total words', 'Number of words lead',
       'Difference in words lead and co-lead', 'Number of male actors', 'Year',
       'Number of female actors', 'Number words male', 'Gross',
       'Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead']

LDA = skl_da.LinearDiscriminantAnalysis()
QDA = skl_da.QuadraticDiscriminantAnalysis()
warnings.filterwarnings('ignore', 'Variables are collinear', )

qda_enew, qda_vec = cv.n_crossvalidation(1039,QDA,df_train,all_inputs,'Lead')
lda_enew, lda_vec = cv.n_crossvalidation(1039,LDA,df_train,all_inputs,'Lead')

print(f" LDA test error with all inputs: {lda_enew}")
print(f" QDA test error with all inputs: {qda_enew}")
#optimal_inputs_discrimanant_analysis(all_inputs)

optimal_inputs = ['Number words female', 'Total words',
 'Difference in words lead and co-lead', 'Number of male actors',
 'Number of female actors', 'Number words male', 'Gross', 'Mean Age Male',
 'Mean Age Female' ,'Age Lead', 'Age Co-Lead']


qda_enew, qda_vec = cv.n_crossvalidation(1039,QDA,df_train,optimal_inputs,'Lead')
lda_enew, lda_vec = cv.n_crossvalidation(1039,LDA,df_train,optimal_inputs,'Lead')

print(f" LDA test error with optimalo inputs: {lda_enew}")
print(f" QDA test error with optimal inputs: {qda_enew}")

#%%
#Production model

x_train = df_train[optimal_inputs]
y_train = df_train["Lead"]
x_test = df_test[optimal_inputs]
model = QDA.fit(x_train, y_train)

Y_predict = model.predict(x_test)
print((Y_predict== 'Male').sum(),(Y_predict== 'Female').sum())
