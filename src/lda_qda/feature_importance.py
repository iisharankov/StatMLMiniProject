#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:36:59 2021

@author: linasundqvist
"""

"""
Created on Mon Nov 15 14:41:01 2021

@author: linasundqvist
"""
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)
import constants as const
import crossvalidation as cv
import numpy as np
import pandas as pd
import sklearn.discriminant_analysis as skl_da
import itertools
import lda_qda
import warnings
import sklearn.neighbors as skl_nb

warnings.filterwarnings('ignore', 'Variables are collinear', )


df_train = pd.read_csv(const.TRAIN_DATASET)
df_test = pd.read_csv(const.TEST_DATASET)
#%%


inputs = ['Number words female', 'Total words', 'Number of words lead',
       'Difference in words lead and co-lead', 'Number of male actors', 'Year',
       'Number of female actors', 'Number words male', 'Gross',
       'Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead']

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
#what is the error by omitting certain inputs?
LDA = skl_da.LinearDiscriminantAnalysis()
QDA = skl_da.QuadraticDiscriminantAnalysis()
knn = skl_nb.KNeighborsClassifier(7)
prob3_inputs = ['Number words female','Number words male', 'Year','Gross']

for i in prob3_inputs:
    inputs = ['Total words','Number words female','Number of words lead',
       'Difference in words lead and co-lead', 'Number of male actors', 'Year',
       'Number of female actors', 'Number words male', 'Gross',
       'Mean Age Male','Mean Age Female', 'Age Lead', 'Age Co-Lead']
    
    #inputs.remove(i)
    enew,evec = cv.crossvalidation(1039,QDA,df_train,inputs,'Lead')
    print( np.mean(enew),i)
    
#%%
# what is the rror when only including some inputs?
for i in prob3_inputs:
    inp =  np.array([i])
    qda_enew,evec = cv.crossvalidation(1039,QDA,df_train,inp,'Lead')

    print('\n Estimated test error for input ',i, ', QDA:', qda_enew)

            