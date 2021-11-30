# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 08:25:28 2021

@author: steli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_ms

import os
import sys

p = os.path.abspath('https://github.com/iisharankov/StatMLMiniProject/tree/main/src/data/train.cvs')
sys.path.insert(1,p)

import constants as const

train_all = pd.read_csv(const.TRAIN_DATASET)
# test_all = pd.read_csv(const.TEST_DATASET)

# Parameter List
params = ['Number words female',#0
          'Total words',#1
          'Number of words lead',#2
          'Difference in words lead and co-lead',#3
          'Number of male actors',#4
          'Year',#5
          'Number of female actors',#6
          'Number words male',#7
          'Gross',#8
          'Mean Age Male',#9
          'Mean Age Female',#10
          'Age Lead',#11
          'Age Co-Lead',#12
          ]

X = train_all[[params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[10],params[11],params[12]]]
y = train_all['Lead']

n_fold=10

model =skl_lm.LogisticRegression(solver='liblinear',max_iter=100000)#as part of the tuning solver = liblinear gives highest accuracy
misclassification = np.zeros(n_fold)

cv = skl_ms.KFold(n_splits = n_fold, random_state=1, shuffle=True)

prediction_mat =[]
all_y =[]
for i,(train_index, val_index) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        misclassification[i] = np.mean(prediction != y_val)
        
        all_y = np.append(all_y,y_val)
        
        cl = model.classes_
        test_prediction = model.predict_proba(X_val)
        prediction_mat = np.append(prediction_mat,np.where(test_prediction[:,0]>=0.5,cl[0],cl[1])) #there is no profound reason not to choose the calssic r=0.5
        
print('confusion matrix:\n')
print(pd.crosstab(prediction_mat, all_y),'\n')
        
err = np.mean(misclassification)

print(f'Error rate for logistic regression: {err*100:.3f}%')
accuracy = 1 - err
print(f'Accuracy of the model: {accuracy*100:.3f}%')    
plt.boxplot(misclassification)
plt.title('cross validation error for logistic regression')
plt.xlabel('LogReg')
plt.ylabel('Validation')
plt.show()    
    
