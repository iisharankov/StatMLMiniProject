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
from sklearn.linear_model import LogisticRegression
import crossvalidation as cv

import warnings

df_train = pd.read_csv(const.TRAIN_DATASET)
df_test = pd.read_csv(const.TEST_DATASET)

#%%
#------ CROSS VALIDATION ------------

#define all inputs to use

all_inputs = ['Number words female', 'Total words', 'Number of words lead',
       'Difference in words lead and co-lead', 'Number of male actors', 'Year',
       'Number of female actors', 'Number words male', 'Gross',
       'Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead']


correlation = df_train.corr().round(2)


optimal_inputs = ['Number words male','Total words', 'Number of words lead',
       'Difference in words lead and co-lead', 'Number of male actors',
       'Number of female actors','Gross',
       'Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead']


#%%
def split_test_train(size,dataframe,output,x=None,**kwargs):
    """
    Divides data from pandas dataframe into two equally sized training and test sets. 
    
    """
    train = np.random.choice(df_train.index,size//2)
    movie_train = dataframe.iloc[train]
    movie_valid = dataframe.iloc[~train]
    y_train = movie_train[output].values.ravel()
    y_test = movie_valid[output].values.ravel()
    
    if x is None:
        x_train = movie_train.loc[:, df_train.columns != output]
        x_test = movie_valid.loc[:, df_train.columns != output]
   
    else:
       x_train = movie_train[x]
       x_test = movie_valid[x]
       

    return y_train, x_train, y_test, x_test


#%%
def ROC(method,r, x_train, y_train, x_test, y_test):
    """" Returnes the false femlae and true female rate given  test and train data, and a method """
    
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


#%%
def lda_crossvalidation(inputs, df_train,n):
    LDA = skl_da.LinearDiscriminantAnalysis()
    lda_enew,enew_vec= cv.n_crossvalidation(n,LDA,df_train,inputs,'Lead')
    #print('\n Estimated test error, LDA:', lda_enew)

    #learn model with all data 
    x_train = df_train[inputs]
    y_train = df_train['Lead']

    learned_lda = LDA.fit(x_train, y_train)
    accuracy = learned_lda.score(x_train,y_train)

    return lda_enew,enew_vec


def qda_crossvalidation(inputs, df_train,n):
    QDA = skl_da.QuadraticDiscriminantAnalysis() 
    output = 'Lead'
    y_train, x_train, y_test, x_test = split_test_train(1039,df_train,output,inputs)

    warnings.filterwarnings('ignore', 'Variables are collinear', )
    qda_enew, enew_vec = cv.n_crossvalidation(n,QDA,df_train,inputs,'Lead')
    #print('\n Estimated test error, QDA: ',qda_enew)
    learned_qda = QDA.fit(x_train, y_train)

    accuracy = learned_qda.score(x_train,y_train)
    #print('\nQDA, accuracy on training data: ',accuracy)
    return qda_enew, enew_vec


#%%

optimal_inputs = ['Number words female', 'Total words',
 'Difference in words lead and co-lead', 'Number of male actors',
 'Number of female actors', 'Number words male', 'Gross', 'Mean Age Male',
 'Mean Age Female' ,'Age Lead', 'Age Co-Lead']

qda_enew, enew_vec = qda_crossvalidation(optimal_inputs, df_train,1039)
print(f"the error for QDA when using optimal inputs: {qda_enew}")
df_train = pd.read_csv(const.TRAIN_DATASET)


qda_enew, qda_vec=qda_crossvalidation(optimal_inputs, df_train, 10)
lda_enew, lda_vec = lda_crossvalidation(optimal_inputs, df_train, 10)

#BOXPLOTS

logreg_vec = np.array( [[0.125],[0.0961538],[0.134615],[0.115385],[0.163462],[0.163462],[0.0576923],[0.134615],[0.134615],[0.15534]] )

kNN_vec = np.array([[0.18269231],[0.125],[0.18269231],[0.14423077],[0.22115385],[0.21153846],[0.19230769],[0.19230769],[0.20192308],[0.16504854]])

tree_vec = 1-np.array([0.81730769,0.83653846, 0.79807692, 0.75961538, 0.86538462, 0.79807692, 0.82692308, 0.81730769, 0.80769231, 0.76699029])

forest_vec = 1-np.array([0.84615385, 0.92307692, 0.82692308, 0.86538462, 0.83653846, 0.81730769,
 0.86538462, 0.79807692, 0.81730769, 0.88349515])

bagging_vec =1- np.array([0.79807692, 0.86538462, 0.79807692, 0.83653846, 0.79807692, 0.75961538,
 0.86538462, 0.80769231, 0.78846154, 0.84466019])

plt.boxplot([logreg_vec.flatten(),lda_vec, qda_vec, kNN_vec.flatten(), tree_vec, forest_vec, bagging_vec])
plt.xticks([1, 2, 3,4,5,6,7], ['logistic \n regression', 'LDA', 'QDA', 'kNN', 'class. \ntree', 'random \nforest', 'bagging'])
plt.show()



#%%
def feat_importance_plots(df_train):
    def female_lead(row):
        if row['Lead']=='Female':
            val = row['Number of words lead']+row['Number words female']
        else:
            val = row['Number words female']
        return val
    
    df_train['No. all words female'] = df_train.apply(female_lead, axis=1)
    df_train['No. all words male'] = df_train['Total words']-df_train['No. all words female']

    plt.scatter(df_train['Year'],df_train['No. all words male']/(df_train['No. all words male']+df_train['No. all words female']))
    plt.show()
    plt.scatter(df_train['No. all words male']/(df_train['No. all words male']+df_train['No. all words female']),df_train['Gross'])
    plt.show()
    
    female_words = sum(df_train['No. all words female'])
    male_words = sum(df_train['No. all words male'])
    print(f"fraction of  words spoken by female {female_words/(female_words+male_words)}")
        
#%%

feat_importance_plots(df_train)

prob3_inputs = ['Number words female','Number words male','Year','Gross']

QDA = skl_da.QuadraticDiscriminantAnalysis()  

rs = 0.01*(np.arange(-1,100)+1)
FF= np.zeros((len(rs),4))
TF = np.zeros((len(rs),4))
j = 0

for i in prob3_inputs:
    inputs = ['Number words female', 'Total words', 'Number of words lead',
       'Difference in words lead and co-lead', 'Number of male actors', 'Year',
       'Number of female actors', 'Number words male', 'Gross',
       'Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead']
    
    inputs.remove(i)
    error,evec = qda_crossvalidation(inputs, df_train,1039)
    print(f"error when omitting {i}: {error}")
    
    np.random.seed(1)
    y_train, x_train, y_test, x_test = split_test_train(1039,df_train,'Lead',inputs)
    
    
    for k in range(len(rs)): 
        FF[k,j],TF[k,j],error = ROC(QDA,rs[k], x_train, y_train, x_test, y_test)
        
        
    j+=1

plt.plot(FF[:,0],TF[:,0],label = 'No. words female')
plt.plot(FF[:,1],TF[:,1],label = 'No. words male')
plt.plot(FF[:,2],TF[:,2], label = 'Year')
plt.plot(FF[:,3],TF[:,3],label = 'Gross')

plt.legend()
plt.show()

#%%
#Error when including only one of each input in prob3_inputs


for i in prob3_inputs:
    inp =  np.array([i])
    qda_enew,evec = cv.crossvalidation(1039,QDA,df_train,inp,'Lead')

    print('\n Estimated test error for input ',i, ', QDA:', qda_enew)
