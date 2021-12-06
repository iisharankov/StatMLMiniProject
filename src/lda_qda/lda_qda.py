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

qda_enew, qda_vec = cv.n_crossvalidation(1039,QDA,df_train,all_inputs,'Lead')
lda_enew, lda_vec = cv.n_crossvalidation(1039,LDA,df_train,all_inputs,'Lead')

print(f" LDA test error with all inputs: {lda_enew}")
print(f" QDA test error with all inputs: {qda_enew}")
#optimal_inputs_discrimanant_analysis(all_inputs)

optimal_inputs = ['Number words female', 'Total words',
 'Difference in words lead and co-lead', 'Number of male actors',
 'Number of female actors', 'Number words male', 'Gross', 'Mean Age Male',
 'Mean Age Female' ,'Age Lead', 'Age Co-Lead']

<<<<<<< HEAD
=======
qda_enew, enew_vec = qda_crossvalidation(optimal_inputs, df_train,1039)
print(f"the error for QDA when using optimal inputs: {qda_enew}")
df_train = pd.read_csv(const.TRAIN_DATASET)


qda_enew, qda_vec=qda_crossvalidation(optimal_inputs, df_train, 10)
lda_enew, lda_vec = lda_crossvalidation(optimal_inputs, df_train, 10)

#BOXPLOTS
logreg_vec = np.array([0.125, 0.0961538, 0.134615, 0.115385, 0.163462, 0.163462, 0.0576923, 0.134615, 0.134615, 0.15534])
kNN_vec = np.array([0.18269231, 0.125, 0.18269231, 0.14423077, 0.22115385, 0.21153846, 0.19230769, 0.19230769, 0.20192308, 0.16504854])
tree_vec = np.array([0.18269231, 0.16346154, 0.20192308, 0.24038462, 0.13461538, 0.20192308, 0.17307692, 0.18269231, 0.19230769, 0.23300971])
forest_vec = np.array([0.15384615, 0.07692308, 0.17307692, 0.13461538, 0.16346154, 0.18269231, 0.13461538, 0.20192308, 0.18269231, 0.11650485])
bagging_vec = np.array([0.20192308, 0.13461538, 0.20192308, 0.16346154, 0.20192308, 0.24038462, 0.13461538, 0.19230769, 0.21153846, 0.15533981])

plt.boxplot([logreg_vec, lda_vec, qda_vec, kNN_vec, tree_vec, forest_vec, bagging_vec])
plt.xticks([1, 2, 3, 4, 5, 6, 7],
           ['logistic \n regression', 'LDA', 'QDA', 'kNN', 'class. \ntree', 'random \nforest', 'bagging'])
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
    ratio_male_words = 100 * (df_train['No. all words male']/(df_train['No. all words male']+df_train['No. all words female']))

    plt.subplot(211)
    plt.scatter(df_train['Year'], ratio_male_words, marker='.')
    plt.title("a) Percentage of script spoken by males vs. time")
    plt.xlabel("Year")
    plt.ylabel("Words spoken by males (%)")
    plt.grid()
    plt.subplot(212)

    plt.scatter(ratio_male_words, df_train['Gross'], marker='.')
    plt.title("b) Effect of male actors on film income")
    plt.xlabel("Words spoken by males (%)")
    plt.ylabel("Gross product of film")
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    female_words = sum(df_train['No. all words female'])
    male_words = sum(df_train['No. all words male'])
    print(f"fraction of  words spoken by female {female_words/(female_words+male_words)}")
        
#%%

feat_importance_plots(df_train)
raise Exception

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
>>>>>>> 5dd1d401138f07b2a6f0eca4188c1c6e34ec9471

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

