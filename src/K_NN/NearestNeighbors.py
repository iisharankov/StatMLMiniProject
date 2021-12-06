import pandas as pd
import numpy as np
import os
import sys
import sklearn.neighbors as skl_nb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

p = os.path.abspath('..')
sys.path.insert(1,p)

import constants as const

train_all = pd.read_csv(const.TRAIN_DATASET)
test_all = pd.read_csv(const.TEST_DATASET)

#------------------------omitting total words and gross
X = train_all[["Number words female", 
               "Number of words lead", 
               "Difference in words lead and co-lead", 
               "Number of male actors", 
               "Year", 
               "Number of female actors", 
               "Number words male", 
               "Mean Age Male", 
               "Mean Age Female", 
               "Age Lead", 
               "Age Co-Lead"]]
y = train_all["Lead"]


#------------------------best k? and validation error with n_fold=1039

K = np.arange(1,50)
n_fold = 1039
n_fold = 10

kf = KFold(n_splits=n_fold, random_state=1, shuffle=True)

miss = np.zeros(len(K))
best_miss = []

for train_index, test_index in kf.split(X):

    X_tr, X_val = X.iloc[train_index], X.iloc[test_index]
    y_tr, y_val = y.iloc[train_index], y.iloc[test_index]
    
    for z, k in enumerate(K):
        model = skl_nb.KNeighborsClassifier(n_neighbors = k)
        model.fit(X_tr, y_tr)
        pred_class = model.predict(X_val)
        err = np.mean(pred_class != y_val)
        miss[z] += err
   
miss /= n_fold
best_k = K[miss == min(miss)]


print('The lowest error was: err = ' + str(round(100*min(miss), 1)) + '%')
print('The k with the lowest error was: k = ' + str(best_k[0]))

y1, y2 = 0.17, 0.30
plt.plot(K, miss, label = "validation error as function of k")
plt.vlines(best_k[0], y1, y2, 'r', label = "minimal error at k = " 
           + str(best_k[0]))
plt.title("Cross validation error for k-NN")
plt.xlabel('k')
plt.ylabel('Validation error')
plt.ylim(y1, y2)
plt.legend()
plt.show()


#------------------------cross validation for the best k (k=7) with n_fold=10
n_fold = 10

models = []
models.append(skl_nb.KNeighborsClassifier(n_neighbors = best_k[0]))

miss_best = np.zeros((n_fold, len(models)))
kf = KFold(n_splits=n_fold, random_state=1, shuffle=True)

for i, (train_index, test_index) in enumerate(kf.split(X)):

    X_tr, X_val = X.iloc[train_index], X.iloc[test_index]
    y_tr, y_val = y.iloc[train_index], y.iloc[test_index]
    
    for m in range(np.shape(models)[0]):
        model = models[m]
        model.fit(X_tr, y_tr)
        pred_class = model.predict(X_val)
        err = np.mean(pred_class != y_val)
        miss_best[i,m] = err

print(miss_best)

plt.boxplot(miss_best)
plt.title("Cross validation error for kNN")
plt.xlabel('k_NN')
plt.ylabel('Validation error')
plt.show()
















