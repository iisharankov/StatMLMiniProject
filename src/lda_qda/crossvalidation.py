#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:27:38 2021

@author: linasundqvist
"""
import numpy as np

def method_error(method,x_train, y_train, x_valid, y_valid):
    model = method.fit(x_train, y_train)
    prediction = model.predict(x_valid)
    error = np.mean(prediction != y_valid)
    return error

def n_crossvalidation(n,method,train_data,inputs,output):
   k = len(train_data)
   all_indices = np.arange(0,k)
   perm_indices = np.random.permutation(all_indices)
   n_split = np.array_split(perm_indices,n)
   e_vec = np.zeros(n)
   i = 0
   
   for validation_indices in n_split:
       validation = train_data.iloc[validation_indices]
       ind_train = np.delete(train_data.index,validation_indices, axis=0)
       training = train_data.iloc[ind_train]
       
       
       
       
       if len(inputs) <=1:
           x_val = validation[inputs].values.reshape(-1, 1)
           x_train = training[inputs].values.reshape(-1, 1)
       else:
           x_train = training[inputs]
           x_val = validation[inputs]
           
       y_val = validation[output].values.ravel()
       y_train = training[output].values.ravel()
       
       e_vec[i]=method_error(method,x_train,y_train, x_val, y_val)
       i+=1
       
   return np.mean(e_vec),e_vec
    

    
