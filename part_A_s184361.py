# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 23:09:13 2022

@author: Jedrzej Konrad Kolbert s184361@student.dtu.dk
"""

# -*- coding: utf-8 -*- 
""" 
Created on Sun Mar 27 11:51:28 2022 
 
@author: Jedrzej Konrad Kolbert s184361@student.dtu.dk 
""" 
# Regression, part a: 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns #function for plotting models

import torch 
from sklearn import model_selection 
from toolbox_02450 import train_neural_net, draw_neural_net,rlr_validate ,ttest_twomodels
from scipy import stats 
from data_prep import * 
from project_lib import * 
 

#%% 
# Normalize data 
mask_r= [0,1,2,4,5,6,7,8] 
y = X[:,[3]] .astype(float)
y= y[:,0]
X = X[:,mask_r].astype(float) 
attributeNames_r = attributeNames[mask_r] 
#%%
 
# Normalize data 
X = stats.zscore(X) 
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
         
N, M = X.shape 
 

 
# Parameters for neural network classifier 
#n_hidden_units = 10      # number of hidden units 
n_replicates = 1        # number of networks trained in each k-fold 
max_iter = 30000 
 
k=0 
K=1
# Parameters for rlr  
lambdas = np.power(10.,range(-5,9)) 
#lambdas = np.logspace(-1, 3, 50)

w = np.empty((M,K)) 
 
 
 
# Setup figure for display of learning curves and error rates in fold 
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5)) 
# Make a list for storing assigned color of learning curve for up to K=10 
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue'] 
 
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss 
 
 
errors = [] # make a list for storing generalizaition error in each loop 
h_unit = [] #optimal number of hidden units

Error_train_rlr = np.empty((K,1)) 
Error_test_rlr = np.empty((K,1)) 
Error_train_nofeatures = np.empty((K,1)) 
Error_test_nofeatures = np.empty((K,1)) 
Table = np.empty((K,6)) 
y_baseline = np.array([])
y_rlr = np.array([])


     


#################################Regularized linear reg################################## 


#Find the optimal lambda
opt_val_err_rlr, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, 10) 

# Estimate weights for the optimal value of lambda, on entire training set 
lambdaI = opt_lambda * np.eye(M)
lambdaI[0,0] = 0 # Do no regularize the bias term 
Xty = X.T @ y 
XtX = X.T @ X 
w[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()  #find weights

# Compute mean squared error with regularization with optimal lambda 
Error_train_rlr[k] = np.square(y-X @ w[:,k]).sum(axis=0)/y.shape[0] 
Error_test_rlr[k] = np.square(y-X @ w[:,k]).sum(axis=0)/y.shape[0] 
y_rlr = np.append(y_rlr,X @ w[:,k]) #rlr prediction test 

###
plt.figure(k, figsize=(12,8))
plt.subplot(1,2,1)
plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
plt.xlabel('Regularization factor')
plt.ylabel('Mean Coefficient Values')
plt.grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
plt.legend(attributeNames_r[1:], loc='best')

plt.subplot(1,2,2)
plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
plt.xlabel('Regularization factor')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Validation error'])
plt.grid()
plt.savefig('part_A.pdf')
