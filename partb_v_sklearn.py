# -*- coding: utf-8 -*- 
""" 
Created on Sun Mar 27 11:51:28 2022 
 
@author: Jedrzej Konrad Kolbert s184361@student.dtu.dk 
""" 
# Regression, part b: 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns #function for plotting models

import torch 
from sklearn import model_selection
from sklearn.neural_network import MLPRegressor
 
from toolbox_02450 import train_neural_net, draw_neural_net,rlr_validate ,ttest_twomodels
from scipy import stats 
from data_prep import * 
from project_lib import * 
 
def plot_models(): 
     
    plt.figure(figsize=(10,10)) 
    y_rlr = X_test_rlr@ w_rlr[:,k]; y_true = y_test
    y_est = y_test_est; 
    axis_range = [np.min([y_rlr, y_true])-1,np.max([y_rlr, y_true])+1] 
    plt.plot(axis_range,axis_range,'k--') 
    plt.plot(y_true, y_est,'og',alpha=.25) 
    plt.plot(y_true, y_rlr,'ob',alpha=.25) 
    plt.plot(y_true, y_true.mean()*np.ones(len(y_true)),'or',alpha=.25) 
    plt.legend(['Perfect estimation','ANN','rlr','baseline']) 
    plt.title('Cross-validation fold'+str(k+1)) 
    plt.ylim(axis_range); plt.xlim(axis_range) 
    plt.xlabel('True value') 
    plt.ylabel('Estimated value') 
    plt.grid() 
 
    plt.show() 
#%% 
# Normalize data 
mask_r= [0,1,2,4,5,6,7,8] 
y = X[:,[3]] .astype(float) 
X = X[:,mask_r].astype(float) 

attributeNames_r = attributeNames[mask_r] 
#%%
 
# Normalize data 
X = stats.zscore(X) 
X_rlr = np.concatenate((np.ones((X.shape[0],1)),X),1)               
N, M = X.shape 
 
# K-fold crossvalidation 
K = 10                   # only three folds to speed up this example 
CV = model_selection.KFold(K, shuffle=True) 
 
# Parameters for neural network classifier 
#n_hidden_units = 10      # number of hidden units 
n_replicates = 1        # number of networks trained in each k-fold 
max_iter = 30000 
 
 
# Parameters for rlr  
lambdas = np.power(10.,range(-5,9)) 
w_rlr = np.empty((M+1,K)) 
 
 
 
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
y_ANN = np.array([])
y_True =np.array([])
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):  
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))     
     
    #################################Baseline############################################### 
    X_train = (X[train_index,:]) 
    y_train = y[train_index][:,0] 
    X_test = (X[test_index,:]) 
    y_test = y[test_index][:,0] 
    # Compute mean squared error without using the input data at all 
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0] 
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0] 
    
    y_baseline= np.append(y_baseline, np.ones(len(y_test))*y_test.mean()) #Baseline prediction
    
    y_True =np.append(y_True,y_test) #True test values
    
    #################################Regularized linear reg################################## 
    X_train = (X_rlr[train_index,:])
    X_test_rlr = (X_rlr[test_index,:]) 
    
    #Find the optimal lambda
    opt_val_err_rlr, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K) 
    
    Table[k,3] = opt_lambda #save opt lambda in table
    # Estimate weights for the optimal value of lambda, on entire training set 
    lambdaI = opt_lambda * np.eye(M+1)
    lambdaI[0,0] = 0 # Do no regularize the bias term 
    Xty = X_train.T @ y_train 
    XtX = X_train.T @ X_train 
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()  #find weights
    
    # Compute mean squared error with regularization with optimal lambda 
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0] 
    Error_test_rlr[k] = np.square(y_test-X_test_rlr @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0] 
    y_rlr = np.append(y_rlr,X_test_rlr @ w_rlr[:,k]) #rlr prediction test 
    ################################ANN########################################
    # Extract training and test set for current CV fold, convert to tensors 
    X_train = (X[train_index,:])
    
    print("####################OPTIMIZING HIDEN UNITS##########################")
    # find optimal value of hiden units
    params = {'hidden_layer_sizes': range(1,10)}
    
    clf = model_selection.GridSearchCV(MLPRegressor(max_iter=50000), params, cv=K)
    clf.fit(X_train, y_train)
    Table[k,1] = clf.best_params_.get('hidden_layer_sizes')
    y_test_est = clf.predict(X_test)
    y_ANN = np.append(y_ANN,clf.predict(X_test))
    
    # Determine errors and errors 
    se = (y_test_est-y_test)**2 # squared error 
    y_ANN = np.append(y_ANN,y_test_est) # ANN prediction
    mse = (sum(se)/len(y_test)) #mean 
    errors.append(mse) # store error rate for current CV fold  
    
    #find best parameters
    print('Logistic Regression parameters: ', clf.best_params_) 
     
     
    plot_models() 
     
     
# Display the MSE across folds 
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list) 
summaries_axes[1].set_xlabel('Fold') 
summaries_axes[1].set_xticks(np.arange(1, K+1)) 
summaries_axes[1].set_ylabel('MSE') 
summaries_axes[1].set_title('Test mean-squared-error') 
'''     
print('Diagram of best neural net in last fold:') 
weights = [net[i].weight.data.numpy().T for i in [0,2]] 
biases = [net[i].bias.data.numpy() for i in [0,2]] 
tf =  [str(net[i]) for i in [1,2]] 
draw_neural_net(weights, biases, tf, attribute_names=attributeNames_r) 
''' 
# Print the average classification error rate 
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4))) 

 
# Setup figure for display of learning curves and error rates in fold 
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5)) 
 
 
#All test plot

plt.figure(figsize=(10,10)) 

axis_range = [np.min([y_rlr, y_True])-1,np.max([y_rlr, y_True])+1] 
plt.plot(axis_range,axis_range,'k--') 
plt.plot(y_True, y_ANN,'og',alpha=.5) 
plt.plot(y_True, y_rlr,'+b',alpha=.5) 
plt.plot(y_True, y_True.mean()*np.ones(len(y_True)),'*r',alpha=.5) 
plt.legend(['Perfect estimation','ANN','rlr','baseline']) 
plt.title('All CV-folds') 
plt.ylim(axis_range); plt.xlim(axis_range) 
plt.xlabel('True value') 
plt.ylabel('Estimated value') 
plt.grid() 
plt.savefig('all_test_partb.pdf') 
plt.show()  
 
 
 
#############################Create the table################################# 
#%%
 
Table[:,0] = np.arange(K) 
Table[:,2] = errors
Table[:,4] = Error_test_rlr[:,0]
Table[:,5] = Error_test_nofeatures[:,0] 
 
#############################Statistics#######################################

#%% Baseline vs rlr
#plot zi's
zdata={'Model': ['Baseline']*len(y_baseline)+['rlr']*len(y_rlr)+['ANN']*len(y_ANN),'Z': np.append(np.append(y_baseline,y_rlr),y_ANN)}
zdf=pd.DataFrame(zdata)
plt.figure(4)
sns.boxplot(x='Model',y='Z', data=zdf)

print('Baseline vs rlr',ttest_twomodels(y_True, y_baseline, y_rlr, alpha=0.05, loss_norm_p=2))
print('Baseline vs ANN',ttest_twomodels(y_True, y_baseline, y_ANN, alpha=0.05, loss_norm_p=2))
print('ANN vs rlr',ttest_twomodels(y_True, y_ANN, y_rlr, alpha=0.05, loss_norm_p=2))