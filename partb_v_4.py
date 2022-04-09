# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 11:51:28 2022

@author: Jedrzej Konrad Kolbert s184361@student.dtu.dk
"""
# Regression, part b:
import matplotlib.pyplot as plt
import numpy as np

import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net,rlr_validate
from scipy import stats
from data_prep import *
from project_lib import *

def plot_models():
    
    plt.figure(figsize=(10,10))
    y_rlr = X_test.data.numpy()@ w_rlr[:,k]; y_true = y_test.data.numpy()[:,0]
    y_est = y_test_est.data.numpy();
    axis_range = [np.min([y_rlr, y_true])-1,np.max([y_rlr, y_true])+1]
    plt.plot(axis_range,axis_range,'k--')
    plt.plot(y_true, y_est,'og',alpha=.25)
    plt.plot(y_true, y_rlr,'ob',alpha=.25)
    plt.plot(y_true, y_true.mean()*np.ones(len(y_true)),'or',alpha=.25)
    plt.legend(['Perfect estimation','ANN','rlr','baseline'])
    plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
    plt.ylim(axis_range); plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()

    plt.show()
#%%
# Normalize data
mask_r= [0,1,2,3,5,6,7]
y = X[:,[8]] .astype(float)
X = X[:,mask_r].astype(float)
attributeNames_r = attributeNames[mask_r]
#%%
C = 2

# Normalize data
X = stats.zscore(X)
                
## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
do_pca_preprocessing = False
if do_pca_preprocessing:
    Y = stats.zscore(X,0)
    U,S,V = np.linalg.svd(Y,full_matrices=False)
    V = V.T
    #Components to be included as features
    k_pca = 10
    X = X @ V[:,:k_pca]
N, M = X.shape

# K-fold crossvalidation
K = 5                   # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Parameters for neural network classifier
n_hidden_units = 5      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000


# Parameters for rlr 
lambdas = np.power(10.,range(-5,9))
w_rlr = np.empty((M,K))



# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss


errors = [] # make a list for storing generalizaition error in each loop
h_unit = []
opt_val_E = []
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
Table = np.empty((K,6))

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

    #################################Regularized linear reg##################################
    opt_val_err_rlr, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K)
    print(opt_lambda)
    
    Table[:,3] = opt_lambda
    Table[:,4] = opt_val_err_rlr
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    
    print("####################OPTIMIZING HIDEN UNITS##########################")
    opt_val_err, n_hidden_units = ANN_validate(X_test,y_test,[3,4],cvf=5)
    h_unit.append(n_hidden_units)
    opt_val_E.append(opt_val_err)
    #print(n_hidden_units)
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))  
    print('Training model of type:\n\n{}\n'.format(str(model())))
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold 
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
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

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))

# Display the Optimal units across folds
summaries_axes[0].bar(np.arange(1, K+1), np.squeeze(np.asarray(h_unit)), color=color_list)
summaries_axes[0].set_xlabel('Fold')
summaries_axes[0].set_xticks(np.arange(1, K+1))
summaries_axes[0].set_ylabel('Optimal no of units')
summaries_axes[0].set_title('Optimal no of units')

# Display the Optimal units across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(opt_val_E)), color=color_list)
summaries_axes[1].set_xlabel('Fold')
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('Optimal no of units')
summaries_axes[1].set_title('Optimal no of units')






#############################Create the table#################################


Table[:,0] = np.arange(K)
Table[:,1] = h_unit
Table[:,2] = opt_val_E
Table[:,5] = Error_test_nofeatures[:,0]


