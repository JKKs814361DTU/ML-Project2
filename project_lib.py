# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:49:41 2022

@author: Jedrzej Konrad Kolbert s184361@student.dtu.dk
"""
import sklearn.metrics.cluster as cluster_metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model
from matplotlib.pyplot import contourf
from matplotlib import cm
from toolbox_02450.statistics import *
from toolbox_02450.similarity import *
from toolbox_02450.categoric2numeric import categoric2numeric
from toolbox_02450.bin_classifier_ensemble import BinClassifierEnsemble
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
def ANN_validate(X,y,h_list,cvf=10):
    ''' Validate ANN model using 'cvf'-fold cross validation.
        Find the optimal hiden units number (minimizing validation error) from 'h_list' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all h_list, MSE train&validation errors for all h_list.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        X       training data set
        y       vector of values
        h_list vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]

    #y = y.squeeze()
         
    # Parameters for neural network classifier

    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000
    errors= np.empty((cvf, len(h_list)))
    for k, (train_index, test_index) in enumerate(CV.split(X, y)):
        print('\nCrossvalidation fold (inner): {0}/{1}'.format(k+1,cvf)) 
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
    
        
        for n_hidden_units in h_list:

            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
            #print('Training model of type:\n\n{}\n'.format(str(model())))
            

               
            

            print("Hiden units number:",n_hidden_units)
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
            errors[k,n_hidden_units-min(h_list)]= (mse*len(y_test)/len(y)) # store error rate for current CV fold        
    opt_val_err = np.min(np.mean(errors,axis=0))
    opt_lambda = h_list[np.argmin(np.mean(errors,axis=0))]
    
    return opt_val_err, opt_lambda

def rlogr_validate(X,y,lambdas,cvf=10):
    ''' Validate rlr model using 'cvf'-fold cross validation.
        Find the optimal regularization factor (minimizing validation error) 
        from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all h_list, MSE train&validation errors for all h_list.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''

    # Fit regularized logistic regression model to training data to predict 
    # the type of wine
    
    #train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]

         
    # Parameters for neural network classifier

    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000
    errors= np.empty((cvf, len(lambdas)))
    for l, (train_index, test_index) in enumerate(CV.split(X, y)):
        print('\nCrossvalidation fold (inner): {0}/{1}'.format(l+1,cvf)) 
        
        #Test & train data
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
            
            mdl.fit(X_train, y_train)
    
            #y_train_est = mdl.predict(X_train).T
            y_test_est = mdl.predict(X_test).T
            
            #train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
            #test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)
            errors[l,k]=np.sum(y_test_est != y_test) / len(y_test)
        
            
    opt_val_err = np.min(np.mean(errors,axis=0))
    opt_lambda = h_list[np.argmin(np.mean(errors,axis=0))]
    return opt_val_err, opt_lambda