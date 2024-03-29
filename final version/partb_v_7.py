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
from toolbox_02450 import train_neural_net, draw_neural_net,rlr_validate ,ttest_twomodels
from scipy import stats 
from data_prep import * 
from project_lib import * 
 
def plot_models(): 
     
    plt.figure(figsize=(10,10)) 
    y_rlr = X_test_rlr@ w_rlr[:,k]; y_true = y_test.data.numpy()[:,0] 
    y_est = y_test_est.data.numpy(); 
    axis_range = [np.min([y_rlr, y_true])-1,np.max([y_rlr, y_true])+1] 
    plt.plot(axis_range,axis_range,'k--') 
    plt.plot(y_true, y_est,'og',alpha=.25) 
    plt.plot(y_true, y_rlr,'ob',alpha=.25) 
    plt.plot(y_true, y_true.mean()*np.ones(len(y_true)),'or',alpha=.25) 
    plt.legend(['Perfect estimation','ANN','rlr','baseline']) 
    plt.title('Cross-validation fold k='+str(k+1)) 
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
#lambdas = np.power(10.,range(-5,9))
lambdas = range(0,100) 

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

#%%
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
    
    ###
    plt.figure(k, figsize=(12,8))
    plt.subplot(1,2,1)
    plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
    plt.xlabel('Regularization factor')
    plt.ylabel('Mean Coefficient Values')
    plt.grid()
    # You can choose to display the legend, but it's omitted for a cleaner 
    # plot, since there are many attributes
    #legend(attributeNames[1:], loc='best')
    
    plt.subplot(1,2,2)
    plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
    plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
    plt.xlabel('Regularization factor')
    plt.ylabel('Squared error (crossvalidation)')
    plt.legend(['Train error','Validation error'])
    plt.grid()
    ################################ANN########################################
    # Extract training and test set for current CV fold, convert to tensors 
    X_train = torch.Tensor(X[train_index,:]) 
    y_train = torch.Tensor(y[train_index]) 
    X_test = torch.Tensor(X[test_index,:]) 
    y_test = torch.Tensor(y[test_index]) 
    
    print("####################OPTIMIZING HIDEN UNITS##########################")
    # find optimal value of hiden units
    opt_val_err, n_hidden_units = ANN_validate(X_test,y_test,range(1,10),cvf=K)
    h_unit.append(n_hidden_units) #update optimal number o units
    
    # Define the model 
    model = lambda: torch.nn.Sequential( 
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units 
                        torch.nn.ReLU(),#torch.nn.Tanh(),   # 1st transfer function, 
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
    y_ANN = np.append(y_ANN,y_test_est.data.numpy()) # ANN prediction
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean 
    errors.append(mse) # store error rate for current CV fold  
    # Display the learning curve for the best net in the current fold 
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k]) 
    h.set_label('CV fold {0}'.format(k+1)) 
    summaries_axes[0].set_xlabel('Iterations') 
    summaries_axes[0].set_xlim((0, max_iter)) 
    summaries_axes[0].set_ylabel('Loss') 
    summaries_axes[0].set_title('Learning curves') 
     
    #plot_models() 
     
     
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
 
# Display the Optimal units across folds 
summaries_axes[0].bar(np.arange(1, K+1), np.squeeze(np.asarray(h_unit)), color=color_list) 
summaries_axes[0].set_xlabel('Fold') 
summaries_axes[0].set_xticks(np.arange(1, K+1)) 
summaries_axes[0].set_ylabel('Optimal no of units') 
summaries_axes[0].set_title('Optimal no of units') 
 
# Display the Optimal units across folds 
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list) 
summaries_axes[1].set_xlabel('Fold') 
summaries_axes[1].set_xticks(np.arange(1, K+1)) 
summaries_axes[1].set_ylabel('Avg error for CV fold') 
summaries_axes[1].set_title('ANN') 
 
 
 
 
 
 
#############################Create the table################################# 
#%%
 
Table[:,0] = np.arange(K)+1 
Table[:,1] = h_unit 
Table[:,2] = errors
Table[:,4] = Error_test_rlr[:,0]
Table[:,5] = Error_test_nofeatures[:,0] 
#%%
import os
Table = Table.round(decimals=3, out=None) 
df2 = pd.DataFrame(Table,columns=['k', 'h', 'E_ANN','lambda','E_rlr','E_base'])
cwd = os.getcwd()
path = cwd + "/part_b_result.csv"
df2.to_csv("part_b_result.csv",index = False)
df3 = pd.DataFrame(np.stack((y_True, y_baseline, y_ANN, y_rlr),axis=-1),columns=['y_true', 'y_baseline', 'y_ANN','y_rlr'])
df3.to_csv("part_b_prediction.csv",index = False)
# Display the learning curve for the best net in the current fold 
h, = summaries_axes[0].plot(learning_curve, color=color_list[k]) 
h.set_label('CV fold {0}'.format(k+1)) 
summaries_axes[0].set_xlabel('Iterations') 
summaries_axes[0].set_xlim((0, max_iter)) 
summaries_axes[0].set_ylabel('Loss') 
summaries_axes[0].set_title('Learning curves') 

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
print('ANN vs rlr',ttest_twomodels(y_True, y_rlr, y_ANN, alpha=0.05, loss_norm_p=2))