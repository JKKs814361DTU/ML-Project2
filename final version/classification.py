# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:34:31 2022

@author: Jedrzej Konrad Kolbert s184361@student.dtu.dk
"""

# Regression, part b: 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import torch 
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from toolbox_02450 import train_neural_net, rlr_validate 
from scipy import stats 
from data_prep import * 
from project_lib import * 


#%% 
# Format data
mask= [0,1,2,3,4,5,6,7,8] 
y = np.uint8(y_CHD)
X = X[:,mask].astype(float) 
attributeNames = attributeNames[mask] 
#%%
 
# Normalize data 
X = stats.zscore(X) 
X_rlr = np.concatenate((np.ones((X.shape[0],1)),X),1)
      
N, M = X.shape 
 
# K-fold crossvalidation 
K = 10                   # only three folds to speed up this example 
CV = model_selection.KFold(K, shuffle=True) 
 
 
# Parameters for rlr  
 
lambdas = np.logspace(-5, 5, 11) #np.logspace(-3, 3, 20)
w_rlr = np.empty((M+1,K)) 
 

#Create used variables
 
Error_test_nofeatures = np.empty((K,1)) 
Table = np.empty((K,6)) 
y_baseline = np.array([])
y_rlr = np.array([])
y_DTC = np.array([])
y_True = np.array([])
w_est = np.array([])
aN= np.array([])
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):  
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))     
     
    #################################Baseline############################################### 
    X_train = (X[train_index,:]) 
    y_train = y[train_index]
    X_test = (X[test_index,:]) 
    y_test = y[test_index] 
    # Compute mean squared error without using the input data at all
    #mdl = LogisticRegression(penalty='l2', C=1e-16 )
    #mdl.fit(np.ones((len(y_train),1)), y_train)
    y_test_est = round(np.sum(y_test)/len(y_test))*np.ones(len(y_test))#mdl.predict(np.ones((len(y_train),1))).T
    y_baseline= np.append(y_baseline, y_test_est)
    y_True = np.append(y_True, y_test)
    #w_est = mdl.coef_[0]
    Error_test_nofeatures[k] = np.sum(y_test_est != y_test) / len(y_test)
    
    #################################Regularized logistic reg################################## 

    
    params = {'C': lambdas}
    
    clf = model_selection.GridSearchCV(LogisticRegression(), params, cv=K)
    clf.fit(X_train, y_train)
    Table[k,3] = clf.best_params_.get('C')
    y_test_est = clf.predict(X_test)
    y_rlr = np.append(y_rlr,clf.predict(X_test))
    Table[k,4] = np.sum(y_test_est != y_test) / len(y_test)
    
    mdl=LogisticRegression(penalty='l2', C=1/10 )

    mdl.fit(X_train, y_train)
    w_est = np.append(w_est,mdl.coef_[0]) 
    aN = np.append(aN,attributeNames)
    #find best parameters
    print('Logistic Regression parameters: ', clf.best_params_) # Now it displays all the parameters selected by the grid search

    ################################Decision Tree########################################
    
    # Fit regression tree classifier, Gini split criterion, no pruning
    dtc = DecisionTreeClassifier(criterion='gini')
    
    parameters = {'max_depth':range(1,10)}
    clf = model_selection.GridSearchCV(dtc, parameters,cv=K)
    clf.fit(X_train,y_train)
    Table[k,1] = clf.best_params_.get('max_depth')
    y_test_est = clf.predict(X_test)
    y_DTC = np.append(y_DTC,clf.predict(X_test))
    Table[k,2] = np.sum(y_test_est != y_test) / len(y_test)
    print('Decision Tree parameters: ', clf.best_params_) # Now it displays all the parameters selected by the grid search

 
#############################Create the table################################# 
 
 
Table[:,0] = np.arange(K)+1

Table[:,5] = Error_test_nofeatures[:,0] 
Table = Table.round(decimals=3, out=None) 

Table = Table.round(decimals=3, out=None) 
#%%
df2 = pd.DataFrame(Table,columns=['k', 'depth', 'E_DTC','lambda','E_log','E_base'])
df2.to_csv("classifiaction.csv")

df3 = pd.DataFrame(np.stack((y_True, y_baseline, y_DTC, y_rlr),axis=-1),columns=['y_true', 'y_baseline', 'y_DTC','y_rlr'])
df3.to_csv("class_prediction.csv")
#############################Statistics#######################################

#%% Baseline vs rlr

#plot zi's
zdata={'Model': ['Baseline']*len(y_baseline)+['rlr']*len(y_rlr)+['DTC']*len(y_DTC),'Prediction avg': np.append(np.append(y_baseline,y_rlr),y_DTC)}
zdf=pd.DataFrame(zdata)
plt.figure(4)
sns.barplot(x='Model',y='Prediction avg', data=zdf)
print('Baseline vs rlr')
print(mcnemar(y_True, y_baseline, y_rlr, alpha=0.05))
print('######################################')
print('Baseline vs DTC')
print(mcnemar(y_True, y_baseline, y_DTC, alpha=0.05))
print('######################################')
print('DTC vs rlr')
print(mcnemar(y_True, y_DTC, y_rlr, alpha=0.05))


########################################Q5############################################
#%%
'''
mdl=LogisticRegression(penalty='l2', C=1/10 )

mdl.fit(X_train, y_train)

y_train_est = mdl.predict(X_train).T
y_test_est = mdl.predict(X_test).T

train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

#w_est = mdl.coef_[0] 
coefficient_norm = np.sqrt(np.sum(w_est**2))

wdata={'Attribute': attributeNames.tolist(), 'weight': w_est}
zdf=pd.DataFrame(zdata)
plt.figure(4,figsize=(10,5))
plt.barh(attributeNames,w_est)
plt.xlabel('Attribte weight')
plt.savefig('weights_logistics.pdf')
'''
#%%
plt.figure(4,figsize=(10,5))
wdata={'Attributes': aN.tolist(),'weights': w_est}
wdf=pd.DataFrame(wdata)
plt.figure(4)
sns.barplot(x='Attributes',y='weights', data=wdf)
plt.savefig('weights_logistics.pdf')
