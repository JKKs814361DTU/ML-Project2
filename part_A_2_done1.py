# exercise 8.1.1 for part A 


from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from scipy.io import loadmat
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from scipy import stats 

import sklearn.linear_model as lm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
Prepare Data
"""
data = pd.read_csv ('data.csv')
data = data.drop(columns='adiposity')
data = data.drop(columns='row.names')
data['famhist'].replace('Present', 1,inplace=True)
data['famhist'].replace('Absent', 0,inplace=True)

"""
Create Heatmap
"""

heat_map = sns.heatmap(data.isnull(), yticklabels = False, cbar = True, cmap = "PuRd", vmin = 0, vmax = 1)
plt.show()
sns.set_style("whitegrid")
sns.countplot(data = data, x = 'chd', palette = 'husl')
plt.show()


X = data.drop({'chd'}, axis = 1)
y = data['chd']
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 121)


"""
Logistic regression is created here
"""

logreg = LogisticRegression(C=1/10)
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)

classification_report(y_test, predictions)

acc=sum((y_test==predictions)*1)/len(y_test)
from sklearn.metrics import plot_confusion_matrix
fig, ax = plt.subplots(figsize=(13, 13))
plot_confusion_matrix(logreg, X_test, y_test,cmap=plt.cm.Blues,ax=ax)


print('\n'+classification_report(y_test, predictions))



"""
ANN is created here
"""
X = data.iloc[:, 0:8].values
y = data.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X = stats.zscore(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 10)


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 100, epochs = 150)

from keras.models import load_model

classifier.save('Heart_disease_model.h5') #Save trained ANN
#classifier = load_model('Heart_disease_model.h5')  #Load trained ANN

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))
plt.show()
sns.heatmap(cm,annot=True)
plt.savefig('heatmap_ANN.png')

"""
Baseline is created here
"""

largest_class_y_train = np.argmax(np.array([len(y_train)-sum(y_train),sum(y_train)]))

Error_test_base_clas = np.sum(largest_class_y_train != y_test.squeeze()) / len(y_test.squeeze())

print("Baseline",1-Error_test_base_clas)


"""
Single Layer Cross Validation
"""

K = 10
M = 10
# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=K-1

internal_cross_validation = K    

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

# Standardize outer fold based on training set, and save the mean and standard
# deviations since they're part of the model (they would be needed for
# making new predictions) - for brevity we won't always store these in the scripts
# mu[k, :] = np.mean(X_train[:, 1:], 0)
# sigma[k, :] = np.std(X_train[:, 1:], 0)

# X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
# X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 

Xty = X_train.T @ y_train
XtX = X_train.T @ X_train

# Compute mean squared error without using the input data at all
Error_train_nofeatures = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
Error_test_nofeatures = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

# Estimate weights for the optimal value of lambda, on entire training set
lambdaI = opt_lambda 
#lambdaI[0,0] = 0 # Do no regularize the bias term
w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
# Compute mean squared error with regularization with optimal lambda
Error_train_rlr = np.square(y_train-X_train @ w_rlr).sum(axis=0)/y_train.shape[0]
Error_test_rlr= np.square(y_test-X_test @ w_rlr).sum(axis=0)/y_test.shape[0]

# # Estimate weights for unregularized linear regression, on entire training set
# w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
# # Compute mean squared error without regularization
# Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
# Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
# # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
# #m = lm.LinearRegression().fit(X_train, y_train)
# #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
# #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

# Display the results for the last cross-validation fold
if k == K-1:
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

# To inspect the used indices, use these print statements
#print('Cross validation fold {0}/{1}:'.format(k+1,K))
#print('Train indices: {0}'.format(train_index))
#print('Test indices: {0}\n'.format(test_index))

    # k+=1
    
plt.show()
# Display results
print('Linear regression without feature selection for Single Layer Cross Validation:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression for Single Layer Cross Validation:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))






"""
Two level is created here
"""

X = np.concatenate((np.ones((data.shape[0],1)),data),1)
M = np.size(data.columns)

M = M+1

from sklearn import model_selection
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0

for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
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
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1
    
plt.show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')


