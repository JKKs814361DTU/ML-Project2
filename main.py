# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:30:19 2022

@author: s184361
"""
#from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import svd
from matplotlib.pyplot import (figure, plot, title, legend, boxplot, xticks, subplot, hist,
                               xlabel,ylabel, ylim, yticks, show,imshow,cm,colorbar)

from scipy.stats import zscore

# Imports the numpy and pandas package, then runs the data_prep code
from data_prep import *
# (requires data structures from ex. 2.1.1)


# Data attributes to be plotted
# %% Scatter plots

for i in [5]:
    for j in [5]:
        if i ==j:
            print('pass')
            pass
      

        # Make another more fancy plot that includes legend, class labels, 
        # attribute names, and a title.
        f = figure()
        title('South African Heart Disease')
        
        for c in range(C):
            # select indices belonging to class c:
            class_mask = y_CHD==c
            plot(X[class_mask,i], X[class_mask,j], '.',alpha=.3)
        
        legend(classNames_CHD)
        xlabel(attributeNames[i])
        ylabel(attributeNames[j])
        
        # Output result to screen
        show()

#%% Box plots
# We start with a box plot of each attribute
figure()
title('CHD: Boxplot')
boxplot(X2)
xticks(range(1,M+1), attributeNames, rotation=45)
X3 = np.delete(X, [9], 1)#No nominal data
X3=np.array(X3,dtype = np.float)
attributeNames_3 = np.delete(attributeNames, [9], 0)
X3[:,1] = X3[:,1]**0.5
#%%Standardized Box-plot

figure(figsize=(12,6),dpi =1200)
title('CHD: Boxplot (standarized), chd ==0')
boxplot(zscore(X3, ddof=1))
xticks(range(1,M), attributeNames_3, rotation=45)

#%% Histograms of all attributes.

figure(figsize=(14,9),dpi =1200)
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    fig = subplot(u,v,i+1)
    hist(X[:,i],bins = 50)
    xlabel(attributeNames[i])
    #ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('CHD: Histogram')
#%% Histograms of all atributes with normal dist
    '''
fig = figure(figsize=(14,7))
#figure.tight_layout(pad=3.0)
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):

    subplot(u,v-1,i+1)
    X=df[attributeNames[i]]
    df[attributeNames[i]].hist(bins=50, density=True)
    
    # Over the histogram, plot the theoretical probability distribution function:
    sd_attribute_Name=np.std(df[attributeNames[i]]) # standard deviation
    mean_attribute_Name=np.mean(df[attributeNames[i]]) # Mean
    
    x = np.linspace(X.min(), X.max(), 1000)
    pdf = stats.norm.pdf(x,loc=mean_attribute_Name,scale=sd_attribute_Name)
    plot(x,pdf,'.',color='red')
    
    # Plot info
    #plt.title(plt_title);
    plt.xlabel(attributeNames[i]);
    plt.ylabel('Probability Distribution');

'''
#%% Histograms of all attributes with potential outliers masked
class_mask = y_CHD==0

figure(figsize=(14,9))
m = [0, 1, 2,4,5,6]
for i in range(len(m)):
    subplot(1,len(m),i+1)

    hist(X3[class_mask ,m[i]],density=True)
    xlabel(attributeNames_3[m[i]])
    #ylim(0, 300) # Make the y-axes equal for improved readability
    if i>0: yticks([])
    if i==0: title('CHD: Histogram (selected attributes)') 


#%% PCA


# Subtract mean value from data
Y = X3 - np.ones((N,1))*X3.mean(axis=0)
Y = Y*(1/np.std(Y,0))#standardize
# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure(dpi=1200)
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#%% Plot PC

V = Vh.T    

# Project the centered data onto principal component space
#Z = Y @ V

z = U*S;
Z = z
# Indices of the principal components to be plotted
for i in [0]:#range(8):
 for j in [1]:#range(8):
    # Plot PCA of the data
    f = figure(dpi=1200)
    title('South African Heart Disease: PCA')
    #Z = array(Z)
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y_CHD==c
        plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    legend(classNames_CHD)
    xlabel('PC{0}'.format(i+1))
    ylabel('PC{0}'.format(j+1))
    
    # Output result to screen
    show()

#%% 

# First 3 PC explain roughly 90%
pcs = np.arange(0,4)
legendStrs = ['PC'+str(e+1) for e in pcs]

bw = .1
r = np.arange(1,M)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames_3)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('South African Heart Disease: PCA Component Coefficients')
plt.show()

# Inspecting the plot, we see that the 2nd principal component has large
# (in magnitude) coefficients for attributes sbp, alcohol and age. We can confirm
# this by looking at it's numerical values directly, too:
print('PC2:')
print(V[:,1].T)

#%%

sns.pairplot(df, hue="chd",diag_kind="hist")