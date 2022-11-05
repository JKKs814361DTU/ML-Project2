# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:59:57 2022

@author: s184361
"""

import numpy as np
import pandas as pd

# Load the csv data using the Pandas library
filename = 'data.csv'
df = pd.read_csv(filename)
df = df.drop(['row.names'], axis=1)#drop the row names since it is not needed
raw_data = df.values  


# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(0, 10) 
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

# extracting the strings for each sample from the famhist loaded from the csv:
classLabels = raw_data[:,4]
# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames_Family = np.unique(classLabels)
classNames_CHD = np.array(['CHD Absent', 'CHD Present'])
# We can assign each type of Iris class with a number by making a
# Python dictionary as so:
classDict_Family = dict(zip(classNames_Family,range(len(classNames_Family))))
classDict_CHD = dict(zip(classNames_CHD,range(len(classNames_CHD))))
#This is the class index vector y:
y_family = np.array([classDict_Family[cl] for cl in classLabels])
y_CHD = X[:,-1]
X2=X
X2[:,4] = y_family
# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames_CHD)