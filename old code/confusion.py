# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 00:59:41 2022

@author: Jedrzej Konrad Kolbert s184361@student.dtu.dk
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
fn='class_prediction.csv'
df = pd.read_csv(fn)

raw_data = df.values
y_true = raw_data[:,1]
y_DTC = raw_data[:,3]
y_rlr = raw_data[:,4]

tn, fp, fn, tp = confusion_matrix(y_true, y_DTC).ravel()

Recall = tp/(tp+fn)
Precision = tp/(tp+fp)
Sepcitivity =tn/(tn+fp)
#%%
tn, fp, fn, tp = confusion_matrix(y_true, y_rlr).ravel()

Recall = tp/(tp+fn)
Precision = tp/(tp+fp)
