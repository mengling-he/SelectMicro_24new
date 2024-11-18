#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 00:00:19 2024

@author: menglinghe
"""
import sys
sys.path.append('../Code')
import loadData 
import RunML
import RunML_continue
import FS
import metric

import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pickle
import matplotlib.pyplot as plt



# Read the data back from the file
with open('../Data/PMI/subset_bact_4taxa_noenv.pkl', 'rb') as file:
    data_subset_4taxa = pickle.load(file)


#for dataset  in data_subset_4taxa:
#    data_subset = dataset
#    for datatype, subset in data_subset.items():
#        print(np.shape(subset))

# read the response variable
    
df_sample= pd.read_csv('../Data/PMI/bact.n.otu.noenv.csv') 
y = df_sample.iloc[:, 0].values 
# Define the threshold
y_threshold = 2500
# Categorize the series based on the threshold
y = np.where(y > y_threshold, 'LONG', 'SHORT')
#print(y)
    

# model
iter =30
cls = ["RF","SVM"]
targetLabel = y

taxlabels = ['OTU', 'class', 'order', 'phylum']
dict_cm_list = []
for i , dataset  in enumerate(data_subset_4taxa):
    print(f"Analysis for {taxlabels[i]}")
    dict_cm = RunML_continue.runClassifier_FScompare(data_subsets= dataset,y= targetLabel,N=iter,classifiers=cls)
    dict_cm_list.append(dict_cm)
    
# Save dictionary to a pickle file
with open('../results/PMI_bact_prediction.pkl', 'wb') as pickle_file:
    pickle.dump(dict_cm_list, pickle_file)

    
    
    
 
