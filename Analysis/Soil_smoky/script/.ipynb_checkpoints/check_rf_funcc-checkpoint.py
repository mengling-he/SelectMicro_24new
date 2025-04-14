#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 00:00:19 2024

@author: menglinghe
"""
import sys
sys.path.append('../../../Code')
import loadData 
import RunML
import RunML_continue
import FS
import metric

import pandas as pd
import numpy as np
import random


data,burn_label,un_label,duration_label,ASVs,df=loadData.loadSoilData("../data/count_table/merge_proportion_asv_smoky_moutain.csv")    

dict_cm_rf = RunML_continue.ML_model_SCV(data, burn_label, "RF", SMOTE=True,k=5)
print('Random Forest')
print(dict_cm_rf)
#print(metric.metric_tb(dict_cm_rf['y_true'],dict_cm_rf['y_pred']))


dict_cm_svm = RunML_continue.ML_model_SCV(data, burn_label, "SVM", SMOTE=True,k=5)
print('SVM')
print(dict_cm_svm)
#print(metric.metric_tb(dict_cm_svm['y_true'],dict_cm_svm['y_pred']))

data_subset = {"AllFeatures":data }
cls = ["RF","SVM", "CatBoost","NB"]
dict_cm = RunML_continue.runClassifier_FScompare(data_subsets= data_subset,y= burn_label,N=30,classifiers=cls,SMOTE=True)
print(metric.metric_sum(dict_cm))
