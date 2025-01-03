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

ASVs = np.array(ASVs)

data=FS.relative_abundance(data)

yList= np.column_stack((burn_label,un_label,duration_label))# y list is a 2D array, each column is a response outcome

print(yList[:5])
      
for i in range(yList.shape[1]):
    print(pd.Series(yList[:,i]).value_counts())
    
 
y_arraylist=[burn_label, un_label, duration_label]
y_index=['burn_label', 'un_label', 'duration_label']

selectedlasso_list = []
for i  in range(len(y_arraylist)):
	selectedlasso = RunML_continue.LassoFS_CV(data,y_arraylist[i])
	selectedlasso_list.append(selectedlasso)


weights=FS.OTU_H_Score_fun(data,yList)
selectedOTU_index, eps=FS.indice_H_multisig(weights,yList)
print(eps)
X_FS = data[:,selectedOTU_index]



cls = ['RF','SVM']
datatypes=['AllFeatures', 'SelectMicro', 'Lasso_finetune']


result = {}
for i in range(len(y_arraylist)):
	result[y_index[i]] = {}
	for clf in cls:
		result[y_index[i]][clf]={}
    		# Store results for each feature selection method
		result[y_index[i]][clf]['AllFeatures'] = RunML_continue.nested_loocv(data,yList[:,i],classifier=clf,smote=True)
		result[y_index[i]][clf]['SelectMicro'] = RunML_continue.nested_loocv(X_FS,yList[:,i],classifier=clf,smote=True)
		result[y_index[i]][clf]['Lasso_finetune'] = RunML_continue.nested_loocv(selectedlasso_list[i][0],yList[:,i],classifier=clf,smote=True)
	

for i in range(len(y_arraylist)):
	print(f"{y_index[i]} analysis result:")
	for clf in cls:
    		for dataset in datatypes:
        		print(f"Mean Accuracy of {clf}_{dataset}:", result[y_index[i]][clf][dataset]["mean_accuracy"])
        		print(f"Std Accuracy of {clf}_{dataset}:",np.std(result[y_index[i]][clf][dataset]["outer_accuracies"]))
        		metric.plot_confusion_matrices(result[y_index[i]][clf][dataset]["y_true"], result[y_index[i]][clf][dataset]["y_pred"],f"{clf}_{dataset}",pos_y=1)

