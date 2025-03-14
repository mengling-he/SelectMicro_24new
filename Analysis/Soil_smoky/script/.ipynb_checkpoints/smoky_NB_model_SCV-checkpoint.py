#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 00:00:19 2024

@author: menglinghe
"""

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder



import sys
sys.path.append('../../../Code')
import loadData 
import RunML
import FS
import metric




data,burn_label,un_label,duration_label,ASVs,df=loadData.loadSoilData("../data/count_table/merge_proportion_asv_smoky_moutain.csv")    

df = pd.DataFrame(data,columns = ASVs)# the function requires a dataframe with feature names

ASVs = np.array(ASVs)


 

le = LabelEncoder()
burn_label = le.fit_transform(burn_label)  
le = LabelEncoder()
un_label = le.fit_transform(un_label)  
le = LabelEncoder()
duration_label = le.fit_transform(duration_label)  



taxlabels = ['Burn','urban','duration']
selectedresult_list = []
for y in [burn_label,un_label,duration_label]:
    selectedresult=FS.SelectMicro_fun(df,y)
    selectedresult_list.append(selectedresult)

selectedOTU_index_Lasso_list = []
selectedOTU_index_FS_lasso_list = []

data_subset_list = []

for index, element in enumerate([burn_label,un_label,duration_label]):
    
    selectedresult = selectedresult_list[index]
    
    selectedOTU_index_FS = selectedresult['selected_indices']

    data = selectedresult['relative_abundance_data']
    data_df = pd.DataFrame(data,columns = ASVs)
    X_FS = selectedresult['selected_data']
    X_FS_df =pd.DataFrame(X_FS,columns = selectedresult['selected_columnames'])

    X_lasso_ft,selectedOTU_index_Lasso  = RunML.LassoFS_CV(data,element)
    X_lasso_ft_df = pd.DataFrame(X_lasso_ft,columns = ASVs[selectedOTU_index_Lasso])
    selectedOTU_index_Lasso_list.append(selectedOTU_index_Lasso)

    X_FS_lasso_ft,xlabel_FS_lasso_ft0  = RunML.LassoFS_CV(X_FS,element)
    selectedOTU_index_FS_lasso = selectedOTU_index_FS[xlabel_FS_lasso_ft0]
    X_FS_lasso_ft_df = pd.DataFrame(X_FS_lasso_ft,columns = ASVs[selectedOTU_index_FS_lasso])
    selectedOTU_index_FS_lasso_list.append(selectedOTU_index_FS_lasso)
    
    data_subset = {"AllFeatures":data_df, 
               "SelectMicro": X_FS_df,
               "Lasso_finetune":X_lasso_ft_df,
                "FS_Lassofinetune":X_FS_lasso_ft_df
               #"Random":data
              }
    data_subset_list.append(data_subset)



for data in data_subset_list:
    print(f'The shape of the full selected dataset is ',np.shape(data['AllFeatures']))
    print(f'The shape of the FS selected dataset is ',np.shape(data['SelectMicro']))
    print(f'The shape of the Lasso_finetune selected dataset is ',np.shape(data['Lasso_finetune']))
    print(f'The shape of the FS_Lasso_finetune selected dataset is ',np.shape(data['FS_Lassofinetune']))
 

for index, element in enumerate([burn_label,un_label,duration_label]):
    targetLabel=element
    for i, (key, value) in enumerate(data_subset_list[index].items()):
        print(taxlabels[index])
        print(key)
        RunML.NB_model_SCV(value, y=element,plot=True, SMOTE=True,y_base = 1)