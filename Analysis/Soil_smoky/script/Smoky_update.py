import sys
sys.path.append('/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code')
import loadData 
import RunML
import FS
import metric
import RunModel

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import os

data,burn_label,un_label,duration_label,ASVs,df=loadData.loadSoilData("/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Soil_smoky/data/count_table/merge_proportion_asv_smoky_moutain.csv")

df = pd.DataFrame(FS.relative_abundance(data),columns = ASVs)# the function requires a dataframe with feature names
cols_name = df.columns

# select the OTUs
taxlabels = ['Burn','Urban','Duration']

selectedresult_list = []
for i,y in enumerate([burn_label,un_label,duration_label]):
    print(f'Analysis of {taxlabels[i]} label-------------------------------------------------------')
    le = LabelEncoder()
    target_variable = le.fit_transform(y)
    selectedresult=FS.SelectMicro_fun(df,target_variable,p_cutoff = 0.10)
    selectedOTU_index= selectedresult['selected_indices']
    X_FS = selectedresult['selected_df']
    
    # Lasso
    X_lasso_ft0,selectedOTU_index_Lasso  = RunML.LassoFS_CV(np.array(df),target_variable)
    X_lasso_ft = pd.DataFrame(X_lasso_ft0, columns=cols_name[selectedOTU_index_Lasso])
    
    # SelectMicro+Lasso
    X_FS_lasso_ft0,xlabel_FS_lasso_ft0  = RunML.LassoFS_CV(np.array(X_FS),target_variable)
    selectedOTU_index_FS_lasso = selectedOTU_index[xlabel_FS_lasso_ft0]
    X_FS_lasso_ft = pd.DataFrame(X_FS_lasso_ft0, columns=cols_name[selectedOTU_index_FS_lasso])

    # final data subset
    data_subset = {"AllFeatures":df,
                   "SelectMicro": X_FS,
                   "Lasso_finetune":X_lasso_ft,
                   "FS_Lassofinetune":X_FS_lasso_ft
                  }
    print(f'The shape of the original dataset is ',np.shape(data))
    print(f'The shape of the SelectMicro dataset is ',np.shape(X_FS))
    print(f'The shape of the Lasso_finetune selected dataset is ',np.shape(X_lasso_ft))
    print(f'The shape of the FS_Lasso_finetune selected dataset is ',np.shape(X_FS_lasso_ft))
    
    # Model-----------------------------------------------------------
    print("5 fold cross validation using Random forest model -----------------------------------------")
    dict_cm_list = []
    save_dir = "/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Soil_smoky/results/"
    for datatype, subset in data_subset.items():
        print(f"Analysis for {datatype}")
        dict_cm = RunModel.RF_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
        #dict_cm = RunModel.XG_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
        #dict_cm = RunModel.SVM_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
        #dict_cm = RunModel.NB_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
        dict_cm_list.append(dict_cm)
        metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=1, n_classes=2, class_label='Class 1',save_path=os.path.join(save_dir, f"RF_{taxlabels[i]}_{datatype}_ROC_class1.png"))
        metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=1, save_path=os.path.join(save_dir, f"RF_{taxlabels[i]}_{datatype}_SHAP_class1.png"))
        
        #metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=2, n_classes=3, class_label='Class 2',save_path =os.path.join(save_dir, f"NB_{datatype}_ROC_class2.png"))
        #metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=2,save_path=os.path.join(save_dir, f"XG_{datatype}_SHAP_class2.png"))
        
        #print("5 fold cross validation using XGBoost model -----------------------------------------")
        #print("5 fold cross validation using NB model -----------------------------------------")# no shap plots for SVM,NB
   
