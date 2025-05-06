import sys
sys.path.append('/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code')
import RunML
import FS
import metric
import RunModel

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import os


# data preprocessing----------------------------------
#data = pd.read_csv('SelectMicro_24/Analysis/Qitta/data/features_genus_db1629.csv',index_col=0)
# delete [, ] or < in colnames which is required in xgboost
#data.columns = data.columns.astype(str).str.replace(r"[\[\]<>]", "_", regex=True)
#cols_name = data.columns

db1629_features_species = pd.read_excel('SelectMicro_24/Analysis/Qitta/data/species/db1629.xlsx', engine='openpyxl', index_col=0)
db1629_features_genus = pd.read_excel('SelectMicro_24/Analysis/Qitta/data/genus/db1629.xlsx', engine='openpyxl', index_col=0)
meta_2 = pd.read_csv('SelectMicro_24/Analysis/Qitta/data/meta_genus_db1629.csv',index_col=0)


df1_aligned = db1629_features_species.loc[meta_2.index.intersection(db1629_features_species.index)].reindex(meta_2.index)
df2_aligned = db1629_features_genus.loc[meta_2.index.intersection(db1629_features_genus.index)].reindex(meta_2.index)

data_list = [df1_aligned,df2_aligned]
tax_levels = ['species', 'genus']

for i, element in enumerate(data_list):
    if not element.index.equals(meta_2.index):
        raise ValueError(f"Features of {tax_levels[i]} have different indexes with metadata")

y_label = meta_2['ibd']
print(pd.Series(y_label).value_counts())


print("Begin Analysis------------------------------------------------------------")

response_cat = ['Multi', 'Binary']

for ii,res in enumerate(response_cat):
    if res=='Multi':
        custom_order = ['nonIBD', 'CD', 'UC']
        custom_mapping = {category: i for i, category in enumerate(custom_order)}
        target_variable = [custom_mapping[category] for category in y_label]
        target_variable = np.array(target_variable)

    if res=='Binary':
        y = y_label.replace({'CD': 'IBD', 'UC': 'IBD'})
        le = LabelEncoder()
        target_variable= le.fit_transform(y)
        
    print(f"Analysis of {response_cat[ii]}-----------------------------------------")   
    print(pd.Series(target_variable).value_counts())
    
    for index, tax_level in enumerate(tax_levels):
        print(f'Analyis in tax = {tax_level}------------------------------------------------------------------------------------------')
        data = data_list[index]
        cols_name = data.columns
    
        data = pd.DataFrame(FS.relative_abundance(data), columns=cols_name)

        # feature select--------------------------------------
        
        # SelectMicro
        selectedresult_strict=FS.SelectMicro_fun(data,target_variable,p_cutoff = 0.05)
        X_FS_strict = selectedresult_strict['selected_df']
        
        selectedresult=FS.SelectMicro_fun(data,target_variable,p_cutoff = 0.10)
        selectedOTU_index= selectedresult['selected_indices']
        X_FS = selectedresult['selected_df']
        #print(f"Number of features selected by SelectMicro {X_FS.shape[1]}")
        
        # Lasso
        X_lasso_ft0,selectedOTU_index_Lasso  = RunML.LassoFS_CV_classification(np.array(data),target_variable)
        X_lasso_ft = pd.DataFrame(X_lasso_ft0, columns=cols_name[selectedOTU_index_Lasso])
        #print(f"Number of features selected by Lasso {X_lasso_ft.shape[1]}")
        
        # SelectMicro+Lasso
        X_FS_lasso_ft0,xlabel_FS_lasso_ft0  = RunML.LassoFS_CV_classification(np.array(X_FS),target_variable)
        selectedOTU_index_FS_lasso = selectedOTU_index[xlabel_FS_lasso_ft0]
        X_FS_lasso_ft = pd.DataFrame(X_FS_lasso_ft0, columns=cols_name[selectedOTU_index_FS_lasso])
        #print(f"Number of features selected by SelectMicro_Lasso {X_FS_lasso_ft.shape[1]}")
        
        # final data subset
        data_subset = {"AllFeatures":data,
                       "SelectMicro_strict": X_FS_strict,
                       "SelectMicro": X_FS,
                       "Lasso_finetune":X_lasso_ft,
                       "FS_Lassofinetune":X_FS_lasso_ft
                      }
        print(f'The shape of the original dataset is ',np.shape(data))
        print(f'The shape of the SelectMicro_strict dataset is ',np.shape(X_FS_strict))
        print(f'The shape of the SelectMicro dataset is ',np.shape(X_FS))
        print(f'The shape of the Lasso_finetune selected dataset is ',np.shape(X_lasso_ft))
        print(f'The shape of the FS_Lasso_finetune selected dataset is ',np.shape(X_FS_lasso_ft))

 # Model-----------------------------------------------------------
        save_dir = "/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Qitta/results/db1629_final/plots"
        
        print("5 fold cross validation using Random forest model -----------------------------------------")
        dict_cm_list = []
        for datatype, subset in data_subset.items():
            print(f"Analysis for {datatype}")
            dict_cm = RunModel.RF_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
            dict_cm_list.append(dict_cm)
            #metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=1, n_classes=3, class_label='Class 1',save_path=os.path.join(save_dir, f"RF_{datatype}_ROC_class1.png"))
            #metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=1, save_path=os.path.join(save_dir, f"RF_{datatype}_SHAP_class1.png"))
            #metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=2, n_classes=3, class_label='Class 2',save_path =os.path.join(save_dir, f"RF_{datatype}_ROC_class2.png"))
            #metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=2,save_path=os.path.join(save_dir, f"RF_{datatype}_SHAP_class2.png"))

        
        print("5 fold cross validation using XGBoost model -----------------------------------------")
        for datatype, subset in data_subset.items():
            print(f"Analysis for {datatype}")
            dict_cm = RunModel.XG_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
    
        print("5 fold cross validation using NB model -----------------------------------------")
        for datatype, subset in data_subset.items():
            print(f"Analysis for {datatype}")
            dict_cm = RunModel.NB_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
        
        print("5 fold cross validation using SVM model -----------------------------------------")
        for datatype, subset in data_subset.items():
            print(f"Analysis for {datatype}")
            dict_cm = RunModel.SVM_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
           
"""


print("Analysis of CD and healthy-----------------------------------------")
data = pd.read_csv('SelectMicro_24/Analysis/Qitta/data/features_genus_db1629.csv',index_col=0)
# delete [, ] or < in colnames which is required in xgboost
data.columns = data.columns.astype(str).str.replace(r"[\[\]<>]", "_", regex=True)
cols_name = data.columns
data = pd.DataFrame(FS.relative_abundance(data), columns=cols_name)

meta_2 = pd.read_csv('SelectMicro_24/Analysis/Qitta/data/meta_genus_db1629.csv',index_col=0)
meta_2 = meta_2.reset_index(drop=False)

# Create a boolean mask for rows where ibd == "UC"
mask = meta_2["ibd"] != "UC"
# Apply the mask to both dataframes
meta_2 = meta_2[mask]
data = data[mask]

y = meta_2['ibd']
print(pd.Series(y).value_counts())

custom_order = ['nonIBD', 'CD']
custom_mapping = {category: i for i, category in enumerate(custom_order)}
target_variable = [custom_mapping[category] for category in y]
target_variable = np.array(target_variable)

# SelectMicro
selectedresult=FS.SelectMicro_fun(data,y,p_cutoff = 0.05)

selectedOTU_index= selectedresult['selected_indices']
X_FS = selectedresult['selected_df']
#print(f"Number of features selected by SelectMicro {X_FS.shape[1]}")

# Lasso
X_lasso_ft0,selectedOTU_index_Lasso  = RunML.LassoFS_CV(np.array(data),target_variable)
X_lasso_ft = pd.DataFrame(X_lasso_ft0, columns=cols_name[selectedOTU_index_Lasso])
#print(f"Number of features selected by Lasso {X_lasso_ft.shape[1]}")

# SelectMicro+Lasso
X_FS_lasso_ft0,xlabel_FS_lasso_ft0  = RunML.LassoFS_CV(np.array(X_FS),target_variable)
selectedOTU_index_FS_lasso = selectedOTU_index[xlabel_FS_lasso_ft0]
X_FS_lasso_ft = pd.DataFrame(X_FS_lasso_ft0, columns=cols_name[selectedOTU_index_FS_lasso])
#print(f"Number of features selected by SelectMicro_Lasso {X_FS_lasso_ft.shape[1]}")

# final data subset
data_subset = {"AllFeatures":data,
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
save_dir = "/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Qitta/results/db1629_update_withbinary"
for datatype, subset in data_subset.items():
    print(f"Analysis for {datatype}")
    dict_cm = RunModel.RF_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
    dict_cm_list.append(dict_cm)
    metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=1, n_classes=2, class_label='CD',save_path=os.path.join(save_dir, f"RF_{datatype}_ROC_binary.png"))
    metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=1, save_path=os.path.join(save_dir, f"RF_{datatype}_SHAP_binary.png"))

print("5 fold cross validation using XGBoost model -----------------------------------------")
#print("5 fold cross validation using NB model -----------------------------------------")# no shap plots for SVM,NB
dict_cm_list = []
save_dir = "/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Qitta/results/db1629_update_withbinary"
for datatype, subset in data_subset.items():
    print(f"Analysis for {datatype}")
    dict_cm = RunModel.XG_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
    #dict_cm = RunModel.SVM_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
    #dict_cm = RunModel.NB_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
    dict_cm_list.append(dict_cm)
    metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=1, n_classes=2, class_label='binary',save_path=os.path.join(save_dir, f"XG_{datatype}_ROC_binary.png"))
    metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=1, save_path=os.path.join(save_dir, f"XG_{datatype}_SHAP_binary.png"))
    
