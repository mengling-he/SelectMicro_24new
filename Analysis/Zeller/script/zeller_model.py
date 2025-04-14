
import sys
sys.path.append('/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code')
import RunML
import FS
import metric
import RunModel

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import os



# data preprocessing----------------------------------
data0 = pd.read_csv('/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Zeller/data/features_table_species.csv',index_col=0)
cols_name = data0.columns
cols_name = np.array([x.split("|")[-1] for x in cols_name])
data = pd.DataFrame(FS.relative_abundance(data0), columns=cols_name)
meta_2 = pd.read_csv('/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Zeller/data/meta_data.csv',index_col=0)
y = meta_2['disease']
print(y.value_counts())

custom_order = ['healthy', 'adenoma', 'CRC']
custom_mapping = {category: i for i, category in enumerate(custom_order)}
target_variable = [custom_mapping[category] for category in y]
target_variable = np.array(target_variable)

# feature select--------------------------------------

# SelectMicro
selectedresult=FS.SelectMicro_fun(data,target_variable,p_cutoff = 0.05)

selectedOTU_index= selectedresult['selected_indices']
X_FS = selectedresult['selected_df']

# Lasso
X_lasso_ft0,selectedOTU_index_Lasso  = RunML.LassoFS_CV(np.array(data),target_variable)
X_lasso_ft = pd.DataFrame(X_lasso_ft0, columns=cols_name[selectedOTU_index_Lasso])



# SelectMicro+Lasso 
X_FS_lasso_ft0,xlabel_FS_lasso_ft0  = RunML.LassoFS_CV(np.array(X_FS),target_variable)
selectedOTU_index_FS_lasso = selectedOTU_index[xlabel_FS_lasso_ft0]
X_FS_lasso_ft = pd.DataFrame(X_FS_lasso_ft0, columns=cols_name[selectedOTU_index_FS_lasso])


# combining environmental: BMI has missing values
env_inf = meta_2[['age','gender','country']].reset_index(drop=True)
# Initialize LabelEncoder
le_gender = LabelEncoder()
env_inf['gender'] = le_gender.fit_transform(env_inf['gender'])
le_country = LabelEncoder()
env_inf['country'] = le_country.fit_transform(env_inf['country'])
print("Class mapping for gender:")
for i, cls in enumerate(le_gender.classes_):
    print(f"{cls} → {i}")
print("Class mapping for country:")
for i, cls in enumerate(le_country.classes_):
    print(f"{cls} → {i}")
X_FS_env = pd.concat([X_FS, env_inf], axis=1)

X_lasso_ft_env = pd.concat([X_lasso_ft, env_inf], axis=1)


# final data subset
data_subset = {"AllFeatures":data,
                "SelectMicro": X_FS,
               "SelectMicro_env":X_FS_env,
               "Lasso_finetune":X_lasso_ft,
               "Lasso_finetune_env":X_lasso_ft_env,
               "FS_Lassofinetune":X_FS_lasso_ft
              }
print(f'The shape of the original dataset is ',np.shape(data))
print(f'The shape of the SelectMicro dataset is ',np.shape(X_FS))
print(f'the shape of the FS_env is',np.shape(X_FS_env))
print(f'The shape of the Lasso_finetune selected dataset is ',np.shape(X_lasso_ft))
print(f'The shape of the Lasso_finetune_env selected dataset is ',np.shape(X_lasso_ft_env))
print(f'The shape of the FS_Lasso_finetune selected dataset is ',np.shape(X_FS_lasso_ft))

# Model-----------------------------------------------------------
dict_cm_list = []
save_dir = "/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Zeller/result_species"

print("5 fold cross validation using Random forest model -----------------------------------------")
for datatype, subset in data_subset.items():
    print(f"Analysis for {datatype}")
    dict_cm = RunModel.RF_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
    
    metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=1, n_classes=3, class_label='Class 1',save_path=os.path.join(save_dir, f"{datatype}_RF_ROC_class1.png"))
    metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=1, save_path=os.path.join(save_dir, f"{datatype}_RF_SHAP_class1.png"))
    
    metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=2, n_classes=3, class_label='Class 2',save_path =os.path.join(save_dir, f"{datatype}_RF_ROC_class2.png"))
    metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=2,save_path=os.path.join(save_dir, f"{datatype}_RF_SHAP_class2.png"))



print("5 fold cross validation using XG model -----------------------------------------")
for datatype, subset in data_subset.items():
    print(f"Analysis for {datatype}")
    dict_cm = RunModel.XG_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
    
    metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=1, n_classes=3, class_label='Class 1',save_path=os.path.join(save_dir, f"{datatype}_XG_ROC_class1.png"))
    metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=1, save_path=os.path.join(save_dir, f"{datatype}_XG_SHAP_class1.png"))
    
    metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=2, n_classes=3, class_label='Class 2',save_path =os.path.join(save_dir, f"{datatype}_XG_ROC_class2.png"))
    metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=2,save_path=os.path.join(save_dir, f"{datatype}_XG_SHAP_class2.png"))


