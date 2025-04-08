
import sys
sys.path.append('/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code')
import RunML
import FS
import metric
import RunModel

import pandas as pd
import numpy as np
import random
import os



# data preprocessing----------------------------------
data = pd.read_csv('/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Zeller/data/features_table_update.csv',index_col=0)
cols_name = data.columns
data = pd.DataFrame(FS.relative_abundance(data), columns=cols_name)
meta_2 = pd.read_csv('/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Zeller/data/meta_data.csv',index_col=0)
y = meta_2['disease']
print(y.value_counts())

custom_order = ['healthy', 'adenoma', 'CRC']
custom_mapping = {category: i for i, category in enumerate(custom_order)}
target_variable = [custom_mapping[category] for category in y]
target_variable = np.array(target_variable)

# feature select--------------------------------------

# SelectMicro
selectedresult=FS.SelectMicro_fun(data,y,p_cutoff = 0.05)

data2 = pd.DataFrame(selectedresult['relative_abundance_data'],columns=cols_name)
selectedOTU_index= selectedresult['selected_indices']
X_FS = selectedresult['selected_df']

# Lasso
X_lasso_ft0,selectedOTU_index_Lasso  = RunML.LassoFS_CV(np.array(data2),target_variable)
X_lasso_ft = pd.DataFrame(X_lasso_ft0, columns=cols_name[selectedOTU_index_Lasso])

# SelectMicro+Lasso
X_FS_lasso_ft0,xlabel_FS_lasso_ft0  = RunML.LassoFS_CV(np.array(X_FS),target_variable)
selectedOTU_index_FS_lasso = selectedOTU_index[xlabel_FS_lasso_ft0]
X_FS_lasso_ft = pd.DataFrame(X_FS_lasso_ft0, columns=cols_name[selectedOTU_index_FS_lasso])

# final data subset
data_subset = {"AllFeatures":data2,
               "SelectMicro": X_FS,
               "Lasso_finetune":X_lasso_ft,
               "FS_Lassofinetune":X_FS_lasso_ft
              }
print(f'The shape of the original dataset is ',np.shape(data))
print(f'The shape of the SelectMicro dataset is ',np.shape(X_FS))
print(f'The shape of the Lasso_finetune selected dataset is ',np.shape(X_lasso_ft))
print(f'The shape of the FS_Lasso_finetune selected dataset is ',np.shape(X_FS_lasso_ft))

# Model-----------------------------------------------------------
dict_cm_list = []
print("5 fold cross validation using Random forest model -----------------------------------------")
save_dir = "/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Zeller/result"
for datatype, subset in data_subset.items():
    print(f"Analysis for {datatype}")
    dict_cm = RunModel.RF_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
    
    metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=1, n_classes=3, class_label='Class 1',save_path=os.path.join(save_dir, f"{datatype}_RF_ROC_class1.png"))
    metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=1, save_path=os.path.join(save_dir, f"{datatype}_RF_SHAP_class1.png"))
    
    metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=2, n_classes=3, class_label='Class 2',save_path =os.path.join(save_dir, f"{datatype}_RF_ROC_class2.png"))
    metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=2,save_path=os.path.join(save_dir, f"{datatype}_RF_SHAP_class2.png"))
