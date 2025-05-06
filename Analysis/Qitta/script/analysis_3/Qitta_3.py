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

tax_levels = ['genus', 'species']
db_names = ['db2151']

# data preprocessing----------------------------------
# read data
for db in db_names:
    for tax_level in tax_levels:
        print(f'Analyis of {db} in tax = {tax_level}-------------------------------------------------------------------------------------------')
        count_filename = f'SelectMicro_24/Analysis/Qitta/data/{tax_level}/data_filtered99/features_{db}.csv'
        meta_filename = f'SelectMicro_24/Analysis/Qitta/data/{tax_level}/data_filtered99/meta_{db}.csv'
        tax_filename = f'SelectMicro_24/Analysis/Qitta/data/{tax_level}/data_filtered99/tax_{db}.csv'
        count_db1269 = pd.read_csv(count_filename,index_col=0)
        meta_db1269 = pd.read_csv(meta_filename,index_col=0)
        tax_db1269 = pd.read_csv(tax_filename,index_col=0)
        tax_db1269.index = tax_db1269.index.astype(str)
        
        
        # relative abundance and response variable
        cols_name = count_db1269.columns
        data = pd.DataFrame(FS.relative_abundance(count_db1269), columns=cols_name)
        y = meta_db1269['ibd']
        print(pd.Series(y).value_counts())
        
        custom_order = ['nonIBD', 'CD', 'UC']
        custom_mapping = {category: i for i, category in enumerate(custom_order)}
        target_variable = [custom_mapping[category] for category in y]
        target_variable = np.array(target_variable)
        
        # feature select--------------------------------------
        
        # SelectMicro
        selectedresult=FS.SelectMicro_fun(data,y,p_cutoff = 0.10)
        selectedOTU_index= selectedresult['selected_indices']
        X_FS = selectedresult['selected_df']
        selected_tax_FS = tax_db1269.loc[X_FS.columns,'Rank']
        #print(f'Select features by H test are: {selected_tax_FS}')
        #print(f"Number of features selected by SelectMicro {X_FS.shape[1]}")

        selectedresult_strict=FS.SelectMicro_fun(data,y,p_cutoff = 0.05)
        selectedOTU_index_strict= selectedresult_strict['selected_indices']
        X_FS_strict = selectedresult_strict['selected_df']
        selected_tax_FS_strict = tax_db1269.loc[X_FS_strict.columns,'Rank']
        
        # Lasso
        X_lasso_ft0,selectedOTU_index_Lasso  = RunML.LassoFS_CV_classification(np.array(data),target_variable)
        X_lasso_ft = pd.DataFrame(X_lasso_ft0, columns=cols_name[selectedOTU_index_Lasso])
        
        # SelectMicro+Lasso
        X_FS_lasso_ft0,xlabel_FS_lasso_ft0  = RunML.LassoFS_CV_classification(np.array(X_FS),target_variable)
        selectedOTU_index_FS_lasso = selectedOTU_index[xlabel_FS_lasso_ft0]
        X_FS_lasso_ft = pd.DataFrame(X_FS_lasso_ft0, columns=cols_name[selectedOTU_index_FS_lasso])
        
        # final data subset
        data_subset = {"AllFeatures":data,
                       "SelectMicro_strict": X_FS_strict,
                       "SelectMicro": X_FS,
                       #"Lasso_finetune":X_lasso_ft,
                       #"FS_Lassofinetune":X_FS_lasso_ft
                      }
        print(f'The shape of the original dataset is ',np.shape(data))
        print(f'The shape of the SelectMicro dataset is ',np.shape(X_FS))
        print(f'The shape of the SelectMicro_strict dataset is ',np.shape(X_FS_strict))
        
        
        # Model-----------------------------------------------------------
        print("5 fold cross validation using Random forest model -----------------------------------------")
        dict_cm_list = []
        save_dir = "/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Qitta/results/analysis_3/plots/higher"
        for datatype, subset in data_subset.items():
            print(f"Analysis for {datatype}")
            dict_cm = RunModel.RF_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
            dict_cm_list.append(dict_cm)
            # Prepare base filename
            base_filename = f"{db}_{tax_level}_RF_{datatype}"

            metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=1, n_classes=3, class_label='Class 1',save_path=os.path.join(save_dir, f"{base_filename}_ROC_class1.png"))
            metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=1, save_path=os.path.join(save_dir, f"{base_filename}_SHAP_class1.png"))
            metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=2, n_classes=3, class_label='Class 2',save_path =os.path.join(save_dir, f"{base_filename}__ROC_class2.png"))
            metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=2,save_path=os.path.join(save_dir, f"{base_filename}_SHAP_class2.png"))
        
        
        print("5 fold cross validation using XGBoost model -----------------------------------------")
        dict_cm_list = []
        save_dir = "/lustre/isaac24/scratch/mhe8/SelectMicro_24/Analysis/Qitta/results/analysis_3/plots/higher"
        for datatype, subset in data_subset.items():
            print(f"Analysis for {datatype}")
            dict_cm = RunModel.XG_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
            #dict_cm = RunModel.SVM_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
            #dict_cm = RunModel.NB_model_SCV_multi(subset, target_variable,SMOTE=True,k=5)
            dict_cm_list.append(dict_cm)
            # Prepare base filename
            base_filename = f"{db}_{tax_level}_XG_{datatype}"
            metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=1, n_classes=3, class_label='Class 1',save_path=os.path.join(save_dir, f"{base_filename}_ROC_class1.png"))
            metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=1, save_path=os.path.join(save_dir, f"{base_filename}_SHAP_class1.png"))
            
            metric.plot_multiclass_roc_cv(dict_cm['y_true'], dict_cm['y_pred_prob'], class_index=2, n_classes=3, class_label='Class 2',save_path =os.path.join(save_dir, f"{base_filename}_ROC_class2.png"))
            metric.plot_SHAP_multiclass(dict_cm['SHAP_full'],dict_cm['x_true'],class_index=2,save_path=os.path.join(save_dir, f"{base_filename}_SHAP_class2.png"))


