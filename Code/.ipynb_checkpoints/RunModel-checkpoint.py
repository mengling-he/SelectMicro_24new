import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler,label_binarize
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier  # Import XGBoost
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, LeaveOneOut
import shap
import time
import sys
import os
sys.path.append('./Code')
import metric

import FS
import RunML

# --------------------------------------------------------------------------------------------------#
# 5 fold cross validation

# 1. Random forest
def RF_model_SCV_multi(X, y, SMOTE=False,k=5):
     # Initialize the classifier
   
    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=777)

    # Lists to store the results
    accuracies = []
    roc_aucs = []
    f_scores = []
    mcc_s = []
    
    shap_values_all = np.empty((0, X.shape[1], len(np.unique(y))))   # Store SHAP values for each fold
    
    # List to store the index of each fold
    test_sets_ix = []

    # # preparation for response
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    idx = 0                          
    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE if specified
        if SMOTE:
            X_train, y_train = RunML.perform_SMOTE(X_train, y_train)
         
        clf = RandomForestClassifier(n_jobs=5, random_state=777)
    
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)## For multiclass, y_prob will have shape (n_samples, n_classes)

        
        # Evaluate the model and save the results for each fold
        accuracy = accuracy_score(y_test, y_pred)
        mcc = metric.mcc_score(y_test,y_pred)
        f_score = f1_score(y_test,y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')# calculate the AUC for each level and  average it
        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)
        mcc_s.append(mcc)
        f_scores.append(f_score)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        shap_values_all = np.concatenate((shap_values_all, shap_values), axis=0) 
        test_sets_ix.append(test_index)
        
        y_true_all.append(y_test)# Using extend to add multiple elements
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)
        # y_prob_all.extend(y_prob.argmax(axis=1))  # For multiclass, take the class with the highest probability
        
        idx += 1
    print("The combined confusion matrix")
    print(confusion_matrix( np.concatenate(y_true_all),  np.concatenate(y_pred_all)))

    # sort x based on the test dataset index
    test_sets_ix = np.concatenate(test_sets_ix)
    X_true_df = pd.DataFrame(X.iloc[test_sets_ix,:], columns=X.columns)
    
    result = {'y_true': y_true_all,
              'y_pred':y_pred_all,
              'y_pred_prob': y_prob_all,
              'x_true':X_true_df,
              'SHAP_full':shap_values_all,
              'mean_accuracy': np.mean(accuracies),
              'std_accuracy':np.std(accuracies),
              'mean_auc': np.mean(roc_aucs),
              'std_auc':np.std(roc_aucs),
              'mean_mcc':np.mean(mcc_s),
              'mean_f':np.mean(f_scores),}
    print('Mean Accuracy: %.3f (%.3f),Mean F1: %.3f (%.3f),Mean MCC: %.3f (%.3f),  Mean AUC: %.3f (%.3f)' % (np.mean(accuracies), np.std(accuracies),np.mean(f_scores), np.std(f_scores),np.mean(mcc_s), np.std(mcc_s),np.mean(roc_aucs), np.std(roc_aucs)))
   
    return result


# 2. XG
def XG_model_SCV_multi(X, y, SMOTE=False,k=5):
     # Initialize the classifier
   
    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=777)

    # Lists to store the results
    accuracies = []
    roc_aucs = []
    f_scores = []
    mcc_s = []
    
    shap_values_all = np.empty((0, X.shape[1], len(np.unique(y))))   # Store SHAP values for each fold
    
    # List to store the index of each fold
    test_sets_ix = []

    # # preparation for response
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    idx = 0                          
    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE if specified
        if SMOTE:
            X_train, y_train = RunML.perform_SMOTE(X_train, y_train)
         
        clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=777)
     
    
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)## For multiclass, y_prob will have shape (n_samples, n_classes)

        
        # Evaluate the model and save the results for each fold
        accuracy = accuracy_score(y_test, y_pred)
        mcc = metric.mcc_score(y_test,y_pred)
        f_score = f1_score(y_test,y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')# calculate the AUC for each level and  average it
        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)
        mcc_s.append(mcc)
        f_scores.append(f_score)

        explainer = shap.TreeExplainer(clf)
        #shap_obj = explainer(X_test)
        #shap_values = shap_obj.values
        shap_values = explainer.shap_values(X_test)
        shap_values_all = np.concatenate((shap_values_all, shap_values), axis=0) 
        test_sets_ix.append(test_index)
        
        y_true_all.append(y_test)# Using extend to add multiple elements
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)
        # y_prob_all.extend(y_prob.argmax(axis=1))  # For multiclass, take the class with the highest probability
        
        idx += 1
    print("The combined confusion matrix")
    print(confusion_matrix( np.concatenate(y_true_all),  np.concatenate(y_pred_all)))

    # sort x based on the test dataset index
    test_sets_ix = np.concatenate(test_sets_ix)
    X_true_df = pd.DataFrame(X.iloc[test_sets_ix,:], columns=X.columns)
    
    result = {'y_true': y_true_all,
              'y_pred':y_pred_all,
              'y_pred_prob': y_prob_all,
              'x_true':X_true_df,
              'SHAP_full':shap_values_all,
              'mean_accuracy': np.mean(accuracies),
              'std_accuracy':np.std(accuracies),
              'mean_auc': np.mean(roc_aucs),
              'std_auc':np.std(roc_aucs),
              'mean_mcc':np.mean(mcc_s),
              'mean_f':np.mean(f_scores),}
    print('Mean Accuracy: %.3f (%.3f),Mean F1: %.3f (%.3f),Mean MCC: %.3f (%.3f),  Mean AUC: %.3f (%.3f)' % (np.mean(accuracies), np.std(accuracies),np.mean(f_scores), np.std(f_scores),np.mean(mcc_s), np.std(mcc_s),np.mean(roc_aucs), np.std(roc_aucs)))
   
    return result

# 3. SVM
def SVM_model_SCV_multi(X, y, SMOTE=False,k=5):
     # Initialize the classifier
   
    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=777)

    # Lists to store the results
    accuracies = []
    roc_aucs = []
    f_scores = []
    mcc_s = []
    
    shap_values_all = np.empty((0, X.shape[1], len(np.unique(y))))   # Store SHAP values for each fold
    
    # List to store the index of each fold
    test_sets_ix = []

    # # preparation for response
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    idx = 0                          
    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE if specified
        if SMOTE:
            X_train, y_train = RunML.perform_SMOTE(X_train, y_train)
         
        #clf = RandomForestClassifier(n_jobs=5, random_state=777)
        clf = svm.SVC(probability=True, random_state=777)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)## For multiclass, y_prob will have shape (n_samples, n_classes)

        
        # Evaluate the model and save the results for each fold
        accuracy = accuracy_score(y_test, y_pred)
        mcc = metric.mcc_score(y_test,y_pred)
        f_score = f1_score(y_test,y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')# calculate the AUC for each level and  average it
        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)
        mcc_s.append(mcc)
        f_scores.append(f_score)

        # disable shap for now since it takes longer time, the code needs to be updated
        """
        explainer = shap.KernelExplainer(clf.predict_proba,X_train)
        shap_values =  explainer.shap_values(X_test)
        
        shap_values_all = np.concatenate((shap_values_all, shap_values), axis=0) 
        """
        test_sets_ix.append(test_index)
        
        y_true_all.append(y_test)# Using extend to add multiple elements
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)
        # y_prob_all.extend(y_prob.argmax(axis=1))  # For multiclass, take the class with the highest probability
        
        idx += 1
    print("The combined confusion matrix")
    print(confusion_matrix( np.concatenate(y_true_all),  np.concatenate(y_pred_all)))

    # sort x based on the test dataset index
    test_sets_ix = np.concatenate(test_sets_ix)
    X_true_df = pd.DataFrame(X.iloc[test_sets_ix,:], columns=X.columns)
    
    result = {'y_true': y_true_all,
              'y_pred':y_pred_all,
              'y_pred_prob': y_prob_all,
              'x_true':X_true_df,
              #'SHAP_full':shap_values_all,
              'mean_accuracy': np.mean(accuracies),
              'std_accuracy':np.std(accuracies),
              'mean_auc': np.mean(roc_aucs),
              'std_auc':np.std(roc_aucs),
              'mean_mcc':np.mean(mcc_s),
              'mean_f':np.mean(f_scores),}
    print('Mean Accuracy: %.3f (%.3f),Mean F1: %.3f (%.3f),Mean MCC: %.3f (%.3f),  Mean AUC: %.3f (%.3f)' % (np.mean(accuracies), np.std(accuracies),np.mean(f_scores), np.std(f_scores),np.mean(mcc_s), np.std(mcc_s),np.mean(roc_aucs), np.std(roc_aucs)))
   
    return result



# 4. NB
def NB_model_SCV_multi(X, y, SMOTE=False,k=5):
     # Initialize the classifier
   
    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=777)

    # Lists to store the results
    accuracies = []
    roc_aucs = []
    f_scores = []
    mcc_s = []
    
    shap_values_all = np.empty((0, X.shape[1]))# KernelExplainer (used for Naïve Bayes) returns a single SHAP value per feature without per-class separation.
    
    # List to store the index of each fold
    test_sets_ix = []

    # # preparation for response
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    idx = 0                          
    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE if specified
        if SMOTE:
            X_train, y_train = RunML.perform_SMOTE(X_train, y_train)

        # Scale Data (GaussianNB assumes normally distributed data)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
         
        clf = GaussianNB()
     
    
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)## For multiclass, y_prob will have shape (n_samples, n_classes)

        
        # Evaluate the model and save the results for each fold
        accuracy = accuracy_score(y_test, y_pred)
        mcc = metric.mcc_score(y_test,y_pred)
        f_score = f1_score(y_test,y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')# calculate the AUC for each level and  average it
        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)
        mcc_s.append(mcc)
        f_scores.append(f_score)

        explainer = shap.Explainer(clf.predict,X_train_scaled)
        shap_obj = explainer(X_test_scaled,max_evals=4000)
        shap_values = shap_obj.values
        print(f"shap shape {shap_values.shape}")
        shap_values_all = np.concatenate((shap_values_all, shap_values), axis=0) 
        test_sets_ix.append(test_index)
        
        y_true_all.append(y_test)# Using extend to add multiple elements
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)
        # y_prob_all.extend(y_prob.argmax(axis=1))  # For multiclass, take the class with the highest probability
        
        idx += 1
    print("The combined confusion matrix")
    print(confusion_matrix( np.concatenate(y_true_all),  np.concatenate(y_pred_all)))

    # sort x based on the test dataset index
    test_sets_ix = np.concatenate(test_sets_ix)
    X_true_df = pd.DataFrame(X.iloc[test_sets_ix,:], columns=X.columns)
    
    result = {'y_true': y_true_all,
              'y_pred':y_pred_all,
              'y_pred_prob': y_prob_all,
              'x_true':X_true_df,
              'SHAP_full':shap_values_all,
              'mean_accuracy': np.mean(accuracies),
              'std_accuracy':np.std(accuracies),
              'mean_auc': np.mean(roc_aucs),
              'std_auc':np.std(roc_aucs),
              'mean_mcc':np.mean(mcc_s),
              'mean_f':np.mean(f_scores),}
    print('Mean Accuracy: %.3f (%.3f),Mean F1: %.3f (%.3f),Mean MCC: %.3f (%.3f),  Mean AUC: %.3f (%.3f)' % (np.mean(accuracies), np.std(accuracies),np.mean(f_scores), np.std(f_scores),np.mean(mcc_s), np.std(mcc_s),np.mean(roc_aucs), np.std(roc_aucs)))
   
    return result



# Test cases
if __name__ == "__main__":
    data = pd.read_csv('SelectMicro_24/Analysis/Qitta/data/features_genus_db1629.csv',index_col=0)
    cols_name = data.columns
    data = pd.DataFrame(FS.relative_abundance(data), columns=cols_name)
    meta_2 = pd.read_csv('SelectMicro_24/Analysis/Qitta/data/meta_genus_db1629.csv',index_col=0)
    y = meta_2['ibd']
    print(pd.Series(y).value_counts())

    custom_order = ['nonIBD', 'CD', 'UC']
    custom_mapping = {category: i for i, category in enumerate(custom_order)}
    target_variable = [custom_mapping[category] for category in y]
    target_variable = np.array(target_variable)

    reusult = NB_model_SCV_multi(data, target_variable, SMOTE=True,k=5)
    print("ROC curve for class 1")
    metric.plot_multiclass_roc_cv(reusult['y_true'], reusult['y_pred_prob'], class_index=1, n_classes=3, class_label='Class 1',save_path="/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code/test_result/ROC_1.png")
    
"""   
    reusult = SVM_model_SCV_multi(data, target_variable, SMOTE=True,k=5)
    print("ROC curve for class 1")
    metric.plot_multiclass_roc_cv(reusult['y_true'], reusult['y_pred_prob'], class_index=1, n_classes=3, class_label='Class 1',save_path="/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code/test_result/ROC_1.png")
    print("SHAP plot for class 1")
    metric.plot_SHAP_multiclass(reusult['SHAP_full'],reusult['x_true'],class_index=1,
                         save_path="/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code/test_result/SHAP_1.png")

    print("ROC curve for class 2")
    metric.plot_multiclass_roc_cv(reusult['y_true'], reusult['y_pred_prob'], class_index=2, n_classes=3, class_label='Class 1',save_path = "/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code/test_result/ROC_2.png")
    print("SHAP plot for class 2")
    metric.plot_SHAP_multiclass(reusult['SHAP_full'],reusult['x_true'],class_index=2,
                         save_path="/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code/test_result/SHAP_2.png")
"""
