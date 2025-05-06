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
# 5 fold cross validation for binary classification

def ML_model_SCV(X, y, classifier_name, SMOTE=False,k=5):
     # Initialize the classifier

    
    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=777)

    # Lists to store the results
    accuracies = []
    roc_aucs = []
    mcc_s = []
    f_scores = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    X = X.values
    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE if specified
        if SMOTE:
            X_train, y_train = perform_SMOTE(X_train, y_train)
     
        if classifier_name.lower() == "rf":
            clf = RandomForestClassifier(n_jobs=5, random_state=777)
        elif classifier_name.lower() == "svm":
            clf = svm.SVC(kernel='rbf', probability=True, random_state=777)
        elif classifier_name.lower() == "catboost":
            clf = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_state=777)
        elif classifier_name.lower() == "nb":
            clf = GaussianNB()
        elif classifier_name.lower() == "xgboost":
            clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=777)
        else:
            raise ValueError("Invalid classifier_name. Please choose 'xgboost', 'catboost','NB', 'RF' or 'svm'.")     

        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        print(roc_auc)
        mcc = metric.mcc_score(y_test,y_pred)
        f_score = f1_score(y_test,y_pred)

        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)
        mcc_s.append(mcc)
        f_scores.append(f_score)
        y_true_all.extend(y_test)# Using extend to add multiple elements
        y_pred_all.extend(y_pred)
        y_prob_all.extend([float(prob) for prob in y_prob])
        #print(f'Fold Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}')

    # Calculate the mean accuracy and ROC AUC
    mean_accuracy = np.mean(accuracies)
    mean_roc_auc = np.mean(roc_aucs)
    mean_mcc = np.mean(mcc_s)
    mean_f = np.mean(f_scores)
    result = {'mean_accuracy': mean_accuracy,
              'std_accuracy':np.std(accuracies),
              'mean_auc': mean_roc_auc,
              'std_auc':np.std(roc_aucs),
              'mean_mcc':mean_mcc,
              'mean_f':mean_f,
              'y_true': y_true_all,
              'y_pred':y_pred_all,
              'y_pred_prob': y_prob_all}
    return result


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
        #y_train_pred = clf.predict(X_train)
        #y_train_prob = clf.predict_proba(X_train)
        #accuracy_train = accuracy_score(y_train, y_train_pred)
        #mcc_train = metric.mcc_score(y_train,y_train_pred)
           
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)## For multiclass, y_prob will have shape (n_samples, n_classes)

        # Evaluate the model and save the results for each fold
        accuracy = accuracy_score(y_test, y_pred)
        mcc = metric.mcc_score(y_test,y_pred)
          
        # Handle binary vs multiclass for ROC AUC
        if len(np.unique(y_train)) == 2:
            # Binary: use prob of positive class only
            f_score = f1_score(y_test,y_pred, pos_label=1, average='binary')
            y_test_prob_pos = y_prob[:, 1]
            roc_auc = roc_auc_score(y_test, y_test_prob_pos)
            #f_score_train = f1_score(y_train,y_train_pred, pos_label=1, average='binary')
            #y_train_prob_pos = y_train_prob[:, 1]
            #roc_auc_train = roc_auc_score(y_train, y_train_prob_pos)
        else:
            # Multiclass
            f_score = f1_score(y_test,y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            #f_score_train = f1_score(y_train,y_train_pred, average='weighted')
            #roc_auc_train = roc_auc_score(y_train, y_train_prob, multi_class='ovr', average='macro')# calculate the AUC for each level and  average it

        # Print results
        # print(f"==== Fold {idx+1} Metrics ====")
        # print(f"Train Accuracy: {accuracy_train:.4f} | Test Accuracy: {accuracy:.4f}")
        # print(f"Train MCC:      {mcc_train:.4f} | Test MCC:      {mcc:.4f}")
        # print(f"Train F1:       {f_score_train:.4f} | Test F1:       {f_score:.4f}")
        # print(f"Train AUC:      {roc_auc_train:.4f} | Test AUC:      {roc_auc:.4f}")

        
        accuracies.append(accuracy)
        mcc_s.append(mcc)
        f_scores.append(f_score)
        roc_aucs.append(roc_auc)

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
    
    if len(np.unique(y)) == 2:
    	shap_values_all = np.empty((0, X.shape[1]))
    else:
    	shap_values_all = np.empty((0, X.shape[1], len(np.unique(y))))
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
          
        # Handle binary vs multiclass for ROC AUC
        if len(np.unique(y_train)) == 2:
            # Binary: use prob of positive class only
            f_score = f1_score(y_test,y_pred, pos_label=1, average='binary')
            y_test_prob_pos = y_prob[:, 1]
            roc_auc = roc_auc_score(y_test, y_test_prob_pos)
            
        else:
            # Multiclass
            f_score = f1_score(y_test,y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    
        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)
        mcc_s.append(mcc)
        f_scores.append(f_score)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        shap_values_all = np.concatenate((shap_values_all, shap_values), axis=0) ####error!!!
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
          
        # Handle binary vs multiclass for ROC AUC
        if len(np.unique(y_train)) == 2:
            # Binary: use prob of positive class only
            f_score = f1_score(y_test,y_pred, pos_label=1, average='binary')
            y_test_prob_pos = y_prob[:, 1]
            roc_auc = roc_auc_score(y_test, y_test_prob_pos)
            
        else:
            # Multiclass
            f_score = f1_score(y_test,y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    
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
          
        # Handle binary vs multiclass for ROC AUC
        if len(np.unique(y_train)) == 2:
            # Binary: use prob of positive class only
            f_score = f1_score(y_test,y_pred, pos_label=1, average='binary')
            y_test_prob_pos = y_prob[:, 1]
            roc_auc = roc_auc_score(y_test, y_test_prob_pos)
            
        else:
            # Multiclass
            f_score = f1_score(y_test,y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    
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

    reusult = RF_model_SCV_multi(data, target_variable, SMOTE=True,k=5)
    print("ROC curve for class 1")
    metric.plot_multiclass_roc_cv(reusult['y_true'], reusult['y_pred_prob'], class_index=1, n_classes=2, class_label='Class 1',save_path="/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code/test_result/ROC_1.png")
    print("SHAP plot for class 1")
    metric.plot_SHAP_multiclass(reusult['SHAP_full'],reusult['x_true'],class_index=1, save_path="/lustre/isaac24/scratch/mhe8/SelectMicro_24/Code/test_result/SHAP_1.png")

    
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
