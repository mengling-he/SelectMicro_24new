import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn import preprocessing, __all__, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier  # Import XGBoost
from imblearn.over_sampling import SMOTE
#from sklearn import svm, datasets
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, LeaveOneOut
import tensorflow as tf
from tensorflow.keras import layers
import shap
import time
import sys
sys.path.append('./Code')
import metric



def split_and_scale_data(features, labels, test_size=0.3, random_state=777):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def perform_SMOTE(X, y, k_neighbors=5, random_state=777):
    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_sm, y_sm = sm.fit_resample(X, y)

    return X_sm, y_sm


# --------------------------------------------------------------------------------------------------#

# +
def LassoFeatureSelection(X,y,alpha=0.05,tol=0.01):
    """
    Perform feature selection using default Lasso regression

    Parameters:
        X: ndarray
            Features.
        y: ndarray
            Labels.
   
    Returns:
        2 results : the selected array and the selected features
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_scaled, y)

    selected_features = np.where(clf.coef_ != 0)[0]
    
    X=X[:,selected_features]
    return X,selected_features


def LassoFS_CV(X,y, param_grid=None):
    if param_grid == None:
        param_grid = {'alpha': np.array([0.001, 0.005, 0.01, 0.05, 0.1])}  # Smaller range of alphas

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize Lasso regression model
    model = linear_model.Lasso(max_iter=10000)
    
    # Set up GridSearchCV
    clf = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    # Perform the grid search
    clf.fit(X_scaled,y)

    # Best Lasso model and alpha
    best_lasso = clf.best_estimator_
    best_score = clf.best_score_
    print(f"Best alpha: {clf.best_params_['alpha']}")
    #print(f"Best cross-validated score (accuracy): {best_score}")

    # Identify selected features (non-zero coefficients)
    selected_features = np.where(best_lasso.coef_ != 0)[0]
    #print(f"Selected features: {selected_features}")
    X=X[:,selected_features]
    return X,selected_features
# --------------------------------------------------------------------------------------------------#


# +

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


# -

# --------------------------------------------------------------------------------------------------#
# an update function of runAUC_FScompare- no fine tune
def runClassifier_FScompare(data_subsets,y,classifiers,SMOTE=False): # fine tune the classifier
# the first input is a dictionary of the 4 dataset with different feature selection method and respect dataset
# y is the response variable
# classifiers: [RF,SVM]
# N is the times of replicates for random selection
# print the AUC  and accuracy, return the prediction results of cross validation 
    # Label encoding to handle 'No'/'Yes' as '0'/'1'
    le = LabelEncoder()
    y = le.fit_transform(y)  
    print("Classes:", le.classes_)#[0,1]
    results = {}#  results  to keep the acc and auc
    results_cm ={} # results  to keep the actual y and predicted y and predicted prob
    
    for datatype, subset in data_subsets.items():
        
        results[datatype] = {}
        results_cm[datatype] = {}    

        for clf in classifiers:
            #print(f"{datatype}_{clf}")
            results_clf = ML_model_SCV(subset, y,clf,SMOTE=SMOTE)
            acc = results_clf['mean_accuracy']
            auc = results_clf['mean_auc']
            y_actual  = results_clf['y_true']
            y_pred = results_clf['y_pred']
            y_prob = results_clf['y_pred_prob']
        
            results[datatype][f"{clf}_Accuracy"] = acc
            results[datatype][f"{clf}_AUC"] = auc
            #results[datatype][f"{clf}_Accuracy_std"] = results_clf['std_accuracy']
            #results[datatype][f"{clf}_AUC_std"] = results_clf['std_auc']
            results[datatype][f"{clf}_mcc"] = results_clf['mean_mcc']
            results[datatype][f"{clf}_F"] = results_clf['mean_f']
            results_cm[datatype][clf] = np.array([y_actual,y_pred,y_prob])
            
    results_df = pd.DataFrame(results).T
        # Adjust display options
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Adjust the display width
    pd.set_option('display.colheader_justify', 'center')  # Center-align column headers

    print(results_df)
    return results_cm # return a dict of dict each dict is for a feature selection method, each sub dict is a dict of arrays, each array is the 3 columns of y for res classifier


# an update function of runAUC_FScompare- no fine tune
def runClassifier_random(data,y,Nselection,clf='RF',iteration=30,SMOTE=False):
    df = data # which is an numpy array not a df
    num_columns = df.shape[1]  # Get the number of columns in the DataFrame
    
    results = []
    random.seed(1992)
    for _ in range(iteration):
        selected_indices = random.sample(range(num_columns), Nselection)
        new_df = df[:, selected_indices]
        
        results_clf = ML_model_SCV(new_df, y,clf,SMOTE=SMOTE)
        y_actual  = results_clf['y_true']
        y_pred = results_clf['y_pred']
        metric_summary_tb = metric.metric_tb(y_actual,y_pred)


        result = {"Accuracy": results_clf['mean_accuracy'],
                  "AUC":results_clf['mean_auc'],
                  "ACC": metric_summary_tb['Accuracy'],
                  "Precision": metric_summary_tb['Precision'],
                  "Recall": metric_summary_tb['Recall'],
                  "MCC": metric_summary_tb['Mcc']}
                  
        results.append(result)
    
    # Iterate over each dictionary in the list
    for result in results:
        for key, value in result.items():
            # Accumulate sums and counts for each key
            sums[key] = sums.get(key, 0) + value
            counts[key] = counts.get(key, 0) + 1

    # Calculate the averages
    average_result = {key: sums[key] / counts[key] for key in sums}

    # Adjust display options
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Adjust the display width
    pd.set_option('display.colheader_justify', 'center')  # Center-align column headers
    

    print(average_result)
    return results_df 


# --------------------------------------------------------------------------------------------------#
# ----------------------------SHAP---------------------------------------------------#


# ### Random Forest Classifier

def RF_model_SCV(X, y, plot=False,SMOTE=False,k=5,y_base = 0):
    # Initialize the classifier
    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=777)

    # Lists to store the results
    accuracies = []
    mcc_s = []
    f_scores = []
    
    # preparation for ROC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
     
    # preparation for SHAP
    enriched_all = pd.DataFrame(X.columns, columns = ['Taxa'])
    shap_values_all = np.empty((0, X.shape[1], len(np.unique(y))))#In tree-based models like Random Forest, SHAP values are computed for each class separately, leading to a 3D array.
    test_sets_ix = []
    idx = 0
    
    # preparation for response
    y_pred_all = []
    y_prob_all = []
    y_true_all = []
    
    fig, ax = plt.subplots(figsize=(6, 6))
    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE if specified
        if SMOTE:
            X_train, y_train = perform_SMOTE(X_train, y_train)
        
        clf = RandomForestClassifier(n_jobs=5, random_state=777)
     
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        y_prob1 = y_prob[:, y_base]
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        #roc_auc = roc_auc_score(y_test, y_prob)# the function result is different with the ROC curve, choose the one from the curve
        mcc = metric.mcc_score(y_test,y_pred)
        f_score = f1_score(y_test,y_pred)
        
        
        # store the result
        accuracies.append(accuracy)
        mcc_s.append(mcc)
        f_scores.append(f_score)
        
        # ROC curve
        viz = RocCurveDisplay.from_estimator(
            clf,
            X_test,
            y_test,
            name=f"iteration {idx}",
            alpha=0.3,
            lw=1,ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # SHAP values
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        shap_obj = explainer(X_test)
        shap_values_all = np.concatenate((shap_values_all, shap_values), axis=0) 
        test_sets_ix.append(test_index)
        
        high_index_df = pd.DataFrame(shap_obj.data, columns=shap_obj.feature_names, index=X_test.index)
        high_index = high_index_df.idxmax()# finds the feature's index with the highest value for each sample.
        shap_1 = pd.DataFrame(shap_values[:,:,y_base], columns=shap_obj.feature_names, index=X_test.index)
    
        enriched = list()
        
        for v, i in high_index.items():
            sv = shap_1[v].loc[i]
            if sv<0:
                sv = "Other"
            else:
                sv = f"Level {y_base}"
            enriched.append(
                {
                    'Taxa': v,
                    'enriched': sv
                }
            )
        enriched = pd.DataFrame(enriched)
        enriched.rename(columns={'enriched': 'enriched{}'.format(idx+1)}, inplace=True)
        enriched_all = enriched_all.merge(enriched, on='Taxa', how='outer')

        idx += 1
        
       
        y_true_all.extend(y_test)# Using extend to add multiple elements
        y_pred_all.extend(y_pred)
        #y_prob_all.extend([float(prob) for prob in y_prob])
    
    # sort x based on the test dataset index
    test_sets_ix = np.concatenate(test_sets_ix)
    x_true_df = pd.DataFrame(X.iloc[test_sets_ix,:], columns=X.columns)
    
    # continue ROC
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()
     
    result = {'y_information':{'y_true':y_true_all,
                              'y_pred':y_pred_all
                               #,'y_prob':
                              },
              'x_true': x_true_df,
              'SHAP':shap_values_all,
             'enrich': enriched_all}
    print('Accuracy: %.3f (%.3f), F1: %.3f (%.3f),MCC: %.3f (%.3f),AUC: %.3f (%.3f)' % 
          (np.mean(accuracies), np.std(accuracies), np.mean(f_scores), np.std(f_scores),
           np.mean(mcc_s), np.std(mcc_s),mean_auc, std_auc))
    if plot:
        shap.summary_plot(shap_values_all[:,:,y_base], x_true_df)
    return result


# ### XGBoost Model

def XGBoost_model_SCV(X, y, plot=False,SMOTE=False,k=5,y_base = 0):
    # Initialize the classifier
    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=777)

    # Lists to store the results
    accuracies = []
    mcc_s = []
    f_scores = []
    
    # preparation for ROC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
     
    # preparation for SHAP
    enriched_all = pd.DataFrame(X.columns, columns = ['Taxa'])
    shap_values_all = np.empty((0, X.shape[1]))
    test_sets_ix = []
    idx = 0
    
    # preparation for response
    y_pred_all = []
    y_prob_all = []
    y_true_all = []
    
    fig, ax = plt.subplots(figsize=(6, 6))
    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE if specified
        if SMOTE:
            X_train, y_train = perform_SMOTE(X_train, y_train)
        
        clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=777)
     
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        y_prob1 = y_prob[:, y_base]
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        #roc_auc = roc_auc_score(y_test, y_prob)# the function result is different with the ROC curve, choose the one from the curve
        mcc = metric.mcc_score(y_test,y_pred)
        f_score = f1_score(y_test,y_pred)
        
        
        # store the result
        accuracies.append(accuracy)
        mcc_s.append(mcc)
        f_scores.append(f_score)
        
        # ROC curve
        viz = RocCurveDisplay.from_estimator(
            clf,
            X_test,
            y_test,
            name=f"iteration {idx}",
            alpha=0.3,
            lw=1,ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # SHAP values
        explainer = shap.TreeExplainer(clf)
        shap_obj = explainer(X_test)
        shap_values = shap_obj.values
        shap_values_all = np.concatenate((shap_values_all, shap_values), axis=0) 
        test_sets_ix.append(test_index)
        
        high_index_df = pd.DataFrame(shap_obj.data, columns=shap_obj.feature_names, index=X_test.index)
        high_index = high_index_df.idxmax()# finds the feature's index with the highest value for each sample.
        shap_1 = pd.DataFrame(shap_values, columns=shap_obj.feature_names, index=X_test.index)
    
        enriched = list()
        
        for v, i in high_index.items():
            sv = shap_1[v].loc[i]
            if sv<0:
                sv = "Other"
            else:
                sv = f"Level {y_base}"
            enriched.append(
                {
                    'Taxa': v,
                    'enriched': sv
                }
            )
        enriched = pd.DataFrame(enriched)
        enriched.rename(columns={'enriched': 'enriched{}'.format(idx+1)}, inplace=True)
        enriched_all = enriched_all.merge(enriched, on='Taxa', how='outer')

        idx += 1
        
       
        y_true_all.extend(y_test)# Using extend to add multiple elements
        y_pred_all.extend(y_pred)
        #y_prob_all.extend([float(prob) for prob in y_prob])
    
    # sort x based on the test dataset index
    test_sets_ix = np.concatenate(test_sets_ix)
    x_true_df = pd.DataFrame(X.iloc[test_sets_ix,:], columns=X.columns)
    
    # continue ROC
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()
     
    result = {'y_information':{'y_true':y_true_all,
                              'y_pred':y_pred_all
                               #,'y_prob':
                              },
              'x_true': x_true_df,
              'SHAP':shap_values_all,
             'enrich': enriched_all}
    print('Accuracy: %.3f (%.3f), F1: %.3f (%.3f),MCC: %.3f (%.3f),AUC: %.3f (%.3f)' % 
          (np.mean(accuracies), np.std(accuracies), np.mean(f_scores), np.std(f_scores),
           np.mean(mcc_s), np.std(mcc_s),mean_auc, std_auc))
    if plot:
        shap.summary_plot(shap_values_all, x_true_df)
    return result


# ### Naive Bayes Classifier

def NB_model_SCV(X, y, plot=False,SMOTE=False,k=5,y_base = 0):
    # Initialize the classifier
    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=777)

    # Lists to store the results
    accuracies = []
    mcc_s = []
    f_scores = []
    
    # preparation for ROC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
     
    # preparation for SHAP
    enriched_all = pd.DataFrame(X.columns, columns = ['Taxa'])
    shap_values_all = np.empty((0, X.shape[1]))# KernelExplainer (used for Naïve Bayes) returns a single SHAP value per feature without per-class separation.
    test_sets_ix = []
    idx = 0
    
    # preparation for response
    y_pred_all = []
    y_prob_all = []
    y_true_all = []
    
    fig, ax = plt.subplots(figsize=(6, 6))
    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE if specified
        if SMOTE:
            X_train, y_train = perform_SMOTE(X_train, y_train)
        
        # Step 3: Scale Data (GaussianNB assumes normally distributed data)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = GaussianNB()
     
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)
        y_prob1 = y_prob[:, y_base]
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        #roc_auc = roc_auc_score(y_test, y_prob)# the function result is different with the ROC curve, choose the one from the curve
        mcc = metric.mcc_score(y_test,y_pred)
        f_score = f1_score(y_test,y_pred)
        
        
        # store the result
        accuracies.append(accuracy)
        mcc_s.append(mcc)
        f_scores.append(f_score)
        
        # ROC curve
        viz = RocCurveDisplay.from_estimator(
            clf,
            X_test_scaled,
            y_test,
            name=f"iteration {idx}",
            alpha=0.3,
            lw=1,ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # SHAP values
        
        
        
        explainer = shap.Explainer(clf.predict,X_train_scaled)
        shap_obj = explainer(X_test,max_evals=4000)
        shap_values = shap_obj.values
        #shap_obj = explainer(X_test)
        shap_values_all = np.concatenate((shap_values_all, shap_values), axis=0) 
        test_sets_ix.append(test_index)
        
        high_index_df = pd.DataFrame(shap_obj.data, columns=shap_obj.feature_names, index=X_test.index)
        high_index = pd.DataFrame(shap_obj.data, columns=shap_obj.feature_names, index=X_test.index).idxmax()# finds the feature's index with the highest value for each sample. returns a seires
        shap_1 = pd.DataFrame(shap_values, columns=shap_obj.feature_names, index=X_test.index)
        enriched = list()
        
        for v, i in high_index.items():
            sv = shap_1[v].loc[i]
            if sv<0:
                sv = "Other"
            else:
                sv = f"Level {y_base}"
            enriched.append(
                {
                    'Taxa': v,
                    'enriched': sv
                }
            )
        enriched = pd.DataFrame(enriched)
        enriched.rename(columns={'enriched': 'enriched{}'.format(idx+1)}, inplace=True)
        enriched_all = enriched_all.merge(enriched, on='Taxa', how='outer')

        idx += 1
        
       
        y_true_all.extend(y_test)# Using extend to add multiple elements
        y_pred_all.extend(y_pred)
        #y_prob_all.extend([float(prob) for prob in y_prob])
    
    # sort x based on the test dataset index
    test_sets_ix = np.concatenate(test_sets_ix)
    x_true_df = pd.DataFrame(X.iloc[test_sets_ix,:], columns=X.columns)
    
    # continue ROC
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()
     
    result = {'y_information':{'y_true':y_true_all,
                              'y_pred':y_pred_all
                               #,'y_prob':
                              },
              'x_true': x_true_df,
              'SHAP':shap_values_all,
             'enrich': enriched_all}
    print('Accuracy: %.3f (%.3f), F1: %.3f (%.3f),MCC: %.3f (%.3f),AUC: %.3f (%.3f)' % 
          (np.mean(accuracies), np.std(accuracies), np.mean(f_scores), np.std(f_scores),
           np.mean(mcc_s), np.std(mcc_s),mean_auc, std_auc))
    if plot:
        shap.summary_plot(shap_values_all, x_true_df)
    return result


# --------------------------------------------------------------------------------------------------#

def nested_loocv(X, y,classifier="SVM", smote=False):
    """
    Perform nested leave-one-out cross-validation. (loocv in parameter tuninng - inner and then loocv in testing - outer)
    suitable for small datasets like smoky data

    Parameters:
        X: ndarray
            Features.
        y: ndarray
            Labels.
        param_grid: dict
            Hyperparameter grid for tuning.
        classifier: str
            "SVM" or "RF" (Random Forest).
        smote: bool
            Whether to apply SMOTE to the training data.

    Returns:
        dict:
            Contains mean accuracy, mean AUC, and the list of metrics for each outer fold.
    """
    outer_loocv = LeaveOneOut()
    inner_loocv = LeaveOneOut()

    # Metrics to store results
    outer_accuracies = []
    #outer_aucs = []

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for train_index, test_index in outer_loocv.split(X):
        # Outer split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE to the training data in the outer loop
        if smote:
            sm = SMOTE(random_state=777)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        # Define the classifier
        if classifier == "SVM":
            model = svm.SVC(probability=True, random_state=777)
            model_param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf', 'linear'],
                'probability': [True]
            }
        elif classifier == "RF":
            model = RandomForestClassifier(random_state=777)
            model_param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion' :['gini', 'entropy'],
        }
        else:
            raise ValueError("Classifier must be 'SVM' or 'RF'.")

        # Inner loop for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=model_param_grid,
            cv=inner_loocv,
            scoring="accuracy",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Evaluate on the outer test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        #auc = roc_auc_score(y_test, y_prob)# not applicable since there is only one sample in the testing for LOOCV

        # Store results
        outer_accuracies.append(accuracy)
        #outer_aucs.append(auc)

        #To append the actual value (not a 1-element array), you access the first (and only) element with [0]
        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])
        y_prob_all.append(y_prob[0])

    # Calculate mean metrics
    mean_accuracy = sum(outer_accuracies) / len(outer_accuracies)
    #mean_auc = sum(outer_aucs) / len(outer_aucs)

    return {
        "mean_accuracy": mean_accuracy,
        #"mean_auc": mean_auc,
        "outer_accuracies": outer_accuracies,
        #"outer_aucs": outer_aucs,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_prob": y_prob_all,
    }

# an update function of runAUC_FScompare
def runClassifierCV_FScompare(data_subsets,y,N,classifiers): # fine tune the classifier
# the first input is a dictionary of the 4 dataset with different feature selection method and respect dataset
# y is the response variable
# classifiers: [RF,SVM]
# N is the times of replicates for random selection and lasso selection 
    Nselection = data_subsets.get('SelectMicro').shape[1] # the number of features selection by the method
    for datatype, subset in data_subsets.items():
        
        if datatype == "AllFeatures" or datatype == "SelectMicro" or datatype == "Lasso":
            for clf in classifiers:
                if clf == "RF":
                    #print(f"Using fine tuned RF classifier on {datatype}")
                    start_time = time.time()
                    y_pred, y_prob = run_RF_kfolds(subset, y, param_grid=None, label=datatype, title=None)
                    print("Accuracy:", accuracy_score(y, y_pred))
                    print("AUC:",roc_auc_score(y, y_prob))
                    end_time = time.time()
                    print(f"{datatype} took {end_time - start_time:.2f} seconds")
                if clf == "SVM":
                    #print(f"Using fine tuned SVM classifier on {datatype}")
                    start_time = time.time()
                    y_pred, y_prob = run_SVM_kfolds(subset, y, param_grid=None, label=datatype, title=None)
                    print("Accuracy:", accuracy_score(y, y_pred))
                    print("AUC:",roc_auc_score(y, y_prob))
                    end_time = time.time()
                    print(f"{datatype} took {end_time - start_time:.2f} seconds")
                else:
                    print("the classifier is not included")
        elif datatype == "Random":
            random.seed(1992)
            df = data_subsets.get('Random')
            columns = df.columns.tolist()
            auc_scores = []
            acc_scores = []
            for _ in range(N):
                selected_columns = random.sample(columns, N)
                new_df = df[selected_columns]
                for clf in classifiers:
                    aucs = []
                    accs = []
                    if clf == "RF":
                        y_pred, y_prob = run_RF_finetune(new_df, y, param_grid=None, label=datatype, title=None)
                        accs.append(accuracy_score(y, y_pred))
                        aucs.append(roc_auc_score(y, y_prob))
                    if clf == "SVM":
                        y_pred, y_prob = run_SVM_finetune(new_df, y, param_grid=None, label=datatype, title=None)
                        accs.append(accuracy_score(y, y_pred))
                        aucs.append(roc_auc_score(y, y_prob))
                    else:
                        print("the classifier is not included")
                acc_scores.append(accs)
                auc_scores.append(aucs)   
            auc_array = np.array(auc_scores)
            mean_rf = np.mean(auc_array[:, 0])
            mean_svm = np.mean(auc_array[:, 1])
            acc_array = np.array(acc_scores)
            mean_rf_acc = np.mean(acc_array[:, 0])
            mean_svm_acc = np.mean(acc_array[:, 1])
            print(f"Using fine tuned RF classifier on {datatype}")
            print(mean_rf_acc)
            print(mean_rf)
            print(f"Using fine tuned SVM classifier on {datatype}")
            print(mean_svm_acc)
            print(mean_svm)
   
        else:
            print("The feature selection type is not included")


""" need to be updated to use a nonused data for test
def run_RF_finetune(X, y, param_grid=None, label=None, title=None, k=5):

    if param_grid == None:
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion' :['gini', 'entropy'],
        }
        

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=777),
        n_jobs=5,
        verbose=0
    )

    print('Building fine tuned RF model for label:', label)
    # Perform the grid search
    grid_search.fit(X, y)

    # Predict with the best model
    best_model = grid_search.best_estimator_
    print('Predicting for label:', label)
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]
    #rocscore = roc_auc_score(y, y_prob)
    #print('Calculating metrics for:', label)
    #print("Accuracy:", accuracy_score(y, y_pred))
    #print("Precision:", precision_score(y, y_pred))
    #print("Recall:", recall_score(y, y_pred))
    #print("AUC:", rocscore)
    return y_pred, y_prob


def run_SVM_finetune(X, y, param_grid=None, label=None, title=None, k=5):
        if param_grid == None:
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf', 'linear'],
                'probability': [True]
            }
        if label == None:
            label = y.name

        grid_search = GridSearchCV(
            estimator=svm.SVC(),
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=777),
            n_jobs=5,
            verbose=0
        )
        print('Building model for label:', label)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        print('Predicting on test data for label:', label)
        y_pred = best_model.predict(X)
        y_prob = best_model.predict_proba(X)[:, 1]
       
       # Evaluate metrics (uncomment if needed)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob)

        print(f'Metrics for {label}:')
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  AUC:       {roc_auc:.4f}")

        return y_pred, y_prob

"""




def plot_dim_reduction(X, y, method='PCA', perplexity=30, n_components=2,datalabel=None):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply dimensionality reduction
    if method.upper() == 'PCA':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X_scaled)
        title = f'PCA Plot {datalabel}'
    elif method.upper() == 'TSNE':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
        title = f't-SNE Plot {datalabel}'
    else:
        raise ValueError("Method should be either 'PCA' or 'tSNE'")

    # Create DataFrame for plotting
    df_plot = pd.DataFrame(X_reduced, columns=[f'Component {i+1}' for i in range(n_components)])
    df_plot['Label'] = y

    # Plot
    plt.figure(figsize=(4, 3))
    for label in np.unique(y):
        subset = df_plot[df_plot['Label'] == label]
        plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], label=f'Class {label}', alpha=0.7)

    plt.xlabel(df_plot.columns[0])
    plt.ylabel(df_plot.columns[1])
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()






