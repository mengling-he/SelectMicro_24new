import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from sklearn import preprocessing, __all__, svm
from imblearn.over_sampling import SMOTE
from sklearn import svm, datasets
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import shap
import time


import RunML
import metric



def split_and_scale_data(features, labels, test_size=0.3, random_state=5):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    X_train_scaled, X_test_scaled = standard_scale(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test



def perform_SMOTE(X, y, k_neighbors=5, random_state=1982):
    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_sm, y_sm = sm.fit_resample(X, y)

    return X_sm, y_sm








def LassoFeatureSelection(X,y,alpha=0.05,tol=0.01):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y= le.fit_transform(y)
    
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_scaled, y)

    selected_features = np.where(clf.coef_ != 0)[0]
    
    X=X[:,selected_features]
    return X,selected_features


"""
def CFValidation_AUCstatistic(X,y,classifier = svm.SVC(kernel='linear', probability=True),k=5):# test this
    cv = StratifiedKFold(n_splits=k, shuffle=True,random_state = 777)
    aucs = []
   # tprs = []
   # mean_fpr = np.linspace(0, 1, 100)
   # fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        # Predict probabilities
        y_prob = classifier.predict_proba(X[test])[:, 1]
        # Calculate AUC
        auc = roc_auc_score(y[test], y_prob)
        aucs.append(auc)
    #return aucs.mean(), aucs.std()
    mean_auc = np.mean(aucs)
    return mean_auc


# calculate the stratified-cross-validation score mean and std
# score : accuracy,f1,roc_auc...
# this function not be tested
def CFValidation_score(X, y,k=5,classifier = svm.SVC(kernel='linear', probability=True),score='accuracy'):# test this
    cv = StratifiedKFold(n_splits=k, shuffle=True,random_state = 777)
    acs = []
    results = cross_val_score(classifier, X_train=X, y_train=y, cv = cv,scoring=score)#scoring, only a single metric is permitted
    return results.mean(),results.std()
"""

"""refer to the function runClassifier_FScompare below
def CalculateAUClist(data_subsets,y,classifiers,N):
# the first input is a dictionary of the 4 dataset with different feature selection method and respect dataset
# y is the response variable
# classifiers is a dictionary of classifiers
# N is the times of replicates for random selection and lasso selection 
    Nselection = data_subsets.get('SelectMicro').shape[1] # the number of features selection by the method, will use this in random selection 
    for datatype, subset in data_subsets.items():
        if datatype == "AllFeatures" or datatype == "SelectMicro":
            start_time = time.time()
            auc_scores = {}
            for name, clf in classifiers.items():
                auc = CFValidation_AUCstatistic(subset,y,classifier= clf)
                auc_scores[name] = auc
            print(f"datatype '{datatype}': {auc_scores}")
            end_time = time.time()
            print(f"{datatype} took {end_time - start_time:.2f} seconds")
        elif datatype == "Random":
            start_time = time.time()
            auc_list  = []
            for ii in range(N):
                auc_scores = {}
                random.seed(1992)
                randomFeatures=random.sample(list(range(np.shape(subset)[1])), Nselection)
                X_randomFeatures=np.array(subset)[:,randomFeatures]
                for name, clf in classifiers.items():
                    auc = CFValidation_AUCstatistic(X_randomFeatures,y,classifier= clf)
                    auc_scores[name] = auc
                auc_list.append(auc_scores)
            auc_pd = pd.DataFrame(auc_list)
            print("Random Selection",auc_pd.mean())
            end_time = time.time()
            print(f"Random Selection took {end_time - start_time:.2f} seconds")
        elif datatype == "Lasso":
            start_time = time.time()
            auc_list  = []
            for ii in range(N):
                auc_scores = {}
                X_Lasso=RunML.LassoFeatureSelection(subset,y)
                for name, clf in classifiers.items():
                    auc = CFValidation_AUCstatistic(X_Lasso,y,classifier= clf)
                    auc_scores[name] = auc
                auc_list.append(auc_scores)
            auc_pd = pd.DataFrame(auc_list)
            print("LASSO Selection",auc_pd.mean())
            end_time = time.time()
            print(f"LASSO Selection took {end_time - start_time:.2f} seconds")
        else:
            print("The feature selection type is not included")

"""

"""refer to the function cross_fold_validation below
def run_RF(X, y, k=5):
     #Initialize the RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Lists to store the results
    accuracies = []
    roc_aucs = []
    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Train the model
        rf.fit(X_train, y_train)
    
        # Make predictions
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
    
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
    
        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)
    return np.mean(accuracies), np.mean(roc_aucs)
"""



def cross_fold_validation(X, y, classifier_name, k=5):
# this function  will use the  default classifier to do cross validarion
# return the mean of accuracy and AUC of all folds. return the shuffled actual y, predicted y and predicted prob
    if classifier_name == "RF":
        clf = RandomForestClassifier(n_jobs=5,random_state=777)
    elif classifier_name == "SVM":
        clf = svm.SVC(kernel='linear', probability=True, random_state=777)
    else:
        print("The classifier is not included")
        return None, None

    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=777)

    # Lists to store the results
    accuracies = []
    roc_aucs = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for train_index, test_index in kf.split(X, y):
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_sm, y_train_sm = perform_SMOTE(X_train, y_train)
        # Train the model
        clf.fit(X_train_sm, y_train_sm)

        # Make predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)
        #print(f'Fold Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}')

    # Calculate the mean accuracy and ROC AUC
    mean_accuracy = np.mean(accuracies)
    mean_roc_auc = np.mean(roc_aucs)

    #print(f'Mean Accuracy: {mean_accuracy:.4f}')
    #print(f'Mean ROC AUC: {mean_roc_auc:.4f}')

    return mean_accuracy, mean_roc_auc, y_true_all, y_pred_all, y_prob_all



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
        # rocscore = roc_auc_score(y, y_prob)

        #print('Calculating metrics for:', label)
        #print("Accuracy:", accuracy_score(y, y_pred))
        #print("Precision:", precision_score(y, y_pred))
        #print("Recall:", recall_score(y, y_pred))
        #print("AUC:", rocscore)
        return y_pred, y_prob


"""
def runAUC_FScompare(data_subsets,y,classifiers,N): # fine tune the classifier
# the first input is a dictionary of the 4 dataset with different feature selection method and respect dataset
# y is the response variable
# classifiers: RF or SVM
# N is the times of replicates for random selection and lasso selection 
    
    Nselection = data_subsets.get('SelectMicro').shape[1] # the number of features selection by the method
    
    if classifiers == "RF":
        print("Using fine tuned RF classifier")
        for datatype, subset in data_subsets.items():
            if datatype == "AllFeatures" or datatype == "SelectMicro" or datatype == "Lasso":
                start_time = time.time()
                y_pred, y_prob = run_RF_kfolds(subset, y, param_grid=None, label=datatype, title=None)
                print("Accuracy:", accuracy_score(y, y_pred))
                print("AUC:",roc_auc_score(y, y_prob))
                end_time = time.time()
                print(auc)
                print(f"{datatype} took {end_time - start_time:.2f} seconds")
                
            elif datatype == "Random":
                start_time = time.time()
                random.seed(1992)
                df = data_subsets.get('Random')
                columns = df.columns.tolist()
                auc_scores = []
                for _ in range(N):
                    selected_columns = random.sample(columns, N)
                    new_df = df[selected_columns]
                    auc = run_RF_kfolds(new_df, y, param_grid=None, label=datatype, title=None, k=5)
                    auc_scores.append(auc)
                if len(auc_scores) >0:
                    print("Random Selection",sum(auc_scores) / len(auc_scores))
                else:
                    print("Random Selection 0")
                end_time = time.time()
                print(f"Random Selection took {end_time - start_time:.2f} seconds")
                
            else:
                print("The feature selection type is not included")
    else:
        print("the classifier is not included")
"""



# an update function of runAUC_FScompare- no fine tune
def runClassifier_FScompare(data_subsets,y,N,classifiers): # fine tune the classifier
# the first input is a dictionary of the 4 dataset with different feature selection method and respect dataset
# y is the response variable
# classifiers: [RF,SVM]
# N is the times of replicates for random selection and lasso selection 
# print the AUC  and accuracy, return the prediction results of cross validation 
    Nselection = data_subsets.get('SelectMicro').shape[1] # the number of features selection by the method
    results = {}#  results  to keep the acc and auc
    results_cm ={} # results  to keep the actual y and predicted y and predicted prob
    for datatype, subset in data_subsets.items():
        results[datatype] = {}
        results_cm[datatype] = {}
        if datatype == "AllFeatures" or datatype == "SelectMicro" or datatype == "Lasso":
            for clf in classifiers:
                acc, auc,  y_actual, y_pred, y_prob = cross_fold_validation(subset, y,clf)
                results[datatype][f"{clf}_Accuracy"] = acc
                results[datatype][f"{clf}_AUC"] = auc
                results_cm[datatype][clf] = np.array([y_actual,y_pred,y_prob])

        elif datatype == "Random":
            df = data_subsets.get('Random')# which is an numpy array not a df
            num_columns = df.shape[1]  # Get the number of columns in the DataFrame
            acc_allrun = []
            auc_allrun = []
            random.seed(1992)
            for _ in range(N):
                acc_eachrun = []
                auc_eachrun = []
                selected_indices = random.sample(range(num_columns), Nselection)
                new_df = df[:, selected_indices]
                for clf in classifiers:
                    acc, auc, _, _, _= cross_fold_validation(new_df, y,clf)
                    acc_eachrun.append(acc)
                    auc_eachrun.append(auc)
                acc_allrun.append(acc_eachrun)
                auc_allrun.append(auc_eachrun)
            acc_allrun = np.array(acc_allrun)
            auc_allrun = np.array(auc_allrun)

            acc_list = np.average(acc_allrun, axis=0)
            auc_list = np.average(auc_allrun, axis=0)
            for index, clf in enumerate(classifiers):
                results[datatype][f"{clf}_Accuracy"] = acc_list[index]
                results[datatype][f"{clf}_AUC"] = auc_list[index]
        else:
            print("The feature selection type is not included")
    results_df = pd.DataFrame(results).T
    print(results_df)
    return results_cm # return a dict of dict each dict is for a feature selection method, each sub dict is a dict of arrays, each array is the 3 columns of y for res classifier




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


def  sharp_value(X,y,classifier_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_sm, y_train_sm = perform_SMOTE(X_train, y_train)

    if classifier_name == "RF":
        clf = RandomForestClassifier(n_jobs=5,random_state=777)
    elif classifier_name == "SVM":
        clf = svm.SVC(kernel='linear', probability=True, random_state=777)
    else:
        raise ValueError('The classifier is not included')

    
    # Train an  model
    model = clf.fit(X_train_sm, y_train_sm)
    
    # Create a SHAP explainer
    explainer = shap.Explainer(model, X_train_sm)
    
    # Calculate SHAP values for the test data
    shap_values = explainer(X_test)
    shap_values_2d = shap_values[:, :, 0]
    # Plot the summary plot
    shap.summary_plot(shap_values_2d, X_test)

    # Plot the waterfall plot for a single prediction
    shap.waterfall_plot(shap_values_2d[0])

    # Plot the dependence plot for a specific feature
    # shap.dependence_plot("MedInc", shap_values, X_test)
    
    



