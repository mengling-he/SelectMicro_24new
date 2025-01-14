import pandas as pd
import numpy as np
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn import preprocessing, __all__, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
#from sklearn import svm, datasets
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, LeaveOneOut
import tensorflow as tf
from tensorflow.keras import layers
import time



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
    
    le = LabelEncoder()
    y= le.fit_transform(y)
    
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_scaled, y)

    selected_features = np.where(clf.coef_ != 0)[0]
    
    X=X[:,selected_features]
    return X,selected_features


def LassoFS_CV(X,y, param_grid=None):
    """
    Perform feature selection using  Lasso regression by tuning alpha using 5-fold CV

    Parameters:
        X: ndarray
            Features.
        y: ndarray
            Labels.
   
    Returns:
        2 results : the selected array and the selected features
    """
    if param_grid == None:
        param_grid = {'alpha': np.array([0.001, 0.005, 0.01, 0.05, 0.1])}  # Smaller range of alphas

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y_encoded= le.fit_transform(y)

    # Initialize Lasso regression model
    model = linear_model.Lasso(max_iter=10000)
    
    # Set up GridSearchCV
    clf = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    # Perform the grid search
    clf.fit(X_scaled,y_encoded)

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

# Define the Autoencoder Model
def create_AE(input_dim=1, latent_dim=100, activation='relu', loss='mae', optimizer='adam'):
    autoencoder = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        layers.Flatten(),
        layers.Dense(latent_dim, activation=activation),
        layers.Dense(input_dim, activation='sigmoid')  # Decoder to reconstruct the input
    ])
    
    autoencoder.compile(loss=loss, optimizer=optimizer, metrics=['mse'])
    
    return autoencoder

# Manual GridSearchCV function
def run_AE(X_train_scaled, X_test_scaled, param_grid=None):
    if param_grid is None:
        param_grid = {
            'input_dim': [X_train_scaled.shape[1]], 
            'latent_dim': [10, 25, 50, 100],
            'activation': ['relu', 'sigmoid', 'tanh'],
            'loss': ['mae'],#, 'binary_crossentropy'],#
            'optimizer': ['sgd', 'adam'],
            'epochs': [10],
            'batch_size': [32]
        }

    # Manually define a function for model training
    def fit_model(input_dim, latent_dim, activation, loss, optimizer, epochs, batch_size):
        autoencoder = create_AE(input_dim=input_dim, latent_dim=latent_dim, activation=activation, 
                                 loss=loss, optimizer=optimizer)
        
        # Fit the model with validation data
        autoencoder.fit(X_train_scaled, X_train_scaled, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_test_scaled, X_test_scaled), verbose=0)
        return autoencoder

    # Custom GridSearchCV logic
    best_model = None
    best_params = None
    best_score = float('inf')

    for latent_dim in param_grid['latent_dim']:
        for activation in param_grid['activation']:
            for loss in param_grid['loss']:
                for optimizer in param_grid['optimizer']:
                    for epochs in param_grid['epochs']:
                        for batch_size in param_grid['batch_size']:
                            
                            # Fit the model with a specific combination of hyperparameters
                            autoencoder = fit_model(X_train_scaled.shape[1], latent_dim, activation, loss, optimizer, epochs, batch_size)
                            
                            # Get the validation loss (first item in the returned list)
                            val_loss = autoencoder.evaluate(X_test_scaled, X_test_scaled, verbose=0)[0]
                            # Track the best parameters based on validation loss
                            if val_loss < best_score:
                                best_score = val_loss
                                best_params = {
                                    'latent_dim': latent_dim,
                                    'activation': activation,
                                    'loss': loss,
                                    'optimizer': optimizer,
                                    'epochs': epochs,
                                    'batch_size': batch_size
                                }
                                best_model = autoencoder
    print(f"latent dimension: {latent_dim}, activation:{activation}, opt:{optimizer}")               
    # Ensure the best model is built
    best_model.predict(X_train_scaled[:1])  # Call the model on one sample to build it

    # **New Fix: Call the encoder with proper outputs**
    encoder_layer = tf.keras.Sequential(best_model.layers[:2])  # Extract only the encoder layers
    
    # Ensure the encoder is also built
    encoder_layer.predict(X_train_scaled[:1])  # Trigger the encoder to initialize

    # Generate latent features from encoder representation
    AE_train = pd.DataFrame(encoder_layer.predict(X_train_scaled))
    AE_train = AE_train.add_prefix('feature_')


    AE_test = pd.DataFrame(encoder_layer.predict(X_test_scaled))
    AE_test = AE_test.add_prefix('feature_')
    
    return AE_train, AE_test




# --------------------------------------------------------------------------------------------------#

def cross_fold_validation(X, y, classifier_name, SMOTE=False,k=5):
    """
    calculate the model results. This functionn is used in another function

    Parameters:
        X: ndarray
            Features.
        y: ndarray
            Labels.
   classifier_name:
        RF or SVM
    SMOTE (bool): Whether to apply SMOTE oversampling (default: False)
    Returns:
        5 results : mean of accuracy and AUC of all folds.  the shuffled actual y, predicted y and predicted prob
    """ 
    if classifier_name == "RF":
        clf = RandomForestClassifier(n_jobs=5,random_state=777)
    elif classifier_name == "SVM":
        clf = svm.SVC(kernel='rbf', probability=True, random_state=777)#{'precomputed', 'poly', 'rbf', 'sigmoid', 'linear'}
    else:
        print("The classifier is not included")
        return None

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

        # Apply SMOTE if specified
        if SMOTE:
            X_train, y_train = perform_SMOTE(X_train, y_train)

        clf.fit(X_train, y_train)

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

    result = {'mean_accuracy': mean_accuracy,
              'mean_auc': mean_roc_auc,
              'y_true': y_true_all,
              'y_pred':y_pred_all,
              'y_pred_prob': y_prob_all}

    return result


def ML_model_SCV(X, y, classifier_name, SMOTE=False,k=5):
    """
    calculate the ML model results using stratefied cross validation. (This functionn is used in another function)

    Parameters:
        X: ndarray
            Features.
        y: ndarray
            Labels.
   classifier_name:
        RF or SVM or catboost or NB(naive bayes) 
    SMOTE (bool): Whether to apply SMOTE oversampling (default: False)
    Returns:
        5 results : average of accuracy and AUC of all folds; the shuffled actual y, predicted y and predicted prob
    """ 
    if classifier_name == "RF":
        clf = RandomForestClassifier(n_jobs=5,random_state=777)
    elif classifier_name == "SVM":
        clf = svm.SVC(kernel='rbf', probability=True, random_state=777)#{'precomputed', 'poly', 'rbf', 'sigmoid', 'linear'}
    
    elif classifier_name == "CatBoost":
        clf = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6,verbose=0, random_state=777)
        
    elif classifier_name == "NB":
        clf = GaussianNB()
    else:
        print("The classifier is not included")
        return None

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

        # Apply SMOTE if specified
        if SMOTE:
            X_train, y_train = perform_SMOTE(X_train, y_train)

        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)

        y_true_all.extend(y_test)# Using extend to add multiple elements
        y_pred_all.extend(y_pred)
        y_prob_all.extend([float(prob) for prob in y_prob])
        #print(f'Fold Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}')

    # Calculate the mean accuracy and ROC AUC
    mean_accuracy = np.mean(accuracies)
    mean_roc_auc = np.mean(roc_aucs)

    result = {'mean_accuracy': mean_accuracy,
              'mean_auc': mean_roc_auc,
              'y_true': y_true_all,
              'y_pred':y_pred_all,
              'y_pred_prob': y_prob_all}
    """
    for key, value in result.items():
        print(f"Key: {key}, Type: {type(value)}")
    """
    return result


# --------------------------------------------------------------------------------------------------#

# an update function of runAUC_FScompare- no fine tune
def runClassifier_FScompare(data_subsets,y,N,classifiers,SMOTE=False): # fine tune the classifier
# the first input is a dictionary of the 4 dataset with different feature selection method and respect dataset
# y is the response variable
# classifiers: [RF,SVM]
# N is the times of replicates for random selection
# print the AUC  and accuracy, return the prediction results of cross validation 
    if 'SelectMicro' in data_subsets and data_subsets['SelectMicro'] is not None:
        Nselection = data_subsets['SelectMicro'].shape[1]  # the number of features selected by the method
    else:
        Nselection = 0
        
    results = {}#  results  to keep the acc and auc
    results_cm ={} # results  to keep the actual y and predicted y and predicted prob
    
    for datatype, subset in data_subsets.items():
        
        results[datatype] = {}
        results_cm[datatype] = {}    

        if datatype == "Random":
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
                    results_clf = ML_model_SCV(new_df, y,clf,SMOTE=SMOTE)
                    acc = results_clf['mean_accuracy']
                    auc = results_clf['mean_auc']
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
                results_cm[datatype][clf] = np.array([y_actual,y_pred,y_prob])
                
    results_df = pd.DataFrame(results).T
    print(results_df)
    return results_cm # return a dict of dict each dict is for a feature selection method, each sub dict is a dict of arrays, each array is the 3 columns of y for res classifier




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











