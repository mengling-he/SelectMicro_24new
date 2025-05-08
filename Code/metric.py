#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:15:05 2024

@author: menglinghe

macro score is better for balanced data
micro score is better for unbalanced data
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import shap
import os

def AUC_multiclass(y_true,y_prob):
	unique_class = set(y_true)
	y_true_bin = label_binarize(y_true, classes=unique_class)
	auc_scores = np.array([roc_auc_score(y_true_bin[:,i],[prob[i] for prob in y_prob]) for i in range(len(unique_class))])
	return auc_scores


def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
    
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict






def accuracy(y_true, y_pred):
    
    """
    Function to calculate accuracy
    -> param y_true: list of true values
    -> param y_pred: list of predicted values
    -> return: accuracy score
    """
    
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == yp:
            
            correct_predictions += 1
    
    #returns accuracy
    return correct_predictions / len(y_true)



# --------------------------------------------------------------------------------------------------#
# TP, TN, FP, FN
def true_positive(y_true, y_pred):
    
    tp = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 1 and yp == 1:
            tp += 1
    
    return tp

def true_negative(y_true, y_pred):
    
    tn = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 0 and yp == 0:
            tn += 1
            
    return tn
def false_positive(y_true, y_pred):
    
    fp = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 0 and yp == 1:
            fp += 1
            
    return fp

def false_negative(y_true, y_pred):
    
    fn = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 1 and yp == 0:
            fn += 1
            
    return fn
# --------------------------------------------------------------------------------------------------#

def macro_precision(y_true, y_pred):

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize precision to 0
    precision = 0
    
    # loop over all classes
    for class_ in list(np.unique(y_true)):
        
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        
        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)
        
        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)
        
        
        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)
        # keep adding precision for all classes
        precision += temp_precision
        
    # calculate and return average precision over all classes
    precision /= num_classes
    
    return precision

def micro_precision(y_true, y_pred):


    # find the number of classes 
    num_classes = len(np.unique(y_true))
    
    # initialize tp and fp to 0
    tp = 0
    fp = 0
    
    # loop over all classes
    for class_ in np.unique(y_true):
        
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)
        
        # calculate false positive for current class
        # and update overall tp
        fp += false_positive(temp_true, temp_pred)
        
    # calculate and return overall precision
    precision = tp / (tp + fp)
    return precision
# --------------------------------------------------------------------------------------------------#

# Computation of macro-averaged recall

def macro_recall(y_true, y_pred):

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize recall to 0
    recall = 0
    
    # loop over all classes
    for class_ in list(np.unique(y_true)):
        
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        
        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)
        
        # compute false negative for current class
        fn = false_negative(temp_true, temp_pred)
        
        
        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)
        
        # keep adding recall for all classes
        recall += temp_recall
        
    # calculate and return average recall over all classes
    recall /= num_classes
    
    return recall

def micro_recall(y_true, y_pred):


    # find the number of classes 
    num_classes = len(np.unique(y_true))
    
    # initialize tp and fp to 0
    tp = 0
    fn = 0
    
    # loop over all classes
    for class_ in np.unique(y_true):
        
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)
        
        # calculate false negative for current class
        # and update overall tp
        fn += false_negative(temp_true, temp_pred)
        
    # calculate and return overall recall
    recall = tp / (tp + fn)
    return recall
# --------------------------------------------------------------------------------------------------#


# F1 Score is the weighted average of Precision and Recall
# Computation of macro-averaged fi score

def macro_f1(y_true, y_pred):

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize f1 to 0
    f1 = 0
    
    # loop over all classes
    for class_ in list(np.unique(y_true)):
        
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        
        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)
        
        # compute false negative for current class
        fn = false_negative(temp_true, temp_pred)
        
        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)
        
        
        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)
        
        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)
        
        
        temp_f1 = 2 * temp_precision * temp_recall / (temp_precision + temp_recall + 1e-6)
        
        # keep adding f1 score for all classes
        f1 += temp_f1
        
    # calculate and return average f1 score over all classes
    f1 /= num_classes
    
    return f1


def micro_f1(y_true, y_pred):


    #micro-averaged precision score
    P = micro_precision(y_true, y_pred)

    #micro-averaged recall score
    R = micro_recall(y_true, y_pred)

    #micro averaged f1 score
    f1 = 2*P*R / (P + R)    

    return f1
# --------------------------------------------------------------------------------------------------#

def mcc_score(y_true, y_pred):# macro score
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize precision to 0
    mcc = 0
    
    # loop over all classes
    for class_ in list(np.unique(y_true)):
        
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        
        # compute tp,tn,fp,fn for current class
        tp = true_positive(temp_true, temp_pred)
        tn = true_negative(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)
        fn = false_negative(temp_true, temp_pred)

        # Calculate the denominator
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        # Check if the denominator is zero to avoid division by zero
        if denominator == 0:
            temp_mcc = np.nan  # or 0, depending on how you want to handle this case
        else:
            temp_mcc = ((tp * tn) - (fp * fn)) / denominator
        
        mcc += temp_mcc
        
    mcc /= num_classes
    
    return mcc
# --------------------------------------------------------------------------------------------------#



def confusion_matrix2(y_true, y_pred):
    plt.figure(figsize = (18,8))
    unique_labels = np.unique(y_true)
    sns.heatmap(metrics.confusion_matrix(y_true, y_pred), annot = True,fmt='d',
                annot_kws={"size": 18},xticklabels = unique_labels, yticklabels = unique_labels, cmap = 'summer')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(fontsize=16)  # Change the size of x-tick labels
    plt.yticks(fontsize=16)
    plt.show()




def metric_tb(y, y_pred):
    acc = accuracy(y, y_pred)
    pos_y = list(np.unique(y))[0]
    neg_y = list(np.unique(y))[1]
    precision = precision_score(y, y_pred, pos_label=pos_y)
    spec = recall_score(y, y_pred, pos_label=neg_y)
    recall = recall_score(y, y_pred, pos_label=pos_y)
    mcc = matthews_corrcoef(y, y_pred)
    
    result = {"Accuracy": acc,
              "Precision": precision,
              "Recall": recall,
              "Specification":spec,
              "Mcc": mcc}
    return result



def metric_sum(prediction_dictionary):

    valuelist = []

    # Iterate over the first level of the dictionary
    for key1, sub_dict in prediction_dictionary.items():
        # Iterate over the second level of the dictionary
        for key2, value in sub_dict.items():
            valuelist.append(((f"{key1}_{key2}"), value))

    first_values = [item[0] for item in valuelist]
    names = first_values

    second_values = [item[1] for item in valuelist]
    actual_list = []
    for i in range(len(second_values)):
        actual = second_values[i][0]
        actual_list.append(actual)

    predict_list = []
    for i in range(len(second_values)):
         predict = second_values[i][1]
         predict_list.append(predict)
    
    # List to store metric results
    results = []

    for i in range(len(second_values)):
        result = metric_tb(actual_list[i], predict_list[i])
        results.append(result)

    # Convert the results into a DataFrame, using dataset_names as the index
    df_results = pd.DataFrame(results, index=names)

    return df_results



# --------------------------------------------------------------------------------------------------#
# ROC curve for different folds from cross validation
def plot_multiclass_roc_cv(y_trues, y_probs, class_index=1, n_classes=3, class_label='Class 1',save_path=None):
   
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    plt.figure(figsize=(8, 6))

    for i in range(len(y_trues)):
        if len(np.unique(y_trues[i])) == 2:
            y_score = np.array(y_probs[i])[:, class_index]
            fpr, tpr, _ = roc_curve(y_trues[i], y_score)
        else:
            y_true_bin = label_binarize(y_trues[i], classes=range(n_classes))# binary of y_true in each fold
            y_score = np.array(y_probs[i])[:, class_index]
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate to mean FPR
        tpr_interp = np.interp(mean_fpr, fpr, tpr)#interpolating TPR values at a common set of FPR points across all folds.
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        plt.plot(fpr, tpr, alpha=0.4, label=f"Fold {i+1} AUC = {roc_auc:.3f}")

    # Mean and Std of TPRs
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f"Mean ROC (AUC = {mean_auc:.2f})", lw=2)

    plt.fill_between(mean_fpr,
                     np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1),
                     color='blue', alpha=0.2, label='± 1 std. dev.')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {class_label}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to: {save_path}")
        plt.close()
    else:
        plt.show()


# --------------------------------------------------------------------------------------------------#
# SHAP plot for multiple class (class is 0,1,2...)
def plot_SHAP_multiclass(shap_value,X_df,class_index=0,save_path=None):
    if shap_value.ndim ==2:
        shap.summary_plot(shap_value, X_df,show=False)
    else:
        shap.summary_plot(shap_value[:,:,class_index], X_df,show=False)
    #shap.summary_plot(shap_value[:,:,class_index], X_df,plot_type="bar",show=False)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP to: {save_path}")
        plt.close()
    else:
        plt.show()


# --------------------------------------------------------------------------------------------------#
def plot_confusion_matrices(y, y_pred,title,pos_y=1):
    acc = accuracy(y, y_pred)
    pos_y = list(np.unique(y))[0]
    precision = precision_score(y, y_pred,pos_label=pos_y)
    recall =recall_score(y, y_pred,pos_label=pos_y)
    #f1 = f1_score(y, y_pred,pos_label=pos_y)
    mcc = matthews_corrcoef(y, y_pred)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y,
        y_pred=y_pred,
        display_labels=sorted(set(y)),  # Use sorted set of labels to ensure consistency
        cmap='Blues'
    )
    
    # Add metrics below the confusion matrix
    metrics_text = (f"Accuracy: {acc:.4f}\n"
                    f"Precision: {precision:.4f}\n"
                    f"Recall: {recall:.4f}\n"
                    #f"F1: {f1:.4f}\n"
                    f"Matthew’s correlation coefficient : {mcc:.4f}\n")
    disp.ax_.text(0.5, -0.2, metrics_text, ha='center', va='top', fontsize=12, transform=disp.ax_.transAxes)

    plt.tight_layout()  # Adjust layout to make room for the metrics
    
    disp.ax_.set_title(title)
    plt.show()


def plot_confusion_matrices2(y, y_pred, title, pos_y=1):
    acc = accuracy(y, y_pred)
    pos_y = list(np.unique(y))[0]
    precision = precision_score(y, y_pred, pos_label=pos_y)
    recall = recall_score(y, y_pred, pos_label=pos_y)
    mcc = matthews_corrcoef(y, y_pred)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y,
        y_pred=y_pred,
        display_labels=sorted(set(y)),  # Use sorted set of labels to ensure consistency
        cmap='Blues'
    )
    
    # Add metrics below the confusion matrix
    metrics_text = (f"Accuracy: {acc:.4f}\n"
                    f"Precision: {precision:.4f}\n"
                    f"Recall: {recall:.4f}\n"
                    f"Matthew’s correlation coefficient: {mcc:.4f}\n")
    disp.ax_.text(0.5, -0.2, metrics_text, ha='center', va='top', fontsize=12, transform=disp.ax_.transAxes)

    plt.tight_layout()  # Adjust layout to make room for the metrics
    disp.ax_.set_title(title)
    # Removed plt.show()

def plotmacro_confusion_matrices(y, y_pred,title):
    acc = accuracy(y, y_pred)
    precision = macro_precision(y, y_pred)
    recall =macro_recall(y, y_pred)
    #f1 = metric.macro_f1(y, y_pred)
    mcc = mcc_score(y, y_pred)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y,
        y_pred=y_pred,
        display_labels=sorted(set(y)),  # Use sorted set of labels to ensure consistency
        cmap='Blues'
    )
    
    # Add metrics below the confusion matrix
    metrics_text = (f"Accuracy: {acc:.4f}\n"
                    f"Precision: {precision:.4f}\n"
                    f"Recall: {recall:.4f}\n"
                    #f"F1: {f1:.4f}\n"
                    f"Matthew’s correlation coefficient : {mcc:.4f}\n")
    disp.ax_.text(0.5, -0.2, metrics_text, ha='center', va='top', fontsize=12, transform=disp.ax_.transAxes)

    plt.tight_layout()  # Adjust layout to make room for the metrics
    
    disp.ax_.set_title(title)
    plt.show()


def plotmicro_confusion_matrices(y, y_pred,title):
    acc = accuracy(y, y_pred)
    precision = micro_precision(y, y_pred)
    recall =micro_recall(y, y_pred)
    #f1 = metric.micro_f1(y, y_pred)
    mcc = mcc_score(y, y_pred)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y,
        y_pred=y_pred,
        display_labels=sorted(set(y)),  # Use sorted set of labels to ensure consistency
        cmap='Blues'
    )
    
    # Add metrics below the confusion matrix
    metrics_text = (f"Accuracy: {acc:.4f}\n"
                    f"Precision: {precision:.4f}\n"
                    f"Recall: {recall:.4f}\n"
                    #f"F1: {f1:.4f}\n"
                    f"Matthew’s correlation coefficient : {mcc:.4f}\n")
    disp.ax_.text(0.5, -0.2, metrics_text, ha='center', va='top', fontsize=12, transform=disp.ax_.transAxes)

    plt.tight_layout()  # Adjust layout to make room for the metrics
    
    disp.ax_.set_title(title)
    plt.show()





def plotcompare(data_dict,metric='Accuracy'):
    # Marker styles for each metric
    markers = {'SelectMicro': 'o', 'Lasso': 's', 'SelectMicrro_Lasso': '^'}
    labels = {'SelectMicro': 'SelectMicro', 'Lasso': 'Lasso', 'SelectMicrro_Lasso': 'SelectMicrro_Lasso'}
    
    # Plot each element in the dataset
    plt.figure(figsize=(8, 6))
    if metric=='AUC':
        j=2
    elif metric == 'F':
        j=3
    else:
        j=1

    # Create a list to collect the line handles for the legend
    line_handles = []

    
    # Iterate over the outer dictionary (keys are 'Smocky_Burn', 'Smocky_urban', etc.)
    for region, metrics in data_dict.items():
        # Collect all points for the region
        x_vals = []
        y_vals = []
        
        # Collect points for each metric
        for method, values in metrics.items():
            x_vals.append(values[0])  # First element (x-coordinate)
            y_vals.append(values[j])  # Second element (y-coordinate)
            
            # Plot the point for each metric with different markers based on the metric name
            plt.scatter(values[0], values[j], marker=markers[method], color='black')
    
       # Plot the line connecting the points for the region and store the line handle for the legend
        line_handle, = plt.plot(x_vals, y_vals, label=f"{region}")
        line_handles.append(line_handle)
    
    # Create custom legend handles for the markers with black color
    legend_handles = [Line2D([0], [0], marker=markers[method], color='black', markerfacecolor='black', markersize=8, label=labels[method]) for method in markers]
    
    # Set labels and title
    plt.xlabel('Number of Features Selected')
    plt.ylabel(metric)
    plt.title(f'{metric} of each Feature Selection Method in each dataset')
    
    # Display the legend with both line handles and marker handles
    plt.legend(handles=line_handles + legend_handles)
    
    
    # Show the plot
    #plt.grid(True)
    plt.show()




def Neg_GINI_origin(X,y):# for a single variable y, calculate the NG for all OTU
    if not (len(X)==len(y)):
        raise ValueError('ERROR! Length of OTU and label have difference length')
    X = X.copy()
    X = np.where(X != 0, 1, 0)
    X_transpose=np.transpose(X)
    ng_List=[]
    for OTU_across_sample in X_transpose:
        df_ng = pd.DataFrame({'OTU': OTU_across_sample,'response': y})
        total_ones = df_ng['OTU'].sum()
        if total_ones==0:
            ng_List.append(0)
        else: 
            ones_in_each_category = df_ng[df_ng['OTU'] == 1].groupby('response').size()
            portion_in_each_category = ones_in_each_category/ones_in_each_category.sum()
            portion_in_each_category = portion_in_each_category.tolist()
            NG = sum([x**2 for x in portion_in_each_category])
            ng_List.append(NG)
    return np.array(ng_List)

    



def Neg_GINI(X,Y):#X is the relative abundance matrix(np.array), each row is a sample; y is the classification results
    # if there is only one response, then y is a 1D array
    # if there is multiple response variable, then y is a 2D array, each column is one variable
    Y = np.asarray(Y)
    if Y.ndim == 1:
        result = Neg_GINI_origin(X,Y)# a 1D array showing the H statistics for all the features
        return result
    elif Y.ndim ==2:# if
        Y_transpose=np.transpose(Y)
        NG_list = []
        for yi in Y_transpose:
            NG_one_class = Neg_GINI_origin(X,yi)
            NG_list.append(NG_one_class)
        NG_combine = np.vstack(NG_list)
        return NG_combine
    else:
        return "Error: The input Y must be a 1D or 2D array."




def  sharp_value(X,y,classifier_name):# need to be update
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




def fisher_discriminant_ratio(features, labels):
    """
    Calculate the Fisher's discriminant ratio (F1) for an entire dataset.

    Parameters:
    - features: A 2D numpy array of feature values (samples x features).
    - labels: A 1D numpy array of class labels corresponding to the samples.

    Returns:
    - F1: Fisher's discriminant ratio for the dataset.
    """
    # Ensure features and labels are numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Get unique classes
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("This implementation only supports two classes.")

    # Calculate class-wise means and overall mean
    overall_mean = np.mean(features, axis=0)
    class_means = [np.mean(features[labels == c], axis=0) for c in classes]

    # Calculate between-class scatter
    between_class_scatter = sum(
        len(features[labels == c]) * np.outer((class_mean - overall_mean), (class_mean - overall_mean))
        for c, class_mean in zip(classes, class_means)
    )

    # Calculate within-class scatter
    within_class_scatter = sum(
        sum(np.outer((sample - class_mean), (sample - class_mean)) for sample in features[labels == c])
        for c, class_mean in zip(classes, class_means)
    )

    # Compute Fisher's discriminant ratio
    F1 = np.trace(between_class_scatter) / np.trace(within_class_scatter)

    return F1





def plot_heatmap(X_df,y_df, tax_prefix='g__', column_cluster = False):
    # both of X_df y_df needs index and colnames
    # add to check the index of both inputs are the same
    if not X_df.index.equals(y_df.index):
        raise ValueError("The indices of X_df and y_df do not match. Please ensure they contain the same samples in the same order.")

    Category = y_df.columns[0]
    sorted_cols = y_df[Category].sort_values().index
    data_matrix_sorted = X_df.loc[sorted_cols]
    group_sorted = y_df.loc[sorted_cols]

    unique_categories = group_sorted[Category].unique()
    palette = sns.color_palette("Set2", len(unique_categories))
    category_colors = dict(zip(unique_categories, palette))
    col_colors = group_sorted[Category].map(category_colors)

    # 🔹 Strip prefix from column names (features)
    def shorten_fn(name):
        if tax_prefix in name:
            return name.split(tax_prefix)[-1]
        return name  # return original if prefix not found

    data_matrix_sorted.columns = [shorten_fn(col) for col in data_matrix_sorted.columns]
    
    # Plot heatmap with annotation (transpose to match R behavior)
    sns.set(style="white")
    cg = sns.clustermap(
        data_matrix_sorted.T,  # Transpose like in R: genes as rows, samples as columns
        col_cluster=column_cluster,     
        row_cluster=True,      
        col_colors=col_colors,
        cmap="vlag",           # or use "coolwarm", "RdBu_r", etc.
        xticklabels=False,
        yticklabels=True
    )
    # Make tick labels smaller
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), fontsize=6)
    #cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xticklabels(), fontsize=6)

    for label in unique_categories:
        cg.ax_col_dendrogram.bar(0, 0, color=category_colors[label],
                                label=label, linewidth=0)
    cg.ax_col_dendrogram.legend(loc="center left", ncol=3, title=Category)
    
    
    plt.show()



