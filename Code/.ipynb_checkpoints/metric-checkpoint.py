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
import matplotlib.pyplot as plt
import seaborn as sns

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


def mcc_score(y_true, y_pred):
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
    


