import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_classifiers(datasets, classifiers):
    results = {}

    for dataset_name, (X, y) in datasets.items():
        results[dataset_name] = {}
        for classifier_name, clf in classifiers.items():
            print(f"Evaluating {classifier_name} on {dataset_name}...")

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            accuracies = []
            roc_aucs = []

            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_prob)

                accuracies.append(accuracy)
                roc_aucs.append(roc_auc)

            mean_accuracy = np.mean(accuracies)
            mean_roc_auc = np.mean(roc_aucs)

            results[dataset_name][f"{classifier_name}_Accuracy"] = mean_accuracy
            results[dataset_name][f"{classifier_name}_AUC"] = mean_roc_auc

    results_df = pd.DataFrame(results).T
    return results_df

# Example datasets
datasets = {
    "Dataset1": (
        np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]),
        np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    ),
    "Dataset2": (
        np.array([[2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]]),
        np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    )
}

# Classifiers to evaluate
classifiers = {
    "RF": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

# Run the evaluation
results_df = evaluate_classifiers(datasets, classifiers)

# Print the results
print(results_df)
