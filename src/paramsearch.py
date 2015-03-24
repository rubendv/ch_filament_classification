from sklearn import datasets
from sklearn.cross_validation import train_test_split, StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, make_scorer, confusion_matrix, recall_score, precision_score, roc_curve
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import scale, normalize
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import json
import os
import sys
import argparse

def load_data(path):
    dataset = pd.read_csv(path, true_values=["true"], false_values=["false"], skiprows=[1])
    dataset.drop(dataset.columns[0], axis=1, inplace=True)

    X = dataset[dataset.columns[:-1]]
    y = dataset[dataset.columns[-1]]
    return scale(X, axis=0), y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-hmi", help="include HMI based parameters", action="store_true")
    args = parser.parse_args()

    if args.with_hmi:
        path = "/home/local/data/DATASET_AIA_HMI.csv"
    else:
        path = "/home/local/data/DATASET_AIA.csv"

    X, y = load_data(path)

    C = 10.0**np.arange(-4, 5)
    gamma = 10.0**np.arange(-4, 5)

    grid = {'SVM': (
                SVC(), 
                [{'kernel': ['rbf'], 'gamma': gamma, 'C': C, 'class_weight': [None, 'auto']}, {'kernel': ['sigmoid'], 'gamma': gamma, 'C': C, 'class_weight': [None, 'auto']}, {'kernel': ['linear'], 'C': C, 'class_weight': [None, 'auto']}, {'kernel': ['poly'], 'C': C, 'gamma': gamma, 'degree': np.arange(2, 6), 'class_weight': [None, 'auto']}]
            ),
            'Linear SVM': (
                LinearSVC(),
                [{'C': C, 'dual': [True, False], 'class_weight': [None, 'auto']}]
            ),
            'Decision Tree': (
                DecisionTreeClassifier(), 
                [{'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_features': [0.1, 0.5, 1.0, "auto", "sqrt", "log2", None], 'min_samples_split': [2, 4, 8, 16], 'max_depth': [2, 4]}]
            ),
            'Random Forest': (
                RandomForestClassifier(),
                [{'criterion': ['gini', 'entropy'], 'max_features': [0.1, 0.5, 1.0, "auto", "sqrt", "log2", None], 'n_estimators': [5, 10, 15, 20, 50, 100], 'min_samples_split': [2, 4, 8, 16]}]
            )
    }

    #for name in 'SVM', 'Linear SVM':
    #    for paramset in grid[name][1]:
    #        paramset['probability'] = [True]

    def tss(y_true, y_pred):
        confmat = confusion_matrix(y_true, y_pred)
        tn = float(confmat[0, 0])
        tp = float(confmat[1, 1])
        fp = float(confmat[0, 1])
        fn = float(confmat[1, 0])
        return tp / (tp + fn) - fp / (fp + tn)

    def tn_rate(y_true, y_pred):
        confmat = confusion_matrix(y_true, y_pred)
        tn = float(confmat[0, 0])
        tp = float(confmat[1, 1])
        fp = float(confmat[0, 1])
        fn = float(confmat[1, 0])
        return tn / (tn + fp)
    def tp_rate(y_true, y_pred):
        confmat = confusion_matrix(y_true, y_pred)
        tn = float(confmat[0, 0])
        tp = float(confmat[1, 1])
        fp = float(confmat[0, 1])
        fn = float(confmat[1, 0])
        return tp / (fn + tp)
    def fp_rate(y_true, y_pred):
        confmat = confusion_matrix(y_true, y_pred)
        tn = float(confmat[0, 0])
        tp = float(confmat[1, 1])
        fp = float(confmat[0, 1])
        fn = float(confmat[1, 0])
        return fp / (fp + tn)

    def accuracy(y_true, y_pred):
        confmat = confusion_matrix(y_true, y_pred)
        tn = float(confmat[0, 0])
        tp = float(confmat[1, 1])
        fp = float(confmat[0, 1])
        fn = float(confmat[1, 0])
        return (tn + tp) / np.sum(confmat)

    def array_to_string(arr):
        return "[" + ", ".join(map(lambda x: "%.5f" % x, arr)) + "]"

    tss_scorer = make_scorer(tss)
    tn_rate_scorer = make_scorer(tn_rate)
    accuracy_scorer = make_scorer(accuracy)

    scorers = [("TSS", tss), ("Precision", precision_score), ("Recall",recall_score)]

    cv_test = StratifiedShuffleSplit(y, n_iter=100, test_size=0.25)

    results = {}
    for scorer in ("tss", "fpr", "tpr", 'roc'):
        results[scorer] = dict([(alg_name, []) for alg_name in grid.keys()]) 

    for i, (train_index, test_index) in enumerate(cv_test):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        cv_parameter = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=0)
        for j, (alg_name, (classifier, tuned_parameters)) in enumerate(grid.items()):
            print("Testing {}".format(alg_name))
            clf = GridSearchCV(classifier, tuned_parameters, cv=cv_parameter, scoring=tss_scorer, n_jobs=-1)
            clf.fit(X_train, y_train)
            if 'SVM' in alg_name:
                y_score = clf.decision_function(X_test)
                roc_fpr, roc_tpr, _ = roc_curve(y_test, y_score)
                results['roc'][alg_name].append({'fpr': list(roc_fpr), 'tpr': list(roc_tpr)})
            y_pred = clf.predict(X_test)
            results["fpr"][alg_name].append(fp_rate(y_test, y_pred))
            results["tpr"][alg_name].append(tp_rate(y_test, y_pred))
            results["tss"][alg_name].append(tss(y_test, y_pred))
        print("Iteration {}/{} done".format(i+1, cv_test.n_iter))
    with open("/tmp/results.json", "w+") as f:
        json.dump(results, f)
    if args.with_hmi:
        suffix = "with"
    else:
        suffix = "without"
    print("Return value: {}".format(os.system("sudo cp /tmp/results.json /home/local/results/results_{}_hmi.json".format(suffix))))
