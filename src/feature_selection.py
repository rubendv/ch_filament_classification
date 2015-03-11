from sklearn import datasets
from sklearn.cross_validation import train_test_split, StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, make_scorer, confusion_matrix, recall_score, precision_score
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
from sklearn.feature_selection import RFECV

def load_data(path):
    dataset = pd.read_csv(path, true_values=["true"], false_values=["false"], skiprows=[1])
    dataset.drop(dataset.columns[0], axis=1, inplace=True)

    X = dataset[dataset.columns[:-1]]
    y = dataset[dataset.columns[-1]]
    return scale(X, axis=0), y, dataset.columns[:-1].values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-hmi", help="include HMI based parameters", action="store_true")
    args = parser.parse_args()

    if args.with_hmi:
        path = "/home/local/data/DATASET_AIA_HMI.csv"
    else:
        path = "/home/local/data/DATASET_AIA.csv"

    X, y, columns = load_data(path)

    C = 10.0**np.arange(-4, 5)
    gamma = 10.0**np.arange(-4, 5)

    grid = [{'C': C, 'dual': [True, False], 'class_weight': [None, 'auto']}]

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

    clf = GridSearchCV(LinearSVC(), grid, cv=StratifiedKFold(y, n_folds=5, shuffle=True), scoring=tss_scorer, n_jobs=-1).fit(X, y).best_estimator_
    selector = RFECV(clf, step=1, cv=StratifiedKFold(y, n_folds=5, shuffle=True), scoring=tss_scorer)
    selector.fit(X, y)

    print(array_to_string(selector.support_))
    print(array_to_string(selector.ranking_))
    for (name, ranking) in sorted(zip(columns, selector.ranking_), key=lambda x: x[1]):
        print("{}. {}".format(ranking, name))
