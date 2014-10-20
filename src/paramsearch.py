from sklearn import datasets
from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, make_scorer, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import scale, normalize
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Loading the Digits dataset
dataset = pd.read_csv("AllDATA.csv", true_values=["yes"], false_values=["no"])

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
shape_parameters = ["Roundedness(GEOMETRICAL)","Symmetry(GEOMETRICAL)","DirDepStdDev(GEOMETRICAL)","Roundness(GEOMETRICAL)","Compactness(GEOMETRICAL)"]
X = dataset[dataset.columns[:-1]]
y = dataset.CH

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    scale(X, axis=0), y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
grid = {'SVM': (
            SVC(), 
            [{'kernel': ['rbf'], 'gamma': [0.0, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}, {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'gamma': [0.0, 1e-3, 1e-4], 'degree': [2, 3, 4, 5]}]
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

scorers = [("TSS", tss_scorer), ("Accuracy", accuracy_scorer), ("TN rate", tn_rate_scorer)]

results = {}

i = 0
scores_array = []
for name, score in scorers:
    print "### Tuning hyper parameters for", name
    for alg_name, (classifier, tuned_parameters) in grid.iteritems():
        print "# Algorithm:", alg_name

        clf = GridSearchCV(classifier, tuned_parameters, cv=StratifiedKFold(y_train, n_folds=10), scoring=score, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        #print("Results on development set:")
        #y_true, y_pred = y_train, clf.predict(X_train)
        #print(classification_report(y_true, y_pred))
        #print "TSS:", tss(y_true, y_pred), "TN rate:", tn_rate(y_true, y_pred)
        #print "Confusion matrix:", confusion_matrix(y_true, y_pred)


        #print("Best parameters set found on development set:")
        #print ""
        #print(clf.best_estimator_)
        #print ""

        if alg_name == 'decision tree':
            with open("tree_%s.dot" % name, "w+") as f:
                export_graphviz(clf.best_estimator_, out_file=f, feature_names=X.columns.values.tolist())
            #print ""

        (params, mean_score, scores) = sorted(clf.grid_scores_, key=lambda x: x[1])[-1]
        #print("Development set: %0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

        #print("Grid scores on development set:")
        #print ""
        #for params, mean_score, scores in clf.grid_scores_:
        #    print("%0.3f (+/-%0.03f) for %r"
        #          % (mean_score, scores.std() / 2, params))
        #print ""

        #print("Detailed classification report:")
        #y_true, y_pred = y_test, clf.predict(X_test)
        #print(classification_report(y_true, y_pred))
        #print "TSS:", tss(y_true, y_pred), "TN rate:", tn_rate(y_true, y_pred)
        #print "Confusion matrix:", confusion_matrix(y_true, y_pred)
        print "Best parameters:", params
        for (scorer_name, scorer2) in scorers:
            scores = cross_val_score(clf.best_estimator_, X_test, y_test, scoring=scorer2, cv=10)
            if score == tss_scorer:
                scores_array.append(scores)
            print("Evaluation using %s: %0.3f (+/-%0.03f): %s" % (scorer_name, scores.mean(), scores.std() / 2, array_to_string(scores)))
        print ""
    print ""
plt.boxplot(scores_array)
plt.ylabel("True Skill Statistic (TSS)")
plt.xlabel("Classifier")
plt.xticks([1, 2, 3], map(lambda x: x[0], grid.iteritems()))
plt.title("Performance")
plt.savefig("figure.pdf")
plt.show()

