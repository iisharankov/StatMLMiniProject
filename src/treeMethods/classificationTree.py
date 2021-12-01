import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline

import src.constants as const

def clean_x_and_y(data):
    # Convert 'Lead' attribute to string, then to boolean and set as y_train
    x = data.loc[:, data.columns != "Lead"]  # Extract lead from x data
    y = data['Lead'].map({'Female': -1, 'Male': 1})  # Convert leads to bools
    return x, y


def classification_tree():
    clf = sklearn.tree.DecisionTreeClassifier(max_depth=7, random_state=0)
    clf.fit(X_train, Y_train)
    return clf


def random_forest():
    rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=4)
    rfc.fit(X_train, Y_train)
    return rfc

def bagging():
    bg_pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))

    # Instantiate the bagging classifier
    bgclassifier = sklearn.ensemble.BaggingClassifier(
        base_estimator=bg_pipeline, n_estimators=100,
        max_features=10, max_samples=100,
        random_state=1, n_jobs=5
    )

    # Fit the bagging classifier
    bgclassifier.fit(X_train, Y_train)
    return bgclassifier


def cross_validation(model, x, y, n):
    cv = sklearn.model_selection.KFold(n_splits=n, random_state=1, shuffle=True)
    scores = sklearn.model_selection.cross_val_score(
        model, x, y, cv=cv, n_jobs=1)

    print(len(scores), scores)
    print(f"cross_validation: {np.mean(np.absolute(scores))}")


if __name__ == "__main__":
    if not os.path.exists(const.TEST_DATASET) or not os.path.exists(const.TRAIN_DATASET):
        print("The file given does not exist, please check the path")
        raise FileNotFoundError
    else:
        data = pd.read_csv(const.TRAIN_DATASET)  # Import dataset
        X_train, Y_train = clean_x_and_y(data)

    gaps = 1 # 104
    n = int(np.ceil(len(X_train) / gaps))

    print("Starting Simple Classification Trees method")
    clf = classification_tree()
    cross_validation(clf, X_train, Y_train, n)


    print("-\n Starting Random Forrest method")
    rfc = random_forest()
    cross_validation(rfc, X_train, Y_train, n)

    print("-\n Starting Bagging method")
    bag = bagging()
    cross_validation(bag, X_train, Y_train, n)
