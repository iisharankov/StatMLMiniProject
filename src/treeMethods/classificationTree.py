import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline

import src.constants as const

def clean_x_and_y(data):
    # Convert 'Lead' attribute to string, then to boolean and set as y_train
    x = data.loc[:, data.columns != "Lead"]  # Extract lead from x data
    # x = data[["Number words female", "Number of female actors", "Number of male actors"]]
    y = data['Lead'].map({'Female': 0, 'Male': 1})  # Convert leads to bools
    return x, y


def classification_tree():
    clf = sklearn.tree.DecisionTreeClassifier(max_depth=7, random_state=R)
    clf.fit(X_train, Y_train)
    # importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 3)})
    # importances = importances.sort_values('importance', ascending=False)
    # print(f"importances are\n {importances}")
    return clf


def random_forest():
    rfc = RandomForestClassifier(n_estimators=n, criterion='entropy', random_state=R)
    rfc.fit(X_train, Y_train)
    return rfc


def bagging():
    bg_pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=R))
    bgclassifier = BaggingClassifier(base_estimator=bg_pipeline, n_estimators=n,
                                     max_features=X_train.shape[1], max_samples=100,
                                     random_state=R, n_jobs=5)

    bgclassifier.fit(X_train, Y_train)
    return bgclassifier


def adaboosting():
    boost = AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier(max_depth=3),
                               n_estimators=n, random_state=R)
    boost.fit(X_train, Y_train)
    return boost


def grad_boosting():
    gb_clf = GradientBoostingClassifier(n_estimators=n, learning_rate=0.44,
                                        max_features=5, max_depth=3, random_state=R)
    gb_clf.fit(X_train, Y_train)
    return gb_clf


def cross_validation(model, x, y, n):
    cv = KFold(n_splits=n, random_state=R, shuffle=True)
    scores = cross_val_score(model, x, y, cv=cv, n_jobs=1)

    accuracy = -1 if isinstance(X_val, int) else round(model.score(X_val, Y_val), 3)
    validation = round(1 - np.mean(np.absolute(scores)), 3)

    # print(len(scores), 1 - np.array(scores))
    print(f"Accu"
          f"racy score={accuracy} - cross_validation= {validation}")


if __name__ == "__main__":
    R = np.random.seed(1)  # Random state
    gaps = 25



    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn


    if not os.path.exists(const.TEST_DATASET) or not os.path.exists(const.TRAIN_DATASET):
        print("The file given does not exist, please check the path")
        raise FileNotFoundError
    else:
        data = pd.read_csv(const.TRAIN_DATASET)  # Import dataset
        X_train, Y_train = clean_x_and_y(data)
        X_val = Y_val = 0

    # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.30, random_state=R)

    n = int(np.ceil(len(X_train) / gaps))
    #
    # print("- - - - Starting Simple Classification Trees method - - - - ")
    # cross_validation(classification_tree(), X_train, Y_train, n)
    #
    # print("- - - - Starting Random Forrest method - - - - ")
    # cross_validation(random_forest(), X_train, Y_train, n)
    #
    # print("- - - - Starting Bagging method - - - - ")
    # cross_validation(bagging(), X_train, Y_train, n)

    print("- - - - Starting Boosting method - - - - ")
    cross_validation(adaboosting(), X_train, Y_train, n)

    print("- - - - Starting Gradient Boosting method - - - - ")
    cross_validation(grad_boosting(), X_train, Y_train, n)
