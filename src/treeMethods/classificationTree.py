import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

import src.constants as const

def clean_x_and_y(data):
    # Convert 'Lead' attribute to string, then to boolean and set as y_train
    # data["Lead"] = data["Lead"].astype("string")  # dtype becomes 'string' from 'object'

    x = data.loc[:, data.columns != "Lead"]  # Extract lead from x data
    y = data['Lead'].map({'Female': 1, 'Male': 0})  # Convert leads to bools

    return x, y

def split_data(data, ratio=0.7):
    split_value = int(np.ceil(len(data) * ratio))

    x_train, y_train = clean_x_and_y(data[0:split_value])
    x_test, y_test = clean_x_and_y(data[split_value:])

    return x_train, y_train, x_test, y_test


def multiple_depths():
    # List of values to try for max_depth:
    max_depth_range = list(range(1, 25))  # List to store the accuracy for each value of max_depth:
    accuracy = []
    for depth in max_depth_range:
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(X_train, Y_train)
        score = clf.score(X_test, Y_test)
        accuracy.append(score)

    print("accuracy is\n", accuracy)

    x = range(len(accuracy))
    plt.plot(x, accuracy)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()


def single_depth(print_importances=False):
    # print(Y_test.Lead[Y_test.Lead==0].count()/ len(Y_test))
    clf = sklearn.tree.DecisionTreeClassifier(max_depth=7, random_state=0)
    clf.fit(X_train, Y_train)
    cross_validation(clf, X_train, Y_train, n)


    # predictions = clf.predict(X_test)
    # print(predictions)  # Predict with X_test

    # The score method returns the accuracy of the model
    # score = clf.score(X_test, Y_test)
    # print(f"score is {score}")

    if print_importances:
        importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 3)})
        importances = importances.sort_values('importance', ascending=False)
        print(f"importances are\n {importances}")

    return clf

###########3
def random_forest():

    # Building  Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=4)
    rfc.fit(X_train, Y_train)
    cross_validation(rfc, X_train, Y_train, n)

    # y_pred = rfc.predict(X_test)

    # for i, (a, b) in enumerate(zip(y_pred, Y_test)):
    #     print(i, b, a)

    # score = rfc.score(X_test, Y_test)
    # print(f"score is {score}")

    # Evaluating on Training / Testing sets
    # print('Training Set Evaluation F1-Score=>', f1_score(Y_train, rfc.predict(X_train)))
    # print('Testing Set Evaluation F1-Score=>', f1_score(Y_test, rfc.predict(X_test)))

    # from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    # print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
    return rfc

def compare(rfc, clf):

    feature_importance = pd.DataFrame({
        'rfc': rfc.feature_importances_,
        'clf': clf.feature_importances_
    }, index=data.drop(columns=['Lead']).columns)
    feature_importance.sort_values(by='rfc', ascending=True, inplace=True)

    index = np.arange(len(feature_importance))
    fig, ax = plt.subplots(figsize=(18, 8))
    rfc_feature = ax.barh(index, feature_importance['rfc'], 0.4, color='purple', label='Random Forest')
    dt_feature = ax.barh(index + 0.4, feature_importance['clf'], 0.4, color='lightgreen', label='Decision Tree')
    ax.set(yticks=index + 0.4, yticklabels=feature_importance.index)

    ax.legend()
    plt.show()


def bagging():
    # Pipeline Estimator
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import BaggingClassifier

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))
    # Instantiate the bagging classifier
    bgclassifier = BaggingClassifier(base_estimator=pipeline, n_estimators=100,
                                     max_features=10,
                                     max_samples=100,
                                     random_state=1, n_jobs=5)
    # Fit the bagging classifier
    bgclassifier.fit(X_train, Y_train)
    cross_validation(bgclassifier, X_train, Y_train)

    # Model scores on test and training data
    print('Model training Score: %.3f' % bgclassifier.score(X_train, Y_train),
          'Model test Score: %.3f, ' % bgclassifier.score(X_test, Y_test))

def cross_validation(model, X, y, n):
    cv = KFold(n_splits=n, random_state=1, shuffle=True)

    # use k-fold CV to evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error',
                             cv=cv, n_jobs=-1)

    # view mean absolute error
    print(f"cross_validation: {np.mean(np.absolute(scores))}")

if __name__ == "__main__":
    if not os.path.exists(const.TEST_DATASET) or not os.path.exists(const.TRAIN_DATASET):
        print("The file given does not exist, please check the path")
        raise FileNotFoundError
    else:
        data = pd.read_csv(const.TRAIN_DATASET)  # Import dataset
        X_train, Y_train, X_test, Y_test = split_data(data, 1)

    n = len(X_train) - 1
    print(len(X_train), n)
    clf = single_depth()


    # print("-")
    # # multiple_depths()
    # print("-")
    #
    rfc = random_forest()
    # print("-")
    #
    # # compare(rfc, clf)
    # print("-")
    bagging()