from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from copy import deepcopy
import pandas as pd
import random as r
import helpers

if __name__ == "__main__":
    r.seed(1)

    train_features = pd.read_csv("train_features.csv")
    train_targets = pd.read_csv("train_targets.csv")
    test_features = pd.read_csv("test_features.csv")

    train_features = helpers.labelEncode(train_features)
    test_features = helpers.labelEncode(test_features)
    train_targets = train_targets.values

    kf = KFold(n_splits=5, shuffle=True)
    best_regressor = None
    best_regressor_error = 999999999
    regressor = LinearRegression() # For loop
    i = 1

    for train, test in kf.split(train_features):
        x_train = train_features[train]
        x_test = train_features[test]

        y_train = train_targets[train].flatten()
        y_test = train_targets[test].flatten()

        regressor.fit(x_train, y_train)

        print("{}. Fold : ".format(i))
        print("Train error : {}".format(mse(y_train, regressor.predict(x_train))))
        print("Train score : {}".format(regressor.score(x_train, y_train)))
        print()

        regressor_error = mse(y_test, regressor.predict(x_test))
        print("Test  error : {}".format(regressor_error))
        print("Test  score : {}".format(regressor.score(x_test, y_test)))
        print("------------------------------------------------")

        if regressor_error < best_regressor_error:
            best_regressor = deepcopy(regressor) # If we don't deepcopy, best_regressor gets updated everytime regressor is updated
            best_regressor_error = regressor_error

        i += 1

    print("Score on test data : {}".format(helpers.testScore(best_regressor)))
    helpers.createSubmission(best_regressor)
