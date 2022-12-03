from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from copy import deepcopy
import random as r
import helpers

DIVIDE_TARGET_BY = 1000

if __name__ == "__main__":
    train_features = helpers.prepareFeatures("train_features.csv", normalize=True)
    train_targets = helpers.prepareTargets("train_targets.csv", divisor=DIVIDE_TARGET_BY)

    test_features = helpers.prepareFeatures("test_features.csv", normalize=True)
    test_targets = helpers.prepareTargets("test_targets.csv", divisor=DIVIDE_TARGET_BY)

    r.seed(1)
    kf = KFold(n_splits=5, shuffle=True)
    best_regressor = None
    best_regressor_error = 9999999999
    regressor = LinearRegression() # For loop
    i = 1

    for train, test in kf.split(train_features):
        x_train, x_test = train_features[train], train_features[test]

        y_train, y_test = train_targets[train], train_targets[test]

        regressor.fit(x_train, y_train)

        print("{}. Fold :".format(i))
        print("Training :")
        helpers.printPerformance(regressor, features=x_train, targets=y_train, divisor=DIVIDE_TARGET_BY)
        print()
        print("Test :")
        test_error = helpers.printPerformance(regressor, features=x_test, targets=y_test, divisor=DIVIDE_TARGET_BY)
        print("------------------------------------------------")

        if test_error < best_regressor_error:
            best_regressor = deepcopy(regressor) # If we don't deepcopy, best_regressor gets updated everytime regressor is updated
            best_regressor_error = test_error

        i += 1

    print("Performance on test data :")
    helpers.printPerformance(best_regressor, features=test_features, targets=test_targets, divisor=DIVIDE_TARGET_BY)
    helpers.createSubmission(best_regressor, test_features=test_features, submissionFile="submission.csv", divisor=DIVIDE_TARGET_BY)
