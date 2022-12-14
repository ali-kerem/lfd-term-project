import helpers
import random as r
from numpy import mean
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor, XGBRFRegressor

if __name__ == "__main__":
    train_features = helpers.prepareFeatures("train_features.csv", normalize=True, ohe_children=False, ohe_region=True)
    train_targets = helpers.prepareTargets("train_targets.csv")

    test_features = helpers.prepareFeatures("test_features.csv", normalize=True, ohe_children=False, ohe_region=True)
    test_targets = helpers.prepareTargets("test_targets.csv")

    r.seed(1)

    models = []
    avg_errors = []

    # Add models
    models.append(LinearRegression())
    models.append(Ridge())
    models.append(XGBRFRegressor())
    models.append(AdaBoostRegressor())
    models.append(XGBRegressor(n_estimators=100,max_depth=2,min_child_weight=1,subsample=1,colsample_bytree=1, learning_rate=0.1))
    models.append(GradientBoostingRegressor(max_depth=2))
    models.append(RandomForestRegressor(max_depth=3, n_estimators=500))


    # Apply 5-fold CV on models
    for model in models:
        errors = helpers.CV(features=train_features, targets=train_targets, model=model, n_splits=5)
        avg_errors.append(mean(errors))

    # Sort by average error in ascending order
    models = [model for _, model in sorted(zip(avg_errors, models))]
    avg_errors.sort()

    # Print errors
    print("   {0:32} Average CV Error".format('Model'))
    for i in range(len(models)):
        print("{0}. {1:30} : {2}".format(i + 1, models[i].__class__.__name__, avg_errors[i]))

    # Pick model with lowest average error
    final_model = models[0]
    final_model.fit(train_features, train_targets)

    print("\nPerformance on test data :")
    helpers.printPerformance(model=final_model, features=test_features, targets=test_targets)

    helpers.createSubmission(model=final_model, test_features=test_features, submissionFile="newsub.csv")
