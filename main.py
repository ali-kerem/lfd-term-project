import helpers
import random as r
from numpy import mean
from sklearn.linear_model import LinearRegression, Ridge, LassoLars
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.neighbors import KNeighborsRegressor

if __name__ == "__main__":
    train_features = helpers.prepareFeatures("train_features.csv", normalize=True, ohe_children=False, ohe_region=True)
    train_targets = helpers.prepareTargets("train_targets.csv")

    test_features = helpers.prepareFeatures("test_features.csv", normalize=True, ohe_children=False, ohe_region=True)

    r.seed(1)

    models = []
    avg_errors = []

    # Add models
    models.append(LinearRegression())
    models.append(Ridge())
    models.append(LassoLars())
    models.append(KNeighborsRegressor())
    models.append(RandomForestRegressor())
    models.append(AdaBoostRegressor())
    models.append(XGBRegressor())
    models.append(GradientBoostingRegressor())
    models.append(XGBRFRegressor())


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

    helpers.createSubmission(model=final_model, test_features=test_features, submissionFile="newsub.csv")
