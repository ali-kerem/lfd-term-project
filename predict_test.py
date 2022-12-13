import helpers
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor

DIVIDE_TARGET_BY = 1000

if __name__ == "__main__":
    train_features = helpers.prepareFeatures("train_features.csv", normalize=True)
    train_targets = helpers.prepareTargets("train_targets.csv", divisor=DIVIDE_TARGET_BY)

    test_features = helpers.prepareFeatures("test_features.csv", normalize=True)
    test_targets = helpers.prepareTargets("test_targets.csv", divisor=DIVIDE_TARGET_BY)

    models = []
    errors = []

    models.append(LinearRegression())
    models.append(Ridge())
    models.append(XGBRFRegressor())
    models.append(AdaBoostRegressor())
    models.append(XGBRegressor(n_estimators=100,max_depth=2,min_child_weight=1,subsample=1,colsample_bytree=1, learning_rate=0.1))
    models.append(GradientBoostingRegressor(max_depth=2))
    models.append(RandomForestRegressor(max_depth = 3, n_estimators=500))

    for model in models:
        model.fit(train_features, train_targets)
        error = mse(model.predict(test_features), test_targets) * (DIVIDE_TARGET_BY ** 2)
        errors.append(error)

    models = [model for _, model in sorted(zip(errors, models))]
    errors.sort()

    print("   {0:32} Error".format('Model'))
    for i in range(len(models)):
        print("{0}. {1:30} : {2}".format(i + 1, models[i].__class__.__name__, errors[i]))