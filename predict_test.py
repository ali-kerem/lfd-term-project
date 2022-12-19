import helpers
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor, XGBRFRegressor


if __name__ == "__main__":
    train_features = helpers.prepareFeatures("train_features.csv", normalize=False, ohe_children=False, ohe_region=True)
    train_targets = helpers.prepareTargets("train_targets.csv")

    test_features = helpers.prepareFeatures("test_features.csv", normalize=False, ohe_children=False, ohe_region=True)
    test_targets = helpers.prepareTargets("test_targets.csv")

    models = []
    errors = []

    xgbrf = XGBRFRegressor(max_depth=4, n_estimators=200, subsample=0.442, colsample_bynode=1, min_child_weight=7)
    gbr = GradientBoostingRegressor(max_depth=2, n_estimators=68, learning_rate=0.228)
    models.append(VotingRegressor([('gbr', gbr), ('xgbrf', xgbrf)]))

    for model in models:
        model.fit(train_features, train_targets)
        error = mse(model.predict(test_features), test_targets)
        errors.append(error)

    models = [model for _, model in sorted(zip(errors, models))]
    errors.sort()

    print("   {0:32} Error".format('Model'))
    for i in range(len(models)):
        print("{0}. {1:30} : {2}".format(i + 1, models[i].__class__.__name__, errors[i]/1000000))

    helpers.createSubmission(model=models[0], test_features=test_features, submissionFile="newsub.csv")

    """
    from sklearn.linear_model import LinearRegression, Ridge, LassoLars, Perceptron
    from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, VotingRegressor, \
        HistGradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import GridSearchCV
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor, XGBRFRegressor
    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    """

    #xgbrf = XGBRFRegressor(colsample_bynode=0.8,learning_rate=1,max_depth=5,num_parallel_tree=100,objective='binary:logistic',subsample=0.8,tree_method='gpu_hist')
    #cbr = CatBoostRegressor(learning_rate=0.1, max_depth=2, n_estimators=150, silent=True)
    #xgb = XGBRegressor(n_estimators=100, max_depth=2, min_child_weight=1, subsample=1, colsample_bytree=1, learning_rate=0.1)

    """rfr = RandomForestRegressor(n_estimators=200, criterion='squared_error', max_depth=5, min_samples_split=1,
                                min_samples_leaf=11, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None,
                                min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                                verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)"""


    #models.append(xgbrf)
    #models.append(cbr)
    #models.append(xgb)
    #models.append(rfr)
    #lgbm = LGBMRegressor(learning_rate=0.056, max_depth=3, n_estimators=200, num_leaves=25)

    """
    cbr = CatBoostRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, silent=True)
    xgb = XGBRegressor(n_estimators=100, max_depth=2, min_child_weight=1, subsample=1, colsample_bytree=1, learning_rate=0.1)
    rfr = RandomForestRegressor(max_depth=3, n_estimators=500)
    """

    """
    params = [{'n_estimators': list(range(10,200,20)),
               'max_depth': list(range(2,4)),
               'learning_rate': np.arange(0.1, 0.20, 0.02) }]
    models.append(GridSearchCV(XGBRegressor(), param_grid=params, scoring='neg_mean_squared_error', cv=5))
    """

    """
    params = [{'n_estimators': list(range(100, 700, 100)),
               'max_depth': list(range(2, 8))}]
    models.append(GridSearchCV(RandomForestRegressor(), param_grid=params, scoring='neg_mean_squared_error', cv=5))
    """

    """
    params = [{'n_estimators': list(range(100, 700, 100)),
               'max_depth': list(range(2, 8))}]
    models.append(GridSearchCV(XGBRegressor(), param_grid=params, scoring='neg_mean_squared_error', cv=5))
    """

    #models.append(HistGradientBoostingRegressor(max_depth=3,learning_rate=0.1))
    #models.append(CatBoostRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42, silent=True))
    #models.append(LGBMRegressor(learning_rate=0.05, max_depth=3, n_estimators=100, num_leaves=25))
    #models.append(XGBRegressor(n_estimators=100,max_depth=2,min_child_weight=1,subsample=1,colsample_bytree=1, learning_rate=0.1))
    #models.append(GradientBoostingRegressor(max_depth=2))
    #models.append(RandomForestRegressor(max_depth = 3, n_estimators=500))
