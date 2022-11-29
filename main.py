import pandas as pd
import random as r
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import helpers


train_features = pd.read_csv("train_features.csv")
train_targets = pd.read_csv("train_targets.csv")
test_features = pd.read_csv("test_features.csv")

train_features = helpers.labelEncode(train_features)
test_features = helpers.labelEncode(test_features)
train_targets = train_targets.values


regressor = LinearRegression()
losses = []

r.seed(1)
kf = KFold(n_splits=5, shuffle=True)
i = 1

for train, test in kf.split(train_features):
    x_train = train_features[train]
    x_test = train_features[test]

    y_train = train_targets[train]
    y_test = train_targets[test]

    regressor.fit(x_train, y_train)

    print("Train error for {}.fold : {}".format(i, mse(y_train, regressor.predict(x_train))))
    print("Test  error for {}.fold : {}".format(i, mse(y_test, regressor.predict(x_test))))
    print("------------------------------------------------")

    i += 1


"""
label = LabelEncoder()
label.fit(train_features.sex.drop_duplicates())
train_features.sex = label.transform(train_features.sex)
label.fit(train_features.smoker.drop_duplicates())
train_features.smoker = label.transform(train_features.smoker)
label.fit(train_features.region.drop_duplicates())
train_features.region = label.transform(train_features.region)
train_features.dtypes

lr = LinearRegression().fit(train_features, train_targets)

train_predictions = lr.predict(train_features)

print(lr.score(train,y_test))
"""