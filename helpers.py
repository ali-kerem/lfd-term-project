from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd


def labelEncode(data):
    le = LabelEncoder()
    data[data.columns[1]] = le.fit_transform(data.iloc[:, 1])
    data[data.columns[4]] = le.fit_transform(data.iloc[:, 4])

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
    data = np.array(ct.fit_transform(data))

    return data


def createSubmission(regressor):
    test_features = pd.read_csv("test_features.csv")
    test_features = labelEncode(test_features)

    with open('submission.csv', 'w') as outFile:
        predictions = regressor.predict(test_features)
        outFile.write("ID,predicted\n")
        for i in range(len(predictions)):
            outFile.write(str(i) + ',' + str(predictions[i]) + "\n")


def testScore(regressor):
    test_features = pd.read_csv("test_features.csv")
    test_features = labelEncode(test_features)

    test_targets = pd.read_csv("test_targets.csv")
    test_targets = test_targets.values

    return mse(test_targets, regressor.predict(test_features))