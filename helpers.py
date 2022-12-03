from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd

DIVIDE_TARGET_BY = 1000


def labelEncode(data):
    le = LabelEncoder()
    data[data.columns[1]] = le.fit_transform(data.iloc[:, 1])
    data[data.columns[4]] = le.fit_transform(data.iloc[:, 4])

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
    data = np.array(ct.fit_transform(data))

    return data

def normalizeFeatures(features):
    features[:,4] = (features[:,4] - features[:,4].mean()) / features[:,4].std() # Normalize age
    features[:,6] = (features[:,6] - features[:,6].mean()) / features[:,6].std() # Normalize bmi
    features[:,7] = (features[:,7] - features[:,7].mean()) / features[:,7].std() # Normalize number of children
    return features

def prepareFeatures(fileName, normalize):
    features = pd.read_csv(fileName)
    features = labelEncode(features)

    if normalize:
        features = normalizeFeatures(features)

    return features

def prepareTargets(fileName):
    targets = pd.read_csv(fileName)
    targets = targets.values
    targets = targets.flatten()
    targets = targets / DIVIDE_TARGET_BY

    return targets

def createSubmission(regressor, test_features, submissionFile):
    with open(submissionFile, 'w') as outFile:
        predictions = regressor.predict(test_features)
        outFile.write("ID,predicted\n")
        for i in range(len(predictions)):
            outFile.write(str(i) + ',' + str(predictions[i] * DIVIDE_TARGET_BY) + "\n")


def printPerformance(regressor, features, targets):
    error = mse(targets, regressor.predict(features)) * (DIVIDE_TARGET_BY ** 2)
    print("Error : {}".format(error))
    print("Score : {}".format(regressor.score(features, targets)))
    return error
