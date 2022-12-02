from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import normalize
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

def prepareFeatures(fileName):
    features = pd.read_csv(fileName)
    features = labelEncode(features)

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

"""
# Normalize the features
def prepareFeatures(fileName):
    features = pd.read_csv(fileName)
    features = labelEncode(features)
    normalized_age = normalize(features[:, [4]], axis=0)
    normalized_bmi_children = normalize(features[:, [6,7]], axis=0)
    features = np.concatenate((features[:,[0,1,2,3]], normalized_age, features[:,[5]], normalized_bmi_children, features[:, [8]]), axis=1)
    #features = normalize(features)
    return features
"""