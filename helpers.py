import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold

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


def prepareTargets(fileName, divisor):
    targets = pd.read_csv(fileName)
    targets = targets.values
    targets = targets.flatten()
    targets = targets / divisor

    return targets


def CV(features, targets, model, n_splits, divisor):
    errors = []
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_index, test_index in kf.split(features):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        model.fit(x_train, y_train)

        error = mse(y_test, model.predict(x_test)) * (divisor ** 2)
        errors.append(error)

    return errors


def createSubmission(model, test_features, submissionFile, divisor):
    with open(submissionFile, 'w') as outFile:
        predictions = model.predict(test_features)
        outFile.write("ID,predicted\n")
        for i in range(len(predictions)):
            outFile.write(str(i) + ',' + str(predictions[i] * divisor) + "\n")


def printPerformance(model, features, targets, divisor):
    error = mse(targets, model.predict(features)) * (divisor ** 2)
    print("Error : {}".format(error))
    print("Score : {}".format(model.score(features, targets)))
    return error
