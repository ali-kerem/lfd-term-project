import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold

def labelEncode(features, ohe_children, ohe_region):
    le = LabelEncoder()
    features['sex'] = le.fit_transform(features['sex'])
    features['smoker'] = le.fit_transform(features['smoker'])

    if ohe_children:
        ohe = OneHotEncoder()
        children_ohe = pd.DataFrame(ohe.fit_transform(features[['children']]).toarray())
        children_ohe.columns = ohe.get_feature_names_out(['children'])
        features = features.join(children_ohe)
        features.drop('children', axis=1, inplace=True)

    if ohe_region:
        ohe = OneHotEncoder()
        region_ohe = pd.DataFrame(ohe.fit_transform(features[['region']]).toarray())
        region_ohe.columns = ohe.get_feature_names_out(['region'])
        features = features.join(region_ohe)
        features.drop('region', axis=1, inplace=True)

    else:
        features['region'] = le.fit_transform(features['region'])

    return features


def normalizeFeatures(features):
    features['age'] = (features['age']-features['age'].mean())/features['age'].std() # Normalize age
    features['bmi'] = (features['bmi']-features['bmi'].mean())/features['bmi'].std() # Normalize bmi

    return features


def prepareFeatures(fileName, normalize, ohe_children, ohe_region):
    features = pd.read_csv(fileName)
    features = labelEncode(features, ohe_children, ohe_region)

    if normalize:
        features = normalizeFeatures(features)

    return features.to_numpy()


def prepareTargets(fileName):
    targets = pd.read_csv(fileName)
    targets = targets.values
    targets = targets.flatten()

    return targets


def CV(features, targets, model, n_splits):
    errors = []
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_index, test_index in kf.split(features):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        model.fit(x_train, y_train)

        error = mse(y_test, model.predict(x_test))
        errors.append(error)

    return errors


def createSubmission(model, test_features, submissionFile):
    with open(submissionFile, 'w') as outFile:
        predictions = model.predict(test_features)
        outFile.write("ID,predicted\n")
        for i in range(len(predictions)):
            outFile.write(str(i) + ',' + str(predictions[i]) + "\n")


def printPerformance(model, features, targets):
    error = mse(targets, model.predict(features))
    print("Error : {}".format(error))
    print("Score : {}".format(model.score(features, targets)))
    return error
