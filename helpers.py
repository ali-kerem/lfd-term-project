from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def labelEncode(data):
    le = LabelEncoder()
    data[data.columns[1]] = le.fit_transform(data.iloc[:, 1])
    data[data.columns[4]] = le.fit_transform(data.iloc[:, 4])

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
    data = np.array(ct.fit_transform(data))

    data = data

    return data

