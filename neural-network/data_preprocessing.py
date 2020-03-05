from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


""" load dataset from sklearn and one-hot encode """
def load_dataset(testing_size: int) -> list:
    iris_data = load_iris()

    x = iris_data.data
    y = iris_data.target.reshape(-1, 1)

    encoder = OneHotEncoder(sparse=False, categories="auto")
    y = encoder.fit_transform(y)

    return train_test_split(x, y, test_size=testing_size)

""" reshape data """
def change_dimensions(data: list) -> list:
    d = []
    for i in data:
        d.append(np.asarray([i]).T)

    return d

""" apply preprocessing """
def preprocess():
    train_x, test_x, train_y, test_y = load_dataset(0.25)
    train_x, test_x, train_y, test_y = change_dimensions(train_x), change_dimensions(test_x), \
                                       change_dimensions(train_y), change_dimensions(test_y)

    return train_x, test_x, train_y, test_y

