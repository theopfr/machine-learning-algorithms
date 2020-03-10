
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


""" load dataset from sklearn and one-hot encode """
def load_dataset(test_size: int) -> list:
    iris_data = load_iris()

    x = iris_data.data
    y = iris_data.target.reshape(-1, 1)

    encoder = OneHotEncoder(sparse=False, categories="auto")
    y = encoder.fit_transform(y)

    return train_test_split(x, y, test_size=test_size)

""" apply preprocessing """
def preprocess(test_size: float=0.1):
    train_x, test_x, train_y, test_y = load_dataset(test_size)

    return train_x, test_x, train_y, test_y


preprocess()