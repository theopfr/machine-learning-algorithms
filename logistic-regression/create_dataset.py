
import numpy as np


def create_dataset(samples_per_class: int) -> list:
    dataset = [[], [], []]
    for i in range(samples_per_class):
        dataset[0].append(np.random.uniform(0, 0.5))
        dataset[1].append(np.random.uniform(0, 0.5))
        dataset[2].append(1)

        dataset[0].append(np.random.uniform(0.5, 1.0))
        dataset[1].append(np.random.uniform(0.5, 1.0))
        dataset[2].append(0)

    return dataset

def split(dataset: list, samples_per_class: int, test_size: float) -> list:
    train_data = [dataset[0][int(samples_per_class * test_size):], dataset[1][int(samples_per_class * test_size):], dataset[2][int(samples_per_class * test_size):]]
    test_data = [dataset[0][:int(samples_per_class * test_size)], dataset[1][:int(samples_per_class * test_size)], dataset[2][:int(samples_per_class * test_size)]]
    return train_data, test_data

def get_dataset(samples_per_class: int=100, test_size: int=0.2):
    dataset = create_dataset(samples_per_class)
    train_data, test_data = split(dataset, samples_per_class, test_size)
    return train_data, test_data
