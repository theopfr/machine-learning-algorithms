
import numpy as np
import matplotlib.pyplot as plt


def create_sine(x_range: int=100, lambda_: float=10, noisy: bool=False, show: bool=False):
    x = np.linspace(0, x_range, num=x_range)

    y = 0.5 * (np.sin(lambda_ * x) + 1)

    if noisy:
        noise = np.random.rand((len(y))) * 0.1
        y += noise

    if show:
        plt.plot(x, y)
        plt.show()

    return y

def data_creation(train_amount: int=500, test_amount: int=100):
    train_data = create_sine(x_range=train_amount, lambda_=0.025, noisy=False, show=False)
    test_data = create_sine(x_range=test_amount, lambda_=0.02, noisy=False)

    return train_data, test_data




