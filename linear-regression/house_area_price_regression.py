
from data_preprocessing import load_csv
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class LinearRegression:
    def __init__(self, dataset_path: str="", amount: int=1000, epochs: int=10000, lr: float=0.00001):
        self.dataset = load_csv(dataset_path)
        self.amount = amount

        self.epochs = epochs
        self.lr = lr

        self.theta_0 = random.uniform(-1, 1)  
        self.theta_1 = random.uniform(-1, 1)

        self.predictions = None

        self.x = np.array(self.dataset["GrLivArea"].tolist()[:self.amount]) / max(np.array(self.dataset["GrLivArea"].tolist()[:self.amount]))
        self.y = np.array(self.dataset["SalePrice"].tolist()[:self.amount]) / max(np.array(self.dataset["SalePrice"].tolist()[:self.amount]))

    """ mean squared error """
    def MSE(self, y_prediction, y_true, deriv=(False, 1)):
        if deriv[0]:
            # deriv[1] is the  derivitive of the fit_function
            return 2 * np.mean(np.subtract(y_true, y_prediction) * -deriv[1])
        return np.mean(np.square(np.subtract(y_true, y_prediction)))

    """ linear function """
    def fit_function(self, t0, t1, x):
        return t0 + (t1 * x)

    """ train model """
    def train(self):
        for epoch in range(self.epochs):

            # predict and calulcate loss
            self.predictions = self.fit_function(self.theta_0, self.theta_1, self.x)
            loss = self.MSE(self.predictions, self.y)
            
            # gradient descent
            delta_theta_0 = self.MSE(self.predictions, self.y, deriv=(True, 1))
            delta_theta_1 = self.MSE(self.predictions, self.y, deriv=(True, self.x))

            self.theta_0 -= self.lr * delta_theta_0
            self.theta_1 -= self.lr * delta_theta_1

            # print loss and plot live time plot
            print("\nepoch", epoch, ", loss:", loss)

    """ plot regression line """
    def plot_regression_line(self):
        variables, predictions = map(list, zip(*sorted(zip(self.x, self.predictions))))
        plt.plot(variables, predictions, "b")
        plt.scatter(self.x, self.y, c="r")
        plt.show()


linearRegression = LinearRegression(dataset_path="dataset/houston_housing/single_variable_dataset/train.csv",
                                    epochs=150000,
                                    lr=0.00075,
                                    amount=500)

linearRegression.train()
linearRegression.plot_regression_line()
