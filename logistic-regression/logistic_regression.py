
import numpy as np
import matplotlib.pyplot as plt
from create_dataset import get_dataset



class LogisticRegression:
    def __init__(self, epochs: int=100, lr: float=0.1):
        self.train_set, self.test_set = get_dataset(samples_per_class=20, test_size=0.2)
        self.epochs = epochs
        self.lr = lr

        # initial parameters
        self.theta_1 = np.random.uniform(-1, 0)
        self.theta_2 = np.random.uniform(-1, 0)
        self.theta_3 = np.random.uniform(-1, 0)

    """ calculate the y value for given x of a linear function """
    def _calculate_function(self, x: int):
        return - ((self.theta_1 * x + self.theta_3) / (self.theta_2))

    """ get distance between point and function """
    def _get_distance(self, point: tuple):
        x, y = point
        return (self.theta_1 * x + self.theta_2 * y + self.theta_3) / (np.sqrt(pow(self.theta_1, 2) + pow(self.theta_2, 2)))

    """ predict with activation function """
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _log_loss(self, y_true: float, y_prediction: float, deriv: tuple=(False, None)):
        n = len(y_prediction)
        y_true, y_prediction = np.array(y_true), np.array(y_prediction)

        if deriv[0]:
            return np.mean(np.subtract(y_true, y_prediction) * deriv[1])
        return - np.mean(y_true * np.log(y_prediction) + (1 - y_true) * np.log(1 - y_prediction))


    """ train model """
    def train(self):
        for epoch in range(self.epochs):
            epoch_loss = []
            
            targets, predictions = [], []
            for idx in range(len(self.train_set[0])):
                x, y = self.train_set[0][idx], self.train_set[1][idx]
                target = self.train_set[2][idx]

                prediction = self._sigmoid(self._get_distance((x, y)))

                predictions.append(prediction)
                targets.append(target)

            loss = self._log_loss(targets, predictions)

            self.theta_1 += self.lr * self._log_loss(targets, predictions, deriv=(True, x))
            self.theta_2 += self.lr * self._log_loss(targets, predictions, deriv=(True, y))
            self.theta_3 += self.lr * self._log_loss(targets, predictions, deriv=(True, 1))

            epoch_loss.append(loss)

            print("loss after epoch [", epoch, "/", self.epochs, "] : ", np.mean(epoch_loss))
    
    """ test model """
    def test(self):
        pass
    
    """ plot the dataset and the line """
    def plot(self):
        # split classes
        blue, red = [], []
        for idx in range(len(self.train_set[0])):
            if self.train_set[2][idx] == 1:
                blue.append((self.train_set[0][idx], self.train_set[1][idx]))
            else:
                red.append((self.train_set[0][idx], self.train_set[1][idx]))

        blue = list(zip(*blue))
        red = list(zip(*red))

        # change plot style
        plt.style.use("ggplot")

        # plot dataset
        plt.scatter(list(blue[0]), list(blue[1]), color="blue")
        plt.scatter(list(red[0]), list(red[1]), color="red")

        # plot function
        xs = np.arange(-1, 2, 0.1)
        ys = [self._calculate_function(x) for x in xs]
        plt.plot(xs, ys, color="black")

        plt.axis("equal")
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.show()


logisticRegression = LogisticRegression(epochs=5000, 
                                        lr=0.1)

logisticRegression.plot()
logisticRegression.train()
logisticRegression.plot()


    