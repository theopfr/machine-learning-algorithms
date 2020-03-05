import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess



class NeuralNetwork:
    def __init__(self, epochs: int=500, lr: float=0.01, dropout_chance: float=0.5, hiddensize: int=32):
        self.epochs = epochs
        self.lr = lr
        self.dropout_chance = dropout_chance
        self.hiddensize = hiddensize
        
        self.train_x, self.test_x, self.train_y, self.test_y = preprocess()

        self.syn0 = 2 * np.random.random((self.hiddensize, 4)) - 1
        self.syn1 = 2 * np.random.random((3, self.hiddensize)) - 1
        self.b = 1

        self.losses = []

    """ sigmoid activation function """
    def _sigmoid(self, x, deriv: bool=False, do_dropout: bool=False):
        if deriv:
            return x * (1 - x)
        s = 1 / (1 + np.exp(-x))

        if do_dropout:
            return self._dropout(s)
        else:
            return s

    """ relu activation function """
    def _relu(self, x, deriv: bool=False, do_dropout: bool=False):
        if deriv:
            return 1/2 * (1 + np.sign(x))
        s = x/2 + 1/2 * x * np.sign(x)

        if do_dropout:
            return self._dropout(s)
        else:
            return s

    """ mean-squared-error loss function """
    def _MSE(self, y_prediction, y_true, deriv: tuple=(False, 1)):
        if deriv[0]:
            return 2 * np.subtract(y_true, y_prediction) * -deriv[1]
        return np.mean(np.square(np.subtract(y_true, y_prediction)))

    """ dropout layer """
    def _dropout(self, mat):
        chance = self.dropout_chance
        percent = chance * 100

        for i in range(len(mat)):
            for j in range(len(mat[0])):
                rand = np.random.randint(101)
                if rand <= percent:
                    mat[i][j] *= 0
                else:
                    pass
        return mat

    """ truth check for test validation """
    def _check(self, output, expected):
        for i in range(len(output)):
            output[i] = output[i].round(0)

        if list(output) == list(expected):
            return 1
        else:
            return 0

    """ train model """
    def train(self):
        print("\n[ training with ", len(self.train_x), " samples ]\n")

        for epoch in range(self.epochs):
            mean_loss = []
            for i in range(len(self.train_x)):

                X = np.asarray(self.train_x[i])
                y = np.asarray(self.train_y[i])

                l0 = X

                l1 = self._relu(np.dot(self.syn0, l0) + self.b, deriv=False, do_dropout=True)

                l2 = self._sigmoid(np.dot(self.syn1, l1) + self.b, deriv=False, do_dropout=False)

                loss = self._MSE(l2, y)
                l2_delta = self._MSE(l2, y, deriv=(True, self._sigmoid(l2, deriv=True, do_dropout=False)))

                l1_error = np.dot(l2_delta.T, self.syn1)
                l1_delta = l1_error.T * self._relu(l1, deriv=True, do_dropout=False)

                self.syn0 -= self.lr * np.dot(l0, l1_delta.T).T
                self.syn1 -= self.lr * np.dot(l1, l2_delta.T).T

                mean_loss.append(loss)

            if epoch % 5 == 0:
                print("loss after epoch [ " + str(epoch + 1) + " /", self.epochs, "] :", np.mean(mean_loss), "\n")

            self.losses.append(np.mean(mean_loss))

    """ test model """
    def test(self):
        correct_classified = 0
        testing_iterations = len(self.test_x)

        for i in range(testing_iterations):

            l1 = self._relu(np.dot(self.syn0, self.test_x[i]) + self.b, deriv=False, do_dropout=False)
            l2 = self._sigmoid(np.dot(self.syn1, l1) + self.b, deriv=False, do_dropout=False)

            correct_classified += self._check(l2, self.test_y[i])

        print("\ntesting with ", testing_iterations, " samples\ncorrect classified: ", correct_classified, " -> ", round((correct_classified / testing_iterations) * 100, 4), "%")

    """ plot loss history """
    def plot_loss(self):
        ys = []
        for i in range(len(self.losses)):
            ys.append(i)

        plt.style.use("ggplot")

        plt.plot(ys, self.losses, "r--")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("loss-curve")
        plt.show()


neuralNetwork = NeuralNetwork(epochs=400, 
                              lr=0.01, 
                              dropout_chance=0.5, 
                              hiddensize=32)

neuralNetwork.train()
neuralNetwork.test()
neuralNetwork.plot_loss()