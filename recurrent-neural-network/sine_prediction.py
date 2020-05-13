
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_creation import data_creation


np.random.seed(23432)


class RNN:
    def __init__(self, train_data: list=[], test_data: list=[], sequence_length: int=15, epochs: int=20, lr: float=1e3, hiddensize: int=64, dropout_chance: float=0.3):
        self.train_data = train_data
        self.test_data = test_data

        self.sequence_length = sequence_length
        self.epochs = epochs
        self.lr = lr
        self.hiddensize = hiddensize
        self.dropout_chance = dropout_chance

        self.weights0 = 2 * np.random.random((self.hiddensize, 1)) - 1
        self.weights1 = 2 * np.random.random((1, self.hiddensize)) - 1
        self.hidden_state_weights = 2 * np.random.random((1, self.hiddensize)) - 1  # for i in range(self.sequence_length)]

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

    def _tanh(self, x, deriv=False, do_dropout=False):
        if deriv:
            return 4 / pow((np.exp(x) + np.exp(-x)), 2)
        s = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x) + 0.000000001)

        if do_dropout:
            return self.dropout(s)
        else:
            return s

    """ mean-squared-error loss function """
    def _MSE(self, y_prediction, y_true, deriv: tuple=(False, 1)):
        if deriv[0]:
            return 2 * np.subtract(y_true, y_prediction) * -deriv[1]
        return np.square(np.subtract(y_true, y_prediction))

    """ log-loss function """
    def _log_loss(self, y_true: float, y_prediction: float, deriv: tuple=(False, None)):
        if deriv[0]:
            return np.subtract(y_true, y_prediction) * deriv[1]
        return - y_true * np.log(y_prediction + (1 - y_true) * np.log(1 - y_prediction))

    """ dropout layer """
    def _dropout(self, mat):
        chance = self.dropout_chance

        mat = np.array(mat)
        mask = np.random.choice([0, 1], size=mat.shape, p=[chance, 1-chance])
        mat = mat * mask
        return mat

    def _normalize(self, x):
        maximum = max(self.train_data)
        x = np.array(x) / maximum
        #x = ((x - min(x)) / (max(x) - min(x)))
        return list(x)


    """ get sample from time-series data by shifting the previous sample by one """
    def _shift_sample(self, dataset: list, idx: int):
        sample = dataset[idx:(idx + self.sequence_length)]
        #sample = self._normalize(sample)
        target = dataset[(self.sequence_length + idx) + 1]

        return sample, target

    def train(self):
        total_loss, validation_accuracy = [], []

        for epoch in range(1, self.epochs+1):
            for i in tqdm(range((len(self.train_data) - self.sequence_length) - 1)):
                
                outputs, hiddenstates, timestep_losses = [], [], []
                hiddenstate = np.zeros((self.hiddensize, 1))

                # load new sample (sliding-window)
                input_sequence, target = self._shift_sample(self.train_data, i)

                for timestep in range(len(input_sequence)):
                    input_value = input_sequence[timestep]

                    hiddenstate = self._sigmoid(np.dot(self.weights0, input_value) + np.dot(self.hidden_state_weights, hiddenstate), do_dropout=False)
                    hiddenstates.append(hiddenstate)

                    output = self._sigmoid(np.dot(self.weights1, hiddenstate))
                    outputs.append(output)

                    loss = self._MSE(target, output)[0][0]
                    timestep_losses.append(loss)

                for timestep in list(range(len(input_sequence)))[::-1]:
                    prediction = outputs[timestep]

                    weights1_delta = self._MSE(prediction, target, deriv=(True, self._sigmoid(prediction, deriv=True)))
                    
                    weights0_error = np.dot(weights1_delta, self.weights1)
                    weights0_delta = weights0_error.T * self._sigmoid(hiddenstates[timestep], deriv=True)

                    weights0_delta = np.dot(input_sequence[timestep], weights0_delta.T).T
                    hidden_delta = np.dot(hiddenstates[timestep-1], weights1_delta.T).T
                    weights1_delta = np.dot(hiddenstates[timestep], weights1_delta.T).T

                    self.weights0 -= self.lr * weights0_delta
                    self.weights1 -= self.lr * weights1_delta
                    self.hidden_state_weights -= self.lr * hidden_delta

            print("epoch:", str(epoch), "/", str(self.epochs), " - loss:", round(sum(timestep_losses), 5))

    def test(self):
        predicted = []

        # self.test_data is acutally the same as self.train_data :/ hm

        for i in tqdm(range(0, (len(self.test_data) - self.sequence_length - 1))):
            hiddenstate = np.zeros((self.hiddensize, 1))

            input_sequence, target = self._shift_sample(self.test_data, i)

            for timestep in range(len(input_sequence)):
                input_value = input_sequence[timestep]

                hiddenstate = self._sigmoid(np.dot(self.weights0, input_value) + np.dot(self.hidden_state_weights, hiddenstate))

                output = self._sigmoid(np.dot(self.weights1, hiddenstate))

            predicted.append(output)

        predicted = list(np.array(predicted).T)[0][0]

        plt.plot(range(len(self.test_data)), self.test_data, c="orange")
        plt.plot(list(range(len(self.test_data)))[self.sequence_length+1:], list(predicted), c="blue")
        plt.show()


if __name__ == "__main__":
    train_set, test_set = data_creation(train_amount=500, test_amount=500)
    
    rnn = RNN(train_data=train_set,
                test_data=test_set,
                sequence_length=5,
                epochs=300,
                lr=0.0001,
                hiddensize=512,
                dropout_chance=0.5)
    
    rnn.train()
    rnn.test()