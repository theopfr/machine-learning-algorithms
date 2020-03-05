import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_preprocessing import preprocess



class KNN:
    def __init__(self, K: int=5):
        self.K = K
        self.train_x, self.test_x, self.train_y, self.test_y = preprocess(test_size=0.2)

    """ get euclidean distance """
    def _distance(self, a: list, b: list) -> float:
        vector_between = np.array(b) - np.array(a)
        return np.sqrt(sum([pow(vector_between[0], 2), pow(vector_between[1], 2), pow(vector_between[2], 2), pow(vector_between[3], 2)]))

    """ classify datapoint """
    def _classify_datapoint(self, dataset: list, labels: list, datapoint: list) -> list:
        distances = []
        for idx in range(len(dataset)):
            distances.append((self._distance(datapoint, dataset[idx]), idx))

        distances = sorted(distances)
        k_nearest_neighbors = distances[:self.K]

        label_count = [0, 0, 0]
        for point, idx in k_nearest_neighbors:
            if list(labels[idx]) == [1, 0, 0]:
                label_count[0] += 1
            elif list(labels[idx]) == [0, 1, 0]:
                label_count[1] += 1
            elif list(labels[idx]) == [0, 0, 1]:
                label_count[2] += 1
        
        if max(label_count) == label_count[0]:
            return [1, 0, 0]
        elif max(label_count) == label_count[1]:
            return [0, 1, 0]
        elif max(label_count) == label_count[2]:
            return [0, 0, 1]

    """ test model """
    def test(self):
        correct = 0
        for idx in range(len(self.test_x)):
            sample, target = self.test_x[idx], self.test_y[idx]
            prediction = self._classify_datapoint(self.train_x, self.train_y, sample)

            print(prediction, target)

            if list(prediction) == list(target):
                correct += 1

        accuracy = round(100 * correct / len(self.test_x), 5)
        print("accuracy: " + str(accuracy) + "%")

        return accuracy

    """ plot dataset """
    def plot_dataset(self):
        all_data = np.concatenate((np.array(self.train_x), np.array(self.test_x)))
        all_labels = np.concatenate((np.array(self.train_y), np.array(self.test_y)))

        setosa, versicolor, virignica = [], [], []
        for i in range(len(all_data)):
            if list(all_labels[i]) == [1, 0, 0]:
                setosa.append(all_data[i])
            elif list(all_labels[i]) == [0, 1, 0]:
                versicolor.append(all_data[i])
            elif list(all_labels[i]) == [0, 0, 1]:
                virignica.append(all_data[i])
        
        setosa_x, setosa_y, setosa_z = np.array(setosa)[:,0], np.array(setosa)[:,1], np.array(setosa)[:,2]
        versicolor_x, versicolor_y, versicolor_z = np.array(versicolor)[:,0], np.array(versicolor)[:,1], np.array(versicolor)[:,2]
        virignica_x, virignica_y, virignica_z = np.array(virignica)[:,0], np.array(virignica)[:,1], np.array(virignica)[:,2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(setosa_x, setosa_y, setosa_z, color="green")
        ax.scatter(versicolor_x, versicolor_y, versicolor_z, color="blue")
        ax.scatter(virignica_x, virignica_y, virignica_z, color="red")

        plt.show()


knn = KNN(K=5)
knn.plot_dataset()
knn.test()
