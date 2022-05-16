# Copyright (c) 2022 Szepol
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from heapq import heappush, heappop


def euclidean_distance(a, b):
    return np.linalg.norm((a - b), 2)


def manhattan_distance(a, b):
    return np.linalg.norm((a - b), 1)


class KNNClassifier:
    __metrics__ = {
        'Euclidean': euclidean_distance,
        'Manhattan': manhattan_distance
    }

    def __init__(self, k, metric=""):
        self.k = k
        self.metric = None
        self.X = None
        self.y = None
        for key, val in self.__metrics__.items():
            if metric == key:
                self.metrics = self.__metrics__.get(key)

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        prediction = np.zeros(len(X))
        for i, x in enumerate(X):
            heap = []
            for j in range(self.X.shape[0]):
                if self.metric == 'ManhattanDistance':
                    distance = manhattan_distance(x, self.X[j, :])
                else:
                    distance = euclidean_distance(x, self.X[j, :])
                heappush(heap, (distance, self.y[j]))
            counter = dict.fromkeys(self.y, 0)
            for _ in range(self.k):
                _, y = heappop(heap)
                counter[y] = counter[y] + 1
            prediction[i] = max(counter, key=counter.get)
        return prediction

    def score(self, X, y):
        y_predicted = self.predict(X)
        accuracy = (y_predicted == y).sum() / len(X)
        return accuracy
