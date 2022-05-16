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


class Network:
    def __init__(self, output_layer, input_layer, inner_layers):
        self.layers = list()
        self.layers.append(input_layer)
        self.layers.extend(inner_layers)
        self.layers.append(output_layer)


class NeuralNetworkClassifier:
    def __init__(self, n_hidden=2, n_iter=1500, learning_rate=0.1):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.activation_function = lambda x: 1 / (1 + np.e ** -x)
        self.d_activation_function = lambda x: self.activation_function(x) * (1 - self.activation_function(x))
        self.neural_net = None

    def fit(self, X, y):
        self.n_class = len(np.unique(y))
        n_input = X.shape[1]
        n_output = len(np.unique(y))
        n_neuron = n_input
        output_layer = {'weights': np.random.sample((n_output, n_neuron)) - 0.5, 'name': 'l_output'}
        input_layer = {'weights': np.random.sample((n_neuron, n_input)) - 0.5, 'name': 'l_input'}
        inner_layers = list()
        for i in range(self.n_hidden - 1):
            hidden_layer = {'weights': np.random.sample((n_neuron, n_neuron)) - 0.5,
                            'name': "l_hidden{0}".format(i + 1)}
            inner_layers.append(hidden_layer)
        self.neural_net = Network(output_layer, input_layer, inner_layers)
        for _ in range(self.n_iter):
            for x, _y in zip(X, y):
                # Forward propagation
                outputs = x
                for layer in self.neural_net.layers:
                    layer['x'] = outputs
                    layer['input'] = np.dot(layer['weights'], outputs)
                    layer['output'] = self.activation_function(layer['input'])
                    outputs = layer['output']
                # Backward propagation
                # Output layer
                target = np.full(n_output, 0.5)
                target[_y] = 0.95
                self.neural_net.layers[-1]['deltas'] = self.d_activation_function(
                    self.neural_net.layers[-1]['input']) * (target - outputs)
                weighted_deltas = np.dot(self.neural_net.layers[-1]['deltas'], self.neural_net.layers[-1]['weights'])
                # Layer L-1 to input layer
                for layer in reversed(self.neural_net.layers[:-1]):
                    layer['deltas'] = self.d_activation_function(layer['input']) * weighted_deltas
                    weighted_deltas = np.dot(layer['deltas'], layer['weights'])
                # Update weight
                for layer in self.neural_net.layers:
                    deltas = self.learning_rate * np.expand_dims(layer['x'], -1).T * np.expand_dims(layer['deltas'], -1)
                    layer['weights'] += deltas

    def predict(self, X):
        prediction = np.zeros(len(X))
        for i, x in enumerate(X):
            outputs = x
            for layer in self.neural_net.layers:
                layer['input'] = np.dot(layer['weights'], outputs)
                layer['output'] = self.activation_function(layer['input'])
                outputs = layer['output']
            prediction[i] = np.argmax(outputs)
        return prediction

    def score(self, X, y):
        y_predicted = self.predict(X)
        accuracy = (y_predicted == y).sum() / len(X)
        return accuracy
