import numpy as np


class Node:
    def __init__(self, label=None, attribute=None):
        self.children = {}
        self.label = label
        self.attribute = attribute


class DecisionTreeClassifier:
    __attr__ = {
        'max_depth': np.inf,
    }

    def __init__(self, **kwargs):
        for k, v in self.__attr__.items():
            if k in kwargs.keys():
                self.__setattr__(k, kwargs[k])
            else:
                self.__setattr__(k, v)
        self.root = None

    def fit(self, X, y):
        self.root = self._decision_tree_learning(X, y, y)

    def _decision_tree_learning(self, X, y, y_parent, depth=0):
        if len(y) == 0:
            return Node(attribute=None, label=np.bincount(y_parent).argmax())
        elif len(np.unique(y)) == 1:
            return Node(attribute=None, label=np.unique(y)[0])
        elif len(X) == 0:
            return Node(attribute=None, label=np.bincount(y).argmax())
        elif depth >= self.max_depth:
            return Node(attribute=None, label=np.bincount(y).argmax())
        else:
            attr = self._importance(X, y)
            node = Node(attribute=attr, label=np.bincount(y).argmax())
            for v in np.unique(X[:, attr]):
                X_branch = X[np.where(X[:, attr] == v)]
                y_branch = y[np.where(X[:, attr] == v)]
                node.children[v] = self._decision_tree_learning(np.delete(X_branch, attr, axis=1), y_branch, y, depth+1)
            return node

    @staticmethod
    def _importance(train, train_labels):
        entropy = np.vectorize(lambda x: 0.0 if x == 0.0 else -x * np.log2(x))
        n = len(train_labels)
        m = train.shape[1]
        k, ns = np.unique(train_labels, return_counts=True)
        p = ns / n
        train_entropy = np.sum(entropy(p))
        gain = {}
        for i in range(m):
            p = {}
            scaled_entropies = list()
            for j in range(n):
                if p.get(train[j, i]) is None:
                    p[train[j, i]] = np.array((0., 0., 0.))
                np.add.at(p[train[j, i]], train_labels[j], 1)
            for key, value in p.items():
                n_value = float(np.sum(value))
                probability = n_value / n
                scaled_value = value / n_value
                scaled_entropies.append(float(probability * np.sum(entropy(scaled_value))))
            gain[i] = train_entropy - np.sum(scaled_entropies)
        return max(gain, key=gain.get)

    def predict(self, X):
        prediction = np.empty(len(X), dtype=object)
        for i, x in enumerate(X):
            node = self.root
            while prediction[i] is None:
                if node.attribute is None:
                    prediction[i] = node.label
                elif node.children.get(x[node.attribute]) is None:
                    prediction[i] = node.label
                else:
                    node = node.children[x[node.attribute]]
        return prediction

    def score(self, X, y):
        y_predicted = self.predict(X)
        accuracy = (y_predicted == y).sum() / len(X)
        return accuracy
