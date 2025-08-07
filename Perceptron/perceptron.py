import numpy as np


class PerceptronClassifier(object):
    """Perceptron classifier.

    Parameters
    ------------
    learning_rate : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        How many iterations to run on training dataset.

    Attributes
    -----------
    weights : 1d-array
        Weights after fitting.
    errors : list
        Classification errors.
    """

    def __init__(self, learning_rate:float=0.01, epochs:int=10):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : PerceptronClassifier
        """

        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        """Calculate network input

        Parameters
        ----------
        X : {array-like}, shape = [n_features]
            Data vector, where n_features is the number of features.

        Returns
        -------
        z : float
            Network input.
        """
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        """Run classification on dataset

        Parameters
        ----------
        X : {array-like}, shape = [n_features]
            Data vector, where n_features is the number of features.

        Returns
        -------
        y : float
            Target value.
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
