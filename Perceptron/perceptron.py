import numpy as np


class PerceptronClassifier(object):
    """Perceptron classifier.

    Parameters
    ------------
    learning_rate : float
        Learning rate (between 0.0 and 1.0)
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    epochs : int
        How many iterations to run on training dataset.

    Attributes
    -----------
    weights : 1d-array
        Weights after fitting.
    errors : list
        Classification errors.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 10,
        shuffle: bool = True,
        random_seed: int = 42,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.shuffle = shuffle
        self.random_seed = random_seed

    def _shuffle(self, X, y):
        """Shuffle training data
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        """
        p = np.random.permutation(len(y))
        return X[p], y[p]

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

        rand_gen = np.random.RandomState(self.random_seed)
        self.weights = rand_gen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.errors = []

        for _ in range(self.epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def _net_input(self, X):
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
        return np.where(self._net_input(X) >= 0.0, 1, -1)
