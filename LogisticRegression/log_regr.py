import numpy as np

class LogisticRegressionClassifier(object):
    """Logistic regression classifier via gradient descent.

    Parameters
    ------------
    learning_rate : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Iterations over the training dataset.

    Attributes
    -----------
    weights : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, learning_rate=0.01, epochs=50):
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
        self : object

        """
        self.weights = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()

            # compute the logistic `cost`
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.5, 1, 0)

    def activation(self, z):
        """Compute sigmoid activation."""
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return sigmoid
