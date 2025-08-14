import numpy as np


class SimpleLinearRegressionModel:
    """
    Simple Linear Regression model for univariate data.

    Parameters
    ----------
    float_precision : int, default=6
        Number of decimal digits to round all input and computed values.

    Attributes
    ----------
    float_precision : int
        Floating point rounding precision.
    indep_vars : list of float
        List of independent variable (input) values.
    dep_vars : list of float
        List of dependent variable (output) values.
    beta_slope : float
        Regression slope coefficient (β).
    alpha_value : float
        Intercept term (α).
    data_size : int
        Number of data points currently in the model.
    """

    def __init__(self, float_precision: int = 6) -> None:
        if float_precision <= 0:
            raise ValueError("precision argument must be positive or 0")

        self.float_precision = float_precision
        self.indep_vars = []
        self.dep_vars = []
        self.beta_slope = 0
        self.alpha_value = 0
        self.data_size = 0

    def _get_average(self, var_list: list[float]) -> float:
        return round(sum(var_list) / self.data_size, self.float_precision)

    def fit(self, data: list[tuple[float, float]]) -> float:
        """
        Set data and fit a simple linear regression line.

        Parameters
        ----------
        data : list of tuple(float, float)
            List of (x, y) pairs to use as model data.

        Returns
        -------
        float
            Coefficient of determination (R²) indicating model goodness-of-fit.
        """
        self.indep_vars = [round(row[0], self.float_precision) for row in data]
        self.dep_vars = [round(row[1], self.float_precision) for row in data]
        self.data_size = len(data)

        x_avg = self._get_average(self.indep_vars)
        y_avg = self._get_average(self.dep_vars)

        nxy = self.data_size * x_avg * y_avg
        product_sum = sum(x * y for x, y in zip(self.indep_vars, self.dep_vars))
        square_sum = sum(x * x for x in self.indep_vars)
        nxsquare_product = self.data_size * (x_avg**2)

        self.beta_slope = (product_sum - nxy) / (square_sum - nxsquare_product)
        self.alpha_value = round(
            y_avg - (self.beta_slope * x_avg), self.float_precision
        )

        y_variance = sum((y - y_avg) ** 2 for y in self.dep_vars) / self.data_size
        e_variance = (
            sum(
                (y - teoric_y) ** 2
                for y, teoric_y in zip(
                    self.dep_vars, (self.predict(x) for x in self.indep_vars)
                )
            )
            / self.data_size
        )

        return 1 - (e_variance / y_variance)

    def predict(self, input_x: float) -> float:
        """
        Predict the output y for a given input x using the trained model.

        Parameters
        ----------
        input_x : float
            Input value.

        Returns
        -------
        float
            Predicted output value.
        """
        return round(
            (self.beta_slope * input_x) + self.alpha_value, self.float_precision
        )


class MultiLinearRegressionModel(SimpleLinearRegressionModel):
    """
    Multiple Linear Regression model for multivariate input data.

    Inherits from SimpleLinearRegressionModel.

    Parameters
    ----------
    float_precision : int, default=6
        Number of decimal digits to round all input and computed values.

    Attributes
    ----------
    beta_values : list of float
        List of regression coefficients (β₁, β₂, ..., βₙ) for each input variable.
    """

    def __init__(self, float_precision: int = 6) -> None:
        super().__init__(float_precision)
        self.beta_values = []

    def fit(self, data: list[tuple[tuple[float], float]]) -> float:
        """
        Set data and fit a multiple linear regression model.

        Parameters
        ----------
        data : list of tuple(tuple of float, float)
            Each item is a tuple (x_vector, y), where x_vector is a tuple of features.

        Returns
        -------
        float
            Coefficient of determination (R²) indicating model goodness-of-fit.
        """
        self.indep_vars = [
            [round(x, self.float_precision) for x in row[0]] for row in data
        ]
        self.dep_vars = [round(row[1], self.float_precision) for row in data]
        self.data_size = len(data)

        x = np.array(self.indep_vars)
        y = np.array(self.dep_vars)

        x_matrix = np.hstack((np.ones((x.shape[0], 1)), x))
        coef_ = np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T @ y

        self.alpha_value = round(coef_[0], self.float_precision)
        self.beta_values = [round(b, self.float_precision) for b in coef_[1:]]

        residuals = y - x_matrix @ coef_
        ssr = np.sum(residuals**2)
        y_mean = np.mean(y)
        sst = np.sum((y - y_mean) ** 2)

        return 1 - (ssr / sst)

    def predict(self, input_x: list[float]) -> float:
        """
        Predict the output y for a given multivariate input vector.

        Parameters
        ----------
        input_x : list of float
            List of input features.

        Returns
        -------
        float
            Predicted output value.
        """
        return round(
            sum(
                [
                    self.alpha_value,
                    *[self.beta_values[i] * input_x[i] for i in range(len(input_x))],
                ]
            ),
            self.float_precision,
        )
