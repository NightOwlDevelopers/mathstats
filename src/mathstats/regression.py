"""Simple linear regression — pure Python implementation."""
import math
from .errors import MathStatsError


class LinearRegression:
    """Ordinary least-squares simple linear regression: y = slope * x + intercept.

    Usage::

        model = LinearRegression()
        model.fit([1, 2, 3], [2, 4, 5])
        print(model.slope, model.intercept)
        print(model.predict(4))
        print(model.r_squared())
    """

    def __init__(self):
        self._slope: float = None
        self._intercept: float = None
        self._xs: list = None
        self._ys: list = None
        self._fitted: bool = False

    def fit(self, xs: list, ys: list) -> 'LinearRegression':
        """Fit the model to the data points (xs, ys).

        Raises MathStatsError on invalid or insufficient input.
        Returns self for method chaining.
        """
        if len(xs) != len(ys):
            raise MathStatsError(
                f"xs and ys must have equal length ({len(xs)} vs {len(ys)})"
            )
        if len(xs) < 2:
            raise MathStatsError("At least 2 data points are required for regression")
        n = len(xs)
        x_bar = sum(xs) / n
        y_bar = sum(ys) / n
        ss_xx = sum((x - x_bar) ** 2 for x in xs)
        if ss_xx == 0.0:
            raise MathStatsError("Cannot fit: all x-values are identical (zero variance)")
        ss_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
        self._slope = ss_xy / ss_xx
        self._intercept = y_bar - self._slope * x_bar
        self._xs = list(xs)
        self._ys = list(ys)
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise MathStatsError("Model has not been fitted yet; call fit() first")

    @property
    def slope(self) -> float:
        """The slope (regression coefficient) of the fitted line."""
        self._check_fitted()
        return self._slope

    @property
    def intercept(self) -> float:
        """The y-intercept of the fitted line."""
        self._check_fitted()
        return self._intercept

    def predict(self, x: float) -> float:
        """Predict the y-value for a given x."""
        self._check_fitted()
        return self._slope * x + self._intercept

    def r_squared(self) -> float:
        """Coefficient of determination R² in [0, 1]."""
        self._check_fitted()
        y_bar = sum(self._ys) / len(self._ys)
        ss_tot = sum((y - y_bar) ** 2 for y in self._ys)
        if ss_tot == 0.0:
            return 1.0
        ss_res = sum(
            (y - self.predict(x)) ** 2 for x, y in zip(self._xs, self._ys)
        )
        return 1.0 - ss_res / ss_tot

    def residuals(self) -> list:
        """Return the list of residuals (y_actual - y_predicted) for training data."""
        self._check_fitted()
        return [y - self.predict(x) for x, y in zip(self._xs, self._ys)]

    def mean_squared_error(self) -> float:
        """Mean squared error on the training data."""
        self._check_fitted()
        res = self.residuals()
        return sum(r ** 2 for r in res) / len(res)

    def root_mean_squared_error(self) -> float:
        """Root mean squared error on the training data."""
        return math.sqrt(self.mean_squared_error())

    def standard_error(self) -> float:
        """Standard error of the estimate (residual standard deviation)."""
        self._check_fitted()
        n = len(self._xs)
        if n < 3:
            raise MathStatsError("Standard error requires at least 3 data points")
        ss_res = sum(r ** 2 for r in self.residuals())
        return math.sqrt(ss_res / (n - 2))

    def __repr__(self) -> str:
        if self._fitted:
            return f"LinearRegression(slope={self._slope:.6g}, intercept={self._intercept:.6g})"
        return "LinearRegression(unfitted)"
