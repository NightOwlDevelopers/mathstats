"""Correlation and covariance functions — pure Python implementations."""
import math
from .errors import MathStatsError


def _check_equal_length(xs: list, ys: list) -> None:
    if len(xs) != len(ys):
        raise MathStatsError(
            f"xs and ys must have the same length ({len(xs)} vs {len(ys)})"
        )
    if len(xs) < 2:
        raise MathStatsError("At least two data points are required")


def _mean(data: list) -> float:
    return sum(data) / len(data)


def covariance(xs: list, ys: list) -> float:
    """Sample covariance Cov(X, Y) = sum((xi - x_bar)(yi - y_bar)) / (n - 1)."""
    _check_equal_length(xs, ys)
    n = len(xs)
    x_bar = _mean(xs)
    y_bar = _mean(ys)
    cov = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / (n - 1)
    return cov


def pearson_r(xs: list, ys: list) -> float:
    """Pearson product-moment correlation coefficient.

    r = Cov(X, Y) / (std(X) * std(Y))
    Returns a value in [-1, 1].
    """
    _check_equal_length(xs, ys)
    n = len(xs)
    x_bar = _mean(xs)
    y_bar = _mean(ys)
    numerator = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    ss_x = sum((x - x_bar) ** 2 for x in xs)
    ss_y = sum((y - y_bar) ** 2 for y in ys)
    denominator = math.sqrt(ss_x * ss_y)
    if denominator == 0.0:
        raise MathStatsError("Cannot compute Pearson r: zero variance in data")
    return numerator / denominator


def _rank(data: list) -> list:
    """Return fractional ranks (1-based, average ties)."""
    n = len(data)
    sorted_vals = sorted(enumerate(data), key=lambda iv: iv[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_vals[j + 1][1] == sorted_vals[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_r(xs: list, ys: list) -> float:
    """Spearman rank correlation coefficient.

    Converts each series to ranks then computes Pearson r on the ranks.
    Handles ties via average-rank method.
    """
    _check_equal_length(xs, ys)
    rx = _rank(xs)
    ry = _rank(ys)
    return pearson_r(rx, ry)


def correlation_matrix(data: list) -> list:
    """Compute the pairwise Pearson correlation matrix for multi-column data.

    *data* is a list of rows (list of lists). Each column is treated as a variable.
    Returns a list-of-lists (n_cols x n_cols) symmetric matrix where entry [i][j]
    is the Pearson r between column i and column j.
    """
    if not data or not data[0]:
        raise MathStatsError("Data must be a non-empty 2-D list")
    n_cols = len(data[0])
    for r_idx, row in enumerate(data):
        if len(row) != n_cols:
            raise MathStatsError(
                f"Row {r_idx} has {len(row)} elements; expected {n_cols}"
            )
    columns = [[row[c] for row in data] for c in range(n_cols)]
    matrix = []
    for i in range(n_cols):
        row_result = []
        for j in range(n_cols):
            if i == j:
                row_result.append(1.0)
            else:
                try:
                    r = pearson_r(columns[i], columns[j])
                except MathStatsError:
                    r = float('nan')
                row_result.append(r)
        matrix.append(row_result)
    return matrix


def kendall_tau(xs: list, ys: list) -> float:
    """Kendall's tau-b rank correlation coefficient.

    Counts concordant (C) and discordant (D) pairs.
    tau_b = (C - D) / sqrt((C + D + T_x)(C + D + T_y))
    where T_x, T_y are the number of ties in xs and ys respectively.
    """
    _check_equal_length(xs, ys)
    n = len(xs)
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if dx == 0:
                    ties_x += 1
                if dy == 0:
                    ties_y += 1
    denom = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    if denom == 0.0:
        raise MathStatsError("Cannot compute Kendall's tau: all pairs are tied")
    return (concordant - discordant) / denom
