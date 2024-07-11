"""General utility functions for the mathstats package."""
import math


def is_matrix_empty(matrix: list) -> bool:
    """Return True if *matrix* is None, empty, or contains no rows."""
    return not matrix or len(matrix) == 0


def transpose(matrix: list) -> list:
    """Return the transpose of a rectangular matrix (list of lists)."""
    if not matrix or not matrix[0]:
        return []
    return [[row[c] for row in matrix] for c in range(len(matrix[0]))]


def flatten(matrix: list) -> list:
    """Flatten a 2-D matrix into a 1-D list (row-major order)."""
    return [x for row in matrix for x in row]


def are_all_finite(data: list) -> bool:
    """Return True only if every value in *data* is finite (not NaN or Inf)."""
    return all(math.isfinite(x) for x in data)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))

