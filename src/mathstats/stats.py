"""Statistical computation routines for matrices and 1-D data series."""
import math
import statistics
from .errors import MathStatsError


def trimmed_mean(data: list, proportion: float = 0.1) -> float:
    """Compute the trimmed (truncated) mean by removing *proportion* of values
    from each tail.

    *proportion* is the fraction of observations to remove from each end
    (e.g. 0.1 removes the lowest 10% and highest 10%).
    """
    if not data:
        raise MathStatsError("Data list cannot be empty")
    if not (0.0 <= proportion < 0.5):
        raise MathStatsError(f"proportion must be in [0, 0.5), got {proportion}")
    n = len(data)
    cut = int(math.floor(n * proportion))
    trimmed = sorted(data)[cut: n - cut] if cut > 0 else sorted(data)
    if not trimmed:
        raise MathStatsError("All data was trimmed; reduce proportion")
    return sum(trimmed) / len(trimmed)


def winsorize(data: list, limits: tuple = (0.05, 0.05)) -> list:
    """Winsorize *data* by clamping values in the tails.

    *limits* is a two-element tuple (lower_fraction, upper_fraction).  Values
    below the lower percentile are replaced by the lower percentile value;
    values above the upper percentile are replaced by the upper percentile value.
    Returns a new list.
    """
    if not data:
        raise MathStatsError("Data list cannot be empty")
    lo_frac, hi_frac = limits
    if not (0.0 <= lo_frac < 0.5 and 0.0 <= hi_frac < 0.5):
        raise MathStatsError("Both limit fractions must be in [0, 0.5)")
    n = len(data)
    sorted_data = sorted(data)
    lo_idx = int(math.floor(n * lo_frac))
    hi_idx = n - int(math.floor(n * hi_frac)) - 1
    lo_val = sorted_data[lo_idx]
    hi_val = sorted_data[hi_idx]
    return [max(lo_val, min(hi_val, x)) for x in data]


def geometric_mean(data: list) -> float:
    """Compute the geometric mean of *data* (all values must be positive)."""
    if not data:
        raise MathStatsError("Data list cannot be empty")
    for i, x in enumerate(data):
        if x <= 0:
            raise MathStatsError(
                f"Geometric mean requires all positive values; got {x} at index {i}"
            )
    log_sum = sum(math.log(x) for x in data)
    return math.exp(log_sum / len(data))


def harmonic_mean(data: list) -> float:
    """Compute the harmonic mean of *data* (all values must be non-zero)."""
    if not data:
        raise MathStatsError("Data list cannot be empty")
    for i, x in enumerate(data):
        if x == 0:
            raise MathStatsError(
                f"Harmonic mean is undefined when any value is zero; found zero at index {i}"
            )
    return len(data) / sum(1.0 / x for x in data)


def mode(data: list) -> list:
    """Return all modes (most frequent values) of *data* as a sorted list.

    Unlike statistics.mode, this returns *all* values that share the highest
    frequency (multi-modal support).
    """
    if not data:
        raise MathStatsError("Data list cannot be empty")
    freq: dict = {}
    for x in data:
        freq[x] = freq.get(x, 0) + 1
    max_count = max(freq.values())
    modes = sorted(k for k, v in freq.items() if v == max_count)
    return modes


def compute_stats(matrix: list, stat: str, axis: str) -> dict:
    if not matrix or not matrix[0]:
        raise MathStatsError("Empty matrix")
    
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    for r_idx, row in enumerate(matrix):
        if len(row) != num_cols:
            raise MathStatsError(f"Row {r_idx} has different length ({len(row)}) than row 0 ({num_cols})")

    if stat in ('stdev', 'all') and (num_rows < 2 or num_cols < 2):
        raise MathStatsError("Error computing stdev: variance requires at least two data points")

    results = {}
    
    def get_mean(data: list) -> float:
        try:
            return statistics.mean(data)
        except Exception as e:
            raise MathStatsError(f"Error computing mean: {e}")

    def get_stdev(data: list) -> float:
        try:
            return statistics.stdev(data)
        except Exception as e:
            raise MathStatsError(f"Error computing stdev: {e}")

    if axis in ('rows', 'all'):
        row_results = []
        for r_idx, row in enumerate(matrix):
            stats = {}
            if stat in ('mean', 'all'):
                stats['mean'] = get_mean(row)
            if stat in ('stdev', 'all'):
                stats['stdev'] = get_stdev(row)
            row_results.append(stats)
        results['rows'] = row_results

    if axis in ('columns', 'all'):
        col_results = []
        for c_idx in range(num_cols):
            col_data = [matrix[r_idx][c_idx] for r_idx in range(num_rows)]
            stats = {}
            if stat in ('mean', 'all'):
                stats['mean'] = get_mean(col_data)
            if stat in ('stdev', 'all'):
                stats['stdev'] = get_stdev(col_data)
            col_results.append(stats)
        results['columns'] = col_results

    return results
