"""Outlier detection methods — pure Python implementations."""
import math
import statistics
from .errors import MathStatsError


def _mean(data: list) -> float:
    return sum(data) / len(data)


def _stdev(data: list) -> float:
    if len(data) < 2:
        raise MathStatsError("Standard deviation requires at least 2 data points")
    return statistics.stdev(data)


def z_score_outliers(data: list, threshold: float = 3.0) -> list:
    """Identify outliers using Z-score method.

    Returns indices of elements whose absolute Z-score exceeds *threshold*.
    A threshold of 3.0 is a common choice (covers ~99.7% of normal distribution).
    """
    if not data:
        raise MathStatsError("Data list cannot be empty")
    if len(data) < 2:
        return []
    if threshold <= 0:
        raise MathStatsError(f"threshold must be positive, got {threshold}")
    mu = _mean(data)
    sigma = _stdev(data)
    if sigma == 0.0:
        return []
    outlier_indices = []
    for i, x in enumerate(data):
        if abs((x - mu) / sigma) > threshold:
            outlier_indices.append(i)
    return outlier_indices


def iqr_outliers(data: list, multiplier: float = 1.5) -> list:
    """Identify outliers using the interquartile range (IQR) / Tukey fence method.

    Returns indices of elements that fall below Q1 - multiplier*IQR or
    above Q3 + multiplier*IQR.  The standard multiplier is 1.5 (mild outliers);
    use 3.0 for extreme outliers.
    """
    if not data:
        raise MathStatsError("Data list cannot be empty")
    if multiplier <= 0:
        raise MathStatsError(f"multiplier must be positive, got {multiplier}")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    lower_half = sorted_data[:mid]
    upper_half = sorted_data[mid + (n % 2):]
    if not lower_half or not upper_half:
        return []
    q1 = _mean(lower_half)
    q3 = _mean(upper_half)
    iqr = q3 - q1
    lower_fence = q1 - multiplier * iqr
    upper_fence = q3 + multiplier * iqr
    return [i for i, x in enumerate(data) if x < lower_fence or x > upper_fence]


def _t_critical(n: int, alpha: float) -> float:
    """Approximate two-tailed t critical value using inverse normal approximation.

    For Grubbs test significance levels; approximation valid for n >= 6.
    """
    p = alpha / (2 * n)
    from math import log, sqrt, erfinv
    z = sqrt(2.0) * erfinv(1.0 - 2.0 * p)
    t = (z * math.sqrt(n - 2)) / math.sqrt(n - 1 - z * z)
    return t


def grubbs_test(data: list, alpha: float = 0.05) -> list:
    """Identify outliers using the Grubbs (ESD) test for a single outlier.

    Returns indices of detected outliers.  For samples smaller than 6 the
    test is unreliable; an empty list is returned in that case.

    Reference: Grubbs, F.E. (1969). Procedures for Detecting Outlying Observations.
    """
    if not data:
        raise MathStatsError("Data list cannot be empty")
    n = len(data)
    if n < 6:
        return []
    mu = _mean(data)
    sigma = _stdev(data)
    if sigma == 0.0:
        return []
    g_scores = [abs(x - mu) / sigma for x in data]
    g_max = max(g_scores)
    try:
        t_crit = _t_critical(n, alpha)
    except Exception:
        return []
    g_critical = ((n - 1) / math.sqrt(n)) * math.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))
    if g_max <= g_critical:
        return []
    return [i for i, g in enumerate(g_scores) if g == g_max]


def hampel_identifier(data: list, window: int = 3, threshold: float = 3.0) -> list:
    """Identify outliers using the Hampel identifier (sliding window MAD).

    For each data point, a local window of size *2*window + 1* centred on it is
    used to compute the median and the median absolute deviation (MAD).  Points
    more than *threshold* scaled MADs from the local median are flagged.

    Returns a list of indices of detected outliers.
    """
    if not data:
        raise MathStatsError("Data list cannot be empty")
    if window < 1:
        raise MathStatsError(f"window must be >= 1, got {window}")
    if threshold <= 0:
        raise MathStatsError(f"threshold must be positive, got {threshold}")
    _SCALE = 1.4826
    n = len(data)
    outliers = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbourhood = sorted(data[lo:hi])
        m = len(neighbourhood)
        if m == 0:
            continue
        median = neighbourhood[m // 2] if m % 2 else (neighbourhood[m // 2 - 1] + neighbourhood[m // 2]) / 2.0
        mad = _SCALE * statistics.median([abs(v - median) for v in neighbourhood])
        if mad == 0.0:
            continue
        if abs(data[i] - median) / mad > threshold:
            outliers.append(i)
    return outliers
