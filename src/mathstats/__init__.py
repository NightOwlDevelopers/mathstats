"""mathstats — Float matrix statistics, distributions, correlation, and regression."""

__version__ = "v1.0.0"

from .stats import (
    compute_stats,
    trimmed_mean,
    winsorize,
    geometric_mean,
    harmonic_mean,
    mode,
)
from .matrix import parse_input
from .distributions import normal_pdf, normal_cdf, binomial_pmf, poisson_pmf, exponential_pdf
from .correlation import pearson_r, spearman_r, covariance, correlation_matrix
from .regression import LinearRegression
from .outliers import z_score_outliers, iqr_outliers, grubbs_test, hampel_identifier

__all__ = [
    'compute_stats',
    'trimmed_mean',
    'winsorize',
    'geometric_mean',
    'harmonic_mean',
    'mode',
    'parse_input',
    'normal_pdf',
    'normal_cdf',
    'binomial_pmf',
    'poisson_pmf',
    'exponential_pdf',
    'pearson_r',
    'spearman_r',
    'covariance',
    'correlation_matrix',
    'LinearRegression',
    'z_score_outliers',
    'iqr_outliers',
    'grubbs_test',
    'hampel_identifier',
]
