"""Probability distribution functions — pure Python implementations."""
import math
from .errors import MathStatsError


def _validate_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise MathStatsError(f"{name} must be finite, got {value}")


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Probability density function of the normal (Gaussian) distribution.

    P(X = x) = 1 / (sigma * sqrt(2*pi)) * exp(-0.5 * ((x - mu) / sigma)^2)
    """
    if sigma <= 0:
        raise MathStatsError(f"sigma must be positive, got {sigma}")
    _validate_finite(x, 'x')
    _validate_finite(mu, 'mu')
    _validate_finite(sigma, 'sigma')
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coef * math.exp(exponent)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Cumulative distribution function of the normal distribution.

    Uses the standard error function: CDF(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2)))).
    """
    if sigma <= 0:
        raise MathStatsError(f"sigma must be positive, got {sigma}")
    _validate_finite(x, 'x')
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Probability mass function of the binomial distribution B(n, p) at k.

    P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)
    """
    if not (0.0 <= p <= 1.0):
        raise MathStatsError(f"p must be in [0, 1], got {p}")
    if n < 0:
        raise MathStatsError(f"n must be non-negative, got {n}")
    if k < 0 or k > n:
        return 0.0
    binom_coef = math.comb(n, k)
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    return binom_coef * (p ** k) * ((1.0 - p) ** (n - k))


def poisson_pmf(k: int, lam: float) -> float:
    """Probability mass function of the Poisson distribution Pois(lam) at k.

    P(X = k) = exp(-lambda) * lambda^k / k!
    """
    if lam < 0:
        raise MathStatsError(f"lambda must be non-negative, got {lam}")
    if k < 0:
        return 0.0
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def exponential_pdf(x: float, lam: float) -> float:
    """Probability density function of the exponential distribution Exp(lam) at x.

    f(x; lambda) = lambda * exp(-lambda * x)  for x >= 0, else 0.
    """
    if lam <= 0:
        raise MathStatsError(f"lambda must be positive, got {lam}")
    if x < 0:
        return 0.0
    return lam * math.exp(-lam * x)


def exponential_cdf(x: float, lam: float) -> float:
    """Cumulative distribution function of the exponential distribution."""
    if lam <= 0:
        raise MathStatsError(f"lambda must be positive, got {lam}")
    if x < 0:
        return 0.0
    return 1.0 - math.exp(-lam * x)


def uniform_pdf(x: float, a: float, b: float) -> float:
    """PDF of the continuous uniform distribution U(a, b)."""
    if a >= b:
        raise MathStatsError(f"a must be less than b, got a={a}, b={b}")
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def uniform_cdf(x: float, a: float, b: float) -> float:
    """CDF of the continuous uniform distribution U(a, b)."""
    if a >= b:
        raise MathStatsError(f"a must be less than b, got a={a}, b={b}")
    if x < a:
        return 0.0
    if x > b:
        return 1.0
    return (x - a) / (b - a)
