"""Two-sided Wilson 95% interval; verified against Wolfram in the eval receipt.

The interval is descriptive for the declared units, not a claim that points
within an episode are independent. A zero denominator is NOT_DEFINED, not 0.
"""
from math import sqrt
from statistics import NormalDist


def wilson(k, n):
    if not (isinstance(k, int) and isinstance(n, int) and 0 <= k <= n):
        raise ValueError("Wilson requires integer 0 <= k <= n")
    if n == 0:
        return None
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    radius = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [0.0 if k == 0 else max(0.0, center - radius),
            1.0 if k == n else min(1.0, center + radius)]
