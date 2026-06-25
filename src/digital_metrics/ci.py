import math

import scipy.stats as st


def calculate_confidence_interval(
    positives: float,
    total: float,
    method: str = "wilson",
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Wilson confidence interval for a proportion.

    Args:
        positives: Number of positive outcomes (e.g. TP).
        total: Total number of trials.
        method: Only "wilson" is supported; other values raise ValueError.
        confidence_level: Desired confidence level, e.g. 0.95.

    Returns:
        (lower_bound, upper_bound) clamped to [0, 1].
    """
    if total == 0:
        return 0.0, 0.0

    if method != "wilson":
        raise ValueError(f"Unknown method: {method!r}. Only 'wilson' is supported.")

    z = st.norm.ppf(1 - (1 - confidence_level) / 2)
    p = positives / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    lower = max(0.0, min(1.0, centre - z * spread))
    upper = max(0.0, min(1.0, centre + z * spread))
    return lower, upper
