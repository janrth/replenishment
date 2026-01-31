"""Service level utilities and conversions."""

from __future__ import annotations

import math

SERVICE_LEVEL_MODE_FACTOR = "factor"
SERVICE_LEVEL_MODE_SERVICE_LEVEL = "service_level"
SERVICE_LEVEL_MODE_STOCKOUT_PROBABILITY = "stockout_probability"

_MODE_ALIASES = {
    SERVICE_LEVEL_MODE_FACTOR: SERVICE_LEVEL_MODE_FACTOR,
    "multiplier": SERVICE_LEVEL_MODE_FACTOR,
    "z": SERVICE_LEVEL_MODE_FACTOR,
    SERVICE_LEVEL_MODE_SERVICE_LEVEL: SERVICE_LEVEL_MODE_SERVICE_LEVEL,
    "cycle_service_level": SERVICE_LEVEL_MODE_SERVICE_LEVEL,
    "csl": SERVICE_LEVEL_MODE_SERVICE_LEVEL,
    SERVICE_LEVEL_MODE_STOCKOUT_PROBABILITY: SERVICE_LEVEL_MODE_STOCKOUT_PROBABILITY,
    "stockout_prob": SERVICE_LEVEL_MODE_STOCKOUT_PROBABILITY,
    "stockout": SERVICE_LEVEL_MODE_STOCKOUT_PROBABILITY,
}


def normalize_service_level_mode(mode: str | None) -> str:
    if mode is None:
        return SERVICE_LEVEL_MODE_FACTOR
    normalized = mode.strip().lower()
    if normalized in _MODE_ALIASES:
        return _MODE_ALIASES[normalized]
    raise ValueError(
        "service_level_mode must be one of: "
        f"{', '.join(sorted(_MODE_ALIASES))}."
    )


def normal_quantile(p: float) -> float:
    """Approximate the standard normal quantile (inverse CDF)."""
    if not 0.0 < p < 1.0:
        raise ValueError("Normal quantile requires 0 < p < 1.")

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
            + c[5]
        ) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
        ) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r
            + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
        + c[5]
    ) / (
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
    )


def service_level_multiplier(value: float, mode: str | None = None) -> float:
    normalized = normalize_service_level_mode(mode)
    if normalized == SERVICE_LEVEL_MODE_FACTOR:
        if value < 0:
            raise ValueError("Service level factors must be non-negative.")
        return float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("Service level probabilities must be between 0 and 1.")
    if normalized == SERVICE_LEVEL_MODE_SERVICE_LEVEL:
        return normal_quantile(float(value))
    if normalized == SERVICE_LEVEL_MODE_STOCKOUT_PROBABILITY:
        return normal_quantile(1.0 - float(value))
    raise ValueError(
        "service_level_mode must be one of: "
        f"{', '.join(sorted(_MODE_ALIASES))}."
    )
