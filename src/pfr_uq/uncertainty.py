"""Uncertainty-quantification utilities.

Separation of error sources used throughout the study
-----------------------------------------------------
* Parametric (input) uncertainty: u, D, k, C_in are random variables with
  stated (illustrative, *assumed*) distributions and are propagated through
  the PDE model (forward propagation).
* Measurement error: an observation is  y = C_true + b + eps  with a fixed
  (unknown) systematic offset b and independent random noise eps ~ N(0, sigma^2).
  It affects quantities *inferred* from data (inverse propagation).
* Numerical (discretisation) error: a deterministic bias of the solver,
  quantified separately by grid/time-step refinement.
* Sampling (integration) error: the error of a Monte Carlo / quasi-Monte Carlo
  estimate of an expectation, quantified with independent replicates.

Monte Carlo (MC) and randomized quasi-Monte Carlo (RQMC) are compared as
*integration rules for the same probability distribution*: both draw points
in the unit cube [0,1)^d that are mapped through the same inverse-CDF
transform, so any difference in error is due to the point set only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm, qmc

__all__ = [
    "LogNormal",
    "InputModel",
    "spawn_generators",
    "unit_cube_sample",
    "nested_estimates",
    "sobol_indices",
    "bootstrap_ci",
    "wilson_interval",
    "measure",
]


@dataclass(frozen=True)
class LogNormal:
    """Log-normal random variable parameterised by its median and coefficient of variation.

    ln X ~ N(ln median, s^2) with s^2 = ln(1 + cv^2), so that sd(X)/E(X) = cv exactly.
    """

    median: float
    cv: float

    def __post_init__(self) -> None:
        if not (self.median > 0 and self.cv >= 0):
            raise ValueError("LogNormal requires median > 0 and cv >= 0.")

    @property
    def s(self) -> float:
        return float(np.sqrt(np.log1p(self.cv**2)))

    @property
    def mean(self) -> float:
        return float(self.median * np.exp(0.5 * self.s**2))

    @property
    def sd(self) -> float:
        return self.mean * self.cv

    def ppf(self, U):
        """Inverse CDF; U in (0,1)."""
        return self.median * np.exp(self.s * norm.ppf(U))

    def quantile_interval(self, level=0.95):
        z = norm.ppf(0.5 + level / 2)
        return self.median * np.exp(-z * self.s), self.median * np.exp(z * self.s)


@dataclass(frozen=True)
class InputModel:
    """Independent input variables, in a fixed order (defines the unit-cube coordinates)."""

    names: tuple
    marginals: tuple

    @classmethod
    def from_dict(cls, spec: dict) -> "InputModel":
        return cls(tuple(spec.keys()), tuple(spec.values()))

    @property
    def dim(self) -> int:
        return len(self.names)

    def transform(self, U) -> dict:
        """Map points U in [0,1)^d (shape (n, d)) to a dict of physical samples."""
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[1] != self.dim:
            raise ValueError(f"Expected samples of shape (n, {self.dim}).")
        # Guard against U == 0 exactly (possible for unscrambled nets) -> -inf.
        U = np.clip(U, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        return {name: marg.ppf(U[:, j]) for j, (name, marg) in enumerate(zip(self.names, self.marginals))}


def spawn_generators(seed: int, n: int) -> list:
    """n statistically independent NumPy generators derived from one documented seed."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def unit_cube_sample(n: int, d: int, method: str, rng: np.random.Generator) -> np.ndarray:
    """Draw n points in [0,1)^d.

    method = "mc"   : independent uniform pseudo-random points (PCG64).
    method = "rqmc" : Sobol' points with Owen-type linear matrix scrambling and
                      digital shift (scipy.stats.qmc.Sobol, scramble=True).
                      n must be a power of two so that the point set is a
                      complete (t, m, d)-net (balance properties hold).
    """
    if method == "mc":
        return rng.random((n, d))
    if method == "rqmc":
        if not _is_power_of_two(n):
            raise ValueError("RQMC sample sizes must be powers of two.")
        try:
            engine = qmc.Sobol(d=d, scramble=True, rng=rng)
        except TypeError:  # SciPy < 1.15 uses `seed`
            engine = qmc.Sobol(d=d, scramble=True, seed=rng)
        return engine.random_base2(int(np.log2(n)))
    raise ValueError("method must be 'mc' or 'rqmc'.")


def nested_estimates(values: np.ndarray, sizes: Sequence[int], stat: Callable = np.mean) -> np.ndarray:
    """Evaluate ``stat`` on the first n rows of ``values`` for each n in ``sizes``.

    For Sobol' sequences the first 2^m points of a 2^M-point sequence form a
    net themselves, so nested prefixes are valid RQMC point sets.
    Returns an array of shape (len(sizes),) + values.shape[1:].
    """
    return np.stack([stat(values[:n], axis=0) for n in sizes])


def sobol_indices(f: Callable, model: InputModel, n: int, method: str, rng: np.random.Generator):
    """First-order and total Sobol' indices with the Saltelli (2010) / Jansen estimators.

    Uses two independent n x d matrices A, B obtained from one 2d-dimensional
    point set, and the d hybrid matrices A_B^(i) (column i taken from B).
    f maps a dict of inputs to an array of shape (n,) or (n, q).
    Cost: n (d + 2) model evaluations.

        S_i  = mean( f(B) * (f(A_B^i) - f(A)) ) / V          (Saltelli 2010)
        ST_i = mean( (f(A) - f(A_B^i))^2 ) / (2 V)            (Jansen 1999)

    Returns dict with arrays S (d,[q]), ST (d,[q]) and V.
    """
    d = model.dim
    U = unit_cube_sample(n, 2 * d, method, rng)
    A, Bm = U[:, :d], U[:, d:]
    fA = np.asarray(f(model.transform(A)))
    fB = np.asarray(f(model.transform(Bm)))
    V = np.var(np.concatenate([fA, fB], axis=0), axis=0, ddof=1)
    S, ST = [], []
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = Bm[:, i]
        fABi = np.asarray(f(model.transform(ABi)))
        S.append(np.mean(fB * (fABi - fA), axis=0) / V)
        ST.append(0.5 * np.mean((fA - fABi) ** 2, axis=0) / V)
    return {"S": np.array(S), "ST": np.array(ST), "V": V}


def bootstrap_ci(data, statistic: Callable, n_boot: int, rng: np.random.Generator, level=0.95, axis=0):
    """Percentile bootstrap confidence interval for ``statistic(data)`` resampling along ``axis``."""
    data = np.asarray(data)
    n = data.shape[axis]
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        reps.append(statistic(np.take(data, idx, axis=axis)))
    reps = np.asarray(reps)
    lo, hi = np.quantile(reps, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return lo, hi


def wilson_interval(successes: int, n: int, level=0.95):
    """Wilson score interval for a binomial proportion."""
    z = norm.ppf(0.5 + level / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def measure(true_values, sigma: float, bias: float, rng: np.random.Generator, size=None):
    """Synthetic sensor readings y = C_true + b + eps, eps ~ N(0, sigma^2) i.i.d.

    ``bias`` is a fixed systematic offset (identical for every reading of the
    instrument); it is *not* random from reading to reading.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    true_values = np.asarray(true_values, dtype=float)
    shape = true_values.shape if size is None else size
    return true_values + bias + sigma * rng.standard_normal(shape)
