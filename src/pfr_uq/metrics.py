"""Error norms, convergence orders, cell averages and curve moments."""
from __future__ import annotations

import numpy as np

__all__ = [
    "norm_l1",
    "norm_l2",
    "norm_linf",
    "error_norms",
    "observed_orders",
    "fitted_order",
    "cell_average",
    "curve_moments",
    "peak_quadratic",
]


def norm_l1(e, h):
    """Discrete L1 norm h * sum |e|."""
    return float(h * np.sum(np.abs(e)))


def norm_l2(e, h):
    """Discrete L2 norm sqrt(h * sum e^2) (approximates the continuous L2 norm)."""
    return float(np.sqrt(h * np.sum(np.asarray(e) ** 2)))


def norm_linf(e):
    """Maximum norm."""
    return float(np.max(np.abs(e)))


def error_norms(numerical, exact, h) -> dict:
    """Return the L1, L2 and Linf norms of numerical - exact."""
    e = np.asarray(numerical) - np.asarray(exact)
    return {"L1": norm_l1(e, h), "L2": norm_l2(e, h), "Linf": norm_linf(e)}


def observed_orders(h, err):
    """Pairwise observed orders p_i = log(E_i/E_{i+1}) / log(h_i/h_{i+1})."""
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


def fitted_order(h, err):
    """Least-squares slope of log(err) against log(h) and its standard error.

    Returns (slope, standard_error, intercept).  The standard error is the
    usual OLS standard error of the slope; with few points it is only a rough
    indication of the spread and is reported as such.
    """
    lh = np.log(np.asarray(h, dtype=float))
    le = np.log(np.asarray(err, dtype=float))
    X = np.vstack([lh, np.ones_like(lh)]).T
    coef, res, *_ = np.linalg.lstsq(X, le, rcond=None)
    n = lh.size
    if n > 2:
        resid = le - X @ coef
        s2 = resid @ resid / (n - 2)
        se = float(np.sqrt(s2 / np.sum((lh - lh.mean()) ** 2)))
    else:
        se = float("nan")
    return float(coef[0]), se, float(coef[1])


def cell_average(func, faces, n_gauss: int = 6):
    """Cell averages of ``func(x)`` over cells [faces[i], faces[i+1]] by Gauss--Legendre quadrature.

    With n_gauss = 6 the quadrature is exact for polynomials of degree 11, so
    for smooth functions the quadrature error is negligible relative to the
    discretisation errors studied here.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    faces = np.asarray(faces, dtype=float)
    a, b = faces[:-1, None], faces[1:, None]
    x = 0.5 * (b - a) * nodes[None, :] + 0.5 * (a + b)
    vals = func(x)
    return 0.5 * np.sum(vals * weights[None, :], axis=1)


def curve_moments(t, c, axis=0):
    """Zeroth moment, mean and variance of a curve c(t) (trapezoidal rule).

    m0 = int c dt,  mean = int t c dt / m0,  var = int (t - mean)^2 c dt / m0.
    ``c`` may be 2-D with time along ``axis``.
    """
    t = np.asarray(t, dtype=float)
    c = np.moveaxis(np.asarray(c, dtype=float), axis, 0)
    tt = t.reshape((-1,) + (1,) * (c.ndim - 1))
    m0 = np.trapezoid(c, t, axis=0)
    mean = np.trapezoid(tt * c, t, axis=0) / m0
    var = np.trapezoid((tt - mean) ** 2 * c, t, axis=0) / m0
    return m0, mean, var


def peak_quadratic(t, c, axis=0):
    """Peak value and time of sampled curve(s) using a three-point parabola through the maximum.

    Returns (c_max, t_max).  For a smooth curve sampled with spacing dt the
    error is O(dt^3) in c_max (parabola through a smooth maximum).
    """
    t = np.asarray(t, dtype=float)
    c = np.moveaxis(np.asarray(c, dtype=float), axis, 0)
    flat = c.reshape(c.shape[0], -1)
    i = np.clip(np.argmax(flat, axis=0), 1, flat.shape[0] - 2)
    cols = np.arange(flat.shape[1])
    y0, y1, y2 = flat[i - 1, cols], flat[i, cols], flat[i + 1, cols]
    dt = t[1] - t[0]
    denom = y0 - 2.0 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(denom != 0, 0.5 * (y0 - y2) / denom, 0.0)
    delta = np.clip(delta, -1.0, 1.0)
    c_max = y1 - 0.25 * (y0 - y2) * delta
    t_max = t[i] + delta * dt
    shape = c.shape[1:]
    return c_max.reshape(shape), t_max.reshape(shape)
