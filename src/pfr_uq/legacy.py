"""Reproduction of the explicit scheme printed in the original manuscript (for audit only).

The original manuscript (``original/main.tex``, Eq. "scheme") stated the update

    C_i^{j+1} = (alpha - beta/2) C_{i-1}^j + (1 - 2 alpha) C_i^j + (alpha + beta/2) C_{i+1}^j + dt a r_i^j

for C_t + u C_x = D C_xx + a r.  Substituting the centred differences into
forward Euler actually gives (alpha + beta/2) on C_{i-1} and (alpha - beta/2)
on C_{i+1}; the printed coefficients correspond to velocity -u, so a pulse is
transported *upstream*.  This module implements both versions on the original
node-based grid with the original boundary treatment (Dirichlet inlet,
second-order one-sided zero-gradient outlet) so that the effect of the sign
error can be demonstrated.  It is not used for any production result.
"""
from __future__ import annotations

import numpy as np

__all__ = ["node_ftcs"]


def node_ftcs(u, D, k, L, N, dt, t_end, C_init, save_times=(), stencil="corrected", C_inlet=0.0):
    """Explicit FTCS on nodes x_i = i h, i = 0..N, as in the original manuscript.

    Parameters
    ----------
    stencil : {"corrected", "original"}
        "corrected": (alpha + beta/2) C_{i-1} + (1 - 2 alpha - kappa) C_i + (alpha - beta/2) C_{i+1}
        "original" : coefficients of C_{i-1} and C_{i+1} interchanged (as printed).
    k : float
        First-order consumption rate; the reaction term is -k C (consumption).

    Returns
    -------
    x : (N+1,) node coordinates
    snapshots : dict time -> (N+1,) profile
    """
    h = L / N
    x = np.arange(N + 1) * h
    alpha, beta, kappa = D * dt / h**2, u * dt / h, k * dt
    if stencil == "corrected":
        w_minus, w_plus = alpha + 0.5 * beta, alpha - 0.5 * beta
    elif stencil == "original":
        w_minus, w_plus = alpha - 0.5 * beta, alpha + 0.5 * beta
    else:
        raise ValueError("stencil must be 'corrected' or 'original'.")
    w_0 = 1.0 - 2.0 * alpha - kappa
    n_steps = int(round(t_end / dt))
    save_steps = {int(round(t / dt)): t for t in save_times}
    C = np.asarray(C_init(x) if callable(C_init) else C_init, dtype=float).copy()
    snapshots = {}
    if 0 in save_steps:
        snapshots[save_steps[0]] = C.copy()
    for n in range(1, n_steps + 1):
        Cn = C.copy()
        Cn[1:-1] = w_minus * C[:-2] + w_0 * C[1:-1] + w_plus * C[2:]
        Cn[0] = C_inlet
        Cn[-1] = (4.0 * Cn[-2] - Cn[-3]) / 3.0
        C = Cn
        if n in save_steps:
            snapshots[save_steps[n]] = C.copy()
        if not np.all(np.isfinite(C)):
            raise FloatingPointError("Legacy scheme produced non-finite values.")
    return x, snapshots
