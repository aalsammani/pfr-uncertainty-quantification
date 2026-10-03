"""Reactor quantities of interest (QoIs) computed with the finite-volume solver.

Two operating scenarios are used in the study.

Steady operation (constant feed C_in):
    C_out   steady exit concentration [mol m^-3]
    X       conversion 1 - C_out / C_in [-]

Pulse injection (feed C_in(t) = C_in exp(-(t - t_c)^2 / (2 w^2)), reactor initially empty):
    peak    maximum exit concentration max_t C(L, t) [mol m^-3]
    t_mean  mean arrival time  int t C(L,t) dt / int C(L,t) dt  [s]
    exposure  int_0^T C(L, t) dt  [mol s m^-3]

The exposure has the exact infinite-time value  C_in w sqrt(2 pi) G(Pe, Da)
(G = steady outlet ratio, the transfer function at zero frequency), up to the
negligible part of the Gaussian feed before t = 0; it is used as an analytical
check of the transient solver inside the uncertainty study.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics, model, solvers

__all__ = ["PulseSettings", "PULSE", "steady_qois", "pulse_qois", "QOI_NAMES", "QOI_LABELS", "QOI_UNITS"]


@dataclass(frozen=True)
class PulseSettings:
    """Numerical and operating settings of the pulse-injection scenario."""

    t_center: float = 5.0   # s, centre of the injected pulse
    width: float = 1.0      # s, standard deviation of the injected pulse
    t_end: float = 80.0     # s, simulated horizon
    dt: float = 0.1         # s, Crank--Nicolson step
    N: int = 100            # cells
    L: float = 1.0          # m


PULSE = PulseSettings()

QOI_NAMES = ("C_out", "X", "peak", "t_mean", "exposure")
QOI_LABELS = {
    "C_out": r"steady exit concentration $C_\mathrm{out}$",
    "X": r"steady conversion $X$",
    "peak": r"peak exit concentration $C_\mathrm{max}$",
    "t_mean": r"mean arrival time $\bar t$",
    "exposure": r"exit exposure $E$",
}
QOI_UNITS = {"C_out": "mol m^-3", "X": "-", "peak": "mol m^-3", "t_mean": "s", "exposure": "mol s m^-3"}


def steady_qois(u, D, k, C_in, N: int = PULSE.N, L: float = PULSE.L) -> dict:
    """Steady exit concentration and conversion from the finite-volume steady solve (batch)."""
    op = solvers.FVOperator(u, D, k, L, N)
    C_in = np.broadcast_to(np.asarray(C_in, dtype=float), (op.B,))
    C = solvers.solve_steady(op, C_in)
    C_out = C[:, -1]
    return {"C_out": C_out, "X": 1.0 - C_out / C_in}


def pulse_qois(u, D, k, C_in, settings: PulseSettings = PULSE, return_curves: bool = False) -> dict:
    """Pulse-response QoIs from one batched Crank--Nicolson simulation."""
    op = solvers.FVOperator(u, D, k, settings.L, settings.N)
    amp = np.broadcast_to(np.asarray(C_in, dtype=float), (op.B,)).copy()

    def inlet(t):
        return amp * model.gaussian_inlet_pulse(t, 1.0, settings.t_center, settings.width)

    res = solvers.integrate(op, 0.0, settings.t_end, settings.dt, inlet=inlet, record_mass_balance=False)
    peak, _ = metrics.peak_quadratic(res.t, res.outlet, axis=0)
    m0, mean, _ = metrics.curve_moments(res.t, res.outlet, axis=0)
    out = {"peak": peak, "t_mean": mean, "exposure": m0, "min_value": res.min_value}
    if return_curves:
        out["t"] = res.t
        out["outlet"] = res.outlet
    return out


def exposure_exact(u, D, k, C_in, settings: PulseSettings = PULSE, L: float | None = None):
    """Infinite-horizon exposure G(Pe, Da) * int_0^inf C_in(t) dt for the pulse fed from t = 0.

    int_0^inf C_in(t) dt = C_in w sqrt(2 pi) Phi(t_c / w), Phi the standard normal CDF.
    """
    from scipy.stats import norm

    L = settings.L if L is None else L
    Pe = model.peclet(u, L, D)
    Da = model.damkohler(k, L, u)
    fed = np.asarray(C_in) * settings.width * np.sqrt(2 * np.pi) * norm.cdf(settings.t_center / settings.width)
    return fed * model.danckwerts_outlet_ratio(Pe, Da)
