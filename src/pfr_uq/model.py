"""Physical model and analytical results for the axial-dispersion reactor.

Governing equation (first-order consumption, constant coefficients)::

    C_t + u C_x = D C_xx - k C,        0 < x < L,  t > 0,

with Danckwerts ("closed--closed") boundary conditions::

    u C(0,t) - D C_x(0,t) = u C_in(t)          (inlet: total-flux continuity)
    C_x(L,t) = 0                              (outlet)

Units used throughout the repository: metres, seconds, mol m^-3.

Dimensionless groups (reactor-length scaling, advective time scale tau = L/u)::

    Pe = u L / D        (axial Peclet number)
    Da = k L / u = k tau (first-order Damkohler number, Da_I)

All analytical expressions below were derived independently and are verified
symbolically in ``notebooks/01_model_verification.ipynb`` and numerically in the
test-suite.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import erf

__all__ = [
    "ReactorParameters",
    "BASELINE",
    "peclet",
    "damkohler",
    "danckwerts_outlet_ratio",
    "danckwerts_profile",
    "pfr_outlet_ratio",
    "cstr_outlet_ratio",
    "damkohler_from_outlet_ratio",
    "closed_vessel_rtd_variance",
    "gaussian_pulse_infinite",
    "gaussian_pulse_infinite_cell_average",
    "gaussian_pulse_noflux_images",
    "gaussian_inlet_pulse",
]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReactorParameters:
    """Physical parameters of the axial-dispersion reactor (SI units).

    Attributes
    ----------
    L : reactor length [m]
    u : mean axial (interstitial) velocity [m s^-1], u >= 0
    D : axial dispersion coefficient [m^2 s^-1], D >= 0
    k : first-order rate constant [s^-1], k >= 0 (consumption)
    C_in : feed concentration [mol m^-3], C_in >= 0
    """

    L: float = 1.0
    u: float = 0.1
    D: float = 0.01
    k: float = 0.1
    C_in: float = 1.0

    def __post_init__(self) -> None:
        for name in ("L", "u", "D", "k", "C_in"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"Parameter {name} must be finite, got {value!r}.")
        if self.L <= 0:
            raise ValueError(f"Reactor length must be positive, got L={self.L}.")
        if self.u < 0:
            raise ValueError("Velocity must be non-negative (flow in +x direction).")
        if self.D < 0:
            raise ValueError(f"Dispersion coefficient must be non-negative, got D={self.D}.")
        if self.k < 0:
            raise ValueError("Rate constant must be non-negative (first-order consumption).")
        if self.C_in < 0:
            raise ValueError("Feed concentration must be non-negative.")

    # Dimensionless groups and time scales -------------------------------
    @property
    def Pe(self) -> float:
        """Axial Peclet number u L / D (inf if D = 0)."""
        return peclet(self.u, self.L, self.D)

    @property
    def Da(self) -> float:
        """First-order Damkohler number k L / u (inf if u = 0 and k > 0)."""
        return damkohler(self.k, self.L, self.u)

    @property
    def tau_adv(self) -> float:
        """Advective (space) time L/u [s]."""
        return self.L / self.u if self.u > 0 else np.inf

    @property
    def tau_disp(self) -> float:
        """Dispersive time L^2/D [s]."""
        return self.L**2 / self.D if self.D > 0 else np.inf

    @property
    def tau_rxn(self) -> float:
        """Reaction time 1/k [s]."""
        return 1.0 / self.k if self.k > 0 else np.inf

    def with_(self, **changes) -> "ReactorParameters":
        """Return a copy with selected parameters changed."""
        return replace(self, **changes)


#: Illustrative baseline used throughout the study (Pe = 10, Da = 1).
BASELINE = ReactorParameters(L=1.0, u=0.1, D=0.01, k=0.1, C_in=1.0)


def peclet(u, L, D):
    """Axial Peclet number Pe = u L / D (vectorised; inf where D == 0)."""
    u, L, D = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (u, L, D)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(D > 0, u * L / np.where(D > 0, D, 1.0), np.inf)
    return out[()] if out.ndim == 0 else out


def damkohler(k, L, u):
    """First-order Damkohler number Da = k L / u (vectorised)."""
    k, L, u = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (k, L, u)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(u > 0, k * L / np.where(u > 0, u, 1.0), np.where(k > 0, np.inf, 0.0))
    return out[()] if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Steady state with Danckwerts boundary conditions
# ---------------------------------------------------------------------------
def _steady_coefficients(Pe, Da):
    """Coefficients of the scaled steady solution.

    phi(xi) = a exp(m_+ (xi - 1)) + b exp(m_- xi),  m_pm = Pe (1 pm q) / 2,
    q = sqrt(1 + 4 Da / Pe).  Writing the growing exponential relative to the
    outlet keeps every exponent non-positive and avoids overflow at large Pe.
    """
    Pe = np.asarray(Pe, dtype=float)
    Da = np.asarray(Da, dtype=float)
    if np.any(Pe <= 0) or np.any(Da < 0):
        raise ValueError("Require Pe > 0 and Da >= 0.")
    q = np.sqrt(1.0 + 4.0 * Da / Pe)
    m_plus = 0.5 * Pe * (1.0 + q)
    m_minus = 0.5 * Pe * (1.0 - q)  # <= 0
    e_minus = np.exp(m_minus)  # exp(m_-), in (0, 1]
    e_plus_rev = np.exp(-m_plus)  # exp(-m_+), in (0, 1)
    # Outlet: phi'(1) = a m_+ + b m_- exp(m_-) = 0
    # Inlet : phi(0) - phi'(0)/Pe = a exp(-m_+)(1-q)/2 + b (1+q)/2 = 1
    # Eliminate a = -b m_- exp(m_-) / m_+.
    ratio = m_minus * e_minus / m_plus  # <= 0
    b = 1.0 / (0.5 * (1.0 + q) - ratio * e_plus_rev * 0.5 * (1.0 - q))
    a = -ratio * b
    return a, b, m_plus, m_minus, q


def danckwerts_profile(xi, Pe, Da):
    """Steady dimensionless profile phi(xi) = C(x)/C_in, xi = x/L.

    Solves  phi''/Pe - phi' - Da phi = 0  with  phi(0) - phi'(0)/Pe = 1 and
    phi'(1) = 0 (Danckwerts).  Scalars Pe, Da; xi array-like in [0, 1].
    """
    a, b, m_plus, m_minus, _ = _steady_coefficients(Pe, Da)
    xi = np.asarray(xi, dtype=float)
    return a * np.exp(m_plus * (xi - 1.0)) + b * np.exp(m_minus * xi)


def danckwerts_outlet_ratio(Pe, Da):
    """Outlet-to-feed ratio C(L)/C_in at steady state (Wehner--Wilhelm).

    Equivalent to 4 q e^{Pe/2} / [(1+q)^2 e^{q Pe/2} - (1-q)^2 e^{-q Pe/2}]
    but evaluated in an overflow-free form.  Vectorised in Pe and Da.
    Pe = inf returns the ideal-PFR limit exp(-Da).
    """
    Pe, Da = np.broadcast_arrays(np.asarray(Pe, dtype=float), np.asarray(Da, dtype=float))
    out = np.empty(Pe.shape, dtype=float)
    finite = np.isfinite(Pe)
    out[~finite] = np.exp(-Da[~finite])
    if np.any(finite):
        Pf, Df = Pe[finite], Da[finite]
        q = np.sqrt(1.0 + 4.0 * Df / Pf)
        num = 4.0 * q * np.exp(0.5 * Pf * (1.0 - q))
        den = (1.0 + q) ** 2 - (1.0 - q) ** 2 * np.exp(-q * Pf)
        out[finite] = num / den
    return out[()] if out.ndim == 0 else out


def pfr_outlet_ratio(Da):
    """Ideal plug-flow limit (Pe -> inf): C(L)/C_in = exp(-Da)."""
    return np.exp(-np.asarray(Da, dtype=float))


def cstr_outlet_ratio(Da):
    """Perfect-mixing limit (Pe -> 0): C(L)/C_in = 1/(1 + Da)."""
    return 1.0 / (1.0 + np.asarray(Da, dtype=float))


def damkohler_from_outlet_ratio(ratio, Pe, tol=1e-13, max_iter=200):
    """Invert ``danckwerts_outlet_ratio`` for Da at fixed Pe (vectorised bisection).

    The outlet ratio is strictly decreasing in Da on (0, inf) with range (0, 1),
    so a unique root exists for ratio in (0, 1).  Entries outside that interval
    (which can arise from noisy measurements) are returned as NaN.
    """
    ratio = np.asarray(ratio, dtype=float)
    Pe = np.broadcast_to(np.asarray(Pe, dtype=float), ratio.shape)
    valid = (ratio > 0) & (ratio < 1) & np.isfinite(ratio)
    lo = np.full(ratio.shape, 0.0)
    hi = np.full(ratio.shape, 1.0)
    # expand upper bracket until f(hi) < ratio
    for _ in range(80):
        need = valid & (danckwerts_outlet_ratio(Pe, hi) > ratio)
        if not np.any(need):
            break
        hi = np.where(need, 2.0 * hi, hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = danckwerts_outlet_ratio(Pe, mid)
        go_right = f_mid > ratio  # ratio decreasing in Da -> root is to the right
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
        width = np.where(valid, (hi - lo) / np.maximum(1.0, hi), 0.0)
        if np.max(width, initial=0.0) < tol:
            break
    out = np.where(valid, 0.5 * (lo + hi), np.nan)
    return out[()] if out.ndim == 0 else out


def closed_vessel_rtd_variance(Pe):
    """Dimensionless RTD variance sigma_theta^2 for the closed--closed dispersion vessel.

    sigma^2 / tau^2 = 2/Pe - (2/Pe^2) (1 - exp(-Pe)), with mean residence time tau = L/u.
    """
    Pe = np.asarray(Pe, dtype=float)
    return 2.0 / Pe - 2.0 / Pe**2 * (1.0 - np.exp(-Pe))


# ---------------------------------------------------------------------------
# Infinite-domain / image solutions used as verification benchmarks
# ---------------------------------------------------------------------------
def gaussian_pulse_infinite(x, t, x0, s0, u=0.0, D=0.0, k=0.0, mass=1.0):
    """Infinite-domain solution for a Gaussian initial pulse.

    C(x,0) = mass * N(x; x0, s0^2) evolves under C_t + u C_x = D C_xx - k C on
    the whole real line to a Gaussian with mean x0 + u t, variance s0^2 + 2 D t
    and mass ``mass * exp(-k t)``.  This is NOT a finite-reactor solution; it is
    used only while boundary effects are negligible (checked explicitly).
    """
    var = s0**2 + 2.0 * D * t
    x = np.asarray(x, dtype=float)
    return mass * np.exp(-k * t) / np.sqrt(2.0 * np.pi * var) * np.exp(-((x - x0 - u * t) ** 2) / (2.0 * var))


def gaussian_pulse_infinite_cell_average(x_faces, t, x0, s0, u=0.0, D=0.0, k=0.0, mass=1.0):
    """Exact cell averages of :func:`gaussian_pulse_infinite` over cells [x_faces[i], x_faces[i+1]]."""
    var = s0**2 + 2.0 * D * t
    sd = np.sqrt(var)
    xf = np.asarray(x_faces, dtype=float)
    cdf = 0.5 * (1.0 + erf((xf - x0 - u * t) / (np.sqrt(2.0) * sd)))
    return mass * np.exp(-k * t) * np.diff(cdf) / np.diff(xf)


def gaussian_pulse_noflux_images(x, t, x0, s0, D, L, n_images=6):
    """Exact solution of C_t = D C_xx on [0, L] with C_x = 0 at both ends.

    Gaussian initial pulse (unit mass); method of images with 2(2 n_images + 1)
    image sources.  Truncation error is far below double precision for the
    parameters used in this study.
    """
    var = s0**2 + 2.0 * D * t
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x)
    for n in range(-n_images, n_images + 1):
        for centre in (x0 + 2 * n * L, -x0 + 2 * n * L):
            total += np.exp(-((x - centre) ** 2) / (2.0 * var))
    return total / np.sqrt(2.0 * np.pi * var)


def gaussian_inlet_pulse(t, amplitude=1.0, t_center=3.0, width=0.5):
    """Smooth tracer/reactant injection C_in(t) = A exp(-(t - t_c)^2 / (2 w^2))."""
    t = np.asarray(t, dtype=float)
    return amplitude * np.exp(-((t - t_center) ** 2) / (2.0 * width**2))
