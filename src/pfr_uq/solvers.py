r"""Finite-volume discretisation and time integration of the ADR reactor model.

Spatial discretisation
----------------------
The reactor [0, L] is divided into N cells of width h = L/N with centres
x_j = (j + 1/2) h, j = 0..N-1.  Integrating the conservation law over cell j
gives the exact balance

    h dC_j/dt = F_{j} - F_{j+1} - k h C_j + h s_j,

where F_f is the total (advective + dispersive) flux through face f at
x_f = f h.  Face fluxes are approximated as follows.

* Inlet face (f = 0): the Danckwerts condition states that the total flux just
  inside the reactor equals the feed flux, so F_0 = u C_in(t) *exactly*.
* Interior faces (1 <= f <= N-1):
      central : F_f = u (C_{f-1} + C_f)/2 - D (C_f - C_{f-1})/h   (2nd order)
      upwind  : F_f = u C_{f-1}           - D (C_f - C_{f-1})/h   (1st order, u >= 0)
* Outlet face (f = N): dispersive flux vanishes (C_x(L) = 0) and the advective
  flux is u C_{N-1}.  Because C_x(L) = 0, C(L) - C(x_{N-1}) = O(h^2).

The semi-discrete system is dC/dt = A C + g(t) with g_0 = (u/h) C_in(t) (+ source).
It conserves the discrete mass M_h = h sum_j C_j exactly:

    dM_h/dt = u C_in - u C_{N-1} - k M_h + h sum_j s_j.

For u >= 0, D >= 0, k >= 0 the symmetric part of A is negative semidefinite
(discrete energy estimate), so the theta-method with theta >= 1/2 is
unconditionally stable in the discrete L2 norm (see ``notebooks/01``).

Time integration
----------------
theta-method:  (I - theta dt A) C^{n+1} = (I + (1-theta) dt A) C^n
                                          + dt [theta g^{n+1} + (1-theta) g^n]
theta = 1/2 Crank--Nicolson (default), theta = 1 backward Euler, theta = 0
forward Euler (explicit; for the FTCS analysis only, stability is checked).
Optional Rannacher start-up replaces the first CN steps by backward-Euler
half-steps to damp non-smooth initial data.

Batches
-------
u, D and k may be arrays of length B; the B independent problems are solved
simultaneously as one block-diagonal sparse system (used for Monte Carlo).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

__all__ = [
    "Grid",
    "FVOperator",
    "TransientResult",
    "integrate",
    "solve_steady",
    "ftcs_amplification",
    "ftcs_max_amplification",
    "ftcs_strictly_stable",
    "ftcs_lax_richtmyer_stable",
    "ftcs_positive",
    "explicit_numbers",
]


# ---------------------------------------------------------------------------
# Grid and operator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    """Uniform cell-centred grid on [0, L] with N cells."""

    L: float
    N: int

    def __post_init__(self) -> None:
        if not (np.isfinite(self.L) and self.L > 0):
            raise ValueError("Grid length L must be positive and finite.")
        if int(self.N) != self.N or self.N < 3:
            raise ValueError("Grid needs an integer number of cells N >= 3.")

    @property
    def h(self) -> float:
        return self.L / self.N

    @property
    def centers(self) -> np.ndarray:
        return (np.arange(self.N) + 0.5) * self.h

    @property
    def faces(self) -> np.ndarray:
        return np.arange(self.N + 1) * self.h


def _as_batch(value, name: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative (got min {arr.min():g}).")
    return arr


class FVOperator:
    """Semi-discrete finite-volume operator for a batch of B parameter sets.

    Parameters
    ----------
    u, D, k : float or array of shape (B,)
        Velocity [m/s], dispersion coefficient [m^2/s], rate constant [1/s].
    L : float
        Reactor length [m] (common to the batch).
    N : int
        Number of cells.
    advection : {"central", "upwind"}
        Interior advective-flux approximation.
    """

    def __init__(self, u, D, k, L: float, N: int, advection: str = "central"):
        if advection not in ("central", "upwind"):
            raise ValueError("advection must be 'central' or 'upwind'.")
        self.grid = Grid(L, int(N))
        u, D, k = np.broadcast_arrays(_as_batch(u, "u"), _as_batch(D, "D"), _as_batch(k, "k"))
        self.u, self.D, self.k = u.copy(), D.copy(), k.copy()
        self.B = u.size
        self.N = int(N)
        self.advection = advection
        self.lower, self.diag, self.upper = self._coefficients()
        self.A = self._assemble()
        self.inlet_gain = self.u / self.grid.h  # g_0 = (u/h) C_in

    # -- assembly ---------------------------------------------------------
    def _coefficients(self):
        """Return (lower, diag, upper) arrays of shape (B, N).

        Row j of the operator: dC_j/dt = lower_j C_{j-1} + diag_j C_j + upper_j C_{j+1}.
        Built face by face from the flux weights, so that the discrete
        conservation form is respected by construction.
        """
        h, N, B = self.grid.h, self.N, self.B
        u = self.u[:, None]
        D = self.D[:, None]
        k = self.k[:, None]
        # interior face f (1..N-1): F_f = wL * C_{f-1} + wR * C_f
        if self.advection == "central":
            wL = np.broadcast_to(0.5 * u + D / h, (B, N - 1))
            wR = np.broadcast_to(0.5 * u - D / h, (B, N - 1))
        else:
            wL = np.broadcast_to(u + D / h, (B, N - 1))
            wR = np.broadcast_to(-D / h, (B, N - 1))
        lower = np.zeros((B, N))
        diag = np.zeros((B, N))
        upper = np.zeros((B, N))
        # +F_j / h contributes to row j for j = 1..N-1 (interior face j)
        lower[:, 1:] += wL / h
        diag[:, 1:] += wR / h
        # -F_{j+1} / h contributes to row j for j = 0..N-2 (interior face j+1)
        diag[:, :-1] -= wL / h
        upper[:, :-1] -= wR / h
        # outlet face N: F_N = u C_{N-1}
        diag[:, -1] -= self.u / h
        # inlet face 0: F_0 = u C_in -> enters g(t), not A
        diag -= k
        return lower, diag, upper

    def _assemble(self) -> sp.csr_matrix:
        B, N = self.B, self.N
        lo = self.lower.copy()
        up = self.upper.copy()
        lo[:, 0] = 0.0  # no coupling across block boundaries
        up[:, -1] = 0.0
        main = self.diag.ravel()
        sub = lo.ravel()[1:]
        sup = up.ravel()[:-1]
        return sp.diags([sub, main, sup], offsets=[-1, 0, 1], shape=(B * N, B * N), format="csr")

    # -- helpers ----------------------------------------------------------
    def forcing(self, C_in, source=None) -> np.ndarray:
        """Flattened forcing vector g for inlet concentration(s) C_in and optional source (B,N)."""
        g = np.zeros((self.B, self.N))
        g[:, 0] = self.inlet_gain * np.broadcast_to(np.asarray(C_in, dtype=float), (self.B,))
        if source is not None:
            g += np.broadcast_to(np.asarray(source, dtype=float), (self.B, self.N))
        return g.ravel()

    def mass(self, C_flat: np.ndarray) -> np.ndarray:
        """Discrete mass per unit cross-section, h * sum_j C_j, for each batch member."""
        return self.grid.h * C_flat.reshape(self.B, self.N).sum(axis=1)

    def mass_rate(self, C_flat, C_in, source=None) -> np.ndarray:
        """Right-hand side of the discrete mass balance u C_in - u C_out - k M (+ source)."""
        C = C_flat.reshape(self.B, self.N)
        rate = self.u * np.broadcast_to(np.asarray(C_in, dtype=float), (self.B,)) - self.u * C[:, -1]
        rate = rate - self.k * self.grid.h * C.sum(axis=1)
        if source is not None:
            rate = rate + self.grid.h * np.broadcast_to(np.asarray(source, dtype=float), (self.B, self.N)).sum(axis=1)
        return rate

    def energy_matrix_max_eig(self) -> float:
        """Largest eigenvalue of the symmetric part (A + A^T)/2 (dense; small N, B=1 only)."""
        if self.B * self.N > 4000:
            raise ValueError("Dense eigenvalue check intended for small systems only.")
        As = self.A.toarray()
        return float(np.linalg.eigvalsh(0.5 * (As + As.T)).max())


# ---------------------------------------------------------------------------
# Explicit-scheme numbers and FTCS (von Neumann) analysis
# ---------------------------------------------------------------------------
def explicit_numbers(u, D, k, h, dt):
    """Return (alpha, beta, kappa) = (D dt/h^2, u dt/h, k dt)."""
    return D * dt / h**2, u * dt / h, k * dt


def ftcs_amplification(alpha, beta, kappa, theta):
    """FTCS amplification factor g = 1 - kappa - 2 alpha (1 - cos th) - i beta sin th."""
    return 1.0 - kappa - 2.0 * alpha * (1.0 - np.cos(theta)) - 1j * beta * np.sin(theta)


def ftcs_max_amplification(alpha, beta, kappa, n_theta: int = 4001):
    """max_theta |g(theta)| evaluated on a uniform grid of wave angles in [0, pi]."""
    th = np.linspace(0.0, np.pi, n_theta)
    alpha, beta, kappa = (np.asarray(v, dtype=float)[..., None] for v in (alpha, beta, kappa))
    return np.abs(ftcs_amplification(alpha, beta, kappa, th)).max(axis=-1)


def ftcs_strictly_stable(alpha, beta, kappa, tol: float = 1e-12):
    """Exact test of |g(theta)| <= 1 for all theta (strict von Neumann condition).

    With w = 1 - cos(theta) in [0, 2] and a = 1 - kappa,
        |g|^2 - 1 = P(w) = (4 alpha^2 - beta^2) w^2 + (2 beta^2 - 4 alpha a) w + (a^2 - 1).
    The scheme is strictly stable iff max_{w in [0,2]} P(w) <= 0; the maximum of
    a quadratic on an interval is attained at an end point or at the vertex.
    For kappa = 0 this reduces to  beta^2 <= 2 alpha <= 1.
    """
    alpha, beta, kappa = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (alpha, beta, kappa)))
    a = 1.0 - kappa
    c2 = 4.0 * alpha**2 - beta**2
    c1 = 2.0 * beta**2 - 4.0 * alpha * a
    c0 = a**2 - 1.0
    P = lambda w: c2 * w**2 + c1 * w + c0  # noqa: E731
    pmax = np.maximum(P(0.0), P(2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        w_star = np.where(c2 < 0, -c1 / (2.0 * c2), -1.0)
    inside = (c2 < 0) & (w_star > 0) & (w_star < 2)
    pmax = np.where(inside, np.maximum(pmax, P(np.where(inside, w_star, 0.0))), pmax)
    out = pmax <= tol
    return out[()] if out.ndim == 0 else out


def ftcs_lax_richtmyer_stable(alpha):
    """Lax--Richtmyer stability of FTCS under refinement at fixed alpha (D > 0): alpha <= 1/2.

    Advection and reaction contribute O(dt) to |g|^2 when alpha is held fixed
    (beta^2 = u^2 alpha dt / D), so they do not affect stability in this
    asymptotic sense; they do matter for the strict (no-growth) condition.
    For D = 0 (pure advection, beta fixed) FTCS is unstable for every beta != 0.
    """
    return np.asarray(alpha) <= 0.5


def ftcs_positive(alpha, beta, kappa):
    """Sufficient condition for positivity / discrete maximum principle of FTCS.

    All stencil weights (alpha + beta/2, 1 - 2 alpha - kappa, alpha - beta/2)
    are non-negative iff |beta| <= 2 alpha (cell Peclet number <= 2) and
    2 alpha + kappa <= 1.
    """
    alpha, beta, kappa = (np.asarray(v, dtype=float) for v in (alpha, beta, kappa))
    return (np.abs(beta) <= 2.0 * alpha + 1e-15) & (2.0 * alpha + kappa <= 1.0 + 1e-15)


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
@dataclass
class TransientResult:
    """Output of :func:`integrate`.

    Attributes
    ----------
    t : (n_steps + 1,) time levels [s]
    outlet : (n_steps + 1, B) exit concentration C_{N-1} (= C(L) + O(h^2))
    inlet : (n_steps + 1, B) imposed feed concentration C_in(t)
    mass : (n_steps + 1, B) discrete mass h sum C_j
    snapshots : {time: (B, N) array}
    final : (B, N) solution at t_end
    mass_balance_residual : max |M^{n+1} - M^n - dt*(theta-weighted mass rate)|
    min_value : minimum concentration over all cells and steps
    """

    t: np.ndarray
    outlet: np.ndarray
    inlet: np.ndarray
    mass: np.ndarray
    snapshots: dict
    final: np.ndarray
    mass_balance_residual: float
    min_value: float
    info: dict = field(default_factory=dict)


def _inlet_values(inlet, t, B):
    val = inlet(t) if callable(inlet) else inlet
    out = np.broadcast_to(np.asarray(val, dtype=float), (B,)).copy()
    if not np.all(np.isfinite(out)):
        raise ValueError(f"Inlet concentration is not finite at t = {t:g}.")
    return out


def _source_values(source, t, B, N):
    """Source term in (N, B) layout (or None)."""
    if source is None:
        return None
    val = source(t) if callable(source) else source
    return np.broadcast_to(np.asarray(val, dtype=float), (B, N)).T


class _ThetaStepper:
    """Linear algebra for one theta-method step in (N, B) memory layout.

    ``apply(x)`` returns (I + c_expl A) x and ``solve(r)`` returns (I - c_impl A)^{-1} r
    for the B independent tridiagonal blocks.  Two equivalent back-ends:
    a vectorised Thomas algorithm (fast for large batches; used only when the
    matrix is diagonally dominant, so no pivoting is required) and SuperLU.
    """

    def __init__(self, op: FVOperator, c_impl: float, c_expl: float, backend: str = "auto"):
        lo, di, up = op.lower.T, op.diag.T, op.upper.T  # (N, B)
        self.N, self.B = op.N, op.B
        self.el, self.ed, self.eu = c_expl * lo, 1.0 + c_expl * di, c_expl * up
        self.c_impl = c_impl
        self.backend = None
        if c_impl == 0.0:
            return
        a, b, c = -c_impl * lo, 1.0 - c_impl * di, -c_impl * up
        a = a.copy()
        c = c.copy()
        a[0] = 0.0
        c[-1] = 0.0
        dominant = bool(np.all(np.abs(b) >= np.abs(a) + np.abs(c)))
        if backend == "auto":
            backend = "thomas" if (self.B >= 8 and dominant) else "splu"
        if backend == "thomas":
            if not dominant:
                raise ValueError("Thomas back-end requires a diagonally dominant matrix.")
            den = np.empty_like(b)
            cp = np.empty_like(b)
            den[0] = b[0]
            cp[0] = c[0] / den[0]
            for j in range(1, self.N):
                den[j] = b[j] - a[j] * cp[j - 1]
                cp[j] = c[j] / den[j]
            self.a, self.den, self.cp = a, den, cp
        else:
            I = sp.identity(self.B * self.N, format="csc")
            self.lu = splu((I - c_impl * op.A).tocsc(), permc_spec="NATURAL")
        self.backend = backend

    def apply(self, x):
        y = self.ed * x
        y[1:] += self.el[1:] * x[:-1]
        y[:-1] += self.eu[:-1] * x[1:]
        return y

    def solve(self, r):
        if self.backend == "thomas":
            x = np.empty_like(r)
            dp = r[0] / self.den[0]
            x[0] = dp
            for j in range(1, self.N):
                dp = (r[j] - self.a[j] * dp) / self.den[j]
                x[j] = dp
            for j in range(self.N - 2, -1, -1):
                x[j] -= self.cp[j] * x[j + 1]
            return x
        flat = self.lu.solve(np.ascontiguousarray(r.T).ravel())
        return flat.reshape(self.B, self.N).T.copy()


def integrate(
    op: FVOperator,
    C0,
    t_end: float,
    dt: float,
    inlet: Callable | float | np.ndarray = 0.0,
    source: Callable | np.ndarray | None = None,
    theta: float = 0.5,
    rannacher_steps: int = 0,
    save_times=(),
    stability: str = "raise",
    check_every: int = 25,
    negative_tol: float | None = None,
    backend: str = "auto",
    record_mass_balance: bool = True,
) -> TransientResult:
    """Integrate dC/dt = A C + g(t) from t = 0 to t_end with the theta-method.

    Parameters
    ----------
    op : FVOperator
    C0 : array (N,) or (B, N)
        Initial cell values (ideally cell averages).
    t_end, dt : float
        Final time and time step [s]; t_end must be an integer multiple of dt.
    inlet : callable t -> (B,) or constant
        Feed concentration C_in(t) entering through the Danckwerts inlet flux.
    source : callable t -> (B, N) or (N,), optional
        Volumetric source (used for manufactured solutions).
    theta : {0, 0.5, 1}
        0.5 Crank--Nicolson, 1 backward Euler, 0 forward Euler (explicit).
    rannacher_steps : int
        Number of initial CN steps replaced by two backward-Euler half-steps each.
    save_times : iterable of float
        Times at which full profiles are stored (must lie on the time grid).
    stability : {"raise", "warn", "ignore"}
        Action when a forward-Euler step violates the strict von Neumann
        condition (central) or the positivity condition (upwind).
    negative_tol : float, optional
        If given, warn when min C < -negative_tol.
    backend : {"auto", "thomas", "splu"}
        Linear-solver back-end (results agree to round-off).
    record_mass_balance : bool
        Evaluate the discrete mass-balance residual at every step.
    """
    if theta not in (0.0, 0.5, 1.0):
        raise ValueError("theta must be 0 (forward Euler), 0.5 (Crank-Nicolson) or 1 (backward Euler).")
    if not (np.isfinite(dt) and dt > 0 and np.isfinite(t_end) and t_end > 0):
        raise ValueError("dt and t_end must be positive and finite.")
    n_steps = int(round(t_end / dt))
    if n_steps < 1 or abs(n_steps * dt - t_end) > 1e-9 * max(1.0, t_end):
        raise ValueError(f"t_end={t_end} is not an integer multiple of dt={dt}.")
    B, N, h = op.B, op.N, op.grid.h

    C = np.array(np.broadcast_to(np.asarray(C0, dtype=float), (B, N)), dtype=float).T.copy()  # (N, B)
    if not np.all(np.isfinite(C)):
        raise ValueError("Initial condition contains non-finite values.")

    # --- explicit-step safeguard ----------------------------------------
    if theta == 0.0:
        alpha, beta, kappa = explicit_numbers(op.u, op.D, op.k, h, dt)
        if op.advection == "central":
            ok = ftcs_strictly_stable(alpha, beta, kappa)
            msg = "strict von Neumann condition (|g| <= 1)"
        else:
            ok = (2 * alpha + beta + kappa) <= 1.0 + 1e-14
            msg = "upwind positivity condition 2 alpha + beta + kappa <= 1"
        if not np.all(ok):
            text = f"Explicit step violates the {msg} for {np.size(ok) - np.count_nonzero(ok)} batch member(s)."
            if stability == "raise":
                raise ValueError(text)
            if stability == "warn":
                warnings.warn(text, RuntimeWarning, stacklevel=2)

    main = _ThetaStepper(op, theta * dt, (1.0 - theta) * dt, backend)
    if rannacher_steps > 0 and theta == 0.5:
        half = _ThetaStepper(op, 0.5 * dt, 0.0, backend)
    else:
        rannacher_steps = 0
        half = None

    save_steps = {}
    for ts in save_times:
        s = int(round(ts / dt))
        if abs(s * dt - ts) > 1e-9 * max(1.0, abs(ts)) or s < 0 or s > n_steps:
            raise ValueError(f"save time {ts} is not on the time grid.")
        save_steps[s] = float(ts)

    t_grid = np.arange(n_steps + 1) * dt
    outlet = np.empty((n_steps + 1, B))
    inlet_hist = np.empty((n_steps + 1, B))
    mass = np.empty((n_steps + 1, B))
    snapshots = {}
    gain = op.inlet_gain
    u, k = op.u, op.k

    def forcing(cin, src):
        g = np.zeros((N, B)) if src is None else np.array(src, dtype=float)
        g[0] += gain * cin
        return g

    def mass_rate(Cs, cin, src):
        rate = u * cin - u * Cs[-1] - k * h * Cs.sum(axis=0)
        if src is not None:
            rate = rate + h * src.sum(axis=0)
        return rate

    cin = _inlet_values(inlet, 0.0, B)
    src = _source_values(source, 0.0, B, N)
    g_old = forcing(cin, src)
    outlet[0] = C[-1]
    inlet_hist[0] = cin
    mass[0] = h * C.sum(axis=0)
    if 0 in save_steps:
        snapshots[save_steps[0]] = C.T.copy()
    min_value = float(C.min())
    max_resid = 0.0

    def step(C, t0, dt_s, th, stepper, cin0, src0, g0):
        t1 = t0 + dt_s
        cin1 = _inlet_values(inlet, t1, B)
        src1 = _source_values(source, t1, B, N)
        g1 = forcing(cin1, src1)
        rhs = stepper.apply(C) + dt_s * (th * g1 + (1.0 - th) * g0)
        Cn = rhs if th == 0.0 else stepper.solve(rhs)
        resid = 0.0
        if record_mass_balance:
            dM = h * (Cn.sum(axis=0) - C.sum(axis=0))
            flux = th * mass_rate(Cn, cin1, src1) + (1.0 - th) * mass_rate(C, cin0, src0)
            resid = float(np.max(np.abs(dM - dt_s * flux)))
        return Cn, t1, cin1, src1, g1, resid

    t = 0.0
    for n in range(1, n_steps + 1):
        if n <= rannacher_steps:
            for _ in range(2):
                C, t, cin, src, g_old, r = step(C, t, 0.5 * dt, 1.0, half, cin, src, g_old)
                max_resid = max(max_resid, r)
        else:
            C, t, cin, src, g_old, r = step(C, t, dt, theta, main, cin, src, g_old)
            max_resid = max(max_resid, r)
        t = t_grid[n]
        outlet[n] = C[-1]
        inlet_hist[n] = cin
        mass[n] = h * C.sum(axis=0)
        if n in save_steps:
            snapshots[save_steps[n]] = C.T.copy()
        min_value = min(min_value, float(C.min()))
        if n % check_every == 0 or n == n_steps:
            if not np.all(np.isfinite(C)):
                raise FloatingPointError(f"Non-finite concentration at step {n} (t = {t:g} s).")

    if negative_tol is not None and min_value < -negative_tol:
        warnings.warn(f"Minimum concentration {min_value:.3e} below -{negative_tol:g}.", RuntimeWarning, stacklevel=2)

    return TransientResult(
        t=t_grid,
        outlet=outlet,
        inlet=inlet_hist,
        mass=mass,
        snapshots=snapshots,
        final=C.T.copy(),
        mass_balance_residual=max_resid,
        min_value=min_value,
        info={"theta": theta, "dt": dt, "n_steps": n_steps, "rannacher_steps": rannacher_steps,
              "N": N, "B": B, "advection": op.advection, "backend": main.backend},
    )


def solve_steady(op: FVOperator, C_in=1.0, source=None) -> np.ndarray:
    """Steady state: solve A C = -g for each batch member. Returns (B, N)."""
    g = op.forcing(C_in, source)
    C = spsolve(op.A.tocsc(), -g)
    C = np.asarray(C).reshape(op.B, op.N)
    if not np.all(np.isfinite(C)):
        raise FloatingPointError("Steady solve produced non-finite values (singular operator?).")
    return C
