"""Fast regression tests for the numerical core (run with `python -m pytest`)."""
import numpy as np
import pytest

from pfr_uq import legacy, metrics, model, solvers, uncertainty

P = model.BASELINE


def test_steady_solution_second_order():
    errs, hs = [], []
    for N in (40, 80, 160, 320):
        op = solvers.FVOperator(P.u, P.D, P.k, P.L, N)
        C = solvers.solve_steady(op, 1.0)[0]
        exact = model.danckwerts_profile(op.grid.centers / P.L, P.Pe, P.Da)
        errs.append(metrics.norm_l2(C - exact, op.grid.h))
        hs.append(op.grid.h)
    p = metrics.observed_orders(hs, errs)
    assert np.all(np.abs(p - 2.0) < 0.05)


def test_outlet_ratio_closed_forms():
    Pe, Da = 7.3, 1.9
    q = np.sqrt(1 + 4 * Da / Pe)
    ww = 4 * q * np.exp(Pe / 2) / ((1 + q) ** 2 * np.exp(q * Pe / 2) - (1 - q) ** 2 * np.exp(-q * Pe / 2))
    assert np.isclose(model.danckwerts_outlet_ratio(Pe, Da), ww, rtol=1e-13)
    assert np.isclose(model.danckwerts_profile(1.0, Pe, Da), ww, rtol=1e-12)
    # limits
    assert np.isclose(model.danckwerts_outlet_ratio(1e6, 2.0), np.exp(-2.0), rtol=1e-4)
    assert np.isclose(model.danckwerts_outlet_ratio(1e-6, 2.0), 1 / 3, rtol=1e-4)
    # no overflow at large Pe
    assert np.isfinite(model.danckwerts_outlet_ratio(5e4, 3.0))


def test_damkohler_inversion_roundtrip():
    Pe = np.array([2.0, 10.0, 200.0])
    Da = np.array([0.3, 1.0, 4.0])
    r = model.danckwerts_outlet_ratio(Pe, Da)
    assert np.allclose(model.damkohler_from_outlet_ratio(r, Pe), Da, rtol=1e-10)
    assert np.isnan(model.damkohler_from_outlet_ratio(1.2, 10.0))


def test_discrete_mass_balance_and_direction():
    op = solvers.FVOperator(P.u, 1e-4, 0.05, P.L, 400)
    x = op.grid.centers
    C0 = np.exp(-((x - 0.2) ** 2) / (2 * 0.03**2))
    res = solvers.integrate(op, C0, 4.0, 0.01, inlet=0.0, save_times=(0.0, 4.0))
    assert res.mass_balance_residual < 1e-13
    c0, c1 = res.snapshots[0.0][0], res.snapshots[4.0][0]
    centroid = lambda c: np.sum(x * c) / np.sum(c)  # noqa: E731
    assert np.isclose(centroid(c1) - centroid(c0), P.u * 4.0, atol=1e-3)


@pytest.mark.parametrize("adv", ["central", "upwind"])
@pytest.mark.parametrize("u,D,k", [(0.1, 0.01, 0.1), (1.0, 1e-4, 0.0), (0.1, 0.0, 0.2), (0.0, 0.01, 0.0)])
def test_energy_stability_symmetric_part(adv, u, D, k):
    op = solvers.FVOperator(u, D, k, 1.0, 60, advection=adv)
    assert op.energy_matrix_max_eig() <= 1e-12


def test_ftcs_strict_condition_matches_numerics():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 0.7, 4000)
    b = rng.uniform(-1.2, 1.2, 4000)
    kap = rng.uniform(0, 0.5, 4000)
    analytic = solvers.ftcs_strictly_stable(a, b, kap)
    numeric = solvers.ftcs_max_amplification(a, b, kap, n_theta=20001) <= 1 + 1e-9
    assert np.mean(analytic == numeric) > 0.998  # disagreements only within grid tolerance of the boundary
    # kappa = 0 reduces to beta^2 <= 2 alpha <= 1
    assert solvers.ftcs_strictly_stable(0.3, 0.7, 0.0)
    assert not solvers.ftcs_strictly_stable(0.2, 0.7, 0.0)
    assert not solvers.ftcs_strictly_stable(0.0, 0.5, 0.0)  # pure advection: unstable


def test_explicit_safeguard_raises():
    op = solvers.FVOperator(0.1, 0.0, 0.0, 1.0, 50)
    with pytest.raises(ValueError):
        solvers.integrate(op, 0.0, 1.0, 0.1, theta=0.0)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        model.ReactorParameters(D=-1.0)
    with pytest.raises(ValueError):
        solvers.FVOperator(0.1, -0.01, 0.1, 1.0, 50)
    with pytest.raises(ValueError):
        solvers.integrate(solvers.FVOperator(0.1, 0.01, 0.1, 1.0, 50), 0.0, 1.0, 0.3)


def test_backends_agree():
    rng = np.random.default_rng(0)
    u = 0.1 * np.exp(0.1 * rng.standard_normal(16))
    op = solvers.FVOperator(u, 0.01, 0.1, 1.0, 80)
    inlet = lambda t: model.gaussian_inlet_pulse(t)  # noqa: E731
    r1 = solvers.integrate(op, 0.0, 10.0, 0.1, inlet=inlet, backend="thomas")
    r2 = solvers.integrate(op, 0.0, 10.0, 0.1, inlet=inlet, backend="splu")
    assert np.max(np.abs(r1.outlet - r2.outlet)) < 1e-12


def test_legacy_original_stencil_moves_upstream():
    gauss = lambda x: np.exp(-((x - 0.5) ** 2) / (2 * 0.04**2))  # noqa: E731
    for stencil, sign in (("original", -1), ("corrected", 1)):
        x, snaps = legacy.node_ftcs(0.1, 1e-3, 0.0, 1.0, 200, 1e-3, 1.0, gauss, save_times=(1.0,), stencil=stencil)
        c = snaps[1.0]
        assert np.sign(np.sum(x * c) / np.sum(c) - 0.5) == sign


def test_lognormal_and_rqmc():
    ln = uncertainty.LogNormal(2.0, 0.3)
    assert np.isclose(ln.sd / ln.mean, 0.3)
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        uncertainty.unit_cube_sample(100, 2, "rqmc", rng)
    U = uncertainty.unit_cube_sample(1024, 3, "rqmc", rng)
    assert U.shape == (1024, 3) and np.all((U >= 0) & (U < 1))
