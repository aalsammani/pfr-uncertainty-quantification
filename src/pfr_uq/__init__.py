"""pfr_uq: verified numerics and uncertainty quantification for the axial-dispersion reactor.

The package solves the one-dimensional advection--dispersion--reaction equation

    dC/dt + u dC/dx = D d^2C/dx^2 - k C,   0 < x < L,

with Danckwerts boundary conditions

    u C(0,t) - D dC/dx(0,t) = u C_in(t),     dC/dx(L,t) = 0,

using a cell-centred finite-volume discretisation and theta-method time stepping
(Crank--Nicolson by default).  Modules:

model        physical parameters, dimensionless groups, analytical solutions
solvers      finite-volume operator, time integration, FTCS stability analysis
metrics      discrete norms, convergence orders, curve moments
uncertainty  input distributions, Monte Carlo / randomized quasi-Monte Carlo,
             Sobol' indices, measurement-error models
plotting     publication figure style and export helpers
legacy       reproduction of the original (erroneous) explicit stencil, for audit
paths        repository-relative paths used by the notebooks
reporting    JSON / CSV / LaTeX output helpers
"""

__version__ = "1.0.0"
