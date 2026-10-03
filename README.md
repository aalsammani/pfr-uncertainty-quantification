
# Numerical, parametric and measurement uncertainty in axial-dispersion reactor simulations

Computational companion to the manuscript

> A. Alsammani, M. A. Y. Mohammed, A. Ali, *Numerical, Parametric, and Measurement Uncertainty in Axial-Dispersion Reactor Simulations: A Verified Finite-Volume and Quasi-Monte Carlo Study* (manuscript in preparation; source in [`manuscript/`](manuscript/)).

The repository contains a small, tested Python package (`pfr_uq`) and four Jupyter notebooks. Together they regenerate every number, table and figure of the manuscript from scratch.

![Uncertainty across the Péclet–Damköhler plane](manuscript/figures/fig07_regime_uq.png)

## Scientific motivation

The axial-dispersion model is the standard one-dimensional description of non-ideal tubular reactors. Predictions made with it are uncertain for several different reasons:

- the parameters (velocity, dispersion coefficient, kinetics) are imperfectly known;
- measurements carry random noise and systematic instrument offsets;
- the numerical solution has a discretization error;
- Monte Carlo estimates add a sampling error.

This project keeps these sources apart. It first verifies the solver rigorously, and only then propagates and compares the uncertainties.

## Mathematical model

For the concentration $C(x,t)$ in a reactor of length $L$, with first-order consumption,

$$
\frac{\partial C}{\partial t}+u\frac{\partial C}{\partial x}=D\frac{\partial^2 C}{\partial x^2}-kC,\qquad 0<x<L,
$$

with Danckwerts (closed–closed) boundary conditions

$$
uC(0^+,t)-D\,\partial_x C(0^+,t)=u\,C_\mathrm{in}(t),\qquad \partial_x C(L,t)=0 .
$$

The dimensionless groups are $\mathrm{Pe}=uL/D$ and $\mathrm{Da}=kL/u$. The steady outlet ratio has the closed form (Wehner–Wilhelm)

$$
\frac{C(L)}{C_\mathrm{in}}=\frac{4q\,e^{\mathrm{Pe}/2}}{(1+q)^2e^{q\mathrm{Pe}/2}-(1-q)^2e^{-q\mathrm{Pe}/2}},\qquad q=\sqrt{1+4\mathrm{Da}/\mathrm{Pe}} .
$$

## Computational methods

| Component | Method |
|---|---|
| Spatial discretisation | cell-centred finite volumes; central advective fluxes (first-order upwind available); the Danckwerts inlet is imposed as an **exact** flux $F_0=uC_\mathrm{in}$ |
| Time integration | $\theta$-method: Crank–Nicolson (production), backward/forward Euler (comparison); optional Rannacher start-up |
| Stability theory | exact strict von Neumann region of explicit FTCS *with reaction*; discrete energy estimate ⇒ unconditional $L^2$ stability of CN on the bounded domain |
| Verification | 6 exact benchmarks (advection, dispersion, reaction, ADR, steady Danckwerts, transfer-function moments) plus manufactured solutions for separate spatial and temporal orders |
| Forward UQ | independent log-normal inputs $(u,D,k,C_\mathrm{in})$; batched PDE solves |
| Sampling | Monte Carlo vs scrambled Sobol' randomised QMC on **identical** input distributions; 32 independent replicates; bootstrap CIs |
| Sensitivity | first-order (Saltelli 2010) and total (Jansen 1999) Sobol' indices; regime maps over $\mathrm{Pe}\in[1,10^3]$, $\mathrm{Da}\in[0.1,10]$ |
| Measurement error | $y=C+b+\varepsilon$ (fixed offset $b$, Gaussian noise $\varepsilon$); inverse problems for $k$ and $(D,k)$; interval coverage |

## Main findings (baseline $\mathrm{Pe}=10$, $\mathrm{Da}=1$; all values from `results/final_results.json`)

- Second-order convergence in space and time is measured, not assumed. Discrete mass balance holds to round-off.
- Explicit FTCS is unstable for pure advection at any Courant number. For advection–dispersion it needs $\beta^2\le2\alpha\le1$, and reaction modifies this region.
- The exit concentration has a CV of 19.1%. Sobol' indices: rate constant 0.75, velocity 0.17, feed 0.07, dispersion 0.01. Dispersion matters mainly for arrival times and for back-mixed reactors at high $\mathrm{Da}$.
- RQMC errors decay as $n^{-1.07}$ to $n^{-1.17}$, against $n^{-1/2}$ for MC. At $n=4096$ this means $2.2\times10^4$ to $6.4\times10^4$ times smaller mean-squared error.
- Ignoring a sensor offset of 1.5% of the peak signal biases the fitted rate constant by −3.3% and drops 95% CI coverage to 0.11. Estimating the offset restores coverage (0.93–0.96).

All observations are **synthetic**, and the input distributions are **assumed** for illustration. No experimental validation is claimed.

## Repository structure

```text
├── src/pfr_uq/            Python package
│   ├── model.py           parameters, dimensionless groups, analytical solutions
│   ├── solvers.py         finite-volume operator, theta-method, FTCS stability analysis
│   ├── qoi.py             reactor quantities of interest
│   ├── uncertainty.py     input distributions, MC/RQMC sampling, Sobol' indices
│   ├── metrics.py         norms, convergence orders, curve moments
│   ├── plotting.py        figure style and export (PDF + PNG)
│   ├── reporting.py       JSON/CSV/LaTeX writers
│   ├── legacy.py          reproduction of the stencil error found in the original draft (audit only)
│   └── paths.py           repository-relative paths
├── notebooks/             01–04: the executable research record (run in order)
├── tests/                 pytest suite for the numerical core
├── results/
│   ├── data/              per-notebook summaries (JSON) and raw arrays/curves (CSV, NPZ)
│   ├── tables/            machine-readable tables (CSV)
│   └── final_results.json all reported quantities
├── manuscript/
│   ├── main_revised_clean.tex / .pdf     revised manuscript
│   ├── main_revised_marked.tex / .pdf    latexdiff against the original
│   ├── references.bib                    Crossref-verified bibliography
│   ├── figures/                          fig01–fig09 (PDF for LaTeX, PNG for preview)
│   └── generated/                        LaTeX tables and value macros written by notebook 04
├── docs/                  audits, revision log, reproducibility map
├── tools/                 run_all.py, reference verification, latexdiff post-processing
└── original/              untouched original manuscript and figures (with SHA-256 checksums)
```

## Installation

Tested with Python 3.14.3 on Windows 11; the code uses only portable, repository-relative paths.

```bash
git clone <repository-url>
cd <repository>
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used, and installs `pfr_uq` in editable mode. Compiling the manuscript also needs a LaTeX distribution with `latexmk`; the marked version additionally needs `latexdiff-so`.

## Reproducing the study

```bash
python -m pytest -q                 # fast unit tests (~15 s)
python tools/run_all.py             # tests + notebooks 01->04 + manuscript (~25 min)
```

Alternatively, open the notebooks in Jupyter and run them **in order**, each from a fresh kernel:

| Notebook | Purpose | Runtime |
|---|---|---|
| `01_model_verification` | symbolic checks of the stencil, amplification factor, steady solution, energy/mass identities; FTCS demonstrations; reproduction of the original stencil error | < 1 min |
| `02_numerical_validation` | benchmarks A–F, manufactured-solution convergence, production-setting accuracy | < 1 min |
| `03_uncertainty_analysis` | forward UQ, MC vs RQMC, Sobol' indices, Pe–Da maps, measurement-error experiments, error budget | ~20 min |
| `04_publication_figures` | all figures, LaTeX tables, value macros and `final_results.json` | < 1 min |

Every random number comes from the master seed `20261002` via `numpy.random.SeedSequence`, so reruns are deterministic. `docs/reproducibility.md` maps every manuscript figure, table and number to its source.

## Outputs

- `manuscript/figures/fig0X_*.pdf|png`: publication figures, generated programmatically and never edited by hand.
- `manuscript/generated/tab_*.tex`: `booktabs` tables; `results_macros.tex` holds every number quoted in the text.
- `results/tables/*.csv`: machine-readable versions of all tables (parameters, convergence, UQ statistics, rates, inference).
- `results/final_results.json`: the complete set of reported quantities.

## Citation

A citation entry is in [`CITATION.cff`](CITATION.cff). Please cite the manuscript once it is published.

## License

The **software** (src/, notebooks/, tools/, tests/) is released under the MIT License ([`LICENSE`](LICENSE)). The manuscript text and figures in `manuscript/`, and the material in `original/`, are **not** covered by that license and remain under the authors' copyright pending publication. Authors should confirm this choice against their institutional and journal requirements before making the repository public.

## Contact

Corresponding author: Abdallah Alsammani, Department of Mathematical Sciences, Delaware State University (aalsammani@desu.edu).
