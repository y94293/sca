# NCPC zone-decoupling: SCA reproduction

Reproduction code for the sequential convex approximation (SCA) calculation in
**Zone-Decoupling Optimization for Highly Robust Non-Coplanar Positive Compensation TEM**
by Tianjiao Yang, Dengke He, Guowei Zhu, and Yuxuan Zhou.

Algorithm version: `NCPC-SCA-CLARABEL-local-1mm-v2.0`.

## Install

Use Python 3.12 on Linux for the closest match to the recorded environment
(Python 3.12.3). Create a fresh virtual environment, then install the three
direct dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The active solver uses native CLARABEL; CVXPY is not required by this extracted
implementation. OpenBLAS, OMP and MKL thread counts are set to one by the runner.

## Reproduce

Run the nominal initialization `(aC, HTR) = (0.430, 0.710) m`:

```bash
python run_sca.py --verify
```

Run all five reported SCA initializations:

```bash
python run_sca.py --all-initials --verify --output-dir results/five_initials
```

Each invocation needs a new output directory. Existing results are preserved.
`--verify` compares freshly calculated geometry, receiver position, objective,
fixed-grid area, nominal mutual inductance and termination status with the
archived SCA values; it also checks convex solver statuses and residuals.
The runner calculates the results rather than loading reference outputs.

The output directory contains one JSON file per initialization and a
`summary.json` with effective settings, environment, full SCA traces and results.
Repeated runs are deterministic in their numerical outputs; runtime depends
on the execution environment. This repository currently covers SCA-only reproduction; the comparative
benchmark scripts and timing records are not yet hosted here.

## Scope of the grid-scan comparison

The manuscript's grid-search baseline evaluates a predefined 41 × 41 design
lattice (1,681 candidates), with 0.005 m spacing in aC and 0.020 m spacing
in HTR. The reported computational comparison applies to that finite-grid
baseline; it is not evidence of a speed advantage over an exhaustive search
at the same 1 mm resolution used for the engineering-neighbour check.

## Expected SCA solutions

| Offset | Actual initial aC / HTR (m) | Final aC / HTR / x0 (m) | Fixed-grid area (mm²) |
| --- | --- | --- | --- |
| -20% | 0.344 / 0.568 | 0.421 / 0.558 / 0.553 | 792 |
| -10% | 0.387 / 0.639 | 0.421 / 0.558 / 0.553 | 792 |
| 0% | 0.430 / 0.710 | 0.422 / 0.573 / 0.563 | 797 |
| +10% | 0.473 / 0.781 | 0.420 / 0.543 / 0.542 | 761 |
| +20% | 0.500 / 0.852 | 0.421 / 0.558 / 0.553 | 792 |

Both nominal initial parameters are multiplied by `1 + offset/100`, rounded to
1 mm, and clipped to the box. At +20%, the requested aC of 0.516 m is clipped
to 0.500 m. Every case terminates at `engineering_neighbour_stall`: no adjacent
in-bound 1 mm design improves the original objective. This termination does not
claim continuous stationarity or global optimality.

## Numerical settings

- Threshold: `Mth = 1e-10 H`.
- Design box: aC in `[0.300, 0.500] m`; HTR in `[0.200, 1.000] m`.
- Spatial grid: 81 × 41 nodes, x in `[0.500, 0.580] m`, dz in
  `[-0.020, 0.020] m`; trapezoidal area weights and 1 mm spacing.
- Tx: square, side 0.500 m, four turns; Cx: square, one turn;
  Rx: circular, radius 0.100 m, ten turns.
- Receiver-circumference quadrature: 64 points; analytic source-segment
  integrals and analytic parameter derivatives.
- Objective: normalized smoothed area minus normalized Laplacian and
  squared threshold-exceedance penalties; coefficients `1, 0.05, 0.05`.
- CLARABEL: QDLDL, one thread, at most 150 iterations; absolute/relative gap
  and feasibility tolerances `1e-11`.

Full settings are in `config/sca_settings.json`. The nominal receiver is chosen
at the nearest feasible 1 mm position to the midpoint of the widest interval
inside the permitted receiver range, with the tie rules in `solver_local.py`.

## Files

| File | Purpose |
| --- | --- |
| `run_sca.py` | Independent nominal/five-initialization entry point |
| `solver_local.py` | Reported trust-region SCA and engineering projection |
| `sca_engine.py` | Per-run physical field and derivative caches |
| `sca_objective.py` | Normalized SCA objective |
| `analytic_jacobian.py` | Analytic source-segment parameter derivatives |
| `tem_sca/native_sca.py` | Native CLARABEL convex subproblem |
| `tem_sca/physics.py` | Signed mutual-inductance model |
| `tem_sca/metrics.py` | Spatial grid and signed threshold intervals |
| `reference/sca_expected.csv` | Five SCA-only numerical reference solutions |
| `SOURCE_PROVENANCE.json` | Original source hashes and retained-symbol checks |



Run the reproduction regression check with:

```bash
python -m unittest discover -s tests -v
```
