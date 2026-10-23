# Exhaustive 1 mm grid-scan benchmark

This directory contains the same-resolution exhaustive grid-scan benchmark used for the revised computational comparison in Section 5.2.5 and Fig. 10.

## Benchmark definition

- Parameter ranges: `aC = 0.300–0.500 m`, `HTR = 0.200–1.000 m`.
- Resolution: `0.001 m` in both parameters.
- Grid size: `201 × 801 = 161,001` parameter pairs.
- Nominal-feasibility screening: 81 permitted receiver positions at zero vertical offset.
- Nominally feasible designs: `23,391`.
- Full objective evaluations: `23,391` per exhaustive run.
- Cross-candidate policy: design-dependent full Tx/Cx fields and objective values are not reused across different candidate geometries. Screening-only one-dimensional profiles, static quadrature nodes, and vectorized numerical operations are retained.
- Timing protocol: 20 strictly serial repetitions, one numerical-library thread per run.

## Recorded result

The 20-run benchmark returned the same optimum in every run:

- `aC = 0.422 m`
- `HTR = 0.573 m`
- objective `J = 0.2490668148527419`
- threshold-satisfying area `797 mm²`
- median wall time `585.3807426653802 s`
- wall-time range `565.939–649.363 s`

The five SCA initializations had median runtimes of approximately `1.90–6.53 s`, corresponding to median runtime reductions of `98.88–99.68%` relative to the common exhaustive-grid median.

## Files

| File | Purpose |
| --- | --- |
| `run_grid1mm_serial20_recorded.py` | Script used for the recorded 20-repeat serial benchmark in the original benchmark working directory. |
| `run_grid1mm_portable.py` | Repository-compatible implementation of the same 1 mm exhaustive-search protocol using the public physics/objective modules. |
| `grid_serial20_summary.json` | Aggregate statistics from the 20 strictly serial exhaustive-grid runs. |
| `sca_100_timings.csv` | Twenty SCA timings for each of the five initialization offsets. |
| `Fig10_summary.csv` | Derived Fig. 10 summary values. |
| `plot_fig10.py` | Reproduces Fig. 10 from the archived timing records and the common grid median. |

## Reproduction

From the repository root, a numerical check can be run with:

```bash
python benchmark_grid1mm/run_grid1mm_portable.py --repeats 1 --output-dir benchmark_grid1mm/results/check
```

To repeat the full timing protocol, use `--repeats 20` on a controlled machine with no competing workload. Absolute timing is hardware- and environment-dependent.

Reproduce Fig. 10 with:

```bash
python benchmark_grid1mm/plot_fig10.py
```

The manuscript timing values are taken from `grid_serial20_summary.json`, not from a newly generated run on a different machine.
