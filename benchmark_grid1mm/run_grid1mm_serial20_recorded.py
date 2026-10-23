#!/usr/bin/env python3
"""
Serial 20-repeat benchmark for the uncached optimized exhaustive 1 mm grid scan.

Expected location:
    /root/autodl-tmp/ncpc_fine_only_20261020/run_grid1mm_serial20.py

Run:
    cd /root/autodl-tmp/ncpc_fine_only_20261020
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /root/autodl-tmp/tem_sca_reconstruction_20261004/.venv/bin/python \
    run_grid1mm_serial20.py

Outputs:
    results/grid1mm_uncached_serial20_YYYYMMDD_HHMMSS/
        repeat_01_grid1mm_uncached.json
        ...
        repeat_20_grid1mm_uncached.json
        all_grid_timings.csv
        summary.json
        run.log
"""

import os

# Force numerical libraries to one thread.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "reference"), str(ROOT / "reference/src")]

from tem_sca.metrics import Grid
from tem_sca.physics import _circle_nodes
from prepared_kernel import PreparedSquareLoop
from search_lattice import terms
from shared_engine import SharedEngine


CFG = json.loads((ROOT / "reference/inputs/search_config.json").read_text())

A_AXIS_MM = tuple(range(300, 501))   # 0.300 ... 0.500 m, 1 mm
H_AXIS_MM = tuple(range(200, 1001))  # 0.200 ... 1.000 m, 1 mm

N_REPEATS = 20
CANDIDATE_PAIRS = len(A_AXIS_MM) * len(H_AXIS_MM)  # 161001


def evaluate_candidate(key, grid):
    """Evaluate one feasible candidate with no cross-candidate full-field cache."""
    a_mm, h_mm = map(int, key)

    a = a_mm * 0.001
    h = h_mm * 0.001

    # Each candidate independently rebuilds its full Tx and Cx fields.
    cx = 10 * PreparedSquareLoop(
        a, grid.x, grid.z, CFG["nphi"]
    ).value()

    tx = 40 * PreparedSquareLoop(
        0.5, grid.x, h + grid.z, CFG["nphi"]
    ).value()

    m_normalized = (tx + cx) / CFG["threshold"]
    obj = terms(m_normalized, grid, CFG)

    return obj


def run_one(repeat_index):
    """
    One complete serial 1 mm exhaustive-grid run.

    Timer includes:
      - spatial-grid setup
      - nominal-feasibility screening
      - all full-field/objective evaluations
      - best-design selection

    Timer excludes:
      - Python interpreter startup/imports
      - JSON/CSV file writing
    """
    wall0 = time.perf_counter()
    cpu0 = time.process_time()

    grid = Grid(
        np.linspace(*CFG["x_range"], CFG["nx"]),
        np.linspace(*CFG["z_range"], CFG["nz"]),
        CFG["length_scale"],
    )

    # Fresh screening engine for every repeat.
    screening = SharedEngine(CFG, grid, None, backend="reuse")

    feasible_keys = screening.feasible_keys(A_AXIS_MM, H_AXIS_MM)

    best_key = None
    best_obj = None

    for key in feasible_keys:
        obj = evaluate_candidate(key, grid)

        if best_obj is None or obj["J"] > best_obj["J"]:
            best_key = key
            best_obj = obj

    wall_seconds = time.perf_counter() - wall0
    cpu_seconds = time.process_time() - cpu0

    result = {
        "repeat": repeat_index,
        "method": "grid1mm_uncached_optimized_serial",
        "seconds": float(wall_seconds),
        "process_cpu_seconds": float(cpu_seconds),
        "candidate_pairs": CANDIDATE_PAIRS,
        "nominally_feasible_candidates": len(feasible_keys),
        "full_objective_evaluations": len(feasible_keys),
        "best_key_mm": [int(best_key[0]), int(best_key[1])],
        "best_aC_m": best_key[0] * 0.001,
        "best_H_m": best_key[1] * 0.001,
        "best_J": float(best_obj["J"]),
        "best_area_mm2": float(best_obj["coarse_area_mm2"]),
        "cache_policy": (
            "No cross-candidate reuse of design-dependent full Tx/Cx fields "
            "or objective values. Static quadrature nodes, vectorized numerical "
            "operations, and nominal-feasibility prescreening are retained."
        ),
    }

    return result


def make_summary(records):
    wall = [r["seconds"] for r in records]
    cpu = [r["process_cpu_seconds"] for r in records]

    # Determinism checks.
    assert len({tuple(r["best_key_mm"]) for r in records}) == 1
    assert len({r["best_J"] for r in records}) == 1
    assert len({r["best_area_mm2"] for r in records}) == 1
    assert len({r["nominally_feasible_candidates"] for r in records}) == 1
    assert len({r["full_objective_evaluations"] for r in records}) == 1

    def stats(x):
        q = statistics.quantiles(x, n=4, method="inclusive")
        return {
            "min": min(x),
            "q1": q[0],
            "median": statistics.median(x),
            "q3": q[2],
            "max": max(x),
            "mean": statistics.mean(x),
            "stdev": statistics.stdev(x),
        }

    return {
        "complete": True,
        "execution_mode": "strictly serial",
        "repeats": len(records),
        "threads_per_run": 1,
        "candidate_pairs": CANDIDATE_PAIRS,
        "nominally_feasible_candidates":
            records[0]["nominally_feasible_candidates"],
        "full_objective_evaluations":
            records[0]["full_objective_evaluations"],
        "best_key_mm": records[0]["best_key_mm"],
        "best_aC_m": records[0]["best_aC_m"],
        "best_H_m": records[0]["best_H_m"],
        "best_J": records[0]["best_J"],
        "best_area_mm2": records[0]["best_area_mm2"],
        "wall_seconds": stats(wall),
        "process_cpu_seconds": stats(cpu),
        "timing_note": (
            "Twenty repeats were executed sequentially in one Python process. "
            "Static quadrature nodes were warmed before measurement. "
            "Each repeat rebuilt its spatial grid, nominal-feasibility screening "
            "state, and every design-dependent full Tx/Cx field and objective."
        ),
    }


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "results" / f"grid1mm_uncached_serial20_{stamp}"
    outdir.mkdir(parents=True, exist_ok=False)

    log_path = outdir / "run.log"

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(f"Output directory: {outdir}")
    log(f"Candidate grid: {len(A_AXIS_MM)} x {len(H_AXIS_MM)} = {CANDIDATE_PAIRS}")
    log("Execution: strictly serial, 20 repeats, one numerical-library thread")

    # Warm only the static quadrature nodes, not any design-dependent field.
    # This avoids first-use overhead while preserving the uncached-grid policy.
    _circle_nodes(CFG["nphi"])
    log("Static quadrature nodes warmed once before timed repeats.")

    records = []

    for repeat in range(1, N_REPEATS + 1):
        log(f"[{repeat:02d}/{N_REPEATS}] starting")

        result = run_one(repeat)

        raw_path = outdir / f"repeat_{repeat:02d}_grid1mm_uncached.json"
        raw_path.write_text(json.dumps(result, indent=2) + "\n")

        records.append(result)

        log(
            f"[{repeat:02d}/{N_REPEATS}] "
            f"wall={result['seconds']:.6f} s, "
            f"cpu={result['process_cpu_seconds']:.6f} s, "
            f"best=({result['best_aC_m']:.3f}, "
            f"{result['best_H_m']:.3f}) m, "
            f"area={result['best_area_mm2']:.3f} mm^2"
        )

    # Combined CSV.
    csv_path = outdir / "all_grid_timings.csv"
    fields = [
        "repeat",
        "method",
        "seconds",
        "process_cpu_seconds",
        "candidate_pairs",
        "nominally_feasible_candidates",
        "full_objective_evaluations",
        "best_aC_m",
        "best_H_m",
        "best_J",
        "best_area_mm2",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fields})

    summary = make_summary(records)
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    log("COMPLETE")
    log(
        "Median serial wall time: "
        f"{summary['wall_seconds']['median']:.6f} s"
    )
    log(
        "Median process CPU time: "
        f"{summary['process_cpu_seconds']['median']:.6f} s"
    )
    log(
        "Best design: "
        f"aC={summary['best_aC_m']:.3f} m, "
        f"H={summary['best_H_m']:.3f} m, "
        f"area={summary['best_area_mm2']:.3f} mm^2"
    )


if __name__ == "__main__":
    main()
