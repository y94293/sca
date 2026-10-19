#!/usr/bin/env python3
"""Run the paper's local 1 mm SCA, without a comparison search or plot pipeline."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
from solver_local import execute

ROOT = Path(__file__).resolve().parent
OFFSETS = [-20, -10, 0, 10, 20]


def safe(value):
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def verify(result, reference):
    best = result['best']
    if not np.allclose(best['theta'], [float(reference['final_aC_m']),
                                     float(reference['final_H_m'])], atol=1e-14, rtol=0):
        raise RuntimeError('Final SCA geometry differs from the archived result')
    for key, old in [('x0', 'final_x0_m'), ('J', 'J'),
                     ('coarse_area_mm2', 'fixed_grid_area_mm2'),
                     ('M0', 'nominal_M_H')]:
        if abs(best[key] - float(reference[old])) >= 1e-12:
            raise RuntimeError(f'Archived SCA agreement failed for {key}')
    if result['status'] != reference['termination']:
        raise RuntimeError('SCA termination status differs from the archive')
    expected = [round(float(reference['actual_aC0_m']) * 1000),
                round(float(reference['actual_H0_m']) * 1000)]
    if result['run']['initialization']['actual_mm'] != expected:
        raise RuntimeError('Rounded/clipped initialization differs from the archive')
    if result['run']['continuous_stationarity_claimed']:
        raise RuntimeError('The engineering-neighbour termination is not continuous stationarity')
    solvers = [row['solver'] for row in result['run']['trace'] if 'solver' in row]
    if not solvers or any(row['solver_name'] != 'CLARABEL' or
                          row['raw_status'] not in ('Solved', 'AlmostSolved') or
                          row['constraint_residual'] > 1e-6 for row in solvers):
        raise RuntimeError('Convex solver status or residual check failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-initials', action='store_true',
                        help='Run all five reported SCA initialization offsets')
    parser.add_argument('--verify', action='store_true',
                        help='Check newly computed numerical outputs against the SCA archive')
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                        help='A new output directory; existing directories are preserved')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config/sca_settings.json').read_text())
    with (ROOT / 'reference/sca_expected.csv').open(newline='') as stream:
        reference = {int(row['offset_percent']): row for row in csv.DictReader(stream)}
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f'Output directory already exists: {args.output_dir}; choose a new directory')
    results = []
    for offset in OFFSETS if args.all_initials else [0]:
        result = execute(cfg, [.430 * (1 + offset / 100), .710 * (1 + offset / 100)])
        result['offset_percent'] = offset
        result['verified'] = False
        if args.verify:
            verify(result, reference[offset])
            result['verified'] = True
        results.append(result)
        (args.output_dir / f'sca_{offset:+d}.json').write_text(
            json.dumps(safe(result), indent=2, allow_nan=False) + '\n')
        best = result['best']
        print(json.dumps(safe(dict(offset_percent=offset, theta_m=best['theta'],
            receiver_x0_m=best['x0'], area_mm2=best['coarse_area_mm2'],
            J=best['J'], status=result['status'], verified=result['verified'])),
            allow_nan=False), flush=True)
    summary = dict(algorithm_version=cfg['version'], config=cfg,
        environment=dict(python=sys.version, platform=platform.platform(),
            versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'clarabel')}),
        results=results)
    (args.output_dir / 'summary.json').write_text(
        json.dumps(safe(summary), indent=2, allow_nan=False) + '\n')
    print(f'Completed {len(results)} SCA run(s). Outputs: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
