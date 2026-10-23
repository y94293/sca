#!/usr/bin/env python3
"""Portable exhaustive 1 mm grid-scan reproduction for the NCPC benchmark.

This repository version uses the public physics/objective modules in the root
repository. It follows the manuscript benchmark protocol:

- aC = 0.300 ... 0.500 m in 1 mm increments (201 values)
- HTR = 0.200 ... 1.000 m in 1 mm increments (801 values)
- 161,001 parameter pairs
- nominal-feasibility screening at the 81 permitted receiver positions
- full-field/objective evaluation for every nominally feasible pair
- no cross-candidate reuse of design-dependent full Tx/Cx fields or objectives
- strictly serial repetitions, one numerical-library thread

Absolute wall time depends on the machine and Python/NumPy implementation.
The recorded manuscript timing statistics are stored in grid_serial20_summary.json.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'

import argparse, csv, json, statistics, time
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))

from tem_sca.metrics import Grid
from tem_sca.physics import loop_mutual, _circle_nodes
from sca_objective import terms

A_AXIS_MM=np.arange(300,501,dtype=int)
H_AXIS_MM=np.arange(200,1001,dtype=int)


def nominal_feasible_keys(cfg):
    x=np.linspace(*cfg['x_range'],cfg['nx'])
    n=cfg['nphi']; threshold=cfg['threshold']
    # Screening-only 1-D caches are permitted by the benchmark protocol.
    cx={int(a):10*loop_mutual(a*.001,a*.001,x,0.,0.,nphi=n) for a in A_AXIS_MM}
    tx={int(h):40*loop_mutual(.5,.5,x,0.,h*.001,nphi=n) for h in H_AXIS_MM}
    keys=[]
    for a in A_AXIS_MM:
        ca=cx[int(a)]
        for h in H_AXIS_MM:
            if np.any(np.abs(tx[int(h)]+ca)<=threshold):
                keys.append((int(a),int(h)))
    return keys


def evaluate_candidate(key,grid,cfg):
    a_mm,h_mm=key; a=a_mm*.001; h=h_mm*.001; n=cfg['nphi']
    # Deliberately rebuild both design-dependent full fields for every candidate.
    cx=10*loop_mutual(a,a,grid.x,0.,grid.z,nphi=n)
    tx=40*loop_mutual(.5,.5,grid.x,0.,h+grid.z,nphi=n)
    return terms((tx+cx)/cfg['threshold'],grid,cfg)


def one_run(cfg):
    t0=time.perf_counter(); c0=time.process_time()
    grid=Grid(np.linspace(*cfg['x_range'],cfg['nx']),
              np.linspace(*cfg['z_range'],cfg['nz']),cfg['length_scale'])
    feasible=nominal_feasible_keys(cfg)
    best_key=None; best=None
    for key in feasible:
        obj=evaluate_candidate(key,grid,cfg)
        if best is None or obj['J']>best['J']:
            best_key,best=key,obj
    return dict(seconds=time.perf_counter()-t0,
                process_cpu_seconds=time.process_time()-c0,
                candidate_pairs=int(len(A_AXIS_MM)*len(H_AXIS_MM)),
                nominally_feasible_candidates=len(feasible),
                full_objective_evaluations=len(feasible),
                best_key_mm=list(best_key),best_aC_m=best_key[0]*.001,
                best_H_m=best_key[1]*.001,best_J=float(best['J']),
                best_area_mm2=float(best['coarse_area_mm2']))


def stats(x):
    q=statistics.quantiles(x,n=4,method='inclusive') if len(x)>1 else [x[0]]*3
    return dict(min=min(x),q1=q[0],median=statistics.median(x),q3=q[2],max=max(x),
                mean=statistics.mean(x),stdev=statistics.stdev(x) if len(x)>1 else 0.0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats',type=int,default=1,help='Use 20 to reproduce the timing protocol.')
    p.add_argument('--output-dir',type=Path,default=None)
    args=p.parse_args()
    cfg=json.loads((ROOT/'config/sca_settings.json').read_text())
    _circle_nodes(cfg['nphi'])
    out=args.output_dir or HERE/'results'/('grid1mm_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i in range(1,args.repeats+1):
        r=one_run(cfg);r['repeat']=i;rows.append(r)
        (out/f'repeat_{i:02d}.json').write_text(json.dumps(r,indent=2)+'\n')
        print(f"[{i:02d}/{args.repeats:02d}] {r['seconds']:.3f} s  best={r['best_key_mm']}  area={r['best_area_mm2']:.3f}",flush=True)
    fields=['repeat','seconds','process_cpu_seconds','candidate_pairs','nominally_feasible_candidates',
            'full_objective_evaluations','best_aC_m','best_H_m','best_J','best_area_mm2']
    with (out/'all_grid_timings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:r[k] for k in fields})
    assert len({tuple(r['best_key_mm']) for r in rows})==1
    summary=dict(complete=True,execution_mode='strictly serial',repeats=len(rows),threads_per_run=1,
                 candidate_pairs=rows[0]['candidate_pairs'],nominally_feasible_candidates=rows[0]['nominally_feasible_candidates'],
                 full_objective_evaluations=rows[0]['full_objective_evaluations'],best_key_mm=rows[0]['best_key_mm'],
                 best_aC_m=rows[0]['best_aC_m'],best_H_m=rows[0]['best_H_m'],best_J=rows[0]['best_J'],
                 best_area_mm2=rows[0]['best_area_mm2'],wall_seconds=stats([r['seconds'] for r in rows]),
                 process_cpu_seconds=stats([r['process_cpu_seconds'] for r in rows]))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    main()
