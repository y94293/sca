"""Box-constrained SCA with local millimetre projection; no global feasibility table."""
import time
import itertools
import numpy as np
from scipy.special import expit
from sca_objective import Grid,mutual
from sca_engine import Engine
from tem_sca.native_sca import NativeConvexStep
from tem_sca.metrics import threshold_intervals


def box_mm(cfg):
    return np.rint(np.array(cfg['lower'])*1000).astype(int),np.rint(np.array(cfg['upper'])*1000).astype(int)


def local_candidates(target,current_mm,radius,cfg):
    lo,hi=box_mm(cfg);span=np.array(cfg['upper'])-cfg['lower']
    center=np.clip(np.rint(np.asarray(target)*1000).astype(int),lo,hi)
    current=np.asarray(current_mm,dtype=int)
    points={tuple(current)}
    for offset in itertools.product([-1,0,1],repeat=2):
        p=center+offset
        if np.any(p<lo) or np.any(p>hi):continue
        if np.linalg.norm((p-current)*.001/span)<=radius+1e-14:points.add(tuple(p))
    return sorted(points)


def execute(cfg,initial):
    tick=time.perf_counter()
    grid=Grid(np.linspace(*cfg['x_range'],cfg['nx']),np.linspace(*cfg['z_range'],cfg['nz']),cfg['length_scale'])
    engine=Engine(cfg,grid,None)
    setup=time.perf_counter()-tick
    run=solve(engine,initial)
    elapsed=time.perf_counter()-tick
    return dict(method='sca',seed=None,seconds=elapsed,setup_seconds=setup,search_seconds=elapsed-setup,
        engine_counts=engine.stats(),nominal_screening_geometries=0,receiver_positions=len(grid.x_axis),
        best=run['engineering'],status=run['status'],run=run)


def solve(engine,initial):
    tick=time.perf_counter();c=engine.cfg;g=engine.grid;s=c['local_sca']
    lo,hi=box_mm(c);lower=np.array(c['lower']);span=np.array(c['upper'])-lower
    current=tuple(np.clip(np.rint(np.asarray(initial)*1000).astype(int),lo,hi))
    actual_start=list(current);cf,cp=c['fixed_normalization_scales']
    conic=NativeConvexStep(g.weights,g.laplacian.shape[0],c['alpha'],c['beta'],c['gamma'],cf,cp)
    radius=s['radius0'];models={};visits={};trace=[];status='iteration_limit'
    m,value=engine.evaluate(current)

    def poll(reason,iteration):
        nonlocal current,m,value,radius
        best=current;mbest=m;vbest=value;checked=[]
        for offset in itertools.product([-1,0,1],repeat=2):
            p=np.array(current)+offset
            if not any(offset) or np.any(p<lo) or np.any(p>hi):continue
            mt,vt=engine.evaluate(tuple(p));checked.append(dict(theta=(p*.001).tolist(),J=vt['J']))
            if vt['J']>vbest['J']+s['gain_tolerance']:best=tuple(p);mbest=mt;vbest=vt
        changed=best!=current
        trace.append(dict(iteration=iteration,action=reason,theta=(np.array(current)*.001).tolist(),
            J=value['J'],trial_theta=(np.array(best)*.001).tolist(),trial_J=vbest['J'],
            actual=vbest['J']-value['J'],accepted=changed,checked=checked))
        if changed:current=best;m=mbest;value=vbest;radius=s['radius0']
        return changed

    for iteration in range(s['max_iterations']):
        state=(current,round(radius,12));visits[state]=visits.get(state,0)+1
        if visits[state]>2:
            if not poll('neighbour_poll_after_stall',iteration):status='engineering_neighbour_stall';break
            continue
        theta=np.array(current)*.001;u=(theta-lower)/span
        if current not in models:
            G=engine.jacobian(current)*span/c['threshold']
            rad=np.sqrt(m*m+c['absolute_smoothing']**2);sigmoid=expit(c['eta']*(1-rad))
            models[current]=dict(m=m,G=G,Lm=g.laplacian@m,LG=g.laplacian@G,
                area_gradient=G.T@(-c['eta']*g.weights*sigmoid*(1-sigmoid)*m/rad),u=u)
        data=models[current];sol=conic.solve(**data,radius=radius)
        log=dict(iteration=iteration,theta=theta.tolist(),J=value['J'],radius=radius,
                 solver={k:v for k,v in sol.items() if k!='d'})
        if sol['status'] not in ['optimal','optimal_inaccurate'] or sol.get('constraint_residual',float('inf'))>1e-6:
            log.update(action='solver_failure',accepted=False);trace.append(log);status='solver_failure';break
        target=lower+(u+sol['d'])*span
        candidates=local_candidates(target,current,radius,c)
        steps=(np.array(candidates)-np.array(current))*.001/span
        ld=data['LG']@steps.T;delta=data['G']@steps.T
        old=np.maximum(abs(m)-1,0);new=np.maximum(abs(m[:,None]+delta)-1,0)
        predictions=c['alpha']*(steps@data['area_gradient'])
        predictions-=c['beta']*(2*data['Lm']@ld+np.sum(ld*ld,axis=0))/len(data['Lm'])/cf
        predictions-=c['gamma']*(g.weights@((new-old[:,None])*(new+old[:,None])))/cp
        idx=int(np.argmax(predictions));candidate=tuple(candidates[idx]);d=steps[idx];pred=float(predictions[idx])
        norm=float(np.linalg.norm(d))
        log.update(continuous_trial_theta=target.tolist(),candidate_count=len(candidates),
                   trial_theta=(np.array(candidate)*.001).tolist(),projected_step_norm=norm,predicted=pred)
        if candidate==current or pred<=s['gain_tolerance']:
            log.update(action='nonpositive_local_model_step',accepted=False);trace.append(log)
            if not poll('neighbour_poll_after_quantization',iteration):status='engineering_neighbour_stall';break
            continue
        mt,vt=engine.evaluate(candidate);actual=vt['J']-value['J'];rho=actual/pred
        accepted=bool(actual>0 and rho>=.1);new_radius=radius
        if rho<.25:new_radius=max(s['radius_min'],radius*.5)
        elif rho>.75 and norm>=.9*radius:new_radius=min(s['radius_max'],radius*2)
        log.update(action='model_step',trial_J=vt['J'],actual=actual,rho=rho,
                   accepted=accepted,next_radius=new_radius)
        trace.append(log)
        if accepted:
            relative_theta=float(np.linalg.norm((np.array(candidate)-np.array(current))*.001)/max(1.,np.linalg.norm(theta)))
            relative_J=abs(actual)/max(1.,abs(value['J']))
            current=candidate;m=mt;value=vt
            if relative_theta<=s['theta_tolerance'] and relative_J<=s['objective_tolerance']:
                status='small_accepted_changes';break
        radius=new_radius

    zero_row=int(np.argmin(abs(g.z_axis)));assert abs(g.z_axis[zero_row])<1e-14
    indices=np.flatnonzero((g.x_axis>=c['lattice']['receiver_range_m'][0]-1e-14)&
                          (g.x_axis<=c['lattice']['receiver_range_m'][1]+1e-14))
    nominal=m.reshape(g.shape)[zero_row,indices]*c['threshold'];idx=int(np.argmin(abs(nominal)))
    receiver_calls=0
    def receiver_field(x):
        nonlocal receiver_calls
        receiver_calls+=1
        return mutual(np.array(current)*.001,x,0.,nphi=c['nphi'])
    intervals=threshold_intervals(g.x_axis[indices],nominal,c['threshold'],receiver_field)
    choices=[]
    for interval in intervals:
        valid=np.flatnonzero((g.x_axis[indices]>=interval['left'])&
                             (g.x_axis[indices]<=interval['right'])&(abs(nominal)<=c['threshold']))
        if len(valid):choices.append((interval,valid))
    receiver_selection=dict(rule='Midpoint of widest threshold interval in the allowed Rx range, rounded to nearest feasible 1 mm receiver position; ties use smaller x0.',interval=None)
    if choices:
        interval,valid=max(choices,key=lambda x:(x[0]['width'],-x[0]['left']))
        idx=int(valid[np.argmin(abs(g.x_axis[indices[valid]]-interval['centre']))])
        receiver_selection['interval']=interval
    best=dict(theta=(np.array(current)*.001).tolist(),x0=float(g.x_axis[indices[idx]]),M0=float(nominal[idx]),
              receiver_selection=receiver_selection,receiver_refinement_evaluations=receiver_calls,
              nominal_feasible=bool(abs(nominal[idx])<=c['threshold']),**value)
    calls=[t['solver'] for t in trace if 'solver' in t]
    return dict(engineering=best,initialization=dict(requested=list(initial),actual_mm=actual_start,
                rule='Box clipping and 1 mm rounding only; nominal feasibility not imposed on intermediate iterates'),
        status=status,trace=trace,solver_subproblems=len(calls),
        solver_iterations=sum(v.get('solver_iterations',0) for v in calls),
        solver_seconds=sum(v.get('solver_seconds',0.) for v in calls),model_builds=len(models),
        seconds=time.perf_counter()-tick,engine_counts=engine.stats(),
        evaluated_keys_mm=[list(k) for k in engine.values],continuous_stationarity_claimed=False)
