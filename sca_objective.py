"""Normalized SCA objective and spatial-grid definitions."""
import numpy as np
from scipy.special import expit
from tem_sca.physics import mutual
from tem_sca.metrics import Grid

def terms(r,g,cfg):
    cf,cp=cfg['fixed_normalization_scales']
    smooth=float(g.weights@expit(cfg['eta']*(1-np.sqrt(r*r+cfg['absolute_smoothing']**2))))
    flat=float(np.mean((g.laplacian@r)**2))/cf
    penalty=float(g.weights@np.maximum(abs(r)-1,0)**2)/cp
    return dict(J=cfg['alpha']*smooth-cfg['beta']*flat-cfg['gamma']*penalty,
                smooth_fraction=smooth,flat_normalized=flat,penalty_normalized=penalty,
                coarse_area_mm2=g.threshold_area(r,1.)*1e6)
