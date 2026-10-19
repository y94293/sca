"""Per-run physical-field and derivative caches for SCA."""
import numpy as np
from tem_sca.physics import loop_mutual
from sca_objective import terms
from analytic_jacobian import loop_derivative

class Engine:
    def __init__(self,cfg,grid,feasible):
        self.cfg=cfg;self.grid=grid;self.feasible=feasible
        self.tx={};self.cx={};self.dtx={};self.dcx={};self.values={};self.gradients={}

    def evaluate(self,key):
        key=tuple(int(k) for k in key)
        if key not in self.values:
            a,h=np.array(key)*.001;g=self.grid;n=self.cfg['nphi']
            if key[0] not in self.cx:
                self.cx[key[0]]=10*loop_mutual(a,a,g.x,0.,g.z,nphi=n)
            if key[1] not in self.tx:
                self.tx[key[1]]=40*loop_mutual(.5,.5,g.x,0.,h+g.z,nphi=n)
            m=(self.tx[key[1]]+self.cx[key[0]])/self.cfg['threshold']
            self.values[key]=(m,terms(m,g,self.cfg))
        return self.values[key]

    def jacobian(self,key):
        key=tuple(int(k) for k in key)
        if key not in self.gradients:
            a,h=np.array(key)*.001;g=self.grid;n=self.cfg['nphi']
            if key[0] not in self.dcx:
                self.dcx[key[0]]=10*loop_derivative(a,g.x,g.z,'side',n)
            if key[1] not in self.dtx:
                self.dtx[key[1]]=40*loop_derivative(.5,g.x,h+g.z,'z',n)
            self.gradients[key]=np.stack([self.dcx[key[0]],self.dtx[key[1]]],axis=-1)
        return self.gradients[key]

    def stats(self):
        return dict(full_objective_evaluations=len(self.values),tx_full_fields=len(self.tx),
                    cx_full_fields=len(self.cx),tx_derivative_fields=len(self.dtx),
                    cx_derivative_fields=len(self.dcx),parameter_jacobians=len(self.gradients))
