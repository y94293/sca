"""Spatial grid and connected signed-field threshold intervals."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.optimize import brentq

@dataclass
class Grid:
    x_axis: np.ndarray
    z_axis: np.ndarray
    length_scale: float=.01

    def __post_init__(self):
        self.x_axis=np.asarray(self.x_axis,float);self.z_axis=np.asarray(self.z_axis,float)
        if min(len(self.x_axis),len(self.z_axis))<3:raise ValueError('Three points per axis required')
        dx=np.diff(self.x_axis);dz=np.diff(self.z_axis)
        if not (np.all(dx>0) and np.all(dz>0) and np.allclose(dx,dx[0]) and np.allclose(dz,dz[0])):
            raise ValueError('Grid axes must be increasing and uniform')
        self.nx=len(self.x_axis);self.nz=len(self.z_axis);self.shape=(self.nz,self.nx)
        xx,zz=np.meshgrid(self.x_axis,self.z_axis);self.x=xx.ravel();self.z=zz.ravel()
        wx=np.ones(self.nx);wz=np.ones(self.nz);wx[[0,-1]]=.5;wz[[0,-1]]=.5
        self.raw_weights=np.outer(wz,wx).ravel()*dx[0]*dz[0]
        self.area=(self.x_axis[-1]-self.x_axis[0])*(self.z_axis[-1]-self.z_axis[0])
        self.weights=self.raw_weights/self.area
        row=[];col=[];data=[];k=0
        sx=self.length_scale**2/dx[0]**2;sz=self.length_scale**2/dz[0]**2
        for j in range(1,self.nz-1):
            for i in range(1,self.nx-1):
                c=j*self.nx+i
                for index,value in [(c,-2*sx-2*sz),(c-1,sx),(c+1,sx),(c-self.nx,sz),(c+self.nx,sz)]:
                    row.append(k);col.append(index);data.append(value)
                k+=1
        self.laplacian=sparse.csr_matrix((data,(row,col)),shape=(k,self.x.size))

    def threshold_area(self,values,threshold):
        return float(np.dot(self.raw_weights,np.abs(np.asarray(values).ravel())<=threshold))


def threshold_intervals(x,values,threshold,fun=None):
    """Intervals in the piecewise-linear SIGNED field; optionally refine crossings.

    Signed crossings are retained even if neither segment endpoint satisfies the
    magnitude threshold. A truncated interval is never labelled a complete width.
    """
    x=np.asarray(x,float);v=np.asarray(values,float)
    if x.ndim!=1 or x.shape!=v.shape or len(x)<2 or not np.all(np.diff(x)>0):raise ValueError('Invalid profile')
    if not np.all(np.isfinite(v)) or threshold<=0:raise ValueError('Invalid values/threshold')
    pieces=[]
    for j,(a,b) in enumerate(zip(v[:-1],v[1:])):
        slope=b-a
        if slope==0:
            if abs(a)>threshold:continue
            t0,t1=0.,1.
        else:
            t0,t1=sorted(((-threshold-a)/slope,(threshold-a)/slope))
            t0=max(0.,t0);t1=min(1.,t1)
            if t1<t0:continue
        left=x[j]+t0*(x[j+1]-x[j]);right=x[j]+t1*(x[j+1]-x[j])
        if fun is not None:
            if 0<t0<1:
                target=np.copysign(threshold,a+t0*slope)
                left=brentq(lambda xx:float(fun(xx))-target,x[j],x[j+1],xtol=1e-13)
            if 0<t1<1:
                target=np.copysign(threshold,a+t1*slope)
                right=brentq(lambda xx:float(fun(xx))-target,x[j],x[j+1],xtol=1e-13)
        if pieces and abs(left-pieces[-1][1])<1e-11:
            pieces[-1][1]=right
        else:pieces.append([left,right])
    out=[]
    for left,right in pieces:
        lt=abs(left-x[0])<1e-12;rt=abs(right-x[-1])<1e-12
        out.append(dict(left=float(left),right=float(right),width=float(right-left),centre=float((left+right)/2),left_truncated=lt,right_truncated=rt,truncated=lt or rt))
    return out
