"""Signed filament-coil mutual inductance in henries for the SCA model."""
from functools import lru_cache
import numpy as np

MU0=4*np.pi*1e-7

@lru_cache(maxsize=16)
def _circle_nodes(nphi):
    if nphi<8: raise ValueError('At least eight angular nodes required')
    phi=np.arange(nphi)*2*np.pi/nphi
    return np.cos(phi),np.sin(phi)


def loop_mutual(side_x,side_y,x,y,z,radius=.1,nphi=64,chunk=256):
    """One rectangular source turn to one circular receiver turn.

    x,y,z describe the receiver centre relative to the source. Do not use for
    intersecting filaments. The source segment integral of 1/r is analytic.
    """
    if min(side_x,side_y,radius)<=0: raise ValueError('Coil sizes must be positive')
    x,y,z=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float),np.asarray(z,dtype=float))
    shape=x.shape;xf,yf,zf=x.ravel(),y.ravel(),z.ravel()
    result=np.empty(xf.size)
    co,si=_circle_nodes(int(nphi))
    a,b=side_x/2,side_y/2
    vertices=[(-a,-b),(a,-b),(a,b),(-a,b),(-a,-b)]
    for start in range(0,xf.size,chunk):
        stop=min(start+chunk,xf.size)
        xx=xf[start:stop,None]+radius*co
        yy=yf[start:stop,None]+radius*si
        zz=zf[start:stop,None]
        ax=np.zeros_like(xx);ay=np.zeros_like(yy)
        for (x0,y0),(x1,y1) in zip(vertices[:-1],vertices[1:]):
            length=np.hypot(x1-x0,y1-y0)
            r0=np.sqrt((xx-x0)**2+(yy-y0)**2+zz**2)
            r1=np.sqrt((xx-x1)**2+(yy-y1)**2+zz**2)
            gap=r0+r1-length
            if np.any(gap<=0): raise ValueError('Receiver intersects source filament')
            line=np.log1p(2*length/gap)
            ax += ((x1-x0)/length)*line
            ay += ((y1-y0)/length)*line
        result[start:stop]=MU0/(4*np.pi)*2*np.pi*radius*np.mean(-si*ax+co*ay,axis=1)
    return result.reshape(shape)


def mutual_components(theta,x,dz,y=0.,nphi=64):
    """Paper turns/sizes. Cx at H, Rx at H+dz; x is absolute lateral Rx centre."""
    ac,height=np.asarray(theta,dtype=float)
    x,dz,y=np.broadcast_arrays(np.asarray(x),np.asarray(dz),np.asarray(y))
    mt=40*loop_mutual(.5,.5,x,y,height+dz,nphi=nphi)
    mc=10*loop_mutual(ac,ac,x,y,dz,nphi=nphi)
    return mt,mc


def mutual(theta,x,dz,y=0.,nphi=64):
    mt,mc=mutual_components(theta,x,dz,y=y,nphi=nphi)
    return mt+mc
