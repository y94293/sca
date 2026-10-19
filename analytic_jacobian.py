"""Analytic parameter derivatives of the rectangular-to-circular Neumann model."""
import numpy as np
from tem_sca.physics import MU0,_circle_nodes

def loop_derivative(side,x,z,parameter,nphi=64,radius=.1,chunk=256):
    x,z=np.broadcast_arrays(np.asarray(x,float),np.asarray(z,float))
    shape=x.shape;xf=x.ravel();zf=z.ravel();out=np.empty(xf.size)
    co,si=_circle_nodes(nphi)
    signs=[(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5),(-.5,-.5)]
    for start in range(0,xf.size,chunk):
        stop=min(start+chunk,xf.size)
        xx=xf[start:stop,None]+radius*co;yy=radius*si;zz=zf[start:stop,None]
        ax=np.zeros_like(xx);ay=np.zeros_like(xx)
        for (sx0,sy0),(sx1,sy1) in zip(signs[:-1],signs[1:]):
            x0,y0=side*sx0,side*sy0;x1,y1=side*sx1,side*sy1
            r0=np.sqrt((xx-x0)**2+(yy-y0)**2+zz**2)
            r1=np.sqrt((xx-x1)**2+(yy-y1)**2+zz**2)
            total=r0+r1
            if parameter=='side':
                ds=(-(xx-x0)*sx0-(yy-y0)*sy0)/r0+(-(xx-x1)*sx1-(yy-y1)*sy1)/r1
                dl=1.
            elif parameter=='z':ds=zz*(1/r0+1/r1);dl=0.
            else:raise ValueError(parameter)
            derivative=2*(total*dl-side*ds)/((total-side)*(total+side))
            ax+=(sx1-sx0)*derivative;ay+=(sy1-sy0)*derivative
        out[start:stop]=MU0/(4*np.pi)*2*np.pi*radius*np.mean(-si*ax+co*ay,axis=1)
    return out.reshape(shape)

def parameter_jacobian(theta,x,dz,nphi=64):
    ac,H=theta
    da=10*loop_derivative(ac,x,dz,'side',nphi)
    dh=40*loop_derivative(.5,x,H+np.asarray(dz),'z',nphi)
    return np.stack([da,dh],axis=-1)
