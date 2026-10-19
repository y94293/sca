"""Native CLARABEL convex subproblem used by the reported local SCA."""
import numpy as np
import clarabel
from scipy import sparse

class NativeConvexStep:
    """Exact quadratic compression and safe hinge-branch aggregation.

    On ||d||<=radius, m_i+G_i d belongs to [m_i-radius*||G_i||,
    m_i+radius*||G_i||]. A squared hinge wholly inside one of its three
    branches is exactly quadratic (or zero); only ambiguous rows need slacks.
    d=radius*y normalizes even very small trust balls to unit radius.
    """
    def __init__(self,weights,nflat,alpha,beta,gamma,flat_scale,penalty_scale,screening=True):
        self.weights=np.asarray(weights);self.nflat=nflat
        self.alpha=alpha;self.beta=beta;self.gamma=gamma
        self.fs=flat_scale;self.ps=penalty_scale;self.screening=screening

    def solve(self,m,G,Lm,LG,area_gradient,u,radius,proximal=None):
        if not np.isfinite(radius) or radius<=0:
            raise ValueError('Trust radius must be positive')
        Q=np.zeros((2,2)) if proximal is None else np.asarray(proximal,float)
        if Q.shape!=(2,2) or not np.all(np.isfinite(Q)) or not np.allclose(Q,Q.T):
            raise ValueError('Proximal curvature must be symmetric 2x2')
        eigen=np.linalg.eigvalsh(Q)
        if eigen[0]<-1e-10*max(1.,np.max(abs(eigen))):
            raise ValueError('Proximal curvature must be positive semidefinite')
        if eigen[0]<0:Q=Q-np.eye(2)*eigen[0]
        m=np.asarray(m);G=np.asarray(G);u=np.asarray(u)
        span=radius*np.linalg.norm(G,axis=1)
        # A small outward rounding guard makes classification conservative.
        guard=32*np.finfo(float).eps*(1+abs(m)+span)
        pos=(m-span>1+guard) if self.screening else np.zeros(len(m),bool)
        neg=(m+span<-1-guard) if self.screening else np.zeros(len(m),bool)
        inactive=(abs(m)+span<1-guard) if self.screening else np.zeros(len(m),bool)
        fixed=pos|neg
        uncertain=~(fixed|inactive) if self.gamma else np.zeros(len(m),bool)
        k=int(uncertain.sum());nvar=2+k
        H=2*self.beta*(LG.T@LG)/self.nflat/self.fs+Q
        q=2*self.beta*(LG.T@Lm)/self.nflat/self.fs-self.alpha*area_gradient
        if np.any(fixed) and self.gamma:
            gf=G[fixed];wf=self.weights[fixed]
            residual=m[fixed]-np.where(pos[fixed],1.,-1.)
            H=H+2*self.gamma*(gf.T@(wf[:,None]*gf))/self.ps
            q=q+2*self.gamma*(gf.T@(wf*residual))/self.ps
        # Every objective coefficient is divided by the same positive number.
        scale=max(float(np.max(abs(q)))*radius,float(np.max(abs(H)))*radius**2,1e-20)
        P=sparse.block_diag((sparse.csc_matrix(H*radius**2/scale),
                             sparse.diags(2*self.gamma*self.weights[uncertain]*radius**2/scale)),format='csc')
        P=sparse.triu(P).tocsc()
        linear=np.r_[q*radius/scale,np.zeros(k)]
        matrices=[];rhs=[]
        if k:
            gu=G[uncertain]/np.sqrt(self.ps)
            eye=sparse.eye(k,format='csc')
            matrices.extend([sparse.hstack((sparse.csc_matrix(gu),-eye)),
                             sparse.hstack((sparse.csc_matrix(-gu),-eye)),
                             sparse.hstack((sparse.csc_matrix((k,2)),-eye))])
            rhs.extend([(1-m[uncertain])/(radius*np.sqrt(self.ps)),
                        (1+m[uncertain])/(radius*np.sqrt(self.ps)),np.zeros(k)])
        boxes=sparse.hstack((sparse.csc_matrix(np.vstack((np.eye(2),-np.eye(2)))),sparse.csc_matrix((4,k))))
        matrices.append(boxes);rhs.append(np.r_[np.minimum((1-u)/radius,1.),np.minimum(u/radius,1.)])
        soc=sparse.csc_matrix(([-1.,-1.],([1,2],[0,1])),shape=(3,nvar))
        matrices.append(soc);rhs.append(np.array([1.,0.,0.]))
        A=sparse.vstack(matrices,format='csc');b=np.concatenate(rhs)
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=150
        settings.max_threads=1;settings.direct_solve_method='qdldl'
        settings.tol_gap_abs=1e-11;settings.tol_gap_rel=1e-11;settings.tol_feas=1e-11
        solver=clarabel.DefaultSolver(P,linear,A,b,[clarabel.NonnegativeConeT(3*k+4),clarabel.SecondOrderConeT(3)],settings)
        solution=solver.solve();status=str(solution.status)
        if status not in ('Solved','AlmostSolved'):
            return dict(status=status,variables=nvar)
        x=np.asarray(solution.x);raw=radius*x[:2]
        d=np.clip(u+raw,0,1)-u
        # Avoid subtractive loss of tiny interior steps when no clipping is needed.
        if np.all(u+raw>=0) and np.all(u+raw<=1):d=raw
        cone_res=max(0.,float(np.max((A@x-b)[:3*k+4])),float(np.linalg.norm(x[:2])-1))
        delta=G@d;old=np.maximum(abs(m)-1,0);new=np.maximum(abs(m+delta)-1,0)
        ld=LG@d
        predicted=self.alpha*np.dot(area_gradient,d)-self.beta*(2*np.dot(Lm,ld)+np.dot(ld,ld))/self.nflat/self.fs
        predicted-=self.gamma*np.dot(self.weights,(new-old)*(new+old))/self.ps+.5*np.dot(d,Q@d)
        return dict(status='optimal' if status=='Solved' else 'optimal_inaccurate',d=d,
                    predicted=float(predicted),constraint_residual=cone_res,
                    solver_iterations=solution.iterations,solver_seconds=solution.solve_time,
                    solver_name='CLARABEL',raw_status=status,primal_residual=float(solution.r_prim),
                    dual_residual=float(solution.r_dual),primal_objective=float(solution.obj_val),
                    dual_objective=float(solution.obj_val_dual),
                    variables=nvar,fixed_penalties=int(fixed.sum()),inactive_penalties=int(inactive.sum()),
                    uncertain_penalties=k,objective_scale=scale)
