"""Two nested Holder bounds on binary subset Fourier coefficients (no subset search)."""
import argparse,json,random
from pathlib import Path
import numpy as np
from quotient_cover import Projection,insert,load


def pack(vs,indices):
    full={}
    for i in indices:insert(full,vs[i])
    groups=[]
    while indices:
        b={};grp=[];rest=[]
        for i in indices:
            if insert(b,vs[i]):grp.append(i)
            else:rest.append(i)
        if len(b)!=len(full):break
        groups.append(grp);indices=rest
    return groups


def design(d,tag='four-lists',n4=101):
    if tag=='four-lists':
        blocks,A,B,_=load(d);ps=sum(blocks,[])
    else:
        g=json.loads((d/'trim.json').read_text())[tag];A,B=g['A'],g['B'];ps=list(map(int,(d/f'trim-{tag}.txt').read_text().split()));random.Random(1).shuffle(ps)
    parts=[[2187,729],[2401,7]];P=Projection(A,B,parts);s=[P.sig(p) for p in ps];idx=[i for i,q in enumerate(P.qs,1) if q%4==1][:n4];u=[sum(((v>>i)&1)<<j for j,i in enumerate(idx)) for v in s];b=pack(s,list(range(len(s))));c=pack(u,sum(b,[]));return ps,P,b,c


def estimate(d):
    ps,P,b,c=design(d);r=len(b);k=len(c);h=P.h;print('layers',r,len(b[0]),k,len(c[0]),flush=True); y=np.array([P.odd(ps[i]) for i in sum(c,[])]);theta=np.arange(h)*np.pi/h
    def f(x):return (abs(np.cos(x))**r+abs(np.sin(x))**r)**(1/r)
    cost=np.log2(f(theta)**k+f(theta+np.pi/4)**k)/k
    bounds=[float(cost[(a*y)%h].sum()) for a in range(h)];v=np.exp2(bounds);print('trivial odd bound',v[0],'other sum',sum(v[1:]),'max nonzero bits',max(bounds[1:]),flush=True);return bounds

def certify(d,n8=24,n4=101,tag='four-lists'):
    from mpmath import iv
    from mpmath.libmp import to_int,round_ceiling
    from collections import Counter
    iv.dps=35
    ps,P,b,c=design(d,tag,n4);quartics=[q for q in P.qs if q%4==1][:n4];idx=[(i,q) for i,q in enumerate(P.qs,1) if q in quartics and q%8==1][:n8]
    v=[sum(((P.sig(p)>>i)&1)<<j for j,(i,q) in enumerate(idx)) for p in ps];groups=pack(v,sum(c,[])) if n8 else [];used=sum(groups or c,[]);y=np.array([P.odd(ps[i]) for i in used]);r,k,l=len(b),len(c),len(groups);h=P.h;I=iv.mpf;scale=2**20
    assert len(c[0])==n4 and (not n8 or len(groups[0])==n8)
    # All the retained Holder factors are periodic. Translate the same exact
    # upper-bound table by the exact grid offsets pi/4 and pi/8.
    grid=8*h
    vals=[(abs(iv.cos(iv.pi*i/grid))**r+abs(iv.sin(iv.pi*i/grid))**r)**(I(1)/r) for i in range(2*h+1)]
    def f(i):
        i=i%(4*h);return vals[min(i,4*h-i)]
    fs=[(f(i)**k+f(i+2*h)**k)**(I(1)/k) for i in range(h+1)]
    def F(i):
        i=i%(2*h);return fs[min(i,2*h-i)]
    def weight(i):return (F(i)**l+F(i+h)**l)**(I(1)/l) if n8 else F(i)
    def ceil_iv(x):return int(to_int(x._mpi_[1],round_ceiling))
    # F2 can exceed 1. Its arguments for *all* remaining characters
    # lie on the grid pi/(8h), so charge dropped F2 factors at the
    # grid maximum. F1<=1 analytically since r>=2.
    assert r>=2
    missing=len(sum(c,[]))-len(used);overhead=missing*max(0,max(ceil_iv(scale*iv.log(x,2)) for x in fs)) if missing else 0
    cost=np.array([ceil_iv(scale*iv.log(weight(8*j),2)) for j in range(h)],dtype=np.int64);assert max(abs(int(x)) for x in cost)*len(y)<2**63
    # Aggregate upward-rounded log2 weights into 1/256-bit bins.
    bins=Counter(-((-(overhead+int(cost[(a*y)%h].sum()))*256)//scale) for a in range(h));upper=sum(n*I(2)**(I(i)/256) for i,n in bins.items());millis=ceil_iv(upper*1000)
    assert millis<2000,'no nontrivial guarantee from this bound';return dict(pool=tag,quartics=quartics,octics=[q for i,q in idx],bases=[r,k,l],dimensions=[len(b[0]),len(c[0]),n8],odd_order=h,sign_rank=len(b[0]),fourier_sum_upper_thousandths=millis,quotient_size=2**(len(b[0])+n4+n8)*h,prime_count=len(ps),histogram=dict(sorted(bins.items())))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir',type=Path,default=Path('runs/lift12/trim'));ap.add_argument('--out',type=Path);ap.add_argument('--octics',type=int,default=24);ap.add_argument('--quartics',type=int,default=101);ap.add_argument('--tag',default='four-lists');a=ap.parse_args()
    if a.out:
        result=certify(a.dir,a.octics,a.quartics,a.tag);a.out.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='histogram'})
    else:estimate(a.dir)
