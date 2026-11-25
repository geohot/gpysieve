"""Discharge unresolved anchor factorizations by *joint* possible order congruences."""
import argparse,json,subprocess
from collections import Counter
from functools import lru_cache
from itertools import product
from math import gcd,lcm,prod
from pathlib import Path
from sympy import divisors
from scale_shapes import indices
from search import mr,is_strong_lucas_prp

@lru_cache(None)
def cyclo(m):return ((1<<m)-1)//prod(cyclo(d) for d in divisors(m) if d<m)

def classes(cs,i,m):
    # Necessary K classes if the i-th prime has exact binary order m.
    c=cs[i];g=gcd(c,m)
    if 2%g:return []
    step=m//g;t=(2//g)*pow(c//g,-1,step)%step;M=lcm(8,2*m);rows=[];two=m&-m
    for k in range(t,M,step):
        if k%2:continue
        q=c*k-1;r=q%8
        if (r==3 and two!=2) or (r==5 and two!=4) or (r==7 and two!=1) or (r==1 and (q-1)%(2*two)):continue
        if prod(a*k-1 for a in cs)%m!=1:continue
        rows.append((k,M,two))
    return rows

def catalogue(src):
    records={r['m']:r for s in src.read_text().splitlines() if (r:=json.loads(s))['D']==2};cert=set()
    @lru_cache(None)
    def component(m):
        if m not in records:return (),True
        v=cyclo(m);ps=[]
        for p in records[m]['primes']:
            assert v%p==0;cert.add(p)
            while v%p==0:v//=p
            if p%m==1:ps.append(p)
        while v>1 and (g:=gcd(v,m))>1:v//=g
        return ps,v!=1
    @lru_cache(None)
    def opts(e):
        ps=set();missing=[]
        for m in divisors(e):
            if m<2:continue
            pp,unk=component(m);ps.update(pp)
            if unk:missing.append(m)
        return ps,missing
    return opts,cert

def certify(cert,gp):
    cmd=f'v={sorted(cert)};print(prod(i=1,#v,isprime(v[i])))\n';r=subprocess.run([gp,'-s','256M','-fq'],input=cmd,text=True,capture_output=True,check=True);assert not r.stderr and r.stdout.strip()=='1',(r.stderr[:1000],r.stdout[:200])

def refine(src,coarse,gp):
    opts,cert=catalogue(src);old=json.loads(coarse.read_text());holes=[];reason=Counter();survivors=[];hits=[];num=0;num64=0
    for cs in map(tuple,old['holes']):
        ks=set();rows=[];es=indices(cs)
        for i,e in enumerate(es):
            if not e:rows.append([(0,2,0)]);continue
            pp,unknown=opts(e);ks.update((p+1)//cs[i] for p in pp if (p+1)%(2*cs[i])==0);rows.append([row for m in unknown for row in classes(cs,i,m)])
        for k in sorted(ks):
            fs=[a*k-1 for a in cs];n=prod(fs)
            if fs[0]<3:continue
            num+=1;num64+=n>=2**64
            if any(e and pow(2,e,q)!=1 for e,q in zip(es,fs)) or not mr(n,[2]):continue
            row=dict(coefficients=cs,K=k,factors=fs,n=n);survivors.append(row)
            if is_strong_lucas_prp(n):hits.append(row)
        if not all(rows):reason['empty_order_branch']+=1;continue
        def compat(a,b):return (not a[2] or not b[2] or a[2]==b[2]) and (a[0]-b[0])%gcd(a[1],b[1])==0
        x,y,z=sorted(rows,key=len)
        if any(compat(a,b) and any(compat(a,c) and compat(b,c) for c in z) for a in x for b in y):holes.append(cs)
        else:reason['joint_incompatibility']+=1
    certify(cert,gp);return dict(box=old['box'],primitive_shapes=old['primitive_shapes'],completed=old['primitive_shapes']-len(holes),complete_box=min((max(t) for t in holes),default=old['box']+1)-1,discharged=reason,remaining=holes,additional_K=num,additional_K_above64=num64,additional_MR=survivors,hits=hits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('coarse',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--gp',default='gp');a=ap.parse_args();r=refine(a.src,a.coarse,a.gp);a.out.write_text(json.dumps(r,separators=(',',':'))+'\n');print({k:v for k,v in r.items() if k not in ('remaining','additional_MR','hits')},'MR',len(r['additional_MR']),'hits',len(r['hits']))
