"""Construct an affine Boolean flat inside a fixed old quotient fiber, with controllable quartic image."""
import argparse,json,random
from math import prod
from pathlib import Path
from quotient_cover import Projection,insert,rotate_decode,solve,target


def indices(v):
    while v:
        b=v&-v;yield b.bit_length()-1;v-=b


def nullbasis(rows,n):
    piv={}
    for x in rows:insert(piv,x)
    def lift(x):
        for j,(v,_) in sorted(piv.items()):
            if (v&x).bit_count()%2:x^=1<<j
        return x
    return [lift(1<<j) for j in range(n) if j not in piv]


def prepare(cert,nq):
    P=Projection(cert['A'],cert['B'],cert['parts']);t,_=target(cert['A'],cert['B']);sw=[]
    for c in cert['blocks']:
        for ab in c['switches']:sw.append(tuple(prod(c['primes'][i] for i in side)%P.m for side in ab))
    base=prod(a for a,b in sw)%P.m;bank=cert['blocks'][0];ids=rotate_decode(bank['reserve'],[P.odd(p) for p in bank['primes']],P.h,(P.odd(t)-P.odd(base))%P.h);res=prod(bank['primes'][i] for i in ids);base=base*res%P.m
    signs=[P.sig(a)^P.sig(b) for a,b in sw];basis={}
    for j,v in enumerate(signs):insert(basis,v,1<<j)
    start=solve(basis,P.sig(t)^P.sig(base));n=len(sw);row=lambda bits:sum((v&1)<<j for j,v in enumerate(bits));constraints=[row(v>>i for v in signs) for i in range(len(P.qs)+1)];qset=[q for q in P.qs if q%4==1][:nq];lows=[];highs=[];baseclass=base
    for j in indices(start):a,b=sw[j];baseclass=baseclass*b*pow(a,-1,P.m)%P.m
    rhs=0
    for i,q in enumerate(qset):
        # Any fixed orientation of the quartic roots works. A matched sign means
        # only the ratio +1 or -1 remains, independent of the orientation.
        root=next(z for g in range(2,q) if (z:=pow(g,(q-1)//4,q)) not in (1,q-1));tab={pow(root,j,q):j for j in range(4)}
        vs=[(1-2*((start>>j)&1))*tab[pow(b*pow(a,-1,q)%q,(q-1)//4,q)]%4 for j,(a,b) in enumerate(sw)];lows.append(row(vs));highs.append(row(v>>1 for v in vs));z=pow(t*pow(baseclass,-1,q)%q,(q-1)//4,q);assert z in (1,q-1);rhs|=int(z!=1)<<i
    return dict(P=P,t=t,start=start,sw=sw,n=n,ids=ids,qset=qset,cons=constraints,lows=lows,highs=highs,rhs=rhs,res=res)


def construct(c,nq):
    p=prepare(c,nq);constraints=p['cons'][:];rng=random.Random(777);basis={};dirs=[]
    for k in range(nq):
        ns=nullbasis(constraints,p['n']);ok=False
        for _ in range(40):
            v=0
            for i in indices(rng.getrandbits(len(ns))):v^=ns[i]
            change=0
            for i,(lo,hi) in enumerate(zip(p['lows'],p['highs'])):
                total=(lo&v).bit_count()+2*(hi&v).bit_count();assert total%2==0;change|=((total>>1)&1)<<i
            if insert(basis,change,1<<k):ok=True;break
        if not ok:raise ValueError(f'rank stalled at {k} directions')
        dirs.append(v);constraints.extend(lo&v for lo in p['lows'])
    mask=p['start'];solution=solve(basis,p['rhs'])
    for j in indices(solution):mask^=dirs[j]
    selected=[c['blocks'][0]['primes'][i] for i in p['ids']];j=0
    for b in c['blocks']:
        for ab in b['switches']:
            selected.extend(b['primes'][i] for i in ab[(mask>>j)&1]);j+=1
    assert len(set(selected))==len(selected);m=p['P'].m;w=prod(selected)%m;t=p['t'];assert p['P'].sig(w)==p['P'].sig(t) and p['P'].odd(w)==p['P'].odd(t) and all(pow(w,(q-1)//4,q)==pow(t,(q-1)//4,q) for q in p['qset']);return dict(qs=p['qset'],reserve=p['ids'],start=hex(p['start']),directions=[hex(v) for v in dirs],target_selection=sorted(selected),full_target=w==t)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('certificate',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--bits',type=int,default=24);a=ap.parse_args();r=construct(json.loads(a.certificate.read_text()),a.bits);a.out.write_text(json.dumps(r,indent=2)+'\n');print(len(r['directions']),len(r['target_selection']),r['full_target'])
