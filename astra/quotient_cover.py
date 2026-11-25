"""Constructive joint zero-one coverage of quadratic characters and an odd CRT quotient."""
import argparse
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from itertools import chain
import json
from math import gcd,lcm,prod
from pathlib import Path
import random
from sympy import factorint, primitive_root
from search import target


def insert(basis,v,mask=0):
    while v:
        j=v.bit_length()-1
        if j not in basis:
            basis[j]=(v,mask);return True
        w,a=basis[j];v^=w;mask^=a
    return False


def solve(basis,v):
    mask=0
    while v:
        w,a=basis[v.bit_length()-1];v^=w;mask^=a
    return mask


class Projection:
    def __init__(self,A,B,parts):
        self.m=lcm(A,B);self.parts=parts;self.qs=sorted(q for q in factorint(self.m) if q!=2)
        assert self.m%4==0 and A%5==0 and B%7==0
        self.h=prod(d for _,d in parts);self.tables=[]
        for modulus,d in parts:
            assert self.m%modulus==0 and len(fs:=factorint(modulus))==1 and d>1 and d%2==1
            q=next(iter(fs));phi=modulus//q*(q-1); assert phi%d==0 and gcd(self.h//d,d)==1
            g=int(primitive_root(modulus));base=pow(g,phi//d,modulus);x=1;table={}
            for i in range(d):table[x]=i;x=x*base%modulus
            assert len(table)==d and x==1
            self.tables.append((modulus,phi//d,table,(self.h//d)*pow(self.h//d,-1,d)))

    @lru_cache(maxsize=12000)
    def sig(self,p):
        assert gcd(p,self.m)==1
        # Component 0: mod-4 sign. Every other bit: Legendre character modulo q.
        return int(p%4==3)+sum(int(pow(p%q,(q-1)//2,q)==q-1)<<i for i,q in enumerate(self.qs,1))

    @lru_cache(maxsize=12000)
    def odd(self,p):
        return sum(table[pow(p%modulus,e,modulus)]*mult for modulus,e,table,mult in self.tables)%self.h


def prefixes(indices,y,h):
    mask=(1<<h)-1;states=[1]
    for i in indices:
        r=states[-1];d=y[i];states.append(r|(((r<<d)|(r>>(h-d)))&mask))
    return states


def rotate_decode(indices,y,h,t):
    st=prefixes(indices,y,h);assert st[-1]>>t&1
    ids=[]
    for i,prev in zip(reversed(indices),reversed(st[:-1])):
        if not (prev>>t&1):ids.append(i);t=(t-y[i])%h
    assert not t;return ids


def xor(ids,col):
    z=0
    for i in ids:z^=col[i]
    return z


def build(ps,projection,seed):
    h=projection.h;x=[projection.sig(p) for p in ps];y=[projection.odd(p) for p in ps];whole={}
    for v in x:insert(whole,v)
    rng=random.Random(seed);pool=list(range(len(ps)));rng.shuffle(pool)
    # Reserve enough columns to generate ALL h residues with binary coefficients.
    reserve=[];bits=1;mask=(1<<h)-1
    while bits!=mask:
        if not pool:return None
        i=pool.pop();reserve.append(i);d=y[i];bits|=((bits<<d)|(bits>>(h-d)))&mask
    basis={};switches=[]
    def accept(a,b):
        ids=a+b
        if len(ids)!=len(set(ids)):return False
        if not insert(basis,xor(ids,x)):return False
        switches.append([a,b]);used=set(ids);pool[:]=[j for j in pool if j not in used];return True
    def take():
        by=defaultdict(list)
        for i in pool:by[y[i]].append(i)
        for i in by.get(0,[]):
            if accept([], [i]):return True
        for i in pool:
            for j in by.get(y[i],()):
                if i<j and accept([i],[j]):return True
            for j in by.get(-y[i]%h,()):
                if i<j and accept([],[i,j]):return True
        for a,i in enumerate(pool):
            for j in pool[a+1:]:
                for k in by.get((y[i]+y[j])%h,()):
                    if accept([k],[i,j]):return True
                for k in by.get(-(y[i]+y[j])%h,()):
                    if accept([],[i,j,k]):return True
        # Birthday collisions of subset sums, only among the unused inputs.
        bysum={}
        for a,i in enumerate(pool):
            for j in pool[a+1:]:
                v=(y[i]+y[j])%h
                if v in bysum and accept(bysum[v],[i,j]):return True
                if (-v)%h in bysum and accept([],bysum[(-v)%h]+[i,j]):return True
                bysum[v]=[i,j]
        return False
    while len(basis)<len(whole):
        if not take():return None
    return dict(primes=ps,reserve=reserve,switches=switches,unused=pool,rank=len(whole),seed=seed)


def decode(c,projection,s,t,free=()):
    ps=c['primes'];x=[projection.sig(p) for p in ps];y=[projection.odd(p) for p in ps];h=projection.h;sw=c['switches'];reserve=c['reserve'];assert set(free)<=set(c['unused'])
    base=list(chain.from_iterable(a for a,b in sw));fixed=list(free);fixed+=rotate_decode(reserve,y,h,(t-sum(y[i] for i in base+fixed))%h)
    basis={}
    for j,(a,b) in enumerate(sw):assert insert(basis,xor(a+b,x),1<<j)
    choice=solve(basis,s^xor(fixed+base,x));ids=sorted(fixed+list(chain.from_iterable(opts[choice>>j&1] for j,opts in enumerate(sw))))
    assert len(set(ids))==len(ids) and xor(ids,x)==s and sum(y[i] for i in ids)%h==t;return ids


def complete(report,projection,partial,block=3):
    """Extend fixed independent choices outside the correction bank to the CRT target quotient."""
    c=report['blocks'][block];ps=c['primes'];partial=set(partial);bank=set(ps)
    allowed={p for b in report['blocks'] for p in b['primes']}-(bank-{ps[i] for i in c['unused']})
    if not partial<=allowed:raise ValueError('partial product uses a reserved or unknown factor')
    ext=partial-bank;t,_=target(report['A'],report['B']);s=projection.sig(t);y=projection.odd(t)
    for p in ext:s^=projection.sig(p);y-=projection.odd(p)
    fixed=[i for i in c['unused'] if ps[i] in partial];inds=decode(c,projection,s,y%projection.h,fixed)
    return sorted(ext|{ps[i] for i in inds})


def load(folder):
    pool=folder/'trim-four-lists.txt';g=json.loads((folder/'trim.json').read_text())['four-lists'];ps=list(map(int,pool.read_text().split()));random.Random(1).shuffle(ps)
    return [ps[i::4] for i in range(4)],g['A'],g['B'],sha256(pool.read_bytes()).hexdigest()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--pool',type=Path,default=Path('runs/lift12/trim'));ap.add_argument('--odd3',type=int,default=729);ap.add_argument('--odd7',type=int,default=7);ap.add_argument('--tries',type=int,default=12);args=ap.parse_args();blocks,A,B,digest=load(args.pool);parts=[[2187,args.odd3],[2401,args.odd7]];P=Projection(A,B,parts);t,_=target(A,B);certs=[]
    for j,ps in enumerate(blocks):
        for seed in range(args.tries):
            c=build(ps,P,seed)
            if c:break
        if not c:raise RuntimeError(f'no certificate found for block {j} (not an impossibility proof)')
        c['target_witness']=decode(c,P,P.sig(t),P.odd(t));certs.append(c);print(j,c['rank'],len(c['reserve']),len(c['unused']),len(c['target_witness']),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(dict(pool_sha256=digest,parts=parts,A=A,B=B,blocks=certs),separators=(',',':'))+'\n')
