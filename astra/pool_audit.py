"""Joint CRT + Selfridge character audit of published D=5 pools. Not a subset solver."""
import argparse,hashlib,json,re
from collections import defaultdict
from functools import lru_cache
from math import gcd,lcm,log2,prod
from pathlib import Path
import numpy as np
from sympy import factorint,isprime,primitive_root
from paper_search import periods
from search import target


def source(path):
    t=' '.join(re.sub('<[^>]+>',' ',path.read_text(encoding='latin1')).split()).split('For a more extreme example, let')[1]
    M,N=(int(x.replace(' ','')) for x in re.match(r' M = ([\d ]+)and N = ([\d ]+)Here',t).groups()); ps=list(map(int,re.search(r'\{([\d,\s]+)\}',t)[1].split(',')))
    return M,N,ps


@lru_cache(None)
def logs(m,g,k):
    x=1;table={}
    for i in range(k):table[x]=i;x=x*g%m
    assert len(table)==k and x==1;return table


def coords(m,vals):
    for q,e in factorint(m).items():
        h=q**e; residues=[v%h for v in vals]
        if q==2:
            if e==1:continue
            signs=[int(x%4==3) for x in residues];yield 2,signs
            if e==2:continue
            residues=[(-x if s else x)%h for x,s in zip(residues,signs)];g=5;k=h//4
        else:g=int(primitive_root(h));k=h-h//q
        table=logs(h,g,k);yield k,[table[x] for x in residues]


def joint(rows):
    # Last column is RHS. Work separately over each primary part of the
    # coordinate group. A pivot in the largest remaining cyclic factor allows
    # substitution into all smaller factors without imposing extra relations.
    ambient=prod(int(k) for k,_ in rows);image=1;obstruct=[]
    for ell in factorint(ambient):
        eq=[];bad=False
        for k,row in rows:
            h=1
            while k%ell==0:k//=ell;h*=ell
            if h>1:
                assert h*h<2**63;eq.append((h,np.asarray(row,dtype=np.int64)%h))
        while eq:
            reduced=[]
            for h,v in eq:
                while h>1 and not (v[:-1]%ell).any():
                    if v[-1]%ell:bad=True
                    v=v//ell;h//=ell
                if h>1:reduced.append((h,v))
            if not reduced:break
            reduced.sort(key=lambda t:t[0]);h,v=reduced.pop();j=int(np.flatnonzero(v[:-1]%ell)[0]);v=(v*pow(int(v[j]),-1,h))%h
            image*=h;eq=[(d,(w-int(w[j])*v)%d) if w[j] else (d,w) for d,w in reduced]
        if bad:obstruct.append(int(ell))
    return dict(ambient=ambient,image=image,obstruction_primes=obstruct,reachable=not obstruct)


def analyze(group,D=5):
    ps,aa,bb=zip(*group);A=lcm(*aa);B=lcm(*bb);t,m=target(A,B);assert all(gcd(p,m)==1 for p in ps)
    rs=list(coords(m,[*ps,t]))+[(2,[1]*(len(ps)+1))] # target -1 for first inert discriminant
    if D==5:assert all(p%5 in (2,3) for p in ps)
    elif D==-7:
        assert all(p%5 and p%7 in (3,5,6) for p in ps)
        rs.append((2,[int(p%5 in (2,3)) for p in ps]+[0])) # earlier D=5 must be +1
    else:raise ValueError(D)
    info=joint(rs);bits=log2(info['image']);return dict(size=len(ps),A=A,B=B,signature=[(aa[0]&-aa[0]).bit_length()-1,(bb[0]&-bb[0]).bit_length()-1],group_bits=bits,surplus_bits=len(ps)-bits,**info)


@lru_cache(None)
def certified(p):
    if p<2**64:return isprime(p)
    if not isprime(p):return False
    fs=factorint(p-1);assert prod(q**e for q,e in fs.items())==p-1 and all(certified(q) for q in fs)
    return all(any(pow(w,p-1,p)==1 and gcd(pow(w,(p-1)//q,p)-1,p)==1 for w in range(2,1000)) for q in fs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('html',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--historic',type=Path,required=True);ap.add_argument('--extra',type=Path);args=ap.parse_args();M,N,ps=source(args.html);historic=list(map(int,args.historic.read_text().split()));extra=list(map(int,args.extra.read_text().split())) if args.extra else []
    assert len(ps)==len(set(ps)); rows={}
    for p in set(ps+historic+extra):
        assert p>2 and certified(p) and p%5 in (2,3);rows[p]=(p,*periods(p))
    groups=defaultdict(list);bad=[];outside=defaultdict(list)
    for p in ps:
        _,a,b=rows[p]
        if M%a or N%b or gcd(M*N,p)!=1:
            bad.append(p);outside[a&-a,b&-b].append(rows[p]);continue
        groups[a&-a,b&-b].append(rows[p])
    cg=[analyze(g) for g in sorted(groups.values(),key=len,reverse=True)]
    orig=[rows[p] for p in historic];assert len(set((a&-a,b&-b) for _,a,b in orig))==1;ag=analyze(orig);A,B=ag['A'],ag['B']
    additions=sorted(p for p in (set(ps+extra)-set(historic)) if gcd(p,A*B)==1 and A%rows[p][1]==B%rows[p][2]==0 and (rows[p][1]&-rows[p][1],rows[p][2]&-rows[p][2])==(2,4))
    enhanced=analyze(orig+[rows[p] for p in additions]);args.out.mkdir(exist_ok=True,parents=True);(args.out/'extended.txt').write_text(''.join(f'{p}\n' for p in sorted(historic+additions)))
    report=dict(outside_published_moduli=[analyze(g) for g in outside.values()],published_M=M,published_N=N,page_sha256=hashlib.sha256(args.html.read_bytes()).hexdigest(),displayed=len(ps),excluded=bad,chen=cg,historic=ag,added=additions,enhanced=enhanced);(args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for g in [*cg,ag,enhanced]:print(g['signature'],g['size'],round(g['group_bits'],2),g['reachable'],g['obstruction_primes'])

if __name__=='__main__':main()
