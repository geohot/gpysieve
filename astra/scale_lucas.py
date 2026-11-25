"""Split anchors => finite gcd; all-inert => CRT/character and Lucas rank sieve."""
import argparse,json,subprocess
from collections import Counter,defaultdict
from functools import lru_cache
from math import gcd,prod
from pathlib import Path
from shape_scan import DS,actual
from scale_shapes import indices
from search import mr,is_strong_lucas_prp

ROOT=Path(__file__).resolve().parent
# Fundamental odd characters (p/q), for p not dividing the discriminants.
CH={5:(5,),-7:(7,),9:(),-11:(11,),13:(13,),-15:(3,5),17:(17,),-19:(19,),21:(3,7)}
SEQ=tuple(CH)

def config(D):
    prev=SEQ[:SEQ.index(D)];Q=(1-D)//4;odds={q for d in (*prev,D) for q in CH[d]};
    if 9 in prev:odds.add(3)
    # odd part of Q is always 1 or an odd prime in this list of D's.
    z=abs(Q)
    while z%2==0:z//=2
    if z>1:odds.add(z)
    return prev,Q,z,sorted(odds)

@lru_cache(None)
def mask_options(cs,D,ell,kmod):
    prev,Q,z,_=config(D);vals=set();nbit=len(prev);rng=range(ell) if kmod<0 else (kmod,)
    for k in rng:
        ps=[(c*k-1)%ell for c in cs]
        if 0 in ps:continue # K>2^40: no prime can equal ell
        neg=sum(int(pow(p,(ell-1)//2,ell)==ell-1)<<i for i,p in enumerate(ps));bits=0
        if neg.bit_count()%2:
            for j,d in enumerate(prev):
                if ell in CH[d]:bits|=1<<j
        if ell in CH[D]:bits|=neg<<nbit
        if ell==z:bits|=neg<<(nbit+3)
        vals.add(bits)
    return vals

def inert_possible(cs,D,r,M):
    prev,Q,z,odds=config(D);j=len(prev);two=(abs(Q)&-abs(Q)).bit_length()-1;states={0}
    for ell in odds:
        vs=mask_options(tuple(c%ell for c in cs),D,ell,r%ell if M%ell==0 else -1);states={a^b for a in states for b in vs}
        if not states:return False
    # Odd prime z contribution uses (z*/p)=(p/z); restore (-1/p) if z=3 mod 4.
    assert M%8==0;qm=0
    for i,c in enumerate(cs):
        p=(c*r-1)%8;s=((Q<0)^(z%4==3)) and p%4==3;v=(bool(s)^bool(two%2 and p in (3,5)));qm|=int(v)<<i
    mn=min(c&-c for c in cs);minmask=sum(int((c&-c)==mn)<<i for i,c in enumerate(cs))
    return any((7<<j)|((qm^target)<<(j+3)) in states for target in (0,minmask))

def scan(progress,gp,Ds):
    groups=defaultdict(set)
    for l in progress.read_text().splitlines():
        a,b,c,r,M,*orders=map(int,l.split());groups[a,b,c].add((r,M))
    report={};hits=[]
    for D in Ds:
        missing=[];stats=Counter();maxG=0;passes=[];survive=[]
        # GP loop over coefficients from stdin. Anchor e is independently
        # bounded by the same binary resultant used in the order-branch sieve.
        cmd=[]
        for cs in groups:
            es=indices(cs);vec=[]
            for i,c in enumerate(cs):
                a,b=cs[:i]+cs[i+1:];vec.append(f'splitanchor({a},{b},{c},{es[i]},{D})')
            cmd.append('print(['+','.join(vec)+'])')
        proc=subprocess.run([gp,'-s','256M','-fq',str(ROOT/'shape.gp'),str(ROOT/'scale_split.gp')],input='\n'.join(cmd)+'\n',capture_output=True,text=True,check=True);assert not proc.stderr,proc.stderr[:500];lines=proc.stdout.splitlines();assert len(lines)==len(groups)
        for (cs,branches),line in zip(groups.items(),lines):
            ks=set();unresolved=False
            for i,(bits,scales) in enumerate(json.loads(line)):
                if bits<0:missing.append([cs,i]);unresolved=True;continue
                maxG=max(bits,maxG);ks.update(scales)
            for k in ks:
                if not any(k%M==r for r,M in branches):continue
                stats['split_compatible_K']+=1;fs=[c*k-1 for c in cs];n=prod(fs)
                if fs[0]>=3 and mr(n,[2]) and actual(n)==D:passes.append([cs,k]);
                if fs[0]>=3 and mr(n,[2]) and actual(n)==D and is_strong_lucas_prp(n):hits.append([cs,k,D])
            inert=[(r,M) for r,M in branches if inert_possible(cs,D,r,M)];stats['inert_progressions']+=len(inert);stats['tested_progressions']+=len(branches)
            if inert or unresolved:survive.append(cs)
        report[D]=dict(counts=stats,split_unfactored=missing,max_split_gcd_bits=maxG,MR=passes,remaining_shapes=survive);print(D,dict(stats),'missing split',len(missing),'remaining shapes',len(survive),flush=True)
    return dict(scope='K above prior bounded screen; split candidates checked on remaining progressions',results=report,hits=hits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('progress',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--gp',default='gp');ap.add_argument('--ds',type=int,nargs='+',default=list(DS));a=ap.parse_args();r=scan(a.progress,a.gp,a.ds);a.out.write_text(json.dumps(r,separators=(',',':'))+'\n')
