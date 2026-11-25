"""Relax cheap template powers, then select those actually worth paying for."""
import argparse,json,subprocess
from pathlib import Path
from math import gcd,lcm,prod
from sympy import factorint
from closure_pool import Closure
from pool_audit import analyze,certified,periods
from mixed import getrow
ROOT=Path(__file__).resolve().parent

def axes(A,B,extra):
    f=dict(factorint(A//2));g=dict(factorint(B//4));assert all(q%4==1 for q in f)
    # Allow (rather than require) higher powers of cheap primes. 5 on the
    # minus side enables D=-7 candidates the D=5 construction cannot use.
    for fs in (f,g):
        for q in fs:
            if q<=extra:fs[q]+=1
    f[5]=6
    return f,g

def discover(A,B,upper,hmax,defect,extra):
    f,g=axes(A,B,extra);text=''.join(f'{len(z)}\n'+''.join(f'{q} {e}\n' for q,e in sorted(z.items())) for z in (f,g))
    lines=subprocess.check_output([str(ROOT/'divisor_lift'),str(upper),str(hmax),str(defect)],input=text,text=True).splitlines();dis={5:set(),-7:set()}
    for line in lines:
        p,ok5,ok7=map(int,line.split());
        if not certified(p):continue
        if ok5:dis[5].add(p)
        if ok7:dis[-7].add(p)
    return dis,2*prod(q**e for q,e in f.items()),4*prod(q**e for q,e in g.items())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--dir',type=Path,default=Path('runs/chen-greene'));ap.add_argument('--upper',type=int,default=10**11);ap.add_argument('--hmax',type=int,default=9);ap.add_argument('--defect',type=int,default=256);ap.add_argument('--extra',type=int,default=71);args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    old=json.loads((args.dir/'audit.json').read_text())['historic'];seeds=set(map(int,(args.dir/'extended.txt').read_text().split()));dis,M,N=discover(old['A'],old['B'],args.upper,args.hmax,args.defect,args.extra);allresults={}
    for D in (5,-7):
        rows=[]
        for p in sorted(dis[D]|seeds):
            if p%8!=3 or gcd(p,M*N)!=1:continue
            if D==5:
                if p%5 not in (2,3):continue
                a,b=periods(p)
            else:
                if p%7 not in (3,5,6) or not p%5:continue
                r=getrow(D,p)
                if not r:continue
                _,a,b=r
            if M%a or N%b:continue
            assert certified(p) and (a&-a,b&-b)==(2,4);rows.append((p,a,b))
        (args.out/f'rows-{D}.json').write_text(json.dumps(rows)+'\n');o=Closure(rows);results={};seen={}
        for price in (0.0,0.5,1.0,1.1):
            ids=o.cut(price) if price else tuple(range(len(rows)))
            if not ids or ids in seen:continue
            seen[ids]=price;core=[rows[i] for i in ids];r=analyze(core,D);results[price]=r;(args.out/f'D{D}-price{price}.txt').write_text(''.join(f'{p}\n' for p,_,_ in core));print(D,price,r['size'],round(r['group_bits'],2),round(r['surplus_bits'],2),r['reachable'],flush=True)
        allresults[D]=results
    (args.out/'report.json').write_text(json.dumps(dict(upper=args.upper,hmax=args.hmax,defect=args.defect,extra=args.extra,results=allresults),indent=2)+'\n')
