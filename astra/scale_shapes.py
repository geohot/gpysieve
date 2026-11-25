"""Eliminate the unbounded scale K in triples p_i=c_i*K-1 by local resultants."""
import argparse,json,subprocess
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
from math import gcd,prod
from pathlib import Path
from sympy import divisors
from search import mr,is_strong_lucas_prp


def indices(cs):
    ret=[]
    for j,c in enumerate(cs):
        a,b=cs[:j]+cs[j+1:];v=abs(2*a*b-c*(a+b))
        while v and (g:=gcd(v,c))>1:v//=g
        ret.append(2*v)
    return ret


def scan(src,B,gp):
    records={r['m']:r for s in src.read_text().splitlines() if (r:=json.loads(s))['D']==2};checked=set()
    @lru_cache(None)
    def cover(e):
        if not e or e not in records:return None
        v=(1<<e)-1;ps={p for m in divisors(e) if m in records for p in records[m]['primes']}
        for p in ps:
            if v%p:raise ValueError('input does not divide bound')
            while v%p==0:v//=p
        if v!=1:return None
        checked.update(ps);return sorted(ps)
    holes=[];completed=0;constructed=0;above=0;ferm=[];hits=[];tot=0;large=0;allmods=0
    for cs in combinations(range(1,B+1),3):
        if gcd(*cs)!=1:continue
        es=indices(cs);opts=[(len(pp),j,pp) for j,e in enumerate(es) if (pp:=cover(e)) is not None];tot+=1
        if not opts:holes.append(cs);continue
        completed+=1;_,j,pp=min(opts)
        for p in pp:
            k,r=divmod(p+1,cs[j]);fs=[a*k-1 for a in cs]
            if r or k%2 or fs[0]<3:continue
            constructed+=1;n=prod(fs);above+=n>=2**64;large=max(large,n.bit_length())
            if any(e and pow(2,e,q)!=1 for e,q in zip(es,fs)):continue
            allmods+=1
            if not mr(n,[2]):continue
            row=dict(coefficients=cs,K=k,factors=fs,n=n);ferm.append(row)
            if is_strong_lucas_prp(n):hits.append(row)
    cmd='print(prod(i=1,#v,isprime(v[i])))';r=subprocess.run([gp,'-s','256M','-fq'],input=f'v={sorted(checked)};{cmd}\n',text=True,capture_output=True,check=True);assert not r.stderr and r.stdout.strip()=='1',r.stderr
    return dict(source_sha256=sha256(src.read_bytes()).hexdigest(),box=B,primitive_shapes=tot,completed=completed,complete_box=min((max(t) for t in holes),default=B+1)-1,holes=holes,certified_anchor_primes=len(checked),constructed=constructed,above64=above,largest_bits=large,local_orders_pass=allmods,MR_pass=ferm,hits=hits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--box',type=int,default=40);ap.add_argument('--gp',default='gp');a=ap.parse_args();r=scan(a.src,a.box,a.gp);a.out.write_text(json.dumps(r,separators=(',',':'))+'\n');print({k:v for k,v in r.items() if k not in ('holes','MR_pass','hits')},'MR',len(r['MR_pass']),'hits',len(r['hits']))
