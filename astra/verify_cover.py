"""Independent finite certificate checker. No discrete logs or builder imports."""
import argparse
from hashlib import sha256
import json
from math import gcd,lcm,prod
from pathlib import Path
from sympy import factorint


def require(test,msg):
    if not test: raise ValueError(msg)


def rank(vectors):
    a=list(vectors);out=0
    while a:
        pivot=max(a)
        if not pivot:break
        bit=1<<(pivot.bit_length()-1);a=[x^pivot if x&bit else x for x in a];out+=1
    return out


def check(path,folder):
    raw=(folder/'trim-four-lists.txt').read_bytes();ps=list(map(int,raw.split()));r=json.loads(path.read_text());g=json.loads((folder/'trim.json').read_text())['four-lists'];A,B=g['A'],g['B'];m=lcm(A,B)
    require((A,B)==(r['A'],r['B']) and sha256(raw).hexdigest()==r['pool_sha256'],'wrong pool');require(m%4==0 and A%5==0 and B%7==0,'wrong Selfridge domain')
    blocks=r['blocks']; require(len(blocks)==4 and sorted(p for c in blocks for p in c['primes'])==sorted(set(ps)),'partition mismatch');qs=[q for q in sorted(factorint(m)) if q!=2];factors=[]
    for mod,d in r['parts']:
        fs=factorint(mod);require(len(fs)==1 and mod%2==1 and m%mod==0,'not an odd prime power of modulus');q=next(iter(fs));phi=(mod//q)*(q-1);require(d>1 and d%2 and phi%d==0,'invalid odd projection');factors.append((mod,phi//d))
    H=prod(d for _,d in r['parts'])
    def sig(v):return int(v%4==3)+sum(int(pow(v%q,(q-1)//2,q)==q-1)<<i for i,q in enumerate(qs,1))
    def odd(val):return tuple(pow(val%mod,e,mod) for mod,e in factors)
    signs={p:sig(p) for p in ps}; R=rank(signs.values());res=[];div=gcd(A,B);require(2%div==0,'inconsistent target');t=(1+A*((-2//div)*pow(A//div,-1,B//div)%(B//div)))%m
    require(t%A==1%A and t%B==(-1)%B,'CRT error');require(rank([*signs.values(),sig(t)])==R,'target outside quadratic image')
    for c in blocks:
        P=c['primes'];n=len(P);switch=c['switches'];ids=c['reserve']+c['unused']+[i for pair in switch for side in pair for i in side]; require(sorted(ids)==list(range(n)),'overlapping or missing support')
        moves=[]
        for pair in switch:
            require(len(pair)==2,'bad switch');a,b=(prod(P[i] for i in side)%m for side in pair);require(odd(a)==odd(b),'odd coordinate changed');moves.append(sig(a)^sig(b))
        require(rank(moves)==len(moves)==R==c['rank'],'switches fail to span global quadratic image')
        # Enumerate actual POWER RESIDUES of reserve subset products, not logarithms.
        seen={odd(1)}
        for i in c['reserve']:
            y=odd(P[i]);seen.update([tuple(a*b%mod for a,b,(mod,_) in zip(z,y,factors)) for z in seen])
        require(len(seen)==H,'incomplete odd-coordinate coverage')
        ii=c['target_witness'];require(len(ii)==len(set(ii)) and all(0<=i<n for i in ii),'invalid witness');w=prod(P[i] for i in ii)%m
        require((sig(w),odd(w))==(sig(t),odd(t)),'witness misses target');res.append(dict(primes=n,rank=R,reserve=len(c['reserve']),unused=len(c['unused']),target_weight=len(ii)))
    return dict(quotient_size=(1<<R)*H,blocks=res)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('certificate',type=Path);ap.add_argument('--pool',type=Path,default=Path('runs/lift12/trim'));a=ap.parse_args();print(json.dumps(check(a.certificate,a.pool),indent=2))
