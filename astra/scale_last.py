"""Targeted D=5 completion of individual exact-order branches. External entries are hints only."""
import argparse,json,subprocess,time,urllib.request
from collections import defaultdict
from hashlib import sha256
from math import gcd,prod
from pathlib import Path
from scale_refine import cyclo
from scale_lucas import inert_possible
from search import mr,is_strong_lucas_prp

def scan(folder,gp,fetch=False):
    db={r['m']:r['primes'] for l in (folder/'hints-big.jsonl').read_text().splitlines() if (r:=json.loads(l))};cache=folder/'factordb';cache.mkdir(exist_ok=True);mem={};cert=set();hits=[];holes=[];ct=0;tested=[];anchors=[]
    def factors(m,remote):
        if (m,remote) in mem:return mem[m,remote]
        N=cyclo(m);pp=set(db.get(m,()));parts=[N]
        if m%8==4:
            t=m//4;g=gcd(N,2**t+1-2**((t+1)//2));parts=[g,N//g]
        for v in parts:
            for p in pp:
                while v%p==0:v//=p
            g=gcd(m,v)
            while g>1:v//=g;g=gcd(m,v)
            if v==1:continue
            if not remote:mem[m,remote]=None;return None
            key=sha256(str(v).encode()).hexdigest();path=cache/(key+'.json')
            if not path.exists() and fetch:
                url='https://factordb.com/api?query='+str(v)
                with urllib.request.urlopen(url,timeout=20) as f:data=f.read()
                path.write_bytes(data);time.sleep(1)
            if not path.exists():return None
            raw=json.loads(path.read_bytes());rawf=[(int(p),int(e)) for p,e in raw['factors']];assert prod(p**e for p,e in rawf)==v
            from sympy import isprime
            if not all(isprime(p) for p,e in rawf):mem[m,remote]=None;return None
            pp.update(p for p,e in rawf)
        # Verify completeness again on the unsplit primitive integer; ignoring
        # q|m is valid since an exact-order-m prime must be 1 mod m.
        for p in pp:
            assert N%p==0
            while N%p==0:N//=p
        while (g:=gcd(m,N))>1:N//=g
        assert N==1;mem[m,remote]=sorted(pp);return mem[m,remote]
    for l in (folder/'merged-orders.txt').read_text().splitlines():
        *cs,r,M,i,j,k=map(int,l.split());cs=tuple(cs)
        if not inert_possible(cs,5,r,M):continue
        ct+=1;options=[]
        for remote in (False,True):
            for c,m in sorted(zip(cs,(i,j,k)),key=lambda t:cyclo(t[1]).bit_length()):
                pp=factors(m,remote)
                if pp is not None:options.append((max((p.bit_length() for p in pp),default=0),c,m,pp))
                if options and remote:break
            if options:break
        if not options:holes.append([cs,r,M,i,j,k]);continue
        _,c,m,pp=min(options);cert.update(pp);anchors.append([cs,r,M,m]);
        for p in pp:
            K,rem=divmod(p+1,c)
            if rem or K%M!=r:continue
            fs=[a*K-1 for a in cs];n=prod(fs);tested.append([cs,K,m]);
            if min(fs)>=3 and mr(n,[2]) and is_strong_lucas_prp(n):hits.append([cs,K])
    # Certify each distinct proposed anchor factor. Failure means no published result.
    cmd=''.join(f'print(isprime({p}))\n' for p in sorted(cert));pr=subprocess.run([gp,'-s','256M','-fq'],input=cmd,text=True,capture_output=True,check=True);assert not pr.stderr and pr.stdout.splitlines()==['1']*len(cert),pr.stderr[:500]
    return dict(D=5,branches=ct,certified=len(cert),max_certified_bits=max(p.bit_length() for p in cert),anchors=anchors,compatible_K=tested,unresolved=holes,hits=hits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--fetch',action='store_true');ap.add_argument('--gp',default='gp');a=ap.parse_args();r=scan(a.folder,a.gp,a.fetch);(a.folder/'last5.json').write_text(json.dumps(r,indent=2)+'\n');print(r['branches'],r['certified'],r['max_certified_bits'],'unresolved',r['unresolved'],'compatible',r['compatible_K'],'hits',r['hits'])
