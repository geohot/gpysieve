"""Any number of cancelling residues: conditional Lucas rank sieve, then an anchored CRT join."""
import argparse,hashlib,json,subprocess
from collections import Counter,defaultdict
from math import gcd,lcm,prod
from pathlib import Path
from sympy import divisors,factorint,isprime,primerange
from order_bank import checked_orders,character,DS,positions
from order_cliques import coprime_product
from search import mr,is_strong_lucas_prp
from campaign import lucas_u,gp_check


def templates(db):
    for m in sorted(db):
        groups=defaultdict(list)
        for j in divisors(m):
            for p in db.get(j,[]):
                if gcd(m,p)==1:groups[j&-j].append((p,j))
        for group in groups.values():
            if sum(p%m!=1 for p,j in group)>=3:yield m,group


def bounded_templates(db,B):
    indices=defaultdict(list)
    for j,ps in db.items():
        if ps:
            for m in range(j,B+1,j):indices[m,j&-j].append(j)
    for (m,s),js in indices.items():
        if lcm(*js)!=m:continue
        group=[(p,j) for j in js for p in db[j] if gcd(m,p)==1]
        if sum(p%m!=1 for p,j in group)>=3 and lcm(*(j for _,j in group))==m:yield m,group


def prefix(d):
    return tuple((5+2*i)*(-1)**i for i in range((abs(d)-5)//2+1))


def metadata(ps,d,bin,reverse=False):
    small=list(primerange(2,6100));req=[];keys=[];tags={};prior=prefix(d);Q=(1-d)//4
    for p in sorted(ps):
        # A zero symbol at an earlier d is a proper gcd for n>max |d| (our scope).
        chars=[character(a,p%abs(a)) for a in prior]
        if 0 in chars or gcd(p,Q)!=1:continue
        eps=1 if reverse else chars[-1];tags[p]=sum((s<0)<<i for i,s in enumerate(chars));t=p-eps;v=t;ls=[]
        for ell in small:
            if v%ell==0:
                ls.append(ell)
                while v%ell==0:v//=ell
            if ell*ell>v:break
        if v>1 and v<1<<64:
            if v<1<<47:ls+=list(factorint(v))
            elif isprime(v):ls.append(v)
        assert len(ls)==len(set(ls)) and all(ell<2**64 and isprime(ell) for ell in ls)
        req.append(f'{p} {0 if reverse else Q} {t} {len(ls)} '+ ' '.join(map(str,ls)));keys.append(p)
    out=subprocess.run([str(bin.resolve())],input='\n'.join(req)+'\n',text=True,capture_output=True,check=True);assert not out.stderr;meta={}
    for p,line in zip(keys,out.stdout.splitlines(),strict=True):
        lo,hi=map(int,line.split());assert lo>0 and hi%lo==0 and (hi//lo)%2==1,(p,d,lo,hi);meta[p]=lo,hi,tags[p]
    return meta


def crt(m,b):
    g=gcd(m,b);assert 2%g==0
    k=(-2//g*pow(m//g,-1,b//g))%(b//g);return 1+m*k,m*(b//g)


def sides(arr,M):
    # For width <= 16 this costs much less than enumerating complete subsets.
    out=[(1,0,0)]
    for i,(p,_,_,_,tag) in enumerate(arr):
        v=p%M;bit=1<<i;out.extend([(v*x%M,t^tag,s|bit) for x,t,s in out])
    return out


def join(m,d,group,meta,counts,hits,gp,maxhalf,reverse=False,near=None):
    by=defaultdict(list);wanted=1<<(len(prefix(d))-1);Q=(1-d)//4
    for p,j in group:
        if p not in meta:continue
        b,t,tag=meta[p]
        if 2%gcd(b,m):continue
        by[b&-b].append((p,j,b,t,tag))
    for bank in by.values():
        # Largest certified divisor first: each nonempty subset has one pivot.
        if lcm(*(v[1] for v in bank))!=m:continue
        bank.sort(key=lambda v:(v[2],v[0]),reverse=True)
        if len(bank)<2 or (not reverse and sum(p%m!=1 for p,*_ in bank)<3):continue
        counts['rank_class_subsets_upper']+=2**len(bank);counts['largest_rank_class']=max(counts['largest_rank_class'],len(bank))
        def anchored(fixed,arr):
            p=prod(v[0] for v in fixed);j=lcm(*(v[1] for v in fixed));b=lcm(*(v[2] for v in fixed));tag=0
            for v in fixed:tag^=v[4]
            target,M=crt(m,b)
            if reverse:target=(-target)%M
            if gcd(p,M)>1:return
            arr=[v for v in arr if gcd(v[0],M)==1 and gcd(v[2],p)==1]
            if len(fixed)+len(arr)<2 or (not reverse and sum(v[0]%m!=1 for v in fixed+arr)<3) or lcm(j,*(v[1] for v in arr))!=m:return
            if len(arr)>2*maxhalf:
                # Every missing maximal prime power must have a user. Partition by
                # the first selected user of the rarest one, including its rank.
                cnts=[[v for v in arr if v[1]%z==0] for q,e in factorint(m).items() if j%(z:=q**e)]
                mandatory=min(cnts,key=len) if cnts else [arr[0]];rest=list(arr)
                for v in mandatory:
                    rest.remove(v);anchored(fixed+[v],rest)
                if not cnts:anchored(fixed,rest)
                counts['recursive_splits']+=1;return
            counts['anchors']+=1;counts['anchor_cube_subsets']+=1<<len(arr)
            cut=len(arr)//2;A=defaultdict(list);target=target*pow(p,-1,M)%M
            for x,tg,s in sides(arr[:cut],M):A[x,tg].append(s)
            counts['half_entries']+=(1<<cut)+(1<<(len(arr)-cut))
            # Build reciprocals multiplicatively (one inversion per input).
            inv=[(pow(v[0],-1,M),*v[1:]) for v in arr[cut:]]
            for y,u,h in sides(inv,M):
                for s in A.get((target*y%M,wanted^tag^u),()):
                    counts['anchor_matches']+=1;rows=fixed+[arr[k] for k in positions(s|(h<<cut))];
                    if len(rows)<2 or (not reverse and sum(v[0]%m!=1 for v in rows)<3) or lcm(*(v[1] for v in rows))!=m:continue
                    if p<=abs(d) and prod(v[0] for v in rows)<=abs(d):continue
                    counts['canonical_matches']+=1;fs=[v[0] for v in rows];n=prod(fs)
                    if any((n+(-1 if reverse else 1))%v[2] for v in rows):continue
                    counts['all_divisors_pass']+=1
                    if near is not None and len(near)<20:near.append(dict(D=d,m=m,factors=fs,orders=[v[1] for v in rows],lower=[v[2] for v in rows],upper=[v[3] for v in rows],bits=n.bit_length()))
                    if reverse:
                        if any(pow(2,(n-1)%v[3],v[0])!=1 for v in rows if v[3]!=v[2]):continue
                        counts['Fermat_pass']+=1
                        if not mr(n,[2]):continue
                    else:
                        if any(lucas_u((n+1)%v[3],Q,v[0]) for v in rows if v[3]!=v[2]):continue
                        counts['ordinary_Lucas_pass']+=1
                    if is_strong_lucas_prp(n):
                        assert mr(n,[2]);ok,trace=gp_check(fs,gp);assert ok,trace;hits.append(dict(n=n,factors=fs,D=d,m=m,trace=trace))
        for i,pivot in enumerate(bank):anchored([pivot],bank[i+1:])


def scan(src,d,bin,gp,limit,B=None):
    db=checked_orders(src);coprime_product(sorted(p for ps in db.values() for p in ps));banks=list(templates(db) if B is None else bounded_templates(db,B));ps={p for _,bank in banks for p,j in bank};data=metadata(ps,d,bin);cnt=Counter();hits=[];near=[]
    for m,group in banks:join(m,d,group,data,cnt,hits,gp,limit,near=near)
    return dict(D=d,order_bound=B,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),input_groups=len(banks),before_subsets=sum(1<<len(bank) for _,bank in banks),factors=len(ps),with_rank_bounds=len(data),exact_ranks=sum(b==t for b,t,_ in data.values()),counts=dict(cnt),lower_bound_survivors=near,hits=hits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--D',type=int,default=5,choices=list(DS));ap.add_argument('--bin',type=Path,default=Path('/tmp/astra-rank'));ap.add_argument('--gp',default='gp');ap.add_argument('--maxhalf',type=int,default=19);ap.add_argument('--bound',type=int);a=ap.parse_args();r=scan(a.src,a.D,a.bin,a.gp,a.maxhalf,a.bound);a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
