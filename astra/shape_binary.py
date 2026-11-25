"""D-independent bounded additive shapes: one unconditional strong base-2 order per factor."""
import argparse,json
from collections import Counter
from math import gcd,lcm,prod
from pathlib import Path
from scale_refine import catalogue,certify
from search import mr,is_strong_lucas_prp

def scan(src,limit,gp):
    opts,cert=catalogue(src);holes=[];counts=Counter();MR=[];hits=[]
    for u in range(2,limit,2):
        for v in range(2,limit-u+1,2):
            offsets=(-u,0,v);centers=set();branches=[];counts['patterns']+=1
            if len({s%3 for s in offsets})==3:
                centers={3-s for s in offsets};counts['forced_factor_3']+=1
            else:
                for s in offsets:
                    e=abs(prod(1+z-s for z in offsets if z!=s)-1)
                    if not e:branches.append([(1,2,0)]);continue
                    pp,unk=opts(e);centers.update(q-s for q in pp);rows=[]
                    for m in unk:
                        two=m&-m;M=lcm(8,2*m)
                        for x in range(1-s,M+1-s,m):
                            q=x+s;r=q%8
                            if not r%2 or (r==3 and two!=2) or (r==5 and two!=4) or (r==7 and two!=1) or (r==1 and (q-1)%(2*two)):continue
                            rows.append((x%M,M,two))
                    branches.append(rows)
                if not all(branches):counts['anchor_complete']+=1
                else:
                    def compatible(a,b):return (not a[2] or not b[2] or a[2]==b[2]) and (a[0]-b[0])%gcd(a[1],b[1])==0
                    x,y,z=sorted(branches,key=len)
                    if any(compatible(a,b) and any(compatible(a,c) and compatible(b,c) for c in z) for a in x for b in y):holes.append([u,v])
                    else:counts['joint_exclusion']+=1
            for p in centers:
                if p-u<3:continue
                n=(p-u)*p*(p+v);counts['candidate_translates']+=1
                if not mr(n,[2]):continue
                MR.append([p-u,p,p+v]);
                if is_strong_lucas_prp(n):hits.append(MR[-1])
    certify(cert,gp);return dict(diameter=limit,counts=counts,first_unresolved_diameter=min(map(sum,holes),default=limit+2),unresolved=holes,MR=MR,hits=hits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--limit',type=int,default=120);ap.add_argument('--gp',default='gp');a=ap.parse_args();r=scan(a.src,a.limit,a.gp);a.out.write_text(json.dumps(r,separators=(',',':'))+'\n');print(r['counts'],r['first_unresolved_diameter'],len(r['unresolved']),len(r['MR']),len(r['hits']))
