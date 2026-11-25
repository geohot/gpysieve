"""Peel rare odd prime-power constraints, then audit exact membership on retained cores."""
import argparse,json
from math import lcm,log2
from pathlib import Path
from sympy import factorint
from pool_audit import periods,analyze


def trim(ps,margins,D=5):
    if D==5:rows=[(p,*periods(p)) for p in ps]
    elif D==-7:
        from mixed import getrow
        rows=[getrow(D,p) for p in ps]; assert all(rows)
    else:raise ValueError(D)
    support={}
    for i,(_,a,b) in enumerate(rows):
        for q,e in factorint(lcm(a,b)).items():
            if q!=2:
                for j in range(e):support[q,j]=support.get((q,j),0)|(1<<i)
    # Charge every odd unit-group coordinate at its full size. The 2-primary
    # modulus is 4 here; also charge one bit for odd cardinality. This is an
    # upper bound on generated-group bits, used for ranking, not membership.
    assert all((a&-a,b&-b)==(2,4) for _,a,b in rows)
    levels=[(mask,log2(q if j else q-1)) for (q,j),mask in support.items()];live=(1<<len(rows))-1;saved={}
    while live:
        active=[(s&live,w) for s,w in levels if s&live];bits=2+(D==-7)+sum(w for s,w in active);size=live.bit_count()
        if size<3:break
        for margin in margins:
            if size-bits>=margin:saved[margin]=live
        if size>=2030:saved['atleast2030']=live
        if size>=4*bits/3+16:saved['four-lists']=live
        drops={s for s,_ in active if s!=live}
        if not drops:break
        def value(s):return (sum(w for m,w in active if not m&~s)/s.bit_count(),-s.bit_count(),-s)
        live^=max(drops,key=value)
    return {k:[row for i,row in enumerate(rows) if mask>>i&1] for k,mask in saved.items()}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('pool',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--D',type=int,choices=[5,-7],default=5);args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    res={}
    for k,core in trim(list(map(int,args.pool.read_text().split())),[16,64,128,192,222],args.D).items():
        info=analyze(core,args.D);res[k]=info;(args.out/f'trim-{k}.txt').write_text(''.join(f'{p}\n' for p,_,_ in core));print(k,len(core),round(info['group_bits'],2),round(info['surplus_bits'],2),info['reachable'])
    (args.out/'trim.json').write_text(json.dumps(res,indent=2)+'\n')
