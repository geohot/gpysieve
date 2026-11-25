"""Recover more admissible entries, without claiming to identify the missing historic entries."""
import argparse,json,re
from collections import defaultdict
from math import gcd
from pathlib import Path
from extend_pool import generate
from pool_audit import analyze,certified,periods,source

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--upper',type=int,default=10**10);ap.add_argument('--hmax',type=int,default=65);args=ap.parse_args();out=args.directory
    page=out/'Baillie-PSW.html';M,N,displayed=source(page);s=' '.join(re.sub('<[^>]+>',' ',page.read_text(encoding='latin1')).split());smaller=set(map(int,re.findall(r'\{([\d,\s]+)\}',s)[0].split(',')))
    audit=json.loads((out/'audit.json').read_text());best=max(audit['chen'],key=lambda g:g['size']);A,B=best['A'],best['B'];new,stats=generate(A,B,args.upper,args.hmax);groups=defaultdict(list)
    for p in sorted(set(displayed)|smaller|set(new)):
        assert certified(p) and p%5 in (2,3);a,b=periods(p)
        if M%a==N%b==0 and gcd(p,M*N)==1:groups[a&-a,b&-b].append((p,a,b))
    ps={p for group in groups.values() for p,_,_ in group};extra=ps-set(displayed);info=[analyze(g) for g in sorted(groups.values(),key=len,reverse=True)]
    (out/'chen-regenerated.txt').write_text(''.join(f'{p}\n' for p in sorted(ps)));(out/'chen-extended.txt').write_text(''.join(f'{p}\n' for p,_,_ in groups[2,4]));(out/'chen-extension.json').write_text(json.dumps(dict(upper=args.upper,hmax=args.hmax,small_list_eligible=sorted(extra&smaller),new_frontier=sorted(extra-smaller),stats=stats,classes=info),indent=2)+'\n');print(len(ps),'eligible; largest:',info[0]['size'],info[0]['group_bits'])
