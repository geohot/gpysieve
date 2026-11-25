"""Independent direct PARI check of deterministic controls (not used in sieve)."""
import argparse,hashlib,json,random,subprocess
from math import prod
from pathlib import Path
from order_cliques import boxes
from order_pairs import pairs
from order_bank import positions

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--gp',default='gp');ap.add_argument('--pairs',action='store_true');a=ap.parse_args();rng=random.Random(18471);samples=[];largest=[]
    seq=pairs(a.src) if a.pairs else ((m,ps,f,[]) for m,ps,f in boxes(a.src))
    for i,(_,ps,forced,need) in enumerate(seq):
        mask=forced|rng.getrandbits(len(ps))
        opt=[j for j in range(len(ps)) if not forced>>j&1];compressed=sum(((mask>>j)&1)<<k for k,j in enumerate(opt))
        if mask.bit_count()<2 or any(not(compressed&z) for z in need):mask=(1<<len(ps))-1
        fs=[ps[j] for j in positions(mask)];pos=i if i<40 else rng.randrange(i+1)
        if pos<40:
            if i<40:samples.append(fs)
            else:samples[pos]=fs
        if len(ps)>len(largest):largest=ps
    samples.append(largest);cmd=''.join('print(bpswtrace(prod(i=1,#v,v[i]),0))\n'.replace('print(',f'v={fs};print(') for fs in samples);r=subprocess.run([a.gp,'-s','256M','-fq',str(Path(__file__).with_name('verify.gp'))],input=cmd,text=True,capture_output=True,check=True);assert not r.stderr and list(map(json.loads,r.stdout.splitlines()))==[[1,0]]*len(samples),r.stderr
    a.out.write_text(json.dumps(dict(seed=18471,pairs=a.pairs,samples=len(samples),max_bits=max(prod(fs).bit_length() for fs in samples),factors=samples,pari_trace_sha256=hashlib.sha256(r.stdout.encode()).hexdigest()),indent=2)+'\n');print('PARI: all',len(samples),'passed MR and failed strong Lucas')
