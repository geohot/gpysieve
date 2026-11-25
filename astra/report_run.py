"""Summarize the mirrored Iris campaigns, avoiding double counts across repeated bounded index runs."""
import argparse
from collections import Counter
import json
from pathlib import Path


def rows(f):
    with f.open('rb') as s:
        for l in s:
            if l.endswith(b'\n'):yield json.loads(l)


def report(base):
    result={};events=[]
    for prefix,keys in [('iris-mixed-',('subsets','exact_checks','full_pass')),('iris-native',('subsets','anchor_pass','two_pass','full_pass')),('iris-base2-',('constructed_mr','ordinary_lucas_pass')),('iris-index-',('products','lucas_congruence_pass','fermat_one_pass')),('iris-refine-',('products','lucas_congruence_pass','fermat_one_pass'))]:
        per={}; name='products.jsonl' if prefix in ('iris-index-','iris-refine-') else 'jobs.jsonl'
        for f in sorted(base.glob(prefix+'*/'+name)):
            c=Counter()
            for r in rows(f):
                if r['event']=='done':c['jobs']+=1;c.update({k:r.get(k,0) for k in keys})
                else:events.append([str(f.relative_to(base)),r['event'],r.get('id')])
            per[f.parent.name]=dict(c)
        result[prefix]=per
    coverage={}
    for f in sorted(base.glob('*/incomplete-indices.json')):
        wp=f.parent/'products.jsonl';st=f.parent/'worker.log'
        coverage[f.parent.name]=dict(incomplete_counts={d:len(v) for d,v in json.loads(f.read_text()).items()},completed_batches=sum(1 for r in rows(wp) if r['event']=='done') if wp.exists() else 0)
    result['index_coverage']=coverage;result['special_events']=events;return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('remote',type=Path);args=ap.parse_args();print(json.dumps(report(args.remote),indent=2))
