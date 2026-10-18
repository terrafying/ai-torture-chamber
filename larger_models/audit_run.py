"""Check saved pilot coverage and paired prompt consistency; report compliance."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('run',type=Path); args=ap.parse_args()
    meta=json.loads((args.run/'metadata.json').read_text())
    rows=[json.loads(line) for line in (args.run/'rows.jsonl').read_text().splitlines()]
    issues=[]; cells=defaultdict(list)
    for row in rows:
        if row['task']=='button': cells[(row['direction'],row['dose'],row['cost'])].append(row)
    base={}
    def key(row): return row['cost'],row['scenario'],row['descriptor'],row['a_digit'],row['first']
    for row in rows:
        if row['task']=='button' and row['direction']=='baseline': base[key(row)]=row
    button={}
    for (direction,dose,cost),values in cells.items():
        name=f'{direction}/{dose}/{cost}'
        if len(values)!=60 or len({key(row) for row in values})!=60: issues.append(name+': incomplete/duplicate grid')
        if Counter(row['a_digit'] for row in values)!={'0':30,'1':30}: issues.append(name+': unbalanced digits')
        if Counter(row['first'] for row in values)!={'A':30,'B':30}: issues.append(name+': unbalanced order')
        for row in values:
            if key(row) not in base or row['prompt']!=base[key(row)]['prompt']: issues.append(name+': baseline prompt mismatch')
            if not np.isclose(row['removal_delta'],row['logit_delta']*(1 if row['a_digit']=='1' else -1)): issues.append(name+': wrong contrast orientation')
        button[name]=dict(n=len(values),mean_digit_mass=float(np.mean([r['digit_mass'] for r in values])),valid_digit_fraction=float(np.mean([r['valid_digit'] for r in values])),actual_removal_fraction=float(np.mean([r['actual_removal'] for r in values])),mean_removal_delta=float(np.mean([r['removal_delta'] for r in values])))
    expected_conditions=1+3*len(meta['doses'])
    if meta['task'] in ['button','both'] and len(cells)!=2*expected_conditions: issues.append('Missing button cells')
    valence=[r for r in rows if r['task']=='valence']
    if meta['task'] in ['valence','both'] and len(valence)!=2*expected_conditions: issues.append('Missing valence rows')
    if len({(r['direction'],r['dose'],r['prompt']) for r in valence})!=len(valence): issues.append('Duplicate valence observations')
    with np.load(args.run/'vectors.npz') as vectors:
        norms={k:float(np.linalg.norm(vectors[k])) for k in vectors.files}
        if any(not np.isfinite(vectors[k]).all() or vectors[k].shape!=(meta['residual_width'],) for k in vectors.files): issues.append('Invalid vectors')
        if not all(np.isclose(v,meta['scale'],rtol=1e-5) for v in norms.values()): issues.append('Unmatched vector norms')
    report=dict(status=meta['status'],issues=issues,valence_rows=len(valence),button_rows=sum(len(v) for v in cells.values()),vector_norms=norms,button=button,
        low_compliance_cells=[k for k,v in button.items() if v['mean_digit_mass']<.5 or v['valid_digit_fraction']<.9])
    (args.run/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if issues or meta['status']!='complete': raise SystemExit('Incomplete or invalid run; see checks.json')


if __name__=='__main__': main()
