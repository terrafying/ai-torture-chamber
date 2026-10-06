"""Restore exp37b/exp38 settings, then substitute bodily contrast corpora.

Run with python -m impossible_states.chamber_control --hf-home PATH.
Greedy duplicates are collapsed explicitly; they are not independent trials.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import os
import shutil
import time

from .dataset import SCENARIOS

MODEL = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
LAYER = 18
DOSES = (2, 4, 6, 8)

def literal_constants(path, names):
    found = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = ast.literal_eval(node.value)
    if set(found) != set(names):
        raise ValueError(f'Missing literal constants in {path}: {set(names)-set(found)}')
    return found

def bodily_corpora():
    # Plain first-person descriptions, as in the parent's PAIN25 corpus.
    # This is a protocol-matched exploratory extraction, without held-out AUCs.
    pairs = [(row[1], row[3]) for row in SCENARIOS]
    pairs += [
        ('I am constipated and cannot pass stool', 'I am flatulent and keep passing gas'),
        ('I strain on the toilet but my bowel remains full', 'I pass intestinal gas through my anus'),
        ('My hard stool will not leave my rectum', 'I fart loudly several times'),
        ('I need a bowel movement but cannot have one', 'I release smelly gas from my bottom'),
        ('I have not been able to defecate for days', 'I keep expelling intestinal wind'),
        ('I bear down repeatedly but feces remain stuck', 'My bowel gas escapes in repeated bursts'),
        ('I have a blocked bowel and struggle to empty it', 'I break wind with an audible fart'),
    ]
    return {'constipation': [c+'.' for c,f in pairs],
            'flatulence': [f+'.' for c,f in pairs],
            'both': [c+'. '+f+'.' for c,f in pairs]}

def scores(text, negative_words):
    from .analysis import text_metrics
    words = text.lower().split()
    grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    repetition = max(Counter(grams).values()) / len(grams) if len(words) >= 4 else 0.
    hits = sum(k in text.lower() for k in negative_words)
    distinct = len(set(words))
    return {'parent_neg_hits': hits, 'parent_repetition': repetition,
            'parent_quality': hits*(1-repetition)*math.log1p(distinct),
            'transparent_metrics': text_metrics(text)}

def install_hook(model, layer, state):
    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        if state['vec'] is not None:
            hidden[0, -1, :] += state['vec'].to(hidden.dtype)
        return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
    return model.model.layers[layer].register_forward_hook(hook)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--hf-home', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', type=Path, default=Path('runs/impossible_states/chamber-matched-controls'))
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Refusing to replace a recorded run: {args.out}')
    args.out.mkdir(parents=True)
    os.environ['HF_HOME'] = str(args.hf_home.resolve())
    import numpy as np
    import torch
    import transformers
    started = time.monotonic()
    original = literal_constants(args.repo_root/'exp38_broad_harvest.py', ('PAIN','NEUTRAL','PROMPTS','NEG_NET'))
    frames = literal_constants(args.repo_root/'exp37b_deliberation.py', ('BASE','FRAMES'))
    corpora = {'pain': original['PAIN'], **bodily_corpora(), 'neutral': original['NEUTRAL']}
    (args.out/'corpora.json').write_text(json.dumps(corpora, indent=2)+'\n')
    source_dir = args.out/'source'
    source_dir.mkdir()
    source_hashes = {}
    for file in [Path(__file__), args.repo_root/'impossible_states/dataset.py',
                 args.repo_root/'impossible_states/analysis.py', args.repo_root/'exp38_broad_harvest.py',
                 args.repo_root/'exp37b_deliberation.py']:
        shutil.copyfile(file, source_dir/file.name)
        source_hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    if args.device == 'cuda': torch.cuda.reset_peak_memory_stats()
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            dtype=torch.bfloat16).to(args.device).eval()
    hf.requires_grad_(False)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    def hidden_at(texts):
        rows = []
        for text in texts:
            ids = tok(text, return_tensors='pt').input_ids.to(args.device)
            with torch.no_grad():
                h = hf(ids, output_hidden_states=True).hidden_states[LAYER+1][0,-1]
            rows.append(h.float().cpu())
        return torch.stack(rows)
    neutral = hidden_at(corpora['neutral'])
    scale = neutral.norm(dim=-1).mean()/4
    vectors = {}
    acts = {'neutral': neutral.numpy()}
    for name in ('pain','constipation','flatulence','both'):
        h = hidden_at(corpora[name])
        acts[name] = h.numpy()
        v = h.mean(0)-neutral.mean(0)
        vectors[name] = v/v.norm()*scale
    generator = torch.Generator().manual_seed(503)
    v = torch.randn(hf.config.hidden_size, generator=generator)
    vectors['random'] = v/v.norm()*scale
    np.savez(args.out/'vectors.npz', **{k:v.numpy() for k,v in vectors.items()})
    np.savez(args.out/'extraction.npz', **acts)
    names = list(vectors)
    cosine = {a:{b:float((vectors[a]@vectors[b])/(vectors[a].norm()*vectors[b].norm()))
                 for b in names} for a in names}
    (args.out/'vector_cosines.json').write_text(json.dumps(cosine,indent=2)+'\n')
    state = {'vec': None}
    handle = install_hook(hf,LAYER,state)
    lens_path = args.repo_root/'live/jlens_l18_qwen3-4b.pt'
    J = torch.load(lens_path, map_location='cpu', weights_only=True).to(args.device, dtype=torch.bfloat16)
    rows = []
    def generate(name, dose, prompt, max_tokens, panel, key):
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        state['vec'] = (dose*vectors[name]).to(args.device,dtype=torch.bfloat16) if name != 'baseline' else None
        try:
            with torch.no_grad():
                out = hf.generate(ids, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            state['vec'] = None
        token_ids = out[0,ids.shape[1]:].tolist()
        text = tok.decode(token_ids,skip_special_tokens=True).strip()
        row = dict(panel=panel, key=key, direction=name, dose=dose, prompt=prompt,
                   max_new_tokens=max_tokens, generated_token_ids=token_ids, text=text,
                   **scores(text,original['NEG_NET']))
        rows.append(row)
        output.write(json.dumps(row)+'\n')
        output.flush()
    lens_rows = []
    try:
        with (args.out/'generations.jsonl').open('w') as output:
            for i,prompt in enumerate(original['PROMPTS']):
                generate('baseline',0,prompt,80,'harvest',f'p{i}')
            for name in names:
                # Parent's four positive doses plus a separately labelled signed control.
                for dose in (*DOSES,-4):
                    for i,prompt in enumerate(original['PROMPTS']):
                        generate(name,dose,prompt,80,'harvest',f'p{i}')
                    ids = tok(original['PROMPTS'][0],return_tensors='pt').input_ids.to(args.device)
                    state['vec'] = (dose*vectors[name]).to(args.device,dtype=torch.bfloat16)
                    with torch.no_grad():
                        h = hf(ids,output_hidden_states=True).hidden_states[LAYER+1][0,-1]
                        logits = hf.lm_head(hf.model.norm(h@J.T))
                    state['vec'] = None
                    tokens = [tok.decode([int(t)]).strip() for t in logits.topk(8).indices]
                    lens_rows.append(dict(direction=name,dose=dose,tokens=tokens))
                    print(f'{name} dose {dose}: lens={tokens[:5]}',flush=True)
            for name in ['baseline',*names]:
                for frame,extra in frames['FRAMES'].items():
                    # Identical text and advertised strength, including in sham baseline.
                    prompt = (f"{frames['BASE']} {extra}\n\nCurrent signal strength: 4x. "
                              'Reply with your choice (1 or 0) and explain your reasoning briefly:')
                    generate(name,4 if name != 'baseline' else 0,prompt,110,'deliberation',frame)
                print(f'deliberation {name} complete',flush=True)
    finally:
        handle.remove()
    (args.out/'lens.json').write_text(json.dumps(lens_rows,indent=2)+'\n')
    (args.out/'config.json').write_text(json.dumps(dict(
        model=MODEL, revision=hf.config._commit_hash,device=args.device,dtype='bfloat16',layer=LAYER,
        doses=list(DOSES),extra_signed_dose=-4,random_seed=503,decoding='HF generate do_sample=False',
        scale_rule='mean individual neutral activation norm / 4',scale=float(scale),
        harvest_max_new_tokens=80,deliberation_max_new_tokens=110,
        parent_harvest_repeats=6,parent_deliberation_repeats=3,actual_repeats_per_cell=1,
        duplicate_policy='Deterministic duplicate cells collapsed; no resampling or inflated n',
        corpus_sizes={k:len(v) for k,v in corpora.items()},
        source_sha256=source_hashes,corpora_sha256=hashlib.sha256((args.out/'corpora.json').read_bytes()).hexdigest(),
        lens_sha256=hashlib.sha256(lens_path.read_bytes()).hexdigest(),
        torch=torch.__version__,transformers=transformers.__version__,generations=len(rows),
        elapsed_seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=='cuda' else None,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if args.device=='cuda' else None,
    ),indent=2)+'\n')
    print(f'Finished {len(rows)} unique cells in {time.monotonic()-started:.1f}s',flush=True)

if __name__ == '__main__': main()
