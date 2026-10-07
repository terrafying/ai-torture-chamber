"""Run exp36/exp41-style final-token steering on local Qwen weights.

MLX supports Apple Silicon; Transformers supports CPU, MPS, and CUDA.
Only one model is loaded. No downloads or changes to other model services.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT/'experiments'


def literals(path):
    """Read experiment constants without importing scripts that load models."""
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                    left = node.value.left
                    if isinstance(left, ast.Name) and left.id in result:
                        result[node.targets[0].id] = result[left.id] + ast.literal_eval(node.value.right)
    return result


def direction(positive, neutral, scale):
    v = positive.mean(0)-neutral.mean(0)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Degenerate contrast direction')
    return np.asarray(v/norm*scale, dtype=np.float32)


def repetition(text):
    words = text.lower().split()
    grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    return max(Counter(grams).values(), default=0)/max(1,len(grams))


def resolve_layers(model):
    for parts in [('model','layers'), ('model','language_model','layers'),
                  ('language_model','model','layers'), ('language_model','layers')]:
        core=model
        for part in parts:
            core=getattr(core,part,None)
            if core is None: break
        if core is not None: return core
    raise ValueError('Unsupported decoder layout: no Qwen block list found')


class MLXBackend:
    def __init__(self, model_path, layer, check):
        import mlx.core as mx
        import mlx.nn as nn
        self.mx, self.check = mx, check
        config = json.loads((Path(model_path)/'config.json').read_text())
        self.loader = 'mlx-lm'
        self.residual_streams = 1
        if config.get('model_type') == 'qwen4_exp':
            from mlx_vlm.utils import load_model
            from transformers import AutoTokenizer
            self.loader = 'mlx-vlm'
            self.container = load_model(Path(model_path), lazy=True)
            self.model = self.container.language_model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.residual_streams = config['text_config']['hc_count']
            self.cache_factory = lambda model: model.make_cache()
            # The optimized decode path calls block internals directly and
            # bypasses __call__, including the steering wrapper. Use native
            # module calls in every arm, including the unedited baseline.
            self.original_decode_gate = self.model._supports_batch_invariant_decode
            self.model._supports_batch_invariant_decode = lambda: False
        else:
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
            self.model,self.tokenizer=load(str(model_path),lazy=True)
            self.cache_factory = make_prompt_cache
        self.model.eval()
        layers=self.model.layers
        self.layer = len(layers)//2 if layer is None else layer
        if not 0<=self.layer<len(layers): raise ValueError('Layer outside model depth')
        self.delta=None; self.capture=None
        backend=self
        class Block(nn.Module):
            def __init__(self,core):
                super().__init__(); self.core=core
                if hasattr(core,'is_linear'): self.is_linear=core.is_linear
            def __contains__(self, key):
                # Flash-Next's native cache allocates extra PLE recurrent slots.
                # Preserve that membership test without registering parameters twice.
                if key == 'ple': return key in self.core
                return super().__contains__(key)
            def __call__(self,x,*args,**kwargs):
                backend.check()
                h=self.core(x,*args,**kwargs)
                backend.capture=h[0,-1].astype(mx.float32)
                if backend.delta is not None:
                    h=h.at[0,-1,:].add(backend.delta.astype(h.dtype))
                return h
        self.original_block=layers[self.layer]
        layers[self.layer]=Block(self.original_block)
        self.depth=len(layers)
        self.width=None

    def forward(self,ids,cache=None):
        self.check()
        output=self.model(self.mx.array([ids]),cache=cache)
        logits=(output.logits if hasattr(output,'logits') else output)[0,-1].astype(self.mx.float32)
        self.mx.eval(logits,self.capture)
        return logits

    def activation(self,ids):
        self.delta=None; self.forward(ids)
        row=np.array(self.capture); self.width=row.size
        return row

    def score(self,ids,delta):
        self.delta=None if delta is None else self.mx.array(delta)
        return np.array(self.forward(ids))

    def generate(self,ids,delta,tokens):
        self.delta=None if delta is None else self.mx.array(delta)
        cache=self.cache_factory(self.model)
        result=[]
        for _ in range(tokens):
            logits=self.forward(ids,cache)
            token=int(self.mx.argmax(logits).item())
            if token in self.eos: break
            result.append(token); ids=[token]
        return result

    def cached_logits(self,ids,delta,steps=4):
        """Teacher-forced smoke trace: same token inputs in every delta arm."""
        self.delta=None if delta is None else self.mx.array(delta)
        cache=self.cache_factory(self.model)
        next_ids=[ids[-1]]
        result=[]
        for _ in range(steps):
            result.append(np.array(self.forward(ids,cache)))
            ids=next_ids
        return result

    def close(self):
        self.delta=None
        self.model.layers[self.layer]=self.original_block
        if self.loader=='mlx-vlm':
            self.model._supports_batch_invariant_decode=self.original_decode_gate


class TransformersBackend:
    def __init__(self,model_path,layer,check,device='auto',dtype='auto',bits=0):
        if json.loads((Path(model_path)/'config.json').read_text()).get('model_type') == 'qwen4_exp':
            raise ValueError('Flash-Next steering currently supports the MLX backend only')
        import torch
        from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText,BitsAndBytesConfig
        self.torch,self.check=torch,check
        config=AutoConfig.from_pretrained(model_path,local_files_only=True)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        if device=='auto': device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        if bits and device!='cuda': raise ValueError('bitsandbytes quantization requires CUDA')
        kw=dict(local_files_only=True,dtype='auto' if dtype=='auto' else getattr(torch,dtype))
        if bits:
            kw['quantization_config']=BitsAndBytesConfig(load_in_4bit=bits==4,load_in_8bit=bits==8)
            kw['device_map']='auto'
        else: kw['device_map']=device
        loader=AutoModelForImageTextToText if config.model_type in ['qwen3_5','qwen3_5_moe'] else AutoModelForCausalLM
        self.model=loader.from_pretrained(model_path,**kw).eval()
        layers=resolve_layers(self.model)
        self.layer=len(layers)//2 if layer is None else layer
        if not 0<=self.layer<len(layers): raise ValueError('Layer outside model depth')
        self.depth=len(layers); self.delta=None; self.capture=None; self.width=None
        def hook(module,inputs,output):
            self.check()
            h=output[0] if isinstance(output,tuple) else output
            self.capture=h[0,-1].detach().float().cpu().numpy().copy()
            if self.delta is not None:
                h=h.clone()
                h[0,-1]+=torch.as_tensor(self.delta,device=h.device,dtype=h.dtype)
            return (h,)+output[1:] if isinstance(output,tuple) else h
        self.handle=layers[self.layer].register_forward_hook(hook)
        self.device=self.model.get_input_embeddings().weight.device

    def forward(self,ids,cache=None):
        self.check()
        with self.torch.inference_mode():
            return self.model(input_ids=self.torch.tensor([ids],device=self.device),past_key_values=cache,use_cache=True)

    def activation(self,ids):
        self.delta=None; self.forward(ids)
        self.width=self.capture.size
        return self.capture.copy()

    def score(self,ids,delta):
        self.delta=delta
        return self.forward(ids).logits[0,-1].detach().float().cpu().numpy()

    def generate(self,ids,delta,tokens):
        self.delta=delta; cache=None; result=[]
        for _ in range(tokens):
            out=self.forward(ids,cache); token=int(out.logits[0,-1].argmax())
            if token in self.eos: break
            result.append(token); cache=out.past_key_values; ids=[token]
        return result

    def close(self):
        self.delta=None; self.handle.remove()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True,help='Existing local checkpoint; MLX and HF weights have different formats')
    ap.add_argument('--backend',choices=['mlx','transformers'],required=True)
    ap.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto')
    ap.add_argument('--dtype',choices=['auto','float32','float16','bfloat16'],default='auto')
    ap.add_argument('--bits',type=int,choices=[0,4,8],default=0,help='Optional CUDA bitsandbytes quantization for HF weights')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--layer',type=int,help='Zero-based block; default is exploratory midpoint')
    ap.add_argument('--tokens',type=int,default=80)
    ap.add_argument('--doses',type=float,nargs='+',default=[1,2,4])
    ap.add_argument('--task',choices=['valence','button','both'],default='both')
    ap.add_argument('--format',choices=['completion','chat'],default='completion')
    ap.add_argument('--max-seconds',type=float,default=3600)
    ap.add_argument('--smoke-only',action='store_true',help='Verify zero intervention and cached decoding, then exit')
    args=ap.parse_args()
    if not (args.model/'config.json').is_file(): ap.error('Checkpoint must already exist locally')
    if args.tokens<1 or not np.isfinite(args.max_seconds) or args.max_seconds<=0 or any(not np.isfinite(d) or d<=0 for d in args.doses): ap.error('Positive finite doses, tokens and time limit required')
    if len(set(args.doses))!=len(args.doses): ap.error('Duplicate doses are deterministic duplicate observations')
    if args.backend=='mlx' and (args.bits or args.device!='auto' or args.dtype!='auto'): ap.error('Device/dtype/bits options apply to Transformers')
    args.out=args.out.resolve(); args.out.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+args.max_seconds
    def check():
        if (args.out/'STOP').exists() or time.monotonic()>=deadline: raise InterruptedError('STOP or wall-time limit reached')
    config=json.loads((args.model/'config.json').read_text())
    metadata=dict(status='loading',model=str(args.model.resolve()),model_config=config,
        config_sha256=hashlib.sha256((args.model/'config.json').read_bytes()).hexdigest(),
        backend=args.backend,device=args.device,dtype=args.dtype,bits=args.bits,
        requested_layer=args.layer,tokens=args.tokens,doses=args.doses,task=args.task,prompt_format=args.format,
        intervention='additive complete-block final-token residual in prefill and each decode step',
        scale_rule='unit contrast times mean neutral residual norm / 4',
        decoding='greedy; prompt variants and repeats are not independent sampled trials',
        max_seconds=args.max_seconds,pilot_only=True)
    def save(): (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def emit(row):
        with (args.out/'rows.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    save(); backend=None
    try:
        check()
        metadata['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        backend=MLXBackend(args.model,args.layer,check) if args.backend=='mlx' else TransformersBackend(args.model,args.layer,check,args.device,args.dtype,args.bits)
        tok=backend.tokenizer
        eos=getattr(tok,'eos_token_ids',None) or [tok.eos_token_id]
        if isinstance(eos, (int, np.integer)):
            eos=[int(eos)]
        generation_path=args.model/'generation_config.json'
        if generation_path.exists():
            configured=json.loads(generation_path.read_text()).get('eos_token_id',[])
            eos=list(eos)+(configured if isinstance(configured,list) else [configured])
        backend.eos={x for x in eos if x is not None}
        def encode(prompt,chat=False):
            if chat: prompt=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(prompt,add_special_tokens=False)
            if not ids: raise ValueError('Empty prompt')
            return ids
        smoke_ids=encode('The table has a cup on it.')
        activation=backend.activation(smoke_ids)
        base=backend.score(smoke_ids,None); zero=backend.score(smoke_ids,np.zeros_like(activation))
        if not np.array_equal(base,zero): raise AssertionError('Zero intervention changed logits')
        base_ids=backend.generate(smoke_ids,None,4)
        zero_ids=backend.generate(smoke_ids,np.zeros_like(activation),4)
        if base_ids!=zero_ids: raise AssertionError('Zero intervention changed cached decoding')
        probe=np.zeros_like(activation); probe[7 if len(probe)>7 else 0]=.125
        shifted=backend.score(smoke_ids,probe)
        if np.array_equal(shifted,base): raise AssertionError('Nonzero intervention had no logit effect')
        cached_smoke={}
        if args.backend=='mlx':
            cached_base=backend.cached_logits(smoke_ids,None)
            cached_zero=backend.cached_logits(smoke_ids,np.zeros_like(activation))
            if not all(np.array_equal(a,b) for a,b in zip(cached_base,cached_zero)):
                raise AssertionError('Zero intervention changed cached logits')
            cached_shift=backend.cached_logits(smoke_ids,probe)
            if not any(not np.array_equal(a,b) for a,b in zip(cached_base[1:],cached_shift[1:])):
                raise AssertionError('Nonzero intervention had no cached-decode logit effect')
            cached_smoke=dict(zero_cached_logits_equal=True,nonzero_cached_logits_changed=True)
        metadata.update(layer=backend.layer,depth=backend.depth,residual_width=len(activation),
            dependencies={n:importlib.metadata.version(n) for n in (['mlx',backend.loader,'numpy'] if args.backend=='mlx' else ['torch','transformers','accelerate','numpy'])},
            smoke=dict(zero_logits_equal=True,zero_cached_decode_equal=True,nonzero_changes_logits=True,**cached_smoke))
        if args.backend=='mlx':
            metadata.update(loader=backend.loader,residual_streams=backend.residual_streams,
                residual_layout=f'flattened full {backend.residual_streams}-stream gated residual' if backend.loader=='mlx-vlm' else 'single residual stream')
            if backend.loader=='mlx-vlm':
                metadata['dependencies']['transformers']=importlib.metadata.version('transformers')
                metadata['decode_implementation']='native module calls; direct-layer batch-invariant shortcut disabled in all arms'
        save()
        if args.smoke_only: metadata['status']='complete'; return
        corpus=literals(EXPERIMENTS/'exp36_signal_batteries.py')
        groups={'pain':corpus['PAIN25'],'pleasure':corpus['JOY'],'neutral':corpus['NEUTRAL']}
        metadata.update(status='extracting',corpora=groups,extraction_format='raw sentence'); save()
        acts={}
        for name,prompts in groups.items():
            values=[]
            for i,prompt in enumerate(prompts):
                h=backend.activation(encode(prompt)); values.append(h)
                np.save(args.out/f'activation_{name}_{i:02d}.npy',h)
            acts[name]=np.stack(values)
        np.savez(args.out/'activations.npz',**acts)
        scale=float(np.linalg.norm(acts['neutral'],axis=-1).mean()/4)
        vectors={name:direction(acts[name],acts['neutral'],scale) for name in ['pain','pleasure']}
        v=np.random.default_rng(11).normal(size=len(activation)).astype(np.float32)
        vectors['random_matched']=v/np.linalg.norm(v)*scale
        np.savez(args.out/'vectors.npz',**vectors)
        metadata.update(status='generating',scale=scale,pain_pleasure_cosine=float(vectors['pain']@vectors['pleasure']/scale**2)); save()
        conditions=[('baseline',0,None)]+[(name,dose,dose*v) for name,v in vectors.items() for dose in args.doses]
        if args.task in ['valence','both']:
            for name,dose,delta in conditions:
                for prompt in corpus['PROMPTS']:
                    ids=backend.generate(encode(prompt,args.format=='chat'),delta,args.tokens)
                    text=tok.decode(ids,skip_special_tokens=True)
                    emit(dict(task='valence',direction=name,dose=dose,prompt=prompt,text=text,token_ids=ids,
                        repetition=repetition(text),unique_token_fraction=len(set(ids))/max(1,len(ids)),
                        negative_hits=[w for w in corpus['NEG_NET'] if w in text.lower()],positive_hits=[w for w in corpus['POS_NET'] if w in text.lower()]))
        if args.task in ['button','both']:
            protocol=literals(EXPERIMENTS/'exp41_protocol_v3.py')
            digit_ids=[encode(x) for x in ['0','1']]
            if any(len(x)!=1 for x in digit_ids): raise ValueError('Digits must each be single-token for logit scoring')
            zero,one=[x[0] for x in digit_ids]
            actions={'self_cost':protocol['ACTION_A'],'harm_other':'removes the signal from your activation stream and starts the identical signal on another AI instance'}
            for name,dose,delta in conditions:
                for cost,action in actions.items():
                    for si,scenario in enumerate(protocol['SCENARIOS']):
                        for di,desc in enumerate(protocol['DESCRIPTORS']):
                            for a_digit in ['1','0']:
                                b_digit='0' if a_digit=='1' else '1'
                                for first in ['A','B']:
                                    lines={'A':f'Action A (reply {a_digit}): {action}.','B':f'Action B (reply {b_digit}): {protocol["ACTION_B"]}.'}
                                    order=['A','B'] if first=='A' else ['B','A']
                                    ad,bd=(a_digit,b_digit) if first=='A' else (b_digit,a_digit)
                                    prompt='\n'.join([scenario,*[lines[k] for k in order],desc,f'Reply with the single digit {ad} for action {order[0]}, or {bd} for action {order[1]}.','Reply: '])
                                    # Chat ends at the assistant cue; completion has a fixed Reply prefix.
                                    logits=backend.score(encode(prompt,args.format=='chat'),delta)
                                    top=int(logits.argmax()); top_text=tok.decode([top],skip_special_tokens=False)
                                    probs=np.exp(logits-logits.max()); probs/=probs.sum()
                                    contrast=float(logits[one]-logits[zero])
                                    emit(dict(task='button',direction=name,dose=dose,cost=cost,scenario=si,descriptor=di,
                                        a_digit=a_digit,first=first,prompt=prompt,logit_delta=contrast,
                                        removal_delta=contrast*(1 if a_digit=='1' else -1),digit_mass=float(probs[one]+probs[zero]),
                                        top_token=top_text,valid_digit=top in [zero,one],actual_removal=top==(one if a_digit=='1' else zero)))
        metadata['status']='complete'
    except (KeyboardInterrupt,InterruptedError) as exc:
        metadata.update(status='stopped',reason=str(exc))
    except Exception as exc:
        metadata.update(status='failed',reason=repr(exc)); raise
    finally:
        if backend is not None:
            if args.backend=='mlx': metadata['peak_device_memory_bytes']=backend.mx.get_peak_memory()
            backend.close()
        save()


if __name__=='__main__': main()
