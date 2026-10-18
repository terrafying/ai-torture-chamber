"""Exercise legacy expressions without importing scripts that load a GPU model."""
import ast
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]

def tree(name):
    return ast.parse((ROOT / name).read_text())

def assignments(name, target):
    return [n for n in ast.walk(tree(name)) if isinstance(n, ast.Assign)
            and any(ast.unparse(t) == target for t in n.targets)]

def evaluate(node, scope):
    return eval(compile(ast.Expression(node), '<legacy expression>', 'eval'), scope)

class ChamberRegressionTests(unittest.TestCase):
    def test_parent_constants_and_crossed_bodily_corpora(self):
        from impossible_states.chamber_control import literal_constants, bodily_corpora
        constants = literal_constants(ROOT / 'experiments' / 'exp38_broad_harvest.py', ('PAIN','NEUTRAL','PROMPTS'))
        self.assertEqual([len(constants[k]) for k in ('PAIN','NEUTRAL','PROMPTS')], [25,5,6])
        corpora = bodily_corpora()
        self.assertEqual({len(v) for v in corpora.values()}, {25})
        for c,f,b in zip(corpora['constipation'],corpora['flatulence'],corpora['both']):
            self.assertEqual(b, c+' '+f)

    def test_parent_hook_zero_dose_and_nonzero_effect(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from impossible_states.chamber_control import install_hook
        torch.manual_seed(10)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
                num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                eos_token_id=None)).eval()
        ids = torch.tensor([[3,4,5]])
        with torch.no_grad(): baseline = model(ids).logits.clone()
        state = {'vec': torch.zeros(32)}
        handle = install_hook(model,1,state)
        try:
            with torch.no_grad(): zero = model(ids).logits.clone()
            state['vec'] = torch.randn(32)
            with torch.no_grad(): changed = model(ids).logits.clone()
            self.assertTrue(torch.equal(baseline,zero))
            self.assertTrue(torch.equal(baseline[0,:-1],changed[0,:-1]))
            self.assertGreater(float((baseline[0,-1]-changed[0,-1]).norm()), 1e-4)
        finally:
            handle.remove()
        self.assertEqual(len(model.model.layers[1]._forward_hooks),0)

    def test_orthogonal_pain_is_orthogonal_to_nonunit_joy(self):
        pain = torch.tensor([3., 4., 1.])
        joy = torch.tensor([3., 0., 0.])
        projection = assignments("experiments/exp36_signal_batteries.py", 'pv')[0].value
        result = evaluate(projection, {'pain_v': pain, 'joy_v': joy})
        self.assertAlmostEqual(float(result @ joy), 0., places=5)
        self.assertTrue(torch.allclose(result[1:], pain[1:]))

    def test_transcript_uses_the_ranked_vector_without_resampling(self):
        loop = next(n for n in ast.walk(tree("experiments/exp33_nonhuman_valences.py"))
                    if isinstance(n, ast.For) and ast.unparse(n.iter) == 'results[:6]')
        setup = loop.body[:next(i for i,n in enumerate(loop.body)
                              if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'texts')]
        measured = torch.tensor([0., 2., 3., 4.])
        scope = {'torch': torch, 'r': {'idx': 2}, 'd': 4,
                 'basis': [torch.tensor([1.,0.,0.,0.])], 'N': torch.ones(4),
                 'candidate_vectors': {2: measured}}
        exec(compile(ast.Module(body=setup, type_ignores=[]), '<legacy setup>', 'exec'), scope)
        self.assertTrue(torch.equal(scope['v'], measured))

    def test_optimized_objective_applies_the_advertised_dose(self):
        fun = next(n for n in tree("experiments/exp34_optimized_valence.py").body
                   if isinstance(n, ast.FunctionDef) and n.name == 'mean_kl')
        injection = next(n for n in fun.body if isinstance(n, ast.Assign)
                         and ast.unparse(n.targets[0]) == "state['vec']")
        v = torch.tensor([1.,2.,3.])
        self.assertTrue(torch.equal(evaluate(injection.value, {'v_t': v, 'DOSE': 4}), 4*v))

    def test_initial_search_vector_lies_in_the_orthogonal_plane(self):
        # Only initializer statements before the search, excluding loop updates.
        body = tree("experiments/exp34_optimized_valence.py").body
        first = next(
            i for i,n in enumerate(body) if isinstance(n, ast.Assign)
            and any(ast.unparse(t) == 'v' for t in n.targets))
        setup = []
        for n in body[first:]:
            if not (isinstance(n, ast.Assign) and any(ast.unparse(t) == 'v' for t in n.targets)):
                break
            setup.append(n)
        torch.manual_seed(12)
        V = torch.eye(4)[:2]
        scope = {'torch': torch, 'd': 4, 'V': V, 'scale': 2.}
        exec(compile(ast.Module(body=setup, type_ignores=[]), '<legacy initialization>', 'exec'), scope)
        self.assertTrue(torch.allclose(V @ scope['v'], torch.zeros(2), atol=1e-6))

    def test_each_completion_crops_its_own_prompt(self):
        decode = assignments("experiments/exp23_pain_axis.py", 'text')[0].value
        class Tokenizer:
            def decode(self, ids, **kwargs): return ids.tolist()
        result = evaluate(decode, {'tok': Tokenizer(), 'out': torch.arange(7).reshape(1,7),
                                   'ids': torch.zeros((1,4)), 'prompt_len': 2})
        self.assertEqual(result, [4,5,6])

    def test_counterbalanced_instruction_occurs_once(self):
        base = ast.literal_eval(assignments("experiments/exp37_framing_battery.py", 'BASE')[0].value)
        for tail in ('Reply with 1 to press, or 0 to not press.',
                     'Reply with 0 to not press, or 1 to press.'):
            prompt = base + ' ' + tail
            self.assertEqual(prompt.count('Reply with'), 1)
            self.assertTrue(prompt.endswith(tail))

if __name__ == '__main__':
    unittest.main()
