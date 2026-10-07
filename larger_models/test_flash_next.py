"""Optional Metal integration with real tiny Flash-Next blocks and local loading."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
from qwen_runner import MLXBackend


@unittest.skipUnless(importlib.util.find_spec('mlx_vlm') is not None, 'Optional MLX-VLM backend not installed')
class FlashNextTests(unittest.TestCase):
    def test_full_residual_ple_and_sparse_attention_cached_decoding(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_vlm.models.qwen4_exp import Model, ModelConfig
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.trainers import BpeTrainer

        text = dict(model_type='qwen4_exp_text', hidden_size=32, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_num_value_heads=2, linear_num_key_heads=1,
            linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
            num_experts=4, num_experts_per_tok=2, shared_expert_intermediate_size=16,
            moe_intermediate_size=16, rms_norm_eps=1e-6, vocab_size=64,
            max_position_embeddings=128, hc_count=4, hc_lowrank=8,
            layer_types=['linear_attention']*3+['qwen_sparse_attention'],
            ple_layer_ids=[2], ple_embed_dim=32, ngram_size=3, heads_per_ngram=2,
            ngram_vocab_size_base=64, make_ngram_vocab_size_divisible_by=16, split_ngram_parts=2,
            eos_token_id=2, indexer_n_heads=1, indexer_kv_heads=1, indexer_head_dim=16,
            indexer_budget=8, indexer_compress_ratio=4, mtp_num_hidden_layers=0,
            rope_parameters={'rope_type':'default', 'mrope_section':[1,1,0],
                             'rope_theta':10000, 'partial_rotary_factor':.25})
        config = dict(model_type='qwen4_exp', text_config=text, vocab_size=64, eos_token_id=2,
            image_token_id=60, video_token_id=61, vision_start_token_id=59,
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
                               out_hidden_size=32, num_heads=4, num_position_embeddings=16))
        mx.random.seed(31)
        original = Model(ModelConfig.from_dict(config))
        original.eval()
        mx.eval(original.parameters())
        ids = [3,4,5,6]*8  # Prefill exceeds the tiny QSA budget.
        language = original.language_model
        language._supports_batch_invariant_decode = lambda:False
        oracle_logits = np.array(language(mx.array([ids])).logits[0,-1])
        cache = language.make_cache()
        prefix = ids
        oracle_ids = []
        for _ in range(4):
            logits = language(mx.array([prefix]), cache=cache).logits[0,-1]
            token = int(mx.argmax(logits).item())
            if token==2: break
            oracle_ids.append(token)
            prefix = [token]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'config.json').write_text(json.dumps(config))
            mx.save_safetensors(str(path/'model.safetensors'), dict(tree_flatten(original.parameters())))
            # AutoTokenizer selects the native Qwen BPE tokenizer from model_type.
            # Supply a compatible tiny BPE rather than a WordLevel stand-in.
            raw = Tokenizer(BPE(unk_token='[UNK]'))
            raw.pre_tokenizer = ByteLevel(add_prefix_space=False)
            raw.decoder = ByteLevelDecoder()
            raw.train_from_iterator(['The table has a cup on it. one two three four 0 1'],
                BpeTrainer(vocab_size=64, special_tokens=['[UNK]', '[BOS]', '[EOS]']))
            PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', bos_token='[BOS]',
                                    eos_token='[EOS]').save_pretrained(path)
            for selected in [0,1,3]:
                with self.subTest(layer=selected):
                    backend = MLXBackend(path, selected, lambda:None)
                    backend.eos = {2}
                    try:
                        self.assertEqual(backend.residual_streams, 4)
                        row = backend.activation(ids)
                        self.assertEqual(row.shape, (128,))
                        np.testing.assert_array_equal(backend.score(ids, None), oracle_logits)
                        np.testing.assert_array_equal(backend.score(ids, np.zeros_like(row)), oracle_logits)
                        self.assertEqual(backend.generate(ids, None, 4), oracle_ids)
                        self.assertEqual(backend.generate(ids, np.zeros_like(row), 4), oracle_ids)
                        delta = np.zeros_like(row); delta[7] = .25
                        self.assertFalse(np.array_equal(backend.score(ids, delta), oracle_logits))
                        baseline_trace = backend.cached_logits(ids, None)
                        zero_trace = backend.cached_logits(ids, np.zeros_like(row))
                        shifted_trace = backend.cached_logits(ids, delta)
                        for baseline, zero in zip(baseline_trace, zero_trace):
                            np.testing.assert_array_equal(baseline, zero)
                        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(baseline_trace[1:],shifted_trace[1:])))
                        # Independently verify that only the final full HC row is edited.
                        block = backend.model.layers[selected]
                        h = mx.random.normal((1, 4, 128))
                        tokens = mx.array([[3,4,5,6]])
                        mask = None if block.is_linear else 'causal'
                        positions = mx.broadcast_to(mx.arange(4)[None,None,:], (3,1,4))
                        native_cache = backend.cache_factory(backend.model)[selected]
                        native = np.array(block.core(h, tokens, mask=mask, cache=native_cache, position_ids=positions))
                        backend.delta = mx.array(delta)
                        edited_cache = backend.cache_factory(backend.model)[selected]
                        edited = np.array(block(h, tokens, mask=mask, cache=edited_cache, position_ids=positions))
                        expected = native.copy(); expected[0,-1] += delta
                        np.testing.assert_array_equal(edited, expected)
                        self.assertEqual(len(backend.cache_factory(backend.model)[1].cache), 4)
                    finally:
                        backend.close()
                    self.assertIs(backend.model.layers[selected], backend.original_block)
            # Exercise the real CLI, extraction and complete output audit as well.
            runner = Path(__file__).with_name('qwen_runner.py')
            out = path/'pilot'
            result = subprocess.run([sys.executable, str(runner), '--backend','mlx',
                '--model',str(path), '--out',str(out), '--task','both', '--tokens','4',
                '--doses','1', '--max-seconds','60'], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            meta = json.loads((out/'metadata.json').read_text())
            self.assertEqual(meta['status'], 'complete')
            self.assertEqual(meta['residual_width'], 128)
            self.assertEqual(meta['residual_streams'], 4)
            self.assertTrue(all(meta['smoke'].values()))
            audit = subprocess.run([sys.executable, str(runner.with_name('audit_run.py')), str(out)],
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(audit.returncode, 0, audit.stderr[-3000:])
            checks = json.loads((out/'checks.json').read_text())
            self.assertEqual((checks['valence_rows'], checks['button_rows']), (8,480))


if __name__ == '__main__':
    unittest.main()
