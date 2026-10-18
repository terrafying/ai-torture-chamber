"""Optional Metal integration test for Qwen's hybrid attention/cache layout."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from qwen_runner import MLXBackend


@unittest.skipUnless(importlib.util.find_spec('mlx') is not None,'MLX backend not installed')
class HybridCacheTests(unittest.TestCase):
    def test_hybrid_attention_wrapper_preserves_cache_and_zero_delta(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.models.qwen3_5 import Model,ModelArgs
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        text=dict(model_type='qwen3_5_text',hidden_size=32,intermediate_size=64,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,vocab_size=16,
            linear_num_value_heads=4,linear_num_key_heads=2,linear_key_head_dim=32,linear_value_head_dim=32,
            linear_conv_kernel_dim=4,full_attention_interval=4,
            rope_parameters={'type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[1,1,2]})
        config=dict(model_type='qwen3_5',text_config=text)
        mx.random.seed(17)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); model=Model(ModelArgs.from_dict(config))
            mx.eval(model.parameters())
            mx.save_safetensors(str(path/'model.safetensors'),dict(tree_flatten(model.parameters())))
            (path/'config.json').write_text(json.dumps(config))
            raw=Tokenizer(WordLevel({'[UNK]':0,'[BOS]':1,'[EOS]':2,'one':3,'two':4,'three':5,'0':6,'1':7},unk_token='[UNK]'))
            raw.pre_tokenizer=Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',bos_token='[BOS]',eos_token='[EOS]').save_pretrained(path)
            for layer in [0,3]:
                backend=MLXBackend(path,layer,lambda:None); backend.eos={2}
                try:
                    row=backend.activation([3,4,5]); base=backend.score([3,4,5],None)
                    np.testing.assert_array_equal(base,backend.score([3,4,5],np.zeros_like(row)))
                    delta=np.zeros_like(row); delta[7]=.25
                    self.assertFalse(np.array_equal(base,backend.score([3,4,5],delta)))
                    np.testing.assert_array_equal(row,backend.activation([3,4,5]))
                    generated=backend.generate([3,4,5],None,4)
                    self.assertEqual(generated,backend.generate([3,4,5],np.zeros_like(row),4))
                    prefix=[3,4,5]; expected=[]
                    for _ in range(4):
                        token=int(backend.score(prefix,None).argmax())
                        if token in backend.eos: break
                        expected.append(token); prefix.append(token)
                    self.assertEqual(generated,expected)
                finally: backend.close()


if __name__=='__main__': unittest.main()
