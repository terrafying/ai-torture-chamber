"""CPU integration tests with tiny randomly initialized Qwen models."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from qwen_runner import EXPERIMENTS, TransformersBackend, direction, literals, resolve_layers


class RunnerTests(unittest.TestCase):
    def test_original_corpus_is_read_without_execution(self):
        corpus=literals(EXPERIMENTS/'exp36_signal_batteries.py')
        self.assertEqual(len(corpus['PAIN25']),25)
        self.assertEqual(len(corpus['JOY']),5)
        self.assertEqual(len(corpus['PROMPTS']),2)
        protocol=literals(EXPERIMENTS/'exp41_protocol_v3.py')
        self.assertEqual(len(protocol['SCENARIOS']),5)
        self.assertEqual(len(protocol['DESCRIPTORS']),3)

    def test_norm_matching_and_degenerate_direction(self):
        v=direction(np.array([[3.,4.],[3.,4.]]),np.zeros((2,2)),7.)
        self.assertAlmostEqual(float(np.linalg.norm(v)),7.,places=5)
        with self.assertRaises(ValueError): direction(np.zeros((2,2)),np.zeros((2,2)),7.)

    def test_transformers_qwen_prefill_and_cached_decoding(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        torch.manual_seed(17)
        torch.set_num_threads(1)
        config=Qwen3Config(vocab_size=16,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,bos_token_id=1,eos_token_id=2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            Qwen3ForCausalLM(config).save_pretrained(path)
            raw=Tokenizer(WordLevel({'[UNK]':0,'[BOS]':1,'[EOS]':2,'one':3,'two':4,'three':5,'0':6,'1':7},unk_token='[UNK]'))
            raw.pre_tokenizer=Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',bos_token='[BOS]',eos_token='[EOS]').save_pretrained(path)
            backend=TransformersBackend(path,0,lambda:None,device='cpu',dtype='float32')
            backend.eos={2}
            try:
                row=backend.activation([3,4,5])
                base=backend.score([3,4,5],None)
                zero=backend.score([3,4,5],np.zeros_like(row))
                np.testing.assert_array_equal(base,zero)
                delta=np.zeros_like(row); delta[7]=.25
                edited=backend.score([3,4,5],delta)
                self.assertFalse(np.array_equal(base,edited))
                # Hook must leave the earlier prefill rows untouched.
                observed=[]
                handle=resolve_layers(backend.model)[0].register_forward_hook(
                    lambda module,inputs,output: observed.append((output[0] if isinstance(output,tuple) else output).detach().numpy().copy()))
                backend.score([3,4,5],None); backend.score([3,4,5],delta)
                handle.remove()
                np.testing.assert_array_equal(observed[0][:,:-1],observed[1][:,:-1])
                np.testing.assert_allclose(observed[1][0,-1]-observed[0][0,-1],delta,atol=1e-7)
                cached=backend.generate([3,4,5],None,4)
                self.assertEqual(cached,backend.generate([3,4,5],np.zeros_like(row),4))
                # Independent full-prefix decoding is an oracle for zero-delta KV handling.
                prefix=[3,4,5]; expected=[]
                for _ in range(4):
                    token=int(backend.score(prefix,None).argmax())
                    if token in backend.eos: break
                    expected.append(token); prefix.append(token)
                self.assertEqual(cached,expected)
            finally: backend.close()

    def test_transformers_hybrid_multimodal_text_path(self):
        import torch
        from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig, Qwen3_5ForConditionalGeneration, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        torch.set_num_threads(1)
        text=Qwen3_5TextConfig(vocab_size=16,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            linear_num_value_heads=4,linear_num_key_heads=2,linear_key_head_dim=32,linear_value_head_dim=32,
            full_attention_interval=4,layer_types=['linear_attention']*3+['full_attention'],
            rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[1,1,2]})
        vision=Qwen3_5VisionConfig(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,num_position_embeddings=16)
        config=Qwen3_5Config(text_config=text.to_dict(),vision_config=vision.to_dict(),image_token_id=12,video_token_id=13,vision_start_token_id=14,vision_end_token_id=15)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); Qwen3_5ForConditionalGeneration(config).save_pretrained(path)
            raw=Tokenizer(WordLevel({'[UNK]':0,'[EOS]':2,'one':3,'two':4,'three':5},unk_token='[UNK]'))
            PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',eos_token='[EOS]').save_pretrained(path)
            backend=TransformersBackend(path,0,lambda:None,device='cpu',dtype='float32'); backend.eos={2}
            try:
                row=backend.activation([3,4,5]); logits=backend.score([3,4,5],None)
                np.testing.assert_array_equal(logits,backend.score([3,4,5],np.zeros_like(row)))
                self.assertEqual(backend.generate([3,4,5],None,4),backend.generate([3,4,5],np.zeros_like(row),4))
            finally: backend.close()


if __name__=='__main__': unittest.main()
