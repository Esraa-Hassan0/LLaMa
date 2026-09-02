import sys
from unittest.mock import MagicMock
sys.modules['sentencepiece'] = MagicMock()

import torch
from model import Transformer, ModelArgs
from inference import LLaMA

class MockTokenizer:
    def __init__(self):
        self.eos_id = 99
        
    def encode(self, prompt, out_type=int, add_bos=True, add_eos=False):
        # Return simple dummy tokens based on the prompt string length
        return [int(c) % 80 + 2 for c in prompt.encode("utf-8")]

    def pad_id(self):
        return 0

    def decode(self, tokens):
        return "".join([chr(t) for t in tokens if t not in (0, 99)])

    def vocab_size(self):
        return 100

def test_inference_and_model():
    print("Initializing ModelArgs...")
    args = ModelArgs(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=100,
        max_batch_size=4,
        max_seq_len=32,
        device="cpu"
    )
    
    print("Building Transformer model...")
    model = Transformer(args).to(args.device)
    
    print("Instantiating MockTokenizer...")
    tokenizer = MockTokenizer()
    
    print("Building LLaMA helper...")
    llama = LLaMA(model, tokenizer, args)
    
    prompts = [
        "Hello space",
        "AI is awesome!",
        "LLaMA 2 test run"
    ]
    
    print(f"Running text_completion with {len(prompts)} prompts...")
    # Using small max_gen_len
    tokens, texts = llama.text_completion(
        prompts,
        temperature=0.6,
        top_p=0.9,
        max_gen_len=10
    )
    
    print("Generation complete successfully!")
    print(f"Generated text list: {texts}")
    
    # Assert return types and lengths
    assert len(tokens) == len(prompts), f"Expected {len(prompts)} output token sequences, got {len(tokens)}"
    assert len(texts) == len(prompts), f"Expected {len(prompts)} output text strings, got {len(texts)}"
    
    # Assert shape limits
    for seq in tokens:
        assert len(seq) <= args.max_seq_len, "Generated sequence length exceeded max_seq_len"
        
    print("All checks passed successfully!")

if __name__ == "__main__":
    test_inference_and_model()
