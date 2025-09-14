import mteb
from mteb.encoder_interface import PromptType
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Union, Dict, Any
import os
from tokenizers import Tokenizer
from model import GPTConfig, GPT

class Tok:
    def __init__(self, tokenizer_json_path: str):
        if not os.path.exists(tokenizer_json_path):
            raise FileNotFoundError(f"Missing tokenizer file at {tokenizer_json_path}")
        self.tk = Tokenizer.from_file(tokenizer_json_path)
    def encode_ids(self, text: str) -> List[int]:
        return self.tk.encode(text, add_special_tokens=False).ids
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

class GPTEmbeddingWrapper:
    """Wrapper to make GPT model compatible with MTEB by adding encode method"""
    
    def __init__(self, model, tokenizer, device="cpu", max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
    def encode(self, 
               sentences: Union[str, List[str]], 
               prompt_type: PromptType = None,
               **kwargs) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
            
        embeddings = []
        
        self.model.eval()
        with torch.no_grad():
            for sentence in sentences:
                # Tokenize
                token_ids = self.tokenizer.encode_ids(sentence)
                
                # Truncate if too long
                if len(token_ids) > self.max_length:
                    token_ids = token_ids[:self.max_length]
                
                # Convert to tensor
                input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                
                # Get model outputs
                logits, hidden_states = self.model(input_ids)
                
                # Use the last hidden state as embedding (mean pooling)
                # Assuming your model returns hidden states as second output
                if hidden_states is not None:
                    embedding = hidden_states.mean(dim=1).squeeze()  # Mean pooling over sequence length
                else:
                    # Fallback: use the embedding layer output
                    # This assumes your model has a token embedding layer
                    with torch.no_grad():
                        embedding = self.model.transformer.wte(input_ids).mean(dim=1).squeeze()
                
                embeddings.append(embedding.cpu().numpy())
        
        return np.array(embeddings)

def _load_checkpoint(out_dir: str, map_location: str):
    ckpt_path = os.path.join(out_dir, "unigram_en_train.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    return torch.load(ckpt_path, map_location=map_location)

def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ["_orig_mod.", "module."]
    for pref in prefixes:
        keys = [k for k in list(sd.keys()) if k.startswith(pref)]
        for k in keys:
            sd[k[len(pref):]] = sd.pop(k)
    return sd

def _infer_vocab_embd_from_sd(sd: Dict[str, torch.Tensor]) -> (int, int):
    cand_keys = [k for k in sd.keys() if k.endswith("lm_head.weight")] + \
                [k for k in sd.keys() if "lm_head" in k and k.endswith(".weight")] + \
                [k for k in sd.keys() if k.endswith("wte.weight")] + \
                [k for k in sd.keys() if "embed_tokens.weight" in k or "tok_embeddings.weight" in k]
    for k in cand_keys:
        t = sd.get(k, None)
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            return t.shape[0], t.shape[1]
    for k, t in sd.items():
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            m, n = t.shape
            if any(s in k for s in ["attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight","mlp.c_proj.weight","ln_f.weight"]):
                return m, n
    raise KeyError("Could not infer vocab/embedding size from state dict")

def _init_model_from_checkpoint(ckpt, device: str):
    sd = _strip_prefixes(dict(ckpt["model"]))
    vocab, n_embd = _infer_vocab_embd_from_sd(sd)
    if "model_args" in ckpt:
        model_args = dict(ckpt["model_args"])
        model_args["vocab_size"] = vocab
        model_args["n_embd"] = n_embd if "n_embd" in model_args else n_embd
    else:
        model_args = dict(n_layer=12, n_head=12, n_embd=n_embd, block_size=1024, bias=False, vocab_size=vocab)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def main():
    tok_env = os.environ.get("TOKENIZER_JSON", "")
    tokenizer_json = tok_env if tok_env else os.path.join(".gitignore/tokenizers/bpe/en", "tokenizer.json")
    tok = Tok(tokenizer_json)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ckpt = _load_checkpoint(".gitignore/out", map_location="cpu")
    model = _init_model_from_checkpoint(ckpt, device=device)
    
    # Wrap the GPT model to make it compatible with MTEB
    embedding_model = GPTEmbeddingWrapper(model, tok, device=device)
    
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")
    evaluation = mteb.MTEB(tasks=benchmark)
    results = evaluation.run(embedding_model)  # Fixed: capture results
    
    output_file = 'results/bpe_en_mmteb.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Create directory if it doesn't exist
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()