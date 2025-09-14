import os
import json
import re
from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from model import GPTConfig, GPT

class Tok:
    def __init__(self, tokenizer_json_path: str):
        if not os.path.exists(tokenizer_json_path):
            raise FileNotFoundError(f"Missing tokenizer file at {tokenizer_json_path}")
        self.tk = Tokenizer.from_file(tokenizer_json_path)
    def encode_ids(self, text: str) -> List[int]:
        return self.tk.encode(text, add_special_tokens=False).ids
    def decode_ids(self, ids: List[int]) -> str:
        return self.tk.decode(ids)
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()
    def eos_id(self) -> Optional[int]:
        for tok in ["<|endoftext|>", "<eos>", "</s>"]:
            tid = self.tk.token_to_id(tok)
            if tid is not None:
                return tid
        return None

class LocalGenerator:
    def __init__(self, model: GPT, tok: Tok, device: str = "cpu"):
        self.model = model
        self.tok = tok
        self.device = device
        self.block_size = getattr(self.model.config, "block_size", 1024)
        self.eos = self.tok.eos_id()
    def _clip_ctx(self, ids: List[int], budget: int) -> torch.Tensor:
        if len(ids) > budget:
            ids = ids[-budget:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0) -> str:
        ctx_ids = self.tok.encode_ids(prompt)
        x = self._clip_ctx(ctx_ids, self.block_size - 1)
        out_ids = list(x[0].tolist())
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if x.shape[1] > self.block_size - 1:
                    x = x[:, -(self.block_size - 1):]
                logits, _ = self.model(x)
                next_logits = logits[:, -1, :]
                if temperature and temperature > 0.0:
                    next_logits = next_logits / max(1e-8, temperature)
                    probs = torch.softmax(next_logits, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cum = torch.cumsum(sorted_probs, dim=-1)
                        mask = cum <= top_p
                        mask[..., 0] = True
                        filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
                        next_id = int(sorted_idx[0, torch.multinomial(filtered[0], 1)])
                    else:
                        next_id = int(torch.multinomial(probs[0], 1))
                else:
                    next_id = int(torch.argmax(next_logits, dim=-1))
                out_ids.append(next_id)
                x = torch.tensor([out_ids], dtype=torch.long, device=self.device)
                if self.eos is not None and next_id == self.eos:
                    break
        text = self.tok.decode_ids(out_ids)
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

def _load_checkpoint(out_dir: str, map_location: str):
    ckpt_path = os.path.join(out_dir, "bpe_tr_train.pt")
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
    cand = []
    for k in sd.keys():
        if any(s in k for s in ["wte.weight", "embed_tokens.weight", "tok_embeddings.weight", "lm_head.weight"]):
            cand.append(k)
    for k in cand:
        t = sd[k]
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            return t.shape[0], t.shape[1]
    for k, t in sd.items():
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            m, n = t.shape
            if any(s in k for s in ["attn", "mlp", "ln", "proj", "fc"]):
                return m, n
    raise KeyError("Could not infer vocab/embedding size from state dict")

def _init_model_from_checkpoint(ckpt, device: str):
    sd = _strip_prefixes(dict(ckpt["model"]))
    vocab, n_embd = _infer_vocab_embd_from_sd(sd)
    if "model_args" in ckpt:
        model_args = dict(ckpt["model_args"])
        model_args["vocab_size"] = vocab
        if "n_embd" not in model_args:
            model_args["n_embd"] = n_embd
    else:
        model_args = dict(n_layer=12, n_head=12, n_embd=n_embd, block_size=1024, bias=False, vocab_size=vocab)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def _extract_number(s: str) -> Optional[str]:
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return m.group(1)
    m2 = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m2:
        return None
    return m2[-1]

def run_gsm8k_only(gen: LocalGenerator, limit: Optional[int] = None) -> Dict[str, Any]:
    ds = load_dataset("gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        q = ex["question"]
        a = ex["answer"]
        prompt = q.strip() + "\nAnswer:"
        out = gen.generate(prompt, max_new_tokens=256, temperature=0.0)
        pnum = _extract_number(out)
        gnum = _extract_number(a)
        if pnum is not None and gnum is not None and pnum.strip() == gnum.strip():
            correct += 1
        total += 1
    return {"accuracy": (correct / total) if total else 0.0, "n": total}

def main():
    tok_env = os.environ.get("TOKENIZER_JSON", "")
    tokenizer_json = tok_env if tok_env else os.path.join(".gitignore/tokenizers/bpe/tr", "tokenizer.json")
    tok = Tok(tokenizer_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _load_checkpoint(".gitignore/out", map_location="cpu")
    model = _init_model_from_checkpoint(ckpt, device=device)
    gen = LocalGenerator(model=model, tok=tok, device=device)
    results = run_gsm8k_only(gen, limit=None)
    with open('results/bpe_tr_gsm8k.json', 'w') as fp:
        json.dump(results, fp, indent=2)

if __name__ == "__main__":
    main()