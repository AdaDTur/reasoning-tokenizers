import os
import json
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
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

class LocalScorer:
    def __init__(self, model: GPT, tok: Tok, device: str = "cpu"):
        self.model = model
        self.tok = tok
        self.device = device
        self.block_size = getattr(self.model.config, "block_size", 1024)
    def _clip_ctx(self, ids: List[int], budget: int) -> torch.Tensor:
        if len(ids) > budget:
            ids = ids[-budget:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    def logprob_of(self, context: str, continuation: str) -> float:
        ctx_ids = self.tok.encode_ids(context)
        x = self._clip_ctx(ctx_ids, self.block_size - 1)
        cont = self.tok.encode_ids(continuation)
        total = 0.0
        self.model.eval()
        with torch.no_grad():
            for tid in cont:
                if x.shape[1] > self.block_size - 1:
                    x = x[:, -(self.block_size - 1):]
                logits, _ = self.model(x)
                lp = F.log_softmax(logits[:, -1, :], dim=-1)[0, tid].item()
                total += float(lp)
                x = torch.cat([x, torch.tensor([[tid]], device=self.device)], dim=1)
        return total

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

def run_hellaswag_only(scorer: LocalScorer, split: str = "validation", limit: Optional[int] = None) -> Dict[str, Any]:
    ds = load_dataset("hellaswag", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        ctx_a = ex.get("ctx_a", "")
        ctx_b = ex.get("ctx_b", "")
        context = (ctx_a + " " + ctx_b).strip() + " "
        endings = list(ex["endings"])
        scores = [scorer.logprob_of(context, endings[i]) for i in range(len(endings))]
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        gold = int(ex["label"])
        correct += int(pred == gold)
        total += 1
    return {"accuracy": (correct / total) if total else 0.0, "n": total}

def main():
    tok_env = os.environ.get("TOKENIZER_JSON", "")
    tokenizer_json = tok_env if tok_env else os.path.join(".gitignore/tokenizers/unigram/en", "tokenizer.json")
    tok = Tok(tokenizer_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _load_checkpoint(".gitignore/out", map_location="cpu")
    model = _init_model_from_checkpoint(ckpt, device=device)
    scorer = LocalScorer(model=model, tok=tok, device=device)
    results = run_hellaswag_only(scorer, split="validation", limit=None)
    with open('results/unigram_hellaswag.json', 'w') as fp:
        json.dump(results, fp, indent=2)

if __name__ == "__main__":
    main()
