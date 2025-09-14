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

def run_mmlu_only(scorer: LocalScorer, limit_per_subset: Optional[int] = None) -> Dict[str, Any]:
    subsets = [
        "abstract_algebra","anatomy","astronomy","business_ethics","clinical_knowledge","college_biology",
        "college_chemistry","college_computer_science","college_mathematics","college_medicine","college_physics",
        "computer_security","conceptual_physics","econometrics","electrical_engineering","elementary_mathematics",
        "formal_logic","global_facts","high_school_biology","high_school_chemistry","high_school_computer_science",
        "high_school_european_history","high_school_geography","high_school_government_and_politics",
        "high_school_macroeconomics","high_school_mathematics","high_school_microeconomics","high_school_physics",
        "high_school_psychology","high_school_statistics","high_school_us_history","high_school_world_history",
        "human_aging","human_sexuality","international_law","jurisprudence","logical_fallacies","machine_learning",
        "management","marketing","medical_genetics","miscellaneous","moral_disputes","moral_scenarios","nutrition",
        "philosophy","prehistory","professional_accounting","professional_law","professional_medicine",
        "professional_psychology","public_relations","security_studies","sociology","us_foreign_policy","virology",
        "world_religions"
    ]
    per_subject = {}
    total_correct, total_count = 0, 0
    for name in subsets:
        try:
            ds = load_dataset("cais/mmlu", name, split="test")
        except Exception:
            continue
        if limit_per_subset:
            ds = ds.select(range(min(limit_per_subset, len(ds))))
        correct, count = 0, 0
        for ex in ds:
            stem = ex["question"]
            choices = list(ex["choices"])
            labels = ["A","B","C","D"][:len(choices)]
            ctx = stem + "\n" + "\n".join([f"{labels[i]}. {choices[i]}" for i in range(len(choices))]) + "\nAnswer:"
            scores = [scorer.logprob_of(ctx + " ", labels[i]) for i in range(len(choices))]
            pred = int(max(range(len(scores)), key=lambda i: scores[i]))
            gold = int(ex["answer"])
            correct += int(pred == gold)
            count += 1
        if count > 0:
            per_subject[name] = {"accuracy": correct / count, "n": count}
            total_correct += correct
            total_count += count
    overall = {"accuracy": (total_correct / total_count) if total_count else 0.0, "n": total_count}
    return {"overall": overall, "per_subject": per_subject}

def main():
    tok_env = os.environ.get("TOKENIZER_JSON", "")
    tokenizer_json = tok_env if tok_env else os.path.join(".gitignore/tokenizers/unigram/en", "tokenizer.json")
    tok = Tok(tokenizer_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _load_checkpoint(".gitignore/out", map_location="cpu")
    model = _init_model_from_checkpoint(ckpt, device=device)
    scorer = LocalScorer(model=model, tok=tok, device=device)
    results = run_mmlu_only(scorer, limit_per_subset=None)
    with open('unigram_mmlu.json', 'w') as fp:
        json.dump(results, fp, indent=2)

if __name__ == "__main__":
    main()
