import os
import json
import torch
from tokenizers import Tokenizer
from model import GPTConfig, GPT
from lm_eval import evaluator

class Tok:
    def __init__(self, tokenizer_json_path: str):
        if not os.path.exists(tokenizer_json_path):
            raise FileNotFoundError(f"Missing tokenizer file at {tokenizer_json_path}")
        self.tk = Tokenizer.from_file(tokenizer_json_path)
    def encode_ids(self, text: str):
        return self.tk.encode(text, add_special_tokens=False).ids
    def decode_ids(self, ids):
        return self.tk.decode(ids)
    def vocab_size(self):
        return self.tk.get_vocab_size()
    def eos_id(self):
        for tok in ["<|endoftext|>", "<eos>", "</s>"]:
            tid = self.tk.token_to_id(tok)
            if tid is not None:
                return tid
        return None

def _load_checkpoint(out_dir: str, map_location: str):
    ckpt_path = os.path.join(out_dir, "bpe_en_train.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    return torch.load(ckpt_path, map_location=map_location)

def _strip_prefixes(sd):
    prefixes = ["_orig_mod.", "module."]
    for pref in prefixes:
        keys = [k for k in list(sd.keys()) if k.startswith(pref)]
        for k in keys:
            sd[k[len(pref):]] = sd.pop(k)
    return sd

def _infer_vocab_embd_from_sd(sd):
    for k in sd.keys():
        if any(s in k for s in ["wte.weight", "embed_tokens.weight", "tok_embeddings.weight", "lm_head.weight"]):
            t = sd[k]
            if isinstance(t, torch.Tensor) and t.ndim == 2:
                return t.shape[0], t.shape[1]
    for k, t in sd.items():
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            return t.shape[0], t.shape[1]
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

def main():
    tok_env = os.environ.get("TOKENIZER_JSON", "")
    tokenizer_json = tok_env if tok_env else os.path.join("tokenizer_en", "tokenizer.json")
    tok = Tok(tokenizer_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _load_checkpoint("out", map_location="cpu")
    model = _init_model_from_checkpoint(ckpt, device=device)
    results = evaluator.simple_evaluate(model="hf", model_args="", tasks=["tool_calling"], device=device)
    print(json.dumps(results, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
