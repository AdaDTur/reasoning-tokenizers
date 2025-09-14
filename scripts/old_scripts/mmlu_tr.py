import os
import json
from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from datasets import get_dataset_config_names
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
    # Fixed: Use wordpiece Turkish checkpoint instead of unigram English
    ckpt_path = os.path.join(out_dir, "wordpiece_tr_train.pt")
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

def run_mmlu_turkish(scorer: LocalScorer, limit_per_subset: Optional[int] = None) -> Dict[str, Any]:
    """
    Run MMLU evaluation on Turkish datasets.
    """
    
    dataset_name = "malhajar/mmlu-tr"
    
    # Fixed: Actually get the available subjects from the dataset
    try:
        available_subjects = get_dataset_config_names(dataset_name)
        print(f"Found {len(available_subjects)} subjects in {dataset_name}")
    except Exception as e:
        print(f"Could not get dataset configs: {e}")
        return {"error": "Could not load dataset configurations"}
    
    per_subject = {}
    total_correct, total_count = 0, 0
    
    for subject in available_subjects:
        try:
            # Fixed: Use the correct dataset_name variable, not dataset_found
            ds = load_dataset(dataset_name, subject, split="test")
            
            if limit_per_subset and len(ds) > 0:
                ds = ds.select(range(min(limit_per_subset, len(ds))))
            
            correct, count = 0, 0
            
            for ex in ds:
                try:
                    # Handle different possible data formats
                    if "question" in ex:
                        stem = ex["question"]
                    elif "soru" in ex:  # Turkish for "question"
                        stem = ex["soru"]
                    elif "text" in ex:
                        stem = ex["text"]
                    else:
                        print(f"Unknown question format in {subject}: available keys {list(ex.keys())}")
                        continue
                    
                    if "choices" in ex:
                        choices = list(ex["choices"])
                    elif "secenekler" in ex:  # Turkish for "choices"
                        choices = list(ex["secenekler"])
                    elif "options" in ex:
                        choices = list(ex["options"])
                    else:
                        print(f"Unknown choices format in {subject}: available keys {list(ex.keys())}")
                        continue
                    
                    # Get the correct answer
                    if "answer" in ex:
                        if isinstance(ex["answer"], int):
                            gold = ex["answer"]
                        else:
                            # Convert letter to number (A=0, B=1, C=2, D=3)
                            answer_str = str(ex["answer"]).upper().strip()
                            if answer_str in ['A', 'B', 'C', 'D', 'E']:
                                gold = ord(answer_str) - ord('A')
                            else:
                                try:
                                    gold = int(answer_str)
                                except:
                                    print(f"Could not parse answer: {ex['answer']}")
                                    continue
                    elif "cevap" in ex:  # Turkish for "answer"
                        if isinstance(ex["cevap"], int):
                            gold = ex["cevap"]
                        else:
                            answer_str = str(ex["cevap"]).upper().strip()
                            if answer_str in ['A', 'B', 'C', 'D', 'E']:
                                gold = ord(answer_str) - ord('A')
                            else:
                                try:
                                    gold = int(answer_str)
                                except:
                                    print(f"Could not parse answer: {ex['cevap']}")
                                    continue
                    else:
                        print(f"Unknown answer format in {subject}: available keys {list(ex.keys())}")
                        continue
                    
                    # Create labels (A, B, C, D, E, etc.)
                    labels = [chr(ord('A') + i) for i in range(len(choices))]
                    
                    # Fixed: Ensure gold answer is within valid range
                    if gold < 0 or gold >= len(choices):
                        print(f"Invalid gold answer {gold} for {len(choices)} choices in {subject}")
                        continue
                    
                    ctx = f"{stem}\n" + "\n".join([f"{labels[i]}. {choices[i]}" for i in range(len(choices))]) + "\nCevap:"
                    
                    # Calculate log probabilities for each choice
                    scores = [scorer.logprob_of(ctx + " ", labels[i]) for i in range(len(choices))]
                    pred = int(max(range(len(scores)), key=lambda i: scores[i]))
                    
                    correct += int(pred == gold)
                    count += 1
                    
                except Exception as e:
                    print(f"Error processing example in {subject}: {e}")
                    continue
            
            if count > 0:
                accuracy = correct / count
                per_subject[subject] = {"accuracy": accuracy, "n": count}
                total_correct += correct
                total_count += count
                print(f"{subject}: {accuracy:.3f} ({correct}/{count})")
            else:
                print(f"{subject}: No valid examples processed")
                
        except Exception as e:
            print(f"Could not load subject {subject}: {e}")
            continue
    
    overall_accuracy = (total_correct / total_count) if total_count > 0 else 0.0
    overall = {"accuracy": overall_accuracy, "n": total_count}
    
    print(f"\nOverall Turkish MMLU Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_count})")
    
    return {
        "overall": overall, 
        "per_subject": per_subject,
        "dataset_used": dataset_name,
        "total_subjects": len(per_subject)
    }

def main():
    tok_env = os.environ.get("TOKENIZER_JSON", "")
    tokenizer_json = tok_env if tok_env else os.path.join(".gitignore/tokenizers/wordpiece/tr", "tokenizer.json")
    tok = Tok(tokenizer_json)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ckpt = _load_checkpoint(".gitignore/out", map_location="cpu")
    model = _init_model_from_checkpoint(ckpt, device=device)
    
    scorer = LocalScorer(model=model, tok=tok, device=device)
    
    print("Starting Turkish MMLU evaluation...")
    results = run_mmlu_turkish(scorer, limit_per_subset=None)
    
    output_file = 'results/wordpiece_tr_mmlu.json'
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()