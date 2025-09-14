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
    
    def generate_answer(self, context: str, max_new_tokens: int = 256) -> str:
        """Generate answer for a given math problem context"""
        ctx_ids = self.tok.encode_ids(context)
        x = self._clip_ctx(ctx_ids, self.block_size - max_new_tokens)
        
        self.model.eval()
        with torch.no_grad():
            generated_ids = []
            for _ in range(max_new_tokens):
                if x.shape[1] >= self.block_size:
                    x = x[:, -self.block_size + 1:]
                
                logits, _ = self.model(x)
                # Use greedy decoding for mathematical reasoning
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
                # Check for end of sequence or stop conditions
                if next_token.item() == self.tok.tk.get_vocab().get("<|endoftext|>", -1):
                    break
                
                generated_ids.append(next_token.item())
                x = torch.cat([x, next_token], dim=1)
            
            # Decode the generated tokens
            if generated_ids:
                generated_text = self.tok.tk.decode(generated_ids)
                return generated_text.strip()
            
        return ""

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

def extract_number_from_answer(answer_text: str) -> Optional[float]:
    """Extract the final numerical answer from GSM8K format answer"""
    # Look for #### followed by a number at the end
    match = re.search(r'####\s*([+-]?\d*\.?\d+)', answer_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Fallback: look for the last number in the text
    numbers = re.findall(r'([+-]?\d*\.?\d+)', answer_text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def extract_number_from_generated_text(generated_text: str) -> Optional[float]:
    """Extract numerical answer from model's generated text"""
    # First try to find #### format
    match = re.search(r'####\s*([+-]?\d*\.?\d+)', generated_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Look for common answer patterns in Turkish
    patterns = [
        r'cevap[:\s]*([+-]?\d*\.?\d+)',
        r'sonuç[:\s]*([+-]?\d*\.?\d+)', 
        r'toplam[:\s]*([+-]?\d*\.?\d+)',
        r'=\s*([+-]?\d*\.?\d+)',
        r'([+-]?\d*\.?\d+)\s*TL',
        r'([+-]?\d*\.?\d+)\s*lira',
        r'([+-]?\d*\.?\d+)\s*kilogram',
        r'([+-]?\d*\.?\d+)\s*metre',
        r'([+-]?\d*\.?\d+)\s*saat',
        r'([+-]?\d*\.?\d+)\s*dakika'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, generated_text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])  # Take the last match
            except ValueError:
                continue
    
    # Final fallback: last number in text
    numbers = re.findall(r'([+-]?\d*\.?\d+)', generated_text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def run_gsm8k_turkish(scorer: LocalScorer, limit_examples: Optional[int] = None) -> Dict[str, Any]:
    """
    Run GSM8K evaluation on Turkish dataset.
    """
    
    dataset_name = "bezir/gsm8k-tr"
    
    try:
        # Load the dataset - try different splits
        try:
            ds = load_dataset(dataset_name, split="test")
            print(f"Loaded test split with {len(ds)} examples")
        except:
            try:
                ds = load_dataset(dataset_name, split="train")
                print(f"Loaded train split with {len(ds)} examples")
            except:
                ds = load_dataset(dataset_name)
                if isinstance(ds, dict):
                    # Take the first available split
                    split_name = list(ds.keys())[0]
                    ds = ds[split_name]
                    print(f"Loaded {split_name} split with {len(ds)} examples")
                else:
                    print(f"Loaded dataset with {len(ds)} examples")
        
    except Exception as e:
        print(f"Could not load dataset {dataset_name}: {e}")
        return {"error": f"Could not load dataset: {e}"}
    
    if limit_examples and len(ds) > limit_examples:
        ds = ds.select(range(limit_examples))
        print(f"Limited to {limit_examples} examples for testing")
    
    correct = 0
    total = 0
    results = []
    
    for i, example in enumerate(ds):
        try:
            # Get question and answer
            if "question" in example:
                question = example["question"]
            elif "soru" in example:
                question = example["soru"]
            else:
                print(f"No question found in example {i}: {list(example.keys())}")
                continue
                
            if "answer" in example:
                gold_answer_text = example["answer"]
            elif "cevap" in example:
                gold_answer_text = example["cevap"]
            else:
                print(f"No answer found in example {i}: {list(example.keys())}")
                continue
            
            # Extract the gold numerical answer
            gold_number = extract_number_from_answer(gold_answer_text)
            if gold_number is None:
                print(f"Could not extract number from gold answer in example {i}")
                continue
            
            # Create prompt for the model
            prompt = f"Soru: {question}\n\nÇözüm:"
            
            # Generate model's answer
            try:
                generated_text = scorer.generate_answer(prompt)
                predicted_number = extract_number_from_generated_text(generated_text)
                
                if predicted_number is None:
                    print(f"Could not extract number from generated text in example {i}")
                    predicted_number = 0  # Default to 0 for failed extractions
                
                # Check if the answer is correct (allowing small floating point errors)
                is_correct = abs(predicted_number - gold_number) < 1e-6
                
                if is_correct:
                    correct += 1
                
                total += 1
                
                # Store detailed results
                results.append({
                    "example_id": i,
                    "question": question,
                    "gold_answer": gold_number,
                    "predicted_answer": predicted_number,
                    "generated_text": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
                    "correct": is_correct
                })
                
                # Print progress
                if (i + 1) % 10 == 0 or i < 10:
                    current_accuracy = correct / total if total > 0 else 0
                    print(f"Example {i+1}: Gold={gold_number}, Pred={predicted_number}, Correct={is_correct} | Running accuracy: {current_accuracy:.3f} ({correct}/{total})")
                    
            except Exception as e:
                print(f"Error generating answer for example {i}: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"\nFinal Results:")
    print(f"Total examples processed: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return {
        "dataset_used": dataset_name,
        "total_examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "detailed_results": results[:10] if len(results) > 10 else results  # Store first 10 for inspection
    }

def main():
    tok_env = os.environ.get("TOKENIZER_JSON", "")
    tokenizer_json = tok_env if tok_env else os.path.join(".gitignore/tokenizers/bpe/tr", "tokenizer.json")
    tok = Tok(tokenizer_json)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ckpt = _load_checkpoint(".gitignore/out", map_location="cpu")
    model = _init_model_from_checkpoint(ckpt, device=device)
    
    scorer = LocalScorer(model=model, tok=tok, device=device)
    
    print("Starting Turkish GSM8K evaluation...")
    # Start with a small number for testing, remove limit for full evaluation
    results = run_gsm8k_turkish(scorer, limit_examples=50)  
    
    output_file = 'bpe_tr_gsm8k_turkish.json'
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()