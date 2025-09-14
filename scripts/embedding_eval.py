#!/usr/bin/env python3
import argparse, os, math, json, time
from typing import List, Optional
import torch
import numpy as np

import mteb
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    MT5EncoderModel,
    T5EncoderModel,
)

# Common nicknames -> HF repo ids (override with --models ids yourself if you want)
ALIAS = {
    "embeddinggemma": "google/embedding-gemma-base",  # HF alias may vary
    "stsb-xlm-r": "sentence-transformers/stsb-xlm-r-multilingual",
    "cohere-v3": "Cohere/embed-multilingual-v3.0",   # if wrapped via HF endpoint
    "jina-v3": "jinaai/jina-embeddings-v3",          # Jina embeddings
    "kalm": "kaist-nlp-lab/kalm-embedding-base",     # KaLM embedding
}

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)  # avoid divide by zero
    pooled = summed / counts
    return pooled.to(torch.float32)

class HFEncoderForMTEB:
    """
    Minimal wrapper so generative or encoder models work as embedding encoders.
    Implements .encode(texts, batch_size, **kwargs) as expected by MTEB.
    """
    def __init__(self, model_id: str, device: Optional[str] = None, fp16: bool = True, max_length: int = 512):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Special handling for mT5/T5 encoder-only usage
        if "mt5" in model_id.lower():
            self.model = MT5EncoderModel.from_pretrained(model_id, torch_dtype=self.dtype)
            self.mode = "encoder"
        elif "t5" in model_id.lower():
            self.model = T5EncoderModel.from_pretrained(model_id, torch_dtype=self.dtype)
            self.mode = "encoder"
        else:
            try:
                self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype)
                self.mode = "encoder"
            except Exception:
                # Use float32 for causal LMs to avoid NaNs during pooling
                self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
                self.mode = "causal"

        self.model.to(self.device)
        self.model.eval()

        # Some tokenizers have no pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                try:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                except Exception:
                    pass

        self.max_length = max_length

    @property
    def model_card_data(self):
        return {"model_id": self.model_id}

    def _forward_batch(self, batch_texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            if self.mode == "encoder":
                out = self.model(**enc)
                hidden = out.last_hidden_state  # [B, T, H]
            else:
                # causal: request hidden states from base model call
                out = self.model.base_model(**enc, output_hidden_states=False, return_dict=True)
                hidden = out.last_hidden_state
            pooled = mean_pool(hidden, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1, eps=1e-12)
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return pooled.detach().cpu().numpy()

    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            vecs = self._forward_batch(batch)
            all_vecs.append(vecs)
        return np.vstack(all_vecs)

def parse_models_arg(models_arg: str) -> List[str]:
    ids = []
    for tok in models_arg.split(","):
        tok = tok.strip()
        ids.append(ALIAS.get(tok.lower(), tok))
    return ids

def main():
    models = "kalm"
    tasks = "XNLI"
    batch_size = 8
    max_length = 100
    output_dir = "results"

    model_ids = parse_models_arg(models)
    task_names = [t.strip() for t in tasks.split(",") if t.strip()]

    # Prepare tasks
    tasks = mteb.get_tasks(tasks=task_names)
    if len(tasks) == 0:
        raise ValueError(f"No MTEB tasks matched: {task_names}. Check names at https://github.com/embeddings-benchmark/mteb")

    for mid in model_ids:
        print(f"\n=== Evaluating: {mid} ===")
        model = HFEncoderForMTEB(mid, max_length=max_length)
        evaluator = mteb.MTEB(tasks=tasks)
        safe_name = mid.replace("/", "__")
        out_dir = os.path.join(output_dir, safe_name)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        results = evaluator.run(
            model,
            encode_kwargs={"batch_size": batch_size},
            output_folder=out_dir,
        )
        dt = time.time() - t0
        # Save a small summary
        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {"model": mid, "elapsed_sec": dt, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
                default=str
            )
        print(f"Saved: {summary_path}  (elapsed: {dt:.1f}s)")

if __name__ == "__main__":
    main()
