import os
import json
from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFScorer:
    def __init__(self, model_name: str = "openai-community/gpt2", device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.block_size = getattr(self.model.config, "n_positions", 1024)
    def _clip_ctx(self, ids: List[int], budget: int) -> torch.Tensor:
        if len(ids) > budget:
            ids = ids[-budget:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    def logprob_of(self, context: str, continuation: str) -> float:
        ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
        cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
        x = self._clip_ctx(ctx_ids, self.block_size - 1)
        total = 0.0
        with torch.no_grad():
            for tid in cont_ids:
                if x.shape[1] > self.block_size - 1:
                    x = x[:, -(self.block_size - 1):]
                logits = self.model(x).logits
                lp = F.log_softmax(logits[:, -1, :], dim=-1)[0, tid].item()
                total += float(lp)
                x = torch.cat([x, torch.tensor([[tid]], device=self.device)], dim=1)
        return total

def run_mmlu_only(scorer: HFScorer, limit_per_subset: Optional[int] = None) -> Dict[str, Any]:
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = HFScorer(model_name="openai-community/gpt2", device=device)
    results = run_mmlu_only(scorer, limit_per_subset=None)
    with open('gpt2_mmlu.json', 'w') as fp:
        json.dump(results, fp, indent=2)

if __name__ == "__main__":
    main()
