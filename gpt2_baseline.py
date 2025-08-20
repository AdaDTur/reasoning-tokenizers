import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

tokenizer.pad_token = tokenizer.eos_token

def generate_answer(prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

def eval_mmlu():
    ds = load_dataset("hendrycks_test", "all", split="test")
    correct, total = 0, 0
    for ex in ds:
        q = ex["question"]
        choices = ex["choices"]
        gold = ex["answer"]

        prompt = q + "\nOptions:\n" + "\n".join([f"{i}. {c}" for i, c in enumerate(choices)]) + "\nAnswer:"
        out = generate_answer(prompt, max_new_tokens=16)

        match = re.search(r"\d", out)
        if match and int(match.group()) == gold:
            correct += 1
        total += 1

        if total % 100 == 0:
            print(f"MMLU progress {total} acc={correct/total:.3f}")

    print(f"MMLU Final Accuracy: {correct/total:.3f}")

def eval_gsm8k():
    ds = load_dataset("gsm8k", "main", split="test")
    correct, total = 0, 0
    for ex in ds:
        q = ex["question"]
        gold = ex["answer"].strip()

        prompt = q + "\nAnswer:"
        out = generate_answer(prompt, max_new_tokens=128)

        pred_nums = re.findall(r"-?\d+(?:\.\d+)?", out)
        gold_nums = re.findall(r"-?\d+(?:\.\d+)?", gold)
        if pred_nums and gold_nums and pred_nums[-1] == gold_nums[-1]:
            correct += 1
        total += 1

        if total % 100 == 0:
            print(f"GSM8K progress {total} acc={correct/total:.3f}")

    print(f"GSM8K Final Accuracy: {correct/total:.3f}")

if __name__ == "__main__":
    eval_mmlu()
    eval_gsm8k()
