import os
import pickle
import time
import math
import numpy as np
import torch
from contextlib import nullcontext
import fire
from model import GPTConfig, GPT
from utils import get_batch


def _load_checkpoint(out_dir: str, map_location: str):
    ckpt_path = os.path.join(out_dir, "bpe_en_train.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if "model_args" not in ckpt or "model" not in ckpt:
        raise KeyError(f"Invalid checkpoint format in {ckpt_path}")
    return ckpt


def _init_model_from_checkpoint(ckpt, device: str, block_size: int | None = None):
    model_args = dict(ckpt["model_args"])
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    if block_size is not None and block_size < model.config.block_size:
        model.crop_block_size(block_size)

    model.to(device)
    model.eval()
    return model


def evaluate(
    out_dir: str = "out",
    dataset: str = "tokenizer_bins/",
    tokenized_validation_input_file: str = "bpe_en_val.bin",
    tokenized_training_input_file: str = "bpe_en_train.bin",
    eval_iters: int = 200,
    batch_size: int = 12,
    block_size: int = 1024,
    dtype: str = "float16",
    device: str = "cuda",
    seed: int = 1337,
    report_every: int = 50,
):
    torch.manual_seed(seed)
    device_type = "cuda" if "cuda" in device and torch.cuda.is_available() else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    amp_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    data_dir = dataset
    val_path = os.path.join(data_dir, tokenized_validation_input_file)
    train_path = os.path.join(data_dir, tokenized_training_input_file)
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation memmap not found: {val_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train memmap not found (needed by get_batch): {train_path}")

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        print(f"[info] meta.pkl found: vocab_size={meta.get('vocab_size', 'unk')}")

    ckpt = _load_checkpoint(out_dir, map_location="cpu" if device_type == "cpu" else device)
    model = _init_model_from_checkpoint(ckpt, device=device, block_size=block_size)

    eval_block = min(block_size, model.config.block_size)
    if eval_block != block_size:
        print(f"[warn] Requested block_size {block_size} > model.block_size {model.config.block_size}; using {eval_block}")

    total_loss = 0.0
    t0 = time.time()
    with torch.no_grad(), amp_ctx:
        for it in range(1, eval_iters + 1):
            X, Y = get_batch(
                split="val",
                train_data=train_data,
                val_data=val_data,
                block_size=eval_block,
                batch_size=batch_size,
                device_type=device_type,
                device=device,
            )
            _, loss = model(X, Y)
            total_loss += loss.item()

            if report_every and (it % report_every == 0 or it == eval_iters):
                running_avg = total_loss / it
                print(f"[{it:4d}/{eval_iters}] avg_loss={running_avg:.4f}  ppl={math.exp(running_avg):.2f}")

    avg_loss = total_loss / eval_iters
    ppl = math.exp(avg_loss)

    dt = time.time() - t0
    tokens_evaluated = eval_iters * batch_size * eval_block
    toks_per_sec = tokens_evaluated / max(dt, 1e-9)

    print("\n=== Evaluation Summary ===")
    print(f"checkpoint:      {os.path.join(out_dir, 'bpe_en_train.pt')}")
    print(f"device/dtype:    {device}/{dtype}")
    print(f"val batches:     {eval_iters}  (batch_size={batch_size}, block_size={eval_block})")
    print(f"avg loss:        {avg_loss:.6f}")
    print(f"perplexity:      {ppl:.4f}")
    print(f"tokens eval'd:   {tokens_evaluated:,}")
    print(f"wall time:       {dt:.2f}s  ({toks_per_sec:,.0f} tok/s)")

    return {"avg_loss": avg_loss, "perplexity": ppl, "tokens": tokens_evaluated, "time_sec": dt}


if __name__ == "__main__":
    fire.Fire(evaluate)
