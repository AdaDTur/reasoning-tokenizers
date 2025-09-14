from huggingface_hub import snapshot_download
import os
from datasets import load_dataset

langs = ['en', 'tr', 'es', 'fr', 'fi']
"""
for lang in langs:
    snapshot_download(
        repo_id="uonlp/CulturaX",
        repo_type="dataset",
        local_dir="~/culturax/",
        local_dir_use_symlinks=False,
        allow_patterns=[f"{lang}/{lang}_part_00000.parquet"],  # specify your target file
        token=os.environ['HF_TOKEN']
    )
"""
# Load MBPP (Mostly Basic Python Problems)
#mbpp = load_dataset("mbpp")
#mbpp["train"].to_parquet("mbpp_train.parquet")
#mbpp["validation"].to_parquet("mbpp_val.parquet")

# Load GSM8K dataset
gsm8k = load_dataset("gsm8k", "main")
gsm8k["train"].to_parquet("gsm8k_train.parquet")