from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="uonlp/CulturaX",
    repo_type="dataset",
    local_dir="culturax",
    local_dir_use_symlinks=False,
    allow_patterns=["en/en_part_00000.parquet"],  # specify your target file
    token=os.environ['HF_TOKEN']
)
