#!/usr/bin/env python3
import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Map ISO-ish language tags → language family
LANG2FAMILY = {
    "ara": "Afro-Asiatic",
    "bul": "Indo-European",
    "deu": "Indo-European",
    "ell": "Indo-European",
    "eng": "Indo-European",
    "spa": "Indo-European",
    "fra": "Indo-European",
    "hin": "Indo-European",
    "rus": "Indo-European",
    "swa": "Niger-Congo",
    "tha": "Kra-Dai",
    "tur": "Turkic",
    "vie": "Austroasiatic",
    "zho": "Sino-Tibetan",
    # add more as needed
}

def load_results(results_dir="results"):
    """Load all summary.json files from subdirectories."""
    data = {}
    for path in glob.glob(os.path.join(results_dir, "*", "summary.json")):
        with open(path, "r") as f:
            res = json.load(f)
        model = res["model"]
        results = res["results"]

        # results may be list[str] if saved with default=str
        # so we need to parse per-language scores
        family_scores = defaultdict(list)
        if isinstance(results, list):
            # stringified results; parse line by line
            for entry in results:
                if isinstance(entry, str) and "hf_subset" in entry:
                    # crude parse for hf_subset + main_score
                    try:
                        lang = entry.split("hf_subset='")[1].split("'")[0]
                        main_score = float(entry.split("main_score': ")[1].split(",")[0])
                        short = lang.split("-")[0]
                        fam = LANG2FAMILY.get(short, "Other")
                        family_scores[fam].append(main_score)
                    except Exception:
                        continue
        elif isinstance(results, dict):
            # normal dict structure
            for task, taskres in results.items():
                if isinstance(taskres, dict) and "test" in taskres:
                    for entry in taskres["test"]:
                        lang = entry.get("hf_subset", "").split("-")[0]
                        score = entry.get("main_score")
                        if lang and score is not None:
                            fam = LANG2FAMILY.get(lang, "Other")
                            family_scores[fam].append(score)

        # average per family
        avg_scores = {fam: np.mean(vals) for fam, vals in family_scores.items()}
        data[model] = avg_scores
    return data

def plot_family_trends(data):
    families = sorted({fam for model in data.values() for fam in model})
    plt.figure(figsize=(10, 6))

    for model, scores in data.items():
        xs, ys = [], []
        for fam in families:
            xs.append(fam)
            ys.append(scores.get(fam, np.nan))
        plt.plot(xs, ys, marker="o", label=model)

        # annotate with values
        for x, y in zip(xs, ys):
            if not np.isnan(y):
                plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Main Score")
    plt.title("Embedding Model Performance by Language Family")
    plt.legend()
    plt.tight_layout()
    plt.savefig('foo.jpg')

if __name__ == "__main__":
    results = load_results("results")
    plot_family_trends(results)
