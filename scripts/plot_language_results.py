#!/usr/bin/env python3
import os, json, glob
import matplotlib.pyplot as plt
import numpy as np
import ast
from collections import defaultdict

# Map language codes to families
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
}

def load_results(results_dir="results"):
    data = {}
    for path in glob.glob(os.path.join(results_dir, "*", "summary.json")):
        with open(path, "r") as f:
            res = json.load(f)
        model = res["model"]
        results = ast.literal_eval(res["results"][0].split(" scores=")[1].split(' evaluation_time')[0])

        family_scores = defaultdict(list)

        # Handle dict results
        for lang in results['test']:
            fam = lang['languages'][0].split('-')[0]
            score = lang['main_score']
            family_scores[fam].append(float(score))

        avg_scores = {fam: np.mean(vals) for fam, vals in family_scores.items()}
        data[model] = avg_scores

    return data

def plot_family_trends(data):
    families = sorted({fam for model in data.values() for fam in model})
    plt.figure(figsize=(16, 6))  # make the figure wider

    for model, scores in data.items():
        xs, ys = [], []
        for fam in families:
            xs.append(fam)
            ys.append(scores.get(fam, np.nan))
        plt.plot(xs, ys, marker="o", label=model)

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Main Score")
    plt.title("Embedding Model Performance by Language Family")

    # Move legend outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space for legend
    plt.savefig('foo.jpg', dpi=300)

if __name__ == "__main__":
    results = load_results("results")
    plot_family_trends(results)
