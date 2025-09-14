import json
import matplotlib.pyplot as plt

with open("results/sentence-transformers__stsb-xlm-r-multilingual/summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"][0]
if isinstance(results, str):
    import re
    scores = re.findall(r"'main_score': ([0-9.]+).*?'languages': \['(.*?)'\]", results, re.DOTALL)
    lang_scores = {lang: float(score) for score, lang in scores}
else:
    lang_scores = {r["languages"][0]: r["main_score"] for r in results}

langs, scores = zip(*sorted(lang_scores.items(), key=lambda x: x[0]))

plt.figure(figsize=(10, 6))
bars = plt.bar(langs, scores, color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Main Score (Accuracy-like)")
plt.title("mT5-base XNLI Results by Language")

for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f"{score:.3f}",
        ha="center", va="bottom", fontsize=9
    )

plt.tight_layout()
plt.savefig('figs/stsb_xlmr_xnli.jpg')
