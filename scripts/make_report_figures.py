"""
Generate publication-ready figures from experiment outputs.
Produces: learning_curves.png, model_comparison.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

COLORS = {"LogisticRegression": "#2196F3", "RandomForest": "#4CAF50", "SVM_RBF": "#FF9800"}
LABELS = {"LogisticRegression": "Logistic Regression", "RandomForest": "Random Forest", "SVM_RBF": "SVM (RBF)"}


def plot_learning_curves():
    """Learning curves: validation accuracy vs training set size."""
    with open(OUTPUTS_DIR / "learning_curves.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, d in data.items():
        sizes = d["train_sizes"]
        val_mean = np.array(d["val_mean"])
        val_std = np.array(d["val_std"])

        ax.plot(sizes, val_mean, "o-", color=COLORS[name], label=LABELS[name], linewidth=2)
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color=COLORS[name])

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("Learning Curves (Stratified 5-Fold CV)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0.7, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / "learning_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_model_comparison():
    """Bar chart: grand mean accuracy +/- std across 5 seeds."""
    with open(OUTPUTS_DIR / "results.json") as f:
        data = json.load(f)

    summary = data["summary"]
    names = list(summary.keys())
    means = [summary[n]["grand_mean"] for n in names]
    stds = [summary[n]["grand_std"] for n in names]
    colors = [COLORS[n] for n in names]
    labels = [LABELS[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.5, alpha=0.85)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (5-Seed Grand Mean)", fontsize=12)
    ax.set_title("Model Comparison: Iris Classification", fontsize=14)
    ax.set_ylim(0.85, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_learning_curves()
    plot_model_comparison()
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
