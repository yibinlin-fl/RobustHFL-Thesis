"""Regenerate plots exclusively from saved V2 CSV tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from scripts.openi.precheck_science import read_rows


def draw(root, out, stage=None):
    out.mkdir(parents=True, exist_ok=True)
    work1, hetero, freq = (root / folder for folder in ("work1", "work2_heterogeneity", "work2_frequency"))
    if stage in (None, "work1") and (work1 / "training_curves.csv").is_file():
        rows = read_rows(work1 / "training_curves.csv")
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
        for strategy in dict.fromkeys(r["strategy"] for r in rows):
            subset = [r for r in rows if r["strategy"] == strategy]
            for ax, label in zip(axs, ("local_ce_loss", "clean_acc", "corrupt_acc")):
                ax.plot([int(r["round"]) for r in subset], [float(r[label]) for r in subset], marker="o", label=strategy)
                ax.set(xlabel="Round", ylabel=label)
        axs[-1].legend(fontsize=7)
        fig.savefig(out / "work1_training_curves.png", dpi=160)
        plt.close(fig)
        rows = read_rows(work1 / "paired_interventions.csv")
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        signals = list(dict.fromkeys(r["signal"] for r in rows))
        distributions = [[100*float(r["gain_robust"]) for r in rows if r["signal"] == signal and
                          r["phase"] == "analysis" and r["arm"] == "downweight"] for signal in signals]
        if all(distributions):
            ax.boxplot(distributions)
            ax.set_xticks(range(1, len(signals)+1), signals, rotation=25)
        ax.axhline(0, color="#777777", linewidth=1)
        ax.set(ylabel="Downweight − normal (percentage points)", xlabel="Private training signal")
        fig.savefig(out / "work1_intervention_plot.png", dpi=160)
        plt.close(fig)
    if stage in (None, "work2_heterogeneity") and (hetero / "kd_collaboration_gains.csv").is_file():
        rows = read_rows(hetero / "kd_collaboration_gains.csv")
        labels = list(dict.fromkeys(r["scenario"] for r in rows))
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for i, label in enumerate(labels):
            subset = [100*float(r["kd_gain"]) for r in rows if r["scenario"] == label]
            ax.scatter([i]*len(subset), subset, color="#6290C8", alpha=.65)
            ax.scatter(i, np.mean(subset), marker="D", color="#24375B")
        ax.axhline(0, color="#777777", linewidth=1)
        ax.set_xticks(range(len(labels)), labels, rotation=18)
        ax.set(ylabel="Std KD − Local CE (percentage points)", xlabel="Training corruption allocation")
        fig.savefig(out / "heterogeneity_comparison.png", dpi=160)
        plt.close(fig)
    if stage in (None, "work2_frequency") and (freq / "delta_r_matrix.csv").is_file():
        rows = read_rows(freq / "delta_r_matrix.csv")
        domains = list(dict.fromkeys(r["slice"] for r in rows))
        fig, axes = plt.subplots(len(domains), 3, figsize=(10, 3.2*len(domains)), squeeze=False,
                                 layout="constrained")
        clients = sorted({int(r["student"]) for r in rows})
        bound = max([.001, *[abs(100*float(r["delta_r"])) for r in rows]])
        for i, domain in enumerate(domains):
            for j, band in enumerate(("full", "low", "high")):
                ax = axes[i, j]
                values = np.full((len(clients), len(clients)), np.nan)
                for s in clients:
                    for k in clients:
                        pairs = [100*float(r["delta_r"]) for r in rows if
                                 (r["slice"], r["band"], int(r["student"]), int(r["teacher"])) == (domain, band, s, k)]
                        if pairs:
                            values[s, k] = np.mean(pairs)
                image = ax.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound)
                ax.set(title=f"{domain} / {band}", xlabel="Teacher", ylabel="Student",
                       xticks=clients, yticks=clients)
        fig.colorbar(image, ax=axes, label="ΔR (percentage points)")
        fig.savefig(out / "delta_r_heatmap.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
        for ax, key in zip(axes, ("mi", "confidence", "discrepancy")):
            for domain in domains:
                points = [r for r in rows if r["slice"] == domain]
                ax.scatter([float(r[key]) for r in points], [100*float(r["delta_r"]) for r in points],
                           label=domain, alpha=.45, s=14)
            ax.axhline(0, color="#777777", linewidth=1)
            ax.set(xlabel=key, ylabel="ΔR (percentage points)")
        axes[-1].legend(fontsize=7)
        fig.savefig(out / "score_vs_gain.png", dpi=160)
        plt.close(fig)
        pixels = freq / "probe" / "frequency_view_pixels.npz"
        if pixels.is_file():
            import torch
            from probes.probe2_freq_delta_r import grid
            with np.load(pixels, allow_pickle=False) as archive:
                grid(out / "frequency_views.png",
                     {band: torch.from_numpy(archive[band]) for band in ("full", "low", "high")}, domains[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    draw(args.run_dir, args.out)
