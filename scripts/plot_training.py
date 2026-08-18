#!/usr/bin/env python3
"""Reads checkpoints/train_log.txt + checkpoints/eval_log.txt and renders PNG plots

Usage: python3 scripts/plot_training.py [--checkpoints-dir checkpoints] [--out-dir report_figures]
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(path, ncols):
    rows = [[] for _ in range(ncols)]
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        for line in csv.reader(f, delimiter="\t"):
            if len(line) < ncols:
                continue
            for i in range(ncols):
                rows[i].append(float(line[i]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", default="checkpoints")
    ap.add_argument("--out-dir", default="report_figures")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # train_log.txt: step, loss, ppl, lr
    tr_step, tr_loss, tr_ppl, tr_lr = read_log(ckpt_dir / "train_log.txt", 4)
    # eval_log.txt: step, loss, ppl
    ev_step, ev_loss, ev_ppl = read_log(ckpt_dir / "eval_log.txt", 3)

    if not tr_step and not ev_step:
        print(f"No logs found under {ckpt_dir}/ (train_log.txt / eval_log.txt). "
              "Run training first.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Loss curve: train vs eval
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if tr_step:
        ax.plot(tr_step, tr_loss, label="Train loss", color="#4C72B0", linewidth=1, alpha=0.85)
    if ev_step:
        ax.plot(ev_step, ev_loss, label="Eval loss", color="#DD8452", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training / Evaluation Loss over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # 2. Perplexity curve
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if tr_step:
        ax.plot(tr_step, tr_ppl, label="Train perplexity", color="#4C72B0", linewidth=1, alpha=0.85)
    if ev_step:
        ax.plot(ev_step, ev_ppl, label="Eval perplexity", color="#DD8452", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Perplexity")
    ax.set_yscale("log")
    ax.set_title("Perplexity over Time (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "perplexity_curve.png", dpi=150)
    plt.close(fig)

    # 3. Learning-rate schedule
    if tr_step:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(tr_step, tr_lr, color="#55A868", linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning Rate Schedule")
        fig.tight_layout()
        fig.savefig(out_dir / "lr_schedule.png", dpi=150)
        plt.close(fig)

    print(f"Wrote figures to {out_dir}/")
    print(f"  train points: {len(tr_step)}, eval points: {len(ev_step)}")
    if ev_loss:
        print(f"  eval loss: {ev_loss[0]:.4f} -> {ev_loss[-1]:.4f} "
              f"(ppl {ev_ppl[0]:.2f} -> {ev_ppl[-1]:.2f})")


if __name__ == "__main__":
    main()
