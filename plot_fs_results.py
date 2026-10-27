# -*- coding: utf-8 -*-
"""HSPSO-FS（Arrhythmia 特征选择）θ=0.5 结果可视化 —— 课程 PPT 图表素材。

数据源：results/results_fs_arrhythmia_theta05.csv（SVM 分类器、5 折分层 CV、
        pop=20、100 迭代、20 次独立运行、θ=0.5）
输出：plots/fs_results_theta05.png（1x2 双面板）
  Panel 1：平均准确率（±std 误差条）vs 全特征 SVM 基线，附 best 标注
  Panel 2：平均子集长度 vs best 子集长度

用法：
    python plot_fs_results.py
    python plot_fs_results.py --csv results/results_fs_arrhythmia_theta05.csv --outdir plots --dpi 150
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser(description="HSPSO-FS theta=0.5 result plot")
    ap.add_argument("--csv", default="results/results_fs_arrhythmia_theta05.csv")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # θ=0.5 两算法各自 20 次运行统计
    stats = {}
    for algo in ["PSO", "HSPSO"]:
        rs = df[df["algorithm"] == algo]
        stats[algo] = dict(
            avg_acc=rs["best_accuracy"].mean(),
            std_acc=rs["best_accuracy"].std(),
            best_acc=rs["best_accuracy"].max(),
            avg_subset=rs["subset_size"].mean(),
            best_subset=rs["subset_size"].min(),
        )
    # 全特征 SVM 基线：用与实验相同的划分再算一次（脚本内可复算，避免硬编码）
    # 简化：基线由实验日志给出（SVM 5 折 CV 全 279 特征 = 0.7943），此处读取自
    # 汇总口径：为保持脚本自洽，直接按论文实验日志记录填入并标注来源。
    baseline = 0.7943

    algos = ["PSO", "HSPSO"]
    labels = ["PSO-FS", "HSPSO-FS"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    fig.suptitle("HSPSO-FS 特征选择复现（Arrhythmia, θ=0.5, SVM 5 折 CV, 20 次运行）",
                 fontsize=14, y=0.98)

    # ---- Panel 1：平均准确率（误差条 + 基线 + best） ----
    ax = axes[0]
    accs = [stats[a]["avg_acc"] * 100 for a in algos]
    errs = [stats[a]["std_acc"] * 100 for a in algos]
    bests = [stats[a]["best_acc"] * 100 for a in algos]
    x = np.arange(len(algos))
    bars = ax.bar(x, accs, yerr=errs, width=0.45, capsize=5,
                  color=["#4C72B0", "#DD8452"])
    ax.axhline(baseline * 100, ls="--", lw=1.2, color="#555555")
    ax.text(0.99, baseline * 100 + 0.6, "全特征基线 %.2f%%" % (baseline * 100),
            fontsize=9, ha="right", color="#333333",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    for i, (a, b) in enumerate(zip(accs, bests)):
        ax.text(i, a + errs[i] + 0.35, "avg %.2f%%" % a,
                ha="center", fontsize=10, fontweight="bold")
        ax.text(i, a - errs[i] - 1.15, "best %.2f%%" % b,
                ha="center", fontsize=8.5, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("分类准确率 (%)")
    ax.set_ylim(74, 90)
    ax.set_title("(a) 平均准确率（±1 std）", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # ---- Panel 2：子集长度（avg 柱 + best 标注） ----
    ax = axes[1]
    avg_sub = [stats[a]["avg_subset"] for a in algos]
    best_sub = [stats[a]["best_subset"] for a in algos]
    bars = ax.bar(x, avg_sub, width=0.45,
                  color=["#4C72B0", "#DD8452"])
    for i, (a, b) in enumerate(zip(avg_sub, best_sub)):
        ax.text(i, a + 2.0, "avg %.1f" % a, ha="center", fontsize=10,
                fontweight="bold")
        ax.text(i, b - 6.0, "best %d" % b, ha="center", fontsize=8.5,
                color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("特征子集长度（个）")
    ax.set_ylim(0, 110)
    ax.set_title("(b) 特征子集长度（avg / best）", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "fs_results_theta05.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print("saved:", out)
    print("PSO    avg_acc=%.4f std=%.4f best=%.4f avg_subset=%.1f best_subset=%d"
          % (stats["PSO"]["avg_acc"], stats["PSO"]["std_acc"],
             stats["PSO"]["best_acc"], stats["PSO"]["avg_subset"],
             stats["PSO"]["best_subset"]))
    print("HSPSO  avg_acc=%.4f std=%.4f best=%.4f avg_subset=%.1f best_subset=%d"
          % (stats["HSPSO"]["avg_acc"], stats["HSPSO"]["std_acc"],
             stats["HSPSO"]["best_acc"], stats["HSPSO"]["avg_subset"],
             stats["HSPSO"]["best_subset"]))


if __name__ == "__main__":
    main()
