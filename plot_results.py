# -*- coding: utf-8 -*-
"""HSPSO 实验结果可视化 —— 为课程 PPT 生成图表素材（Yao et al. 2024 复现项目）。

用法：
    python plot_results.py                      # 读 results/，输出 plots/
    python plot_results.py --results-dir results --outdir plots --dpi 150

输出（plots/ 目录）：
    01_error_comparison.png      分组柱状图：15 函数 × 4 算法 的均值误差（log 轴，越小越好）
    02_convergence_F01-F05.png   收敛曲线（误差 log 轴）：F1-F5
    03_convergence_F06-F10.png   收敛曲线：F6-F10
    04_convergence_F11-F15.png   收敛曲线：F11-F15

数据口径：误差 = 均值(或每代全局最优均值) − 全局最优；F13 用修正后的真值 −133.20633
（论文 Table 1 标 0 有误，见 README）；误差 <= 0 的按 1e-12 下限显示（log 轴）。
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Windows 中文字体；axes.unicode_minus 避免负号显示为方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 论文 Table 1 全局最优；F13 修正为真值（单维 h_min=-4.440211 x 30 维）
OPT = {"F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0, "F5": 0.0, "F6": 0.0,
       "F7": 0.0, "F8": -418.9828872724339 * 30.0, "F9": 0.0, "F10": 0.0,
       "F11": 0.0, "F12": 0.0, "F13": -133.20633326681094,
       "F14": -1.0316284535, "F15": 0.39788735772973816}
ALGOS = ["PSO", "DAIW", "FA", "HSPSO"]
FLOOR = 1e-12


def _floor(arr):
    """log 轴下限：误差 <= 0 的置为 FLOOR。"""
    return np.maximum(np.asarray(arr, dtype=float), FLOOR)


def plot_error_comparison(summary_path, outdir, dpi):
    df = pd.read_csv(summary_path)
    fns = sorted(df["function"].unique(), key=lambda s: int(s[1:]))
    x = np.arange(len(fns))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, algo in enumerate(ALGOS):
        vals = [float(df[(df["function"] == fn) & (df["algorithm"] == algo)]["mean"].iloc[0])
                - OPT[fn] for fn in fns]
        ax.bar(x + (i - 1.5) * width, _floor(vals), width, label=algo)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(fns, fontsize=11)
    ax.set_xlabel("CEC-2005 benchmark function")
    ax.set_ylabel("mean error (log)")
    ax.set_title("CEC-2005 mean error comparison (HSPSO vs baselines, 30 runs)")
    ax.legend()
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out = outdir / "01_error_comparison.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_convergence(conv_path, outdir, dpi, fns, name):
    df = pd.read_csv(conv_path)
    fig, axes = plt.subplots(1, len(fns), figsize=(4.2 * len(fns), 3.6))
    for ax, fn in zip(axes, fns):
        for algo in ALGOS:
            sub = df[(df["function"] == fn) & (df["algorithm"] == algo)]
            err = _floor(sub["mean_best"].values - OPT[fn])
            ax.plot(sub["iteration"].values, err, lw=1.8, label=algo)
        ax.set_yscale("log")
        ax.set_title(fn, fontsize=12)
        ax.set_xlabel("Iterations")
        if ax is axes[0]:
            ax.set_ylabel("mean error (log)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Convergence curves (mean best error over 30 runs)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = outdir / name
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="HSPSO reproduction: plot experiment results")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    res = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = res / "summary_cec2005.csv"
    conv = res / "convergence_cec2005.csv"
    if not summary.exists() or not conv.exists():
        raise FileNotFoundError("Missing results_cec2005 CSVs in %s (run run_experiments.py first)" % res)

    outs = [plot_error_comparison(summary, outdir, args.dpi)]
    fns = sorted(pd.read_csv(summary)["function"].unique(), key=lambda s: int(s[1:]))
    groups = [("02_convergence_F01-F05.png", fns[0:5]),
              ("03_convergence_F06-F10.png", fns[5:10]),
              ("04_convergence_F11-F15.png", fns[10:15])]
    for name, g in groups:
        outs.append(plot_convergence(conv, outdir, args.dpi, g, name))
    for o in outs:
        print("saved:", o)


if __name__ == "__main__":
    main()
