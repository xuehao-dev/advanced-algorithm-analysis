# -*- coding: utf-8 -*-
"""HSPSO 论文复现的实验入口（Yao et al. 2024, Sci Rep 14:24928）。

默认实验：CEC-2005（论文 Table 1 的 15 个函数）× PSO/DAIW/FA/HSPSO × 30 次独立运行，
N=20, T=100（与论文实验设置一致），进程级并行加速（结果与串行逐位一致）。

用法示例：
    python run_experiments.py                          # 完整实验（15 函数 × 30 次 × 4 算法）
    python run_experiments.py --smoke                  # 快速自检（F1/F9，3 次，小规模）
    python run_experiments.py --algorithms PSO HSPSO --runs 10
    python run_experiments.py --no-hj                  # 消融：关闭 Hooke-Jeeves
    python run_experiments.py --workers 8 --outdir results

输出（默认 results/ 目录，tag 默认 cec2005）：
    results_{tag}.csv        逐次运行原始结果
    summary_{tag}.csv        各算法×函数 的最优/均值/标准差/中位数
    pvalues_{tag}.csv        HSPSO vs 各基线 的 Mann-Whitney U 秩和检验 p 值
    convergence_{tag}.csv    各算法×函数 的迭代收敛曲线（30 次运行均值）
"""
from __future__ import annotations
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from algorithms import ALGORITHMS, ALGO_OFFSET
from benchmarks import get_benchmarks


def _run_task(task):
    """在子进程中执行一次 (函数, 算法, 次数) 实验。

    每个任务用独立 RNG；初始种群 X0 在同一 (函数, 次数) 下对所有算法共享，
    保证公平比较；结果与串行执行逐位一致。
    """
    idx, func_idx, fname, fun, dim, lb, ub, opt, algo, run, seed, pop, iters, cfg = task
    seed_fr = seed + func_idx * 100000 + run
    base = np.random.default_rng(seed_fr)
    x0 = base.uniform(lb, ub, (pop, dim))                      # 所有算法共享的初始种群
    arng = np.random.default_rng(seed_fr * 7 + ALGO_OFFSET[algo])

    start = time.perf_counter()
    if algo == "HSPSO":
        res = ALGORITHMS[algo](fun, dim, lb, ub, x0, arng, pop=pop, iters=iters,
                               mutation_rate=cfg["mutation_rate"], b=cfg["b"],
                               hj_on=cfg["hj_on"], hj_nmax=cfg["hj_nmax"],
                               hj_delta0=cfg["hj_delta0"], hj_tol=cfg["hj_tol"])
    elif algo == "DAIW":
        res = ALGORITHMS[algo](fun, dim, lb, ub, x0, arng, pop=pop, iters=iters, b=cfg["b"])
    else:
        res = ALGORITHMS[algo](fun, dim, lb, ub, x0, arng, pop=pop, iters=iters)
    elapsed = time.perf_counter() - start

    row = {"algorithm": algo, "function": fname, "dimension": dim, "run": run + 1,
           "best_fitness": res.best_f, "error": res.best_f - opt,
           "nfev": res.nfev, "seconds": elapsed}
    return idx, row, list(res.history)


def run_experiments(benchmarks, algorithms, runs, pop, iters, seed, outdir, tag,
                    smoke=False, workers=None, mutation_rate=0.1, b=1.0,
                    hj_on=True, hj_nmax=2, hj_delta0=None, hj_tol=1e-8):
    if smoke:
        benchmarks = get_benchmarks(["F1", "F9"])
        runs = min(runs, 3)
        pop = min(pop, 10)
        iters = min(iters, 20)

    cfg = {"mutation_rate": mutation_rate, "b": b, "hj_on": hj_on,
           "hj_nmax": hj_nmax, "hj_delta0": hj_delta0, "hj_tol": hj_tol}

    tasks = []
    for func_idx, (fname, fun, dim, lb, ub, opt) in enumerate(benchmarks):
        for algo in algorithms:
            for run in range(runs):
                tasks.append((len(tasks), func_idx, fname, fun, dim, lb, ub, opt,
                              algo, run, seed, pop, iters, cfg))

    total = len(tasks)
    nproc = workers if workers and workers > 0 else (os.cpu_count() or 1)
    if smoke:
        nproc = 1                                       # 冒烟任务小，进程开销占主导

    results = {}
    print(f"total tasks: {total}, workers: {nproc}", flush=True)
    if nproc == 1:
        for t in tasks:
            idx, row, hist = _run_task(t)
            results[idx] = (row, hist)
            print(f"[{len(results)}/{total}] {row['algorithm']:5s} {row['function']:3s} "
                  f"run={row['run']:02d}/{runs} f={row['best_fitness']:.6g} nfev={row['nfev']}",
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futures = [ex.submit(_run_task, t) for t in tasks]
            for fut in as_completed(futures):
                idx, row, hist = fut.result()
                results[idx] = (row, hist)
                print(f"[{len(results)}/{total}] {row['algorithm']:5s} {row['function']:3s} "
                      f"run={row['run']:02d}/{runs} f={row['best_fitness']:.6g} nfev={row['nfev']}",
                      flush=True)

    ordered = [results[i] for i in range(total)]
    rows = [r for r, _ in ordered]
    hists = {r["algorithm"] + "|" + r["function"]: [] for r, _ in ordered}
    for (row, hist) in ordered:
        hists[row["algorithm"] + "|" + row["function"]].append(np.asarray(hist, dtype=float))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 原始结果
    raw_path = outdir / f"results_{tag}.csv"
    with raw_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 2) 汇总：各算法×函数 最优/均值/标准差/中位数
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = df.groupby(["algorithm", "function"]).agg(
        best_best=("best_fitness", "min"),
        mean=("best_fitness", "mean"),
        std=("best_fitness", "std"),
        median=("best_fitness", "median"),
        mean_nfev=("nfev", "mean"),
        mean_seconds=("seconds", "mean")).reset_index()
    summ_path = outdir / f"summary_{tag}.csv"
    summ.to_csv(summ_path, index=False, encoding="utf-8-sig")

    # 3) 秩和检验：HSPSO vs 各基线（Mann-Whitney U，双侧）
    from scipy.stats import mannwhitneyu
    p_rows = []
    for fname in df["function"].unique():
        hs = df[(df["function"] == fname) & (df["algorithm"] == "HSPSO")]["best_fitness"].values
        for algo in algorithms:
            if algo == "HSPSO":
                continue
            bs = df[(df["function"] == fname) & (df["algorithm"] == algo)]["best_fitness"].values
            try:
                p = float(mannwhitneyu(hs, bs, alternative="two-sided").pvalue)
            except ValueError:
                p = float("nan")
            p_rows.append({"function": fname, "baseline": algo, "p_value": p,
                           "hspso_mean": float(hs.mean()), "baseline_mean": float(bs.mean()),
                           "significant": bool(p < 0.05)})
    p_path = outdir / f"pvalues_{tag}.csv"
    pd.DataFrame(p_rows).to_csv(p_path, index=False, encoding="utf-8-sig")

    # 4) 收敛曲线：各算法×函数，每代全局最优（30 次运行均值）
    conv_rows = []
    for key, arrs in hists.items():
        algo, fname = key.split("|")
        mean_curve = np.mean(np.vstack(arrs), axis=0)     # shape (iters+1,)
        for it in range(1, len(mean_curve)):
            conv_rows.append({"algorithm": algo, "function": fname,
                              "iteration": it, "mean_best": float(mean_curve[it])})
    conv_path = outdir / f"convergence_{tag}.csv"
    pd.DataFrame(conv_rows).to_csv(conv_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved:\n  {raw_path}\n  {summ_path}\n  {p_path}\n  {conv_path}")


def main():
    ap = argparse.ArgumentParser(description="HSPSO (Yao et al. 2024) reproduction experiments")
    ap.add_argument("--functions", nargs="+", default=None,
                    help="Benchmark names to run (default: all F1..F15)")
    ap.add_argument("--algorithms", nargs="+", default=["PSO", "DAIW", "FA", "HSPSO"],
                    choices=["PSO", "DAIW", "FA", "HSPSO"])
    ap.add_argument("--runs", type=int, default=30, help="independent runs (paper: 30)")
    ap.add_argument("--pop", type=int, default=20, help="population size (paper: 20)")
    ap.add_argument("--iters", type=int, default=100, help="max iterations (paper: 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tag", default="cec2005")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes (default: CPU count; 1 = serial)")
    ap.add_argument("--smoke", action="store_true", help="quick self-check on F1/F9")
    ap.add_argument("--no-hj", action="store_true", help="disable Hooke-Jeeves (ablation)")
    ap.add_argument("--mutation-rate", type=float, default=0.1,
                    help="Cauchy mutation rate (paper omits the value; default 0.1)")
    ap.add_argument("--b", type=float, default=1.0, help="damping factor in Eq.(6), b in [0,1]")
    ap.add_argument("--hj-nmax", type=int, default=2, help="Hooke-Jeeves max sweeps per call")
    ap.add_argument("--hj-delta0", type=float, default=None,
                    help="Hooke-Jeeves initial step (default: 0.05*(ub-lb))")
    args = ap.parse_args()

    bmarks = get_benchmarks(args.functions)
    run_experiments(bmarks, args.algorithms, args.runs, args.pop, args.iters,
                    args.seed, args.outdir, args.tag, smoke=args.smoke,
                    workers=args.workers, mutation_rate=args.mutation_rate,
                    b=args.b, hj_on=not args.no_hj, hj_nmax=args.hj_nmax,
                    hj_delta0=args.hj_delta0)


if __name__ == "__main__":
    main()
