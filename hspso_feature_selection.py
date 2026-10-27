# -*- coding: utf-8 -*-
"""HSPSO-FS：Arrhythmia 特征选择应用复现 —— Yao et al. 2024 (Sci Rep 14:24928)。

论文应用章节要点：UCI Arrhythmia 数据集 279 特征、452 样本、16 类合并为二分类
（正常/异常）；适应度（式 9）Fitness = θ(1−Accuracy) + (1−θ)·FS/N；实验设置
pop=20、100 代、20 次独立运行、维度=特征数；对比 PSO-FS 与 HSPSO-FS。

本脚本为独立文件，不修改项目其他脚本，输出独立 CSV（results_fs_arrhythmia.csv），
与现有 CEC-2005 实验数据完全隔离。

论文未写明、由本脚本假设的细节（答辩需说明）：
  1) 分类器：SVM（sklearn 默认 SVC：RBF 核、C=1.0、gamma='scale'）——论文结论
     章节明确 "HSPSO is combined with Support Vector Machine (SVM) for feature
     selection"，但未给 kernel/C/划分等参数，故采用 sklearn 默认；
  2) θ=0.5 / 0.9（--theta 可调）——论文只给 θ∈[0,1]，未给具体值；
  3) 评估方式：5 折分层交叉验证（--folds 可调）——论文未写 train/test 划分；
  4) 缺失值 '?' 用列中位数填充，类别属性按数值处理，折内标准化；二分类采用
     class=1（正常，245 例）vs class 2-16（异常，207 例）——按真实数据分布，非
     网上流传的 "16=正常"（本数据 class 16 仅 22 例）；
  5) HSPSO 在 FS 场景关闭 Hooke-Jeeves（279 维逐粒子 HJ 成本不可接受，且论文
     未说明 FS 应用是否包含 HJ），保留自适应权重+反向学习+柯西变异；
  6) 特征选择映射：连续位置 x∈[0,1]^N，mask = x > 0.5。

用法：
  python hspso_feature_selection.py             # 完整实验（20 次 × 2 算法）
  python hspso_feature_selection.py --smoke     # 快速自检（验证数据与管线）
  python hspso_feature_selection.py --runs 20 --theta 0.9 --workers 8
  python hspso_feature_selection.py --data D:\\path\\arrhythmia.data   # 用本地数据

输出：
  results/results_fs_arrhythmia.csv  每次运行的 适应度/准确率/子集长度/nfev/耗时
  （--tag T 时输出 results_fs_arrhythmia_T.csv，用于多组 θ 参数对比）
  控制台摘要：全特征基线 + 各算法 Best/Avg 准确率、平均子集长度（对齐论文 Table 5 叙述）
"""
from __future__ import annotations
import argparse
import csv
import os
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import algorithms  # 只读复用标准 PSO 与 HSPSO（不修改该文件）

# UCI Arrhythmia 老路径（逗号分隔，279 属性 + 1 类，'?' 为缺失）
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
N_FEATURES = 279


# ---------------------------------------------------------------------------
# 数据加载与预处理
# ---------------------------------------------------------------------------
def _read_csv(source, source_name):
    import pandas as pd
    df = pd.read_csv(source, header=None, sep=",")
    df = df.apply(pd.to_numeric, errors="coerce")          # '?' -> NaN
    df = df.dropna(how="all")                               # 去掉可能的表头行
    df = df.dropna(subset=[df.columns[-1]])                 # 类标签不能缺失
    if df.shape[1] != N_FEATURES + 1:
        raise ValueError(
            "%s: expected %d columns (279 features + class), got %d"
            % (source_name, N_FEATURES + 1, df.shape[1]))
    X = df.iloc[:, :N_FEATURES].to_numpy(dtype=float)
    y = df.iloc[:, N_FEATURES].to_numpy(dtype=float)
    print("loaded %s: %d rows x %d features" % (source_name, X.shape[0], X.shape[1]))
    return X, y


def load_data(path=None):
    """按优先级加载：--data 指定 > 脚本目录 arrhythmia.data 缓存 > UCI 下载 > OpenML。"""
    cache = Path(__file__).resolve().parent / "arrhythmia.data"
    if path:
        return _read_csv(path, "local file")
    if cache.exists():
        return _read_csv(cache, "local arrhythmia.data")
    try:
        print("downloading from UCI archive ...")
        req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("latin-1")
        with cache.open("w", encoding="utf-8") as f:
            f.write(raw)
        return _read_csv(cache, "UCI download")
    except Exception as e:
        print("UCI download failed (%s); trying OpenML ..." % e)
    try:
        from sklearn.datasets import fetch_openml
        df = fetch_openml(name="arrhythmia", version=1, as_frame=True)
        X = df.data.to_numpy(dtype=float)
        y = df.target.astype(float).to_numpy()
        print("loaded OpenML arrhythmia: %d rows x %d features" % (X.shape[0], X.shape[1]))
        return X, y
    except Exception as e:
        raise RuntimeError(
            "Cannot obtain Arrhythmia data. Download arrhythmia.data from UCI "
            "(or run: python -c \"from sklearn.datasets import fetch_openml; "
            "fetch_openml(name='arrhythmia')\") and retry with --data. Error: %s" % e)


def preprocess(X, y):
    """缺失值中位数填充、二分类（class=1 正常/其余=异常）、折内标准化在 fitness 中做。

    注意：UCI Arrhythmia 的类别编码中 **1 = 正常（245 例）**，2-16 = 各类心律失常
    （共 207 例）；网上流传的 "16=正常" 说法与真实数据分布不符（本数据集 class 16 仅 22 例）。
    """
    from sklearn.impute import SimpleImputer
    X = SimpleImputer(strategy="median").fit_transform(X)
    y_bin = np.where(y == 1.0, 0, 1)
    print("binary labels: normal=%d abnormal=%d" % (int((y_bin == 0).sum()),
                                                    int((y_bin == 1).sum())))
    return X, y_bin


def cv_accuracy(mask, X, y, folds, seed):
    """固定划分的 k 折分层交叉验证 SVM 准确率（折内标准化，确定性）。

    论文结论明确 FS 应用使用 SVM，未给 kernel/C/gamma，本脚本采用
    sklearn 默认 SVC()（RBF 核、C=1.0、gamma='scale'）。
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    if int(mask.sum()) == 0:
        return 0.5                                   # 空特征子集：按随机猜测计
    Xs = X[:, mask]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(Xs, y):
        sc = StandardScaler().fit(Xs[tr])
        clf = SVC().fit(sc.transform(Xs[tr]), y[tr])
        accs.append(clf.score(sc.transform(Xs[te]), y[te]))
    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# 子进程任务：(算法, 次数) 一次独立运行
# ---------------------------------------------------------------------------
def _run_task(task):
    (idx, algo, run, seed, pop, iters, theta, folds, X, y) = task
    N = X.shape[1]
    seed_r = seed + run * 100000
    base = np.random.default_rng(seed_r)
    x0 = base.uniform(0.0, 1.0, (pop, N))              # PSO/HSPSO 共享初始种群
    arng = np.random.default_rng(seed_r * 7 + (0 if algo == "PSO" else 3))

    def fun(x):
        # 式(9)：Fitness = θ×(1−Accuracy) + (1−θ)×FS/N
        mask = x > 0.5
        acc = cv_accuracy(mask, X, y, folds, seed)
        return theta * (1.0 - acc) + (1.0 - theta) * int(mask.sum()) / N

    start = time.perf_counter()
    if algo == "HSPSO":
        res = algorithms.hspso(fun, N, 0.0, 1.0, x0, arng, pop=pop, iters=iters,
                               hj_on=False, mutation_rate=0.1)
    else:  # PSO（线性惯性权重，论文式 1-2）
        res = algorithms.pso(fun, N, 0.0, 1.0, x0, arng, pop=pop, iters=iters)
    elapsed = time.perf_counter() - start

    mask = res.best_x > 0.5
    acc = cv_accuracy(mask, X, y, folds, seed)
    return idx, {
        "algorithm": algo, "run": run + 1,
        "best_fitness": res.best_f,
        "best_accuracy": acc,
        "subset_size": int(mask.sum()),
        "nfev": res.nfev, "seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
FIELDNAMES = ["algorithm", "run", "best_fitness", "best_accuracy",
              "subset_size", "nfev", "seconds"]


def run_fs(algorithms_list, runs, pop, iters, theta, folds, seed,
           X, y, outdir, workers, smoke, tag=""):
    N = X.shape[1]
    tasks = [(i, algo, r, seed, pop, iters, theta, folds, X, y)
             for i, (algo, r) in enumerate((a, r)
             for a in algorithms_list for r in range(runs))]
    total = len(tasks)
    nproc = workers if workers and workers > 0 else (os.cpu_count() or 1)
    if smoke:
        nproc = 1

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / ("results_fs_arrhythmia%s.csv" % ("_" + tag if tag else ""))

    results = {}

    def save():
        # 增量保存：每完成一个任务立即写盘，中断时也保留已跑部分
        rows = [results[i] for i in range(total) if i in results]
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(rows)

    print("tasks=%d workers=%d theta=%.1f classifier=SVC -> %s"
          % (total, nproc, theta, out.name), flush=True)
    if nproc == 1:
        for t in tasks:
            i, row = _run_task(t)
            results[i] = row
            save()
            print("[%d/%d] %-5s run=%02d acc=%.4f size=%d" %
                  (len(results), total, row["algorithm"], row["run"],
                   row["best_accuracy"], row["subset_size"]), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futs = [ex.submit(_run_task, t) for t in tasks]
            for fut in as_completed(futs):
                i, row = fut.result()
                results[i] = row
                save()
                print("[%d/%d] %-5s run=%02d acc=%.4f size=%d" %
                      (len(results), total, row["algorithm"], row["run"],
                       row["best_accuracy"], row["subset_size"]), flush=True)

    rows = [results[i] for i in range(total)]

    # 摘要
    baseline = cv_accuracy(np.ones(N, dtype=bool), X, y, folds, seed)
    print("\nbaseline (all %d features, SVM, %d-fold CV): accuracy=%.4f"
          % (N, folds, baseline))
    for algo in algorithms_list:
        rs = [r for r in rows if r["algorithm"] == algo]
        accs = [r["best_accuracy"] for r in rs]
        sizes = [r["subset_size"] for r in rs]
        print("%-6s best_acc=%.4f avg_acc=%.4f avg_subset=%.1f (vs baseline %.1f)"
              % (algo, max(accs), np.mean(accs), np.mean(sizes), N))
    print("\nsaved:", out)


def main():
    ap = argparse.ArgumentParser(description="HSPSO-FS: Arrhythmia feature selection (Yao 2024)")
    ap.add_argument("--data", default=None, help="local arrhythmia.data path")
    ap.add_argument("--algorithms", nargs="+", default=["PSO", "HSPSO"],
                    choices=["PSO", "HSPSO"])
    ap.add_argument("--runs", type=int, default=20, help="independent runs (paper: 20)")
    ap.add_argument("--pop", type=int, default=20, help="population size (paper: 20)")
    ap.add_argument("--iters", type=int, default=100, help="iterations (paper: 100)")
    ap.add_argument("--theta", type=float, default=0.9,
                    help="weight of accuracy in Eq.(9), theta in [0,1] (paper omits value)")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tag", default="",
                    help="output file suffix: results_fs_arrhythmia{_tag}.csv")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers (default: CPU count; 1 = serial)")
    ap.add_argument("--smoke", action="store_true", help="quick pipeline self-check")
    args = ap.parse_args()

    X_raw, y_raw = load_data(args.data)
    X, y = preprocess(X_raw, y_raw)

    if args.smoke:
        args.runs, args.pop, args.iters, args.folds = 2, 10, 10, 3

    run_fs(args.algorithms, args.runs, args.pop, args.iters, args.theta,
           args.folds, args.seed, X, y, args.outdir, args.workers,
           args.smoke, args.tag)


if __name__ == "__main__":
    main()
