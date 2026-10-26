# -*- coding: utf-8 -*-
"""HSPSO 及其对比基线算法 —— Yao et al. 2024 (Sci Rep 14:24928) 复现。

包含：标准 PSO（式 1-2）、DAIW-PSO（自适应惯性权重，式 3-6 修正版）、
FA（萤火虫算法，补充基线）、HSPSO（自适应权重 + 反向学习 + 柯西变异 + Hooke-Jeeves）。

论文笔误与歧义的复现处理（详见 README）：
  1) 式(6) 惯性权重分子用原始迭代数 t，按字面会在 t≈38 后变负并崩到 -39 ——
     复现采用归一化分子 t/T_max，使 w 恒在 [0.4, 0.8]；
  2) Algorithm 1 第 30 行的 w 公式与式(6) 矛盾且会发散 —— 不使用，以修正的式(6) 为准；
  3) 式(7)/伪代码第 32-36 行：当前适应度劣于 pbest 时交换认知/社会项（即“反向学习”），
     方程与伪代码一致，正文文字描述与之相反，以方程+伪代码为准；
  4) 变异率 mutation_rate 论文未给数值，默认取 0.1（可调）；
  5) 实验设置的 “v_min=1, v_max=-1” 为笔误，实现中不显式限速，仅做位置边界裁剪。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List
import numpy as np


@dataclass
class Result:
    best_x: np.ndarray
    best_f: float
    nfev: int
    history: List[float]      # 每代（含初始化）的全局最优，history[0]=初始化最优


# ---------------------------------------------------------------------------
# Hooke-Jeeves 局部搜索（论文 Algorithm 2，Fig. 3）
# ---------------------------------------------------------------------------
def hooke_jeeves(fun: Callable, x0: np.ndarray, lb: float, ub: float,
                 delta0: float, nmax: int = 2, tol: float = 1e-8):
    """对单点做 Hooke-Jeeves 模式搜索，返回 (精修后的点, 本次函数评估次数)。

    按论文 Algorithm 2：逐坐标方向先试 +Δ 再试 -Δ，有改进则接受；整轮无改进则 Δ 减半。
    """
    x = np.clip(np.asarray(x0, dtype=float), lb, ub)
    f = float(fun(x))
    ne = 1
    delta = float(delta0)
    dim = len(x)
    for _ in range(int(nmax)):
        x_best, f_best = x.copy(), f
        for d in range(dim):
            xt = x_best.copy()
            xt[d] = float(np.clip(x_best[d] + delta, lb, ub))
            if xt[d] != x_best[d]:
                ft = float(fun(xt)); ne += 1
                if ft < f_best:
                    x_best, f_best = xt, ft
                else:
                    xt2 = x_best.copy()
                    xt2[d] = float(np.clip(x_best[d] - delta, lb, ub))
                    if xt2[d] != x_best[d]:
                        ft2 = float(fun(xt2)); ne += 1
                        if ft2 < f_best:
                            x_best, f_best = xt2, ft2
        if np.any(x_best != x):
            x, f = x_best, f_best
        else:
            delta = delta / 2.0
            if delta < tol:
                break
    return x, ne


# ---------------------------------------------------------------------------
# 标准 PSO（论文式 1-2：线性衰减惯性权重，c1=c2=1.5）
# ---------------------------------------------------------------------------
def pso(fun: Callable, dim: int, lb: float, ub: float, x0: np.ndarray,
        rng: np.random.Generator, pop: int = 20, iters: int = 100,
        w_lo: float = 0.4, w_hi: float = 0.8, c1: float = 1.5, c2: float = 1.5) -> Result:
    span = ub - lb
    x = np.array(x0, dtype=float, copy=True)
    v = rng.uniform(-0.2, 0.2, (pop, dim)) * span
    fx = np.array([fun(z) for z in x], dtype=float)
    nfev = pop
    pb, pbf = x.copy(), fx.copy()
    gi = int(np.argmin(pbf)); gb, gbf = pb[gi].copy(), float(pbf[gi])
    hist = [gbf]
    for t in range(1, iters + 1):
        w = w_hi - (w_hi - w_lo) * (t / iters)              # 式(2) 线性衰减
        r1, r2 = rng.random((pop, dim)), rng.random((pop, dim))
        v = w * v + c1 * r1 * (pb - x) + c2 * r2 * (gb - x)
        x = np.clip(x + v, lb, ub)
        fx_new = np.array([fun(z) for z in x], dtype=float); nfev += pop
        imp = fx_new < pbf
        pb[imp] = x[imp]; pbf[imp] = fx_new[imp]
        j = int(np.argmin(pbf))
        if pbf[j] < gbf:
            gb, gbf = pb[j].copy(), float(pbf[j])
        hist.append(gbf)
    return Result(gb, gbf, nfev, hist)


# ---------------------------------------------------------------------------
# DAIW-PSO（论文式 3-6 的自适应惯性权重 + 标准 PSO 更新）
# ---------------------------------------------------------------------------
def daiw_pso(fun: Callable, dim: int, lb: float, ub: float, x0: np.ndarray,
             rng: np.random.Generator, pop: int = 20, iters: int = 100,
             w_lo: float = 0.4, w_hi: float = 0.8, c1: float = 1.5, c2: float = 1.5,
             b: float = 1.0) -> Result:
    span = ub - lb
    x = np.array(x0, dtype=float, copy=True)
    v = rng.uniform(-0.2, 0.2, (pop, dim)) * span
    fx = np.array([fun(z) for z in x], dtype=float)
    nfev = pop
    pb, pbf = x.copy(), fx.copy()
    gi = int(np.argmin(pbf)); gb, gbf = pb[gi].copy(), float(pbf[gi])
    hist = [gbf]
    prev_std = None
    for t in range(1, iters + 1):
        # 式(3) 种群适应度标准差 STF；式(4) 适应性离散度 H（除零保护）
        std = float(np.std(fx))
        # 式(4) H：除零保护（prev_std 与 std 任一为 0 时取 1.0，避免 2t/(H*iters) 除零）
        if prev_std is not None and prev_std > 1e-12 and std > 1e-12:
            H = std / prev_std
        else:
            H = 1.0
        prev_std = std
        # 式(6) 修正版：分子归一化为 t/T_max，w 恒在 [w_lo, w_hi]
        w = w_hi - (w_hi - w_lo) * (t / iters) / \
            (1.0 + np.exp(-10.0 * b * (2.0 * t / (H * iters) - 1.0)))
        w = float(np.clip(w, w_lo, w_hi))
        r1, r2 = rng.random((pop, dim)), rng.random((pop, dim))
        v = w * v + c1 * r1 * (pb - x) + c2 * r2 * (gb - x)
        x = np.clip(x + v, lb, ub)
        fx_new = np.array([fun(z) for z in x], dtype=float); nfev += pop
        imp = fx_new < pbf
        pb[imp] = x[imp]; pbf[imp] = fx_new[imp]
        j = int(np.argmin(pbf))
        if pbf[j] < gbf:
            gb, gbf = pb[j].copy(), float(pbf[j])
        hist.append(gbf)
    return Result(gb, gbf, nfev, hist)


# ---------------------------------------------------------------------------
# FA 萤火虫算法（补充基线；简化实现，README 已注明）
# ---------------------------------------------------------------------------
def fa(fun: Callable, dim: int, lb: float, ub: float, x0: np.ndarray,
       rng: np.random.Generator, pop: int = 20, iters: int = 100,
       beta0: float = 1.0, alpha0: float = 0.5, decay: float = 0.95) -> Result:
    span = ub - lb
    gamma = 1.0 / (span * span)                     # 吸收系数按搜索范围自适应
    alpha = alpha0 * span
    x = np.array(x0, dtype=float, copy=True)
    fx = np.array([fun(z) for z in x], dtype=float)
    nfev = pop
    gi = int(np.argmin(fx)); gb, gbf = x[gi].copy(), float(fx[gi])
    hist = [gbf]
    for _ in range(1, iters + 1):
        for i in range(pop):
            for j in range(pop):
                if fx[j] < fx[i]:                    # 更亮的萤火虫吸引
                    r2 = float(np.sum((x[i] - x[j]) ** 2))
                    beta = beta0 * np.exp(-gamma * r2)
                    x[i] = np.clip(x[i] + beta * (x[j] - x[i])
                                   + alpha * (rng.random(dim) - 0.5), lb, ub)
        fx = np.array([fun(z) for z in x], dtype=float); nfev += pop
        gi = int(np.argmin(fx))
        if fx[gi] < gbf:
            gb, gbf = x[gi].copy(), float(fx[gi])
        alpha *= decay
        hist.append(gbf)
    return Result(gb, gbf, nfev, hist)


# ---------------------------------------------------------------------------
# HSPSO（论文 Algorithm 1：自适应权重 + 反向学习 + 柯西变异 + Hooke-Jeeves）
# ---------------------------------------------------------------------------
def hspso(fun: Callable, dim: int, lb: float, ub: float, x0: np.ndarray,
          rng: np.random.Generator, pop: int = 20, iters: int = 100,
          c1: float = 1.5, c2: float = 1.5, w_lo: float = 0.4, w_hi: float = 0.8,
          b: float = 1.0, mutation_rate: float = 0.1, cauchy_gamma: float = 0.3,
          hj_on: bool = True, hj_delta0: float = None, hj_nmax: int = 2,
          hj_tol: float = 1e-8) -> Result:
    span = ub - lb
    if hj_delta0 is None:
        hj_delta0 = 0.05 * span
    x = np.array(x0, dtype=float, copy=True)
    v = rng.uniform(-0.2, 0.2, (pop, dim)) * span
    fx = np.array([fun(z) for z in x], dtype=float)
    nfev = pop
    pb, pbf = x.copy(), fx.copy()
    gi = int(np.argmin(pbf)); gb, gbf = pb[gi].copy(), float(pbf[gi])
    hist = [gbf]
    prev_std = None
    for t in range(1, iters + 1):
        # 式(3)(4)：种群适应度标准差与适应性离散度（除零保护）
        std = float(np.std(fx))
        # 式(4) H：除零保护（prev_std 与 std 任一为 0 时取 1.0，避免 2t/(H*iters) 除零）
        if prev_std is not None and prev_std > 1e-12 and std > 1e-12:
            H = std / prev_std
        else:
            H = 1.0
        prev_std = std
        # 式(6) 修正版：分子归一化，w 恒在 [w_lo, w_hi]
        w = w_hi - (w_hi - w_lo) * (t / iters) / \
            (1.0 + np.exp(-10.0 * b * (2.0 * t / (H * iters) - 1.0)))
        w = float(np.clip(w, w_lo, w_hi))

        for i in range(pop):
            # 式(7) / Algorithm 1 第 32-36 行：反向学习
            if fx[i] <= pbf[i]:                       # 当前不劣于历史最优：标准速度更新
                v[i] = w * v[i] + c1 * rng.random(dim) * (pb[i] - x[i]) \
                       + c2 * rng.random(dim) * (gb - x[i])
            else:                                     # 当前劣于历史最优：交换认知/社会项
                v[i] = w * v[i] + c1 * rng.random(dim) * (gb - x[i]) \
                       + c2 * rng.random(dim) * (pb[i] - x[i])
            # 式(8) 柯西变异（按伪代码顺序，先于位置更新执行）
            if rng.random() < mutation_rate:
                cauchy = np.tan((rng.random(dim) - 0.5) * np.pi)
                x[i] = np.clip(x[i] * (1.0 + cauchy_gamma * cauchy), lb, ub)
            # 位置更新 + 边界裁剪
            x[i] = np.clip(x[i] + v[i], lb, ub)
            # Hooke-Jeeves 局部精修（论文 Algorithm 1 第 45-46 行）
            if hj_on:
                x[i], ne = hooke_jeeves(fun, x[i], lb, ub, hj_delta0, hj_nmax, hj_tol)
                nfev += ne

        fx = np.array([fun(z) for z in x], dtype=float); nfev += pop
        imp = fx < pbf
        pb[imp] = x[imp]; pbf[imp] = fx[imp]
        j = int(np.argmin(pbf))
        if pbf[j] < gbf:
            gb, gbf = pb[j].copy(), float(pbf[j])
        hist.append(gbf)
    return Result(gb, gbf, nfev, hist)


# 算法注册表：(名称, 调用函数)
ALGORITHMS = {
    "PSO":   pso,
    "DAIW":  daiw_pso,
    "FA":    fa,
    "HSPSO": hspso,
}

# 用于区分各算法内部随机流（初始种群 X0 各算法共享，保证公平）
ALGO_OFFSET = {"PSO": 0, "DAIW": 1, "FA": 2, "HSPSO": 3}
