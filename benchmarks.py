# -*- coding: utf-8 -*-
"""CEC-2005（论文 Table 1）基准函数 —— Yao et al. 2024《Research on hybrid strategy
particle swarm optimization algorithm and its applications》(Sci Rep 14:24928) 复现用。

注意：论文把这张表标注为 "CEC-2005"，但表内 F1–F15 的表达式是经典（未平移/未旋转）测试函数。
这里严格按论文 Table 1 的表达式实现，并给出论文列出的取值范围与全局最优值。全部为最小化问题。

论文 Table 1 一览（Dim / Range / Min）：
  F1  Sphere                  30 [-100,100] 0
  F2  Schwefel 2.22           30 [-10,10]   0
  F3  Schwefel 1.2            30 [-100,100] 0
  F4  Schwefel 2.21           30 [-100,100] 0
  F5  Rosenbrock              30 [-10,10]   0
  F6  Step                    30 [-100,100] 0
  F7  Quartic（含噪声）       30 [-1.28,1.28] 0
  F8  Schwefel 2.26           30 [-500,500] -418.9828872724339*n
  F9  Rastrigin               30 [-5.12,5.12] 0
  F10 Griewank                30 [-600,600] 0
  F11 Generalized Penalized 1 30 [-50,50]   0
  F12 Generalized Penalized 2 30 [-50,50]   0
  F13 (x·sin(x)+0.1|x|)       30 [-10,10]   0
  F14 Six-Hump Camel          2  [-5,5]     -1.0316284535
  F15 Branin                  2  [-5,5]     0.39788735772973816

复现修正记录（详见 README）：
  * F7 含 U(0,1) 随机噪声，结果天然带随机性；
  * F12 (Penalized 2) 论文打印式疑似截断，采用标准定义；
  * F15 (Branin) 论文打印式缺括号，采用标准定义。
"""
from __future__ import annotations
import numpy as np


def _u(x, a, k, m):
    """Penalized 函数使用的 u(x,a,k,m) 分段罚函数。"""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    gt = x > a
    lt = x < -a
    out[gt] = k * (x[gt] - a) ** m
    out[lt] = k * (-x[lt] - a) ** m
    return out


def f1(x):
    """Sphere: sum(x_i^2)"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def f2(x):
    """Schwefel 2.22: sum|x_i| + prod|x_i|"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))


def f3(x):
    """Schwefel 1.2: sum( (sum_{j=1..i} x_j)^2 )"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.cumsum(x) ** 2))


def f4(x):
    """Schwefel 2.21: max|x_i|"""
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x)))


def f5(x):
    """Rosenbrock: sum(100(x_{i+1}-x_i^2)^2 + (x_i-1)^2)"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def f6(x):
    """Step: sum( floor(x_i+0.5)^2 )"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.floor(x + 0.5) ** 2))


def f7(x):
    """Quartic with noise: sum(i·x_i^4) + U(0,1)"""
    x = np.asarray(x, dtype=float)
    noise = np.random.uniform(0.0, 1.0)
    return float(np.sum(np.arange(1, len(x) + 1) * x ** 4) + noise)


def f8(x):
    """Schwefel 2.26: sum(-x_i·sin(sqrt(|x_i|)))"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))


def f9(x):
    """Rastrigin: sum(x_i^2 - 10cos(2πx_i) + 10)"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x) + 10.0))


def f10(x):
    """Griewank: 1 + sum(x^2)/4000 - prod(cos(x_i/sqrt(i)))"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    idx = np.sqrt(np.arange(1, n + 1, dtype=float))
    return float(1.0 + np.sum(x * x) / 4000.0 - np.prod(np.cos(x / idx)))


def f11(x):
    """Generalized Penalized 1（论文式与标准定义一致）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = 1.0 + (x + 1.0) / 4.0
    term = (np.pi / n) * (10.0 * np.sin(np.pi * y[0]) ** 2
                          + np.sum((y[:-1] - 1.0) ** 2 * (1.0 + np.sin(np.pi * y[1:]) ** 2))
                          + (y[-1] - 1.0) ** 2)
    return float(term + np.sum(_u(x, 10.0, 100.0, 4.0)))


def f12(x):
    """Generalized Penalized 2（采用标准定义；论文打印式疑似截断）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = np.sin(3.0 * np.pi * x[0]) ** 2 \
        + np.sum((x[:-1] - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x[1:]) ** 2)) \
        + (x[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x[-1]) ** 2)
    return float(0.1 * s + np.sum(_u(x, 5.0, 100.0, 4.0)))


def f13(x):
    """F13: sum(x_i·sin(x_i) + 0.1|x_i|)（论文 Table 1 原式）。"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * np.sin(x) + 0.1 * np.abs(x)))


def f14(x):
    """Six-Hump Camel Back（2 维）。"""
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    return float(4.0 * x1 ** 2 - 2.1 * x1 ** 4 + (1.0 / 3.0) * x1 ** 6
                 + x1 * x2 - 4.0 * x2 ** 2 + 4.0 * x2 ** 4)


def f15(x):
    """Branin（2 维，标准定义）。"""
    x = np.asarray(x, dtype=float)
    x1, x2 = x[0], x[1]
    a = 5.1 / (4.0 * np.pi ** 2)
    b = 5.0 / np.pi
    return float((x2 - a * x1 ** 2 + b * x1 - 6.0) ** 2
                 + 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(x1) + 10.0)


# (名称, 函数, 维度, 下界, 上界, 全局最优值)
BENCHMARKS = [
    ("F1",  f1,  30, -100.0,  100.0, 0.0),
    ("F2",  f2,  30,  -10.0,   10.0, 0.0),
    ("F3",  f3,  30, -100.0,  100.0, 0.0),
    ("F4",  f4,  30, -100.0,  100.0, 0.0),
    ("F5",  f5,  30,  -10.0,   10.0, 0.0),
    ("F6",  f6,  30, -100.0,  100.0, 0.0),
    ("F7",  f7,  30,   -1.28,   1.28, 0.0),
    ("F8",  f8,  30, -500.0,  500.0, -418.9828872724339 * 30.0),
    ("F9",  f9,  30,   -5.12,   5.12, 0.0),
    ("F10", f10, 30, -600.0,  600.0, 0.0),
    ("F11", f11, 30,  -50.0,   50.0, 0.0),
    ("F12", f12, 30,  -50.0,   50.0, 0.0),
    ("F13", f13, 30,  -10.0,   10.0, 0.0),
    ("F14", f14,  2,   -5.0,    5.0, -1.0316284535),
    ("F15", f15,  2,   -5.0,    5.0, 0.39788735772973816),
]


def get_benchmarks(names=None):
    """返回基准函数列表；names 为 None 时返回全部 15 个。"""
    if not names:
        return list(BENCHMARKS)
    wanted = set(names)
    return [b for b in BENCHMARKS if b[0] in wanted]
