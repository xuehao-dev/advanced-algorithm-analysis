# HSPSO 论文复现项目

复现论文：Yao, J., Luo, X., Li, F., et al. (2024). *Research on hybrid strategy Particle Swarm
Optimization algorithm and its applications*. **Scientific Reports 14:24928**.
DOI: 10.1038/s41598-024-76010-y

目标：完整复现论文提出的 **HSPSO（Hybrid Strategy PSO）** 算法，并在 CEC-2005 基准函数上
与标准 PSO、DAIW-PSO、FA 做对比实验与统计检验，为课程汇报（PPT + 代码演示 + 答辩）提供
可复现的实验数据与图表素材。

---

## 文件结构

```
高级算法分析_project/
├── benchmarks.py          CEC-2005（论文 Table 1）15 个基准函数
├── algorithms.py          标准 PSO、DAIW-PSO、FA、HSPSO + Hooke-Jeeves
├── run_experiments.py     实验入口（多进程并行、CSV 输出、秩和检验、收敛曲线）
├── README.md              本说明
└── results/               实验输出（运行时自动创建）
```

## 快速开始

```bash
pip install numpy scipy pandas

# 完整实验：15 函数 × 30 次 × 4 算法（N=20, T=100，与论文一致）
python run_experiments.py

# 快速自检
python run_experiments.py --smoke

# 只跑两个算法 / 关闭 Hooke-Jeeves（消融实验）
python run_experiments.py --algorithms PSO HSPSO
python run_experiments.py --no-hj
```

输出文件（`results/` 目录）：
- `results_cec2005.csv`：逐次运行原始结果
- `summary_cec2005.csv`：各算法×函数的最优/均值/标准差/中位数
- `pvalues_cec2005.csv`：HSPSO 对各基线的 Mann-Whitney U 秩和检验 p 值（<0.05 为显著）
- `convergence_cec2005.csv`：每代全局最优均值（画收敛曲线用）

## 算法实现要点（对应论文公式）

| 策略 | 论文公式 | 本实现 |
| --- | --- | --- |
| 标准 PSO 基线 | 式 (1)(2)，ω 线性 0.8→0.4 | `pso()` |
| 自适应惯性权重 | 式 (3)-(6)，STF 标准差 + H 离散度 + sigmoid 权重 | `daiw_pso()` / HSPSO 内 |
| 反向学习 | 式 (7) | HSPSO 速度更新：当前适应度劣于 pbest 时交换认知/社会项 |
| 柯西变异 | 式 (8)，γ=0.3 | HSPSO 内，`mutation_rate` 可调 |
| Hooke-Jeeves | Algorithm 2 (Fig.3) | `hooke_jeeves()`，逐粒子调用 |

实验设置与论文一致：N=20、T=100、c1=c2=1.5、ω∈[0.4,0.8]、30 次独立运行、
各算法共享同一初始种群（保证公平）。

## 论文笔误 / 歧义清单（复现时的关键决策，答辩素材）

1. **式 (6) 惯性权重会变负（重要）**：论文打印的分子是原始迭代数 t（非归一化），按字面
   实现时 ω = 0.8 − 0.4t/(1+exp(...))，在 t≈38 后跌破 0，t=100 时 ≈ −39，算法直接发散。
   **处理：分子改用归一化的 t/T_max**（与论文式 (2) 一致），ω 恒在 [0.4, 0.8]。
2. **Algorithm 1 第 30 行的 w 公式与式 (6) 矛盾**：伪代码里的公式恒大于 0.8 且会指数发散，
   判定为错印。**处理：以修正后的式 (6) 为准，伪代码那一行不使用。**
3. **反向学习的条件（式 7 vs 正文文字）**：式 (7) 与 Algorithm 1 第 32-36 行一致——
   “当前适应度劣于 pbest 时交换认知/社会项”；正文散文描述与之相反。**处理：以方程+伪代码为准。**
4. **变异率未给出数值**：论文只写 “Set inertia weight w and mutation rate”，未给 mutation_rate
   取值。**处理：默认取 0.1（`--mutation-rate` 可调），答辩中如实说明这是假设。**
5. **速度上下限笔误**：实验设置写 “v_min=1, v_max=−1”（符号写反且数值可疑）。**处理：不显式限速，
   只做位置边界裁剪**（这也是多数 PSO 实现的常见做法）。
6. **F12 / F15 公式印刷疑截断**：Penalized 2 与 Branin 采用标准定义实现。
7. **F13 的 Table 1 最小值标注有误（实验后核实）**：论文标 F13 最小值为 0，但
   h(x)=x·sin(x)+0.1|x| 在 [-10,10] 上的单维最小值为 **−4.440211**（位于边界 x=±10），
   30 维全局最优 ≈ **−133.20633**。本实验 HSPSO 30/30 次独立运行全部收敛到 −133.2（std=0），
   与解析值吻合。**注意：当前 results CSV 的 error 列按论文的 opt=0 计算，F13 行会显示为负，
   做 PPT 表格时请用修正后的最优值 −133.20633 重新换算 error。**
7. **MFE 声明与伪代码矛盾**：论文声明最大评估次数 MFE=N×T=2000，但伪代码要求每个粒子每代
   都跑 Hooke-Jeeves（代价远超 2000）。**处理：按伪代码逐粒子执行（hj_nmax=2），实际 nfev
   如实记录在 CSV 中；可用 `--no-hj` 做消融对比。**
8. **F7 含随机噪声**：Quartic 函数自带 U(0,1) 噪声，结果天然有随机性，不保证逐位复现。

## 补充说明

- **FA（萤火虫）** 是补充基线（论文 CEC-2005 表格含 FA），为简化实现（吸收系数按搜索范围自适应）。
- **ACO / BOA** 未实现：连续域 ACO 在论文中定义不清，BOA 可后续补充；论文 CEC-2014 表本身
  只用 PSO/DAIW-PSO/HBF-PSO/HSPSO 四算法，本项目的四算法对比规模与之相当。
- **评估预算差异（答辩必讲）**：完整实验实测 HSPSO 平均 nfev ≈ 185,000（区间 18k–236k），
  三个基线均为 2,020（N×T）——HSPSO 约多 90 倍评估次数，来源是"每个粒子每代都跑
  Hooke-Jeeves"（论文 Algorithm 1 如此设计，与其 MFE=N×T=2000 的声明矛盾）。结论是
  "以预算换精度"；如需等预算对比，可做消融（`--no-hj`，或只对 gbest 做 HJ）。
- 并行：默认按 CPU 核数并行（`--workers N` 控制），各任务独立种子，结果与串行逐位一致。
