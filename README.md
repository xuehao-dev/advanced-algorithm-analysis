# HSPSO 论文复现项目

复现论文：Yao, J., Luo, X., Li, F., et al. (2024). *Research on hybrid strategy Particle Swarm
Optimization algorithm and its applications*. **Scientific Reports 14:24928**.
DOI: 10.1038/s41598-024-76010-y

目标：完整复现论文提出的 **HSPSO（Hybrid Strategy PSO）** 算法，在 CEC-2005 基准函数上与
标准 PSO、DAIW-PSO、FA 做对比实验与统计检验，并在 UCI Arrhythmia 数据集上复现论文的
特征选择应用（HSPSO-FS），为课程汇报（PPT + 代码演示 + 答辩）提供可复现的实验数据与图表素材。

---

## 文件与目录结构（每个脚本 / 文件夹的用途）

```
高级算法分析_project/
├── benchmarks.py              基准函数库：论文 Table 1 的 15 个 CEC-2005 函数
│                              （F1–F15，含标准表达式与注释；F13 用修正真值说明）
├── algorithms.py              算法实现库：标准 PSO、DAIW-PSO、FA、HSPSO + Hooke-Jeeves
│                              （所有公式对应论文式 (1)–(8) 与 Algorithm 2；ALGORITHMS 注册表）
├── run_experiments.py         CEC-2005 复现实验入口：15 函数 × 4 算法 × 30 次独立运行，
│                              进程级并行（--workers），输出 4 份 CSV（原始/汇总/秩和/收敛）
├── plot_results.py            CEC 结果出图：读 results/ 生成 4 张图
│                              （01_error_comparison + 02/03/04_convergence 三段收敛曲线）
├── hspso_feature_selection.py 特征选择应用（HSPSO-FS）：Arrhythmia 数据集（279 特征/452 样本）
│                              ，SVM 分类器 + 5 折分层 CV，式(9) 适应度（θ 可调），
│                              --tag 区分输出文件，边跑边增量保存 CSV
├── plot_fs_results.py         FS 结果出图：读 results_fs_arrhythmia_theta05.csv
│                              生成双面板图（准确率±std + 子集长度 avg/best）
├── README.md                  本说明（项目结构、用途、复现决策、坑位记录）
├── PPT_OUTLINE.md             PPT 大纲与三人分工（33 页 blueprint，含每页素材来源）
├── arrhythmia.data            UCI Arrhythmia 数据集本地缓存（452 行 × 280 列，'?' 缺失）
│
├── results/                   实验输出（自动生成）
│   ├── results_cec2005.csv        CEC-2005 逐次运行原始结果（30 次 × 15 函数 × 4 算法）
│   ├── summary_cec2005.csv        各算法×函数汇总：最优/均值/标准差/中位数
│   ├── pvalues_cec2005.csv        HSPSO 对各基线的 Mann-Whitney U 秩和检验 p 值
│   ├── convergence_cec2005.csv    每代全局最优均值（画收敛曲线用）
│   └── results_fs_arrhythmia_theta05.csv  FS 应用结果（θ=0.5、SVM、20 次×2 算法）
├── plots/                    出图脚本生成的成品图（PPT 直接取用）
│   ├── 01_error_comparison.png       15 函数 × 4 算法误差柱状图（log 轴）
│   ├── 02/03/04_convergence_*.png    收敛曲线（F1–F5 / F6–F10 / F11–F15）
│   └── fs_results_theta05.png        FS 应用双面板图（准确率 + 子集长度）
└── paper_figures/            论文 PDF 中裁剪的 9 张现成图表（PPT 介绍论文用）
    ├── table4_dataset_info.png           Arrhythmia 数据集信息表
    ├── table5_pso_hspso_fs.png           PSO-FS vs HSPSO-FS 结果表
    ├── table6_literature_compare.png     与 7 种文献算法对比表
    ├── fig4_3d_functions_cec2005.png     CEC-2005 六个 3D 函数曲面
    ├── fig5_convergence_cec2005.png      CEC-2005 收敛曲线（论文原图）
    ├── fig6_convergence_cec2014.png      CEC-2014 收敛曲线（论文原图）
    └── table2/table3_*.png               CEC-2005 / CEC-2014 结果大表
```

环境与配置目录（非代码产物）：
- `.venv/`：项目虚拟环境（numpy/scipy/pandas/matplotlib/scikit-learn 已安装）
- `.idea/`：PyCharm 项目配置（IDE 自动生成）
- `.git/`、`.gitignore`：Git 版本管理

---

## 快速开始

```bash
# 激活虚拟环境（或直接使用其中解释器）
.venv\Scripts\python.exe -m pip install numpy scipy pandas scikit-learn matplotlib

# ===== CEC-2005 复现实验 =====
# 完整实验：15 函数 × 30 次 × 4 算法（N=20, T=100，与论文一致）
python run_experiments.py
# 快速自检（2 次 × 小规模）
python run_experiments.py --smoke
# 只跑指定算法 / 关闭 Hooke-Jeeves（消融）
python run_experiments.py --algorithms PSO HSPSO
python run_experiments.py --no-hj

# ===== 特征选择应用（HSPSO-FS）=====
# 完整实验：20 次 × 2 算法，θ=0.5，SVM，5 折 CV（论文未给 θ，本组取 0.5/0.9 两组对比）
python hspso_feature_selection.py --runs 20 --theta 0.5 --tag theta05 --workers 4
python hspso_feature_selection.py --runs 20 --theta 0.9 --tag theta09 --workers 4
# 快速自检
python hspso_feature_selection.py --smoke

# ===== 出图 =====
python plot_results.py      # 生成 CEC 4 张图到 plots/
python plot_fs_results.py   # 生成 FS 双面板图到 plots/
```

输出文件说明见上方目录结构；`results/` 与 `plots/` 目录在脚本首次运行时自动创建。

---

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

**特征选择应用（HSPSO-FS）假设清单**（论文未写明、本组自行确定的细节，答辩需说明）：
1. 分类器：**SVM**（sklearn 默认 SVC：RBF 核、C=1.0、gamma='scale'）——论文结论章节明确
   使用 SVM，但未给 kernel/C 等参数；
2. θ：论文只给 θ∈[0,1]，本组取 **θ=0.5 与 θ=0.9** 两组对比（θ 越接近 0 越强调稀疏）；
3. 评估方式：5 折分层交叉验证（折内标准化）；
4. 缺失值 '?' 用列中位数填充；二分类：class=1（正常，245 例）vs class 2–16（异常，207 例）；
5. FS 场景关闭 Hooke-Jeeves（279 维逐粒子 HJ 成本不可接受），保留其余三策略；
6. 特征映射：x∈[0,1]^N，mask = x > 0.5。

---

## 论文笔误 / 歧义清单（复现时的关键决策，技术记录）

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
   与解析值吻合。**注意：results CSV 的 error 列按论文的 opt=0 计算，F13 行会显示为负，
   做 PPT 表格时请用修正后的最优值 −133.20633 重新换算 error。**
8. **MFE 声明与伪代码矛盾**：论文声明最大评估次数 MFE=N×T=2000，但伪代码要求每个粒子每代
   都跑 Hooke-Jeeves（代价远超 2000）。**处理：按伪代码逐粒子执行（hj_nmax=2），实际 nfev
   如实记录在 CSV 中；可用 `--no-hj` 做消融对比。**
9. **F7 含随机噪声**：Quartic 函数自带 U(0,1) 噪声，结果天然有随机性，不保证逐位复现。

---

## 补充说明

- **FA（萤火虫）** 是补充基线（论文 CEC-2005 表格含 FA），为简化实现（吸收系数按搜索范围自适应）。
- **ACO / BOA 未实现**：连续域 ACO 在论文中定义不清，BOA 可后续补充；论文 CEC-2014 表本身
  只用 PSO/DAIW-PSO/HBF-PSO/HSPSO 四算法，本项目的四算法对比规模与之相当。
- **评估预算差异（技术记录）**：完整实验实测 HSPSO 平均 nfev ≈ 185,000（区间 18k–236k），
  三个基线均为 2,020（N×T）——HSPSO 约多 90 倍评估次数，来源是"每个粒子每代都跑
  Hooke-Jeeves"（论文 Algorithm 1 如此设计，与其 MFE=N×T=2000 的声明矛盾）。结论是
  "以预算换精度"；如需等预算对比，可做消融（`--no-hj`，或只对 gbest 做 HJ）。
- 并行：默认按 CPU 核数并行（`--workers N` 控制），各任务独立种子，结果与串行逐位一致。
- **已删除的过时产物**：早期 KNN 版 FS 实验（式 9 缺稀疏项的 bug 版）与其 θ=0.9 备份
  （results_fs_arrhythmia.csv / results_fs_arrhythmia_theta09.csv）已清理，PPT 只使用
  θ=0.5 的 SVM 版结果。
