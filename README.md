# 基于 Cauchy 变异的粒子群优化算法

本项目围绕 Yao 等人 2024 年文章中的 **Cauchy 变异机制**展开，范围控制为一条容易复现和答辩的课程作业主线：

1. 标准全局最优粒子群优化（PSO）；
2. 停滞触发的 Cauchy-PSO；
3. 基于群体多样性动态调整触发阈值的 Dynamic-Cauchy-PSO；
4. 经典基准函数对比、收敛曲线和汇总图；
5. 以标准 PSO 为“去掉 Cauchy 机制”的消融对照；
6. Cauchy 强度和停滞阈值的参数敏感性实验。

项目不声称完整复现论文中的 HS-PSO，也不在当前版本加入 CEC 官方完整套件或 UCI 心律失常特征选择。这样可以把课程重点放在一个改进机制上，并保证每个实验结果都能解释。

## 算法

标准 PSO 使用速度和位置更新公式：

```text
v(t+1) = w v(t) + c1 r1 (pbest - x(t)) + c2 r2 (gbest - x(t))
x(t+1) = x(t) + v(t+1)
```

当全局最优值连续 `stagnation_limit` 次没有改善时，Cauchy-PSO 随机选择一部分粒子，并执行论文中的变异公式：

```text
x_new = x * (1 + gamma * Cauchy(0, 1))
```

Cauchy 分布具有长尾，偶尔可以产生较大跳跃，帮助粒子离开局部最优区域。变异后的位置会被截断到搜索边界内，只有更好的位置才会更新历史最优解。

Dynamic-Cauchy-PSO 是本项目的小改进：

- 群体已经很集中时，缩短停滞等待时间，尽快变异；
- 群体仍然分散时，延长等待时间，避免打断正常探索。

## 文件

- `cauchy_pso.py`：算法、测试函数、对比实验、消融实验、参数敏感性实验和绘图。
- `requirements.txt`：NumPy 和 Matplotlib 依赖。
- `results/experiment_results.csv`：每次运行和每个算法汇总结果。
- `results/convergence_plots/`：每个函数的平均收敛曲线和总均值对比柱状图。
- `results/sensitivity_results.csv`：参数敏感性结果。
- `results/sensitivity_heatmap.png`：参数敏感性热力图。

## 安装

```powershell
pip install -r requirements.txt
```

## 运行

快速演示：

```powershell
python cauchy_pso.py --dimensions 10 --particles 20 --iterations 100 --runs 5 --plot
```

正式实验：

```powershell
python cauchy_pso.py --dimensions 30 --particles 30 --iterations 500 --runs 30 --plot
```

加上 Cauchy 参数敏感性实验：

```powershell
python cauchy_pso.py --dimensions 30 --particles 30 --iterations 500 --runs 30 --plot --sensitivity
```

敏感性实验默认在 Rastrigin 函数上比较：

- `gamma = 0.1, 0.3, 0.5`；
- `stagnation_limit = 10, 20, 50`。

## 如何写实验结论

每个函数至少报告：

1. 最优值；
2. 30 次运行的平均值；
3. 标准差；
4. 平均运行时间；
5. Cauchy 变异次数；
6. 平均收敛曲线。

最小化问题中，平均值越低越好。不要只挑选 Cauchy-PSO 胜出的函数，而应同时报告它变差的函数，并解释 Cauchy 扰动可能过强的原因。

## 参考文献

Yao, J., Luo, X., Li, F., et al. Research on hybrid strategy particle swarm optimization algorithm and its applications. *Scientific Reports*, 2024, 14:24928. DOI: 10.1038/s41598-024-76010-y.
