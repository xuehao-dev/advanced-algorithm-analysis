"""标准 PSO、停滞触发的 Cauchy-PSO 和动态阈值 Cauchy-PSO。

本项目采用 Yao 等人（2024）文章中的 Cauchy 变异公式：

    x' = x * (1 + gamma * Cauchy(0, 1))

论文伪代码没有完全规定统一的变异触发条件，因此本项目采用清晰可复现的规则：
当全局最优值连续 ``stagnation_limit`` 次迭代没有改善时，对部分粒子执行 Cauchy 变异。
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


Objective = Callable[[np.ndarray], np.ndarray]


def configure_chinese_font() -> str:
    """自动选择系统中文字体，避免 Matplotlib 绘图文字显示为方框。"""

    # 延迟导入 Matplotlib，使只运行算法而不绘图时仍然只需要 NumPy。
    import matplotlib
    from matplotlib import font_manager

    # 按常见程度排列 Windows 和跨平台环境中可能存在的中文字体。
    preferred_fonts = (
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "DengXian",
        "Source Han Serif SC",
        "Noto Sans CJK SC",
    )
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font for font in preferred_fonts if font in available_fonts), None)
    if selected_font is None:
        raise RuntimeError("未找到可显示中文的 Matplotlib 字体，请安装微软雅黑、黑体或思源黑体")

    # 设置中文字体和负号显示规则，防止坐标轴负号变成方框。
    matplotlib.rcParams["font.sans-serif"] = [selected_font]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return selected_font


def display_algorithm_name(algorithm: str) -> str:
    """将代码中的算法标识转换为图表中更易读的中文名称。"""

    names = {
        "PSO": "标准 PSO",
        "Cauchy-PSO": "Cauchy 变异 PSO",
        "Dynamic-Cauchy-PSO": "动态 Cauchy 变异 PSO",
    }
    return names.get(algorithm, algorithm)


@dataclass(frozen=True)
class Benchmark:
    """表示一个最小化基准函数及其搜索边界。"""

    name: str
    objective: Objective
    lower: float
    upper: float


@dataclass
class OptimizationResult:
    """保存一次优化运行产生的结果。"""

    best_position: np.ndarray
    best_value: float
    convergence: np.ndarray
    mutation_events: int = 0
    mutation_calls: int = 0


def _as_batch(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        return values[None, :]
    if values.ndim != 2:
        raise ValueError("粒子位置必须是 (dimensions,) 或 (particles, dimensions) 形状")
    return values


def sphere(x: np.ndarray) -> np.ndarray:
    x = _as_batch(x)
    return np.sum(x * x, axis=1)


def schwefel_2_22(x: np.ndarray) -> np.ndarray:
    x = _as_batch(x)
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    x = _as_batch(x)
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    x = _as_batch(x)
    return 10.0 * x.shape[1] + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x), axis=1)


def noncontinuous_rastrigin(x: np.ndarray) -> np.ndarray:
    x = _as_batch(x)
    rounded = np.where(np.abs(x) < 0.5, x, np.round(2.0 * x) / 2.0)
    return rastrigin(rounded)


def ackley(x: np.ndarray) -> np.ndarray:
    x = _as_batch(x)
    dimension = x.shape[1]
    mean_square = np.sum(x * x, axis=1) / dimension
    mean_cosine = np.sum(np.cos(2.0 * np.pi * x), axis=1) / dimension
    return -20.0 * np.exp(-0.2 * np.sqrt(mean_square)) - np.exp(mean_cosine) + 20.0 + np.e


def griewank(x: np.ndarray) -> np.ndarray:
    x = _as_batch(x)
    indexes = np.sqrt(np.arange(1, x.shape[1] + 1, dtype=float))
    return np.sum(x * x, axis=1) / 4000.0 - np.prod(np.cos(x / indexes), axis=1) + 1.0


def schwefel(x: np.ndarray) -> np.ndarray:
    x = _as_batch(x)
    return 418.9829 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def generalized_penalized(x: np.ndarray) -> np.ndarray:
    x = _as_batch(x)
    y = 1.0 + (x + 1.0) / 4.0
    core = (np.pi / x.shape[1]) * (
        10.0 * np.sin(np.pi * y[:, 0]) ** 2
        + np.sum((y[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * y[:, 1:]) ** 2), axis=1)
        + (y[:, -1] - 1.0) ** 2
    )

    def penalty(values: np.ndarray) -> np.ndarray:
        return np.where(values > 10.0, 100.0 * (values - 10.0) ** 4, 0.0) + np.where(
            values < -10.0, 100.0 * (-values - 10.0) ** 4, 0.0
        )

    return core + np.sum(penalty(x), axis=1)


def default_benchmarks() -> tuple[Benchmark, ...]:
    """返回一组同时覆盖单峰和多峰情况的紧凑基准函数。"""

    return (
        Benchmark("Sphere", sphere, -100.0, 100.0),
        Benchmark("Schwefel_2_22", schwefel_2_22, -10.0, 10.0),
        Benchmark("Rosenbrock", rosenbrock, -30.0, 30.0),
        Benchmark("Rastrigin", rastrigin, -5.12, 5.12),
        Benchmark("Noncontinuous_Rastrigin", noncontinuous_rastrigin, -5.12, 5.12),
        Benchmark("Ackley", ackley, -32.0, 32.0),
        Benchmark("Griewank", griewank, -600.0, 600.0),
        Benchmark("Schwefel", schwefel, -500.0, 500.0),
        Benchmark("Generalized_Penalized", generalized_penalized, -50.0, 50.0),
    )


class ParticleSwarmOptimizer:
    """用于最小化问题的标准全局最优 PSO。"""

    algorithm_name = "PSO"

    def __init__(
        self,
        objective: Objective,
        lower: float,
        upper: float,
        dimensions: int,
        particles: int = 30,
        iterations: int = 500,
        c1: float = 2.0,
        c2: float = 2.0,
        w_max: float = 0.9,
        w_min: float = 0.4,
        rng: np.random.Generator | None = None,
    ) -> None:
        if dimensions < 2 or particles < 2 or iterations < 1:
            raise ValueError("维度数和粒子数必须至少为 2，迭代次数必须为正数")
        if lower >= upper:
            raise ValueError("搜索下界必须小于搜索上界")
        self.objective = objective
        self.lower = float(lower)
        self.upper = float(upper)
        self.dimensions = dimensions
        self.particles = particles
        self.iterations = iterations
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.rng = rng or np.random.default_rng()
        self._span = self.upper - self.lower

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        positions = self.rng.uniform(self.lower, self.upper, size=(self.particles, self.dimensions))
        velocities = self.rng.uniform(-0.2 * self._span, 0.2 * self._span, size=positions.shape)
        return positions, velocities

    def _evaluate(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(self.objective(positions), dtype=float)
        if values.shape != (positions.shape[0],):
            raise ValueError("目标函数必须为每个粒子返回一个适应度值")
        return values

    def _update_global_best(
        self,
        positions: np.ndarray,
        fitness: np.ndarray,
        best_position: np.ndarray,
        best_value: float,
    ) -> tuple[np.ndarray, float, bool]:
        index = int(np.argmin(fitness))
        if fitness[index] < best_value:
            return positions[index].copy(), float(fitness[index]), True
        return best_position, best_value, False

    def optimize(
        self,
        initial_positions: np.ndarray | None = None,
        initial_velocities: np.ndarray | None = None,
    ) -> OptimizationResult:
        if initial_positions is None or initial_velocities is None:
            positions, velocities = self.initial_state()
        else:
            positions = np.array(initial_positions, dtype=float, copy=True)
            velocities = np.array(initial_velocities, dtype=float, copy=True)
        if positions.shape != (self.particles, self.dimensions):
            raise ValueError("初始位置数组的形状不符合要求")
        if velocities.shape != positions.shape:
            raise ValueError("初始速度数组必须与初始位置数组具有相同形状")

        positions = np.clip(positions, self.lower, self.upper)
        fitness = self._evaluate(positions)
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        best_position, best_value, _ = self._update_global_best(
            pbest_positions, pbest_fitness, positions[0], float("inf")
        )
        convergence: list[float] = []

        for iteration in range(self.iterations):
            weight = self.w_max - (self.w_max - self.w_min) * iteration / max(1, self.iterations - 1)
            r1 = self.rng.random((self.particles, self.dimensions))
            r2 = self.rng.random((self.particles, self.dimensions))
            velocities = (
                weight * velocities
                + self.c1 * r1 * (pbest_positions - positions)
                + self.c2 * r2 * (best_position - positions)
            )
            velocities = np.clip(velocities, -self._span, self._span)
            positions = np.clip(positions + velocities, self.lower, self.upper)
            fitness = self._evaluate(positions)

            improved = fitness < pbest_fitness
            pbest_positions[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]
            best_position, best_value, _ = self._update_global_best(
                pbest_positions, pbest_fitness, best_position, best_value
            )
            convergence.append(best_value)

        return OptimizationResult(best_position, best_value, np.asarray(convergence))


class CauchyMutationPSO(ParticleSwarmOptimizer):
    """加入参考论文乘法式 Cauchy 变异的 PSO。"""

    algorithm_name = "Cauchy-PSO"

    def __init__(
        self,
        *args,
        gamma: float = 0.3,
        mutation_rate: float = 0.2,
        stagnation_limit: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if gamma <= 0 or not 0 < mutation_rate <= 1 or stagnation_limit < 1:
            raise ValueError("gamma 必须为正数，mutation_rate 必须在 (0, 1] 内，停滞阈值必须为正数")
        self.gamma = gamma
        self.mutation_rate = mutation_rate
        self.stagnation_limit = stagnation_limit

    def _mutate(self, positions: np.ndarray) -> int:
        selected = self.rng.random(self.particles) < self.mutation_rate
        if not np.any(selected):
            selected[int(self.rng.integers(0, self.particles))] = True
        uniform = self.rng.random((int(np.sum(selected)), self.dimensions))
        cauchy = np.tan(np.pi * (uniform - 0.5))
        positions[selected] *= 1.0 + self.gamma * cauchy
        positions[selected] = np.clip(positions[selected], self.lower, self.upper)
        return int(np.sum(selected))

    def _stagnation_threshold(self, positions: np.ndarray) -> int:
        """返回触发变异所需的连续停滞迭代次数。"""

        return self.stagnation_limit

    def optimize(
        self,
        initial_positions: np.ndarray | None = None,
        initial_velocities: np.ndarray | None = None,
    ) -> OptimizationResult:
        if initial_positions is None or initial_velocities is None:
            positions, velocities = self.initial_state()
        else:
            positions = np.array(initial_positions, dtype=float, copy=True)
            velocities = np.array(initial_velocities, dtype=float, copy=True)
        if positions.shape != (self.particles, self.dimensions):
            raise ValueError("初始位置数组的形状不符合要求")
        if velocities.shape != positions.shape:
            raise ValueError("初始速度数组必须与初始位置数组具有相同形状")

        positions = np.clip(positions, self.lower, self.upper)
        fitness = self._evaluate(positions)
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        best_position, best_value, _ = self._update_global_best(
            pbest_positions, pbest_fitness, positions[0], float("inf")
        )
        convergence: list[float] = []
        stagnation = 0
        mutation_events = 0
        mutation_calls = 0

        for iteration in range(self.iterations):
            previous_best = best_value
            weight = self.w_max - (self.w_max - self.w_min) * iteration / max(1, self.iterations - 1)
            r1 = self.rng.random((self.particles, self.dimensions))
            r2 = self.rng.random((self.particles, self.dimensions))
            velocities = (
                weight * velocities
                + self.c1 * r1 * (pbest_positions - positions)
                + self.c2 * r2 * (best_position - positions)
            )
            velocities = np.clip(velocities, -self._span, self._span)
            positions = np.clip(positions + velocities, self.lower, self.upper)
            fitness = self._evaluate(positions)

            improved = fitness < pbest_fitness
            pbest_positions[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]
            best_position, best_value, _ = self._update_global_best(
                pbest_positions, pbest_fitness, best_position, best_value
            )

            if best_value < previous_best:
                stagnation = 0
            else:
                stagnation += 1
            if stagnation >= self._stagnation_threshold(positions):
                mutation_events += self._mutate(positions)
                mutation_calls += 1
                fitness = self._evaluate(positions)
                improved = fitness < pbest_fitness
                pbest_positions[improved] = positions[improved]
                pbest_fitness[improved] = fitness[improved]
                best_position, best_value, _ = self._update_global_best(
                    pbest_positions, pbest_fitness, best_position, best_value
                )
                stagnation = 0
            convergence.append(best_value)

        return OptimizationResult(
            best_position,
            best_value,
            np.asarray(convergence),
            mutation_events,
            mutation_calls,
        )


class DynamicCauchyMutationPSO(CauchyMutationPSO):
    """根据粒子群多样性自适应调整触发阈值的 Cauchy-PSO。

    这是本课程项目中的小改进：粒子已经聚集时提前触发变异，粒子仍然分散时延长等待，
    从而减少不必要的随机跳跃。
    """

    algorithm_name = "Dynamic-Cauchy-PSO"

    def _stagnation_threshold(self, positions: np.ndarray) -> int:
        # 将平均坐标离散程度归一化，使规则不受基准函数数值尺度影响。
        diversity = float(np.mean(np.std(positions, axis=0)) / self._span)
        # 群体多样性较低时，说明可能陷入局部最优，因此缩短等待时间。
        if diversity < 0.03:
            return max(5, self.stagnation_limit // 2)
        # 群体多样性较高时，说明仍在探索，因此延长等待时间。
        if diversity > 0.15:
            return max(self.stagnation_limit, int(round(self.stagnation_limit * 1.5)))
        # 处于中间范围时，使用默认触发阈值。
        return self.stagnation_limit


def _make_optimizer(
    algorithm: str,
    benchmark: Benchmark,
    dimensions: int,
    particles: int,
    iterations: int,
    rng: np.random.Generator,
    stagnation_limit: int,
) -> ParticleSwarmOptimizer:
    common = dict(
        objective=benchmark.objective,
        lower=benchmark.lower,
        upper=benchmark.upper,
        dimensions=dimensions,
        particles=particles,
        iterations=iterations,
        rng=rng,
    )
    if algorithm == "PSO":
        return ParticleSwarmOptimizer(**common)
    if algorithm == "Cauchy-PSO":
        return CauchyMutationPSO(**common, stagnation_limit=stagnation_limit)
    if algorithm == "Dynamic-Cauchy-PSO":
        return DynamicCauchyMutationPSO(**common, stagnation_limit=stagnation_limit)
    raise ValueError(f"未知的算法名称：{algorithm}")


def run_experiments(
    benchmarks: Iterable[Benchmark],
    dimensions: int = 30,
    particles: int = 30,
    iterations: int = 500,
    runs: int = 30,
    seed: int = 2024,
    stagnation_limit: int = 20,
    algorithms: tuple[str, ...] = ("PSO", "Cauchy-PSO", "Dynamic-Cauchy-PSO"),
) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    """使用相同的初始种群，保证不同算法之间公平比较。"""

    if runs < 1:
        raise ValueError("独立运行次数必须为正数")
    rows: list[dict[str, object]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}

    for benchmark_index, benchmark in enumerate(benchmarks):
        # 为每种算法分别保存结果，这样无需重新读取 CSV 就能生成汇总信息。
        values: dict[str, list[float]] = {algorithm: [] for algorithm in algorithms}
        convergence_values: dict[str, list[np.ndarray]] = {algorithm: [] for algorithm in algorithms}
        mutation_counts: dict[str, list[int]] = {algorithm: [] for algorithm in algorithms}
        mutation_call_counts: dict[str, list[int]] = {algorithm: [] for algorithm in algorithms}
        runtimes: dict[str, list[float]] = {algorithm: [] for algorithm in algorithms}
        for run_index in range(runs):
            run_seed = seed + benchmark_index * 100_000 + run_index
            init_rng = np.random.default_rng(run_seed)
            initial_positions = init_rng.uniform(
                benchmark.lower, benchmark.upper, size=(particles, dimensions)
            )
            span = benchmark.upper - benchmark.lower
            initial_velocities = init_rng.uniform(
                -0.2 * span, 0.2 * span, size=(particles, dimensions)
            )
            for algorithm_index, algorithm in enumerate(algorithms):
                optimizer = _make_optimizer(
                    algorithm,
                    benchmark,
                    dimensions,
                    particles,
                    iterations,
                    np.random.default_rng(run_seed + 10_000 + algorithm_index),
                    stagnation_limit,
                )
                started = time.perf_counter()
                result = optimizer.optimize(initial_positions, initial_velocities)
                elapsed = time.perf_counter() - started
                values[algorithm].append(result.best_value)
                convergence_values[algorithm].append(result.convergence)
                mutation_counts[algorithm].append(result.mutation_events)
                mutation_call_counts[algorithm].append(result.mutation_calls)
                runtimes[algorithm].append(elapsed)
                rows.append(
                    {
                        "function": benchmark.name,
                        "algorithm": algorithm,
                        "run": run_index + 1,
                        "best_value": result.best_value,
                        "runtime_seconds": elapsed,
                        "mutation_events": result.mutation_events,
                        "mutation_calls": result.mutation_calls,
                    }
                )

        for algorithm in algorithms:
            values_array = np.asarray(values[algorithm])
            rows.append(
                {
                    "function": benchmark.name,
                    "algorithm": algorithm,
                    "run": "summary",
                    "best_value": float(np.min(values_array)),
                    "mean_value": float(np.mean(values_array)),
                    "std_value": float(np.std(values_array, ddof=1)) if runs > 1 else 0.0,
                    "runtime_seconds": float(np.sum(runtimes[algorithm])),
                    "mean_mutation_events": float(np.mean(mutation_counts[algorithm])),
                    "mean_mutation_calls": float(np.mean(mutation_call_counts[algorithm])),
                }
            )
        curves[benchmark.name] = {
            algorithm: np.mean(np.vstack(convergence_values[algorithm]), axis=0)
            for algorithm in algorithms
        }
    return rows, curves


def run_sensitivity_experiment(
    benchmark: Benchmark,
    dimensions: int,
    particles: int,
    iterations: int,
    runs: int,
    seed: int,
    gamma_values: tuple[float, ...] = (0.1, 0.3, 0.5),
    stagnation_values: tuple[int, ...] = (10, 20, 50),
) -> list[dict[str, object]]:
    """测量 Cauchy 强度和触发时机对一个基准函数的影响。"""

    rows: list[dict[str, object]] = []
    for gamma in gamma_values:
        for stagnation_limit in stagnation_values:
            best_values: list[float] = []
            mutation_calls: list[int] = []
            for run_index in range(runs):
                run_seed = seed + run_index
                init_rng = np.random.default_rng(run_seed)
                positions = init_rng.uniform(
                    benchmark.lower, benchmark.upper, size=(particles, dimensions)
                )
                span = benchmark.upper - benchmark.lower
                velocities = init_rng.uniform(-0.2 * span, 0.2 * span, size=positions.shape)
                optimizer = CauchyMutationPSO(
                    objective=benchmark.objective,
                    lower=benchmark.lower,
                    upper=benchmark.upper,
                    dimensions=dimensions,
                    particles=particles,
                    iterations=iterations,
                    gamma=gamma,
                    stagnation_limit=stagnation_limit,
                    rng=np.random.default_rng(run_seed + 10_000),
                )
                result = optimizer.optimize(positions, velocities)
                best_values.append(result.best_value)
                mutation_calls.append(result.mutation_calls)
            rows.append(
                {
                    "function": benchmark.name,
                    "gamma": gamma,
                    "stagnation_limit": stagnation_limit,
                    "mean_value": float(np.mean(best_values)),
                    "std_value": float(np.std(best_values, ddof=1)) if runs > 1 else 0.0,
                    "mean_mutation_calls": float(np.mean(mutation_calls)),
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "function",
        "algorithm",
        "run",
        "best_value",
        "mean_value",
        "std_value",
        "runtime_seconds",
        "mutation_events",
        "mutation_calls",
        "mean_mutation_events",
        "mean_mutation_calls",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_sensitivity_csv(rows: list[dict[str, object]], path: Path) -> None:
    """将参数敏感性网格以紧凑的 CSV 格式写入文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["function", "gamma", "stagnation_limit", "mean_value", "std_value", "mean_mutation_calls"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    curves: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    summary_rows: list[dict[str, object]] | None = None,
) -> None:
    """写入收敛曲线；如果提供汇总结果，则额外写入柱状图。"""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("使用 --plot 需要安装 matplotlib，请执行 pip install matplotlib") from exc

    # 在创建图表前设置中文字体，保证标题、坐标轴和图例都能正常显示。
    configure_chinese_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, algorithms in curves.items():
        figure, axis = plt.subplots(figsize=(8, 5))
        for algorithm, curve in algorithms.items():
            axis.semilogy(
                np.maximum(curve, 1e-300),
                label=display_algorithm_name(algorithm),
            )
        axis.set_title(f"{name}：平均收敛曲线")
        axis.set_xlabel("迭代次数")
        axis.set_ylabel("当前最优目标值")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{name}.png", dpi=160)
        plt.close(figure)

    # 汇总图让主要对比结果一眼可见，无需打开 CSV 文件。
    if summary_rows:
        functions = sorted({str(row["function"]) for row in summary_rows})
        algorithms = list(dict.fromkeys(str(row["algorithm"]) for row in summary_rows))
        positions = np.arange(len(functions))
        width = 0.8 / max(1, len(algorithms))
        figure, axis = plt.subplots(figsize=(12, 6))
        for index, algorithm in enumerate(algorithms):
            means = []
            for function in functions:
                match = next(
                    row
                    for row in summary_rows
                    if row["function"] == function and row["algorithm"] == algorithm
                )
                means.append(max(float(match["mean_value"]), 1e-300))
            axis.bar(
                positions + index * width,
                means,
                width=width,
                label=display_algorithm_name(algorithm),
            )
        axis.set_yscale("log")
        axis.set_xticks(positions + width * (len(algorithms) - 1) / 2)
        axis.set_xticklabels(functions, rotation=45, ha="right")
        axis.set_ylabel("平均目标值（对数刻度）")
        axis.set_title("各算法平均性能对比")
        axis.grid(axis="y", which="both", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "mean_performance_comparison.png", dpi=160)
        plt.close(figure)


def write_sensitivity_plot(rows: list[dict[str, object]], path: Path) -> None:
    """绘制 gamma 和停滞阈值敏感性热力图。"""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("使用 --plot 需要安装 matplotlib，请执行 pip install matplotlib") from exc

    # 热力图同样使用统一的中文字体配置。
    configure_chinese_font()
    gammas = sorted({float(row["gamma"]) for row in rows})
    limits = sorted({int(row["stagnation_limit"]) for row in rows})
    matrix = np.empty((len(gammas), len(limits)))
    for gamma_index, gamma in enumerate(gammas):
        for limit_index, limit in enumerate(limits):
            match = next(
                row
                for row in rows
                if float(row["gamma"]) == gamma and int(row["stagnation_limit"]) == limit
            )
            matrix[gamma_index, limit_index] = float(match["mean_value"])

    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(np.log10(np.maximum(matrix, 1e-300)), cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(limits)), labels=[str(value) for value in limits])
    axis.set_yticks(np.arange(len(gammas)), labels=[str(value) for value in gammas])
    axis.set_xlabel("停滞触发阈值")
    axis.set_ylabel("Cauchy 变异强度 gamma")
    axis.set_title("Cauchy 参数敏感性：平均目标值的 log10")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2g}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="平均目标值的 log10")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较加入 Cauchy 变异的多个 PSO 版本")
    parser.add_argument("--dimensions", type=int, default=30)
    parser.add_argument("--particles", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--stagnation-limit", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--plot", action="store_true", help="同时写入 PNG 格式的收敛曲线")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="在 Rastrigin 函数上运行 gamma 和停滞阈值敏感性实验",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, curves = run_experiments(
        default_benchmarks(),
        dimensions=args.dimensions,
        particles=args.particles,
        iterations=args.iterations,
        runs=args.runs,
        seed=args.seed,
        stagnation_limit=args.stagnation_limit,
    )
    output_dir = args.output_dir
    write_csv(rows, output_dir / "experiment_results.csv")
    summary_rows = [row for row in rows if row["run"] == "summary"]
    if args.plot:
        write_plots(curves, output_dir / "convergence_plots", summary_rows)

    if args.sensitivity:
        sensitivity_rows = run_sensitivity_experiment(
            default_benchmarks()[3],
            dimensions=args.dimensions,
            particles=args.particles,
            iterations=args.iterations,
            runs=args.runs,
            seed=args.seed,
        )
        write_sensitivity_csv(sensitivity_rows, output_dir / "sensitivity_results.csv")
        if args.plot:
            write_sensitivity_plot(sensitivity_rows, output_dir / "sensitivity_heatmap.png")

    print("函数,算法,最优值,平均值,标准差,平均变异粒子数,平均变异触发次数")
    for row in summary_rows:
        print(
            f"{row['function']},{row['algorithm']},{row['best_value']:.6e},"
            f"{row['mean_value']:.6e},{row['std_value']:.6e},"
            f"{row['mean_mutation_events']:.2f},{row['mean_mutation_calls']:.2f}"
        )
    print(f"\nDetailed results: {output_dir / 'experiment_results.csv'}")
    if args.sensitivity:
        print(f"Sensitivity results: {output_dir / 'sensitivity_results.csv'}")


if __name__ == "__main__":
    main()
