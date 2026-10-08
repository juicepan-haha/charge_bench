"""两变量参数扫描：找出「哪个算法在什么条件下胜出」。

方案 §5 的核心要求：不只给算法排一次名，而是通过改变场景参数观察适用边界。

设计要点：

1. **评分规则事前写定，不得事后更改**（方案 §5 公平性原则 4）。规则以文本形式
   存进 SweepResult.score_rule，与结果一起持久化，事后无法悄悄改口径。

2. **硬约束优先于任何优化目标**（方案 §5 原则 3）。任何种子下出现约束违规的算法
   在该格点直接取消资格，不参与排名 —— 不允许把「违规供电」当成更好的结果。

3. **一律跨种子聚合**（方案 §5 原则 2）。单次运行会被随机性掩盖差异，扫描的每个
   格点都对每个算法跑全部种子，取均值并记录波动范围。

4. **车位可行性单独标注**。`port_occupancy_estimate > 1` 的格点会产生大量丢弃，
   满足率的分母随之缩水，属于场景可行性问题（方案 §5 原则 5），必须与算法表现区分开。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Iterable, Sequence

import numpy as np

from .experiments import run_experiment
from .schemas import (
    METRIC_DIRECTIONS,
    SCHEMA_VERSION,
    MetricAggregate,
    RunResult,
    Scenario,
    SweepCell,
    SweepResult,
)
from .storage import ResultStore

#: 扫描轴名 → 对应的 Scenario 字段名。目前两轴与字段同名，保留映射是为了将来扩展。
X_FIELD = "load_intensity"
Y_FIELD = "deadline_tightness"


def _fmt(value: float) -> str:
    """把浮点参数压成短而稳定的字符串，用于拼 scenario_id。"""
    return f"{value:g}".replace(".", "p")


def grid_scenario_id(base_id: str, x: float, y: float) -> str:
    return f"{base_id}_L{_fmt(x)}_T{_fmt(y)}"


def grid_scenarios(
    base: Scenario, x_values: Sequence[float], y_values: Sequence[float]
) -> list[Scenario]:
    """展开扫描网格。id 确定性生成，因此同一网格点的 run_id 可跨机比对。"""
    out: list[Scenario] = []
    for y in y_values:
        for x in x_values:
            out.append(
                base.with_updates(
                    scenario_id=grid_scenario_id(base.scenario_id, x, y),
                    **{X_FIELD: x, Y_FIELD: y},
                )
            )
    return out


def _aggregate(runs: list[RunResult], primary_metric: str) -> MetricAggregate:
    primary = [getattr(r.metrics, primary_metric) for r in runs]
    return MetricAggregate(
        n_seeds=len(runs),
        demand_satisfaction_rate_mean=float(
            np.mean([r.metrics.demand_satisfaction_rate for r in runs])
        ),
        session_completion_rate_mean=float(
            np.mean([r.metrics.session_completion_rate for r in runs])
        ),
        energy_cost_cny_mean=float(
            np.mean([r.metrics.energy_cost_cny for r in runs])
        ),
        peak_power_kw_mean=float(np.mean([r.metrics.peak_power_kw for r in runs])),
        mean_delivery_ratio_mean=float(
            np.mean([r.metrics.mean_delivery_ratio for r in runs])
        ),
        worst_delivery_ratio_mean=float(
            np.mean([r.metrics.worst_delivery_ratio for r in runs])
        ),
        constraint_violations_max=int(
            max(r.metrics.constraint_violations for r in runs)
        ),
        spread=float(max(primary) - min(primary)),
    )


def _pick_winner(
    aggregates: dict[str, MetricAggregate], primary_metric: str
) -> tuple[str | None, list[str]]:
    """按写定的评分规则选出胜者。

    返回 (胜者, 被取消资格的算法)。规则见 ``build_score_rule``。
    """
    disqualified = sorted(
        name for name, agg in aggregates.items() if agg.constraint_violations_max > 0
    )
    eligible = {k: v for k, v in aggregates.items() if k not in disqualified}
    if not eligible:
        return None, disqualified

    if primary_metric not in METRIC_DIRECTIONS:
        raise KeyError(
            f"primary_metric 必须是 {sorted(METRIC_DIRECTIONS)} 之一，收到 {primary_metric!r}"
        )
    direction = METRIC_DIRECTIONS[primary_metric]

    def sort_key(item: tuple[str, MetricAggregate]):
        name, agg = item
        mean = getattr(agg, f"{primary_metric}_mean")
        # 主目标优先，其次波动更小者（更稳健），最后按名字保证确定性
        primary = -mean if direction == "max" else mean
        return (primary, agg.spread, name)

    ordered = sorted(eligible.items(), key=sort_key)
    return ordered[0][0], disqualified


def build_score_rule(primary_metric: str) -> str:
    """生成人类可读的评分规则文本。事前写定，随结果一起持久化。"""
    direction = METRIC_DIRECTIONS[primary_metric]
    better = "越大越好" if direction == "max" else "越小越好"
    return (
        f"1) 硬约束优先：任何种子下 constraint_violations > 0 的算法在该格点取消资格"
        f"（方案 §5 原则 3，不允许把违规供电当成更优结果）；"
        f"2) 主目标：{primary_metric} 的跨种子均值，方向为{better}；"
        f"3) 均值相同时，取跨种子波动（spread）更小者 —— 均值相同但波动大的算法不可信；"
        f"4) 仍相同则按算法名升序取首个，保证确定性。"
        f"越限判定采用相对容差 max(1e-3 A, 限值×1e-4)，用于吸收调度器二分搜索的浮点残差，"
        f"详见 metrics.violation_threshold_a。"
    )


def run_sweep(
    base: Scenario,
    x_values: Sequence[float],
    y_values: Sequence[float],
    algorithms: Iterable[str],
    seeds: Sequence[int],
    primary_metric: str = "session_completion_rate",
    store: ResultStore | None = None,
) -> SweepResult:
    """对 (X, Y) 两个变量的网格逐点评测全部算法，选出每格的胜者。

    Args:
        base: 基准场景。网格点由它的 ``load_intensity`` / ``deadline_tightness`` 替换而来。
        primary_metric: 排名依据，取值见 ``schemas.METRIC_DIRECTIONS``。
        store: 给定时复用已算结果并把新结果落盘。**强烈建议提供** —— 扫描的仿真次数是
            网格点数 × 算法数 × 种子数，缓存能把重复扫描的成本降到接近零。
    """
    if primary_metric not in METRIC_DIRECTIONS:
        raise KeyError(
            f"primary_metric 必须是 {sorted(METRIC_DIRECTIONS)} 之一，收到 {primary_metric!r}"
        )
    if not seeds:
        raise ValueError("扫描必须给定至少一个种子，否则无法区分算法差异与随机噪声")

    algorithm_list = list(algorithms)
    seed_list = list(seeds)
    cells: list[SweepCell] = []
    all_run_ids: list[str] = []

    for y in y_values:
        for x in x_values:
            scenario = base.with_updates(
                scenario_id=grid_scenario_id(base.scenario_id, x, y),
                **{X_FIELD: x, Y_FIELD: y},
            )
            by_algorithm: dict[str, list[RunResult]] = {}
            for algorithm in algorithm_list:
                runs = []
                for seed in seed_list:
                    cached = (
                        store.find_existing(scenario.scenario_hash, algorithm, seed)
                        if store is not None
                        else None
                    )
                    runs.append(
                        cached
                        if cached is not None
                        else run_experiment(scenario, algorithm, seed=seed)
                    )
                by_algorithm[algorithm] = runs
                all_run_ids.extend(r.run_id for r in runs)

            if store is not None:
                for runs in by_algorithm.values():
                    for run in runs:
                        store.save_run(run)

            aggregates = {
                name: _aggregate(runs, primary_metric)
                for name, runs in by_algorithm.items()
            }
            winner, disqualified = _pick_winner(aggregates, primary_metric)

            dropped = int(
                np.mean(
                    [
                        r.metrics.sessions_dropped
                        for runs in by_algorithm.values()
                        for r in runs
                    ]
                ).round()
            )
            cells.append(
                SweepCell(
                    load_intensity=x,
                    deadline_tightness=y,
                    port_occupancy_estimate=scenario.port_occupancy_estimate,
                    port_constrained=scenario.port_occupancy_estimate > 1.0,
                    sessions_dropped=dropped,
                    aggregates=aggregates,
                    winner=winner,
                    disqualified=disqualified,
                )
            )

    payload = json.dumps(
        {
            "base": base.scenario_hash,
            "x": list(x_values),
            "y": list(y_values),
            "algorithms": sorted(algorithm_list),
            "seeds": sorted(seed_list),
            "primary_metric": primary_metric,
            "schema_version": SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    return SweepResult(
        sweep_id="sweep--" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10],
        created_at=dt.datetime.now(),
        base_scenario_id=base.scenario_id,
        x_name=X_FIELD,
        x_values=list(x_values),
        y_name=Y_FIELD,
        y_values=list(y_values),
        algorithms=algorithm_list,
        seeds=seed_list,
        primary_metric=primary_metric,
        metric_direction=METRIC_DIRECTIONS[primary_metric],
        score_rule=build_score_rule(primary_metric),
        cells=cells,
        run_ids=sorted(set(all_run_ids)),
    )


def rerank_sweep(sweep: SweepResult, primary_metric: str) -> SweepResult:
    """换一个优化目标重新排名，**不重跑任何仿真**。

    因为 ``MetricAggregate`` 已经存下了全部指标的跨种子均值，而一次实验的
    物理结果与用哪个指标排名无关 —— 「换目标 → 换胜者」这件事本身不需要新数据。
    这既省算力，也让论点更锋利：同一批 run_id，只是换了把尺子。
    """
    if primary_metric not in METRIC_DIRECTIONS:
        raise KeyError(
            f"primary_metric 必须是 {sorted(METRIC_DIRECTIONS)} 之一，收到 {primary_metric!r}"
        )
    if primary_metric == sweep.primary_metric:
        return sweep

    cells: list[SweepCell] = []
    for cell in sweep.cells:
        winner, disqualified = _pick_winner(cell.aggregates, primary_metric)
        cells.append(cell.model_copy(update={"winner": winner, "disqualified": disqualified}))

    payload = json.dumps(
        {
            "reused": sweep.sweep_id,
            "primary_metric": primary_metric,
            "schema_version": SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sweep.model_copy(
        update={
            "sweep_id": "sweep--" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10],
            "primary_metric": primary_metric,
            "metric_direction": METRIC_DIRECTIONS[primary_metric],
            "score_rule": build_score_rule(primary_metric),
            "cells": cells,
        }
    )


# ----------------------------------------------------------------------
# 终端展示
# ----------------------------------------------------------------------


def format_winner_grid(sweep: SweepResult) -> str:
    """把胜者矩阵打成 ASCII 表，终端里直接能看结论。"""
    winners, constrained = sweep.winner_grid()

    lines = [
        f"算法适用区域图  （主目标 {sweep.primary_metric}，{sweep.metric_direction}）",
        f"X = {sweep.x_name}   Y = {sweep.y_name}",
        "",
        " " * 14 + "".join(f"{x:>12g}" for x in sweep.x_values),
    ]
    for y, row, crow in zip(sweep.y_values, winners, constrained):
        cells = []
        for name, is_constrained in zip(row, crow):
            label = name or "-"
            if is_constrained:
                label += "*"
            cells.append(f"{label:>12}")
        lines.append(f"{y:>12g}  " + "".join(cells))

    lines.append("")
    lines.append("  * = 车位占用率 > 1，该格点丢弃显著，满足率结论需谨慎")
    return "\n".join(lines)


def format_sweep_detail(sweep: SweepResult) -> str:
    """逐格点的详细指标，供报告与排查使用。"""
    lines = [
        f"{'L':>6} {'T':>6} {'occ':>6} {'drop':>5}  "
        + "".join(f"{a:>22}" for a in sweep.algorithms),
        "  " + "主目标均值(波动)".rjust(22) * len(sweep.algorithms),
        "-" * (28 + 22 * len(sweep.algorithms)),
    ]
    for y in sweep.y_values:
        for x in sweep.x_values:
            cell = sweep.cell(x, y)
            parts = []
            for a in sweep.algorithms:
                agg = cell.aggregates.get(a)
                if agg is None:
                    parts.append(f"{'-':>22}")
                    continue
                value = getattr(agg, f"{sweep.primary_metric}_mean")
                mark = "!" if a in cell.disqualified else ("*" if a == cell.winner else " ")
                parts.append(f"{value:>15.3f}({agg.spread:.3f}){mark}")
            lines.append(
                f"{x:>6g} {y:>6g} {cell.port_occupancy_estimate:>6.2f} "
                f"{cell.sessions_dropped:>5d}  " + "".join(parts)
            )
    lines.append("")
    lines.append("  标记：* = 该格点胜者   ! = 因约束违规取消资格")
    return "\n".join(lines)
