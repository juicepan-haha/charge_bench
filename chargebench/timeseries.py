"""时序曲线提取，供 UI 的过程回放与负荷曲线绘制使用。

**为什么不把曲线塞进 RunResult**：一次扫描有上百次运行，把每个运行的
(站点 × 时间步) 矩阵都持久化会让结果库膨胀几十倍，而时序只在「看某一次运行的过程」
时才需要。因此曲线按需重算 —— 场景与种子都是确定性的，重算必然得到同一条曲线，
不存在"存下来才对得上"的问题。

已知边界：仿真的终点是**最后一个拔枪事件**，不等于 `window_hours`
（见 docs/PROTOCOL.md §9）。因此曲线长度通常略短于 `scenario.periods`。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np

import acnportal.acnsim.analysis as acnportal_analysis

from .adapter import build_scenario, build_simulator
from .algorithms import build_algorithm
from .schemas import BENCHMARK_DATE, Scenario
from .tariff import CNTimeOfUseTariff, simulation_start


@dataclass(frozen=True)
class LoadCurves:
    """一次运行的时序结果。序列长度一致，可直接逐点对齐。"""

    scenario_id: str
    algorithm: str
    seed: int
    start: dt.datetime
    period_min: float
    #: 相对窗口起点的时刻 [h]，长度 = n_periods
    hours: list[float]
    #: 聚合充电负荷 [kW]
    aggregate_kw: list[float]
    #: 每步电价 [CNY/kWh]
    price_cny_per_kwh: list[float]
    #: 每步所处时段：valley / flat / peak
    segments: list[str]
    #: 站点聚合功率上限 [kW]，用于在图上标出瓶颈
    limit_kw: float
    #: 每步仍在充电的车辆数
    active_evs: list[int]
    #: 站点 ID → 各步功率 [kW]（过程回放用；站点多时可不必绘制）
    per_station_kw: dict[str, list[float]]
    #: 每步电价 × 负荷累加得到的累计电费 [CNY]
    cumulative_cost_cny: list[float]

    @property
    def n_periods(self) -> int:
        return len(self.hours)

    @property
    def duration_hours(self) -> float:
        return self.hours[-1] if self.hours else 0.0


def simulate_curves(
    scenario: Scenario,
    algorithm: str,
    seed: int | None = None,
    algorithm_params: dict[str, Any] | None = None,
    include_per_station: bool = True,
) -> LoadCurves:
    """重跑一次仿真并提取时序曲线。

    与 ``run_experiment`` 走完全相同的构造路径，因此指标与曲线必然来自同一次物理过程；
    只是这里额外把时间维度的数据带出来。
    """
    effective_seed = scenario.seed if seed is None else seed
    if effective_seed != scenario.seed:
        scenario = scenario.with_updates(seed=effective_seed)

    start = simulation_start(scenario.start_hour, BENCHMARK_DATE)
    artifacts = build_scenario(scenario)
    simulator = build_simulator(artifacts, build_algorithm(algorithm, algorithm_params or {}), start)
    simulator.run()

    aggregate = np.asarray(
        acnportal_analysis.aggregate_power(simulator), dtype=float
    )
    n = len(aggregate)
    if n == 0:
        raise ValueError(f"算法 {algorithm} 的仿真没有任何时间步，无法提取曲线")

    tariff = CNTimeOfUseTariff(scenario.price_profile)
    prices = tariff.get_tariffs(start, n, scenario.time_step_min)
    segments = tariff.segments(start, n, scenario.time_step_min)

    hours = [i * scenario.time_step_min / 60.0 for i in range(n)]
    hours_per_step = scenario.time_step_min / 60.0
    energy_kwh = aggregate * hours_per_step
    cumulative = np.cumsum(energy_kwh * np.asarray(prices, dtype=float))

    rates = np.asarray(simulator.charging_rates, dtype=float)
    active = (
        [int(np.count_nonzero(rates[:, i] > 1e-9)) for i in range(n)]
        if rates.size
        else [0] * n
    )

    per_station: dict[str, list[float]] = {}
    if include_per_station and rates.size:
        voltages = np.asarray(simulator.network._voltages, dtype=float)
        for idx, station in enumerate(simulator.network.station_ids):
            per_station[station] = list(rates[idx, :] * voltages[idx] / 1000.0)

    return LoadCurves(
        scenario_id=scenario.scenario_id,
        algorithm=algorithm,
        seed=effective_seed,
        start=start,
        period_min=scenario.time_step_min,
        hours=hours,
        aggregate_kw=list(aggregate),
        price_cny_per_kwh=[float(p) for p in prices],
        segments=list(segments),
        limit_kw=scenario.effective_supply_kw,
        active_evs=active,
        per_station_kw=per_station,
        cumulative_cost_cny=[float(c) for c in cumulative],
    )


def summarize(curves: LoadCurves) -> dict[str, float]:
    """曲线的几个概括量，便于在 UI 上做小卡片。"""
    aggregate = np.asarray(curves.aggregate_kw, dtype=float)
    limit = curves.limit_kw or 1.0
    return {
        "peak_kw": float(aggregate.max()),
        "mean_kw": float(aggregate.mean()),
        "utilization": float(aggregate.mean() / limit),
        "energy_kwh": float(aggregate.sum() * curves.period_min / 60.0),
        "cost_cny": float(curves.cumulative_cost_cny[-1]) if curves.cumulative_cost_cny else 0.0,
        "duration_hours": curves.duration_hours,
    }
