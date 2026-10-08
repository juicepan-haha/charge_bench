"""指标口径的唯一来源。

**任何地方需要指标数字都必须调用本模块**，不允许在 UI / Agent / 报表里各自重算 ——
否则口径会分叉，排名将随报表不同而改变。

口径定义（改口径必须同步改 docs/PROTOCOL.md）：

需求满足率（能量口径）
    实际交付电量 / 请求电量。分母只含**已排定车位**的会话；
    因无空位被丢弃的会话不计入分母，单列为 sessions_dropped。
    对应 ACN-Sim ``proportion_of_energy_delivered``。

按时完成率（车辆口径）
    剩余需求低于完成阈值的会话占比。阈值是**绝对剩余电量 [kWh]**，
    不是百分比 —— 当成百分比会静默改变排名（PLAN.md §6 坑 #4）。
    对应 ACN-Sim ``proportion_of_demands_met``。

总用电成本
    按分时电价对每步聚合功率计费，单位 **CNY**。
    ACN-Sim 内置电价是美元，故本项目的 tariff 由 tariff.py 提供（PLAN.md §6 坑 #7）。

峰值负荷
    ``aggregate_power`` 的返回值**已经是 kW**，不要再除以 1000（PLAN.md §6 坑 #5）。
"""

from __future__ import annotations

import numpy as np

import acnportal.acnsim.analysis as acnsim_analysis

from .adapter import ScenarioArtifacts
from .schemas import Metrics, PriceProfile
from .tariff import CNTimeOfUseTariff

#: 判定「按时完成」的剩余电量阈值 [kWh]（绝对量，非比例）。
#: 0.1 kWh 相当于「基本充满」。
COMPLETION_THRESHOLD_KWH = 0.1

#: 约束越限判定容差。必须用**相对**容差：ACN-Sim 调度器内部用二分搜索求解功率分配，
#: 结果会带有约 1e-7 相对量级的浮点残差。实测在 272.7 A 的限值上出现 1.99e-05 A 的
#: 超出（相对 7.3e-8），这是数值噪声而非违规。若用绝对容差 1e-6 A 判定，
#: 会把完全正常的运行误判为违规 —— 而扫描的评分规则又用「违规即取消资格」，
#: 于是一个浮点残差就能静默翻转整张热力图的胜者。故取相对容差为主、绝对下限兜底。
VIOLATION_RELATIVE_TOLERANCE = 1e-4  # 相对限值的 0.01%
VIOLATION_ABSOLUTE_FLOOR_A = 1e-3  # 限值极小时不至于退化为零容差


def violation_threshold_a(limit_a: float) -> float:
    """给定约束限值，返回判定为「越限」所需的最小超出量 [A]。"""
    return max(VIOLATION_ABSOLUTE_FLOOR_A, VIOLATION_RELATIVE_TOLERANCE * abs(limit_a))


def compute_metrics(
    sim,
    artifacts: ScenarioArtifacts,
    runtime_s: float,
    price_profile: PriceProfile | None = None,
    completion_threshold_kwh: float = COMPLETION_THRESHOLD_KWH,
) -> Metrics:
    """从一次已完成的仿真中提取标准化指标。"""
    profile = price_profile or artifacts.scenario.price_profile
    history = sim.ev_history

    if not history:
        raise ValueError(
            "仿真没有任何车辆会话进入 ev_history，无法计算指标。"
            f"（生成 {len(artifacts.sessions_generated)} 个会话，"
            f"丢弃 {artifacts.n_dropped} 个，排定 {len(artifacts.sessions_scheduled)} 个）"
        )

    energy_requested = acnsim_analysis.total_energy_requested(sim)
    energy_delivered = acnsim_analysis.total_energy_delivered(sim)

    # --- 单车交付比：仅在已排定会话上统计 ---
    # 被丢弃的会话从未进入仿真，把它们计为 0 会让 worst_delivery_ratio 恒为 0，
    # 失去观察稳健性的作用。场景可行性问题由 sessions_dropped 单独承担。
    ratios = np.array(
        [
            min(ev.energy_delivered / ev.requested_energy, 1.0)
            for ev in history.values()
            if ev.requested_energy > 0
        ]
    )

    # --- 约束越限 ---
    violations, max_violation_a = _constraint_violations(sim)

    # --- 成本 ---
    tariff = CNTimeOfUseTariff(profile)
    cost = float(acnsim_analysis.energy_cost(sim, tariff))

    # --- 峰值负荷（aggregate_power 已是 kW） ---
    aggregate = acnsim_analysis.aggregate_power(sim)
    peak_kw = float(np.max(aggregate)) if len(aggregate) else 0.0

    return Metrics(
        demand_satisfaction_rate=float(
            acnsim_analysis.proportion_of_energy_delivered(sim)
        ),
        energy_delivered_kwh=float(energy_delivered),
        energy_requested_kwh=float(energy_requested),
        session_completion_rate=float(
            acnsim_analysis.proportion_of_demands_met(
                sim, threshold=completion_threshold_kwh
            )
        ),
        mean_delivery_ratio=float(np.mean(ratios)),
        worst_delivery_ratio=float(np.min(ratios)),
        energy_cost_cny=cost,
        peak_power_kw=peak_kw,
        constraint_violations=violations,
        max_violation_a=max_violation_a,
        sessions_generated=len(artifacts.sessions_generated),
        sessions_scheduled=len(artifacts.sessions_scheduled),
        sessions_dropped=artifacts.n_dropped,
        runtime_s=float(runtime_s),
    )


def _constraint_violations(sim) -> tuple[int, float]:
    """统计越限的时间步数与最大越限幅度。

    ``constraint_currents`` 返回 {约束名: 数组} 的字典（不是单个数组），
    约束限值按 network.constraint_index 的顺序取自 network.magnitudes。

    只有超出量大于 ``violation_threshold_a(limit)`` 的时间步才计入违规；
    容差的来由见该函数的文档串（调度器浮点残差不应被当成违规）。
    """
    currents = acnsim_analysis.constraint_currents(sim)
    if not currents:
        return 0, 0.0

    limits = list(sim.network.magnitudes)
    index = list(sim.network.constraint_index)

    total_steps = 0
    max_over = 0.0
    for name, series in currents.items():
        limit = float(limits[index.index(name)])
        overshoot = np.asarray(series, dtype=float) - limit
        # 超限幅度本身也过滤：只把真正越限的部分算作违规幅度
        over_real = overshoot[overshoot > violation_threshold_a(limit)]
        total_steps += int(over_real.size)
        if over_real.size:
            max_over = max(max_over, float(np.max(over_real)))
    return total_steps, max_over
