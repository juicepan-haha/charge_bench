"""指标口径测试。

这些断言把 PROTOCOL.md 的口径定义固定下来 —— 口径漂移会让排名随报表而变，
是本项目最容易失去可信度的地方。
"""

import numpy as np
import pytest

from chargebench.adapter import build_scenario
from chargebench.algorithms import CORE_ALGORITHMS
from chargebench.experiments import run_experiment
from chargebench.metrics import COMPLETION_THRESHOLD_KWH
from chargebench.schemas import Scenario

from .conftest import load_scenario


@pytest.fixture(scope="module")
def demo_result():
    return run_experiment(load_scenario(), "EDF")


# ----------------------------------------------------------------------
# 台账一致性
# ----------------------------------------------------------------------


def test_energy_ledger_is_consistent(demo_result):
    m = demo_result.metrics
    assert 0.0 <= m.energy_delivered_kwh <= m.energy_requested_kwh * (1 + 1e-9)
    assert m.demand_satisfaction_rate == pytest.approx(
        m.energy_delivered_kwh / m.energy_requested_kwh, rel=1e-9
    )


def test_requested_energy_matches_scheduled_sessions(demo_result):
    """请求电量必须等于已排定会话的请求量之和 —— 被丢弃的车不计入分母。"""
    artifacts = build_scenario(load_scenario())
    expected = sum(r.requested_energy_kwh for r in artifacts.sessions_scheduled)
    assert demo_result.metrics.energy_requested_kwh == pytest.approx(expected, rel=1e-9)


def test_delivery_ratios_bounded(demo_result):
    m = demo_result.metrics
    assert 0.0 <= m.worst_delivery_ratio <= m.mean_delivery_ratio <= 1.0 + 1e-9


def test_session_counters(demo_result):
    m = demo_result.metrics
    assert m.sessions_generated == m.sessions_scheduled + m.sessions_dropped
    assert m.sessions_scheduled > 0


# ----------------------------------------------------------------------
# 电网指标
# ----------------------------------------------------------------------


def test_peak_never_exceeds_network_limit(demo_result):
    """峰值负荷单位是 kW。若误除以 1000 会得到 0.06 这种量级，此处即会暴露。"""
    limit_kw = load_scenario().effective_supply_kw
    assert 0 < demo_result.metrics.peak_power_kw <= limit_kw * (1 + 1e-9)


def test_no_constraint_violations_in_feasible_scenario(demo_result):
    assert demo_result.metrics.constraint_violations == 0
    assert demo_result.metrics.max_violation_a == pytest.approx(0.0)


def test_peak_reaches_limit_when_oversubscribed(demo_result):
    """本演示场景刻意标定为约束生效，峰值应当贴住上限。"""
    assert demo_result.metrics.peak_power_kw == pytest.approx(
        load_scenario().effective_supply_kw, rel=1e-6
    )


# ----------------------------------------------------------------------
# 成本
# ----------------------------------------------------------------------


def test_cost_is_in_cny_range(demo_result):
    """成本必须落在人民币电价的合理区间内 —— 用美元电价会得到约 1/7 的数值。"""
    profile = load_scenario().price_profile
    delivered = demo_result.metrics.energy_delivered_kwh
    cost = demo_result.metrics.energy_cost_cny
    assert profile.valley_cny_per_kwh * delivered <= cost <= profile.peak_cny_per_kwh * delivered


def test_cost_tracks_price_profile(demo_scenario: Scenario):
    """全谷段价格下的成本应当等于谷价 × 电量。"""
    flat_valley = demo_scenario.with_updates(
        price_profile={"kind": "flat", "flat_cny_per_kwh": 0.5}
    )
    result = run_experiment(flat_valley, "EDF")
    expected = 0.5 * result.metrics.energy_delivered_kwh
    assert result.metrics.energy_cost_cny == pytest.approx(expected, rel=1e-6)


# ----------------------------------------------------------------------
# 完成率阈值的口径
# ----------------------------------------------------------------------


def test_completion_threshold_is_absolute_energy_not_percentage(demo_scenario: Scenario):
    """阈值是绝对剩余电量 [kWh]。放宽阈值只会让更多会话被判为完成（单调不减）。"""
    from chargebench.adapter import build_simulator
    from chargebench.algorithms import build_algorithm
    from chargebench.metrics import compute_metrics
    from chargebench.tariff import simulation_start
    from chargebench.schemas import BENCHMARK_DATE

    artifacts = build_scenario(demo_scenario)
    sim = build_simulator(
        artifacts,
        build_algorithm("EDF", {}),
        simulation_start(demo_scenario.start_hour, BENCHMARK_DATE),
    )
    sim.run()

    strict = compute_metrics(sim, artifacts, 0.0, demo_scenario.price_profile, 0.1)
    loose = compute_metrics(sim, artifacts, 0.0, demo_scenario.price_profile, 20.0)
    assert loose.session_completion_rate >= strict.session_completion_rate
    assert loose.session_completion_rate == pytest.approx(1.0), "阈值放宽到 20 kWh 应全部算完成"
    assert COMPLETION_THRESHOLD_KWH == 0.1


def test_completion_rate_is_independent_of_delivery_volume(demo_scenario: Scenario):
    """两个口径确实在测不同东西：总电量几乎相同，但车辆口径差别显著。

    这正是「约束成为瓶颈时不能只报总电量」的实证（PLAN.md §3）。
    """
    results = {name: run_experiment(demo_scenario, name) for name in CORE_ALGORITHMS}
    delivered = [r.metrics.energy_delivered_kwh for r in results.values()]
    completion = [r.metrics.session_completion_rate for r in results.values()]

    assert (max(delivered) - min(delivered)) / np.mean(delivered) < 0.05, (
        "总交付电量在约束瓶颈下应当几乎相同"
    )
    assert max(completion) - min(completion) > 0.2, (
        "车辆口径必须能区分出算法的分配差异，否则指标设计失败"
    )
