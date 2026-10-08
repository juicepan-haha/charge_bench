"""时序曲线测试。

曲线是「过程回放」的数据来源。它必须与同一配置的指标结果一致 ——
否则 UI 上会同时出现互相矛盾的两个数字。
"""

import pytest

from chargebench.algorithms import CORE_ALGORITHMS
from chargebench.experiments import run_experiment
from chargebench.metrics import COMPLETION_THRESHOLD_KWH
from chargebench.schemas import Scenario
from chargebench.timeseries import LoadCurves, simulate_curves, summarize

from .conftest import load_scenario


@pytest.fixture(scope="module")
def curves() -> LoadCurves:
    return simulate_curves(load_scenario(), "EDF")


# ----------------------------------------------------------------------
# 结构
# ----------------------------------------------------------------------


def test_series_lengths_align(curves: LoadCurves):
    n = curves.n_periods
    assert n > 0
    assert len(curves.aggregate_kw) == n
    assert len(curves.price_cny_per_kwh) == n
    assert len(curves.segments) == n
    assert len(curves.active_evs) == n
    assert len(curves.cumulative_cost_cny) == n
    for station_series in curves.per_station_kw.values():
        assert len(station_series) == n


def test_per_station_covers_all_stations(curves: LoadCurves):
    assert len(curves.per_station_kw) == load_scenario().n_ports
    assert set(curves.per_station_kw) == {f"PS-{i:03d}" for i in range(load_scenario().n_ports)}


def test_hours_are_monotonic_with_correct_step(curves: LoadCurves):
    step = load_scenario().time_step_min / 60.0
    assert curves.hours[0] == pytest.approx(0.0)
    for a, b in zip(curves.hours, curves.hours[1:]):
        assert b - a == pytest.approx(step)


def test_per_station_sums_to_aggregate(curves: LoadCurves):
    """各站点功率之和必须等于聚合负荷，否则 UI 上的两张图会互相矛盾。"""
    for i in range(curves.n_periods):
        total = sum(series[i] for series in curves.per_station_kw.values())
        assert total == pytest.approx(curves.aggregate_kw[i], abs=1e-6)


def test_include_per_station_can_be_disabled():
    lean = simulate_curves(load_scenario(), "EDF", include_per_station=False)
    assert lean.per_station_kw == {}
    assert lean.n_periods > 0


# ----------------------------------------------------------------------
# 与指标结果一致
# ----------------------------------------------------------------------


def test_energy_matches_metrics(curves: LoadCurves):
    """曲线积分出的电量必须与同配置的指标一致。"""
    result = run_experiment(load_scenario(), "EDF")
    assert summarize(curves)["energy_kwh"] == pytest.approx(
        result.metrics.energy_delivered_kwh, rel=1e-9
    )


def test_cost_matches_metrics(curves: LoadCurves):
    result = run_experiment(load_scenario(), "EDF")
    assert summarize(curves)["cost_cny"] == pytest.approx(
        result.metrics.energy_cost_cny, rel=1e-9
    )


def test_peak_matches_metrics(curves: LoadCurves):
    result = run_experiment(load_scenario(), "EDF")
    assert summarize(curves)["peak_kw"] == pytest.approx(result.metrics.peak_power_kw, rel=1e-9)


def test_cumulative_cost_is_monotonic(curves: LoadCurves):
    for a, b in zip(curves.cumulative_cost_cny, curves.cumulative_cost_cny[1:]):
        assert b >= a - 1e-9


def test_peak_respects_limit(curves: LoadCurves):
    assert max(curves.aggregate_kw) <= curves.limit_kw * (1 + 1e-9)


def test_segments_are_known_values(curves: LoadCurves):
    assert set(curves.segments) <= {"valley", "flat", "peak"}


def test_duration_is_shorter_than_window(curves: LoadCurves):
    """仿真终点是最后一个拔枪事件，因此通常略短于窗口（PROTOCOL §7 已知简化）。"""
    assert 0 < curves.duration_hours <= load_scenario().window_hours


# ----------------------------------------------------------------------
# 可复现性
# ----------------------------------------------------------------------


def test_curves_are_reproducible(curves: LoadCurves):
    again = simulate_curves(load_scenario(), "EDF")
    assert again.aggregate_kw == curves.aggregate_kw
    assert again.cumulative_cost_cny == curves.cumulative_cost_cny


def test_seed_changes_curves(curves: LoadCurves):
    other = simulate_curves(load_scenario(), "EDF", seed=43)
    assert other.aggregate_kw != curves.aggregate_kw
    assert other.seed == 43


def test_seed_override_does_not_mutate_scenario(demo_scenario: Scenario):
    simulate_curves(demo_scenario, "EDF", seed=999)
    assert demo_scenario.seed == 42


def test_all_algorithms_produce_curves():
    for name in CORE_ALGORITHMS:
        curves = simulate_curves(load_scenario(), name)
        assert curves.algorithm == name
        assert curves.n_periods > 0


# ----------------------------------------------------------------------
# 概括量
# ----------------------------------------------------------------------


def test_summary_fields(curves: LoadCurves):
    summary = summarize(curves)
    assert set(summary) == {
        "peak_kw", "mean_kw", "utilization", "energy_kwh", "cost_cny", "duration_hours"
    }
    assert 0 < summary["utilization"] <= 1 + 1e-9, "利用率应在 (0,1]，超过 1 说明越限"


def test_active_evs_within_port_count(curves: LoadCurves):
    assert all(0 <= n <= load_scenario().n_ports for n in curves.active_evs)
    assert max(curves.active_evs) > 0
