"""场景契约测试：取值范围、非法组合、派生量与哈希稳定性。"""

import json

import pytest
from pydantic import ValidationError

from chargebench.schemas import (
    BENCHMARK_DATE,
    SCHEMA_VERSION,
    ArrivalMode,
    PriceProfile,
    RunResult,
    Scenario,
)

from .conftest import load_scenario

VALID = dict(
    scenario_id="t",
    n_ports=10,
    port_max_power_kw=7.0,
    window_hours=8.0,
    n_sessions=20,
    load_intensity=1.0,
    deadline_tightness=0.5,
)


def test_demo_scenario_is_valid(demo_scenario: Scenario):
    assert demo_scenario.scenario_id
    assert demo_scenario.periods > 0


def test_schema_version_declared(demo_scenario: Scenario):
    """协议版本必须进入结果，UI 与 Agent 才能判断兼容性。"""
    assert SCHEMA_VERSION == "1.0"
    assert BENCHMARK_DATE.weekday() == 0, "基准日必须是周一，否则电价时段会漂移"


def test_unknown_field_rejected():
    """extra=forbid：拼错字段名必须报错，而不是被静默忽略。"""
    with pytest.raises(ValidationError):
        Scenario(**VALID, typo_field=1)


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"network_limit_kw": 3.0}, "网络上限低于单桩功率，任何车都无法额定充电"),
        ({"time_step_min": 0}, "时间步必须为正"),
        ({"deadline_tightness": 0}, "紧迫度必须为正"),
        ({"deadline_tightness": 1.5}, "紧迫度上限为 1"),
        ({"load_intensity": 0}, "负荷强度必须为正"),
        ({"n_ports": 0}, "至少一个车位"),
        ({"start_hour": 24}, "起始小时必须 < 24"),
    ],
)
def test_invalid_combinations_rejected(override, reason):
    with pytest.raises(ValidationError):
        Scenario(**{**VALID, **override})


def test_window_must_hold_at_least_two_steps():
    with pytest.raises(ValidationError, match="窗口过短"):
        Scenario(**{**VALID, "window_hours": 0.05, "time_step_min": 5.0})


def test_price_order_enforced():
    with pytest.raises(ValidationError, match="谷 < 平 < 峰"):
        PriceProfile(valley_cny_per_kwh=0.9, flat_cny_per_kwh=0.5, peak_cny_per_kwh=1.1)


@pytest.mark.parametrize("bad", [(8.0, 8.0), (-1.0, 5.0), (10.0, 25.0)])
def test_price_windows_validated(bad):
    with pytest.raises(ValidationError):
        PriceProfile(valley_hours=[bad])


def test_arrival_mode_enum():
    assert Scenario(**{**VALID, "arrival_mode": "front_loaded"}).arrival_mode is ArrivalMode.front_loaded
    with pytest.raises(ValidationError):
        Scenario(**{**VALID, "arrival_mode": "nonexistent"})


# ----------------------------------------------------------------------
# 派生量
# ----------------------------------------------------------------------


def test_effective_supply_takes_the_tighter_bound():
    no_limit = Scenario(**{**VALID, "network_limit_kw": None})
    assert no_limit.effective_supply_kw == pytest.approx(10 * 7.0)

    limited = Scenario(**{**VALID, "network_limit_kw": 40.0})
    assert limited.effective_supply_kw == pytest.approx(40.0)

    loose = Scenario(**{**VALID, "network_limit_kw": 500.0})
    assert loose.effective_supply_kw == pytest.approx(70.0), "桩总和更紧时应取桩总和"


def test_load_intensity_scales_requested_energy():
    a = Scenario(**{**VALID, "load_intensity": 0.5})
    b = Scenario(**{**VALID, "load_intensity": 1.5})
    assert b.total_requested_kwh == pytest.approx(3 * a.total_requested_kwh)


def test_port_occupancy_is_independent_of_session_count():
    """占用率只由 λ、功率与紧迫度决定，与车辆数无关（PLAN.md 阶段 4 选扫描范围的关键）。"""
    few = Scenario(**{**VALID, "n_sessions": 10})
    many = Scenario(**{**VALID, "n_sessions": 400})
    assert few.port_occupancy_estimate == pytest.approx(many.port_occupancy_estimate)


def test_port_occupancy_predicts_drops():
    """>1 必然丢弃：构造一个占用率远超 1 的场景，验证确有丢弃。"""
    from chargebench.adapter import build_scenario

    oversubscribed = Scenario(**{**VALID, "load_intensity": 3.0, "deadline_tightness": 0.5})
    assert oversubscribed.port_occupancy_estimate > 1.0
    assert build_scenario(oversubscribed).n_dropped > 0


# ----------------------------------------------------------------------
# 哈希与 run_id
# ----------------------------------------------------------------------


def test_scenario_hash_is_stable_and_sensitive():
    a = Scenario(**VALID)
    b = Scenario(**VALID)
    c = Scenario(**{**VALID, "load_intensity": 1.0001})
    assert a.scenario_hash == b.scenario_hash
    assert a.scenario_hash != c.scenario_hash


def test_scenario_hash_ignores_field_order():
    reordered = json.loads(json.dumps(VALID))
    items = list(reordered.items())[::-1]
    assert Scenario(**dict(items)).scenario_hash == Scenario(**VALID).scenario_hash


def test_run_id_is_deterministic_and_sensitive(demo_scenario: Scenario):
    base = RunResult.make_run_id(demo_scenario, "EDF", {}, 42)
    assert base == RunResult.make_run_id(demo_scenario, "EDF", {}, 42)
    assert base != RunResult.make_run_id(demo_scenario, "FCFS", {}, 42)
    assert base != RunResult.make_run_id(demo_scenario, "EDF", {}, 43)
    assert base != RunResult.make_run_id(demo_scenario, "EDF", {"foo": 1}, 42)
    assert base.startswith(f"{demo_scenario.scenario_id}--EDF--")


def test_demo_config_file_matches_validated_schema():
    """configs/demo_scenario.json 必须能被 schema 接受 —— 它是全队共享的样例输入。"""
    assert load_scenario().scenario_id == "campus_baseline_v1"
