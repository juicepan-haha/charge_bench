"""Adapter 测试：场景展开的确定性、车位分配的物理正确性、网络约束的合法性。

这些不变量是公平比较的前提 —— 一旦破坏，所有算法对比都失去意义。
"""

from collections import defaultdict

import numpy as np
import pytest

from chargebench.adapter import (
    build_events,
    build_network,
    build_scenario,
    generate_sessions,
)
from chargebench.schemas import ArrivalMode, Scenario

from .conftest import load_scenario


# ----------------------------------------------------------------------
# 确定性
# ----------------------------------------------------------------------


def test_scenario_expansion_is_deterministic(demo_scenario: Scenario):
    """同一场景两次展开必须逐位相同，否则算法对比不可复现。"""
    a = build_scenario(demo_scenario)
    b = build_scenario(demo_scenario)
    assert a.sessions_generated == b.sessions_generated
    assert a.sessions_scheduled == b.sessions_scheduled
    assert a.n_dropped == b.n_dropped


def test_different_seeds_give_different_sessions(demo_scenario: Scenario):
    a = build_scenario(demo_scenario)
    b = build_scenario(demo_scenario.with_updates(seed=demo_scenario.seed + 1))
    assert [r.requested_energy_kwh for r in a.sessions_generated] != [
        r.requested_energy_kwh for r in b.sessions_generated
    ]


def test_events_are_rebuilt_fresh_each_call(demo_scenario: Scenario):
    """仿真会改写 EV 状态，因此不同算法必须各自拿到干净的队列。"""
    artifacts = build_scenario(demo_scenario)
    first = build_events(artifacts.sessions_scheduled, demo_scenario)
    ev_first = next(iter(first._queue))[1].ev

    # 模拟被前一次仿真污染
    ev_first.charge(pilot=1.0, voltage=demo_scenario.voltage_v, period=demo_scenario.time_step_min)
    assert ev_first.energy_delivered > 0

    second = build_events(artifacts.sessions_scheduled, demo_scenario)
    ev_second = next(iter(second._queue))[1].ev
    assert ev_second.energy_delivered == pytest.approx(0.0), "事件队列未重建，算法会读到脏状态"


# ----------------------------------------------------------------------
# 车位分配
# ----------------------------------------------------------------------


def test_accounting_adds_up(demo_scenario: Scenario):
    artifacts = build_scenario(demo_scenario)
    assert len(artifacts.sessions_scheduled) + artifacts.n_dropped == len(
        artifacts.sessions_generated
    )


def test_no_double_booking(demo_scenario: Scenario):
    """同一车位的占用区间不得重叠 —— 有限车位分配的核心不变量。"""
    artifacts = build_scenario(demo_scenario)
    by_station: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rec in artifacts.sessions_scheduled:
        by_station[rec.station_id].append((rec.arrival_period, rec.departure_period))

    for station, intervals in by_station.items():
        intervals.sort()
        for (a1, d1), (a2, d2) in zip(intervals, intervals[1:]):
            assert a2 >= d1, f"{station} 上 ({a1},{d1}) 与 ({a2},{d2}) 重叠"


def test_station_ids_are_valid_and_bounded(demo_scenario: Scenario):
    artifacts = build_scenario(demo_scenario)
    valid = {f"PS-{i:03d}" for i in range(demo_scenario.n_ports)}
    used = {r.station_id for r in artifacts.sessions_scheduled}
    assert used <= valid, "出现了未注册的车位编号"
    assert used == valid, "应尽可能用满车位"


def test_session_intervals_stay_inside_window(demo_scenario: Scenario):
    artifacts = build_scenario(demo_scenario)
    for rec in artifacts.sessions_scheduled:
        assert 0 <= rec.arrival_period < rec.departure_period
        assert rec.departure_period <= demo_scenario.periods


def test_drops_appear_when_oversubscribed(demo_scenario: Scenario):
    oversubscribed = demo_scenario.with_updates(load_intensity=4.0, deadline_tightness=1.0)
    assert build_scenario(oversubscribed).n_dropped > 0


def test_ports_are_reused_when_turnover_allows(demo_scenario: Scenario):
    """车位应当被多轮复用，而不是一车一位。"""
    artifacts = build_scenario(demo_scenario)
    per_station: dict[str, int] = defaultdict(int)
    for rec in artifacts.sessions_scheduled:
        per_station[rec.station_id] += 1
    assert max(per_station.values()) >= 2, "没有发生车位复用，说明车位分配或场景标定有误"


# ----------------------------------------------------------------------
# 到达分布
# ----------------------------------------------------------------------


def test_arrival_modes_order_the_mean_arrival(demo_scenario: Scenario):
    early = build_scenario(demo_scenario.with_updates(arrival_mode=ArrivalMode.front_loaded))
    late = build_scenario(demo_scenario.with_updates(arrival_mode=ArrivalMode.back_loaded))
    mean_early = np.mean([r.arrival_period for r in early.sessions_generated])
    mean_late = np.mean([r.arrival_period for r in late.sessions_generated])
    assert mean_early < mean_late


# ----------------------------------------------------------------------
# 网络
# ----------------------------------------------------------------------


def test_network_always_has_a_constraint(demo_scenario: Scenario):
    """PLAN.md §6 坑 #9：constraint_matrix 为 None 会让仿真无法启动。"""
    network = build_network(demo_scenario)
    assert network.constraint_matrix is not None
    assert len(network.constraint_index) >= 1


def test_network_limit_is_never_none_in_practice(demo_scenario: Scenario):
    """即使场景不设聚合上限，也要有一条不生效的约束行把矩阵撑起来。"""
    unlimited = demo_scenario.with_updates(network_limit_kw=None)
    network = build_network(unlimited)
    assert network.constraint_matrix is not None
    limit_a = float(network.magnitudes[0])
    port_sum_a = unlimited.n_ports * unlimited.port_max_power_kw * 1000 / unlimited.voltage_v
    assert limit_a == pytest.approx(port_sum_a), "无网络上限时约束应恰好等于桩总和（不生效）"


def test_port_power_converted_to_amps(demo_scenario: Scenario):
    """上层用 kW，ACN-Sim 的 EVSE 用安培 —— 换算必须正确。"""
    network = build_network(demo_scenario)
    expected_a = demo_scenario.port_max_power_kw * 1000 / demo_scenario.voltage_v
    for station in network.station_ids:
        assert network.max_pilot_signals[network.station_ids.index(station)] == pytest.approx(expected_a)


# ----------------------------------------------------------------------
# 仍处于窗口内的场景
# ----------------------------------------------------------------------


def test_stay_clamping_is_counted():
    """停留时长超过窗口时会被截断，必须计数，否则会把截断误读成算法表现。"""
    long_stay = Scenario(
        scenario_id="clamp",
        n_ports=5,
        port_max_power_kw=7.0,
        window_hours=4.0,
        n_sessions=10,
        load_intensity=1.0,
        deadline_tightness=0.1,  # stay = 10 × t_req，远超 4 小时窗口
        seed=1,
    )
    artifacts = build_scenario(long_stay)
    assert artifacts.n_stay_clamped > 0


def test_no_clamping_in_demo_scenario(demo_scenario: Scenario):
    assert build_scenario(demo_scenario).n_stay_clamped == 0
