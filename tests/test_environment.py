"""阶段 0 验收：环境与单位契约的可执行断言。

本文件把 PLAN.md §1 与 §2 里两个"不会报错、只会静默给出错误数字"的坑变成会失败的测试。
任何人改动依赖或适配层后，这些测试应当立刻报警。
"""

import datetime as dt

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. 依赖契约：setuptools<81
# ---------------------------------------------------------------------------


def test_pkg_resources_available():
    """acnportal/acnsim/base.py:17 依赖 pkg_resources，setuptools 81+ 已移除。

    这条如果失败，说明环境里的 setuptools 版本过高 —— 见 PLAN.md §1 与 pyproject.toml。
    """
    import pkg_resources  # noqa: F401


def test_acnportal_imports():
    """仿真后端必须可导入且核心类齐全。"""
    import acnportal.acnsim as acnsim

    for name in ("Simulator", "ChargingNetwork", "EVSE", "EV", "Battery"):
        assert hasattr(acnsim, name), f"acnsim 缺少 {name}"


# ---------------------------------------------------------------------------
# 2. 单位契约：Battery 是 kWh/kW，不是教程里的 A*periods
# ---------------------------------------------------------------------------


def test_battery_power_is_kw():
    """Battery 的功率单位必须是 kW。

    16 A @ 208 V = 3.328 kW。若实现按 A*periods 处理，这里会得到完全不同的数值。
    """
    from acnportal.acnsim import Battery

    batt = Battery(capacity=10.0, init_charge=0.0, max_power=10.0)
    rate_a = batt.charge(pilot=16.0, voltage=208.0, period=30.0)

    assert rate_a == pytest.approx(16.0)  # 返回值是电流 [A]
    assert batt.current_charging_power == pytest.approx(16.0 * 208.0 / 1000.0)


def test_energy_units_end_to_end():
    """单桩满功率充电：请求 1 小时满功率电量，应当在恰好 1 小时内被完全满足。

    这是最灵敏的单位探针 —— 若 requested_energy 被误当成 A*periods，
    需求满足率会从 1.0 掉到 ~0.02。
    """
    from acnportal.acnsim import (
        Battery,
        ChargingNetwork,
        EV,
        EVSE,
        Simulator,
    )
    from acnportal.acnsim.events import EventQueue, PluginEvent
    from acnportal.acnsim.network import Current
    from acnportal.algorithms import UncontrolledCharging
    import acnportal.acnsim.analysis as an

    max_rate_a, voltage, period = 32.0, 208.0, 5.0
    max_kw = max_rate_a * voltage / 1000.0  # 6.656 kW

    cn = ChargingNetwork()
    cn.register_evse(EVSE("PS-001", max_rate=max_rate_a), voltage, 0)
    # 必须至少有一条约束：ChargingNetwork.constraint_matrix 在 add_constraint 之前是 None，
    # 而 InfrastructureInfo._validate 会无条件访问它的 .shape，导致仿真无法启动（见 PLAN.md §6 坑 #9）。
    # 单桩的桩级限流本身就是物理约束，这里如实建模。
    cn.add_constraint(Current(["PS-001"]), max_rate_a, "pilot_limit")

    # 请求量 = 满功率充电 1 小时的电量；窗口给 2 小时，充裕。
    ev = EV(
        arrival=0,
        departure=int(2 * 60 / period),
        requested_energy=max_kw * 1.0,
        station_id="PS-001",
        session_id="unit-probe",
        battery=Battery(max_kw * 1.0, 0.0, max_kw),
    )
    sim = Simulator(
        cn,
        UncontrolledCharging(),
        EventQueue([PluginEvent(ev.arrival, ev)]),
        dt.datetime(2026, 10, 9, 8, 0),
        period=period,
        verbose=False,
    )
    sim.run()

    assert an.total_energy_delivered(sim) == pytest.approx(max_kw, rel=1e-6)
    assert an.proportion_of_energy_delivered(sim) == pytest.approx(1.0, rel=1e-6)
    assert an.proportion_of_demands_met(sim, threshold=1e-3) == 1.0


# ---------------------------------------------------------------------------
# 3. 网络约束契约
# ---------------------------------------------------------------------------


def test_aggregate_constraint_never_violated():
    """聚合约束必须真正生效：峰值不超限，且违规计数为 0。

    注意 aggregate_power() 已是 kW，不要再除 1000（PLAN.md §6 坑 #5）。
    """
    from acnportal.acnsim import (
        Battery,
        ChargingNetwork,
        EV,
        EVSE,
        Simulator,
    )
    from acnportal.acnsim.events import EventQueue, PluginEvent
    from acnportal.acnsim.network import Current
    from acnportal.algorithms import SortedSchedulingAlgo, earliest_deadline_first
    import acnportal.acnsim.analysis as an

    n_ports, max_rate_a, voltage, period = 6, 32.0, 208.0, 5.0
    limit_a = 80.0
    limit_kw = limit_a * voltage / 1000.0

    cn = ChargingNetwork()
    for i in range(n_ports):
        cn.register_evse(EVSE(f"PS-{i:03d}", max_rate=max_rate_a), voltage, 0)
    # 负载系数用 Current(station_ids)，限值是独立参数（不是 Current(limit)）
    cn.add_constraint(Current([f"PS-{i:03d}" for i in range(n_ports)]), limit_a, "transformer")

    # 6 辆车同时到达且都要求满充 —— 总需求远超 80 A，约束必然成为瓶颈。
    max_kw = max_rate_a * voltage / 1000.0
    evs = [
        EV(i, i + 24, max_kw * 1.0, f"PS-{i:03d}", f"s{i}", Battery(max_kw, 0.0, max_kw))
        for i in range(n_ports)
    ]
    sim = Simulator(
        cn,
        SortedSchedulingAlgo(earliest_deadline_first),
        EventQueue([PluginEvent(ev.arrival, ev) for ev in evs]),
        dt.datetime(2026, 10, 9, 8, 0),
        period=period,
        verbose=False,
    )
    sim.run()

    peak_kw = an.aggregate_power(sim).max()
    assert peak_kw <= limit_kw * (1 + 1e-9), f"峰值 {peak_kw} kW 超过约束 {limit_kw} kW"

    currents = an.constraint_currents(sim)
    assert currents, "constraint_currents 应返回 {约束名: 数组} 的字典"
    for name, series in currents.items():
        assert np.all(np.asarray(series) <= limit_a * (1 + 1e-9)), f"约束 {name} 被违反"


# ---------------------------------------------------------------------------
# 4. 算法与可复现性契约
# ---------------------------------------------------------------------------


def test_three_algorithms_available():
    """方案 §5 要求的 FCFS / EDF / LLF 三个官方排序算法必须可用。"""
    from acnportal.algorithms import (
        SortedSchedulingAlgo,
        earliest_deadline_first,
        first_come_first_served,
        least_laxity_first,
    )

    for sort_fn in (first_come_first_served, earliest_deadline_first, least_laxity_first):
        assert isinstance(SortedSchedulingAlgo(sort_fn), SortedSchedulingAlgo)


def test_same_seed_same_scenario():
    """公平性原则的前提：同一种子必须重建出逐位相同的车辆会话。"""
    def build(seed):
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 35.0, size=(50, 3))

    a, b, c = build(42), build(42), build(43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c), "不同种子应产生不同场景"


def test_metric_threshold_is_absolute_energy():
    """proportion_of_demands_met 的 threshold 是绝对剩余电量 [kWh], 不是百分比。

    见 PLAN.md §6 坑 #4 —— 当成百分比会静默改变排名。
    """
    import inspect

    import acnportal.acnsim.analysis as an

    sig = inspect.signature(an.proportion_of_demands_met)
    assert "threshold" in sig.parameters
    # 默认值 0.1 是 0.1 kWh（≈"基本充满"），而非 10%
    assert sig.parameters["threshold"].default == 0.1
