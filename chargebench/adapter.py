"""Adapter：把 Scenario 映射为 ACN-Sim 的网络与事件队列。

本模块是项目的技术核心，也是唯一允许 import acnportal 仿真对象的地方
（schemas / metrics / tariff 都不依赖 ACN-Sim 内部对象）。

三件必须自己做、ACN-Sim 不提供的事：
  1. **有限车位分配** —— ACN-Sim 的 ``StochasticEvents._convert_ev_matrix`` 给每辆车
     分配独立站点（源码注释写着 "Infinite space assumption"），与真实停车场不符。
  2. **至少一条网络约束** —— ``ChargingNetwork.constraint_matrix`` 在 add_constraint 之前
     是 None，而 ``InfrastructureInfo._validate`` 会无条件访问它的 ``.shape``，
     导致「没有约束的网络无法启动仿真」（PLAN.md §6 坑 #9）。
  3. **kWh/kW → A 的换算** —— EVSE 按安培建模，上层一律用 kW/kWh。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from acnportal.acnsim import Battery, ChargingNetwork, EV, EVSE, Simulator
from acnportal.acnsim.events import EventQueue, PluginEvent
from acnportal.acnsim.network import Current

from .schemas import ArrivalMode, Scenario

#: 到达位置的 Beta 分布形状参数（见 ArrivalMode 文档）
_ARRIVAL_SHAPE: dict[ArrivalMode, tuple[float, float]] = {
    ArrivalMode.uniform: (1.0, 1.0),
    ArrivalMode.front_loaded: (2.0, 4.0),
    ArrivalMode.back_loaded: (4.0, 2.0),
}


@dataclass(frozen=True)
class SessionRecord:
    """一辆车的会话请求。尚未分配车位，是算法无法看见的中间产物。"""

    session_id: str
    requested_energy_kwh: float
    arrival_period: int
    departure_period: int
    station_id: str = ""  # 由 assign_ports 填入
    stay_clamped: bool = False


@dataclass
class ScenarioArtifacts:
    """一个场景的确定性展开结果。"""

    scenario: Scenario
    sessions_generated: list[SessionRecord]
    sessions_scheduled: list[SessionRecord]
    n_dropped: int
    n_stay_clamped: int
    horizon_periods: int

    def build_network(self) -> ChargingNetwork:
        return build_network(self.scenario)

    def build_events(self) -> EventQueue:
        return build_events(self.sessions_scheduled, self.scenario)


# ----------------------------------------------------------------------
# 场景展开
# ----------------------------------------------------------------------


def generate_sessions(scenario: Scenario) -> list[SessionRecord]:
    """由场景参数确定性地生成车辆会话（座位分配之前）。

    数学关系：
        单车站满功率所需时长  t_req = e / port_max_power
        可停留时长            stay  = t_req / deadline_tightness
        到达时刻              arrival ~ (W - stay) * Beta(shape)

    到达时刻按「剩余可用区间」缩放，从而在窗口内精确保持 deadline_tightness；
    代价是到达分布形状在窗口尾部被轻微压缩，这是为了保证扫描轴可解释而做的取舍。
    """
    rng = np.random.default_rng(scenario.seed)
    n = scenario.n_sessions
    mean_energy = scenario.total_requested_kwh / n

    if scenario.energy_cv == 0.0:
        energies = np.full(n, mean_energy)
    else:
        # 对数正态，保证均值为 mean_energy、变异系数为 energy_cv、且恒为正
        sigma2 = np.log(1.0 + scenario.energy_cv**2)
        sigma = np.sqrt(sigma2)
        energies = mean_energy * np.exp(-sigma2 / 2 + sigma * rng.standard_normal(n))

    a_shape, b_shape = _ARRIVAL_SHAPE[scenario.arrival_mode]
    positions = rng.beta(a_shape, b_shape, size=n)

    records: list[SessionRecord] = []
    for i, (energy, pos) in enumerate(zip(energies, positions)):
        required_h = energy / scenario.port_max_power_kw
        stay_h = required_h / scenario.deadline_tightness

        clamped = stay_h > scenario.window_hours
        if clamped:
            # 车辆不能停到窗口之外。截断会使该车的实际紧迫度低于设定值，
            # 因此单独计数并上报，避免把截断误读成算法表现。
            stay_h = scenario.window_hours

        arrival_h = pos * max(scenario.window_hours - stay_h, 0.0)
        departure_h = arrival_h + stay_h

        periods_per_hour = 60.0 / scenario.time_step_min
        arrival_p = int(round(arrival_h * periods_per_hour))
        departure_p = int(round(departure_h * periods_per_hour))
        arrival_p = max(0, min(arrival_p, scenario.periods - 1))
        departure_p = max(arrival_p + 1, min(departure_p, scenario.periods))

        records.append(
            SessionRecord(
                session_id=f"s{i:05d}",
                requested_energy_kwh=float(energy),
                arrival_period=arrival_p,
                departure_period=departure_p,
                stay_clamped=bool(clamped),
            )
        )
    return records


def assign_ports(
    sessions: list[SessionRecord], n_ports: int
) -> tuple[list[SessionRecord], int]:
    """贪心分配车位：按到达时间顺序，选最早空出且已空闲的车位。

    到达时无空闲车位的会话被丢弃并计数 —— 这属于场景可行性问题，
    按方案 §5 公平性原则 5 不能归因于算法，故必须单独可见。

    返回 (已排定会话, 丢弃数)。结果对输入顺序确定，不依赖任何随机性。
    """
    free_at = [0] * n_ports
    scheduled: list[SessionRecord] = []
    dropped = 0

    for rec in sorted(sessions, key=lambda r: (r.arrival_period, r.session_id)):
        # 在所有空闲车位中选释放最早的那个；并列时取编号最小者以保证确定性
        best_port = -1
        best_free = None
        for port in range(n_ports):
            if free_at[port] <= rec.arrival_period:
                if best_free is None or free_at[port] < best_free:
                    best_free = free_at[port]
                    best_port = port
        if best_port < 0:
            dropped += 1
            continue
        free_at[best_port] = rec.departure_period
        scheduled.append(
            SessionRecord(
                session_id=rec.session_id,
                requested_energy_kwh=rec.requested_energy_kwh,
                arrival_period=rec.arrival_period,
                departure_period=rec.departure_period,
                station_id=f"PS-{best_port:03d}",
                stay_clamped=rec.stay_clamped,
            )
        )
    return scheduled, dropped


def build_scenario(scenario: Scenario) -> ScenarioArtifacts:
    """把一个 Scenario 确定性展开为可运行的全部产物。"""
    generated = generate_sessions(scenario)
    scheduled, dropped = assign_ports(generated, scenario.n_ports)
    horizon = max((r.departure_period for r in scheduled), default=0)
    return ScenarioArtifacts(
        scenario=scenario,
        sessions_generated=generated,
        sessions_scheduled=scheduled,
        n_dropped=dropped,
        n_stay_clamped=sum(r.stay_clamped for r in generated),
        horizon_periods=horizon,
    )


# ----------------------------------------------------------------------
# ACN-Sim 对象构造
# ----------------------------------------------------------------------


def build_network(scenario: Scenario) -> ChargingNetwork:
    """构造充电网络：n_ports 个充电桩 + 一条聚合功率约束。"""
    network = ChargingNetwork()
    max_rate_a = scenario.port_max_power_kw * 1000.0 / scenario.voltage_v
    for i in range(scenario.n_ports):
        network.register_evse(EVSE(f"PS-{i:03d}", max_rate=max_rate_a), scenario.voltage_v, 0)

    # 始终登记一条聚合约束（PLAN.md §6 坑 #9：没有约束的网络无法启动仿真）。
    # 限值为站点可交付功率换算成安培。network_limit_kw 为 None 时该约束不会真正生效，
    # 各桩的上限仍由 EVSE.max_rate 保证。
    limit_a = scenario.effective_supply_kw * 1000.0 / scenario.voltage_v
    network.add_constraint(
        Current([f"PS-{i:03d}" for i in range(scenario.n_ports)]), limit_a, "site_limit"
    )
    return network


def build_events(sessions: list[SessionRecord], scenario: Scenario) -> EventQueue:
    """构造事件队列。每次调用都重建全新的 EV 对象 —— 仿真会改写 EV 状态，
    因此不同算法必须各自拿到一份干净的队列，否则后跑的算法会看到脏数据。"""
    events = []
    for rec in sessions:
        battery = Battery(
            capacity=rec.requested_energy_kwh,
            init_charge=0.0,
            max_power=scenario.port_max_power_kw,
        )
        ev = EV(
            arrival=rec.arrival_period,
            departure=rec.departure_period,
            requested_energy=rec.requested_energy_kwh,
            station_id=rec.station_id,
            session_id=rec.session_id,
            battery=battery,
        )
        events.append(PluginEvent(ev.arrival, ev))
    return EventQueue(events)


def build_simulator(
    artifacts: ScenarioArtifacts, scheduler, start_datetime
) -> Simulator:
    """把场景产物装配成可运行的 Simulator。"""
    return Simulator(
        artifacts.build_network(),
        scheduler,
        artifacts.build_events(),
        start_datetime,
        period=artifacts.scenario.time_step_min,
        verbose=False,
    )
