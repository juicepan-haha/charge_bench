"""ChargeBench AI — 环境健康检查 / 已验证的端到端基线（阶段 0 产物）。

用有限车位 + 网络聚合功率约束跑通 FCFS/EDF/LLF 三算法对比，并提取方案 §5 的核心指标。
2026-10-09 在本机实测通过（Python 3.14.4 / acnportal 0.3.3 / numpy 2.5.3 / pandas 3.0.6）。

运行：
    python3 -m venv .venv
    .venv/bin/pip install "acnportal==0.3.3" "setuptools<81"
    .venv/bin/python scratch/acn_e2e_verified.py

预期输出（约束瓶颈下总电量相同、分布不同）：
    algo    demand_met%   satis%  energy_kWh   peak_kW   viol
    FCFS           38.5     14.3      156.97     16.64      0
    EDF            38.5     19.0      156.97     16.64      0
    LLF            38.5      0.0      156.97     16.64      0

单位约定（务必遵守；在线教程写的是已被替换的旧单位）：
    EVSE.max_rate [A] · Battery(capacity, init_charge) [kWh] · Battery(max_power) [kW]
    EV.requested_energy [kWh] · add_constraint(Current(ids), limit) 的 limit [A]

本脚本同时是阶段 2 的 Adapter 骨架：build_sessions() 里的车位分配必须自己写，
ACN-Sim 的 _convert_ev_matrix 默认假设无限车位。
"""
import datetime as dt
import numpy as np
import acnportal.acnsim as acnsim
from acnportal.acnsim import Simulator, ChargingNetwork, EVSE, EV, Battery
from acnportal.acnsim.events import EventQueue, PluginEvent
from acnportal.acnsim.network import Current
from acnportal.algorithms import (
    SortedSchedulingAlgo,
    first_come_first_served,
    earliest_deadline_first,
    least_laxity_first,
)
import acnportal.acnsim.analysis as an

PERIOD = 5          # minutes
N_PORTS = 6
VOLTAGE = 208.0
MAX_RATE = 32.0     # A per port
NETWORK_LIMIT_A = 80.0   # aggregate constraint (A across all ports)
N_SESSIONS = 30
SEED = 42

def build_network():
    cn = ChargingNetwork()
    for i in range(N_PORTS):
        cn.register_evse(EVSE(f"PS-{i:03d}", max_rate=MAX_RATE), VOLTAGE, 0)
    # aggregate "transformer" constraint: sum over all stations <= limit
    cn.add_constraint(Current([f"PS-{i:03d}" for i in range(N_PORTS)]), NETWORK_LIMIT_A, "transformer")
    return cn

def build_sessions(rng):
    """Generate synthetic sessions; assign to ports respecting occupancy."""
    hours = 10.0                      # simulated window: 10 hours
    periods_total = int(hours * 60 / PERIOD)
    raw = []
    for i in range(N_SESSIONS):
        arrival_p = int(rng.integers(0, periods_total - 12))
        stay_p = int(rng.integers(6, 48))          # 30 min .. 4 h
        departure_p = min(arrival_p + stay_p, periods_total)
        # energy requested: kWh
        req_kwh = float(rng.uniform(5.0, 35.0))
        raw.append((arrival_p, departure_p, req_kwh))
    raw.sort(key=lambda x: x[0])
    # greedy port assignment
    free_at = {f"PS-{i:03d}": 0 for i in range(N_PORTS)}
    sessions = []
    for idx, (a, d, e) in enumerate(raw):
        cands = [s for s, t in free_at.items() if t <= a]
        if not cands:
            continue                                    # drop: no free port on arrival
        st = min(cands, key=lambda s: free_at[s])
        free_at[st] = d
        sessions.append((st, f"session_{idx}", a, d, e))
    return sessions

def make_events(sessions):
    evs = []
    for st, sid, a, d, e_kwh in sessions:
        # acnportal 0.3.3: Battery/EV energies in kWh, power in kW
        max_kw = MAX_RATE * VOLTAGE / 1000.0
        batt = Battery(e_kwh, 0.0, max_kw)
        evs.append(EV(a, d, e_kwh, st, sid, batt))
    return EventQueue([PluginEvent(ev.arrival, ev) for ev in evs])

ALGOS = {
    "FCFS": first_come_first_served,
    "EDF":  earliest_deadline_first,
    "LLF":  least_laxity_first,
}

def run_one(cn, events, name, sort_fn):
    sched = SortedSchedulingAlgo(sort_fn)
    sim = Simulator(cn, sched, events, dt.datetime(2026, 10, 9), period=PERIOD, verbose=False)
    sim.run()
    return sim

rng = np.random.default_rng(SEED)
sessions = build_sessions(rng)
print(f"generated {len(sessions)} sessions on {N_PORTS} ports (limit {NETWORK_LIMIT_A}A)")

print(f"\n{'algo':6} {'demand_met%':>12} {'satis%':>8} {'energy_kWh':>11} {'peak_kW':>9} {'viol':>6}")
for name, fn in ALGOS.items():
    cn = build_network()
    events = make_events(sessions)
    sim = run_one(cn, events, name, fn)
    sat = an.proportion_of_demands_met(sim, threshold=0.99) * 100
    dem = an.proportion_of_energy_delivered(sim) * 100
    ene = an.total_energy_delivered(sim)
    peak = an.aggregate_power(sim).max()               # already kW
    # constraint violations
    ccur = an.constraint_currents(sim)
    viol = sum(int(np.sum(v > NETWORK_LIMIT_A * 1.0001)) for v in ccur.values())
    print(f"{name:6} {dem:12.1f} {sat:8.1f} {ene:11.2f} {peak:9.2f} {viol:6d}")
print("\nOK: end-to-end run completed")
