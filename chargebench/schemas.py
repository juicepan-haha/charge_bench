"""ChargeBench 输入/输出契约（协议版本 1.0）。

设计原则（PLAN.md 阶段 1）：
  * 上层（UI / AI Agent / MCP）不依赖 ACN-Sim 内部对象，只依赖本文件的模型。
  * 一次实验可用一份 JSON 完整描述、复现、分享。
  * 所有字段显式声明单位、取值范围；非法组合在构造时即报错，不留到仿真运行中途。

单位约定（与 PLAN.md §2 一致，也是 ACN-Sim 0.3.3 的原生单位）：
    能量 kWh · 功率 kW · 电流 A · 时长 s · 时间步 min

用户可见字段一律用 kW/kWh —— 安培只出现在 Adapter 内部（ACN-Sim 的 EVSE 按安培建模）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"

#: 基准日期。固定为周一，使跨机器、跨天运行的仿真时间戳与分时电价完全一致。
#: 注意：本项目的分时电价只按「一天中的小时」划分，不区分工作日/周末（见 PROTOCOL.md「已知简化」）。
BENCHMARK_DATE = dt.date(2026, 1, 5)


class ArrivalMode(str, Enum):
    """车辆到达在仿真窗口内的时间分布形状。

    Adapter 用 Beta 分布生成归一化到达位置 u ∈ [0,1]，再映射到可用到达区间：
        uniform      Beta(1, 1) —— 均匀到达
        front_loaded Beta(2, 4) —— 前段集中（默认 8 点开始 ⇒ 上午高峰）
        back_loaded  Beta(4, 2) —— 后段集中（默认 8 点开始 ⇒ 傍晚高峰）
    """

    uniform = "uniform"
    front_loaded = "front_loaded"
    back_loaded = "back_loaded"


class PriceKind(str, Enum):
    flat = "flat"
    tou = "tou"


class PriceProfile(BaseModel):
    """分时电价。单位：CNY/kWh。

    内置电价全部为人民币 —— ACN-Sim 自带的 TariffSchedule 是美国加州电价（美元），
    直接使用会让成本指标币种错误（PLAN.md §6 坑 #7），故本项目自己实现。
    """

    model_config = {"extra": "forbid"}

    kind: PriceKind = PriceKind.tou
    currency: str = "CNY"
    flat_cny_per_kwh: float = Field(0.70, gt=0, description="平段电价")
    valley_cny_per_kwh: float = Field(0.30, gt=0, description="谷段电价")
    peak_cny_per_kwh: float = Field(1.10, gt=0, description="峰段电价")
    valley_hours: list[tuple[float, float]] = Field(
        default=[(0.0, 8.0)], description="谷段时段，绝对小时，左闭右开"
    )
    peak_hours: list[tuple[float, float]] = Field(
        default=[(10.0, 15.0), (18.0, 21.0)], description="峰段时段，绝对小时，左闭右开"
    )

    @field_validator("valley_hours", "peak_hours")
    @classmethod
    def _check_hours(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for lo, hi in v:
            if not (0.0 <= lo < hi <= 24.0):
                raise ValueError(
                    f"时段必须满足 0 <= 起点 < 终点 <= 24，收到 ({lo}, {hi})"
                )
        return v

    @model_validator(mode="after")
    def _check_price_order(self) -> "PriceProfile":
        if not (
            self.valley_cny_per_kwh
            < self.flat_cny_per_kwh
            < self.peak_cny_per_kwh
        ):
            raise ValueError(
                "电价必须满足 谷 < 平 < 峰，收到 "
                f"{self.valley_cny_per_kwh} / {self.flat_cny_per_kwh} / {self.peak_cny_per_kwh}"
            )
        return self


class Scenario(BaseModel):
    """一次实验的场景。所有字段即是可复现实验的完整输入。"""

    model_config = {"extra": "forbid"}

    scenario_id: str = Field(description="人类可读的场景标识")

    # --- 基础设施 ---
    n_ports: int = Field(ge=1, le=500, description="充电车位数量")
    port_max_power_kw: float = Field(gt=0, le=1000, description="单桩最大功率 [kW]")
    voltage_v: float = Field(220.0, gt=0, description="额定电压 [V]，中国单相 220 / 三相 380")
    network_limit_kw: Optional[float] = Field(
        None,
        gt=0,
        description="站点聚合功率上限 [kW]。None 表示不限（仅受各桩功率约束）",
    )

    # --- 时间 ---
    time_step_min: float = Field(5.0, gt=0, le=60, description="仿真时间步长 [min]")
    start_hour: float = Field(8.0, ge=0, lt=24, description="窗口起始时刻（一天中的小时）")
    window_hours: float = Field(10.0, gt=0, le=48, description="仿真窗口长度 [h]")

    # --- 负荷与紧迫度（阶段 4 的两个扫描轴）---
    n_sessions: int = Field(ge=1, le=10000, description="生成的车辆会话数")
    load_intensity: float = Field(
        gt=0,
        le=10,
        description="X 轴·负荷强度 = 总请求电量 / 站点窗口内可供电量。>1 表示必然供不应求",
    )
    deadline_tightness: float = Field(
        gt=0,
        le=1.0,
        description="Y 轴·时间紧迫度 = 满功率所需充电时长 / 可停留时长。=1 表示刚好够（叠加聚合约束后必然不可行）",
    )
    energy_cv: float = Field(
        0.0, ge=0.0, lt=1.0, description="单车请求电量的变异系数。0 表示所有车请求量完全相同"
    )

    arrival_mode: ArrivalMode = ArrivalMode.uniform
    price_profile: PriceProfile = Field(default_factory=PriceProfile)
    seed: int = Field(42, ge=0, description="随机种子。相同种子必须重建出逐位相同的场景")

    # ------------------------------------------------------------------
    # 派生量（不参与序列化与哈希，避免污染契约）
    # ------------------------------------------------------------------

    @property
    def periods(self) -> int:
        """窗口被划分成的时间步数。"""
        return int(self.window_hours * 60 / self.time_step_min)

    @property
    def effective_supply_kw(self) -> float:
        """站点在任一时刻的最大可交付功率 [kW]，取「桩总和」与「网络上限」的较小者。"""
        port_sum = self.n_ports * self.port_max_power_kw
        if self.network_limit_kw is None:
            return port_sum
        return min(port_sum, self.network_limit_kw)

    @property
    def supply_capability_kwh(self) -> float:
        """窗口内站点理论可交付的最大电量 [kWh]，即负荷强度的分母。"""
        return self.effective_supply_kw * self.window_hours

    @property
    def total_requested_kwh(self) -> float:
        """按负荷强度换算出的总请求电量 [kWh]。"""
        return self.load_intensity * self.supply_capability_kwh

    @property
    def port_occupancy_estimate(self) -> float:
        """预计车位占用率（泊位小时需求 / 泊位小时供给）。

        推导：总请求电量 = λ·S（S = effective_supply_kw · W），单车能量 e = λ·S/N，
        单车满功率时长 t_req = e/R，停留时长 stay = t_req/τ，
        于是总泊位需求 = N·stay = λ·S/(R·τ)，除以供给 P·W 得

            占用率 = λ · effective_supply_kw / (P · R · τ)

        与车辆数 N 无关 —— N 只改变粒度，不改变总占用量。
        **占用率 > 1 时必然出现因无空闲车位而被丢弃的会话**，此时 sessions_dropped
        会显著大于 0，需求满足率的分母也随之缩水。设定扫描范围时务必先看这个值。
        """
        return (
            self.load_intensity
            * self.effective_supply_kw
            / (self.n_ports * self.port_max_power_kw * self.deadline_tightness)
        )

    # ------------------------------------------------------------------
    # 非法组合校验
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_feasibility(self) -> "Scenario":
        if (
            self.network_limit_kw is not None
            and self.network_limit_kw < self.port_max_power_kw
        ):
            raise ValueError(
                f"network_limit_kw ({self.network_limit_kw}) 小于单桩功率 "
                f"({self.port_max_power_kw})：任何一辆车都无法以额定功率充电，"
                "场景退化。请提高网络上限或降低单桩功率。"
            )
        if self.periods < 2:
            raise ValueError(
                f"窗口过短：window_hours={self.window_hours} / time_step_min="
                f"{self.time_step_min} 只划分出 {self.periods} 个时间步。"
            )
        return self

    def canonical_json(self) -> str:
        """规范化 JSON 串，用于哈希与跨机器比对。"""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def with_updates(self, **changes: Any) -> "Scenario":
        """返回一个经过**完整校验**的新场景。

        不要用 ``model_copy(update=...)`` 改本场景：pydantic v2 的 model_copy 跳过校验，
        会把 ``kind="flat"`` 存成裸字符串而不是枚举成员，于是下游比较悄然失配、
        静默退回默认行为（实测踩到过）。本方法把改动重新走一遍模型构造。
        """
        return Scenario(**{**self.model_dump(mode="json"), **changes})

    @property
    def scenario_hash(self) -> str:
        return hashlib.sha1(self.canonical_json().encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# 结果
# ----------------------------------------------------------------------


class Metrics(BaseModel):
    """一次运行的标准化指标。

    口径定义详见 chargebench/metrics.py 与 docs/PROTOCOL.md —— 改口径必须同步改两处。
    """

    model_config = {"extra": "forbid"}

    # 能量口径
    demand_satisfaction_rate: float = Field(
        description="需求满足率（能量口径）= 实际交付电量 / 请求电量"
    )
    energy_delivered_kwh: float
    energy_requested_kwh: float

    # 车辆口径
    session_completion_rate: float = Field(
        description="按时完成率（车辆口径）= 剩余需求低于完成阈值的会话占比"
    )
    mean_delivery_ratio: float = Field(
        description="单车交付比的均值，仅统计已排定车位的会话"
    )
    worst_delivery_ratio: float = Field(
        description="最差单车交付比（仅统计已排定会话），用于观察稳健性"
    )

    # 成本与电网
    energy_cost_cny: float = Field(description="总用电成本 [CNY]")
    peak_power_kw: float = Field(description="仿真时段内最大聚合负荷 [kW]")
    constraint_violations: int = Field(description="越限的时间步数（所有约束合计）")
    max_violation_a: float = Field(description="最大越限幅度 [A]，未越限时为 0")

    # 场景可行性（不可归因于算法，必须单独可见，见方案 §5 公平性原则 5）
    sessions_generated: int
    sessions_scheduled: int
    sessions_dropped: int = Field(description="因到达时无空闲车位而未进入仿真的会话数")

    runtime_s: float


class Traceability(BaseModel):
    """可追溯元数据。方案 §4 要求每个 run_id 可关联到依赖与执行环境。"""

    model_config = {"extra": "forbid"}

    schema_version: str = SCHEMA_VERSION
    acnportal_version: str
    python_version: str
    numpy_version: str
    pandas_version: str
    benchmark_date: str = Field(description="仿真基准日，固定为周一")
    simulation_start: dt.datetime = Field(description="仿真窗口起点（由基准日 + start_hour 决定）")
    executed_at: dt.datetime = Field(description="实际执行时刻，仅作记录，不参与 run_id")


class RunResult(BaseModel):
    """一次单算法运行的结果。UI 与 AI Agent 只消费这个结构。"""

    model_config = {"extra": "forbid"}

    run_id: str = Field(description="由场景+算法+参数+种子确定性导出，同配置重复运行得到同一 id")
    scenario_id: str
    scenario_hash: str
    algorithm: str
    algorithm_params: dict[str, Any] = Field(default_factory=dict)
    seed: int
    metrics: Metrics
    traceability: Traceability

    @staticmethod
    def make_run_id(
        scenario: Scenario, algorithm: str, algorithm_params: dict[str, Any], seed: int
    ) -> str:
        """确定性 run_id：同配置在任何机器上都得到同一个 id，便于幂等重跑与结果比对。"""
        payload = json.dumps(
            {
                "scenario": scenario.model_dump(mode="json"),
                "algorithm": algorithm,
                "algorithm_params": algorithm_params,
                "seed": seed,
                "schema_version": SCHEMA_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
        return f"{scenario.scenario_id}--{algorithm}--{digest}"


class BatchResult(BaseModel):
    """一次批量比较的结果：同一场景集合 × 多算法 × 多种子。"""

    model_config = {"extra": "forbid"}

    batch_id: str
    schema_version: str = SCHEMA_VERSION
    created_at: dt.datetime
    scenario_ids: list[str]
    algorithms: list[str]
    seeds: list[int]
    runs: list[RunResult]
    truncated: bool = Field(
        False,
        description=(
            "是否因超出时间预算而提前结束。为 True 时 runs 只是计划集合的一部分 —— "
            "**不得**据此得出「某算法更优」的结论，因为候选之间可能跑的次数并不对等。"
        ),
    )


# ----------------------------------------------------------------------
# 参数扫描与适用边界
# ----------------------------------------------------------------------

#: 可参与排名的指标及其方向。"max" = 越大越好，"min" = 越小越好。
METRIC_DIRECTIONS: dict[str, str] = {
    "demand_satisfaction_rate": "max",
    "session_completion_rate": "max",
    "mean_delivery_ratio": "max",
    "worst_delivery_ratio": "max",
    "energy_cost_cny": "min",
    "peak_power_kw": "min",
}


class MetricAggregate(BaseModel):
    """某个算法在某个网格点上、跨多个种子的聚合指标。

    单次运行会被随机性掩盖差异，因此扫描一律跨种子取均值，并给出波动范围
    （方案 §5 公平性原则 2）。
    """

    model_config = {"extra": "forbid"}

    n_seeds: int
    demand_satisfaction_rate_mean: float
    session_completion_rate_mean: float
    energy_cost_cny_mean: float
    peak_power_kw_mean: float
    mean_delivery_ratio_mean: float
    worst_delivery_ratio_mean: float
    constraint_violations_max: int = Field(
        description="各种子中的最大违规步数。>0 即取消该算法在本格点的资格"
    )
    #: 主指标的跨种子极差 —— 稳健性。均值相同的两个算法，波动小的更可信。
    spread: float = Field(description="主指标在多种子间的极差 (max - min)")


class SweepCell(BaseModel):
    """扫描网格的一个格点。"""

    model_config = {"extra": "forbid"}

    load_intensity: float
    deadline_tightness: float
    #: 预计车位占用率。>1 表示该格点必然出现丢弃，属于场景可行性问题而非算法问题
    port_occupancy_estimate: float
    port_constrained: bool = Field(
        description="占用率 > 1：该格点的丢弃会显著影响满足率分母，结论需谨慎"
    )
    sessions_dropped: int = Field(description="实际被丢弃的会话数（各种子均值）")
    aggregates: dict[str, MetricAggregate] = Field(
        description="算法名 → 跨种子聚合指标"
    )
    winner: Optional[str] = Field(
        default=None, description="按评分规则胜出的算法；全部被取消资格时为 None"
    )
    disqualified: list[str] = Field(
        default_factory=list, description="因出现约束违规被取消资格的算法"
    )


class SweepResult(BaseModel):
    """两变量参数扫描的结果，用于绘制算法适用区域图。"""

    model_config = {"extra": "forbid"}

    sweep_id: str
    schema_version: str = SCHEMA_VERSION
    created_at: dt.datetime
    base_scenario_id: str
    x_name: str
    x_values: list[float]
    y_name: str
    y_values: list[float]
    algorithms: list[str]
    seeds: list[int]
    primary_metric: str = Field(description="排名依据的指标名")
    metric_direction: str = Field(description="max 或 min")
    score_rule: str = Field(description="人类可读的评分规则，事前写定，不得事后更改")
    cells: list[SweepCell]
    run_ids: list[str] = Field(description="本次扫描涉及的全部 run_id，用于追溯")

    def cell(self, x: float, y: float) -> SweepCell:
        for c in self.cells:
            if c.load_intensity == x and c.deadline_tightness == y:
                return c
        raise KeyError(f"网格点 ({x}, {y}) 不存在")

    def winner_grid(self) -> tuple[list[list[Optional[str]]], list[list[bool]]]:
        """返回 (胜者矩阵, 是否受车位约束矩阵)，行对应 y、列对应 x，供热力图直接使用。"""
        rows: list[list[Optional[str]]] = []
        constrained: list[list[bool]] = []
        for y in self.y_values:
            row: list[Optional[str]] = []
            crow: list[bool] = []
            for x in self.x_values:
                c = self.cell(x, y)
                row.append(c.winner)
                crow.append(c.port_constrained)
            rows.append(row)
            constrained.append(crow)
        return rows, constrained

    def metric_grid(self, algorithm: str) -> list[list[float]]:
        """返回某算法主指标的取值矩阵，缺失处填 NaN。"""
        rows: list[list[float]] = []
        for y in self.y_values:
            row: list[float] = []
            for x in self.x_values:
                agg = self.cell(x, y).aggregates.get(algorithm)
                row.append(float("nan") if agg is None else getattr(agg, f"{self.primary_metric}_mean", float("nan")))
            rows.append(row)
        return rows


# ----------------------------------------------------------------------
# Agent 实验闭环（阶段 6）
# ----------------------------------------------------------------------


class SortKey(str, Enum):
    """候选调度策略可用的排序键白名单。

    **刻意做成白名单而不是允许执行代码**：方案 §7 的工程约束要求「禁止任意 Python 代码
    执行」。LLM 因此只能从这套词汇里组合策略，而不能生成代码。这样既安全，
    也让每个假设都能用一句话说清楚。
    """

    arrival = "arrival"
    estimated_departure = "estimated_departure"
    laxity = "laxity"
    remaining_demand = "remaining_demand"
    delivery_ratio = "delivery_ratio"


class CandidateSpec(BaseModel):
    """一个候选调度策略。即 Agent 提出的「假设」的可执行形式。"""

    model_config = {"extra": "forbid"}

    name: str = Field(description="策略名，进入 run_id")
    primary: SortKey = Field(description="主排序键")
    direction: Literal["asc", "desc"] = "asc"
    completion_first: bool = Field(
        False,
        description=(
            "先排出能在剩余停留时间内充满的车辆。用于检验「给 LLF 加完成度优先回退」"
            "这类假设 —— LLF 的松弛时间每周期重算会把能量摊薄，加上这条可能恢复完成率。"
        ),
    )
    tiebreak: Optional[SortKey] = Field(None, description="主键相同时的次级排序键")
    rationale: str = Field("", description="提出该策略的理由，写进报告供人核对")
    baseline: bool = Field(
        False,
        description=(
            "是否为注册表里的官方基准算法。为 True 时按名字调用注册表，"
            "**不复用组合器** —— 否则就变成拿我的重实现去和官方实现对打，排名没有意义。"
        ),
    )


class BudgetSpec(BaseModel):
    """实验预算。方案 §6：任何自动执行都必须有资源上限。"""

    model_config = {"extra": "forbid"}

    max_rounds: int = Field(4, ge=1, le=50, description="最大批次数")
    max_simulations: int = Field(200, ge=1, le=100000, description="总仿真调用次数上限")
    max_seconds: float = Field(120.0, gt=0, description="墙钟时间上限")
    patience: int = Field(2, ge=1, le=20, description="连续多少轮无实质改善即停止")
    min_improvement: float = Field(
        0.01, gt=0, description="主目标提升多少才算「实质改善」"
    )


class CandidateOutcome(BaseModel):
    """一个候选策略在基准场景上的成绩。所有数字来自 metrics 模块。"""

    model_config = {"extra": "forbid"}

    spec: CandidateSpec
    run_ids: list[str]
    session_completion_rate: float
    demand_satisfaction_rate: float
    energy_cost_cny: float
    peak_power_kw: float
    constraint_violations: int
    spread: float = Field(description="主指标跨种子极差，稳健性参考")
    n_simulations: int


class ValidationResult(BaseModel):
    """在保留场景上的复验。方案 §5 公平性原则 6：避免对实验集过拟合。"""

    model_config = {"extra": "forbid"}

    candidate_name: str
    scenario_ids: list[str]
    candidate_completion_rate: float
    best_baseline_name: str
    best_baseline_completion_rate: float
    improvement: float = Field(description="候选 − 基准，在保留场景上的差值")
    passed: bool = Field(description="是否仍然优于基准")
    run_ids: list[str]


class RoundRecord(BaseModel):
    """一轮实验的记录。"""

    model_config = {"extra": "forbid"}

    round_index: int
    outcomes: list[CandidateOutcome]
    best_so_far: str
    best_value: float
    improved: bool = Field(description="本轮是否带来实质改善")
    simulations_used: int = Field(description="累计仿真次数")


class AgentReport(BaseModel):
    """AI 实验闭环的产出。

    **措辞纪律**：结论只能表述为「在已测试范围与预算内的最佳可行候选」，
    不得声称全局最优 —— 见 agent_loop.build_conclusion 与对应测试。
    """

    model_config = {"extra": "forbid"}

    report_id: str
    schema_version: str = SCHEMA_VERSION
    created_at: dt.datetime
    primary_metric: str
    primary_direction: str
    budget: BudgetSpec
    stop_reason: str
    rounds: list[RoundRecord]
    baseline_outcomes: list[CandidateOutcome]
    best_candidate: Optional[CandidateOutcome]
    validation: Optional[ValidationResult]
    conclusion: str = Field(description="经过措辞纪律约束的结论")
    caveats: list[str] = Field(
        default_factory=list, description="必须写进报告的限制条件"
    )
    simulations_used: int
    elapsed_seconds: float
    all_run_ids: list[str]
