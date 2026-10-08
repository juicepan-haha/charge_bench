"""AI 实验闭环：提出假设 → 调用仿真 → 分析结果 → 决定下一步。

方案 §6 要求 AI 变成实验研究员，而不是写总结。三条纪律贯穿本模块：

1. **LLM 不参与任何数值计算。** 它只能从 ``SortKey`` 这套受控词汇里组合候选策略，
   所有指标都来自 ``metrics`` 模块。方案 §7 的「禁止任意 Python 代码执行」也因此满足：
   LLM 输出的是结构化规格，不是代码。
2. **任何自动执行都有资源上限**（方案 §6）。批次数、仿真调用数、墙钟时间、连续无改善
   轮数四重闸门，任一触发即停。
3. **措辞必须收敛**。除非有严格求解器证明，只能说「在已测试范围与预算内的最佳可行候选」。
   这条由 ``build_conclusion`` 保证并有测试守护 —— 它明文禁止「全局最优」这类表述。

另有一条工程纪律：**AI 提出的改进必须在保留场景上复验**（方案 §5 原则 6），
否则很容易把「对实验集过拟合」当成「算法变好了」。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import numpy as np
from acnportal.algorithms import SortedSchedulingAlgo
from acnportal.acnsim.interface import SessionInfo

from .algorithms import CORE_ALGORITHMS, build_algorithm
from .experiments import run_with_scheduler
from .schemas import (
    METRIC_DIRECTIONS,
    SCHEMA_VERSION,
    AgentReport,
    BudgetSpec,
    CandidateOutcome,
    CandidateSpec,
    RoundRecord,
    RunResult,
    Scenario,
    SortKey,
    ValidationResult,
)
from .storage import ResultStore

#: 基准算法对应的排序意图描述。仅用于报告展示，实际运行仍走注册表里的官方实现。
BASELINE_SPECS: dict[str, CandidateSpec] = {
    "FCFS": CandidateSpec(
        name="FCFS", primary=SortKey.arrival, direction="asc", baseline=True,
        rationale="官方基准：先到先服务。",
    ),
    "EDF": CandidateSpec(
        name="EDF", primary=SortKey.estimated_departure, direction="asc", baseline=True,
        rationale="官方基准：最早离站优先。",
    ),
    "LLF": CandidateSpec(
        name="LLF", primary=SortKey.laxity, direction="asc", baseline=True,
        rationale=(
            "官方基准：最小松弛时间优先。松弛时间每周期重算，"
            "已获供电的车辆会被逐渐降级，因而可能把能量摊薄。"
        ),
    ),
}


# ----------------------------------------------------------------------
# 受控词汇 → 可执行调度器
# ----------------------------------------------------------------------


def _laxity_periods(ev: SessionInfo, iface) -> float:
    """剩余松弛时间 [period]，与 ACN-Sim 官方 least_laxity_first 的定义一致。"""
    max_rate = iface.max_pilot_signal(ev.station_id)
    if max_rate <= 0:
        return float("-inf")
    return (ev.estimated_departure - iface.current_time) - (
        iface.remaining_amp_periods(ev) / max_rate
    )


def _key_value(key: SortKey, ev: SessionInfo, iface) -> float:
    if key is SortKey.arrival:
        return float(ev.arrival)
    if key is SortKey.estimated_departure:
        return float(ev.estimated_departure)
    if key is SortKey.laxity:
        return _laxity_periods(ev, iface)
    if key is SortKey.remaining_demand:
        return float(ev.remaining_demand)
    if key is SortKey.delivery_ratio:
        if ev.requested_energy <= 0:
            return 0.0
        return float(ev.energy_delivered / ev.requested_energy)
    raise KeyError(f"未实现的排序键 {key!r}")  # pragma: no cover - 枚举已封闭


def build_sort_fn(spec: CandidateSpec) -> Callable[[list, Any], list]:
    """把候选规格编译成一个排序函数。

    ``completion_first`` 实现方案 §6 提到的那个具体假设：先排出**能在剩余停留时间内
    充满**的车辆（即松弛时间 >= 0）。LLF 的松弛时间每周期重算会把能量摊薄到多辆车，
    加上这条回退规则后，能充满的车会先被喂饱。
    """

    def sort_fn(evs: list, iface) -> list:
        def key(ev: SessionInfo):
            parts: list[float] = []
            if spec.completion_first:
                # 0 排在前：能充满的优先
                parts.append(0.0 if _laxity_periods(ev, iface) >= 0 else 1.0)
            primary = _key_value(spec.primary, ev, iface)
            parts.append(-primary if spec.direction == "desc" else primary)
            if spec.tiebreak is not None:
                parts.append(_key_value(spec.tiebreak, ev, iface))
            # 最后的键保证排序稳定且确定，不依赖输入顺序
            parts.append(float(ev.arrival))
            return tuple(parts)

        return sorted(evs, key=key)

    return sort_fn


def build_spec_algorithm(spec: CandidateSpec):
    """按候选规格构造调度器。基准算法走注册表，不复用组合器。"""
    if spec.baseline:
        return build_algorithm(spec.name, {})
    return SortedSchedulingAlgo(build_sort_fn(spec))


# ----------------------------------------------------------------------
# 预算账本
# ----------------------------------------------------------------------


class BudgetExhausted(RuntimeError):
    """预算耗尽。由循环内部捕获用来正常收尾，不外泄给调用方。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Ledger:
    """资源账本。方案 §6：任何自动执行都必须有资源上限。"""

    budget: BudgetSpec
    simulations: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def remaining_simulations(self) -> int:
        return self.budget.max_simulations - self.simulations

    @property
    def remaining_seconds(self) -> float:
        return self.budget.max_seconds - self.elapsed

    def check(self, needed: int = 1) -> None:
        """在开跑之前检查额度，超了就抛。"""
        if needed > self.remaining_simulations:
            raise BudgetExhausted("simulations")
        if self.remaining_seconds <= 0:
            raise BudgetExhausted("seconds")

    def charge(self, n: int) -> None:
        self.simulations += n


# ----------------------------------------------------------------------
# 提议者
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalContext:
    """提议者能看到的全部信息。刻意只给结构化结果，不透露仿真内部状态。"""

    primary_metric: str
    direction: str
    baseline_best_name: str
    baseline_best_value: float
    best_name: str
    best_value: float
    tried_names: tuple[str, ...]
    history: tuple[tuple[str, float], ...]
    remaining_rounds: int
    remaining_simulations: int
    default_scenario_id: str


class Proposer(Protocol):
    def propose(self, context: ProposalContext, batch_size: int) -> list[CandidateSpec]:
        """返回下一批要测的候选。返回空列表表示「没有更多假设」，循环随即收尾。"""
        ...


#: 启发式提议者的先验顺序：先测「完成度优先」这条针对性假设，再逐步放宽。
#: 顺序是写死的常量，不随时间或结果变化 —— 这样整个搜索可复现、可解释。
_HEURISTIC_ORDER: tuple[tuple[SortKey, bool, Optional[SortKey]], ...] = (
    # 第一梯队：直接检验「给松弛类策略加完成度优先回退」这个假设
    (SortKey.laxity, True, None),
    (SortKey.estimated_departure, True, None),
    (SortKey.laxity, True, SortKey.estimated_departure),
    # 第二梯队：其它主键 + 完成度优先
    (SortKey.arrival, True, None),
    (SortKey.remaining_demand, True, None),
    (SortKey.estimated_departure, True, SortKey.laxity),
    (SortKey.laxity, True, SortKey.arrival),
    # 第三梯队：不带完成度优先的纯排序变体
    (SortKey.delivery_ratio, False, None),
    (SortKey.remaining_demand, False, SortKey.estimated_departure),
    (SortKey.laxity, False, SortKey.arrival),
    (SortKey.delivery_ratio, False, SortKey.estimated_departure),
    (SortKey.remaining_demand, False, None),
)


def _spec_name(primary: SortKey, completion_first: bool, tiebreak: Optional[SortKey]) -> str:
    tag = "CF" if completion_first else "P"
    tail = f"+{tiebreak.value}" if tiebreak else ""
    return f"{primary.value}-{tag}{tail}"


class HeuristicProposer:
    """确定性搜索提议者，不依赖任何外部服务。

    方案 §8 要求「先保障离线回退」：AI 服务不可用时，演示仍要能完整跑完闭环。
    因此内层数值搜索（这个类）与外层 LLM 推理（``LLMProposer``）是分离的。
    """

    def propose(self, context: ProposalContext, batch_size: int) -> list[CandidateSpec]:
        out: list[CandidateSpec] = []
        for primary, completion_first, tiebreak in _HEURISTIC_ORDER:
            name = _spec_name(primary, completion_first, tiebreak)
            if name in context.tried_names:
                continue
            rationale = (
                f"启发式搜索：主键 {primary.value}"
                + ("，并优先排出能在剩余时间内充满的车辆" if completion_first else "")
                + (f"，次级键 {tiebreak.value}" if tiebreak else "")
                + "。"
            )
            out.append(
                CandidateSpec(
                    name=name,
                    primary=primary,
                    direction="asc",
                    completion_first=completion_first,
                    tiebreak=tiebreak,
                    rationale=rationale,
                )
            )
            if len(out) >= batch_size:
                break
        return out


_LLM_SYSTEM_PROMPT = """你是充电调度实验的研究助理。你的唯一职责是决定下一批要测什么策略。

严格约束：
- 你只能从给定的排序键词汇里组合策略，**不得生成代码**。
- 你不得计算或推断任何指标数值；数值一律由仿真引擎给出。
- 你的输出必须是 JSON 数组，每个元素形如：
  {"name": "...", "primary": "<排序键>", "direction": "asc|desc",
   "completion_first": true|false, "tiebreak": "<排序键>|null", "rationale": "..."}
- 不要输出 JSON 以外的任何内容。"""


class LLMProposer:
    """由外部 LLM 驱动的提议者。

    ``complete`` 是一个「提示词进、文本出」的可调用对象，由调用方注入具体的模型客户端。
    这样本模块不绑定任何厂商 SDK，也便于在测试里用一个确定性的假实现替换。

    **严格校验**：LLM 的输出必须能被 ``CandidateSpec`` 完整解析，未知字段直接报错。
    解析失败的条目被丢弃并记录到 ``rejected``，不会污染实验 —— 这是「参数白名单」的落点。
    """

    def __init__(self, complete: Callable[[str], str]):
        self._complete = complete
        self.rejected: list[str] = []
        self.last_raw: str = ""

    def propose(self, context: ProposalContext, batch_size: int) -> list[CandidateSpec]:
        self.rejected = []
        prompt = self._build_prompt(context, batch_size)
        try:
            raw = self._complete(prompt)
        except Exception as exc:  # noqa: BLE001 - 外部服务失败不该让闭环崩掉
            self.rejected.append(f"调用失败：{exc}")
            return []
        self.last_raw = raw

        try:
            payload = json.loads(_extract_json_array(raw))
        except Exception as exc:  # noqa: BLE001
            self.rejected.append(f"响应不是合法 JSON 数组：{exc}")
            return []
        if not isinstance(payload, list):
            self.rejected.append("响应不是 JSON 数组")
            return []

        out: list[CandidateSpec] = []
        for index, item in enumerate(payload):
            try:
                spec = CandidateSpec(**item)
            except Exception as exc:  # noqa: BLE001 - 单条不合法不影响其余
                self.rejected.append(f"第 {index} 条不合法：{exc}")
                continue
            if spec.name in context.tried_names:
                self.rejected.append(f"{spec.name} 已测过")
                continue
            out.append(spec)
            if len(out) >= batch_size:
                break
        return out

    @staticmethod
    def _build_prompt(context: ProposalContext, batch_size: int) -> str:
        keys = ", ".join(k.value for k in SortKey)
        history = "\n".join(
            f"  - {name}: {value:.4f}" for name, value in context.history
        ) or "  （尚无）"
        return (
            f"{_LLM_SYSTEM_PROMPT}\n\n"
            f"场景：{context.default_scenario_id}\n"
            f"主目标：{context.primary_metric}（{context.direction}）\n"
            f"可用排序键：{keys}\n\n"
            f"基准成绩：\n"
            + "\n".join(
                f"  - {n}: {v:.4f}"
                for n, v in context.history
                if n in BASELINE_SPECS
            )
            + f"\n\n已有全部结果：\n{history}\n\n"
            f"已测策略：{', '.join(context.tried_names) or '（无）'}\n"
            f"剩余轮数 {context.remaining_rounds}，剩余仿真次数 {context.remaining_simulations}。\n"
            f"请提出至多 {batch_size} 个新策略。"
        )


def _extract_json_array(text: str) -> str:
    """从模型响应里抠出 JSON 数组。容忍 ```json 围栏与前后寒暄。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [ln for ln in stripped.splitlines() if not ln.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    start, end = stripped.find("["), stripped.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("响应里找不到 JSON 数组")
    return stripped[start : end + 1]


# ----------------------------------------------------------------------
# 评测
# ----------------------------------------------------------------------


def _seed_spread(runs: list[RunResult], metric: str) -> float:
    """稳健性：**每个场景内**跨种子的极差，再对场景取均值。

    不能把所有 run 池在一起算极差 —— 那样量到的主要是「场景之间的差异」，
    而不是「同场景下种子带来的噪声」。实测池化口径会给出 0.48 这种被场景差异
    灌水的数字，掩盖真正的种子波动。
    """
    by_scenario: dict[str, list[float]] = {}
    for r in runs:
        by_scenario.setdefault(r.scenario_id, []).append(getattr(r.metrics, metric))
    spreads = [max(v) - min(v) for v in by_scenario.values() if len(v) > 1]
    return float(np.mean(spreads)) if spreads else 0.0


def _outcome_from_runs(spec: CandidateSpec, runs: list[RunResult]) -> CandidateOutcome:
    return CandidateOutcome(
        spec=spec,
        run_ids=[r.run_id for r in runs],
        session_completion_rate=float(
            np.mean([r.metrics.session_completion_rate for r in runs])
        ),
        demand_satisfaction_rate=float(
            np.mean([r.metrics.demand_satisfaction_rate for r in runs])
        ),
        energy_cost_cny=float(np.mean([r.metrics.energy_cost_cny for r in runs])),
        peak_power_kw=float(np.mean([r.metrics.peak_power_kw for r in runs])),
        constraint_violations=int(
            max(r.metrics.constraint_violations for r in runs)
        ),
        spread=_seed_spread(runs, PRIMARY_METRIC_HOLDER[0]),
        n_simulations=len(runs),
    )


#: 主指标名在聚合时用到的占位。用单元素列表是为了让内部函数能读到循环外的选择，
#: 又不至于把 metric 名穿透每一层调用。
PRIMARY_METRIC_HOLDER: list[str] = ["session_completion_rate"]


def evaluate_spec(
    spec: CandidateSpec,
    scenarios: Sequence[Scenario],
    seeds: Sequence[int],
    ledger: Ledger,
    store: ResultStore | None = None,
) -> CandidateOutcome:
    """在一组场景上评测一个候选，跨种子聚合。

    所有指标来自 ``RunResult.metrics``，本函数只做均值与极差 —— 不做任何自己的计算。
    """
    runs: list[RunResult] = []
    for scenario in scenarios:
        for seed in seeds:
            ledger.check()
            factory = (
                (lambda s=spec: build_spec_algorithm(s))
                if not spec.baseline
                else (lambda s=spec: build_algorithm(s.name, {}))
            )
            cached = (
                store.find_existing(scenario.scenario_hash, spec.name, seed)
                if store is not None
                else None
            )
            if cached is not None:
                runs.append(cached)
                continue
            result = run_with_scheduler(
                scenario,
                algorithm_label=spec.name,
                scheduler_factory=factory,
                algorithm_params=_params_signature(spec),
                seed=seed,
            )
            ledger.charge(1)
            if store is not None:
                store.save_run(result)
            runs.append(result)
    return _outcome_from_runs(spec, runs)


def _params_signature(spec: CandidateSpec) -> dict[str, Any]:
    """候选规格里影响行为的字段，写进 run_id。

    不含 ``name`` 与 ``rationale`` —— 改名或改措辞不该产生新的实验记录。
    """
    return {
        "primary": spec.primary.value,
        "direction": spec.direction,
        "completion_first": spec.completion_first,
        "tiebreak": spec.tiebreak.value if spec.tiebreak else None,
        "baseline": spec.baseline,
    }


# ----------------------------------------------------------------------
# 循环
# ----------------------------------------------------------------------


def run_agent_loop(
    scenarios: Sequence[Scenario],
    holdout_scenarios: Sequence[Scenario],
    base_algorithms: Sequence[str] = CORE_ALGORITHMS,
    proposer: Proposer | None = None,
    budget: BudgetSpec | None = None,
    primary_metric: str = "session_completion_rate",
    seeds: Sequence[int] | None = None,
    store: ResultStore | None = None,
    target: float | None = None,
    batch_size: int = 3,
) -> AgentReport:
    """跑完整的 AI 实验闭环。

    Args:
        scenarios: 用于提出与筛选假设的场景（实验集）。
        holdout_scenarios: 保留场景，只用来复验最终候选（方案 §5 原则 6）。
            **不得**参与筛选 —— 否则保留场景就失去了意义。
        proposer: 提议者。为 None 时用 ``HeuristicProposer``（离线回退）。
        budget: 资源上限。为 None 时用默认值。
        target: 主目标达到该值即提前停止。
        batch_size: 每轮评测多少个候选。

    返回的 ``AgentReport`` 含结论文本与必须随附的限制说明（``caveats``）。
    """
    if not scenarios:
        raise ValueError("至少需要一个实验场景")
    if primary_metric not in METRIC_DIRECTIONS:
        raise KeyError(
            f"primary_metric 必须是 {sorted(METRIC_DIRECTIONS)} 之一，收到 {primary_metric!r}"
        )

    PRIMARY_METRIC_HOLDER[0] = primary_metric
    budget = budget or BudgetSpec()
    seeds = list(seeds) if seeds else [scenarios[0].seed]
    proposer = proposer or HeuristicProposer()
    ledger = Ledger(budget=budget)

    tried: dict[str, CandidateOutcome] = {}
    all_run_ids: list[str] = []
    rounds: list[RoundRecord] = []
    stop_reason = "completed"

    # --- 基准轮 -------------------------------------------------------
    baseline_outcomes: list[CandidateOutcome] = []
    for name in base_algorithms:
        spec = BASELINE_SPECS.get(name) or CandidateSpec(
            name=name, primary=SortKey.arrival, baseline=True,
            rationale=f"注册表算法 {name}。",
        )
        try:
            ledger.check(needed=len(scenarios) * len(seeds))
        except BudgetExhausted as exc:
            stop_reason = exc.reason
            break
        outcome = evaluate_spec(spec, scenarios, seeds, ledger, store)
        baseline_outcomes.append(outcome)
        tried[name] = outcome
        all_run_ids.extend(outcome.run_ids)

    if not baseline_outcomes:
        raise BudgetExhausted(
            f"预算不足以跑完任何基准算法（需要 {len(scenarios) * len(seeds)} 次仿真，"
            f"上限 {budget.max_simulations}）"
        )

    def _rank(outcome: CandidateOutcome) -> float:
        """候选排序键。**出现约束违规的一律排到最后**（返回 +inf）。

        方案 §5 公平性原则 3：不允许把违规供电当成更优结果。若只看主指标，
        一个把网络撑爆的策略可能赢下排名 —— 那等于奖励了错误行为。
        这与扫描模块的「违规即取消资格」是同一条规则。
        """
        if outcome.constraint_violations > 0:
            return float("inf")
        value = _metric_of(outcome, primary_metric)
        return -value if METRIC_DIRECTIONS[primary_metric] == "max" else value

    best = min(tried.values(), key=_rank)
    baseline_best = min(baseline_outcomes, key=_rank)

    # --- 探索轮 -------------------------------------------------------
    round_index = 0
    no_improvement_streak = 0
    while round_index < budget.max_rounds:
        if stop_reason != "completed":
            break
        if target is not None and _metric_of(best, primary_metric) >= target:
            stop_reason = "target_reached"
            break
        if no_improvement_streak >= budget.patience:
            stop_reason = "no_improvement"
            break

        context = ProposalContext(
            primary_metric=primary_metric,
            direction=METRIC_DIRECTIONS[primary_metric],
            baseline_best_name=baseline_best.spec.name,
            baseline_best_value=_metric_of(baseline_best, primary_metric),
            best_name=best.spec.name,
            best_value=_metric_of(best, primary_metric),
            tried_names=tuple(sorted(tried)),
            history=tuple(
                (name, _metric_of(o, primary_metric)) for name, o in sorted(tried.items())
            ),
            remaining_rounds=budget.max_rounds - round_index,
            remaining_simulations=ledger.remaining_simulations,
            default_scenario_id=scenarios[0].scenario_id,
        )

        proposals = proposer.propose(context, batch_size)
        if not proposals and isinstance(proposer, LLMProposer):
            # 离线回退：LLM 不可用时不该让闭环停摆（方案 §8 保底方案）
            proposals = HeuristicProposer().propose(context, batch_size)
        if not proposals:
            stop_reason = "no_more_hypotheses"
            break

        previous_best_value = _metric_of(best, primary_metric)
        outcomes: list[CandidateOutcome] = []
        for spec in proposals:
            try:
                ledger.check(needed=len(scenarios) * len(seeds))
            except BudgetExhausted as exc:
                stop_reason = exc.reason
                break
            outcome = evaluate_spec(spec, scenarios, seeds, ledger, store)
            outcomes.append(outcome)
            tried[spec.name] = outcome
            all_run_ids.extend(outcome.run_ids)

        if not outcomes:
            break

        # 本轮最好的候选
        round_best = min(outcomes, key=_rank)
        # 只有优于此前最好、且提升幅度达到阈值，才算「实质改善」——
        # 否则跨种子噪声也会被当成进展，循环会一直「有改善」而停不下来。
        gain = (
            _metric_of(round_best, primary_metric) - previous_best_value
        ) * (1.0 if METRIC_DIRECTIONS[primary_metric] == "max" else -1.0)
        improved = gain >= budget.min_improvement
        if _rank(round_best) < _rank(best) and gain > 0:
            best = round_best

        no_improvement_streak = 0 if improved else no_improvement_streak + 1
        rounds.append(
            RoundRecord(
                round_index=round_index,
                outcomes=outcomes,
                best_so_far=best.spec.name,
                best_value=_metric_of(best, primary_metric),
                improved=improved,
                simulations_used=ledger.simulations,
            )
        )
        round_index += 1

    if stop_reason == "completed" and round_index >= budget.max_rounds:
        stop_reason = "max_rounds"

    # --- 保留场景复验 -------------------------------------------------
    validation: Optional[ValidationResult] = None
    best_is_new = best.spec.name not in {o.spec.name for o in baseline_outcomes}
    if best_is_new and holdout_scenarios:
        try:
            ledger.check(needed=2 * len(holdout_scenarios) * len(seeds))
        except BudgetExhausted:
            stop_reason = "budget_before_validation"
        else:
            cand = evaluate_spec(best.spec, holdout_scenarios, seeds, ledger, store)
            base_holdout = evaluate_spec(
                baseline_best.spec, holdout_scenarios, seeds, ledger, store
            )
            all_run_ids.extend(cand.run_ids + base_holdout.run_ids)
            improvement = _metric_of(cand, primary_metric) - _metric_of(
                base_holdout, primary_metric
            )
            validation = ValidationResult(
                candidate_name=best.spec.name,
                scenario_ids=[s.scenario_id for s in holdout_scenarios],
                candidate_completion_rate=cand.session_completion_rate,
                best_baseline_name=base_holdout.spec.name,
                best_baseline_completion_rate=base_holdout.session_completion_rate,
                improvement=float(improvement),
                passed=improvement > 0,
                run_ids=cand.run_ids + base_holdout.run_ids,
            )

    conclusion, caveats = build_conclusion(
        best=best,
        baseline_best=baseline_best,
        validation=validation,
        primary_metric=primary_metric,
        primary_direction=METRIC_DIRECTIONS[primary_metric],
        stop_reason=stop_reason,
        budget=budget,
        n_candidates=len(tried) - len(baseline_outcomes),
        n_scenarios=len(scenarios),
        n_holdout=len(holdout_scenarios),
        n_simulations=ledger.simulations,
        elapsed_seconds=ledger.elapsed,
    )

    payload = json.dumps(
        {
            "scenarios": [s.scenario_hash for s in scenarios],
            "holdout": [s.scenario_hash for s in holdout_scenarios],
            "metric": primary_metric,
            "seeds": sorted(seeds),
            "budget": budget.model_dump(mode="json"),
            "schema_version": SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    return AgentReport(
        report_id="agent--" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10],
        created_at=dt.datetime.now(),
        primary_metric=primary_metric,
        primary_direction=METRIC_DIRECTIONS[primary_metric],
        budget=budget,
        stop_reason=stop_reason,
        rounds=rounds,
        baseline_outcomes=baseline_outcomes,
        best_candidate=best if best_is_new else None,
        validation=validation,
        conclusion=conclusion,
        caveats=caveats,
        simulations_used=ledger.simulations,
        elapsed_seconds=round(ledger.elapsed, 3),
        all_run_ids=sorted(set(all_run_ids)),
    )


def _metric_of(outcome: CandidateOutcome, metric: str) -> float:
    return float(getattr(outcome, metric))


# ----------------------------------------------------------------------
# 结论措辞
# ----------------------------------------------------------------------

#: 除非有严格求解器证明最优性，否则不得**断言**以下表述。
#: 注意：这些词出现在否定句里（「不是全局最优」）是**要求**的免责声明，不算违规。
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "全局最优",
    "最优解",
    "数学最优",
    "证明了最优",
    "guaranteed optimal",
    "globally optimal",
)

#: 否定标记。出现在被禁短语之前若干字符内即视为否定句。
_NEGATIONS: tuple[str, ...] = ("不", "非", "未", "无", "别", "莫", "cannot", "not ", "no ")


def claims_optimality(text: str, lookback: int = 5) -> list[str]:
    """找出**断言**最优性的片段。否定形式不算。

    朴素的子串检查会把「本结论不是全局最优」这句必需的免责声明也判成违规 ——
    那样的测试要么被绕过，要么逼着人删掉免责声明。所以必须区分断言与否定。
    """
    hits: list[str] = []
    for phrase in FORBIDDEN_PHRASES:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index == -1:
                break
            window = text[max(0, index - lookback) : index]
            if not any(neg in window for neg in _NEGATIONS):
                hits.append(text[max(0, index - 8) : index + len(phrase) + 6])
            start = index + len(phrase)
    return hits


#: 判断「某项指标变差」的阈值：相对 2% 或绝对 0.01，取较宽者，避免把噪声写成取舍。
_TRADEOFF_RELATIVE = 0.02
_TRADEOFF_ABSOLUTE = 0.01

_TRADEOFF_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("demand_satisfaction_rate", "max", "需求满足率（能量口径）"),
    ("energy_cost_cny", "min", "总用电成本"),
    ("peak_power_kw", "min", "峰值负荷"),
    ("constraint_violations", "min", "约束越限步数"),
)


def detect_tradeoffs(
    best: CandidateOutcome, baseline: CandidateOutcome
) -> list[str]:
    """找出候选相对基准**变差**的指标。

    只报主指标是不够的：实测 `estimated_departure-CF` 把按时完成率从 0.691 抬到 0.876，
    但需求满足率同时从 0.904 掉到 0.868 —— 它是把电量集中喂给「能喂饱」的车，
    总交付电量反而下降。这是一个真实取舍，必须写进结论，否则就是过度声明。
    """
    out: list[str] = []
    for field_name, direction, label in _TRADEOFF_FIELDS:
        candidate_value = float(getattr(best, field_name))
        baseline_value = float(getattr(baseline, field_name))
        worse = (
            candidate_value < baseline_value
            if direction == "max"
            else candidate_value > baseline_value
        )
        if not worse:
            continue
        scale = max(abs(baseline_value), 1e-9)
        if abs(candidate_value - baseline_value) / scale < _TRADEOFF_RELATIVE and abs(
            candidate_value - baseline_value
        ) < _TRADEOFF_ABSOLUTE:
            continue
        arrow = "降到" if direction == "max" else "升到"
        out.append(
            f"{label}由 {baseline_value:.4f} {arrow} {candidate_value:.4f}"
            f"（基准 {baseline.spec.name}）"
        )
    return out


def build_conclusion(
    best: CandidateOutcome,
    baseline_best: CandidateOutcome,
    validation: Optional[ValidationResult],
    primary_metric: str,
    primary_direction: str,
    stop_reason: str,
    budget: BudgetSpec,
    n_candidates: int,
    n_scenarios: int,
    n_holdout: int,
    n_simulations: int,
    elapsed_seconds: float,
) -> tuple[str, list[str]]:
    """生成结论文本与必须随附的限制说明。

    措辞纪律：只能说「在已测试范围与预算内的最佳可行候选」。方案 §6 明确要求
    不得声称全局最优 —— 我们没有严格求解器证明，任何「最优」的说法都是过度声明。
    """
    best_value = _metric_of(best, primary_metric)
    base_value = _metric_of(baseline_best, primary_metric)
    delta = best_value - base_value
    better = "高于" if delta > 0 else ("低于" if delta < 0 else "持平于")

    lines = [
        f"在已测试范围与预算内，表现最好的可行候选是 **{best.spec.name}**"
        f"（{primary_metric} = {best_value:.4f}），"
        f"{better}官方基准中最优的 {baseline_best.spec.name}（{base_value:.4f}），"
        f"差值 {delta:+.4f}。",
        f"搜索范围：{n_scenarios} 个实验场景，共评测 {n_candidates} 个候选策略，"
        f"消耗 {n_simulations} 次仿真、{elapsed_seconds:.1f} 秒。",
    ]

    if best.spec.rationale:
        lines.append(f"该候选的策略含义：{best.spec.rationale}")

    tradeoffs = detect_tradeoffs(best, baseline_best)
    if tradeoffs:
        lines.append(
            "**这是一次有代价的改进**，相对基准同时变差的指标："
            + "；".join(tradeoffs)
            + "。主目标的提升并非免费，选型时需按实际运营目标权衡。"
        )
    else:
        lines.append("未观测到相对基准明显变差的其他指标。")

    if validation is not None:
        verdict = "仍然胜出" if validation.passed else "未能保持优势"
        lines.append(
            f"在**未参与筛选**的保留场景（{', '.join(validation.scenario_ids)}）上复验，"
            f"该候选{verdict}：{validation.candidate_completion_rate:.4f} 对 "
            f"{validation.best_baseline_name} 的 "
            f"{validation.best_baseline_completion_rate:.4f}"
            f"（差值 {validation.improvement:+.4f}）。"
        )
    elif best.spec.baseline:
        lines.append(
            "未发现优于基准的候选 —— 在该范围内，官方基准算法已经足够好。"
            "这本身是有价值的结论：它说明新增调度逻辑没有带来可测量的收益。"
        )
    else:
        lines.append("未做保留场景复验，该候选的优势尚不能排除对实验集过拟合的可能。")

    caveats = [
        f"本结论只在已测试的 {n_scenarios} 个实验场景与给定预算内成立，"
        f"**不是全局最优**，也不构成对未测试场景的保证。",
        f"停止原因：{stop_reason}"
        f"（上限 {budget.max_rounds} 轮 / {budget.max_simulations} 次仿真 / "
        f"{budget.max_seconds:.0f} 秒）。",
    ]
    if n_holdout == 0:
        caveats.append("本次未提供保留场景，无法排除对实验集过拟合。")
    elif validation is None:
        caveats.append("候选未做保留场景复验，优势待确认。")
    elif not validation.passed:
        caveats.append(
            "候选在保留场景上的优势未获确认，报告引用时应标注为「待进一步验证」。"
        )
    if detect_tradeoffs(best, baseline_best):
        caveats.append(
            "该候选以牺牲其他指标为代价换取主目标提升，结论不应表述为「全面更优」。"
        )
    caveats.append(
        "候选策略来自受控的排序键词汇组合，不是任意代码；"
        "其行为差异全部来自分配优先级，不改变仿真物理模型，也不涉及真实桩控。"
    )

    return "\n".join(lines), caveats
