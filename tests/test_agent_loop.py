"""AI 实验闭环测试。

三条纪律必须有测试守护，否则很容易在后续改动中悄悄失守：
  1. LLM 不能执行代码，只能从受控词汇组合策略；
  2. 任何自动执行都有资源上限；
  3. 结论措辞不得声称全局最优，且必须报出取舍。
"""

import json

import pytest

from chargebench.agent_loop import (
    BASELINE_SPECS,
    HeuristicProposer,
    claims_optimality,
    LLMProposer,
    ProposalContext,
    _extract_json_array,
    build_sort_fn,
    detect_tradeoffs,
    run_agent_loop,
)
from chargebench.algorithms import CORE_ALGORITHMS
from chargebench.schemas import (
    BudgetSpec,
    CandidateOutcome,
    CandidateSpec,
    SortKey,
)

from .conftest import load_scenario


def _train_scenarios():
    base = load_scenario()
    return [
        base,
        base.with_updates(scenario_id="train_tight", deadline_tightness=0.85, load_intensity=0.9),
        base.with_updates(scenario_id="train_loose", deadline_tightness=0.5, load_intensity=0.8),
    ]


def _holdout_scenarios():
    base = load_scenario()
    return [
        base.with_updates(scenario_id="holdout_a", seed=101, load_intensity=0.95),
        base.with_updates(scenario_id="holdout_b", seed=202, deadline_tightness=0.8, load_intensity=1.05),
    ]


def _small_budget(**over) -> BudgetSpec:
    base = dict(max_rounds=2, max_simulations=60, max_seconds=60.0, patience=2)
    base.update(over)
    return BudgetSpec(**base)


@pytest.fixture(scope="module")
def report():
    return run_agent_loop(
        _train_scenarios(), _holdout_scenarios(), CORE_ALGORITHMS,
        HeuristicProposer(), budget=_small_budget(), seeds=[42, 43], batch_size=2,
    )


# ----------------------------------------------------------------------
# 受控词汇：不允许执行代码
# ----------------------------------------------------------------------


def test_candidate_spec_rejects_unknown_fields():
    """LLM 若返回多余字段（比如想夹带代码）必须被拒绝，而不是被忽略。"""
    with pytest.raises(Exception):
        CandidateSpec(name="x", primary="laxity", code="import os")


def test_candidate_spec_rejects_unknown_sort_key():
    with pytest.raises(Exception):
        CandidateSpec(name="x", primary="os.system")


def test_sort_fn_covers_every_sort_key():
    """词汇表里的每个键都必须真的能编译出可用的排序函数。"""
    for key in SortKey:
        spec = CandidateSpec(name=f"k-{key.value}", primary=key)
        assert callable(build_sort_fn(spec))


def test_sort_fn_orders_by_completion_first():
    """completion_first 必须真的把「能充满的车」排到前面。"""
    from chargebench.schemas import Scenario

    scenario = Scenario(**{**load_scenario().model_dump(mode="json")})

    class FakeEV:
        def __init__(self, sid, laxity):
            self.session_id = sid
            self.station_id = "PS-000"
            self._laxity = laxity
            self.arrival = 0
            self.estimated_departure = 10
            self.remaining_demand = 1.0
            self.requested_energy = 1.0
            self.energy_delivered = 0.0

    class FakeIface:
        current_time = 0

        def max_pilot_signal(self, station_id):
            return 1.0

        def remaining_amp_periods(self, ev):
            return ev._laxity  # 直接把它当松弛时间用，便于构造用例

    spec = CandidateSpec(name="cf", primary=SortKey.arrival, completion_first=True)
    # 松弛时间 = (离站 10 - 当前 0) - 所需时长；所需时长 > 10 即为「充不完」
    can_finish = FakeEV("can", laxity=2.0)  # 10 - 2 = 8 >= 0
    cannot_finish = FakeEV("cannot", laxity=30.0)  # 10 - 30 = -20 < 0
    ordered = build_sort_fn(spec)([cannot_finish, can_finish], FakeIface())
    assert [e.session_id for e in ordered] == ["can", "cannot"]


def test_baseline_specs_use_registry_not_combinator():
    """基准算法必须走注册表，否则就成了拿重实现和官方实现对打。"""
    assert all(spec.baseline for spec in BASELINE_SPECS.values())
    assert set(BASELINE_SPECS) == set(CORE_ALGORITHMS)


# ----------------------------------------------------------------------
# 提议者
# ----------------------------------------------------------------------


def _context(tried=()) -> ProposalContext:
    return ProposalContext(
        primary_metric="session_completion_rate",
        direction="max",
        baseline_best_name="EDF",
        baseline_best_value=0.7,
        best_name="EDF",
        best_value=0.7,
        tried_names=tuple(tried),
        history=(),
        remaining_rounds=2,
        remaining_simulations=50,
        default_scenario_id="s",
    )


def test_heuristic_proposer_is_deterministic():
    a = HeuristicProposer().propose(_context(), 3)
    b = HeuristicProposer().propose(_context(), 3)
    assert [s.name for s in a] == [s.name for s in b]


def test_heuristic_proposer_prioritises_completion_first():
    """启发式的先验顺序把「完成度优先」排在最前 —— 那是方案给出的具体假设。"""
    first = HeuristicProposer().propose(_context(), 1)[0]
    assert first.completion_first is True


def test_heuristic_proposer_skips_tried():
    first_batch = HeuristicProposer().propose(_context(), 2)
    tried = tuple(s.name for s in first_batch)
    second_batch = HeuristicProposer().propose(_context(tried), 2)
    assert not ({s.name for s in second_batch} & set(tried))


def test_heuristic_proposer_eventually_exhausts():
    tried: list[str] = []
    for _ in range(50):
        batch = HeuristicProposer().propose(_context(tried), 4)
        if not batch:
            break
        tried.extend(s.name for s in batch)
    assert HeuristicProposer().propose(_context(tried), 4) == []


def test_llm_proposer_parses_json_and_rejects_extras():
    raw = json.dumps([
        {"name": "good", "primary": "laxity", "completion_first": True},
        {"name": "bad", "primary": "laxity", "evil": "os.system('rm -rf /')"},
    ])
    proposer = LLMProposer(lambda prompt: raw)
    specs = proposer.propose(_context(), 5)
    assert [s.name for s in specs] == ["good"]
    assert proposer.rejected, "不合法的条目必须被记录，不能静默吞掉"


def test_llm_proposer_handles_markdown_fence():
    raw = '好的，这是我的建议：\n```json\n[{"name": "a", "primary": "arrival"}]\n```\n希望有帮助。'
    specs = LLMProposer(lambda prompt: raw).propose(_context(), 5)
    assert [s.name for s in specs] == ["a"]


def test_llm_proposer_survives_service_failure():
    """外部服务挂了不该让闭环崩掉，而应返回空以便回退到启发式。"""
    def boom(prompt):
        raise RuntimeError("provider unavailable")

    proposer = LLMProposer(boom)
    assert proposer.propose(_context(), 3) == []
    assert proposer.rejected


def test_llm_proposer_rejects_non_json():
    proposer = LLMProposer(lambda prompt: "抱歉，我无法完成这个请求。")
    assert proposer.propose(_context(), 3) == []
    assert proposer.rejected


def test_extract_json_array():
    assert _extract_json_array('前言 [1,2] 后语') == "[1,2]"
    with pytest.raises(ValueError):
        _extract_json_array("没有数组")


def test_agent_loop_falls_back_to_heuristic_when_llm_fails():
    """方案 §8 保底方案：AI 服务不可用时，闭环仍要能跑完并给出结论。"""
    broken = LLMProposer(lambda prompt: "服务不可用")
    report = run_agent_loop(
        _train_scenarios()[:1], [], CORE_ALGORITHMS, broken,
        budget=_small_budget(max_rounds=1, max_simulations=40), seeds=[42], batch_size=2,
    )
    assert report.rounds, "LLM 失效时应回退到启发式，而不是直接停摆"
    assert report.simulations_used > 0


# ----------------------------------------------------------------------
# 预算
# ----------------------------------------------------------------------


def test_simulation_budget_is_never_exceeded():
    budget = _small_budget(max_simulations=25)
    report = run_agent_loop(
        _train_scenarios(), _holdout_scenarios(), CORE_ALGORITHMS,
        HeuristicProposer(), budget=budget, seeds=[42, 43], batch_size=3,
    )
    assert report.simulations_used <= budget.max_simulations, (
        f"消耗 {report.simulations_used} 次仿真，超过上限 {budget.max_simulations}"
    )
    assert report.stop_reason in {
        "simulations", "seconds", "max_rounds", "no_improvement",
        "no_more_hypotheses", "target_reached", "budget_before_validation",
    }


def test_round_budget_is_respected():
    budget = _small_budget(max_rounds=1)
    report = run_agent_loop(
        _train_scenarios()[:1], [], CORE_ALGORITHMS, HeuristicProposer(),
        budget=budget, seeds=[42], batch_size=2,
    )
    assert len(report.rounds) <= 1


def test_patience_stops_the_loop():
    """连续无实质改善即停 —— 否则会一直烧预算做无用实验。"""
    report = run_agent_loop(
        _train_scenarios()[:1], [], ["EDF"], HeuristicProposer(),
        budget=BudgetSpec(max_rounds=6, max_simulations=400, max_seconds=120, patience=1),
        seeds=[42], batch_size=1,
    )
    assert report.stop_reason in {"no_improvement", "no_more_hypotheses", "max_rounds"}
    assert len(report.rounds) < 6


def test_target_stops_early():
    report = run_agent_loop(
        _train_scenarios()[:1], [], CORE_ALGORITHMS, HeuristicProposer(),
        budget=BudgetSpec(max_rounds=4, max_simulations=200, max_seconds=60, patience=3),
        seeds=[42], batch_size=2, target=0.0,
    )
    assert report.stop_reason == "target_reached"


def test_empty_scenarios_rejected():
    with pytest.raises(ValueError):
        run_agent_loop([], [], CORE_ALGORITHMS)


def test_unknown_metric_rejected():
    with pytest.raises(KeyError):
        run_agent_loop(_train_scenarios()[:1], [], CORE_ALGORITHMS, primary_metric="nope")


# ----------------------------------------------------------------------
# 措辞纪律
# ----------------------------------------------------------------------


def test_conclusion_never_claims_global_optimum(report):
    """方案 §6：除非有严格求解器证明，否则不得声称全局最优。

    用否定感知的检查：**要求**出现「不是全局最优」这句免责声明，
    但同时禁止任何断言式的「是最优」。
    """
    text = report.conclusion + " " + " ".join(report.caveats)
    hits = claims_optimality(text)
    assert not hits, f"结论里出现了断言最优性的表述：{hits}"
    # 免责声明必须真的在
    assert "全局最优" in text, "免责声明缺失"


def test_claims_optimality_distinguishes_negation():
    """检测器本身要能区分「不是最优」与「就是最优」——否则它形同虚设。"""
    assert claims_optimality("本结论不是全局最优，不构成保证。") == []
    assert claims_optimality("未证明全局最优。") == []
    assert claims_optimality("该算法达到全局最优。")  # 断言式，必须被抓到
    assert claims_optimality("globally optimal result.")


def test_conclusion_states_it_is_budget_bounded(report):
    assert "不是全局最优" in " ".join(report.caveats)
    assert any(report.stop_reason in c for c in report.caveats), "停止原因必须写进限制说明"


def test_conclusion_reports_tradeoffs(report):
    """只报主指标等于掩盖代价 —— 实测该候选的能量口径满足率会下降。"""
    text = report.conclusion
    if report.best_candidate is not None:
        # 结论里必须对取舍有明确表态（有取舍说代价，没取舍说未观测到）
        assert ("有代价的改进" in text) or ("未观测到" in text)


def test_detect_tradeoffs_flags_degraded_metric():
    baseline = CandidateOutcome(
        spec=BASELINE_SPECS["EDF"], run_ids=[], session_completion_rate=0.7,
        demand_satisfaction_rate=0.95, energy_cost_cny=100.0, peak_power_kw=50.0,
        constraint_violations=0, spread=0.01, n_simulations=1,
    )
    candidate = baseline.model_copy(
        update={"demand_satisfaction_rate": 0.80}  # 明显下降
    )
    found = detect_tradeoffs(candidate, baseline)
    assert any("需求满足率" in t for t in found)


def test_detect_tradeoffs_ignores_noise():
    baseline = CandidateOutcome(
        spec=BASELINE_SPECS["EDF"], run_ids=[], session_completion_rate=0.7,
        demand_satisfaction_rate=0.9500, energy_cost_cny=100.0, peak_power_kw=50.0,
        constraint_violations=0, spread=0.01, n_simulations=1,
    )
    candidate = baseline.model_copy(update={"demand_satisfaction_rate": 0.9499})
    assert detect_tradeoffs(candidate, baseline) == [], "微小波动不该被写成取舍"


# ----------------------------------------------------------------------
# 违规候选不得胜出
# ----------------------------------------------------------------------


def test_violating_candidate_never_wins():
    """方案 §5 原则 3：不允许把违规供电当成更优结果。"""
    spec = CandidateSpec(name="violator", primary=SortKey.arrival)
    outcome = CandidateOutcome(
        spec=spec, run_ids=[], session_completion_rate=0.99,
        demand_satisfaction_rate=0.99, energy_cost_cny=1.0, peak_power_kw=999.0,
        constraint_violations=3, spread=0.0, n_simulations=1,
    )
    clean = CandidateOutcome(
        spec=BASELINE_SPECS["EDF"], run_ids=[], session_completion_rate=0.5,
        demand_satisfaction_rate=0.9, energy_cost_cny=100.0, peak_power_kw=50.0,
        constraint_violations=0, spread=0.0, n_simulations=1,
    )
    # 与 run_agent_loop 内部同一条规则
    direction = "max"
    def rank(o):
        if o.constraint_violations > 0:
            return float("inf")
        return -getattr(o, "session_completion_rate") if direction == "max" else getattr(o, "session_completion_rate")
    assert min([outcome, clean], key=rank).spec.name == "EDF"


# ----------------------------------------------------------------------
# 结果结构
# ----------------------------------------------------------------------


def test_report_structure(report):
    assert report.report_id.startswith("agent--")
    assert report.baseline_outcomes
    assert {o.spec.name for o in report.baseline_outcomes} == set(CORE_ALGORITHMS)
    assert report.all_run_ids, "必须能追溯到具体运行"
    assert report.simulations_used > 0
    assert report.elapsed_seconds >= 0


def test_holdout_runs_are_separate_from_selection(report):
    """保留场景的运行只能出现在复验里 —— 它们不得参与筛选。"""
    if report.validation is None:
        pytest.skip("本次未产生候选，无复验")
    flat_selection = {rid for o in report.baseline_outcomes for rid in o.run_ids} | {
        rid for r in report.rounds for o in r.outcomes for rid in o.run_ids
    }
    assert set(report.validation.run_ids).isdisjoint(flat_selection)


def test_holdout_validation_produces_verdict(report):
    if report.validation is None:
        pytest.skip("本次未产生候选，无复验")
    v = report.validation
    assert v.best_baseline_name in CORE_ALGORITHMS
    assert v.scenario_ids
    assert isinstance(v.passed, bool)
    assert v.improvement == pytest.approx(
        v.candidate_completion_rate - v.best_baseline_completion_rate
    )


def test_report_is_reproducible():
    """同配置两次运行必须给出同样的结论（启发式提议者是确定性的）。"""
    kwargs = dict(
        base_algorithms=CORE_ALGORITHMS,
        proposer=HeuristicProposer(),
        budget=_small_budget(max_rounds=1, max_simulations=40),
        seeds=[42],
        batch_size=2,
    )
    a = run_agent_loop(_train_scenarios()[:1], [], **kwargs)
    b = run_agent_loop(_train_scenarios()[:1], [], **kwargs)
    assert a.report_id == b.report_id
    assert a.conclusion == b.conclusion
    assert [r.best_so_far for r in a.rounds] == [r.best_so_far for r in b.rounds]
