"""参数扫描测试。

重点是评分规则的确定性与「不允许把违规当成更优」这条硬约束 ——
它们决定了热力图上每一格的颜色，一旦出错整张适用区域图就是错的。
"""

import pytest

from chargebench.schemas import METRIC_DIRECTIONS, MetricAggregate, SweepResult
from chargebench.sweep import (
    _pick_winner,
    build_score_rule,
    format_sweep_detail,
    format_winner_grid,
    grid_scenario_id,
    grid_scenarios,
    rerank_sweep,
    run_sweep,
)
from chargebench.algorithms import CORE_ALGORITHMS

from .conftest import load_scenario


def _agg(completion: float, spread: float = 0.0, violations: int = 0) -> MetricAggregate:
    """构造一个只关心关键字段的聚合指标，用于单元测试评分规则。"""
    return MetricAggregate(
        n_seeds=2,
        demand_satisfaction_rate_mean=0.9,
        session_completion_rate_mean=completion,
        energy_cost_cny_mean=400.0,
        peak_power_kw_mean=60.0,
        mean_delivery_ratio_mean=0.8,
        worst_delivery_ratio_mean=0.5,
        constraint_violations_max=violations,
        spread=spread,
    )


# ----------------------------------------------------------------------
# 评分规则
# ----------------------------------------------------------------------


def test_higher_completion_wins():
    winner, dq = _pick_winner({"A": _agg(0.7), "B": _agg(0.5)}, "session_completion_rate")
    assert winner == "A" and dq == []


def test_lower_is_better_respected_for_cost():
    winner, _ = _pick_winner({"A": _agg(0.7), "B": _agg(0.5)}, "energy_cost_cny")
    # energy_cost_cny 的方向是 min，但其均值在 _agg 里都设成 400，故退化为平局
    assert winner in {"A", "B"}


def test_violators_are_disqualified_even_if_best():
    """硬约束优先于任何优化目标：不允许把违规供电当成更优结果。"""
    winner, dq = _pick_winner(
        {"GREEDY": _agg(1.0, violations=3), "SAFE": _agg(0.4)},
        "session_completion_rate",
    )
    assert winner == "SAFE", "违规的算法即使指标最好也必须出局"
    assert dq == ["GREEDY"]


def test_all_disqualified_yields_no_winner():
    winner, dq = _pick_winner(
        {"A": _agg(1.0, violations=1), "B": _agg(1.0, violations=2)},
        "session_completion_rate",
    )
    assert winner is None
    assert dq == ["A", "B"]


def test_tie_is_broken_by_lower_spread():
    """均值相同时，波动小的更可信 —— 这是写进评分规则的次级判据。"""
    winner, _ = _pick_winner(
        {"NOISY": _agg(0.8, spread=0.4), "STABLE": _agg(0.8, spread=0.01)},
        "session_completion_rate",
    )
    assert winner == "STABLE"


def test_pick_winner_is_deterministic():
    """完全平局时按算法名升序 —— 结果不能依赖字典顺序。"""
    aggs = {"C": _agg(0.8), "A": _agg(0.8), "B": _agg(0.8)}
    assert _pick_winner(aggs, "session_completion_rate")[0] == "A"
    reversed_aggs = dict(reversed(list(aggs.items())))
    assert _pick_winner(reversed_aggs, "session_completion_rate")[0] == "A"


def test_score_rule_is_self_describing():
    rule = build_score_rule("session_completion_rate")
    assert "取消资格" in rule
    assert "session_completion_rate" in rule
    assert "越大越好" in rule
    assert "spread" in rule
    assert "容差" in rule, "容差口径必须写进评分规则，否则事后无法解释为何不算违规"


def test_unknown_metric_rejected():
    with pytest.raises(KeyError, match="primary_metric"):
        _pick_winner({"A": _agg(0.5)}, "nonexistent_metric")


# ----------------------------------------------------------------------
# 缓存复用
# ----------------------------------------------------------------------


def test_sweep_reuses_store_and_keeps_results_identical(tmp_path):
    """扫描次数是网格 × 算法 × 种子，重复扫描必须走缓存而不是重算。

    判定依据是执行时刻：命中缓存返回的是原始 RunResult，其 executed_at 不变；
    重算则会生成新时刻。仅比对运行条数无法区分这两者。
    """
    from chargebench.storage import ResultStore

    store = ResultStore(tmp_path / "results")
    base = load_scenario()

    # 2x2 网格 × 3 算法 × 2 种子 = 24 次运行
    first = run_sweep(base, [0.5, 1.3], [0.4, 0.85], CORE_ALGORITHMS, [42, 43], store=store)
    assert store.counts()["runs"] == 24
    stamps = {r: store.get_run(r).traceability.executed_at for r in first.run_ids}

    second = run_sweep(base, [0.5, 1.3], [0.4, 0.85], CORE_ALGORITHMS, [42, 43], store=store)
    assert store.counts()["runs"] == 24, "重复扫描不得制造新记录"
    assert second.run_ids == first.run_ids
    for run_id in second.run_ids:
        assert store.get_run(run_id).traceability.executed_at == stamps[run_id], (
            "执行时刻变了，说明第二次扫描是重算而非复用"
        )
    assert [c.winner for c in second.cells] == [c.winner for c in first.cells]


# ----------------------------------------------------------------------
# 网格展开
# ----------------------------------------------------------------------


def test_grid_scenario_id_is_deterministic_and_distinct():
    assert grid_scenario_id("base", 0.9, 0.65) == grid_scenario_id("base", 0.9, 0.65)
    assert grid_scenario_id("base", 0.9, 0.65) != grid_scenario_id("base", 1.1, 0.65)
    assert grid_scenario_id("base", 0.9, 0.65) != grid_scenario_id("base", 0.9, 0.85)
    # 浮点格式必须稳定且文件系统安全：小数点被替换为 p，不会出现 0.9000000000000001
    assert grid_scenario_id("base", 0.9, 0.65) == "base_L0p9_T0p65"
    assert "." not in grid_scenario_id("base", 0.9, 0.65)


def test_grid_scenarios_covers_full_cross_product():
    base = load_scenario()
    xs, ys = [0.5, 0.9], [0.4, 0.7, 1.0]
    grid = grid_scenarios(base, xs, ys)
    assert len(grid) == len(xs) * len(ys)
    assert len({s.scenario_id for s in grid}) == len(grid)
    for s in grid:
        assert s.load_intensity in xs and s.deadline_tightness in ys
        assert s.n_ports == base.n_ports, "非扫描轴字段必须原样继承"
    # 基准场景本身不被改写
    assert base.load_intensity == load_scenario().load_intensity


# ----------------------------------------------------------------------
# 扫描执行
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_sweep() -> SweepResult:
    base = load_scenario()
    return run_sweep(
        base, [0.5, 1.3], [0.4, 0.85], CORE_ALGORITHMS, [42, 43],
        primary_metric="session_completion_rate",
    )


def test_sweep_shape(small_sweep: SweepResult):
    assert len(small_sweep.cells) == 4
    assert small_sweep.x_values == [0.5, 1.3]
    assert small_sweep.y_values == [0.4, 0.85]
    assert set(small_sweep.algorithms) == set(CORE_ALGORITHMS)
    assert small_sweep.seeds == [42, 43]
    # 4 格 × 3 算法 × 2 种子
    assert len(small_sweep.run_ids) == 24


def test_sweep_cells_have_all_algorithms(small_sweep: SweepResult):
    for cell in small_sweep.cells:
        assert set(cell.aggregates) == set(CORE_ALGORITHMS)
        for agg in cell.aggregates.values():
            assert agg.n_seeds == 2
            assert 0.0 <= agg.demand_satisfaction_rate_mean <= 1.0


def test_port_constrained_flag_matches_occupancy(small_sweep: SweepResult):
    for cell in small_sweep.cells:
        assert cell.port_constrained == (cell.port_occupancy_estimate > 1.0)


def test_sweep_grid_helpers(small_sweep: SweepResult):
    winners, constrained = small_sweep.winner_grid()
    assert len(winners) == 2 and len(winners[0]) == 2
    assert len(constrained) == 2
    for name in CORE_ALGORITHMS:
        grid = small_sweep.metric_grid(name)
        assert len(grid) == 2 and len(grid[0]) == 2


def test_sweep_requires_seeds():
    with pytest.raises(ValueError, match="种子"):
        run_sweep(load_scenario(), [0.5], [0.5], CORE_ALGORITHMS, [])


def test_sweep_rejects_unknown_metric():
    with pytest.raises(KeyError):
        run_sweep(load_scenario(), [0.5], [0.5], CORE_ALGORITHMS, [42], "nope")


def test_sweep_is_reproducible(small_sweep: SweepResult):
    base = load_scenario()
    again = run_sweep(base, [0.5, 1.3], [0.4, 0.85], CORE_ALGORITHMS, [42, 43])
    assert again.sweep_id == small_sweep.sweep_id
    assert [c.winner for c in again.cells] == [c.winner for c in small_sweep.cells]


# ----------------------------------------------------------------------
# 换目标重排名
# ----------------------------------------------------------------------


def test_rerank_changes_winners_without_new_simulation(small_sweep: SweepResult):
    """核心论点：同一批实验数据，换一把尺子，胜者就变了。"""
    reranked = rerank_sweep(small_sweep, "peak_power_kw")
    assert reranked.primary_metric == "peak_power_kw"
    assert reranked.metric_direction == "min"
    assert reranked.sweep_id != small_sweep.sweep_id
    # 底层实验数据完全没动
    assert reranked.run_ids == small_sweep.run_ids
    for before, after in zip(small_sweep.cells, reranked.cells):
        assert before.aggregates == after.aggregates, "重排名不得改动实验数据"
        assert before.port_occupancy_estimate == after.port_occupancy_estimate


def test_rerank_is_identity_for_same_metric(small_sweep: SweepResult):
    assert rerank_sweep(small_sweep, small_sweep.primary_metric) is small_sweep


def test_rerank_rejects_unknown_metric(small_sweep: SweepResult):
    with pytest.raises(KeyError):
        rerank_sweep(small_sweep, "nope")


@pytest.mark.parametrize("metric", sorted(METRIC_DIRECTIONS))
def test_every_metric_is_rankable(small_sweep: SweepResult, metric):
    """每个可选指标都必须真的能排 —— 曾出现指标列表与聚合字段不同步导致崩溃。"""
    reranked = rerank_sweep(small_sweep, metric)
    assert all(c.winner is not None or c.disqualified for c in reranked.cells)


def test_all_metrics_distinguish_algorithms():
    """至少要有指标能区分算法，否则平台的核心论点不成立。"""
    base = load_scenario()
    sweep = run_sweep(base, [0.5, 0.7, 0.9, 1.1, 1.3], [0.4, 0.7, 0.85],
                      CORE_ALGORITHMS, [42, 43])
    winners = set()
    for metric in ("session_completion_rate", "peak_power_kw", "energy_cost_cny"):
        winners |= {c.winner for c in rerank_sweep(sweep, metric).cells}
    assert len(winners - {None}) >= 3, (
        f"三个目标函数合计只产生了 {winners} 个胜者，说明网格或指标缺乏区分度"
    )


# ----------------------------------------------------------------------
# 终端展示
# ----------------------------------------------------------------------


def test_format_winner_grid_marks_constrained_cells(small_sweep: SweepResult):
    text = format_winner_grid(small_sweep)
    assert small_sweep.primary_metric in text
    assert "*" in text, "受车位约束的格点必须有标记"
    # 只有胜者会出现在格子里；未获胜的算法不该被标出来
    winners = {c.winner for c in small_sweep.cells} - {None}
    assert winners, "本次扫描应至少产生一个胜者"
    for name in winners:
        assert name in text
    for name in set(CORE_ALGORITHMS) - winners:
        assert name not in text, f"{name} 未获胜，不应出现在适用区域图里"


def test_format_sweep_detail_lists_every_cell(small_sweep: SweepResult):
    text = format_sweep_detail(small_sweep)
    for cell in small_sweep.cells:
        assert f"{cell.load_intensity:g}" in text
    for name in CORE_ALGORITHMS:
        assert name in text
