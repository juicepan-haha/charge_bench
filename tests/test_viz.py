"""可视化测试。

图表是交付给评委看的产物，因此至少要保证：能画出来、文件写到盘上、
标签不会退化成豆腐块（缺中文字体时自动切英文而不是画成空框）。
"""

import warnings

import pytest

from chargebench.algorithms import CORE_ALGORITHMS
from chargebench.schemas import SweepResult
from chargebench.sweep import rerank_sweep, run_sweep
from chargebench.timeseries import simulate_curves
from chargebench.viz import (
    _setup_fonts,
    axis_label,
    metric_label,
    plot_load_curve,
    plot_load_curves_side_by_side,
    plot_metric_panels,
    plot_objective_comparison,
    plot_winner_map,
    render_objective_comparison,
    render_sweep,
    save,
)

from .conftest import load_scenario


@pytest.fixture(scope="module")
def sweep() -> SweepResult:
    base = load_scenario()
    return run_sweep(
        base, [0.5, 0.9, 1.3], [0.4, 0.7, 0.85], CORE_ALGORITHMS, [42, 43]
    )


# ----------------------------------------------------------------------
# 字体与标签
# ----------------------------------------------------------------------


def test_font_setup_returns_a_known_language():
    """返回 zh 或 en。若中文字体缺失必须退回 en，而不是画成豆腐块。"""
    assert _setup_fonts() in {"zh", "en"}


def test_metric_label_never_empty():
    for lang in ("zh", "en"):
        for metric in (
            "session_completion_rate",
            "peak_power_kw",
            "energy_cost_cny",
            "demand_satisfaction_rate",
            "worst_delivery_ratio",
        ):
            assert metric_label(metric, lang)
    # 未登记的指标回退到原始名，而不是空白
    assert metric_label("brand_new_metric", "zh") == "brand_new_metric"


def test_axis_label_falls_back():
    assert axis_label("load_intensity", "zh") == "负荷强度"
    assert axis_label("unknown_axis", "zh") == "unknown_axis"


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------


def test_winner_map_renders(sweep: SweepResult):
    fig = plot_winner_map(sweep)
    assert fig.axes, "应至少有一个坐标轴"
    assert "算法适用区域" in fig.axes[0].get_title() or "suitability" in fig.axes[0].get_title()


def test_metric_panels_has_one_axis_per_algorithm(sweep: SweepResult):
    fig = plot_metric_panels(sweep)
    data_axes = [ax for ax in fig.axes if ax.get_images()]
    assert len(data_axes) == len(CORE_ALGORITHMS)
    titles = {ax.get_title() for ax in data_axes}
    assert titles == set(CORE_ALGORITHMS)


def test_objective_comparison_renders(sweep: SweepResult):
    peak = rerank_sweep(sweep, "peak_power_kw")
    fig = plot_objective_comparison([sweep, peak])
    data_axes = [ax for ax in fig.axes if ax.get_images()]
    assert len(data_axes) == 2


def test_objective_comparison_requires_input():
    with pytest.raises(ValueError, match="至少需要一个"):
        plot_objective_comparison([])


def test_single_algorithm_sweep_still_renders():
    """只有一个算法时也必须能出图，不能因为图例宽度为零而崩。"""
    base = load_scenario()
    single = run_sweep(base, [0.5, 0.9], [0.5, 0.85], ["EDF"], [42])
    assert plot_winner_map(single).axes
    assert plot_metric_panels(single).axes


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------


def test_render_sweep_writes_two_files(sweep: SweepResult, tmp_path):
    written = render_sweep(sweep, tmp_path)
    assert len(written) == 2
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 5_000, "PNG 太小，可能画成了空白图"
        assert path.suffix == ".png"
    assert any("winner-map" in p.name for p in written)
    assert any("metric-panels" in p.name for p in written)


def test_render_objective_comparison_writes_file(sweep: SweepResult, tmp_path):
    peak = rerank_sweep(sweep, "peak_power_kw")
    path = render_objective_comparison([sweep, peak], tmp_path)
    assert path.exists()
    assert path.stat().st_size > 5_000
    assert "session_completion_rate" in path.name
    assert "peak_power_kw" in path.name


def test_render_creates_missing_directory(sweep: SweepResult, tmp_path):
    target = tmp_path / "deep" / "nested"
    written = render_sweep(sweep, target)
    assert all(p.exists() for p in written)


def test_save_returns_path(sweep: SweepResult, tmp_path):
    path = save(plot_winner_map(sweep), tmp_path / "x" / "fig.png")
    assert path.exists() and path.name == "fig.png"


# ----------------------------------------------------------------------
# 豆腐块检测
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_figure",
    [
        lambda s: plot_winner_map(s),
        lambda s: plot_metric_panels(s),
        lambda s: plot_objective_comparison([s, rerank_sweep(s, "peak_power_kw")]),
    ],
)
def test_chinese_labels_actually_render(sweep: SweepResult, tmp_path, make_figure):
    """中文字符串存在 ≠ 渲染成功。

    matplotlib 在**绘制时**（savefig）才发出缺字形警告，所以必须在 savefig 期间捕获。
    断言标题里有中文是没用的 —— 字符串永远是中文，渲染成豆腐块照样通过。
    这条测试专门防这个：曾出现整张图的中文全是空框，而所有断言仍然全绿。
    """
    fig = make_figure(sweep)  # 不传 lang，走自动检测
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save(fig, tmp_path / "render-check.png")

    missing = [str(w.message) for w in caught if "missing from font" in str(w.message)]
    assert not missing, (
        f"有 {len(missing)} 个字形渲染失败，图上的中文会显示为空框。"
        f"首个：{missing[0] if missing else ''}"
    )


def test_load_curve_labels_render(sweep: SweepResult, tmp_path):
    """负荷曲线的中文标签同样要真的画得出来。"""
    from chargebench.timeseries import simulate_curves

    curves = simulate_curves(load_scenario(), "EDF")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save(plot_load_curve(curves), tmp_path / "curve.png")
    missing = [w for w in caught if "missing from font" in str(w.message)]
    assert not missing, f"负荷曲线有 {len(missing)} 个字形渲染失败"


def test_resolve_lang_falls_back_to_english_without_cjk_font(monkeypatch):
    """没有中文字体时必须退回英文标签（渲染得出来），而不是硬画中文（豆腐块）。"""
    import chargebench.viz as viz

    monkeypatch.setattr(viz, "_FONT_LANG", "en")
    assert viz._resolve_lang("zh") == "en", "缺中文字体时不该硬用中文标签"
    assert viz._resolve_lang("en") == "en"


def test_resolve_lang_honours_explicit_english():
    import chargebench.viz as viz

    assert viz._resolve_lang("en") == "en"


# ----------------------------------------------------------------------
# i18n 纪律
# ----------------------------------------------------------------------


def _all_texts(fig) -> list[str]:
    """收集图里所有会画出来的文字（标题、轴标签、图例、刻度、注释）。"""
    out: list[str] = []
    for ax in fig.axes:
        out.append(ax.get_title())
        out.append(ax.get_xlabel())
        out.append(ax.get_ylabel())
        out.extend(t.get_text() for t in ax.texts)
        legend = ax.get_legend()
        if legend is not None:
            out.extend(t.get_text() for t in legend.get_texts())
        out.extend(t.get_text() for t in ax.get_xticklabels())
        out.extend(t.get_text() for t in ax.get_yticklabels())
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        out.append(suptitle.get_text())
    for leg in fig.legends:
        out.extend(t.get_text() for t in leg.get_texts())
    return [t for t in out if t]


def _cjk_in(text: str) -> list[str]:
    return [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]


@pytest.mark.parametrize(
    "make_figure",
    [
        lambda s: plot_winner_map(s, lang="en"),
        lambda s: plot_metric_panels(s, lang="en"),
        lambda s: plot_objective_comparison([s, rerank_sweep(s, "peak_power_kw")], lang="en"),
        lambda s: plot_load_curve(simulate_curves(load_scenario(), "EDF"), lang="en"),
        lambda s: plot_load_curves_side_by_side(
            [simulate_curves(load_scenario(), a) for a in ("FCFS", "EDF")], lang="en"
        ),
    ],
)
def test_english_mode_contains_no_chinese(sweep: SweepResult, make_figure):
    """英文模式下不得出现任何中文。

    硬编码的中文字符串会绕过标签字典 —— 在缺少中文字体的机器上被强行画出，
    结果是空框。这条测试强制所有会显示的文字都走标签字典。
    """
    fig = make_figure(sweep)
    offenders = []
    for text in _all_texts(fig):
        bad = _cjk_in(text)
        if bad:
            offenders.append(f"{text!r} 含 {''.join(bad[:6])}")
    assert not offenders, "英文模式下出现了中文（硬编码字符串）：\n" + "\n".join(offenders)


def test_chinese_mode_has_chinese_where_expected(sweep: SweepResult):
    """反向确认：中文模式下确实有中文，避免上一条测试因为「两边都没字」而空过。"""
    fig = plot_winner_map(sweep, lang="zh")
    assert any(_cjk_in(t) for t in _all_texts(fig))


# ----------------------------------------------------------------------
# Agent 闭环图
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent_report():
    from chargebench.agent_loop import HeuristicProposer, run_agent_loop
    from chargebench.schemas import BudgetSpec

    base = load_scenario()
    return run_agent_loop(
        [base, base.with_updates(scenario_id="t_loose", load_intensity=0.8)],
        [base.with_updates(scenario_id="h_a", seed=101)],
        CORE_ALGORITHMS, HeuristicProposer(),
        budget=BudgetSpec(max_rounds=1, max_simulations=40, max_seconds=60, patience=1),
        seeds=[42], batch_size=2,
    )


def test_agent_report_chart_renders(agent_report):
    from chargebench.viz import plot_agent_report

    fig = plot_agent_report(agent_report)
    data_axes = [ax for ax in fig.axes if ax.get_xlabel() or ax.get_ylabel()]
    assert len(data_axes) >= 2, "应有候选对比与逐轮收敛两张子图"


def test_agent_report_chart_labels_render(agent_report, tmp_path):
    from chargebench.viz import plot_agent_report

    fig = plot_agent_report(agent_report)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save(fig, tmp_path / "agent.png")
    missing = [w for w in caught if "missing from font" in str(w.message)]
    assert not missing, f"Agent 图有 {len(missing)} 个字形渲染失败"


def test_agent_report_chart_english_mode_has_no_chinese(agent_report):
    from chargebench.viz import plot_agent_report

    fig = plot_agent_report(agent_report, lang="en")
    offenders = [t for t in _all_texts(fig) if _cjk_in(t)]
    assert not offenders, f"英文模式下出现中文：{offenders[:3]}"


def test_agent_report_chart_without_rounds():
    """没有候选轮次时也必须能出图，而不是崩在 None 上。"""
    from chargebench.agent_loop import HeuristicProposer, run_agent_loop
    from chargebench.schemas import BudgetSpec
    from chargebench.viz import plot_agent_report

    base = load_scenario()
    report = run_agent_loop(
        [base], [], CORE_ALGORITHMS, HeuristicProposer(),
        budget=BudgetSpec(max_rounds=1, max_simulations=3, max_seconds=60, patience=1),
        seeds=[42], batch_size=1,
    )
    assert plot_agent_report(report).axes


def test_render_agent_report_writes_file(agent_report, tmp_path):
    from chargebench.viz import render_agent_report

    path = render_agent_report(agent_report, tmp_path)
    assert path.exists()
    assert path.stat().st_size > 10_000
    assert agent_report.report_id in path.name
