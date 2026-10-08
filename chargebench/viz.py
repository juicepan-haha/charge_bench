"""适用边界可视化。

三张图对应三个层次的结论：

1. ``plot_winner_map``      —— 算法适用区域图。每个格点标出胜者。
2. ``plot_metric_panels``   —— 每个算法的主目标取值小图，看性能如何随参数连续变化。
3. ``plot_objective_comparison`` —— **最有说服力的一张**：同一批实验数据、
   换一个优化目标，胜者地图完全不同。这是平台存在的理由的直接证据。

中文字体：优先使用 Noto Sans CJK，缺失时自动退回英文标签（不画成豆腐块）。
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib
import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from .schemas import SweepResult

#: 算法配色（超出 4 个算法时循环）
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]

_CJK_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "Droid Sans Fallback",
)

_LABELS = {
    "zh": {
        "x": "负荷强度（请求电量 / 可供电量）",
        "y": "时间紧迫度（所需充电时长 / 可停留时长）",
        "winner": "算法适用区域",
        "constrained": "车位占用率 > 1\n（丢弃显著，结论需谨慎）",
        "legend": "胜者",
        "no_winner": "全部取消资格",
        "score": "主目标",
        "direction_max": "越大越好",
        "direction_min": "越小越好",
        "seeds": "种子数",
        "grid": "网格",
        "objective": "优化目标",
        "objective_headline": "同一批数据，换一个优化目标，胜者就变了 — 这正是平台存在的理由",
        "winner_outline": "白框 = 该算法在此格点胜出",
        "axis": {
            "load_intensity": "负荷强度",
            "deadline_tightness": "时间紧迫度",
        },
        "metric": {
            "demand_satisfaction_rate": "需求满足率（能量口径）",
            "session_completion_rate": "按时完成率（车辆口径）",
            "mean_delivery_ratio": "平均单车交付比",
            "worst_delivery_ratio": "最差单车交付比",
            "energy_cost_cny": "总用电成本",
            "peak_power_kw": "峰值负荷",
        },
    },
    "en": {
        "x": "Load intensity (demand / supply)",
        "y": "Deadline tightness (required / available time)",
        "winner": "Algorithm suitability map",
        "constrained": "Port occupancy > 1\n(drops significant)",
        "legend": "Winner",
        "no_winner": "all disqualified",
        "score": "Primary metric",
        "direction_max": "higher is better",
        "direction_min": "lower is better",
        "seeds": "seeds",
        "grid": "grid",
        "objective": "Objective",
        "objective_headline": "Same experiments, different objective, different winner",
        "winner_outline": "white outline = this algorithm won that cell",
        "axis": {
            "load_intensity": "Load intensity",
            "deadline_tightness": "Deadline tightness",
        },
        "metric": {},
    },
}


def metric_label(metric: str, lang: str) -> str:
    """指标的显示名。缺失时回退到原始字段名，不会画成空白。"""
    return _LABELS[lang]["metric"].get(metric, metric)


def axis_label(field: str, lang: str) -> str:
    """扫描轴的显示名（短名，用于小图的轴标题）。"""
    return _LABELS[lang]["axis"].get(field, field)


def _setup_fonts() -> str:
    """选一个可用的中日韩字体；返回实际使用的语言标签键。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in _CJK_CANDIDATES:
        if candidate in available:
            rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            return "zh"
    return "en"


_FONT_LANG: str | None = None


def _resolve_lang(lang: str | None) -> str:
    """确定标签语言，并保证字体已就位。

    绘图函数必须自己调用本函数，不能依赖调用方先设置字体 —— CLI 走 render_* 包装
    会设置，但 Streamlit 与测试是直接调 plot_* 的。

    曾因此踩坑：整张图的中文渲染成豆腐块，而断言只检查「标题里有中文字符串」——
    字符串永远是中文，豆腐块照样通过，测试全绿但图是废的。
    所以这里做两件事：字体检测结果缓存一次；请求中文但没有中文字体时强制退回英文标签。
    """
    global _FONT_LANG
    if _FONT_LANG is None:
        _FONT_LANG = _setup_fonts()
    if lang is None:
        return _FONT_LANG
    if lang == "zh" and _FONT_LANG != "zh":
        return "en"
    return lang


#: 无胜者格点的底色
_BAD_COLOR = "#DDDDDD"


def _with_bad_color(cmap):
    """给「无胜者」格点设置底色。

    matplotlib 已弃用 ``Colormap.set_bad``，改用 ``with_extremes``；
    但旧版没有该方法，故做一次兼容回退，避免为了一个配色崩掉整张图。
    """
    if hasattr(cmap, "with_extremes"):
        return cmap.with_extremes(bad=_BAD_COLOR)
    cmap.set_bad(_BAD_COLOR)
    return cmap


def _winner_colors(sweep: SweepResult) -> dict[str, str]:
    return {
        name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(sweep.algorithms)
    }


def _annotate(ax, text: str, x: float, y: float, color: str = "white", size: int = 9, weight: str = "bold"):
    ax.text(
        x, y, text, ha="center", va="center", color=color,
        fontsize=size, fontweight=weight,
    )


# ----------------------------------------------------------------------
# 图 1：算法适用区域图
# ----------------------------------------------------------------------


def plot_winner_map(sweep: SweepResult, ax=None, lang: str | None = None):
    """绘制算法适用区域图。受车位约束的格点加灰色边框与斜线提示。"""
    lang = _resolve_lang(lang)
    L = _LABELS[lang]
    winners, constrained = sweep.winner_grid()
    colors = _winner_colors(sweep)

    # 胜者 → 整数索引；无胜者用 -1
    index_of = {name: i for i, name in enumerate(sweep.algorithms)}
    grid = np.array(
        [[-1 if c is None else index_of[c] for c in row] for row in winners],
        dtype=float,
    )

    if ax is None:
        fig = Figure(figsize=(7.2, 5.4), layout="constrained")
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    cmap = _with_bad_color(ListedColormap([colors[name] for name in sweep.algorithms]))
    masked = np.ma.masked_where(grid < 0, grid)

    ax.imshow(
        masked, cmap=cmap, vmin=0, vmax=len(sweep.algorithms) - 1,
        origin="lower", aspect="auto",
    )

    for row_i, row in enumerate(winners):
        for col_i, name in enumerate(row):
            cell_label = name or "—"
            if name is None:
                _annotate(ax, cell_label, col_i, row_i, color="#666666")
            else:
                _annotate(ax, cell_label, col_i, row_i)
            if constrained[row_i][col_i]:
                ax.add_patch(
                    Rectangle(
                        (col_i - 0.5, row_i - 0.5), 1, 1, fill=False,
                        edgecolor="#333333", linewidth=2.0, linestyle=":",
                    )
                )

    ax.set_xticks(range(len(sweep.x_values)))
    ax.set_xticklabels([f"{v:g}" for v in sweep.x_values])
    ax.set_yticks(range(len(sweep.y_values)))
    ax.set_yticklabels([f"{v:g}" for v in sweep.y_values])
    ax.set_xlabel(L["x"])
    ax.set_ylabel(L["y"])
    ax.set_title(
        f"{L['winner']}  ·  {L['score']}：{metric_label(sweep.primary_metric, lang)} "
        f"（{L['direction_' + sweep.metric_direction]}）"
    )
    ax.set_xticks(np.arange(-0.5, len(sweep.x_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sweep.y_values), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    handles = [Patch(facecolor=colors[n], label=n) for n in sweep.algorithms]
    if any(c is None for row in winners for c in row):
        handles.append(Patch(facecolor="#DDDDDD", label=L["no_winner"]))
    handles.append(
        Patch(facecolor="white", edgecolor="#333333", linestyle=":", label=L["constrained"])
    )
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=False, fontsize=9,
    )
    return fig


# ----------------------------------------------------------------------
# 图 2：逐算法指标小图
# ----------------------------------------------------------------------


def plot_metric_panels(sweep: SweepResult, fig: Figure | None = None, lang: str | None = None):
    """每个算法一张小图，显示主目标随场景参数的连续变化，并框出各自胜出的格点。"""
    lang = _resolve_lang(lang)
    L = _LABELS[lang]
    n = len(sweep.algorithms)
    if fig is None:
        fig = Figure(figsize=(4.0 * n, 4.2), layout="constrained")

    grids = {name: np.array(sweep.metric_grid(name)) for name in sweep.algorithms}
    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-9

    winners, _ = sweep.winner_grid()

    for i, name in enumerate(sweep.algorithms):
        ax = fig.add_subplot(1, n, i + 1)
        grid = grids[name]
        im = ax.imshow(
            grid, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )
        for row_i, row in enumerate(winners):
            for col_i, winner in enumerate(row):
                if winner == name:
                    ax.add_patch(
                        Rectangle(
                            (col_i - 0.5, row_i - 0.5), 1, 1, fill=False,
                            edgecolor="white", linewidth=2.5,
                        )
                    )
                _annotate(
                    ax, f"{grid[row_i][col_i]:.2f}", col_i, row_i,
                    color="white", size=8,
                )
        ax.set_xticks(range(len(sweep.x_values)))
        ax.set_xticklabels([f"{v:g}" for v in sweep.x_values], fontsize=8)
        ax.set_yticks(range(len(sweep.y_values)))
        ax.set_yticklabels([f"{v:g}" for v in sweep.y_values], fontsize=8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(axis_label(sweep.x_name, lang), fontsize=8)
        if i == 0:
            ax.set_ylabel(axis_label(sweep.y_name, lang), fontsize=8)

    fig.suptitle(
        f"{metric_label(sweep.primary_metric, lang)}"
        f"（{L['direction_' + sweep.metric_direction]}）  ·  {L['winner_outline']}"
    )
    fig.colorbar(im, ax=fig.axes, shrink=0.85, label=metric_label(sweep.primary_metric, lang))
    return fig


# ----------------------------------------------------------------------
# 图 3：换目标 → 换胜者（最有说服力的一张）
# ----------------------------------------------------------------------


def plot_objective_comparison(
    sweeps: Sequence[SweepResult], fig: Figure | None = None, lang: str | None = None
):
    """并列展示不同优化目标下的胜者地图。

    这是平台核心论点的直接证据：同一批场景、同一批算法，只换优化目标，
    「哪个算法最好」的答案就变了。
    """
    if not sweeps:
        raise ValueError("至少需要一个扫描结果")
    lang = _resolve_lang(lang)
    L = _LABELS[lang]
    if fig is None:
        # 这张图版式复杂（figure 级标题 + 每面板标题 + 底部换行图例），
        # 交给布局引擎管会互相压住：suptitle 会盖住中间面板标题，图例盖住 x 轴标签，
        # 且 engine.set(rect=...) 实测不可靠。因此显式切分水平带状区域，完全确定性。
        fig = Figure(figsize=(6.0 * len(sweeps), 5.4))

    # 跨图统一配色，保证同名算法颜色一致
    all_algorithms: list[str] = []
    for sweep in sweeps:
        for name in sweep.algorithms:
            if name not in all_algorithms:
                all_algorithms.append(name)
    colors = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(all_algorithms)}

    for i, sweep in enumerate(sweeps):
        ax = fig.add_subplot(1, len(sweeps), i + 1)
        winners, constrained = sweep.winner_grid()
        index_of = {name: j for j, name in enumerate(sorted(all_algorithms))}
        grid = np.array(
            [[-1 if c is None else index_of[c] for c in row] for row in winners],
            dtype=float,
        )
        cmap = _with_bad_color(ListedColormap([colors[n] for n in sorted(all_algorithms)]))
        ax.imshow(
            np.ma.masked_where(grid < 0, grid), cmap=cmap,
            vmin=0, vmax=len(all_algorithms) - 1, origin="lower", aspect="auto",
        )
        for row_i, row in enumerate(winners):
            for col_i, name in enumerate(row):
                _annotate(ax, name or "—", col_i, row_i,
                          color="white" if name else "#666666")
                if constrained[row_i][col_i]:
                    ax.add_patch(
                        Rectangle((col_i - 0.5, row_i - 0.5), 1, 1, fill=False,
                                  edgecolor="#333333", linewidth=2.0, linestyle=":")
                    )
        ax.set_xticks(range(len(sweep.x_values)))
        ax.set_xticklabels([f"{v:g}" for v in sweep.x_values])
        ax.set_yticks(range(len(sweep.y_values)))
        ax.set_yticklabels([f"{v:g}" for v in sweep.y_values])
        ax.set_xlabel(L["x"], fontsize=9)
        if i == 0:
            ax.set_ylabel(L["y"], fontsize=9)
        ax.set_title(
            f"{L['objective']}：{metric_label(sweep.primary_metric, lang)}\n"
            f"（{L['direction_' + sweep.metric_direction]}）",
            fontsize=11,
        )
        ax.set_xticks(np.arange(-0.5, len(sweep.x_values), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(sweep.y_values), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

    handles = [Patch(facecolor=colors[n], label=n) for n in sorted(all_algorithms)]
    handles.append(Patch(facecolor="white", edgecolor="#333333", linestyle=":", label=L["constrained"]))
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles), frameon=False,
        bbox_to_anchor=(0.5, 0.035), fontsize=10,
    )
    # 用 fig.text 而不是 suptitle：constrained layout 会接管 suptitle 的位置，
    # 传入的 y 被忽略，结果压在中间那排面板标题上。fig.text 不受布局引擎管理。
    # 竖直带状划分：标题带 [0.86,1.0) / 面板标题+热力图 [0.20,0.86) / 图例带 [0,0.20)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.78, bottom=0.235, wspace=0.16)
    fig.text(
        0.5, 0.965, _LABELS[lang]["objective_headline"],
        ha="center", va="top", fontsize=12.5,
    )
    return fig


# ----------------------------------------------------------------------
# 负荷曲线（过程回放）
# ----------------------------------------------------------------------

#: 电价时段底色（谷=绿、平=灰、峰=红），极低透明度，只作氛围提示
_SEGMENT_TINT = {"valley": "#55A868", "flat": "#BBBBBB", "peak": "#C44E52"}

_CURVE_LABELS = {
    "zh": {
        "power": "聚合充电负荷 [kW]",
        "time": "窗口内时刻 [h]",
        "limit": "站点功率上限",
        "cost": "累计电费 [CNY]",
        "price": "电价 [CNY/kWh]",
        "valley": "谷段",
        "flat": "平段",
        "peak": "峰段",
        "active": "在充车辆数",
        "shade": "底色为电价时段",
        "peak_short": "峰值",
        "mean_short": "均值",
        "load_shapes": "同一场景下各算法塑造的负荷形状",
    },
    "en": {
        "power": "Aggregate charging load [kW]",
        "time": "Hours into window",
        "limit": "Site power limit",
        "cost": "Cumulative cost [CNY]",
        "price": "Price [CNY/kWh]",
        "valley": "valley",
        "flat": "flat",
        "peak": "peak",
        "active": "Active EVs",
        "shade": "background = price segment",
        "peak_short": "peak",
        "mean_short": "mean",
        "load_shapes": "Load shape produced by each algorithm in the same scenario",
    },
}


def _shade_segments(ax, curves, lang: str) -> None:
    """按电价时段给背景上色。相邻同段的时间步合并成一块，避免画出上千个矩形。"""
    lang = _resolve_lang(lang)
    L = _CURVE_LABELS[lang]
    start_idx = 0
    seen: set[str] = set()
    for i in range(1, len(curves.segments) + 1):
        if i == len(curves.segments) or curves.segments[i] != curves.segments[start_idx]:
            segment = curves.segments[start_idx]
            ax.axvspan(
                curves.hours[start_idx], curves.hours[min(i, len(curves.hours) - 1)],
                color=_SEGMENT_TINT.get(segment, "#BBBBBB"),
                alpha=0.10, linewidth=0, zorder=0,
            )
            seen.add(segment)
            start_idx = i
    # 只在图上说明一次，避免图例被三段色块淹没
    if seen:
        ax.text(
            0.995, 1.02, L["shade"], transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#666666",
        )


def plot_load_curve(curves, fig: Figure | None = None, lang: str | None = None):
    """单次运行的负荷曲线：聚合负荷 + 站点上限 + 电价时段底色 + 累计电费副轴。

    这是最直观的一张图 —— 能直接看出算法是在谷段填负荷，还是把负荷堆在峰段。
    """
    lang = _resolve_lang(lang)
    L = _CURVE_LABELS[lang]
    if fig is None:
        fig = Figure(figsize=(9.5, 4.6), layout="constrained")

    ax = fig.add_subplot(111)
    _shade_segments(ax, curves, lang)

    ax.fill_between(
        curves.hours, curves.aggregate_kw, step="post",
        color="#4C72B0", alpha=0.35, zorder=2,
    )
    ax.step(
        curves.hours, curves.aggregate_kw, where="post",
        color="#4C72B0", linewidth=1.6, zorder=3, label=L["power"],
    )
    ax.axhline(
        curves.limit_kw, color="#C44E52", linestyle="--", linewidth=1.4, zorder=4,
        label=f"{L['limit']} {curves.limit_kw:g} kW",
    )
    ax.set_xlabel(L["time"])
    ax.set_ylabel(L["power"])
    ax.set_ylim(0, max(curves.limit_kw, max(curves.aggregate_kw)) * 1.18)
    ax.set_xlim(0, max(curves.duration_hours, 0.1))
    ax.grid(alpha=0.25, zorder=1)

    # 副轴：累计电费，用于说明"省了多少"
    ax2 = ax.twinx()
    ax2.plot(
        curves.hours, curves.cumulative_cost_cny,
        color="#55A868", linewidth=1.6, linestyle="-", zorder=5,
    )
    ax2.set_ylabel(L["cost"], color="#3F7F4F")
    ax2.tick_params(axis="y", colors="#3F7F4F")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", framealpha=0.9, fontsize=9)
    ax.set_title(
        f"{curves.scenario_id} · {curves.algorithm} · seed={curves.seed}"
    )
    return fig


def plot_load_curves_side_by_side(curves_list, lang: str | None = None):
    """多个算法的负荷曲线并排。看算法如何塑造负荷形状，一眼可见差异。"""
    if not curves_list:
        raise ValueError("至少需要一条曲线")
    lang = _resolve_lang(lang)
    L = _CURVE_LABELS[lang]
    n = len(curves_list)

    fig = Figure(figsize=(4.6 * n, 4.0), layout="constrained")
    for i, curves in enumerate(curves_list):
        ax = fig.add_subplot(1, n, i + 1)
        _shade_segments(ax, curves, lang)
        ax.fill_between(
            curves.hours, curves.aggregate_kw, step="post",
            color="#4C72B0", alpha=0.35, zorder=2,
        )
        ax.step(
            curves.hours, curves.aggregate_kw, where="post",
            color="#4C72B0", linewidth=1.5, zorder=3,
        )
        ax.axhline(
            curves.limit_kw, color="#C44E52", linestyle="--", linewidth=1.3, zorder=4
        )
        ax.set_xlabel(L["time"], fontsize=9)
        if i == 0:
            ax.set_ylabel(L["power"], fontsize=9)
        ax.set_ylim(0, max(curves.limit_kw, max(curves.aggregate_kw)) * 1.12)
        ax.set_xlim(0, max(curves.duration_hours, 0.1))
        ax.grid(alpha=0.25, zorder=1)
        peak = max(curves.aggregate_kw)
        mean = sum(curves.aggregate_kw) / len(curves.aggregate_kw)
        ax.set_title(
            f"{curves.algorithm}\n"
            f"{L['peak_short']} {peak:.1f} kW · {L['mean_short']} {mean:.1f} kW",
            fontsize=10,
        )
    fig.suptitle(L["load_shapes"])
    return fig


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------


def save(fig: Figure, path: str | pathlib.Path, dpi: int = 160) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ----------------------------------------------------------------------
# Agent 闭环：候选对比与收敛
# ----------------------------------------------------------------------

_AGENT_LABELS = {
    "zh": {
        "candidates": "候选策略对比",
        "rounds": "逐轮最好成绩",
        "baseline": "官方基准",
        "candidate": "候选策略",
        "winner": "最终候选",
        "holdout": "保留场景复验",
        "score": "主目标",
        "round": "轮次",
        "value": "取值",
    },
    "en": {
        "candidates": "Candidate comparison",
        "rounds": "Best-so-far by round",
        "baseline": "baseline",
        "candidate": "candidate",
        "winner": "final candidate",
        "holdout": "holdout validation",
        "score": "primary metric",
        "round": "round",
        "value": "value",
    },
}


def plot_agent_report(report, lang: str | None = None):
    """Agent 闭环结果：左图候选对比，右图逐轮收敛。

    左图把基准与候选放在同一根轴上 —— 只看结论文字无法判断提升幅度是否显著。
    """
    lang = _resolve_lang(lang)
    L = _AGENT_LABELS[lang]
    metric = report.primary_metric
    higher_better = report.primary_direction == "max"

    baselines = [(o.spec.name, getattr(o, metric)) for o in report.baseline_outcomes]
    candidates = [
        (o.spec.name, getattr(o, metric))
        for r in report.rounds
        for o in r.outcomes
    ]
    best_name = report.best_candidate.spec.name if report.best_candidate else None

    fig = Figure(figsize=(12.5, 5.0), layout="constrained")

    # ---- 左：候选对比 ----
    ax1 = fig.add_subplot(1, 2, 1)
    labels = [n for n, _ in baselines] + [n for n, _ in candidates]
    values = [v for _, v in baselines] + [v for _, v in candidates]
    colors = ["#999999"] * len(baselines) + ["#4C72B0"] * len(candidates)
    if best_name is not None:
        colors = [
            "#55A868" if n == best_name else c
            for n, c in zip(labels, colors)
        ]
    ypos = np.arange(len(labels))[::-1]
    ax1.barh(ypos, values, color=colors, height=0.62)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel(f"{L['score']}: {metric}")
    ax1.set_title(L["candidates"], fontsize=11)
    for y, v in zip(ypos, values):
        ax1.text(v, y, f" {v:.3f}", va="center", fontsize=8)
    # 右侧留足空白，否则图例（lower right）会压住倒数几根柱子
    ax1.set_xlim(0, max(values) * 1.55 if values else 1.0)
    ax1.grid(axis="x", alpha=0.25)
    handles = [
        Patch(facecolor="#999999", label=L["baseline"]),
        Patch(facecolor="#4C72B0", label=L["candidate"]),
        Patch(facecolor="#55A868", label=L["winner"]),
    ]
    ax1.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    ax1.set_ylim(-0.7, len(labels) - 0.3)

    # ---- 右：逐轮收敛 ----
    ax2 = fig.add_subplot(1, 2, 2)
    base_best = (
        max(v for _, v in baselines) if higher_better else min(v for _, v in baselines)
    )
    rounds = [r.round_index + 1 for r in report.rounds]
    bests = [r.best_value for r in report.rounds]
    ax2.axhline(
        base_best, color="#999999", linestyle="--", linewidth=1.4,
        label=f"{L['baseline']} {base_best:.3f}",
    )
    if rounds:
        ax2.plot(rounds, bests, marker="o", color="#4C72B0", linewidth=1.8)
        for x, y, r in zip(rounds, bests, report.rounds):
            marker = "★" if r.improved else ""
            if marker:
                ax2.annotate(
                    marker, (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", color="#55A868", fontsize=13,
                )
        ax2.set_xticks(rounds)
    else:
        ax2.text(
            0.5, 0.5, "未产生候选轮次", transform=ax2.transAxes,
            ha="center", va="center", color="#666666",
        )
    ax2.set_xlabel(L["round"])
    ax2.set_ylabel(f"{L['score']} ({L['value']})")
    ax2.set_title(L["rounds"], fontsize=11)
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, fontsize=9, loc="center right")
    # 给星标留出上方余量，否则会被裁掉
    series = bests + [base_best] if rounds else [base_best]
    lo, hi = min(series), max(series)
    pad = max((hi - lo) * 0.28, 0.02)
    ax2.set_ylim(lo - pad, hi + pad)

    if report.validation is not None:
        v = report.validation
        verdict = "✓" if v.passed else "✗"
        fig.text(
            0.5, -0.02,
            f"{L['holdout']}: {v.candidate_name} {v.candidate_completion_rate:.3f} vs "
            f"{v.best_baseline_name} {v.best_baseline_completion_rate:.3f}  {verdict}",
            ha="center", va="top", fontsize=10,
        )
    return fig


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------


def render_agent_report(report, out_dir: str | pathlib.Path = "reports") -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    lang = _resolve_lang(None)
    return save(plot_agent_report(report, lang=lang), out_dir / f"{report.report_id}.png")


def render_curves(curves_list, out_dir: str | pathlib.Path = "reports") -> list[pathlib.Path]:
    """产出负荷曲线图：逐算法并排 + 单算法详图（含累计电费副轴）。"""
    if not curves_list:
        raise ValueError("至少需要一条曲线")
    out_dir = pathlib.Path(out_dir)
    lang = _resolve_lang(None)
    first = curves_list[0]
    stem = f"load-curves--{first.scenario_id}--seed{first.seed}"
    written = [
        save(plot_load_curves_side_by_side(curves_list, lang=lang), out_dir / f"{stem}.png")
    ]
    for curves in curves_list:
        written.append(
            save(
                plot_load_curve(curves, lang=lang),
                out_dir
                / f"load-curve--{curves.scenario_id}--{curves.algorithm}--seed{curves.seed}.png",
            )
        )
    return written


def render_sweep(sweep: SweepResult, out_dir: str | pathlib.Path = "reports") -> list[pathlib.Path]:
    """产出该次扫描的标准两张图，返回文件路径。"""
    out_dir = pathlib.Path(out_dir)
    lang = _resolve_lang(None)
    written = []
    fig1 = plot_winner_map(sweep, lang=lang)
    written.append(save(fig1, out_dir / f"{sweep.sweep_id}--winner-map.png"))
    fig2 = plot_metric_panels(sweep, lang=lang)
    written.append(save(fig2, out_dir / f"{sweep.sweep_id}--metric-panels.png"))
    return written


def render_objective_comparison(
    sweeps: Sequence[SweepResult], out_dir: str | pathlib.Path = "reports", name: str | None = None
) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    lang = _resolve_lang(None)
    fig = plot_objective_comparison(sweeps, lang=lang)
    stem = name or ("vs-".join(s.primary_metric for s in sweeps))
    return save(fig, out_dir / f"objective-comparison--{stem}.png")
