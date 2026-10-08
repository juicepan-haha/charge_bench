"""ChargeBench AI — Streamlit 仪表板。

启动：
    .venv/bin/streamlit run app.py

设计原则：**不碰代码也能走完「选场景 → 跑实验 → 看结论」。**
所有数字都来自 `chargebench` 包，页面不做任何自定义计算 —— 否则口径会与报告分叉。
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import streamlit as st

from chargebench.adapter import build_scenario
from chargebench.algorithms import CORE_ALGORITHMS, list_algorithms
from chargebench.experiments import run_batch
from chargebench.schemas import METRIC_DIRECTIONS, SCHEMA_VERSION, BatchResult, Scenario
from chargebench.storage import ResultStore
from chargebench.sweep import (
    format_winner_grid,
    rerank_sweep,
    run_sweep,
)
from chargebench.timeseries import simulate_curves, summarize
from chargebench.viz import (
    plot_load_curve,
    plot_load_curves_side_by_side,
    plot_metric_panels,
    plot_objective_comparison,
    plot_winner_map,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_STORE = PROJECT_ROOT / "results"

st.set_page_config(page_title="ChargeBench AI", page_icon="🔌", layout="wide")


# ----------------------------------------------------------------------
# 数据层（缓存的是数据，不是图 —— Figure 不适合跨版本缓存）
# ----------------------------------------------------------------------


def _scenario_files() -> dict[str, pathlib.Path]:
    return {p.stem: p for p in sorted(CONFIG_DIR.glob("*.json"))}


@st.cache_data(show_spinner=False)
def _load_scenario_text(path: str) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")


def _scenario_from(path: str) -> Scenario:
    return Scenario(**json.loads(_load_scenario_text(path)))


@st.cache_data(show_spinner="正在运行仿真…")
def _cached_batch(scenario_json: str, algorithms: tuple[str, ...], seeds: tuple[int, ...]) -> BatchResult:
    scenario = Scenario(**json.loads(scenario_json))
    return run_batch([scenario], algorithms, seeds=list(seeds) or None)


@st.cache_data(show_spinner="正在扫描参数网格…（每个格点都会跑全部算法与种子）")
def _cached_sweep(
    scenario_json: str,
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
    algorithms: tuple[str, ...],
    seeds: tuple[int, ...],
    metric: str,
) -> dict:
    scenario = Scenario(**json.loads(scenario_json))
    sweep = run_sweep(
        scenario, list(x_values), list(y_values), list(algorithms), list(seeds),
        primary_metric=metric,
    )
    # 顺带把「换目标」的两种排名也算好，供对比图使用
    alternates = [
        rerank_sweep(sweep, m).model_dump(mode="json")
        for m in ("peak_power_kw", "energy_cost_cny")
        if m != metric
    ]
    return {"primary": sweep.model_dump(mode="json"), "alternates": alternates}


@st.cache_data(show_spinner="正在重跑仿真以提取曲线…")
def _cached_curves(scenario_json: str, algorithm: str, seed: int) -> dict:
    scenario = Scenario(**json.loads(scenario_json))
    return simulate_curves(scenario, algorithm, seed=seed).__dict__


def _curves_from(payload: dict):
    from chargebench.timeseries import LoadCurves

    return LoadCurves(**payload)


# ----------------------------------------------------------------------
# 侧边栏
# ----------------------------------------------------------------------


def _sidebar() -> dict:
    st.sidebar.title("🔌 ChargeBench AI")
    st.sidebar.caption(f"协议版本 {SCHEMA_VERSION}")

    files = _scenario_files()
    if not files:
        st.sidebar.error(f"configs/ 下没有场景文件：{CONFIG_DIR}")
        st.stop()

    labels = list(files)
    choice = st.sidebar.selectbox("场景", labels, index=0)
    scenario_path = str(files[choice])
    scenario = _scenario_from(scenario_path)

    st.sidebar.divider()
    st.sidebar.subheader("场景概览")
    art = build_scenario(scenario)
    occupancy = scenario.port_occupancy_estimate
    st.sidebar.metric("预计车位占用率", f"{occupancy:.2f}")
    if occupancy > 1.0:
        st.sidebar.warning(
            "占用率 > 1：会有车辆因无空闲车位被丢弃，满足率的分母随之缩水。"
            "这属于场景可行性问题，**不能归因于算法**。"
        )

    st.sidebar.caption(
        f"车位 {scenario.n_ports} · 单桩 {scenario.port_max_power_kw:g} kW · "
        f"站点上限 {scenario.effective_supply_kw:g} kW\n\n"
        f"窗口 {scenario.window_hours:g} h × {scenario.time_step_min:g} min = "
        f"{scenario.periods} 步\n\n"
        f"负荷强度 {scenario.load_intensity:g} · 紧迫度 {scenario.deadline_tightness:g} · "
        f"到达 {scenario.arrival_mode.value}"
    )
    st.sidebar.caption(
        f"生成 {len(art.sessions_generated)} 个会话 · 排定 {len(art.sessions_scheduled)} · "
        f"丢弃 {art.n_dropped}"
    )

    st.sidebar.divider()
    st.sidebar.subheader("实验设置")
    algorithms = st.sidebar.multiselect(
        "算法", [a["name"] for a in list_algorithms()], default=list(CORE_ALGORITHMS)
    )
    if not algorithms:
        st.sidebar.error("至少选择一个算法")
        st.stop()

    seeds_text = st.sidebar.text_input("种子（逗号分隔，跨种子看波动）", str(scenario.seed))
    try:
        seeds = tuple(int(s.strip()) for s in seeds_text.split(",") if s.strip())
    except ValueError:
        st.sidebar.error("种子必须是整数")
        st.stop()
    if not seeds:
        seeds = (scenario.seed,)

    metric = st.sidebar.selectbox(
        "排名依据的指标",
        list(METRIC_DIRECTIONS),
        index=list(METRIC_DIRECTIONS).index("session_completion_rate"),
        help="换一个目标，胜者可能会变 —— 这是本平台要说明的核心事实",
    )

    with st.sidebar.expander("算法能看到哪些信息（公平性核对）"):
        for entry in list_algorithms():
            st.markdown(f"**{entry['name']}** — {entry['summary']}")
            st.caption("可见信息：" + "、".join(entry["visible_info"]))

    return {
        "scenario": scenario,
        "scenario_json": scenario.canonical_json(),
        "algorithms": tuple(algorithms),
        "seeds": seeds,
        "metric": metric,
    }


# ----------------------------------------------------------------------
# 标签页 1：批量对比
# ----------------------------------------------------------------------


def _metrics_frame(batch: BatchResult) -> pd.DataFrame:
    rows = []
    for run in batch.runs:
        m = run.metrics
        rows.append(
            {
                "算法": run.algorithm,
                "种子": run.seed,
                "需求满足率": round(m.demand_satisfaction_rate, 4),
                "按时完成率": round(m.session_completion_rate, 4),
                "最差单车交付比": round(m.worst_delivery_ratio, 4),
                "交付电量 kWh": round(m.energy_delivered_kwh, 2),
                "电费 CNY": round(m.energy_cost_cny, 2),
                "峰值 kW": round(m.peak_power_kw, 2),
                "越限步数": m.constraint_violations,
                "丢弃车辆": m.sessions_dropped,
            }
        )
    return pd.DataFrame(rows)


def _tab_batch(cfg: dict) -> None:
    st.subheader("批量对比")
    st.caption(
        "同一场景、同一批车辆会话，横向比较多个算法。"
        "**总交付电量往往几乎相同** —— 差异体现在「这些电怎么分配」，"
        "所以要同时看能量口径与车辆口径。"
    )

    batch = _cached_batch(cfg["scenario_json"], cfg["algorithms"], cfg["seeds"])
    frame = _metrics_frame(batch)
    st.dataframe(frame, width="stretch", hide_index=True)

    if len(cfg["seeds"]) > 1:
        st.markdown("**跨种子均值与波动**（方案 §5 公平性原则 2：单次运行会被随机性掩盖差异）")
        grouped = frame.groupby("算法").agg(
            完成率均值=("按时完成率", "mean"),
            完成率最小=("按时完成率", "min"),
            完成率最大=("按时完成率", "max"),
            满足率均值=("需求满足率", "mean"),
            电费均值=("电费 CNY", "mean"),
        )
        grouped["完成率波动"] = grouped["完成率最大"] - grouped["完成率最小"]
        st.dataframe(grouped.round(4), width="stretch")

    st.markdown("**按时完成率对比**")
    if len(cfg["seeds"]) > 1:
        st.bar_chart(frame, x="算法", y="按时完成率", color="种子", stack=False)
    else:
        # 只有一个种子时用 color 会生成一条无意义的连续色标（42.000000 的滑块），
        # 还会挤得 x 轴标签竖排，故不加颜色编码
        st.bar_chart(frame, x="算法", y="按时完成率")

    best = frame.loc[frame["按时完成率"].idxmax()]
    st.success(
        f"本轮按时完成率最高：**{best['算法']}**（{best['按时完成率']:.3f}，seed={best['种子']}）。"
        f"注意这只是一次运行的排名，换目标或换种子都可能变。"
    )

    st.markdown(
        "**同一个表格里还藏着一个反直觉的结果**：LLF 的按时完成率最低（0.236），"
        "但它的「最差单车交付比」反而最高。因为 LLF 的松弛时间每周期重算，"
        "能量被摊薄到更多车上 —— 没有车被完全饿死，却也没有车被喂饱。"
        "只报一个指标会完全看不到这件事。"
    )

    with st.expander("可追溯性：这些数字是用什么跑出来的"):
        first = batch.runs[0]
        st.json(first.traceability.model_dump(mode="json"))
        st.caption("每个 run_id 都能据此定位到场景、算法、参数、种子与依赖版本。")


# ----------------------------------------------------------------------
# 标签页 2：过程回放
# ----------------------------------------------------------------------


def _tab_replay(cfg: dict) -> None:
    st.subheader("过程回放")
    st.caption(
        "把一次运行的负荷曲线摊开看。底色是电价时段，红色虚线是站点功率上限。"
    )
    scenario = cfg["scenario"]
    seed = st.selectbox("种子", cfg["seeds"], key="replay_seed")

    curves_by_algo = {}
    error_slots = []
    for algorithm in cfg["algorithms"]:
        slot = st.empty()
        try:
            curves_by_algo[algorithm] = _curves_from(
                _cached_curves(cfg["scenario_json"], algorithm, seed)
            )
        except Exception as exc:  # noqa: BLE001 - UI 层需要把失败展示出来而不是崩掉
            error_slots.append((slot, algorithm, exc))
    for slot, algorithm, exc in error_slots:
        slot.error(f"{algorithm} 无法提取曲线：{exc}")

    if not curves_by_algo:
        return

    cards = st.columns(len(curves_by_algo))
    for column, (algorithm, curves) in zip(cards, curves_by_algo.items()):
        summary = summarize(curves)
        column.metric(
            f"{algorithm} · 峰值",
            f"{summary['peak_kw']:.1f} kW",
            help=f"均值 {summary['mean_kw']:.1f} kW · 利用率 {summary['utilization']:.1%}",
        )
    st.caption(
        "峰值通常都贴住站点上限 —— 因为这三个算法都是「贪心填充」策略："
        "只要有额度就尽早充。它们只在**争抢激烈**时因分配优先级不同而分化，"
        "本身并不做价格套利，所以负荷形状相近。"
    )

    if len(curves_by_algo) == 1:
        st.pyplot(plot_load_curve(next(iter(curves_by_algo.values()))))
    else:
        st.pyplot(plot_load_curves_side_by_side(list(curves_by_algo.values())))

    if len(curves_by_algo) > 1:
        st.markdown("**累计电费**")
        cost_frame = pd.DataFrame(
            {name: c.cumulative_cost_cny for name, c in curves_by_algo.items()},
            index=[round(h, 2) for h in next(iter(curves_by_algo.values())).hours],
        )
        cost_frame.index.name = "窗口内时刻 [h]"
        st.line_chart(cost_frame)

    with st.expander("站点级负荷时间线（车位 × 时间）"):
        chosen = st.selectbox("查看哪个算法", list(curves_by_algo), key="station_algo")
        curves = curves_by_algo[chosen]
        station_frame = pd.DataFrame(
            curves.per_station_kw,
            index=[round(h, 2) for h in curves.hours],
        )
        station_frame.index.name = "窗口内时刻 [h]"
        st.area_chart(station_frame)
        st.caption(
            f"共 {len(curves.per_station_kw)} 个车位。"
            "仿真的终点是最后一个拔枪事件，所以横轴通常略短于窗口长度。"
        )


# ----------------------------------------------------------------------
# 标签页 3：适用边界
# ----------------------------------------------------------------------


def _tab_sweep(cfg: dict) -> None:
    st.subheader("适用边界扫描")
    st.caption(
        "在「负荷强度 × 时间紧迫度」网格上逐点评测，找出**哪个算法在什么条件下胜出**。"
        "评分规则事前写定并随结果持久化，不会看结果后改口径。"
    )
    scenario = cfg["scenario"]

    col1, col2 = st.columns(2)
    with col1:
        x_values = st.slider(
            "负荷强度 X 轴",
            min_value=0.2, max_value=2.0, value=(0.5, 1.3), step=0.1,
            help="总请求电量 / 站点窗口内可供电量",
        )
        x_list = [round(x_values[0] + i * 0.2, 2) for i in range(
            int((x_values[1] - x_values[0]) / 0.2) + 1) if
            round(x_values[0] + i * 0.2, 2) <= x_values[1] + 1e-9]
    with col2:
        y_values = st.slider(
            "时间紧迫度 Y 轴",
            min_value=0.2, max_value=1.0, value=(0.4, 0.85), step=0.05,
            help="满功率所需充电时长 / 可停留时长",
        )
        y_list = [round(y_values[0] + i * 0.15, 2) for i in range(
            int((y_values[1] - y_values[0]) / 0.15) + 1) if
            round(y_values[0] + i * 0.15, 2) <= y_values[1] + 1e-9]

    n_runs = len(x_list) * len(y_list) * len(cfg["algorithms"]) * len(cfg["seeds"])
    st.info(f"本次将运行 **{n_runs}** 次仿真（{len(x_list)}×{len(y_list)} 格 × "
            f"{len(cfg['algorithms'])} 算法 × {len(cfg['seeds'])} 种子）。")

    if not st.button("开始扫描", type="primary"):
        st.caption("点击「开始扫描」运行。结果会被缓存，调整坐标轴不会重跑。")
        return

    payload = _cached_sweep(
        cfg["scenario_json"], tuple(x_list), tuple(y_list),
        cfg["algorithms"], cfg["seeds"], cfg["metric"],
    )
    primary = payload["primary"]
    alternates = payload["alternates"]

    constrained = [c for c in primary["cells"] if c["port_constrained"]]
    if constrained:
        st.warning(
            f"{len(constrained)}/{len(primary['cells'])} 个格点车位占用率 > 1，"
            "会有车辆因无空位被丢弃。这些格点的满足率分母缩水，结论需谨慎 —— "
            "图中用虚线框标出。"
        )

    from chargebench.schemas import SweepResult

    sweep = SweepResult(**primary)
    st.pyplot(plot_winner_map(sweep))

    st.markdown(f"**评分规则**（事前写定，随结果持久化）  \n{sweep.score_rule}")
    st.code(format_winner_grid(sweep), language=None)

    # 换目标 → 换胜者。实验数据完全相同，只是换了把尺子。
    if alternates:
        st.divider()
        st.markdown("### 换一个优化目标，胜者就变了")
        st.caption(
            "下面这些面板与上面**共享同一份实验数据**（同一批 run_id），"
            "只是换了排名依据。这是平台要说明的核心事实："
            "「哪个算法最好」没有唯一答案，取决于你把什么当目标。"
        )
        others = [SweepResult(**a) for a in alternates]
        st.pyplot(plot_objective_comparison([sweep, *others]))
        for other in others:
            with st.expander(f"{other.primary_metric} 的逐格明细"):
                st.code(format_winner_grid(other), language=None)

    st.divider()
    st.markdown("### 逐算法指标")
    st.pyplot(plot_metric_panels(sweep))


# ----------------------------------------------------------------------
# 标签页 4：结果库
# ----------------------------------------------------------------------


def _tab_store(cfg: dict) -> None:
    st.subheader("结果库")
    store_path = st.text_input("结果库目录", str(DEFAULT_STORE))
    store = ResultStore(store_path)
    counts = store.counts()

    col1, col2 = st.columns(2)
    col1.metric("运行记录", counts["runs"])
    col2.metric("批次", counts["batches"])

    if counts["runs"] == 0:
        st.info(
            "结果库还是空的。上面几个标签页的结果默认不落盘 —— "
            "需要持久化请用命令行：\n\n"
            "```bash\npython -m chargebench.cli run --scenario configs/demo_scenario.json \\\n"
            "    --algorithm FCFS EDF LLF --store results/\n```"
        )
        return

    batches = store.list_batches()
    if batches:
        st.markdown("**批次**")
        st.dataframe(pd.DataFrame(batches), width="stretch", hide_index=True)

    algorithms = sorted({r.algorithm for r in store.list_runs()})
    pick = st.selectbox("按算法筛选", ["（全部）", *algorithms])
    runs = store.list_runs(algorithm=None if pick == "（全部）" else pick)
    st.dataframe(
        pd.DataFrame([r.__dict__ for r in runs]).round(4),
        width="stretch", hide_index=True,
    )

    run_ids = [r.run_id for r in runs]
    chosen = st.selectbox("查看某条记录的完整来源", run_ids)
    if chosen:
        st.code(store.describe_run(chosen), language=None)
        st.caption("缓存复用的判据是 executed_at 不变 —— "
                   "同配置重跑返回的是原始结果，而不是重新计算。")


# ----------------------------------------------------------------------


def main() -> None:
    cfg = _sidebar()

    st.title("ChargeBench AI")
    st.markdown(
        "**不开发「唯一最好的充电调度算法」，而是搭建一套让不同算法在统一约束下公平竞争、"
        "自动探索适用边界，并由 AI 提出可验证改进建议的实验平台。**"
    )

    tab1, tab2, tab3, tab4 = st.tabs(["📊 批量对比", "📈 过程回放", "🗺️ 适用边界", "🗄️ 结果库"])
    with tab1:
        _tab_batch(cfg)
    with tab2:
        _tab_replay(cfg)
    with tab3:
        _tab_sweep(cfg)
    with tab4:
        _tab_store(cfg)

    st.divider()
    st.caption(
        "仿真后端为外部开源项目 ACN-Sim（BSD-3-Clause，Caltech ACN Portal），"
        "本项目仅通过其公开 API 使用。我们新增的是统一评测协议、边界探索与 Agent 闭环。"
        f"　协议版本 {SCHEMA_VERSION} · 接口定义见 docs/PROTOCOL.md"
    )


if __name__ == "__main__":
    main()
