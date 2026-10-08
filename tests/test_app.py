"""仪表板应用层测试。

Streamlit 的渲染无法在单元测试里完整覆盖（那是浏览器验收的事），
但**数据层必须与核心 API 一致** —— 否则页面上会出现和报告不一样的数字。
本文件守住这条：UI 只是展示，不做任何自定义计算。
"""

import json
import warnings

import pytest

# 以 bare mode 导入 Streamlit 应用会产生 ScriptRunContext 提示，与测试无关
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*No runtime found.*")

import app  # noqa: E402

from chargebench.algorithms import CORE_ALGORITHMS  # noqa: E402
from chargebench.experiments import run_batch  # noqa: E402
from chargebench.schemas import Scenario  # noqa: E402

from .conftest import CONFIG_DIR, PROJECT_ROOT, load_scenario  # noqa: E402


def test_app_imports_without_streamlit_runtime():
    """应用必须能在 bare mode 下导入 —— 否则测试与其他工具都无法引用它的数据层。"""
    assert callable(app.main)
    assert callable(app._metrics_frame)


def test_scenario_files_discovered():
    found = app._scenario_files()
    assert found, f"configs/ 下应至少有一个场景文件：{CONFIG_DIR}"
    assert "demo_scenario" in found


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIG_DIR.glob("*.json")))
def test_every_config_loads_into_schema(name: str):
    """界面下拉里能选到的每个场景都必须能通过 schema 校验。"""
    scenario = Scenario(**json.loads((CONFIG_DIR / name).read_text(encoding="utf-8")))
    assert scenario.scenario_id


def test_metrics_frame_matches_core_api():
    """表格里的数字必须与 run_batch 的输出逐字一致 —— UI 不得自行计算。"""
    scenario = load_scenario()
    batch = run_batch([scenario], CORE_ALGORITHMS)
    frame = app._metrics_frame(batch)

    assert len(frame) == len(batch.runs)
    by_algorithm = {row["算法"]: row for _, row in frame.iterrows()}
    for run in batch.runs:
        row = by_algorithm[run.algorithm]
        assert row["种子"] == run.seed
        assert row["需求满足率"] == pytest.approx(
            round(run.metrics.demand_satisfaction_rate, 4)
        )
        assert row["按时完成率"] == pytest.approx(
            round(run.metrics.session_completion_rate, 4)
        )
        assert row["交付电量 kWh"] == pytest.approx(
            round(run.metrics.energy_delivered_kwh, 2)
        )
        assert row["越限步数"] == run.metrics.constraint_violations
        assert row["丢弃车辆"] == run.metrics.sessions_dropped


def test_metrics_frame_columns_are_stable():
    """列名会被 UI 与截图引用，改名应当是一次有意识的决定。"""
    frame = app._metrics_frame(run_batch([load_scenario()], ["EDF"]))
    assert list(frame.columns) == [
        "算法",
        "种子",
        "需求满足率",
        "按时完成率",
        "最差单车交付比",
        "交付电量 kWh",
        "电费 CNY",
        "峰值 kW",
        "越限步数",
        "丢弃车辆",
    ]


def test_metrics_frame_handles_multiple_seeds():
    batch = run_batch([load_scenario()], ["EDF"], seeds=[42, 43])
    frame = app._metrics_frame(batch)
    assert len(frame) == 2
    assert set(frame["种子"]) == {42, 43}


def test_curves_round_trip_through_app_payload():
    """应用把曲线序列化成 dict 交给缓存，必须能还原成 LoadCurves。"""
    from chargebench.timeseries import simulate_curves

    curves = simulate_curves(load_scenario(), "EDF")
    restored = app._curves_from(dict(curves.__dict__))
    assert restored.aggregate_kw == curves.aggregate_kw
    assert restored.n_periods == curves.n_periods


def test_default_store_path_is_project_local():
    assert app.DEFAULT_STORE.is_absolute()
    assert app.DEFAULT_STORE.name == "results"


# ----------------------------------------------------------------------
# 演示准备脚本
# ----------------------------------------------------------------------


def test_demo_script_constants_are_consistent_with_configs():
    """演示脚本引用的场景必须存在且能通过校验 —— 否则现场才会发现跑不起来。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prepare_demo", PROJECT_ROOT / "scripts" / "prepare_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = module.load_base()
    assert base.scenario_id == load_scenario().scenario_id
    assert len(module.SWEEP_X) >= 2 and len(module.SWEEP_Y) >= 2
    assert module.DEMO_SEEDS
    assert all(s.port_occupancy_estimate > 0 for s in module.train_scenarios(base))
    assert len(module.holdout_scenarios(base)) >= 1


def test_demo_train_and_holdout_are_disjoint():
    """实验集与保留场景不能重叠，否则复验没有意义（方案 §5 原则 6）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prepare_demo", PROJECT_ROOT / "scripts" / "prepare_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = module.load_base()
    train_ids = {s.scenario_id for s in module.train_scenarios(base)}
    holdout_ids = {s.scenario_id for s in module.holdout_scenarios(base)}
    assert not (train_ids & holdout_ids)
    train_hashes = {s.scenario_hash for s in module.train_scenarios(base)}
    holdout_hashes = {s.scenario_hash for s in module.holdout_scenarios(base)}
    assert not (train_hashes & holdout_hashes), "保留场景与实验集内容相同，复验无意义"


def test_demo_objectives_are_rankable():
    """演示里用作对比的目标函数必须都在可排名指标里。"""
    import importlib.util

    from chargebench.schemas import METRIC_DIRECTIONS

    spec = importlib.util.spec_from_file_location(
        "prepare_demo", PROJECT_ROOT / "scripts" / "prepare_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert set(module.OBJECTIVES) <= set(METRIC_DIRECTIONS)
