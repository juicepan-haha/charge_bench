"""可复现性与公平性测试。

方案 §5 的公平性原则要能由代码证明，而不是靠口头承诺：
  1. 每个算法看到同样的车辆会话与基础设施；
  2. 随机场景固定种子，并跨种子给均值/区间；
  5. 场景不可行不能归因于算法。
"""

import subprocess
import sys

import pytest

from chargebench.algorithms import CORE_ALGORITHMS
from chargebench.experiments import run_batch, run_experiment
from chargebench.schemas import Scenario

from .conftest import PROJECT_ROOT, load_scenario


# ----------------------------------------------------------------------
# 同配置必须逐位可复现
# ----------------------------------------------------------------------


def test_same_config_reproduces_identical_metrics(demo_scenario: Scenario):
    first = run_experiment(demo_scenario, "EDF")
    second = run_experiment(demo_scenario, "EDF")

    assert first.run_id == second.run_id
    # executed_at 记录真实执行时刻，必然不同；其余字段必须完全一致
    assert first.metrics.model_dump(exclude={"runtime_s"}) == second.metrics.model_dump(
        exclude={"runtime_s"}
    )


def test_run_id_is_stable_across_processes(demo_scenario: Scenario):
    """跨进程稳定 —— 若哈希受 PYTHONHASHSEED 或字典顺序影响，这里会失败。"""
    in_process = run_experiment(demo_scenario, "EDF").run_id

    script = (
        "import json, sys; sys.path.insert(0, '.');"
        "from chargebench.experiments import run_experiment;"
        "from chargebench.schemas import Scenario;"
        "print(run_experiment(Scenario(**json.load(open('configs/demo_scenario.json'))), 'EDF').run_id)"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == in_process


def test_different_seed_changes_outcome(demo_scenario: Scenario):
    a = run_experiment(demo_scenario, "EDF", seed=42)
    b = run_experiment(demo_scenario, "EDF", seed=43)
    assert a.run_id != b.run_id
    assert a.metrics.model_dump() != b.metrics.model_dump()
    assert a.seed == 42 and b.seed == 43


def test_seed_override_does_not_mutate_input(demo_scenario: Scenario):
    run_experiment(demo_scenario, "EDF", seed=999)
    assert demo_scenario.seed == 42, "run_experiment 不应改写调用方的场景对象"


# ----------------------------------------------------------------------
# 公平性
# ----------------------------------------------------------------------


def test_all_algorithms_see_identical_scenario(demo_scenario: Scenario):
    """公平性核心断言：请求电量总和相同 ⇒ 它们拿到的是同一批车辆会话。"""
    results = [run_experiment(demo_scenario, name) for name in CORE_ALGORITHMS]
    requested = {r.metrics.energy_requested_kwh for r in results}
    scheduled = {r.metrics.sessions_scheduled for r in results}
    dropped = {r.metrics.sessions_dropped for r in results}
    generated = {r.metrics.sessions_generated for r in results}

    assert len(requested) == 1, f"各算法看到的请求电量不一致：{requested}"
    assert len(scheduled) == 1
    assert len(dropped) == 1
    assert len(generated) == 1


def test_scenario_hash_shared_across_algorithms(demo_scenario: Scenario):
    hashes = {run_experiment(demo_scenario, name).scenario_hash for name in CORE_ALGORITHMS}
    assert len(hashes) == 1


def test_dropped_sessions_are_visible_not_hidden(demo_scenario: Scenario):
    """场景不可行必须单独可见，且各算法看到的丢弃数一致（与算法无关）。"""
    artifacts_drops = run_experiment(demo_scenario, "EDF").metrics.sessions_dropped
    for name in CORE_ALGORITHMS:
        assert run_experiment(demo_scenario, name).metrics.sessions_dropped == artifacts_drops


def test_batch_covers_scenarios_algorithms_and_seeds(demo_scenario: Scenario):
    second = demo_scenario.with_updates(scenario_id="campus_baseline_v1_alt")
    batch = run_batch([demo_scenario, second], CORE_ALGORITHMS, seeds=[42, 43])

    assert len(batch.runs) == 2 * len(CORE_ALGORITHMS) * 2
    assert batch.seeds == [42, 43]
    assert batch.algorithms == sorted(CORE_ALGORITHMS)
    assert sorted(batch.scenario_ids) == sorted({"campus_baseline_v1", "campus_baseline_v1_alt"})
    assert len({r.run_id for r in batch.runs}) == len(batch.runs)


def test_batch_id_is_deterministic(demo_scenario: Scenario):
    a = run_batch([demo_scenario], CORE_ALGORITHMS, seeds=[42])
    b = run_batch([demo_scenario], CORE_ALGORITHMS, seeds=[42])
    assert a.batch_id == b.batch_id


def test_multi_seed_reveals_spread(demo_scenario: Scenario):
    """单次运行会被随机性掩盖差异，跨种子必须能看出波动（方案 §5 原则 2）。"""
    rates = [
        run_experiment(demo_scenario, "EDF", seed=s).metrics.demand_satisfaction_rate
        for s in (42, 43, 44, 45, 46)
    ]
    assert len(set(rates)) > 1, "跨种子结果完全相同，说明种子没有真正生效"
    assert min(rates) > 0.5 and max(rates) < 1.0, "演示场景应落在有区分度的区间"


def test_algorithm_params_validated(demo_scenario: Scenario):
    with pytest.raises(KeyError):
        run_experiment(demo_scenario, "EDF", algorithm_params={"nonexistent_param": 1})
    with pytest.raises(KeyError):
        run_experiment(demo_scenario, "NOPE")
