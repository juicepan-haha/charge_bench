"""结果存储测试。

阶段 3 的判据是「这个结果当时是用什么配置跑出来的」必须能一句话查出来，
且重复实验不能污染结果库。本文件把这两条钉死。
"""

import json

import pytest

from chargebench.experiments import run_batch, run_experiment
from chargebench.storage import ResultMismatchError, ResultStore

from .conftest import load_scenario


@pytest.fixture
def store(tmp_path) -> ResultStore:
    return ResultStore(tmp_path / "results")


@pytest.fixture
def one_run(demo_scenario):
    return run_experiment(demo_scenario, "EDF")


# ----------------------------------------------------------------------
# 往返
# ----------------------------------------------------------------------


def test_run_round_trips_through_store(store: ResultStore, one_run):
    store.save_run(one_run)
    restored = store.get_run(one_run.run_id)
    assert restored.model_dump() == one_run.model_dump()


def test_raw_json_is_written(store: ResultStore, one_run):
    """原始 JSON 必须独立于 SQLite 存在 —— 便于复现与分享。"""
    store.save_run(one_run)
    path = store.runs_dir / f"{one_run.run_id}.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == one_run.run_id
    assert payload["traceability"]["acnportal_version"]


def test_get_run_unknown_id_raises(store: ResultStore):
    with pytest.raises(KeyError):
        store.get_run("nonexistent")


# ----------------------------------------------------------------------
# 幂等性与一致性校验
# ----------------------------------------------------------------------


def test_save_is_idempotent(store: ResultStore, one_run):
    store.save_run(one_run)
    store.save_run(one_run)
    assert store.counts()["runs"] == 1


def test_runtime_and_timestamp_differences_are_tolerated(store: ResultStore, demo_scenario):
    """runtime_s 与 executed_at 必然逐次不同，不该被当成内容冲突。"""
    first = run_experiment(demo_scenario, "EDF")
    second = run_experiment(demo_scenario, "EDF")
    store.save_run(first)
    store.save_run(second)  # 不应抛错
    assert store.counts()["runs"] == 1


def test_metrics_difference_is_rejected(store: ResultStore, one_run):
    """同 run_id 不同内容 = 实验不可复现，必须报错而不是静默覆盖。"""
    store.save_run(one_run)
    tampered = one_run.model_copy(
        update={
            "metrics": one_run.metrics.model_copy(
                update={"energy_cost_cny": one_run.metrics.energy_cost_cny + 1.0}
            )
        }
    )
    with pytest.raises(ResultMismatchError, match="非确定性"):
        store.save_run(tampered)


# ----------------------------------------------------------------------
# 批次与查询
# ----------------------------------------------------------------------


def test_batch_round_trips(store: ResultStore, demo_scenario):
    batch = run_batch([demo_scenario], ["FCFS", "EDF", "LLF"], seeds=[42, 43])
    store.save_batch(batch)

    restored = store.get_batch(batch.batch_id)
    assert restored.batch_id == batch.batch_id
    assert len(restored.runs) == len(batch.runs)
    assert store.counts() == {"batches": 1, "runs": 6}


def test_list_runs_filters(store: ResultStore, demo_scenario):
    batch = run_batch([demo_scenario], ["FCFS", "EDF", "LLF"])
    store.save_batch(batch)

    assert len(store.list_runs()) == 3
    assert len(store.list_runs(algorithm="EDF")) == 1
    assert len(store.list_runs(scenario_id=demo_scenario.scenario_id)) == 3
    assert store.list_runs(scenario_id="nope") == []
    assert len(store.list_runs(batch_id=batch.batch_id)) == 3


def test_list_runs_orders_by_metric(store: ResultStore, demo_scenario):
    store.save_batch(run_batch([demo_scenario], ["FCFS", "EDF", "LLF"]))
    ranked = store.list_runs(order_by="session_completion_rate")
    rates = [r.session_completion_rate for r in ranked]
    assert rates == sorted(rates), "按完成率排序必须真的有序，供排名使用"


def test_order_by_rejects_injection(store: ResultStore):
    with pytest.raises(ValueError, match="order_by"):
        store.list_runs(order_by="run_id; DROP TABLE runs")


def test_find_existing_reuses_computation(store: ResultStore, demo_scenario):
    """缓存命中是 Agent 闭环的成本闸门，必须真的命中。"""
    assert store.find_existing(demo_scenario.scenario_hash, "EDF", 42) is None

    result = run_experiment(demo_scenario, "EDF")
    store.save_run(result)

    found = store.find_existing(demo_scenario.scenario_hash, "EDF", 42)
    assert found is not None
    assert found.run_id == result.run_id
    # 换一个维度就不该命中
    assert store.find_existing(demo_scenario.scenario_hash, "FCFS", 42) is None
    assert store.find_existing(demo_scenario.scenario_hash, "EDF", 43) is None


def test_run_batch_with_store_skips_recomputation(store: ResultStore, demo_scenario):
    first = run_batch([demo_scenario], ["FCFS", "EDF"], store=store)
    assert store.counts()["runs"] == 2

    second = run_batch([demo_scenario], ["FCFS", "EDF"], store=store)
    assert store.counts()["runs"] == 2, "重复运行不应在结果库里制造新记录"
    assert {r.run_id for r in second.runs} == {r.run_id for r in first.runs}
    # 复用得到的结果内容必须一致（runtime_s / executed_at 除外）
    a = {r.run_id: r for r in first.runs}
    for run in second.runs:
        assert run.metrics.model_dump(exclude={"runtime_s"}) == a[
            run.run_id
        ].metrics.model_dump(exclude={"runtime_s"})


def test_run_batch_with_store_extends_batch(store: ResultStore, demo_scenario):
    """新增算法时只算新的，已有结果复用。"""
    run_batch([demo_scenario], ["FCFS"], store=store)
    assert store.counts()["runs"] == 1
    run_batch([demo_scenario], ["FCFS", "EDF", "LLF"], store=store)
    assert store.counts()["runs"] == 3


# ----------------------------------------------------------------------
# 可追溯性
# ----------------------------------------------------------------------


def test_describe_run_answers_what_config_was_used(store: ResultStore, one_run):
    """阶段 3 的核心判据：一句话说清这个结果当时是怎么跑出来的。"""
    store.save_run(one_run)
    text = store.describe_run(one_run.run_id)

    assert one_run.run_id in text
    assert one_run.scenario_id in text
    assert "EDF" in text
    assert str(one_run.seed) in text
    assert one_run.traceability.acnportal_version in text
    assert one_run.traceability.benchmark_date in text
    assert "acnportal" in text and "python" in text


def test_format_table_contains_all_runs(store: ResultStore, demo_scenario):
    store.save_batch(run_batch([demo_scenario], ["FCFS", "EDF", "LLF"]))
    table = store.format_table()
    for algorithm in ("FCFS", "EDF", "LLF"):
        assert algorithm in table
    assert "complete" in table


def test_list_batches_reports_run_counts(store: ResultStore, demo_scenario):
    batch = run_batch([demo_scenario], ["FCFS", "EDF"], store=store)
    listed = store.list_batches()
    assert len(listed) == 1
    assert listed[0]["batch_id"] == batch.batch_id
    assert listed[0]["n_runs"] == 2
    assert listed[0]["algorithms"] == ["EDF", "FCFS"]


def test_empty_store_is_safe(store: ResultStore):
    assert store.counts() == {"batches": 0, "runs": 0}
    assert store.list_runs() == []
    assert store.list_batches() == []
    assert store.format_table() == "（无匹配结果）"


# ----------------------------------------------------------------------
# 资源
# ----------------------------------------------------------------------


def test_no_connection_leak(store: ResultStore, one_run):
    """每次存储操作都必须关闭连接。

    `with sqlite3.connect(...) as conn` 只提交事务、不关闭连接 —— 用它实现会每次操作
    泄漏一个句柄。Store 是长驻的（Streamlit 里每个交互都会调用），泄漏会持续累积。
    """
    import gc
    import warnings

    store.save_run(one_run)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        for _ in range(8):
            store.counts()
            store.list_runs()
            store.list_batches()
            store.get_run(one_run.run_id)
        store.save_run(one_run)  # 幂等路径同样不得泄漏
        gc.collect()

    leaked = [str(w.message) for w in caught if "unclosed database" in str(w.message)]
    assert not leaked, f"泄漏了 {len(leaked)} 个数据库连接：{leaked[:2]}"


def test_writes_are_committed(store: ResultStore, one_run):
    """提交语义不能因为改用新的事务封装而丢失 —— 换一个连接重新读必须能读到。"""
    store.save_run(one_run)
    fresh = ResultStore(store.root)
    assert fresh.get_run(one_run.run_id).run_id == one_run.run_id


def test_failed_write_does_not_leave_partial_state(store: ResultStore, one_run):
    """写入失败时回滚，不得留下半条记录。"""
    from chargebench.storage import ResultMismatchError

    store.save_run(one_run)
    tampered = one_run.model_copy(
        update={
            "metrics": one_run.metrics.model_copy(
                update={"energy_cost_cny": one_run.metrics.energy_cost_cny + 5.0}
            )
        }
    )
    with pytest.raises(ResultMismatchError):
        store.save_run(tampered)
    # 原记录必须完好无损
    assert store.get_run(one_run.run_id).metrics.energy_cost_cny == pytest.approx(
        one_run.metrics.energy_cost_cny
    )
