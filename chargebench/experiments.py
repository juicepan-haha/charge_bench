"""实验执行入口。

上层（UI / CLI / Agent / MCP）只调用本模块的 ``run_experiment`` 与 ``run_batch``，
不直接接触 ACN-Sim。

公平性由构造方式保证，而非靠调用者自觉：
  * 每个算法都从 ``build_scenario`` 重新展开场景、重建网络与事件队列，
    因此不存在「后跑的算法拿到被前一个算法改写过的 EV」这种脏数据问题；
  * 场景展开完全由 ``Scenario.seed`` 决定，同一 (scenario, seed) 必然得到同一批会话；
  * ``run_id`` 是 (场景, 算法, 参数, 种子) 的确定性函数，同配置在任何机器上都一致。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .adapter import build_scenario, build_simulator
from .algorithms import build_algorithm
from .metrics import compute_metrics
from .schemas import (
    BENCHMARK_DATE,
    SCHEMA_VERSION,
    BatchResult,
    RunResult,
    Scenario,
    Traceability,
)
from .storage import ResultStore
from .tariff import simulation_start


def _traceability(start: dt.datetime) -> Traceability:
    return Traceability(
        schema_version=SCHEMA_VERSION,
        acnportal_version=importlib.metadata.version("acnportal"),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        benchmark_date=BENCHMARK_DATE.isoformat(),
        simulation_start=start,
        executed_at=dt.datetime.now(),
    )


def run_experiment(
    scenario: Scenario,
    algorithm: str,
    algorithm_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> RunResult:
    """运行一次仿真并返回标准化结果。

    Args:
        scenario: 场景定义。
        algorithm: 算法名，见 ``algorithms.list_algorithms()``。
        algorithm_params: 对算法默认参数的覆盖，未知参数会直接报错。
        seed: 覆盖场景种子，用于「同一场景跨多个种子取均值」（方案 §5 公平性原则 2）。
              为 None 时使用 ``scenario.seed``。

    本函数保持无副作用：不读也不写任何存储。需要复用/落盘请用 ``run_batch(store=...)``。
    """
    params = dict(algorithm_params or {})
    return run_with_scheduler(
        scenario,
        algorithm_label=algorithm,
        algorithm_params=params,
        seed=seed,
        scheduler_factory=lambda: build_algorithm(algorithm, params),
    )


def run_with_scheduler(
    scenario: Scenario,
    algorithm_label: str,
    scheduler_factory: Callable[[], Any],
    algorithm_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> RunResult:
    """用**任意调度器**跑一次仿真。

    供 Agent 闭环评测候选策略使用 —— 候选策略不在注册表里，无法按名字构造。
    `run_experiment` 也走这条路径，因此两者共用同一套构造与埋点，指标口径必然一致。

    Args:
        algorithm_label: 写进 run_id 与结果的算法标识。候选策略用它自己的名字，
            因此不同候选的 run_id 天然互不相同且可追溯。
        scheduler_factory: 无参可调用对象，返回一个 BaseAlgorithm 实例。做成工厂而非
            直接传实例，是为了每次运行都能拿到全新实例（调度器会在运行中持有状态）。
    """
    params = dict(algorithm_params or {})
    effective_seed = scenario.seed if seed is None else seed
    if effective_seed != scenario.seed:
        scenario = scenario.with_updates(seed=effective_seed)

    start = simulation_start(scenario.start_hour, BENCHMARK_DATE)
    artifacts = build_scenario(scenario)
    simulator = build_simulator(artifacts, scheduler_factory(), start)

    started = time.perf_counter()
    simulator.run()
    runtime_s = time.perf_counter() - started

    metrics = compute_metrics(simulator, artifacts, runtime_s, scenario.price_profile)

    return RunResult(
        run_id=RunResult.make_run_id(scenario, algorithm_label, params, effective_seed),
        scenario_id=scenario.scenario_id,
        scenario_hash=scenario.scenario_hash,
        algorithm=algorithm_label,
        algorithm_params=params,
        seed=effective_seed,
        metrics=metrics,
        traceability=_traceability(start),
    )


def run_batch(
    scenarios: Sequence[Scenario],
    algorithms: Iterable[str],
    seeds: Sequence[int] | None = None,
    store: ResultStore | None = None,
    deadline_seconds: float | None = None,
) -> BatchResult:
    """在同样的场景集合上比较多个算法。

    ``seeds`` 给定时，每个 (场景, 算法) 组合会在每个种子上各跑一次 —— 单次结果
    会被随机性掩盖差异，必须跨种子取均值与波动范围（方案 §5 公平性原则 2）。

    ``store`` 给定时，命中缓存的 (场景, 算法, 参数, 种子) 组合会直接复用已有结果，
    并把新算的结果落盘。复用是透明的：由于 run_id 是配置的确定性函数，
    复用与重算得到的 RunResult 除 ``runtime_s`` / ``executed_at`` 外完全一致，
    而存储层会强制校验这一点（不一致会抛 ``ResultMismatchError``）。
    """
    algorithm_list = list(algorithms)
    seeds = list(seeds) if seeds is not None else [None]  # type: ignore[list-item]
    started = time.perf_counter()
    truncated = False

    runs: list[RunResult] = []
    for scenario in scenarios:
        for algorithm in algorithm_list:
            for seed in seeds:
                if (
                    deadline_seconds is not None
                    and time.perf_counter() - started > deadline_seconds
                ):
                    truncated = True
                    break
                effective_seed = scenario.seed if seed is None else seed
                cached = None
                if store is not None:
                    cached = store.find_existing(
                        scenario.scenario_hash, algorithm, effective_seed
                    )
                if cached is not None:
                    runs.append(cached)
                    continue

                result = run_experiment(scenario, algorithm, seed=seed)
                if store is not None:
                    store.save_run(result)
                runs.append(result)
            if truncated:
                break
        if truncated:
            break

    if not runs:
        raise TimeoutError(
            f"在 {deadline_seconds}s 内没有跑完任何一次运行。请提高时间预算或减小批次规模。"
        )

    effective_seeds = sorted({r.seed for r in runs})
    payload = json.dumps(
        {
            "scenario_hashes": sorted({r.scenario_hash for r in runs}),
            "algorithms": sorted(algorithm_list),
            "seeds": effective_seeds,
            "schema_version": SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    batch_id = "batch--" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    batch = BatchResult(
        batch_id=batch_id,
        created_at=dt.datetime.now(),
        scenario_ids=sorted({r.scenario_id for r in runs}),
        algorithms=sorted(algorithm_list),
        seeds=effective_seeds,
        runs=runs,
        truncated=truncated,
    )
    if store is not None:
        store.save_batch(batch)
    return batch
