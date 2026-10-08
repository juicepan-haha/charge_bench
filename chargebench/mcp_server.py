"""MCP Server：把平台的评测能力暴露给支持 MCP 的 AI 客户端。

方案 §7 的工程约束，逐条落在代码里：

| 约束 | 落点 |
|---|---|
| 参数白名单 | 所有入参都用 pydantic 模型校验，``extra="forbid"`` |
| 禁止任意 Python 代码执行 | 算法只来自注册表；候选策略只来自 ``SortKey`` 受控词汇；本模块不 ``eval`` 任何输入 |
| 限制批次规模 | ``ServerLimits`` 对场景数/算法数/种子数/总运行数设上限，超出即拒 |
| 超时与取消 | 每次调用带 ``deadline_seconds``，超时返回**部分结果并标记 truncated** |
| 结果持久化 | 结果写入服务端固定的结果库；工具返回 ``run_id``/``batch_id`` 供追溯 |
| 错误码标准化 | ``ErrorCode`` 枚举 + 统一返回信封 ``{ok, data}`` / ``{ok: false, error}`` |

**安全边界**：结果库路径由服务端在构造时固定，**不接受客户端传入**。
否则客户端就能借工具读写任意路径。同理，场景可以由客户端内联传入（经 schema 校验），
也可以按名字从 ``configs/`` 白名单加载，但不能传任意文件路径。

**架构**：工具实现是普通同步函数，MCP 注册只是薄薄一层。因此测试可以直接调用工具函数，
不需要起传输层 —— ``tests/test_mcp_server.py`` 就是这么做的。
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from .agent_loop import BASELINE_SPECS
from .algorithms import ALGORITHMS, CORE_ALGORITHMS, list_algorithms
from .experiments import run_batch
from .schemas import (
    METRIC_DIRECTIONS,
    SCHEMA_VERSION,
    SortKey,
    Scenario,
    SweepResult,
)
from .storage import ResultStore
from .sweep import grid_scenario_id, run_sweep

SERVER_NAME = "chargebench"
SERVER_VERSION = "0.1.0"


class ErrorCode(str, Enum):
    """标准化错误码。客户端据此分支处理，不必解析错误文案。"""

    invalid_argument = "invalid_argument"
    unknown_algorithm = "unknown_algorithm"
    unknown_scenario = "unknown_scenario"
    not_found = "not_found"
    budget_exceeded = "budget_exceeded"
    timeout = "timeout"
    internal_error = "internal_error"


@dataclass(frozen=True)
class ServerLimits:
    """服务端的硬上限。方案 §7 要求「限制批次规模」。"""

    max_scenarios: int = 20
    max_algorithms: int = 8
    max_seeds: int = 8
    max_runs_per_call: int = 400
    default_deadline_seconds: float = 120.0
    max_deadline_seconds: float = 900.0


class BatchRequest(BaseModel):
    """``run_batch`` 的入参。白名单校验，未知字段直接拒绝。"""

    model_config = {"extra": "forbid"}

    scenario: Optional[dict[str, Any]] = Field(
        None, description="内联场景对象（按 Scenario schema 校验）。与 config_name 二选一"
    )
    config_name: Optional[str] = Field(
        None, description="configs/ 下的场景名（白名单）。与 scenario 二选一"
    )
    algorithms: list[str] = Field(
        default_factory=lambda: list(CORE_ALGORITHMS), max_length=8
    )
    seeds: Optional[list[int]] = Field(None, max_length=8)
    deadline_seconds: Optional[float] = Field(None, gt=0)
    parallel_scenarios: int = Field(
        1, ge=1, le=5,
        description="在同一批里额外派生几个负荷强度扰动场景，用于看跨工况稳健性",
    )


class SweepRequest(BaseModel):
    model_config = {"extra": "forbid"}

    scenario: Optional[dict[str, Any]] = None
    config_name: Optional[str] = None
    x_values: list[float] = Field(min_length=2, max_length=12, description="负荷强度网格")
    y_values: list[float] = Field(min_length=2, max_length=12, description="时间紧迫度网格")
    algorithms: list[str] = Field(
        default_factory=lambda: list(CORE_ALGORITHMS), max_length=8
    )
    seeds: Optional[list[int]] = Field(None, max_length=8)
    primary_metric: str = "session_completion_rate"
    deadline_seconds: Optional[float] = Field(None, gt=0)


# ----------------------------------------------------------------------
# 返回信封
# ----------------------------------------------------------------------


def ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data}


def err(code: ErrorCode, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code.value, "message": message, **extra}}


# ----------------------------------------------------------------------
# 工具实现
# ----------------------------------------------------------------------


class ChargeBenchTools:
    """工具集。与 MCP 传输层解耦，可直接在测试与脚本中调用。"""

    def __init__(
        self,
        store_dir: str | pathlib.Path,
        config_dir: str | pathlib.Path | None = None,
        limits: ServerLimits | None = None,
    ):
        self.store = ResultStore(store_dir)
        self.config_dir = pathlib.Path(config_dir) if config_dir else None
        self.limits = limits or ServerLimits()

    # ------------------------------------------------------------------

    def list_algorithms(self) -> dict[str, Any]:
        """列出可用算法、各自可见的信息，以及候选策略的受控词汇。"""
        return ok(
            {
                "schema_version": SCHEMA_VERSION,
                "algorithms": list_algorithms(),
                "core_algorithms": list(CORE_ALGORITHMS),
                "metrics": {
                    "available": sorted(METRIC_DIRECTIONS),
                    "directions": METRIC_DIRECTIONS,
                },
                "candidate_vocabulary": {
                    "note": (
                        "候选策略只能由此词汇组合，服务端不执行任何客户端提供的代码。"
                    ),
                    "sort_keys": [k.value for k in SortKey],
                    "options": ["direction: asc|desc", "completion_first: bool",
                                "tiebreak: <sort_key>|null"],
                    "baselines": {
                        name: spec.rationale for name, spec in BASELINE_SPECS.items()
                    },
                },
                "limits": {
                    "max_scenarios": self.limits.max_scenarios,
                    "max_algorithms": self.limits.max_algorithms,
                    "max_seeds": self.limits.max_seeds,
                    "max_runs_per_call": self.limits.max_runs_per_call,
                    "max_deadline_seconds": self.limits.max_deadline_seconds,
                },
            }
        )

    # ------------------------------------------------------------------

    def run_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        """在给定场景上比较多个算法。返回 batch_id 与各运行的指标摘要。"""
        try:
            req = BatchRequest(**request)
        except ValidationError as exc:
            return err(ErrorCode.invalid_argument, _validation_message(exc))

        resolved = self._resolve_scenarios(req.scenario, req.config_name, req.parallel_scenarios)
        if isinstance(resolved, dict):
            return resolved  # 已经是错误信封
        scenarios = resolved

        unknown = [a for a in req.algorithms if a not in ALGORITHMS]
        if unknown:
            return err(
                ErrorCode.unknown_algorithm,
                f"未知算法 {unknown}。可用：{sorted(ALGORITHMS)}",
                available=sorted(ALGORITHMS),
            )
        if not req.algorithms:
            return err(ErrorCode.invalid_argument, "algorithms 不能为空")

        seeds = req.seeds or [scenarios[0].seed]
        planned = len(scenarios) * len(req.algorithms) * len(seeds)
        if planned > self.limits.max_runs_per_call:
            return err(
                ErrorCode.budget_exceeded,
                f"本次将运行 {planned} 次仿真，超过单次调用上限 "
                f"{self.limits.max_runs_per_call}。请减少场景、算法或种子。",
                planned_runs=planned,
                limit=self.limits.max_runs_per_call,
            )

        deadline = self._deadline(req.deadline_seconds)
        try:
            batch = run_batch(scenarios, req.algorithms, seeds=seeds,
                              store=self.store, deadline_seconds=deadline)
        except TimeoutError as exc:
            return err(ErrorCode.timeout, str(exc), deadline_seconds=deadline)

        return ok(
            {
                "batch_id": batch.batch_id,
                "truncated": batch.truncated,
                "planned_runs": planned,
                "completed_runs": len(batch.runs),
                "scenario_ids": batch.scenario_ids,
                "algorithms": batch.algorithms,
                "seeds": batch.seeds,
                "runs": [_run_summary(r) for r in batch.runs],
                "store": str(self.store.root),
                "note": (
                    "结果已持久化；可用 get_results 按 batch_id 取回，"
                    "或用 compare_experiments 做逐指标对比。"
                ),
            }
        )

    # ------------------------------------------------------------------

    def get_results(self, batch_id: str) -> dict[str, Any]:
        """按 batch_id 取回结构化结果与数据位置。"""
        try:
            batch = self.store.get_batch(batch_id)
        except KeyError:
            return err(ErrorCode.not_found, f"未找到 batch_id {batch_id!r}")

        return ok(
            {
                "batch_id": batch.batch_id,
                "truncated": batch.truncated,
                "scenario_ids": batch.scenario_ids,
                "algorithms": batch.algorithms,
                "seeds": batch.seeds,
                "runs": [_run_summary(r) for r in batch.runs],
                "json_paths": [
                    str(self.store.runs_dir / f"{r.run_id}.json") for r in batch.runs
                ],
                "note": "每条 run 的 run_id 可追溯到场景、算法、参数、种子与依赖版本。",
            }
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        """取回单次运行的完整结果。"""
        try:
            run = self.store.get_run(run_id)
        except KeyError:
            return err(ErrorCode.not_found, f"未找到 run_id {run_id!r}")
        return ok(json.loads(run.model_dump_json()))

    # ------------------------------------------------------------------

    def compare_experiments(self, run_ids: list[str]) -> dict[str, Any]:
        """逐指标对比若干次运行。

        **不给出单一「谁更好」的结论** —— 方案 §0 的核心事实就是排名取决于目标函数。
        这里逐指标分别标出胜者，把「取决于你把什么当目标」直接摆在结果里。
        """
        if not run_ids:
            return err(ErrorCode.invalid_argument, "run_ids 不能为空")
        if len(run_ids) > self.limits.max_runs_per_call:
            return err(
                ErrorCode.budget_exceeded,
                f"一次最多对比 {self.limits.max_runs_per_call} 条运行",
            )

        runs = []
        for rid in run_ids:
            try:
                runs.append(self.store.get_run(rid))
            except KeyError:
                return err(ErrorCode.not_found, f"未找到 run_id {rid!r}")

        # 只有定义了方向的指标才排名。**约束违规不参与排名** —— 平台的规则是
        # 「违规即取消资格」（方案 §5 原则 3），而不是「违规少者胜出」；
        # 给违规数排个名本身就是误导。同理，违规候选不参与任何指标的胜者评选。
        disqualified = sorted(
            r.run_id for r in runs if r.metrics.constraint_violations > 0
        )
        eligible = [r for r in runs if r.metrics.constraint_violations == 0]

        ranked_metrics = sorted(METRIC_DIRECTIONS)
        table: dict[str, dict[str, float]] = {}
        winners: dict[str, Optional[str]] = {}
        for metric in ranked_metrics:
            table[metric] = {r.run_id: float(getattr(r.metrics, metric)) for r in runs}
            if not eligible:
                winners[metric] = None
                continue
            best = min(
                eligible,
                key=lambda r: (
                    -getattr(r.metrics, metric)
                    if METRIC_DIRECTIONS[metric] == "max"
                    else getattr(r.metrics, metric)
                ),
            )
            winners[metric] = best.run_id

        # 同一场景集合才可比 —— 不同场景的数字放一起比是没有意义的
        scenario_hashes = {r.scenario_hash for r in runs}
        incomparable = len(scenario_hashes) > 1
        decided = [w for w in winners.values() if w is not None]

        payload: dict[str, Any] = {
            "values": table,
            "diagnostics": {
                "constraint_violations": {
                    r.run_id: r.metrics.constraint_violations for r in runs
                },
                "sessions_dropped": {
                    r.run_id: r.metrics.sessions_dropped for r in runs
                },
            },
            "disqualified": disqualified,
            "winner_by_metric": winners,
            "winner_disagrees": len(set(decided)) > 1,
            "note": (
                "不同指标可能给出不同的胜者 —— 这正是平台要说明的事实，"
                "因此这里不做单一排名。约束违规的候选已被排除在胜者评选之外"
                "（违规即取消资格，而不是违规少者胜出）。"
            ),
        }
        if incomparable:
            payload["warning"] = (
                "这些运行来自不同场景，指标不可直接互相比较。"
                "请只在同一 scenario_hash 内对比。"
            )
            payload["scenario_hashes"] = sorted(scenario_hashes)
        return ok(payload)

    # ------------------------------------------------------------------

    def parameter_sweep(self, request: dict[str, Any]) -> dict[str, Any]:
        """两变量扫描，找算法适用边界。返回胜者地图与评分规则。"""
        try:
            req = SweepRequest(**request)
        except ValidationError as exc:
            return err(ErrorCode.invalid_argument, _validation_message(exc))

        if req.primary_metric not in METRIC_DIRECTIONS:
            return err(
                ErrorCode.invalid_argument,
                f"primary_metric 必须是 {sorted(METRIC_DIRECTIONS)} 之一",
                available=sorted(METRIC_DIRECTIONS),
            )

        resolved = self._resolve_scenarios(req.scenario, req.config_name, 1)
        if isinstance(resolved, dict):
            return resolved
        base = resolved[0]

        unknown = [a for a in req.algorithms if a not in ALGORITHMS]
        if unknown:
            return err(ErrorCode.unknown_algorithm, f"未知算法 {unknown}",
                       available=sorted(ALGORITHMS))

        seeds = req.seeds or [base.seed]
        planned = len(req.x_values) * len(req.y_values) * len(req.algorithms) * len(seeds)
        if planned > self.limits.max_runs_per_call:
            return err(
                ErrorCode.budget_exceeded,
                f"网格共需 {planned} 次仿真，超过单次调用上限 "
                f"{self.limits.max_runs_per_call}。请缩小网格或减少种子。",
                planned_runs=planned,
                limit=self.limits.max_runs_per_call,
            )

        deadline = self._deadline(req.deadline_seconds)
        started = dt.datetime.now()
        sweep = run_sweep(
            base, req.x_values, req.y_values, req.algorithms, seeds,
            primary_metric=req.primary_metric, store=self.store,
        )

        winners, constrained = sweep.winner_grid()
        return ok(
            {
                "sweep_id": sweep.sweep_id,
                "base_scenario_id": sweep.base_scenario_id,
                "primary_metric": sweep.primary_metric,
                "metric_direction": sweep.metric_direction,
                "score_rule": sweep.score_rule,
                "x_name": sweep.x_name, "x_values": sweep.x_values,
                "y_name": sweep.y_name, "y_values": sweep.y_values,
                "algorithms": sweep.algorithms,
                "seeds": sweep.seeds,
                "winner_grid": winners,
                "port_constrained_grid": constrained,
                "cells": [
                    {
                        "load_intensity": c.load_intensity,
                        "deadline_tightness": c.deadline_tightness,
                        "port_occupancy_estimate": round(c.port_occupancy_estimate, 4),
                        "port_constrained": c.port_constrained,
                        "sessions_dropped": c.sessions_dropped,
                        "winner": c.winner,
                        "disqualified": c.disqualified,
                        "aggregates": {
                            name: agg.model_dump(mode="json")
                            for name, agg in c.aggregates.items()
                        },
                    }
                    for c in sweep.cells
                ],
                "elapsed_seconds": round((dt.datetime.now() - started).total_seconds(), 3),
                "deadline_seconds": deadline,
                "n_run_ids": len(sweep.run_ids),
                "note": (
                    "评分规则事前写定并随结果持久化，不会看结果后改口径。"
                    "port_constrained 为 true 的格点有大量车辆因无空位被丢弃，"
                    "满足率结论需谨慎。"
                ),
            }
        )

    # ------------------------------------------------------------------

    def list_batches(self, limit: int = 20) -> dict[str, Any]:
        """列出结果库里的批次。"""
        batches = self.store.list_batches()[: max(1, min(limit, 100))]
        return ok({"batches": batches, "counts": self.store.counts()})

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _deadline(self, requested: float | None) -> float:
        value = requested or self.limits.default_deadline_seconds
        return min(value, self.limits.max_deadline_seconds)

    def _resolve_scenarios(
        self, inline: dict[str, Any] | None, config_name: str | None, extra: int
    ) -> list[Scenario] | dict[str, Any]:
        """解析场景来源。返回 Scenario 列表，或一个错误信封。"""
        if (inline is None) == (config_name is None):
            return err(
                ErrorCode.invalid_argument,
                "必须且只能提供 scenario（内联对象）或 config_name（白名单名）之一",
            )

        if inline is not None:
            try:
                base = Scenario(**inline)
            except ValidationError as exc:
                return err(ErrorCode.invalid_argument, _validation_message(exc))
        else:
            base = self._load_config(config_name)  # type: ignore[arg-type]
            if isinstance(base, dict):
                return base

        scenarios = [base]
        for index in range(1, extra):
            scenarios.append(
                base.with_updates(
                    scenario_id=f"{base.scenario_id}_v{index + 1}",
                    load_intensity=round(base.load_intensity * (1.0 + 0.15 * index), 4),
                )
            )
        if len(scenarios) > self.limits.max_scenarios:
            return err(
                ErrorCode.budget_exceeded,
                f"场景数 {len(scenarios)} 超过上限 {self.limits.max_scenarios}",
            )
        return scenarios

    def _load_config(self, name: str) -> Scenario | dict[str, Any]:
        """按白名单名加载 configs/ 下的场景。

        只接受**文件名主干**，并拒绝任何路径分隔符 —— 客户端不能借此读取任意文件。
        """
        if self.config_dir is None:
            return err(ErrorCode.invalid_argument, "服务端未配置 configs 目录，请改用内联 scenario")
        if (
            pathlib.Path(name).name != name
            or "/" in name
            or "\\" in name
            or name.startswith(".")
        ):
            return err(
                ErrorCode.invalid_argument,
                f"config_name 只能是 configs/ 下的文件名主干，不接受路径。收到 {name!r}",
            )
        path = self.config_dir / f"{name}.json"
        if not path.is_file():
            available = sorted(p.stem for p in self.config_dir.glob("*.json"))
            return err(
                ErrorCode.unknown_scenario,
                f"未找到场景 {name!r}",
                available=available,
            )
        try:
            return Scenario(**json.loads(path.read_text(encoding="utf-8")))
        except ValidationError as exc:
            return err(ErrorCode.invalid_argument, f"场景文件不合法：{_validation_message(exc)}")


def _run_summary(run) -> dict[str, Any]:
    m = run.metrics
    return {
        "run_id": run.run_id,
        "scenario_id": run.scenario_id,
        "scenario_hash": run.scenario_hash,
        "algorithm": run.algorithm,
        "seed": run.seed,
        "metrics": {
            "demand_satisfaction_rate": round(m.demand_satisfaction_rate, 6),
            "session_completion_rate": round(m.session_completion_rate, 6),
            "energy_delivered_kwh": round(m.energy_delivered_kwh, 4),
            "energy_cost_cny": round(m.energy_cost_cny, 4),
            "peak_power_kw": round(m.peak_power_kw, 4),
            "constraint_violations": m.constraint_violations,
            "sessions_dropped": m.sessions_dropped,
            "runtime_s": round(m.runtime_s, 4),
        },
    }


def _validation_message(exc: ValidationError) -> str:
    """把 pydantic 的报错压成一句可读的话，避免把堆栈丢给客户端。"""
    parts = []
    for item in exc.errors()[:5]:
        location = ".".join(str(x) for x in item.get("loc", ())) or "(root)"
        parts.append(f"{location}: {item.get('msg')}")
    extra = "" if len(exc.errors()) <= 5 else f"（另有 {len(exc.errors()) - 5} 处）"
    return "；".join(parts) + extra


# ----------------------------------------------------------------------
# MCP 注册
# ----------------------------------------------------------------------


def build_server(
    store_dir: str | pathlib.Path,
    config_dir: str | pathlib.Path | None = None,
    limits: ServerLimits | None = None,
):
    """构造并返回一个配置好的 MCPServer。

    工具层的所有约束都在 ``ChargeBenchTools`` 里，这里只做注册。
    """
    from mcp.server.mcpserver import MCPServer

    tools = ChargeBenchTools(store_dir, config_dir, limits)
    server = MCPServer(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "新能源汽车充电调度评测平台。所有定量结果都来自仿真引擎，"
            "任何指标都不会由模型推断产生。\n"
            "注意：算法没有单一最优 —— 换一个优化目标，胜者就会变。"
            "对比结果时请先看清目标是哪一个。"
        ),
    )

    @server.tool(
        name="list_algorithms",
        description=(
            "列出可用算法、各自可见的信息，以及候选策略的受控词汇与服务端限制。"
            "开始任何实验前先调用它。"
        ),
    )
    def _list_algorithms() -> dict:
        return tools.list_algorithms()

    @server.tool(
        name="run_batch",
        description=(
            "在给定场景上比较多个算法。传入内联 scenario 对象或 config_name。"
            "返回 batch_id、各运行的指标摘要与数据位置。"
            "所有数字来自仿真，不做推断。"
        ),
    )
    def _run_batch(
        scenario: Optional[dict] = None,
        config_name: Optional[str] = None,
        algorithms: Optional[list[str]] = None,
        seeds: Optional[list[int]] = None,
        deadline_seconds: Optional[float] = None,
        parallel_scenarios: int = 1,
    ) -> dict:
        return tools.run_batch(
            {
                "scenario": scenario,
                "config_name": config_name,
                "algorithms": algorithms if algorithms is not None else list(CORE_ALGORITHMS),
                "seeds": seeds,
                "deadline_seconds": deadline_seconds,
                "parallel_scenarios": parallel_scenarios,
            }
        )

    @server.tool(
        name="get_results",
        description="按 batch_id 取回结构化结果与每条运行的 JSON 落盘位置。",
    )
    def _get_results(batch_id: str) -> dict:
        return tools.get_results(batch_id)

    @server.tool(
        name="get_run",
        description="按 run_id 取回单次运行的完整结果与依赖版本等可追溯信息。",
    )
    def _get_run(run_id: str) -> dict:
        return tools.get_run(run_id)

    @server.tool(
        name="compare_experiments",
        description=(
            "逐指标对比若干次运行。**不返回单一排名** —— 不同指标可能给出不同胜者，"
            "这是本平台的既定事实。请按目标函数解读。"
        ),
    )
    def _compare_experiments(run_ids: list[str]) -> dict:
        return tools.compare_experiments(run_ids)

    @server.tool(
        name="parameter_sweep",
        description=(
            "在「负荷强度 × 时间紧迫度」网格上逐点评测，返回算法适用区域图数据"
            "（winner_grid）与事前写定的评分规则。网格点数×算法数×种子数受服务端上限约束。"
        ),
    )
    def _parameter_sweep(
        scenario: Optional[dict] = None,
        config_name: Optional[str] = None,
        x_values: Optional[list[float]] = None,
        y_values: Optional[list[float]] = None,
        algorithms: Optional[list[str]] = None,
        seeds: Optional[list[int]] = None,
        primary_metric: str = "session_completion_rate",
        deadline_seconds: Optional[float] = None,
    ) -> dict:
        return tools.parameter_sweep(
            {
                "scenario": scenario,
                "config_name": config_name,
                "x_values": x_values,
                "y_values": y_values,
                "algorithms": algorithms if algorithms is not None else list(CORE_ALGORITHMS),
                "seeds": seeds,
                "primary_metric": primary_metric,
                "deadline_seconds": deadline_seconds,
            }
        )

    @server.tool(
        name="list_batches",
        description="列出结果库中已记录的批次与统计。",
    )
    def _list_batches(limit: int = 20) -> dict:
        return tools.list_batches(limit)

    server._chargebench_tools = tools  # 供测试直接访问工具层
    return server


def main() -> None:
    """以 stdio 传输启动服务端（方案 §7：先实现本机调用）。"""
    import argparse

    parser = argparse.ArgumentParser(prog="chargebench-mcp", description="ChargeBench MCP Server")
    parser.add_argument("--store", default="results", help="结果库目录")
    parser.add_argument("--configs", default="configs", help="场景配置目录")
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio", "sse", "streamable-http"]
    )
    parser.add_argument("--max-runs", type=int, default=ServerLimits.max_runs_per_call)
    args = parser.parse_args()

    limits = ServerLimits(max_runs_per_call=args.max_runs)
    server = build_server(args.store, args.configs, limits)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
