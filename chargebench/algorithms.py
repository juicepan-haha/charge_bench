"""算法注册表。

方案 §5 的公平性原则 1 要求「每个算法看到的车辆会话、基础设施与外部信息一致」。
这里把每个算法实际使用的信息显式写成元数据，而不是留在实现细节里 —— 评测结果
一旦被质疑，可以直接从注册表回答「它们是否在同等信息下比较」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from acnportal.algorithms import (
    BaseAlgorithm,
    SortedSchedulingAlgo,
    UncontrolledCharging,
    earliest_deadline_first,
    first_come_first_served,
    least_laxity_first,
)


@dataclass(frozen=True)
class AlgorithmSpec:
    """一个可评测算法的完整描述。"""

    name: str
    summary: str
    #: 该算法用于排序/决策的字段，用于核对公平性
    visible_info: tuple[str, ...]
    builder: Callable[[dict[str, Any]], BaseAlgorithm]
    params: dict[str, dict[str, Any]] = field(default_factory=dict)


def _sorted(sort_fn: Callable) -> Callable[[dict[str, Any]], BaseAlgorithm]:
    def build(params: dict[str, Any]) -> BaseAlgorithm:
        return SortedSchedulingAlgo(sort_fn, **params)

    return build


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "FCFS": AlgorithmSpec(
        name="FCFS",
        summary="先到先服务：按到达时间顺序分配充电功率。",
        visible_info=("arrival",),
        builder=_sorted(first_come_first_served),
        params={
            "allow_overcharging": False,
            "uninterrupted_charging": False,
            "estimate_max_rate": False,
        },
    ),
    "EDF": AlgorithmSpec(
        name="EDF",
        summary="最早离站优先：优先满足预计离开时间更早的车辆。",
        visible_info=("arrival", "estimated_departure"),
        builder=_sorted(earliest_deadline_first),
        params={
            "allow_overcharging": False,
            "uninterrupted_charging": False,
            "estimate_max_rate": False,
        },
    ),
    "LLF": AlgorithmSpec(
        name="LLF",
        summary=(
            "最小松弛时间优先：按 LAX = (预计离站 - 当前时刻) - 剩余需求/最大功率 升序分配。"
            "松弛时间每周期重算，因此已获供电的车辆会被逐渐降级。"
        ),
        visible_info=("arrival", "estimated_departure", "remaining_demand", "max_rate"),
        builder=_sorted(least_laxity_first),
        params={
            "allow_overcharging": False,
            "uninterrupted_charging": False,
            "estimate_max_rate": False,
        },
    ),
    "UNCONTROLLED": AlgorithmSpec(
        name="UNCONTROLLED",
        summary=(
            "无控制对照：插枪即满功率充电。不构成调度策略，"
            "作为「不做任何调度」的参考基准，用于量化调度带来的改善。"
        ),
        visible_info=("max_rate",),
        builder=lambda params: UncontrolledCharging(),
        params={},
    ),
}

#: 方案 §5 规定的最小算法集合，任何评测都至少包含这三个
CORE_ALGORITHMS: tuple[str, ...] = ("FCFS", "EDF", "LLF")


def build_algorithm(name: str, params: dict[str, Any] | None = None) -> BaseAlgorithm:
    """构造算法实例。未知名称或未知参数立即报错，不做静默降级。"""
    if name not in ALGORITHMS:
        raise KeyError(f"未知算法 {name!r}，可用：{sorted(ALGORITHMS)}")
    spec = ALGORITHMS[name]
    merged = dict(spec.params)
    if params:
        unknown = set(params) - set(spec.params)
        if unknown:
            raise KeyError(f"算法 {name} 不支持的参数：{sorted(unknown)}")
        merged.update(params)
    return spec.builder(merged)


def list_algorithms() -> list[dict[str, Any]]:
    """算法清单，供 UI 下拉框与 MCP 的 list_algorithms 工具直接消费。"""
    return [
        {
            "name": spec.name,
            "summary": spec.summary,
            "visible_info": list(spec.visible_info),
            "params": spec.params,
        }
        for spec in ALGORITHMS.values()
    ]
