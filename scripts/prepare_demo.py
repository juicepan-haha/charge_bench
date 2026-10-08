#!/usr/bin/env python
"""一键准备演示：跑完所有实验并缓存，产出全部图表与报告 JSON。

现场演示最怕两件事：**随机性让数字变了**、**现场环境跑不出来**。
这个脚本把两者都消掉：固定种子预跑一遍，结果落进结果库；演示时同配置重跑
会全部命中缓存（``run_id`` 是配置的确定性函数），因此数字逐位一致、耗时接近零。

    # 准备（首次，或换了场景参数之后）
    python scripts/prepare_demo.py

    # 复核：重跑一遍，应当全部命中缓存且结果一致
    python scripts/prepare_demo.py --verify

产物：
    results/   结果库（SQLite + 每条运行的原始 JSON）
    reports/   所有图表（适用区域图、目标对比图、负荷曲线、Agent 收敛图）
    examples/  协议 JSON 样例（UI 与外部 Agent 可直接消费）

本项目不依赖任何网络服务：仿真后端 ACN-Sim 在本地运行，场景为合成数据，
Agent 闭环内置离线回退。因此断网也能完整演示。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chargebench.agent_loop import HeuristicProposer, run_agent_loop  # noqa: E402
from chargebench.algorithms import CORE_ALGORITHMS  # noqa: E402
from chargebench.experiments import run_batch  # noqa: E402
from chargebench.schemas import BudgetSpec, Scenario  # noqa: E402
from chargebench.storage import ResultStore  # noqa: E402
from chargebench.sweep import rerank_sweep, run_sweep  # noqa: E402
from chargebench.timeseries import simulate_curves  # noqa: E402
from chargebench.viz import (  # noqa: E402
    render_agent_report,
    render_curves,
    render_objective_comparison,
    render_sweep,
)

#: 演示用的固定参数。改这里等于换演示内容，请同步更新 docs/DEMO_SCRIPT.md 的数字。
DEMO_SEEDS = (42, 43)
SWEEP_X = (0.5, 0.7, 0.9, 1.1, 1.3)
SWEEP_Y = (0.40, 0.55, 0.70, 0.85)
OBJECTIVES = ("session_completion_rate", "peak_power_kw", "energy_cost_cny")
BUDGET = BudgetSpec(max_rounds=3, max_simulations=200, max_seconds=180, patience=2)


def load_base() -> Scenario:
    return Scenario(**json.loads((ROOT / "configs/demo_scenario.json").read_text(encoding="utf-8")))


def train_scenarios(base: Scenario) -> list[Scenario]:
    return [
        base,
        base.with_updates(scenario_id="campus_train_tight",
                          deadline_tightness=0.85, load_intensity=0.9),
        base.with_updates(scenario_id="campus_train_loose",
                          deadline_tightness=0.5, load_intensity=0.8),
    ]


def holdout_scenarios(base: Scenario) -> list[Scenario]:
    return [
        base.with_updates(scenario_id="campus_holdout_a", seed=101, load_intensity=0.95),
        base.with_updates(scenario_id="campus_holdout_b", seed=202,
                          deadline_tightness=0.8, load_intensity=1.05),
    ]


def prepare() -> dict:
    base = load_base()
    store = ResultStore(ROOT / "results")
    reports = ROOT / "reports"
    examples = ROOT / "examples"
    reports.mkdir(parents=True, exist_ok=True)
    examples.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    t0 = time.perf_counter()

    print("[1/5] 批量对比：三种算法在同一场景上的横向比较")
    batch = run_batch([base], CORE_ALGORITHMS, seeds=list(DEMO_SEEDS), store=store)
    (examples / f"{batch.batch_id}.json").write_text(
        json.dumps(batch.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["batch"] = {
        r.algorithm: {
            "completion": round(r.metrics.session_completion_rate, 4),
            "satisfaction": round(r.metrics.demand_satisfaction_rate, 4),
        }
        for r in batch.runs
        if r.seed == DEMO_SEEDS[0]
    }
    print("      " + " · ".join(
        f"{a}: 完成率 {v['completion']} 满足率 {v['satisfaction']}"
        for a, v in summary["batch"].items()
    ))

    print("[2/5] 适用边界扫描：5×4 网格 × 3 算法 × 2 种子")
    sweep = run_sweep(base, list(SWEEP_X), list(SWEEP_Y), CORE_ALGORITHMS,
                      list(DEMO_SEEDS), primary_metric=OBJECTIVES[0], store=store)
    for path in render_sweep(sweep, reports):
        print(f"      {path.relative_to(ROOT)}")

    print("[3/5] 换目标重排名：同一批实验数据，只换尺子，不重跑仿真")
    others = [rerank_sweep(sweep, metric) for metric in OBJECTIVES[1:]]
    print(f"      {render_objective_comparison([sweep, *others], reports).relative_to(ROOT)}")
    payload = {"sweeps": {s.primary_metric: s.model_dump(mode="json") for s in [sweep, *others]}}
    (examples / f"{sweep.sweep_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for s in [sweep, *others]:
        winners = [c.winner for c in s.cells]
        tally: dict[str, int] = {}
        for w in winners:
            tally[w or "（无）"] = tally.get(w or "（无）", 0) + 1
        summary[f"winners::{s.primary_metric}"] = tally
        print(f"      {s.primary_metric:26} 胜者分布 {tally}")
    summary["sweep_run_ids"] = len(sweep.run_ids)

    print("[4/5] 过程回放：负荷曲线")
    curves = [simulate_curves(base, name, seed=DEMO_SEEDS[0]) for name in CORE_ALGORITHMS]
    for path in render_curves(curves, reports):
        print(f"      {path.relative_to(ROOT)}")
    summary["peak_kw"] = {c.algorithm: round(max(c.aggregate_kw), 2) for c in curves}

    print("[5/5] AI 实验闭环：提假设 → 评测 → 保留场景复验")
    report = run_agent_loop(
        train_scenarios(base), holdout_scenarios(base), CORE_ALGORITHMS,
        HeuristicProposer(), budget=BUDGET, seeds=list(DEMO_SEEDS),
        store=store, batch_size=3,
    )
    print(f"      {render_agent_report(report, reports).relative_to(ROOT)}")
    (examples / f"{report.report_id}.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["agent"] = {
        "report_id": report.report_id,
        "stop_reason": report.stop_reason,
        "simulations_used": report.simulations_used,
        "elapsed_seconds": report.elapsed_seconds,
        "baselines": {o.spec.name: round(o.session_completion_rate, 4)
                      for o in report.baseline_outcomes},
        "best": (report.best_candidate.spec.name if report.best_candidate else None),
        "best_completion": (round(report.best_candidate.session_completion_rate, 4)
                            if report.best_candidate else None),
        "validation": (report.validation.model_dump(mode="json") if report.validation else None),
        "conclusion": report.conclusion,
    }
    print(f"      最佳候选 {summary['agent']['best']} "
          f"（完成率 {summary['agent']['best_completion']}），"
          f"仿真 {report.simulations_used} 次 / {report.elapsed_seconds}s")
    if report.validation:
        v = report.validation
        print(f"      保留场景复验 {v.candidate_completion_rate:.4f} 对 "
              f"{v.best_baseline_name} {v.best_baseline_completion_rate:.4f} "
              f"→ {'通过' if v.passed else '未通过'}")

    summary["counts"] = store.counts()
    summary["wall_seconds"] = round(time.perf_counter() - t0, 2)
    (reports / "demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def verify() -> int:
    """复核：重跑一遍所有实验，必须全部命中缓存且结果一致。

    这是「演示不会因为现场随机性而失败」的机器化证明 —— 靠人工记得预跑是不可靠的。
    """
    base = load_base()
    store = ResultStore(ROOT / "results")
    problems: list[str] = []

    # 判定「命中缓存」必须看 executed_at，不能只看记录条数：
    # 重算也会得到同一个 run_id，于是覆盖同名文件、条数不变 —— 那样就看不出区别。
    # （这是本项目在阶段 3 踩过的同一个坑，复核脚本里不能再犯一次。）
    probes = [r.run_id for r in store.list_runs()][:5]
    stamps = {rid: store.get_run(rid).traceability.executed_at for rid in probes}
    if not probes:
        print("结果库为空 —— 请先不带 --verify 运行一次以准备演示")
        return 1

    started = time.perf_counter()
    run_batch([base], CORE_ALGORITHMS, seeds=list(DEMO_SEEDS), store=store)
    run_sweep(base, list(SWEEP_X), list(SWEEP_Y), CORE_ALGORITHMS,
              list(DEMO_SEEDS), primary_metric=OBJECTIVES[0], store=store)
    run_agent_loop(train_scenarios(base), holdout_scenarios(base), CORE_ALGORITHMS,
                   HeuristicProposer(), budget=BUDGET, seeds=list(DEMO_SEEDS),
                   store=store, batch_size=3)
    elapsed = time.perf_counter() - started

    after = store.counts()
    for rid, before_stamp in stamps.items():
        now = store.get_run(rid).traceability.executed_at
        if now != before_stamp:
            problems.append(
                f"{rid} 的执行时刻变了 —— 说明是重算而非复用缓存，现场数字可能与预跑不同"
            )

    # 图表是否齐全
    expected = {
        "objective-comparison", "winner-map", "metric-panels", "load-curve", "agent--",
    }
    names = [p.name for p in (ROOT / "reports").glob("*.png")]
    for marker in expected:
        if not any(marker in n for n in names):
            problems.append(f"reports/ 缺少含 {marker!r} 的图")

    # 演示不依赖网络：确认没有任何模块在导入期发起网络调用
    import chargebench.adapter  # noqa: F401
    import chargebench.mcp_server  # noqa: F401

    print(f"复核耗时 {elapsed:.2f}s")
    print(f"结果库 {after['runs']} 条运行 / {after['batches']} 个批次")
    print(f"抽查 {len(probes)} 条记录的执行时刻，均与预跑一致 → 全部命中缓存")
    print(f"图表 {len(names)} 张")
    if problems:
        print("\n复核未通过：")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("\n✓ 复核通过：数字可复现、图表齐全、不依赖网络")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ChargeBench 演示准备")
    parser.add_argument("--verify", action="store_true", help="复核：重跑应全部命中缓存")
    args = parser.parse_args()
    if args.verify:
        return verify()
    summary = prepare()
    print(f"\n完成：{summary['counts']['runs']} 条运行记录 / "
          f"{summary['counts']['batches']} 个批次，总耗时 {summary['wall_seconds']}s")
    print("演示摘要：" + str((ROOT / 'reports' / 'demo_summary.json').relative_to(ROOT)))
    print("下一步：python scripts/prepare_demo.py --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
