"""命令行入口。

让不写 Python 的队友（UI / 报告）也能直接产出并查询协议规定的 JSON：

    python -m chargebench.cli list-algorithms
    python -m chargebench.cli run --scenario configs/demo_scenario.json \
        --algorithm FCFS EDF LLF --store results/
    python -m chargebench.cli results --store results/
    python -m chargebench.cli describe --store results/ <run_id>

输出严格遵循 docs/PROTOCOL.md 定义的 BatchResult / RunResult 结构。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

from .algorithms import CORE_ALGORITHMS, list_algorithms
from .experiments import run_batch
from .schemas import Scenario
from .sweep import X_FIELD, Y_FIELD
from .storage import ResultStore

# acnportal 依赖已被 setuptools 弃用的 pkg_resources（PLAN.md §1 已锁定 setuptools<81）。
# 这是固定依赖的良性提示，会在命令行输出里盖住真正的信息，故仅在 CLI 层静默。
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

DEFAULT_STORE = "results"


def _load_scenario(path: str) -> Scenario:
    return Scenario(**json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------
# list-algorithms
# ----------------------------------------------------------------------


def _cmd_list_algorithms(_args: argparse.Namespace) -> int:
    print(json.dumps(list_algorithms(), ensure_ascii=False, indent=2))
    return 0


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    scenarios = [_load_scenario(p) for p in args.scenario]
    store = ResultStore(args.store) if args.store else None
    batch = run_batch(scenarios, args.algorithm, seeds=args.seeds or None, store=store)

    payload = batch.model_dump(mode="json")
    if args.out and args.out != "-":
        out_dir = pathlib.Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{batch.batch_id}.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"写入 {target}", file=sys.stderr)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if store is not None:
        print(f"已存入 {store.root}（{batch.batch_id}）", file=sys.stderr)
    if not args.quiet:
        _print_summary(batch)
    return 0


def _print_summary(batch) -> None:
    """打印人类可读的对比表到 stderr，不污染 stdout 的 JSON。"""
    header = (
        f"{'scenario':22} {'algo':11} {'seed':>5} {'satis':>7} {'complete':>9} "
        f"{'kWh':>8} {'cost':>9} {'peak':>7} {'drop':>5}"
    )
    print(header, file=sys.stderr)
    for run in batch.runs:
        m = run.metrics
        print(
            f"{run.scenario_id:22} {run.algorithm:11} {run.seed:5d} "
            f"{m.demand_satisfaction_rate:7.3f} {m.session_completion_rate:9.3f} "
            f"{m.energy_delivered_kwh:8.1f} {m.energy_cost_cny:9.2f} "
            f"{m.peak_power_kw:7.2f} {m.sessions_dropped:5d}",
            file=sys.stderr,
        )


# ----------------------------------------------------------------------
# results / describe / batches
# ----------------------------------------------------------------------


def _cmd_results(args: argparse.Namespace) -> int:
    store = ResultStore(args.store)
    if args.json:
        runs = store.list_runs(
            batch_id=args.batch, scenario_id=args.scenario, algorithm=args.algorithm
        )
        print(json.dumps([r.__dict__ for r in runs], ensure_ascii=False, indent=2))
        return 0
    print(store.format_table(
        batch_id=args.batch, scenario_id=args.scenario, algorithm=args.algorithm
    ))
    counts = store.counts()
    print(f"\n共 {counts['runs']} 条运行记录，{counts['batches']} 个批次（{store.root}）")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    store = ResultStore(args.store)
    print(store.describe_run(args.run_id))
    return 0


def _cmd_batches(args: argparse.Namespace) -> int:
    store = ResultStore(args.store)
    batches = store.list_batches()
    if not batches:
        print(f"（{store.root} 下没有批次记录）")
        return 0
    print(f"{'batch_id':22} {'created_at':26} {'runs':>5}  scenarios")
    print("-" * 88)
    for b in batches:
        print(
            f"{b['batch_id']:22} {b['created_at'][:19]:26} {b['n_runs']:5d}  "
            f"{', '.join(b['scenario_ids'])}"
        )
    return 0


# ----------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------


def _cmd_sweep(args: argparse.Namespace) -> int:
    import json as _json

    from .algorithms import CORE_ALGORITHMS
    from .sweep import (
        format_sweep_detail,
        format_winner_grid,
        rerank_sweep,
        run_sweep,
    )

    base = _load_scenario(args.scenario[0])
    algorithms = args.algorithm or list(CORE_ALGORITHMS)
    seeds = args.seeds or [base.seed]
    store = ResultStore(args.store) if args.store else None

    sweep = run_sweep(
        base, args.x, args.y, algorithms, seeds,
        primary_metric=args.metric, store=store,
    )

    payload: dict = {"sweeps": {sweep.primary_metric: sweep.model_dump(mode="json")}}

    # 换目标重排名不需要重跑仿真 —— 实验数据与用哪个指标排名无关
    for extra in args.compare or []:
        payload["sweeps"][extra] = rerank_sweep(sweep, extra).model_dump(mode="json")

    if args.out:
        out_dir = pathlib.Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{sweep.sweep_id}.json"
        target.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {target}", file=sys.stderr)
    elif not args.plot:
        print(_json.dumps(payload, ensure_ascii=False, indent=2))

    if store is not None:
        print(f"已存入 {store.root}（{len(sweep.run_ids)} 条运行记录）", file=sys.stderr)

    if not args.quiet:
        print(file=sys.stderr)
        print(format_winner_grid(sweep), file=sys.stderr)
    # --detail 独立于 --quiet：只想要明细时不该被迫同时看区域图
    if args.detail:
        print(file=sys.stderr)
        print(format_sweep_detail(sweep), file=sys.stderr)

    if args.plot:
        from .viz import render_objective_comparison, render_sweep

        for path in render_sweep(sweep, args.plot):
            print(f"写出 {path}", file=sys.stderr)
        if args.compare:
            others = [rerank_sweep(sweep, m) for m in args.compare]
            path = render_objective_comparison([sweep, *others], args.plot)
            print(f"写出 {path}", file=sys.stderr)
    return 0


# ----------------------------------------------------------------------
# curves
# ----------------------------------------------------------------------


def _cmd_curves(args: argparse.Namespace) -> int:
    from .algorithms import CORE_ALGORITHMS
    from .timeseries import simulate_curves, summarize
    from .viz import render_curves

    scenario = _load_scenario(args.scenario[0])
    algorithms = args.algorithm or list(CORE_ALGORITHMS)
    seed = args.seeds[0] if args.seeds else scenario.seed

    curves = []
    for name in algorithms:
        try:
            curves.append(simulate_curves(scenario, name, seed=seed))
        except Exception as exc:  # noqa: BLE001 - 命令行需要把失败报出来而不是整体崩掉
            print(f"{name} 无法提取曲线：{exc}", file=sys.stderr)
    if not curves:
        print("没有任何算法产出曲线", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"{'算法':8} {'峰值kW':>8} {'均值kW':>8} {'利用率':>8} {'电量kWh':>9} {'电费CNY':>9}", file=sys.stderr)
        for c in curves:
            s = summarize(c)
            print(
                f"{c.algorithm:8} {s['peak_kw']:8.2f} {s['mean_kw']:8.2f} "
                f"{s['utilization']:8.3f} {s['energy_kwh']:9.2f} {s['cost_cny']:9.2f}",
                file=sys.stderr,
            )

    wrote_something = False
    if args.plot:
        for path in render_curves(curves, args.plot):
            print(f"写出 {path}", file=sys.stderr)
        wrote_something = True

    if args.out:
        out_dir = pathlib.Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            c.algorithm: {
                "scenario_id": c.scenario_id,
                "seed": c.seed,
                "hours": c.hours,
                "aggregate_kw": c.aggregate_kw,
                "price_cny_per_kwh": c.price_cny_per_kwh,
                "segments": c.segments,
                "limit_kw": c.limit_kw,
                "cumulative_cost_cny": c.cumulative_cost_cny,
                "summary": summarize(c),
            }
            for c in curves
        }
        target = out_dir / f"curves--{scenario.scenario_id}--seed{seed}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {target}", file=sys.stderr)
        wrote_something = True

    if not wrote_something:
        print(json.dumps(
            {c.algorithm: summarize(c) for c in curves}, ensure_ascii=False, indent=2
        ))
    return 0


# ----------------------------------------------------------------------
# agent
# ----------------------------------------------------------------------


def _auto_holdout(base):
    """从基准场景派生保留场景。

    保留场景必须与实验集**不同**，否则复验没有意义（方案 §5 原则 6）。
    这里用「换种子 + 轻微扰动参数」派生：既不重复实验集，也仍在同一工况族内，
    因此结论仍是「该工况族内泛化」而非跨工况外推。
    """
    return [
        base.with_updates(scenario_id=f"{base.scenario_id}_holdout_seed{base.seed + 101}",
                          seed=base.seed + 101),
        base.with_updates(scenario_id=f"{base.scenario_id}_holdout_shifted",
                          seed=base.seed + 202,
                          load_intensity=round(base.load_intensity * 0.95, 3)),
    ]


def _cmd_agent(args: argparse.Namespace) -> int:
    from .agent_loop import HeuristicProposer, LLMProposer, run_agent_loop
    from .schemas import BudgetSpec

    base = _load_scenario(args.scenario[0])
    store = ResultStore(args.store) if args.store else None

    # 实验集：基准场景 + 若干参数扰动，用于提出与筛选假设
    train = [base]
    if args.train_variants:
        train.append(base.with_updates(
            scenario_id=f"{base.scenario_id}_tight",
            deadline_tightness=round(min(base.deadline_tightness * 1.3, 1.0), 3),
            load_intensity=round(base.load_intensity * 0.9, 3),
        ))
        train.append(base.with_updates(
            scenario_id=f"{base.scenario_id}_loose",
            deadline_tightness=round(base.deadline_tightness * 0.8, 3),
            load_intensity=round(base.load_intensity * 0.8, 3),
        ))

    holdout = [_load_scenario(p) for p in args.holdout] if args.holdout else _auto_holdout(base)

    budget = BudgetSpec(
        max_rounds=args.rounds,
        max_simulations=args.max_simulations,
        max_seconds=args.max_seconds,
        patience=args.patience,
    )
    proposer = HeuristicProposer()
    if args.llm_command:
        proposer = LLMProposer(_shell_llm(args.llm_command))
        print(f"使用外部模型提议者：{args.llm_command}", file=sys.stderr)

    report = run_agent_loop(
        train, holdout, args.algorithm or list(CORE_ALGORITHMS), proposer,
        budget=budget, seeds=args.seeds or [base.seed],
        store=store, target=args.target, batch_size=args.batch_size,
    )

    if args.out:
        out_dir = pathlib.Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        target_path = out_dir / f"{report.report_id}.json"
        target_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"写入 {target_path}", file=sys.stderr)
    elif not args.plot:
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))

    if not args.quiet:
        print(file=sys.stderr)
        print(f"停止原因 {report.stop_reason} · 仿真 {report.simulations_used} 次 · "
              f"耗时 {report.elapsed_seconds:.1f}s", file=sys.stderr)
        print(f"\n{'策略':34} {'完成率':>8} {'满足率':>8} {'波动':>7} {'越限':>5}", file=sys.stderr)
        for o in report.baseline_outcomes:
            print(f"{o.spec.name + '（基准）':34} {o.session_completion_rate:8.4f} "
                  f"{o.demand_satisfaction_rate:8.4f} {o.spread:7.4f} {o.constraint_violations:5d}",
                  file=sys.stderr)
        for r in report.rounds:
            for o in r.outcomes:
                print(f"{o.spec.name:34} {o.session_completion_rate:8.4f} "
                      f"{o.demand_satisfaction_rate:8.4f} {o.spread:7.4f} "
                      f"{o.constraint_violations:5d}", file=sys.stderr)
        print(file=sys.stderr)
        print(report.conclusion, file=sys.stderr)
        print(file=sys.stderr)
        for c in report.caveats:
            print(f"  · {c}", file=sys.stderr)

    if args.plot:
        from .viz import render_agent_report

        print(f"写出 {render_agent_report(report, args.plot)}", file=sys.stderr)
    return 0


def _shell_llm(command: str):
    """把一个 shell 命令包装成「提示词进、文本出」的调用。

    刻意不绑定任何厂商 SDK：谁能把提示词喂给模型并打印回复，谁就能当提议者。
    命令通过 stdin 接提示词、stdout 出文本。
    """
    import shlex
    import subprocess

    argv = shlex.split(command)

    def complete(prompt: str) -> str:
        proc = subprocess.run(
            argv, input=prompt, capture_output=True, text=True, timeout=120
        )
        if proc.returncode != 0:
            raise RuntimeError(f"模型命令退出码 {proc.returncode}：{proc.stderr[:200]}")
        return proc.stdout

    return complete


# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chargebench", description="ChargeBench AI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-algorithms", help="列出可用算法及其可见信息").set_defaults(
        func=_cmd_list_algorithms
    )

    run = sub.add_parser("run", help="运行实验并输出协议 JSON")
    run.add_argument(
        "--scenario", action="append", required=True, help="场景 JSON 路径，可重复指定"
    )
    run.add_argument(
        "--algorithm", nargs="+", required=True, help="算法名，可多个（如 FCFS EDF LLF）"
    )
    run.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="覆盖场景种子的种子列表"
    )
    run.add_argument("--out", default=None, help="输出目录；省略则打印到 stdout")
    run.add_argument(
        "--store", default=None, help=f"结果库目录，给出则落盘并复用已算结果（如 {DEFAULT_STORE}）"
    )
    run.add_argument("--quiet", action="store_true", help="不打印汇总表")
    run.set_defaults(func=_cmd_run)

    results = sub.add_parser("results", help="查询结果库")
    results.add_argument("--store", default=DEFAULT_STORE)
    results.add_argument("--batch", default=None, help="限定批次")
    results.add_argument("--scenario", default=None, help="限定场景")
    results.add_argument("--algorithm", default=None, help="限定算法")
    results.add_argument("--json", action="store_true", help="输出 JSON 而非表格")
    results.set_defaults(func=_cmd_results)

    describe = sub.add_parser("describe", help="说明某个结果是用什么配置跑出来的")
    describe.add_argument("run_id")
    describe.add_argument("--store", default=DEFAULT_STORE)
    describe.set_defaults(func=_cmd_describe)

    batches = sub.add_parser("batches", help="列出已记录的批次")
    batches.add_argument("--store", default=DEFAULT_STORE)
    batches.set_defaults(func=_cmd_batches)

    sweep = sub.add_parser("sweep", help="两变量参数扫描，找算法适用边界")
    sweep.add_argument("--scenario", action="append", required=True, help="基准场景 JSON")
    sweep.add_argument(
        "--algorithm", nargs="+", default=None, help="算法名，默认 FCFS EDF LLF"
    )
    sweep.add_argument("--x", nargs="+", type=float, required=True,
                       help=f"X 轴（{X_FIELD}）取值列表")
    sweep.add_argument("--y", nargs="+", type=float, required=True,
                       help=f"Y 轴（{Y_FIELD}）取值列表")
    sweep.add_argument("--seeds", nargs="+", type=int, default=None,
                       help="种子列表；省略则用场景自带的那个")
    sweep.add_argument("--metric", default="session_completion_rate",
                       help="排名依据的指标，见 docs/PROTOCOL.md §4")
    sweep.add_argument("--compare", nargs="+", default=None,
                       help="额外目标函数，复用同一批实验重新排名（不重跑仿真）")
    sweep.add_argument("--plot", default=None, help="图表输出目录，如 reports/")
    sweep.add_argument("--out", default=None, help="扫描结果 JSON 输出目录")
    sweep.add_argument("--store", default=None, help="结果库目录，强烈建议提供以复用缓存")
    sweep.add_argument("--detail", action="store_true", help="打印逐格点明细")
    sweep.add_argument("--quiet", action="store_true")
    sweep.set_defaults(func=_cmd_sweep)

    curves = sub.add_parser("curves", help="导出某次运行的负荷曲线（过程回放）")
    curves.add_argument("--scenario", action="append", required=True, help="场景 JSON")
    curves.add_argument("--algorithm", nargs="+", default=None, help="算法名，默认 FCFS EDF LLF")
    curves.add_argument("--seeds", nargs="+", type=int, default=None, help="取第一个种子")
    curves.add_argument("--plot", default=None, help="图表输出目录，如 reports/")
    curves.add_argument("--out", default=None, help="曲线 JSON 输出目录")
    curves.add_argument("--quiet", action="store_true", help="不打印汇总表")
    curves.set_defaults(func=_cmd_curves)

    agent = sub.add_parser("agent", help="AI 实验闭环：提假设 → 跑仿真 → 复验")
    agent.add_argument("--scenario", action="append", required=True, help="基准场景 JSON")
    agent.add_argument("--algorithm", nargs="+", default=None, help="基准算法，默认 FCFS EDF LLF")
    agent.add_argument("--seeds", nargs="+", type=int, default=None, help="实验种子列表")
    agent.add_argument("--rounds", type=int, default=3, help="最大批次数")
    agent.add_argument("--max-simulations", type=int, default=200, help="仿真调用次数上限")
    agent.add_argument("--max-seconds", type=float, default=120.0, help="墙钟时间上限")
    agent.add_argument("--patience", type=int, default=2, help="连续多少轮无实质改善即停")
    agent.add_argument("--batch-size", type=int, default=3, help="每轮评测几个候选")
    agent.add_argument("--target", type=float, default=None, help="主目标达到即停")
    agent.add_argument("--train-variants", action="store_true",
                       help="额外派生两个参数扰动的实验场景（强烈建议开启）")
    agent.add_argument("--holdout", action="append", default=None,
                       help="保留场景 JSON（可重复）；省略则自动派生")
    agent.add_argument("--llm-command", default=None,
                       help="外部模型命令（stdin 收提示词、stdout 出 JSON），省略则用内置启发式")
    agent.add_argument("--plot", default=None, help="图表输出目录")
    agent.add_argument("--out", default=None, help="报告 JSON 输出目录")
    agent.add_argument("--store", default=None, help="结果库目录，用于复用缓存")
    agent.add_argument("--quiet", action="store_true")
    agent.set_defaults(func=_cmd_agent)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
