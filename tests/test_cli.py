"""CLI 与协议往返测试。

UI / Agent 消费的是落盘的 JSON，而不是 Python 对象。因此必须验证：
写出的 JSON 能被 schema 重新解析回来，且字段不丢、口径不变。
"""

import json
import subprocess
import sys

import pytest

from chargebench.algorithms import CORE_ALGORITHMS
from chargebench.schemas import BatchResult, Scenario

from .conftest import PROJECT_ROOT


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "chargebench.cli", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def test_list_algorithms_emits_valid_json():
    out = _run_cli("list-algorithms")
    payload = json.loads(out.stdout)
    names = [entry["name"] for entry in payload]
    for core in CORE_ALGORITHMS:
        assert core in names
    for entry in payload:
        assert entry["summary"], "每个算法都要有可读说明"
        assert entry["visible_info"], "必须声明算法能看到什么信息，否则无法核对公平性"


def test_run_writes_roundtrippable_batch_result(tmp_path):
    _run_cli(
        "run",
        "--scenario",
        "configs/demo_scenario.json",
        "--algorithm",
        *CORE_ALGORITHMS,
        "--out",
        str(tmp_path),
    )
    files = list(tmp_path.glob("batch--*.json"))
    assert len(files) == 1

    payload = json.loads(files[0].read_text(encoding="utf-8"))
    # 关键断言：落盘的 JSON 必须能原样解析回协议模型，字段不丢
    batch = BatchResult(**payload)

    assert len(batch.runs) == len(CORE_ALGORITHMS)
    assert batch.schema_version == "1.0"
    assert {r.algorithm for r in batch.runs} == set(CORE_ALGORITHMS)
    assert all(r.traceability.acnportal_version for r in batch.runs)


def test_run_stdout_is_pure_json():
    """stdout 必须是可直接 json.loads 的纯 JSON，汇总表走 stderr。"""
    out = _run_cli(
        "run", "--scenario", "configs/demo_scenario.json", "--algorithm", "EDF"
    )
    batch = BatchResult(**json.loads(out.stdout))
    assert len(batch.runs) == 1
    assert "scenario" in out.stderr, "人类可读的汇总表应输出到 stderr"


def test_run_with_multiple_seeds(tmp_path):
    _run_cli(
        "run",
        "--scenario",
        "configs/demo_scenario.json",
        "--algorithm",
        "EDF",
        "--seeds",
        "42",
        "43",
        "--out",
        str(tmp_path),
    )
    batch = BatchResult(**json.loads(next(tmp_path.glob("batch--*.json")).read_text()))
    assert batch.seeds == [42, 43]
    assert len(batch.runs) == 2
    rates = {r.metrics.demand_satisfaction_rate for r in batch.runs}
    assert len(rates) == 2, "不同种子应给出不同结果"


def test_run_rejects_invalid_scenario(tmp_path):
    bad = tmp_path / "bad.json"
    payload = json.loads((PROJECT_ROOT / "configs/demo_scenario.json").read_text())
    payload["network_limit_kw"] = 1.0  # 低于单桩功率，场景退化
    bad.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "chargebench.cli", "run", "--scenario", str(bad),
         "--algorithm", "EDF"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "network_limit_kw" in result.stderr


def test_example_artifacts_match_schema():
    """仓库里 examples/ 下的样例必须始终是合法协议文档。"""
    examples = sorted((PROJECT_ROOT / "examples").glob("batch--*.json"))
    if not examples:
        pytest.skip("examples/ 尚未生成，运行 CLI 后会补上")
    for path in examples:
        BatchResult(**json.loads(path.read_text(encoding="utf-8")))


# ----------------------------------------------------------------------
# 结果库子命令
# ----------------------------------------------------------------------


def test_run_with_store_then_query(tmp_path):
    store_dir = tmp_path / "results"
    _run_cli(
        "run", "--scenario", "configs/demo_scenario.json",
        "--algorithm", *CORE_ALGORITHMS, "--store", str(store_dir),
    )
    assert (store_dir / "results.db").exists()
    assert len(list((store_dir / "runs").glob("*.json"))) == len(CORE_ALGORITHMS)

    out = _run_cli("results", "--store", str(store_dir))
    for algorithm in CORE_ALGORITHMS:
        assert algorithm in out.stdout
    assert "3 条运行记录" in out.stdout


def test_results_json_mode(tmp_path):
    store_dir = tmp_path / "results"
    _run_cli("run", "--scenario", "configs/demo_scenario.json",
             "--algorithm", "EDF", "--store", str(store_dir))
    out = _run_cli("results", "--store", str(store_dir), "--json")
    rows = json.loads(out.stdout)
    assert len(rows) == 1
    assert rows[0]["algorithm"] == "EDF"
    assert rows[0]["demand_satisfaction_rate"] > 0


def test_results_filters(tmp_path):
    store_dir = tmp_path / "results"
    _run_cli("run", "--scenario", "configs/demo_scenario.json",
             "--algorithm", *CORE_ALGORITHMS, "--store", str(store_dir))
    out = _run_cli("results", "--store", str(store_dir), "--algorithm", "EDF", "--json")
    assert len(json.loads(out.stdout)) == 1

    out = _run_cli("results", "--store", str(store_dir), "--algorithm", "NOPE", "--json")
    assert json.loads(out.stdout) == []


def test_store_reuses_existing_results(tmp_path):
    """重复运行同一配置不得在结果库里制造新记录。"""
    store_dir = tmp_path / "results"
    args = ("run", "--scenario", "configs/demo_scenario.json",
            "--algorithm", "EDF", "--store", str(store_dir))
    _run_cli(*args)
    _run_cli(*args)
    out = _run_cli("results", "--store", str(store_dir), "--json")
    assert len(json.loads(out.stdout)) == 1


def test_describe_reports_config(tmp_path):
    """阶段 3 判据：一句话说清某个结果当时是用什么配置跑出来的。"""
    store_dir = tmp_path / "results"
    _run_cli("run", "--scenario", "configs/demo_scenario.json",
             "--algorithm", "EDF", "--store", str(store_dir))

    run_id = json.loads(
        _run_cli("results", "--store", str(store_dir), "--json").stdout
    )[0]["run_id"]

    out = _run_cli("describe", run_id, "--store", str(store_dir))
    assert run_id in out.stdout
    assert "campus_baseline_v1" in out.stdout
    assert "acnportal" in out.stdout
    assert "2026-01-05" in out.stdout


def test_batches_lists_recorded_batches(tmp_path):
    store_dir = tmp_path / "results"
    _run_cli("run", "--scenario", "configs/demo_scenario.json",
             "--algorithm", "EDF", "--store", str(store_dir))
    out = _run_cli("batches", "--store", str(store_dir))
    assert "batch--" in out.stdout
    assert "campus_baseline_v1" in out.stdout


def test_batches_on_empty_store_is_safe(tmp_path):
    out = _run_cli("batches", "--store", str(tmp_path / "nothing"))
    assert "没有批次记录" in out.stdout


# ----------------------------------------------------------------------
# sweep 子命令
# ----------------------------------------------------------------------

SWEEP_ARGS = (
    "--scenario", "configs/demo_scenario.json",
    "--x", "0.5", "1.3",
    "--y", "0.4", "0.85",
    "--seeds", "42", "43",
)


def test_sweep_prints_grid_and_writes_json(tmp_path):
    out = _run_cli("sweep", *SWEEP_ARGS, "--out", str(tmp_path), "--quiet")
    # 汇总表与 JSON 走不同流：stdout 只保留汇总，JSON 落盘
    payload = json.loads(next(tmp_path.glob("sweep--*.json")).read_text(encoding="utf-8"))
    sweep = payload["sweeps"]["session_completion_rate"]
    assert len(sweep["cells"]) == 4
    assert sweep["score_rule"], "评分规则必须随结果持久化，否则事后无法解释排名"
    assert sweep["run_ids"], "必须能追溯到具体运行"


def test_sweep_prints_ascii_grid():
    out = _run_cli("sweep", *SWEEP_ARGS)
    assert "算法适用区域图" in out.stderr
    assert "load_intensity" in out.stderr


def test_sweep_detail_flag():
    out = _run_cli("sweep", *SWEEP_ARGS, "--detail", "--quiet")
    assert "取消资格" in out.stderr or "胜者" in out.stderr


def test_sweep_compare_reuses_experiments(tmp_path):
    """--compare 换目标重排名，不应重跑仿真：两条 sweep 的 run_ids 必须完全一致。"""
    _run_cli(
        "sweep", *SWEEP_ARGS,
        "--metric", "session_completion_rate",
        "--compare", "peak_power_kw",
        "--out", str(tmp_path), "--quiet",
    )
    payload = json.loads(next(tmp_path.glob("sweep--*.json")).read_text(encoding="utf-8"))
    a = payload["sweeps"]["session_completion_rate"]
    b = payload["sweeps"]["peak_power_kw"]
    assert a["run_ids"] == b["run_ids"], "换目标重排名不得产生新的实验"
    assert a["sweep_id"] != b["sweep_id"]
    assert b["metric_direction"] == "min"
    assert a["primary_metric"] != b["primary_metric"]


def test_sweep_plots_and_stores(tmp_path):
    store_dir = tmp_path / "results"
    plot_dir = tmp_path / "reports"
    _run_cli(
        "sweep", *SWEEP_ARGS,
        "--store", str(store_dir),
        "--plot", str(plot_dir),
        "--compare", "peak_power_kw",
        "--quiet",
    )
    assert (store_dir / "results.db").exists()
    # 2x2 网格 × 3 算法 × 2 种子 = 24 条
    assert len(list((store_dir / "runs").glob("*.json"))) == 24
    assert len(list(plot_dir.glob("*.png"))) == 3, "应产出区域图、指标小图、目标对比图"


def test_sweep_store_reuses_cached_runs(tmp_path):
    """缓存复用必须真的生效：命中缓存返回的是原始结果，executed_at 不会变。

    只比对文件条数无法区分「复用」与「重算后覆盖同名文件」，故用执行时刻来判定。
    """
    store_dir = tmp_path / "results"
    args = ("sweep", *SWEEP_ARGS, "--store", str(store_dir), "--quiet")
    _run_cli(*args)
    files = sorted((store_dir / "runs").glob("*.json"))
    assert len(files) == 24
    first_times = {
        f.name: json.loads(f.read_text(encoding="utf-8"))["traceability"]["executed_at"]
        for f in files
    }

    _run_cli(*args)
    after = sorted((store_dir / "runs").glob("*.json"))
    assert len(after) == 24, "重复扫描不得制造新记录"

    for f in after:
        now = json.loads(f.read_text(encoding="utf-8"))["traceability"]["executed_at"]
        assert now == first_times[f.name], (
            f"{f.name} 的执行时刻变了，说明第二次扫描是重算而非复用缓存"
        )


def test_sweep_rejects_unknown_metric():
    result = subprocess.run(
        [sys.executable, "-m", "chargebench.cli", "sweep", *SWEEP_ARGS,
         "--metric", "not_a_metric"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "primary_metric" in result.stderr


# ----------------------------------------------------------------------
# agent 子命令
# ----------------------------------------------------------------------

AGENT_ARGS = (
    "--scenario", "configs/demo_scenario.json",
    "--seeds", "42",
    "--rounds", "1",
    "--max-simulations", "60",
    "--batch-size", "2",
    "--train-variants",
)


def test_agent_writes_report(tmp_path):
    _run_cli("agent", *AGENT_ARGS, "--out", str(tmp_path), "--quiet")
    payload = json.loads(next(tmp_path.glob("agent--*.json")).read_text(encoding="utf-8"))
    assert payload["rounds"], "应至少产生一轮候选评测"
    assert payload["baseline_outcomes"]
    assert payload["caveats"]
    assert payload["all_run_ids"]


def test_agent_prints_conclusion_and_caveats():
    out = _run_cli("agent", *AGENT_ARGS)
    assert "在已测试范围与预算内" in out.stderr
    assert "不是全局最优" in out.stderr, "免责声明必须打印出来"


def test_agent_never_claims_global_optimum():
    """措辞纪律要对**命令行输出**同样生效，而不只是报告对象。"""
    from chargebench.agent_loop import claims_optimality

    out = _run_cli("agent", *AGENT_ARGS)
    assert not claims_optimality(out.stderr)


def test_agent_respects_simulation_budget():
    out = _run_cli(
        "agent", "--scenario", "configs/demo_scenario.json",
        "--seeds", "42", "--rounds", "5", "--max-simulations", "20",
        "--train-variants", "--quiet",
    )
    # stdout 是 JSON（无 --out/--plot 时），解析出来核对账目
    report = json.loads(out.stdout)
    assert report["simulations_used"] <= 20


def test_agent_plots(tmp_path):
    _run_cli("agent", *AGENT_ARGS, "--plot", str(tmp_path), "--quiet")
    pngs = list(tmp_path.glob("agent--*.png"))
    assert len(pngs) == 1
    assert pngs[0].stat().st_size > 10_000


def test_agent_with_external_llm_command(tmp_path):
    """外部模型路径要真的跑得通 —— 用一条 shell 命令冒充模型来解释 JSON。

    这条测试同时证明：LLM 只能产出结构化规格，无法夹带代码。
    """
    # 用脚本文件而不是 python3 -c：JSON 里的引号会和外层 shell 引号打架，
    # 那样测的就是引号转义而不是被测逻辑了
    script = tmp_path / "fake_model.py"
    script.write_text(
        "import json, sys\n"
        "sys.stdin.read()\n"
        "print(json.dumps([\n"
        '    {"name": "llm-pick", "primary": "laxity", "completion_first": True,\n'
        '     "rationale": "来自外部模型的建议"},\n'
        '    {"name": "evil", "primary": "laxity", "__import__": "os"},\n'
        "]))\n",
        encoding="utf-8",
    )
    _run_cli(
        "agent", *AGENT_ARGS,
        "--llm-command", f"{sys.executable} {script}",
        "--out", str(tmp_path), "--quiet",
    )
    report = json.loads(next(tmp_path.glob("agent--*.json")).read_text(encoding="utf-8"))
    names = [o["spec"]["name"] for r in report["rounds"] for o in r["outcomes"]]
    assert "llm-pick" in names, "外部模型给出的合法策略应当被评测"
    assert "evil" not in names, "夹带额外字段的条目必须被拒绝"


def test_agent_llm_command_failure_falls_back(tmp_path):
    """模型命令失败时必须回退到启发式，而不是让整个闭环失败。"""
    _run_cli(
        "agent", *AGENT_ARGS,
        "--llm-command", "python3 -c 'import sys; sys.exit(3)'",
        "--out", str(tmp_path), "--quiet",
    )
    report = json.loads(next(tmp_path.glob("agent--*.json")).read_text(encoding="utf-8"))
    assert report["rounds"], "模型失效时应回退到启发式继续探索"
