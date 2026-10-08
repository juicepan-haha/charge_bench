"""MCP Server 测试。

工具层与传输层解耦，因此这里直接调用工具函数，不需要起 MCP 会话。
另有一组测试通过 ``server.call_tool`` 走真实的 MCP 调用路径，确认注册没有漏参。

**安全边界**（方案 §7 的工程约束）逐条有测试守护：
参数白名单、禁止任意代码、批次上限、超时、错误码标准化、结果库路径不由客户端控制。
"""

import asyncio
import json
import sys

import pytest

from chargebench.mcp_server import (
    ChargeBenchTools,
    ErrorCode,
    ServerLimits,
    build_server,
)
from chargebench.schemas import Scenario

from .conftest import CONFIG_DIR, PROJECT_ROOT, load_scenario


@pytest.fixture
def tools(tmp_path) -> ChargeBenchTools:
    # 显式给一个小上限，便于验证批次规模闸门。（默认上限是 400）
    return ChargeBenchTools(tmp_path / "results", CONFIG_DIR,
                            ServerLimits(max_runs_per_call=60))


@pytest.fixture
def server(tmp_path):
    return build_server(tmp_path / "results", CONFIG_DIR, ServerLimits(max_runs_per_call=60))


def call(server, name: str, args: dict) -> dict:
    """走真实 MCP 调用路径并解出信封。"""
    result = asyncio.run(server.call_tool(name, args))
    return json.loads(result.content[0].text)


# ----------------------------------------------------------------------
# 注册
# ----------------------------------------------------------------------


def test_all_tools_registered(server):
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {
        "list_algorithms", "run_batch", "get_results", "get_run",
        "compare_experiments", "parameter_sweep", "list_batches",
    }


def test_server_instructions_state_objective_dependence(server):
    """平台的核心事实必须写进服务端说明，让接入的 Agent 一开始就知道。"""
    text = server.instructions or ""
    assert "目标" in text
    assert "仿真" in text, "必须说明指标来自仿真而非模型推断"


def test_tools_layer_is_decoupled_from_transport(tools):
    """工具实现是普通同步函数，可直接调用 —— 这是可测试性的基础。"""
    payload = tools.list_algorithms()
    assert payload["ok"] is True


def test_no_client_supplied_path_parameter(server):
    """结果库路径由服务端固定。工具的入参里不得出现路径类参数，
    否则客户端就能借工具读写任意位置。"""
    for tool in asyncio.run(server.list_tools()):
        # mcp 2.x 用 snake_case 的 input_schema；v1 的 inputSchema 已不存在
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        props = set(schema.get("properties", {}))
        assert not (props & {"store", "store_dir", "path", "out", "output_dir"}), (
            f"工具 {tool.name} 暴露了路径类参数：{props}"
        )


# ----------------------------------------------------------------------
# list_algorithms
# ----------------------------------------------------------------------


def test_list_algorithms_payload(tools):
    data = tools.list_algorithms()["data"]
    names = {a["name"] for a in data["algorithms"]}
    assert {"FCFS", "EDF", "LLF"} <= names
    assert data["candidate_vocabulary"]["sort_keys"]
    assert data["limits"]["max_runs_per_call"] > 0
    assert data["metrics"]["directions"]


def test_list_algorithms_advertises_visible_info(tools):
    """公平性核对信息要能通过 MCP 拿到，外部 Agent 才能自行判断可比性。"""
    for entry in tools.list_algorithms()["data"]["algorithms"]:
        assert entry["visible_info"], f"{entry['name']} 未声明可见信息"


# ----------------------------------------------------------------------
# run_batch
# ----------------------------------------------------------------------


def test_run_batch_happy_path(tools):
    payload = tools.run_batch(
        {"config_name": "demo_scenario", "algorithms": ["FCFS", "EDF", "LLF"], "seeds": [42]}
    )
    assert payload["ok"] is True
    data = payload["data"]
    assert data["completed_runs"] == 3
    assert data["batch_id"].startswith("batch--")
    assert data["truncated"] is False
    for run in data["runs"]:
        assert run["run_id"]
        assert 0.0 <= run["metrics"]["demand_satisfaction_rate"] <= 1.0


def test_run_batch_with_inline_scenario(tools):
    payload = tools.run_batch(
        {"scenario": load_scenario().model_dump(mode="json"), "algorithms": ["EDF"]}
    )
    assert payload["ok"] is True
    assert payload["data"]["completed_runs"] == 1


def test_run_batch_persists_results(tools):
    batch_id = tools.run_batch({"config_name": "demo_scenario", "algorithms": ["EDF"]})["data"][
        "batch_id"
    ]
    assert tools.get_results(batch_id)["ok"] is True
    assert tools.store.counts()["runs"] >= 1


def test_run_batch_defaults_to_core_algorithms(tools):
    data = tools.run_batch({"config_name": "demo_scenario"})["data"]
    assert set(data["algorithms"]) == {"FCFS", "EDF", "LLF"}


def test_parallel_scenarios_derives_variants(tools):
    data = tools.run_batch(
        {"config_name": "demo_scenario", "algorithms": ["EDF"], "parallel_scenarios": 3}
    )["data"]
    assert len(data["scenario_ids"]) == 3
    assert data["completed_runs"] == 3


# ----------------------------------------------------------------------
# 错误码标准化
# ----------------------------------------------------------------------


def test_unknown_algorithm_rejected(tools):
    payload = tools.run_batch({"config_name": "demo_scenario", "algorithms": ["os.system"]})
    assert payload["ok"] is False
    assert payload["error"]["code"] == ErrorCode.unknown_algorithm.value
    assert "available" in payload["error"]


def test_unknown_scenario_rejected(tools):
    payload = tools.run_batch({"config_name": "no_such_scenario"})
    assert payload["error"]["code"] == ErrorCode.unknown_scenario.value
    assert payload["error"]["available"], "应告知可用的场景名"


def test_path_traversal_rejected(tools):
    """客户端不能借 config_name 读任意文件。"""
    for attack in ["../../etc/passwd", "/etc/passwd", "..%2Fsecret", ".hidden", "a/b"]:
        payload = tools.run_batch({"config_name": attack})
        assert payload["ok"] is False, f"{attack} 竟然被接受了"
        assert payload["error"]["code"] in {
            ErrorCode.invalid_argument.value, ErrorCode.unknown_scenario.value
        }


def test_both_sources_rejected(tools):
    payload = tools.run_batch(
        {"scenario": load_scenario().model_dump(mode="json"), "config_name": "demo_scenario"}
    )
    assert payload["error"]["code"] == ErrorCode.invalid_argument.value


def test_neither_source_rejected(tools):
    payload = tools.run_batch({})
    assert payload["error"]["code"] == ErrorCode.invalid_argument.value


def test_scenario_with_extra_field_rejected(tools):
    """场景是**数据**，由 schema 严格校验 —— 多余字段必须报错而不是被忽略。"""
    dirty = {**load_scenario().model_dump(mode="json"), "evil": 1}
    payload = tools.run_batch({"scenario": dirty, "algorithms": ["EDF"]})
    assert payload["error"]["code"] == ErrorCode.invalid_argument.value


def test_scenario_with_invalid_value_rejected(tools):
    dirty = {**load_scenario().model_dump(mode="json"), "deadline_tightness": 5.0}
    payload = tools.run_batch({"scenario": dirty, "algorithms": ["EDF"]})
    assert payload["error"]["code"] == ErrorCode.invalid_argument.value


def test_budget_exceeded_before_running(tools):
    """超预算必须在**开跑之前**拒绝，而不是跑一半再报错。"""
    payload = tools.run_batch(
        {
            "config_name": "demo_scenario",
            "algorithms": ["FCFS", "EDF", "LLF"],
            "seeds": list(range(1, 9)),
            "parallel_scenarios": 5,
        }
    )
    assert payload["error"]["code"] == ErrorCode.budget_exceeded.value
    assert payload["error"]["planned_runs"] > 60
    assert tools.store.counts()["runs"] == 0, "被拒的请求不该产生任何结果"


def test_not_found_error_codes(tools):
    assert tools.get_results("nope")["error"]["code"] == ErrorCode.not_found.value
    assert tools.get_run("nope")["error"]["code"] == ErrorCode.not_found.value
    assert tools.compare_experiments(["nope"])["error"]["code"] == ErrorCode.not_found.value


def test_empty_run_ids_rejected(tools):
    assert tools.compare_experiments([])["error"]["code"] == ErrorCode.invalid_argument.value


def test_error_envelope_shape(tools):
    payload = tools.get_results("nope")
    assert set(payload) == {"ok", "error"}
    assert set(payload["error"]) >= {"code", "message"}


# ----------------------------------------------------------------------
# get_results / get_run
# ----------------------------------------------------------------------


def test_get_results_returns_traceable_paths(tools):
    data = tools.run_batch({"config_name": "demo_scenario", "algorithms": ["EDF"]})["data"]
    out = tools.get_results(data["batch_id"])["data"]
    assert out["runs"][0]["run_id"] == data["runs"][0]["run_id"]
    for path in out["json_paths"]:
        assert path.endswith(".json")


def test_get_run_includes_traceability(tools):
    run_id = tools.run_batch({"config_name": "demo_scenario", "algorithms": ["EDF"]})["data"][
        "runs"
    ][0]["run_id"]
    data = tools.get_run(run_id)["data"]
    assert data["traceability"]["acnportal_version"]
    assert data["traceability"]["schema_version"]


# ----------------------------------------------------------------------
# compare_experiments
# ----------------------------------------------------------------------


def test_compare_reports_per_metric_winners(tools):
    data = tools.run_batch(
        {"config_name": "demo_scenario", "algorithms": ["FCFS", "EDF", "LLF"], "seeds": [42]}
    )["data"]
    out = tools.compare_experiments([r["run_id"] for r in data["runs"]])["data"]

    assert set(out["winner_by_metric"]) == {
        "demand_satisfaction_rate", "session_completion_rate", "mean_delivery_ratio",
        "worst_delivery_ratio", "energy_cost_cny", "peak_power_kw",
    }
    # 不同指标给出不同胜者 —— 这就是平台要说明的事实
    assert out["winner_disagrees"] is True


def test_compare_does_not_rank_constraint_violations(tools):
    """违规不参与排名：平台规则是违规即取消资格，而不是违规少者胜出。"""
    data = tools.run_batch(
        {"config_name": "demo_scenario", "algorithms": ["FCFS", "EDF"], "seeds": [42]}
    )["data"]
    out = tools.compare_experiments([r["run_id"] for r in data["runs"]])["data"]
    assert "constraint_violations" not in out["winner_by_metric"]
    assert "constraint_violations" in out["diagnostics"]
    assert out["disqualified"] == []


def test_compare_warns_on_mixed_scenarios(tools):
    """不同场景的数字放一起比是没有意义的，必须明确警告。"""
    base = load_scenario()
    first = tools.run_batch({"scenario": base.model_dump(mode="json"), "algorithms": ["EDF"]})
    second = tools.run_batch(
        {"scenario": base.with_updates(scenario_id="other").model_dump(mode="json"),
         "algorithms": ["EDF"]}
    )
    ids = [first["data"]["runs"][0]["run_id"], second["data"]["runs"][0]["run_id"]]
    out = tools.compare_experiments(ids)["data"]
    assert "warning" in out
    assert len(out["scenario_hashes"]) == 2


# ----------------------------------------------------------------------
# parameter_sweep
# ----------------------------------------------------------------------


def test_parameter_sweep_returns_grid_and_rule(tools):
    payload = tools.parameter_sweep(
        {
            "config_name": "demo_scenario",
            "x_values": [0.5, 0.9, 1.3],
            "y_values": [0.4, 0.7],
            "seeds": [42],
        }
    )
    assert payload["ok"] is True
    data = payload["data"]
    assert len(data["winner_grid"]) == 2
    assert len(data["winner_grid"][0]) == 3
    assert len(data["port_constrained_grid"]) == 2
    assert data["score_rule"], "评分规则必须随结果返回，事后无法改口径"
    assert len(data["cells"]) == 6


def test_parameter_sweep_rejects_unknown_metric(tools):
    payload = tools.parameter_sweep(
        {"config_name": "demo_scenario", "x_values": [0.5, 0.9], "y_values": [0.4, 0.7],
         "primary_metric": "nonsense"}
    )
    assert payload["error"]["code"] == ErrorCode.invalid_argument.value
    assert "available" in payload["error"]


def test_parameter_sweep_enforces_grid_budget(tools):
    payload = tools.parameter_sweep(
        {
            "config_name": "demo_scenario",
            "x_values": [0.4 + i * 0.1 for i in range(12)],
            "y_values": [0.3 + i * 0.05 for i in range(12)],
            "algorithms": ["FCFS", "EDF", "LLF"],
            "seeds": [1, 2, 3, 4],
        }
    )
    assert payload["error"]["code"] == ErrorCode.budget_exceeded.value


def test_parameter_sweep_rejects_short_grid(tools):
    payload = tools.parameter_sweep(
        {"config_name": "demo_scenario", "x_values": [0.5], "y_values": [0.4, 0.7]}
    )
    assert payload["error"]["code"] == ErrorCode.invalid_argument.value


# ----------------------------------------------------------------------
# 禁止任意代码执行
# ----------------------------------------------------------------------


def test_only_registry_algorithms_accepted(tools):
    """算法必须来自注册表。客户端无法注入自己的可调用对象或代码。"""
    for attack in ["__import__('os').system('id')", "exec", "lambda: None", ""]:
        payload = tools.run_batch({"config_name": "demo_scenario", "algorithms": [attack]})
        assert payload["ok"] is False


def test_candidate_vocabulary_is_closed(tools):
    """候选策略只能来自受控词汇，且词汇表是有限枚举。"""
    vocab = tools.list_algorithms()["data"]["candidate_vocabulary"]
    assert set(vocab["sort_keys"]) == {
        "arrival", "estimated_departure", "laxity", "remaining_demand", "delivery_ratio"
    }
    assert vocab["baselines"], "基准算法要说明其策略含义，供外部 Agent 理解"


def test_no_eval_or_exec_in_module():
    """本模块不得含 eval/exec/compile —— 这是「禁止任意代码执行」的静态检查。"""
    import chargebench.mcp_server as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("eval(", "exec(", "compile(", "__import__("):
        assert forbidden not in source, f"源码里出现了 {forbidden}"


# ----------------------------------------------------------------------
# 超时
# ----------------------------------------------------------------------


def test_deadline_caps_requested_value(tmp_path):
    tools = ChargeBenchTools(tmp_path / "r", CONFIG_DIR,
                             ServerLimits(max_deadline_seconds=5.0))
    assert tools._deadline(None) == 5.0
    assert tools._deadline(1000.0) == 5.0, "客户端不能要求超过服务端上限的时间预算"
    assert tools._deadline(2.0) == 2.0


def test_deadline_produces_truncated_or_error(tmp_path):
    """极短的时间预算下，要么返回带 truncated 的部分结果，要么返回超时错误码。"""
    tools = ChargeBenchTools(tmp_path / "r", CONFIG_DIR)
    payload = tools.run_batch(
        {
            "config_name": "demo_scenario",
            "algorithms": ["FCFS", "EDF", "LLF"],
            "seeds": [1, 2, 3],
            "deadline_seconds": 0.001,
        }
    )
    if payload["ok"]:
        assert payload["data"]["truncated"] is True
        assert payload["data"]["completed_runs"] < 9
    else:
        assert payload["error"]["code"] == ErrorCode.timeout.value


# ----------------------------------------------------------------------
# list_batches
# ----------------------------------------------------------------------


def test_list_batches(tools):
    tools.run_batch({"config_name": "demo_scenario", "algorithms": ["EDF"]})
    data = tools.list_batches()["data"]
    assert data["counts"]["batches"] == 1
    assert data["batches"][0]["batch_id"].startswith("batch--")


def test_list_batches_on_empty_store(tmp_path):
    fresh = ChargeBenchTools(tmp_path / "empty", CONFIG_DIR)
    data = fresh.list_batches()["data"]
    assert data["batches"] == []
    assert data["counts"] == {"batches": 0, "runs": 0}


# ----------------------------------------------------------------------
# 通过真实 MCP 调用路径的冒烟
# ----------------------------------------------------------------------


def test_call_tool_roundtrip(server):
    payload = call(server, "run_batch",
                   {"config_name": "demo_scenario", "algorithms": ["EDF"], "seeds": [42]})
    assert payload["ok"] is True
    batch_id = payload["data"]["batch_id"]
    assert call(server, "get_results", {"batch_id": batch_id})["ok"] is True
    assert call(server, "list_batches", {})["ok"] is True


def test_call_tool_surfaces_error_codes(server):
    payload = call(server, "run_batch", {"config_name": "../../etc/passwd"})
    assert payload["ok"] is False
    assert payload["error"]["code"] in {
        "invalid_argument", "unknown_scenario"
    }


# ----------------------------------------------------------------------
# stdio 传输集成测试（真正的验收：起子进程、按 MCP 协议握手）
# ----------------------------------------------------------------------


def test_server_over_stdio_transport(tmp_path):
    """前面都是进程内调用工具；这一条走**真实传输**：起子进程、握手、列出工具、调用。

    mcp 2.x 全面改为 snake_case：`MCPServer`（v1 是 FastMCP）、`input_schema`、
    `server_info`。混用 v1 示例会在这里直接报 AttributeError。
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run() -> dict:
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m", "chargebench.mcp_server",
                "--store", str(tmp_path / "stdio_store"),
                "--configs", str(CONFIG_DIR),
            ],
            cwd=str(PROJECT_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = await session.list_tools()
                batch = await session.call_tool(
                    "run_batch",
                    {"config_name": "demo_scenario", "algorithms": ["EDF"], "seeds": [42]},
                )
                denied = await session.call_tool(
                    "run_batch", {"config_name": "../../etc/passwd"}
                )
                return {
                    "name": init.server_info.name,
                    "instructions": init.instructions or "",
                    "tools": sorted(t.name for t in tools.tools),
                    "batch": json.loads(batch.content[0].text),
                    "denied": json.loads(denied.content[0].text),
                }

    out = asyncio.run(run())

    assert out["name"] == "chargebench"
    assert "仿真" in out["instructions"]
    assert out["tools"] == [
        "compare_experiments", "get_results", "get_run", "list_algorithms",
        "list_batches", "parameter_sweep", "run_batch",
    ]
    assert out["batch"]["ok"] is True
    assert out["batch"]["data"]["completed_runs"] == 1
    assert out["denied"]["ok"] is False
    assert out["denied"]["error"]["code"] in {"invalid_argument", "unknown_scenario"}


def test_stdio_store_is_server_fixed(tmp_path):
    """客户端无法改结果库位置 —— 落盘位置只由服务端启动参数决定。"""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    fixed = tmp_path / "fixed_store"

    async def run() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "chargebench.mcp_server", "--store", str(fixed),
                  "--configs", str(CONFIG_DIR)],
            cwd=str(PROJECT_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool(
                    "run_batch", {"config_name": "demo_scenario", "algorithms": ["EDF"]}
                )

    asyncio.run(run())
    assert (fixed / "results.db").exists(), "结果必须落在服务端指定的位置"
