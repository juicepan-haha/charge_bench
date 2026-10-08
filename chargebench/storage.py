"""结果持久化。

设计要点：

1. **双写**：SQLite 用于查询与聚合（阶段 4 的扫描、阶段 5 的 UI 都要按条件筛结果），
   同时把每份完整 RunResult 落成原始 JSON —— 方案 §4 要求一次实验能"用简单 JSON
   描述、复现和分享"，SQLite 文件不便于分享，JSON 才便于。

2. **常用指标扁平化进列**：把满足率、完成率、成本、峰值等提成真实列，
   于是"哪个算法在这个区间胜出"变成一条 SQL，而不是把全表读进内存再筛。

3. **幂等写入 + 一致性校验**：run_id 是 (场景, 算法, 参数, 种子, 协议版本) 的确定性函数，
   因此同配置重跑必然命中同一个 run_id。写入时若发现同 run_id 但内容不同，
   说明出现了非确定性 —— 这是严重问题，必须报错而不是静默覆盖。
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .schemas import BatchResult, RunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id       TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    scenario_ids   TEXT NOT NULL,
    algorithms     TEXT NOT NULL,
    seeds          TEXT NOT NULL,
    payload        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id                   TEXT PRIMARY KEY,
    batch_id                 TEXT,
    scenario_id              TEXT NOT NULL,
    scenario_hash            TEXT NOT NULL,
    algorithm                TEXT NOT NULL,
    algorithm_params         TEXT NOT NULL,
    seed                     INTEGER NOT NULL,
    schema_version           TEXT NOT NULL,
    executed_at              TEXT NOT NULL,
    acnportal_version        TEXT NOT NULL,
    demand_satisfaction_rate REAL NOT NULL,
    session_completion_rate  REAL NOT NULL,
    energy_delivered_kwh     REAL NOT NULL,
    energy_requested_kwh     REAL NOT NULL,
    energy_cost_cny          REAL NOT NULL,
    peak_power_kw            REAL NOT NULL,
    constraint_violations    INTEGER NOT NULL,
    sessions_dropped         INTEGER NOT NULL,
    runtime_s                REAL NOT NULL,
    payload                  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_batch    ON runs(batch_id);
CREATE INDEX IF NOT EXISTS idx_runs_scenario ON runs(scenario_id, algorithm, seed);
CREATE INDEX IF NOT EXISTS idx_runs_config   ON runs(scenario_hash, algorithm, seed);
"""


@dataclass(frozen=True)
class RunSummary:
    """结果摘要。UI 列表与 Agent 决策消费这个结构，不需要反序列化完整结果。"""

    run_id: str
    batch_id: str | None
    scenario_id: str
    scenario_hash: str
    algorithm: str
    seed: int
    demand_satisfaction_rate: float
    session_completion_rate: float
    energy_delivered_kwh: float
    energy_cost_cny: float
    peak_power_kw: float
    constraint_violations: int
    sessions_dropped: int


def _comparable(run: RunResult) -> dict[str, Any]:
    """用于一致性比较的规范化表示。

    剔除必然随时间变化、与实验内容无关的字段：runtime_s（机器负载）与
    executed_at（真实执行时刻）。
    """
    payload = run.model_dump(mode="json")
    payload["metrics"].pop("runtime_s", None)
    payload["traceability"].pop("executed_at", None)
    return payload


class ResultMismatchError(RuntimeError):
    """同一 run_id 出现不同内容 —— 说明实验不可复现，必须人工介入。"""


class ResultStore:
    """实验结果的持久化与查询。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "results.db"
        with self._session() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """一次性连接：成功提交、失败回滚、无论如何关闭。

        不能用 ``with sqlite3.connect(...) as conn`` —— sqlite3 的上下文管理器只管
        事务提交/回滚，**不会关闭连接**。那样每次操作都泄漏一个句柄，
        在 Streamlit 这种长驻进程里会持续累积（实测泄漏并被 ResourceWarning 抓到）。
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def save_run(self, run: RunResult, batch_id: str | None = None) -> None:
        """写入单次运行。幂等：同 run_id 重复写入内容一致时是空操作。

        已存在且带有内容差异时会抛 ``ResultMismatchError``；仅 batch_id 从空变为有值时
        会补写关联 —— ``run_batch`` 会先逐条落盘（防中途崩溃丢进度），最后才登记批次，
        因此这条路径必须能补上批次号。
        """
        existing = self._existing_payload(run.run_id)
        if existing is not None:
            if existing != _comparable(run):
                raise ResultMismatchError(
                    f"run_id {run.run_id} 已存在且内容不同。\n"
                    "run_id 由 (场景, 算法, 参数, 种子, 协议版本) 确定性导出，"
                    "同 id 不同内容说明实验出现了非确定性 —— 这是可复现性的根基建问题，"
                    "请先排查再覆盖。"
                )
            self._attach_batch(run.run_id, batch_id)
            return

        m = run.metrics
        with self._session() as conn:
            conn.execute(
                """
                INSERT INTO runs VALUES (
                    :run_id, :batch_id, :scenario_id, :scenario_hash, :algorithm,
                    :algorithm_params, :seed, :schema_version, :executed_at,
                    :acnportal_version, :demand_satisfaction_rate, :session_completion_rate,
                    :energy_delivered_kwh, :energy_requested_kwh, :energy_cost_cny,
                    :peak_power_kw, :constraint_violations, :sessions_dropped,
                    :runtime_s, :payload
                )
                """,
                {
                    "run_id": run.run_id,
                    "batch_id": batch_id,
                    "scenario_id": run.scenario_id,
                    "scenario_hash": run.scenario_hash,
                    "algorithm": run.algorithm,
                    "algorithm_params": json.dumps(run.algorithm_params, sort_keys=True),
                    "seed": run.seed,
                    "schema_version": run.traceability.schema_version,
                    "executed_at": run.traceability.executed_at.isoformat(),
                    "acnportal_version": run.traceability.acnportal_version,
                    "demand_satisfaction_rate": m.demand_satisfaction_rate,
                    "session_completion_rate": m.session_completion_rate,
                    "energy_delivered_kwh": m.energy_delivered_kwh,
                    "energy_requested_kwh": m.energy_requested_kwh,
                    "energy_cost_cny": m.energy_cost_cny,
                    "peak_power_kw": m.peak_power_kw,
                    "constraint_violations": m.constraint_violations,
                    "sessions_dropped": m.sessions_dropped,
                    "runtime_s": m.runtime_s,
                    "payload": json.dumps(run.model_dump(mode="json"), ensure_ascii=False),
                },
            )

        # 原始 JSON 另存一份：便于复现、分享与跨机比对，不依赖 SQLite
        (self.runs_dir / f"{run.run_id}.json").write_text(
            json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _attach_batch(self, run_id: str, batch_id: str | None) -> None:
        """把已存在的运行关联到批次。只在原先没有批次号时补写，不覆盖已有归属。"""
        if batch_id is None:
            return
        with self._session() as conn:
            conn.execute(
                "UPDATE runs SET batch_id = ? WHERE run_id = ? AND batch_id IS NULL",
                (batch_id, run_id),
            )

    def save_batch(self, batch: BatchResult) -> None:
        """写入整批结果。批内每次运行都会落盘。"""
        with self._session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?,?,?)",
                (
                    batch.batch_id,
                    batch.schema_version,
                    batch.created_at.isoformat(),
                    json.dumps(batch.scenario_ids, ensure_ascii=False),
                    json.dumps(batch.algorithms, ensure_ascii=False),
                    json.dumps(batch.seeds),
                    json.dumps(batch.model_dump(mode="json"), ensure_ascii=False),
                ),
            )
        for run in batch.runs:
            self.save_run(run, batch_id=batch.batch_id)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def _existing_payload(self, run_id: str) -> dict[str, Any] | None:
        with self._session() as conn:
            row = conn.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        payload["metrics"].pop("runtime_s", None)
        payload["traceability"].pop("executed_at", None)
        return payload

    def get_run(self, run_id: str) -> RunResult:
        """按 run_id 取回完整结果。优先读原始 JSON，保证与分享出去的文件逐字一致。"""
        json_path = self.runs_dir / f"{run_id}.json"
        if json_path.exists():
            return RunResult(**json.loads(json_path.read_text(encoding="utf-8")))
        with self._session() as conn:
            row = conn.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"未找到 run_id {run_id}")
        return RunResult(**json.loads(row["payload"]))

    def get_batch(self, batch_id: str) -> BatchResult:
        with self._session() as conn:
            row = conn.execute(
                "SELECT payload FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"未找到 batch_id {batch_id}")
        return BatchResult(**json.loads(row["payload"]))

    def list_runs(
        self,
        batch_id: str | None = None,
        scenario_id: str | None = None,
        algorithm: str | None = None,
        seed: int | None = None,
        order_by: str = "scenario_id, algorithm, seed",
    ) -> list[RunSummary]:
        """按条件查询结果摘要。order_by 只接受白名单列名，杜绝字符串拼接注入。"""
        allowed = {
            "scenario_id", "algorithm", "seed", "demand_satisfaction_rate",
            "session_completion_rate", "energy_cost_cny", "peak_power_kw", "run_id",
        }
        columns = [c.strip() for c in order_by.split(",") if c.strip()]
        if not columns or any(c not in allowed for c in columns):
            raise ValueError(f"order_by 只支持 {sorted(allowed)}（可逗号分隔多列），收到 {order_by!r}")
        order_clause = ", ".join(columns)

        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("batch_id", batch_id),
            ("scenario_id", scenario_id),
            ("algorithm", algorithm),
            ("seed", seed),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._session() as conn:
            rows = conn.execute(
                f"""
                SELECT run_id, batch_id, scenario_id, scenario_hash, algorithm, seed,
                       demand_satisfaction_rate, session_completion_rate,
                       energy_delivered_kwh, energy_cost_cny, peak_power_kw,
                       constraint_violations, sessions_dropped
                FROM runs {where} ORDER BY {order_by}
                """,
                params,
            ).fetchall()
        return [RunSummary(**dict(row)) for row in rows]

    def find_existing(
        self, scenario_hash: str, algorithm: str, seed: int, algorithm_params: dict[str, Any] | None = None
    ) -> RunResult | None:
        """按配置查找已算过的结果，用于跳过重复计算。

        这是 Agent 闭环（阶段 6）反复试错时的成本闸门：预算有限，
        算过的配置必须直接复用，不能重复烧仿真时间。
        """
        params_json = json.dumps(algorithm_params or {}, sort_keys=True)
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT run_id FROM runs
                WHERE scenario_hash = ? AND algorithm = ? AND seed = ?
                  AND algorithm_params = ?
                """,
                (scenario_hash, algorithm, seed, params_json),
            ).fetchone()
        return self.get_run(row["run_id"]) if row else None

    def list_batches(self) -> list[dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT batch_id, schema_version, created_at, scenario_ids, algorithms, seeds,"
                " (SELECT COUNT(*) FROM runs WHERE runs.batch_id = batches.batch_id) AS n_runs"
                " FROM batches ORDER BY created_at DESC"
            ).fetchall()
        out = []
        for row in rows:
            entry = dict(row)
            for key in ("scenario_ids", "algorithms", "seeds"):
                entry[key] = json.loads(entry[key])
            out.append(entry)
        return out

    def counts(self) -> dict[str, int]:
        with self._session() as conn:
            return {
                "batches": conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0],
                "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            }

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------

    def format_table(
        self,
        batch_id: str | None = None,
        scenario_id: str | None = None,
        algorithm: str | None = None,
    ) -> str:
        """人类可读的对比表。judge 判据是「这个结果当时用什么配置跑出来的」能一眼看出。"""
        runs = self.list_runs(
            batch_id=batch_id, scenario_id=scenario_id, algorithm=algorithm
        )
        if not runs:
            return "（无匹配结果）"

        lines = [
            f"{'scenario':22} {'algo':11} {'seed':>5} {'satis':>7} {'complete':>9} "
            f"{'kWh':>8} {'cost':>9} {'peak':>7} {'viol':>5} {'drop':>5}",
            "-" * 96,
        ]
        for r in runs:
            lines.append(
                f"{r.scenario_id:22} {r.algorithm:11} {r.seed:5d} "
                f"{r.demand_satisfaction_rate:7.3f} {r.session_completion_rate:9.3f} "
                f"{r.energy_delivered_kwh:8.1f} {r.energy_cost_cny:9.2f} "
                f"{r.peak_power_kw:7.2f} {r.constraint_violations:5d} {r.sessions_dropped:5d}"
            )
        return "\n".join(lines)

    def describe_run(self, run_id: str) -> str:
        """一句话说清「这个结果当时是用什么跑出来的」。"""
        run = self.get_run(run_id)
        t = run.traceability
        return (
            f"{run.run_id}\n"
            f"  场景      {run.scenario_id}  (hash {run.scenario_hash[:12]})\n"
            f"  算法      {run.algorithm}  参数 {run.algorithm_params or '默认'}\n"
            f"  种子      {run.seed}\n"
            f"  协议版本  {t.schema_version}\n"
            f"  依赖      acnportal {t.acnportal_version} / python {t.python_version} / "
            f"numpy {t.numpy_version} / pandas {t.pandas_version}\n"
            f"  基准日    {t.benchmark_date}  仿真起点 {t.simulation_start.isoformat()}\n"
            f"  执行时刻  {t.executed_at.isoformat()}\n"
            f"  结果      satis={run.metrics.demand_satisfaction_rate:.3f} "
            f"complete={run.metrics.session_completion_rate:.3f} "
            f"cost={run.metrics.energy_cost_cny:.2f} CNY peak={run.metrics.peak_power_kw:.2f} kW"
        )

    def iter_run_files(self) -> Iterator[Path]:
        return iter(sorted(self.runs_dir.glob("*.json")))
