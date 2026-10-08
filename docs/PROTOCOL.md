# ChargeBench 输入/输出协议 v1.0

> 本文件是 UI、AI Agent、报告三条线共同的接口契约。
> **改字段或改口径必须同时改 `chargebench/schemas.py`、`chargebench/metrics.py` 和本文件，并升版本号。**
>
> 权威实现：`chargebench/schemas.py`（结构）· `chargebench/metrics.py`（口径）· `chargebench/adapter.py`（映射规则）

---

## 0. 快速上手

```bash
# 列出可用算法及其「能看到哪些信息」（公平性核对用）
python -m chargebench.cli list-algorithms

# 跑一次批量对比，输出协议 JSON
python -m chargebench.cli run --scenario configs/demo_scenario.json \
    --algorithm FCFS EDF LLF --out examples/

# 跨种子取均值与波动（方案 §5 公平性原则 2）
python -m chargebench.cli run --scenario configs/demo_scenario.json \
    --algorithm EDF --seeds 42 43 44 45 46

# 落盘并复用已算结果（重复运行同一配置不会重复计算）
python -m chargebench.cli run --scenario configs/demo_scenario.json \
    --algorithm FCFS EDF LLF --store results/

# 查询结果库
python -m chargebench.cli results  --store results/
python -m chargebench.cli batches  --store results/
python -m chargebench.cli describe <run_id> --store results/   # 这个结果当时怎么跑出来的

# 两变量扫描 + 适用边界热力图（换目标重排名不重跑仿真）
python -m chargebench.cli sweep --scenario configs/demo_scenario.json \
    --x 0.5 0.7 0.9 1.1 1.3 --y 0.40 0.55 0.70 0.85 \
    --seeds 42 43 --metric session_completion_rate \
    --compare peak_power_kw energy_cost_cny \
    --store results/ --plot reports/ --out results/ --detail
```

样例产出见 `examples/batch--*.json`。UI 可直接用它做图表原型，**不需要等真实数据接入**。

### 结果库的保证

`ResultStore` 写入时强制校验可复现性：同 `run_id` 但内容不同（`runtime_s` / `executed_at`
除外）会抛 `ResultMismatchError`。由于 `run_id` 是配置的确定性函数，这条校验实际上把
「实验必须可复现」从约定变成了强制——一旦出现非确定性，立刻暴露而不是静默覆盖。

缓存复用的判定依据是 `executed_at`：命中缓存返回的是原始结果，其执行时刻不变；
重算则会生成新时刻。仅比对运行条数无法区分这两者。

---

## 1. Scenario（输入）

单位一律为 **kWh / kW / A**。安培只出现在 Adapter 内部的 EVSE 建模，协议层不暴露。

| 字段 | 类型 | 默认 | 取值范围 | 含义 |
|---|---|---|---|---|
| `scenario_id` | str | — | — | 场景标识，进入 `run_id` |
| `n_ports` | int | — | 1–500 | 充电车位数 |
| `port_max_power_kw` | float | — | >0 | 单桩最大功率 [kW] |
| `voltage_v` | float | 220 | >0 | 额定电压 [V]，中国单相 220 / 三相 380 |
| `network_limit_kw` | float? | null | >0 | 站点聚合功率上限 [kW]；null 表示不限 |
| `time_step_min` | float | 5 | 0–60 | 仿真时间步长 [min] |
| `start_hour` | float | 8.0 | [0,24) | 窗口起始时刻 |
| `window_hours` | float | 10.0 | 0–48 | 仿真窗口长度 [h] |
| `n_sessions` | int | — | 1–10000 | 生成的车辆会话数 |
| `load_intensity` | float | — | 0–10 | **X 轴**，见 §2 |
| `deadline_tightness` | float | — | (0,1] | **Y 轴**，见 §2 |
| `energy_cv` | float | 0.0 | [0,1) | 单车请求电量的变异系数；0 = 全部相同 |
| `arrival_mode` | enum | `uniform` | `uniform` / `front_loaded` / `back_loaded` | 到达时间分布 |
| `price_profile` | object | 见 §1.2 | — | 分时电价 |
| `seed` | int | 42 | ≥0 | 随机种子 |

`extra="forbid"`：拼错字段名会**报错**而不是被静默忽略。

### 1.1 两个扫描轴的精确含义

**X 轴 · `load_intensity`（负荷强度）**

```
supply_capability_kwh = min(n_ports × port_max_power_kw, network_limit_kw) × window_hours
total_requested_kwh   = load_intensity × supply_capability_kwh
```

即「总请求电量 / 窗口内站点可供电量」。>1 表示必然供不应求。

**Y 轴 · `deadline_tightness`（时间紧迫度）**

```
required_h = 单车请求电量 / port_max_power_kw      # 满功率所需时长
stay_h     = required_h / deadline_tightness        # 可停留时长
```

=1 表示「刚好够满功率充完」，叠加聚合约束后必然不可行。

### 1.2 `price_profile`

| 字段 | 默认 | 说明 |
|---|---|---|
| `kind` | `tou` | `flat`（单一电价）或 `tou`（分时） |
| `currency` | `CNY` | 固定人民币 |
| `flat_cny_per_kwh` | 0.70 | 平段 |
| `valley_cny_per_kwh` | 0.30 | 谷段 |
| `peak_cny_per_kwh` | 1.10 | 峰段 |
| `valley_hours` | `[[0,8]]` | 谷段时段，绝对小时，左闭右开 |
| `peak_hours` | `[[10,15],[18,21]]` | 峰段时段 |

校验：`谷 < 平 < 峰`；每段满足 `0 ≤ 起点 < 终点 ≤ 24`。

> ACN-Sim 内置电价是美国加州电价（**美元**），直接使用会让成本指标币种错误，
> 故本项目自实现 `chargebench/tariff.py`。

---

## 2. 派生量与「选扫描范围」的必备公式

这些不是配置项，而是由上面字段算出来的量。**阶段 4 设定扫描网格前必须先看占用率。**

| 派生量 | 公式 |
|---|---|
| `periods` | `window_hours × 60 / time_step_min` |
| `effective_supply_kw` | `min(n_ports × port_max_power_kw, network_limit_kw)` |
| `supply_capability_kwh` | `effective_supply_kw × window_hours` |
| `total_requested_kwh` | `load_intensity × supply_capability_kwh` |
| `port_occupancy_estimate` | `load_intensity × effective_supply_kw / (n_ports × port_max_power_kw × deadline_tightness)` |

**`port_occupancy_estimate > 1` 时必然出现因无空闲车位而被丢弃的会话。** 推导见
`Scenario.port_occupancy_estimate` 的文档串。注意它与 `n_sessions` 无关——车辆数只改变
粒度，不改变总占用量。等价地，无丢弃条件是：

```
load_intensity ≤ deadline_tightness × (n_ports × port_max_power_kw / network_limit_kw)
```

演示场景 `campus_baseline_v1` 标定在占用率 0.659、`load_intensity = 1.0`：
网络约束真正生效，且仅丢弃 5/60 辆车——算法差异不会被丢弃噪声淹没。

---

## 3. RunResult（输出）

```jsonc
{
  "run_id": "campus_baseline_v1--EDF--2223c71dba",
  "scenario_id": "campus_baseline_v1",
  "scenario_hash": "…",            // 场景全部输入的 sha1，用于确认「跑的是同一个场景」
  "algorithm": "EDF",
  "algorithm_params": {},
  "seed": 42,
  "metrics": { /* 见 §4 */ },
  "traceability": {
    "schema_version": "1.0",
    "acnportal_version": "0.3.3",
    "python_version": "3.14.4",
    "numpy_version": "2.5.3",
    "pandas_version": "3.0.6",
    "benchmark_date": "2026-01-05",
    "simulation_start": "2026-01-05T08:00:00",
    "executed_at": "2026-10-09T00:36:12"
  }
}
```

`run_id` 的形式为 `{scenario_id}--{algorithm}--{10位哈希}`，哈希由
(场景, 算法, 参数, 种子, 协议版本) 确定性导出：

* 同配置在任何机器上都得到同一个 `run_id`，便于幂等重跑与跨机比对；
* `executed_at` 记录真实执行时刻，**不参与** `run_id`。

`BatchResult` 是 `RunResult` 的集合，额外带 `batch_id`、`created_at`、`scenario_ids`、
`algorithms`、`seeds`。`batch_id` 同样确定性导出。

---

## 4. 指标口径（唯一权威定义）

**任何地方需要指标数字都必须调用 `chargebench/metrics.py`。**
不允许在 UI / Agent / 报表里各自重算——否则口径会分叉，排名将随报表而变。

| 指标 | 口径 | 对应 ACN-Sim 函数 |
|---|---|---|
| `demand_satisfaction_rate` | **能量口径**：交付电量 / 请求电量。分母只含已排定车位的会话 | `proportion_of_energy_delivered` |
| `session_completion_rate` | **车辆口径**：剩余需求低于阈值的会话占比 | `proportion_of_demands_met` |
| `mean_delivery_ratio` / `worst_delivery_ratio` | 单车交付比（上限截断到 1.0），**仅统计已排定会话** | 自行计算 |
| `energy_cost_cny` | 按分时电价对每步聚合功率计费，单位 **CNY** | `energy_cost` |
| `peak_power_kw` | 窗口内最大聚合负荷 [kW] | `aggregate_power().max()` |
| `constraint_violations` | 越限的时间步数（所有约束合计） | `constraint_currents` |
| `max_violation_a` | 最大越限幅度 [A] | 同上 |
| `sessions_dropped` | 到达时无空闲车位而未进入仿真的会话数 | — |

### 三个必须记住的陷阱

1. **完成阈值是绝对电量，不是百分比。** `COMPLETION_THRESHOLD_KWH = 0.1` 表示
   「剩余不足 0.1 kWh 即算充满」。当成 10% 会静默改变排名。

2. **`peak_power_kw` 已经是 kW。** ACN-Sim 的 `aggregate_power` 返回值就是 kW，
   再除以 1000 会得到 0.06 这种量级。

3. **约束成为瓶颈时，总交付电量几乎不随算法变化。** 实测演示场景：

   | 算法 | 总交付 kWh | 需求满足率 | 按时完成率 |
   |---|---|---|---|
   | FCFS | 497.0 | 0.903 | 0.564 |
   | EDF | 496.6 | 0.903 | **0.691** |
   | LLF | 502.5 | 0.913 | 0.236 |

   总量差异 < 2%，但车辆口径相差 0.45。**只报总电量/总成本会得出「算法无差异」的错误结论**，
   必须并报分布型指标。`tests/test_metrics.py::test_completion_rate_is_independent_of_delivery_volume`
   把这个性质固定成了回归测试。

### 场景可行性不归因于算法

`sessions_dropped`、`sessions_generated`、`sessions_scheduled` 单列，不混入满足率分母。
方案 §5 公平性原则 5：场景本身不可行时不能归因于某个算法。
各算法看到的丢弃数必然相同（丢弃发生在算法介入之前），这条也有测试守护。

---

## 5. SweepResult（两变量参数扫描）

方案 §5 的核心交付：不只给算法排一次名，而是找出「哪个算法在什么条件下胜出」。

```jsonc
{
  "sweep_id": "sweep--b527cb11f1",
  "schema_version": "1.0",
  "base_scenario_id": "campus_baseline_v1",
  "x_name": "load_intensity",       "x_values": [0.5, 0.7, 0.9, 1.1, 1.3],
  "y_name": "deadline_tightness",   "y_values": [0.40, 0.55, 0.70, 0.85],
  "algorithms": ["FCFS", "EDF", "LLF"],
  "seeds": [42, 43],
  "primary_metric": "session_completion_rate",
  "metric_direction": "max",
  "score_rule": "1) 硬约束优先：…（事前写定，随结果持久化）",
  "cells": [
    {
      "load_intensity": 0.5, "deadline_tightness": 0.40,
      "port_occupancy_estimate": 0.41, "port_constrained": false,
      "sessions_dropped": 0,
      "aggregates": {
        "EDF": { "n_seeds": 2, "session_completion_rate_mean": 1.0, "spread": 0.0, … }
      },
      "winner": "EDF", "disqualified": []
    }
  ],
  "run_ids": ["…"]   // 本次扫描的全部运行，可逐条追溯到具体配置
}
```

### 评分规则（事前写定，不得事后更改）

规则文本存入 `score_rule` 字段与结果一起持久化，事后无法悄悄改口径
（方案 §5 公平性原则 4）。规则依次为：

1. **硬约束优先**：任何种子下 `constraint_violations > 0` 的算法在该格点取消资格。
   不允许把「违规供电」当成更好的结果（方案 §5 原则 3）。
2. **主目标**：`primary_metric` 的跨种子均值，方向见 `METRIC_DIRECTIONS`。
3. **平局**：取跨种子波动 `spread` 更小者 —— 均值相同但波动大的算法不可信。
4. **仍平局**：按算法名升序取首个，保证结果不依赖字典顺序。

越限判定采用**相对容差** `max(1e-3 A, 限值 × 1e-4)`，用于吸收 ACN-Sim 调度器
二分搜索留下的浮点残差（实测约 1e-7 相对量级）。若用绝对容差判定，
一个浮点噪声就会经由「违规即取消资格」静默翻转整张热力图的颜色。

### 换目标重排名不重跑仿真

`rerank_sweep(sweep, new_metric)` 只重新排序，不产生任何新实验 —— 一次实验的物理结果
与用哪个指标排名无关，而 `MetricAggregate` 已存下全部指标的跨种子均值。
因此 `--compare` 出来的多个 sweep **共享同一份 `run_ids`**，只是换了把尺子。

### 实测结论

同一批数据，换目标就换胜者（5×4 网格）：

| 目标函数 | 胜者分布 |
|---|---|
| `session_completion_rate` | EDF 17 / LLF 2 / FCFS 1 |
| `peak_power_kw` | FCFS 7 / LLF 7 / EDF 6 |
| `energy_cost_cny` | EDF 11 / FCFS 6 / LLF 3 |

即「哪个算法最好」没有唯一答案，取决于把什么当目标。这是平台核心论点的直接证据，
也是 `reports/objective-comparison--*.png` 那张图要传达的内容。

### 车位可行性单独标注

`port_constrained = port_occupancy_estimate > 1` 的格点用虚线框标出。这些格点会产生
大量丢弃，满足率的分母随之缩水，属场景可行性问题而非算法问题
（方案 §5 原则 5）。看热力图时务必先看这个标记。

---

## 6. AgentReport（AI 实验闭环）

方案 §6 要求 AI 成为实验研究员。闭环的产出结构如下：

```jsonc
{
  "report_id": "agent--d24d0335c8",
  "primary_metric": "session_completion_rate",
  "primary_direction": "max",
  "budget": { "max_rounds": 3, "max_simulations": 200, "max_seconds": 120.0,
              "patience": 2, "min_improvement": 0.01 },
  "stop_reason": "max_rounds",
  "baseline_outcomes": [ /* CandidateOutcome，官方基准 */ ],
  "rounds": [ { "round_index": 0, "outcomes": [...], "best_so_far": "...",
                "best_value": 0.9205, "improved": true, "simulations_used": 36 } ],
  "best_candidate": { "spec": { "name": "estimated_departure-CF+laxity", ... }, ... },
  "validation": { "candidate_name": "...", "best_baseline_name": "EDF",
                  "improvement": 0.2235, "passed": true, "scenario_ids": [...] },
  "conclusion": "在已测试范围与预算内，表现最好的可行候选是 ...",
  "caveats": [ "本结论只在已测试的 3 个实验场景与给定预算内成立，**不是全局最优** ..." ],
  "simulations_used": 77,
  "elapsed_seconds": 7.6,
  "all_run_ids": [ "..." ]
}
```

### 候选策略的表达方式：受控词汇，不是代码

`CandidateSpec` 只能由下列词汇组合而成 —— LLM 因此无法夹带代码（方案 §7
「禁止任意 Python 代码执行」），且每个假设都能用一句话说清：

| 字段 | 取值 | 说明 |
|---|---|---|
| `primary` | `arrival` / `estimated_departure` / `laxity` / `remaining_demand` / `delivery_ratio` | 主排序键 |
| `direction` | `asc` / `desc` | 排序方向 |
| `completion_first` | bool | 先排出**能在剩余停留时间内充满**的车辆（即松弛时间 ≥ 0） |
| `tiebreak` | 同上排序键或 null | 次级排序键 |
| `baseline` | bool | 为 true 时按名字走官方注册表，**不复用组合器** |

`baseline=true` 这一点很重要：基准必须调用 ACN-Sim 的官方实现，
否则就成了拿我们的重实现和官方实现对打，排名没有意义。

### 措辞纪律

`conclusion` 只能表述为「在已测试范围与预算内的最佳可行候选」。
`claims_optimality()` 以**否定感知**的方式检查措辞 —— 它区分「该算法达到全局最优」（违规）
与「本结论不是全局最优」（必需的免责声明）。测试直接复用这个函数，因此纪律不是靠人工 review。

`caveats` 是强制随附的，至少包含：结论的范围限制、停止原因。
若候选以牺牲其他指标为代价，还会追加「不应表述为全面更优」。

### 取舍必须报出

`detect_tradeoffs()` 会逐项比对候选与基准，把**变差**的指标写进结论。
实测中「完成度优先」类候选把按时完成率从 0.7874 抬到 0.9205，
但能量口径满足率从 0.9419 降到 0.9291。只报主指标等于掩盖代价。

---

## 7. MCP Server（对外工具接口）

方案 §7：MCP 是标准工具访问协议，**不是** Agent 推理系统本身。本项目先实现本地 Python API，
再把同一套 API 封装为 MCP 工具。

```bash
python -m chargebench.mcp_server --store results --configs configs
```

| Tool | 入参 | 返回 |
|---|---|---|
| `list_algorithms` | 无 | 算法列表、各自可见信息、候选策略词汇、服务端上限 |
| `run_batch` | `scenario` 内联对象或 `config_name`；`algorithms`；`seeds`；`deadline_seconds`；`parallel_scenarios` | `batch_id`、各运行指标摘要、数据位置 |
| `get_results` | `batch_id` | 结构化结果 + 每条运行的 JSON 落盘路径 |
| `get_run` | `run_id` | 单次运行完整结果与依赖版本 |
| `compare_experiments` | `run_ids` | **逐指标**胜者 + 诊断信息 + 不一致提示 |
| `parameter_sweep` | 网格 X/Y、算法、种子、主指标 | 胜者地图、评分规则、逐格明细 |
| `list_batches` | `limit` | 已记录的批次与统计 |

### 返回信封与错误码

所有工具返回统一信封，成功 `{ok: true, data: {...}}`，失败 `{ok: false, error: {code, message, ...}}`。

| 错误码 | 含义 |
|---|---|
| `invalid_argument` | 参数或内联场景不合法（含未知字段、路径类输入） |
| `unknown_algorithm` | 算法不在注册表内 |
| `unknown_scenario` | `config_name` 不在白名单内 |
| `not_found` | `batch_id` / `run_id` 不存在 |
| `budget_exceeded` | 批次规模超出服务端上限（**开跑前**即拒） |
| `timeout` | 在给定时间预算内一次都没跑完 |
| `internal_error` | 其他 |

### 安全边界

1. **结果库路径由服务端启动参数固定，不是工具参数。** 否则客户端能借工具读写任意路径。
   有测试遍历工具 schema，断言入参里不出现 `store` / `path` / `out` 等字段。
2. **`config_name` 只接受文件名主干**，拒绝路径分隔符与点开头。场景也可以内联传入
   （经 schema 校验），但不能传文件路径。
3. **不执行任何客户端提供的代码。** 算法只来自注册表；候选策略只来自 `SortKey` 受控词汇。
   源码有静态检查禁止 `eval` / `exec` / `compile` / `__import__`。
4. **超时返回部分结果并标记 `truncated`。** 此时 `runs` 只是计划集合的一部分 ——
   **不得**据此得出「某算法更优」的结论，因为候选之间跑的次数可能并不对等。

### `compare_experiments` 为什么不给单一排名

不同指标会给出不同的胜者（实测 `demand_satisfaction_rate` 与 `energy_cost_cny` 的胜者就不同）。
只给一个排名等于替调用方选定了目标函数，而那正是平台要避免的事。
另外**约束违规不参与排名** —— 平台规则是「违规即取消资格」，不是「违规少者胜出」。

> **mcp 2.x 与 v1 的改名**：`FastMCP` → `MCPServer`、`inputSchema` → `input_schema`、
> `serverInfo` → `server_info`。混用 v1 示例会直接报 `AttributeError`。

---

## 8. 公平性由构造保证

| 原则（方案 §5） | 实现方式 |
|---|---|
| 1. 各算法看到同样的会话与设施 | 每个算法都从 `build_scenario` 重新展开场景；测试断言各算法的 `energy_requested_kwh` / `sessions_scheduled` 完全一致 |
| 2. 固定种子、跨种子给区间 | `Scenario.seed` 决定全部随机性；`run_batch(seeds=[...])` 支持多种子；`run_id` 随种子变化 |
| 3. 不允许把约束违规当更优结果 | `constraint_violations` 是**独立指标**，不参与任何评分 |
| 4. 事前定义硬约束与主目标 | 口径写死在本文件 §4；改口径须升协议版本 |
| 5. 场景不可行不归因于算法 | 丢弃数单列，见 §4 |
| 6. 用保留场景验证 AI 改进 | 阶段 6 实现 |

**另有一层实现级保证**：仿真会改写 EV 状态，因此 `build_events()` 每次调用都重建全新的
EV 对象。若复用同一个事件队列，后跑的算法会读到被前一个算法改写过的脏数据——
这一点有测试守护（`test_events_are_rebuilt_fresh_each_call`）。

---

## 9. 已知简化（写进报告，避免过度解读结论）

1. **算法看到完美信息**：`estimated_departure == departure`，即假设用户预告的离站时间完全准确。
   真实场景中预告会偏差，EDF/LLF 的优势会缩水。
2. **电价不区分工作日/周末**，只按一天中的小时划分。
3. **车辆只能停在分配到的固定车位**，不支持换桩或共享充电枪。
4. **到达时刻按「剩余可用区间」缩放**，以在窗口内精确保持 `deadline_tightness`；
   代价是到达分布形状在窗口尾部被轻微压缩。
5. **停留时长超过窗口时被截断**，并计入 `n_stay_clamped`（见 `ScenarioArtifacts`）。
   截断会降低该车的实际紧迫度，因此必须单独可见。
6. **仿真终点 = 最后一个拔枪事件**，不等于 `window_hours`。

---

## 10. 扩展协议的正确姿势

1. 改 `chargebench/schemas.py` 的模型（记得同时改校验）；
2. 若涉及指标，改 `chargebench/metrics.py` 的口径函数；
3. 更新本文件，并把 `SCHEMA_VERSION` 加一；
4. 在 `tests/` 补一条断言把新行为钉住；
5. 重跑 `python -m chargebench.cli run … --out examples/` 刷新样例。
