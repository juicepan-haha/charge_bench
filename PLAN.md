# ChargeBench AI｜分步实施计划

> 本文档是《ChargeBench_AI_项目方案.md》的落地版。方案回答"做什么"，本文回答"按什么顺序做、每步的验收标准、以及已经踩掉的坑"。
>
> **前置结论：方案里标记的头号风险（ACN-Sim 兼容性）已在 2026-10-09 实测解除。** 证据见 §1，可直接进入开发。

---

## 1. 已验证的环境基线（实测，非推测）

本机实测通过，可直接复现：

```bash
python3 -m venv .venv
.venv/bin/pip install "acnportal==0.3.3" "setuptools<81"
```

| 项目 | 实测结果 |
|---|---|
| Python | 3.14.4（系统自带，`/usr/bin/python3.14`） |
| acnportal | 0.3.3（PyPI wheel，纯 Python） |
| 传递依赖 | numpy 2.5.3 / pandas 3.0.6 / scipy 1.18.1 / scikit-learn 1.9.1 / matplotlib 3.11.2 |
| 库自带测试 | **146 passed, 12468 subtests passed**（`acnsim/tests`、`algorithms/tests`、`network/tests`） |
| 端到端仿真 | 跑通：有限车位 + 网络功率约束 + FCFS/EDF/LLF 对比 + 指标提取 |
| 单次运行耗时 | 6 车位/19 会话 = 52 ms；20/80 = 273 ms；50/219 = 761 ms |

**⚠️ 必须固定 `setuptools<81`。** `acnportal` 在 `acnsim/base.py:17` 里 `import pkg_resources`，而 setuptools 81+ 已彻底移除该模块。装最新的 setuptools（实测 84.0.0）会导致 `ModuleNotFoundError: No module named 'pkg_resources'`，整个包无法导入。这是一个会在比赛当天炸掉的坑，必须写进 `pyproject.toml` 并锁定。

**⚠️ 官方在线教程与安装版单位不一致。** 教程（2019 年）用 A·periods，但 0.3.3 的 `Battery` / `EV` 已改为 kWh/kW。照抄教程会得到量级完全错误的指标（实测满足率会从 38.5% 掉到 0.7%）。正确约定见 §2。

复现脚本见 `scratch/acn_e2e_verified.py`。

---

## 2. 必须钉死的单位与 API 约定

这一节是整个 Adapter 层的正确性基础。写错任何一条，指标就会静默错误——不会报错，只会给出看起来合理但完全错的数字。

**单位系统（0.3.3 实测）**

| 对象 | 字段 | 单位 | 说明 |
|---|---|---|---|
| `EVSE(station_id, max_rate)` | `max_rate` | **A（安培）** | 单桩最大电流 |
| `Battery(capacity, init_charge, max_power)` | 前两个 | **kWh** | 电池容量 / 初始电量 |
| `Battery(...)` | `max_power` | **kW** | 车载最大充电功率 |
| `EV(arrival, departure, requested_energy, ...)` | `arrival`/`departure` | **period 序号（int）** | 不是 datetime |
| `EV(...)` | `requested_energy` | **kWh** | 用户请求电量 |
| `EV.energy_delivered` | — | **kWh** | 由 `(A × V)/1000 × (period/60)` 累加 |
| `add_constraint(...)` | `limit` | **A** | 聚合电流上限 |

单桩最大功率换算：`max_kw = max_rate_A × voltage_V / 1000`。

**API 签名（实测，与教程有出入）**

```python
cn = ChargingNetwork()
cn.register_evse(EVSE("PS-001", max_rate=32), voltage=208, phase_angle=0)
cn.add_constraint(Current(["PS-001", "PS-002"]), limit=80, name="transformer")
#                  ↑ 负载系数，非限值；限值是独立参数（旧文档写成 Current(limit) 是错的）

sim = Simulator(network, scheduler, events, start_dt, period=5, verbose=False)
sim.run()
```

**可用的官方算法**（`acnportal.algorithms`，与方案的 FCFS/EDF/LLF 一一对应）：

- `first_come_first_served` — FCFS
- `earliest_deadline_first` — EDF
- `least_laxity_first` — LLF

统一用 `SortedSchedulingAlgo(sort_fn)` 包装。默认 `max_recompute = 1`（每周期重算一次）。

---

## 3. 指标口径（方案 §5 的落地映射）

好消息：方案里列的核心指标，ACN-Sim 的 `acnsim.analysis` 基本原生覆盖，不需要自己算物理量。

| 方案指标 | 原生函数 | 口径陷阱 |
|---|---|---|
| 需求满足率 | `proportion_of_energy_delivered(sim)` | 按**总电量**统计。要按车辆统计需自己写 |
| 按时完成率 | `proportion_of_demands_met(sim, threshold)` | ⚠️ `threshold` 是**绝对剩余电量（kWh）**，不是百分比。默认 0.1 表示"剩余不足 0.1 kWh 即算充满" |
| 总用电成本 | `energy_cost(sim, tariff)` | ⚠️ 返回值单位是**美元**。内置电价是美国加州 PGE/SCE，**没有人民币电价** |
| 峰值负荷 | `aggregate_power(sim).max()` | 返回值**已经是 kW**，不要再除以 1000 |
| 约束违规 | `constraint_currents(sim)` | 返回 `Dict[约束名, np.ndarray]`，不是单个数组 |
| 总交付/请求电量 | `total_energy_delivered` / `total_energy_requested` | 单位 kWh，可直接用于对账 |

**人民币电价方案**：内置 `TimeOfUseTariff` 只读包内 JSON（美式电价）。用一个鸭子类型的自定义类即可，实测可行：

```python
class CNTariff:
    def get_tariffs(self, start_dt, length, period):  # 返回每周期单价 CNY/kWh
        ...
```

`energy_cost(sim, tariff)` 只调用 `tariff.get_tariffs(...)`，不检查类型。实测输出 89.36 CNY（均价 0.569 CNY/kWh，峰谷分时），合理。

**一个必须写进文档的测量学结论**：当网络功率约束成为瓶颈时，**所有算法的总交付电量完全相同**（实测 FCFS/EDF/LLF 均为 156.97 kWh）。差异只体现在"这些电怎么分配"。因此：

- 只报"总电量/总成本"会得出"算法无差异"的错误结论；
- 必须同时报告**分布型指标**（满足率、按时完成率）。实测同一批数据下 EDF 19.0% > FCFS 14.3% > LLF 0.0%。

---

## 4. 分阶段实施步骤

每阶段末尾给出**验收标准**。严格按序，因为后一阶段依赖前一阶段的产物。括号内是方案 §8 的 24 小时时段对照。

### 阶段 0：环境固化（0–3 小时对应项）✅ 已完成

**已产出**（2026-10-09 落地并验证）
1. `pyproject.toml` — 锁 `acnportal==0.3.3`、`setuptools<81`、`numpy<3`、`pandas<4`
2. `requirements.lock.txt` — `pip freeze` 全量快照
3. `.venv/` — 项目独立虚拟环境（`.gitignore` 已排除）
4. `scratch/acn_e2e_verified.py` — 环境健康检查 + 阶段 2 Adapter 骨架
5. `tests/test_environment.py` — **8 项契约测试全部通过**：依赖上界、kWh/kW 单位、约束不越限、三算法可用、同种子可复现、指标 threshold 口径

**验收（已完成）**：`.venv/bin/python -m pytest tests/ -q` → 8 passed；`.venv/bin/python scratch/acn_e2e_verified.py` → 输出见 §7。

**为什么这一步不能省**：方案的"保底方案"提到要预留容器/锁依赖方案。现在锁，比赛当天不慌。

**新环境重建方式**：`python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt && .venv/bin/pip install -e ".[dev]"`

---

### 阶段 1：先定协议，再做实现（3–4 小时）✅ 已完成

**核心原则**（方案 §4）：上层不依赖 ACN-Sim 内部对象。一次实验能用简单 JSON 描述、复现、分享。

**已产出**
1. `chargebench/schemas.py` — pydantic v2 定义 `Scenario` / `RunResult` / `BatchResult` / `Metrics` / `Traceability`，含逐字段单位、取值范围与非法组合校验（`extra="forbid"`，拼错字段名会报错）
2. `configs/demo_scenario.json` — 已标定到关键区间的真实场景
3. `docs/PROTOCOL.md` — 接口契约。含两个扫描轴的精确含义、指标口径的权威定义、公平性保证、**已知简化清单**（写报告时要照抄，避免过度解读结论）
4. `chargebench/cli.py` — 命令行入口，让不写 Python 的队友也能直接产出协议 JSON
5. `examples/batch--*.json` — 真实跑出的样例结果，UI 可直接拿去做图表原型

**三个关键设计决定（已定，改动成本高）**
- **抽象参数 → 事件的映射规则写进了文档**（`docs/PROTOCOL.md` §1.1），不留悬念：
  `required_h = 能量 / 桩功率`，`stay_h = required_h / deadline_tightness`。
- **单位统一 kWh / kW**，安培只出现在 Adapter 内部的 EVSE 建模。用户可见字段一律 kW/kWh。
- **`run_id` 确定性导出**：`{scenario_id}--{algorithm}--{hash}`，哈希含场景+算法+参数+种子+协议版本。同配置在任何机器上都得到同一 id，支持幂等重跑与跨机比对。

**额外发现**：pydantic v2 的 `model_copy(update=...)` **跳过校验**，会把 `kind="flat"` 存成裸字符串而非枚举成员，
导致下游比较静默失配、悄悄退回默认行为。已提供 `Scenario.with_updates()` 走完整校验，并在 `docs/PROTOCOL.md` 记录。

**验收（已完成）**：`examples/batch--*.json` 能被 `BatchResult` 原样解析回模型（`tests/test_cli.py` 守护往返）。

**新发现的关键公式**（阶段 4 选扫描范围前必看）：

```
port_occupancy_estimate = load_intensity × effective_supply_kw / (n_ports × port_max_power_kw × deadline_tightness)
```

**> 1 时必然出现因无空闲车位被丢弃的会话**，且与车辆数无关。等价地，无丢弃条件为
`load_intensity ≤ deadline_tightness × (n_ports × port_max_power_kw / network_limit_kw)`。

---

### 阶段 2：Adapter + 单次运行（对应"必须完成"）✅ 已完成

**已产出**
1. `chargebench/adapter.py` — 技术核心：`Scenario` → 网络 + 事件队列，含**有限车位分配**
2. `chargebench/algorithms.py` — 算法注册表 FCFS / EDF / LLF / UNCONTROLLED，**显式声明每个算法能看到哪些信息**（公平性核对用）
3. `chargebench/metrics.py` — 指标口径的唯一来源
4. `chargebench/tariff.py` — 人民币分时电价（鸭子类型适配，不继承 ACN-Sim 基类）
5. `chargebench/experiments.py` — `run_experiment` / `run_batch`

**⚠️ 车位分配是自己写的。** ACN-Sim 的 `StochasticEvents._convert_ev_matrix` 给每辆车分配独立站点（源码注释写着 "Infinite space assumption"）。实测的贪心策略：按到达时间排序，选「在车辆到达时已空闲」的车位中最早释放的那个；无空位则丢弃并计数。**不双重占用**这一不变量有测试守护。

**⚠️ 每次 `build_events()` 都重建全新 EV 对象。** 仿真会改写 EV 状态，复用同一事件队列会让后跑的算法读到被前一个算法改写的脏数据。这一点有测试守护。

**验收（已完成）**：方案 §14 的第一个可验收任务已通过，并已远超它——现在是协议化、可复现、有 75 项测试守护的完整垂直切片：

```
scenario               algo    seed   satis  complete      kWh      cost    peak  drop
campus_baseline_v1     FCFS      42   0.903     0.564    497.0    467.87   60.00     5
campus_baseline_v1     EDF       42   0.903     0.691    496.6    467.64   60.00     5
campus_baseline_v1     LLF       42   0.913     0.236    502.5    471.74   60.00     5
```

场景标定在占用率 0.659、`load_intensity = 1.0`：网络约束真正生效（峰值贴住 60 kW 上限），
仅丢弃 5/60 辆车。**总交付电量三者几乎相同（差异 < 2%），但按时完成率相差 0.45** ——
这正是方案 §5 要求「提前定义口径」的实证。

---

### 阶段 3：批量运行 + 结果存储（3–7 小时）✅ 已完成

**已产出**
1. `chargebench/experiments.py` — `run_experiment()` / `run_batch()`，支持跨种子
2. `chargebench/metrics.py` — 指标口径唯一来源（阶段 2 已完成）
3. `chargebench/storage.py` — SQLite（查询）+ 原始 JSON（复现/分享）双写
4. `chargebench/cli.py` — 增加 `--store` / `results` / `describe` / `batches` 子命令
5. `tests/test_storage.py` + `tests/test_cli.py` — 17 + 14 项守护

**关键实现决定**

- **双写**：SQLite 供查询聚合（阶段 4 扫描、阶段 5 UI 都要按条件筛），每份完整结果另存
  原始 JSON —— 方案 §4 要求实验能"用简单 JSON 描述、复现和分享"，`results.db` 不便分享。
- **常用指标扁平化进真实列**：于是"哪个算法在这个区间胜出"是一条 SQL，
  而不是把全表读进内存再筛。`order_by` 走列名白名单，杜绝字符串拼接注入。
- **幂等写入 + 非确定性检测**：`run_id` 是配置的确定性函数，同配置重跑必然命中同一 id。
  写入时若发现同 id 但内容不同（`runtime_s` / `executed_at` 除外），说明实验出现了
  **非确定性**——必须报 `ResultMismatchError` 而不是静默覆盖。这把可复现性从"约定"变成了"强制"。
- **缓存复用是 Agent 闭环的成本闸门**：`run_batch(store=...)` 命中已算配置直接复用。
  实测重复运行同一批 9 次仿真，结果库仍为 9 条记录。

**验收（已完成）**

```
$ chargebench run --scenario configs/demo_scenario.json --algorithm FCFS EDF LLF \
      --seeds 42 43 44 --store results/
$ chargebench describe campus_baseline_v1--EDF--5570508019
  ...
  依赖      acnportal 0.3.3 / python 3.14.4 / numpy 2.5.3 / pandas 3.0.6
  基准日    2026-01-05  仿真起点 2026-01-05T08:00:00
  结果      satis=0.903 complete=0.691 cost=467.64 CNY peak=60.00 kW
```

跨 3 个种子的完成率：

| 算法 | 均值 | 最小 | 最大 | 波动 |
|---|---|---|---|---|
| EDF | 0.691 | 0.650 | 0.732 | 0.082 |
| FCFS | 0.620 | 0.564 | 0.679 | 0.115 |
| LLF | 0.431 | 0.236 | 0.607 | **0.371** |

**这是阶段 3 最有价值的产出**：LLF 不仅均值更差，波动还大 4 倍。单次运行完全看不出这一点
（seed=43 时 LLF 的 0.450 接近 FCFS），必须跨种子才暴露。这直接印证了方案 §5 公平性原则 2，
也是阶段 4 热力图之外值得单独讲的一条结论。

---

### 阶段 4：两变量参数扫描 + 适用边界热力图（7–11 小时）✅ 已完成

**已产出**
1. `chargebench/sweep.py` — `run_sweep()` 两变量扫描、`rerank_sweep()` 换目标重排名、ASCII 区域图
2. `chargebench/viz.py` — 三张图：算法适用区域图、逐算法指标小图、**目标对比图**
3. `chargebench/schemas.py` — `SweepResult` / `SweepCell` / `MetricAggregate` 纳入协议
4. `chargebench/cli.py` — `sweep` 子命令（`--x` / `--y` / `--metric` / `--compare` / `--plot` / `--store`）
5. `tests/test_sweep.py`（30 项）、`tests/test_viz.py`（15 项）

**三个关键设计决定**

- **评分规则事前写定并随结果持久化**（方案 §5 原则 4）。规则文本存进 `SweepResult.score_rule`，
  与结果一起落盘，事后无法悄悄改口径。规则为：硬约束优先 → 主目标跨种子均值 → 平局取波动更小者
  → 仍平局按算法名升序。
- **`rerank_sweep()` 换目标不重跑仿真**。一次实验的物理结果与用哪个指标排名无关，而
  `MetricAggregate` 已存下全部 6 个指标的均值，因此换尺子只需重新排序。这既省算力，
  也让论点更锋利：**同一批 `run_id`，只是换了把尺子。**
- **车位可行性单独标注**。占用率 > 1 的格点打虚线框，提示丢弃会缩水满足率的分母
  （方案 §5 原则 5：场景不可行不归因于算法）。

**实测发现（本次最有价值的产出）**

平台的核心论点得到了直接证据：**同一批实验数据，换一个优化目标，胜者就变了。**

| 目标函数 | 胜者分布（5×4 网格） |
|---|---|
| `session_completion_rate` | EDF 17 / LLF 2 / FCFS 1 —— EDF 近乎通吃 |
| `peak_power_kw` | FCFS 7 / LLF 7 / EDF 6 —— 三方均衡 |
| `energy_cost_cny` | EDF 11 / FCFS 6 / LLF 3 |

即「哪个算法最好」这个问题**没有唯一答案，取决于你把什么当目标**。
这正是方案 §0 卖点 1「公平评测」与卖点 2「边界探索」要共同说明的事。

**⚠️ 途中修掉的一个会静默翻转热力图的 bug**：约束越限判定原先用绝对容差 `1e-6 A`，
而 ACN-Sim 调度器用二分搜索求解功率分配，会留下约 1e-7 相对量级的浮点残差 ——
实测在 272.7 A 的限值上出现 1.99e-05 A 的超出。由于评分规则是「违规即取消资格」，
这个数值噪声会把完全正常的算法踢出排名，**静默改变整张热力图的颜色**。
已改为相对容差 `max(1e-3 A, 限值×1e-4)`（`metrics.violation_threshold_a`），并写进评分规则文本。

**性能实测**：单次运行 0.05–0.76 秒。实测 5×4 网格 × 3 算法 × 2 种子 = 120 次运行，
全量约 6 秒；配上 `--store` 缓存后重复扫描接近免费。

**验收（已完成）**：三张图产出在 `reports/`，中文标签正常渲染（字体缺失时自动退回英文，
不会画成豆腐块），且「为什么这一格是这个算法赢」能从 `--detail` 的逐格明细里读出来。

---

### 阶段 5：Streamlit 仪表板（11–15 小时）✅ 已完成

**已产出**
1. `app.py` — 四个标签页：批量对比 / 过程回放 / 适用边界 / 结果库
2. `chargebench/timeseries.py` — 负荷曲线提取（聚合负荷、电价时段、逐站点功率、累计电费）
3. `chargebench/viz.py` — 新增 `plot_load_curve` 与 `plot_load_curves_side_by_side`
4. `tests/test_timeseries.py`（18 项）、`tests/test_app.py`（8 项）、`tests/test_viz.py` 扩到 24 项

**设计决定**

- **UI 不做任何自定义计算**。页面只调用 `chargebench` 包的函数并展示结果。
  `tests/test_app.py::test_metrics_frame_matches_core_api` 断言表格数值与 `run_batch`
  逐字一致 —— 否则页面上会出现和报告不一样的数字。
- **曲线按需重算而不落盘**。一次扫描上百次运行，把每个运行的 (站点 × 时间步) 矩阵
  都持久化会让结果库膨胀几十倍，而时序只在「看某次运行的过程」时才需要。
  场景与种子都是确定性的，重算必然得到同一条曲线。
- **换目标重排名不重跑仿真**（沿用阶段 4 的 `rerank_sweep`）。界面上的三个目标面板
  共享同一份 `run_id`，只是换了把尺子。

**验收（已完成，浏览器实测）**

- 侧边栏场景概览显示占用率 0.66，与 `Scenario.port_occupancy_estimate` 一致
- 批量对比标签页自动跑出结果：`本轮按时完成率最高：EDF（0.691，seed=42）`
- 适用边界标签页点击「开始扫描」后跑完 60 次仿真，产出胜者地图、评分规则、
  ASCII 区域图、三目标对比图，**胜者分布与命令行完全一致**
- 中文全程正常渲染

**途中修掉的两个渲染缺陷**（都是只有在真实渲染/浏览器里看才发现，断言全绿也照样存在）

1. **豆腐块**：绘图函数依赖调用方先设置字体，而 Streamlit 与测试是直接调 `plot_*` 的。
   结果是整张图的中文渲染成空框，**而断言只检查「标题里有中文字符串」—— 字符串永远是中文，
   豆腐块照样通过，测试全绿但图是废的**。现在字体检测在绘图函数内部完成并缓存，
   且无中文字体时强制退回英文标签。同时补了 `test_chinese_labels_actually_render`
   （在 **savefig 期间**捕获缺字形警告）与 `test_english_mode_contains_no_chinese`
   （英文模式下不得出现任何中文，强制所有可见文字走标签字典。
   这条当场抓出三处硬编码中文标题）。
2. **图例/标题互相压住**：`layout="constrained"` 不会为 figure 级图例和 suptitle 留位，
   而 `engine.set(rect=...)` 实测不生效。这张图的版式改为显式切分竖直带状区域，
   完全确定性。

**另有一条实测结论值得写进报告**：FCFS / EDF / LLF 都是「贪心填充」策略 ——
只要有额度就尽早充。所以**负荷曲线形状相近，峰值都贴住站点上限**；它们只在争抢激烈时
因分配优先级不同而分化，本身并不做价格套利。想让负荷形状明显不同需要引入价格感知算法
（方案 §8 的「可选加分」项）。

---

### 阶段 6：AI 实验闭环最小演示（15–19 小时）✅ 已完成

**已产出**
1. `chargebench/agent_loop.py` — 完整的「提出假设 → 调用仿真 → 分析 → 决策」闭环
2. `chargebench/schemas.py` — `SortKey` / `CandidateSpec` / `BudgetSpec` / `AgentReport` / `ValidationResult`
3. `chargebench/experiments.py` — 新增 `run_with_scheduler()`，让闭环能评测注册表之外的候选策略
4. `chargebench/cli.py` — `agent` 子命令，含 `--llm-command` 外部模型接入
5. `chargebench/viz.py` — `plot_agent_report()`：候选对比 + 逐轮收敛
6. `tests/test_agent_loop.py`（32 项）、`test_cli.py` 增 7 项、`test_viz.py` 增 5 项

**关键设计决定**

- **LLM 不能写代码，只能组合受控词汇。** 方案 §7 要求「禁止任意 Python 代码执行」，
  而方案 §6 要 LLM 提假设 —— 两者合起来指向同一个设计：给 LLM 一套排序键白名单
  （`SortKey`：arrival / estimated_departure / laxity / remaining_demand / delivery_ratio）
  加一个 `completion_first` 开关，它输出的永远是结构化规格。
  额外字段（比如夹带 `__import__`）会被 pydantic 的 `extra="forbid"` 直接拒绝。
- **内层数值搜索与外层 LLM 推理分离**（方案 §6 的两个循环层次）。`HeuristicProposer`
  是确定性的内层搜索，`LLMProposer` 是外层推理；后者失效时自动回退到前者，
  满足方案 §8 的「先保障离线回退」。
- **四重预算闸门**：批次数、仿真调用数、墙钟时间、连续无改善轮数。任一触发即停。

**实测结论（这就是方案 §6 想要的那个闭环）**

内置启发式的先验顺序把「完成度优先」排在最前 —— 正好对应方案里点名的那个假设。
在 3 个实验场景 × 2 种子上跑 3 轮，77 次仿真、7.6 秒：

| 策略 | 按时完成率 | 需求满足率 | 种子波动 |
|---|---|---|---|
| FCFS（基准） | 0.7462 | 0.9423 | 0.1187 |
| EDF（基准） | 0.7874 | 0.9419 | 0.0859 |
| LLF（基准） | 0.5949 | 0.9485 | 0.1990 |
| **estimated_departure-CF+laxity（候选）** | **0.9205** | 0.9291 | **0.0480** |

* 保留场景复验：**0.9041 对 EDF 的 0.6806，通过**（差值 +0.2235）
* 完成率相对 EDF 提升 +0.1331，同时种子波动从 0.0859 降到 0.0480 —— 不只是更好，也更稳
* **但这是一次有代价的改进**：能量口径满足率从 0.9419 降到 0.9291。该策略把电量集中
  喂给「能在剩余时间内喂饱」的车，总交付电量反而略降。

**⚠️ 两个只有实测才会暴露的问题**

1. **我的第一版结论掩盖了取舍。** 只报主指标时，输出是「表现最好的候选」——
   听起来是全面更优，实际是拿满足率换了完成率。已加入 `detect_tradeoffs()`：
   结论必须列出同时变差的指标，并在 caveats 里注明「不应表述为全面更优」。
2. **措辞纪律的测试差点是假的。** 朴素的子串检查会把必需的免责声明
   「本结论**不是**全局最优」判成违规 —— 那样的测试要么被绕过，要么逼人删掉免责声明。
   已改为否定感知的 `claims_optimality()`，并专门测试它能否区分
   「不是最优」与「就是最优」。

另有一处公平性补漏：候选选优原先只看主指标，**出现约束违规的候选仍可能胜出**。
已改为违规即排到最后（返回 `+inf`），与扫描模块的规则一致（方案 §5 原则 3）。

**验收（已完成）**：AI 基于真实结果提出假设 → 调用引擎评测 → 在未参与筛选的保留场景上
完成复验。全程每个数字都可经 `run_id` 追溯到具体配置。

---

### 阶段 7：MCP Server ✅ 已完成

**已产出**
1. `chargebench/mcp_server.py` — 7 个工具 + 标准错误码 + 服务端硬上限
2. `chargebench/schemas.py` — `BatchResult` 增加 `truncated` 标记
3. `chargebench/experiments.py` — `run_batch` 支持 `deadline_seconds`
4. `tests/test_mcp_server.py` — 42 项，含 **stdio 传输集成测试**（起子进程、按 MCP 协议握手）

**工具集**：`list_algorithms` / `run_batch` / `get_results` / `get_run` /
`compare_experiments` / `parameter_sweep` / `list_batches`

**方案 §7 的六条工程约束逐条落点**

| 约束 | 落点 |
|---|---|
| 参数白名单 | 工具参数由 SDK 按签名校验类型；场景**数据**由 pydantic 模型 `extra="forbid"` 严格校验 |
| 禁止任意 Python 代码执行 | 算法只来自注册表；候选策略只来自 `SortKey` 受控词汇；源码有静态检查禁止 `eval/exec/compile/__import__` |
| 限制批次规模 | `ServerLimits` 对场景数/算法数/种子数/总运行数设上限，**开跑之前**即拒 |
| 超时与取消 | 每次调用带 `deadline_seconds`，超时返回部分结果并标记 `truncated`；客户端请求的时间预算会被服务端上限截断 |
| 结果持久化 | 写入服务端固定的结果库，工具返回 `run_id`/`batch_id` 与 JSON 落盘路径 |
| 错误码标准化 | `ErrorCode` 枚举 + 统一信封 `{ok, data}` / `{ok: false, error: {code, message, ...}}` |

**安全边界（两层，实测确认）**
- **结果库路径由服务端启动参数固定，不作为工具参数暴露** —— 否则客户端就能借工具读写任意路径。
  有测试遍历工具 schema，断言入参里不出现 `store`/`path`/`out` 等字段。
- `config_name` 只接受 `configs/` 下的**文件名主干**，拒绝任何路径分隔符与点开头。
  实测 `../../etc/passwd`、`/etc/passwd`、`.hidden` 全部被拒。
  场景也可以由客户端**内联传入**（经 schema 校验），但不能传文件路径。

**⚠️ mcp 2.x 与 v1 的三处改名**（方案恰好提醒过"避免混用 v1/v2 示例"，实测全中）：
`FastMCP` → **`MCPServer`**、`inputSchema` → **`input_schema`**、`serverInfo` → **`server_info`**。
其中 `mcp.server.fastmcp` 会主动报错并指出改名，但字段级改名只会给出 `AttributeError`。
写法一律以 `mcp>=2,<3` 为准，已写入 `pyproject.toml` 注释。

**一个顺带修掉的设计错误**：`compare_experiments` 最初给所有指标都评了胜者，
包括 `constraint_violations`。但平台的规则是「违规即**取消资格**」，不是「违规少者胜出」——
给违规数排名本身就是误导。已改为：违规候选被排除在胜者评选之外，违规数只作为诊断信息返回。

**验收（已完成）**：用真实 MCP 客户端经 stdio 传输连接服务端，握手成功、列出 7 个工具、
调用 `run_batch` 跑出结果、路径穿越被拒。**先本机调用已完整，远程服务的鉴权与资源隔离
按方案留待后续。**

---

### 阶段 8：收口 ✅ 已完成

**已产出**
1. `scripts/prepare_demo.py` — 一键跑完全部实验并缓存、产出所有图表与演示摘要
2. `docs/DEMO_SCRIPT.md` — 3 分钟演讲稿（按方案 §11 的五段结构），含预判问答与「不要说的话」
3. `README.md` — 原创贡献声明；`tests/test_app.py` 增 3 项守护演示脚本与场景的一致性
4. 全量回归 **275 项**

**演示不依赖现场网络与随机性（机器化证明，非人工承诺）**

```bash
python scripts/prepare_demo.py            # 预跑并缓存：203 条运行 / 20.6s
python scripts/prepare_demo.py --verify   # 复核：抽查执行时刻全部命中缓存 / 12.4s
```

`--verify` 抽查已缓存记录的 **`executed_at`**，而不是比记录条数 ——
重算也会得到同一个 `run_id` 并覆盖同名文件，条数不变，**只看条数区分不出「命中缓存」与「重算」**。
（这是本项目在阶段 3 踩过的同一个坑，复核脚本里差点又犯一次。）

不依赖网络的依据：仿真后端 ACN-Sim 本地运行、场景为合成数据（不使用需要 token 的 ACN-Data）、
AI 闭环内置确定性回退。

**原创贡献声明**（README 顶部与结尾各一处）

明确标注 ACN-Sim 为外部开源基础设施（BSD-3-Clause，Caltech ACN Portal），
本项目仅通过其公开 API 使用、不修改其实现；我们新增的是四层：
统一评测协议、确定性可追溯的实验管理、适用边界的系统扫描、AI 实验闭环。

**验收（已完成）**：演讲稿按 30/40/50/40/20 秒分段，每段标注对应屏幕内容与实际数字；
结论措辞已由 `claims_optimality()` 约束，讲稿末尾列出「不要说的话」并对应到会失败的测试。

---

## 5. 代码组织（✅ 标记为已完成，其余为待建）

```text
charge_bench/
├── README.md                   # ✅ 快速上手
├── PLAN.md                     # ✅ 本文档
├── pyproject.toml              # ✅ 锁 acnportal==0.3.3 + setuptools<81
├── requirements.lock.txt       # ✅ 已验证依赖快照
├── .gitignore                  # ✅
├── docs/
│   └── PROTOCOL.md             # ✅ 输入/输出契约（全队共享）
├── configs/
│   └── demo_scenario.json      # ✅ 已标定到关键区间
├── examples/
│   └── batch--*.json           # ✅ 真实样例结果，UI 可直接用
├── chargebench/
│   ├── schemas.py              # ✅ Scenario / RunResult / BatchResult / Metrics
│   ├── adapter.py              # ✅ ★ 技术核心：场景→网络/事件，含车位分配
│   ├── algorithms.py           # ✅ 算法注册表 + 可见信息声明 + 参数校验
│   ├── metrics.py              # ✅ ★ 指标口径唯一来源
│   ├── tariff.py               # ✅ 人民币分时电价
│   ├── experiments.py          # ✅ run_experiment / run_batch
│   ├── cli.py                  # ✅ 命令行入口
│   ├── storage.py              # ⬜ 阶段 3：SQLite + 原始 JSON
│   ├── sweep.py                # ⬜ 阶段 4：两变量扫描
│   ├── viz.py                  # ⬜ 阶段 4：热力图
│   ├── agent_loop.py           # ⬜ 阶段 6：AI 闭环 + 预算控制
│   └── mcp_server.py           # ⬜ 阶段 7
├── app.py                      # ⬜ 阶段 5：Streamlit
├── scratch/
│   └── acn_e2e_verified.py     # ✅ 环境健康检查（不依赖本项目代码，独立可跑）
├── tests/                      # ✅ 75 项，全绿
│   ├── test_environment.py     #   依赖上界 / 单位契约
│   ├── test_schemas.py         #   取值范围 / 非法组合 / 哈希稳定性
│   ├── test_adapter.py         #   确定性 / 不双重占用 / 车位复用
│   ├── test_metrics.py         #   口径
│   ├── test_reproducibility.py #   同配置可复现 / 跨进程 / 公平性
│   └── test_cli.py             #   协议往返
└── reports/                    # ⬜ 阶段 8
```

---

## 6. 已识别的坑（按危险程度排序）

| # | 坑 | 后果 | 对策 |
|---|---|---|---|
| 1 | 不锁 `setuptools<81` | 包无法导入，比赛当天炸 | 写进 `pyproject.toml`（**已实测确认**） |
| 2 | 照抄 2019 教程的 A·periods 单位 | 指标静默错误，量级差 50 倍 | 以 §2 表为准；`test_adapter` 加单位断言 |
| 3 | 只报总电量/总成本 | 约束瓶颈下所有算法数字相同 → 误判"算法无差异" | 必须并报分布型指标（**已实测确认**） |
| 4 | `proportion_of_demands_met` 的 threshold 当成百分比 | 口径错误，排名颠倒 | threshold 是绝对 kWh；写进 `metrics.py` 文档 |
| 5 | `aggregate_power` 重复除 1000 | 峰值负荷差 1000 倍 | 返回值已是 kW（**已实测确认**） |
| 6 | 用 `_convert_ev_matrix` 默认行为 | 无限车位假设，与真实场景不符 | 自己实现车位分配 + 丢弃计数 |
| 7 | 内置电价是美元 | 成本指标币种错误 | `tariff.py` 自定义人民币分时电价 |
| 8 | ACN-Data 需要 API token | 离线演示失败 | MVP 全用合成场景；不依赖网络 |
| 9 | 网络没有任何约束就启动仿真 | `AttributeError: 'NoneType' object has no attribute 'shape'`（`InfrastructureInfo._validate` 无条件访问 `constraint_matrix.shape`） | Adapter 必须始终至少产出一条约束。如实建模：每桩一条桩级限流约束 + 可选聚合约束（**已实测确认**，见 `tests/test_environment.py`） |

---

## 7. 当前里程碑状态

阶段 0–2 已完成，项目现在是一个**协议化、可复现、有 75 项测试守护的垂直切片**：
场景 JSON 进 → 结果 JSON 出，中间不依赖任何未验证的假设。

```bash
.venv/bin/python -m chargebench.cli run \
    --scenario configs/demo_scenario.json --algorithm FCFS EDF LLF --out examples/
```

```
scenario               algo    seed   satis  complete      kWh      cost    peak  drop
campus_baseline_v1     FCFS      42   0.903     0.564    497.0    467.87   60.00     5
campus_baseline_v1     EDF       42   0.903     0.691    496.6    467.64   60.00     5
campus_baseline_v1     LLF       42   0.913     0.236    502.5    471.74   60.00     5
```

**下一步**：阶段 3（批量运行 + 结果存储）。

批量本身已可用（`run_batch` + CLI 的 `--seeds`），阶段 3 要补的是**持久化与查询**：
`chargebench/storage.py`，把结果落到 SQLite（便于按 `batch_id` 查询）并另存原始 JSON（便于复现）。
判据是「这个结果当时是用什么配置跑出来的」必须能一句话查出来，而不是翻聊天记录。
