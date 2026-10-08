# ChargeBench AI

新能源汽车充电调度智能仿真评测平台 — AI-powered EV Charging Scheduling Benchmark & Simulation Lab

> **不开发"唯一最好的充电调度算法"，而是搭建一套让不同算法在统一约束下公平竞争、
> 自动探索适用边界，并由 AI 提出可验证改进建议的实验平台。**

完整方案见 [方案文档](#)，分步实施计划见 [PLAN.md](PLAN.md)，接口契约见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

---

## 快速上手

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/pip install -e ".[dev]"

# 环境自检
.venv/bin/python -m pytest tests/ -q

# 跑一次三算法对比
.venv/bin/python -m chargebench.cli run \
    --scenario configs/demo_scenario.json \
    --algorithm FCFS EDF LLF --out examples/
```

输出：

```
scenario               algo    seed   satis  complete      kWh      cost    peak  drop
campus_baseline_v1     FCFS      42   0.903     0.564    497.0    467.87   60.00     5
campus_baseline_v1     EDF       42   0.903     0.691    496.6    467.64   60.00     5
campus_baseline_v1     LLF       42   0.913     0.236    502.5    471.74   60.00     5
```

注意这一组数字说明了平台存在的理由：**总交付电量三者几乎相同（差异 < 2%），
但按时完成率相差 0.45**。当网络功率约束成为瓶颈时，差异只体现在"这些电怎么分配"。
只报总电量会得出"算法无差异"的错误结论。

---

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境固化（锁依赖、契约测试） | ✅ 已完成 |
| 1 | 输入/输出协议 | ✅ 已完成 |
| 2 | Adapter + 单次运行 + 指标 | ✅ 已完成 |
| 3 | 批量运行 + 结果存储 | ✅ 已完成 |
| 4 | 两变量扫描 + 适用边界热力图 | ✅ 已完成 |
| 5 | Streamlit 仪表板 | ✅ 已完成 |
| 6 | AI 实验闭环 | ✅ 已完成 |
| 7 | MCP Server | ✅ 已完成 |
| 8 | 收口（报告 / 演示 / 回归） | ✅ 已完成 |

测试：**278 项全绿**。演示脚本见 [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)。

### 结果库

```bash
# 落盘并复用已算结果（重复运行不会重复计算）
python -m chargebench.cli run --scenario configs/demo_scenario.json \
    --algorithm FCFS EDF LLF --seeds 42 43 44 --store results/

python -m chargebench.cli results  --store results/
python -m chargebench.cli batches  --store results/
python -m chargebench.cli describe <run_id> --store results/   # 这个结果当时怎么跑出来的
```

跨 3 个种子的按时完成率——注意 LLF 不只均值更差，波动还大 4 倍，
单次运行完全看不出这一点：

| 算法 | 均值 | 最小 | 最大 | 波动 |
|---|---|---|---|---|
| EDF | 0.691 | 0.650 | 0.732 | 0.082 |
| FCFS | 0.620 | 0.564 | 0.679 | 0.115 |
| LLF | 0.431 | 0.236 | 0.607 | **0.371** |

### 仪表板

```bash
.venv/bin/streamlit run app.py
```

四个标签页：**批量对比**（指标表 + 跨种子波动）、**过程回放**（负荷曲线 + 逐站点时间线）、
**适用边界**（两变量扫描 + 换目标对比）、**结果库**（查询 + 追溯）。
界面不做任何自定义计算，数字与命令行逐字一致。

### 一键准备演示

```bash
python scripts/prepare_demo.py            # 跑完全部实验并缓存 + 产出所有图表
python scripts/prepare_demo.py --verify   # 复核：重跑应全部命中缓存、数字逐位一致
```

现场演示最怕随机性让数字变了。这个脚本固定种子预跑一遍，`--verify` 会抽查已缓存记录的
**执行时刻**（只看记录条数区分不出"命中缓存"与"重算覆盖同名文件"），
确认现场数字与预跑逐位相同。**本项目不依赖任何网络服务**，断网也能完整演示。

### MCP Server

```bash
.venv/bin/pip install -e ".[mcp]"
python -m chargebench.mcp_server --store results --configs configs
```

7 个工具：`list_algorithms` / `run_batch` / `get_results` / `get_run` /
`compare_experiments` / `parameter_sweep` / `list_batches`。

服务端的硬约束：结果库路径不由客户端控制、算法只来自注册表、候选策略只来自受控词汇、
批次规模与时间预算都有上限、错误码标准化。`compare_experiments` **不返回单一排名** ——
不同指标可能给出不同胜者，这正是平台要说明的事实。

> **mcp 2.x 与 v1 的改名**：`FastMCP`→`MCPServer`、`inputSchema`→`input_schema`、
> `serverInfo`→`server_info`。混用会直接报 `AttributeError`。

### AI 实验闭环

```bash
python -m chargebench.cli agent --scenario configs/demo_scenario.json \
    --seeds 42 43 --rounds 3 --train-variants \
    --store results/ --plot reports/ --out examples/
```

闭环会提出候选策略 → 调用仿真评测 → 在**未参与筛选的保留场景**上复验。实测 77 次仿真、
7.6 秒内找到的候选把按时完成率从 EDF 的 0.7874 提到 **0.9205**，保留场景复验通过
（0.9041 对 0.6806），且种子波动更小。

**但结论会同时告诉你代价**：该候选的能量口径满足率从 0.9419 降到 0.9291 ——
它是把电量集中喂给「能喂饱」的车。平台不把有取舍的改进说成全面更优。

LLM 只能从受控的排序键词汇里组合策略，不能生成代码（`--llm-command` 可接入外部模型，
不可用时自动回退到内置启发式）。

### 适用边界扫描

```bash
python -m chargebench.cli sweep --scenario configs/demo_scenario.json \
    --x 0.5 0.7 0.9 1.1 1.3 --y 0.40 0.55 0.70 0.85 \
    --seeds 42 43 --metric session_completion_rate \
    --compare peak_power_kw energy_cost_cny \
    --store results/ --plot reports/ --detail
```

产出三张图到 `reports/`：算法适用区域图、逐算法指标小图、**目标对比图**。

核心结论——**同一批实验数据，换一个优化目标，胜者就变了**：

| 目标函数 | 胜者分布（5×4 网格） |
|---|---|
| 按时完成率 | EDF 17 / LLF 2 / FCFS 1 |
| 峰值负荷 | FCFS 7 / LLF 7 / EDF 6 |
| 总用电成本 | EDF 11 / FCFS 6 / LLF 3 |

即「哪个算法最好」没有唯一答案，取决于你把什么当目标。`--compare` 出来的多个结果
共享同一份 `run_ids`——因为换目标只是换把尺子，不需要重跑任何仿真。

---

## 原创贡献声明

仿真后端为外部开源项目 **ACN-Sim**（BSD-3-Clause，Caltech ACN Portal）。
本项目仅通过其公开 API 使用，**不修改其实现**，也不把 ACN-Sim 已有的实验伪装成原创研究。

我们新增的是四层：

1. **统一评测协议** —— 场景与结果用 JSON 完整描述，`run_id` 确定性导出，
   任何一次实验都能复现、比对、分享（`docs/PROTOCOL.md`）。
2. **确定性可追溯的实验管理** —— 同配置在任何机器上得到同一 `run_id`；
   写入时强制校验可复现性，出现非确定性立刻报错而非静默覆盖（`chargebench/storage.py`）。
3. **适用边界的系统扫描** —— 两变量网格扫描 + 事前写定的评分规则 +
   换目标重排名（不重跑仿真），回答「哪个算法在什么条件下胜出」（`chargebench/sweep.py`）。
4. **AI 实验闭环** —— 提出假设 → 调用仿真 → 分析 → 保留场景复验，
   带四重预算闸门与措辞纪律；LLM 只能在受控词汇内组合策略，不能生成代码
   （`chargebench/agent_loop.py`）。

## 目录

```
configs/      场景配置
docs/         协议与设计文档
examples/     真实样例结果（UI 可直接消费）
chargebench/  核心包
scratch/      环境健康检查（不依赖本项目代码，可独立运行）
tests/        测试
```

## 依赖说明

仿真后端为外部开源项目 **ACN-Sim**（BSD-3-Clause，Caltech ACN Portal）。
本项目仅通过其公开 API 使用，不修改其实现；我们新增的是统一评测协议、边界探索与 Agent 闭环。

两个必须注意的约束（详见 [PLAN.md](PLAN.md) §1、§2）：

1. **`setuptools` 必须 `<81`** —— `acnportal` 依赖已被移除的 `pkg_resources`，否则无法导入。
2. **单位是 kWh / kW**（不是 2019 年教程里的 A·periods）—— 照抄教程会静默产生约 50 倍的指标误差。
