"""人民币分时电价。

为什么不用 ACN-Sim 自带的 TimeOfUseTariff：
  1. 它只读包内 JSON，全是美国加州电价（PGE/SCE），金额单位是美元；
  2. 其构造函数要求 tariff_dir 下的文件，接受不了运行时定义的时段。
直接使用会让成本指标币种错误（PLAN.md §6 坑 #7）。

本类只实现 ACN-Sim ``energy_cost(sim, tariff)`` 实际调用的那一个方法
``get_tariffs(start, length, period)``，属于鸭子类型适配，不继承其基类。
"""

from __future__ import annotations

import datetime as dt

from .schemas import PriceKind, PriceProfile

#: 时段标签，供 UI 与调试使用
SEGMENT_VALLEY = "valley"
SEGMENT_FLAT = "flat"
SEGMENT_PEAK = "peak"


def _in_windows(hour: float, windows: list[tuple[float, float]]) -> bool:
    """hour 是否落在任一 [起点, 终点) 区间内。"""
    return any(lo <= hour < hi for lo, hi in windows)


class CNTimeOfUseTariff:
    """按「一天中的小时」划分时段的人民币电价。

    不区分工作日/周末（见 PROTOCOL.md「已知简化」）。窗口跨零点时由 datetime 算术自然处理。
    """

    def __init__(self, profile: PriceProfile):
        self.profile = profile

    def segment_at(self, when: dt.datetime) -> str:
        # 用 == 而非 is 比较：若 profile 未经校验构造（例如 model_copy 绕过了验证），
        # kind 可能是裸字符串而非枚举成员，用 is 会静默失配并退回默认时段逻辑。
        if self.profile.kind == PriceKind.flat:
            return SEGMENT_FLAT
        hour = when.hour + when.minute / 60 + when.second / 3600
        if _in_windows(hour, self.profile.valley_hours):
            return SEGMENT_VALLEY
        if _in_windows(hour, self.profile.peak_hours):
            return SEGMENT_PEAK
        return SEGMENT_FLAT

    def price_at(self, when: dt.datetime) -> float:
        """返回该时刻的电价 [CNY/kWh]。"""
        segment = self.segment_at(when)
        p = self.profile
        return {
            SEGMENT_VALLEY: p.valley_cny_per_kwh,
            SEGMENT_FLAT: p.flat_cny_per_kwh,
            SEGMENT_PEAK: p.peak_cny_per_kwh,
        }[segment]

    # --- ACN-Sim 适配接口 ---------------------------------------------------

    def get_tariffs(
        self, start: dt.datetime, length: int, period: float
    ) -> list[float]:
        """返回从 start 起、连续 length 个时间步（每步 period 分钟）的电价序列。

        与 ``acnportal.acnsim.analysis.energy_cost`` 的期望签名一致：
            energy_cost = Σ price_i * power_i * (period / 60)
        """
        return [
            self.price_at(start + dt.timedelta(minutes=i * period))
            for i in range(length)
        ]

    def segments(
        self, start: dt.datetime, length: int, period: float
    ) -> list[str]:
        """返回每步所处的时段标签，供图表标注峰谷。"""
        return [
            self.segment_at(start + dt.timedelta(minutes=i * period))
            for i in range(length)
        ]


def simulation_start(scenario_start_hour: float, benchmark_date: dt.date) -> dt.datetime:
    """由基准日 + 窗口起始小时构造仿真起点。

    用基准日而非「今天」，是为了让同一种子在任何一天运行都得到完全相同的时间戳与电价。
    """
    whole_hours = int(scenario_start_hour)
    minutes = int(round((scenario_start_hour - whole_hours) * 60))
    return dt.datetime.combine(benchmark_date, dt.time(whole_hours, minutes))
