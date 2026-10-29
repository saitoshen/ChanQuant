"""季节性因子：把「月份」当成一个横截面变量。

## 两类东西，别混在一起

1. **统计季节性**（`season_*`）：直接从该品种自己的历史里算 —— 过去若干年同一
   个月的平均收益、这个平均值的稳定性、当月已经走了多少 vs 历史同月该走多少。
   纯数据驱动，不看品种是什么，也不需要任何外部知识。

2. **基本面日历**（`season_cal*`）：农产品看收割/播种、工业品看开工/取暖、软商品
   看榨季/停割。这些窗口来自产业常识而不是价格历史，写死在一张表里（见
   `_CALENDAR`），品种对不上就返回 0 —— 宁可没有信号，也不要拿别的品种的日历
   硬套。

## PIT 怎么守

统计季节性最容易犯的错是「用全样本的月度均值」。这里一律用**严格往年**：
第 Y 年 M 月的均值只由 <Y 年的 M 月算出（`expanding().mean().shift(1)`），
不足 3 年给 NaN。`shift(1)` 是按年位移，不是按 bar —— 当年还没走完就拿来算，
等于提前知道这个月会怎么走。

`season_gap` 用的是「本月至今」，所以还要按已过交易日折算历史均值，不能拿整月
的均值去比半个月的涨幅。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_YEARS = 3        # 至少 3 个完整年度才给值
DAYS_PER_MONTH = 21  # 折算「本月至今」用


# ---------------------------------------------------------------- 基础工具
def _nan(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index, dtype=float)


def _day(df: pd.DataFrame) -> pd.Series:
    if "trading_day" in df.columns:
        d = pd.to_datetime(df["trading_day"], errors="coerce")
        if d.notna().any():
            return d
    return pd.Series(pd.DatetimeIndex(df.index).normalize(), index=df.index)


def _daily_close(df: pd.DataFrame) -> pd.Series:
    """压成日频收盘。日线本来一天一根，这一步是恒等变换。"""
    g = _day(df)
    c = pd.to_numeric(df["close"], errors="coerce")
    s = c.groupby(g).last()
    s.index = pd.DatetimeIndex(s.index)
    return s


def _month_table(dc: pd.Series) -> pd.DataFrame:
    """年 × 月的收益表：行是年份、列是月份，值是当月累计收益。"""
    r = dc.pct_change()
    idx = pd.DatetimeIndex(dc.index)
    return r.groupby([idx.year, idx.month]).sum().unstack()


def _prior(table: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """每个 (年, 月) 的「往年同月」统计量，严格不含当年。"""
    exp = table.expanding(min_periods=1)
    val = exp.mean() if how == "mean" else exp.std(ddof=1)
    val = val.shift(1)
    n = table.notna().cumsum().shift(1)
    return val.where(n >= MIN_YEARS)


def _map_back(df: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """把年×月表映射回每根 bar（同一交易月内取值相同）。"""
    day = _day(df)
    idx = pd.DatetimeIndex(day)
    key = pd.MultiIndex.from_arrays([idx.year, idx.month])
    vals = table.stack().reindex(key)
    return pd.Series(vals.to_numpy(dtype=float), index=df.index)


def _map_daily(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    g = _day(df)
    return pd.Series(g.map(s).to_numpy(dtype=float), index=df.index)


# ---------------------------------------------------------------- 统计季节性
def f_season_avg(df: pd.DataFrame) -> pd.Series:
    """当月的历史平均月收益（过去 ≥3 年，不含当年）。

    这是「平均月度曲线」在那一天的值。正值说明这个品种在这个月历史上偏强，
    比如螺纹 3-4 月、豆粕 3-5 月南美天气窗口。
    """
    dc = _daily_close(df)
    if len(dc) < 250:
        return _nan(df)
    return _map_back(df, _prior(_month_table(dc), "mean"))


def f_season_ir(df: pd.DataFrame) -> pd.Series:
    """季节性的稳定性：历史同月平均收益 / 该月历史标准差。

    同样的 +2% 均值，有的是年年如此，有的是某一年暴涨拉起来的。用标准差一除
    就能分开 —— 只有稳定重现的才配叫季节性，一次性行情叫趋势。
    """
    dc = _daily_close(df)
    if len(dc) < 250:
        return _nan(df)
    tab = _month_table(dc)
    mean = _prior(tab, "mean")
    sd = _prior(tab, "std")
    return _map_back(df, mean / sd.replace(0.0, np.nan))


def f_season_rank(df: pd.DataFrame) -> pd.Series:
    """当月历史均值在 12 个月里的分位（-0.5 ~ 0.5）。

    跨品种可比：铜的 3 月均值可能是 +1.2%、豆粕是 +3%，绝对值没法比，但在各自
    12 个月里的排位是可比的。0.4 表示「这是本品种一年里最好的两个月之一」。
    """
    dc = _daily_close(df)
    if len(dc) < 250:
        return _nan(df)
    prior = _prior(_month_table(dc), "mean")
    return _map_back(df, prior.rank(axis=1, pct=True) - 0.5)


def f_season_gap(df: pd.DataFrame) -> pd.Series:
    """当月已走出的收益 − 历史同月该走的部分。

    本月至今涨了 3%，而这个月历史上平均只涨 1%、已经过了三分之二，那就是
    「季节性该给的已经给完了还在涨」—— 超额部分要么是基本面新信息，要么是透支。
    按已过交易日折算，不拿整月均值去比半个月。
    """
    dc = _daily_close(df)
    if len(dc) < 250:
        return _nan(df)
    avg = _prior(_month_table(dc), "mean")
    mkey = pd.DatetimeIndex(dc.index).to_period("M")
    m0 = dc.groupby(mkey).transform("first")
    mtd = dc / m0 - 1.0
    elapsed = pd.Series(mkey, index=dc.index).groupby(mkey).cumcount() + 1
    frac = (elapsed / float(DAYS_PER_MONTH)).clip(0.0, 1.0)
    avg_bar = _map_back(df, avg)          # 先铺到 bar，再按 bar 所在月的进度折算
    frac_bar = _map_daily(df, frac)
    return _map_daily(df, mtd) - avg_bar * frac_bar


# ---------------------------------------------------------------- 基本面日历
# 品种 → [(起始月, 结束月, 方向)]，跨年窗口写两段（比如 11-1 月写成 11-12 和 1-1）。
# 方向 +1 = 产业旺季/利多窗口，-1 = 淡季/利空窗口。
#
# 这张表是**产业常识**，不是从价格拟合出来的：螺纹金三银四、九银十的开工旺季；
# 豆粕 9-11 月北美收割压力、3-5 月南美天气炒作；白糖 10-12 月开榨、6-8 月消费
# 旺季；天胶 11-2 月停割；原油 6-8 月驾驶季。写死是有意的 —— 让模型或数据自己
# 找「哪个月涨得多」只会找到过拟合的月份。
_CALENDAR: dict[str, list[tuple[int, int, int]]] = {
    "RB": [(3, 5, 1), (9, 11, 1), (1, 2, -1), (6, 7, -1)],
    "HC": [(3, 5, 1), (9, 11, 1), (1, 2, -1), (6, 7, -1)],
    "I": [(3, 5, 1), (9, 11, 1), (6, 7, -1)],
    "J": [(11, 12, 1), (1, 1, 1), (5, 7, -1)],
    "JM": [(11, 12, 1), (1, 1, 1), (5, 7, -1)],
    "CU": [(3, 5, 1), (11, 12, -1)],
    "AL": [(3, 5, 1), (11, 12, -1)],
    "ZN": [(3, 5, 1), (11, 12, -1)],
    "AU": [(12, 12, 1), (1, 2, 1), (6, 8, -1)],
    "AG": [(12, 12, 1), (1, 2, 1), (6, 8, -1)],
    "SC": [(6, 8, 1), (1, 3, -1)],
    "M": [(3, 5, 1), (9, 11, -1)],
    "Y": [(1, 3, 1), (9, 11, -1)],
    "P": [(1, 3, 1), (9, 11, -1)],
    "A": [(3, 5, 1), (9, 11, -1)],
    "C": [(4, 6, 1), (9, 11, -1)],
    "SR": [(6, 8, 1), (10, 12, -1)],
    "CF": [(9, 11, 1), (4, 6, -1)],
    "AP": [(3, 5, 1), (10, 11, -1)],
    "JD": [(7, 9, 1), (2, 4, -1)],
    "RU": [(11, 12, 1), (1, 2, 1), (5, 8, -1)],
    "TA": [(3, 5, 1), (9, 10, 1)],
    "EG": [(3, 5, 1), (9, 10, 1)],
    "MA": [(9, 11, 1), (3, 5, -1)],
    "FG": [(4, 5, 1), (9, 10, 1), (1, 2, -1)],
    "SA": [(6, 8, 1), (1, 2, -1)],
    "PP": [(3, 5, 1), (11, 12, -1)],
    "L": [(3, 5, 1), (11, 12, -1)],
    "V": [(3, 5, 1), (11, 12, -1)],
    "OI": [(6, 8, 1), (10, 12, -1)],
    "RM": [(6, 9, 1), (11, 12, -1), (1, 1, -1)],
    "SF": [(3, 5, 1), (11, 12, -1)],
    "SM": [(3, 5, 1), (11, 12, -1)],
}


def _symbol(df: pd.DataFrame) -> str:
    """品种代码。加载器把它挂在 `df.attrs["symbol"]` 上；取不到就返回空串。"""
    sym = ""
    try:
        sym = str(df.attrs.get("symbol") or "")
    except Exception:  # noqa: BLE001
        sym = ""
    if not sym and "symbol" in df.columns and len(df):
        sym = str(df["symbol"].iloc[0])
    return sym.upper()


def f_season_cal(df: pd.DataFrame) -> pd.Series:
    """品种基本面季节窗口：+1 旺季 / -1 淡季 / 0 无窗口（表里没这个品种则 NaN）。

    只在 `_CALENDAR` 里有记录的品种上非零。别把它当方向信号用 ——
    它的价值在**当条件**：同样一个突破，发生在金三银四和发生在六月，胜率不一样。
    """
    sym = _symbol(df)
    wins = _CALENDAR.get(sym)
    if not wins:
        # 表里没有这个品种就返回 NaN 而不是 0：0 会被当成「有值、只是没窗口」，
        # 整列常数过一遍 z-score 变 NaN，白白占一个因子位还让人以为能选
        return _nan(df)
    month = pd.DatetimeIndex(_day(df)).month
    out = np.zeros(len(df), dtype=float)
    for m0, m1, sign in wins:
        out[(month >= m0) & (month <= m1)] = float(sign)
    return pd.Series(out, index=df.index)


def f_season_cal_mom(df: pd.DataFrame) -> pd.Series:
    """季节窗口 × 20 日动量：只在产业旺季做多趋势、淡季做空趋势。

    动量单用的问题是「趋势在淡季也照样延续」，把季节当门控（而不是加成）能砍掉
    一半无效信号。窗口外的日子直接 0，不参与打分。
    """
    cal = f_season_cal(df)
    c = pd.to_numeric(df["close"], errors="coerce")
    mom = c / c.shift(20) - 1.0
    return cal * mom


# ---------------------------------------------------------------- 注册
def register_into(FACTORS, FactorSpec) -> int:
    specs = [
        ("season_avg", "季节均值", f_season_avg, "季节性",
         "过去 3 年以上同月平均收益，即平均月度曲线在当月的值。"),
        ("season_ir", "季节稳定性", f_season_ir, "季节性",
         "同月历史均值 / 标准差，区分「年年如此」和「某一年拉起来的」。"),
        ("season_rank", "季节分位", f_season_rank, "季节性",
         "当月均值在 12 个月中的分位，跨品种可比。"),
        ("season_gap", "季节偏离", f_season_gap, "季节性",
         "本月至今收益减去历史同月已该走的部分，超额部分是新信息或透支。"),
        ("season_cal", "品种季节窗口", f_season_cal, "品种季节",
         "产业日历：+1 旺季 / -1 淡季 / 0 无窗口，仅覆盖有基本面季节的品种。"),
        ("season_cal_mom", "季节窗口×动量", f_season_cal_mom, "品种季节",
         "季节窗口当门控的 20 日动量：旺季做多趋势、淡季做空趋势。"),
    ]
    n = 0
    for key, label, func, cat, desc in specs:
        FACTORS[key] = FactorSpec(key, label, func, cat, desc)
        n += 1
    return n
