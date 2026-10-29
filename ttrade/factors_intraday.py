"""日内因子：只有分钟级行情才算得出来的一类。

## 为什么单开一类

日线上「当天」只有一根 bar，`open` 和 `close` 之间发生了什么完全看不到：是
开盘冲高一路走弱，还是低开翻红收在最高，日线只能给出一个净涨跌。日内因子要的
正是这段被抹掉的过程 —— 现价相对当日成交均价贵不贵（VWAP 偏离）、成交量堆在
高价区还是低价区（量价位置）、当天走了多少、隔夜跳空吃掉多少。

## 日线上这些因子必须是 NaN

`_intraday()` 判断「平均每天至少两根 bar」，不满足就整列返回 NaN。这不是偷懒：
日线上 `close/open-1` 就是当日涨跌幅，`(h+l+c)/3` 当 VWAP 也不成立（没有日内
成交分布）。硬算出来的数会进因子筛选、还会被当成「有效因子」参与合成，比留空
更糟。`factors.filter_available()` 会自动把这类因子从日线标的的候选里剔掉。

## PIT

每个量都只用「当前 bar 及之前」的信息：`cumsum` / `cummax` / `cummin` 全部按
交易日分组向后累计，`overnight_gap` 用的是上一交易日的收盘。没有任何前视。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 基础工具
def _nan(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index, dtype=float)


def _day(df: pd.DataFrame) -> pd.Series:
    """每根 bar 属于哪个交易日。

    天勤数据带 `trading_day` 列（夜盘 21:00 的 bar 归到**下一个**交易日，这是
    交易所口径，不能拿自然日代替）；Tushare 日线没有这列，退回索引的日期。
    """
    if "trading_day" in df.columns:
        d = pd.to_datetime(df["trading_day"], errors="coerce")
        if d.notna().any():
            return d
    return pd.Series(pd.DatetimeIndex(df.index).normalize(), index=df.index)


def _intraday(df: pd.DataFrame) -> bool:
    """是不是分钟级数据：平均每个交易日至少两根 bar。"""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 20:
        return False
    return len(idx) >= 2 * idx.normalize().nunique()


def _hhmm(df: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(df.index).strftime("%H:%M"), index=df.index)


def _to_day(df: pd.DataFrame, s: pd.Series, how: str = "last") -> pd.Series:
    """把 bar 级序列压成日频，再铺回每根 bar（同一交易日内取值相同）。"""
    g = _day(df)
    return pd.Series(g.map(s.groupby(g).agg(how)).to_numpy(float), index=df.index)


def _window_ret(df: pd.DataFrame, t0: str, t1: str, min_bars: int = 1) -> pd.Series:
    """每个交易日 [t0, t1] 这段的收益：窗口内第一根开盘 → 最后一根收盘。

    按「收盘时刻」落进窗口，所以 5m 的 09:45 那根算在 09:45-09:50；1h 的窗口
    可能只圈到一根 bar，退化成那根自己的涨跌 —— 精度下降但方向不错，比直接
    返回 NaN 有用。
    """
    if not _intraday(df):
        return _nan(df)
    hh = _hhmm(df)
    m = (hh >= t0) & (hh <= t1)
    if not m.any():
        return _nan(df)
    g = _day(df)
    sub = df.loc[m]
    o = pd.to_numeric(sub["open"], errors="coerce").groupby(g[m]).first()
    c = pd.to_numeric(sub["close"], errors="coerce").groupby(g[m]).last()
    r = c / o - 1.0
    # 窗口里当天只有一根 bar 的，说明这个周期切不出这么细的窗口（1h 的 bar 本身
    # 就是一小时），这时候「窗口收益」等于「这根 bar 自己的涨跌」，挂着时段的名
    # 字卖别的货，不如留空
    n = sub.groupby(g[m]).size().reindex(r.index)
    r = r.where(n >= max(1, int(min_bars)))
    return pd.Series(g.map(r).to_numpy(float), index=df.index)


# ---------------------------------------------------------------- 日内
def f_vwap_gap(df: pd.DataFrame) -> pd.Series:
    """现价相对当日 VWAP 的偏离。

    VWAP 用典型价 (H+L+C)/3 按成交量加权累计 —— 库里没有成交额，典型价比收盘价
    更接近真实成交重心。持续在 VWAP 之上说明买盘愿意追价，是日内多头占优的直接
    证据；日线上这个量恒等于「收盘相对典型价」，没有信息。
    """
    if not _intraday(df):
        return _nan(df)
    g = _day(df)
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * v).groupby(g).cumsum()
    vv = v.groupby(g).cumsum()
    vwap = pv / vv.replace(0.0, np.nan)
    return pd.to_numeric(df["close"], errors="coerce") / vwap - 1.0


def f_pv_pos(df: pd.DataFrame) -> pd.Series:
    """量价位置：成交量加权的「收盘在当日振幅中的位置」，减去 0.5。

    把成交量当权重，等于问「今天堆的量大头发生在高价区还是低价区」。放量在
    高位（>0）说明有承接、放量在低位（<0）说明在派发 —— 比单纯的收盘位置多了
    一层量的确认，也是日内「量价配合」最直接的度量。
    """
    if not _intraday(df):
        return _nan(df)
    g = _day(df)
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    pos = ((df["close"] - df["low"]) / rng).clip(0.0, 1.0)
    num = (pos * v).groupby(g).cumsum()
    den = v.groupby(g).cumsum()
    return num / den.replace(0.0, np.nan) - 0.5


def f_range_break(df: pd.DataFrame) -> pd.Series:
    """收盘相对当日开盘的位移 / 当日已实现振幅。

    取值在 -1~1 之间：+1 是「当天最高点收盘」，-1 是「最低点收盘」。趋势日会
    顶到两端，震荡日贴在 0 附近。比 `close_pos`（10 日均值）更即时。
    """
    if not _intraday(df):
        return _nan(df)
    g = _day(df)
    c = pd.to_numeric(df["close"], errors="coerce")
    o0 = pd.to_numeric(df["open"], errors="coerce").groupby(g).transform("first")
    hi = pd.to_numeric(df["high"], errors="coerce").groupby(g).cummax()
    lo = pd.to_numeric(df["low"], errors="coerce").groupby(g).cummin()
    return (c - o0) / (hi - lo).replace(0.0, np.nan)


def f_intraday_mom(df: pd.DataFrame) -> pd.Series:
    """日内动量：现价 / 当日开盘 - 1，把隔夜跳空剔掉。

    日线的 `mom_20` 把隔夜和日内揉在一起。隔夜跳空多半是外盘和消息驱动，日内
    才是本国资金的态度；两者分开看，能区分「外盘带的」和「自己走的」。
    """
    if not _intraday(df):
        return _nan(df)
    g = _day(df)
    o0 = pd.to_numeric(df["open"], errors="coerce").groupby(g).transform("first")
    return pd.to_numeric(df["close"], errors="coerce") / o0 - 1.0


def f_overnight_gap(df: pd.DataFrame) -> pd.Series:
    """隔夜跳空：当日开盘 / 上一交易日收盘 - 1。

    国内夜盘 21:00 就开盘，所谓「隔夜」实际是上一交易日 15:00 到当晚 21:00 这段
    的缺口，反映的是外盘与消息。连续同向跳空往往意味着有资金在抢跑。
    """
    if not _intraday(df):
        return _nan(df)
    g = _day(df)
    c = pd.to_numeric(df["close"], errors="coerce")
    dc = c.groupby(g).last()
    prev = dc.shift(1)
    o0 = pd.to_numeric(df["open"], errors="coerce").groupby(g).transform("first")
    return pd.Series(g.map(o0.groupby(g).first() / prev - 1.0).to_numpy(float),
                     index=df.index)


def f_close_drive(df: pd.DataFrame) -> pd.Series:
    """尾盘驱动：当日最后一个小时的收益 / 当日振幅。

    收盘前这段时间是资金定调的时候（要留仓的、要平仓的都在这个窗口动手）。尾盘
    拉升且幅度占全天振幅比例高，说明有人愿意持仓过夜；反之是尾盘砸盘出货。
    用「最后一小时」而不是半小时，是为了 1h 周期也圈得住（5m 上同理可用）。
    """
    if not _intraday(df):
        return _nan(df)
    last = _window_ret(df, "14:00", "15:00")
    g = _day(df)
    hi = pd.to_numeric(df["high"], errors="coerce").groupby(g).transform("max")
    lo = pd.to_numeric(df["low"], errors="coerce").groupby(g).transform("min")
    return last / ((hi - lo) / pd.to_numeric(df["close"], errors="coerce")).replace(
        0.0, np.nan)


# ---------------------------------------------------------------- 时段
def f_trend_window(df: pd.DataFrame) -> pd.Series:
    """活跃时段收益：09:45–10:15。

    开盘头半小时是集合竞价的余波，噪音大；09:45 之后到 10:15 这段是当天第一波
    真正的方向选择，隔夜的挂单已经消化完、日内资金开始表态。这段的方向对全天
    的延续性最好，所以归「利于趋势」的一类。
    """
    return _window_ret(df, "09:45", "10:15", min_bars=3)


def f_quiet_window(df: pd.DataFrame) -> pd.Series:
    """非活跃时段收益：13:30–14:00，取负。

    午后开盘这段是一天里最清淡的窗口（11:30 收盘到 13:30 开盘，外盘也没新消息），
    这个时段的单边推动多半是流动性不足造成的过度反应，随后容易回补。所以取负
    号，让「高分 = 预期反转」。
    """
    return -_window_ret(df, "13:30", "14:00", min_bars=3)


def f_first_hour(df: pd.DataFrame) -> pd.Series:
    """开盘首小时收益：09:00–10:00。日内的「隔夜信息消化」窗口。"""
    return _window_ret(df, "09:00", "10:00")


def f_night_ret(df: pd.DataFrame) -> pd.Series:
    """夜盘收益：21:00–23:00。外盘驱动的时段，和日盘是两拨资金。"""
    return _window_ret(df, "21:00", "23:00")


def f_session_spread(df: pd.DataFrame) -> pd.Series:
    """日夜盘强弱差：夜盘收益 − 日盘收益（09:00–15:00）。

    两段由不同资金主导，谁在主导往往决定第二天开盘的偏向：夜盘强、日盘弱，
    说明外盘在带、国内在抛，次日容易低开回补。
    """
    return _window_ret(df, "21:00", "23:00") - _window_ret(df, "09:00", "15:00")


# ---------------------------------------------------------------- 注册
def register_into(FACTORS, FactorSpec) -> int:
    """把这一类因子挂进主库。由 `factors.py` 在末尾调用。"""
    specs = [
        ("vwap_gap", "VWAP 偏离", f_vwap_gap, "日内",
         "现价相对当日成交量加权均价（典型价）的偏离，日内多头占优的直接度量。"),
        ("pv_pos", "量价位置", f_pv_pos, "日内",
         "成交量加权的收盘位置减 0.5：量大头在高价区为正、在低价区为负。"),
        ("range_break", "日内区间突破", f_range_break, "日内",
         "收盘相对开盘的位移 / 当日已实现振幅，±1 表示收在最高/最低。"),
        ("intraday_mom", "日内动量", f_intraday_mom, "日内",
         "现价 / 当日开盘 - 1，剔掉隔夜跳空后的日内方向。"),
        ("overnight_gap", "隔夜跳空", f_overnight_gap, "日内",
         "当日开盘 / 上一交易日收盘 - 1，外盘与消息驱动的缺口。"),
        ("close_drive", "尾盘驱动", f_close_drive, "日内",
         "最后一个小时收益占当日振幅的比例，持仓意愿的度量。"),
        ("trend_window", "活跃时段 09:45-10:15", f_trend_window, "时段",
         "开盘噪音过后的第一波方向选择，对全天延续性最好。"),
        ("quiet_window", "非活跃时段反转 13:30-14:00", f_quiet_window, "时段",
         "午后最清淡的半小时，取负：清淡时段的推动多半过度反应。", -1),
        ("first_hour", "开盘首小时", f_first_hour, "时段",
         "09:00-10:00 收益，隔夜信息的消化窗口。"),
        ("night_ret", "夜盘时段 21:00-23:00", f_night_ret, "时段",
         "外盘驱动的时段，与日盘是两拨资金。"),
        ("session_spread", "日夜盘强弱差", f_session_spread, "时段",
         "夜盘收益减日盘收益，外盘主导还是国内主导。"),
    ]
    n = 0
    for item in specs:
        key, label, func, cat, desc = item[:5]
        hint = item[5] if len(item) > 5 else 1
        FACTORS[key] = FactorSpec(key, label, func, cat, desc, hint)
        n += 1
    return n
