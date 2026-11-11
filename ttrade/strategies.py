"""趋势跟踪策略集合。

约定：每个策略函数返回带 `signal` 列的 DataFrame。
signal 表示「当日收盘后确定的目标仓位」：
    1  = 满仓做多
    0  = 空仓
    -1 = 满仓做空
真正的成交由回测引擎在下一交易日开盘执行，因此不存在未来函数。

期货天然双向，因此策略默认给出对称信号（多/空）。若只想做多，
由回测引擎的 `BacktestConfig.long_only` 把 -1 裁成 0，策略本身不做区分。
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 基础指标
def sma(s: pd.Series, n: int) -> pd.Series:
    n = int(n)
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    n = int(n)
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder 平滑 ATR。"""
    n = int(n)
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _position_state(
    long_in: pd.Series,
    long_out: pd.Series,
    short_in: pd.Series | None = None,
    short_out: pd.Series | None = None,
) -> np.ndarray:
    """事件驱动的持仓状态机：1=多，-1=空，0=平。

    处理顺序：先判断离场（含「反向入场」触发的离场），再判断入场，
    因此「多头 → 空头」的反手可以在同一根 K 线上完成。
    """
    li = long_in.fillna(False).to_numpy(dtype=bool)
    lo = long_out.fillna(False).to_numpy(dtype=bool)
    n = len(li)
    if short_in is None:
        si = np.zeros(n, dtype=bool)
    else:
        si = short_in.fillna(False).to_numpy(dtype=bool)
    if short_out is None:
        so = np.zeros(n, dtype=bool)
    else:
        so = short_out.fillna(False).to_numpy(dtype=bool)

    pos = np.zeros(n, dtype=int)
    cur = 0
    for i in range(n):
        if cur == 1 and (lo[i] or si[i]):
            cur = 0
        elif cur == -1 and (so[i] or li[i]):
            cur = 0
        if cur == 0:
            if li[i]:
                cur = 1
            elif si[i]:
                cur = -1
        pos[i] = cur
    return pos


# ---------------------------------------------------------------- 基准策略
# 买入持有 / 卖出持有是**基准**，不是 alpha 来源。它们的作用是给其它策略
# 一个「什么都不做」的对照：一个趋势策略如果跑不赢同品种的买入持有，
# 那它赚的是 beta 而不是择时，加杠杆只会放大回撤。
def strat_buy_hold(df: pd.DataFrame) -> pd.DataFrame:
    """买入持有：第一根 K 线就满仓做多，之后一直不动。

    不看任何指标，也不设离场 —— 它度量的是「这个品种自己涨了多少」。
    回测引擎在下一根开盘成交，所以第一根的 signal=1 对应第二根开盘建仓。
    """
    out = df.copy()
    out["signal"] = np.ones(len(out), dtype=int)
    return out


def strat_sell_hold(df: pd.DataFrame) -> pd.DataFrame:
    """卖出持有：第一根 K 线就满仓做空，之后一直不动。

    商品期货天然可做空，所以「一直做空」是一个合法的基准。它等于买入持有
    的镜像（扣掉资金成本之前，收益正好相反），用来判断一个多头策略是不是
    只是「在下跌品种上做反了」。
    """
    out = df.copy()
    out["signal"] = -np.ones(len(out), dtype=int)
    return out


# ---------------------------------------------------------------- 策略实现
def strat_ma_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """双均线：快线在慢线上方做多，下方做空。对称反手，始终在市。"""
    out = df.copy()
    f, s = sma(out["close"], fast), sma(out["close"], slow)
    out["fast_ma"], out["slow_ma"] = f, s
    sig = pd.Series(0, index=out.index, dtype=int)
    valid = s.notna() & f.notna()
    sig[valid] = np.where(f[valid] > s[valid], 1, -1)
    out["signal"] = sig
    return out


def strat_donchian(df: pd.DataFrame, entry: int = 55, exit: int = 20) -> pd.DataFrame:
    """唐奇安通道突破（海龟法则核心）：双向。

    做多：突破 N 日新高；做空：跌破 N 日新低；
    离场：反向跌破/突破 M 日通道。
    """
    out = df.copy()
    entry, exit = int(entry), int(exit)
    up_e = out["high"].rolling(entry, min_periods=entry).max().shift(1)
    dn_e = out["low"].rolling(entry, min_periods=entry).min().shift(1)
    up_x = out["high"].rolling(exit, min_periods=exit).max().shift(1)
    dn_x = out["low"].rolling(exit, min_periods=exit).min().shift(1)
    out["upper"], out["lower"] = up_e, dn_e
    out["exit_up"], out["exit_dn"] = up_x, dn_x

    close = out["close"]
    out["signal"] = _position_state(
        long_in=close > up_e,
        long_out=close < dn_x,
        short_in=close < dn_e,
        short_out=close > up_x,
    )
    return out


def strat_atr_channel(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """ATR 通道（Keltner 变体）：双向。

    做多：收盘 > 均线 + k×ATR；做空：收盘 < 均线 - k×ATR；
    离场：回到均线另一侧。
    """
    out = df.copy()
    mid = sma(out["close"], n)
    a = atr(out, n)
    upper, lower = mid + k * a, mid - k * a
    out["mid"], out["upper"], out["lower"] = mid, upper, lower

    close = out["close"]
    out["signal"] = _position_state(
        long_in=close > upper,
        long_out=close < mid,
        short_in=close < lower,
        short_out=close > mid,
    )
    return out


def strat_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_n: int = 9) -> pd.DataFrame:
    """MACD 趋势：双向。

    做多：DIF 在零轴上方且上穿 DEA；做空：DIF 在零轴下方且下穿 DEA；
    离场：反向交叉或回到零轴另一侧。
    """
    out = df.copy()
    dif = ema(out["close"], fast) - ema(out["close"], slow)
    dea = ema(dif, signal_n)
    out["dif"], out["dea"] = dif, dea
    out["hist"] = dif - dea

    out["signal"] = _position_state(
        long_in=(dif > dea) & (dif > 0),
        long_out=(dif < dea) | (dif < 0),
        short_in=(dif < dea) & (dif < 0),
        short_out=(dif > dea) | (dif > 0),
    )
    return out


def strat_turtle(
    df: pd.DataFrame,
    entry: int = 55,
    exit: int = 20,
    atr_n: int = 20,
    atr_mult: float = 2.5,
) -> pd.DataFrame:
    """海龟完整版：唐奇安突破入场 + N 倍 ATR 移动止损，双向。

    多头止损线 = 持仓期最高价 - k×ATR（只上移不下移）；
    空头止损线 = 持仓期最低价 + k×ATR（只下移不上移）。
    """
    out = df.copy()
    entry, exit = int(entry), int(exit)
    a = atr(out, atr_n)
    up_e = out["high"].rolling(entry, min_periods=entry).max().shift(1)
    dn_e = out["low"].rolling(entry, min_periods=entry).min().shift(1)
    up_x = out["high"].rolling(exit, min_periods=exit).max().shift(1)
    dn_x = out["low"].rolling(exit, min_periods=exit).min().shift(1)
    out["upper"], out["lower"], out["atr"] = up_e, dn_e, a

    close = out["close"].to_numpy()
    high = out["high"].to_numpy()
    low = out["low"].to_numpy()
    up_e_a, dn_e_a = up_e.to_numpy(), dn_e.to_numpy()
    up_x_a, dn_x_a = up_x.to_numpy(), dn_x.to_numpy()
    av = a.to_numpy()

    n = len(out)
    sig = np.zeros(n, dtype=int)
    pos = 0
    extreme = np.nan   # 多头记最高价，空头记最低价
    stop = np.nan

    for i in range(n):
        if np.isnan(av[i]):
            sig[i] = 0
            continue
        if pos == 0:
            if not np.isnan(up_e_a[i]) and close[i] > up_e_a[i]:
                pos, extreme = 1, high[i]
                stop = close[i] - atr_mult * av[i]
            elif not np.isnan(dn_e_a[i]) and close[i] < dn_e_a[i]:
                pos, extreme = -1, low[i]
                stop = close[i] + atr_mult * av[i]
        elif pos == 1:
            extreme = max(extreme, high[i])
            trail = extreme - atr_mult * av[i]
            stop = max(stop, trail) if not np.isnan(stop) else trail
            if close[i] < stop or (not np.isnan(dn_x_a[i]) and close[i] < dn_x_a[i]):
                pos, extreme, stop = 0, np.nan, np.nan
        else:
            extreme = min(extreme, low[i])
            trail = extreme + atr_mult * av[i]
            stop = min(stop, trail) if not np.isnan(stop) else trail
            if close[i] > stop or (not np.isnan(up_x_a[i]) and close[i] > up_x_a[i]):
                pos, extreme, stop = 0, np.nan, np.nan
        sig[i] = pos
    out["signal"] = sig
    return out


def strat_sbbox_mini(
    df: pd.DataFrame,
    entry: int = 240,
    exit_n: int = 720,
    vol_ratio: float = 0.4,
    range_n: int = 1,
    atr_n: int = 14,
    stop_mult: float = 5.0,
) -> pd.DataFrame:
    """箱体 + 波幅双确认突破（迷你版），设计用于 5 分钟周期。

    入场要**两个条件同时成立**：收盘突破 `entry` 根箱体，且相对当日开盘的偏离
    不小于 `vol_ratio` 倍的参考波幅。只看突破的话，窄幅横盘里那些一个点的高点
    也算突破，5 分钟上一天能开十几次；加一条「偏离够大」就把它们滤掉了。

    离场两条：收盘反向走 `stop_mult` 倍 ATR（止损），或者反向突破 `exit_n` 根
    箱体（趋势结束）。信号在收盘确定，下一根开盘成交 —— 和平台其它策略一致。

    「当日」按**交易日**算，不是自然日：走 `_trading_days()`，优先用数据里带的
    `trading_day` 列（交易所口径），没有该列才退回「整体 +6h 再取日期」。
    🔴 不要在这里自己写 `index + 6h`：周五 21:00 的夜盘加 6h 会落到**周六**，
    于是周一的「当日开盘」错取成周一 09:00 的开盘（丢掉隔夜跳空），周一的
    「昨日全天振幅」错取成周五夜盘那一小段 —— RB/SA 上实测分别有 13.2% / 12.8%
    的 bar 受影响，波幅基准被低估 45~49%，周一的开仓门槛被系统放松。
    """
    out = df.copy()
    c, h, l, o = out["close"], out["high"], out["low"], out["open"]

    hi = c.rolling(entry, min_periods=entry).max().shift(1)
    lo = c.rolling(entry, min_periods=entry).min().shift(1)
    hi_x = c.rolling(exit_n, min_periods=exit_n).max().shift(1)
    lo_x = c.rolling(exit_n, min_periods=exit_n).min().shift(1)

    day_key = _trading_days(out)
    day_open = o.groupby(day_key).transform("first")
    dev = c - day_open

    ref_range = (
        h.rolling(range_n, min_periods=range_n).max().shift(1)
        - l.rolling(range_n, min_periods=range_n).min().shift(1)
    )
    thr = vol_ratio * ref_range
    a = atr(out, atr_n)

    cu, au = c.to_numpy(), a.to_numpy()
    bu, bd = (c > hi).to_numpy(), (c < lo).to_numpy()
    ub, db = (dev > thr).to_numpy(), (-dev > thr).to_numpy()
    xu, xd = (c < lo_x).to_numpy(), (c > hi_x).to_numpy()

    n = len(out)
    sig = np.zeros(n, dtype=int)
    pos = 0
    epx = np.nan
    for i in range(n):
        new_pos = pos
        if pos == 1:
            if cu[i] < epx - stop_mult * au[i] or xu[i]:
                new_pos = 0
        elif pos == -1:
            if cu[i] > epx + stop_mult * au[i] or xd[i]:
                new_pos = 0
        if pos == 0:
            if bu[i] and ub[i]:
                new_pos = 1
            elif bd[i] and db[i]:
                new_pos = -1
        if new_pos != pos:
            pos = new_pos
            epx = cu[i] if pos != 0 else np.nan
        sig[i] = pos

    out["signal"] = sig
    return out


def strat_sbbox_ib(
    df: pd.DataFrame,
    entry_n: int = 20,
    exit_n: int = 60,
    vol_ratio: float = 0.4,
    range_n: int = 1,
    atr_n: int = 75,
    stop_mult: float = 1.5,
) -> pd.DataFrame:
    """箱体 + 波幅双突破，**K 线内成交**（`_ib` = intrabar）。

    和 `strat_sbbox_mini` 的区别不在条件，在**成交时点**：

    - `sbbox_mini` 等收盘确认，下一根开盘才成交；
    - 这一条在价格**碰到触发价的那一刻**就成交，不等这根收盘。

    ## 入场：两个条件必须**同时**满足（且，不是或）

        ① 箱体突破   价格 > 前 entry_n 根的最高价
        ② 波幅突破   价格 > 当日开盘 + vol_ratio × 昨日全天高低波幅

    两个条件都是「价格向上越过某个价位」，所以「同时成立」等价于「越过两者中
    更高的那个」：

        做多  up = max(①的价位, ②的价位)    ← 越过 up 就同时越过了 ① 和 ②
        做空  dn = min(①的价位, ②的价位)

    🔴 取 max 是**为了在盘中判断「两个条件同时成立」**，不是二选一：只要价格
    只碰到其中一条线（比如只破了箱体、还没涨够 40% 波幅），就**不进**。
    测试里有一条逐笔断言：每一笔成交价减去两条线都 > 0（见
    `selftest` 的「K 线内进场是且不是或」）。

    ## 止损

    `stop_mult` × ATR(`atr_n`)，绑在**成交价**上，K 线内碰到就按止损价成交。
    ATR 周期可调区间 **10~50**，默认 **20**。周期越短，止损贴得越近：1h 上
    20 根就是 20 小时，跟着最近的波动走；5 分钟上 20 根只有 100 分钟，止损会
    非常紧。

    🔴 **进场当根不出场**：进场那一根只负责建仓，止损从**下一根**才开始判
    （`BacktestConfig.intrabar_samebar_stop` 默认 `False`，引擎的 ① ② 两处
    出场都带 `i > entry_bar` 守卫）。原因：进场那根 bar 的 `low` 很可能是
    **建仓之前**做出来的 —— 进场要求价格摸到触发价，而 low 往往在开盘那一刻
    就已经成交过。拿整根 bar 的 low 去判，止损距离比这根振幅小的时候几乎必然
    误杀，一次都没持仓却记一笔亏损。那个开关打开只是「悲观上界」的诊断口径。

    🔴 ATR 取**上一根**（`shift(1)`）：当根的 high/low/close 在 K 线走的过程中
    还没定型，拿当根的 ATR 去定当根的止损就是前视，回测会好得没有道理。

    ## 出场

    当根**收盘价**反向突破前 `exit_n` 根收盘价箱体 ⇒ **下一根开盘**平仓。
    这一条不盘中化：用户明确要求「下一根 K 线开盘价格出场」，而且收盘箱体
    本身也要收盘才成立。

    ## 不反手

    持有多单时**完全看不到空头信号** —— 空头触发价再被打穿，也只当没发生；
    只有多单真出场（ATR 止损 / 反向突破平仓）之后，**下一根起**才可能开空。
    持空单反之，只有止损或信号出场之后才可能做多。

    这不是保守，是「反手」在 K 线内根本没法验证：一根 5 分钟线里「先止损平多
    再跌破开空」和「先跌破开空再被止损」，盘口长得一样。所以平仓当根一律不再
    开新仓（引擎侧同一开关 `BacktestConfig.intrabar_reentry_samebar`），
    和 `sbbox_mini` 那套「收盘确认 → 下一根开盘」的语义一致。

    「当日」按**交易日**算：走 `_trading_days()`，优先用数据里带的 `trading_day`
    列（交易所口径），没有该列才退回「整体 +6h 再取日期」。🔴 别在这里自己写
    `index + 6h` —— 周五 21:00 加 6h 落到周六，周一的「当日开盘」和「昨日振幅」
    都会错位（RB/SA 实测 13.2% / 12.8% 的 bar 受影响）。
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        for k in ("entry_long", "entry_short", "exit_long", "exit_short", "stop_dist"):
            out[k] = pd.Series(dtype=float)
        out["signal"] = pd.Series(dtype=int)
        return out

    c, h, l, o = out["close"], out["high"], out["low"], out["open"]

    # ---- 入场箱体：前 entry_n 根的高/低点，shift(1) 把当根排除在外 ----
    box_hi = h.rolling(entry_n, min_periods=entry_n).max().shift(1)
    box_lo = l.rolling(entry_n, min_periods=entry_n).min().shift(1)

    # ---- 出场箱体：前 exit_n 根的**收盘价**（不是高低点） ----
    c_hi = c.rolling(exit_n, min_periods=exit_n).max().shift(1)
    c_lo = c.rolling(exit_n, min_periods=exit_n).min().shift(1)

    # ---- 当日开盘（按交易日，夜盘归次日） ----
    day_key = _trading_days(out)
    day_open = o.groupby(day_key).transform("first")

    # ---- 昨日高低波幅：按交易日聚合全天 high.max − low.min，再 shift(1) ----
    # 直接对 bar 做 rolling(max) − rolling(min) 会跨交易日把「昨天下午 + 今天
    # 早上」混成一段，波幅基准偏小；必须先在交易日这一层聚合。
    per_day = pd.DataFrame({"h": h.to_numpy(dtype=float),
                            "l": l.to_numpy(dtype=float)}, index=day_key)
    day_rng = (per_day["h"].groupby(level=0).max()
               - per_day["l"].groupby(level=0).min())
    # range_n 个交易日的均值，再整体前移一天 ⇒ 用到的都是「昨天及更早」
    ref_rng = day_rng.rolling(range_n, min_periods=1).mean().shift(1)
    yday = pd.Series(day_key).map(ref_rng).to_numpy(dtype=float)

    # ---- 触发价 ----
    box_hi_a = box_hi.to_numpy(dtype=float)
    box_lo_a = box_lo.to_numpy(dtype=float)
    do_a = day_open.to_numpy(dtype=float)
    band = float(vol_ratio) * yday
    up = np.maximum(box_hi_a, do_a + band)
    dn = np.minimum(box_lo_a, do_a - band)

    # ---- 止损距离（价格单位）= stop_mult × 上一根 ATR ----
    stop_dist = (float(stop_mult) * atr(out, int(atr_n)).shift(1))
    sd_a = stop_dist.to_numpy(dtype=float)

    # 任一输入缺失（箱体没暖起来 / 第一个交易日没有「昨日」/ ATR 还没到周期）
    # ⇒ 这根不触发。填 0 或填当根价格都会凭空造出一个触发点。
    # **ATR 也是必要条件**：没有止损距离就不该进场，否则这一笔没有止损，
    # 只会被后面的「反向突破」兜底，跟这套规则不是一回事
    bad = ~(np.isfinite(box_hi_a) & np.isfinite(do_a) & np.isfinite(yday)
            & np.isfinite(sd_a) & (sd_a > 0))
    up[bad] = np.nan
    dn[bad] = np.nan

    # ---- 出场标志：当根收盘反向突破 exit_n 根收盘箱体，引擎读 i-1 那一格 ----
    xl = (c < c_lo).to_numpy()
    xs = (c > c_hi).to_numpy()

    # ---- `signal` 列：只是给信号图看的持仓状态 ----
    # 真正的成交由引擎按同一套规则、带滑点和手数重放。这里的状态机是它的
    # 精简镜像（用 open 近似成交价、不管滑点），**不要**拿它当回测结果用。
    o_a = o.to_numpy(dtype=float)
    h_a = h.to_numpy(dtype=float)
    l_a = l.to_numpy(dtype=float)
    sig = np.zeros(n, dtype=int)
    pos = 0
    stop = np.nan
    for i in range(n):
        closed = False
        if pos == 1 and i > 0 and xl[i - 1]:
            pos, stop, closed = 0, np.nan, True
        elif pos == -1 and i > 0 and xs[i - 1]:
            pos, stop, closed = 0, np.nan, True
        if pos == 1 and np.isfinite(stop) and l_a[i] <= stop:
            pos, stop, closed = 0, np.nan, True
        elif pos == -1 and np.isfinite(stop) and h_a[i] >= stop:
            pos, stop, closed = 0, np.nan, True
        # 本根刚平过仓 ⇒ 本根不再开新仓（下一根才重新看触发价）。不这么写，
        # 一根 K 线里「止损平多」紧接着「跌破开空」就会被记成反手 —— 而这两件
        # 事在盘口上的先后根本分不出来。`sbbox_mini` 那套本来就是这个语义，
        # 这里和它对齐
        if pos == 0 and not closed:
            lu = bool(np.isfinite(up[i])) and h_a[i] >= up[i]
            su = bool(np.isfinite(dn[i])) and l_a[i] <= dn[i]
            # 同一根两头都打穿：分不出先后，宁可这单不做
            if lu and not su:
                fill = max(o_a[i], up[i])
                pos = 1
                stop = fill - sd_a[i]
                # 🔴 这里**不判同根止损**。这根 bar 的 low 很可能是进场之前做出来的
                # （进场要求价格摸到 up，而 low 在开盘那一刻就已成交过），拿它判
                # 止损会凭空少掉一笔。引擎的口径是 `intrabar_samebar_stop=False`
                # （默认），镜像必须跟它一致 —— 否则「信号图上有仓、回测里没这笔」
                # 和「回测里有仓、信号图上没有」都会出现。
            elif su and not lu:
                fill = min(o_a[i], dn[i])
                pos = -1
                stop = fill + sd_a[i]
        sig[i] = pos

    out["entry_long"] = up
    out["entry_short"] = dn
    out["exit_long"] = xl.astype(float)
    out["exit_short"] = xs.astype(float)
    out["stop_dist"] = sd_a
    out["signal"] = sig
    return out


def strat_im_orb(
    df: pd.DataFrame,
    entry_start: int = 580,
    entry_end: int = 620,
    exit_min: int = 575,
    vol_ratio: float = 0.4,
    range_n: int = 1,
    atr_n: int = 14,
    stop_mult: float = 3.0,
) -> pd.DataFrame:
    """开盘区间突破（ORB）· 时段版，为股指日内设计。

    每天只用 `entry_start`~`entry_end` 这几分钟（默认 09:40~10:20，单位是
    从 0 点起的分钟数）开仓：价格碰到「当日开盘 ± `vol_ratio` × 前 N 日全天振幅」
    就顺着方向进，`stop_mult` 倍 ATR 止损，第二天 `exit_min`（默认 09:35）平。

    时段是硬约束，不是优化出来的：开盘头十分钟价格还在找方向，全天什么时候
    都可能突破 —— 但突破得越晚，离收盘越近，赚的钱越不够覆盖隔夜风险。

    拿日线数据跑这个策略会自动退化成「次日收盘平」：日线没有分钟，`ms` 全是同一个
    值，`intraday` 判假。留这条退路是为了让用户在日线上也能看个大概，不至于报错。
    """
    out = df.copy()
    out["signal"] = 0
    n = len(out)
    if n == 0:
        return out
    idx = pd.DatetimeIndex(out.index)
    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)
    ms = (idx.hour * 60 + idx.minute).to_numpy()
    days = idx.normalize().to_numpy()
    intraday = np.unique(ms).size > 1

    uday, dd = np.unique(days, return_inverse=True)
    dd = np.asarray(dd).reshape(-1)
    nd = len(uday)
    day_hi = np.full(nd, -np.inf)
    day_lo = np.full(nd, np.inf)
    np.maximum.at(day_hi, dd, h)
    np.minimum.at(day_lo, dd, l)
    day_open = np.full(nd, np.nan)
    seen = np.zeros(nd, dtype=bool)
    for i in range(n):
        k = dd[i]
        if not seen[k]:
            seen[k] = True
            day_open[k] = o[i]

    # 参考振幅取**前** range_n 个交易日的全天振幅，绝不能用当天的 —— 当天的
    # 高低点是收盘后才知道的，拿它当门槛就是未来函数
    rng_day = day_hi - day_lo
    rn = max(int(range_n), 1)
    prev_rng = np.full(nd, np.nan)
    if nd > rn:
        prev_rng[rn:] = rng_day[: nd - rn]
    upper = day_open[dd] + vol_ratio * prev_rng[dd]
    lower = day_open[dd] - vol_ratio * prev_rng[dd]
    atr_v = atr(out, atr_n).to_numpy(dtype=float)

    sig = np.zeros(n, dtype=int)

    if intraday:
        pos = 0
        entry_px = np.nan
        entry_atr = np.nan
        entry_day = -1
        last = -1
        for i in range(n):
            if pos != 0:
                hit = False
                if entry_atr == entry_atr:
                    if pos > 0 and l[i] <= entry_px - stop_mult * entry_atr:
                        hit = True
                    if pos < 0 and h[i] >= entry_px + stop_mult * entry_atr:
                        hit = True
                out_time = (ms[i] == exit_min) and (dd[i] > entry_day)
                if hit or out_time:
                    pos = 0
                    entry_px = np.nan
                    entry_atr = np.nan
                    entry_day = -1
            if pos == 0 and entry_start <= ms[i] <= entry_end and dd[i] != last:
                if upper[i] == upper[i] and h[i] >= upper[i]:
                    pos = 1
                    entry_px = c[i]
                    entry_atr = atr_v[i]
                    entry_day = dd[i]
                    last = dd[i]
                elif lower[i] == lower[i] and l[i] <= lower[i]:
                    pos = -1
                    entry_px = c[i]
                    entry_atr = atr_v[i]
                    entry_day = dd[i]
                    last = dd[i]
            sig[i] = pos
    else:
        # 日线：一天一根，开盘价就是全天第一个价，突破后只能等收盘或次日平
        pos = 0
        entry_day = -1
        for i in range(n):
            k = dd[i]
            if pos != 0 and k > entry_day:
                pos = 0
                entry_day = -1
            if pos == 0 and upper[i] == upper[i]:
                up_hit = h[i] >= upper[i]
                dn_hit = l[i] <= lower[i]
                if up_hit and dn_hit:
                    # 一根 K 线里上下都碰到：按「先到的那一边」判，用距离近似
                    pos = 1 if (upper[i] - l[i]) < (h[i] - lower[i]) else -1
                elif up_hit:
                    pos = 1
                elif dn_hit:
                    pos = -1
                if pos != 0:
                    entry_day = k
            sig[i] = pos

    out["signal"] = sig
    return out


# ------------------------------------------------- 「波幅突破」共用的交易日切分
def _trading_days(df: pd.DataFrame) -> np.ndarray:
    """K 线时间 → 交易日（`datetime64` 的日期部分）。

    **优先用数据里带的 `trading_day` 列**：天勤的分钟线本来就带这一列，它是
    交易所口径的真值 —— 周五 21:00 的夜盘属于下周一，节假日顺延，这些靠时间
    推算都会错。没有这一列（股票、手工造的日线）才退回下面的经验规则。

    🔴 **该列并非全品种可信**（2026-11 实测）：夜盘跨零点收盘的 13 个品种
    （AG/AU/CU/AL/NI/PB/SN/ZN/SS/SC/BC/AD/AO）有 8~11% 的 bar 的 `trading_day`
    落在**周六/周日** —— 天勤没把周五夜盘往前滚到周一。23:00 收盘的品种
    （RB/SA/M…）该列是对的。所以「周五夜盘归周一」这条只对 23:00 收盘的品种成立；
    要彻底修得改成「交易日 = 有日盘的自然日，夜盘前滚到下一个有日盘的日子」，
    不看这一列。

    经验规则：整体加 6 小时再取日期。期货交易日从**夜盘 21:00** 开始，不是
    自然日 00:00：21:30 那根属于第二天，凌晨 02:00 那根也属于第二天。
    21:30+6h=次日 03:30，02:00+6h=当日 08:00，白天各时段都在自己那天；
    无夜盘品种（只有 09:00-15:00）加不加都是同一天，不受影响。
    🔴 但**周五夜盘必错**：周五 21:00 + 6h = 周六。这条只配当兜底，别当正解。

    切错的话「当日开盘价」和「前一日波幅」两处都会错位 —— 而这两个数正是
    波幅突破的全部依据。
    """
    if "trading_day" in df.columns:
        td = pd.to_datetime(df["trading_day"], errors="coerce")
        if td.notna().any():
            return td.to_numpy()
    return (pd.DatetimeIndex(df.index) + pd.Timedelta(hours=6)).normalize().to_numpy()


def _day_blocks(day_key: np.ndarray, high: np.ndarray, low: np.ndarray,
                open_: np.ndarray):
    """按交易日汇总：每日首根 open、当日最高 / 最低，以及**前一日**波幅。

    返回 `(daily_open, prev_range, day_index)`：
    - `daily_open[i]` 第 i 根所属交易日的首根 open（夜盘品种即 21:00 那根）
    - `prev_range[i]` 该交易日的**前一交易日**全天振幅（high-low），首日 NaN
    - `day_index[i]` 第 i 根属于第几个交易日（0 起，用来判「换日了没有」）

    前一日波幅用当日 high/low 聚合后整体右移一格算，绝不拿当日的：
    当日高低点收盘后才知道，用它当门槛就是未来函数。
    """
    uday, dd = np.unique(day_key, return_inverse=True)
    dd = np.asarray(dd).reshape(-1)
    nd = len(uday)
    day_hi = np.full(nd, -np.inf)
    day_lo = np.full(nd, np.inf)
    np.maximum.at(day_hi, dd, high)
    np.minimum.at(day_lo, dd, low)

    daily_open = np.full(nd, np.nan)
    seen = np.zeros(nd, dtype=bool)
    for i in range(len(dd)):
        k = dd[i]
        if not seen[k]:
            seen[k] = True
            daily_open[k] = open_[i]

    prev_range = np.full(nd, np.nan)
    if nd > 1:
        prev_range[1:] = day_hi[:-1] - day_lo[:-1]
    return daily_open[dd], prev_range[dd], dd


# ------------------------------------------- 「波幅突破」日盘 / 夜盘的共用骨架
def _is_fin_futures(day_key: np.ndarray, ms: np.ndarray, has_night: bool) -> bool:
    """金融期货（股指 / 国债）从数据本身判，不读品种配置。

    两条同时成立才算：① 整段数据没有夜盘；② 交易日的第一根 K 线在 09:15 或更晚。
    商品期货 09:00 开盘（540 分），股指 09:30（570）、国债 09:15（555）——
    555 这条线正好把两类分开，而策略是纯函数，只认 `df`。

    为什么要单独认这一类：它们上午开盘晚（09:15 / 09:30），09:05 那根 K 线
    根本不存在，出场时刻必须另算。
    """
    if has_night:
        return False
    uday, inv = np.unique(day_key, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    first = np.full(len(uday), np.nan)
    seen = np.zeros(len(uday), dtype=bool)
    for i in range(len(inv)):
        k = inv[i]
        if not seen[k]:
            seen[k] = True
            first[k] = ms[i]
    if len(first) == 0:
        return False
    return bool(np.nanmedian(first) >= 555)


def _break_exit_hit(kind: str, fin: bool, has_night: bool, ms_i: int,
                    cur_day: int, entry_day: int,
                    cal_i, entry_cal) -> bool:
    """定时平仓：这一根 K 线到平仓时刻了没有。

    时刻全部写死、不做参数 —— 它们是交易所作息，不是策略偏好：

    - **日盘仓**：有夜盘的商品当晚 21:05 平；当晚没有夜盘（节前）就顺延到下一个
      开盘日的 09:05；无夜盘的商品次日 09:05；金融期货次日 09:35
    - **夜盘仓**：进场后的第一个 09:05

    「当晚」用**日历日**判、不用交易日：21:05 那根属于下一个交易日，但它是进场
    那天傍晚的夜盘，这才是「当晚」的意思。节前那晚根本没有 21:05 的 K 线，
    条件自然不成立，自动落到 09:05 那条 —— 这就是「遇节假日顺延到下一个开盘
    的白天 09:05」，不需要单独写顺延逻辑。

    判「到点」用 `>=` 而不是卡一个窄区间，并限定在日盘时段内（`< 1260`）：
    5 分钟档正好落在 21:05 / 09:05 那根，60 分钟档落到 22:00 / 10:00 那根，
    不用为不同周期写两套。日盘仓另外要求 `cur_day > entry_day` —— 定时平最早
    也是下一个交易日，同一天 13:00 的 K 线不能被当成「次日 09:05 到了」。
    夜盘仓反过来不能用交易日判：21:30 进的仓，次日早上 09:05 属于**同一个**
    交易日，所以只看「日盘时段的第一根」。
    """
    if kind == "night":
        return 545 <= ms_i < 1260
    if cur_day <= entry_day:
        return False
    if fin:
        return 575 <= ms_i < 1260      # 次日 09:35
    if not has_night:
        return 545 <= ms_i < 1260      # 次日 09:05
    return (ms_i >= 1265 and cal_i == entry_cal) or (545 <= ms_i < 1260)


def _ria_break_core(df: pd.DataFrame, *, ratio: float, atr_n: int,
                    stop_mult: float, hold_days: int,
                    window: tuple, kind: str) -> pd.DataFrame:
    """日盘 / 夜盘两条 RIA 波幅突破的共用实现。

    两者只有两处不同：`window`（开仓窗口）和 `kind`（决定出场时刻）。判据、
    止损、持仓上限完全一样 —— 拆成两条策略是为了让参数和出场时间各管各的，
    而不是因为逻辑不同。
    """
    out = df.copy()
    n = len(out)
    out["signal"] = 0
    if n == 0:
        return out

    idx = pd.DatetimeIndex(out.index)
    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)
    ms = (idx.hour * 60 + idx.minute).to_numpy()
    cal = idx.normalize().to_numpy()
    a = atr(out, atr_n).to_numpy(dtype=float)

    day_key = _trading_days(out)
    daily_open, prev_range, dd = _day_blocks(day_key, h, l, o)
    # 有没有夜盘从数据本身判：出现过 21:00 之后的 K 线就是有夜盘
    has_night = bool((ms >= 21 * 60).any())
    fin = _is_fin_futures(day_key, ms, has_night)

    w0, w1 = int(window[0]), int(window[1])
    ratio = float(ratio)
    stop_mult = float(stop_mult)
    hold_days = max(int(hold_days), 1)

    sig = np.zeros(n, dtype=int)
    pos = 0
    entry_px = np.nan
    entry_atr = np.nan
    entry_day = -1
    entry_cal = None
    last_entry = -1             # 已开过仓的交易日序号：一天最多一笔

    for i in range(n):
        cur_day = dd[i]
        if pos != 0:
            hit = False
            if entry_atr == entry_atr:
                if pos > 0 and c[i] <= entry_px - stop_mult * entry_atr:
                    hit = True
                elif pos < 0 and c[i] >= entry_px + stop_mult * entry_atr:
                    hit = True
            timed = _break_exit_hit(kind, fin, has_night, int(ms[i]),
                                    int(cur_day), int(entry_day),
                                    cal[i], entry_cal)
            if cur_day - entry_day >= hold_days:
                timed = True
            if hit or timed:
                pos = 0
                entry_px = entry_atr = np.nan
                entry_day = -1
                entry_cal = None
        if pos == 0 and last_entry != cur_day and w0 <= ms[i] <= w1:
            pr = prev_range[i]
            base = daily_open[i]
            if pr == pr and pr > 0 and base == base:
                dev = c[i] - base
                thr = ratio * pr
                if dev >= thr:
                    pos = 1
                elif dev <= -thr:
                    pos = -1
                if pos != 0:
                    entry_px = c[i]
                    entry_atr = a[i]
                    entry_day = cur_day
                    entry_cal = cal[i]
                    last_entry = cur_day
        sig[i] = pos

    out["signal"] = sig
    return out


def strat_ria_break_day(
    df: pd.DataFrame,
    ratio: float = 0.40,
    atr_n: int = 20,
    stop_mult: float = 5.0,
    day_start: int = 580,
    day_end: int = 640,
    hold_days: int = 20,
) -> pd.DataFrame:
    """RIA 波幅突破·日盘（future_quant `RIABreak` 的日盘时序腿）。

    - 突破判据：`当前收盘 − 当日首根开盘` 的偏离，超过 `ratio` × **前一交易日振幅**
    - 开仓窗口：日盘 09:40–10:40（默认 580–640 分钟），窗口内每根都看，
      **每个交易日最多一笔**
    - 进场：信号在 K 线收盘确认，下一根开盘成交（引擎统一处理）
    - 出场：① 进场价 ∓ `stop_mult` × ATR（进场时的 ATR，收盘价穿线即出）
      ② 定时平 —— **时刻写死，不是参数**：有夜盘的商品当晚 21:05 平；当晚
      没有夜盘（节前）就顺延到下一个开盘日的 09:05；无夜盘的商品次日 09:05；
      金融期货（股指 / 国债）次日 09:35 ③ 超过 `hold_days` 个交易日兜底平

    为什么用「前一交易日振幅」而不是 ATR：振幅是这一品种昨天真走过的距离，
    隔夜跳空、涨跌停、临近交割这些非常态都直接反映在里面；ATR 是平滑过的，
    遇到突发行情会慢半拍，阈值偏小就会天天开仓。

    「有夜盘」和「是不是金融期货」都从数据本身判，见 `_break_exit_hit`
    与 `_is_fin_futures` —— 策略是纯函数，只认 `df`，不读品种配置。

    默认参数取自 future_quant 的 60 分钟档（`small_lookback=2`、万 0.5 费率那组），
    在本平台上要按品种重新调 —— 参数是可调的，别把默认值当结论。
    """
    return _ria_break_core(df, ratio=ratio, atr_n=atr_n, stop_mult=stop_mult,
                           hold_days=hold_days,
                           window=(day_start, day_end), kind="day")


def strat_ria_break_night(
    df: pd.DataFrame,
    ratio: float = 0.40,
    atr_n: int = 20,
    stop_mult: float = 5.0,
    night_start: int = 1270,
    night_end: int = 1330,
    hold_days: int = 20,
) -> pd.DataFrame:
    """RIA 波幅突破·夜盘（future_quant `RIABreak` 的夜盘时序腿）。

    判据和日盘版完全一样，只有两处不同：

    - 开仓窗口是**夜盘自己的**：21:10–22:10（默认 1270–1330 分钟），
      每个交易日的夜盘窗口最多一笔
    - 出场统一次日 **09:05**，不看当晚有没有夜盘 —— 夜盘仓本来就是「隔夜」的，
      第二天早上第一根就平

    止损、持仓上限、判据全部与日盘版共用同一份实现，参数却各管各的：日盘调
    出来的 `ratio` 直接套到夜盘通常不合适，夜盘波动率结构不一样。

    只能用在有夜盘的品种上。没有夜盘的品种（股指、国债、苹果一类）在这条策略
    上永远不会开仓 —— 窗口里根本没有 K 线。
    """
    return _ria_break_core(df, ratio=ratio, atr_n=atr_n, stop_mult=stop_mult,
                           hold_days=hold_days,
                           window=(night_start, night_end), kind="night")


def strat_cta_break(
    df: pd.DataFrame,
    ratio: float = 0.40,
    check_min: int = 640,
    atr_n: int = 20,
    stop_mult: float = 5.0,
    exit_min: int = 575,
    hold_bars: int = 96,
) -> pd.DataFrame:
    """CTA 短周期波幅突破（检查点版）。

    移植自 future_quant `cta_short/rule_day_backtest.py` 与 `rule_night_backtest.py`。
    和 `ria_break_day` 是同一条判据（`收盘 − 当日开盘` 相对前一交易日振幅），区别只在
    **什么时候看**：

    - RIA 在窗口内每根都看、每个窗口最多一笔；这条一天只看**一个检查点**
      （默认 10:40 = 640 分钟），到点那根的偏离过阈值就进
    - 出场：`stop_mult` × ATR 止损 + 次日 `exit_min`（默认 09:35）定时平
      + 最长持有 `hold_bars` 根

    夜盘版不用另写一条策略：把 `check_min` 设成 21:30（1290）、`exit_min` 设成
    09:05（545）就是夜盘口径 —— 两者除了时刻没有任何差别。

    「一天只看一次」不是偷懒：检查点越晚，价格越接近当天方向定型，假突破越少；
    但也越接近收盘，赚的钱不够覆盖隔夜风险。10:40 是 future_quant 扫出来的平衡点。
    """
    out = df.copy()
    n = len(out)
    out["signal"] = 0
    if n == 0:
        return out

    idx = pd.DatetimeIndex(out.index)
    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)
    ms = (idx.hour * 60 + idx.minute).to_numpy()
    a = atr(out, atr_n).to_numpy(dtype=float)

    daily_open, prev_range, dd = _day_blocks(_trading_days(out), h, l, o)

    ck = int(check_min)
    ex = int(exit_min)
    stop_mult = float(stop_mult)
    ratio = float(ratio)
    hold_bars = max(int(hold_bars), 1)

    sig = np.zeros(n, dtype=int)
    pos = 0
    entry_px = np.nan
    entry_atr = np.nan
    entry_day = -1
    entry_i = -1
    checked = -1                # 已经看过检查点的交易日序号

    for i in range(n):
        cur_day = dd[i]
        if pos != 0:
            hit = False
            if entry_atr == entry_atr:
                if pos > 0 and c[i] <= entry_px - stop_mult * entry_atr:
                    hit = True
                elif pos < 0 and c[i] >= entry_px + stop_mult * entry_atr:
                    hit = True
            timed = ms[i] == ex and cur_day > entry_day
            if hit or timed or (i - entry_i) >= hold_bars:
                pos = 0
                entry_px = entry_atr = np.nan
                entry_day = -1
                entry_i = -1
        # 检查点：当天的第一根 `ms >= check_min` 的 K 线，一天只看这一次
        if pos == 0 and cur_day != checked and ms[i] >= ck:
            checked = cur_day
            pr = prev_range[i]
            base = daily_open[i]
            if pr == pr and pr > 0 and base == base:
                dev = (c[i] - base) / pr
                if dev >= ratio:
                    pos = 1
                elif dev <= -ratio:
                    pos = -1
                if pos != 0:
                    entry_px = c[i]
                    entry_atr = a[i]
                    entry_day = cur_day
                    entry_i = i
        sig[i] = pos

    out["signal"] = sig
    return out


# ---------------------------------------------------------------- A 股选股型
def _limit_pct(is_st, board):
    """涨跌停幅度。`board`：0 主板 / 1 创业板 / 2 科创板 / 3 北交所。

    🔴 ST 会**改涨跌停幅度**（主板 5%），所以「昨日是否涨停」这件事必须先知道
    当时是不是 ST。创业板/科创板注册制之后不论 ST 都是 20%，北交所 30%。
    """
    st = np.asarray(is_st, dtype=bool)
    b = np.asarray(board, dtype=float)
    return np.where(b == 3, 0.30,
                    np.where(st & (b == 0), 0.05,
                             np.where((b == 1) | (b == 2), 0.20, 0.10)))


def _round2(x):
    """价格四舍五入到分。**不能用 `np.round`**：它是银行家舍入（.005 → 偶数），
    交易所是四舍五入，差一分钱就会把「涨停」判成「没涨停」。"""
    return np.floor(np.asarray(x, dtype=float) * 100.0 + 0.5) / 100.0


def strat_a_vol_breakout(df: pd.DataFrame, max_price: float = 80.0,
                         gain_lo: float = 0.0, gain_hi: float = 9.8,
                         vol_mult: float = 3.0, amt_min: float = 5000.0,
                         pe_min: float = 0.0, pe_max: float = 25.0,
                         hold_days: int = 5) -> pd.DataFrame:
    """A 股·放量启动（选股型，日线）。

    当日收盘同时满足下面八条 ⇒ 记一个信号，**下一根开盘买**，持有 `hold_days`
    根后开盘卖：

    1. 非 ST（按**当时**的曾用名判，不是现在是不是 ST）
    2. 非北交所
    3. 昨日未涨停（按板块 + 当日 ST 状态还原昨天的涨停价）
    4. 股价 < `max_price`（默认 80）
    5. 今日涨幅在 [`gain_lo`, `gain_hi`]（默认 0 ~ 9.8%）
    6. 今日成交量 ≥ `vol_mult` × 昨日（默认 3 倍）
    7. `pe_min` < PE(TTM) ≤ `pe_max`（默认 0 ~ 25；**负 PE 是亏损，不算「低估值」**）
    8. 今日成交额 ≥ `amt_min` 万元（默认 5000 万）

    🔴 需要的列都是 A 股日线专有的（`pct_chg` / `volume` / `amount` / `pe_ttm`
    / `is_st` / `board_code`）。缺列直接报错、**不静默降级** —— 「ST 过滤悄悄
    没生效」这种错不会报错，只会让结果看着更好。
    """
    out = df.copy()
    need = ("pct_chg", "volume", "amount", "pe_ttm", "is_st", "board_code")
    miss = [c for c in need if c not in out.columns]
    if miss:
        raise ValueError(
            "A股·放量启动 只能跑 A 股日线，缺列：" + "、".join(miss)
            + "（需要 Tushare 日线 + 基本面 + 曾用名 ST 记录）"
        )

    n = len(out)
    c = out["close"].to_numpy(dtype=float)
    pct = pd.to_numeric(out["pct_chg"], errors="coerce").to_numpy(dtype=float)
    vol = pd.to_numeric(out["volume"], errors="coerce").to_numpy(dtype=float)
    # Tushare `daily.amount` 单位是**千元**，参数按「万元」给，这里换成同一量纲
    amt = pd.to_numeric(out["amount"], errors="coerce").to_numpy(dtype=float) / 10.0
    pe = pd.to_numeric(out["pe_ttm"], errors="coerce").to_numpy(dtype=float)
    is_st = pd.Series(out["is_st"]).fillna(False).astype(bool).to_numpy()
    board = pd.to_numeric(out["board_code"], errors="coerce").to_numpy(dtype=float)

    # 昨日成交量 / 收盘价
    prev_vol = np.r_[np.nan, vol[:-1]]
    prev_close = np.r_[np.nan, c[:-1]]
    prev2_close = np.r_[np.nan, np.nan, c[:-2]]
    # 昨天的涨停价：用**前天**收盘 × (1 + 昨天的幅度)。幅度用昨天的 ST 状态，
    # 因为 ST 是当天起效的
    prev_st = np.r_[False, is_st[:-1]]
    yday_up = _round2(prev2_close * (1.0 + _limit_pct(prev_st, board)))
    yday_limit_up = prev_close >= yday_up - 1e-6

    with np.errstate(invalid="ignore"):
        ok = (
            (~is_st)                                  # 1 非 ST
            & np.isin(board, (0.0, 1.0, 2.0))         # 2 非北交所（板块已知）
            & (~yday_limit_up)                        # 3 昨日未涨停
            & (c < float(max_price))                  # 4 股价
            & (pct >= float(gain_lo)) & (pct <= float(gain_hi))   # 5 今日涨幅
            & (vol >= float(vol_mult) * prev_vol)     # 6 放量
            & (pe > float(pe_min)) & (pe <= float(pe_max))        # 7 低估值
            & (amt >= float(amt_min))                 # 8 成交额（万元）
        )
    ok = np.asarray(ok, dtype=bool)

    # 持仓状态机：满足条件当根出信号，引擎在**下一根开盘**成交；连持 hold_days 根
    hold = max(int(hold_days), 1)
    sig = np.zeros(n, dtype=int)
    left = 0
    for i in range(n):
        if left > 0:
            sig[i] = 1
            left -= 1
        elif ok[i]:
            sig[i] = 1
            left = hold - 1

    out["signal"] = sig
    # 逐根写「为什么进」：交易记录页的备注列直接读它。只给真正触发的那几根写，
    # 其余留空 —— 几千根都挂一句话会把表撑爆，也看不出重点
    note = np.full(n, "", dtype=object)
    if ok.any():
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = vol / prev_vol
        for i in np.flatnonzero(ok):
            note[i] = (f"放量启动：涨幅 {pct[i]:.2f}%·量比 {ratio[i]:.2f}"
                       f"·PE {pe[i]:.1f}")
    out["entry_note"] = note
    return out


# 网格的**按品种默认方向**（`follow=-1` 时查这张表，没列出的按顺势）。
# 依据是 2026-11-11 同一段样本（5m、挂单成本、禁同根、14:55 强平、格 0.6%/3 档）：
#   股指四兄弟 IM/IC/IF/IH 这段是单边上涨（+27%~+71%），顺势 +129%~+155%、
#   逆势基本全部爆仓；白糖 SR 是震荡（-9.3%），逆势 +42.5%（卡玛 2.49）、
#   顺势 -33.8%。所以方向不是「哪种更好」的问题，而是品种行情性质的问题。
# 🔴 这是**样本内的结论**，不是永久属性：品种从趋势转震荡时要重新判。
#    NR 两条路都不赚（逆势扫 12 组最好 -21.4%），放进来是因为用户指定，
#    实盘前必须自己复核。
GRID_FOLLOW_BY_SYMBOL: dict[str, int] = {
    "IM": 1, "IC": 1, "IF": 1, "IH": 1,     # 股指：顺势
    "SR": 0, "NR": 0,                        # 白糖 / 20 号胶：逆势
}
# 🔴表里没有的品种**默认走逆势**（2026-11-11 全市场体检改的）。
# 原来这里兜底是 1（顺势）—— 等于把股指的顺势口径套到全部 72 个非股指品种上，
# 实测顺势在非股指上中位累计 −43.9%（78 个品种只有 7 个正），逆势 −16.5%
# （15 个正）。而且**没有任何品种在两个方向上都赚钱** ⇒ 方向本来就该按品种定，
# 兜底值必须选「天然属于网格」的那一个：网格赚的是价格回中枢，逆势才是它的本义。
# 详见 MEMORY #112。

# 分品种定稿参数（2026-11-11 全历史寻优，笔数均 >= 1000 笔，见 MEMORY #109f）。
# 为什么必须分品种：**格宽要按「单 bar 波幅」定** —— 股指 5m 波幅大、格宽要放到
# 0.56~0.75%，白糖只要 0.47%；而中枢长度取决于品种的「记忆长度」：股指 36 根
# （0.75 日）最优，上证 50 要 60 根、白糖要 96 根。同一套参数套全品种，实测
# IC 累计 +28% 会变成 −5%（中枢一改就翻负）。
#
# 每项含义：ma_n 中枢根数 / grid_n 档数 / grid_mode 格宽口径
#          （1=5mATR, 2=日线ATR）/ atr_mult 倍数 / hold 持有过夜 / flat_pct 震荡阈值
GRID_PRESET_BY_SYMBOL: dict[str, dict] = {
    "IM": {"ma_n": 36, "grid_n": 4, "grid_mode": 1, "atr_mult": 3.0, "hold": 0,
           "flat_pct": 0.0},
    "IC": {"ma_n": 36, "grid_n": 4, "grid_mode": 2, "atr_mult": 0.4, "hold": 0,
           "flat_pct": 0.0},
    "IF": {"ma_n": 36, "grid_n": 4, "grid_mode": 1, "atr_mult": 3.0, "hold": 0,
           "flat_pct": 0.0},
    "IH": {"ma_n": 60, "grid_n": 4, "grid_mode": 2, "atr_mult": 0.8, "hold": 0,
           "flat_pct": 0.0},
    "SR": {"ma_n": 96, "grid_n": 5, "grid_mode": 1, "atr_mult": 4.0, "hold": 1,
           "flat_pct": 0.0},
}
# 表里没有的品种：按「股指那一档」的中性配置兜底，不隔夜、**开震荡过滤**（保守）。
# 🔴`flat_pct` 兜底 1.0 是 2026-11-11 全市场体检的结果：78 个品种里绝大多数是
# 「趋势段做网格」被拖死的，开 1% 的震荡过滤是唯一能把爆仓数压到 0 的开关。
# 但那 5 个定稿品种**实测被过滤拖坏**（组合 27.65% → 9.02%），所以它们显式写 0。
# 判据：过滤是「救大面积」+「伤尖子」，只能按品种分，不能全局开。
GRID_PRESET_FALLBACK: dict = {"ma_n": 36, "grid_n": 4, "grid_mode": 1,
                              "atr_mult": 3.0, "hold": 0, "flat_pct": 1.0}


def strat_grid(df: pd.DataFrame, ma_n: int = -1, grid_n: int = -1,
               grid_pct: float = 0.5, flat_pct: float = -1.0,
               flat_min: int = 895, night: int = 0,
               follow: int = -1, hold: int = -1,
               grid_mode: int = -1, atr_n: int = 300,
               atr_nd: int = 10, atr_mult: float = -1.0,
               net_stop: float = 0.0, grid_side: int = 2,
               mid_band: float = 0.0) -> pd.DataFrame:
    """网格交易：一格一档。**方向默认按品种**（股指顺势、其余逆势）。**为日内 5 分钟设计**。

    规则来自网格交易的标准定义，只把「起始价」换成了滚动中枢：

    - **中枢**：`ma_n` 均线，用**上一根**收盘算，不看当根（不能用未来）。
    - **方向（`follow`）**，这一条决定整个策略的性质（`-1` = 按品种，见
      `GRID_FOLLOW_BY_SYMBOL`；表里没有的品种兜底 **逆势**）：
      - `follow=1` **顺势**：价格在中枢**之上**做多头网格、**之下**做空头
        网格。档位挂在**趋势那一侧** —— 中枢之上第 k 档 = `中枢 + k×格距`，突破
        一档买一份；止盈挂在**再上一格**。
      - `follow=0` **逆势**：中枢**之下**做多、之上做空，档位挂在**回调侧**
        （中枢之下第 k 档 = `中枢 − k×格距`），止盈挂在**回到中枢的那一格**。
        这是教科书网格，「跌买涨卖」，2026-11-11 之前的默认值。
      🔴 两种方向不只是信号取负：**档位挂在哪一侧、止盈朝哪边走**都要一起翻。
      只把 signal 取负而不动几何，结果是「顺势方向 + 逆势价位」，档位永远挂在
      价格到不了的那一侧 —— 实测中证 1000 两年只成交 53 笔，是结构错位不是策略。
    - **档位**：以中枢为 0 档，每偏离一个「格宽」算一档，最多 `grid_n` 档。
      格宽有三种口径（`grid_mode`）：
      - `0`（默认）固定百分比 `grid_pct`% —— 简单，但同一个数套到不同品种上
        含义完全不一样（实测股指要 1.0%、白糖要 0.30%）。
      - `1` = `atr_n` 根 **5 分钟** ATR × `atr_mult`（默认 300 根 ≈ 6 个交易日）。
      - `2` = `atr_nd` 根 **日线** ATR × `atr_mult`。
      ATR 口径的好处是格宽跟着波动率自己走，**一套参数能跨品种**；代价是
      波动率跳变时档位价会整体平移（实盘要按「当根算出的档位」重挂单）。
      🔴两条 ATR 都 `shift(1)`，不许用当根的波动决定当根挂哪。
    - **`flat_pct`**：趋势过滤。只有中枢在 `ma_n` 根内的变化率不超过它才开网格，
      否则空仓。`0` = 不过滤；`-1`（默认）= 按品种（定稿 5 个 0、其余 1.0）。
      🔴全市场实测：开 1% 能把爆仓数从 4 压到 0，但会把这 5 个定稿品种的组合收益
      从 27.65% 拖到 9.02% —— **救大面积、伤尖子**，所以只能按品种分。
    - **中枢为什么不用固定起始价**：固定起始价在实盘里没法复现 —— 今天开跑和
      昨天开跑的网格是两套不同的档位价，回测和实盘永远对不上。改成滚动中枢后，
      任何一根 K 线都能独立算出同一张网。

    `signal` 是**档位号**而不是 -1/0/1：`+k` = 持 k 份多头，`-k` = k 份空头。
    「一份」是多少手由回测的手数口径决定 —— 用「固定手数」就是每格 1 手，和
    `grid_volume_long = [0,1,2,...]` 完全一致。回测走 `run_grid_intrabar`
    （K 线内挂单成交），实盘走 `st.target = signal × lots`。

    ## 日内约束（这一族参数就是为 5 分钟周期加的）

    - `flat_min`：**平仓时刻**，从 0 点起的分钟数，默认 **895 = 14:55**。
      回测是「T 根收盘定信号、T+1 根开盘成交」，而 5m bar 的时间戳是**起点**
      （日盘最后一根是 14:55，覆盖 14:55~15:00）。想「14:55 手上必须是空的」，
      就得在 **14:50 那根**把 signal 打成 0 —— 下面按「下一根的开盘时刻 >=
      flat_min」清零，正好落在这里。改成别的时刻就按那个时刻自动前推一根。
    - **跨时段一律清零**：夜盘转日盘、日盘转夜盘都不留仓。日盘的午休
      （11:30~13:30）不算跨时段，可以持仓过午休 —— 那没有隔夜风险。
    - `night`：0 = 只做日盘（默认），1 = 夜盘也做。夜盘同样在时段末尾清零，
      不会把仓带进第二天。
    - `hold`：**1 = 持有过夜，只在方向翻转时强平**。这时上面三条时段规则
      （`flat_min` / 跨时段 / 夜盘清零）**全部关掉**，仓位一直拿到「信号反号」
      那一根才用开盘价市价全平；单档止盈照旧。⚠️ 代价是隔夜跳空全部暴露 ——
      日盘平仓省掉的是强平亏损，换来的是跳空风险，两者要分开看。
    - `net_stop`：**破网止损**（单位=格宽）。价格越过最外一档再走这么多格宽，
      就把整张网全平、当天不再开新网。`0` = 关闭（老行为）。
    - **日线及以上自动跳过整段约束**：日线的时间戳是当天零点，套「日盘」判据
      会把信号整条清零。判据是相邻 bar 的中位间隔（>= 1 天即视为非日内）。

    ## 三种方向结构（`grid_side`）

    原来的划分只有两块：价格站的那一侧算「多头区」、另一侧算「空头区」，同一时刻
    只有一个方向有网。但网格真正赚钱的地方是**震荡** —— 中枢附近来回抽的时候，
    上面挂着卖、下面挂着买，两个方向都能收一格。所以把原划分**中间切出一段**
    做无方向网格：

    - `grid_side=0` **只做多头**：只有多头区有网（等价于把空头信号清零）。跌下来
      接、回到中枢止盈，价格在另一侧时干等。适合「只做多」的产品约束。
    - `grid_side=1` **只做空头**：镜像。
    - `grid_side=2` **多空都做**（默认）：`|收盘 − 中枢| < mid_band × 格宽` 的
      **中间带**里，多空两张网同时挂着（`grid_dual=1`）；出了中间带只留原先那一
      侧的网，另一侧市价平掉。⚠️`mid_band` **默认 0 = 中间带关闭**，也就是「按
      价格位置在两个方向之间轮换」的原行为 —— 实测把带拉开（≥2 格）在股指上
      一律变差，只有白糖在 1~3 格时略好，所以这一条是**opt-in**，不是默认。
      🔴**逆势几何下这条「出带平掉另一侧」是赚钱的动作**：价格往下走出带时，上方
      那排空腿全在浮盈里（它们的止盈价在中枢附近，还没碰到），出带正好把浮盈兑现；
      否则要等价格涨回来才止盈，等于把已经赚到的让回去。顺势几何下反过来 —— 出带
      平掉的是亏的那一侧，等于一层趋势止损。

    `mid_band` 单位是**格宽**（不是百分比），`3.0` = 中枢上下三格以内双向。`0`
    （默认）= **关闭中间带**，退化成原来的单边网、结果逐位不变；给很大的数等于
    全程双向（等于不做方向判断）。
    """
    # ---- 方向：`follow` 取 1 / 0 / -1（按品种自动）----
    # 中枢之外的第二个决策点：同一套档位规则，方向反了就是另一条策略 ——
    # 趋势品种顺势才赚、震荡品种逆势才赚（同一样本里白糖逆势 +42%、顺势 -34%）。
    # `df.attrs["symbol"]` 由数据层挂上（因子层也靠它认品种），切片与复制都会
    # 保留；读不到就按顺势。
    _sym = str((getattr(df, "attrs", None) or {}).get("symbol") or "").upper()
    # ---- 分品种定稿参数：任一参数填 -1 就走 `GRID_PRESET_BY_SYMBOL` ----
    # 和 `follow=-1` 是同一套约定：**留 -1 = 按品种自动**，显式填数字就覆盖。
    # 这样「一套默认值跑批」和「单品种手调」用同一份代码，不用维护两条路径。
    _pre = GRID_PRESET_BY_SYMBOL.get(_sym, GRID_PRESET_FALLBACK)
    if int(ma_n) < 0:
        ma_n = _pre["ma_n"]
    if int(grid_n) < 0:
        grid_n = _pre["grid_n"]
    if int(grid_mode) < 0:
        grid_mode = _pre["grid_mode"]
    if float(atr_mult) < 0:
        atr_mult = _pre["atr_mult"]
    if int(hold) < 0:
        hold = _pre["hold"]
    if float(flat_pct) < 0:
        # 震荡阈值也走「-1 = 按品种」：定稿的 5 个显式 0（过滤会拖坏它们），
        # 其余品种兜底 1.0（趋势段不开网，是唯一能把爆仓压到 0 的开关）。
        flat_pct = _pre["flat_pct"]
    follow = int(follow)
    if follow < 0:
        # 🔴兜底 0（逆势），不是 1。网格赚的是「价格回中枢」，逆势是它的本义；
        #    顺势只对股指成立，而股指已经在表里显式列了。
        follow = int(GRID_FOLLOW_BY_SYMBOL.get(_sym, 0))
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    n = int(ma_n)
    gmax = max(int(grid_n), 1)
    pct = float(grid_pct) / 100.0
    out["center"] = sma(close, n).shift(1)

    # ---- 格宽：按百分比，还是按 ATR ----
    # `grid_mode=0` 固定百分比（`grid_pct`）；`1` = `atr_n` 根 5 分钟 ATR；
    # `2` = `atr_nd` 根**日线** ATR。后两种都再乘 `atr_mult`。
    # 🔴为什么值得换：固定百分比在低波品种上太宽（网格从不成交）、在高波品种上
    #    又太窄（一个 bar 扫穿整张网 + 手续费吃掉）。实测同一套 `grid_pct` 必须
    #    给股指配 1.0%、给白糖配 0.30% —— 差 3 倍多。ATR 定档就是把这一步
    #    交给波动率自己算。
    # 🔴两处都必须 `shift(1)`：第 i 根的档位若用到第 i 根的 ATR，就是拿当根
    #    的波动去决定当根挂哪 —— 前视。日线口径再 shift 一次日线，保证「今天
    #    用昨天及以前的日线 ATR」。
    mode = int(grid_mode)
    step = None
    if mode in (1, 2) and "high" in out.columns and "low" in out.columns:
        hi = pd.to_numeric(out["high"], errors="coerce").astype(float)
        lo = pd.to_numeric(out["low"], errors="coerce").astype(float)
        pc = close.shift(1)
        tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
        if mode == 1:
            atr = tr.rolling(max(int(atr_n), 2),
                             min_periods=max(2, int(atr_n) // 3)).mean().shift(1)
        else:
            # 日线口径：先按**自然日**聚合。夜盘 21:00 的 bar 会归到当天而不是
            # 下一个交易日，日线 bar 因此不是严格意义上的「交易日」bar —— 但
            # 它只用历史价、且整体 shift 一根，不会前视；对 ATR 量级的影响
            # 远小于「用当根 ATR」的代价（最后那一种直接是前视）。
            d = pd.DataFrame({"h": hi, "l": lo, "c": close})
            dd = d.resample("1D").agg({"h": "max", "l": "min", "c": "last"}).dropna()
            dpc = dd["c"].shift(1)
            dtr = pd.concat([dd["h"] - dd["l"], (dd["h"] - dpc).abs(),
                             (dd["l"] - dpc).abs()], axis=1).max(axis=1)
            datr = dtr.rolling(max(int(atr_nd), 2),
                               min_periods=max(2, int(atr_nd) // 2)).mean().shift(1)
            key = pd.DatetimeIndex(out.index).normalize()
            atr = pd.Series(pd.Series(datr.to_numpy(), index=dd.index)
                            .reindex(key).to_numpy(), index=out.index).ffill()
        step = (atr * float(atr_mult)).where(lambda s: s > 0)
    if step is None:
        # 固定百分比（默认）：中枢还没算出来时（预热段）置 NaN 留空仓
        step = (out["center"] * pct).where(lambda s: s > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        down = (out["center"] - close) / step
        up = (close - out["center"]) / step

    lv_long = np.floor(down.clip(lower=0, upper=float(gmax)))
    lv_short = np.floor(up.clip(lower=0, upper=float(gmax)))
    lv_long = lv_long.fillna(0).to_numpy()
    lv_short = lv_short.fillna(0).to_numpy()

    valid = out["center"].notna().to_numpy() & (close > 0).to_numpy()
    below = (close <= out["center"]).to_numpy()
    # 方向：顺势 = 跟着价格站的那一侧（之上做多、之下做空）；逆势 = 反过来。
    # 档位**绝对值**不变，变的是符号 —— 也就是「这一侧的偏离算多头还是空头」。
    # 档位到底挂在中心哪一侧、止盈朝哪边走，由回测按 `grid_follow` 一起算，
    # 只翻信号不翻几何会变成「顺势方向 + 逆势价位」，挂单永远成交不了。
    if int(follow):
        sig = np.where(below, -lv_long, lv_short)
    else:
        sig = np.where(below, lv_long, -lv_short)
    sig = np.where(valid, sig, 0).astype(int)

    # ---- 三种方向结构（`grid_side`）：只多 / 只空 / 带中间震荡带的多空都做 ----
    # `dev` = 价格偏离中枢几个格宽（带符号，正 = 在中枢上方）。中间带按它划。
    # 🔴用**未取整**的 `dev` 而不是档位号：档位号在 ±1 格内恒为 0，拿它判「在不在
    #    中枢附近」会把整个中间带压成一条线。
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = (close - out["center"]) / step
    dev_np = dev.to_numpy(dtype=float)
    _sid = int(grid_side)
    # 「中间带」= 价格还没走远的这一段，多空两张网一起挂。
    # 🔴`mid_band=0` 必须是**干干净净的关闭**（退化成原单边网，逐位不变）：
    #    写成 `<=` 时 `dev` 恰好等于 0 的那几根会翻成双向 —— 手数随之被摊成一半，
    #    整条曲线就变了（实测 SR 31.11% → 15.22%，只因为几根 `close == center`）。
    dual = (np.isfinite(dev_np) & (np.abs(dev_np) < float(mid_band)) & valid
            if float(mid_band) > 0 else np.zeros(len(sig), dtype=bool))
    if _sid == 0:
        # 只做多头：把空头信号整条清零。留 `dual=False`，否则出带规则会把
        # 本来就不存在的空腿当成「该平的反向腿」。
        sig = np.maximum(sig, 0).astype(int)
        dual = np.zeros(len(sig), dtype=bool)
    elif _sid == 1:
        sig = np.minimum(sig, 0).astype(int)
        dual = np.zeros(len(sig), dtype=bool)

    # 震荡过滤：中枢自己在这一段里的变化率。趋势段不开网
    calm = None
    if float(flat_pct) > 0:
        chg = (out["center"] / out["center"].shift(n) - 1.0).abs().to_numpy()
        calm = np.isfinite(chg) & (chg <= float(flat_pct) / 100.0)
        sig = np.where(calm, sig, 0).astype(int)

    # ---- 日内时段约束 ----
    idx = pd.DatetimeIndex(out.index)
    ms = (np.asarray(idx.hour) * 60 + np.asarray(idx.minute)).astype(int)
    # 日盘 08:00~16:00（含各品种 09:00 / 09:30 开盘差异），其余算夜盘
    is_day = (ms >= 480) & (ms < 960)
    total = len(out)
    # 日线及以上没有「日内时段」这回事：日线的时间戳一律是当天零点，套上
    # 「日盘 08:00~16:00」的判据会把整条信号清零（回测/自检里全是 0 档）。
    # 用相邻 bar 的中位间隔当判据 —— 5m 数据的中位间隔就是 5 分钟，日线是
    # 1440 分钟（周末空档不改变中位数）。
    step_min = 1440.0
    if total > 2:
        # 用 `total_seconds()` 而不是拿 `asi8` 自己除：`asi8` 的单位跟着索引的
        # 分辨率走（ns / us 都见过），除错一个量级就会把日线当成 1 分钟数据。
        gap = (pd.Series(idx).diff().dt.total_seconds() / 60.0).to_numpy()
        gap = gap[np.isfinite(gap) & (gap > 0)]
        if gap.size:
            step_min = float(np.median(gap))
    # `force` 要在两个分支外先建好：`hold=1` 时不进时段块，下面算 `grid_flat`
    # 还要用它（全 False = 没有任何一根是「被时段规则打掉」的）。
    force = np.zeros(total, dtype=bool)
    if total and step_min < 1440 and not int(hold):
        # 这一根的信号是在**下一根的开盘**成交的，所以时段判定要看下一根
        exec_ms = np.full(total, -1, dtype=int)
        cross = np.zeros(total, dtype=bool)
        if total > 1:
            exec_ms[:-1] = ms[1:]
            cross[:-1] = is_day[1:] != is_day[:-1]
        if not int(night):
            sig = np.where(is_day, sig, 0).astype(int)
        # ① 跨时段（夜→日 / 日→夜）不留仓；② 日盘里下一根已经到/过了平仓时刻；
        # ③ 最后一根没有下一根可成交，也按清仓处理
        force = cross | (is_day & (exec_ms >= int(flat_min))) | (exec_ms < 0)
        if not int(night):
            # ④ **本时段最后一根**也不留单。上面 ② 是靠「下一根的分钟数 >= 895」
            #    把 14:55 那根顺带打掉的，而节假日前一天没有夜盘，下一根直接
            #    跳到节后 09:00（540 < 895）⇒ 漏网，等于给节后开盘挂了一张隔夜
            #    单。判据用「下一根的时钟比这一根更早」＝换场/换日，不依赖数值。
            back = (exec_ms >= 0) & (exec_ms < ms)
            force = force | back
        sig = np.where(force, 0, sig).astype(int)

    out["signal"] = sig
    # ---- 挂单成交模型要用的两列 ----
    # `grid_step`：一格在**价格**上是多少（中枢 × 间距%）。K 线内成交要自己算
    #   第 k 档挂在哪个价位上，只知道「档位号」是不够的。
    # `grid_flat`：这一根的信号是**被时段/过滤规则强制打掉**的（不是「价格回到
    #   中枢自然出网」）。回测据此区分「下一根开盘市价全平」和「靠止盈单逐档
    #   平」—— 尾盘强平只能走市价，不能指望挂单成交。
    out["grid_step"] = step.astype(float)
    # 方向模式：1=顺势（档位挂趋势侧、止盈再走一格）/ 0=逆势（档位挂回调侧、
    # 止盈回中枢）。回测必须知道这一位，否则光看档位号算不出该挂在哪个价位。
    out["grid_follow"] = float(int(follow))
    # 这张网最多铺几档。回测的挂单模型要按「预先挂好的整排单」成交，不能拿
    # 收盘档位的最大值当上限 —— 那会随着行情漂移（一路上涨的样本可能只观测到
    # 2 档，等于偷偷把网缩小了）。
    out["grid_cap"] = float(gmax)
    # 破网止损：价格越过「最外一档再走 `net_stop` 个格宽」就全部平掉，当天不再
    # 开新网（`net_stop=0` = 关闭）。0 档到最外档是 `grid_n` 个格宽，所以它的
    # 绝对距离是 `(grid_n + net_stop) × 格宽`。
    # 🔴为什么网格必须要这一条：打满档位之后就再没有仓位约束了，价格继续单边走
    #    就是「满仓扛单」，直到 14:55 强平（或一直扛过夜）。这是网格亏损的主要
    #    来源，也是 `flat_pct` 过滤治不了的那一块。
    # 由回测引擎执行（它才有 high/low 路径），策略只负责把参数传下去。
    out["grid_net_stop"] = float(max(float(net_stop), 0.0))
    # 中间震荡带：这一根允许**多空两张网同时挂着**（`grid_side=2` 才有意义）。
    # 回测引擎读它决定「这一根建几个方向的腿」，以及「另一侧该不该市价平掉」。
    out["grid_dual"] = dual.astype(float)
    # 方向结构（0 只多 / 1 只空 / 2 多空都做）。回测只用来算手数摊薄：双向时每个
    # 方向分到一半预算，不然中间带里两条腿叠起来的名义敞口就是两倍。
    out["grid_side"] = float(_sid)
    grid_flat = np.zeros(len(out), dtype=bool)
    if total and step_min < 1440 and not int(hold):
        grid_flat = force.copy()
    if calm is not None:
        grid_flat = grid_flat | (~calm)
    out["grid_flat"] = grid_flat
    # 逐档逐根不好标备注（几千根都在网里），只标「换了档」的那几根 —— 交易记录
    # 的备注列读的就是这一列
    note = np.full(len(out), "", dtype=object)
    prev = np.concatenate([[0], sig[:-1]])
    _up = "上穿" if int(follow) else "下穿"
    _dn = "下穿" if int(follow) else "上穿"
    for i in np.flatnonzero(sig != prev):
        k = int(sig[i])
        if k == 0:
            note[i] = "回到中枢 / 出网 / 日内平仓，清仓"
        elif k > 0:
            note[i] = f"{_up}中枢第 {k} 档：持多 {k} 份"
        else:
            note[i] = f"{_dn}中枢第 {-k} 档：持空 {-k} 份"
    out["entry_note"] = note
    return out


# ---------------------------------------------------------------- 注册表
@dataclass
class StrategySpec:
    key: str
    label: str
    func: Callable[..., pd.DataFrame]
    params: dict = field(default_factory=dict)
    # 参数元信息：name -> (最小, 最大, 步长/类型)
    param_meta: dict = field(default_factory=dict)
    desc: str = ""
    bidirectional: bool = True   # 是否支持做空
    # 只在这些市场里有意义（`us` / `cn` / `cn_stock`）。**空 = 不限制**。
    # 研究页的策略下拉按它过滤：A 股选股策略挂在期货品种上跑不出东西（缺
    # `pct_chg` / `pe_ttm` / `is_st` 这些列，会直接报错），摆在期货的下拉里
    # 只是让人点一次、看一次报错。
    markets: tuple = ()
    # 是否按「K 线内触发价」成交（回测走 `run_intrabar_backtest`，实盘走
    # `PortfolioRunner._intrabar_tick`）。为真时策略必须输出 `entry_long` /
    # `entry_short` / `stop_dist` / `exit_long` / `exit_short` 这几列 ——
    # 只给 `signal` 的旧策略一律保持 False，走「收盘定信号、下一根开盘成交」
    intrabar: bool = False
    # 用户注册的策略自带源码：`exec` 出来的函数没有源文件，`inspect.getsource()`
    # 取不到，库页要展示代码就只能存一份
    code: str = ""
    # 参数中文名的**本策略覆盖**。同一个参数名在不同策略里含义不同时必填 ——
    # `range_n` 在 `sbbox_mini` 里是「根 bar」（1 = 上一根 K 线的振幅），在
    # `sbbox_ib` 里是「交易日」（1 = 昨天一整天）。共用一条「参考根数」会让
    # 人以为是同一回事，实测量级差 10 倍。
    param_labels: dict = field(default_factory=dict)


# 参数的中文名。研究页的滑杆、组合页的「设置参数」、手册都从这里取 ——
# 同一个 `fast` 在两个界面叫两个名字，用户会以为是两个不同的东西
PARAM_LABELS = {
    "fast": "快线周期", "slow": "慢线周期", "signal_n": "信号周期",
    "entry": "入场周期", "exit": "离场周期", "n": "通道周期",
    "k": "ATR 倍数", "atr_n": "ATR 周期", "atr_mult": "止损倍数",
    # 下面几条是 sbbox_mini / im_orb 的参数。标签要短：研究页那个滑杆标签
    # 只有 62px，超过六个汉字就被截断
    "exit_n": "离场箱体", "vol_ratio": "波幅倍数", "range_n": "参考根数",
    "stop_mult": "止损倍数", "entry_n": "入场箱体",
    "entry_start": "开仓起(分)", "entry_end": "开仓止(分)", "exit_min": "平仓(分)",
    # 波幅突破那一族（ria_break_day / ria_break_night / cta_break）
    "ratio": "波幅倍数", "check_min": "确认(分)",     "hold_days": "持仓(天)",
    "hold_bars": "持仓(根)",
    # A 股选股型（a_vol_breakout）。滑杆标签只有 62px，超过六个汉字会被截断
    "max_price": "股价上限", "gain_lo": "涨幅下限", "gain_hi": "涨幅上限",
    "vol_mult": "量比", "amt_min": "成交额(万)",
    "pe_min": "PE 下限", "pe_max": "PE 上限",
    "day_start": "日窗起(分)", "day_end": "日窗止(分)",
    "night_start": "夜窗起(分)", "night_end": "夜窗止(分)",
}


def param_label(spec: "StrategySpec | None", name: str) -> str:
    """参数在界面上显示的中文名。

    先看策略自己的 `param_labels` 覆盖，再退回全局表。**同一个参数名在两个策略
    里含义不同时必须用覆盖**，否则滑杆上写着同一个词、底下是两个量级的东西。
    """
    if spec is not None:
        hit = getattr(spec, "param_labels", {}).get(name)
        if hit:
            return hit
    return PARAM_LABELS.get(name, name)


STRATEGIES: dict[str, StrategySpec] = {
    # 基准两条放最前面：策略库里第一条就是「买入持有」，用户一眼能看到
    # 对照组的入口，不用翻到最后
    "buy_hold": StrategySpec(
        key="buy_hold",
        label="买入持有（基准）",
        func=strat_buy_hold,
        params={},
        param_meta={},
        desc="第一根 K 线满仓做多、之后一直不动。不看指标，是其它策略的对照基准。",
        bidirectional=False,
    ),
    "sell_hold": StrategySpec(
        key="sell_hold",
        label="卖出持有（基准）",
        func=strat_sell_hold,
        params={},
        param_meta={},
        desc="第一根 K 线满仓做空、之后一直不动。买入持有的镜像，用来判断多头策略是不是只是做反了。",
        bidirectional=False,
    ),
    "ma_cross": StrategySpec(
        key="ma_cross",
        label="双均线交叉",
        func=strat_ma_cross,
        params={"fast": 50, "slow": 200},
        param_meta={"fast": (5, 200, 5), "slow": (20, 400, 10)},
        desc="快线上穿慢线做多、下穿做空。对称反手，始终在市。",
    ),
    "grid_trade": StrategySpec(
        key="grid_trade",
        label="网格交易（日内档位）",
        func=strat_grid,
        params={"ma_n": -1, "grid_n": -1, "grid_pct": 0.6, "flat_pct": -1.0,
                "flat_min": 895, "night": 0, "follow": -1, "hold": -1,
                "grid_mode": -1, "atr_n": 300, "atr_nd": 10, "atr_mult": -1.0,
                "net_stop": 0.0, "grid_side": 2, "mid_band": 0.0},
        param_meta={
            "ma_n": (-1, 120, 1),
            "grid_n": (-1, 20, 1),
            "grid_pct": (0.05, 3.0, 0.05),
            "flat_pct": (-1.0, 10.0, 0.5),
            "flat_min": (600, 900, 5),
            "night": (0, 1, 1),
            "follow": (-1, 1, 1),
            "hold": (-1, 1, 1),
            "grid_mode": (-1, 2, 1),
            "atr_n": (20, 600, 10),
            "atr_nd": (2, 60, 1),
            "atr_mult": (-1.0, 4.0, 0.1),
            "net_stop": (0.0, 5.0, 0.5),
            "grid_side": (0, 2, 1),
            "mid_band": (0.0, 12.0, 0.5),
        },
        param_labels={"ma_n": "中枢周期(-1按品种)", "grid_n": "网格档数(-1按品种)",
                      "grid_pct": "每格间距(%)", "flat_pct": "震荡阈值(%)-1按品种",
                      "flat_min": "平仓(分)", "night": "做夜盘(0/1)",
                      "follow": "方向(1顺/0逆/-1按品种)",
                      "hold": "持有过夜(-1按品种)",
                      "grid_mode": "格宽口径(-1按品种/0百分比/1五分钟ATR/2日线ATR)",
                      "atr_n": "5mATR根数", "atr_nd": "日线ATR天数",
                      "atr_mult": "ATR倍数(-1按品种)",
                      "net_stop": "破网止损(格宽,0=关)",
                      "grid_side": "方向结构(0只多/1只空/2多空都做)",
                      "mid_band": "中间震荡带宽度(格宽)"},
        desc="一格一档，**为日内 5 分钟设计**：`ma_n` 均线当中枢。\n"
             "「顺势(1)/逆势(0)」**默认 1 = 顺势**：价格在中枢**之上**做多头网格"
             "（档位挂在中心之上，突破一档买一份，止盈再走一格），**之下**做空头"
             "镜像。逆势(0) 是教科书网格「跌买涨卖」，档位挂回调侧、止盈回中枢。"
             "🔴两种不只是信号取负，档位挂在哪一侧要一起翻 —— 只翻信号会变成"
             "「顺势方向 + 逆势价位」，单子永远挂不到价（实测中证 1000 两年 53 笔）。\n"
             "「方向结构」三选一：**0 只做多头 / 1 只做空头 / 2 多空都做**。多空都做"
             "时，中枢附近 `中间带宽度` 格以内**两张网同时挂**（上面卖、下面买，"
             "震荡区间两头都收一格），出了中间带只留原先那一侧、另一侧市价平掉。"
             "逆势几何下出带平掉的正好是浮盈中的那一侧，等于把震荡带的利润兑现。\n"
             "「平仓(分)」默认 895 = **14:55 清仓**，按「下一根开盘成交」自动前推"
             "一根（即 14:50 那根出平仓信号）；夜盘转日盘、日盘转夜盘一律不留仓，"
             "午休可以持仓。「做夜盘(0/1)」默认 0 = 只做日盘。\n"
             "signal 是**档位号**（+k = 持 k 份多），所以手数口径要用「固定手数」，"
             "每格 1 手才等于文章里的 grid_volume = 1..N。\n"
             "「震荡阈值」>0 时只在均线走平的震荡段开网。\n"
             "每格间距不能小于成本：手续费 + 滑点折成价格往往是 0.1~0.3%。\n"
             "🔴回测成本请选**「挂单」档**（0.1 跳滑点）：网格每一档都是限价挂单，"
             "成交价就是挂单价，不是穿过去再追回来。\n"
             "🔴成交是**K 线内挂单**：突破/砸到第 k 档就在那个价位成交、走到止盈价"
             "就在那平，不等下一根开盘。但**同一根 K 线绝不进出** —— 一根 bar 的"
             "high/low 先后分不出来，「先买后卖」是拿 OHLC 编出来的来回。\n"
             "🔴**逆势版实测在 5m 日内约束下没有正期望**（72 组只有 2 组微正）："
             "止盈单笔均赚 262~2434 元，但 14:55 强平的单笔均亏 92~3692 元、笔数是"
             "止盈的 4~5 倍 —— 日内强平把「不止损等回本」这个网格唯一的优势砍掉了。",
    ),
    "donchian": StrategySpec(
        key="donchian",
        label="唐奇安通道突破",
        func=strat_donchian,
        params={"entry": 55, "exit": 20},
        param_meta={"entry": (10, 120, 5), "exit": (5, 60, 5)},
        desc="突破 N 日新高做多、跌破 N 日新低做空，反向 M 日通道离场。",
    ),
    "atr_channel": StrategySpec(
        key="atr_channel",
        label="ATR 通道突破",
        func=strat_atr_channel,
        params={"n": 20, "k": 2.0},
        param_meta={"n": (10, 100, 5), "k": (1.0, 4.0, 0.5)},
        desc="收盘突破 均线±k×ATR 双向开仓，回到均线离场，自适应波动率。",
    ),
    "macd": StrategySpec(
        key="macd",
        label="MACD 趋势",
        func=strat_macd,
        params={"fast": 12, "slow": 26, "signal_n": 9},
        param_meta={"fast": (5, 20, 1), "slow": (15, 60, 1), "signal_n": (5, 15, 1)},
        desc="零轴上金叉做多、零轴下死叉做空，反向交叉或回零轴离场。",
    ),
    "turtle": StrategySpec(
        key="turtle",
        label="海龟（突破+ATR止损）",
        func=strat_turtle,
        params={"entry": 55, "exit": 20, "atr_n": 20, "atr_mult": 2.5},
        param_meta={
            "entry": (20, 120, 5),
            "exit": (5, 60, 5),
            "atr_n": (10, 40, 2),
            "atr_mult": (1.5, 5.0, 0.5),
        },
        desc="唐奇安双向突破 + 2.5×ATR 移动止损，单笔风险可控。",
    ),
    # 下面两条原本是用户在「ChanAgent」里生成的，2026-10 转为内置：
    # 用的人多、逻辑也稳定了，再留在用户库里意味着换台机器就没了
    "sbbox_mini": StrategySpec(
        key="sbbox_mini",
        label="箱体+波幅突破·迷你",
        func=strat_sbbox_mini,
        params={"entry": 240, "exit_n": 720, "vol_ratio": 0.4,
                "range_n": 1, "atr_n": 14, "stop_mult": 5.0},
        param_meta={
            "entry": (60, 720, 20),
            "exit_n": (240, 1440, 20),
            "vol_ratio": (0.1, 1.5, 0.05),
            "range_n": (1, 96, 1),
            "atr_n": (5, 60, 1),
            "stop_mult": (2.0, 12.0, 0.5),
        },
        desc="箱体突破 + 相对当日开盘的波幅确认，双条件同时成立才入场；"
             "反向 5×ATR 或反向 720 根箱体离场。设计用于 5 分钟周期。",
    ),
    "sbbox_ib": StrategySpec(
        key="sbbox_ib",
        label="箱体+波幅突破·K线内",
        func=strat_sbbox_ib,
        intrabar=True,
        params={"entry_n": 20, "exit_n": 60, "vol_ratio": 0.4,
                "range_n": 1, "atr_n": 20, "stop_mult": 1.5},
        param_meta={
            "entry_n": (5, 240, 5),
            "exit_n": (10, 480, 10),
            "vol_ratio": (0.1, 1.5, 0.05),
            "range_n": (1, 10, 1),
            "atr_n": (10, 50, 1),
            "stop_mult": (0.5, 5.0, 0.5),
        },
        # 🔴 这个 range_n 的单位是**交易日**（前 range_n 个交易日的平均全天振幅），
        # 和 sbbox_mini 的「根 bar」不是一个东西，标签必须分开
        param_labels={"range_n": "参考天数"},
        desc="两个条件**同时**满足才进：成交价要同时越过「前 20 根箱体上沿」和"
             "「当日开盘 ± 昨幅×40%」，也就是越过两者中更高的那个（碰价即进，"
             "不等收盘）。1.5×ATR（周期 10~50，默认 20）盘中止损；收盘反向突破 "
             "60 根收盘箱体则下一根开盘平。平仓当根不反手，下一根才重新看触发价。",
    ),
    "im_orb": StrategySpec(
        key="im_orb",
        label="IM 波幅突破·时段版",
        func=strat_im_orb,
        params={"entry_start": 580, "entry_end": 620, "exit_min": 575,
                "vol_ratio": 0.4, "range_n": 1, "atr_n": 14, "stop_mult": 3.0},
        param_meta={
            "entry_start": (540, 600, 5),
            "entry_end": (600, 690, 5),
            "exit_min": (555, 600, 5),
            "vol_ratio": (0.1, 1.0, 0.05),
            "range_n": (1, 5, 1),
            "atr_n": (5, 60, 1),
            "stop_mult": (1.0, 6.0, 0.5),
        },
        desc="日内 09:40-10:20 触及「当日开盘 ± 0.4×昨日振幅」开仓，3×ATR 止损，"
             "次日 09:35 平；日线上自动退化为次日收盘平。",
    ),
    # 下面三条从 future_quant 移植：同一条判据（收盘相对当日开盘的偏离 ÷ 前一日
    # 振幅），差别只在「什么时候看」—— 两条是窗口内每根都看（日盘 / 夜盘各一条，
    # 出场时刻不同），一条是一天只看一个检查点
    "ria_break_day": StrategySpec(
        key="ria_break_day",
        label="RIA 波幅突破·日盘",
        func=strat_ria_break_day,
        params={"ratio": 0.40, "atr_n": 20, "stop_mult": 5.0,
                "day_start": 580, "day_end": 640, "hold_days": 20},
        param_meta={
            "ratio": (0.1, 1.5, 0.05),
            "atr_n": (5, 60, 1),
            "stop_mult": (2.0, 12.0, 0.5),
            "day_start": (540, 620, 5),
            "day_end": (580, 700, 5),
            "hold_days": (1, 60, 1),
        },
        desc="future_quant RIABreak 的日盘时序腿：09:40-10:40 窗口内，收盘相对当日"
             "开盘的偏离超过 0.4×前一日振幅就顺向开仓，一天最多一笔；5×ATR 止损 + "
             "定时平（有夜盘商品当晚 21:05、无夜盘商品次日 09:05、金融期货次日 09:35，"
             "遇节假日顺延到下一个开盘日的 09:05）。",
    ),
    "ria_break_night": StrategySpec(
        key="ria_break_night",
        label="RIA 波幅突破·夜盘",
        func=strat_ria_break_night,
        params={"ratio": 0.40, "atr_n": 20, "stop_mult": 5.0,
                "night_start": 1270, "night_end": 1330, "hold_days": 20},
        param_meta={
            "ratio": (0.1, 1.5, 0.05),
            "atr_n": (5, 60, 1),
            "stop_mult": (2.0, 12.0, 0.5),
            "night_start": (1260, 1320, 5),
            "night_end": (1300, 1380, 5),
            "hold_days": (1, 60, 1),
        },
        desc="future_quant RIABreak 的夜盘时序腿：21:10-22:10 窗口内，收盘相对当日"
             "开盘的偏离超过 0.4×前一日振幅就顺向开仓，一夜最多一笔；5×ATR 止损 + "
             "次日 09:05 定时平。无夜盘品种不会开仓。",
    ),
    "cta_break": StrategySpec(
        key="cta_break",
        label="CTA 波幅突破·检查点",
        func=strat_cta_break,
        params={"ratio": 0.40, "check_min": 640, "atr_n": 20,
                "stop_mult": 5.0, "exit_min": 575, "hold_bars": 96},
        param_meta={
            "ratio": (0.1, 1.5, 0.05),
            "check_min": (540, 1380, 5),
            "atr_n": (5, 60, 1),
            "stop_mult": (2.0, 12.0, 0.5),
            "exit_min": (540, 600, 5),
            "hold_bars": (24, 480, 12),
        },
        desc="future_quant cta_short 的规则时序：一天只看一个检查点（默认 10:40），"
             "偏离超过 0.4×前一日振幅就顺向开仓；5×ATR 止损 + 次日 09:35 定时平。"
             "把确认时刻改成 21:30、平仓改成 09:05 就是夜盘版。",
    ),
    # ---- A 股选股型（只做多、只吃日线）----
    "a_vol_breakout": StrategySpec(
        key="a_vol_breakout",
        label="A股·放量启动",
        func=strat_a_vol_breakout,
        params={"max_price": 80.0, "gain_lo": 0.0, "gain_hi": 9.8,
                "vol_mult": 3.0, "amt_min": 5000.0,
                "pe_min": 0.0, "pe_max": 25.0, "hold_days": 5},
        param_meta={
            "max_price": (5, 200, 5),
            "gain_lo": (0, 5, 0.5),
            "gain_hi": (5, 20, 0.2),
            "vol_mult": (1.5, 10.0, 0.5),
            "amt_min": (1000, 50000, 500),
            "pe_min": (0, 20, 1),
            "pe_max": (5, 100, 5),
            "hold_days": (1, 30, 1),
        },
        bidirectional=False,        # A 股融券做空不现实，只做多
        markets=("cn_stock",),      # 只能跑 A 股日线，别摆在期货的下拉里
        param_labels={"hold_days": "持有(根)"},
        desc="八条同时满足才买：非 ST（按**当时**的曾用名判）、非北交所、昨日未涨停、"
             "股价低于上限、今日涨幅在上限内、今日成交量≥昨日的 N 倍、PE(TTM) 在上限内、"
             "成交额达标。当日收盘出信号，**下一根开盘买**，持有 N 根后开盘卖。"
             "只能跑 A 股日线（需要 pct_chg / pe_ttm / is_st / board_code 这几列）。",
    ),
}


def run_strategy(df: pd.DataFrame, key: str, params: dict | None = None,
                 reverse: bool = False) -> pd.DataFrame:
    """跑一条内置规则策略，返回带 `signal` 列的 df。

    `reverse=True` 把信号整体取反（多 ↔ 空、空 ↔ 多、空仓不变）。这件事必须
    在**这里**做，不能放在界面里乘一下：研究页、组合回放、实盘三条路都要同一条
    曲线，界面里乘的话实盘还是按原方向下单。
    """
    spec = STRATEGIES[key]
    p = dict(spec.params)
    if params:
        p.update({k: v for k, v in params.items() if k in p})
    out = spec.func(df, **p)
    if reverse:
        out["signal"] = -out["signal"].astype(int)
        # K 线内那一族的多空方向藏在触发价 / 出场标志里，`-signal` 翻不动它们。
        # 不换的话「反向」只翻了个没人读的展示列，回测还是按原方向成交 ——
        # 这种错不会报错，只会让研究页的「反向」按钮静默失效
        for a, b in (("entry_long", "entry_short"), ("exit_long", "exit_short")):
            if a not in out.columns and b not in out.columns:
                continue
            va = (out[a].to_numpy(dtype=float) if a in out.columns
                  else np.full(len(out), np.nan))
            vb = (out[b].to_numpy(dtype=float) if b in out.columns
                  else np.full(len(out), np.nan))
            out[a], out[b] = vb, va
    return out


def next_levels(df: pd.DataFrame, key: str, params: dict | None = None,
                reverse: bool = False) -> dict:
    """**正在走的这根** K 线的 K 线内参数：触发价 / 止损距离 / 出场标志。

    实盘只能拿已收盘的 bar 算指标，却要在还没走完的这根里判断价格有没有碰到
    触发价。所以这里把「正在走的这根」（`df` 的最后一行）一起算进去：

    - `entry_*` / `stop_dist` 取**最后一行**。策略里这些量全部走 `shift(1)`
      或「上一根 ATR」，最后一行拿到的正是「用已收盘数据算出来、给当前这根
      用」的那组值；
    - `exit_*` 取**倒数第二行**。那两列的含义是「这根收盘决定、下一根开盘
      执行」，所以当前这根要执行的是**上一根**留下的标志。

    `df` 最后一行是正在走的 bar，它的 high/low 还不完整 —— 对触发价没影响
    （`shift(1)` 已经把它排除），但 `day_open` 要用它：一天第一根 K 线开盘
    那一刻，「当日开盘价」就是它自己，拿上一交易日的收盘去凑就是错的。

    返回空 dict 表示这条策略不支持 K 线内成交（`spec.intrabar` 为假）。
    """
    spec = STRATEGIES.get(str(key or ""))
    if spec is None or not getattr(spec, "intrabar", False):
        return {}
    if df is None or len(df) < 2:
        return {}
    p = dict(spec.params)
    if params:
        p.update({k: v for k, v in params.items() if k in p})
    out = spec.func(df, **p)
    if reverse:
        for a, b in (("entry_long", "entry_short"), ("exit_long", "exit_short")):
            va = out[a].to_numpy(dtype=float).copy()
            vb = out[b].to_numpy(dtype=float).copy()
            out[a], out[b] = vb, va

    def num(row, name: str) -> float:
        try:
            v = float(row.get(name, float("nan")))
        except (TypeError, ValueError):
            return float("nan")
        return v if math.isfinite(v) else float("nan")

    last, prev = out.iloc[-1], out.iloc[-2]
    return {
        "entry_long": num(last, "entry_long"),
        "entry_short": num(last, "entry_short"),
        "stop_dist": num(last, "stop_dist"),
        "exit_long": bool(num(prev, "exit_long") > 0.5),
        "exit_short": bool(num(prev, "exit_short") > 0.5),
    }


_MINE_TAIL = re.compile(r"\s*[·•・\-—_]\s*(我的|内置)\s*$")

def _norm(s: str) -> str:
    t = _MINE_TAIL.sub("", str(s or "").strip())
    t = unicodedata.normalize("NFKC", t)
    for tail in ("（我的）", "(我的)", "（内置）", "(内置)"):
        t = t.replace(tail, "")
    return re.sub(r"\s+", "", t).lower()


def resolve_key(name: str) -> str:
    """把用户或模型写的策略名换成 `STRATEGIES` 里的 key，换不出来返回空串。

    三种写法都认：代号（`ma_cross`）、显示名（`双均线交叉`）、以及左侧列表里
    带归属尾巴的显示名（`双均线交叉 · 内置`）。后者是渲染时加的，不是名字的
    一部分 —— 模型从截图里认名字，看到的正是带尾巴的那一版。

    **策略库里的模板不走这里**：那是一整份配置（品种、周期、参数、手数都定了），
    不是一条规则策略，用 `strategy_store.find_library` 找。
    """
    raw = str(name or "").strip()
    if not raw:
        return ""
    if raw in STRATEGIES:
        return raw
    want = _norm(raw)
    if not want:
        return ""
    for k, spec in STRATEGIES.items():
        if _norm(k) == want or _norm(spec.label) == want:
            return k
    return ""


def available_text(limit: int = 8) -> str:
    """可用策略的一行摘要，报错时带上，让模型/用户下一轮能自己改对。"""
    items = [f"{k}（{s.label}）" for k, s in list(STRATEGIES.items())[:limit]]
    more = len(STRATEGIES) - len(items)
    return "、".join(items) + (f" 等 {len(STRATEGIES)} 条" if more > 0 else "")


# ---------------------------------------------------------------- 用户自定义策略
def _load_user_strategies() -> int:
    """把用户目录里的自定义策略挂进来。

    和 `factors.py` 同一个理由放在文件**末尾**：`userstrategies` 要
    `from ttrade.strategies import StrategySpec`，这时本模块已经执行到底。
    失败一律吞掉 —— 用户手改 JSON 改坏了，不该让程序起不来。
    """
    try:
        from ttrade import userstrategies as US

        return US.register_all_into(STRATEGIES)
    except Exception:  # noqa: BLE001
        return 0


USER_STRATEGY_COUNT = _load_user_strategies()
