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

    「当日」按**交易日**算，不是自然日：`index + 6h` 再取日期，夜盘 21:00 之后
    的 K 线就归到第二个交易日了。不这么切的话，晚上九点的开盘价会和早上九点的
    算成同一天，波幅基准整个错位。
    """
    out = df.copy()
    c, h, l, o = out["close"], out["high"], out["low"], out["open"]

    hi = c.rolling(entry, min_periods=entry).max().shift(1)
    lo = c.rolling(entry, min_periods=entry).min().shift(1)
    hi_x = c.rolling(exit_n, min_periods=exit_n).max().shift(1)
    lo_x = c.rolling(exit_n, min_periods=exit_n).min().shift(1)

    day_key = (out.index + pd.Timedelta(hours=6)).normalize().to_numpy()
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
    - 这一条在价格**碰到触发价的那一刻**就成交 —— 突破 20 根箱体上沿、或者
      相对当日开盘涨过昨日波幅的 40%，谁先满足谁触发。

    ## 入场

    两个条件都表示「价格向上越过某个价位」：

        箱体突破    价格 > 前 entry_n 根的最高价
        波幅突破    价格 > 当日开盘 + vol_ratio × 昨日高低波幅

    价格越过**两者中更高的那个**，两个条件必然同时成立，所以触发价取

        做多  up = max(前 entry_n 根最高, 当日开盘 + vol_ratio × 昨日波幅)
        做空  dn = min(前 entry_n 根最低, 当日开盘 - vol_ratio × 昨日波幅)

    取 max 而不是「两个条件都成立才算」，是因为后者只能收盘才知道成没成 ——
    那又退回 `sbbox_mini` 了。用户要的是「K 线突破的那个价格进场」。

    ## 止损

    `stop_mult` × ATR(`atr_n`)，绑在**成交价**上，K 线内碰到就按止损价成交。
    ATR 周期默认 75（用户给的区间 50~100 的中点偏上：短周期噪音多，取长一点
    止损不容易被一根毛刺扫掉）。

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

    「当日」按**交易日**算（`index + 6h` 再取日期），夜盘 21:00 之后归次日 ——
    晚上九点的开盘价要是和早上九点算成同一天，波幅基准整个错位。
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
    day_key = (pd.DatetimeIndex(out.index) + pd.Timedelta(hours=6)).normalize()
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
                if l_a[i] <= stop:      # 同根先入场后止损，按止损价成交
                    pos, stop = 0, np.nan
            elif su and not lu:
                fill = min(o_a[i], dn[i])
                pos = -1
                stop = fill + sd_a[i]
                if h_a[i] >= stop:
                    pos, stop = 0, np.nan
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

    经验规则：整体加 6 小时再取日期。期货交易日从**夜盘 21:00** 开始，不是
    自然日 00:00：21:30 那根属于第二天，凌晨 02:00 那根也属于第二天。
    21:30+6h=次日 03:30，02:00+6h=当日 08:00，白天各时段都在自己那天；
    无夜盘品种（只有 09:00-15:00）加不加都是同一天，不受影响。

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
    # 是否按「K 线内触发价」成交（回测走 `run_intrabar_backtest`，实盘走
    # `PortfolioRunner._intrabar_tick`）。为真时策略必须输出 `entry_long` /
    # `entry_short` / `stop_dist` / `exit_long` / `exit_short` 这几列 ——
    # 只给 `signal` 的旧策略一律保持 False，走「收盘定信号、下一根开盘成交」
    intrabar: bool = False
    # 用户注册的策略自带源码：`exec` 出来的函数没有源文件，`inspect.getsource()`
    # 取不到，库页要展示代码就只能存一份
    code: str = ""


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
    "ratio": "波幅倍数", "check_min": "确认(分)", "hold_days": "持仓(天)",
    "hold_bars": "持仓(根)",
    "day_start": "日窗起(分)", "day_end": "日窗止(分)",
    "night_start": "夜窗起(分)", "night_end": "夜窗止(分)",
}


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
                "range_n": 1, "atr_n": 75, "stop_mult": 1.5},
        param_meta={
            "entry_n": (5, 240, 5),
            "exit_n": (10, 480, 10),
            "vol_ratio": (0.1, 1.5, 0.05),
            "range_n": (1, 10, 1),
            "atr_n": (50, 100, 5),
            "stop_mult": (0.5, 5.0, 0.5),
        },
        desc="K 线内成交：价格碰到「20 根箱体上沿」或「当日开盘 ± 昨幅×40%」"
             "更高的那个就进，1.5×ATR(75) 盘中止损；收盘反向突破 60 根收盘"
             "箱体则下一根开盘平。与 sbbox_mini 条件相近，但不等收盘。",
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
