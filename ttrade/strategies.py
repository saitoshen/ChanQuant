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
    # 用户注册的策略自带源码：`exec` 出来的函数没有源文件，`inspect.getsource()`
    # 取不到，库页要展示代码就只能存一份
    code: str = ""


# 参数的中文名。研究页的滑杆、组合页的「设置参数」、手册都从这里取 ——
# 同一个 `fast` 在两个界面叫两个名字，用户会以为是两个不同的东西
PARAM_LABELS = {
    "fast": "快线周期", "slow": "慢线周期", "signal_n": "信号周期",
    "entry": "入场周期", "exit": "离场周期", "n": "通道周期",
    "k": "ATR 倍数", "atr_n": "ATR 周期", "atr_mult": "止损倍数",
}


STRATEGIES: dict[str, StrategySpec] = {
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
}


def run_strategy(df: pd.DataFrame, key: str, params: dict | None = None) -> pd.DataFrame:
    spec = STRATEGIES[key]
    p = dict(spec.params)
    if params:
        p.update({k: v for k, v in params.items() if k in p})
    return spec.func(df, **p)


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
