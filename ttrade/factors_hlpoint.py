"""HL-Point 的 36 个 1h 因子（从 ``future_quant/strategy/hl_point`` 移植）。

## 来源

``D:/CozeTask/future_quant/strategy/hl_point/`` 那套策略，真源是
``tools_ml.py`` 的常量区：

```python
FEATS = [...]        # 27 个价格特征
OI_FEATS = [...]     #  5 个持仓量特征
ROLL_FEATS = [...]   #  4 个展期特征
FUSE = FEATS + EXTRA_FEATS      # 27 + 5 + 4 = 36
```

⚠️ **别和 ``cross_section/`` 那套搞混**：那是日频 59 个量价因子 + 77 列 Rust
（本仓库的 ``ttrade/factors_hl.py`` 移的是那一套）。两套都叫「HL point」，
但周期、因子、建模范式完全不同。本模块只负责 **1h / 36 因子** 这一套。

## 三十六因子与窗口（全部照抄 ``HLConfig`` 默认值）

| 类别 | 个数 | 因子 | 关键窗口 |
|---|---|---|---|
| 多均线偏离度 | 5 | ``dev_z_20/60/120/agg``、``dev_spread`` | ``ma=(20,60,120)``、``dev_zwin=250``、``atr=14`` |
| 影线 | 8 | ``up_sh``、``dn_sh``、``up_sh_atr``、``dn_sh_atr``、``up_sh_int``、``dn_sh_int``、``shadow_high_z``、``shadow_low_z`` | ``shadow_lookback=20``、``shadow_zwin=250`` |
| 趋势结构 | 7 | ``slope_atr``、``r2``、``slope_chg_atr``、``dist_hi``、``dist_lo``、``ext_hi``、``ext_lo`` | ``trend_lookback=60``、``ext_lookback=250`` |
| 通道 | 1 | ``ch_slope_atr`` | ``channel_lookback=30``、``channel_k=2.0`` |
| 关键位 | 6 | ``hit_high``、``hit_low``、``clus_high``、``clus_low``、``level_high``、``level_low`` | ``pivot_left=right=3``、``recent_m=6``、``band_tol_atr=0.75``、``band_min_touch=2``、``level_hit_atr=1.0`` |
| 持仓量 | 5 | ``oi_chg_5``、``oi_chg_20``、``oi_z_60``、``oi_ratio_60``、``oi_px_sync_5`` | ``60`` 根 |
| 展期 | 4 | ``roll``、``roll_ann``、``curv``、``roll_chg_20`` | ``20`` 交易日 |

## 三条必须守住的口径

1. **ATR 是 Wilder ATR**（``tr.ewm(alpha=1/n, adjust=False, min_periods=n)``），
   不是 ``tr.rolling(n).mean()``。影线 / 斜率 / 距离类因子全部除以它做归一，
   换成简单均值会让全部 27 个价格因子的量纲一起漂。
2. **``rolling_z`` 用 ``std(ddof=0)`` 且 ``min_periods=满窗``**。唯一的例外是
   ``oi_z_60``，原项目用的是 pandas 默认 ``ddof=1`` —— 照抄，不要「顺手统一」。
3. **关键位因子必须按「确认时点」可见**。``pivot_events`` 用
   ``rolling(center=True)`` 找分形高低点（这本身是对的，分形定义就需要右侧），
   但**只有 ``confirm_pos = bar_pos + right`` 之后的 bar 才准用**。少了这一步
   就是最典型的未来函数：当下的 pivot 价格里含着未来 3 根 K 线的最高价。

## 与库里已有因子的去重

``roll`` / ``roll_ann`` / ``curv`` 三个**不重复注册**，直接映射到库里的主力口径：

| HL-Point | 库里已有 | 差异 |
|---|---|---|
| ``roll`` | ``roll_main_oi`` | 库里多一道 5 日均值（换腿当天有台阶） |
| ``roll_ann`` | ``roll_main_annual`` | 同上，且 365/(dmonth×30) 只差 1.39% 常数，z-score 后等价 |
| ``curv`` | ``roll_main_curv`` | 同上 |
| ``roll_chg_20`` | — | **新建**：库里 ``roll_main_change`` 差分的是**年化值**，这里是**价差率**本身 |

``ext_hi`` / ``ext_lo`` 是**绝对价格水平量**（250 根内的最高/最低价），和
``tools_ml.py`` 里「只用已归一化特征，避免跨品种价格水平泄漏」的注释自相矛盾。
照实注册以便与原文 1:1 对照，但建库脚本默认**不勾选**它们 —— 截面 z-score
会把绝对价格变成「本品种在池中的价格排名」，等于塞了一个品种哑变量。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
# 窗口参数（= HLConfig 默认值，改这里不要改函数体）
# ══════════════════════════════════════════════════════════════════════════
ATR_N = 14
MA_PERIODS = (20, 60, 120)
DEV_ZWIN = 250
SHADOW_LOOKBACK = 20
SHADOW_ZWIN = 250
TREND_LOOKBACK = 60
CHANNEL_LOOKBACK = 30
#: ``ext_lookback`` 在 ``build_features`` 里是
#: ``max(cfg.level_lookback, cfg.trend_lookback) = max(250, 60) = 250``
EXT_LOOKBACK = 250
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
RECENT_M = 6
BAND_TOL_ATR = 0.75
BAND_MIN_TOUCH = 2
LEVEL_HIT_ATR = 1.0


# ══════════════════════════════════════════════════════════════════════════
# 基础工具（自包含，不 import factors.py —— 那边末尾会反向 import 本模块）
# ══════════════════════════════════════════════════════════════════════════
def _nan(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index, dtype=float)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """取衍生数据列，没有就整列 NaN（``filter_available`` 会自动剔掉）。"""
    if name not in df.columns:
        return _nan(df)
    return pd.to_numeric(df[name], errors="coerce").astype(float)


def _oi(df: pd.DataFrame) -> pd.Series:
    """持仓量。日线 CSV 叫 ``open_interest``，天勤分钟线叫 ``open_oi``。"""
    for name in ("open_interest", "open_oi"):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").astype(float)
    return _nan(df)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def _truth(df: pd.DataFrame) -> pd.Series:
    """Wilder ATR（和 ``indicators.atr`` 逐位一致）。"""
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / ATR_N, adjust=False, min_periods=ATR_N).mean()


def _rz(s: pd.Series, win: int, ddof: int = 0) -> pd.Series:
    """滚动 z（``min_periods = 满窗``）。``ddof=0`` 是 ``rolling_z`` 的口径。"""
    m = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std(ddof=ddof)
    return _safe_div(s - m, sd)


def _norm01(x, cap: float = 1.0) -> np.ndarray:
    """线性压到 0~1，``cap`` 处封顶，NaN → 0（= ``indicators.norm01``）。"""
    arr = np.asarray(x, dtype=float)
    out = np.clip(arr / float(cap), 0.0, 1.0)
    return np.where(np.isfinite(out), out, 0.0)


def _ols(y: pd.Series, L: int):
    """滚动 OLS ``y ~ a + b·x``（x = 0..L-1）。

    返回 ``(slope, intercept, resid_std, r2)``，前 ``L-1`` 个为 NaN。
    含 NaN 的窗口整窗置 NaN（原项目口径，不做插补）。

    为什么按 6.5 万根分块：``sliding_window_view`` 本身不拷贝，但后面几个
    ``.mean(axis=1)`` / 残差运算会 materialize 出 ``(n, L)`` 的矩阵。5m 上
    30 万根 × 60 列 = 1.4 亿 float，直接顶掉内存。
    """
    v = np.asarray(y, dtype=float)
    n = len(v)
    slope = np.full(n, np.nan)
    inter = np.full(n, np.nan)
    rstd = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    L = int(L)
    if n < L or L < 2:
        return slope, inter, rstd, r2

    x = np.arange(L, dtype=float)
    xm = x.mean()
    xc = x - xm
    den = float((xc ** 2).sum())
    step = 65536
    for a in range(0, n - L + 1, step):
        b = min(a + step, n - L + 1)
        seg = v[a:b + L - 1]
        sw = np.lib.stride_tricks.sliding_window_view(seg, L)
        bad = ~np.isfinite(sw).all(axis=1)
        Y = np.where(np.isfinite(sw), sw, 0.0)
        bb = (Y @ xc) / den
        ybar = Y.mean(axis=1)
        aa = ybar - bb * xm
        resid = Y - (aa[:, None] + bb[:, None] * x)
        ss_res = (resid ** 2).sum(axis=1)
        ss_tot = ((Y - ybar[:, None]) ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = 1.0 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)
        w = ~bad
        # 🔴 下标必须 +L-1：`sw[j]` 是「从 a+j **开始**、长 L」的窗口，它描述的是
        # 第 a+j+L-1 根（窗口**结束**处）的状态。写成 `slope[a:b]` 就等于每个
        # 斜率都偷了未来 L−1 根 —— 实测 slope_atr 的横截面 IC 会从 ~0.01 飙到
        # **0.64**（ch_slope_atr 0.75），整条净值凭空多赚 239%。
        sl = slice(a + L - 1, b + L - 1)
        slope[sl] = np.where(w, bb, np.nan)
        inter[sl] = np.where(w, aa, np.nan)
        rstd[sl] = np.where(w, np.sqrt(ss_res / L), np.nan)
        r2[sl] = np.where(w, rr, np.nan)
    return slope, inter, rstd, r2


def _trade_day(df: pd.DataFrame) -> pd.Series:
    """每根 bar 属于哪个交易日（夜盘 21:00 归次日）。"""
    idx = pd.DatetimeIndex(df.index)
    if "trading_day" in df.columns:
        d = pd.to_datetime(df["trading_day"], errors="coerce")
        if d.notna().any():
            nat = d.isna()
            if nat.any():
                guess = pd.Series(idx.normalize(), index=df.index)
                guess = guess.where(idx.hour < 20, guess + pd.Timedelta(days=1))
                d = d.fillna(guess)
            return d
    return pd.Series(idx.normalize(), index=df.index)


def _day_diff(df: pd.DataFrame, s: pd.Series, n: int = 20) -> pd.Series:
    """按**交易日**取 n 天差分，再映射回每根 bar。

    ``roll`` 这类衍生量本身「一天一个快照」，``auxdata.attach_intraday`` 把它
    ``shift(1)`` 之后**广播到当天每根 bar**。所以 ``groupby(day).last()`` 取到的
    不是「当天最后一根的未来值」，而是本来就已知的那个日频快照 —— 前提是**入参
    在同一个交易日内恒定**。1h 上 ``shift(20)`` 只有 3.3 天，必须压日频才不会
    口径漂掉。

    🔴 推论：如果把一个**逐根变化**的序列喂进来（比如把价差当分钟量算），
    ``.last()`` 就会取到当天收盘的值 —— 那才是未来函数。这里的调用方
    （``f_roll_chg_20``）只喂 ``aux`` 的日频快照列。
    """
    idx = s.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return s.diff(n)
    day = _trade_day(df)
    if len(day) == len(day.unique()):
        return s.diff(n)
    daily = s.groupby(day).last()
    return pd.Series(np.asarray(day.map(daily.diff(n)), dtype=float), index=idx)


# ══════════════════════════════════════════════════════════════════════════
# PIT 分形高低点（关键位因子的地基）
# ══════════════════════════════════════════════════════════════════════════
def _pivot_events(df: pd.DataFrame, left: int = PIVOT_LEFT, right: int = PIVOT_RIGHT):
    """分形高低点事件表：``(confirm_pos, price, kind)``，按 ``confirm_pos`` 升序。

    🔴 **PIT 的全部秘密在 ``confirm_pos = bar_pos + right``**：
    第 ``pos`` 根是不是分形高点，要用 ``pos+1 .. pos+right`` 这几根的最高价才判得出来，
    所以这个信息最早在 ``pos+right`` 收盘时才可见。拿 ``bar_pos`` 当可见时点
    就是把未来 3 根 K 线的最高价塞进今天的因子。

    这个函数本身允许用 ``center=True``（分形定义决定的），但返回的可见时点是
    ``confirm_pos``，调用方只准用它。
    """
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    n = len(df)
    L = int(left) + int(right) + 1
    hh = pd.Series(h).rolling(L, center=True, min_periods=L).max().values
    ll = pd.Series(l).rolling(L, center=True, min_periods=L).min().values
    with np.errstate(invalid="ignore"):
        is_ph = (h >= hh) & np.isfinite(hh)
        is_pl = (l <= ll) & np.isfinite(ll)

    pos, val, kind = [], [], []
    for raw_pos in np.flatnonzero(is_ph):
        c = int(raw_pos) + int(right)
        if c < n:
            pos.append(c)
            val.append(float(h[raw_pos]))
            kind.append("H")
    for raw_pos in np.flatnonzero(is_pl):
        c = int(raw_pos) + int(right)
        if c < n:
            pos.append(c)
            val.append(float(l[raw_pos]))
            kind.append("L")
    if not pos:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype="<U1")
    order = np.argsort(np.asarray(pos), kind="stable")
    return (np.asarray(pos, dtype=int)[order],
            np.asarray(val, dtype=float)[order],
            np.asarray(kind)[order])


def _step_values(n: int, event_pos: np.ndarray, event_val: np.ndarray, back: int) -> np.ndarray:
    """第 ``i`` 根 bar 上「倒数第 ``back`` 个**已确认**事件」的值；不足则 NaN。

    用 ``cumsum(bincount(...))`` 拿到「到 i 为止已确认了几个」，再减 ``back``
    就是事件列表里的下标 —— 全程向量化，30 万根也不用循环。
    """
    out = np.full(n, np.nan)
    if len(event_pos) == 0:
        return out
    m = np.cumsum(np.bincount(event_pos, minlength=n))[:n]
    idx = m - int(back)
    ok = idx >= 0
    if ok.any():
        out[ok] = event_val[idx[ok]]
    return out


def _pivot_frame(df: pd.DataFrame, m: int = RECENT_M):
    """``last_ph/last_pl`` + ``ph_1..ph_M``/``pl_1..pl_M``（全是 PIT 可见值）。"""
    pos, val, kind = _pivot_events(df)
    n = len(df)
    hp, hv = pos[kind == "H"], val[kind == "H"]
    lp, lv = pos[kind == "L"], val[kind == "L"]
    last_ph = _step_values(n, hp, hv, 1)
    last_pl = _step_values(n, lp, lv, 1)
    PH = np.column_stack([_step_values(n, hp, hv, j) for j in range(1, m + 1)])
    PL = np.column_stack([_step_values(n, lp, lv, j) for j in range(1, m + 1)])
    return last_ph, last_pl, PH, PL


# ══════════════════════════════════════════════════════════════════════════
# 一、多均线偏离度（5）
# ══════════════════════════════════════════════════════════════════════════
def _dev_block(df: pd.DataFrame, a: pd.Series):
    """``dev_k = (close − MA_k)/ATR`` 再各自滚动 z。"""
    close = df["close"].astype(float)
    devs, zs = {}, []
    for k in MA_PERIODS:
        ma = close.rolling(int(k), min_periods=int(k)).mean()
        d = _safe_div(close - ma, a)
        devs[k] = d
        zs.append(_rz(d, DEV_ZWIN))
    agg = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
    spread = devs[MA_PERIODS[0]] - devs[MA_PERIODS[-1]]
    return {f"dev_z_{k}": z for k, z in zip(MA_PERIODS, zs)}, agg, spread


def f_dev_z_20(df: pd.DataFrame) -> pd.Series:
    a = _truth(df)
    return _dev_block(df, a)[0]["dev_z_20"]


def f_dev_z_60(df: pd.DataFrame) -> pd.Series:
    a = _truth(df)
    return _dev_block(df, a)[0]["dev_z_60"]


def f_dev_z_120(df: pd.DataFrame) -> pd.Series:
    a = _truth(df)
    return _dev_block(df, a)[0]["dev_z_120"]


def f_dev_z_agg(df: pd.DataFrame) -> pd.Series:
    """三条均线偏离度 z 的等权均值 —— HL 报告里重要性 30.5%，排第一。"""
    a = _truth(df)
    return _dev_block(df, a)[1]


def f_dev_spread(df: pd.DataFrame) -> pd.Series:
    """近端均线减远端均线的原始偏离度差（不取 z）。

    >0 表示短均线比长均线更「张嘴」，趋势末端的喇叭口形态。
    """
    a = _truth(df)
    return _dev_block(df, a)[2]


# ══════════════════════════════════════════════════════════════════════════
# 二、上下影线（8）
# ══════════════════════════════════════════════════════════════════════════
def _shadow_block(df: pd.DataFrame, a: pd.Series):
    h, l, o, c = (df[k].astype(float) for k in ("high", "low", "open", "close"))
    rng = (h - l)
    rng = rng.where(rng > 0)
    body_hi = pd.concat([o, c], axis=1).max(axis=1)
    body_lo = pd.concat([o, c], axis=1).min(axis=1)
    up_sh = (h - body_hi) / rng
    dn_sh = (body_lo - l) / rng
    up_atr = _safe_div(h - body_hi, a)
    dn_atr = _safe_div(body_lo - l, a)
    mp = max(3, SHADOW_LOOKBACK // 2)
    up_int = up_sh.rolling(SHADOW_LOOKBACK, min_periods=mp).mean()
    dn_int = dn_sh.rolling(SHADOW_LOOKBACK, min_periods=mp).mean()
    return {
        "up_sh": up_sh,
        "dn_sh": dn_sh,
        "up_sh_atr": up_atr,
        "dn_sh_atr": dn_atr,
        "up_sh_int": up_int,
        "dn_sh_int": dn_int,
        "shadow_high_z": _rz(up_int, SHADOW_ZWIN),
        "shadow_low_z": _rz(dn_int, SHADOW_ZWIN),
    }


def _shadow(df: pd.DataFrame, key: str) -> pd.Series:
    return _shadow_block(df, _truth(df))[key]


def f_up_sh(df: pd.DataFrame) -> pd.Series:
    """上影线占当根振幅比例，0~1。"""
    return _shadow(df, "up_sh")


def f_dn_sh(df: pd.DataFrame) -> pd.Series:
    """下影线占当根振幅比例，0~1。"""
    return _shadow(df, "dn_sh")


def f_up_sh_atr(df: pd.DataFrame) -> pd.Series:
    """上影线绝对长度 / ATR —— 捕捉「实体很小但上影极长」的针形。"""
    return _shadow(df, "up_sh_atr")


def f_dn_sh_atr(df: pd.DataFrame) -> pd.Series:
    """下影线绝对长度 / ATR。"""
    return _shadow(df, "dn_sh_atr")


def f_up_sh_int(df: pd.DataFrame) -> pd.Series:
    """20 根内上影线强度的均值（「上涨过程中反复被打头」）。"""
    return _shadow(df, "up_sh_int")


def f_dn_sh_int(df: pd.DataFrame) -> pd.Series:
    """20 根内下影线强度的均值。"""
    return _shadow(df, "dn_sh_int")


def f_shadow_high_z(df: pd.DataFrame) -> pd.Series:
    """上影线强度的滚动 z（250 根窗口）。"""
    return _shadow(df, "shadow_high_z")


def f_shadow_low_z(df: pd.DataFrame) -> pd.Series:
    """下影线强度的滚动 z（250 根窗口）。"""
    return _shadow(df, "shadow_low_z")


# ══════════════════════════════════════════════════════════════════════════
# 三、趋势结构（7）+ 通道（1）
# ══════════════════════════════════════════════════════════════════════════
def _trend_block(df: pd.DataFrame, a: pd.Series):
    close = df["close"].astype(float)
    slope, _i, _r, r2 = _ols(close, TREND_LOOKBACK)
    slope = pd.Series(slope, index=df.index)
    half = max(3, TREND_LOOKBACK // 2)
    s_prev = pd.Series(slope, index=df.index).shift(half)
    rmax = df["high"].astype(float).rolling(EXT_LOOKBACK, min_periods=EXT_LOOKBACK).max()
    rmin = df["low"].astype(float).rolling(EXT_LOOKBACK, min_periods=EXT_LOOKBACK).min()
    return {
        # slope × 窗口长度 / ATR = 「这段趋势一共走了几个 ATR」，无量纲
        "slope_atr": _safe_div(slope * TREND_LOOKBACK, a),
        "r2": pd.Series(r2, index=df.index),
        "slope_chg_atr": _safe_div((slope - s_prev) * half, a),
        "dist_hi": _safe_div(close - rmax, a),      # <=0，0 = 正在 250 根最高
        "dist_lo": _safe_div(close - rmin, a),      # >=0，0 = 正在 250 根最低
        "ext_hi": rmax,                              # ⚠️ 绝对价格水平量
        "ext_lo": rmin,
    }


def _trend(df: pd.DataFrame, key: str) -> pd.Series:
    return _trend_block(df, _truth(df))[key]


def f_slope_atr(df: pd.DataFrame) -> pd.Series:
    """60 根 OLS 斜率的「窗口总涨幅」折成 ATR 倍数（>0 上行）。"""
    return _trend(df, "slope_atr")


def f_r2(df: pd.DataFrame) -> pd.Series:
    """60 根回归的 R²，趋势「干净」程度。"""
    return _trend(df, "r2")


def f_slope_chg_atr(df: pd.DataFrame) -> pd.Series:
    """斜率的变化量（正 = 加速，负 = 拐头），HL 报告重要性 15.8%。"""
    return _trend(df, "slope_chg_atr")


def f_dist_hi(df: pd.DataFrame) -> pd.Series:
    """收盘距 250 根最高价的 ATR 距离，0 = 正在阶段最高。"""
    return _trend(df, "dist_hi")


def f_dist_lo(df: pd.DataFrame) -> pd.Series:
    """收盘距 250 根最低价的 ATR 距离，0 = 正在阶段最低。"""
    return _trend(df, "dist_lo")


def f_ext_hi(df: pd.DataFrame) -> pd.Series:
    """250 根内最高价（绝对价格）。

    ⚠️ 原项目把它列进「已归一化特征」是不准确的 —— 这是价格水平量。截面
    z-score 后会退化成「本品种在池中的价格排名」，等于一个品种哑变量。
    为与原文 1:1 对照才注册；建库脚本默认不勾选。
    """
    return _trend(df, "ext_hi")


def f_ext_lo(df: pd.DataFrame) -> pd.Series:
    """250 根内最低价（绝对价格）。同 ``ext_hi`` 的注意事项。"""
    return _trend(df, "ext_lo")


def f_ch_slope_atr(df: pd.DataFrame) -> pd.Series:
    """30 根回归通道的斜率 / ATR（每根 bar 走几个 ATR）。

    和 ``slope_atr`` 不是重复：那个是 60 根、且乘了窗口长度（看总涨幅）；
    这个是 30 根、单根速度（看当下斜率陡不陡）。
    """
    slope, _i, _r, _r2 = _ols(df["close"].astype(float), CHANNEL_LOOKBACK)
    return _safe_div(pd.Series(slope, index=df.index), _truth(df))


# ══════════════════════════════════════════════════════════════════════════
# 四、关键位（6）
# ══════════════════════════════════════════════════════════════════════════
def _level_block(df: pd.DataFrame, a: pd.Series):
    last_ph, last_pl, PH, PL = _pivot_frame(df, RECENT_M)
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    av = a.values
    # 预热期「一个 pivot 都还没确认」时整行是 NaN，nanmin / nansum 会各报一条
    # RuntimeWarning。这是正常现象不是 bug，静音掉免得刷屏掩盖真问题。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            d_hi = np.nanmin(np.abs(PH - high[:, None]), axis=1)
            d_lo = np.nanmin(np.abs(PL - low[:, None]), axis=1)
        hit_high = d_hi / av
        hit_low = d_lo / av

        tol = (BAND_TOL_ATR * av)[:, None]
        with np.errstate(invalid="ignore"):
            # 全 NaN 行在预热期是正常的：nansum 会给 0，正好表示「没有同带触点」
            clus_h = np.nansum(np.abs(PH - last_ph[:, None]) <= tol, axis=1)
            clus_l = np.nansum(np.abs(PL - last_pl[:, None]) <= tol, axis=1)

    # 贴线 60% + 聚集 40%（原项目 weight，不要改比例）
    hit_part_h = 1.0 - _norm01(np.where(np.isfinite(hit_high), hit_high, np.inf), LEVEL_HIT_ATR)
    hit_part_l = 1.0 - _norm01(np.where(np.isfinite(hit_low), hit_low, np.inf), LEVEL_HIT_ATR)
    clus_part_h = _norm01(np.maximum(clus_h - 1, 0), max(1, BAND_MIN_TOUCH))
    clus_part_l = _norm01(np.maximum(clus_l - 1, 0), max(1, BAND_MIN_TOUCH))
    return {
        "hit_high": pd.Series(hit_high, index=df.index),
        "hit_low": pd.Series(hit_low, index=df.index),
        "clus_high": pd.Series(clus_h, index=df.index),
        "clus_low": pd.Series(clus_l, index=df.index),
        "level_high": pd.Series(np.clip(0.60 * hit_part_h + 0.40 * clus_part_h, 0, 1), index=df.index),
        "level_low": pd.Series(np.clip(0.60 * hit_part_l + 0.40 * clus_part_l, 0, 1), index=df.index),
    }


def _level(df: pd.DataFrame, key: str) -> pd.Series:
    return _level_block(df, _truth(df))[key]


def f_hit_high(df: pd.DataFrame) -> pd.Series:
    """当根最高价距最近一个已确认 pivot high（6 个里最近的那个）的 ATR 距离。"""
    return _level(df, "hit_high")


def f_hit_low(df: pd.DataFrame) -> pd.Series:
    """当根最低价距最近一个已确认 pivot low 的 ATR 距离。"""
    return _level(df, "hit_low")


def f_clus_high(df: pd.DataFrame) -> pd.Series:
    """最近 6 个 pivot high 里，与「最近那条」同带（±0.75×ATR）的个数。"""
    return _level(df, "clus_high")


def f_clus_low(df: pd.DataFrame) -> pd.Series:
    """最近 6 个 pivot low 里与最近一条同带的个数。"""
    return _level(df, "clus_low")


def f_level_high(df: pd.DataFrame) -> pd.Series:
    """压力线得分 0~1：贴线 60% + 被反复触及 40%。"""
    return _level(df, "level_high")


def f_level_low(df: pd.DataFrame) -> pd.Series:
    """支撑线得分 0~1：贴线 60% + 被反复触及 40%。"""
    return _level(df, "level_low")


# ══════════════════════════════════════════════════════════════════════════
# 五、持仓量（5）
# ══════════════════════════════════════════════════════════════════════════
def _oi_block(df: pd.DataFrame):
    """持仓量类特征，全部无量纲。

    注意这些是**按 bar** 算的（``shift(5)`` = 5 根，1h 上约 5/6 个交易日），
    原项目就是这么写的 —— 不要「顺手」改成按交易日。
    """
    s = _oi(df).where(lambda x: x > 0)
    c = df["close"].astype(float)
    r1 = s.pct_change()
    mu = r1.rolling(60, min_periods=60).mean()
    sd = r1.rolling(60, min_periods=60).std(ddof=1)      # 原项目用 pandas 默认 ddof=1
    return {
        "oi_chg_5": _safe_div(s, s.shift(5)) - 1.0,
        "oi_chg_20": _safe_div(s, s.shift(20)) - 1.0,
        "oi_z_60": _safe_div(r1 - mu, sd),
        "oi_ratio_60": _safe_div(s, s.rolling(60, min_periods=60).mean()) - 1.0,
        "oi_px_sync_5": np.sign(c.pct_change(5)) * np.sign(s.pct_change(5)),
    }


def _oif(df: pd.DataFrame, key: str) -> pd.Series:
    return _oi_block(df)[key]


def f_oi_chg_5(df: pd.DataFrame) -> pd.Series:
    """持仓量 5 根变化率。"""
    return _oif(df, "oi_chg_5")


def f_oi_chg_20(df: pd.DataFrame) -> pd.Series:
    """持仓量 20 根变化率。"""
    return _oif(df, "oi_chg_20")


def f_oi_z_60(df: pd.DataFrame) -> pd.Series:
    """持仓量单根变化率的 60 根 z（ddof=1，照抄原文）。"""
    return _oif(df, "oi_z_60")


def f_oi_ratio_60(df: pd.DataFrame) -> pd.Series:
    """持仓量相对 60 根均值的偏离 —— HL 报告重要性 26.4%，排第二。"""
    return _oif(df, "oi_ratio_60")


def f_oi_px_sync_5(df: pd.DataFrame) -> pd.Series:
    """价与仓 5 根变化方向的符号乘积（+1 同向、−1 背离、0 有一边没动）。"""
    return _oif(df, "oi_px_sync_5")


# ══════════════════════════════════════════════════════════════════════════
# 六、展期（1；另 3 个映射到库里的 roll_main_*）
# ══════════════════════════════════════════════════════════════════════════
def _main_legs(df: pd.DataFrame):
    """主力腿：先试 ``o6_``，整列取不到退 ``o3_``。

    和 ``factors._main_legs`` 同一个选腿口径（近端=持仓最大的腿，远端=它之后
    持仓最大的腿），这里只取前三列 + 到期日，够算 ``roll`` 就够了。
    """
    for pre in ("o6_", "o3_"):
        c1 = _col(df, f"{pre}1_close")
        if c1.notna().sum() > 0:
            return (c1, _col(df, f"{pre}2_close"), _col(df, f"{pre}3_close"),
                    _col(df, f"{pre}1_dte"), _col(df, f"{pre}2_dte"), _col(df, f"{pre}3_dte"))
    nan = _nan(df)
    return nan, nan, nan, nan, nan, nan


def f_roll_chg_20(df: pd.DataFrame) -> pd.Series:
    """主力/次主力价差率的 20 交易日变化（差分的不是年化值）。

    🔴 库里 ``roll_main_change`` 差分的是 **年化** roll，这里是 **价差率** 本身。
    两者差一个 ``365/间隔天数`` 的时变系数（间隔随换腿变），不是常倍数，
    z-score 消不掉 ⇒ 必须独立注册，不能复用。
    """
    c1, c2, _c3, _d1, _d2, _d3 = _main_legs(df)
    ry = _safe_div(c1 - c2, c2)
    return _day_diff(df, ry, 20)


# ══════════════════════════════════════════════════════════════════════════
# 注册
# ══════════════════════════════════════════════════════════════════════════
def register_into(FACTORS, FactorSpec) -> int:
    """把 33 个新因子灌进 ``FACTORS``；返回注册个数。

    ``roll`` / ``roll_ann`` / ``curv`` 不在这里 —— 它们映射到库里的
    ``roll_main_oi`` / ``roll_main_annual`` / ``roll_main_curv``（见模块文档）。
    """
    specs = [
        # ---- 多均线偏离度（5）----
        ("dev_z_20", "偏离度 z · MA20", f_dev_z_20, "偏离度",
         "(收盘−MA20)/ATR 的 250 根滚动 z，短均线偏离。"),
        ("dev_z_60", "偏离度 z · MA60", f_dev_z_60, "偏离度",
         "(收盘−MA60)/ATR 的 250 根滚动 z。"),
        ("dev_z_120", "偏离度 z · MA120", f_dev_z_120, "偏离度",
         "(收盘−MA120)/ATR 的 250 根滚动 z，长均线偏离。"),
        ("dev_z_agg", "偏离度 z · 聚合", f_dev_z_agg, "偏离度",
         "三条均线偏离度 z 的等权均值，HL-Point 里重要性 30.5% 排第一。"),
        ("dev_spread", "偏离度张角", f_dev_spread, "偏离度",
         "dev_20 − dev_120（原始 ATR 偏离度，不取 z），>0 为短均线张嘴。"),
        # ---- 影线（8）----
        ("up_sh", "上影线占比", f_up_sh, "影线", "(高−实体顶)/(高−低)，0~1。"),
        ("dn_sh", "下影线占比", f_dn_sh, "影线", "(实体底−低)/(高−低)，0~1。"),
        ("up_sh_atr", "上影线 / ATR", f_up_sh_atr, "影线", "上影绝对长度折成 ATR 倍数。"),
        ("dn_sh_atr", "下影线 / ATR", f_dn_sh_atr, "影线", "下影绝对长度折成 ATR 倍数。"),
        ("up_sh_int", "上影线强度 20", f_up_sh_int, "影线", "20 根上影占比均值。"),
        ("dn_sh_int", "下影线强度 20", f_dn_sh_int, "影线", "20 根下影占比均值。"),
        ("shadow_high_z", "上影强度 z", f_shadow_high_z, "影线", "上影强度 20 根均值的 250 根滚动 z。"),
        ("shadow_low_z", "下影强度 z", f_shadow_low_z, "影线", "下影强度 20 根均值的 250 根滚动 z。"),
        # ---- 趋势结构（7）----
        ("slope_atr", "趋势斜率 ATR", f_slope_atr, "趋势结构",
         "60 根 OLS 斜率×60/ATR，即这段趋势走了几个 ATR。"),
        ("r2", "趋势拟合度", f_r2, "趋势结构", "60 根回归 R²，越大趋势越干净。"),
        ("slope_chg_atr", "斜率变化 ATR", f_slope_chg_atr, "趋势结构",
         "斜率相对 30 根前的加速量（×ATR），正=加速、负=拐头。HL-Point 重要性 15.8%。"),
        ("dist_hi", "距阶段高 ATR", f_dist_hi, "趋势结构",
         "收盘距 250 根最高价的 ATR 距离，0=正在阶段最高。"),
        ("dist_lo", "距阶段低 ATR", f_dist_lo, "趋势结构",
         "收盘距 250 根最低价的 ATR 距离，0=正在阶段最低。"),
        ("ext_hi", "阶段最高价", f_ext_hi, "趋势结构",
         "250 根内最高价（绝对价格）。价格水平量，截面 z 后近似品种哑变量，慎用。"),
        ("ext_lo", "阶段最低价", f_ext_lo, "趋势结构",
         "250 根内最低价（绝对价格）。同 ext_hi 的注意事项。"),
        # ---- 通道（1）----
        ("ch_slope_atr", "通道斜率 ATR", f_ch_slope_atr, "趋势结构",
         "30 根回归通道的单根斜率 / ATR，看当下斜率陡不陡。"),
        # ---- 关键位（6）----
        ("hit_high", "贴压力线 ATR", f_hit_high, "关键位",
         "当根最高价距最近已确认 pivot high 的 ATR 距离，越小越贴线。"),
        ("hit_low", "贴支撑线 ATR", f_hit_low, "关键位",
         "当根最低价距最近已确认 pivot low 的 ATR 距离。"),
        ("clus_high", "压力带聚集度", f_clus_high, "关键位",
         "近 6 个 pivot high 中与最近一条同带（±0.75×ATR）的个数。"),
        ("clus_low", "支撑带聚集度", f_clus_low, "关键位",
         "近 6 个 pivot low 中与最近一条同带的个数。"),
        ("level_high", "压力线得分", f_level_high, "关键位",
         "0~1：贴线 60% + 反复触及 40%。"),
        ("level_low", "支撑线得分", f_level_low, "关键位",
         "0~1：贴线 60% + 反复触及 40%。"),
        # ---- 持仓量（5）----
        ("oi_chg_5", "持仓变化 5 根", f_oi_chg_5, "量能",
         "持仓量 5 根变化率（bar 口径，非交易日）。", 1, ("cn",)),
        ("oi_chg_20", "持仓变化 20 根", f_oi_chg_20, "量能",
         "持仓量 20 根变化率。", 1, ("cn",)),
        ("oi_z_60", "持仓变化 z60", f_oi_z_60, "量能",
         "单根持仓变化率的 60 根 z（ddof=1）。", 1, ("cn",)),
        ("oi_ratio_60", "持仓偏离 60", f_oi_ratio_60, "量能",
         "持仓量 / 60 根均值 − 1。HL-Point 重要性 26.4% 排第二。", 1, ("cn",)),
        ("oi_px_sync_5", "量仓同向 5", f_oi_px_sync_5, "量能",
         "价与仓 5 根变化方向的符号乘积，+1 同向、−1 背离。", 1, ("cn",)),
        # ---- 展期（1）----
        ("roll_chg_20", "展期价差变化 20", f_roll_chg_20, "展期",
         "主力/次主力价差率的 20 交易日差分（不是年化值的差分）。", 1, ("cn",)),
    ]
    n = 0
    for item in specs:
        key, label, func, cat, desc = item[:5]
        hint = item[5] if len(item) > 5 else 1
        mk = item[6] if len(item) > 6 else ()
        if key in FACTORS:          # 幂等：重复调用不重复注册
            continue
        FACTORS[key] = FactorSpec(key, label, func, cat, desc, hint, markets=mk)
        n += 1
    return n
