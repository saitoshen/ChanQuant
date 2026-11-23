"""HL point 截面对冲策略用到的因子（从 future_quant 移植）。

## 来源与口径

`D:/CozeTask/future_quant/cross_section/` 那套 HL point 策略，量价因子全部来自
`rust_futures/src/lib.rs` 的 `compute_single_stock_features()`，输出 **77 列**：

- Group A (0-19) 基础量价 20 个
- Group B (20-37) Fama-French 18 个 —— **期货恒 0**（财务列填 0，`has_financial`
  为假），标准化时 std=0 被跳过，整列恒 0，对模型零贡献。**不移植**
- Group C (38-47) MACD 风洞共振 10 个
- Group D (48-57) 几何结构 10 个
- Group E (58-64) 缩量短线 7 个
- Group F (65-71) 动量+反转+成交额 7 个
- Group H (72-76) 日线级日内 5 个

🔴 顺手记一个原项目的坑：它的 Python 侧 `RUST_PRICE_VOL_FEATURES` 只写了 **59 个
名字**，却按 `enumerate` 顺序去贴 77 列的值 ⇒ 从第 21 个名字开始全部错位一格
（列名叫 `funnel_daily` 的那一列装的其实是 `mc_chg_5`）。本模块按 **Rust 源码的
真实定义**移植，不复刻这个错位。

## 去重原则

和库里已有因子**公式等价**的一律不重复注册，只做映射（对照表见
`docs/HL_POINT_FACTORS.md`）：

| HL point | 库里已有 | 说明 |
|---|---|---|
| `mom_20d` / `mom_60d` | `mom_20` / `mom_60` | 同一个东西 |
| `vol_20d` | `vol_20` | 同为年化已实现波动 |
| `atr` | `atr_pct` | 库里是占比版，跨品种更可比 |
| `rsi` | `rsi_14` | 同周期 |
| `macd_hist` | `macd_hist` | 库里已除收盘价 |
| `bb_middle` / `price_vs_sma20` | `ma_gap_20` | 都是 close/MA20−1 |
| `price_vs_sma60` | `ma_gap_60` | 同上 |
| `dev_20` | `bb_pos_20` | 差一个常数 2，z-score 后完全等价 |
| `pos_20` | `pos_20` | 同一个区间位置 |
| `close_position` | `close_pos` | 库里是 10 日均值 |
| `gap_continuation` | `gap` | 库里是 10 日均值 |
| `vol_ratio_20` / `vol_ratio_5d` | `vol_surge` | 都是 量/20 日均量 |
| `close_vs_open_ratio` | `mom_1` | 日线上恒等 |
| `reversal_1d` | `mom_1` | 只差符号，方向由滚动 IC 定 |
| `turnover_est_ma5` | `log_amount_20` | 量纲不同，含义相近 |
| `term_spread_near_far` | `term_near_far` | 同 |
| `term_spread_main_far` | `roll_main_oi` | 同族（主力-远月价差） |
| `roll_yield` | `roll_yield` | 同 |
| `oi_chg_1d/5d` | `oi_change_1` / `oi_change_5` | 同 |
| `basis_rate` | `spot_basis` | 同 |
| `warehouse_rank` / `inventory_rank` | `war_pct` / `inv_pct` | 同为三年滚动分位 |
| `warehouse_chg_5` | `war_chg_5` | 同 |
| `inventory_chg_1w` | `inv_chg_5` | 周频近似 |
| `profit_chg_4w` | `profit_chg_5` | 同 |
| `delivery_chg_mom` | `deliv_chg_30` | 同族 |

**不移植的**（数据源在 ChanQuant 侧不存在，注册了也只是整列 NaN）：
持仓排名 5 个（`net_pos_ratio` / `long_oi_share` / `short_oi_share` /
`long_chg_ratio` / `short_chg_ratio`，要天勤龙虎榜 `symbol_rank_daily`）、
Group B 财务 18 个、`log_price`（纯价位，跨品种不可比）。

## PIT

价格类全部只有 `shift(正数)` / `rolling`，没有一处看未来。
基本面（现货 / 仓单 / 库存 / 交割量 / 利润）沿用 `factors_ext._fund` 的
`lag_days=1` —— EDB 当天收盘后才更新，白天用当天的值就是未来函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 和 factors_ext 一样，本模块**不** import 主模块（主模块末尾才 import 本模块，
# 反向 import 就是循环）。EDB 那几条要用的取数逻辑直接复用 factors_ext 的。
from ttrade import factors_ext as _FX


# ---------------------------------------------------------------- 通用工具
def _safe_div(a, b):
    return a / b.replace(0.0, np.nan)


def _vol(df: pd.DataFrame) -> pd.Series:
    """成交量。两条数据链路的列名不一样（`volume` / `vol`）。"""
    for c in ("volume", "vol"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _oi(df: pd.DataFrame) -> pd.Series:
    """持仓量。列名有 `open_interest` / `open_oi` / `oi` 三种写法。"""
    for c in ("open_interest", "open_oi", "oi"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _tr(df: pd.DataFrame) -> pd.Series:
    """真实波幅 TR = max(H−L, |H−PC|, |L−PC|)。"""
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)


def _macd(c: pd.Series, fast: int, slow: int, sig: int):
    """MACD 三件套。和库里 `f_macd_hist` 同一套 ewm 口径（不是通达信的 SMA）。"""
    dif = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=sig, adjust=False).mean()
    return dif, dea, dif - dea


def _slope(s: pd.Series, n: int) -> pd.Series:
    """滚动线性回归斜率（x = 0..n−1）。

    为什么不用 `rolling.apply`：日线上无所谓，但分钟线上 30 万根逐个调 Python
    函数是分钟级的。这里用 `sliding_window_view` 一次算一片，再按 6.5 万根分块，
    避免一次性materialize 出 30 万 × 60 的矩阵把内存顶掉。
    """
    v = np.asarray(s, dtype=float)
    m = len(v)
    out = np.full(m, np.nan)
    if m < n:
        return pd.Series(out, index=s.index)
    x = np.arange(n, dtype=float)
    xm = x.mean()
    den = float(((x - xm) ** 2).sum())
    step = 65536
    for a in range(0, m - n + 1, step):
        b = min(a + step, m - n + 1)
        seg = v[a:b + n - 1]
        sw = np.lib.stride_tricks.sliding_window_view(seg, n)
        out[a:b] = ((sw - sw.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / den
    return pd.Series(out, index=s.index)


def _funnel(hist: pd.Series, n: int) -> pd.Series:
    """MACD 风洞：柱值在最近 n 根里的相对位置，线性映射到 −1~+1。

    原实现（Rust `funnel_signal`）就是把 `(h − min)/(max − min)` 拉成 `2p − 1`，
    range 为 0 时留 0。形状上像「漏斗」收口，名字由此而来。
    """
    lo = hist.rolling(n, min_periods=n).min()
    hi = hist.rolling(n, min_periods=n).max()
    rng = hi - lo
    up = _safe_div(hist - lo, rng)
    dn = _safe_div(hi - hist, rng)
    return (up - dn).where(rng.abs() > 1e-12, 0.0)


# ---------------------------------------------------------------- MACD 风洞
# 这一族在原项目里是「多周期共振」的核心：日/周/月三条 MACD 柱各看自己的风洞
# 位置，加权合成一个共振分。它和单纯的 `macd_hist` 不是一回事 —— 后者看的是
# 「柱子有多长」，这里看的是「柱子在最近一段里处在什么位置」。
def f_macd_dif(df):
    """DIF（快慢线之差）/ 收盘价。归一化是为了跨品种可比。"""
    dif, _, _ = _macd(df["close"], 12, 26, 9)
    return _safe_div(dif, df["close"])


def f_macd_dea(df):
    """DEA（DIF 的 9 日 EMA）/ 收盘价。"""
    _, dea, _ = _macd(df["close"], 12, 26, 9)
    return _safe_div(dea, df["close"])


def f_macd_hist_chg(df):
    """MACD 柱的一阶差分 / 收盘价。柱子这一根在变长还是变短。"""
    _, _, h = _macd(df["close"], 12, 26, 9)
    return _safe_div(h - h.shift(1), df["close"])


def f_macd_hist_acc(df):
    """MACD 柱的二阶差分 / 收盘价。柱子变化的速度本身在加速还是减速。"""
    _, _, h = _macd(df["close"], 12, 26, 9)
    d1 = h - h.shift(1)
    return _safe_div(d1 - d1.shift(1), df["close"])


def f_macd_hist_lag1(df):
    """T−1 根的 MACD 柱 / 收盘价。

    原项目叫 `tg_macd_hist`，用在「当天盘中决策时只能看到昨天收盘」的场景。
    我们这边因子本身是「bar 收盘后可得」的粒度，所以它在这里的意义变成
    「柱子的一阶滞后」—— 对隔夜跳空类的组合仍然有用。
    """
    _, _, h = _macd(df["close"], 12, 26, 9)
    return _safe_div(h, df["close"]).shift(1)


def f_macd_funnel_d(df):
    """日线 MACD 柱的风洞位置（回看 5 根）。"""
    _, _, h = _macd(df["close"], 12, 26, 9)
    return _funnel(h, 5)


def f_macd_funnel_w(df):
    """「周线」MACD 柱的风洞位置（参数 21/55/9，回看 8 根）。

    原项目直接在日线上套 21/55/9 的参数，而不是真的重采样到周线 —— 效果上
    相当于把周期拉长约 2 倍。这里照抄，**不要**擅自改成 `resample("W")`：
    那是另一条策略，换了就不是复现了。
    """
    _, _, h = _macd(df["close"], 21, 55, 9)
    return _funnel(h, 8)


def f_macd_funnel_m(df):
    """「月线」MACD 柱的风洞位置（参数 55/144/9，回看 13 根）。"""
    _, _, h = _macd(df["close"], 55, 144, 9)
    return _funnel(h, 13)


def f_macd_resonance(df):
    """日 / 周 / 月三档风洞的共振分：0.4·日 + 0.4·周 + 0.2·月。

    权重不对称是有道理的：越长的周期越平滑、也越滞后，给它两成只用来
    「别和大方向作对」，不参与选时。
    """
    _, _, hd = _macd(df["close"], 12, 26, 9)
    _, _, hw = _macd(df["close"], 21, 55, 9)
    _, _, hm = _macd(df["close"], 55, 144, 9)
    return _funnel(hd, 5) * 0.4 + _funnel(hw, 8) * 0.4 + _funnel(hm, 13) * 0.2


def f_macd_cross_up(df):
    """日线 MACD 柱上穿 0 的当根记 1（金叉）。离散事件，其余为 0。"""
    _, _, h = _macd(df["close"], 12, 26, 9)
    return ((h > 0) & (h.shift(1) <= 0)).astype(float)


def f_macd_cross_dn(df):
    """日线 MACD 柱下穿 0 的当根记 1（死叉）。"""
    _, _, h = _macd(df["close"], 12, 26, 9)
    return ((h < 0) & (h.shift(1) >= 0)).astype(float)


def f_macd_at_trough(df):
    """日线 MACD 柱当前是最近 10 根里的最低点记 1。

    和 `cross_up` 互补：金叉是「确认转了」，这里是「正在最低点」，
    更早一步、也更早被噪声骗。
    """
    _, _, h = _macd(df["close"], 12, 26, 9)
    lo = h.rolling(10, min_periods=10).min()
    return (h <= lo).astype(float)


def f_macd_at_peak(df):
    """日线 MACD 柱当前是最近 10 根里的最高点记 1。"""
    _, _, h = _macd(df["close"], 12, 26, 9)
    hi = h.rolling(10, min_periods=10).max()
    return (h >= hi).astype(float)


# ---------------------------------------------------------------- 趋势 / 动量
def f_mom_5(df):
    """5 日涨跌。库里 1/3/10/20 都有，唯独缺 5 日这一档 —— 一周正好是
    「资金进来又出去」的最短完整周期，短线反转和趋势都用得上。"""
    c = df["close"]
    return c / c.shift(5) - 1.0


def f_mom_accel(df):
    """动量加速度：最近 5 日收益 − 前 15 日收益。

    趋势的「二阶导」。价格还在涨但涨得越来越慢，是趋势末段的典型特征 ——
    单看 `mom_20` 看不出来。
    """
    c = df["close"]
    recent = c / c.shift(5) - 1.0
    prev = c.shift(5) / c.shift(20) - 1.0
    return recent - prev


def f_mom_diverge(df):
    """动量背离：`C/C[-5]` 与 `C[-5]/C[-20]` 两个**比值**之差。

    🔴 它和 `mom_accel` 在数学上**完全相等**：`(a−1) − (b−1) ≡ a − b`，两个
    「减 1」互相抵消。原项目（`lib.rs` 的 Group F）把这两条当两个不同特征喂进
    LightGBM，实际是同一列算了两遍 —— 树模型里等于白占一个特征名额。
    这里保留函数以备对照，但**不注册**（见 `register_into` 末尾的说明）。
    """
    c = df["close"]
    return c / c.shift(5) - c.shift(5) / c.shift(20)


def f_slope_20(df):
    """20 日收盘线性回归斜率 / 收盘价。

    和 `ma_slope_20` 的区别：那条是「MA20 相对 10 日前」，本质是均线的差；
    这条是对 20 个收盘点做最小二乘，对两端的极值不敏感。
    """
    return _safe_div(_slope(df["close"], 20), df["close"])


def f_slope_60(df):
    """60 日收盘线性回归斜率 / 收盘价。"""
    return _safe_div(_slope(df["close"], 60), df["close"])


def f_ma_align(df):
    """均线排列分：四条判据（C>MA5、MA5>MA10、MA10>MA20、MA20>MA60）。

    每满足一条 +1，减 2 后除 2 ⇒ 取值 −1 / −0.5 / 0 / 0.5 / 1。
    比单一动量更「结构化」：多头排列的五档里，−1 是空头完整排列。
    """
    c = df["close"]
    ma5 = c.rolling(5, min_periods=5).mean()
    ma10 = c.rolling(10, min_periods=10).mean()
    ma20 = c.rolling(20, min_periods=20).mean()
    ma60 = c.rolling(60, min_periods=60).mean()
    score = (
        (c > ma5).astype(float)
        + (ma5 > ma10).astype(float)
        + (ma10 > ma20).astype(float)
        + (ma20 > ma60).astype(float)
    )
    return (score - 2.0) / 2.0


# ---------------------------------------------------------------- 形态 / 几何
def f_pos_60(df):
    """60 日区间位置：(C − LL60) / (HH60 − LL60)。`pos_20` 的长周期版。"""
    hh = df["high"].rolling(60, min_periods=60).max()
    ll = df["low"].rolling(60, min_periods=60).min()
    return _safe_div(df["close"] - ll, hh - ll)


def f_new_high_10(df):
    """当根最高价就是最近 10 根最高的，记 1。突破的离散版。"""
    hh = df["high"].rolling(10, min_periods=10).max()
    return (df["high"] >= hh).astype(float)


def f_bb_width_20(df):
    """布林带宽度：4σ20 / MA20（上轨−下轨 相对中轨）。

    波动率的「相对」表达 —— 和 `vol_20` 不同，它只用收盘价、不含跳空，
    而且在趋势行情里会先于收益率波动率收窄。
    """
    c = df["close"]
    m = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=1)
    return _safe_div(4.0 * sd, m)


def f_bb_width_60(df):
    """布林带宽度 60 日。"""
    c = df["close"]
    m = c.rolling(60, min_periods=60).mean()
    sd = c.rolling(60, min_periods=60).std(ddof=1)
    return _safe_div(4.0 * sd, m)


def f_dev_60(df):
    """60 日标准化偏离：(C − MA60) / σ60。

    `bb_pos_20` 是它的 20 日版（差常数 2）。60 日这一档更能代表「中期脱离了
    均值多远」，是均值回归的主要读数。
    """
    c = df["close"]
    m = c.rolling(60, min_periods=60).mean()
    sd = c.rolling(60, min_periods=60).std(ddof=1)
    return _safe_div(c - m, sd)


def f_high_ma_gap_20(df):
    """最高价相对 MA20 的偏离：H / MA20 − 1。

    收盘价偏离（`ma_gap_20`）会漏掉「冲高回落」——盘中确实摸到了高位，
    但收盘按回去了。这条专门量上影线。
    """
    m = df["close"].rolling(20, min_periods=20).mean()
    return _safe_div(df["high"], m) - 1.0


# ---------------------------------------------------------------- 波动
def f_vol_5(df):
    """5 日年化已实现波动率。库里只有 10 / 20 日，5 日这一档对波动突变最快。"""
    r = df["close"].pct_change()
    return r.rolling(5, min_periods=5).std(ddof=1) * np.sqrt(252.0)


# ---------------------------------------------------------------- 量能 / 缩量
def f_vol_surge_5(df):
    """当日量 / 5 日均量。

    和 `vol_surge`（当日量 / **20** 日均量）是一对：20 日的基准包含了自己，
    突发放大时会被自己抬高；5 日基准反应更快，适合看「今天相对这几天」。
    """
    v = _vol(df)
    return _safe_div(v, v.rolling(5, min_periods=5).mean())


def f_vol_shrink(df):
    """缩量信号：5 日均量低于 20 日均量的 60% 记 1。

    阈值 0.6 是原项目拍的，不是拟合出来的 —— 换品种换周期都别指望它最优，
    它的价值在于把「显著缩量」这种状态离散化，让树模型少切几刀。
    """
    v = _vol(df)
    m5 = v.rolling(5, min_periods=5).mean()
    m20 = v.rolling(20, min_periods=20).mean()
    return ((m5 < m20 * 0.6) & m20.notna() & m5.notna()).astype(float)


def f_atr_ratio(df):
    """ATR 收缩比：MA5(TR) / MA20(TR)。<1 = 波动在收敛。"""
    tr = _tr(df)
    return _safe_div(tr.rolling(5, min_periods=5).mean(),
                     tr.rolling(20, min_periods=20).mean())


def f_range_ratio(df):
    """振幅收缩比：5 日振幅 / 20 日振幅，振幅 = (HH−LL)/收盘。

    和 `atr_ratio` 的区别是分母用收盘价而不是真实波幅：它不含跳空，
    纯看「K 线本身的长度在不在缩」。
    """
    c = df["close"]
    r5 = _safe_div(
        df["high"].rolling(5, min_periods=5).max() - df["low"].rolling(5, min_periods=5).min(),
        c,
    )
    r20 = _safe_div(
        df["high"].rolling(20, min_periods=20).max() - df["low"].rolling(20, min_periods=20).min(),
        c,
    )
    return _safe_div(r5, r20)


def f_vol_price_diverge(df):
    """量价背离：5 日上涨 **且** 显著缩量，同时成立记 1。

    「涨得没量」在商品上比在股票上更可靠 —— 期货的量是和持仓一起看的，
    缩量上涨多半是平空推上去的，没有新多进场。
    """
    c = df["close"]
    mom5 = c / c.shift(5) - 1.0
    v = _vol(df)
    m5 = v.rolling(5, min_periods=5).mean()
    m20 = v.rolling(20, min_periods=20).mean()
    shrink = (m5 < m20 * 0.6) & m5.notna() & m20.notna()
    return ((mom5 > 0.0) & shrink).astype(float)


def f_contraction_score(df):
    """综合缩量评分：量、真实波幅、振幅三条收缩比的加权合成分，映射到 −1~+1。

    三个分量各自先截断再归一（量比 0~3、波幅比 0~2、振幅比 0~2），
    所以极端值不会把总分带飞。越高 = 越「安静」，是突破前的典型状态。
    """
    v = _vol(df)
    vr = _safe_div(v, v.rolling(20, min_periods=20).mean()).clip(0.0, 3.0) / 3.0
    tr = _tr(df)
    ar = _safe_div(tr.rolling(5, min_periods=5).mean(),
                   tr.rolling(20, min_periods=20).mean()).clip(0.0, 2.0) / 2.0
    c = df["close"]
    r5 = _safe_div(
        df["high"].rolling(5, min_periods=5).max() - df["low"].rolling(5, min_periods=5).min(), c)
    r20 = _safe_div(
        df["high"].rolling(20, min_periods=20).max() - df["low"].rolling(20, min_periods=20).min(), c)
    rr = _safe_div(r5, r20).clip(0.0, 2.0) / 2.0
    return ((1.0 - vr) * 0.4 + (1.0 - ar) * 0.3 + (1.0 - rr) * 0.3) * 2.0 - 1.0


def f_oi_change_10(df):
    """持仓量 10 日变化率。库里只有 1 / 5 日，10 日对应「两周的资金态度」。"""
    oi = _oi(df)
    return oi.where(oi > 0).pct_change(10)


def f_oi_vol_ratio(df):
    """持仓量 / 成交量。

    同样是「这笔量沉淀下来没有」：比值高说明成交没被平掉、是新开仓在堆；
    比值低说明当日来回打、日内投机为主。原项目用它做「资金是否认真」的读数。
    """
    return _safe_div(_oi(df).where(_oi(df) > 0), _vol(df).where(_vol(df) > 0))


def f_oi_price_corr_5(df):
    """持仓量与收盘价的 5 日相关性。

    正 = 价涨仓增（多头真进场），负 = 价涨仓减（空头在平）。
    和 `pv_corr` 不是一回事：那条是「收益 vs 成交量变化」。
    """
    oi = _oi(df).where(_oi(df) > 0)
    return oi.rolling(5, min_periods=5).corr(df["close"])


# ---------------------------------------------------------------- 日内（日线口径）
def f_intraday_strength(df):
    """日内强度：(C − O) / (H − L)。收盘落在当日振幅的哪个位置。

    和 `close_pos`（(C−L)/(H−L) 的 10 日均值）不同：这条的分子是开盘到收盘的
    净移动，量的是「方向」；`close_pos` 量的是「收盘在区间里的绝对位置」。
    一根一开盘就跳高、之后一路走低的 K 线，`close_pos` 很高但这里是负的。
    """
    return _safe_div(df["close"] - df["open"], df["high"] - df["low"])


def f_gap_fill_ratio(df):
    """跳空回补比例：(C − O) / (O − PC)。

    +1 = 跳空之后一路走完全程，−1 = 完全回补并反向。开盘跳空越大，这个比值
    越接近「回补进度」，是假突破最直接的读数。

    🔴 分母用 `|O − PC| > 收盘价 × 1e-4` 兜一道：原项目只挡 `1e-8` 的绝对零，
    真遇到「几乎平开」的 bar（夜盘断价、主力换月接缝）分母是个毫厘之数，
    比值能蹦到上万 —— 一根这样的 bar 就能把整列的 z-score 带歪。
    """
    gap = df["open"] - df["close"].shift(1)
    ok = gap.abs() > (df["close"].abs() * 1e-4)
    return _safe_div(df["close"] - df["open"], gap.where(ok))


# ---------------------------------------------------------------- 基本面（EDB）
# 全部走 `factors_ext._fund`：它负责「按品种取序列 + as-of 到 bar + 滞后 1 个
# 日历日」。自己再写一套就一定会和 `war_pct` 那批差一天（#184 的老坑）。
def f_spot_mom_5(df):
    """现货价的 5 日动量。期货还没动、现货先动，是基差走强的领先读数。"""
    return _FX._day_pct(df, _FX._fund(df, "spot_price"), 5)


def f_spot_mom_20(df):
    """现货价的 20 日动量。"""
    return _FX._day_pct(df, _FX._fund(df, "spot_price"), 20)


def f_warehouse_chg_20(df):
    """仓单数量 20 日变化率。`war_chg_5` 是一个月内，这条是一个月跨度。"""
    return _FX._day_pct(df, _FX._fund(df, "warehouse"), 20)


def f_inventory_yoy(df):
    """期货库存同比（52 周 ≈ 364 个日历日）。

    库存是**周频**，直接 `pct_change(52)` 在日线上量到的是 52 个交易日，
    必须先按自然日压频再算 —— `_day_pct` 做的就是这件事。
    """
    return _FX._day_pct(df, _FX._fund(df, "inventory"), 364)


def f_profit_pct(df):
    """加工利润的三年滚动分位（0~1）。

    `profit_val` 是绝对水平（元/吨，跨品种不可比），这条换成「相对自己历史
    的贵贱」。利润分位低 = 上游快减产了，是远月的供应端利多。
    """
    return _FX._pct_rank(df, _FX._fund(df, "profit"))


def f_deliv_pct(df):
    """交割量的滚动分位（0~1）。月频指标，观察数少，`_pct_rank` 的 minp 放宽。"""
    return _FX._pct_rank(df, _FX._fund(df, "delivery"), n=60, minp=24)


# ---------------------------------------------------------------- 注册
def register_into(FACTORS, FactorSpec) -> int:
    specs = [
        # ---- MACD 风洞（原项目 Group C）----
        ("macd_dif", "MACD DIF", f_macd_dif, "MACD 风洞",
         "快慢线之差除以收盘价，MACD 的原始形态，归一化后跨品种可比。"),
        ("macd_dea", "MACD DEA", f_macd_dea, "MACD 风洞",
         "DIF 的 9 日 EMA 除以收盘价，比 DIF 平滑一档。"),
        ("macd_hist_chg", "MACD 柱变化", f_macd_hist_chg, "MACD 风洞",
         "柱值一阶差分/收盘价 —— 柱子这一根在变长还是变短。"),
        ("macd_hist_acc", "MACD 柱加速度", f_macd_hist_acc, "MACD 风洞",
         "柱值二阶差分/收盘价 —— 柱子变化本身在加速还是减速。"),
        ("macd_hist_lag1", "MACD 柱（滞后一根）", f_macd_hist_lag1, "MACD 风洞",
         "T−1 根的 MACD 柱，原项目的 tg_macd_hist，用于隔夜跳空类组合。"),
        ("macd_funnel_d", "MACD 风洞·日", f_macd_funnel_d, "MACD 风洞",
         "日线柱在最近 5 根里的相对位置，映射到 −1~+1。", 1),
        ("macd_funnel_w", "MACD 风洞·周", f_macd_funnel_w, "MACD 风洞",
         "21/55/9 参数的柱在最近 8 根里的相对位置。", 1),
        ("macd_funnel_m", "MACD 风洞·月", f_macd_funnel_m, "MACD 风洞",
         "55/144/9 参数的柱在最近 13 根里的相对位置。", 1),
        ("macd_resonance", "MACD 三周期共振", f_macd_resonance, "MACD 风洞",
         "0.4·日 + 0.4·周 + 0.2·月三档风洞的加权合成分。", 1),
        ("macd_cross_up", "MACD 金叉", f_macd_cross_up, "MACD 风洞",
         "柱值上穿 0 的当根记 1，离散事件。", 1),
        ("macd_cross_dn", "MACD 死叉", f_macd_cross_dn, "MACD 风洞",
         "柱值下穿 0 的当根记 1。", -1),
        ("macd_at_trough", "MACD 柱谷底", f_macd_at_trough, "MACD 风洞",
         "柱值是最近 10 根最低点记 1，比金叉早一步。", -1),
        ("macd_at_peak", "MACD 柱峰顶", f_macd_at_peak, "MACD 风洞",
         "柱值是最近 10 根最高点记 1。", 1),
        # ---- 趋势 / 动量 ----
        ("mom_5", "动量 5 日", f_mom_5, "趋势", "5 日涨跌，库里 1/3/10/20 之外的那一档。", 1),
        ("mom_accel", "动量加速度", f_mom_accel, "趋势",
         "最近 5 日收益减前 15 日收益，趋势的二阶导。", 1),
        # `mom_diverge` 不注册：它和 mom_accel 数学上完全相等（见函数注释）。
        # 原项目把同一列当两个特征，复现时没必要跟着多一列。
        ("slope_20", "回归斜率 20", f_slope_20, "趋势",
         "20 日收盘最小二乘斜率/收盘价，对两端极值不敏感。", 1),
        ("slope_60", "回归斜率 60", f_slope_60, "趋势", "60 日回归斜率/收盘价。", 1),
        ("ma_align", "均线排列分", f_ma_align, "趋势",
         "C>MA5>MA10>MA20>MA60 四条判据的得分，−1~+1 五档。", 1),
        # ---- 形态 / 几何 ----
        ("pos_60", "区间位置 60", f_pos_60, "形态", "收盘在 60 日高低区间中的位置。", 1),
        ("new_high_10", "10 日新高", f_new_high_10, "形态",
         "当根最高价创 10 日新高记 1，突破的离散版。", 1),
        ("bb_width_20", "布林带宽 20", f_bb_width_20, "波动",
         "4σ20/MA20，收盘价口径的相对波动，不含跳空。", -1),
        ("bb_width_60", "布林带宽 60", f_bb_width_60, "波动", "4σ60/MA60。", -1),
        ("dev_60", "标准化偏离 60", f_dev_60, "反转",
         "(C−MA60)/σ60，均值回归的主要读数，bb_pos_20 的长周期版。", -1),
        ("high_ma_gap_20", "最高价偏离 20", f_high_ma_gap_20, "形态",
         "H/MA20−1，专门量上影线，收盘价偏离会漏掉冲高回落。", 1),
        # ---- 波动 ----
        ("vol_5", "波动率 5", f_vol_5, "波动",
         "5 日年化已实现波动率，对波动突变反应最快的一档。", -1),
        # ---- 量能 ----
        ("vol_surge_5", "放量倍数 5", f_vol_surge_5, "量能",
         "当日量/5 日均量，vol_surge 的短基准版，不会被自己抬高。", 1),
        ("vol_shrink", "缩量信号", f_vol_shrink, "量能",
         "5 日均量低于 20 日均量的 60% 记 1，显著缩量的离散化。", -1),
        ("atr_ratio", "ATR 收缩比", f_atr_ratio, "量能",
         "MA5(TR)/MA20(TR)，<1 表示波动在收敛。", -1),
        ("range_ratio", "振幅收缩比", f_range_ratio, "量能",
         "5 日振幅/20 日振幅，分母用收盘价，纯看 K 线长度。", -1),
        ("vol_price_diverge", "量价背离", f_vol_price_diverge, "量能",
         "5 日上涨且显著缩量同时成立，缩量上涨多为平空推动。", 1),
        ("contraction_score", "综合缩量评分", f_contraction_score, "量能",
         "量/波幅/振幅三条收缩比的加权合成，−1~+1，越高越安静。", 1),
        ("oi_change_10", "持仓变化 10 日", f_oi_change_10, "量能",
         "持仓量 10 日变化率，两周的资金态度。", 1, ("cn",)),
        ("oi_vol_ratio", "持仓成交比", f_oi_vol_ratio, "量能",
         "持仓量/成交量，衡量当日的量是沉淀成仓还是日内来回打。", 1, ("cn",)),
        ("oi_price_corr_5", "持仓价格相关 5", f_oi_price_corr_5, "量能",
         "持仓量与收盘价的 5 日相关性，正=价涨仓增（多头真进场）。", 1, ("cn",)),
        # ---- 日内（日线口径）----
        ("intraday_strength", "日内强度", f_intraday_strength, "日内",
         "(C−O)/(H−L)，开盘到收盘的净移动占振幅的比例。", 1),
        ("gap_fill_ratio", "跳空回补比例", f_gap_fill_ratio, "日内",
         "(C−O)/(O−PC)，+1 走完全程、−1 完全回补，假突破的直接读数。"),
        # ---- 基本面 ----
        ("spot_mom_5", "现货动量 5", f_spot_mom_5, "基本面",
         "现货价 5 日动量，期货未动现货先动是基差走强的领先读数。", 1, ("cn",)),
        ("spot_mom_20", "现货动量 20", f_spot_mom_20, "基本面",
         "现货价 20 日动量。", 1, ("cn",)),
        ("warehouse_chg_20", "仓单变化 20 日", f_warehouse_chg_20, "基本面",
         "仓单数量 20 日变化率，war_chg_5 的长跨度版。", -1, ("cn",)),
        ("inventory_yoy", "库存同比", f_inventory_yoy, "基本面",
         "期货库存 52 周同比，周频指标按自然日压频后计算。", -1, ("cn",)),
        ("profit_pct", "利润分位", f_profit_pct, "基本面",
         "加工利润的三年滚动分位，低分位=上游快减产，远月供应端利多。", 1, ("cn",)),
        ("deliv_pct", "交割量分位", f_deliv_pct, "基本面",
         "交割量的滚动分位，月频指标，观察数少所以窗口放宽。", 1, ("cn",)),
    ]
    n = 0
    for item in specs:
        key, label, func, cat, desc = item[:5]
        hint = item[5] if len(item) > 5 else 1
        mk = item[6] if len(item) > 6 else ()
        FACTORS[key] = FactorSpec(key, label, func, cat, desc, hint, markets=mk)
        n += 1
    return n
