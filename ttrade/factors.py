"""因子库：全部按 PIT（Point-In-Time）方式构造。

## 什么叫 PIT，为什么这里必须守

PIT = 在 T 日收盘后，你只能知道 ≤T 的信息。这条约束在因子研究里最容易破，
而且破了之后回测结果会好得离谱、实盘一分钱赚不到。本项目守三条铁律：

1. **计算只用向后看的窗口**：所有 `rolling` / `ewm` / `shift(正数)`，
   绝不出现 `shift(-n)`（前向收益除外，那只用于评估，不参与信号）。
2. **标准化用滚动窗口**：`z-score` 的均值/标准差取过去 252 日，而不是全样本。
   全样本 z-score 会把未来的分布信息泄漏到今天的因子值里 —— 这是最隐蔽的
   一类未来函数，肉眼看不出来，IC 能虚高 30% 以上。
3. **分位、因子方向（IC 符号）同样滚动**：用扩张窗口的 IC 决定「高分做多还是
   做空」，绝不用全样本 IC。用全样本 IC 定向等于提前知道哪个因子管用。

因子函数只负责算出**原始值**，标准化在 `standardize()` 里统一做。
因子值允许有 NaN（预热期），下游一律 dropna 后再用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------- 基础工具
def _ret(c: pd.Series, n: int) -> pd.Series:
    return c / c.shift(n) - 1.0


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """分母为 0 时返回 NaN，避免 inf 污染后续 z-score。"""
    return a / b.replace(0.0, np.nan)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _nan_series(df: pd.DataFrame) -> pd.Series:
    """数据里没有这一列时（比如美股没有持仓量）返回全 NaN。"""
    return pd.Series(np.nan, index=df.index, dtype=float)


# ---------------------------------------------------------------- 趋势 / 动量
def f_mom_20(df: pd.DataFrame) -> pd.Series:
    """20 日动量。短期趋势延续。"""
    return _ret(df["close"], 20)


def f_mom_60(df: pd.DataFrame) -> pd.Series:
    """60 日动量。中期趋势延续，是商品期货上最经典的一类因子。"""
    return _ret(df["close"], 60)


def f_mom_120(df: pd.DataFrame) -> pd.Series:
    """120 日动量。半年趋势。"""
    return _ret(df["close"], 120)


def f_mom_250(df: pd.DataFrame) -> pd.Series:
    """250 日动量。年度趋势，最慢也最稳。"""
    return _ret(df["close"], 250)


def f_ma_gap_20(df: pd.DataFrame) -> pd.Series:
    """收盘价相对 20 日均线的偏离度。"""
    c = df["close"]
    return _safe_div(c, c.rolling(20, min_periods=20).mean()) - 1.0


def f_ma_gap_60(df: pd.DataFrame) -> pd.Series:
    """收盘价相对 60 日均线的偏离度。"""
    c = df["close"]
    return _safe_div(c, c.rolling(60, min_periods=60).mean()) - 1.0


def f_ma_slope_20(df: pd.DataFrame) -> pd.Series:
    """20 日均线自身的斜率（相对 10 日前）。比价格动量更平滑。"""
    m = df["close"].rolling(20, min_periods=20).mean()
    return _ret(m, 10)


def f_breakout_pos(df: pd.DataFrame) -> pd.Series:
    """55 日通道位置：0=贴通道下沿，1=贴上沿。唐奇安突破的连续化版本。"""
    hh = df["high"].rolling(55, min_periods=55).max()
    ll = df["low"].rolling(55, min_periods=55).min()
    return _safe_div(df["close"] - ll, hh - ll)


def f_efficiency(df: pd.DataFrame) -> pd.Series:
    """Kaufman 效率比：净位移 / 总路程。衡量趋势的「直」不「直」。"""
    n = 20
    c = df["close"]
    net = (c - c.shift(n)).abs()
    path = c.diff().abs().rolling(n, min_periods=n).sum()
    return _safe_div(net, path)


# ---------------------------------------------------------------- 反转 / 超买超卖
def f_rev_5(df: pd.DataFrame) -> pd.Series:
    """5 日反转：短期涨多了倾向回落。方向与动量相反。"""
    return -_ret(df["close"], 5)


def f_rev_20(df: pd.DataFrame) -> pd.Series:
    """20 日反转。"""
    return -_ret(df["close"], 20)


def f_rsi_14(df: pd.DataFrame) -> pd.Series:
    """RSI(14)，0~100。标准化后使用，不要直接比阈值。"""
    c = df["close"]
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return 100.0 - 100.0 / (1.0 + _safe_div(up, dn))


def f_bb_pos_20(df: pd.DataFrame) -> pd.Series:
    """布林带位置：(close - 中轨) / (2×标准差)。±1 表示贴上/下轨。"""
    c = df["close"]
    m = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=1)
    return _safe_div(c - m, 2.0 * sd)


def f_stoch_14(df: pd.DataFrame) -> pd.Series:
    """随机指标 %K(14)：收盘在近 14 日高低区间中的位置。"""
    ll = df["low"].rolling(14, min_periods=14).min()
    hh = df["high"].rolling(14, min_periods=14).max()
    return _safe_div(df["close"] - ll, hh - ll)


# ---------------------------------------------------------------- 波动率
def f_vol_20(df: pd.DataFrame) -> pd.Series:
    """20 日已实现波动率（年化）。低波动因子的原料。"""
    r = df["close"].pct_change()
    return r.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(TRADING_DAYS)


def f_vol_ratio(df: pd.DataFrame) -> pd.Series:
    """短/长波动比：>1 说明波动在放大，常伴随趋势启动或恐慌。"""
    r = df["close"].pct_change()
    s = r.rolling(10, min_periods=10).std(ddof=1)
    l = r.rolling(60, min_periods=60).std(ddof=1)
    return _safe_div(s, l)


def f_atr_pct(df: pd.DataFrame) -> pd.Series:
    """ATR / 收盘价。跨品种可比的波动率度量。"""
    return _safe_div(_atr(df, 14), df["close"])


def f_downside_ratio(df: pd.DataFrame) -> pd.Series:
    """下行波动占比：只算负收益的标准差 / 全样本标准差。偏度代理。"""
    r = df["close"].pct_change()
    dn = r.where(r < 0, 0.0).rolling(20, min_periods=20).std(ddof=1)
    tot = r.rolling(20, min_periods=20).std(ddof=1)
    return _safe_div(dn, tot)


def f_maxdd_60(df: pd.DataFrame) -> pd.Series:
    """60 日内的回撤深度（当前价距 60 日高点的距离）。"""
    hh = df["close"].rolling(60, min_periods=60).max()
    return _safe_div(df["close"], hh) - 1.0


# ---------------------------------------------------------------- 量能
def f_vol_surge(df: pd.DataFrame) -> pd.Series:
    """成交量相对 20 日均量的放大倍数减 1。放量常是趋势确认。"""
    v = df["volume"]
    return _safe_div(v, v.rolling(20, min_periods=20).mean()) - 1.0


def f_pv_corr(df: pd.DataFrame) -> pd.Series:
    """20 日价量相关性。正相关=价涨量增（健康），负相关=背离。"""
    r = df["close"].pct_change()
    v = df["volume"].pct_change()
    return r.rolling(20, min_periods=20).corr(v)


def f_obv_slope(df: pd.DataFrame) -> pd.Series:
    """OBV 的 20 日变化 / 20 日总量。量能累积速度。"""
    sign = np.sign(df["close"].diff()).fillna(0.0)
    obv = (sign * df["volume"]).cumsum()
    return _safe_div(obv.diff(20), df["volume"].rolling(20, min_periods=20).sum())


def f_illiq(df: pd.DataFrame) -> pd.Series:
    """Amihud 非流动性：|收益| / 成交额，20 日均值。数值小，乘 1e9 便于展示。"""
    if "amount" not in df.columns:
        return _nan_series(df)
    r = df["close"].pct_change().abs()
    return (_safe_div(r, df["amount"]) * 1e9).rolling(20, min_periods=20).mean()


def f_oi_change(df: pd.DataFrame) -> pd.Series:
    """持仓量 20 日变化率。期货特有：增仓上涨比减仓上涨更可信。"""
    if "open_interest" not in df.columns:
        return _nan_series(df)
    oi = pd.to_numeric(df["open_interest"], errors="coerce")
    oi = oi.where(oi > 0)
    return oi.pct_change(20)


# ---------------------------------------------------------------- 形态 / 结构
def f_hl_range(df: pd.DataFrame) -> pd.Series:
    """20 日平均振幅 (high-low)/close。"""
    return _safe_div(df["high"] - df["low"], df["close"]).rolling(
        20, min_periods=20
    ).mean()


def f_close_pos(df: pd.DataFrame) -> pd.Series:
    """10 日平均「收盘在当日振幅中的位置」。>0.5 说明收盘偏强。"""
    return _safe_div(df["close"] - df["low"], df["high"] - df["low"]).rolling(
        10, min_periods=10
    ).mean()


def f_gap(df: pd.DataFrame) -> pd.Series:
    """10 日平均跳空幅度。"""
    return _safe_div(df["open"], df["close"].shift(1)).sub(1.0).rolling(
        10, min_periods=10
    ).mean()


def f_macd_hist(df: pd.DataFrame) -> pd.Series:
    """MACD 柱 / 收盘价。把趋势强度归一化成跨品种可比的量。"""
    c = df["close"]
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    return _safe_div(dif - dea, c)


# ---------------------------------------------------------------- 注册表
@dataclass
class FactorSpec:
    key: str
    label: str
    func: Callable[[pd.DataFrame], pd.Series]
    category: str
    desc: str
    # 仅作展示提示：历史上这类因子高分倾向涨(+1)还是跌(-1)。
    # **回测里绝不直接用**，方向一律由滚动 IC 决定，否则就是用未来信息。
    hint_dir: int = 1


FACTORS: dict[str, FactorSpec] = {}


def _reg(spec: FactorSpec) -> None:
    FACTORS[spec.key] = spec


_reg(FactorSpec("mom_20", "动量 20 日", f_mom_20, "趋势", "20 日涨跌幅，短周期趋势延续。"))
_reg(FactorSpec("mom_60", "动量 60 日", f_mom_60, "趋势", "60 日涨跌幅，商品上最经典的趋势因子。"))
_reg(FactorSpec("mom_120", "动量 120 日", f_mom_120, "趋势", "120 日涨跌幅，半年趋势。"))
_reg(FactorSpec("mom_250", "动量 250 日", f_mom_250, "趋势", "250 日涨跌幅，年度趋势。"))
_reg(FactorSpec("ma_gap_20", "偏离 20 日均线", f_ma_gap_20, "趋势", "收盘相对 20 日均线的偏离度。"))
_reg(FactorSpec("ma_gap_60", "偏离 60 日均线", f_ma_gap_60, "趋势", "收盘相对 60 日均线的偏离度。"))
_reg(FactorSpec("ma_slope_20", "均线斜率 20", f_ma_slope_20, "趋势", "20 日均线相对 10 日前的斜率，比价格更平滑。"))
_reg(FactorSpec("breakout_pos", "通道位置 55", f_breakout_pos, "趋势", "收盘在 55 日高低区间中的位置，唐奇安突破的连续化。"))
_reg(FactorSpec("efficiency", "效率比 20", f_efficiency, "趋势", "净位移/总路程，衡量趋势的顺滑程度。"))

_reg(FactorSpec("rev_5", "反转 5 日", f_rev_5, "反转", "5 日跌幅（取负），短期超跌反弹。", -1))
_reg(FactorSpec("rev_20", "反转 20 日", f_rev_20, "反转", "20 日跌幅（取负）。", -1))
_reg(FactorSpec("rsi_14", "RSI 14", f_rsi_14, "反转", "相对强弱指标，标准化后使用。", -1))
_reg(FactorSpec("bb_pos_20", "布林带位置", f_bb_pos_20, "反转", "收盘相对布林带中轨的标准化偏离。", -1))
_reg(FactorSpec("stoch_14", "随机指标 14", f_stoch_14, "反转", "收盘在近 14 日高低区间中的位置。", -1))

_reg(FactorSpec("vol_20", "波动率 20", f_vol_20, "波动", "20 日年化已实现波动率，低波动因子的原料。", -1))
_reg(FactorSpec("vol_ratio", "短长波动比", f_vol_ratio, "波动", "10 日波动 / 60 日波动，衡量波动放大。", -1))
_reg(FactorSpec("atr_pct", "ATR 占比", f_atr_pct, "波动", "ATR/收盘价，跨品种可比的波动度量。", -1))
_reg(FactorSpec("downside_ratio", "下行波动占比", f_downside_ratio, "波动", "负收益波动 / 总波动，偏度代理。", -1))
_reg(FactorSpec("maxdd_60", "60 日回撤", f_maxdd_60, "波动", "当前价距 60 日高点的距离。", -1))

_reg(FactorSpec("vol_surge", "放量倍数", f_vol_surge, "量能", "成交量相对 20 日均量的放大倍数。"))
_reg(FactorSpec("pv_corr", "价量相关 20", f_pv_corr, "量能", "20 日收益与成交量变化的相关性。"))
_reg(FactorSpec("obv_slope", "OBV 斜率", f_obv_slope, "量能", "OBV 的 20 日变化 / 20 日总量。"))
_reg(FactorSpec("illiq", "非流动性", f_illiq, "量能", "Amihud 非流动性指标，需成交额。", -1))
_reg(FactorSpec("oi_change", "持仓量变化", f_oi_change, "量能", "20 日持仓量变化率，期货特有。"))

_reg(FactorSpec("hl_range", "平均振幅", f_hl_range, "形态", "20 日平均 (高-低)/收盘。", -1))
_reg(FactorSpec("close_pos", "收盘位置", f_close_pos, "形态", "10 日平均收盘在当日振幅中的位置。"))
_reg(FactorSpec("gap", "跳空幅度", f_gap, "形态", "10 日平均跳空幅度。"))
_reg(FactorSpec("macd_hist", "MACD 柱", f_macd_hist, "形态", "MACD 柱/收盘价，趋势强度的归一化表达。"))


CATEGORIES: list[str] = ["趋势", "反转", "波动", "量能", "形态"]

DEFAULT_SINGLE = "mom_60"
DEFAULT_MULTI = ["mom_20", "mom_60", "mom_120", "rev_5", "vol_20", "breakout_pos"]


# ---------------------------------------------------------------- 计算
def raw(df: pd.DataFrame, key: str) -> pd.Series:
    """算单个因子的原始值。返回与 df 同索引的 Series。"""
    spec = FACTORS.get(key)
    if spec is None:
        raise KeyError(f"未知因子 {key}")
    s = spec.func(df)
    s = pd.to_numeric(s, errors="coerce")
    # inf 一律当缺失：除零、极端值都会产生 inf，进了 z-score 会污染整段
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.astype(float)


def raw_frame(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """批量算原始因子值。列名即因子 key。"""
    if not keys:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame({k: raw(df, k) for k in keys}, index=df.index)


def standardize(
    s: pd.Series,
    window: int = 252,
    method: str = "z",
    clip: float = 4.0,
) -> pd.Series:
    """滚动标准化。**PIT 的核心**：统计量只取 ≤T 的窗口。

    - `method="z"`   : 滚动 z-score，按 ±clip 截断（等价于温和的缩尾）
    - `method="rank"` : 滚动分位，映射到 [-0.5, 0.5]，对异常值更稳健

    不用全样本均值/标准差 —— 那是把未来的分布信息泄漏到今天。
    """
    window = int(window)
    mp = max(20, window // 4)
    roll = s.rolling(window, min_periods=mp)
    if method == "rank":
        return (roll.rank(pct=True) - 0.5).astype(float)
    m = roll.mean()
    sd = roll.std(ddof=1)
    z = _safe_div(s - m, sd)
    return z.clip(-clip, clip).astype(float)


def factor_frame(
    df: pd.DataFrame,
    keys: list[str],
    window: int = 252,
    method: str = "z",
) -> pd.DataFrame:
    """标准化后的因子矩阵，可直接喂给策略或模型。"""
    if not keys:
        return pd.DataFrame(index=df.index)
    out = {}
    for k in keys:
        out[k] = standardize(raw(df, k), window=window, method=method)
    return pd.DataFrame(out, index=df.index)


def composite(
    fmat: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """把多个标准化因子加权合成一个分数。默认等权。

    权重里缺的因子按 0 处理；全部缺失的行返回 NaN（预热期）。
    """
    if fmat.empty or fmat.shape[1] == 0:
        return pd.Series(dtype=float)
    w = weights or {c: 1.0 for c in fmat.columns}
    total = sum(abs(w.get(c, 0.0)) for c in fmat.columns)
    if total <= 0:
        return pd.Series(np.nan, index=fmat.index, dtype=float)
    acc = pd.Series(0.0, index=fmat.index)
    cnt = pd.Series(0.0, index=fmat.index)
    for c in fmat.columns:
        wc = w.get(c, 0.0)
        if wc == 0:
            continue
        v = fmat[c]
        acc = acc.add(v * wc, fill_value=0.0)
        cnt = cnt.add(v.notna().astype(float) * abs(wc), fill_value=0.0)
    # 按「实际有值的权重」归一，避免某个因子缺失把整行拉成 0
    return _safe_div(acc, cnt)


def available(df: pd.DataFrame, key: str, min_obs: int = 60) -> bool:
    """这个因子在当前数据上是否有足够观测值（如美股没有持仓量）。"""
    try:
        return raw(df, key).notna().sum() >= min_obs
    except Exception:
        return False


def filter_available(df: pd.DataFrame, keys: list[str], min_obs: int = 60) -> list[str]:
    return [k for k in keys if k in FACTORS and available(df, k, min_obs)]
