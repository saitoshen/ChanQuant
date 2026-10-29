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


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """取衍生数据列（期限结构 / 期权），没有就返回全 NaN。

    这些列由 `auxdata.attach()` 在加载时并进来，美股没有、期货里没下过
    衍生数据的品种也没有。统一返回 NaN 而不是报错，`filter_available()`
    就会自动把这些因子从该标的的因子列表里剔掉。
    """
    if name not in df.columns:
        return _nan_series(df)
    return pd.to_numeric(df[name], errors="coerce").astype(float)


def _oi(df: pd.DataFrame) -> pd.Series:
    """持仓量，统一取法。

    **两条数据链路的列名不一样**：Tushare 日线 CSV 叫 `open_interest`，天勤
    分钟线叫 `open_oi`（还多一个收盘时的 `close_oi`）。因子直接写
    `df["open_interest"]` 在分钟线上就是 KeyError / 全 NaN —— 曾经 `oi_change`
    在 1h/5m 上整列空着，看起来像「天勤不给持仓量」，其实数据一直在。
    """
    for name in ("open_interest", "open_oi"):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").astype(float)
    return _nan_series(df)


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """ln(a/b)，任一侧非正时返回 NaN。"""
    ok = (a > 0) & (b > 0)
    return np.log(a.where(ok) / b.where(ok))


def _day_diff(s: pd.Series, n: int = 20) -> pd.Series:
    """按**交易日**取 n 天变化。

    日线上 `diff(n)` 本来就是 n 天。分钟线上不是：5m 的 `diff(20)` 只是 100
    分钟，而衍生数据（期限结构 / 期权 IV）本身是「一天一个快照」，由
    `auxdata.attach_intraday` 铺到当天每根 bar 上 —— 直接 diff 得到的是一串 0，
    每天开盘那几根再跳一下，看着像信号其实是采样伪影。

    这里先压成日频、diff 完再映射回每根 bar，两个周期口径就一致了。
    """
    idx = s.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return s.diff(n)
    day = idx.normalize()
    if len(day) == len(day.unique()):        # 一天一根，本来就是日线
        return s.diff(n)
    daily = s.groupby(day).last()
    return pd.Series(np.asarray(day.map(daily.diff(n)), dtype=float), index=idx)


# ---------------------------------------------------------------- 展期 / 期限结构
# 展期因子全部来自「同一天、不同到期日」的合约价格，这是主力连续合约给不了的
# 信息。数据来源见 `ttrade/contracts.py`。
#
# 期限结构有**三套口径**，共用下面这套 `pre` 前缀实现：
#   "c"   → c1/c2/c3：当日最近、次近、第三近的三个合约（原口径）
#   "o3_" → o3_1..3：先从持仓量前 3 里挑，再按到期日取最近三腿
#   "o6_" → o6_1..3：先从持仓量前 6 里挑，再按到期日取最近三腿
#
# 为什么要多两套：主力合约月份是跳跃的（螺纹/热卷是 1/5/10，豆粕/棕榈是 1/5/9），
# 纯按到期日选会让 c1 常常落在冷门月上 —— 螺纹有七成的日子主力根本不在 c1/c2/c3
# 里，c1 的持仓量只占全品种的 0.1%，棕榈有 27.7% 的日子 c1 成交量为 0。筛活跃能
# 把这种空壳合约滤掉。但实测这不是全局更优（全 81 品种上只改善 43% 的品种、
# 方向一致性还下降），所以三套并列而不是替换，让用户在因子分析页按品种自己比。
def _roll_yield(df: pd.DataFrame, pre: str = "c") -> pd.Series:
    return _safe_div(_col(df, f"{pre}1_close"), _col(df, f"{pre}2_close")) - 1.0


def _roll_annual(df: pd.DataFrame, pre: str = "c") -> pd.Series:
    ry = _roll_yield(df, pre)
    gap = (_col(df, f"{pre}2_dte") - _col(df, f"{pre}1_dte")).clip(lower=1.0)
    return ry * 365.0 / gap


def _roll_curv(df: pd.DataFrame, pre: str = "c") -> pd.Series:
    l1 = _safe_log_ratio(_col(df, f"{pre}1_close"), _col(df, f"{pre}2_close"))
    l2 = _safe_log_ratio(_col(df, f"{pre}2_close"), _col(df, f"{pre}3_close"))
    g1 = (_col(df, f"{pre}2_dte") - _col(df, f"{pre}1_dte")).clip(lower=1.0)
    g2 = (_col(df, f"{pre}3_dte") - _col(df, f"{pre}2_dte")).clip(lower=1.0)
    return (l2 / g2 - l1 / g1) * 365.0


def _roll_change(df: pd.DataFrame, pre: str = "c") -> pd.Series:
    return _day_diff(_roll_annual(df, pre), 20)


def f_roll_yield(df: pd.DataFrame) -> pd.Series:
    """展期收益率：近月相对次近月的价差率 (P1/P2 - 1)。

    现货溢价（backwardation）时为正——近月比远月贵，多头移仓能赚到价差；
    期货溢价（contango）时为负，长期持有多头要付出移仓成本。
    这是商品期货上最经典的一类横截面/时序因子。
    """
    return _roll_yield(df, "c")


def f_roll_annual(df: pd.DataFrame) -> pd.Series:
    """展期年化收益率：把 P1/P2 的价差按两个合约的到期间隔折算成年化。

    不同品种、不同时点的近远月间隔差异很大（14 天到 160 天都有），
    不年化就没法横向比较——同样是 1% 的价差，隔 15 天和隔 150 天含义完全不同。
    """
    return _roll_annual(df, "c")


def f_roll_curv(df: pd.DataFrame) -> pd.Series:
    """展期曲率：近端与远端年化斜率之差。

    近端斜率 k1 = ln(P1/P2)/(T2-T1)，远端 k2 = ln(P2/P3)/(T3-T2)，曲率 = k2-k1。
    用「斜率之差」而不是三点的二阶差分，是因为三个合约的到期间隔通常不等距，
    直接做二阶差分会被间隔差异污染。曲率为负说明曲线在近端更陡（近月紧张），
    为正说明远端更陡。
    """
    return _roll_curv(df, "c")


def f_roll_change(df: pd.DataFrame) -> pd.Series:
    """展期变化：年化展期收益率的 20 日变化。

    用年化值而不是原始价差，是为了减少换月日的跳变——主力合约一换月，
    c1/c2 直接换成另一对合约，原始价差会出现台阶；年化后台阶小得多。
    """
    return _roll_change(df, "c")


def f_roll_intraday(df: pd.DataFrame) -> pd.Series:
    """日内价差漂移：收盘时的 c1/c2 价差 − 当天第一根 bar 的同一对价差。

    只有分钟级真实合约才算得出来（`tqcurve` 构建期限结构时顺手写进
    `data_tq/curve/` 的 `roll_intra` 列）。正的表示近月当天相对远月走强，
    是「谁在被买」的即时读数，日线的展期收益率把这个过程抹平了。

    实测很弱：次日 IC 约 0.011（t≈2.1），5 日以上归零。别当主力因子用，
    适合当短线择时的辅助项，或者和 `roll_yield` 一起看「水平高 + 还在走强」。
    """
    return _col(df, "roll_intra")


def f_near_oi_share(df: pd.DataFrame) -> pd.Series:
    """近月持仓占比 c1_oi / (c1_oi + c2_oi)。

    接近 1 说明资金都挤在近月，往往对应现货紧张或交割博弈；
    偏低说明主力已经移到远月，近月价格参考意义下降。

    只在原口径上提供 —— o3_ 那套的三腿本来就是「持仓量前 3」，再算占比等于
    「最大的占前两个的比例」，恒在 0.5~0.7 之间，没有信息量。
    """
    a, b = _col(df, "c1_oi"), _col(df, "c2_oi")
    return _safe_div(a, a + b)


# ---- 筛活跃口径：同一套逻辑，换个列前缀 ----
def f_roll_yield_o3(df: pd.DataFrame) -> pd.Series:
    """展期收益率（持仓量前 3 口径）：只从当日持仓最大的 3 个合约里取最近两腿。"""
    return _roll_yield(df, "o3_")


def f_roll_annual_o3(df: pd.DataFrame) -> pd.Series:
    """展期年化（持仓量前 3 口径）。"""
    return _roll_annual(df, "o3_")


def f_roll_curv_o3(df: pd.DataFrame) -> pd.Series:
    """展期曲率（持仓量前 3 口径）。三腿都落在活跃月上，报价噪声比原口径小。"""
    return _roll_curv(df, "o3_")


def f_roll_change_o3(df: pd.DataFrame) -> pd.Series:
    """展期变化（持仓量前 3 口径）。"""
    return _roll_change(df, "o3_")


def f_roll_yield_o6(df: pd.DataFrame) -> pd.Series:
    """展期收益率（持仓量前 6 口径）。

    候选池比 OI3 宽，第三腿更可能落在真正活跃的远月上；代价是换腿更频繁
    （实测日均 7% vs 原口径 5%）。在流动性前 40 的品种池上这一档全面优于原口径，
    但全品种看方向一致性不如原口径 —— 所以是并列提供，不做替换。
    """
    return _roll_yield(df, "o6_")


def f_roll_annual_o6(df: pd.DataFrame) -> pd.Series:
    """展期年化（持仓量前 6 口径）。"""
    return _roll_annual(df, "o6_")


def f_roll_curv_o6(df: pd.DataFrame) -> pd.Series:
    """展期曲率（持仓量前 6 口径）。"""
    return _roll_curv(df, "o6_")


def f_roll_change_o6(df: pd.DataFrame) -> pd.Series:
    """展期变化（持仓量前 6 口径）。"""
    return _roll_change(df, "o6_")


def f_roll_main_oi(df: pd.DataFrame) -> pd.Series:
    """主力/次主力价差率：近端取持仓最大的腿，远端取它**之后**持仓最大的腿。

    和上面那套「按到期日排三腿再取前两腿」不是一回事：这里两端都由持仓量决定，
    所以近端一定是最活跃的合约、远端一定比它更远。中国商品期货的主力月份是跳跃的
    （螺纹只做 1/5/10），按到期日排出来的「近月」经常是个空壳合约，持仓口径不会。

    先试 6 腿（o6_），没数据退回 3 腿（o3_）—— 只下了衍生数据基础档的品种只有
    o3_。取不到任何一条腿时整列 NaN，`filter_available()` 会自动把它剔掉。

    最后做 5 日均值：单日价差在换腿当天会跳一下，平滑之后才是「期限结构的陡峭度」
    而不是「今天轮到哪个合约」。
    """
    def pick(pre: str):
        close = np.column_stack([
            _col(df, f"{pre}{i}_close").to_numpy(dtype=float) for i in (1, 2, 3)
        ])
        oi = np.column_stack([
            _col(df, f"{pre}{i}_oi").to_numpy(dtype=float) for i in (1, 2, 3)
        ])
        ok = np.isfinite(close) & np.isfinite(oi)
        # 用 -inf 占位，argmax 就会跳过缺失的腿；全缺时 has_* 负责把结果打成 NaN
        i_main = np.where(ok, oi, -np.inf).argmax(axis=1)
        has_main = ok.any(axis=1)
        after = (np.arange(3)[None, :] > i_main[:, None]) & ok
        far = np.where(after, oi, -np.inf)
        has_far = np.isfinite(far).any(axis=1)
        i_far = far.argmax(axis=1)
        rows = np.arange(len(df))
        near = pd.Series(np.where(has_main, close[rows, i_main], np.nan), index=df.index)
        nextm = pd.Series(
            np.where(has_main & has_far, close[rows, i_far], np.nan), index=df.index
        )
        return near, nextm

    near, nextm = pick("o6_")
    if near.notna().sum() == 0:
        near, nextm = pick("o3_")
    return _safe_div(near - nextm, nextm).rolling(5, min_periods=3).mean()


# ---------------------------------------------------------------- 期权 / 波动率曲面
# 期权因子是唯一「不靠价格历史就能得到」的一类信息：市场对未来波动的定价
# 直接写在期权价格里。数据来源见 `ttrade/options.py`。
def f_iv_atm(df: pd.DataFrame) -> pd.Series:
    """近月平值隐含波动率（ATM IV）。

    由当日近月期权整条微笑做二次拟合后在 K=F 处取值，不是单点插值。
    IV 高说明市场为未来波动付了高价，常见于恐慌或事件前。
    """
    return _col(df, "iv_near")


def f_iv_term(df: pd.DataFrame) -> pd.Series:
    """IV 期限结构斜率：(IV_次月 - IV_近月) / 间隔天数 × 30，即「每月多少 IV 点」。

    正值＝远月更贵（正常，波动率曲线通常上倾）；负值＝近月恐慌，
    是压力期最典型的信号之一（现货紧张、事件冲击）。
    """
    iv1, iv2 = _col(df, "iv_near"), _col(df, "iv_next")
    gap = (_col(df, "dte_next") - _col(df, "dte_near")).clip(lower=1.0)
    return (iv2 - iv1) / gap * 30.0


def f_iv_skew(df: pd.DataFrame) -> pd.Series:
    """偏度：IV 微笑在平值处对 ln(K/F) 的一阶导 dIV/dlnK。

    正值＝虚值看涨比虚值看跌贵（市场怕涨）；负值＝虚值看跌更贵（市场怕跌）。
    商品期权多数时间是负偏，因为下跌往往伴随波动率跳升。
    """
    return _col(df, "skew")


def f_iv_rv(df: pd.DataFrame) -> pd.Series:
    """方差风险溢价：近月 ATM IV 减去 20 日已实现波动率。

    期权通常比事后实现的波动更贵（卖方收溢价），所以这个差值长期偏正。
    差值被压到 0 附近甚至转负，说明期权相对便宜，或市场已进入高波动状态。
    两个量都是年化的，可以直接相减。
    """
    return _col(df, "iv_near") - f_vol_20(df)


def f_iv_change(df: pd.DataFrame) -> pd.Series:
    """IV 的 20 日变化。衡量波动率预期的动量。"""
    return _day_diff(_col(df, "iv_near"), 20)


def f_pcr(df: pd.DataFrame) -> pd.Series:
    """看跌/看涨持仓比（Put/Call Ratio），取对数。

    >1 说明看跌期权持仓更多，情绪偏空。取对数是因为这个比值天然是
    乘性的（0.5 与 2.0 是同一件事的两个方向），取完对数才左右对称。
    """
    r = _col(df, "pcr")
    return np.log(r.where(r > 0))


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
    oi = _oi(df)
    return oi.where(oi > 0).pct_change(20)


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
    # 用户自定义因子（Agent 生成）自带公式与源码：它们不在 FACTOR_DOCS 里，
    # `inspect.getsource()` 也取不到 —— `exec` 出来的函数没有对应文件。
    formula: str = ""
    code: str = ""


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

# 展期：需要真实合约（同一天不同到期日），来自 data_futures_aux
_reg(FactorSpec("roll_yield", "展期收益率", f_roll_yield, "展期",
                "近月/次近月价差率，现货溢价为正、期货溢价为负。"))
_reg(FactorSpec("roll_annual", "展期年化", f_roll_annual, "展期",
                "展期收益率按近远月间隔折算成年化，跨品种可比。"))
_reg(FactorSpec("roll_curv", "展期曲率", f_roll_curv, "展期",
                "近端与远端年化斜率之差，刻画期限结构弯曲程度。"))
_reg(FactorSpec("roll_change", "展期变化", f_roll_change, "展期",
                "年化展期收益率的 20 日变化，反映曲线陡峭度的动量。"))
_reg(FactorSpec("near_oi_share", "近月持仓占比", f_near_oi_share, "展期",
                "近月持仓 / (近月+次近) 持仓，衡量资金是否挤在近月。"))
# 日内口径的展期：需要分钟级真实合约（天勤），Tushare 那份 aux 里没有这一列
_reg(FactorSpec("roll_intraday", "展期·日内价差漂移", f_roll_intraday, "展期",
                "收盘价差减当日开盘价差，近月当天相对远月的强弱变化，需分钟级合约。"))

# 展期·OI3 / OI6：同样四个公式，但换一套选合约的口径。
# 上面的「展期」是按到期日取最近三腿；这两类是先按持仓量取前 3 / 前 6 个活跃合约，
# 再在其中按到期日排三腿。原因是中国商品期货的主力月份是跳跃的（螺纹/热卷/铁矿
# 只做 1/5/10，豆粕/棕榈/豆油只做 1/5/9），纯按到期日常常整条曲线都落在冷门月上。
# 三套并列而不是替换：全 81 个品种上原口径的方向一致性最好（IC>0 占 79%），
# OI 口径在流动性前 40 的品种上更强，谁合适取决于做什么品种，让用户自己比。
_reg(FactorSpec("roll_yield_o3", "展期收益率·OI3", f_roll_yield_o3, "展期 OI3",
                "持仓量前 3 合约中的近月/次近月价差率。"))
_reg(FactorSpec("roll_annual_o3", "展期年化·OI3", f_roll_annual_o3, "展期 OI3",
                "OI3 口径的展期收益率按近远月间隔年化。"))
_reg(FactorSpec("roll_curv_o3", "展期曲率·OI3", f_roll_curv_o3, "展期 OI3",
                "OI3 口径近端与远端年化斜率之差。"))
_reg(FactorSpec("roll_change_o3", "展期变化·OI3", f_roll_change_o3, "展期 OI3",
                "OI3 口径年化展期收益率的 20 日变化。"))
_reg(FactorSpec("roll_yield_o6", "展期收益率·OI6", f_roll_yield_o6, "展期 OI6",
                "持仓量前 6 合约中的近月/次近月价差率。"))
_reg(FactorSpec("roll_annual_o6", "展期年化·OI6", f_roll_annual_o6, "展期 OI6",
                "OI6 口径的展期收益率按近远月间隔年化。"))
_reg(FactorSpec("roll_curv_o6", "展期曲率·OI6", f_roll_curv_o6, "展期 OI6",
                "OI6 口径近端与远端年化斜率之差。"))
_reg(FactorSpec("roll_change_o6", "展期变化·OI6", f_roll_change_o6, "展期 OI6",
                "OI6 口径年化展期收益率的 20 日变化。"))
# 主力/次主力：两端都按持仓量选，和上面「按到期日排三腿」是两套选合约逻辑。
# 归在「展期」里（用户自己建的那条也在这类），但依赖 o6_ 那套列，没有时退回 o3_
_reg(FactorSpec("roll_main_oi", "展期收益率·主力次主力（持仓量口径）", f_roll_main_oi, "展期",
                "近端取持仓最大的腿、远端取其后持仓最大的腿，5 日均值。正值=backwardation。"))

# 期权：需要期权日线反解出的波动率曲面，来自 data_futures_aux
_reg(FactorSpec("iv_atm", "平值 IV", f_iv_atm, "期权",
                "近月平值隐含波动率，来自期权微笑的二次拟合。"))
_reg(FactorSpec("iv_term", "IV 期限结构", f_iv_term, "期权",
                "次月与近月 IV 的月度斜率，负值代表近月恐慌。"))
_reg(FactorSpec("iv_skew", "IV 偏度", f_iv_skew, "期权",
                "IV 对 ln(K/F) 的斜率，负偏＝市场更怕跌。"))
_reg(FactorSpec("iv_rv", "IV-RV 价差", f_iv_rv, "期权",
                "平值 IV 减 20 日已实现波动率，即方差风险溢价。"))
_reg(FactorSpec("iv_change", "IV 变化", f_iv_change, "期权",
                "平值 IV 的 20 日变化，波动率预期的动量。"))
_reg(FactorSpec("pcr", "看跌看涨持仓比", f_pcr, "期权",
                "Put/Call 持仓比取对数，>0 表示情绪偏空。"))


CATEGORIES: list[str] = [
    "趋势", "反转", "波动", "量能", "形态",
    "日内", "时段",
    "展期", "展期 OI3", "展期 OI6", "期权",
    "季节性", "品种季节",
]

DEFAULT_SINGLE = "mom_60"
DEFAULT_MULTI = ["mom_20", "mom_60", "mom_120", "rev_5", "vol_20", "breakout_pos"]

# 需要额外数据（真实合约 / 期权）才能算的因子。美股没有这些数据，
# 界面上要把它们和「算不出来」区分开，而不是显示成一堆空行。
# 展期三个类别（原口径 / OI3 / OI6）都读同一批 aux 列，用前缀匹配一并收进来，
# 免得以后再加一档（比如 OI9）时忘了同步这里。
AUX_KEYS = [k for k, s in FACTORS.items()
            if s.category.startswith("展期") or s.category == "期权"]


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


# ---------------------------------------------------------------- 用户自定义因子
def _load_user_factors() -> int:
    """把用户目录里的自定义因子挂进来。

    放在文件**末尾**：`userfactors` 要 `from ttrade.factors import FactorSpec`，
    这时本模块已经执行到最底下，`FactorSpec` / `_col` 这些都在了。放在开头会
    变成循环导入。

    失败一律吞掉 —— 用户手改 JSON 改坏了，不该让整个程序起不来。
    """
    try:
        from ttrade import userfactors as UF

        return UF.register_all_into(FACTORS)
    except Exception:  # noqa: BLE001
        return 0


USER_FACTOR_COUNT = _load_user_factors()

# 日内 / 时段 / 季节性三类单独成文件：主文件已经八百行，再往里塞二十个因子
# 会变成一锅粥。它们在模块末尾注册，是因为要用这里的 `FactorSpec` —— 放开头
# 就是循环导入。
from ttrade import factors_intraday as _FI   # noqa: E402
from ttrade import factors_seasonal as _FS   # noqa: E402

_INTRA_FACTORS = _FI.register_into(FACTORS, FactorSpec)
_SEASON_FACTORS = _FS.register_into(FACTORS, FactorSpec)

# 用户因子可能带来新类别（默认叫「自定义」）。`CATEGORIES` 是界面上分组用的
# 静态列表，不补进去的话新因子在「策略研究」页的因子下拉里根本不出现 ——
# 注册成功了却选不到，比注册失败还难排查。
for _spec in FACTORS.values():
    if _spec.category not in CATEGORIES:
        CATEGORIES.append(_spec.category)
