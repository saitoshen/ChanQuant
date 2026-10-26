"""期权日线与隐含波动率。

## 只保留聚合结果，不落原始期权行情

一天的全市场期权日线约 1.8 万行，2017 年至今累计约 2500 万行（1.5GB+）。
这些数据对因子研究没有直接价值——有用的是**波动率曲面的几个截面**：
平值 IV、期限结构、偏度、看跌看涨持仓比。所以默认把每天的期权行情读进来、
算完就丢，只把每品种一行的摘要落盘（约 15MB）。

需要原始期权明细做别的研究时，用 `--keep-raw` 另存到 `data_options_raw/`。

## 定价模型

国内商品期权是**期货期权**，标的不是现货而是某个具体月份的期货合约，
因此用 Black-76（带贴现的 Black 模型）：

    c = e^{-rT}[F·N(d1) - K·N(d2)]
    p = e^{-rT}[K·N(-d2) - F·N(-d1)]

反解用二分法（对 sigma 单调，二分比 Newton 稳，不会因为 vega→0 发散），
向量化到整天的合约上，50 次迭代足够收敛到 1e-9。

**近似说明**：大商所/郑商所的期权是美式，用欧式 Black-76 反解会低估美式
权利金对应的时间价值，因此算出的 IV 在深度实值处偏低。对「平值附近、剩余
期限 15~180 天」这个区间，美式与欧式差异很小，做因子足够；要精确定价得用
BAW 或二叉树，那是另一件事。

## 微笑拟合

不对单点做插值。深度虚值期权价格只有一两个最小变动价位，反解出的 IV 会
离谱（实测某天 TA 的 90% 虚值 put 解出 58% 的 IV，而平值只有 13%）。

改成在 ±20% 的 log-moneyness 区间内用**二次多项式加权最小二乘**拟合整条微笑，
权重取 sqrt(持仓量)：
- 平值 IV = 拟合曲线在 m=0 处的值
- 偏度 = 一阶导 dIV/dlnK（m=0 处的斜率）
- 曲率 = 二阶导

这样单个报价的噪声会被摊平，且「偏度」有了明确的经济含义：标的价格下跌时
IV 上升多少。

## 三条口径，每一条都是踩坑之后才定下来的

1. **标的价 F 用期货结算价**，不是收盘价。期权结算价与期货结算价是同一次
   交易所结算的产物；用收盘价会让同一行权价的 call/put 解出相差近 10 个
   IV 点（2017-12-29 豆粕：平价隐含远期 2852 = 结算价，收盘价 2828）。
2. **只取虚值**（K≥F 取 call，K<F 取 put）。实值期权几乎没有时间价值，
   一个最小变动价位就能让 σ 翻几倍（白糖 K=7400 的实值 put 解出 31%，
   平值只有 10%）。
3. **权利金严格大于 1 元**。贴着最小变动价位的报价是取整产物，不是价格
   （沪铜 2019-06-03 的 4 个 1.0 元报价把曲率从 0 拉到 6.2）。

判断这三条有没有做对，靠的是可验证的恒等式而不是肉眼看曲线：
put-call 平价反解出的隐含远期必须等于结算价，同一行权价的 call/put IV
必须几乎相等。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import futures as fut
from .contracts import SUFFIX_EX, merge_write, write_atomic

PAGE = 5000
OPT_PAGE = 15000
SLEEP = 0.05
RISK_FREE = 0.02        # 无风险利率，对 15~180 天期权的影响 < 0.1 个 IV 点
# 权利金必须**高于** 1 元（严格大于）。贴着最小变动价位的报价没有信息量：
# 沪铜 2019-06-03 有 4 个报价恰好是 1.0 元，反解出的 IV 从 0.144 一路爬到 0.218，
# 而同一张微笑在 ±8% 以内是平的（0.150）—— 拟合曲率被这 4 个点从 0 拉到 6.2。
MIN_PRICE = 1.0
# 剩余期限下限取 15 天，不是 7 天。临近到期的期权，±20% moneyness 覆盖的都是
# 好几个标准差之外的行权价，IV 会在两翼同时爆掉：实测 2024-06 三个交易日，
# DTE ≤ 15 的拟合曲率中位数是 15.7，DTE > 15 的是 −0.05，差 300 倍。这不是
# 市场信号，纯粹是「快到期」这个日历效应，留在因子序列里会变成一个锯齿。
# 抬到 15 天，覆盖的品种数不变（40 个），曲率 p90 从 12.6 降到 5.1。
MIN_DTE = 15
MAX_DTE = 180
MONEY_BAND = 0.20       # log-moneyness 拟合区间 ±20%
MIN_POINTS = 5          # 一条微笑至少要这么多个报价点（二次拟合 3 个参数，5 点剩 2 个自由度）
IV_LO, IV_HI = 0.005, 3.0
IV_ITERS = 50

# 国内商品期权从 2017-03-31（大商所豆粕）开始，之前只有中金所股指期权且标的是
# 指数不是期货。早于这个日期不用请求 opt_daily —— 20 年里有近 3000 个交易日
# 是必然空的，白等 25 分钟。
OPT_START = "20170101"

OPT_AUX_COLS = ["date", "iv_near", "iv_next", "dte_near", "dte_next",
                "skew", "curv", "pcr", "n_opt"]


# ------------------------------------------------------------------ 基础信息
def fetch_basic(pro=None, cache_path: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """全部商品期权合约的静态信息。

    返回列：ts_code / exchange / und（标的期货合约，如 M2401.DCE）/ call_put /
    strike / mat（到期日）。

    `opt_code` 是标的期货代码（大商所带 `OP` 前缀，如 `OPM2401.DCE`），
    去掉前缀就是能直接和 `fut_daily` 对齐的 ts_code。
    """
    pro = pro or fut.get_pro()
    if cache_path and not refresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            if len(df) > 1000:
                df["mat"] = pd.to_datetime(df["mat"], errors="coerce")
                return df
        except Exception:
            pass

    frames: list[pd.DataFrame] = []
    for ex in ("DCE", "CZCE", "SHFE", "INE", "GFEX"):
        offset = 0
        while True:
            try:
                d = pro.opt_basic(exchange=ex, offset=offset, limit=PAGE)
            except Exception:
                break
            if d is None or d.empty:
                break
            frames.append(d)
            if len(d) < PAGE:
                break
            offset += PAGE
            time.sleep(SLEEP)
    if not frames:
        raise RuntimeError("拉取期权合约清单失败：opt_basic 没有返回任何数据")

    b = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
    out = pd.DataFrame({
        "ts_code": b["ts_code"].astype(str),
        "exchange": b["exchange"].astype(str),
        "und": b["opt_code"].astype(str).str.replace(r"^OP", "", regex=True),
        "call_put": b["call_put"].astype(str),
        "strike": pd.to_numeric(b["exercise_price"], errors="coerce"),
        "mat": pd.to_datetime(b["maturity_date"], format="%Y%m%d", errors="coerce"),
    })
    out = out[(out["strike"] > 0) & out["mat"].notna() & out["und"].str.contains(r"\d", na=False)]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        write_atomic(cache_path, out)
    return out


def fetch_day(pro, day: str, retries: int = 3) -> pd.DataFrame:
    """某一天全市场的期权日线（分页，单页上限 15000）。"""
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        last = None
        df = None
        for attempt in range(retries):
            try:
                df = pro.opt_daily(trade_date=day, offset=offset, limit=OPT_PAGE)
                break
            except Exception as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        if df is None:
            raise RuntimeError(f"{day} 期权取数失败：{last}")
        if df.empty:
            break
        frames.append(df)
        if len(df) < OPT_PAGE:
            break
        offset += OPT_PAGE
        time.sleep(SLEEP)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
    out = pd.DataFrame({
        "ts_code": raw["ts_code"].astype(str),
        "close": pd.to_numeric(raw.get("close"), errors="coerce"),
        "settle": pd.to_numeric(raw.get("settle"), errors="coerce"),
        "volume": pd.to_numeric(raw.get("vol"), errors="coerce"),
        "oi": pd.to_numeric(raw.get("oi"), errors="coerce"),
    })
    return out


# ------------------------------------------------------------------ 定价与反解
def black76(F, K, T, sigma, r=RISK_FREE, is_call=True):
    """期货期权理论价（Black-76）。F/K/T/sigma 都可广播。"""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    s = np.maximum(sigma, 1e-6)
    t = np.maximum(T, 1e-8)
    sq = s * np.sqrt(t)
    d1 = (np.log(F / K) + 0.5 * sq ** 2) / sq
    d2 = d1 - sq
    disc = np.exp(-r * t)
    call = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


def implied_vol(price, F, K, T, is_call, lo=IV_LO, hi=IV_HI, iters=IV_ITERS):
    """二分法反解隐含波动率。价格低于内在价值时返回 lo（会被上层当噪声剔掉）。"""
    price = np.asarray(price, dtype=float)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = black76(F, K, T, mid, RISK_FREE, is_call) < price
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def fit_smile(money, iv, weight) -> tuple[float, float, float] | None:
    """二次拟合 IV 微笑，返回 (平值 IV, 偏度, 曲率)。

    `money` = ln(K/F)。用二次而不是三次：三次在边缘会外插出离谱的值，
    而我们要的平值和斜率都在 m=0 附近，二次足够且更稳。
    """
    if len(money) < MIN_POINTS or len(np.unique(np.round(money, 4))) < 3:
        return None
    try:
        coef = np.polyfit(np.asarray(money, float), np.asarray(iv, float), 2,
                          w=np.asarray(weight, float))
    except Exception:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    atm = float(coef[2])
    skew = float(coef[1])
    curv = float(2.0 * coef[0])
    if not (IV_LO < atm < IV_HI):
        return None
    return atm, skew, curv


# ------------------------------------------------------------------ 每日聚合
def day_metrics(day: str, opt_day: pd.DataFrame, basic: pd.DataFrame,
                fut_close: dict[str, float]) -> pd.DataFrame:
    """把一天的期权横截面压成「每品种一行」的波动率摘要。

    `fut_close`：当日真实合约的**结算价**（ts_code → settle），来自 `contracts.fetch_day`
    经 `contracts.settle_map` 转换。复用同一份数据，避免为了拿标的价格再请求
    一次 `fut_daily`。用结算价而不是收盘价的原因见 `settle_map` 的注释。
    """
    empty = pd.DataFrame(columns=["date", "symbol", "exchange"] + OPT_AUX_COLS[1:])
    if opt_day is None or opt_day.empty or basic.empty:
        return empty

    d = pd.to_datetime(day, format="%Y%m%d")
    m = opt_day.merge(basic, on="ts_code", how="inner")
    if m.empty:
        return empty
    m["F"] = m["und"].map(fut_close)
    m["px"] = m["settle"].where(m["settle"] > 0, m["close"])
    m["dte"] = (m["mat"] - d).dt.days
    m = m[(m["F"] > 0) & (m["px"] > MIN_PRICE)
          & (m["dte"] >= MIN_DTE) & (m["dte"] <= MAX_DTE)].copy()
    if m.empty:
        return empty

    m["money"] = np.log(m["strike"] / m["F"])
    m = m[m["money"].abs() <= MONEY_BAND]
    if m.empty:
        return empty

    m["iv"] = implied_vol(m["px"].to_numpy(), m["F"].to_numpy(), m["strike"].to_numpy(),
                          m["dte"].to_numpy() / 365.0, (m["call_put"] == "C").to_numpy())
    m = m[(m["iv"] > IV_LO * 1.2) & (m["iv"] < IV_HI * 0.95)]
    if m.empty:
        return empty

    m["symbol"] = m["und"].str.extract(r"^([A-Za-z]+)")[0]
    m["exchange"] = m["ts_code"].str.split(".").str[-1].map(SUFFIX_EX).fillna("")

    # PCR 用**全部**行权价（含实值）的持仓量：深度实值 put 的持仓恰恰是对冲盘
    # 最集中的地方，只用虚值会把它漏掉，比率就失去意义了。所以先算好再筛虚值。
    _oi = pd.to_numeric(m["oi"], errors="coerce").fillna(0.0)
    put_oi = _oi.where(m["call_put"] == "P", 0.0).groupby(m["symbol"]).sum()
    call_oi = _oi.where(m["call_put"] == "C", 0.0).groupby(m["symbol"]).sum()
    pcr_map = (put_oi / call_oi.where(call_oi > 0)).to_dict()

    # 只用虚值：K ≥ F 取 call，K < F 取 put。
    # 实值期权几乎没有时间价值，一个最小变动价位的误差就能让反解出的 σ 翻几倍
    # —— 实测白糖 2017-12-29 的 K=7400 实值 put 解出 31% IV，而平值只有 10%，
    # 这些点会把二次拟合拉成一条假的高曲率曲线（曲率 12 → 4.2）。
    # 同时这也避免了同一行权价的 call/put 各算一次、把每个点重复计一遍。
    otm = ((m["strike"] >= m["F"]) & (m["call_put"] == "C")) | (
        (m["strike"] < m["F"]) & (m["call_put"] == "P")
    )
    m = m[otm]
    if m.empty:
        return empty

    m["w"] = np.sqrt(pd.to_numeric(m["oi"], errors="coerce").fillna(0.0).clip(lower=1.0))

    rows: list[dict] = []
    for (sym, ex), g in m.groupby(["symbol", "exchange"], sort=False):
        exps = sorted(g["mat"].unique())
        rec: dict = {"date": d, "symbol": sym, "exchange": ex}
        fits: list[tuple[int, float, float, float]] = []
        for exp in exps[:2]:
            gg = g[g["mat"] == exp]
            fit = fit_smile(gg["money"].to_numpy(), gg["iv"].to_numpy(), gg["w"].to_numpy())
            if fit is None:
                continue
            fits.append((int(gg["dte"].iloc[0]), *fit))
        if not fits:
            continue
        # 近月：剩余期限最短的那条；次近月：第二条
        fits.sort(key=lambda x: x[0])
        dte, iv, skew, curv = fits[0]
        rec["dte_near"], rec["iv_near"] = dte, round(iv, 4)
        rec["skew"], rec["curv"] = round(skew, 4), round(curv, 4)
        if len(fits) > 1:
            rec["dte_next"], rec["iv_next"] = fits[1][0], round(fits[1][1], 4)
        p = pcr_map.get(sym)
        rec["pcr"] = round(float(p), 3) if p is not None and np.isfinite(p) else np.nan
        # 参与拟合的报价点数（虚值），用来判断这个品种当天够不够活跃
        rec["n_opt"] = int(len(g))
        rows.append(rec)
    if not rows:
        return empty
    out = pd.DataFrame(rows)
    for c in OPT_AUX_COLS[1:]:
        if c not in out.columns:
            out[c] = np.nan
    return out[["date", "symbol", "exchange"] + OPT_AUX_COLS[1:]]


def append_raw(raw_dir: str, day_df: pd.DataFrame, basic: pd.DataFrame) -> None:
    """把当天的期权原始明细按品种追加到 `data_options_raw/`（仅在 --keep-raw 时调用）。"""
    if day_df is None or day_df.empty:
        return
    os.makedirs(raw_dir, exist_ok=True)
    m = day_df.merge(basic[["ts_code", "exchange", "und"]], on="ts_code", how="inner")
    if m.empty:
        return
    m["symbol"] = m["und"].str.extract(r"^([A-Za-z]+)")[0]
    for (sym, ex), g in m.groupby(["symbol", "exchange"], sort=False):
        path = os.path.join(raw_dir, f"opt_{sym}_{ex}.csv")
        merge_write(path, g.drop(columns=["symbol", "exchange"]), ["date", "ts_code"])
