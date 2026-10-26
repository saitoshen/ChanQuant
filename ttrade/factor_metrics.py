"""因子评估：IC / ICIR / 分位收益 / IC 衰减 / 换手。

## 几个必须说清楚的口径问题

**1. 前向收益怎么算**

因子在 T 日收盘后算出，信号在 T+1 开盘成交。所以「这个因子能赚多少」对应的
收益区间是 `open[T+1] → close[T+h]`，不是 `close[T] → close[T+h]`。
后者包含了 T→T+1 这一段你根本吃不到的行情，会把 IC 虚抬一截（尤其对反转类因子）。

**2. IC 为什么用滚动序列**

单资产没有横截面，教科书里的「每期在所有股票上算一个 IC」在这里退化成一个数。
可行的做法是把 IC 做成时间序列：用 252 日滚动窗口算因子与前向收益的相关系数，
得到一条 IC 曲线。它的均值/标准差之比就是 ICIR。

**3. t 值必须做 Newey-West 调整**

滚动窗口的 IC 序列前后高度重叠（相邻两个窗口共用 251 天），自相关接近 1。
直接拿 `mean/std×sqrt(n)` 算 t 值，会把 t=2 的东西算成 t=20。这里用
Newey-West 异方差自相关一致估计，滞后阶数按经验取 `4*(n/100)^(2/9)`。
**没做这个调整的因子报告，t 值基本都是假的。**

**4. 分位收益的分组也要 PIT**

用全样本分位数分组，等于提前知道因子的完整分布。这里用滚动分位
（`rolling(window).rank(pct=True)`）把每天映射到 0~1，再切成 n 桶。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import ttrade.factors as F


# ---------------------------------------------------------------- 前向收益
def forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """T 日信号对应的可交易前向收益：T+1 开盘买入 → T+h 收盘卖出。

    `close.shift(-h) / open.shift(-1) - 1`。h=1 时是 T+1 当日的开→收，
    h=5 时是 T+1 开盘到 T+5 收盘。两端都对齐到「T 日能下单」的现实。
    """
    h = max(1, int(horizon))
    return df["close"].shift(-h) / df["open"].shift(-1) - 1.0


# ---------------------------------------------------------------- IC
def spearman(a: pd.Series, b: pd.Series, min_n: int = 10) -> float:
    """秩相关。pandas 的 rank 默认 average，与 scipy.stats.spearmanr 一致。

    `min_n` 可下调：分位收益只有 5 个桶，用默认的 10 会直接返回 NaN。
    """
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(d) < max(3, int(min_n)):
        return float("nan")
    ra, rb = d["a"].rank(), d["b"].rank()
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return float("nan")
    return float(ra.corr(rb))


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West 调整后的 t 值。重叠窗口的 IC 序列必须用它。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12:
        return float("nan")
    if lags is None:
        # 经验公式，n=2500 时约 8 阶，足够吃掉滚动窗口造成的自相关
        lags = int(round(4 * (n / 100.0) ** (2.0 / 9.0)))
        lags = max(1, min(lags, n // 4))
    mu = x.mean()
    e = x - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        gl = float(e[L:] @ e[:-L]) / n
        var += 2.0 * w * gl
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def ic_series(
    factor: pd.Series, fwd: pd.Series, window: int = 252
) -> pd.Series:
    """滚动 IC 序列。

    用滚动 Pearson（因子已标准化，与秩相关差别很小），因为滚动 Spearman
    要对每个窗口重新排序，5000 根 × 252 窗口会慢到没法在 GUI 里用。
    单点 IC（`ic_full`）仍然是严格的 Spearman。
    """
    w = int(window)
    mp = max(30, w // 2)
    d = pd.concat([factor.rename("f"), fwd.rename("r")], axis=1)
    return d["f"].rolling(w, min_periods=mp).corr(d["r"])


def ic_stats(
    factor: pd.Series, fwd: pd.Series, window: int = 252
) -> dict:
    """一次性给出因子的全部核心指标。"""
    d = pd.concat([factor.rename("f"), fwd.rename("r")], axis=1).dropna()
    out = {
        "n": int(len(d)),
        "ic": float("nan"),
        "ic_mean": float("nan"),
        "ic_std": float("nan"),
        "ir": float("nan"),
        "ic_pos": float("nan"),
        "t": float("nan"),
        "hit": float("nan"),
        "turnover": float("nan"),
    }
    if len(d) < 60:
        return out

    out["ic"] = spearman(d["f"], d["r"])

    ser = ic_series(factor, fwd, window).dropna()
    if len(ser) >= 12:
        mu, sd = float(ser.mean()), float(ser.std(ddof=1))
        out["ic_mean"] = mu
        out["ic_std"] = sd
        out["ir"] = mu / sd if sd > 0 else float("nan")
        out["ic_pos"] = float((ser > 0).mean())
        out["t"] = newey_west_t(ser.to_numpy())

    # 方向胜率：因子符号与前向收益符号一致的比例（用滚动中位数去中心，避免
    # 因子整体偏正时胜率被系统性拉高）
    med = d["f"].rolling(int(window), min_periods=30).median()
    sgn_f = np.sign(d["f"] - med)
    sgn_r = np.sign(d["r"])
    ok = (sgn_f != 0) & (sgn_r != 0)
    if ok.sum() > 30:
        out["hit"] = float((sgn_f[ok] == sgn_r[ok]).mean())

    # 换手：因子符号（相对滚动中位数）的日均变化率
    flip = (sgn_f.diff().abs() > 0).astype(float)
    out["turnover"] = float(flip.mean())

    return out


def ic_decay(
    df: pd.DataFrame, key: str, horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 60),
    window: int = 252, method: str = "z",
) -> pd.Series:
    """不同持有期下的单点 IC。衰减慢的因子更适合低频、成本敏感的策略。"""
    f = F.standardize(F.raw(df, key), window=window, method=method)
    vals = {}
    for h in horizons:
        vals[h] = spearman(f, forward_return(df, h))
    return pd.Series(vals, name="ic").rename_axis("horizon")


def quantile_returns(
    factor: pd.Series, fwd: pd.Series, n_buckets: int = 5, window: int = 252
) -> pd.Series:
    """PIT 分桶后的各组平均前向收益。

    分位用滚动窗口算，因此每一桶的成员随日期变化，不会泄漏未来分布。
    """
    n = int(n_buckets)
    w = int(window)
    mp = max(30, w // 4)
    pct = factor.rolling(w, min_periods=mp).rank(pct=True)
    # 0~1 → 1~n
    bucket = np.ceil(pct * n).clip(1, n)
    d = pd.DataFrame({"b": bucket, "r": fwd}).dropna()
    if d.empty:
        return pd.Series(dtype=float)
    g = d.groupby("b")["r"].mean()
    g.index = g.index.astype(int)
    return g.reindex(range(1, n + 1))


def monotonicity(qret: pd.Series) -> float:
    """分位收益的单调性：桶序号与平均收益的秩相关。±1 为完全单调。"""
    q = qret.dropna()
    if len(q) < 3:
        return float("nan")
    idx = pd.Series(np.arange(len(q), dtype=float), index=q.index)
    # 只有 5 个桶，不能套用默认的最小样本量
    return spearman(idx, q, min_n=3)


def turnover_of(signal: np.ndarray) -> float:
    """信号换手率：|Δsignal| 的均值 / 2（0~1，1 表示每天都全仓翻）。"""
    s = np.asarray(signal, dtype=float)
    if len(s) < 2:
        return float("nan")
    return float(np.abs(np.diff(s)).mean() / 2.0)


# ---------------------------------------------------------------- 批量报告
def factor_report(
    df: pd.DataFrame,
    keys: list[str],
    horizon: int = 5,
    window: int = 252,
    method: str = "z",
) -> pd.DataFrame:
    """对一批因子出完整评估表。GUI 的「因子分析」页直接用这个结果。"""
    fwd = forward_return(df, horizon)
    rows = []
    for k in keys:
        spec = F.FACTORS.get(k)
        if spec is None:
            continue
        try:
            f = F.standardize(F.raw(df, k), window=window, method=method)
        except Exception as exc:  # 单个因子出错不影响其它
            rows.append({"key": k, "label": spec.label, "category": spec.category,
                         "error": str(exc)[:60]})
            continue
        st = ic_stats(f, fwd, window)
        q = quantile_returns(f, fwd, 5, window)
        rows.append(
            {
                "key": k,
                "label": spec.label,
                "category": spec.category,
                "n": st["n"],
                "ic": st["ic"],
                "ic_mean": st["ic_mean"],
                "ic_std": st["ic_std"],
                "ir": st["ir"],
                "ic_pos": st["ic_pos"],
                "t": st["t"],
                "hit": st["hit"],
                "turnover": st["turnover"],
                "mono": monotonicity(q),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and "ic" in out:
        # 排序用 |IC|：正反方向都能用，方向由滚动 IC 决定
        out = out.reindex(out["ic"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def format_report(rep: pd.DataFrame) -> pd.DataFrame:
    """把原始数值转成适合直接显示的百分比/保留位数表格。"""
    if rep.empty:
        return rep
    out = rep.copy()
    for col in ("ic", "ic_mean", "ic_std", "ir", "ic_pos", "hit", "turnover", "mono"):
        if col in out:
            out[col] = out[col].map(
                lambda v: "" if pd.isna(v) else f"{v:.3f}"
            )
    if "t" in out:
        out["t"] = out["t"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return out
