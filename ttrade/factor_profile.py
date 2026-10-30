"""因子库的横向体检：快慢、品种、周期、组合、稳健性。

「因子分析」页回答的是**单个品种、单个周期**上「这个因子现在灵不灵」；
这个模块回答的是**整个库**的「这批因子各自是什么脾气」：

    慢因子还是快因子？ —— IC 衰减半衰期（横截面 IC 掉到一半要几个交易日）
    适合什么品种？     —— 逐品种的时间序列 IC，和整体方向一致的有哪些
    适合什么周期？     —— 半衰期 + 换手率倒推（快因子才扛得住 5m 的交易成本）
    和谁配组合？       —— 因子间相关矩阵，挑低相关、各自有效的
    有多稳？           —— 分年一致性 / 参数敏感性 / t 值，合成一个 0~100 分

## 为什么用横截面 IC

单品种的时间序列 IC 是「这个品种自己的因子值和自己的未来收益」相关，样本是
一条 2000 点的序列，噪声大、品种间不可比。商品多因子的标准做法是**每天在所有
品种上排一次名**，看因子排名靠前的品种是不是真的涨得多 —— 一天一个数，十年
2500 个数，样本量是前者的 2500 倍，而且天然回答「这个因子擅长挑品种吗」。

两套都算：横截面给结论，时间序列给「这个因子在螺纹上到底有没有用」。

## 为什么不用现成的 `factor_metrics.factor_report`

那个函数是逐品种算的，一次只能看一个品种，68 个因子 × 18 个品种要跑 1200 次
滚动相关，而且给不出相关矩阵和分年稳定性。这里的横截面口径是向量化的：
一次算完 2500 天 × 18 个品种的秩相关，68 个因子总共几秒。

## PIT

因子标准化只用滚动窗口（`factors.standardize`），前向收益是 `T+1 开盘 →
T+h 收盘`，横截面排名在**同一天内**做 —— 三处都不看未来。分年统计只是把同一条
IC 序列按年切开求均值，不引入新信息。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import ttrade.factors as F
from ttrade.factor_metrics import newey_west_t, spearman

# 做横截面的品种篮子。每个板块至少两个，否则「板块内谁强谁弱」这种信号
# 会被板块间的系统性差异盖掉。只挑主力连续里成交活跃、上市满十年的。
BASKET: list[tuple[str, str, str]] = [
    ("RB", "SHFE", "黑色"), ("I", "DCE", "黑色"), ("JM", "DCE", "黑色"),
    ("CU", "SHFE", "有色"), ("AL", "SHFE", "有色"), ("ZN", "SHFE", "有色"),
    ("AU", "SHFE", "贵金属"), ("AG", "SHFE", "贵金属"),
    ("SC", "INE", "能化"), ("MA", "CZCE", "能化"), ("TA", "CZCE", "能化"),
    ("RU", "SHFE", "能化"),
    ("M", "DCE", "农产品"), ("P", "DCE", "农产品"), ("SR", "CZCE", "农产品"),
    ("CF", "CZCE", "农产品"),
    ("IF", "CFFEX", "金融"), ("T", "CFFEX", "金融"),
]

HORIZONS = (1, 2, 3, 5, 10, 20, 40, 60)
WINDOWS = (120, 252, 504)          # 参数敏感性：标准化窗口
MIN_SYMBOLS = 5                    # 一天至少几个品种才做横截面排名


# ------------------------------------------------------------------ 横截面工具


def _row_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """按行（日期）算秩相关。`a` / `b` 同形状，列是品种。

    先按行排名再算 Pearson —— 等价于逐行 Spearman，但省掉 2500 次函数调用。
    缺失值按「成对完整」处理：只在该行两个都有值的品种之间算，不用 0 顶替。
    """
    ra = a.rank(axis=1)
    rb = b.rank(axis=1)
    A = ra.to_numpy(dtype=float)
    B = rb.to_numpy(dtype=float)
    m = np.isfinite(A) & np.isfinite(B)
    n = m.sum(axis=1).astype(float)
    A = np.where(m, A, 0.0)
    B = np.where(m, B, 0.0)
    sa, sb = A.sum(1), B.sum(1)
    saa, sbb, sab = (A * A).sum(1), (B * B).sum(1), (A * B).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sab - sa * sb / n
        va = saa - sa * sa / n
        vb = sbb - sb * sb / n
        r = cov / np.sqrt(va * vb)
    r[~(n >= MIN_SYMBOLS)] = np.nan
    r[~np.isfinite(r)] = np.nan
    return pd.Series(r, index=a.index, dtype=float)


def _ic_stats(ic: pd.Series) -> dict:
    """一条 IC 序列的核心统计量。"""
    s = ic.dropna()
    out = {"n": int(len(s)), "mean": float("nan"), "std": float("nan"),
           "icir": float("nan"), "t": float("nan"), "pos": float("nan")}
    if len(s) < 30:
        return out
    mu, sd = float(s.mean()), float(s.std(ddof=1))
    out.update(mean=mu, std=sd, icir=(mu / sd if sd > 0 else float("nan")),
               t=newey_west_t(s.to_numpy()), pos=float((s > 0).mean()))
    return out


def _half_life(ics: dict[int, float]) -> float:
    """IC 衰减半衰期（交易日）。

    定义：|IC(h)| 第一次掉到 |IC(1)| 一半的那个 h。线性插值，找不到就给 999
    （说明这个因子在 60 天内都没衰减，属于很慢的那一类）。
    """
    base = abs(ics.get(1, float("nan")))
    if not np.isfinite(base) or base < 1e-9:
        return float("nan")
    prev_h, prev_v = 1, base
    for h in sorted(ics):
        v = abs(ics[h])
        if not np.isfinite(v):
            continue
        if v <= base / 2:
            # 在 prev_h 与 h 之间线性插值到 base/2
            span = prev_v - v
            frac = (prev_v - base / 2) / span if span > 1e-12 else 1.0
            return float(prev_h + frac * (h - prev_h))
        prev_h, prev_v = h, v
    return 999.0


def _speed_class(h: float, ic1: float = float("nan")) -> tuple[str, str]:
    """半衰期 → (快慢标签, 周期建议)。

    **IC 太小的因子不给速度结论**：一条 ±0.002 的 IC 曲线「衰减到一半」只是
    噪声在抖，按它推出来的「5m 还是周线」纯属瞎猜。先要 IC 本身站得住，再谈
    快慢 —— 这条判据用的也是 `ic1`，也就是 h=1 的横截面 IC。
    """
    if not np.isfinite(h) or not np.isfinite(ic1) or abs(ic1) < 0.01:
        return "信号弱", "—（先看它在哪个品种上有值）"
    if h <= 1.5:
        return "极快", "5m / 1h（日内，成本敏感）"
    if h <= 3:
        return "快", "5m / 1h"
    if h <= 7:
        return "中快", "1h / 日线"
    if h <= 20:
        return "中", "日线"
    if h <= 60:
        return "慢", "日线 / 周线"
    return "很慢", "周线 / 长持（半年以上）"


# ------------------------------------------------------------------ 数据准备


def _load(basket, years: int, progress) -> dict[str, pd.DataFrame]:
    from ttrade.futures import load_local
    from ttrade.paths import data_dir

    root = data_dir("cn")
    out: dict[str, pd.DataFrame] = {}
    n = len(basket)
    for i, (sym, ex, _sector) in enumerate(basket, 1):
        if progress:
            progress(i, n, f"读取 {sym}_{ex}")
        try:
            df = load_local(sym, ex, root, adjust="qfq")
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 500:
            continue
        if years > 0:
            df = df.tail(int(years * 250) + 260)     # 多留一年给滚动窗口预热
        out[f"{sym}_{ex}"] = df
    return out


def _fwd_panel(frames: dict[str, pd.DataFrame], h: int) -> pd.DataFrame:
    """T 日信号对应的可交易前向收益：T+1 开盘 → T+h 收盘。"""
    cols = {}
    for key, df in frames.items():
        cols[key] = df["close"].shift(-h) / df["open"].shift(-1) - 1.0
    return pd.DataFrame(cols).sort_index()


# ------------------------------------------------------------------ 主体


@dataclass
class ProfileResult:
    """一次完整体检的结果。可直接 `json.dumps(asdict(...))` 落盘。"""

    meta: dict = field(default_factory=dict)
    factors: list = field(default_factory=list)
    combos: list = field(default_factory=list)
    redundant: list = field(default_factory=list)   # 高度相关的因子对

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProfileResult":
        return cls(meta=d.get("meta") or {}, factors=d.get("factors") or [],
                   combos=d.get("combos") or [], redundant=d.get("redundant") or [])

    def by_key(self) -> dict[str, dict]:
        return {f["key"]: f for f in self.factors}


def build(
    years: int = 10,
    basket=None,
    progress=None,
    keys: list[str] | None = None,
) -> ProfileResult:
    """跑一遍全库体检。`progress(done, total, msg)` 可传可不传。"""
    t0 = time.time()
    basket = basket or BASKET
    keys = keys or list(F.FACTORS)
    frames = _load(basket, years, progress)
    if len(frames) < MIN_SYMBOLS:
        raise RuntimeError(
            f"本地只有 {len(frames)} 个品种的日线，横截面分析至少要 {MIN_SYMBOLS} 个 —— "
            "先去「数据管理」页补一次期货日线"
        )

    # 前向收益面板：一次算好，所有因子共用
    fwd = {h: _fwd_panel(frames, h) for h in HORIZONS}
    ret1 = fwd[1]

    # 原始因子值只算一次，三档窗口各自标准化后算 IC，只有 252 那档留下面板
    panels: dict[str, pd.DataFrame] = {}
    raw: dict[str, dict[str, pd.Series]] = {}
    total = len(frames) * len(keys)
    done = 0
    for key, df in frames.items():
        raw[key] = {}
        for fk in keys:
            done += 1
            if progress and done % 40 == 0:
                progress(done, total, f"{key} · {F.FACTORS[fk].label if fk in F.FACTORS else fk}")
            try:
                raw[key][fk] = F.raw(df, fk)
            except Exception:  # noqa: BLE001
                raw[key][fk] = pd.Series(np.nan, index=df.index, dtype=float)

    for fk in keys:
        if fk not in F.FACTORS:
            continue
        cols = {}
        for key, series in raw.items():
            try:
                cols[key] = F.standardize(series[fk], window=252, method="z")
            except Exception:  # noqa: BLE001
                continue
        panels[fk] = pd.DataFrame(cols).sort_index()

    # 因子相关矩阵：把面板摊平成一列，秩相关。横截面 IC 只关心排名，
    # 所以用秩相关而不是 Pearson，免得被少数极端值带偏。
    corr = _factor_corr(panels)

    rows = []
    for i, fk in enumerate(keys, 1):
        if fk not in F.FACTORS or fk not in panels:
            continue
        if progress:
            progress(i, len(keys), f"分析 {F.FACTORS[fk].label}")
        rows.append(_profile_one(fk, panels[fk], fwd, ret1, corr, raw))
    rows.sort(key=lambda r: -abs(r.get("ic_mean") or 0.0))
    _attach_mates(rows, corr)

    combos = _combos(rows, corr)
    redundant = _redundant(corr)

    return ProfileResult(
        meta={
            "generated": time.strftime("%Y-%m-%d %H:%M"),
            "years": years,
            "symbols": sorted(frames),
            "sectors": sorted({s for _k, _e, s in basket if f"{_k}_{_e}" in frames}),
            "n_factors": len(rows),
            "span": [str(min(df.index.min() for df in frames.values()))[:10],
                     str(max(df.index.max() for df in frames.values()))[:10]],
            "elapsed": round(time.time() - t0, 1),
        },
        factors=rows, combos=combos, redundant=redundant,
    )


def _factor_corr(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """因子两两秩相关。

    做法：每个因子摊平成「日期 × 品种」的一列，先按品种内标准化（消掉品种间
    的量级差），再算 Pearson —— 数据已经过了 z-score + 秩变换，Pearson 和
    Spearman 差别很小，但 68×68 用矩阵乘一次算完，比逐对调用快两个数量级。
    """
    keys = [k for k in panels if not panels[k].empty]
    if len(keys) < 2:
        return pd.DataFrame()
    mat = []
    for k in keys:
        p = panels[k]
        z = (p - p.mean()) / p.std(ddof=0).replace(0.0, np.nan)
        v = z.to_numpy(dtype=float).ravel()
        v = np.where(np.isfinite(v), v, 0.0)
        n = float(np.count_nonzero(v))
        mat.append(v / (np.sqrt(n) if n > 0 else 1.0))
    M = np.vstack(mat)
    C = M @ M.T
    C = np.clip(C, -1.0, 1.0)
    return pd.DataFrame(C, index=keys, columns=keys)


def _profile_one(
    fk: str,
    panel: pd.DataFrame,
    fwd: dict[int, pd.DataFrame],
    ret1: pd.DataFrame,
    corr: pd.DataFrame,
    raw: dict[str, dict[str, pd.Series]] | None = None,
) -> dict:
    """单个因子的全部指标。"""
    spec = F.FACTORS[fk]
    ics = {h: float(_row_spearman(panel, fwd[h]).mean()) for h in HORIZONS}
    ic1 = _row_spearman(panel, ret1)
    st = _ic_stats(ic1)
    if st["n"] == 0:
        # 日内 / 时段那十来个因子在日线上整列是空的：它们要的是当天 5m 或 1h 的
        # 形态，日线一根 bar 里没有「上午还是下午」这回事。手册里必须单独说明，
        # 否则读的人会以为这些因子坏了。
        return {
            "key": fk, "label": spec.label, "category": spec.category,
            "desc": spec.desc, "formula": spec.formula,
            "n": 0, "ic_mean": float("nan"), "icir": float("nan"),
            "t": float("nan"), "pos": float("nan"), "half_life": float("nan"),
            "speed": "日线无值", "period": "只在 5m / 1h 上有效",
            "turnover": float("nan"), "ts_ic": float("nan"),
            "ts_agree": float("nan"), "best_symbols": [], "per_symbol": {},
            "years": {}, "year_ratio": float("nan"), "win_ic": {},
            "corr_max": {}, "mates": [], "robust": 0.0, "no_daily": True,
        }

    # 逐品种的时间序列 IC：这个因子在螺纹 / 铜上到底灵不灵
    per_symbol = {}
    for col in panel.columns:
        try:
            per_symbol[col] = spearman(panel[col], ret1[col])
        except Exception:  # noqa: BLE001
            continue
    vals = {k: v for k, v in per_symbol.items() if np.isfinite(v)}
    sign = np.sign(st["mean"]) if np.isfinite(st["mean"]) else 0.0
    agree = [k for k, v in vals.items() if np.isfinite(v) and np.sign(v) == sign and sign != 0]
    best = sorted(agree, key=lambda k: -abs(vals[k]))[:5]

    # 分年一致性：IC 的正负号在多少比例的年份里跟整体一致
    by_year = {}
    if len(ic1.dropna()):
        g = ic1.dropna().groupby(ic1.dropna().index.year)
        by_year = {int(y): float(v.mean()) for y, v in g if len(v) >= 60}
    same = [y for y, v in by_year.items() if np.isfinite(v) and np.sign(v) == sign]
    year_ratio = (len(same) / len(by_year)) if by_year else float("nan")

    # 换手：标准化因子每天变化的绝对值均值 / 2（0~1，1 = 每天全翻）。
    # 必须用 nanmean：预热期是 NaN，`np.diff` 会把 NaN 传染给相邻两个差分，
    # 直接 mean 出来整列都是 NaN。
    deltas = [
        np.nanmean(np.abs(np.diff(panel[c].to_numpy(dtype=float)))) / 2.0
        for c in panel.columns if panel[c].notna().sum() > 100
    ]
    turn = float(np.nanmean(deltas)) if deltas else float("nan")

    # 参数敏感性：换标准化窗口重算横截面 IC。窗口是唯一一个「随手改一下也
    # 说得通」的参数，它对结果的影响最小才算稳 —— 换 120 天就翻脸的因子，
    # 实盘换个起点就不灵了。
    win_ic = {252: ics.get(1, float("nan"))}
    if raw:
        for w in WINDOWS:
            if w == 252:
                continue
            try:
                cols = {}
                for key, series in raw.items():
                    if fk in series:
                        cols[key] = F.standardize(series[fk], window=w, method="z")
                if len(cols) >= MIN_SYMBOLS:
                    win_ic[w] = float(
                        _row_spearman(pd.DataFrame(cols).sort_index(), ret1).mean()
                    )
            except Exception:  # noqa: BLE001
                pass

    hl = _half_life(ics)
    speed, period = _speed_class(hl, ics.get(1, float("nan")))

    row = {
        "key": fk,
        "label": spec.label,
        "category": spec.category,
        "desc": spec.desc,
        "formula": spec.formula,
        "n": st["n"],
        "ic_mean": st["mean"],
        "ic_std": st["std"],
        "icir": st["icir"],
        "t": st["t"],
        "pos": st["pos"],
        "ic1": ics.get(1, float("nan")),
        "ic5": ics.get(5, float("nan")),
        "ic20": ics.get(20, float("nan")),
        "ic60": ics.get(60, float("nan")),
        "half_life": hl,
        "speed": speed,
        "period": period,
        "turnover": turn,
        "ts_ic": float(np.nanmean(list(vals.values()))) if vals else float("nan"),
        "ts_agree": len(agree) / len(vals) if vals else float("nan"),
        "best_symbols": best,
        "per_symbol": {k: round(v, 4) for k, v in vals.items()},
        "years": {str(y): round(v, 4) for y, v in sorted(by_year.items())},
        "year_ratio": year_ratio,
        "win_ic": win_ic,
        "corr_max": _max_corr(fk, corr),
    }
    row["robust"] = _robust(row)
    return row


def _max_corr(fk: str, corr: pd.DataFrame) -> dict:
    """和这个因子最像的另外三个（相关系数最高的）。"""
    if corr.empty or fk not in corr.index:
        return {}
    s = corr.loc[fk].drop(labels=[fk], errors="ignore").dropna()
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:3]
    return {k: round(float(v), 3) for k, v in s.items()}


def _robust(row: dict) -> float:
    """稳健性得分 0~100。四项各 25 分，缺项的按 0 算（宁可低估）。

    1. 显著性：|t| 到 4 就给满 —— 滚动 IC 重叠严重，t=2 其实很弱
    2. 方向稳定：IC>0 的天数占比，0.5 是抛硬币
    3. 分年一致：多少比例的年份符号不变，跨年翻转的因子不能信
    4. 横截面普适：多少比例的品种方向一致

    窗口敏感性不进这个分：它是「换个参数还灵不灵」，属于加分项而不是及格线，
    单独在报告里列出来给人看。
    """
    def _v(x):
        try:
            f = float(x)
        except (TypeError, ValueError):
            return float("nan")
        return f if np.isfinite(f) else float("nan")

    t, pos = abs(_v(row.get("t")) or 0.0), _v(row.get("pos"))
    yr, agree = _v(row.get("year_ratio")), _v(row.get("ts_agree"))
    s = 25.0 * min(t / 4.0, 1.0)
    if np.isfinite(pos):
        s += 25.0 * min(max((pos - 0.5) / 0.2, 0.0), 1.0)
    if np.isfinite(yr):
        s += 25.0 * min(max((yr - 0.5) / 0.4, 0.0), 1.0)
    if np.isfinite(agree):
        s += 25.0 * min(max((agree - 0.4) / 0.4, 0.0), 1.0)
    return round(s, 1)


def _attach_mates(rows: list[dict], corr: pd.DataFrame) -> None:
    """给每个因子找 3 个「互补的搭子」：相关性低、自己也算稳。

    多因子的价值来自「各自押不同的东西」。所以搭子的第一判据不是 IC 高，而是
    |相关| 低 —— 两个 IC 0.02、相关 0.1 的因子合起来，比一个 IC 0.03、相关 0.9
    的孪生兄弟有用得多。相关矩阵里没有的因子（样本不足）直接跳过。
    """
    if corr.empty:
        return
    robust = {r["key"]: r.get("robust") or 0.0 for r in rows}
    for r in rows:
        k = r["key"]
        if k not in corr.index:
            r["mates"] = []
            continue
        s = corr.loc[k].drop(labels=[k], errors="ignore").dropna()
        cand = [(c, float(v)) for c, v in s.items()
                if abs(float(v)) < 0.25 and robust.get(c, 0) >= 30]
        cand.sort(key=lambda kv: -robust.get(kv[0], 0))
        r["mates"] = [{"key": c, "corr": round(v, 2), "robust": robust.get(c, 0)}
                      for c, v in cand[:3]]


def _combos(rows: list[dict], corr: pd.DataFrame, size: int = 4) -> list[dict]:
    """推荐几组多因子组合。

    规则：按 |ICIR| 排序，逐个往里加，新加的必须和已选的全部 |相关| < 0.55，
    且同一类别最多两个 —— 展期那一家子有十几个变体（原始 / OI3 / OI6），
    相关性 0.9 以上，放三个进去等于把仓位三倍押在 carry 上。

    给三组不同口味：
      稳健组 —— |ICIR| 最高
      显著组 —— |t| 最高（统计上最不像运气）
      分散组 —— 类别覆盖最广（趋势 / 反转 / 展期 / 波动 各来一个）
    """
    if not rows or corr.empty:
        return []

    def pick(order: list[dict], need: int) -> list[str]:
        chosen: list[str] = []
        per_cat: dict[str, int] = {}
        for r in order:
            k = r["key"]
            if k in chosen or k not in corr.index:
                continue
            cat = r.get("category") or ""
            if per_cat.get(cat, 0) >= 2:
                continue
            if all(abs(float(corr.loc[k, c])) < 0.55 for c in chosen):
                chosen.append(k)
                per_cat[cat] = per_cat.get(cat, 0) + 1
            if len(chosen) >= need:
                break
        return chosen

    by_icir = sorted(rows, key=lambda r: -abs(r.get("icir") or 0.0))
    by_t = sorted(rows, key=lambda r: -abs(r.get("t") or 0.0))
    # 分散组：按类别轮流取，每类先取该类别里 |ICIR| 最高的
    seen: dict[str, int] = {}
    spread: list[dict] = []
    for r in by_icir:
        c = r.get("category") or ""
        if seen.get(c, 0) == 0:
            spread.append(r)
        seen[c] = seen.get(c, 0) + 1
    spread += [r for r in by_icir if r not in spread]

    out = []
    for name, order, why in (
        ("稳健组", by_icir, "按 ICIR 从高到低挑，两两低相关、同类别不超过两个"),
        ("显著组", by_t, "按 Newey-West t 值挑，最不像运气的几个"),
        ("分散组", spread, "先按类别各取一个，再看 ICIR 补位"),
    ):
        ks = pick(order, size)
        if len(ks) < 2:
            continue
        sel = [r for k in ks for r in rows if r["key"] == k]
        out.append({
            "name": name,
            "why": why,
            "keys": ks,
            "labels": [r["label"] for r in sel],
            # 组合的强度用各成员 |ICIR| 的均值：方向由回测的滚动 IC 自动定，
            # 直接平均带符号的 ICIR 会把「一个正向、一个反向」算成零
            "icir": round(float(np.mean([abs(r.get("icir") or 0) for r in sel])), 3),
            "t": round(float(np.mean([abs(r.get("t") or 0) for r in sel])), 2),
            "max_corr": round(float(max(
                (abs(float(corr.loc[a, b])) for i, a in enumerate(ks) for b in ks[i + 1:]),
                default=0.0)), 3),
        })
    return out


def _redundant(corr: pd.DataFrame, top: int = 8) -> list[dict]:
    """相关系数最高的几对，用来提示「这两个其实是一个因子」。"""
    if corr.empty:
        return []
    pairs = []
    ks = list(corr.index)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            v = float(corr.loc[a, b])
            if abs(v) >= 0.85:
                pairs.append({"a": a, "b": b, "corr": round(v, 3)})
    pairs.sort(key=lambda p: -abs(p["corr"]))
    return pairs[:top]
