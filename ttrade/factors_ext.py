"""补充因子：从 future_quant 那套 CTA 短周期 / RIA 选品种里移植过来的。

## 三块内容，来源不同、口径不同

1. **CTA 短周期**（`cta_short/factor_extra.py`）—— 14 个日线量价 + 6 个期限结构
   + 9 个基本面。它的口径是「日内 10:40 决策」：白天样本只能用 T-1 的日线，
   夜盘样本可以用当天。我们这边因子是**按 bar 算**的，bar 本身已经是「收盘后
   可得」的粒度，所以不需要那套 `_search_prev`，直接用当根 bar 及以前。

2. **RIA 选品种**（`multi_symbol_bd/symbol_selector.py`）—— 7 个横截面选品种
   因子（波动/趋势/活跃度/量仓比/成交额/持仓变化/成交量变化）。它们是给「今天
   该做哪几个品种」打分用的，单品种时序上也有意义（比如成交额水平）。

3. **板块指数**（`cta_short/factor_extra.py` 的 `INDEX_MOM_FEATS`）—— 成交额
   加权的板块/全市场指数动量与波动。这类因子在同一板块内所有品种上取值相同，
   截面 IC 上会被当成「板块哑变量」，价值在于**时序择时**和**给单品种做条件**。

## 去重原则

原项目里有 4 个（`dmom20` / `dvol20` / `datr14` / `drsi14`）和库里已有的
`mom_20` / `vol_20` / `atr_pct` / `rsi_14` 是同一个东西，`dmom5` 是 `rev_5`
取反 —— 这几个一律不重复注册。`ts_basis_sub` 与 `roll_main_oi` 是同一件事
（主力-次主力价差），也不重复。剩下的按公式实现。

## PIT

- 价格类：全部只用 `shift(正数)` / `rolling`，没有一处 `shift(-n)`。
- 基本面（EDB）：仓单/库存/利润/现货都是**当天收盘后**才发布，所以一律取
  `T-1` 及以前的值（`lag_days=1`）。这一条比价格因子更要紧：EDB 的日频数据
  如果按当天取，10:40 的样本就提前知道了当天下午才更新的仓单。
- 板块指数：权重用 `amount.shift(1)`，当天的成交额不参与当天权重。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 通用工具


def _nan(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index, dtype=float)


def _safe_div(a, b):
    return a / b.replace(0.0, np.nan)


def _symbol(df: pd.DataFrame) -> str:
    """品种代码（`RB`）。加载器挂在 `df.attrs["symbol"]` 上。"""
    sym = ""
    try:
        sym = str(df.attrs.get("symbol") or "")
    except Exception:  # noqa: BLE001
        sym = ""
    if not sym and "symbol" in df.columns and len(df):
        sym = str(df["symbol"].iloc[0])
    return sym.upper()


def _day(df: pd.DataFrame) -> pd.Series:
    """每根 bar 属于哪个交易日。夜盘 21:00 归下一个交易日。

    和 `factors._trade_day` 是同一个口径，各写一份是因为本模块不 import 主模块
    （主模块末尾才 import 本模块，反向 import 就是循环）。
    """
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


def _day_diff(df: pd.DataFrame, s: pd.Series, n: int) -> pd.Series:
    """按**交易日**取 n 天变化（分钟线上不能直接 diff(n)）。"""
    idx = s.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return s.diff(n)
    day = _day(df)
    if len(day) == len(day.unique()):
        return s.diff(n)
    daily = s.groupby(day).last()
    return pd.Series(np.asarray(day.map(daily.diff(n)), dtype=float), index=idx)


def _asof(series: pd.Series, index, lag_days: int = 0) -> pd.Series:
    """把「一天一个值」的外部序列 as-of 铺到 bar 上。

    `lag_days` 是**日历日**滞后：EDB 的仓单、库存、利润都是当天收盘后才更新，
    白天用当天的值就是未来函数，所以调用方一律传 1。

    实现用 `merge_asof`：外部序列的日期可能落在周末/节假日（仓单是周频），
    直接 `reindex` 会整段 NaN，必须「取不晚于目标日的最后一个值」。
    """
    if series is None or len(series) == 0:
        return pd.Series(np.nan, index=index, dtype=float)
    left = pd.DataFrame({"t": pd.DatetimeIndex(index).normalize()})
    if lag_days:
        left["t"] = left["t"] - pd.Timedelta(days=int(lag_days))
    right = pd.DataFrame({
        "t": pd.DatetimeIndex(series.index).normalize(),
        "v": pd.to_numeric(series.to_numpy(), errors="coerce"),
    }).dropna(subset=["t"]).sort_values("t")
    # 同一天有多个值（周频指标偶发重复）时保留最后一个
    right = right.groupby("t", as_index=False).last()
    out = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward")
    out = out.reindex(left.index)
    return pd.Series(out["v"].to_numpy(dtype=float), index=index)


# ---------------------------------------------------------------- CTA 日线量价
def _ret(c: pd.Series, n: int) -> pd.Series:
    return c / c.shift(n) - 1.0


def f_mom_1(df):
    """1 日涨跌。日内噪声最大的一档，短线反转的主要原料。"""
    return _ret(df["close"], 1)


def f_mom_3(df):
    """3 日涨跌。"""
    return _ret(df["close"], 3)


def f_mom_10(df):
    """10 日（两周）涨跌。比 20 日更快，比 5 日噪声小。"""
    return _ret(df["close"], 10)


def f_vol_10(df):
    """10 日年化已实现波动率。短窗口，对波动突变反应快。"""
    r = df["close"].pct_change()
    return r.rolling(10, min_periods=10).std(ddof=1) * np.sqrt(252.0)


def f_pos_20(df):
    """20 日区间位置：(C − LL20) / (HH20 − LL20)。

    `breakout_pos` 是 55 日、`stoch_14` 是 14 日，20 日是中间档：短到能跟上
    一个月的波段，长到不会被一两根 K 线带偏。
    """
    ll = df["low"].rolling(20, min_periods=20).min()
    hh = df["high"].rolling(20, min_periods=20).max()
    return _safe_div(df["close"] - ll, hh - ll)


def f_vol_ratio_5_20(df):
    """5 日均量 / 20 日均量。

    和 `vol_surge`（当日量 / 20 日均量）互补：那条看「今天有没有异动」，
    这条看「最近一周整体是不是在放量」，对单日噪声更稳。
    """
    v = df["volume"]
    return _safe_div(v.rolling(5, min_periods=5).mean(),
                     v.rolling(20, min_periods=20).mean())


def f_oi_change_1(df):
    """持仓量 1 日变化率。增仓突破比减仓突破可信，这是最直接的一档。"""
    oi = _oi(df)
    return oi.where(oi > 0).pct_change(1)


def f_oi_change_5(df):
    """持仓量 5 日变化率。一周的资金进出。"""
    oi = _oi(df)
    return oi.where(oi > 0).pct_change(5)


def f_range_1(df):
    """单根 K 线的振幅 (H−L)/C。`hl_range` 是 20 日均值，这条是当根。"""
    return _safe_div(df["high"] - df["low"], df["close"])


def _oi(df: pd.DataFrame) -> pd.Series:
    """持仓量。两条数据链路的列名不一样（`open_interest` / `open_oi`）。"""
    for name in ("open_interest", "open_oi"):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").astype(float)
    return _nan(df)


# ---------------------------------------------------------------- 期限结构（CTA 口径）
def _leg(df: pd.DataFrame, i: int, field: str) -> pd.Series:
    """取第 i 条腿的某个字段，优先 o6_ 口径，退回 o3_，再退回 c。"""
    for pre in ("o6_", "o3_", "c"):
        name = f"{pre}{i}_{field}" if pre != "c" else f"c{i}_{field}"
        if name in df.columns:
            s = pd.to_numeric(df[name], errors="coerce")
            if s.notna().any():
                return s.astype(float)
    return _nan(df)


def f_term_slope(df):
    """期限结构斜率：(P2 − P3) / P2，即次近月对第三近月的价差率。

    `roll_curv` 用的是「年化斜率之差」（对数、按到期日归一），这条是**原始价差
    率**，不做年化 —— 在换月密集的品种上它更稳，因为不除以一个很小的 dte 间隔。
    """
    p2, p3 = _leg(df, 2, "close"), _leg(df, 3, "close")
    return _safe_div(p2 - p3, p2)


def f_main_oi_share(df):
    """主力持仓占比：近月持仓 / 三条腿持仓之和。

    接近 1 说明资金全挤在最近的那个活跃月上（交割博弈、现货紧张）；
    偏低说明主力已经移到远月，近月价格的参考意义下降。
    和 `near_oi_share` 的区别是分母用了三条腿，不只看前两条。
    """
    a = _leg(df, 1, "oi")
    b = _leg(df, 2, "oi")
    c = _leg(df, 3, "oi")
    tot = a.fillna(0) + b.fillna(0) + c.fillna(0)
    return _safe_div(a, tot.where(tot > 0))


def f_term_near_far(df):
    """近远月价差率：(P1 − P3) / P1。整条曲线两端的距离。

    `roll_yield` 只看 P1/P2 这一段，近端一抖动整条就翻；用两端能滤掉近月的
    单日噪声，代价是对「曲线中段凸起」不敏感。
    """
    p1, p3 = _leg(df, 1, "close"), _leg(df, 3, "close")
    return _safe_div(p1 - p3, p1)


# ---------------------------------------------------------------- RIA 选品种
def f_ma_cross_5_20(df):
    """(MA5 − MA20) / MA20。RIA 的 `trend` 因子。

    和 `ma_gap_20`（收盘对均线）不同：这里是**两条均线之间**的距离，
    已经被两次平滑过，方向切换比价格慢一拍，但假信号少得多。
    """
    c = df["close"]
    ma5 = c.rolling(5, min_periods=5).mean()
    ma20 = c.rolling(20, min_periods=20).mean()
    return _safe_div(ma5 - ma20, ma20)


def f_abs_ret_20(df):
    """20 日平均绝对收益。RIA 的 `activity`。

    不是波动率（没有平方、不年化），就是「每天平均动多少」。对跳空和厚尾
    不如标准差敏感，反而更像「这个品种好不好做」的直接度量。
    """
    return df["close"].pct_change().abs().rolling(20, min_periods=20).mean()


def f_turnover_20(df):
    """20 日均量 / 20 日均持仓。RIA 的 `volume_oi`，即换手率。

    分子分母**必须同窗口**。原项目早期版本分子是 20 日均量、分母是当日持仓，
    而单日成交量 ±50% 是常态，因子被单日噪声主导 —— 那个坑值得记一笔。
    """
    v = df["volume"].rolling(20, min_periods=20).mean()
    oi = _oi(df).rolling(20, min_periods=20).mean()
    return _safe_div(v, oi.where(oi > 0))


def f_log_amount_20(df):
    """20 日成交额合计取对数。RIA 的 `amount`，衡量「这个品种有多大」。

    取对数是因为成交额的分布跨三个数量级（股指几万亿、苹果几亿），
    不取对数的话横截面排序完全被头部品种主导。
    """
    if "amount" in df.columns:
        s = pd.to_numeric(df["amount"], errors="coerce").rolling(
            20, min_periods=20).sum()
    else:
        s = (pd.to_numeric(df["volume"], errors="coerce") * df["close"]).rolling(
            20, min_periods=20).sum()
    return np.log1p(s.where(s > 0))


def f_oi_change_robust(df):
    """换月稳健的 20 日持仓变化率。

    888 主力连续在换月日持仓会从旧主力跳到新主力（实测 ag888 单日 +1216%），
    直接取「20 日首末点差」只要有一端落在跳变日就整段失真。这里改成逐日
    对数变化率累加，单日截断到 ±15% —— 挡掉换月跳变，保留真实的建仓/减仓。

    分钟线上**用前一日的日线值**，不用当日。当日持仓要等 15:00 收盘才成立，
    按「当日最后一根」聚合等于把收盘持仓摊给早上每一根 bar —— 截断到 09:55
    重算，前 12 个值就变了。日线尺度的因子本来也不需要日内更新。
    """
    oi = _oi(df)
    day = _day(df)
    intraday = len(day) != len(day.unique())
    if intraday:
        oi = oi.groupby(day).last().shift(1)
    oi = oi.where(oi > 0)
    pct = oi.pct_change().replace([np.inf, -np.inf], np.nan)
    lr = np.log1p(pct.clip(-0.5, 1.0)).clip(-0.15, 0.15)
    out = np.expm1(lr.rolling(20, min_periods=15).sum())
    if intraday:
        out = pd.Series(np.asarray(day.map(out), dtype=float), index=df.index)
    return out


def f_vol_change_20(df):
    """成交量变化：近 10 日均量 / 前 10 日均量 − 1。RIA 的 `volume_change`。"""
    v = df["volume"]
    recent = v.rolling(10, min_periods=10).mean()
    earlier = v.shift(10).rolling(10, min_periods=10).mean()
    return _safe_div(recent, earlier) - 1.0


# ---------------------------------------------------------------- 基本面（EDB）
_edb_cache: dict[str, dict[str, pd.Series]] = {}
# 类别 → 实际挑中的那条序列的名字。研报要写清口径：同样是 `profit` 类别，
# 螺纹库里只有「成本(含税)」，写「加工利润」就是错的。
_edb_name_cache: dict[str, dict[str, str]] = {}
_edb_meta_cache: list[pd.DataFrame] = []
# 汇总口径的标志词。按这个顺序找，命中越靠前越像「全国平均」。
_AGG_WORDS = ("平均", "全国", "合计", "总计", "综合")


def _edb_meta() -> pd.DataFrame:
    """`edb.meta()` 每次都会开一次 duckdb 连接。95 个品种各来一次就是 95 次
    连接 —— 一次因子计算能拖到好几秒。这里只取一次。"""
    if _edb_meta_cache:
        return _edb_meta_cache[0]
    try:
        from ttrade import edb as E

        m = E.meta()
    except Exception:  # noqa: BLE001
        m = pd.DataFrame()
    _edb_meta_cache.append(m)
    return m


def _edb_pick(variety: str) -> dict[str, pd.Series]:
    """某个品种的 EDB 序列，按类别各挑一条。

    同一类别下往往有十几条（螺纹的成本按省份分了 13 条、豆粕的压榨利润按港口
    分了 16 条）。挑法是：先找**汇总口径**（名字里带「平均 / 全国 / 合计 /
    总计 / 综合」的），在观测数不少于最长那条一半的候选里挑观测最多的；一条
    汇总都没有，才退回「观测数最多」。

    为什么不直接挑观测最多的：螺纹的 `profit` 里第一条是「安徽」。单省成本当
    全国成本用，做因子还能凑合（变化方向大体同步），写进研报就是错的 —— 读者
    会以为 3445 是螺纹的平均成本。而这些序列的起始日和频率基本一致（都是同一
    张表拆出来的），挑汇总口径并不会少掉历史。
    """
    key = variety.lower()
    if key in _edb_cache:
        return _edb_cache[key]
    out: dict[str, pd.Series] = {}
    names: dict[str, str] = {}
    try:
        from ttrade import edb as E

        m = _edb_meta()
        if m.empty:
            _edb_cache[key] = out
            return out
        sub = m[m["variety"].astype(str).str.lower() == key]
        if sub.empty:
            _edb_cache[key] = out
            return out
        ids = [int(x) for x in sub["edb_id"]]
        wide = E.load(ids=ids, wide=True)
        if wide.empty:
            _edb_cache[key] = out
            return out
        for cat in sub["category"].unique():
            cand: list[tuple[int, int, str]] = []
            for eid, nm in zip(sub[sub["category"] == cat]["edb_id"],
                               sub[sub["category"] == cat]["name"]):
                eid = int(eid)
                if eid not in wide.columns:
                    continue
                n = int(wide[eid].notna().sum())
                if n > 0:
                    cand.append((eid, n, str(nm)))
            if not cand:
                continue
            top_n = max(n for _, n, _ in cand)
            agg = [c for c in cand if any(w in c[2] for w in _AGG_WORDS)
                   and c[1] >= top_n * 0.5]
            best = max(agg, key=lambda c: c[1]) if agg else max(cand, key=lambda c: c[1])
            out[str(cat)] = wide[best[0]].dropna()
            names[str(cat)] = best[2]
    except Exception:  # noqa: BLE001
        pass
    _edb_cache[key] = out
    _edb_name_cache[key] = names
    return out


def _edb_name(variety: str, cat: str, fallback: str = "") -> str:
    """某个品种在某个类别下**实际挑中**的那条序列的全名。"""
    key = str(variety).lower()
    _edb_pick(key)
    return _edb_name_cache.get(key, {}).get(cat, fallback)


def clear_cache() -> None:
    """数据更新后调用。EDB 补了新数据，旧缓存会一直返回补之前那一段。"""
    _edb_cache.clear()
    _edb_name_cache.clear()
    _edb_meta_cache.clear()
    _sector_cache.clear()


def _fund(df: pd.DataFrame, cat: str, lag: int = 1) -> pd.Series:
    sym = _symbol(df)
    if not sym:
        return _nan(df)
    s = _edb_pick(sym).get(cat)
    if s is None or len(s) == 0:
        return _nan(df)
    return _asof(s, df.index, lag_days=lag)


def f_war_chg_1(df):
    """仓单数量 1 日变化率。仓单增加 = 现货往交易所流 = 供应宽松。"""
    s = _fund(df, "warehouse")
    return s.pct_change(1)


def f_war_chg_5(df):
    """仓单数量 5 日变化率。一周的注册/注销节奏。"""
    s = _fund(df, "warehouse")
    return s.pct_change(5)


def f_war_oi(df):
    """仓单 / 期货持仓量。实物库存相对盘面规模的比值。

    比值高说明「可交割的货相对持仓很多」，近月逼仓的难度大；
    比值低说明盘面持仓远大于库存，交割月容易出挤仓行情。
    """
    s = _fund(df, "warehouse")
    oi = _oi(df)
    return _safe_div(s, oi.where(oi > 0))


def f_spot_basis(df):
    """现货基差率：(现货价 − 期货收盘) / 期货收盘。

    正值 = 现货升水（backwardation 的现货端读数），负值 = 期货升水。
    和展期因子是两回事：展期看的是**期货曲线内部**的斜率，基差看的是
    现货和盘面的差。两者同时为正才是「现货紧张」的完整证据。
    """
    s = _fund(df, "spot_price")
    return _safe_div(s - df["close"], df["close"])


def f_spot_basis_chg_5(df):
    """基差率的 5 日变化。现货相对盘面在走强还是走弱。"""
    s = _fund(df, "spot_price")
    basis = _safe_div(s - df["close"], df["close"])
    return basis - basis.shift(5)


def f_profit_val(df):
    """加工利润（元/吨）的绝对水平。

    跨品种不可比（螺纹的钢厂利润和豆粕的压榨利润不是一个量纲），
    但**同一品种的时序**上很有用：利润为负时上游会减产，是远月的供应端利多。
    """
    return _fund(df, "profit")


def f_profit_chg_5(df):
    """加工利润的 5 日变化率。用绝对值做分母，利润为负时符号才不会翻。"""
    s = _fund(df, "profit")
    return _safe_div(s - s.shift(5), s.shift(5).abs())


def f_inv_chg_5(df):
    """期货库存 5 日变化率。和仓单类似但口径更宽（含非注册仓单）。"""
    s = _fund(df, "inventory")
    return s.pct_change(5)


def f_deliv_chg_30(df):
    """交割量 30 日变化率。月频指标，30 天正好对上一个月。

    交割量突然放大往往意味着「有货要交」或者「逼仓成功」，对近月合约的
    影响远大于远月，所以这条因子只对近月有解释力。
    """
    s = _fund(df, "delivery")
    return _safe_div(s - s.shift(30), s.shift(30).abs())


# ---------------------------------------------------------------- 板块 / 行业指数
# 板块划分沿用 future_quant 的 SECTORS，按我们的品种代码（大写）补齐。
SECTORS: dict[str, list[str]] = {
    "贵金属": ["AG", "AU"],
    "有色金属": ["CU", "AL", "ZN", "PB", "NI", "SN", "BC", "AO"],
    "黑色": ["RB", "HC", "I", "J", "JM", "FG", "SF", "SM", "SS"],
    "能化": ["SC", "BU", "FU", "LU", "MA", "TA", "EB", "PP", "L", "PG", "EG",
             "SA", "PF", "EC", "PX", "V", "UR", "RU", "BR", "SP", "PR"],
    "农产品": ["A", "M", "C", "CS", "RM", "P", "OI", "Y", "JD", "LH", "AP", "CJ",
               "CF", "SR", "PK", "B", "CY", "WH", "RI", "JR", "PM", "RS"],
    "新能源": ["LC", "PS", "SI"],
    "股指": ["IF", "IH", "IC", "IM"],
    "国债": ["T", "TF", "TS", "TL"],
}

# 品种 → 板块。反向索引，避免每次遍历。
_SECTOR_OF: dict[str, str] = {}
for _sec, _syms in SECTORS.items():
    for _s in _syms:
        _SECTOR_OF[_s] = _sec

_sector_cache: dict[str, dict] = {}


def _data_dir() -> str:
    from ttrade.paths import data_dir as _dd

    return _dd("cn")


def _sector_panel() -> dict:
    """全市场日收益面板 + 板块指数收益。

    读 95 个 CSV 只取 4 列，一次约 0.4 秒；算完之后整个进程复用。
    缓存键带上 `_manifest.json` 的 mtime —— 补完数据文件变了就该重算。

    权重用 `amount.shift(1)`：当天的成交额要等收盘才知道，拿它给当天加权
    等于用未来信息决定当天的指数。
    """
    d = _data_dir()
    try:
        stamp = str(os.path.getmtime(os.path.join(d, "_manifest.json")))
    except OSError:
        stamp = "0"
    hit = _sector_cache.get(stamp)
    if hit is not None:
        return hit

    closes, amounts = {}, {}
    try:
        names = [f for f in os.listdir(d) if f.endswith(".csv")]
    except OSError:
        names = []
    for fn in names:
        path = os.path.join(d, fn)
        try:
            df = pd.read_csv(path, usecols=lambda c: c in
                             ("date", "close", "amount", "adj_factor"))
        except Exception:  # noqa: BLE001
            continue
        if df.empty or "close" not in df.columns:
            continue
        sym = fn.split("_")[0].upper()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates("date").set_index("date")
        c = pd.to_numeric(df["close"], errors="coerce")
        if "adj_factor" in df.columns:
            f = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
            c = c * f
        if c.notna().sum() < 250:      # 覆盖太短的新品种不进指数
            continue
        closes[sym] = c
        if "amount" in df.columns:
            amounts[sym] = pd.to_numeric(df["amount"], errors="coerce")

    panel: dict = {"ret": pd.DataFrame(), "amt": pd.DataFrame(), "idx": {}}
    if not closes:
        _sector_cache[stamp] = panel
        return panel
    px = pd.DataFrame(closes).sort_index()
    ret = px.pct_change().replace([np.inf, -np.inf], np.nan)
    amt = pd.DataFrame(amounts).reindex(px.index)
    # 成交额缺失（早期 CSV 没有 amount 列）时退回等权，别让整段变 NaN
    w_all = amt.shift(1)
    panel["ret"], panel["amt"] = ret, w_all

    def _idx(syms: list[str]) -> pd.Series:
        use = [s for s in syms if s in ret.columns]
        if len(use) < 2:
            return pd.Series(dtype=float)
        sub, w = ret[use], w_all[use]
        wsum = w.sum(axis=1)
        eq = sub.mean(axis=1)
        wavg = (sub * w).sum(axis=1) / wsum.where(wsum > 0)
        return wavg.fillna(eq)

    for sec, syms in SECTORS.items():
        s = _idx(syms)
        if len(s):
            panel["idx"][f"sector_{sec}"] = s
    # 商品指数：剔除股指和国债，否则「商品」会被利率和权益主导
    comm = [s for s in ret.columns
            if _SECTOR_OF.get(s) not in ("股指", "国债")]
    s = _idx(comm)
    if len(s):
        panel["idx"]["commodity"] = s

    _sector_cache[stamp] = panel
    return panel


def _map_index(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    """把日频指数序列铺到 bar 上（分钟线上整体后移一个交易日）。

    和 `factors_seasonal._map_daily` 同样的理由：指数用的是**当天收盘**算的，
    分钟线的 09:00 那根拿到当天收盘的指数就是未来函数。
    """
    if s is None or len(s) == 0:
        return _nan(df)
    day = _day(df)
    idx = pd.DatetimeIndex(df.index)
    if len(day) != len(day.unique()):
        days = pd.DatetimeIndex(sorted(set(day)))
        day = day.map(pd.Series(days[:-1], index=days[1:]))
    return pd.Series(day.map(s).to_numpy(dtype=float), index=idx)


def _sector_ret(df: pd.DataFrame) -> pd.Series:
    panel = _sector_panel()
    sec = _SECTOR_OF.get(_symbol(df))
    if not sec:
        return _nan(df)
    return _map_index(df, panel["idx"].get(f"sector_{sec}"))


def f_sector_mom_10(df):
    """所属板块指数的 10 日动量。

    同一个板块里所有品种取值相同 —— 它衡量的是**板块 beta**，不是品种 alpha。
    截面策略里它更适合当「风险暴露」而不是选品种的打分项。
    """
    r = _sector_ret(df)
    return r.rolling(10, min_periods=8).sum()


def f_sector_ret_20(df):
    """所属板块指数的 20 日累计收益。"""
    r = _sector_ret(df)
    return r.rolling(20, min_periods=15).sum()


def f_sector_vol_20(df):
    """所属板块指数的 20 日年化波动。板块在恐慌时，个股的动量最容易失效。"""
    r = _sector_ret(df)
    return r.rolling(20, min_periods=15).std(ddof=1) * np.sqrt(244.0)


def f_sector_corr_20(df):
    """品种与所属板块指数的 20 日相关性。

    相关性突然掉下来，说明这个品种在走独立行情（自己的基本面驱动），
    这时候用板块动量去解释它会完全失效。
    """
    r = _sector_ret(df)
    own = df["close"].pct_change()
    if len(own) != len(r):
        return _nan(df)
    return own.rolling(20, min_periods=15).corr(r)


def f_commodity_mom_10(df):
    """全市场商品指数 10 日动量（剔除股指和国债）。

    商品整体的牛熊，是单品种趋势最强的「顺风/逆风」变量。
    """
    panel = _sector_panel()
    return _map_index(df, panel["idx"].get("commodity")).rolling(
        10, min_periods=8).sum()


def f_commodity_vol_20(df):
    """全市场商品指数 20 日年化波动。"""
    panel = _sector_panel()
    return _map_index(df, panel["idx"].get("commodity")).rolling(
        20, min_periods=15).std(ddof=1) * np.sqrt(244.0)


# ---------------------------------------------------------------- 注册
# 公式单独一张表：specs 是定长元组（最后一个可选 hint），把公式塞进去会让
# 「第 6 个元素是 hint 还是 formula」变得含糊。这里按 key 查，缺了就用空串，
# 详情面板会显示「未收录公式」。
FORMULAS = {
    "mom_1": "Mom(1) = P_t / P_{t−1} − 1",
    "mom_3": "Mom(3) = P_t / P_{t−3} − 1",
    "mom_10": "Mom(10) = P_t / P_{t−10} − 1",
    "ma_cross_5_20": "(MA_5 − MA_20) / MA_20",
    "vol_10": "σ(r, 10) × √244，r 为日收益",
    "abs_ret_20": "mean(|r|, 20)",
    "pos_20": "(P_t − LL_20) / (HH_20 − LL_20)",
    "range_1": "(High − Low) / Close",
    "vol_ratio_5_20": "mean(Vol, 5) / mean(Vol, 20)",
    "oi_change_1": "OI_t / OI_{t−1} − 1",
    "oi_change_5": "OI_t / OI_{t−5} − 1",
    "oi_change_robust": "expm1( Σ_{k=1..20} clip(ln(1+ΔOI_k), ±0.15) )",
    "turnover_20": "mean(Vol, 20) / mean(OI, 20)",
    "log_amount_20": "ln(1 + Σ_{20} Amount)",
    "vol_change_20": "mean(Vol, t−9..t) / mean(Vol, t−19..t−10) − 1",
    "term_slope": "(P_2 − P_3) / P_2，P_i 为第 i 近月价",
    "main_oi_share": "OI_近月 / (OI_1 + OI_2 + OI_3)",
    "term_near_far": "(P_1 − P_3) / P_1",
    "war_chg_1": "WAR_t / WAR_{t−1} − 1（T−1 口径）",
    "war_chg_5": "WAR_t / WAR_{t−5} − 1（T−1 口径）",
    "war_oi": "WAR / OI（均为 T−1）",
    "spot_basis": "(Spot − Fut) / Fut（均为 T−1）",
    "spot_basis_chg_5": "Basis_t − Basis_{t−5}",
    "profit_val": "Profit（元/吨，T−1）",
    "profit_chg_5": "(Profit_t − Profit_{t−5}) / |Profit_{t−5}|",
    "inv_chg_5": "INV_t / INV_{t−5} − 1（T−1 口径）",
    "deliv_chg_30": "DLV_t / DLV_{t−30} − 1（月频，T−1 口径）",
    "sector_mom_10": "Σ_{10} r_sector，r_sector 为板块指数日收益",
    "sector_ret_20": "P_sector,t / P_sector,{t−20} − 1",
    "sector_vol_20": "σ(r_sector, 20) × √244",
    "sector_corr_20": "corr(r_own, r_sector, 20)",
    "commodity_mom_10": "Σ_{10} r_comm，r_comm 为全市场商品指数日收益",
    "commodity_vol_20": "σ(r_comm, 20) × √244",
}


def register_into(FACTORS, FactorSpec) -> int:
    specs = [
        # ---- 趋势 / 波动 / 形态 / 量能（CTA 日线 + RIA）----
        ("mom_1", "动量 1 日", f_mom_1, "趋势", "1 日涨跌幅，短线反转的原料。"),
        ("mom_3", "动量 3 日", f_mom_3, "趋势", "3 日涨跌幅。"),
        ("mom_10", "动量 10 日", f_mom_10, "趋势", "10 日涨跌幅，两周趋势。"),
        ("ma_cross_5_20", "均线交叉 5/20", f_ma_cross_5_20, "趋势",
         "(MA5−MA20)/MA20，两条均线的距离，比价格动量平滑。"),
        ("vol_10", "波动率 10", f_vol_10, "波动",
         "10 日年化已实现波动率，短窗口对波动突变反应快。", -1),
        ("abs_ret_20", "平均绝对收益 20", f_abs_ret_20, "波动",
         "20 日平均 |日收益|，不年化的「每天动多少」，RIA 活跃度因子。", -1),
        ("pos_20", "区间位置 20", f_pos_20, "形态",
         "收盘在 20 日高低区间中的位置，突破强度的连续化。"),
        ("range_1", "单日振幅", f_range_1, "形态",
         "当根 (高−低)/收盘，hl_range 是它的 20 日均值。", -1),
        ("vol_ratio_5_20", "量比 5/20", f_vol_ratio_5_20, "量能",
         "5 日均量 / 20 日均量，看最近一周是否整体放量。"),
        ("oi_change_1", "持仓变化 1 日", f_oi_change_1, "量能",
         "持仓量 1 日变化率，最直接的增仓/减仓读数。"),
        ("oi_change_5", "持仓变化 5 日", f_oi_change_5, "量能",
         "持仓量 5 日变化率，一周的资金进出。"),
        ("oi_change_robust", "持仓变化·换月稳健", f_oi_change_robust, "量能",
         "逐日对数变化率累加、单日截断 ±15%，挡掉主力换月的跳变。"),
        ("turnover_20", "换手率 20", f_turnover_20, "量能",
         "20 日均量 / 20 日均持仓，RIA 的 volume_oi。"),
        ("log_amount_20", "成交额 20（对数）", f_log_amount_20, "量能",
         "20 日成交额合计取对数，衡量品种容量，RIA 的 amount。"),
        ("vol_change_20", "成交量变化 20", f_vol_change_20, "量能",
         "近 10 日均量 / 前 10 日均量 − 1，RIA 的 volume_change。"),
        # ---- 展期（CTA 期限结构口径）----
        ("term_slope", "期限斜率 2-3", f_term_slope, "展期",
         "(P2−P3)/P2 原始价差率，不年化，换月密集品种上更稳。"),
        ("main_oi_share", "近月持仓占比 3 腿", f_main_oi_share, "展期",
         "近月持仓 / 三条腿持仓之和，衡量资金是否挤在最近那个活跃月上。"),
        ("term_near_far", "近远月价差", f_term_near_far, "展期",
         "(P1−P3)/P1，整条曲线两端的距离，比只看 P1/P2 抗噪。"),
        # ---- 基本面 ----
        ("war_chg_1", "仓单变化 1 日", f_war_chg_1, "基本面",
         "仓单数量 1 日变化率，仓单增加＝供应宽松。", -1),
        ("war_chg_5", "仓单变化 5 日", f_war_chg_5, "基本面",
         "仓单数量 5 日变化率。", -1),
        ("war_oi", "仓单持仓比", f_war_oi, "基本面",
         "仓单 / 期货持仓，可交割库存相对盘面规模，低＝易挤仓。"),
        ("spot_basis", "现货基差率", f_spot_basis, "基本面",
         "(现货 − 期货)/期货，正值＝现货升水。"),
        ("spot_basis_chg_5", "基差变化 5 日", f_spot_basis_chg_5, "基本面",
         "基差率的 5 日变化，现货相对盘面在走强还是走弱。"),
        ("profit_val", "加工利润", f_profit_val, "基本面",
         "加工利润绝对水平（元/吨），跨品种不可比，同品种时序可比。"),
        ("profit_chg_5", "加工利润变化 5 日", f_profit_chg_5, "基本面",
         "加工利润 5 日变化率，分母取绝对值，利润为负时符号不翻。"),
        ("inv_chg_5", "库存变化 5 日", f_inv_chg_5, "基本面",
         "期货库存 5 日变化率，口径比仓单宽。", -1),
        ("deliv_chg_30", "交割量变化 30 日", f_deliv_chg_30, "基本面",
         "交割量 30 日变化率，月频指标，近月合约的解释力更强。"),
        # ---- 板块行业 ----
        ("sector_mom_10", "板块动量 10", f_sector_mom_10, "板块行业",
         "所属板块指数的 10 日动量，同板块内取值相同，是板块 beta。"),
        ("sector_ret_20", "板块收益 20", f_sector_ret_20, "板块行业",
         "所属板块指数的 20 日累计收益。"),
        ("sector_vol_20", "板块波动 20", f_sector_vol_20, "板块行业",
         "所属板块指数的 20 日年化波动。", -1),
        ("sector_corr_20", "板块相关性 20", f_sector_corr_20, "板块行业",
         "品种与所属板块指数的 20 日相关性，掉下来＝在走独立行情。"),
        ("commodity_mom_10", "商品指数动量 10", f_commodity_mom_10, "板块行业",
         "全市场商品指数（剔除股指/国债）10 日动量。"),
        ("commodity_vol_20", "商品指数波动 20", f_commodity_vol_20, "板块行业",
         "全市场商品指数 20 日年化波动。", -1),
    ]
    n = 0
    # 展期 / 基本面 / 板块行业三类依赖期货衍生数据（真实合约、仓单、库存、基差、
    # 板块指数），A 股和美股上没有这些列 → 标 ("cn",) 让 filter_by_market 过滤掉。
    _FUTURES_ONLY_CATS = {"展期", "基本面", "板块行业"}
    # 这几个挂在「量能」下但算的是**持仓量**（成交量/持仓量、持仓变化），股票没有
    # 持仓量这一说，同样只能给期货用。按 key 单独标，不能按类别一刀切 ——
    # 同一个「量能」类里的 log_amount_20 / vol_change_20 是纯成交量的，股票也能算。
    _FUTURES_ONLY_KEYS = {"oi_change_1", "oi_change_5", "oi_change_robust", "turnover_20"}
    for key, label, func, cat, desc, *rest in specs:
        hint = rest[0] if rest else 1
        mk = ("cn",) if (cat in _FUTURES_ONLY_CATS or key in _FUTURES_ONLY_KEYS) else ()
        FACTORS[key] = FactorSpec(key, label, func, cat, desc, hint,
                                  formula=FORMULAS.get(key, ""), markets=mk)
        n += 1
    return n
