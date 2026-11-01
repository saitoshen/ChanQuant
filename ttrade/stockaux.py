"""A 股衍生数据（基本面 / 资金流 / 复权因子）的读取与挂载。

## 和 `auxdata.py` 是同一个套路，只是数据源不同

期货的衍生数据（期限结构、期权 IV）来自 `data_futures_aux/*.csv`；股票的
衍生数据（PE/PB/换手率、资金流、复权因子）来自 `data_tq/stock.duckdb` 的
三张表。**挂载方式完全一致**：加载 K 线时把衍生列 join 到 df 上，因子函数
退化成读列，`filter_available()` 自动把没数据的因子剔掉。

这样 `factors.raw(df, key)` 的签名不用动，策略 / 机器学习 / 因子分析页三条
链路一行都不用改。

## PIT：基本面一定要滞后

`daily_basic` 的 PE / 换手率 / 市值是**收盘后**才发布的。日线上按当天 join
等于「用今天收盘的数据决定今天收盘的信号」——信号在收盘那一刻产生、下一根
开盘成交，这条链路上它是合法的（和 `auxdata.attach` 对日线的处理一致）。

**分钟线上不能按当天**：上午 10:00 的 bar 不可能知道当天收盘后的换手率。
所以分钟级统一取**上一个交易日**的快照（`attach_intraday` 的做法）。

## 资金流的单位

Tushare `moneyflow` 的 `*_vol` 单位是**手**，`net_mf_amount` 单位是**万元**。
两者量纲不同，所以这里不直接拼，而是先算出**占比**再挂上去：
`mf_net_ratio` = 净流入额 / 成交额，`mf_lg_ratio` = 大单+超大单净买入手数 /
全部手数。占比是尺度无关的，跨股票可比，也不会因为市值大小而天然偏大。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stockstore as SS

# 挂到 df 上的列名 → 来源表字段
BASIC_MAP = {
    "pe": "pe",
    "pe_ttm": "pe_ttm",
    "pb": "pb",
    "ps_ttm": "ps_ttm",
    "total_mv": "total_mv",
    "circ_mv": "circ_mv",
    "turnover_rate": "turnover_rate",
    "turnover_rate_f": "turnover_rate_f",
    "dv_ratio": "dv_ratio",
    "dv_ttm": "dv_ttm",
}

# 资金流原始列（占比在 `_moneyflow_frame` 里算好再挂）
FLOW_COLS = [
    "mf_net_ratio",      # 净流入额 / 成交额
    "mf_lg_ratio",       # (大单+超大单) 净买入手数 / 全部手数
    "mf_elg_ratio",      # 超大单净买入手数 / 全部手数
    "mf_sm_ratio",       # 小单净买入手数 / 全部手数
]

AUX_COLS = list(BASIC_MAP) + FLOW_COLS + ["adj_factor", "concept_count", "board_code"]

# 板块 → 哑变量。不是「好因子」，是控制变量：创业板/科创板的波动中枢和涨跌幅
# 限制都和主板不同，混在一起做截面排序时它天然占优或吃亏。
BOARD_CODE = {"主板": 0.0, "中小板": 0.0, "创业板": 1.0, "科创板": 2.0, "北交所": 3.0}


def _board_code(ts_code: str) -> float:
    """从 stock_meta 取板块，映射成哑变量。"""
    try:
        df = SS.load_meta()
    except Exception:  # noqa: BLE001
        return np.nan
    if df is None or df.empty or "board" not in df.columns:
        return np.nan
    row = df[df["ts_code"] == ts_code]
    if row.empty:
        return np.nan
    return BOARD_CODE.get(str(row["board"].iloc[0]), np.nan)


def _basic_frame(ts_code: str) -> pd.DataFrame:
    """基本面宽表，索引为交易日。"""
    df = SS.load_basic(ts_code)
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=pd.to_datetime(df["dt"]))
    for dst, src in BASIC_MAP.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=float)
    out.index.name = "date"
    return out[~out.index.duplicated(keep="last")].sort_index()


def _moneyflow_frame(ts_code: str) -> pd.DataFrame:
    """资金流宽表（已换算成占比），索引为交易日。"""
    df = SS.load_moneyflow(ts_code)
    if df is None or df.empty:
        return pd.DataFrame()

    def num(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce").astype(float)

    buy = num("buy_sm_vol") + num("buy_md_vol") + num("buy_lg_vol") + num("buy_elg_vol")
    sell = num("sell_sm_vol") + num("sell_md_vol") + num("sell_lg_vol") + num("sell_elg_vol")
    total = (buy + sell).replace(0.0, np.nan)

    out = pd.DataFrame(index=pd.to_datetime(df["dt"]))
    # 一律 to_numpy：源 Series 是 RangeIndex，out 是 DatetimeIndex，直接赋值会
    # 按索引对齐 → 整列 NaN（这个坑在 `_basic_frame` 里已经踩过一次）。
    out["mf_lg_ratio"] = (((num("buy_lg_vol") + num("buy_elg_vol"))
                           - (num("sell_lg_vol") + num("sell_elg_vol"))) / total).to_numpy()
    out["mf_elg_ratio"] = ((num("buy_elg_vol") - num("sell_elg_vol")) / total).to_numpy()
    out["mf_sm_ratio"] = ((num("buy_sm_vol") - num("sell_sm_vol")) / total).to_numpy()
    # 净流入额（万元）要除以成交额（元）才是无量纲占比，成交额在 K 线上，
    # 这里先原样带着，`attach()` 里再算。
    out["_net_amount"] = num("net_mf_amount").to_numpy()
    out.index.name = "date"
    return out[~out.index.duplicated(keep="last")].sort_index()


def _adj_series(ts_code: str) -> pd.Series:
    s = SS.load_one_adj(ts_code)
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return s


def _concept_count(ts_code: str) -> float:
    """这只股票挂了多少个概念板块。主题热度的粗代理。"""
    try:
        df = SS.load_concept()
    except Exception:  # noqa: BLE001
        return np.nan
    if df is None or df.empty or "ts_code" not in df.columns:
        return np.nan
    return float((df["ts_code"] == ts_code).sum())


def attach(df: pd.DataFrame, ts_code: str, intraday: bool = False) -> pd.DataFrame:
    """把基本面 / 资金流 / 复权因子 join 到 K 线上。缺数据时原样返回。

    用 `join(how="left")`：df 的索引是 date，行数和顺序都不能变 —— K 线渲染、
    信号对齐都依赖「第 i 行就是第 i 根 bar」。

    `intraday=True` 时所有快照整体后移一个交易日：分钟 bar 不能提前知道当天
    收盘后才知道的基本面 / 资金流。
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    parts = [p for p in (_basic_frame(ts_code), _moneyflow_frame(ts_code)) if not p.empty]
    if not parts:
        return df

    # 资金流净额要除以成交额才有意义；成交额就在 df 上（Tushare 日线有 amount）
    if "amount" in df.columns:
        for part in parts:
            if "_net_amount" in part.columns:
                amt = pd.to_numeric(df["amount"], errors="coerce").astype(float)
                amt = amt.replace(0.0, np.nan)
                # 万元 → 元 再除，得到无量纲占比
                part["mf_net_ratio"] = (part["_net_amount"] * 1e4) / amt.reindex(part.index)
            part.drop(columns=[c for c in ("_net_amount",) if c in part.columns],
                      inplace=True)

    out = df
    for part in parts:
        part = part.copy()
        if intraday:
            # 分钟 bar：只用上一个及更早交易日的快照
            part.index = part.index + pd.Timedelta(days=1)
        cols = [c for c in part.columns if c not in out.columns]
        if not cols:
            continue
        # 分钟 bar 的索引是带时分秒的时间戳，直接 join 对不上：先把快照按
        # 「日期」落到当天每根 bar 上（reindex 到 df 的 normalize 后再回填）
        if intraday:
            day = pd.Series(out.index.normalize(), index=out.index)
            mapped = part.reindex(pd.DatetimeIndex(day.unique()))
            for c in cols:
                out[c] = day.map(mapped[c]).to_numpy()
        else:
            out = out.join(part[cols], how="left")

    # 复权因子：除权除息事件检测要用
    adj = _adj_series(ts_code)
    if len(adj) and "adj_factor" not in out.columns:
        if intraday:
            day = pd.Series(out.index.normalize(), index=out.index)
            mapped = adj.reindex(pd.DatetimeIndex(day.unique()))
            out["adj_factor"] = day.map(mapped).to_numpy()
        else:
            out = out.join(adj.rename("adj_factor"), how="left")

    # 概念板块数量 / 板块哑变量都是常量，广播成整列
    if "concept_count" not in out.columns:
        n = _concept_count(ts_code)
        if np.isfinite(n):
            out["concept_count"] = n
    if "board_code" not in out.columns:
        b = _board_code(ts_code)
        if np.isfinite(b):
            out["board_code"] = b
    return out
