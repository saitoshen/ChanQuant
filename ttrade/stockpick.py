"""A 股「按条件选股」：给截面策略挑一个能跑得动的品种池。

为什么需要这一层：期货只有 78 个品种，勾选框摆得下；A 股 5577 只，把
5577 个复选框塞进弹出面板，界面会直接卡死，人也找不到想选的那只。

所以 A 股走**条件筛选**：先按板块 / 行业 / 成交额 / 市值 / 关键词收窄，再
在结果里勾。默认条件是「近 20 日日均成交额 ≥ 1 亿」—— 流动性太差的票，
回测里滑点和涨跌停会把结论完全带偏，先剔掉比事后解释便宜得多。

数据全部来自本地 `stock.duckdb`，不联网：

- `stock_meta`：代码 / 名称 / 行业 / 板块
- `stock_basic_ts`：最新一个交易日的总市值、换手率
- `stock_bars_ts`：近 N 个交易日的日均成交额
"""

from __future__ import annotations

import pandas as pd

from ttrade import stockstore as SS

# 板块代号 → 中文。和 `stock_meta.board` 的取值一致
BOARDS = [
    ("主板", "主板"),
    ("创业板", "创业板"),
    ("科创板", "科创板"),
    ("北交所", "北交所"),
]


def snapshot(days: int = 20) -> pd.DataFrame:
    """全市场最新快照：代码 / 名称 / 行业 / 板块 / 市值 / 换手 / 日均成交额。

    成交额按**最近 days 个交易日**的均值算（默认 20 日），单位统一成「亿元」——
    表里全是亿级数字时，用户不需要再心算量纲。
    """
    if not SS.exists():
        return pd.DataFrame(columns=["ts_code", "name", "industry", "board",
                                     "total_mv", "turnover_rate", "amount"])
    with SS.connect(read_only=True) as con:
        meta = con.execute(
            "SELECT ts_code, name, industry, board FROM stock_meta"
        ).fetchdf()
        # 最新交易日的市值 / 换手率。整表 900 多万行，先取 max(dt) 再按天取，
        # 走的是 (ts_code, dt) 主键，一次只读 5000 多行
        last = con.execute("SELECT max(dt) FROM stock_basic_ts").fetchone()[0]
        basic = pd.DataFrame(columns=["ts_code", "total_mv", "turnover_rate"])
        if last is not None:
            basic = con.execute(
                "SELECT ts_code, total_mv, turnover_rate FROM stock_basic_ts "
                "WHERE dt = ?", [last]
            ).fetchdf()
        # 日均成交额：只扫最近 days 个交易日，别去动八年的全表
        amt = pd.DataFrame(columns=["ts_code", "amount"])
        days_row = con.execute(
            "SELECT DISTINCT dt FROM stock_bars_ts ORDER BY dt DESC LIMIT ?", [int(days)]
        ).fetchall()
        if days_row:
            first = days_row[-1][0]
            amt = con.execute(
                "SELECT ts_code, avg(amount) AS amount FROM stock_bars_ts "
                "WHERE dt >= ? GROUP BY 1", [first]
            ).fetchdf()

    df = meta.merge(basic, on="ts_code", how="left").merge(amt, on="ts_code", how="left")
    # Tushare 的 amount 是千元。换成亿元：×1000 / 1e8
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce") * 1000 / 1e8
    df["total_mv"] = pd.to_numeric(df.get("total_mv"), errors="coerce") / 1e4   # 万元 → 亿
    df["turnover_rate"] = pd.to_numeric(df.get("turnover_rate"), errors="coerce")
    df["board"] = df["board"].fillna("")
    df["industry"] = df["industry"].fillna("")
    return df


def industries(df: pd.DataFrame) -> list[str]:
    """行业列表，按股票数从多到少。"""
    if df is None or df.empty:
        return []
    vc = df["industry"].replace("", pd.NA).dropna().value_counts()
    return [str(k) for k in vc.index]


def filter_frame(df: pd.DataFrame, boards=None, industry: str = "",
                 min_amount: float = 0.0, min_mv: float = 0.0,
                 max_mv: float = 0.0, keyword: str = "",
                 limit: int = 300) -> pd.DataFrame:
    """按条件收窄。`min_amount` / `min_mv` / `max_mv` 的单位都是「亿」。

    排序用成交额降序：同样是「符合条件」，先看流动性好的，挑池子时也更可能
    挑到真能交易的。
    """
    if df is None or df.empty:
        return df
    out = df
    if boards:
        keep = [b for b in boards if b]
        if keep:
            out = out[out["board"].isin(keep)]
    if industry:
        out = out[out["industry"] == industry]
    if min_amount > 0:
        out = out[out["amount"].fillna(0) >= float(min_amount)]
    if min_mv > 0:
        out = out[out["total_mv"].fillna(0) >= float(min_mv)]
    if max_mv > 0:
        out = out[out["total_mv"].fillna(0) <= float(max_mv)]
    kw = str(keyword or "").strip()
    if kw:
        hit = (out["ts_code"].astype(str).str.contains(kw, case=False, na=False)
               | out["name"].astype(str).str.contains(kw, case=False, na=False))
        out = out[hit]
    out = out.sort_values("amount", ascending=False, na_position="last")
    return out.head(int(limit)) if limit else out
