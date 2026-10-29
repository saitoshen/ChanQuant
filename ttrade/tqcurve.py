"""用天勤的真实合约 K 线构建期限结构（展期因子要用的那份）。

## 为什么要有这个

展期类因子（`roll_yield` / `roll_annual` / `roll_curv` / `roll_change`，以及
`o3_` / `o6_` 两套活跃口径）读的是 `data_futures_aux/<品种>.csv`，那份文件一直是
`auxbuild.py` 从 **Tushare** 的合约日线聚合出来的。没有 Tushare token 的人，
分钟线全下完了，展期因子还是整列「无数据」—— 卡在一条完全不相干的链路上。

天勤的真实合约（`contract_bars`）本来就已经落库了，用它同样能算近/次近/第三近
的价差，所以这里把同一套口径重新实现一遍，写到 `data_tq/curve/`：

    data_tq/curve/RB_SHFE.csv        列与 data_futures_aux/RB_SHFE.csv 完全一致

读取端（`auxdata.load_curve`）在 aux 文件缺失时回退到这里，两套互不覆盖 ——
Tushare 那份是 20 年、天勤这份是十年出头，混着写会让同一天的展期值随机来自两个
源，谁也说不清哪个对。

## 口径与 `contracts.build_curve` 完全一致

直接复用 `contracts.build_curve`：它只要求「一天的横截面」长成
`date / ts_code / close / open_interest` 加上一份 `basic`（含 `fut_code`、
`exchange`、`delist`）。天勤这边缺的只有 `delist`（到期日），从合约代码的
年月推 —— 商品的最后交易日是交割月的第 15 个自然日前后，股指是第三个周五，
这里统一取「交割月最后一天」，只影响 `dte` 这个**排序与过滤**用的量，不影响价格。
"""

from __future__ import annotations

import os

import pandas as pd

from . import contracts as C
from . import tqstore as S
from .paths import data_dir


def curve_dir() -> str:
    """天勤期限结构的落盘目录：`data_tq/curve/`。"""
    return os.path.join(os.path.dirname(data_dir("cn").rstrip("/\\")), "data_tq", "curve")


def curve_path(symbol: str, exchange: str) -> str:
    return os.path.join(curve_dir(), f"{symbol}_{exchange}.csv")


def _delivery(contract: str) -> pd.Timestamp | None:
    """`RB2601.SHF` → 2026-01-15（交割月的最后交易日）。

    国内商品期货的最后交易日基本是交割月 15 日（遇节假日顺延），股指是第三个
    周五。这里统一取 15 日 —— 比取月末准得多：取月末的话 RB2609 在 9 月 11 日
    还「剩 19 天」，会把它当成近月留着，而 Tushare 那份（真实摘牌日）这时候
    已经把它筛掉了，两边算出来的 c1 直接差一个合约。
    """
    import re

    m = re.search(r"(\d{4})\.", str(contract))
    if not m:
        return None
    yymm = int(m.group(1))
    year, month = 2000 + yymm // 100, yymm % 100
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=15)


def build(symbol: str, exchange: str, period: str = "1d") -> tuple[pd.DataFrame, str]:
    """算一个品种的期限结构。返回 (表, 说明文字)；没有合约数据时表为空。"""
    if not S.exists():
        return pd.DataFrame(columns=C.CURVE_COLS), "仓库不存在"
    with S.connect(read_only=True) as con:
        df = con.execute(
            "SELECT contract, trading_day, dt, close, close_oi FROM contract_bars"
            " WHERE symbol=? AND exchange=? AND period=? AND trading_day IS NOT NULL"
            " ORDER BY contract, dt",
            [symbol, exchange, period],
        ).df()
    if df.empty:
        return pd.DataFrame(columns=C.CURVE_COLS), f"没有 {period} 真实合约数据"

    # 分钟级合约一天有几十根，期限结构要的是「每天一根」。取当天最后一根 ——
    # 和 Tushare 那份（日线收盘）口径一致。不聚合的话 build_curve 会把同一天
    # 的每根 bar 当成一个独立合约，算出来的 c1/c2 全是同一时刻的同一批合约。
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["close_oi"] = pd.to_numeric(df["close_oi"], errors="coerce").fillna(0.0)
    df = (
        df.dropna(subset=["close"])
        .groupby(["contract", "trading_day"], as_index=False)
        .agg(close=("close", "last"), open_interest=("close_oi", "last"))
    )
    df["date"] = pd.to_datetime(df["trading_day"])
    df["ts_code"] = df["contract"].astype(str)
    df = df[["date", "ts_code", "close", "open_interest"]]

    basic = pd.DataFrame({"ts_code": sorted(df["ts_code"].unique())})
    basic["fut_code"] = symbol
    basic["exchange"] = exchange
    basic["delist"] = basic["ts_code"].map(_delivery)
    basic = basic.dropna(subset=["delist"])
    if basic.empty:
        return pd.DataFrame(columns=C.CURVE_COLS), "合约代码解析不出交割月"

    out = C.build_curve(df, basic)
    if out.empty:
        return out, "合约数不足（每天至少要两个有持仓的合约）"
    # build_curve 会把分组键 symbol / exchange 一起写进记录；aux 文件的列里
    # 没有这两个（品种就是文件名），带出去会让 join 到 K 线上多两列字符串
    out = out.drop(columns=[c for c in ("symbol", "exchange") if c in out.columns])
    return out, f"{len(out)} 个交易日"


def build_all(pairs: list[tuple[str, str]], period: str = "1d", on_row=None,
              cancel=None) -> dict:
    """批量构建并写盘。返回 {品种_交易所: 行数}。"""
    os.makedirs(curve_dir(), exist_ok=True)
    out: dict[str, int] = {}
    for sym, ex in pairs:
        if cancel is not None and cancel():
            break
        key = f"{sym}_{ex}"
        try:
            df, note = build(sym, ex, period)
        except Exception as exc:  # noqa: BLE001
            out[key] = 0
            if on_row is not None:
                on_row(key, 0, f"失败：{type(exc).__name__}: {str(exc)[:60]}")
            continue
        n = 0
        if not df.empty:
            n = C.merge_write(curve_path(sym, ex), df, ["date"])
        out[key] = n
        if on_row is not None:
            on_row(key, n, note)
    return out
