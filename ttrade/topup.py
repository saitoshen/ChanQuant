"""把本地行情补到最新。

引擎订阅的 K 线本来就是实时的（`get_kline_serial` 会自己回填最近 3000 根），
策略算信号不缺数据。这个模块解决的是另一件事：**本地库停在几天前**。

组合页打开一条策略，K 线图画的是库里的历史 —— 库里最后一根是上次下载的那天，
图就一直停在那儿，看着像「策略没在跟行情」。所以策略一启动，顺手把引擎手上
这份实时序列写回库：下次打开、下次回测，数据就是最新的。

## 只按源周期落盘

库里的 15m / 30m / 2h / 4h 不是独立存的，是 5m / 1h 合并出来的（见
`tqdata.PERIODS`）。把 15m 的 bar 硬写进 15m 表，读的时候只会去找 5m，
白写一场 —— 界面还是「没有数据」。所以补库一律落到源周期上；而引擎订阅
本来就订的是源周期（见 `PortfolioRunner._subscribe`），这里直接复用那条
序列，不用另外再订一份。

## 正在走的那根不写

`get_kline_serial` 的最后一根是活 bar，OHLC 要到收盘才定。写进库之后再补数据
只会接「严格晚于库里的」，这根半成品就永久留在库尾，回测拿它当收盘价算信号。
所以凡是还没到结束时刻的 bar 一律丢掉。
"""

from __future__ import annotations

import pandas as pd

from ttrade import tqstore
from ttrade.tqdata import source_period, trading_days
from ttrade.tqlive import bars_from_serial

# 能落库的源周期。1d 不在这里：日线走另一条链路，别混进来
SOURCE_PERIODS = ("5m", "1h")


def source_of(period: str) -> str:
    """这个周期在库里应该落到哪个源周期上。"""
    return source_period(period)


def drop_live_tail(df: pd.DataFrame) -> pd.DataFrame:
    """去掉最后一根还没走完的 bar。"""
    if df is None or df.empty:
        return df
    now = pd.Timestamp.now()
    if df.index[-1] > now:
        return df.iloc[:-1]
    return df


def save(serial, symbol: str, exchange: str, period: str) -> dict:
    """把一份实时序列里**库里还没有**的部分写进 DuckDB。

    返回 `{"added": n, "src": "5m", "last": Timestamp|None, "error": "",
    "ok": bool}`。`ok=False` 表示这次没拿到可用数据（比如刚订阅、序列还没
    填充），调用方可以下一轮再试；出错只回 error，不抛 —— 补数据失败不该把
    策略打死。
    """
    src = source_of(period)
    out = {"added": 0, "src": src, "last": None, "error": "", "ok": False}
    if src not in SOURCE_PERIODS:
        out["error"] = f"{period} 不是可落库的源周期"
        out["ok"] = True          # 不重试：这是配置问题，不是「还没数据」
        return out
    try:
        df = bars_from_serial(serial, src)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if df is None or df.empty:
        return out
    df = drop_live_tail(df)
    if df.empty:
        return out
    try:
        last = tqstore.last_dt(symbol, exchange, src)
    except Exception:  # noqa: BLE001
        last = None
    if last is not None:
        # 只写严格晚于库里的。重叠那段库里那份 `adj_factor` 是对的，用 1.0
        # 盖过去会让复权价在接缝处跳一下
        df = df[df.index > last]
    out["ok"] = True
    if df.empty:
        return out
    df = df.copy()
    # 后复权序列的最新一段因子恒为 1（见 `tqfetch.adj_factor_series`），实时段
    # 天然落在这一段里。真要换月了，下次全量下载会重算这张表
    df["adj_factor"] = 1.0
    df["trading_day"] = trading_days(df.index).to_numpy()
    d = df.reset_index().rename(columns={"datetime": "dt"})
    try:
        with tqstore.connect() as con:
            out["added"] = tqstore.upsert_bars(con, symbol, exchange, src, d)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"写库失败：{type(exc).__name__}: {exc}"
        return out
    out["last"] = df.index[-1]
    return out
