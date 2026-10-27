"""天勤行情仓库（DuckDB）。

为什么不再用 CSV：5 分钟全品种十年是**千万行**级别，按品种一个 CSV 会有两个问题 ——
读一个品种要把整个文件解析一遍（几十万行 × 12 列，几百毫秒），而横截面计算
（同一时刻所有品种）得打开 78 个文件。DuckDB 是列存 + 压缩，同一个文件里
按 (symbol, period, dt) 建主键索引，单品种查询是毫秒级，全市场切片是一条 SQL。

**5m 和 1h 放同一张表**（用 `period` 列区分），不分开建表：

- 结构完全一致，分表只会让读取层多一个分支、多一处可能不一致；
- 1h 只占 5m 的 1/12 空间，合表带来的体积代价可以忽略；
- 以后要加 1d 或 tick，加个 period 值就行，不用改 schema。

表结构：

    bars          主连（已复权因子列，原始 OHLC）+ period
    contract_bars 真实合约（不复权，单合约本身没有换月）
    switch        主力切换表（天勤口径）
    meta          键值对，记 schema 版本和数据覆盖

写入用 `INSERT OR REPLACE`，主键冲突就覆盖 —— 增量更新时同一根 bar 会被重下，
这样天然幂等。
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pandas as pd

from .paths import data_dir

SCHEMA_VERSION = "1"

BAR_COLS = [
    "symbol", "exchange", "period", "dt", "trading_day",
    "open", "high", "low", "close",
    "volume", "open_oi", "close_oi", "adj_factor",
]

DDL = [
    """
    CREATE TABLE IF NOT EXISTS bars (
        symbol      VARCHAR NOT NULL,
        exchange    VARCHAR NOT NULL,
        period      VARCHAR NOT NULL,
        dt          TIMESTAMP NOT NULL,
        trading_day DATE,
        open        DOUBLE,
        high        DOUBLE,
        low         DOUBLE,
        close       DOUBLE,
        volume      BIGINT,
        open_oi     BIGINT,
        close_oi    BIGINT,
        adj_factor  DOUBLE DEFAULT 1.0,
        PRIMARY KEY (symbol, exchange, period, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS contract_bars (
        symbol      VARCHAR NOT NULL,
        exchange    VARCHAR NOT NULL,
        period      VARCHAR NOT NULL,
        contract    VARCHAR NOT NULL,
        dt          TIMESTAMP NOT NULL,
        trading_day DATE,
        open        DOUBLE,
        high        DOUBLE,
        low         DOUBLE,
        close       DOUBLE,
        volume      BIGINT,
        open_oi     BIGINT,
        close_oi    BIGINT,
        PRIMARY KEY (symbol, exchange, period, contract, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS switch (
        symbol       VARCHAR NOT NULL,
        exchange     VARCHAR NOT NULL,
        switch_date  DATE NOT NULL,
        old_tq       VARCHAR,
        new_tq       VARCHAR,
        old_ts       VARCHAR,
        new_ts       VARCHAR,
        ratio        DOUBLE,
        old_close    DOUBLE,
        new_close    DOUBLE,
        price_src    VARCHAR,
        PRIMARY KEY (symbol, exchange, switch_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (k VARCHAR PRIMARY KEY, v VARCHAR)
    """,
]


def db_path() -> str:
    return os.path.join(os.path.dirname(data_dir("cn").rstrip("/\\")), "data_tq", "tq.duckdb")


def exists() -> bool:
    return os.path.isfile(db_path())


@contextmanager
def connect(read_only: bool = False):
    """打开仓库。只读模式允许多个进程同时读（GUI 与下载脚本并存）。

    写入时 DuckDB 是**单写多读**：同一时刻只允许一个写连接。所以下载脚本要写、
    GUI 只读，两边不冲突；如果两个下载脚本同时跑，后一个会拿到
    「Conflicting lock」错误 —— 这是预期行为，不要用重试掩盖。
    """
    import duckdb

    p = db_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    con = duckdb.connect(p, read_only=read_only)
    try:
        if not read_only:
            for ddl in DDL:
                con.execute(ddl)
            con.execute(
                "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", [SCHEMA_VERSION]
            )
        yield con
    finally:
        con.close()


# ------------------------------------------------------------------ 主连


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("volume", "open_oi", "close_oi"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")
    for c in ("open", "high", "low", "close", "adj_factor"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    if "adj_factor" not in out.columns:
        out["adj_factor"] = 1.0
    out["dt"] = pd.to_datetime(out["dt"])
    if "trading_day" in out.columns:
        out["trading_day"] = pd.to_datetime(out["trading_day"]).dt.date
    return out


def upsert_bars(con, symbol: str, exchange: str, period: str, df: pd.DataFrame) -> int:
    """写入主连行情。df 需要 dt/open/high/low/close/volume/open_oi/close_oi[/trading_day]。"""
    if df is None or df.empty:
        return 0
    d = _norm(df)
    d["symbol"] = symbol
    d["exchange"] = exchange
    d["period"] = period
    if "trading_day" not in d.columns:
        from .tqdata import trading_days

        d["trading_day"] = trading_days(d["dt"]).dt.date.to_numpy()
    d = d[BAR_COLS]
    con.register("_tmp_bars", d)
    con.execute(
        "INSERT OR REPLACE INTO bars SELECT symbol, exchange, period, dt, trading_day,"
        " open, high, low, close, volume, open_oi, close_oi, adj_factor FROM _tmp_bars"
    )
    con.unregister("_tmp_bars")
    return len(d)


def load_bars(
    symbol: str, exchange: str, period: str, start=None, end=None
) -> pd.DataFrame:
    """读主连行情，index=dt。没有这个品种时返回空 DataFrame（不抛错）。"""
    if not exists():
        return pd.DataFrame()
    sql = (
        "SELECT dt, open, high, low, close, volume, open_oi, close_oi, adj_factor,"
        " trading_day FROM bars WHERE symbol=? AND exchange=? AND period=?"
    )
    args: list = [symbol, exchange, period]
    if start is not None:
        sql += " AND dt >= ?"
        args.append(pd.Timestamp(start))
    if end is not None:
        sql += " AND dt <= ?"
        args.append(pd.Timestamp(end))
    sql += " ORDER BY dt"
    with connect(read_only=True) as con:
        df = con.execute(sql, args).df()
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["dt"])
    return df.set_index("dt")


def last_dt(symbol: str, exchange: str, period: str):
    if not exists():
        return None
    with connect(read_only=True) as con:
        r = con.execute(
            "SELECT max(dt) FROM bars WHERE symbol=? AND exchange=? AND period=?",
            [symbol, exchange, period],
        ).fetchone()
    return pd.Timestamp(r[0]) if r and r[0] is not None else None


def bar_count(symbol: str, exchange: str, period: str) -> int:
    if not exists():
        return 0
    with connect(read_only=True) as con:
        r = con.execute(
            "SELECT count(*) FROM bars WHERE symbol=? AND exchange=? AND period=?",
            [symbol, exchange, period],
        ).fetchone()
    return int(r[0]) if r else 0


# ------------------------------------------------------------------ 切换表


def upsert_switch(con, symbol: str, exchange: str, rec: pd.DataFrame) -> int:
    if rec is None or rec.empty:
        return 0
    d = rec.copy()
    d["symbol"] = symbol
    d["exchange"] = exchange
    d["switch_date"] = pd.to_datetime(d["switch_date"]).dt.date
    for c in ("old_tq", "new_tq", "old_ts", "new_ts"):
        if c not in d.columns:
            d[c] = None
    for c in ("ratio", "old_close", "new_close", "price_src"):
        if c not in d.columns:
            d[c] = None
    d = d[
        ["symbol", "exchange", "switch_date", "old_tq", "new_tq",
         "old_ts", "new_ts", "ratio", "old_close", "new_close", "price_src"]
    ]
    con.register("_tmp_sw", d)
    con.execute("INSERT OR REPLACE INTO switch SELECT * FROM _tmp_sw")
    con.unregister("_tmp_sw")
    return len(d)


def load_switch(symbol: str, exchange: str) -> pd.DataFrame:
    if not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT * FROM switch WHERE symbol=? AND exchange=? ORDER BY switch_date",
            [symbol, exchange],
        ).df()
    if df.empty:
        return df
    df["switch_date"] = pd.to_datetime(df["switch_date"])
    return df


def switch_symbols() -> list[tuple[str, str]]:
    if not exists():
        return []
    with connect(read_only=True) as con:
        rows = con.execute("SELECT DISTINCT symbol, exchange FROM switch ORDER BY 1,2").fetchall()
    return [(r[0], r[1]) for r in rows]


# ------------------------------------------------------------------ 真实合约


def upsert_contracts(con, symbol: str, exchange: str, period: str, df: pd.DataFrame) -> int:
    """df 需要 contract/dt/OHLCV。真实合约不复权，adj_factor 不落库。"""
    if df is None or df.empty:
        return 0
    d = df.copy()
    d["symbol"] = symbol
    d["exchange"] = exchange
    d["period"] = period
    d["dt"] = pd.to_datetime(d["dt"])
    if "trading_day" not in d.columns:
        from .tqdata import trading_days

        d["trading_day"] = trading_days(d["dt"]).dt.date.to_numpy()
    else:
        d["trading_day"] = pd.to_datetime(d["trading_day"]).dt.date
    for c in ("volume", "open_oi", "close_oi"):
        if c not in d.columns:
            d[c] = 0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype("int64")
    d = d[["symbol", "exchange", "period", "contract", "dt", "trading_day",
           "open", "high", "low", "close", "volume", "open_oi", "close_oi"]]
    con.register("_tmp_cb", d)
    con.execute("INSERT OR REPLACE INTO contract_bars SELECT * FROM _tmp_cb")
    con.unregister("_tmp_cb")
    return len(d)


def load_contracts(symbol: str, exchange: str, period: str,
                   contract: str | None = None) -> pd.DataFrame:
    if not exists():
        return pd.DataFrame()
    sql = ("SELECT contract, dt, open, high, low, close, volume, open_oi, close_oi"
           " FROM contract_bars WHERE symbol=? AND exchange=? AND period=?")
    args: list = [symbol, exchange, period]
    if contract:
        sql += " AND contract=?"
        args.append(contract)
    sql += " ORDER BY contract, dt"
    with connect(read_only=True) as con:
        df = con.execute(sql, args).df()
    if not df.empty:
        df["dt"] = pd.to_datetime(df["dt"])
    return df


# ------------------------------------------------------------------ 统计


def stats() -> dict:
    if not exists():
        return {"exists": False, "path": db_path()}
    out = {"exists": True, "path": db_path(),
           "size_mb": round(os.path.getsize(db_path()) / 1e6, 1)}
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT period, count(*) n, count(DISTINCT symbol) s,"
            " min(dt) a, max(dt) b FROM bars GROUP BY period ORDER BY period"
        ).fetchall()
        out["periods"] = {
            r[0]: {"rows": int(r[1]), "symbols": int(r[2]), "first": str(r[3]), "last": str(r[4])}
            for r in rows
        }
        r = con.execute("SELECT count(*), count(DISTINCT symbol) FROM switch").fetchone()
        out["switch"] = {"rows": int(r[0]), "symbols": int(r[1])}
        r = con.execute(
            "SELECT count(*), count(DISTINCT contract) FROM contract_bars"
        ).fetchone()
        out["contracts"] = {"rows": int(r[0]), "contracts": int(r[1])}
    return out
