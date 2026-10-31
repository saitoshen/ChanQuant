"""天勤期权仓库（DuckDB，独立文件）。

**为什么单独一个库**：期权和期货 K 线的键完全不同（期权是 contract + strike +
到期日），塞进 `tq.duckdb` 只会让那个 11GB 的库再翻一倍，而它要随包分发、整库
下载。期权这份是「想研究期权才下」的可选数据，体积又是十几 GB 量级，必须分开。

表结构：

    option_meta        合约定义：品种 / 交易所 / 认购认沽 / 行权价 / 到期日 / 标的
    option_bars        期权 K 线，主键 (contract, period, dt)
    option_underlying  标的 K 线（期权定价要用的 F），主键 (contract, period, dt)
    option_iv          按品种聚合的隐含波动率摘要，主键 (variety, period, dt)

**标的单独一张表**：算 IV 需要同一时刻的标的价格。把标的的 bar 混进 `option_bars`
会让「这个合约是期权还是期货」变成靠代码猜，取数时到处要排除；单独一张表就没
这个问题，而且标的只有几千个合约，体积可以忽略。

**period 是 `1d` / `1h` / `5m`**，和行情库的 `1d`/`1h`/`5m` 同一套叫法，方便
两边 join。
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import pandas as pd

from .paths import data_dir

SCHEMA_VERSION = "1"

DB_NAME = "option.duckdb"

# 期权 K 线的字段。天勤的 DataDownloader 输出是「每个合约一组 .open/.high/...」，
# 这里统一成长表 —— 宽表在 200 个合约时就是 1600 列，DuckDB 存这种表纯属自找麻烦。
BAR_COLS = ["contract", "period", "dt",
            "open", "high", "low", "close",
            "volume", "open_oi", "close_oi"]

IV_COLS = ["iv_near", "iv_next", "dte_near", "dte_next",
           "skew", "curv", "pcr", "n_opt"]

DDL = [
    """
    CREATE TABLE IF NOT EXISTS option_meta (
        contract    VARCHAR PRIMARY KEY,
        variety     VARCHAR,
        exchange    VARCHAR,
        opt_class   VARCHAR,
        strike      DOUBLE,
        expiry      DATE,
        underlying  VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS option_bars (
        contract    VARCHAR NOT NULL,
        period      VARCHAR NOT NULL,
        dt          TIMESTAMP NOT NULL,
        open        DOUBLE,
        high        DOUBLE,
        low         DOUBLE,
        close       DOUBLE,
        volume      BIGINT,
        open_oi     BIGINT,
        close_oi    BIGINT,
        PRIMARY KEY (contract, period, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS option_underlying (
        contract    VARCHAR NOT NULL,
        period      VARCHAR NOT NULL,
        dt          TIMESTAMP NOT NULL,
        close       DOUBLE,
        PRIMARY KEY (contract, period, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS option_iv (
        variety     VARCHAR NOT NULL,
        period      VARCHAR NOT NULL,
        dt          TIMESTAMP NOT NULL,
        iv_near     DOUBLE,
        iv_next     DOUBLE,
        dte_near    INTEGER,
        dte_next    INTEGER,
        skew        DOUBLE,
        curv        DOUBLE,
        pcr         DOUBLE,
        n_opt       INTEGER,
        PRIMARY KEY (variety, period, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS option_meta_kv (
        k VARCHAR PRIMARY KEY,
        v VARCHAR
    )
    """,
]


def db_path() -> str:
    """`TRENDDESK_OPT_DB` 可以指到别处（测试用临时库、或者放到空间更大的盘）。"""
    env = os.environ.get("TRENDDESK_OPT_DB")
    if env:
        return env
    return os.path.join(
        os.path.dirname(data_dir("cn").rstrip("/\\")), "data_tq", DB_NAME
    )


def exists() -> bool:
    return os.path.isfile(db_path())


@contextmanager
def connect(read_only: bool = False):
    import duckdb

    p = db_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    last = None
    for attempt in range(3 if read_only else 1):
        try:
            con = duckdb.connect(p, read_only=read_only)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < 2:
                time.sleep(0.4)
            else:
                raise
    else:  # pragma: no cover
        raise last
    try:
        if not read_only:
            for ddl in DDL:
                con.execute(ddl)
            con.execute(
                "INSERT OR REPLACE INTO option_meta_kv VALUES ('schema_version', ?)",
                [SCHEMA_VERSION],
            )
        yield con
    finally:
        con.close()


def _insert_df(con, table: str, df: pd.DataFrame) -> None:
    """列存批量写入。

    和 EDB 那边同一个坑：DuckDB 的 `executemany` 是逐行执行预处理语句，
    几十万行能跑几十分钟。`register` 之后走的是批量扫描，快两个数量级。
    """
    if df is None or len(df) == 0:
        return
    con.register("_opt_batch", df)
    try:
        cols = ", ".join(df.columns)
        con.execute(f"INSERT OR REPLACE INTO {table} SELECT {cols} FROM _opt_batch")
    finally:
        con.unregister("_opt_batch")


# ------------------------------------------------------------------ 写入


def upsert_meta(con, df: pd.DataFrame) -> int:
    """`df` 需要 contract/variety/exchange/opt_class/strike/expiry/underlying。"""
    if df is None or len(df) == 0:
        return 0
    d = df[["contract", "variety", "exchange", "opt_class",
            "strike", "expiry", "underlying"]].copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce").astype(float)
    d["expiry"] = pd.to_datetime(d["expiry"]).dt.date
    _insert_df(con, "option_meta", d)
    return len(d)


def upsert_bars(con, df: pd.DataFrame) -> int:
    if df is None or len(df) == 0:
        return 0
    d = df[BAR_COLS].copy()
    for c in ("volume", "open_oi", "close_oi"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype("int64")
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    d["dt"] = pd.to_datetime(d["dt"])
    d = d.dropna(subset=["close"])
    _insert_df(con, "option_bars", d)
    return len(d)


def upsert_underlying(con, df: pd.DataFrame) -> int:
    if df is None or len(df) == 0:
        return 0
    d = df[["contract", "period", "dt", "close"]].copy()
    d["close"] = pd.to_numeric(d["close"], errors="coerce").astype(float)
    d["dt"] = pd.to_datetime(d["dt"])
    d = d.dropna(subset=["close"])
    _insert_df(con, "option_underlying", d)
    return len(d)


def upsert_iv(con, df: pd.DataFrame) -> int:
    if df is None or len(df) == 0:
        return 0
    d = df[["variety", "period", "dt"] + IV_COLS].copy()
    d["dt"] = pd.to_datetime(d["dt"])
    d["dte_near"] = pd.to_numeric(d["dte_near"], errors="coerce").fillna(0).astype("int64")
    d["dte_next"] = pd.to_numeric(d["dte_next"], errors="coerce").fillna(0).astype("int64")
    d["n_opt"] = pd.to_numeric(d["n_opt"], errors="coerce").fillna(0).astype("int64")
    _insert_df(con, "option_iv", d)
    return len(d)


def set_meta(con, k: str, v: str) -> None:
    con.execute("INSERT OR REPLACE INTO option_meta_kv VALUES (?,?)", [k, str(v)])


# ------------------------------------------------------------------ 读取


def last_dt(con, contract: str, period: str):
    r = con.execute(
        "SELECT max(dt) FROM option_bars WHERE contract = ? AND period = ?",
        [contract, period],
    ).fetchone()
    return r[0] if r else None


def done_contracts(period: str) -> dict[str, pd.Timestamp]:
    """已经下到最后的合约 → 最后一根时间。断点续传靠它。"""
    if not exists():
        return {}
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT contract, max(dt) FROM option_bars WHERE period = ? GROUP BY 1",
            [period],
        ).fetchall()
    return {c: pd.Timestamp(d) for c, d in rows if d is not None}


def load_meta(con=None, varieties: list[str] | None = None) -> pd.DataFrame:
    if con is not None:
        sql = "SELECT * FROM option_meta"
        args = []
        if varieties:
            sql += " WHERE variety IN (" + ",".join("?" * len(varieties)) + ")"
            args = list(varieties)
        return con.execute(sql, args).df()
    if not exists():
        return pd.DataFrame()
    with connect(read_only=True) as c:
        return load_meta(c, varieties)


def load_bars(contracts: list[str], period: str) -> pd.DataFrame:
    if not contracts or not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        q = ",".join("?" * len(contracts))
        df = con.execute(
            f"SELECT * FROM option_bars WHERE period = ? AND contract IN ({q})"
            " ORDER BY dt, contract",
            [period] + list(contracts),
        ).df()
    return df


def load_underlying(contracts: list[str], period: str) -> pd.DataFrame:
    if not contracts or not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        q = ",".join("?" * len(contracts))
        return con.execute(
            f"SELECT * FROM option_underlying WHERE period = ? AND contract IN ({q})"
            " ORDER BY dt, contract",
            [period] + list(contracts),
        ).df()


def load_iv(varieties: list[str] | None = None, period: str = "1d",
            start=None, end=None) -> pd.DataFrame:
    """IV 摘要。列：variety, period, dt, iv_near, iv_next, dte_near, ..."""
    if not exists():
        return pd.DataFrame()
    where, args = ["period = ?"], [period]
    if varieties:
        where.append("variety IN (" + ",".join("?" * len(varieties)) + ")")
        args += list(varieties)
    if start is not None:
        where.append("dt >= ?")
        args.append(pd.Timestamp(start))
    if end is not None:
        where.append("dt <= ?")
        args.append(pd.Timestamp(end))
    with connect(read_only=True) as con:
        return con.execute(
            "SELECT * FROM option_iv WHERE " + " AND ".join(where) + " ORDER BY dt, variety",
            args,
        ).df()


def iv_wide(period: str = "1d", field: str = "iv_near") -> pd.DataFrame:
    """`index=dt, columns=variety` 的宽表，直接喂给因子计算。"""
    df = load_iv(period=period)
    if df.empty:
        return df
    return df.pivot_table(index="dt", columns="variety", values=field, aggfunc="last")


def stats() -> dict:
    if not exists():
        return {"exists": False, "contracts": 0, "bars": {}, "iv": {},
                "size_mb": 0.0}
    with connect(read_only=True) as con:
        contracts = con.execute("SELECT count(*) FROM option_meta").fetchone()[0]
        bars = dict(con.execute(
            "SELECT period, count(*) FROM option_bars GROUP BY 1 ORDER BY 1"
        ).fetchall())
        ivs = dict(con.execute(
            "SELECT period, count(*) FROM option_iv GROUP BY 1 ORDER BY 1"
        ).fetchall())
        first, last = con.execute(
            "SELECT min(dt), max(dt) FROM option_bars"
        ).fetchone()
    size = os.path.getsize(db_path()) / 1024 / 1024
    return {"exists": True, "contracts": int(contracts or 0),
            "bars": {k: int(v) for k, v in bars.items()},
            "iv": {k: int(v) for k, v in ivs.items()},
            "first": first, "last": last,
            "size_mb": round(size, 1)}


def varieties() -> list[str]:
    if not exists():
        return []
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT DISTINCT variety FROM option_meta ORDER BY 1"
        ).fetchall()
    return [r[0] for r in rows]


def chains(varieties: list[str] | None = None) -> dict[str, list[str]]:
    """`{链代码: [合约...]}`。链代码就是标的合约，如 `DCE.m1709`。"""
    df = load_meta(varieties=varieties)
    if df.empty:
        return {}
    out: dict[str, list[str]] = {}
    for und, g in df.groupby("underlying", sort=False):
        out[str(und)] = sorted(g["contract"].tolist())
    return out
