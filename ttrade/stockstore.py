"""A股股票仓库（DuckDB，独立文件）。

**为什么单独一个库**：和期货/期权同理——股票的合约体系、数据维度和期货完全不同
（5000+ 只股票 vs 100 个品种，股票有基本面/概念板块/资金流等期货没有的数据），
塞进 `tq.duckdb` 只会让那个库再翻几倍。独立库也方便「不想研究股票就不下」。

**天勤 vs Tushare 分表**：两个数据源的 K 线口径不同（天勤 close vs Tushare 收盘价），
混在一张表里会出错。天勤 K 线存 `stock_bars_tq`，Tushare 日线存 `stock_bars_ts`，
各有各的复权因子表。

表结构：

    stock_meta        股票代码：ts_code / name / exchange(SSE/SZSE) / board / industry / list_date / delist_date
    stock_bars_tq     天勤 K 线：ts_code / period(1d/1h/5m) / dt / OHLCV
    stock_bars_ts     Tushare 日线：ts_code / dt / OHLCV / amount / pct_chg
    stock_adj         复权因子（Tushare）：ts_code / dt / adj_factor
    stock_basic_ts    Tushare 基本面：ts_code / dt / pe / pb / total_mv / circ_mv / turnover_rate / dv_ratio
    stock_concept     同花顺概念板块：concept_name / ts_code / in_date
    stock_moneyflow   资金流（Tushare）：ts_code / dt / buy_sm_vol / ... / net_mf_amount
    stock_meta_kv     键值表（schema 版本等）

**period 是 `1d` / `1h` / `5m`**，和行情库的叫法一致，方便跨表 join。

**复权**：天勤支持实时复权（请求时指定 "F"/"B"），Tushare 需要手动用 `adj_factor`
算前复权价。`load_bars()` 返回时附上 `adj_factor` 列，下游用 `auxdata.attach()` 的
同一套机制。天勤和 Tushare 的复权因子**不通用**（基准日不同），所以分两张表。
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import pandas as pd

from .paths import data_dir

SCHEMA_VERSION = "1"

DB_NAME = "stock.duckdb"

# 天勤 K 线字段。和期权一样统一成长表——宽表在 5000 个股票时就是 4 万列
BAR_COLS_TQ = ["ts_code", "period", "dt",
               "open", "high", "low", "close",
               "volume", "amount"]

# Tushare 日线字段。比天勤多了 amount（成交额）和 pct_chg（涨跌幅）
BAR_COLS_TS = ["ts_code", "dt",
               "open", "high", "low", "close",
               "volume", "amount", "pct_chg", "change"]

# 基本面字段（Tushare daily_basic）
BASIC_COLS = ["ts_code", "dt",
              "pe", "pe_ttm", "pb", "ps", "ps_ttm",
              "total_mv", "circ_mv", "turnover_rate", "turnover_rate_f",
              "dv_ratio", "dv_ttm"]

# 资金流字段（Tushare moneyflow）
MONEYFLOW_COLS = ["ts_code", "dt",
                  "buy_sm_vol", "sell_sm_vol", "buy_md_vol", "sell_md_vol",
                  "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol",
                  "net_mf_amount"]

DDL = [
    """
    CREATE TABLE IF NOT EXISTS stock_meta (
        ts_code     VARCHAR PRIMARY KEY,
        name        VARCHAR,
        exchange    VARCHAR,
        board       VARCHAR,
        industry    VARCHAR,
        list_date   DATE,
        delist_date DATE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_bars_tq (
        ts_code     VARCHAR NOT NULL,
        period      VARCHAR NOT NULL,
        dt          TIMESTAMP NOT NULL,
        open        DOUBLE,
        high        DOUBLE,
        low         DOUBLE,
        close       DOUBLE,
        volume      BIGINT,
        amount      DOUBLE,
        PRIMARY KEY (ts_code, period, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_bars_ts (
        ts_code     VARCHAR NOT NULL,
        dt          DATE NOT NULL,
        open        DOUBLE,
        high        DOUBLE,
        low         DOUBLE,
        close       DOUBLE,
        volume      BIGINT,
        amount      DOUBLE,
        pct_chg     DOUBLE,
        change      DOUBLE,
        PRIMARY KEY (ts_code, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_adj (
        ts_code     VARCHAR NOT NULL,
        dt          DATE NOT NULL,
        adj_factor  DOUBLE,
        PRIMARY KEY (ts_code, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_basic_ts (
        ts_code         VARCHAR NOT NULL,
        dt              DATE NOT NULL,
        pe              DOUBLE,
        pe_ttm          DOUBLE,
        pb              DOUBLE,
        ps              DOUBLE,
        ps_ttm          DOUBLE,
        total_mv        DOUBLE,
        circ_mv         DOUBLE,
        turnover_rate   DOUBLE,
        turnover_rate_f DOUBLE,
        dv_ratio        DOUBLE,
        dv_ttm          DOUBLE,
        PRIMARY KEY (ts_code, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_concept (
        concept_name    VARCHAR NOT NULL,
        ts_code         VARCHAR NOT NULL,
        in_date         DATE,
        PRIMARY KEY (concept_name, ts_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_moneyflow (
        ts_code         VARCHAR NOT NULL,
        dt              DATE NOT NULL,
        buy_sm_vol      DOUBLE,
        sell_sm_vol     DOUBLE,
        buy_md_vol      DOUBLE,
        sell_md_vol     DOUBLE,
        buy_lg_vol      DOUBLE,
        sell_lg_vol     DOUBLE,
        buy_elg_vol     DOUBLE,
        sell_elg_vol    DOUBLE,
        net_mf_amount   DOUBLE,
        PRIMARY KEY (ts_code, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_meta_kv (
        k VARCHAR PRIMARY KEY,
        v VARCHAR
    )
    """,
]


def db_path() -> str:
    """`TRENDDESK_STOCK_DB` 可以指到别处（测试用临时库、或者放到空间更大的盘）。"""
    env = os.environ.get("TRENDDESK_STOCK_DB")
    if env:
        return env
    return os.path.join(
        os.path.dirname(data_dir("cn").rstrip("/\\")), "data_tq", DB_NAME
    )


def exists() -> bool:
    return os.path.isfile(db_path())


@contextmanager
def connect(read_only: bool = False):
    """连接股票库。读写锁策略和 optstore 一致：DuckDB 一写多读。"""
    import duckdb

    p = db_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    attempts, delay = (3, 0.4) if read_only else (6, 2.0)
    last = None
    con = None
    for i in range(attempts):
        try:
            con = duckdb.connect(p, read_only=read_only)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < attempts - 1:
                time.sleep(delay)
    if con is None:
        raise IOError(
            f"打不开股票库（{'读' if read_only else '写'}）—— 另一个程序正占着它："
            f"{last}"
        ) from last
    try:
        if not read_only:
            for ddl in DDL:
                con.execute(ddl)
            con.execute(
                "INSERT OR REPLACE INTO stock_meta_kv VALUES ('schema_version', ?)",
                [SCHEMA_VERSION],
            )
        yield con
    finally:
        con.close()


def _insert_df(con, table: str, df: pd.DataFrame) -> None:
    """列存批量写入。和 optstore / edb 同一个模式。

    DuckDB 的 `executemany` 逐行执行，几万行能跑几分钟。
    `register` 之后走批量扫描，快两个数量级。
    """
    if df is None or len(df) == 0:
        return
    con.register("_stock_batch", df)
    try:
        cols = ", ".join(df.columns)
        con.execute(f"INSERT OR REPLACE INTO {table} SELECT {cols} FROM _stock_batch")
    finally:
        con.unregister("_stock_batch")


# ------------------------------------------------------------------ 写入

def upsert_meta(con, df: pd.DataFrame) -> int:
    """`df` 需要 ts_code/name/exchange/board/industry/list_date/delist_date。"""
    if df is None or len(df) == 0:
        return 0
    d = df[["ts_code", "name", "exchange", "board", "industry",
            "list_date", "delist_date"]].copy()
    d["ts_code"] = d["ts_code"].astype(str)
    d["name"] = d["name"].astype(str)
    d["exchange"] = d["exchange"].astype(str)
    d["board"] = d["board"].fillna("").astype(str)
    d["industry"] = d["industry"].fillna("").astype(str)
    d["list_date"] = pd.to_datetime(d["list_date"], errors="coerce").dt.date
    d["delist_date"] = pd.to_datetime(d["delist_date"], errors="coerce").dt.date
    _insert_df(con, "stock_meta", d)
    return len(d)


def upsert_bars_tq(con, df: pd.DataFrame) -> int:
    """天勤 K 线。列：ts_code/period/dt/OHLCV/amount。"""
    if df is None or len(df) == 0:
        return 0
    d = df[BAR_COLS_TQ].copy()
    for c in ("open", "high", "low", "close", "amount"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0).astype("int64")
    d["dt"] = pd.to_datetime(d["dt"])
    d["ts_code"] = d["ts_code"].astype(str)
    d["period"] = d["period"].astype(str)
    d = d.dropna(subset=["close"])
    _insert_df(con, "stock_bars_tq", d)
    return len(d)


def upsert_bars_ts(con, df: pd.DataFrame) -> int:
    """Tushare 日线。列：ts_code/dt/OHLCV/amount/pct_chg/change。"""
    if df is None or len(df) == 0:
        return 0
    d = df[BAR_COLS_TS].copy()
    for c in ("open", "high", "low", "close", "amount", "pct_chg", "change"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0).astype("int64")
    d["dt"] = pd.to_datetime(d["dt"]).dt.date
    d["ts_code"] = d["ts_code"].astype(str)
    d = d.dropna(subset=["close"])
    _insert_df(con, "stock_bars_ts", d)
    return len(d)


def upsert_adj(con, df: pd.DataFrame) -> int:
    """复权因子（Tushare）。列：ts_code/dt/adj_factor。"""
    if df is None or len(df) == 0:
        return 0
    d = df[["ts_code", "dt", "adj_factor"]].copy()
    d["adj_factor"] = pd.to_numeric(d["adj_factor"], errors="coerce").astype(float)
    d["dt"] = pd.to_datetime(d["dt"]).dt.date
    d["ts_code"] = d["ts_code"].astype(str)
    d = d.dropna(subset=["adj_factor"])
    _insert_df(con, "stock_adj", d)
    return len(d)


def upsert_basic(con, df: pd.DataFrame) -> int:
    """基本面（Tushare daily_basic）。"""
    if df is None or len(df) == 0:
        return 0
    cols = [c for c in BASIC_COLS if c in df.columns]
    d = df[cols].copy()
    for c in cols:
        if c not in ("ts_code", "dt"):
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    d["dt"] = pd.to_datetime(d["dt"]).dt.date
    d["ts_code"] = d["ts_code"].astype(str)
    # 补齐缺失列
    for c in BASIC_COLS:
        if c not in d.columns:
            d[c] = None
    d = d[BASIC_COLS]
    _insert_df(con, "stock_basic_ts", d)
    return len(d)


def upsert_moneyflow(con, df: pd.DataFrame) -> int:
    """资金流（Tushare moneyflow）。"""
    if df is None or len(df) == 0:
        return 0
    cols = [c for c in MONEYFLOW_COLS if c in df.columns]
    d = df[cols].copy()
    for c in cols:
        if c not in ("ts_code", "dt"):
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    d["dt"] = pd.to_datetime(d["dt"]).dt.date
    d["ts_code"] = d["ts_code"].astype(str)
    for c in MONEYFLOW_COLS:
        if c not in d.columns:
            d[c] = None
    d = d[MONEYFLOW_COLS]
    _insert_df(con, "stock_moneyflow", d)
    return len(d)


def upsert_concept(con, df: pd.DataFrame) -> int:
    """同花顺概念板块。列：concept_name/ts_code/in_date。"""
    if df is None or len(df) == 0:
        return 0
    d = df[["concept_name", "ts_code", "in_date"]].copy()
    d["concept_name"] = d["concept_name"].astype(str)
    d["ts_code"] = d["ts_code"].astype(str)
    d["in_date"] = pd.to_datetime(d["in_date"], errors="coerce").dt.date
    _insert_df(con, "stock_concept", d)
    return len(d)


def set_meta(con, k: str, v: str) -> None:
    con.execute("INSERT OR REPLACE INTO stock_meta_kv VALUES (?,?)", [k, str(v)])


# ------------------------------------------------------------------ 读取

def last_dt_tq(con, ts_code: str, period: str):
    """天勤 K 线的最后日期（断点续传用）。"""
    r = con.execute(
        "SELECT max(dt) FROM stock_bars_tq WHERE ts_code = ? AND period = ?",
        [ts_code, period],
    ).fetchone()
    return r[0] if r else None


def last_dt_ts(con, ts_code: str):
    """Tushare 日线的最后日期。"""
    r = con.execute(
        "SELECT max(dt) FROM stock_bars_ts WHERE ts_code = ?",
        [ts_code],
    ).fetchone()
    return r[0] if r else None


def done_codes_tq(period: str) -> dict[str, pd.Timestamp]:
    """天勤已下载完的股票 → 最后一根时间。断点续传靠它。"""
    if not exists():
        return {}
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT ts_code, max(dt) FROM stock_bars_tq WHERE period = ? GROUP BY 1",
            [period],
        ).fetchall()
    return {c: pd.Timestamp(d) for c, d in rows if d is not None}


def done_codes_ts() -> dict[str, pd.Timestamp]:
    """Tushare 已下载完的股票 → 最后一根日期。"""
    if not exists():
        return {}
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT ts_code, max(dt) FROM stock_bars_ts GROUP BY 1"
        ).fetchall()
    return {c: pd.Timestamp(d) for c, d in rows if d is not None}


def load_meta(con=None, exchanges: list[str] | None = None) -> pd.DataFrame:
    """股票代码表。"""
    if con is not None:
        sql = "SELECT * FROM stock_meta"
        args = []
        if exchanges:
            sql += " WHERE exchange IN (" + ",".join("?" * len(exchanges)) + ")"
            args = list(exchanges)
        return con.execute(sql, args).df()
    if not exists():
        return pd.DataFrame()
    with connect(read_only=True) as c:
        return load_meta(c, exchanges)


def load_bars_tq(ts_codes: list[str], period: str) -> pd.DataFrame:
    """天勤 K 线。返回长表（ts_code/period/dt/OHLCV/amount）。"""
    if not ts_codes or not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        q = ",".join("?" * len(ts_codes))
        return con.execute(
            f"SELECT * FROM stock_bars_tq WHERE period = ? AND ts_code IN ({q})"
            " ORDER BY dt, ts_code",
            [period] + list(ts_codes),
        ).df()


def load_bars_ts(ts_codes: list[str]) -> pd.DataFrame:
    """Tushare 日线。返回长表。"""
    if not ts_codes or not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        q = ",".join("?" * len(ts_codes))
        return con.execute(
            f"SELECT * FROM stock_bars_ts WHERE ts_code IN ({q})"
            " ORDER BY dt, ts_code",
            list(ts_codes),
        ).df()


def load_adj(ts_code: str) -> pd.DataFrame:
    """单只股票的复权因子。"""
    if not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        return con.execute(
            "SELECT * FROM stock_adj WHERE ts_code = ? ORDER BY dt",
            [ts_code],
        ).df()


def load_basic(ts_code: str) -> pd.DataFrame:
    """单只股票的基本面。"""
    if not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        return con.execute(
            "SELECT * FROM stock_basic_ts WHERE ts_code = ? ORDER BY dt",
            [ts_code],
        ).df()


def load_moneyflow(ts_code: str) -> pd.DataFrame:
    """单只股票的资金流。"""
    if not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        return con.execute(
            "SELECT * FROM stock_moneyflow WHERE ts_code = ? ORDER BY dt",
            [ts_code],
        ).df()


def load_concept(concept_name: str | None = None) -> pd.DataFrame:
    """概念板块。不传 concept_name 就返回全部。"""
    if not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        if concept_name:
            return con.execute(
                "SELECT * FROM stock_concept WHERE concept_name = ? ORDER BY ts_code",
                [concept_name],
            ).df()
        return con.execute(
            "SELECT * FROM stock_concept ORDER BY concept_name, ts_code"
        ).df()


def stock_list(exchange: str | None = None) -> list[str]:
    """库里有的股票代码列表。"""
    if not exists():
        return []
    with connect(read_only=True) as con:
        if exchange:
            rows = con.execute(
                "SELECT ts_code FROM stock_meta WHERE exchange = ? ORDER BY 1",
                [exchange],
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT ts_code FROM stock_meta ORDER BY 1"
            ).fetchall()
    return [r[0] for r in rows]


def bars_wide(ts_code: str, period: str = "1d", source: str = "tq") -> pd.DataFrame:
    """单只股票的 K 线，index=dt，方便喂给因子计算。

    `source`：`"tq"` 天勤 / `"ts"` Tushare（只有日线）。
    天勤 K 线返回 OHLCV+amount；Tushare 日线返回 OHLCV+amount+pct_chg。
    """
    if source == "ts":
        df = load_bars_ts([ts_code])
    else:
        df = load_bars_tq([ts_code], period)
    if df.empty:
        return df
    df = df.sort_values("dt")
    if "dt" in df.columns:
        df = df.set_index("dt")
    return df


def stats() -> dict:
    """库的统计信息。"""
    if not exists():
        return {"exists": False, "stocks": 0, "bars_tq": {}, "bars_ts": 0,
                "adj": 0, "basic": 0, "moneyflow": 0, "concept": 0,
                "size_mb": 0.0}
    with connect(read_only=True) as con:
        stocks = con.execute("SELECT count(*) FROM stock_meta").fetchone()[0]
        bars_tq = dict(con.execute(
            "SELECT period, count(*) FROM stock_bars_tq GROUP BY 1 ORDER BY 1"
        ).fetchall())
        bars_ts = con.execute("SELECT count(*) FROM stock_bars_ts").fetchone()[0]
        adj = con.execute("SELECT count(*) FROM stock_adj").fetchone()[0]
        basic = con.execute("SELECT count(*) FROM stock_basic_ts").fetchone()[0]
        mf = con.execute("SELECT count(*) FROM stock_moneyflow").fetchone()[0]
        concept = con.execute("SELECT count(*) FROM stock_concept").fetchone()[0]
        first, last = con.execute(
            "SELECT min(dt), max(dt) FROM stock_bars_tq"
        ).fetchone()
    size = os.path.getsize(db_path()) / 1024 / 1024
    return {"exists": True, "stocks": int(stocks or 0),
            "bars_tq": {k: int(v) for k, v in bars_tq.items()},
            "bars_ts": int(bars_ts or 0), "adj": int(adj or 0),
            "basic": int(basic or 0), "moneyflow": int(mf or 0),
            "concept": int(concept or 0),
            "first": first, "last": last,
            "size_mb": round(size, 1)}


def exchanges() -> list[str]:
    """库里有数据的交易所列表。"""
    if not exists():
        return []
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT DISTINCT exchange FROM stock_meta ORDER BY 1"
        ).fetchall()
    return [r[0] for r in rows if r[0]]


# ------------------------------------------------------------------ GUI 便捷封装
# 上面的 load_* 函数返回的是长表（带 ts_code 列、支持多只），适合批量分析。
# GUI 一次只看一只股票，需要的是单只的宽表 DataFrame（dt/open/high/low/close/
# volume），和期货/美股的格式一致。以下封装做这层转换。


def load_one_tq(ts_code: str, period: str) -> pd.DataFrame:
    """单只股票的天勤 K 线，返回宽表（dt/open/high/low/close/volume/amount）。"""
    if not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        return con.execute(
            "SELECT dt, open, high, low, close, volume, amount"
            " FROM stock_bars_tq"
            " WHERE ts_code = ? AND period = ?"
            " ORDER BY dt",
            [ts_code, period],
        ).df()


def load_one_ts(ts_code: str) -> pd.DataFrame:
    """单只股票的 Tushare 日线，返回宽表。"""
    if not exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        return con.execute(
            "SELECT dt, open, high, low, close, volume, amount, pct_chg"
            " FROM stock_bars_ts"
            " WHERE ts_code = ?"
            " ORDER BY dt",
            [ts_code],
        ).df()


def load_one_adj(ts_code: str) -> pd.Series:
    """单只股票的复权因子，返回以 dt 为 index 的 Series。"""
    if not exists():
        return pd.Series(dtype=float)
    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT dt, adj_factor FROM stock_adj"
            " WHERE ts_code = ? ORDER BY dt",
            [ts_code],
        ).df()
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("dt")["adj_factor"].astype(float)
    s.index = pd.to_datetime(s.index)
    return s


def available_periods(ts_code: str) -> list[str]:
    """该股票在库里有哪种周期的天勤数据。"""
    if not exists():
        return []
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT DISTINCT period FROM stock_bars_tq"
            " WHERE ts_code = ? ORDER BY 1",
            [ts_code],
        ).fetchall()
    return [r[0] for r in rows if r[0]]
