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

import json
import os
import time
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
    """仓库路径。`TRENDDESK_TQ_DB` 可以指向别处 —— 测试用临时库、或者把仓库放到
    空间更大的盘上，都不需要改代码。"""
    env = os.environ.get("TRENDDESK_TQ_DB")
    if env:
        return env
    return os.path.join(os.path.dirname(data_dir("cn").rstrip("/\\")), "data_tq", "tq.duckdb")


def exists() -> bool:
    return os.path.isfile(db_path())


@contextmanager
def connect(read_only: bool = False):
    """打开仓库。

    **DuckDB 的文件锁是排他的**：一个进程以读写方式打开后，别的进程连只读都打不开
    （报 `Cannot open file ... 另一个程序正在使用此文件`）。所以下载脚本在跑的时候，
    界面读不了库 —— 这不是配置问题，是单文件数据库的固有行为。

    因此只读打开会先重试几次（下载器是「一批一个写连接」，批次之间有空隙），
    实在拿不到就把错误往上抛，由界面提示「正在下载」，而不是假装没数据。
    写入仍然是单写多读：同一时刻只允许一个写连接，第二个写会拿到 Conflicting lock，
    这是预期行为，不要用重试掩盖。
    """
    import duckdb

    p = db_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    last = None
    for attempt in range(3 if read_only else 1):
        try:
            con = duckdb.connect(p, read_only=read_only)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == 0:
                time.sleep(0.4)
            else:
                raise
    else:  # pragma: no cover - 循环必然 break 或 raise
        raise last
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


def first_dt(symbol: str, exchange: str, period: str):
    """库里这个品种该周期最早的一根。

    增量更新只看 `last_dt` 是不够的：早期用短窗口（两年）下过的品种，之后每次
    都只补最新几根，前面缺的那几年永远补不上，而且日志显示「+14 行」看着像正常
    增量。RB 的 1h 就这样停在两年前。
    """
    if not exists():
        return None
    with connect(read_only=True) as con:
        r = con.execute(
            "SELECT min(dt) FROM bars WHERE symbol=? AND exchange=? AND period=?",
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


def set_adj_factor(con, symbol: str, exchange: str, period: str,
                   trading_day, factor) -> int:
    """按**交易日**整段重写 `adj_factor`，返回该品种该周期覆盖的 bar 数。

    为什么需要这个：`adj_factor` 是**派生数据**，完全由切换表决定，而切换表会变
    —— 补算、重算、每来一次新换月都会变。只在写 K 线那一刻算一次是不够的：后补的
    切换记录不会回头修已经落库的历史 bar。

    实测踩到过：78 个品种里 CJ、LH 两个整段因子停在 1.0（写库时它们的切换表还是
    空的），复权后换月处留着 3% 的假跳空，比不复权还难看；而 `write_main_many` 的
    增量只重写最近 3 天，历史段永远不会被修。

    传的是「交易日 → 因子」的映射而不是逐行 bar：同一个交易日内的所有 bar 因子相同
    （换月发生在夜盘 21:00，按交易日对齐才不会切在半天里），5m 十年也才两千多个
    交易日，UPDATE 的 join 表很小。
    """
    d = pd.DataFrame({
        "trading_day": pd.to_datetime(pd.Series(trading_day)).dt.date,
        "adj_factor": pd.to_numeric(pd.Series(factor), errors="coerce").astype(float),
    }).dropna()
    if d.empty:
        return 0
    # 同一交易日只留一行：UPDATE ... FROM 遇到重复匹配行时结果不确定
    d = d.drop_duplicates("trading_day")
    con.register("_tmp_adj", d)
    try:
        # `.fetchall()` 是必须的：DuckDB 的 execute 惰性求值，UPDATE 的结果不消费
        # 就会挂在连接上，下一条 execute 报 unfinished result。
        con.execute(
            "UPDATE bars SET adj_factor = _tmp_adj.adj_factor FROM _tmp_adj"
            " WHERE bars.symbol=? AND bars.exchange=? AND bars.period=?"
            " AND bars.trading_day = _tmp_adj.trading_day",
            [symbol, exchange, period],
        ).fetchall()
    finally:
        con.unregister("_tmp_adj")
    # 必须 fetchall 而不是 fetchone：只取一行不会把结果集关掉，连接仍处于
    # 「有未完成结果」的状态，调用方下一条 execute/register 直接报
    # `ResourceInUseException`（下载跑到第二个品种就崩，报的是 upsert 里那行 register）。
    r = con.execute(
        "SELECT count(*) FROM bars WHERE symbol=? AND exchange=? AND period=?",
        [symbol, exchange, period],
    ).fetchall()
    return int(r[0][0]) if r else 0


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


def periods_of(symbol: str, exchange: str, con=None) -> set[str]:
    """这个品种在库里已经落了哪些源周期。

    界面每换一个品种要判断 6 个周期有没有数据，逐个调 `bar_count()` 就是 6 次
    开库查询；这里一次查完。注意返回的是**源周期**（5m / 1h），15m/2h 这些
    是本地合并出来的，由 `tqdata` 负责映射。

    传入 `con` 就在这个连接上查 —— 批量重算因子的循环自己持着写连接，这时候
    再开一个只读连接会撞上 DuckDB 的排他锁。
    """
    if con is not None:
        rows = con.execute(
            "SELECT DISTINCT period FROM bars WHERE symbol=? AND exchange=?",
            [symbol, exchange],
        ).fetchall()
        return {r[0] for r in rows}
    if not exists():
        return set()
    try:
        with connect(read_only=True) as conn:
            rows = conn.execute(
                "SELECT DISTINCT period FROM bars WHERE symbol=? AND exchange=?",
                [symbol, exchange],
            ).fetchall()
    except Exception:  # noqa: BLE001
        # 下载脚本正持锁时拿不到。界面据此标「未下载」会短暂误报，但比抛异常好 ——
        # 下一次刷新就恢复了，而异常会让整张图消失。
        return set()
    return {r[0] for r in rows}


def contract_last_dt(symbol: str, exchange: str, period: str) -> dict[str, pd.Timestamp]:
    """每个真实合约在库里已有的最后一根时间，用于增量跳过。

    没有这个，每次重跑都要把几百个合约从头下一遍 —— 真实合约十年是几千个
    合约、上亿行，重下要几小时。
    """
    if not exists():
        return {}
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT contract, max(dt) FROM contract_bars"
            " WHERE symbol=? AND exchange=? AND period=? GROUP BY contract",
            [symbol, exchange, period],
        ).fetchall()
    return {r[0]: pd.Timestamp(r[1]) for r in rows if r[1] is not None}


def meta_get(k: str, default=None):
    """读 meta 里的一个键。JSON 能解析就返回对象，否则返回原字符串。"""
    if not exists():
        return default
    with connect(read_only=True) as con:
        r = con.execute("SELECT v FROM meta WHERE k=?", [k]).fetchone()
    if r is None or r[0] is None:
        return default
    try:
        return json.loads(r[0])
    except (ValueError, TypeError):
        return r[0]


def meta_set(k: str, v) -> None:
    """写 meta。dict / list 自动转 JSON，省得调用方到处 dumps。"""
    val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", [k, val])


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
