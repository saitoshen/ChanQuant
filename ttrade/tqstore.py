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

**两个库**：安装包自带 `tq_lite.duckdb`（约 270MB：主连 5m/1h/1d + 真实合约日线 +
切换表），完整库 `tq.duckdb`（约 11GB，多出 5m/1h 的真实合约明细）由用户按需下载。
读取时**优先完整库**，没有再退到轻量库；**写入永远写完整库** —— 随包的轻量库可能
落在只读目录（macOS 的 `.app`），也不该被增量更新改坏。轻量库缺 `contract_bars`
的分钟级明细，所以 `stats()` / `load_contracts()` 都按「表可能不存在」来写。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

import pandas as pd

from .paths import data_dir

SCHEMA_VERSION = "1"

DB_NAME = "tq.duckdb"          # 完整库：主连 + 真实合约（5m/1h/1d）
LITE_NAME = "tq_lite.duckdb"   # 随包轻量库：主连三周期 + 真实合约日线

BAR_COLS = [
    "symbol", "exchange", "period", "dt", "trading_day",
    "open", "high", "low", "close",
    "volume", "open_oi", "close_oi", "adj_factor",
]

# `all_contracts()` 的进程内缓存：周期 → (键, DataFrame)。键里带库文件 mtime，
# 补完库自动失效，不需要谁记着来清。
_CON_CACHE: dict = {}

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


def _db_dir() -> str:
    return os.path.join(os.path.dirname(data_dir("cn").rstrip("/\\")), "data_tq")


def db_path(read_only: bool = True) -> str:
    """仓库路径。

    `TRENDDESK_TQ_DB` 可以指向别处 —— 测试用临时库、或者把仓库放到空间更大的盘上，
    都不需要改代码。设了它就不再走轻量库回退（指向谁就是谁）。

    没设时：**写**永远落到完整库；**读**先看完整库在不在，没有才退到随包的
    `tq_lite.duckdb`。顺序反过来会让用户下载的完整库被忽略 —— 那个文件 11GB，
    下完却还在读 270MB 的轻量库，等于白下。
    """
    env = os.environ.get("TRENDDESK_TQ_DB")
    if env:
        return env
    base = _db_dir()
    full = os.path.join(base, DB_NAME)
    if not read_only:
        return full
    if os.path.isfile(full):
        return full
    lite = os.path.join(base, LITE_NAME)
    if os.path.isfile(lite):
        return lite
    return full


def source() -> str:
    """当前读的是哪个库：`full` 完整库 / `lite` 随包轻量库 / `none` 都没有。"""
    p = db_path(read_only=True)
    if not os.path.isfile(p):
        return "none"
    return "lite" if os.path.basename(p) == LITE_NAME else "full"


def exists() -> bool:
    return os.path.isfile(db_path())


def _seed_full_from_lite(full: str) -> None:
    """第一次写完整库时，先把随包的轻量库搬过去当底子。

    不搬会出事：用户装完就有十年主连（读的是轻量库），点一次「补近 1 天」会新建
    一个只有这一天数据的 `tq.duckdb` —— 而读取优先完整库，十年历史于是凭空消失，
    界面上只剩一根 K 线。

    **不能直接拷文件**：随包的轻量库是不建主键的（带主键 1.2GB，查询还更慢），
    而写库全靠 `INSERT OR REPLACE`，没有唯一约束直接抛 BinderException。所以这里
    按 DDL 建一张带主键的空表再灌数据 —— 一次性约 28 秒，只在「第一次往轻量库上
    补数据」时发生。

    先写 `.seed` 再 `os.replace`：中途被打断时不会留下一个半截的 tq.duckdb 被下次
    当成正常库打开。
    """
    lite = os.path.join(os.path.dirname(full), LITE_NAME)
    if os.path.exists(full) or not os.path.isfile(lite):
        return
    import duckdb

    tmp = full + ".seed"
    for p in (tmp, tmp + ".wal"):
        if os.path.exists(p):
            os.remove(p)
    con = duckdb.connect(tmp)
    try:
        con.execute(f"ATTACH '{lite}' AS lite (READ_ONLY)")
        have = {
            r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables"
                " WHERE table_catalog = 'lite'"
            ).fetchall()
        }
        for ddl in DDL:
            con.execute(ddl)
        # 🔴 不能一条 INSERT 把 bars（打包版 1079 万行）整表搬过去：DuckDB 会把
        # 整张表一次性物化，打包版首次写库直接 `Failed to commit: Allocation
        # failure`。按 rowid 分段拷贝，内存占用 bounded，失败也不会留下半份库。
        step = 200_000
        for t in ("bars", "contract_bars", "switch", "meta"):
            if t not in have:
                continue
            n = int(con.execute(f"SELECT count(*) FROM lite.{t}").fetchone()[0])
            for lo in range(0, n, step):
                con.execute(
                    f"INSERT INTO {t} SELECT * FROM lite.{t}"
                    " WHERE rowid >= ? AND rowid < ?", [lo, lo + step])
        con.execute("DETACH lite")
    except Exception:
        con.close()
        for p in (tmp, tmp + ".wal"):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise
    else:
        con.close()
    os.replace(tmp, full)


def _blocked_by_reader(exc: BaseException) -> bool:
    """这个异常是不是「被别的**读**连接挡住」（Windows 共享冲突），而不是写写冲突。

    DuckDB / Windows 的文案跟着版本和语言变，这里放宽到几个关键片段：
    中文系统是「另一个程序正在使用此文件」，英文是 `being used by another process`；
    `Conflicting lock` 是**写写**冲突（第二个下载器），必须原样抛出去，不能重试。
    """
    msg = str(exc)
    if "Conflicting lock" in msg or "conflicting" in msg.lower():
        return False
    for needle in ("另一个程序正在使用此文件", "正由另一进程使用",
                   "being used by another process", "used by another process",
                   "另一个程序正在使用", "sharing violation"):
        if needle.lower() in msg.lower():
            return True
    return False


@contextmanager
def connect(read_only: bool = False):
    """打开仓库。

    **DuckDB 的文件锁是排他的**：一个进程以读写方式打开后，别的进程连只读都打不开
    （报 `Cannot open file ... 另一个程序正在使用此文件`）。所以下载脚本在跑的时候，
    界面读不了库 —— 这不是配置问题，是单文件数据库的固有行为。

    因此只读打开会先重试几次（下载器是「一批一个写连接」，批次之间有空隙），
    实在拿不到就把错误往上抛，由界面提示「正在下载」，而不是假装没数据。

    **写入也要重试，但只重试「被界面读连接短暂挡住」这一种**：界面是按查询
    open/close 的，两个查询之间有几毫秒的空隙；定时补库恰好落在这个空隙里就会
    拿到 `另一个程序正在使用此文件`（2026-11-19 现场：GUI 开着时补 5m，写到第 9
    个品种就断了，前 8 个已入库 —— 看着像「补了一半」，实际是被一次读挡了一下）。
    这种重试几秒就能过。真正的 `Conflicting lock`（另一个**写**连接）仍然不重试：
    那说明确实有第二个下载器在跑，掩盖它只会让两边互相踩。
    """
    import duckdb

    p = db_path(read_only=read_only)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not read_only and not os.environ.get("TRENDDESK_TQ_DB"):
        _seed_full_from_lite(p)
    last = None
    tries = 3 if read_only else 12
    for attempt in range(tries):
        try:
            con = duckdb.connect(p, read_only=read_only)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if read_only:
                if attempt == 0:
                    time.sleep(0.4)
                else:
                    raise
                continue
            if not _blocked_by_reader(exc) or attempt == tries - 1:
                raise
            time.sleep(0.5)          # 最多等 5.5 秒，够跨过界面的查询间隙
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


# 交易日历（进程级缓存）。日历来自**日线表**：全库 1d 只有十几万行，扫一遍
# 几十毫秒，比给 5m 的九百万行建索引便宜得多。日线一个交易日一根，任一品种
# 都在，所以并起来就是整张交易所日历 —— 节假日自然不在里面。
_CAL_CACHE: dict = {"days": None, "last": None}


def calendar(refresh: bool = False) -> set:
    """全库出现过的**交易日**集合（`datetime.date`）。

    用来判断「某段 K 线是不是缺了几天」。不能拿自然日算：春节、国庆一周不开市，
    按自然日会把正常的长假当成数据缺口，然后天天报错。

    `1d` 表为空时退回空集合 —— 调用方要按「不知道日历」处理，别把整段判成缺。
    """
    today = pd.Timestamp.now().normalize().date()
    if (not refresh and _CAL_CACHE["days"] is not None
            and _CAL_CACHE["last"] == today):
        return _CAL_CACHE["days"]
    days: set = set()
    if exists():
        try:
            with connect(read_only=True) as con:
                r = con.execute(
                    "SELECT DISTINCT trading_day FROM bars "
                    "WHERE period='1d' AND trading_day IS NOT NULL"
                ).df()
            if not r.empty:
                days = {pd.Timestamp(x).date() for x in r["trading_day"]}
        except Exception:  # noqa: BLE001
            days = set()
    if days:
        _CAL_CACHE["days"], _CAL_CACHE["last"] = days, today
    return days


def coverage(period: str) -> dict:
    """每个「品种_交易所」在该周期的覆盖情况：`(首日, 末日, 有数据的交易日数)`。

    一次扫表分组，**别逐品种查** —— 5m 表九百万行，78 个品种各查一次要几十秒；
    这条按 `period` 过滤后一次聚合，实测 0.5 秒。

    存在的理由：**「补了多少行」不能证明「补全了」**。增量只要按库尾往后接，
    中间断的那几天就永远不会被碰，而日志每天照样显示「+14 行」。
    """
    if not exists():
        return {}
    out: dict = {}
    try:
        with connect(read_only=True) as con:
            r = con.execute(
                "SELECT symbol, exchange, min(trading_day) AS lo, "
                "       max(trading_day) AS hi, "
                "       COUNT(DISTINCT trading_day) AS n "
                "FROM bars WHERE period=? AND trading_day IS NOT NULL "
                "GROUP BY 1, 2",
                [period],
            ).df()
    except Exception:  # noqa: BLE001
        return {}
    for _, row in r.iterrows():
        if row["lo"] is None or row["hi"] is None:
            continue
        out[(row["symbol"], row["exchange"])] = (
            pd.Timestamp(row["lo"]).date(),
            pd.Timestamp(row["hi"]).date(),
            int(row["n"]),
        )
    return out


def miss_days(symbol: str, exchange: str, period: str, cal: set | None = None) -> list:
    """这个品种该周期的 K 线里**缺掉的交易日**（首尾之间，拿日线日历比）。

    2026-11-17 的现场：IF 的 5m 停在 11-10，每日定时任务按「近 1 天」增量，
    中间 11-11~11-13 再也不碰 —— 策略把实时段直接贴在 11-10 后面算指标，
    信号与回测对不上，账户里有仓的被平掉，而回测里还「持着」。

    只抓**整天缺失**：一天的 bar 只少几根是半天市或停牌，不是数据缺口。
    """
    if not exists():
        return []
    try:
        with connect(read_only=True) as con:
            r = con.execute(
                "SELECT DISTINCT trading_day FROM bars "
                "WHERE symbol=? AND exchange=? AND period=? AND trading_day IS NOT NULL",
                [symbol, exchange, period],
            ).df()
    except Exception:  # noqa: BLE001
        return []
    if r.empty:
        return []
    have = {pd.Timestamp(x).date() for x in r["trading_day"]}
    cal = calendar() if cal is None else cal
    if not cal:
        return []
    lo, hi = min(have), max(have)
    hi = min(hi, max(cal))
    return sorted(d for d in cal if lo <= d <= hi and d not in have)


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


def list_contracts(symbol: str, exchange: str, period: str) -> pd.DataFrame:
    """库里有哪些真实合约，各自的起止时间和根数。

    界面上的「具体合约」下拉就读这个 —— 只列**库里有数据的**，没下过的合约
    摆出来点了也是空图。列：contract / first / last / rows。

    下载器正持锁时返回空表（和 `periods_of` 同一个理由：让界面显示「未下载」
    而不是抛异常把整张图弄没）。
    """
    empty = pd.DataFrame(columns=["contract", "first", "last", "rows"])
    if not exists():
        return empty
    try:
        with connect(read_only=True) as con:
            if not _has_table(con, "contract_bars"):
                return empty
            df = con.execute(
                "SELECT contract, min(dt) AS first, max(dt) AS last, count(*) AS rows"
                " FROM contract_bars WHERE symbol=? AND exchange=? AND period=?"
                " GROUP BY contract ORDER BY contract",
                [symbol, exchange, period],
            ).df()
    except Exception:      # noqa: BLE001
        return empty
    if not df.empty:
        df["first"] = pd.to_datetime(df["first"])
        df["last"] = pd.to_datetime(df["last"])
    return df


def all_contracts(period: str = "1d") -> pd.DataFrame:
    """库里**全部品种**的真实合约：contract / symbol / exchange / last。

    下单面板的合约补全读这个。和 `list_contracts()` 的区别是那个按品种查（图上的
    「具体合约」下拉），这个跨品种一次拿全 —— 下单时要输的合约跟当前图上的品种
    没有关系（图上看着 rb、手上要下 m2601 是常事）。

    结果按 `_CON_CACHE` 缓存：合约表一天只变一次（挂牌/摘牌），而这里要扫
    一百万行 `contract_bars`，每次下拉都查一遍是几百毫秒。缓存键带上库文件的
    mtime，补完库自动失效。
    """
    empty = pd.DataFrame(columns=["contract", "symbol", "exchange", "last"])
    if not exists():
        return empty
    try:
        stamp = os.path.getmtime(db_path())
    except OSError:
        stamp = 0.0
    key = f"{period}|{stamp}"
    hit = _CON_CACHE.get(period)
    if hit is not None and hit[0] == key:
        return hit[1].copy()
    try:
        with connect(read_only=True) as con:
            if not _has_table(con, "contract_bars"):
                return empty
            df = con.execute(
                "SELECT contract, any_value(symbol) AS symbol,"
                " any_value(exchange) AS exchange, max(dt) AS last"
                " FROM contract_bars WHERE period=?"
                " GROUP BY contract ORDER BY contract",
                [period],
            ).df()
    except Exception:      # noqa: BLE001
        return empty
    if not df.empty:
        df["last"] = pd.to_datetime(df["last"])
    _CON_CACHE[period] = (key, df)
    return df.copy()


def contract_last_bar(contract: str, period: str = "1d") -> dict | None:
    """某个真实合约的最后一根 bar：`{dt, close, ...}`，查不到返回 None。

    下单面板在**没连柜台**的时候拿它填「最新价」—— 所以只查一行，不走
    `load_contracts()`（那个会把整个合约十年的行都读出来）。调用方必须把
    「这是本地收盘价、不是现价」写在界面上，不然盘中会当成现价用。
    """
    c = str(contract or "").strip()
    if not c or not exists():
        return None
    try:
        with connect(read_only=True) as con:
            if not _has_table(con, "contract_bars"):
                return None
            row = con.execute(
                "SELECT dt, close FROM contract_bars"
                " WHERE contract=? AND period=? ORDER BY dt DESC LIMIT 1",
                [c, period],
            ).fetchone()
    except Exception:      # noqa: BLE001
        return None
    if not row or row[1] is None:
        return None
    return {"dt": pd.Timestamp(row[0]), "close": float(row[1])}


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
        # 轻量库只带真实合约的日线，分钟级明细要下载完整库才有
        if not _has_table(con, "contract_bars"):
            return pd.DataFrame()
        df = con.execute(sql, args).df()
    if not df.empty:
        df["dt"] = pd.to_datetime(df["dt"])
    return df


# ------------------------------------------------------------------ 统计


def _has_table(con, name: str) -> bool:
    """表在不在。轻量库没有 `contract_bars` 的分钟级明细，不能假定它存在 ——
    直接查会抛 `CatalogException`，而 `stats()` 是界面首屏要调的。"""
    r = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(r and r[0])


def stats() -> dict:
    if not exists():
        return {"exists": False, "path": db_path(), "kind": "none"}
    out = {"exists": True, "path": db_path(), "kind": source(),
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
        if _has_table(con, "contract_bars"):
            r = con.execute(
                "SELECT count(*), count(DISTINCT contract) FROM contract_bars"
            ).fetchone()
            out["contracts"] = {"rows": int(r[0]), "contracts": int(r[1])}
        else:
            out["contracts"] = {"rows": 0, "contracts": 0}
    return out
