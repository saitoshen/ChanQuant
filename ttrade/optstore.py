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


# 期货品种 → 期权品种。绝大多数同名，只有股指三家**完全不一样**：
#   沪深300 期货 IF ↔ 期权 IO（标的指数 000300）
#   上证50  期货 IH ↔ 期权 HO（000016）
#   中证1000 期货 IM ↔ 期权 MO（000852）
# 名字对不上的根源是命名口径不同：期货按**合约**命名，期权按**标的指数**命名。
# 中证500（IC）至今没有对应的场内期权，所以不在这张表里。
INDEX_FUT_OPT = {"IF": "IO", "IH": "HO", "IM": "MO"}


def _base_symbol(symbol: str) -> str:
    """`IF_CFFEX` / `if` / `IF` → `IF`。交易所后缀和大小写都按品种代码归一。

    🔴 必须先判有没有下划线：`"IF".rpartition("_")` 返回 `("", "", "IF")`，
    照 `ex` 那一支剥下去就把整个品种切空了（返回空串，所有品种都查不到数据）。
    """
    s = str(symbol or "").strip()
    if "_" in s:
        sym, _, ex = s.rpartition("_")
        if ex and ex.isalpha() and ex.isupper():
            s = sym
    return s.upper()


def opt_variety(symbol: str) -> str:
    """期货品种名 → 期权品种名（股指走映射，商品原样返回大写）。"""
    return INDEX_FUT_OPT.get(_base_symbol(symbol), _base_symbol(symbol))


def iv_candidates(symbol: str) -> list[str]:
    """按期货品种给出**库里可能用的**期权品种写法，按优先级排好。

    股指只有映射那一档；商品库里的写法不统一 —— 大商所 / 上期所 / 能源中心
    是小写（`cu` / `m` / `sc`），郑商所是大写（`CF` / `MA`），所以两种都试。
    """
    base = _base_symbol(symbol)
    if base in INDEX_FUT_OPT:
        return [INDEX_FUT_OPT[base]]      # 股指：IO / HO / MO，别再去试 IF
    return [base, base.lower()]


def iv_span(symbol: str, period: str = "1d"):
    """这个品种 / 周期在库里的 IV 覆盖：`(行数, 最早, 最晚, 天数)`；没有返回 None。

    只做聚合查询，**不拉数据** —— 判「分钟 bar 该用哪一档」时会连问好几档，
    每档都 `load_iv()` 一遍就是几百毫秒起步。判覆盖度必须看两个维度：

    - **最晚到哪天**：库里 5m 档各家差别很大（IO 只到 2020-11、HO 一直到 2026-10、
      MO 没有），只看「有没有」会挑到一个停更五年的档
    - **一共多少天**（第 4 个元素）：`download_options.py --periods 5m` 只能从
      天勤回补约两个月，补完之后「最晚一条」是新的、看着完全合格，但历史深度
      只有 40 来天。只判新旧的闸门会放它进来，于是前面两个月 5m、后面四年日频，
      精度前后不一致 —— 这正是 `_iv_period_for` 要拦的东西
    """
    if not exists():
        return None
    for v in iv_candidates(symbol):
        try:
            with connect(read_only=True) as con:
                row = con.execute(
                    "SELECT count(*), min(dt), max(dt),"
                    " count(DISTINCT CAST(dt AS DATE)) FROM option_iv"
                    " WHERE variety = ? AND period = ?",
                    [v, period],
                ).fetchone()
        except Exception:      # noqa: BLE001 - 库正被写、文件不在，都当「没有」
            return None
        if row and row[0]:
            return (int(row[0]), pd.Timestamp(row[1]), pd.Timestamp(row[2]),
                    int(row[3]))
    return None


def has_iv(symbol: str, period: str = "1d") -> bool:
    """库里这个品种 / 周期有没有 IV 摘要（`iv_span` 的布尔版）。"""
    return iv_span(symbol, period) is not None


def load_iv_for(symbol: str, period: str = "1d", **kw) -> pd.DataFrame:
    """按**期货品种**取 IV 摘要（`period` 档：`1d` / `1h` / `5m`）。

    返回 `index=dt`、列正好 `IV_COLS` 的表（`variety` / `period` 这两个键列
    去掉）—— 和 `auxdata.load_opt()` 读 Tushare 那份 CSV 的形状一致，可以直接
    顶替。找不到数据返回空表，不报错。

    存在的理由：股指期权的标的是**指数**不是期货合约，Tushare 那条
    `opt_daily + settle_map` 的链路拿不到 F，于是 `data_futures_aux/opt_IF_*.csv`
    根本不会生成 —— 股指期货的 IV 因子一直是空的。本地天勤库有 IO / HO / MO，
    只是名字对不上，映射一下就能用。
    """
    for v in iv_candidates(symbol):
        df = load_iv(varieties=[v], period=period, **kw)
        if not df.empty:
            return (df.set_index("dt")[list(IV_COLS)]
                    .sort_index().drop_duplicates())
    return pd.DataFrame()



@contextmanager
def connect(read_only: bool = False):
    import duckdb

    p = db_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    # DuckDB 是「一写多读」，读写**不能并存**。两个真实场景都会撞上：
    # 界面在「期权合约」下拉里挑链（读），后台下载器正好要写；或者反过来。
    # 读的那边一般几十毫秒就关连接，所以重试几次基本都能过去。
    # 写锁等得更久一些：下载器跑一条链要几秒，等它写完这条就能进。
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
            f"打不开期权库（{'读' if read_only else '写'}）—— 另一个程序正占着它："
            f"{last}"
        ) from last
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


def chain_expiries() -> dict[str, pd.Timestamp]:
    """每条链（= 标的合约）的到期日：`{'CZCE.ZC406': 2024-06-05, ...}`。

    **离线一次查询**，用来在联网之前就把「整条链都在回溯窗口之前」的筛掉。

    为什么值得单独一个函数：库里 2939 条链里有两千多条是已摘牌的，而判断一条链
    老不老原本只能 `get_quote` 问一次天勤（`chain_info`）—— 「只补最近 1 天」的
    情形下，光是为了跳过它们就要发两千多个网络请求。这份表在下载时已经写进
    `option_meta` 了，直接查就行。

    查不到（库不存在 / 表还空 / 被写锁占着）就返回空字典，调用方退回「每条链都
    联网问一次」的老路径 —— 少一个优化，不影响正确性。
    """
    if not exists():
        return {}
    try:
        with connect(read_only=True) as con:
            rows = con.execute(
                "SELECT underlying, max(expiry) FROM option_meta "
                "WHERE underlying IS NOT NULL GROUP BY 1"
            ).fetchall()
    except Exception:  # noqa: BLE001 - 锁被占 / 表还没建：当作没有
        return {}
    out: dict[str, pd.Timestamp] = {}
    for u, e in rows:
        if isinstance(u, str) and u and e is not None:
            out[u] = pd.Timestamp(e)
    return out


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


def list_months(variety: str | None = None) -> pd.DataFrame | None:
    """库里有哪几条期权链（品种 + 标的 + 到期月）。

    界面上的「月份」下拉读这个。列：variety / underlying / expiry / n / first / last。
    `expiry` 是到期日，`n` 是这条链上的合约数 —— 链长 20 和 200 的差别很大，
    摆出来用户才知道该挑哪个月。

    **一条链 = 标的 + 到期日，不是只有标的**。商品期权的标的自带月份
    （`DCE.m2405`），两者一一对应；但中金所三个股指期权所有月份共用同一个
    标的（`SSE.000300`），只按标的聚合会把 24 个月压成一条，行权价也跟着
    混成两千多个 —— 挑出来的合约可能根本不在你想看的那个月。

    **返回值三态**：

    - `None` —— 库在，但打不开（下载器正持写锁）。界面应该说「正在下载」，
      而不是「还没有数据」—— 后者会让人以为下失败了
    - 空表 —— 库在、能读，但一条链都没有
    - 有数据
    """
    empty = pd.DataFrame(columns=["variety", "underlying", "expiry", "n",
                                  "first", "last"])
    if not exists():
        return empty
    try:
        with connect(read_only=True) as con:
            # `count(*)` 会被 LEFT JOIN 的行数放大成「合约数 × 根数」，
            # 必须数不同的合约
            sql = ("SELECT m.variety, m.underlying, m.expiry,"
                   " count(DISTINCT m.contract) AS n,"
                   " min(b.dt) AS first, max(b.dt) AS last"
                   " FROM option_meta m"
                   " LEFT JOIN option_bars b ON b.contract = m.contract"
                   "   AND b.period = '1d'")
            args: list = []
            if variety:
                sql += " WHERE m.variety = ?"
                args.append(variety)
            sql += (" GROUP BY m.variety, m.underlying, m.expiry"
                    " ORDER BY m.variety, m.expiry, m.underlying")
            df = con.execute(sql, args).df()
    except Exception:      # noqa: BLE001
        return None
    if not df.empty:
        df["expiry"] = pd.to_datetime(df["expiry"])
        for c in ("first", "last"):
            df[c] = pd.to_datetime(df[c])
    return df


def list_strikes(variety: str, underlying: str,
                 expiry=None) -> pd.DataFrame:
    """一条链上的全部行权价。

    列：strike / cp / contract / rows。`rows` 是日线根数 —— 用来判断这个合约
    是不是从来没成交过（天勤会给「合约存在但没数据」的壳）。

    `expiry` 一定要带：股指期权一个月里行权价就上百个，24 个月混在一起是
    两千多个，同一个行权价会出现 24 次（对应 24 个到期日）。
    """
    empty = pd.DataFrame(columns=["strike", "cp", "contract", "rows"])
    if not exists():
        return empty
    try:
        with connect(read_only=True) as con:
            sql = ("SELECT m.strike, m.opt_class AS cp, m.contract,"
                   " count(b.dt) AS rows"
                   " FROM option_meta m"
                   " LEFT JOIN option_bars b ON b.contract = m.contract"
                   "   AND b.period = '1d'"
                   " WHERE m.variety = ? AND m.underlying = ?")
            args: list = [variety, underlying]
            if expiry is not None:
                sql += " AND m.expiry = ?"
                args.append(pd.Timestamp(expiry))
            sql += (" GROUP BY m.strike, m.opt_class, m.contract"
                    " ORDER BY m.strike, m.opt_class")
            df = con.execute(sql, args).df()
    except Exception:      # noqa: BLE001
        return empty
    return df


def underlying_close(underlying: str, period: str = "1d") -> float | None:
    """标的最新收盘价。界面拿它决定「默认落在哪个行权价」。"""
    if not exists():
        return None
    try:
        with connect(read_only=True) as con:
            r = con.execute(
                "SELECT close FROM option_underlying WHERE contract=? AND period=?"
                " ORDER BY dt DESC LIMIT 1",
                [underlying, period],
            ).fetchone()
    except Exception:      # noqa: BLE001
        return None
    return float(r[0]) if r and r[0] is not None else None


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
