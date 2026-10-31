"""天勤 EDB 基本面指标（DuckDB）。

EDB（`edb.shinnytech.com`）是基本面指标库 —— 现货价、仓单、库存、交割量、
加工利润。指标只有数字 ID，中文名和分类只存在于网页上，所以指标清单是**随包
的一份 JSON**（`ttrade/edb_indicators.json`，243 个 ID），不在运行时抓网页。

**为什么单独一个库**：EDB 和 K 线没有任何共用的键，塞进 `tq.duckdb` 只会让那个
11GB 的库再大一块，而它要随包分发、整库下载，用户可能只想下行情。

表分成「指标定义」和「数值」两张：

    edb_meta    edb_id 主键 + 中文名 / 品种 / 分类 / 单位 / 频率
    edb_series  (edb_id, dt) 主键 + value

分开的理由：数值每天变，指标定义一年变不了两次。分表之后刷新定义不用重写
几百万行数值，按品种取数也只需要 join 一次 meta。

**频率不是统一的**：现货价是日频，仓单和库存多为周频，交割量按月。`edb_series`
里只存「有值的日子」，不做自然日补齐 —— 补齐该在读取层按需做（前向填充到
交易日），存进来会凭空多出几十万行 NaN。

**踩过的坑**（来自 future_quant 那套 MySQL 版本）：

- `query_edb_data` 返回的 index 是**字符串**（`'2026-09-04'`），不是 Timestamp。
  直接和 `datetime.date` 比永远不相等，增量过滤会 100% 失效 —— 这里统一在
  入口 `pd.to_datetime` 一次。
- 内部有 30 秒超时，长区间 × 100 个 ID 会偶发 `TqTimeoutError`。所以按 100 个
  一批，失败退避重试，不让一个批次打断整轮。
- `variety` 大小写不统一（`SC` 和 `sc` 并存）。这里统一 `lower()`，原始拼写留在
  `edb_meta.variety_raw` 里备查。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import date, timedelta

import pandas as pd

from .paths import data_dir

SCHEMA_VERSION = "1"

DB_NAME = "edb.duckdb"

# 天勤对单个品种的指标回溯深度不一致，2010 是「能取到就取到」的起点。
# 更早的日子接口会给空，不会报错。
DEFAULT_START = date(2010, 1, 1)

# 单次 `query_edb_data` 的 ID 上限。超过会直接报参数错误，不是慢一点的问题。
MAX_IDS = 100

# 增量重算的回看天数。周频指标（仓单）在两次运行之间可能有修订，
# 往前多查几天覆盖住，代价可以忽略（整批查询本来就只有一两秒）
OVERLAP_DAYS = 30

CATEGORY_LABEL = {
    "spot_price": "现货价",
    "warehouse": "仓单数量",
    "inventory": "期货库存",
    "delivery": "交割量",
    "profit": "加工利润",
}

DDL = [
    """
    CREATE TABLE IF NOT EXISTS edb_meta (
        edb_id      INTEGER PRIMARY KEY,
        name        VARCHAR,
        variety     VARCHAR,
        variety_raw VARCHAR,
        category    VARCHAR,
        unit        VARCHAR,
        freq        VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edb_series (
        edb_id  INTEGER NOT NULL,
        dt      DATE    NOT NULL,
        value   DOUBLE,
        PRIMARY KEY (edb_id, dt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edb_meta_kv (
        k VARCHAR PRIMARY KEY,
        v VARCHAR
    )
    """,
]


# ------------------------------------------------------------------ 路径


def db_path() -> str:
    """EDB 库路径。`TRENDDESK_EDB_DB` 可以指到别处（测试用临时库）。"""
    env = os.environ.get("TRENDDESK_EDB_DB")
    if env:
        return env
    return os.path.join(
        os.path.dirname(data_dir("cn").rstrip("/\\")), "data_tq", DB_NAME
    )


def exists() -> bool:
    return os.path.isfile(db_path())


@contextmanager
def connect(read_only: bool = False):
    """打开 EDB 库。

    和 `tqstore.connect` 一样：DuckDB 的文件锁是排他的，写的时候读不了。
    只读打开重试两次，让「下载刚结束、连接还没释放」这种瞬态自己过去。
    """
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
                "INSERT OR REPLACE INTO edb_meta_kv VALUES ('schema_version', ?)",
                [SCHEMA_VERSION],
            )
        yield con
    finally:
        con.close()


# ------------------------------------------------------------------ 指标清单


def indicators_path() -> str:
    """随包的指标清单路径。

    打包后 `__file__` 落在 PYZ 里，指向一个不存在的 `.pyc` 路径 —— 所以先看
    `__file__` 旁边，找不到再退回 `bundle_root()/ttrade/`。两个都找不到就直接
    报错：静默返回空清单会让「更新基本面」变成一次什么都不做的空跑。
    """
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "edb_indicators.json")
    if os.path.isfile(here):
        return here
    from .paths import bundle_root

    alt = os.path.join(bundle_root(), "ttrade", "edb_indicators.json")
    if os.path.isfile(alt):
        return alt
    return here          # 交给调用方抛 FileNotFoundError，带上真实路径


def load_indicators(path: str | None = None) -> dict[int, dict]:
    """读随包的指标清单，返回 `{edb_id: meta}`。

    清单的键是 `品种_ID`（如 `OI_6980`），品种从键里取，取不到才退回 meta 里
    的 `variety` 字段 —— 老版本清单有几条没写这个字段。
    """
    p = path or indicators_path()
    with open(p, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    out: dict[int, dict] = {}
    for cat, items in cfg.items():
        if not isinstance(items, dict):
            continue
        for key, meta in items.items():
            try:
                eid = int(meta.get("id"))
            except (TypeError, ValueError):
                continue
            raw = str(meta.get("variety") or "")
            if not raw and "_" in str(key):
                raw = str(key).split("_")[0]
            out[eid] = {
                "name": str(meta.get("name") or ""),
                "variety": raw.lower(),
                "variety_raw": raw,
                "category": cat,
                "unit": str(meta.get("unit") or ""),
                "freq": str(meta.get("freq") or ""),
            }
    return out


def _insert_df(con, table: str, df: pd.DataFrame) -> None:
    """把 DataFrame 整块灌进表里，冲突就覆盖。

    **不能用 `executemany`**：DuckDB 那个是逐行执行预处理语句，5 万行要跑几分钟
    （实测 52,874 行 100 秒还没完）。`register` 之后走的是列存批量写入，同样
    的数据不到一秒。这是 DuckDB 和 sqlite 最不一样的地方。
    """
    if df is None or len(df) == 0:
        return
    con.register("_edb_batch", df)
    try:
        cols = ", ".join(df.columns)
        con.execute(f"INSERT OR REPLACE INTO {table} SELECT {cols} FROM _edb_batch")
    finally:
        con.unregister("_edb_batch")


def upsert_meta(con, indicators: dict[int, dict]) -> int:
    if not indicators:
        return 0
    df = pd.DataFrame(
        [(int(eid), m["name"], m["variety"], m["variety_raw"],
          m["category"], m["unit"], m["freq"]) for eid, m in indicators.items()],
        columns=COLS_META,
    )
    _insert_df(con, "edb_meta", df)
    return len(df)


# ------------------------------------------------------------------ 取数


def _to_date(x) -> date | None:
    """天勤返回的日期是字符串，也可能是 Timestamp —— 统一成 `date`。"""
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, date):
        return x
    s = str(x).strip().replace("/", "-")
    if len(s) < 10:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def fetch(api, ids: list[int], start: date, end: date,
          retries: int = 3, base_delay: float = 4.0) -> pd.DataFrame:
    """带退避重试的 `query_edb_data`。

    内部有 30 秒超时，全区间 × 100 个 ID 偶发会撞上。这里退避 4s / 8s 重试，
    仍失败就把异常抛给调用方 —— 由上层决定「跳过这一批继续」还是整轮失败。
    """
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return api.query_edb_data(ids=list(ids), start_dt=start, end_dt=end)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries:
                time.sleep(base_delay * attempt)
    raise RuntimeError(f"EDB 查询失败（已重试 {retries} 次）：{last}")


def _long(df: pd.DataFrame, indicators: dict[int, dict]) -> pd.DataFrame:
    """宽表（index=日期字符串，columns=ID）→ `[edb_id, dt, value]` 长表。"""
    empty = pd.DataFrame(columns=["edb_id", "dt", "value"])
    if df is None or len(df) == 0:
        return empty

    # 天勤给的 index 是**字符串**。`pd.isna` 对 Index 返回的是 ndarray，
    # 所以下面一律用布尔数组而不是 pandas 的布尔 Series —— 两者的 `.to_numpy`
    # 行为不一样，混着写会得到「ndarray 没有 to_numpy」这种莫名其妙的错。
    idx = pd.to_datetime(pd.Index(df.index), errors="coerce")
    mask = ~pd.isna(idx).astype(bool)
    if not mask.any():
        return empty
    body = df.copy()
    body.index = idx
    body = body[mask]

    out: list[pd.DataFrame] = []
    for col in body.columns:
        try:
            eid = int(col)
        except (TypeError, ValueError):
            continue
        if eid not in indicators:
            continue          # 接口会顺手返回几个没要的 ID，别写进来
        s = pd.to_numeric(body[col], errors="coerce")
        s = s[~pd.isna(s)]
        if s.empty:
            continue
        out.append(pd.DataFrame({
            "edb_id": eid,
            "dt": [d.date() for d in s.index],
            "value": s.to_numpy(dtype=float),
        }))
    if not out:
        return empty
    return pd.concat(out, ignore_index=True)


def _last_dates(con) -> dict[int, date]:
    rows = con.execute("SELECT edb_id, max(dt) FROM edb_series GROUP BY 1").fetchall()
    out: dict[int, date] = {}
    for eid, dt in rows:
        d = _to_date(dt)
        if d is not None:
            out[eid] = d
    return out


def update(api, start: date | None = None, end: date | None = None,
           incremental: bool = True, progress=None,
           cancel=None) -> dict:
    """把指标拉进库。

    增量口径：**按批**算起点，取这一批里最早的那个「最后一天」再往前推
    `OVERLAP_DAYS`。按批而不是按 ID 是因为一次请求能带 100 个 ID，逐个 ID 查
    等于把请求数乘 100 倍；而整批都从 2010 重查也只有一两秒，没必要精算。

    返回 `{ids, rows, batches, failed, start, end}`。
    """
    indicators = load_indicators()
    ids = sorted(indicators)
    start = start or DEFAULT_START
    end = end or date.today()

    stats = {"ids": len(ids), "rows": 0, "batches": 0,
             "failed": 0, "start": start, "end": end}

    with connect() as con:
        upsert_meta(con, indicators)
        last = _last_dates(con) if incremental else {}
        total = (len(ids) + MAX_IDS - 1) // MAX_IDS

        for i in range(0, len(ids), MAX_IDS):
            if cancel is not None and cancel():
                break
            batch = ids[i:i + MAX_IDS]
            n = i // MAX_IDS + 1
            if progress is not None:
                progress(n - 1, total, f"EDB 第 {n}/{total} 批（{len(batch)} 个指标）")

            # 批内最早的最后一天 → 起点；全都没下过就从 start 开始
            known = [last[e] for e in batch if e in last]
            lo = min(known) - timedelta(days=OVERLAP_DAYS) if known else start
            if lo < start:
                lo = start
            if lo > end:
                lo = start          # 库里比 end 还新（改过系统时间），退回全量

            try:
                df = fetch(api, batch, lo, end)
            except Exception:      # noqa: BLE001
                # 一个批次失败不该把整轮打断：另外 142 个指标是好的
                stats["failed"] += 1
                if progress is not None:
                    progress(n, total, f"EDB 第 {n}/{total} 批失败，跳过")
                continue

            stats["batches"] += 1
            long = _long(df, indicators)
            if len(long):
                _insert_df(con, "edb_series", long)
                stats["rows"] += len(long)
            if progress is not None:
                progress(n, total,
                         f"EDB 第 {n}/{total} 批　+{len(long):,} 行")

        con.execute(
            "INSERT OR REPLACE INTO edb_meta_kv VALUES ('last_update', ?)",
            [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
        )
    return stats


# ------------------------------------------------------------------ 读取


COLS_META = ["edb_id", "name", "variety", "variety_raw", "category", "unit", "freq"]


def meta() -> pd.DataFrame:
    """指标定义表。`category_label` 是中文分类，界面直接用。"""
    if not exists():
        return pd.DataFrame(columns=COLS_META + ["category_label"])
    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT edb_id, name, variety, variety_raw, category, unit, freq"
            " FROM edb_meta ORDER BY category, edb_id"
        ).df()
    df["category_label"] = df["category"].map(
        lambda c: CATEGORY_LABEL.get(str(c), str(c))
    )
    return df


def load(ids: list[int] | None = None,
         varieties: list[str] | None = None,
         categories: list[str] | None = None,
         start: date | str | None = None,
         end: date | str | None = None,
         wide: bool = True) -> pd.DataFrame:
    """读指标数值。

    `wide=True` 返回 `index=dt, columns=edb_id` 的宽表 —— 和 `query_edb_data`
    同形，方便和原来的代码对齐；`wide=False` 返回 `[edb_id, dt, value]`。

    `varieties` / `categories` 都按小写比对（库里 `variety` 已经归一化过）。
    """
    if not exists():
        return pd.DataFrame()

    where, args = [], []
    if ids:
        where.append("edb_id IN (" + ",".join("?" * len(ids)) + ")")
        args += [int(x) for x in ids]
    if varieties:
        low = [str(v).lower() for v in varieties]
        where.append(
            "edb_id IN (SELECT edb_id FROM edb_meta WHERE lower(variety) IN ("
            + ",".join("?" * len(low)) + "))"
        )
        args += low
    if categories:
        where.append(
            "edb_id IN (SELECT edb_id FROM edb_meta WHERE category IN ("
            + ",".join("?" * len(categories)) + "))"
        )
        args += [str(c) for c in categories]
    if start is not None:
        where.append("dt >= ?")
        args.append(pd.Timestamp(start).date())
    if end is not None:
        where.append("dt <= ?")
        args.append(pd.Timestamp(end).date())

    sql = "SELECT edb_id, dt, value FROM edb_series"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY dt, edb_id"

    with connect(read_only=True) as con:
        df = con.execute(sql, args).df()
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["dt"])
    if not wide:
        return df
    out = df.pivot_table(index="dt", columns="edb_id", values="value", aggfunc="last")
    out.columns.name = None
    return out.sort_index()


def stats() -> dict:
    """概览：指标数、行数、覆盖区间、库体积。"""
    if not exists():
        return {"exists": False, "ids": 0, "rows": 0,
                "first": None, "last": None, "size_mb": 0.0}
    with connect(read_only=True) as con:
        ids = con.execute("SELECT count(*) FROM edb_meta").fetchone()[0]
        rows, first, last = con.execute(
            "SELECT count(*), min(dt), max(dt) FROM edb_series"
        ).fetchone()
    size = os.path.getsize(db_path()) / 1024 / 1024
    return {"exists": True, "ids": int(ids or 0), "rows": int(rows or 0),
            "first": first, "last": last, "size_mb": round(size, 1)}


def varieties() -> list[str]:
    """库里有哪些品种（小写）。"""
    if not exists():
        return []
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT DISTINCT variety FROM edb_meta WHERE variety <> '' ORDER BY 1"
        ).fetchall()
    return [r[0] for r in rows]
