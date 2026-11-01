"""从 Tushare 下载 A 股股票数据：全历史日线 + 复权因子 + 基本面 + 概念板块 + 资金流。

这个模块是**下载逻辑本体**，`download_stock_ts.py` 只是它的命令行壳子。
界面（「数据管理 → A 股」）也直接调这里的 `run()`，所以下载能力不能只活在
脚本里 —— 打包后的 exe 里没有 .py 文件，subprocess 调脚本这条路走不通。

写 `data_tq/stock.duckdb` 的 Tushare 专用表（和天勤数据分开表）。

## Tushare 股票接口

| 接口 | 用途 | 频率限制 |
|------|------|----------|
| `stock_basic` | 股票列表（代码/名称/板块/行业/上市日） | 每分钟 |
| `daily` | 日线行情（不复权） | 每分钟 500 次 |
| `adj_factor` | 复权因子 | 每分钟 500 次 |
| `daily_basic` | 每日指标（pe/pb/市值/换手率） | 每分钟 500 次 |
| `ths_index` | 同花顺概念板块列表 | 每天 |
| `ths_member` | 板块成分股 | 每天 |
| `moneyflow` | 个股资金流向 | 每分钟 500 次 |

## 全历史策略

- 日线：按 `trade_date` 分页（`pro.daily(trade_date=YYYYMMDD)`），每页最多 5000 条
  （一天的全部股票约 5000 行），一个交易日一次请求
- 复权因子：同样按 `trade_date` 分页
- 基本面：同上
- 资金流：同上
- 概念板块：先取列表再逐个取成分

## 代码格式

Tushare 股票代码：`600000.SH` / `000001.SZ`（交易所后缀两位）
天勤股票代码：`SSE.600000` / `SZSE.000001`

两者需要转换。统一用天勤格式做 `ts_code`（`SSE.600000`），Tushare 请求时转成
Tushare 格式（`600000.SH`）。

用法：

    python download_stock_ts.py --stats
    python download_stock_ts.py --daily                 # 全历史日线
    python download_stock_ts.py --daily --start 20200101
    python download_stock_ts.py --adj                   # 复权因子
    python download_stock_ts.py --basic                 # 基本面
    python download_stock_ts.py --concept               # 概念板块
    python download_stock_ts.py --moneyflow             # 资金流
    python download_stock_ts.py --all                   # 全部
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

from ttrade import stockstore as SS
from ttrade.futures import get_pro, read_token

PAGE = 5000      # daily 单次上限
SLEEP = 0.15     # 频率限制：每分钟 500 次 ≈ 0.12s/次，留余量

# 天勤 → Tushare 交易所映射
EXCHANGE_MAP = {
    "SSE": "SH",
    "SZSE": "SZ",
    "BSE": "BJ",
}
# Tushare → 天勤
EXCHANGE_MAP_REV = {v: k for k, v in EXCHANGE_MAP.items()}


def tq_to_ts(ts_code: str) -> str:
    """`SSE.600000` → `600000.SH`。"""
    ex, _, code = ts_code.partition(".")
    suffix = EXCHANGE_MAP.get(ex, "")
    return f"{code}.{suffix}" if suffix else ts_code


def ts_to_tq(ts_code: str) -> str:
    """`600000.SH` → `SSE.600000`。"""
    code, _, suffix = ts_code.partition(".")
    ex = EXCHANGE_MAP_REV.get(suffix, "")
    return f"{ex}.{code}" if ex else ts_code


# ------------------------------------------------------------------ 元数据

def fetch_basic(pro=None) -> pd.DataFrame:
    """股票列表：ts_code/name/exchange/board/industry/list_date/delist_date。

    Tushare 的 `stock_basic` 返回的 ts_code 是 `600000.SH` 格式，
    转成天勤格式 `SSE.600000`。

    `exchange_id` 并不可靠 —— 传 `SH` 也会带回 `920285.BJ` 这类北交所代码，
    结果就是「交易所列写 SSE、代号却是 .BJ」，界面上既不在沪市也不在深市，
    点开还是空图。所以交易所一律**按代号后缀判**，请求回来的行只留后缀对得上
    的那个交易所。
    """
    pro = pro or get_pro()
    rows = []
    for ex_suffix, ex_name in (("SH", "SSE"), ("SZ", "SZSE"), ("BJ", "BSE")):
        try:
            df = pro.stock_basic(exchange_id=ex_suffix,
                                 fields="ts_code,symbol,name,area,industry,"
                                        "list_date,delist_date")
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        keep = df["ts_code"].astype(str).str.upper().str.endswith("." + ex_suffix)
        df = df[keep]
        if df.empty:
            continue
        df["exchange"] = ex_name
        df["ts_code"] = df["ts_code"].apply(ts_to_tq)
        # 板块：从代码判断（科创板 688/创业板 300/主板其他）
        df["board"] = df["ts_code"].apply(_guess_board)
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
        df["delist_date"] = pd.to_datetime(df.get("delist_date"), errors="coerce")
        rows.append(df[["ts_code", "name", "exchange", "board", "industry",
                        "list_date", "delist_date"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _guess_board(ts_code: str) -> str:
    """从代码猜板块。`SSE.688xxx` = 科创板，`SZSE.300xxx` = 创业板。"""
    ex, _, code = ts_code.partition(".")
    if ex == "BSE":
        return "北交所"
    if code.startswith("688"):
        return "科创板"
    if code.startswith("300") or code.startswith("301"):
        return "创业板"
    if code.startswith("000") or code.startswith("001") or code.startswith("002"):
        return "主板"
    return ""


# ------------------------------------------------------------------ 日线

def fetch_daily(pro, trade_date: str) -> pd.DataFrame:
    """某一天全市场日线（不复权）。`trade_date` 格式 `YYYYMMDD`。"""
    try:
        df = pro.daily(trade_date=trade_date)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ts_code"] = df["ts_code"].apply(ts_to_tq)
    df["dt"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    # 字段映射
    out = pd.DataFrame({
        "ts_code": df["ts_code"],
        "dt": df["dt"].dt.date,
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["vol"], errors="coerce").fillna(0),
        "amount": pd.to_numeric(df["amount"], errors="coerce"),
        "pct_chg": pd.to_numeric(df["pct_chg"], errors="coerce"),
        "change": pd.to_numeric(df.get("change"), errors="coerce"),
    })
    return out.dropna(subset=["close"])


def fetch_adj(pro, trade_date: str) -> pd.DataFrame:
    """某一天全市场复权因子。"""
    try:
        df = pro.adj_factor(trade_date=trade_date)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ts_code"] = df["ts_code"].apply(ts_to_tq)
    df["dt"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return pd.DataFrame({
        "ts_code": df["ts_code"],
        "dt": df["dt"].dt.date,
        "adj_factor": pd.to_numeric(df["adj_factor"], errors="coerce"),
    }).dropna(subset=["adj_factor"])


def fetch_basic_daily(pro, trade_date: str) -> pd.DataFrame:
    """某一天全市场基本面（daily_basic）。"""
    try:
        df = pro.daily_basic(trade_date=trade_date)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ts_code"] = df["ts_code"].apply(ts_to_tq)
    df["dt"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    # 只取我们关心的列
    cols = {c: c for c in SS.BASIC_COLS if c in df.columns and c not in ("ts_code", "dt")}
    out = pd.DataFrame({"ts_code": df["ts_code"], "dt": df["dt"].dt.date})
    for c in SS.BASIC_COLS:
        if c in df.columns and c not in ("ts_code", "dt"):
            out[c] = pd.to_numeric(df[c], errors="coerce")
        elif c not in ("ts_code", "dt"):
            out[c] = None
    return out


def fetch_moneyflow(pro, trade_date: str) -> pd.DataFrame:
    """某一天全市场资金流。"""
    try:
        df = pro.moneyflow(trade_date=trade_date)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ts_code"] = df["ts_code"].apply(ts_to_tq)
    df["dt"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    out = pd.DataFrame({"ts_code": df["ts_code"], "dt": df["dt"].dt.date})
    for c in SS.MONEYFLOW_COLS:
        if c in df.columns and c not in ("ts_code", "dt"):
            out[c] = pd.to_numeric(df[c], errors="coerce")
        elif c not in ("ts_code", "dt"):
            out[c] = None
    return out


# ------------------------------------------------------------------ 概念板块

def fetch_concepts(pro) -> pd.DataFrame:
    """同花顺概念板块列表 + 成分股。返回 (concept_name, ts_code, in_date) 长表。

    字段名跟 Tushare 文档对不上，实测（2026-11）返回的是：

        ths_index  → ts_code / name / count / exchange / list_date / type
        ths_member → ts_code(板块代码) / con_code(成分股) / con_name

    文档写的是 `ths_name` 和成分股的 `ts_code`，照抄会拿到两列空字符串 —— 于是
    每个板块都被 `if not name` 跳过，最后「写入 0 行」且不报错。所以这里两个
    字段都做兼容：先取文档名，取不到再取实测名。
    """
    rows = []
    try:
        idx = pro.ths_index(exchange="A", type="N")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if idx is None or idx.empty:
        return pd.DataFrame()
    for _, r in idx.iterrows():
        ts_code = str(r.get("ts_code", ""))
        name = str(r.get("ths_name") or r.get("name") or "")
        if not ts_code or not name:
            continue
        try:
            members = pro.ths_member(ts_code=ts_code)
        except Exception:  # noqa: BLE001
            continue
        if members is None or members.empty:
            continue
        for _, m in members.iterrows():
            # 成分股：文档叫 ts_code，实测叫 con_code
            code = str(m.get("con_code") or m.get("ts_code") or "")
            if not code:
                continue
            rows.append({
                "concept_name": name,
                "ts_code": ts_to_tq(code),
                "in_date": m.get("in_date"),
            })
        time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 交易日历

def trade_calendar(pro, start: str, end: str) -> list[str]:
    """交易日列表（YYYYMMDD）。"""
    try:
        df = pro.trade_cal(exchange="SSE", start_date=start, end_date=end,
                           is_open="1")
    except Exception:  # noqa: BLE001
        # 退回工作日推算
        d0 = date.fromisoformat(f"{start[:4]}-{start[4:6]}-{start[6:8]}")
        d1 = date.fromisoformat(f"{end[:4]}-{end[4:6]}-{end[6:8]}")
        days = []
        cur = d0
        while cur <= d1:
            if cur.weekday() < 5:
                days.append(cur.strftime("%Y%m%d"))
            cur += timedelta(days=1)
        return days
    if df is None or df.empty:
        return []
    return sorted(df["cal_date"].astype(str).tolist())


# ------------------------------------------------------------------ 写入

def _write(df: pd.DataFrame, upsert_fn, table_name: str) -> int:
    if df is None or df.empty:
        return 0
    with SS.connect() as con:
        n = upsert_fn(con, df)
    return n


# ------------------------------------------------------------------ 入口

def run_daily(start: str | None = None, end: str | None = None,
              cancel=None, progress=None) -> dict:
    """下载全历史日线。"""
    pro = get_pro()
    today = date.today().strftime("%Y%m%d")
    start = start or "20100101"
    end = end or today

    cal = trade_calendar(pro, start, end)
    if not cal:
        return {"done": 0, "rows": 0, "errors": ["交易日历为空"]}

    done = SS.done_codes_ts()
    last_done = max(done.values()) if done else None
    if last_done:
        last_str = pd.Timestamp(last_done).strftime("%Y%m%d")
        cal = [d for d in cal if d > last_str]
        print(f"增量：从 {last_str} 开始，{len(cal)} 个交易日")

    n_rows = 0
    n_days = 0
    errors = []
    for i, td in enumerate(cal):
        if cancel is not None and cancel():
            break
        try:
            df = fetch_daily(pro, td)
            if not df.empty:
                n = _write(df, SS.upsert_bars_ts, "stock_bars_ts")
                n_rows += n
                n_days += 1
            # 复权因子同日写
            adj = fetch_adj(pro, td)
            if not adj.empty:
                _write(adj, SS.upsert_adj, "stock_adj")
            time.sleep(SLEEP)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{td}: {exc}")
        if progress:
            progress(i + 1, len(cal))

    return {"done": n_days, "rows": n_rows, "errors": errors[:20]}


def run_basic(start: str | None = None, end: str | None = None,
              cancel=None, progress=None) -> dict:
    """下载基本面（daily_basic）。"""
    pro = get_pro()
    today = date.today().strftime("%Y%m%d")
    start = start or "20100101"
    end = end or today

    cal = trade_calendar(pro, start, end)
    if not cal:
        return {"done": 0, "rows": 0, "errors": ["交易日历为空"]}

    n_rows = 0
    n_days = 0
    errors = []
    for i, td in enumerate(cal):
        if cancel is not None and cancel():
            break
        try:
            df = fetch_basic_daily(pro, td)
            if not df.empty:
                n = _write(df, SS.upsert_basic, "stock_basic_ts")
                n_rows += n
                n_days += 1
            time.sleep(SLEEP)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{td}: {exc}")
        if progress:
            progress(i + 1, len(cal))

    return {"done": n_days, "rows": n_rows, "errors": errors[:20]}


def run_moneyflow(start: str | None = None, end: str | None = None,
                  cancel=None, progress=None) -> dict:
    """下载资金流。"""
    pro = get_pro()
    today = date.today().strftime("%Y%m%d")
    start = start or "20200101"
    end = end or today

    cal = trade_calendar(pro, start, end)
    if not cal:
        return {"done": 0, "rows": 0, "errors": ["交易日历为空"]}

    n_rows = 0
    n_days = 0
    errors = []
    for i, td in enumerate(cal):
        if cancel is not None and cancel():
            break
        try:
            df = fetch_moneyflow(pro, td)
            if not df.empty:
                n = _write(df, SS.upsert_moneyflow, "stock_moneyflow")
                n_rows += n
                n_days += 1
            time.sleep(SLEEP)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{td}: {exc}")
        if progress:
            progress(i + 1, len(cal))

    return {"done": n_days, "rows": n_rows, "errors": errors[:20]}


def run_concept(cancel=None) -> dict:
    """下载概念板块。"""
    pro = get_pro()
    try:
        df = fetch_concepts(pro)
    except Exception as exc:  # noqa: BLE001
        return {"done": 0, "rows": 0, "errors": [str(exc)]}
    if df.empty:
        return {"done": 0, "rows": 0, "errors": ["没有概念数据"]}
    n = _write(df, SS.upsert_concept, "stock_concept")
    return {"done": n, "rows": n, "errors": []}


def run_meta() -> dict:
    """下载股票元数据。"""
    try:
        df = fetch_basic()
    except Exception as exc:  # noqa: BLE001
        return {"done": 0, "rows": 0, "errors": [str(exc)]}
    if df.empty:
        return {"done": 0, "rows": 0, "errors": ["没有元数据"]}
    n = _write(df, SS.upsert_meta, "stock_meta")
    return {"done": n, "rows": n, "errors": []}



# ------------------------------------------------------------------ 界面入口

# 下载项：(键, 名称)。顺序就是执行顺序 —— 元数据必须最先（后面的接口都靠它
# 把代码转成天勤格式），概念板块放最后（它是全市场一次请求，跑得快但行数最多）
PARTS = [
    ("meta", "股票列表"),
    ("daily", "日线 + 复权因子"),
    ("basic", "基本面（PE/PB/换手/市值）"),
    ("moneyflow", "资金流"),
    ("concept", "概念板块"),
]


def stats_text() -> str:
    """库现状，给界面显示。一行一项，查不到就说「没下」。"""
    st = SS.stats()
    if not st.get("exists"):
        return "还没有股票库（data_tq/stock.duckdb）—— 先点下面的「更新数据」"
    lines = [f"股票 {st['stocks']:,} 只　库 {st['size_mb']} MB"]
    tq = st.get("bars_tq") or {}
    if tq:
        for p, n in sorted(tq.items()):
            lines.append(f"天勤 {p} K 线 {n:,} 行")
    else:
        lines.append("天勤分钟线：未下载（1h / 5m 需要天勤专业版）")
    lines.append(f"Tushare 日线 {st['bars_ts']:,} 行　复权因子 {st['adj']:,} 行")
    lines.append(f"基本面 {st['basic']:,} 行　资金流 {st['moneyflow']:,} 行　"
                 f"概念 {st['concept']:,} 条")
    if st.get("first"):
        lines.append(f"覆盖 {st['first']} → {st['last']}")
    exs = st.get("exchanges") or []
    if exs:
        lines.append("交易所：" + "、".join(exs))
    return "\n".join(lines)


def run(parts=None, start: str | None = None, end: str | None = None,
        progress=None, cancel=None) -> dict:
    """按顺序跑若干下载项。返回 `{键: 结果}`。

    `progress(done, total, msg)`：`done/total` 是**下载项**的进度（5 项就是
    1/5、2/5…），单个接口内部的天数进度拼进 `msg` 里 —— 界面只有一根进度条，
    两层进度混在一起会来回跳。
    """
    todo = [p for p in (parts or [k for k, _ in PARTS]) if p in dict(PARTS)]
    if not todo:
        return {}
    label = dict(PARTS)
    out: dict[str, dict] = {}
    total = len(todo)
    for i, key in enumerate(todo):
        if cancel is not None and cancel():
            out[key] = {"done": 0, "rows": 0, "errors": ["已取消"]}
            continue
        progress and progress(i, total, f"{label[key]}…")

        def inner(a, b, _key=key, _i=i):
            progress and progress(_i, total, f"{label[_key]}　{a}/{b} 个交易日")

        try:
            if key == "meta":
                st = run_meta()
            elif key == "daily":
                st = run_daily(start=start, end=end, cancel=cancel,
                               progress=lambda a, b: inner(a, b))
            elif key == "basic":
                st = run_basic(start=start, end=end, cancel=cancel,
                               progress=lambda a, b: inner(a, b))
            elif key == "moneyflow":
                st = run_moneyflow(start=start, end=end, cancel=cancel,
                                   progress=lambda a, b: inner(a, b))
            else:
                st = run_concept(cancel=cancel)
        except Exception as exc:  # noqa: BLE001 - 一项失败不该拖垮其余项
            st = {"done": 0, "rows": 0, "errors": [f"{type(exc).__name__}: {exc}"]}
        out[key] = st
        progress and progress(i + 1, total,
                              f"{label[key]} 完成（{st.get('rows', 0):,} 行）")
    return out
