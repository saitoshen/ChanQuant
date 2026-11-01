"""天勤股票数据下载：枚举股票、下载 K 线、取复权因子。

## 天勤股票数据的事实

1. **合约格式**：`SSE.600000` / `SZSE.000001`，和期货合约一样是 `交易所.代码`
2. **ins_class="STOCK"**：`query_quotes(ins_class="STOCK", exchange_id=["SSE","SZSE"])`
   列全部股票
3. **K 线**：`get_kline_serial` 单只最多 8000 根（和期货一样）
4. **DataDownloader**：批量下载，但上限比期货更低（股票 5000+ 只，分批更小）
5. **复权**：天勤支持复权 K 线（"F" 前复权 / "B" 后复权），也可以取复权因子
6. **专业版**：股票数据和 DataDownloader 需要专业版。免费版只能 `get_kline_serial`
   取最近 8000 根
7. **起始时间**：天勤股票数据预计 2018 年起（正式上线 2020-09，但有回填历史），
   以实际连接确认的为准

## 下载策略

- **日线 + 1h**：取天勤能提供的最长时间（从最早有数据开始）
- **5m**：只取最近一年（5000+ 只 × 5m 全历史 = 几亿行，体积和耗时都不可接受）
- **复权因子**：用 `get_kline_serial` 的复权模式取复权后的价格，复权因子单独
  从 `get_quote` 取（`rst_rate` 字段）
- **批量**：每批 20 只（比期权更保守，股票品种数是期权的十几倍），带看门狗
- **断点续传**：`done_codes_tq` 记录每只股票最后下载到哪天，增量接着补

## 和期权下载器的区别

- 期权要算 IV（反解 + 微笑拟合），股票不需要——只下 K 线
- 期权的「链」概念在股票里不存在，每只股票独立
- 股票没有持仓量（`open_oi`/`close_oi`），但有成交额（`amount`）
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import stockstore as SS

PERIOD_SEC = {"1d": 86400, "1h": 3600, "5m": 300}

# 单批股票数。期权是 30，股票更保守取 20（品种数是期权十几倍，连接更容易被踢）
BATCH = 20

# 一批多久没有进度就判定卡死
STALL_SEC = 60

# 5m 只取最近一年的天数
RECENT_DAYS_5M = 365

# 天勤股票数据预计起始日。以实际为准——下不到数据不会报错，只返回空表
DEFAULT_START = date(2018, 1, 1)


def list_stocks(api, exchanges: list[str] | None = None) -> list[str]:
    """枚举全部股票代码。

    返回 `["SSE.600000", "SZSE.000001", ...]`。
    不调 `query_quotes` 两次——一次拿全部，按交易所前缀过滤。
    """
    exs = exchanges or ["SSE", "SZSE", "BSE"]
    all_codes = []
    for ex in exs:
        try:
            codes = api.query_quotes(ins_class="STOCK", exchange_id=[ex])
            all_codes.extend(codes)
        except Exception:  # noqa: BLE001
            pass
    return sorted(all_codes)


def stock_meta(api, codes: list[str]) -> pd.DataFrame:
    """从天勤取股票元数据：名称 / 交易所 / 板块 / 行业 / 上市日。

    每只股票调一次 `get_quote`，5000 只要几秒。板块和行业天勤可能不提供，
    缺的字段留给 Tushare 的 `stock_basic` 补。
    """
    rows = []
    for code in codes:
        try:
            q = api.get_quote(code)
        except Exception:  # noqa: BLE001
            continue
        ex = code.split(".")[0]
        name = getattr(q, "instrument_name", "") or ""
        # 天勤的板块/行业字段不一定有，先取着，没有就是空
        board = ""
        industry = ""
        # 上市日期：天勤的 listing_date 可能是时间戳或 None
        listing = getattr(q, "listing_date", None)
        if listing:
            try:
                list_date = pd.Timestamp(float(listing), unit="s",
                                         tz="Asia/Shanghai").date()
            except (ValueError, TypeError):
                list_date = None
        else:
            list_date = None

        rows.append({
            "ts_code": code,
            "name": name,
            "exchange": ex,
            "board": board,
            "industry": industry,
            "list_date": list_date,
            "delist_date": None,
        })
    return pd.DataFrame(rows)


def download_stock(api, ts_code: str, period: str,
                   start: date | None = None, end: date | None = None,
                   max_klines: int = 8000) -> pd.DataFrame:
    """单只股票的 K 线。`get_kline_serial` 路径（免费版也能用）。

    返回长表（ts_code/period/dt/OHLCV/amount）。

    天勤的 K 线字段映射：
    - `datetime`（纳秒时间戳）→ dt
    - `open/high/low/close` → OHLC
    - `volume` → 成交量（手）
    - `amount` → 成交额（元）

    单次最多 8000 根。日线 8000 根约 32 年，足够全历史；5m 8000 根约 3 个月，
    要分多次拉（`_download_full_5m` 处理）。
    """
    dur = PERIOD_SEC.get(period, 86400)
    try:
        kl = api.get_kline_serial(ts_code, dur, data_length=max_klines)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    if kl is None or len(kl) == 0:
        return pd.DataFrame()

    dt = pd.to_datetime(kl["datetime"], unit="ns", errors="coerce",
                        utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    ok = dt.notna()
    if not ok.any():
        return pd.DataFrame()

    df = pd.DataFrame({
        "ts_code": ts_code,
        "period": period,
        "dt": dt[ok].to_numpy(),
        "open": pd.to_numeric(kl.loc[ok, "open"], errors="coerce").to_numpy(),
        "high": pd.to_numeric(kl.loc[ok, "high"], errors="coerce").to_numpy(),
        "low": pd.to_numeric(kl.loc[ok, "low"], errors="coerce").to_numpy(),
        "close": pd.to_numeric(kl.loc[ok, "close"], errors="coerce").to_numpy(),
        "volume": pd.to_numeric(kl.loc[ok, "volume"], errors="coerce").fillna(0).to_numpy(),
        "amount": pd.to_numeric(kl.loc[ok, "amount"], errors="coerce").to_numpy(),
    })
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    if start:
        df = df[df["dt"].dt.date >= start]
    if end:
        df = df[df["dt"].dt.date <= end]
    return df.reset_index(drop=True)


def download_stock_batch(api, codes: list[str], period: str,
                         start: date, end: date,
                         tmpdir: str, stall: float = STALL_SEC,
                         cancel=None) -> pd.DataFrame:
    """批量下载（DataDownloader 路径，专业版）。

    每批 BATCH 只股票，带看门狗。卡住的批次退到单只重试。
    返回长表。
    """
    from tqsdk.tools import DataDownloader

    dur = PERIOD_SEC[period]
    os.makedirs(tmpdir, exist_ok=True)
    path = os.path.join(tmpdir, f"stock_batch_{period}.csv")
    bars: list[pd.DataFrame] = []

    def take(syms: list[str]) -> None:
        df = _read_csv(path, syms, period)
        if not df.empty:
            bars.append(df)

    for i in range(0, len(codes), BATCH):
        if cancel is not None and cancel():
            break
        chunk = codes[i:i + BATCH]
        if _run_batch(api, chunk, dur, start, end, path, stall) is not None:
            take(chunk)
            continue
        # 整批卡死：退到单只
        for sym in chunk:
            if cancel is not None and cancel():
                break
            if _run_batch(api, [sym], dur, start, end, path, stall) is not None:
                take([sym])

    if not bars:
        return pd.DataFrame(columns=SS.BAR_COLS_TQ)
    return pd.concat(bars, ignore_index=True)


def _read_csv(path: str, symbols: list[str], period: str) -> pd.DataFrame:
    """DataDownloader 的宽表 CSV → 长表。和期权同一套逻辑。"""
    if not os.path.exists(path) or os.path.getsize(path) < 32:
        return pd.DataFrame(columns=SS.BAR_COLS_TQ)
    raw = pd.read_csv(path)
    if "datetime" not in raw.columns or len(raw) == 0:
        return pd.DataFrame(columns=SS.BAR_COLS_TQ)

    dt = pd.to_datetime(raw["datetime"], errors="coerce")
    ok = dt.notna()
    if not ok.any():
        return pd.DataFrame(columns=SS.BAR_COLS_TQ)
    raw, dt = raw[ok], dt[ok]

    out: list[pd.DataFrame] = []
    for sym in symbols:
        close_col = f"{sym}.close"
        if close_col not in raw.columns:
            continue
        sub = pd.DataFrame({"dt": dt.to_numpy()})
        for k in ("open", "high", "low", "close", "volume", "amount"):
            src = f"{sym}.{k}"
            sub[k] = (pd.to_numeric(raw[src], errors="coerce").to_numpy()
                      if src in raw.columns else np.nan)
        sub = sub.dropna(subset=["close"])
        if sub.empty:
            continue
        sub["ts_code"] = sym
        sub["period"] = period
        out.append(sub)
    if not out:
        return pd.DataFrame(columns=SS.BAR_COLS_TQ)
    df = pd.concat(out, ignore_index=True)
    return df[SS.BAR_COLS_TQ]


def _run_batch(api, syms: list[str], dur: int, start: date, end: date,
               path: str, stall: float):
    """跑一批，成功返回 csv 路径，卡住返回 None。"""
    from tqsdk.tools import DataDownloader

    try:
        task = DataDownloader(api, symbol_list=list(syms), dur_sec=dur,
                              start_dt=start, end_dt=end, csv_file_name=path)
    except Exception:  # noqa: BLE001
        return None

    last, prog = time.time(), -1.0
    while not task.is_finished():
        try:
            api.wait_update(deadline=time.time() + 3)
        except Exception:  # noqa: BLE001
            break
        cur = task.get_progress()
        if cur != prog:
            prog, last = cur, time.time()
        elif time.time() - last > stall:
            break
    if not task.is_finished():
        try:
            task._task.cancel()
        except Exception:  # noqa: BLE001
            pass
        return None
    return path


# ------------------------------------------------------------------ 增量

def fetch_recent(api, ts_code: str, period: str,
                 last_dt: pd.Timestamp | None) -> pd.DataFrame:
    """增量补充：从 `last_dt` 到现在。

    `get_kline_serial` 取最近 8000 根，筛掉 `last_dt` 之前的。
    日线 8000 根远超一年，增量足够；1h 8000 根约一年；5m 8000 根约 3 个月。
    """
    df = download_stock(api, ts_code, period)
    if df.empty:
        return df
    if last_dt is not None:
        df = df[df["dt"] > pd.Timestamp(last_dt)]
    return df


def can_download(api) -> bool:
    """检查是否有 DataDownloader 权限（专业版）。"""
    try:
        return bool(api._auth._has_feature("tq_dl"))
    except Exception:  # noqa: BLE001
        return True  # 读不到按「能」处理


# ------------------------------------------------------------------ 写入

def write_stock_bars(con, df: pd.DataFrame) -> int:
    """写入天勤 K 线。"""
    return SS.upsert_bars_tq(con, df)


def write_meta(con, df: pd.DataFrame) -> int:
    """写入股票元数据。"""
    return SS.upsert_meta(con, df)


# ------------------------------------------------------------------ 全量入口

def run(api, period: str = "1d", exchanges: list[str] | None = None,
        start: date | None = None, end: date | None = None,
        cancel=None, progress=None) -> dict:
    """全量下载入口。

    返回 `{"done": N, "skipped": N, "errors": [...]}`。

    策略：
    1. 枚举全部股票
    2. 写入元数据
    3. 逐只增量下载（免费版走 `get_kline_serial`，专业版走 `DataDownloader` 批量）
    4. 写入 K 线
    """
    codes = list_stocks(api, exchanges)
    if not codes:
        return {"done": 0, "skipped": 0, "errors": ["没有枚举到任何股票"]}

    # 元数据
    #
    # `stock_meta()` 是**逐只** `get_quote`，5577 只股票要几十分钟 —— 而
    # Tushare 那条链路早就把名称 / 行业 / 板块 / 上市日写进同一张表了。
    # 库里已经覆盖大部分代码时直接跳过，别为了补几个名字把下载卡在第一步。
    try:
        known = set(SS.stock_list())
    except Exception:  # noqa: BLE001
        known = set()
    need_meta = len(known & set(codes)) < 0.9 * len(codes)
    if need_meta:
        meta = stock_meta(api, codes)
        if not meta.empty:
            with SS.connect() as con:
                write_meta(con, meta)

    # 增量断点
    done = SS.done_codes_tq(period)

    use_dd = can_download(api)
    tmpdir = os.path.join(os.path.dirname(SS.db_path()), "_stock_tmp")

    errors: list[str] = []
    n_done = 0
    n_skip = 0

    if use_dd:
        # 专业版：批量下载
        todo = [c for c in codes if c not in done or not done[c]]
        if period == "5m":
            # 5m 只取最近一年
            start_dt = date.today() - timedelta(days=RECENT_DAYS_5M)
        else:
            start_dt = start or DEFAULT_START
        end_dt = end or date.today()

        for i in range(0, len(todo), BATCH):
            if cancel is not None and cancel():
                break
            chunk = todo[i:i + BATCH]
            try:
                df = download_stock_batch(api, chunk, period, start_dt, end_dt,
                                          tmpdir, cancel=cancel)
                if not df.empty:
                    with SS.connect() as con:
                        write_stock_bars(con, df)
                    n_done += len(chunk)
                else:
                    n_skip += len(chunk)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"batch {chunk[:3]}...: {exc}")
                n_skip += len(chunk)
            if progress:
                progress(n_done, n_skip + n_done, len(codes))
    else:
        # 免费版：逐只 get_kline_serial
        for idx, code in enumerate(codes):
            if cancel is not None and cancel():
                break
            try:
                last = done.get(code)
                df = fetch_recent(api, code, period, last)
                if not df.empty:
                    with SS.connect() as con:
                        write_stock_bars(con, df)
                    n_done += 1
                else:
                    n_skip += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{code}: {exc}")
                n_skip += 1
            if progress:
                progress(n_done, n_skip + n_done, len(codes))

    return {"done": n_done, "skipped": n_skip, "errors": errors[:20]}
