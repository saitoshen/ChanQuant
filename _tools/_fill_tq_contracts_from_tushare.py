# -*- coding: utf-8 -*-
"""从 Tushare 侧真实合约日线补天勤 contract_bars（只填空缺，不覆盖已有）。

数据源 auto（默认）：
1. 优先读 MySQL `future_quant.daily_data` JOIN `futures_contracts`
   （本机这张表 2016–2021 每年 6.4 万~17 万行，覆盖完整）。
2. MySQL 里缺的交易日，再走 Tushare `pro.fut_daily(trade_date=day)` 下载。
3. 统一 `INSERT OR IGNORE` 写入 `tqstore.contract_bars` period='1d'，
   已有 key 一律保留，不覆盖天勤原生数据。

合约名转换（关键）：
- MySQL `A1607` + `DCE`      -> `A1607.DCE`
- MySQL `RB2009` + `SHFE`    -> `RB2009.SHF`
- MySQL `SR901` + `CZCE`     -> 按交易日消歧 -> `SR1901.ZCE` / `SR2901.ZCE`
- `a000` / `AP000` 这类主力连续（尾号 000）跳过。
- Tushare 接口返回的就是 4 位 ts_code，直接清洗后写入。

用法：
    python _tools/_fill_tq_contracts_from_tushare.py --start 2016-01-01 --end 2021-12-31
    python _tools/_fill_tq_contracts_from_tushare.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FQ = "D:/CozeTask/future_quant"
for p in (ROOT, FQ):
    if p not in sys.path:
        sys.path.insert(0, p)

from ttrade import contracts as C, futures as F, tqstore as S  # noqa: E402
from ttrade.tqsdk_symbols import to_ts  # noqa: E402

TS_SUFFIX = {"SHFE": "SHF", "DCE": "DCE", "CZCE": "ZCE",
             "CFFEX": "CFX", "INE": "INE", "GFEX": "GFE"}
CODE_RE = re.compile(r"^([A-Za-z]+)(\d{3,4})$")
COLS = ["symbol", "exchange", "period", "contract", "dt", "trading_day",
        "open", "high", "low", "close", "volume", "open_oi", "close_oi"]


def to_ts_code(sym: str, exchange: str, day) -> tuple[str, str] | None:
    """MySQL 代码 -> (tqstore.symbol, Tushare ts_code)。None 表示跳过。"""
    m = CODE_RE.match(str(sym).strip())
    if not m:
        return None
    prod, digits = m.group(1).upper(), m.group(2)
    if digits == "000":
        return None
    ex = str(exchange).strip().upper()
    suffix = TS_SUFFIX.get(ex)
    if suffix is None:
        return None
    if len(digits) == 3:
        # 只有郑商所历史上有 3 位月份；其它交易所的 3 位基本是主连/脏数据。
        if ex != "CZCE":
            return None
        try:
            ts_code = to_ts(f"CZCE.{prod}{digits}", day)
        except Exception:
            return None
    else:
        ts_code = f"{prod}{digits}.{suffix}"
    return prod, ts_code


def make_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """标准字段 -> contract_bars 表结构。rows 需带 symbol/exchange/contract/date/OHLCV/OI。"""
    d = rows.copy()
    d["period"] = "1d"
    d["dt"] = pd.to_datetime(d["date"]).dt.normalize()
    d["trading_day"] = d["dt"].dt.date
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("volume", "open_interest"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype("int64")
    d["open_oi"] = d["open_interest"]
    d["close_oi"] = d["open_interest"]
    # 无效价格不写，避免把坏 bar 灌进库
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[d["close"] > 0]
    return d[["symbol", "exchange", "period", "contract", "dt", "trading_day",
              "open", "high", "low", "close", "volume", "open_oi", "close_oi"]]


def insert_ignore(con, d: pd.DataFrame) -> int:
    if d is None or d.empty:
        return 0
    d = d.drop_duplicates(
        subset=["symbol", "exchange", "period", "contract", "dt"], keep="last")
    con.register("_tushare_fill", d[COLS])
    try:
        # DuckDB 的 execute() 返回连接自身（不是独立 result 对象），
        # fetchone() 取到最后一行后仍留着“未完成结果”，下一条 execute /
        # unregister 会抛 ResourceInUseException。统一用 fetchall() 彻底消费；
        # 绝不能对返回值调 close()——那会把整个连接关掉。
        before = con.execute(
            "SELECT count(*) FROM contract_bars WHERE period='1d'").fetchall()[0][0]
        con.execute(
            "INSERT OR IGNORE INTO contract_bars SELECT * FROM _tushare_fill"
        ).fetchall()
        after = con.execute(
            "SELECT count(*) FROM contract_bars WHERE period='1d'").fetchall()[0][0]
        return int(after - before)
    finally:
        con.unregister("_tushare_fill")


def mysql_rows(start, end, pairs: set[str] | None, chunksize: int):
    """读 MySQL 真实合约日线，产出标准 frame。连接失败返回 None 让上层回退 Tushare。"""
    try:
        from sqlalchemy import text
        from data.db.db_manager import DBManager
    except Exception as exc:  # noqa: BLE001
        print(f"  MySQL 不可用（{type(exc).__name__}），改走 Tushare", flush=True)
        return None
    sql = text(
        "SELECT c.symbol, c.exchange, d.trading_date AS date, "
        "       d.open_price AS open, d.high_price AS high, "
        "       d.low_price AS low, d.close_price AS close, "
        "       d.volume, d.open_interest "
        "FROM daily_data d JOIN futures_contracts c ON c.id=d.contract_id "
        "WHERE d.trading_date >= :s AND d.trading_date < :e "
        "ORDER BY d.trading_date, c.symbol"
    )
    try:
        db = DBManager()
        with db.engine.connect() as conn:
            frames = []
            n_all = n_keep = 0
            for chunk in pd.read_sql(sql, conn,
                                     params={"s": start, "e": end},
                                     chunksize=chunksize):
                n_all += len(chunk)
                out = []
                for row in chunk.itertuples(index=False):
                    conv = to_ts_code(row.symbol, row.exchange, row.date)
                    if conv is None:
                        continue
                    sym, ts_code = conv
                    if pairs is not None and f"{sym}_{row.exchange}" not in pairs:
                        continue
                    out.append((sym, row.exchange, ts_code, row.date,
                                row.open, row.high, row.low, row.close,
                                row.volume, row.open_interest))
                n_keep += len(out)
                if out:
                    frames.append(pd.DataFrame(
                        out, columns=["symbol", "exchange", "contract", "date",
                                      "open", "high", "low", "close",
                                      "volume", "open_interest"]))
            print(f"  MySQL 读到 {n_all} 行，转换后可用 {n_keep} 行", flush=True)
            return make_frame(pd.concat(frames, ignore_index=True)) if frames else make_frame(
                pd.DataFrame(columns=["symbol", "exchange", "contract", "date",
                                      "open", "high", "low", "close",
                                      "volume", "open_interest"]))
    except Exception as exc:  # noqa: BLE001
        print(f"  MySQL 读取失败（{type(exc).__name__}: {str(exc)[:80]}），改走 Tushare",
              flush=True)
        return None


def tushare_rows(days, pairs: set[str] | None) -> pd.DataFrame:
    pro = F.get_pro()
    frames = []
    for i, day in enumerate(days, 1):
        try:
            raw = C.fetch_day(pro, day.strftime("%Y%m%d"))
        except Exception as exc:  # noqa: BLE001
            print(f"  Tushare {day.date()} 失败：{type(exc).__name__}: {exc}", flush=True)
            continue
        if raw.empty:
            continue
        out = []
        for r in raw.itertuples(index=False):
            code = str(r.ts_code)
            if "." not in code:
                continue
            body, suffix = code.rsplit(".", 1)
            ex = {v: k for k, v in TS_SUFFIX.items()}.get(suffix)
            if ex is None:
                continue
            m = CODE_RE.match(body.upper())
            if not m or m.group(2) == "000":
                continue
            sym = m.group(1)
            if pairs is not None and f"{sym}_{ex}" not in pairs:
                continue
            out.append((sym, ex, code, pd.Timestamp(day),
                        r.open, r.high, r.low, r.close, r.volume, 0))
        if out:
            frames.append(pd.DataFrame(
                out, columns=["symbol", "exchange", "contract", "date",
                              "open", "high", "low", "close",
                              "volume", "open_interest"]))
        if i % 100 == 0:
            print(f"  Tushare 已处理 {i}/{len(days)} 天", flush=True)
    cols = ["symbol", "exchange", "contract", "date", "open", "high",
            "low", "close", "volume", "open_interest"]
    return make_frame(pd.concat(frames, ignore_index=True)) if frames else make_frame(
        pd.DataFrame(columns=cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--source", choices=["auto", "mysql", "tushare"], default="auto")
    ap.add_argument("--symbols", default="", help="逗号分隔 品种_交易所，默认天勤库全部品种")
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--stage-csv", default="",
                    help="下载后先落 CSV（数据库被占用时用），不写库")
    ap.add_argument("--import-csv", default="",
                    help="从之前 stage 的 CSV 直接写库")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.import_csv:
        d = pd.read_csv(args.import_csv)
        d["dt"] = pd.to_datetime(d["dt"])
        d["trading_day"] = pd.to_datetime(d["trading_day"]).dt.date
        print(f"从 {args.import_csv} 读入 {len(d)} 行，INSERT OR IGNORE 写库", flush=True)
        cm = S.connect()
        wcon = cm.__enter__()
        try:
            added = 0
            for (sym, ex), g in d.groupby(["symbol", "exchange"], sort=False):
                added += insert_ignore(wcon, g)
            print(f"实际新增 {added} 行", flush=True)
        finally:
            cm.__exit__(None, None, None)
        return

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()
    cm = S.connect(read_only=True)
    con = cm.__enter__()
    pairs = {f"{r[0]}_{r[1]}" for r in con.execute(
        "SELECT DISTINCT symbol, exchange FROM contract_bars").fetchall()}
    cm.__exit__(None, None, None)
    if args.symbols:
        pairs &= {s.strip() for s in args.symbols.split(",") if s.strip()}

    print(f"补数区间 {start} ~ {end}，目标品种 {len(pairs)} 个，source={args.source}",
          flush=True)

    mysql_df = None
    if args.source in ("auto", "mysql"):
        mysql_df = mysql_rows(start, end, pairs, args.chunksize)
    if mysql_df is None and args.source == "mysql":
        raise RuntimeError("指定了 mysql 但读不到 MySQL 表")
    if mysql_df is None and args.source in ("auto", "tushare"):
        cal = [pd.Timestamp(d).date() for d in S.calendar()
               if start <= pd.Timestamp(d).date() < end]
        if not cal:
            from ttrade.tqdata import trading_days
            idx = pd.bdate_range(start, end)
            cal = trading_days(pd.Series(idx)).dt.date.tolist()
        print(f"  Tushare 兜底：需下载 {len(cal)} 个交易日", flush=True)
        mysql_df = tushare_rows(
            [pd.Timestamp(d) for d in cal if pd.Timestamp(d) < pd.Timestamp(end)],
            pairs)

    if mysql_df is None or mysql_df.empty:
        print("没有可写数据，结束", flush=True)
        return
    print(f"待写入 {len(mysql_df)} 行（INSERT OR IGNORE）", flush=True)
    if args.dry_run:
        print(mysql_df.groupby(mysql_df["dt"].dt.year).size().to_string(), flush=True)
        return
    if args.stage_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.stage_csv)), exist_ok=True)
        mysql_df.to_csv(args.stage_csv, index=False)
        print(f"已暂存到 {args.stage_csv}，等数据库空闲后用 --import-csv 写库", flush=True)
        return

    write_cm = S.connect()
    wcon = write_cm.__enter__()
    try:
        added = 0
        for (sym, ex), g in mysql_df.groupby(["symbol", "exchange"], sort=False):
            n = insert_ignore(wcon, g)
            added += n
        print(f"实际新增 {added} 行", flush=True)
    finally:
        write_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
