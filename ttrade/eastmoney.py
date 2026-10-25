"""东方财富美股行情下载器。

数据来源：东方财富 push2his 接口（美股日线，支持前/后复权）。
回测统一使用后复权（fqt=2），保证历史收益率连续、不出现负价格。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import requests

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
SEARCH_URL = "https://searchapi.eastmoney.com/api/suggest/get"
SEARCH_TOKEN = "D43BF722C8E33BDC906FB84D85E326E8"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}

# klines 每行字段顺序：日期,开,收,高,低,量,额
COLS = ["date", "open", "close", "high", "low", "volume", "amount"]


@dataclass
class Security:
    symbol: str
    name: str
    secid: str  # 形如 105.AAPL
    exchange: str


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def resolve_secid(symbol: str, session: requests.Session | None = None) -> Security | None:
    """通过搜索接口把股票代码解析成东财 secid（105=NASDAQ 106=NYSE 107=AMEX）。"""
    s = session or _session()
    try:
        r = s.get(
            SEARCH_URL,
            params={"input": symbol, "type": 14, "token": SEARCH_TOKEN, "count": 10},
            timeout=15,
        )
        payload = r.json()
    except Exception:
        return None

    rows = (payload.get("QuotationCodeTable") or {}).get("Data") or []
    target = symbol.upper().replace(".", "").replace("-", "")
    for row in rows:
        if row.get("Classify") != "UsStock":
            continue
        code = (row.get("Code") or "").upper()
        if code.replace(".", "").replace("-", "") != target:
            continue
        quote_id = row.get("QuoteID") or ""
        if "." not in quote_id:
            continue
        return Security(
            symbol=symbol.upper(),
            name=row.get("Name") or symbol.upper(),
            secid=quote_id,
            exchange=row.get("JYS") or "",
        )
    return None


def fetch_daily(
    secid: str,
    start: str | None = None,
    end: str = "20500101",
    adjust: int = 2,
    session: requests.Session | None = None,
    retries: int = 5,
) -> pd.DataFrame:
    """拉取日线。

    adjust: 0=不复权 1=前复权 2=后复权
    """
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": 101,
        "fqt": adjust,
        "beg": start or "0",
        "end": end,
        "lmt": 10000,
    }

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            s = session or _session()
            r = s.get(KLINE_URL, params=params, timeout=40)
            payload = r.json()
            data = payload.get("data")
            if not data or not data.get("klines"):
                raise ValueError(f"空数据: {payload.get('rc')} {secid}")
            rows = [line.split(",") for line in data["klines"]]
            df = pd.DataFrame(rows, columns=COLS)
            for col in COLS[1:]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["open", "close", "high", "low"])
            df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
            df.attrs["name"] = data.get("name", "")
            df.attrs["secid"] = secid
            return df
        except Exception as exc:  # 网络抖动 / 代理断开，退避后重建连接
            last_err = exc
            session = None
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"下载失败 {secid}: {last_err}")


def download_universe(
    symbols: Iterable[str],
    out_dir: str,
    years: int = 20,
    adjust: int = 2,
    sleep: float = 0.15,
    verbose: bool = True,
) -> list[dict]:
    """批量下载并落地 CSV，返回清单。"""
    os.makedirs(out_dir, exist_ok=True)
    start = (date.today() - timedelta(days=int(years * 365.25) + 30)).strftime("%Y%m%d")
    session = _session()
    manifest: list[dict] = []

    for sym in symbols:
        try:
            sec = resolve_secid(sym, session)
            if sec is None:
                if verbose:
                    print(f"  [跳过] {sym}: 未找到美股代码")
                continue
            df = fetch_daily(sec.secid, start=start, adjust=adjust, session=session)
            if len(df) < 250:
                if verbose:
                    print(f"  [跳过] {sym}: 数据不足({len(df)}行)")
                continue
            path = os.path.join(out_dir, f"{sec.symbol.replace('.', '_')}.csv")
            out = df.copy()
            out.insert(0, "symbol", sec.symbol)
            out.to_csv(path, index=False, encoding="utf-8-sig")
            manifest.append(
                {
                    "symbol": sec.symbol,
                    "name": sec.name,
                    "exchange": sec.exchange,
                    "secid": sec.secid,
                    "rows": len(df),
                    "start": df["date"].iloc[0].strftime("%Y-%m-%d"),
                    "end": df["date"].iloc[-1].strftime("%Y-%m-%d"),
                    "file": os.path.basename(path),
                }
            )
            if verbose:
                print(
                    f"  [OK] {sec.symbol:<6} {sec.name:<12} {len(df):>5}行 "
                    f"{df['date'].iloc[0]:%Y-%m-%d} ~ {df['date'].iloc[-1]:%Y-%m-%d}"
                )
        except Exception as exc:
            if verbose:
                print(f"  [失败] {sym}: {exc}")
        time.sleep(sleep)

    if manifest:
        mpath = os.path.join(out_dir, "_manifest.json")
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def load_local(symbol: str, data_dir: str, clean: bool = True) -> pd.DataFrame:
    """读取本地缓存的 CSV（默认做数据质量修复）。"""
    path = os.path.join(data_dir, f"{symbol.replace('.', '_')}.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df = df.set_index("date")

    if clean:
        from ttrade.quality import clean_prices

        df = clean_prices(df)
    return df
