#!/usr/bin/env python
"""Dukascopy 汇率/指数日内外盘数据 → 5m K 线 + 5m 因子。

为什么单独做这一套
------------------
系统里 `dxy` / `usdcnh` 是 Tushare 的**日频**序列（`macro_series` 表），铺到 5m
只能按自然日 as-of（`factors_macro._asof(lag_days=1)`），于是：

  · 周一的 5m bar 用「周日那一行」，周二才用上「周一那一行」；
  · 周五那一行因为被周日行顶掉，几乎永远进不了因子；
  · Tushare 这几周还整段缺周五（2026-10 起），缺口会静默穿透到因子。

Dukascopy 给的是**分钟级原始成交**，可以自己聚合成 5m，滞后只剩一分钟级别，
不再需要「日频压日频」那套 as-of 规则。

接口（2026-11 实测可用，不是老的 .bi5 二进制）
----------------------------------------------
    https://jetta.dukascopy.com/v1/candles/minute/{code}/BID/{Y}/{M}/{D}

`{code}` 是 `/v1/instruments` 里的 *code* 字段（斜杠换成连字符）：
    USD/CNH          → USD-CNH          （priceScale 5，multiplier 1e-5）
    DOLLAR.IDX/USD   → DOLLAR.IDX-USD   （priceScale 3，multiplier 1e-3）

必须带 `Referer: https://widgets.dukascopy.com/en/historical-data-export`，
否则未缓存的响应会被挡（429）。

响应是**增量编码**，照 `dukascopy-node` 的 decode 逻辑还原：

    {"timestamp": <基准 ms UTC>, "multiplier": 1e-5, "shift": 60000,
     "open"/"high"/"low"/"close": 第一根的**真实价**,
     "times": [相对上一根的**bar 数**步长],
     "opens"/"highs"/"lows"/"closes": [相对上一根的**价格增量**(units)],
     "volumes": [...]}

  · 价格是 units 累加：`units += delta`，真价 = `units * multiplier`
  · 时间也是累加：`t += times[i] * shift`
  · `times[i]` 是「跳过了几根」，不是毫秒偏移；缺成交的分钟被压缩掉了

时区（用户特别关心的一点）
--------------------------
Dukascopy 的时间戳是 **UTC 毫秒**，不是伦敦本地时间，所以：

    北京时间 = UTC + 8    ← 永远 +8，冬夏令时都一样

夏令时影响的是「伦敦 09:00 对应北京几点」，不影响时间戳本身。只有当你手里的
CSV 是用**伦敦本地时间**标注的（别的数据渠道常见），才需要按 3 月最后一个
周日 / 10 月最后一个周日的规则在 +8 和 +7 之间切换 —— 这就是 `--tz london`。

中国不实行夏令时，`Asia/Shanghai` 永远 UTC+8。

用法
----
    python _tools/duka_fx/duka_fx.py fetch  --start 2026-01-01 --end 2026-11-17
    python _tools/duka_fx/duka_fx.py factor
    python _tools/duka_fx/duka_fx.py all    --start 2026-01-01

数据落 `data_duka/`，原始 JSON 缓存在 `data_duka/_cache/`（断点续传，重跑不重复请求）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "data_duka")
CACHE_DIR = os.path.join(OUT_DIR, "_cache")

API = "https://jetta.dukascopy.com/v1/candles/minute/{code}/BID/{y}/{m}/{d}"
HEADERS = {
    "Referer": "https://widgets.dukascopy.com/en/historical-data-export",
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/155.0.0.0 Safari/537.36"),
}

# ── 品种 ────────────────────────────────────────────────────────────────────
# 名字 → (Dukascopy code, 落盘简称)
SYMBOLS = {
    "usdcnh": ("USD-CNH", "USDCNH"),
    "dxy": ("DOLLAR.IDX-USD", "DOLLARIDX"),
}

BAR = "5min"
BARS_PER_DAY = 288          # 24h / 5min


# ── 下载 ────────────────────────────────────────────────────────────────────
def _cache_path(code: str, y: int, m: int, d: int) -> str:
    p = os.path.join(CACHE_DIR, code.replace("/", "_"), "%04d" % y, "%02d" % m)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, "%02d.json" % d)


def fetch_day(code: str, day: date, retries: int = 4, timeout: int = 45) -> dict | None:
    """取一天的一分钟数据（原始 JSON）。有缓存直接读。

    返回 None 表示这一天确实没有数据（周末休市 / 尚未发布 / 404）；
    抛异常表示网络问题，由调用方决定要不要继续。
    """
    cp = _cache_path(code, day.year, day.month, day.day)
    if os.path.exists(cp):
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001 —— 缓存坏了就重下
            os.remove(cp)

    url = API.format(code=code, y=day.year, m=day.month, d=day.day)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            payload = json.loads(raw.decode("utf-8"))
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            return payload
        except urllib.error.HTTPError as e:
            body = e.read()[:200].decode("utf-8", "ignore")
            if e.code in (400, 404):
                # "Unknown instrument" / 该日无数据：不是网络问题，别重试
                return None
            last = "HTTP %d %s" % (e.code, body)
        except Exception as e:  # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, e)
        time.sleep(1.5 * (i + 1))       # 退避；代理链路会间歇性超时
    raise RuntimeError("%s %s 取数失败：%s" % (code, day, last))


def decode_candles(payload: dict) -> pd.DataFrame:
    """增量编码 → (utc_ms, open, high, low, close, volume)。

    严格照 `dukascopy-node` 的 normaliseCandles：价格 units 累加、时间按 bar
    数步进。写错这里会得到一条**看着像价格、其实是乱码**的曲线，所以下面
    每一句都对着源码来，不做「优化」。
    """
    times = payload.get("times") or []
    n = len(times)
    if n == 0:
        return pd.DataFrame(columns=["utc_ms", "open", "high", "low", "close", "volume"])

    mult = float(payload["multiplier"])
    shift = int(payload["shift"])

    ou = round(float(payload["open"]) / mult)
    hu = round(float(payload["high"]) / mult)
    lu = round(float(payload["low"]) / mult)
    cu = round(float(payload["close"]) / mult)

    t = int(payload["timestamp"])
    opens = payload["opens"]
    highs = payload["highs"]
    lows = payload["lows"]
    closes = payload["closes"]
    vols = payload.get("volumes") or [0.0] * n

    rows = []
    for i in range(n):
        t += int(times[i]) * shift
        ou += int(opens[i])
        hu += int(highs[i])
        lu += int(lows[i])
        cu += int(closes[i])
        rows.append((t, ou * mult, hu * mult, lu * mult, cu * mult, float(vols[i])))
    return pd.DataFrame(rows, columns=["utc_ms", "open", "high", "low", "close", "volume"])


def fetch_range(name: str, start: date, end: date, workers: int = 6,
                retries: int = 3, timeout: int = 25) -> pd.DataFrame:
    """批量取一段日期，返回 1 分钟帧（index = UTC 毫秒）。"""
    code, _ = SYMBOLS[name]
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    print("  %s（%s）：%d 天，并发 %d" % (name, code, len(days), workers))

    frames, fails, empty = [], [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_day, code, d, retries, timeout): d for d in days}
        done = 0
        for fu in as_completed(futs):
            d = futs[fu]
            done += 1
            try:
                p = fu.result()
            except Exception as e:  # noqa: BLE001
                fails.append((d, str(e)))
                continue
            if p is None:
                empty += 1
                continue
            f = decode_candles(p)
            if len(f):
                frames.append(f)
            if done % 20 == 0 or done == len(days):
                print("    %d/%d  已取 %d 天  失败 %d  %ds"
                      % (done, len(days), len(frames), len(fails), time.time() - t0),
                      flush=True)

    if fails:
        print("  ! %d 天失败（前 3 条）：" % len(fails))
        for d, msg in fails[:3]:
            print("     %s  %s" % (d, msg))
    if not frames:
        return pd.DataFrame(columns=["utc_ms", "open", "high", "low", "close", "volume"])

    df = pd.concat(frames, ignore_index=True).drop_duplicates("utc_ms").sort_values("utc_ms")
    print("  1m 合计 %d 根：%s → %s（无数据 %d 天）"
          % (len(df), pd.to_datetime(df["utc_ms"].iloc[0], unit="ms", utc=True),
             pd.to_datetime(df["utc_ms"].iloc[-1], unit="ms", utc=True), empty))
    return df


# ── 时区 ────────────────────────────────────────────────────────────────────
def to_beijing(ms: pd.Series, tz: str = "utc") -> pd.DatetimeIndex:
    """UTC 毫秒 → naive 北京时间（与本系统其它表的 datetime 同口径）。

    `tz="utc"`    ：Dukascopy 原生口径，固定 +8。
    `tz="london"` ：数据是**伦敦本地时间**标注时才用；由 zoneinfo 自动处理
                    夏令时（3 月最后一个周日 01:00 UTC 起 +7，10 月最后一个
                    周日 01:00 UTC 起 +8）。中国无夏令时，目标侧恒为 UTC+8。
    """
    utc = pd.to_datetime(pd.Series(ms).astype("int64"), unit="ms", utc=True)
    if tz == "london":
        naive = utc.dt.tz_localize(None)          # 当作伦敦墙上时间
        lon = naive.dt.tz_localize("Europe/London", ambiguous="NaT",
                                   nonexistent="shift_forward")
        return pd.DatetimeIndex(lon.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None))
    return pd.DatetimeIndex(utc.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None))


# ── 聚合 ────────────────────────────────────────────────────────────────────
def to_5m(df1m: pd.DataFrame, tz: str = "utc") -> pd.DataFrame:
    """1 分钟 → 5 分钟（北京时间索引）。

    5 分钟网格在 UTC 与北京时间下是**同一套**（+8h 是 5 分钟的整数倍），所以
    先把时间戳换算到北京时间再 resample 不会错位；反过来先 resample 再平移也
    一样，这里选前者，索引直接就是系统口径。
    """
    if not len(df1m):
        return pd.DataFrame()
    out = df1m.copy()
    out["dt"] = to_beijing(out["utc_ms"], tz)
    out = out.set_index("dt").drop(columns=["utc_ms"]).sort_index()
    g = out.resample(BAR, label="left", closed="left")
    bars = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "n_min": g["close"].count(),
    }).dropna(subset=["open", "close"])
    bars["n_min"] = bars["n_min"].astype(int)
    return bars


# ── 因子 ────────────────────────────────────────────────────────────────────
def _span_ok(idx: pd.DatetimeIndex, n: int, tol: float = 1.6) -> pd.Series:
    """第 i 根往前 n 根之间，时间跨度是否「正常」（没跨周末/长假）。

    跨缺口时 `close/close.shift(n)` 算出来的是**周末累计变动**，当成 5 分钟
    变动喂给模型就是污染。宁可置 NaN，也不给一个假的短周期信号。
    """
    step = pd.Timedelta(BAR)
    d = pd.Series(idx, index=idx).diff(n)
    return (d <= step * n * tol) & d.notna()


def build_factors(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """一条 5m 序列 → 一组 5m 因子（全部只用过去，PIT）。"""
    c = bars["close"].astype(float)
    idx = bars.index
    out = pd.DataFrame(index=idx)

    for n, tag in ((1, "1"), (3, "3"), (6, "6"), (12, "12"), (48, "48")):
        v = c / c.shift(n) - 1.0
        out["%s_chg_%s" % (prefix, tag)] = v.where(_span_ok(idx, n))

    # 相对 48 根（4 小时）的 z：均值/标准差都用过去，shift(1) 把当根排除
    m = c.rolling(48, min_periods=36).mean().shift(1)
    s = c.rolling(48, min_periods=36).std().shift(1)
    out["%s_z_48" % prefix] = ((c - m) / s.replace(0, np.nan)).where(_span_ok(idx, 1))

    # 48 根已实现波动（年化：一天 288 根，一年 252 天）
    r1 = (c / c.shift(1) - 1.0).where(_span_ok(idx, 1))
    out["%s_rv_48" % prefix] = r1.rolling(48, min_periods=36).std() * np.sqrt(
        BARS_PER_DAY * 252)

    # 扩张分位（与 factors_macro 的口径一致：只用过去，不回看未来）
    out["%s_pct" % prefix] = c.expanding(min_periods=BARS_PER_DAY * 20).rank(pct=True)

    return out


# ── 主流程 ──────────────────────────────────────────────────────────────────
def save_bars(name: str, bars: pd.DataFrame) -> str:
    """按时间索引**增量合并**落盘。

    🔴 不能直接覆盖：`fetch` 每次只处理 `--start/--end` 这一段，覆盖写会把之前
    抓好的年份整段抹掉（2026-11-18 实测：补 2020 年那一次，把已有的 2026 数据
    全冲没了）。缓存里明明还有，但 parquet 里已经没了 —— 这种丢法没有任何报错。
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "%s_5m.parquet" % SYMBOLS[name][1])
    merged = bars
    if os.path.exists(p):
        old = pd.read_parquet(p)
        keep = old[~old.index.isin(bars.index)]
        merged = pd.concat([keep, bars]).sort_index()
        if len(merged) > len(bars):
            print("    （并入已有 %d 根，合计 %d 根）" % (len(keep), len(merged)))
    merged.to_parquet(p)
    print("  → %s  %d 根  %s → %s"
          % (p, len(merged), merged.index[0], merged.index[-1]))
    return p


def cmd_fetch(args):
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    for name in args.symbols:
        print("[fetch] %s" % name)
        df1 = fetch_range(name, start, end, workers=args.workers,
                          retries=args.retries, timeout=args.timeout)
        if not len(df1):
            print("  ! 没有取到数据")
            continue
        bars = to_5m(df1, args.tz)
        save_bars(name, bars)


def cmd_factor(args):
    start = args.start
    frames = []
    for name in args.symbols:
        p = os.path.join(OUT_DIR, "%s_5m.parquet" % SYMBOLS[name][1])
        if not os.path.exists(p):
            print("  ! 缺少 %s，先跑 fetch" % p)
            continue
        bars = pd.read_parquet(p)
        frames.append(build_factors(bars, "fx_" + name))
    if not frames:
        return
    f = pd.concat(frames, axis=1).sort_index()
    if start:
        f = f[f.index >= pd.Timestamp(start)]
    p = os.path.join(OUT_DIR, "fx_5m_factors.parquet")
    f.to_parquet(p)
    print("[factor] → %s  %d 根 × %d 列  %s → %s"
          % (p, len(f), f.shape[1], f.index[0], f.index[-1]))
    print("  列：%s" % ", ".join(f.columns))
    print("  覆盖率：")
    for col in f.columns:
        print("    %-22s %5.1f%%" % (col, 100.0 * f[col].notna().mean()))


def cmd_rebuild(args):
    """只用本地缓存重建 parquet —— **一个请求都不发**。

    链路是间歇性的（同一批请求里既有 200 也有 502），下载和落盘必须解耦：
    先让 `fetch` 尽量把 JSON 攒进缓存，攒不动的部分留着，下次再补；只要缓存
    在，重建就是纯本地操作，随时可以重跑。
    """
    import glob as _glob

    for name in args.symbols:
        code, _ = SYMBOLS[name]
        base = os.path.join(CACHE_DIR, code.replace("/", "_"))
        files = sorted(_glob.glob(os.path.join(base, "*", "*", "*.json")))
        if not files:
            print("  ! %s 没有缓存" % name)
            continue
        frames = []
        for p in files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    frames.append(decode_candles(json.load(f)))
            except Exception as e:  # noqa: BLE001
                print("  ! 跳过坏缓存 %s：%s" % (p, e))
        df1 = pd.concat([f for f in frames if len(f)], ignore_index=True)
        df1 = df1.drop_duplicates("utc_ms").sort_values("utc_ms")
        print("[rebuild] %s：缓存 %d 天 → 1m %d 根" % (name, len(files), len(df1)))
        bars = to_5m(df1, args.tz)
        save_bars(name, bars)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dukascopy 汇率 → 5m 因子")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--symbols", nargs="+", default=list(SYMBOLS),
                       choices=list(SYMBOLS))
        p.add_argument("--tz", default="utc", choices=["utc", "london"],
                       help="utc=Dukascopy 原生（固定+8）；london=数据是伦敦本地时间")
        p.add_argument("--start", default=None)

    f = sub.add_parser("fetch", help="下载 1m 并聚合成 5m")
    common(f)
    f.add_argument("--end", default=None)
    f.add_argument("--workers", type=int, default=6)
    f.add_argument("--retries", type=int, default=3)
    f.add_argument("--timeout", type=int, default=25)
    f.set_defaults(func=cmd_fetch, start="2026-01-01")

    g = sub.add_parser("factor", help="用本地 5m 生成因子")
    common(g)
    g.set_defaults(func=cmd_factor)

    b = sub.add_parser("rebuild", help="只用本地缓存重建 5m（不发请求）")
    common(b)
    b.set_defaults(func=cmd_rebuild)

    a = sub.add_parser("all", help="fetch + factor")
    common(a)
    a.add_argument("--end", default=None)
    a.add_argument("--workers", type=int, default=6)
    a.add_argument("--retries", type=int, default=3)
    a.add_argument("--timeout", type=int, default=25)
    a.set_defaults(func=lambda x: (cmd_fetch(x), cmd_factor(x)), start="2026-01-01")

    args = ap.parse_args(argv)
    # `start` 只对 fetch/all 有默认值（用 set_defaults 设的）；factor/rebuild
    # 必须保持 None —— 否则会把已生成的全历史因子悄悄截掉，而且截得很自然：
    # 2026-11-18 实测，factor 默认带着 "2026-01-01" 过滤，6 年数据只剩 11 个月。
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
