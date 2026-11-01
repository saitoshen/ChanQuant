"""宏观数据：抓取、存储、读取。

## 为什么单开一组表，塞进 `edb.duckdb`

EDB 那个库本来就是「基本面/宏观指标」的容器，只是原来的 243 个指标全是**品种
级**的（螺纹的仓单、豆粕的压榨利润）。宏观指标是**市场级**的（SHIBOR、CPI、
美元指数），没有品种维度 —— 所以另开两张表，不跟 `edb_meta` / `edb_series`
混在一起，免得 `edb.varieties()` 之类的接口被空品种污染。

    macro_meta    code 主键 + 中文名 / 频率 / 单位 / 数据源
    macro_series  (code, dt) 主键 + value

## 数据源有两家，因为单靠一家凑不齐 23 个

- **Tushare**：SHIBOR、美债收益率曲线、实际收益率、离岸人民币、标普/纳指/
  恒生/日经、CPI/PPI/M0M1M2/GDP、社融。优点是稳定、字段干净。
- **东方财富**：制造业 PMI（Tushare 那个接口在当前账号上字段对不上）、
  美元指数（Tushare 的 `fx_daily` 没有 UDI）。两家都是公开的行情/宏观接口。

**LPR 两家都没有**。它只在中国货币网的接口里，那个路径已经 404。所以
`lpr_1y` / `lpr_5y` 两个 code 保留了定义但抓不到数据 —— 因子照样注册，
读出来是 NaN，`filter_available()` 会把它们从可用列表里剔掉。**不做假数据**：
拿 SHIBOR 1Y 顶替 LPR 会让「利率因子」看起来齐全，实际是错的。

## PIT：日期存的是**发布日**，不是统计期

CPI 一月的值要到二月十号左右才公布。如果按「统计期」存成 2024-01-31，
回测里一月底就能看到一月的 CPI —— 那是未来函数，而且是最容易忽略的一种。

所以月频/季频指标一律换算成**发布日**再入库：

- PMI：当月最后一天（统计局月末当天发布）
- CPI / PPI / M0M1M2 / 社融：次月 10 日（惯例是 9-15 日之间）
- GDP：季后 20 天
- LPR：当月 20 日

这个换算是**近似**的，误差在几天量级。因子层再统一滞后 1 个交易日，把
「当天发布、当天就能用」这种乐观假设也挡掉。
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import edb as E

SCHEMA_VERSION = "1"

# 单次请求的起始日期。SHIBOR 从 2006 年开始，美债从 1990 年，都够用。
DEFAULT_START = date(2006, 1, 1)

# 增量回看天数：宏观数据会被修订（尤其 GDP、社融），往前多抓一点覆盖住
OVERLAP_DAYS = 400

DDL = [
    """
    CREATE TABLE IF NOT EXISTS macro_meta (
        code   VARCHAR PRIMARY KEY,
        name   VARCHAR,
        freq   VARCHAR,
        unit   VARCHAR,
        source VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS macro_series (
        code  VARCHAR NOT NULL,
        dt    DATE    NOT NULL,
        value DOUBLE,
        PRIMARY KEY (code, dt)
    )
    """,
]

# ------------------------------------------------------------------ 定义
# code -> (中文名, 频率, 单位, 数据源说明)
# 频率：D 日 / M 月 / Q 季
SERIES: dict[str, tuple[str, str, str, str]] = {
    "shibor_on":  ("SHIBOR 隔夜", "D", "%", "tushare.shibor"),
    "shibor_3m":  ("SHIBOR 3M", "D", "%", "tushare.shibor"),
    "lpr_1y":     ("LPR 1 年", "M", "%", "（暂无数据源）"),
    "lpr_5y":     ("LPR 5 年", "M", "%", "（暂无数据源）"),
    "us_y2":      ("美债 2 年", "D", "%", "tushare.us_tycr"),
    "us_y10":     ("美债 10 年", "D", "%", "tushare.us_tycr"),
    "us_10y2y":   ("美债 10Y−2Y", "D", "%", "派生：us_y10 − us_y2"),
    "us_real10":  ("美债实际 10 年", "D", "%", "tushare.us_trycr"),
    "dxy":        ("美元指数", "D", "点", "eastmoney.UDI"),
    "usdcnh":     ("美元兑离岸人民币", "D", "元", "tushare.fx_daily"),
    "spx":        ("标普 500", "D", "点", "tushare.index_global"),
    "ixic":       ("纳斯达克", "D", "点", "tushare.index_global"),
    "hsi":        ("恒生指数", "D", "点", "tushare.index_global"),
    "n225":       ("日经 225", "D", "点", "tushare.index_global"),
    "spx_vol20":  ("标普 20 日波动", "D", "%", "派生：spx 收益的 20 日年化标准差"),
    "cn_pmi_mfg": ("制造业 PMI", "M", "点", "eastmoney.RPT_ECONOMY_PMI"),
    "cn_cpi_yoy": ("CPI 同比", "M", "%", "tushare.cn_cpi"),
    "cn_ppi_yoy": ("PPI 同比", "M", "%", "tushare.cn_ppi"),
    "cn_m1_yoy":  ("M1 同比", "M", "%", "tushare.cn_m"),
    "cn_m2_yoy":  ("M2 同比", "M", "%", "tushare.cn_m"),
    "cn_sf_inc":  ("社融增量", "M", "亿元", "tushare.sf_month"),
    "cn_sf_cum":  ("社融存量", "M", "亿元", "tushare.sf_month"),
    "cn_gdp_yoy": ("GDP 同比", "Q", "%", "tushare.cn_gdp"),
}

CODES = list(SERIES)


# ------------------------------------------------------------------ 存储
def _connect(read_only: bool = False):
    """复用 `edb.connect`：同一个文件，同一套锁重试逻辑。"""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        with E.connect(read_only=read_only) as con:
            if not read_only:
                for ddl in DDL:
                    con.execute(ddl)
            yield con

    return _cm()


def _insert(con, table: str, df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        return
    con.register("_macro_batch", df)
    try:
        con.execute(
            f"INSERT OR REPLACE INTO {table} SELECT "
            + ", ".join(df.columns)
            + " FROM _macro_batch"
        )
    finally:
        con.unregister("_macro_batch")


def upsert_meta(con) -> int:
    df = pd.DataFrame(
        [(c, n, f, u, s) for c, (n, f, u, s) in SERIES.items()],
        columns=["code", "name", "freq", "unit", "source"],
    )
    _insert(con, "macro_meta", df)
    return len(df)


# ------------------------------------------------------------------ 抓取
def _retry(fn, *a, **kw):
    """网络请求重试。宏观接口偶尔 502，不值得为一次抖动丢掉整轮。"""
    last = None
    for i in range(3):
        try:
            return fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < 2:
                time.sleep(1.5 * (i + 1))
    raise last  # type: ignore[misc]


def _span(start: date, end: date, years: int = 3):
    """把区间切成年份块。**Tushare 单次返回上限 2000 行**，日频十年就是
    2500 行 —— 不分块的话拿到的是「最近 2000 行」，前面五年静默消失。
    这个问题不会报错，只会让因子的历史莫名变短。"""
    cur = start
    while cur < end:
        nxt = min(date(cur.year + years, 1, 1) - timedelta(days=1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def _tushare_series(api, code: str) -> pd.Series:
    """从 Tushare 拉一条日频序列。返回 index=date 的 float Series。"""
    s, e = DEFAULT_START, date.today()

    if code in ("shibor_on", "shibor_3m"):
        field = "on" if code == "shibor_on" else "3m"
        parts = [_pick(_retry(api.shibor, start_date=a.strftime("%Y%m%d"),
                              end_date=b.strftime("%Y%m%d")), "date", field)
                 for a, b in _span(s, e)]
        return _concat(parts)

    if code in ("us_y2", "us_y10"):
        field = "y2" if code == "us_y2" else "y10"
        parts = [_pick(_retry(api.us_tycr, start_date=a.strftime("%Y%m%d"),
                              end_date=b.strftime("%Y%m%d")), "date", field)
                 for a, b in _span(s, e)]
        return _concat(parts)

    if code == "us_real10":
        parts = [_pick(_retry(api.us_trycr, start_date=a.strftime("%Y%m%d"),
                              end_date=b.strftime("%Y%m%d")), "date", "y10")
                 for a, b in _span(s, e)]
        return _concat(parts)

    if code in ("usdcnh", "dxy"):
        if code == "dxy":
            return _dxy(api)
        # 美元指数在 Tushare 的外汇表里叫 `USDOLLAR.FXCM`，但它 2023-06 就断更了，
        # 而且量纲是 12947 这种（不是 104 那种 DXY 点数）—— 所以不用它，
        # 改成用六个构成货币按 ICE 官方公式合成，见 `_dxy()`。
        parts = []
        for a, b in _span(s, e):
            df = _retry(api.fx_daily, ts_code="USDCNH.FXCM",
                        start_date=a.strftime("%Y%m%d"),
                        end_date=b.strftime("%Y%m%d"))
            if df is None or len(df) == 0:
                continue
            # 外汇给的是买卖价，取中间价：只取一边会带进半个点差的水平偏移
            mid = (pd.to_numeric(df["bid_close"], errors="coerce")
                   + pd.to_numeric(df["ask_close"], errors="coerce")) / 2.0
            parts.append(_finish(df["trade_date"], mid))
        return _concat(parts)

    if code in ("spx", "ixic", "hsi", "n225"):
        tag = {"spx": "SPX", "ixic": "IXIC", "hsi": "HSI", "n225": "N225"}[code]
        parts = [_pick(_retry(api.index_global, ts_code=tag,
                              start_date=a.strftime("%Y%m%d"),
                              end_date=b.strftime("%Y%m%d")),
                       "trade_date", "close")
                 for a, b in _span(s, e)]
        return _concat(parts)

    if code in ("cn_cpi_yoy", "cn_ppi_yoy", "cn_m1_yoy", "cn_m2_yoy",
                "cn_sf_inc", "cn_sf_cum", "cn_gdp_yoy", "cn_pmi_mfg"):
        return _cn_monthly(api, code)

    return pd.Series(dtype=float)


def _concat(parts: list[pd.Series]) -> pd.Series:
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


# ICE 美元指数公式：六个构成货币的几何加权
#   DXY = 50.14348112 × EURUSD^-0.576 × USDJPY^0.136 × GBPUSD^-0.119
#         × USDCAD^0.091 × USDSEK^0.042 × USDCHF^0.036
# 权重之和正好是 0，常数 50.14 让指数在 1973 年 3 月基准日等于 100。
_DXY_BASKET = [
    ("EURUSD.FXCM", -0.576), ("USDJPY.FXCM", 0.136), ("GBPUSD.FXCM", -0.119),
    ("USDCAD.FXCM", 0.091), ("USDSEK.FXCM", 0.042), ("USDCHF.FXCM", 0.036),
]


def _fx_mid(api, ts_code: str, start: date, end: date) -> pd.Series:
    parts = []
    for a, b in _span(start, end):
        df = _retry(api.fx_daily, ts_code=ts_code,
                    start_date=a.strftime("%Y%m%d"),
                    end_date=b.strftime("%Y%m%d"))
        if df is None or len(df) == 0:
            continue
        mid = (pd.to_numeric(df["bid_close"], errors="coerce")
               + pd.to_numeric(df["ask_close"], errors="coerce")) / 2.0
        parts.append(_finish(df["trade_date"], mid))
    return _concat(parts)


def _dxy(api) -> pd.Series:
    """用六个构成货币合成美元指数。

    为什么不用现成的：Tushare 的 `USDOLLAR.FXCM` 2023-06 断更，量纲还是
    12947 这种；东方财富的 `100.UDI` 网络不稳（代理经常断）。自己合成
    反而最稳 —— 六个货币对全都有 2006 年以来的日线，公式是公开的。
    """
    frames = {}
    for code, w in _DXY_BASKET:
        s = _fx_mid(api, code, DEFAULT_START, date.today())
        if len(s) == 0:
            return pd.Series(dtype=float)
        frames[w] = s
    df = pd.DataFrame({str(w): s for w, s in frames.items()}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    acc = pd.Series(np.log(50.14348112), index=df.index)
    for w, s in frames.items():
        acc = acc + float(w) * np.log(df[str(w)])
    return np.exp(acc)


def _pick(df, date_col: str, value_col: str) -> pd.Series:
    if df is None or len(df) == 0 or value_col not in df.columns:
        return pd.Series(dtype=float)
    return _finish(df[date_col], pd.to_numeric(df[value_col], errors="coerce"))


def _finish(raw_dates, values) -> pd.Series:
    d = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
    s = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(d))
    s = s[~s.index.isna()].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def _month_end(month: str) -> pd.Timestamp:
    return pd.Timestamp(month + "01") + pd.offsets.MonthEnd(0)


def _cn_monthly(api, code: str) -> pd.Series:
    """中国月频/季频指标：统一换算成**发布日**再返回。

    见模块开头的 PIT 说明。这里不追求精确到天，只要不早于真实发布日太多。
    """
    start, end = "200601", pd.Timestamp.today().strftime("%Y%m")

    if code == "cn_pmi_mfg":
        df = _retry(api.cn_pmi, start_m=start, end_m=end,
                    fields="month,PMI010000")
        if df is None or len(df) == 0 or "PMI010000" not in df.columns:
            return pd.Series(dtype=float)
        pub = df["month"].map(_month_end)
        return _finish(pub, pd.to_numeric(df["PMI010000"], errors="coerce"))

    if code == "cn_cpi_yoy":
        df = _retry(api.cn_cpi, start_m=start, end_m=end)
        return _monthly_pub(df, "month", "nt_yoy", lag_months=1)

    if code == "cn_ppi_yoy":
        df = _retry(api.cn_ppi, start_m=start, end_m=end)
        return _monthly_pub(df, "month", "ppi_yoy", lag_months=1)

    if code in ("cn_m1_yoy", "cn_m2_yoy"):
        field = "m1_yoy" if code == "cn_m1_yoy" else "m2_yoy"
        df = _retry(api.cn_m, start_m=start, end_m=end)
        return _monthly_pub(df, "month", field, lag_months=1)

    if code in ("cn_sf_inc", "cn_sf_cum"):
        field = "inc_month" if code == "cn_sf_inc" else "stk_endval"
        df = _retry(api.sf_month, start_m=start, end_m=end)
        return _monthly_pub(df, "month", field, lag_months=1)

    if code == "cn_gdp_yoy":
        y = int(start[:4])
        start_q = f"{y}Q1"
        end_q = f"{pd.Timestamp.today().year}Q4"
        df = _retry(api.cn_gdp, start_q=start_q, end_q=end_q)
        if df is None or len(df) == 0:
            return pd.Series(dtype=float)
        pub = df["quarter"].map(_quarter_pub)
        return _finish(pub, pd.to_numeric(df["gdp_yoy"], errors="coerce"))

    return pd.Series(dtype=float)


def _monthly_pub(df, month_col: str, value_col: str,
                 lag_months: int = 1) -> pd.Series:
    if df is None or len(df) == 0 or value_col not in df.columns:
        return pd.Series(dtype=float)
    m = df[month_col].astype(str)
    pub = pd.to_datetime(m + "01", errors="coerce") + pd.offsets.MonthBegin(lag_months)
    pub = pub + pd.Timedelta(days=9)      # 次月 10 号左右发布
    return _finish(pub, pd.to_numeric(df[value_col], errors="coerce"))


def _quarter_pub(q) -> pd.Timestamp:
    """'2024Q1' → 2024-04-20（季后 20 天，GDP 初值惯例在 4 月中下旬）。"""
    try:
        s = str(q)
        y, k = int(s[:4]), int(s[-1])
        end = pd.Timestamp(year=y, month=3 * k, day=1) + pd.offsets.MonthEnd(0)
        return end + pd.Timedelta(days=20)
    except Exception:  # noqa: BLE001
        return pd.NaT


_EM_HEAD = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Referer": "https://data.eastmoney.com/",
}


def _em_kline(secid: str, start: date) -> pd.Series:
    """东方财富日线（前收价）。用于美元指数这种 Tushare 没有的序列。"""
    import requests

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    q = {
        "secid": secid, "fields1": "f1,f2,f3",
        "fields2": "f51,f53", "klt": "101", "fqt": "1",
        "beg": start.strftime("%Y%m%d"),
        "end": date.today().strftime("%Y%m%d"),
    }
    r = _retry(requests.get, url, params=q, headers=_EM_HEAD, timeout=20)
    j = r.json() or {}
    data = j.get("data") or {}
    rows = data.get("klines") or []
    if not rows:
        return pd.Series(dtype=float)
    dates, vals = [], []
    for line in rows:
        p = line.split(",")
        if len(p) < 2:
            continue
        dates.append(p[0])
        vals.append(pd.to_numeric(p[1], errors="coerce"))
    return _finish(dates, pd.Series(vals, dtype=float))


def _em_report(report: str, cols: str, date_col: str) -> pd.DataFrame:
    """东方财富数据中心报表（宏观月度数据）。"""
    import requests

    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    q = {
        "reportName": report, "columns": cols, "pageNumber": 1,
        "pageSize": 500, "sortColumns": date_col, "sortTypes": "-1",
    }
    r = _retry(requests.get, url, params=q, headers=_EM_HEAD, timeout=25)
    j = r.json() or {}
    data = (j.get("result") or {}).get("data") or []
    return pd.DataFrame(data)


def _em_pmi(start: date) -> pd.Series:
    df = _em_report("RPT_ECONOMY_PMI", "REPORT_DATE,TIME,MAKE_INDEX", "REPORT_DATE")
    if df is None or df.empty or "MAKE_INDEX" not in df.columns:
        return pd.Series(dtype=float)
    # 东财给的 REPORT_DATE 是「统计月」的 1 号；PMI 是当月最后一天发布
    pub = pd.to_datetime(df["REPORT_DATE"], errors="coerce") + pd.offsets.MonthEnd(0)
    return _finish(pub, pd.to_numeric(df["MAKE_INDEX"], errors="coerce"))


def fetch(code: str, api=None) -> pd.Series:
    """拉一条宏观序列。取不到就返回空 Series（不抛）。"""
    if code in ("lpr_1y", "lpr_5y"):
        return pd.Series(dtype=float)
    try:
        if code == "cn_pmi_mfg":
            return _em_pmi(DEFAULT_START)
        if code == "us_10y2y":
            # 派生序列不单独抓，由 update() 在最后合成
            return pd.Series(dtype=float)
        if code == "spx_vol20":
            return pd.Series(dtype=float)
        if api is None:
            from .futures import get_pro

            api = get_pro()
        return _tushare_series(api, code)
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)


def _derive(store: dict[str, pd.Series]) -> None:
    """合成派生序列：10Y−2Y 利差、标普 20 日波动。"""
    y2, y10 = store.get("us_y2"), store.get("us_y10")
    if y2 is not None and y10 is not None and len(y2) and len(y10):
        j = pd.concat([y2.rename("a"), y10.rename("b")], axis=1).dropna()
        if len(j):
            store["us_10y2y"] = (j["b"] - j["a"]).sort_index()

    spx = store.get("spx")
    if spx is not None and len(spx) > 25:
        r = spx.pct_change()
        store["spx_vol20"] = (r.rolling(20, min_periods=15).std(ddof=1)
                              * (244.0 ** 0.5) * 100.0).dropna()


def update(api=None, codes: list[str] | None = None,
           incremental: bool = True, replace: bool = False,
           progress=None, cancel=None) -> dict:
    """抓一轮并写库。返回统计字典。

    增量口径：库里最后一天往前推 `OVERLAP_DAYS` 天再抓一次。宏观数据会被修订
    （GDP 初值/终值、社融），只看「有没有新日期」会把修订漏掉。

    `replace=True` 先把这些 code 的旧值删干净再写。**换数据源时必须用它**：
    美元指数原来走 FXCM（量纲 12947），后来改成六货币合成（量纲 102），
    不清空的话两段数据混在一条序列里，分位会跳成噪声。
    """
    codes = [c for c in (codes or CODES) if c in SERIES]
    stats = {"codes": len(codes), "rows": 0, "ok": 0, "empty": 0, "failed": 0,
             "detail": {}}

    last: dict[str, date] = {}
    if incremental and not replace:
        try:
            with _connect(read_only=True) as con:
                for c, dt in con.execute(
                        "SELECT code, max(dt) FROM macro_series GROUP BY 1").fetchall():
                    d = E._to_date(dt)
                    if d:
                        last[str(c)] = d
        except Exception:  # noqa: BLE001
            last = {}

    store: dict[str, pd.Series] = {}
    for i, code in enumerate(codes):
        if cancel is not None and cancel():
            break
        if progress is not None:
            progress(i, len(codes), f"宏观 {i + 1}/{len(codes)}　{SERIES[code][0]}")
        s = fetch(code, api=api)
        if s is None or len(s) == 0:
            stats["empty"] += 1
            stats["detail"][code] = "无数据"
            continue
        cut = last.get(code)
        if cut is not None:
            s = s[s.index >= pd.Timestamp(cut) - pd.Timedelta(days=OVERLAP_DAYS)]
        if len(s) == 0:
            stats["empty"] += 1
            continue
        store[code] = s
        stats["ok"] += 1
        stats["detail"][code] = f"{len(s)} 行"

    _derive(store)
    # 派生序列在循环里没有对应 code，补记一笔，否则统计里永远是「缺两个」
    for code in ("us_10y2y", "spx_vol20"):
        if code in store and stats["detail"].get(code) == "无数据":
            stats["detail"][code] = f"{len(store[code])} 行（派生）"
            stats["ok"] += 1
            stats["empty"] = max(0, stats["empty"] - 1)

    if store:
        with _connect() as con:
            upsert_meta(con)
            if replace:
                con.execute(
                    "DELETE FROM macro_series WHERE code IN ("
                    + ",".join("?" * len(store)) + ")", list(store))
            for code, s in store.items():
                df = pd.DataFrame({
                    "code": code,
                    "dt": [d.date() for d in pd.DatetimeIndex(s.index)],
                    "value": s.to_numpy(dtype=float),
                }).dropna(subset=["value"])
                if df.empty:
                    continue
                _insert(con, "macro_series", df)
                stats["rows"] += len(df)
            con.execute(
                "INSERT OR REPLACE INTO edb_meta_kv VALUES ('macro_last_update', ?)",
                [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
            )

    if progress is not None:
        progress(len(codes), len(codes), "宏观完成")
    return stats


# ------------------------------------------------------------------ 读取
def load(codes: list[str] | None = None, wide: bool = True,
         start=None, end=None) -> pd.DataFrame:
    """读宏观序列。`wide=True` 返回 index=dt、columns=code 的宽表。"""
    if not E.exists():
        return pd.DataFrame()
    where, args = [], []
    if codes:
        where.append("code IN (" + ",".join("?" * len(codes)) + ")")
        args += list(codes)
    if start is not None:
        where.append("dt >= ?")
        args.append(pd.Timestamp(start).date())
    if end is not None:
        where.append("dt <= ?")
        args.append(pd.Timestamp(end).date())
    sql = "SELECT code, dt, value FROM macro_series"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY dt, code"
    try:
        with _connect(read_only=True) as con:
            df = con.execute(sql, args).df()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["dt"])
    if not wide:
        return df
    out = df.pivot_table(index="dt", columns="code", values="value", aggfunc="last")
    out.columns.name = None
    return out.sort_index()


def series(code: str) -> pd.Series:
    df = load(codes=[code], wide=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("dt")["value"].astype(float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def stats() -> dict:
    if not E.exists():
        return {"exists": False, "codes": 0, "rows": 0, "first": None, "last": None}
    try:
        with _connect(read_only=True) as con:
            n = con.execute(
                "SELECT count(DISTINCT code) FROM macro_series").fetchone()[0]
            rows, first, last = con.execute(
                "SELECT count(*), min(dt), max(dt) FROM macro_series").fetchone()
    except Exception:  # noqa: BLE001
        return {"exists": True, "codes": 0, "rows": 0, "first": None, "last": None}
    return {"exists": True, "codes": int(n or 0), "rows": int(rows or 0),
            "first": first, "last": last}


def coverage() -> pd.DataFrame:
    """每个 code 有多少行、覆盖到哪天。界面和体检都用它。"""
    rows = []
    if not E.exists():
        return pd.DataFrame(columns=["code", "name", "freq", "n", "first", "last"])
    try:
        with _connect(read_only=True) as con:
            got = {str(c): (int(n), a, b) for c, n, a, b in con.execute(
                "SELECT code, count(*), min(dt), max(dt) "
                "FROM macro_series GROUP BY 1").fetchall()}
    except Exception:  # noqa: BLE001
        got = {}
    for code, (name, freq, _u, _s) in SERIES.items():
        n, a, b = got.get(code, (0, None, None))
        rows.append({"code": code, "name": name, "freq": freq,
                     "n": n, "first": a, "last": b})
    return pd.DataFrame(rows)
