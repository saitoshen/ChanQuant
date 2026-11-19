"""美债 10 年实际利率：直接从美国财政部取（不经 Tushare）。

## 为什么自己取

财政部是**原始出处**，Tushare 的 `us_trycr` 就是转发它 —— 逐值实测完全一致
（2026-11：11-06 2.63 / 11-09 2.66 / 11-12 2.86 / 11-13 2.91 / 11-16 2.92 /
11-17 2.89）。但**发布晚一天**：

    2026-11-19 08:50  Tushare 最新 = 11-17 (2.89)
                     财政部最新 = 11-18 (2.88)

财政部每天约 15:50 UTC（北京 23:50）更新当天收盘。所以美债这一路比 Tushare
早整整一天可用，对「早晨跑盘前、要拿隔夜美债」的场景是实质差别。

## 三个接口都试过

| 接口 | 可用 | 说明 |
|---|---|---|
| `api.fiscaldata.treasury.gov` | ✗ | **没有日度实际利率曲线**。它的数据集是 `avg_interest_rates`（按券种的平均利率，月/季频）、`debt_to_the_penny`、`daily_treasury_statement`，不覆盖这条序列。 |
| CSV 按年 | ✓ | `.../daily-treasury-rates.csv/{年}/all?type=daily_treasury_real_yield_curve`，一次拿整年，默认走这个。 |
| XML 按年月 | ✓ | OData/Atom feed，字段 `d:NEW_DATE` + `d:TC_10YEAR`，按 `field_tdr_date_value_month=YYYYMM` 取，留作备用（`use_xml=True`）。 |

## 口径

- 序列是 **Daily Treasury Real Yield Curve Rates**（TIPS 实际利率），取 `10 YR`
  列，单位 %（不是百分数的小数）。与 `macro_series` 里的 `us_real10` 同口径。
- 周末与联邦假日**没有行**：实测 2026-11-12 直接跳到 11-10（11-11 退伍军人节），
  与 Tushare 的缺口一致。
- 2003-01 起有 10 年期（更早没有 TIPS 10Y 报价）。

## 用法

    python -m ttrade.macro_treasury --check       # 只看，和库里对比，不写
    python -m ttrade.macro_treasury --update      # 增量写进 edb.duckdb
    python -m ttrade.macro_treasury --full        # 2003 起全量重拉
    python -m ttrade.macro_treasury --xml         # 改用 XML 接口
"""

from __future__ import annotations

import io
import os
import re
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

# ── 数据源 ──────────────────────────────────────────────────────────────────
BASE = ("https://home.treasury.gov/resource-center/"
        "data-chart-center/interest-rates")
CSV_URL = (BASE + "/daily-treasury-rates.csv/{y}/all"
           "?type=daily_treasury_real_yield_curve"
           "&field_tdr_date_value={y}&page&_format=csv")
XML_URL = (BASE + "/pages/xml?data=daily_treasury_real_yield_curve"
           "&field_tdr_date_value_month={ym}")

UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/120.0 Safari/537.36"),
      "Accept": "text/csv,application/xml,*/*"}

COL_CSV = "10 YR"          # CSV 的表头
TAG_XML = "TC_10YEAR"      # XML 的字段（d:TC_10YEAR）
TAG_DATE = "NEW_DATE"

START_YEAR = 2003          # 10 年期 TIPS 报价从此开始
CODE = "us_real10"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(_ROOT, "data_tq", "edb.duckdb")


# ── 取数 ────────────────────────────────────────────────────────────────────
def _get(url: str, timeout: int = 30, retries: int = 3) -> str | None:
    """拿文本。404 当「这一年没有数据」返回 None；其它错重试。"""
    import requests

    last = ""
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200:
                t = r.text
                return t if t.strip() else None
            if r.status_code == 404:
                return None
            last = "HTTP %d" % r.status_code
        except Exception as e:  # noqa: BLE001
            last = str(e) or e.__class__.__name__
        time.sleep(1.0 + i)          # 财政部偶尔 502，退避重试
    raise RuntimeError("取 %s 失败：%s" % (url, last))


def _to_series(idx, vals) -> pd.Series:
    s = pd.Series(np.asarray(vals, dtype=float),
                  index=pd.DatetimeIndex(idx))
    s = s[s.index.notna()].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def parse_csv(text: str) -> pd.Series:
    """CSV → Series。日期是 `MM/DD/YYYY`，10 年期列名 `10 YR`。"""
    if not text:
        return pd.Series(dtype=float)
    df = pd.read_csv(io.StringIO(text))
    if "Date" not in df.columns:
        return pd.Series(dtype=float)
    col = COL_CSV if COL_CSV in df.columns else next(
        (c for c in df.columns if c.replace(" ", "").upper() in ("10YR", "R10YEAR")),
        None)
    if col is None:
        return pd.Series(dtype=float)
    d = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    return _to_series(d, pd.to_numeric(df[col], errors="coerce"))


def parse_xml(text: str) -> pd.Series:
    """OData/Atom feed → Series。

    不引 lxml，直接正则扫 `<d:NEW_DATE>2026-11-18T00:00:00</d:NEW_DATE>` 与
    `<d:TC_10YEAR m:type="Edm.Double">2.88</d:TC_10YEAR>`：这两个字段在一条
    `<entry>` 里成对出现，按文档顺序配对即可（实测 2026-11 的 12 条全部对上）。
    """
    if not text:
        return pd.Series(dtype=float)
    dates = re.findall(r"<d:%s[^>]*>([^<]+)</d:%s>" % (TAG_DATE, TAG_DATE), text)
    vals = re.findall(r"<d:%s[^>]*>([^<]+)</d:%s>" % (TAG_XML, TAG_XML), text)
    if not dates or len(dates) != len(vals):
        return pd.Series(dtype=float)
    d = pd.to_datetime(dates, errors="coerce")
    return _to_series(d, pd.to_numeric(pd.Series(vals), errors="coerce"))


def fetch_year(year: int, use_xml: bool = False) -> pd.Series:
    """拉一年。没有数据（未来年份 / 2003 之前）返回空 Series。"""
    if use_xml:
        parts = []
        for m in range(1, 13):
            ym = "%04d%02d" % (year, m)
            if ym > datetime.now().strftime("%Y%m"):
                break
            try:
                txt = _get(XML_URL.format(ym=ym))
            except RuntimeError:
                continue
            s = parse_xml(txt or "")
            if len(s):
                parts.append(s)
        return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    txt = _get(CSV_URL.format(y=year))
    return parse_csv(txt or "")


def fetch(start_year: int = START_YEAR, end_year: int | None = None,
          use_xml: bool = False, progress=None) -> pd.Series:
    """拉 [start_year, end_year] 的 10 年期实际利率，按年循环。"""
    end_year = end_year or date.today().year
    parts = []
    for y in range(int(start_year), int(end_year) + 1):
        if progress is not None:
            progress(y)
        try:
            s = fetch_year(y, use_xml=use_xml)
        except RuntimeError:
            continue                  # 单年失败不拖垮整轮
        if len(s):
            parts.append(s)
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="last")]


def fetch_recent(n_years: int = 2, use_xml: bool = False) -> pd.Series:
    """增量口径：只拉最近 n 年（财政部按年给整年，没法只取几天）。"""
    return fetch(date.today().year - int(n_years) + 1, date.today().year,
                 use_xml=use_xml)


# ── 与库对比 / 写库 ─────────────────────────────────────────────────────────
def from_db(code: str = CODE, db: str = DEFAULT_DB) -> pd.Series:
    import duckdb

    con = duckdb.connect(db, read_only=True)
    try:
        rows = con.execute(
            "SELECT dt, value FROM macro_series WHERE code = ? ORDER BY dt",
            [code]).fetchall()
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)
    finally:
        con.close()
    if not rows:
        return pd.Series(dtype=float)
    return _to_series([r[0] for r in rows], [r[1] for r in rows])


def compare(series: pd.Series, code: str = CODE, db: str = DEFAULT_DB) -> dict:
    """新拉到的 vs 库里已有的：重叠段差多少、库里缺哪几天。"""
    old = from_db(code, db)
    if not len(series):
        return {"ok": False, "reason": "新数据为空"}
    if not len(old):
        return {"ok": True, "old": 0, "new": len(series),
                "new_range": (series.index[0].date(), series.index[-1].date())}
    j = pd.concat([series.rename("new"), old.rename("old")], axis=1,
                  sort=True).dropna()
    diff = (j["new"] - j["old"]).abs() if len(j) else pd.Series(dtype=float)
    new_days = [d for d in series.index if d not in old.index]
    # 「库里有、源没有」只统计源覆盖的那段，否则增量模式下会报一堆历史年份
    lo = series.index[0]
    old_days = [d for d in old.index if d not in series.index and d >= lo]
    fmt = lambda xs: "、".join(str(d.date()) for d in xs)  # noqa: E731
    return {
        "ok": True,
        "old": len(old), "new": len(series),
        "overlap": len(j),
        "max_diff": float(diff.max()) if len(diff) else float("nan"),
        "n_diff": int((diff > 1e-9).sum()) if len(diff) else 0,
        "db_last": old.index[-1].date(),
        "src_last": series.index[-1].date(),
        "new_days": new_days[-10:],
        "only_in_db": len(old_days),
        "only_in_db_sample": old_days[:5],
        "_fmt": fmt,
    }


def update_to_db(series: pd.Series | None = None, code: str = CODE,
                 db: str = DEFAULT_DB, replace: bool = False,
                 use_xml: bool = False) -> int:
    """写进 `macro_series`（主键 code+dt，重复即覆盖）。返回写入行数。"""
    import duckdb

    if series is None:
        series = fetch_recent(2, use_xml=use_xml)
    if not len(series):
        return 0
    df = pd.DataFrame({
        "code": code,
        "dt": [d.date() if hasattr(d, "date") else d for d in series.index],
        "value": series.to_numpy(dtype=float),
    })
    con = duckdb.connect(db)
    try:
        if replace:
            con.execute("DELETE FROM macro_series WHERE code = ?", [code])
        con.register("_mt", df)
        con.execute("INSERT OR REPLACE INTO macro_series "
                    "SELECT code, dt, value FROM _mt")
        con.unregister("_mt")
    finally:
        con.close()
    return len(df)


# ── 发布时刻探针（每小时采样，定出真实的「北京几点更新」） ──────────────────
# 财政部官方只说「usually by 6:00 PM Eastern Time」，实测又拿不到 —— feed 的
# `updated` 是静态元数据（11-18T15:48Z 那天美东才 10:48，债市刚开盘，不可能有
# 当天收盘）。唯一可靠的办法是长期采样：每小时抓一次，记「D 那行第一次出现的
# 采样时刻」，一周就能看出分布的众数与尾部。
PROBE_LOG = os.path.join(_ROOT, "data_tq", "treasury_probe.csv")


def _feed_updated(month: str | None = None) -> str:
    """XML feed 的 `updated` 戳。**只做记录，不能当发布时刻**（原因见上）。"""
    ym = month or datetime.now().strftime("%Y%m")
    try:
        txt = _get(XML_URL.format(ym=ym))
    except RuntimeError:
        return ""
    if not txt:
        return ""
    m = re.search(r"<updated>([^<]+)</updated>", txt)
    return m.group(1) if m else ""


def probe(log_path: str = PROBE_LOG, with_xml: bool = True) -> dict:
    """采样一次并追加到 CSV。返回本次记录。"""
    t0 = time.time()
    y = date.today().year
    try:
        s = fetch_year(y)
    except RuntimeError as e:
        s = pd.Series(dtype=float)
        err = str(e)[:120]
    else:
        err = ""
    rec = {
        "sampled_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "latest_date": s.index[-1].strftime("%Y-%m-%d") if len(s) else "",
        "latest_value": ("%.2f" % float(s.iloc[-1])) if len(s) else "",
        "n_rows": int(len(s)),
        "feed_updated": _feed_updated() if with_xml else "",
        "elapsed_ms": int((time.time() - t0) * 1000),
        "err": err,
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    row = pd.DataFrame([rec])
    if os.path.exists(log_path):
        try:
            old = pd.read_csv(log_path, dtype=str)
            row = pd.concat([old, row.astype(str)], ignore_index=True)
        except Exception:  # noqa: BLE001
            pass                      # 日志坏了就重开，不因为写日志失败丢采样
    row.to_csv(log_path, index=False, encoding="utf-8-sig")
    return rec


def probe_report(log_path: str = PROBE_LOG, last_days: int = 14) -> str:
    """按「数据日期」分组，看每个日期第一次被看到的采样时刻 = 发布时刻的上界。"""
    if not os.path.exists(log_path):
        return "还没有采样记录：%s" % log_path
    df = pd.read_csv(log_path, dtype=str)
    if not len(df):
        return "日志为空"
    df = df[df["latest_date"].astype(str).str.len() == 10].copy()
    if not len(df):
        return "日志里没有有效日期"
    df["ts"] = pd.to_datetime(df["sampled_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df["bj"] = df["ts"].dt.tz_convert("Asia/Shanghai")

    lines = ["采样 %d 条  %s → %s" % (
        len(df), df["bj"].iloc[0].strftime("%m-%d %H:%M"),
        df["bj"].iloc[-1].strftime("%m-%d %H:%M")), ""]
    first = (df.groupby("latest_date")["bj"].min().sort_index().tail(last_days))
    lines.append("%-12s %-6s %-6s %s" % ("数据日期", "首次见于", "值", "距数据日"))
    for d, t in first.items():
        v = df[df["latest_date"] == d]["latest_value"].iloc[-1]
        lag = (t.tz_localize(None).normalize() - pd.Timestamp(d)).days
        lines.append("%-12s %-6s %-6s 北京 %s（D+%d）"
                     % (d, t.strftime("%H:%M"), v, t.strftime("%m-%d %H:%M"), lag))
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    use_xml = "--xml" in argv

    if "--probe-report" in argv:
        print(probe_report())
        return 0
    if "--probe" in argv:
        rec = probe(with_xml=("--no-xml" not in argv))
        print("采样 %s" % rec["sampled_at"])
        if rec["err"]:
            print("  失败：%s" % rec["err"])
            return 1
        print("  最新 %s = %s（当年 %d 行，%d ms，feed_updated %s）"
              % (rec["latest_date"], rec["latest_value"], rec["n_rows"],
                 rec["elapsed_ms"], rec["feed_updated"] or "—"))
        prev = None
        try:
            df = pd.read_csv(PROBE_LOG, dtype=str)
            if len(df) > 1:
                prev = df["latest_date"].iloc[-2]
        except Exception:  # noqa: BLE001
            pass
        if prev and prev != rec["latest_date"]:
            print("  🔴 新日期出现：%s → %s（这就是它第一次被看到的时刻）"
                  % (prev, rec["latest_date"]))
        return 0

    if "--full" in argv:
        mode = "full"
    elif "--update" in argv:
        mode = "update"
    else:
        mode = "check"

    src = "XML(按月)" if use_xml else "CSV(按年)"
    print("数据源：美国财政部 Daily Treasury Real Yield Curve，10 YR  ·  接口 %s" % src)

    if mode == "full":
        print("拉 2003 年起全量 …")
        s = fetch(START_YEAR, use_xml=use_xml,
                  progress=lambda y: print("  %d" % y, end="", flush=True))
        print()
    else:
        s = fetch_recent(2, use_xml=use_xml)

    if not len(s):
        print("没取到数据")
        return 1
    print("取到 %d 条  %s → %s" % (len(s), s.index[0].date(), s.index[-1].date()))
    print("最近 6 天：")
    for k, v in s.tail(6).items():
        print("  %s  %.2f" % (k.date(), v))

    info = compare(s)
    print("\n与库里 us_real10 对比：")
    if not info.get("ok"):
        print("  %s" % info.get("reason"))
    elif "overlap" not in info:
        print("  库里没有这条序列，将新建 %d 行" % info["new"])
    else:
        fmt = info.get("_fmt") or (lambda xs: str(xs))
        print("  库里 %d 行，最新 %s；源最新 %s"
              % (info["old"], info["db_last"], info["src_last"]))
        print("  重叠 %d 天，最大差异 %.4f，有差异 %d 天"
              % (info["overlap"], info["max_diff"], info["n_diff"]))
        if info["new_days"]:
            print("  源有、库里没有：%s" % fmt(info["new_days"]))
        if info["only_in_db"]:
            print("  库里有、源没有：%d 天（例 %s）"
                  % (info["only_in_db"], fmt(info["only_in_db_sample"])))

    if mode in ("full", "update"):
        n = update_to_db(s, replace=(mode == "full"), use_xml=use_xml)
        print("\n已写入 %d 行（%s）" % (n, "全量替换" if mode == "full" else "增量覆盖"))
    else:
        print("\n（只检查，未写库；要写加 --update）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
