"""updater 的逻辑自测（不联网，全部打桩）。

    python smoke_test_updater.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import updater  # noqa: E402
from ttrade import us_tushare as us_t  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ OK ] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def mkdir() -> str:
    return tempfile.mkdtemp(prefix="upd_")


def weekdays(start: str, n: int) -> list[pd.Timestamp]:
    d = pd.Timestamp(start)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def next_days(last: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    out, d = [], last
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


def write_us(dirpath: str, sym: str, days, close: float = 100.0) -> None:
    pd.DataFrame(
        {
            "symbol": sym,
            "date": [x.strftime("%Y-%m-%d") for x in days],
            "open": close, "close": close, "high": close + 1, "low": close - 1,
            "volume": 1000, "amount": 1e5,
        }
    ).to_csv(os.path.join(dirpath, f"{sym}.csv"), index=False, encoding="utf-8-sig")


def write_manifest(dirpath: str, rows: list[dict]) -> None:
    with open(os.path.join(dirpath, "_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)


def em_stub(price: float):
    """模拟东财：从 start 一直返回到今天（真实接口就是这么干的）。"""

    def fake(secid, start=None, adjust=2, session=None, retries=5):
        d = pd.to_datetime(start, format="%Y%m%d")
        today = pd.Timestamp(date.today())
        days = []
        while d <= today:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        df = pd.DataFrame(
            {
                "date": pd.to_datetime([x.strftime("%Y-%m-%d") for x in days]),
                "open": price, "close": price, "high": price, "low": price,
                "volume": 1, "amount": 1,
            }
        )
        df.attrs["secid"] = secid
        return df

    return fake


def ts_stub(price: float):
    def fake(ts_code, start, end, cancel=None, retries=2):
        d = pd.to_datetime(start, format="%Y%m%d")
        days = weekdays(d.strftime("%Y-%m-%d"), 6)
        return pd.DataFrame(
            {
                "date": pd.to_datetime([x.strftime("%Y-%m-%d") for x in days]),
                "open": price, "close": price, "high": price, "low": price,
                "volume": 1, "amount": 1,
            }
        )

    return fake


# ------------------------------------------------------------------ 用例
def test_window_start() -> None:
    print("\n[1] 区间起点计算")
    today = date.today()
    s1 = updater._window_start(31, today - timedelta(days=1))
    check("近1月≈31天前", abs((today - s1).days - 31) <= 1, str(s1))
    s2 = updater._window_start(31, today - timedelta(days=200))
    check("本地落后时回溯到本地尾部", (today - s2).days > 180, str(s2))
    s3 = updater._window_start(0, today)
    check("全部≈20年", 7200 < (today - s3).days < 7500, str(s3))


def test_us_merge() -> None:
    print("\n[2] 美股增量合并（东财）")
    d = mkdir()
    try:
        days = weekdays("2025-01-01", 100)
        write_us(d, "AAPL", days)
        write_manifest(d, [{"symbol": "AAPL", "name": "苹果", "exchange": "NASDAQ",
                            "secid": "105.AAPL", "source": "eastmoney"}])
        orig = updater.em.fetch_daily
        updater.em.fetch_daily = em_stub(200.0)
        try:
            res = updater.update_us(d, "1m", source="eastmoney", min_interval=0)
        finally:
            updater.em.fetch_daily = orig

        out = pd.read_csv(os.path.join(d, "AAPL.csv"))
        check("无失败", res.failed == 0, f"{res.errors[:1]}")
        check("行数增加", len(out) > 100, f"{len(out)} 行")
        check("日期无重复", out["date"].is_unique)
        check("日期有序", out["date"].is_monotonic_increasing)
        check("重叠被覆盖", float(out["close"].iloc[-1]) == 200.0, str(out["close"].iloc[-1]))
        check("symbol 未丢", out["symbol"].notna().all())
        check("无 .tmp 残留", not any(f.endswith(".tmp") for f in os.listdir(d)))
        mf = json.load(open(os.path.join(d, "_manifest.json"), encoding="utf-8"))
        rec = [x for x in mf if x.get("symbol") == "AAPL"][0]
        check("清单已更新", rec["rows"] == len(out) and rec["source"] == "eastmoney", str(rec))
        check("来源计数", res.sources == {"eastmoney": 1}, str(res.sources))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_us_switch_source() -> None:
    print("\n[3] 换数据源必须整段重下（不能把未复权和后复权拼一起）")
    d = mkdir()
    try:
        days = weekdays("2025-01-01", 100)
        write_us(d, "AAPL", days, close=3000.0)   # 模拟后复权的高价位
        write_manifest(d, [{"symbol": "AAPL", "name": "苹果", "exchange": "NASDAQ",
                            "secid": "105.AAPL", "source": "eastmoney"}])

        def ts_short(ts_code, start, end, cancel=None, retries=2):
            # 未复权原始价：只给最近 6 天
            dd = weekdays("2026-10-01", 6)
            return pd.DataFrame(
                {
                    "date": pd.to_datetime([x.strftime("%Y-%m-%d") for x in dd]),
                    "open": 200.0, "close": 200.0, "high": 200.0, "low": 200.0,
                    "volume": 1, "amount": 1,
                }
            )

        ot = updater.us_t.fetch_daily
        updater.us_t.fetch_daily = ts_short
        try:
            res = updater.update_us(d, "1m", source="tushare", min_interval=0)
        finally:
            updater.us_t.fetch_daily = ot

        out = pd.read_csv(os.path.join(d, "AAPL.csv"))
        check("无失败", res.failed == 0, f"{res.errors[:1]}")
        check("整段替换（不再是 100 行）", len(out) == 6, f"{len(out)} 行")
        check("价格基准已换成未复权", float(out["close"].max()) == 200.0,
              str(float(out["close"].max())))
        mf = json.load(open(os.path.join(d, "_manifest.json"), encoding="utf-8"))
        rec = [x for x in mf if x.get("symbol") == "AAPL"][0]
        check("来源已标记为 tushare", rec["source"] == "tushare", str(rec.get("source")))
        check("adjusted=False", rec["adjusted"] is False, str(rec.get("adjusted")))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_us_fallback_to_tushare() -> None:
    print("\n[4] 东财失败 → Tushare 兜底（auto 模式）")
    d = mkdir()
    try:
        days = weekdays("2025-01-01", 60)
        for s_ in ("AAPL", "MSFT"):
            write_us(d, s_, days)
        write_manifest(d, [{"symbol": s_, "name": "", "exchange": "", "secid": f"105.{s_}",
                            "source": "eastmoney"} for s_ in ("AAPL", "MSFT")])

        def em_boom(*a, **k):
            raise RuntimeError("东财不可达")

        ot, oe = updater.us_t.fetch_daily, updater.em.fetch_daily
        updater.us_t.fetch_daily = ts_stub(200.0)
        updater.em.fetch_daily = em_boom
        updater.us_t.reset_availability()
        try:
            res = updater.update_us(d, "1m", source="auto", min_interval=0)
        finally:
            updater.us_t.fetch_daily = ot
            updater.em.fetch_daily = oe
            updater.us_t.reset_availability()

        check("全部成功（靠兜底）", res.failed == 0 and res.ok == 2,
              f"ok={res.ok} failed={res.failed} {res.errors[:1]}")
        check("来源全是 Tushare", res.sources == {"tushare": 2}, str(res.sources))
        check("有兜底提示", any("兜底" in n for n in res.notes), str(res.notes[:1]))
        out = pd.read_csv(os.path.join(d, "AAPL.csv"))
        check("换源后整段替换", len(out) == 6, f"{len(out)} 行")
        mf = json.load(open(os.path.join(d, "_manifest.json"), encoding="utf-8"))
        rec = [x for x in mf if x.get("symbol") == "AAPL"][0]
        check("来源标记为 tushare", rec["source"] == "tushare", str(rec.get("source")))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_us_both_fail() -> None:
    print("\n[5] 东财和 Tushare 都失败 → 如实报错，且 Tushare 只试一次")
    d = mkdir()
    try:
        days = weekdays("2025-01-01", 30)
        for s_ in ("AAPL", "MSFT", "NVDA"):
            write_us(d, s_, days)
        write_manifest(d, [{"symbol": s_, "secid": f"105.{s_}", "source": "eastmoney"}
                           for s_ in ("AAPL", "MSFT", "NVDA")])

        calls = {"n": 0}

        def em_boom(*a, **k):
            raise RuntimeError("东财不可达")

        def ts_quota(ts_code, start, end, cancel=None, retries=2):
            calls["n"] += 1
            raise us_t.QuotaExceeded("抱歉，您访问接口(us_daily)频率超限(5次/天)", True)

        ot, oe = updater.us_t.fetch_daily, updater.em.fetch_daily
        updater.us_t.fetch_daily = ts_quota
        updater.em.fetch_daily = em_boom
        updater.us_t.reset_availability()
        try:
            try:
                res = updater.update_us(d, "1m", source="auto", min_interval=0)
            except updater.NetworkAbort:
                res = None
        finally:
            updater.us_t.fetch_daily = ot
            updater.em.fetch_daily = oe
            updater.us_t.reset_availability()

        check("Tushare 只试 1 次就熔断", calls["n"] == 1, f"调用 {calls['n']} 次")
        check("判定为整体失败", res is None or res.failed >= 3,
              "NetworkAbort" if res is None else f"failed={res.failed}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_futures_merge() -> None:
    print("\n[5] 期货增量合并 + 复权因子拼接")
    d = mkdir()
    try:
        days = weekdays("2025-01-01", 80)
        pd.DataFrame(
            {
                "ts_code": "CU.SHFE", "name": "沪铜",
                "date": [x.strftime("%Y-%m-%d") for x in days],
                "open": 70000.0, "high": 71000.0, "low": 69000.0, "close": 70500.0,
                "volume": 100, "amount": 1e6, "open_interest": 50,
                "adj_factor": [2.0] * 80,
            }
        ).to_csv(os.path.join(d, "CU_SHFE.csv"), index=False, encoding="utf-8-sig")
        last = days[-1]
        new_days = next_days(last, 4)

        def fake_daily(ts_code, s, e, pro=None, retries=2, pause=0.12):
            return pd.DataFrame(
                {
                    "date": pd.to_datetime([y.strftime("%Y-%m-%d") for y in new_days]),
                    "open": 80000.0, "high": 81000.0, "low": 79000.0, "close": 80500.0,
                    "volume": 200, "amount": 2e6, "open_interest": 60,
                }
            )

        def fake_adj(ts_code, s, e, pro=None, retries=2, pause=0.12):
            return pd.DataFrame(
                {
                    "date": pd.to_datetime([y.strftime("%Y-%m-%d") for y in new_days]),
                    "adj_close": 80500.0 * 2.5,
                }
            )

        od, oa, og = updater.fut.fetch_daily, updater.fut.fetch_adj_close, updater.fut.get_pro
        updater.fut.fetch_daily = fake_daily
        updater.fut.fetch_adj_close = fake_adj
        updater.fut.get_pro = lambda: object()
        try:
            res = updater.update_futures(d, "1m", include_new=False)
        finally:
            updater.fut.fetch_daily = od
            updater.fut.fetch_adj_close = oa
            updater.fut.get_pro = og

        out = pd.read_csv(os.path.join(d, "CU_SHFE.csv"))
        check("无失败", res.failed == 0, f"{res.errors[:1]}")
        check("行数增加", len(out) > 80, f"{len(out)} 行")
        check("日期无重复", out["date"].is_unique)
        old_part = out[out["date"] <= last.strftime("%Y-%m-%d")]
        new_part = out[out["date"] > last.strftime("%Y-%m-%d")]
        check("老因子未变", float(old_part["adj_factor"].iloc[0]) == 2.0)
        check("新因子=2.5",
              len(new_part) > 0 and abs(float(new_part["adj_factor"].iloc[-1]) - 2.5) < 1e-6,
              str(new_part["adj_factor"].tolist()))
        check("ts_code/name 未丢",
              out["ts_code"].notna().all() and out["name"].notna().all())
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_adj_empty_fallback() -> None:
    print("\n[6] 复权数据为空时沿用旧因子")
    d = mkdir()
    try:
        days = weekdays("2025-01-01", 50)
        pd.DataFrame(
            {
                "ts_code": "LC.GFE", "name": "碳酸锂",
                "date": [x.strftime("%Y-%m-%d") for x in days],
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                "volume": 1, "amount": 1, "open_interest": 1,
                "adj_factor": [3.2] * 50,
            }
        ).to_csv(os.path.join(d, "LC_GFE.csv"), index=False, encoding="utf-8-sig")
        nd = next_days(days[-1], 2)

        def fake_daily(ts_code, s, e, pro=None, retries=2, pause=0.12):
            return pd.DataFrame(
                {
                    "date": pd.to_datetime([y.strftime("%Y-%m-%d") for y in nd]),
                    "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                    "volume": 1, "amount": 1, "open_interest": 1,
                }
            )

        od, oa, og = updater.fut.fetch_daily, updater.fut.fetch_adj_close, updater.fut.get_pro
        updater.fut.fetch_daily = fake_daily
        updater.fut.fetch_adj_close = lambda *a, **k: pd.DataFrame()
        updater.fut.get_pro = lambda: object()
        try:
            updater.update_futures(d, "1m", include_new=False)
        finally:
            updater.fut.fetch_daily = od
            updater.fut.fetch_adj_close = oa
            updater.fut.get_pro = og

        out = pd.read_csv(os.path.join(d, "LC_GFE.csv"))
        check("新行因子沿用 3.2", abs(float(out["adj_factor"].iloc[-1]) - 3.2) < 1e-6,
              str(float(out["adj_factor"].iloc[-1])))
        check("因子序列连续", out["adj_factor"].nunique() == 1,
              str(sorted(out["adj_factor"].unique())))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_abort_on_network() -> None:
    print("\n[7] 连续失败自动中止")
    d = mkdir()
    try:
        days = weekdays("2025-06-01", 10)
        for sym in ["A", "B", "C", "D", "E"]:
            write_us(d, sym, days)
        write_manifest(d, [{"symbol": s, "secid": f"105.{s}", "source": "eastmoney"}
                           for s in ["A", "B", "C", "D", "E"]])

        def boom(*a, **k):
            raise RuntimeError("模拟断网")

        orig = updater.em.fetch_daily
        updater.em.fetch_daily = boom
        aborted = False
        try:
            updater.update_us(d, "1m", source="eastmoney", min_interval=0)
        except updater.NetworkAbort as exc:
            aborted = True
            print(f"       中止信息：{str(exc)[:60]}…")
        finally:
            updater.em.fetch_daily = orig
        check("3 个失败后中止", aborted)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_rate_limiter() -> None:
    print("\n[8] Tushare 美股限速与取消")
    import time as _t

    us_t.set_rate_limit(1.0)
    us_t.reset_call_clock()
    calls = {"n": 0}

    def fake_pro_call():
        calls["n"] += 1

    t0 = _t.monotonic()
    us_t._throttle()
    us_t._throttle()
    dt = _t.monotonic() - t0
    check("第二次调用被推迟约 1 秒", dt >= 0.9, f"{dt:.2f}s")

    us_t.reset_call_clock()
    us_t._throttle()
    t1 = _t.monotonic()
    cancelled = False
    try:
        us_t._throttle(cancel=lambda: True)
    except us_t.Cancelled:
        cancelled = True
    dt2 = _t.monotonic() - t1
    check("等待中可被取消", cancelled and dt2 < 0.5, f"{dt2:.2f}s")
    us_t.set_rate_limit(0)


def test_quota_parsing() -> None:
    print("\n[9] 额度报错分类")
    a = us_t._is_quota_error("抱歉，您访问接口(us_daily)频率超限(5次/天)")
    b = us_t._is_quota_error("抱歉，您每分钟最多访问该接口1次")
    c = us_t._is_quota_error("Connection reset by peer")
    check("每天额度 → 熔断", a == (True, True), str(a))
    check("每分钟超频 → 不熔断", b == (True, False), str(b))
    check("网络错误不算额度", c == (False, False), str(c))


if __name__ == "__main__":
    test_window_start()
    test_us_merge()
    test_us_switch_source()
    test_us_fallback_to_tushare()
    test_us_both_fail()
    test_futures_merge()
    test_adj_empty_fallback()
    test_abort_on_network()
    test_rate_limiter()
    test_quota_parsing()
    print(f"\n结果：{PASS} 通过，{FAIL} 失败")
    raise SystemExit(0 if FAIL == 0 else 1)
