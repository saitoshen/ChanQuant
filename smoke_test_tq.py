"""天勤数据链路的离线自测：切换表、复权因子、多周期合并。

不联网、不依赖天勤账户，只读 `data_tq/` 里已经下好的文件。有数据就全量校验，
没数据就跳过（打包后的环境本来就没有这份数据，不该因此报错）。

    python smoke_test_tq.py            # 校验全部已下载品种
    python smoke_test_tq.py RB_SHFE    # 只校验一个
"""

from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np  # noqa: F401  （保留给后续数值校验用）
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import tqdata as T  # noqa: E402
from ttrade import tqstore as S  # noqa: E402

_ok = _fail = _skip = 0
_msgs: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _ok, _fail
    if cond:
        _ok += 1
    else:
        _fail += 1
        _msgs.append(f"[FAIL] {name}  {detail}")


def skip(name: str, why: str) -> None:
    global _skip
    _skip += 1
    _msgs.append(f"[SKIP] {name}  {why}")


def keys() -> list[str]:
    return [f"{s}_{e}" for s, e in S.switch_symbols()]


def check_one(key: str) -> None:
    sym, ex = key.rsplit("_", 1)
    sw = S.load_switch(sym, ex)
    check(f"{key} 切换表非空", len(sw) > 0, f"{len(sw)} 条")
    if not len(sw):
        return
    r = pd.to_numeric(sw["ratio"], errors="coerce")
    check(f"{key} ratio 落在合理区间", bool(((r > 0.5) & (r < 2.0)).all()),
          f"min={r.min():.4f} max={r.max():.4f}")
    check(f"{key} ratio 无缺失", bool(r.notna().all()), f"缺 {int(r.isna().sum())} 条")
    # 主连正常只会向远月滚，但天勤按持仓/成交额判定，偶尔会在换月当天翻回来
    # （IC 2024-12-19 切到 2503，次日又切回 2501）。这种回切本身是对的 ——
    # 复权比率是在每次切换的那一刻量的，连乘能精确抵消 —— 所以只要求占比很小，
    # 占比大就说明切换表解析错了。
    later = sw["new_ts"].str.extract(r"(\d{4})\.")[0].astype(int)
    earlier = sw["old_ts"].str.extract(r"(\d{4})\.")[0].astype(int)
    back = int((later <= earlier).sum())
    check(f"{key} 回切占比很小", back <= max(2, len(sw) * 0.05),
          f"{back}/{len(sw)} 次回切")

    for period in ("5m", "1h"):
        if not T.has_period(sym, ex, period):
            continue
        df = T.load_main(sym, ex, period, adjust="none")
        check(f"{key} {period} 有数据", len(df) > 100, f"{len(df)} 行")
        if len(df) < 100:
            continue
        check(f"{key} {period} 价格为正",
              bool((df[["open", "high", "low", "close"]] > 0).all().all()))
        check(f"{key} {period} high≥low",
              bool((df["high"] >= df["low"]).all()))
        check(f"{key} {period} 时间戳唯一", bool(df.index.is_unique))

        # 后复权因子 = 之后所有换月比率的连乘，所以它的单调性取决于换月方向：
        # 商品多为 contango（远月更贵，ratio>1），因子随时间递减；股指远月贴水
        # （ratio<1），因子随时间递增。两种都对，能保证的只有「最新段为 1」和
        # 「量级合理」—— 连续性由下面的跳空检查负责。
        f = df["adj_factor"].to_numpy(float)
        check(f"{key} {period} 复权因子量级合理", bool(((f > 0.2) & (f < 5.0)).all()),
              f"min={f.min():.4f} max={f.max():.4f}")
        check(f"{key} {period} 最新段因子为 1", abs(f[-1] - 1.0) < 1e-9, f"{f[-1]}")

        # 复权后，换月 bar 的跳空不应显著大于平常 bar
        adj = df["close"] * df["adj_factor"]
        ret = adj.pct_change().abs()
        swd = set(pd.to_datetime(sw["switch_date"]).dt.normalize())
        is_sw = pd.Series(df["trading_day"]).isin(swd).to_numpy()
        if is_sw.sum() >= 3:
            m_sw = float(ret[is_sw].mean())
            m_norm = float(ret[~is_sw].mean())
            check(f"{key} {period} 换月跳空已消除",
                  m_sw < max(m_norm * 3, 0.002),
                  f"换月 {m_sw:.5f} vs 平常 {m_norm:.5f}")


def check_resample(key: str) -> None:
    sym, ex = key.rsplit("_", 1)
    if not T.has_period(sym, ex, "5m"):
        return
    d5 = T.load_main(sym, ex, "5m")
    for period, n in (("15m", 3), ("30m", 6)):
        if not T.has_period(sym, ex, period):
            continue
        dn = T.load_main(sym, ex, period)
        check(f"{key} {period} 成交量守恒",
              abs(float(dn["volume"].sum()) - float(d5["volume"].sum())) < 1e-6,
              f"{dn['volume'].sum():.0f} vs {d5['volume'].sum():.0f}")
        check(f"{key} {period} 最高价不低于 5m",
              float(dn["high"].max()) >= float(d5["high"].max()) - 1e-9)
        check(f"{key} {period} 行数约为 5m 的 1/{n}",
              abs(len(dn) - len(d5) / n) <= len(d5) * 0.02,
              f"{len(dn)} vs {len(d5) / n:.0f}")


def check_trading_day() -> None:
    """夜盘 bar 必须归到下一个有日盘的交易日。"""
    idx = pd.DatetimeIndex(
        ["2026-09-01 21:00", "2026-09-01 22:00", "2026-09-02 09:00", "2026-09-02 14:00"]
    )
    td = T.trading_days(pd.Series(idx)).dt.normalize().tolist()
    want = [pd.Timestamp("2026-09-02")] * 4
    check("交易日映射（夜盘归次日）", td == want, f"{[str(x.date()) for x in td]}")
    # 周五夜盘要归到下周一（用真实的 2026-10-23 周五 / 10-26 周一）
    idx2 = pd.DatetimeIndex(["2026-10-23 21:00", "2026-10-26 09:00"])
    td2 = T.trading_days(pd.Series(idx2)).dt.normalize().tolist()
    check("交易日映射（周五夜盘归下周一）",
          td2[0] == pd.Timestamp("2026-10-26") and td2[1] == pd.Timestamp("2026-10-26"),
          f"{[str(x.date()) for x in td2]}")


# ------------------------------------------------- 真实合约增量（不联网）


def _fake_frame(end: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(end=end, periods=48, freq="5min")
    return pd.DataFrame({
        "dt": idx, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
        "volume": 10, "open_oi": 5, "close_oi": 6,
    })


class _FakeApi:
    def __init__(self, conts: list[str]) -> None:
        self.conts = conts
        self.calls = 0

    def query_quotes(self, **_kw):
        return list(self.conts)

    def wait_update(self) -> None:
        self.calls += 1
        if self.calls > 200:
            raise RuntimeError("测试里不该跑这么久")


def check_contract_incremental() -> None:
    """真实合约的四种跳过路径 + 临时文件清理。

    十年全品种能不能重跑，全靠这段逻辑：写错就每次重下几千个合约，写反了（把该下
    的也跳过）数据就永远缺一块，而且两种情况都不会报错。
    """
    import tempfile

    try:
        from ttrade import tqfetch as D
    except Exception as exc:
        skip("真实合约增量", f"无法导入 tqfetch：{exc}")
        return

    old_env = os.environ.get("TRENDDESK_TQ_DB")
    tmpdir = tempfile.mkdtemp(prefix="tqtest_")
    os.environ["TRENDDESK_TQ_DB"] = os.path.join(tmpdir, "t.duckdb")
    real_dl = D.download_many
    try:
        today = pd.Timestamp.today().normalize()
        # 用 1 年区间：start ≈ 今天-365 天，于是 2024/2025 上半年的合约落在区间外
        conts = ["SHFE.rb2401", "SHFE.rb2506",   # 区间外
                 "SHFE.rb2609",                  # 区间内、已到期、但一笔没成交
                 "SHFE.rb2512",                  # 区间内、已到期、有数据
                 "SHFE.rb2611"]                  # 区间内、未到期
        seen: list[list[str]] = []

        def fake_dl(api, jobs, period, out_dir, tag, max_parallel=8):
            seen.append([s for s, _, _ in jobs])
            out = {}
            for sym, _s, _e in jobs:
                if sym.endswith("2609"):
                    continue                     # 模拟摘牌时就没数据
                out[sym] = _fake_frame(today - pd.Timedelta(hours=1))
            return out

        D.download_many = fake_dl
        api = _FakeApi(conts)

        r1 = D.write_contracts(api, "RB", "SHFE", "5m", 1)
        check("增量①首跑：区间外合约被过滤", r1["old"] == 2, f"old={r1['old']}")
        check("增量①首跑：只下区间内的 3 个", r1["contracts"] == 2 and r1["blank"] == 1,
              f"contracts={r1['contracts']} blank={r1['blank']}")
        check("增量①首跑：确实写了行", r1["rows"] > 0, f"{r1['rows']} 行")
        check("增量①首跑：请求了 3 个合约", len(seen) and len(seen[0]) == 3, f"{seen}")

        r2 = D.write_contracts(api, "RB", "SHFE", "5m", 1)
        check("增量②重跑：已到期有数据的跳过", r2["skipped"] >= 1, f"skipped={r2['skipped']}")
        check("增量②重跑：空合约不再请求", r2["blank"] >= 1 and r2["contracts"] == 0,
              f"blank={r2['blank']} contracts={r2['contracts']}")
        check("增量②重跑：没有新行", r2["rows"] == 0, f"{r2['rows']} 行")

        r3 = D.write_contracts(api, "RB", "SHFE", "5m", 1)
        check("增量③幂等：第三次结果与第二次一致",
              (r3["rows"], r3["contracts"]) == (r2["rows"], r2["contracts"]),
              f"{r3}")

        # ---- 临时文件：正常结束与异常中断都不能留下 _dl_*.csv ----
        import sys as _sys
        import types

        class _Task:
            def __init__(self, api, symbol_list, dur_sec, start_dt, end_dt, csv_file_name):
                self.sym = symbol_list
                self.path = csv_file_name
                df = _fake_frame(pd.Timestamp("2026-01-05 15:00"))
                df = df.rename(columns={c: f"{self.sym}.{c}" for c in df.columns})
                df.to_csv(self.path, index=False)

            def is_finished(self) -> bool:
                return True

        fake_tools = types.ModuleType("tqsdk.tools")
        fake_tools.DataDownloader = _Task
        fake_pkg = types.ModuleType("tqsdk")
        fake_pkg.tools = fake_tools
        saved = {k: _sys.modules.get(k) for k in ("tqsdk", "tqsdk.tools")}
        _sys.modules["tqsdk"] = fake_pkg
        _sys.modules["tqsdk.tools"] = fake_tools
        try:
            jobs = [("SHFE.rb2601", date(2025, 1, 1), date(2026, 1, 1)),
                    ("SHFE.rb2605", date(2025, 1, 1), date(2026, 1, 1))]
            got = D.download_many(_FakeApi([]), jobs, "5m", tmpdir, "T", max_parallel=2)
            left = [f for f in os.listdir(tmpdir) if f.startswith("_dl_")]
            check("临时文件：正常结束全部删除", not left, f"残留 {left}")
            check("临时文件：读回的列已去掉合约前缀",
                  len(got) == 2 and "close" in next(iter(got.values())).columns,
                  f"{list(got)}")

            class _HangApi(_FakeApi):
                def wait_update(self) -> None:
                    raise RuntimeError("与 wss://api.shinnytech.com 的连接失败")

            class _HangTask(_Task):
                def is_finished(self) -> bool:
                    return False

            fake_tools.DataDownloader = _HangTask
            try:
                D.download_many(_HangApi([]), jobs, "5m", tmpdir, "T", max_parallel=2)
            except RuntimeError:
                pass
            left = [f for f in os.listdir(tmpdir) if f.startswith("_dl_")]
            check("临时文件：断线中断也要清干净", not left, f"残留 {left}")
        finally:
            for k, v in saved.items():
                if v is None:
                    _sys.modules.pop(k, None)
                else:
                    _sys.modules[k] = v
    finally:
        D.download_many = real_dl
        if old_env is None:
            os.environ.pop("TRENDDESK_TQ_DB", None)
        else:
            os.environ["TRENDDESK_TQ_DB"] = old_env
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    ks = keys()
    if not ks:
        print("仓库里还没有切换表，先跑 download_tq.py --switch --all")
        return 1
    if len(sys.argv) > 1:
        ks = [k for k in ks if k == sys.argv[1]] or ks

    check_trading_day()
    check_contract_incremental()
    for k in ks:
        try:
            check_one(k)
            check_resample(k)
        except Exception as exc:
            check(f"{k} 读取", False, f"{type(exc).__name__}: {exc}")

    print("\n".join(_msgs))
    print("=" * 60)
    print(f"通过 {_ok} 项，失败 {_fail} 项，跳过 {_skip} 项")
    print("=" * 60)
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
