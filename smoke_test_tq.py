"""天勤数据链路的离线自测：切换表、复权因子、多周期合并。

不联网、不依赖天勤账户，只读 `data_tq/` 里已经下好的文件。有数据就全量校验，
没数据就跳过（打包后的环境本来就没有这份数据，不该因此报错）。

    python smoke_test_tq.py            # 校验全部已下载品种
    python smoke_test_tq.py RB_SHFE    # 只校验一个
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import date

import numpy as np  # noqa: F401  （保留给后续数值校验用）
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import tqdata as T  # noqa: E402
from ttrade import tqstore as S  # noqa: E402
from ttrade.tqfetch import adj_factor_series, factor_at  # noqa: E402

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
        # （ratio<1），因子随时间递增。两种都对，唯一能要求的是「量级不失控」——
        # 十年三十多次换月的连乘本来就可以到 0.19（铁矿石实测），连续性由下面的
        # 跳空检查负责。这里只拦 NaN / 0 / 爆炸这种明显的错。
        f = df["adj_factor"].to_numpy(float)
        check(f"{key} {period} 复权因子有限且为正",
              bool(np.isfinite(f).all() and (f > 0).all()),
              f"min={f.min():.4g} max={f.max():.4g}")
        check(f"{key} {period} 复权因子量级合理", bool(((f > 0.01) & (f < 100.0)).all()),
              f"min={f.min():.4f} max={f.max():.4f}")

        # 真正该钉死的不变式：**库里存的因子必须等于按当前切换表现算的因子**。
        # 「最新段恒为 1」是个假不变式 —— 它只在没有待换月时成立，而且抓不到最
        # 要命的那类 bug：切换表补全后历史 bar 的因子没跟着改（CJ、LH 曾整段停在
        # 1.0，复权后换月处留 3% 假跳空，比不复权还难看）。
        want = factor_at(adj_factor_series(sw), pd.Series(df["trading_day"]))
        check(f"{key} {period} 库中因子与切换表一致",
              bool(np.allclose(f, want, rtol=0, atol=1e-12)),
              f"最大偏差 {float(np.abs(f - want).max()):.3g}")

        # 待换月（切换日 > 最后一根 bar 的交易日）时，最新一段的因子就等于那次换月
        # 的比率 —— 这是对的：今晚 21:00 主连就换了，提前乘上去序列才不会在夜盘
        # 开盘跳一下。没有待换月时，最新段因子必须正好是 1。
        last_td = pd.Timestamp(df["trading_day"].iloc[-1])
        sw_dates = pd.to_datetime(sw["switch_date"])
        if sw_dates.max() > last_td:
            check(f"{key} {period} 待换月时因子等于最后比率",
                  abs(f[-1] - float(sw["ratio"].iloc[-1])) < 1e-9, f"{f[-1]}")
        else:
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


def check_live_helpers() -> None:
    """实时行情的纯逻辑：占位行、时区、与本地历史的拼接。

    这三件事都不需要联网，但每一件写错都很难在界面上看出来 ——
    占位行会让图上多一根塌到 0 的 K 线，时区错 8 小时会让实时段和本地历史
    错开一大截，拼接时覆盖本地最后一根会把复权因子换成 1.0。
    """
    try:
        from ttrade import tqlive as L
    except Exception as exc:
        skip("实时行情纯逻辑", f"无法导入 tqlive：{exc}")
        return

    ns = lambda s: int(pd.Timestamp(s).value)  # noqa: E731  UTC 纳秒
    raw = pd.DataFrame({
        "datetime": [0, 0, ns("2026-10-27 09:00"), ns("2026-10-27 10:00")],
        "open": [np.nan, np.nan, 100.0, 102.0],
        "high": [np.nan, np.nan, 103.0, 104.0],
        "low": [np.nan, np.nan, 99.0, 101.0],
        "close": [np.nan, np.nan, 102.0, 103.0],
        "volume": [0, 0, 10, 20],
        "open_oi": [0, 0, 5, 6],
        "close_oi": [0, 0, 5, 7],
    })
    bars = L.bars_from_serial(raw, "1h")
    check("实时①启动占位行被剔除", len(bars) == 2, f"{len(bars)} 行")
    check("实时①占位行不参与计算", bool(bars["close"].notna().all()))
    check("实时①UTC 纳秒转北京时间",
          bars.index[0] == pd.Timestamp("2026-10-27 17:00"),
          f"{bars.index[0]}")
    check("实时①索引无重复", bool(bars.index.is_unique))

    # 与本地历史拼接：本地最后一根 17:00，实时有 17:00 和 18:00 两根
    local = pd.DataFrame(
        {
            "open": [100.0], "high": [103.0], "low": [99.0], "close": [102.0],
            "volume": [10.0], "open_oi": [5.0], "close_oi": [5.0],
            "adj_factor": [0.97], "trading_day": [pd.Timestamp("2026-10-27").date()],
        },
        index=pd.DatetimeIndex(["2026-10-27 17:00"], name="datetime"),
    )
    merged = L.merge_live(local, bars)
    check("实时②只接严格晚于本地的 bar", len(merged) == 2, f"{len(merged)} 行")
    check("实时②不覆盖本地最后一根",
          float(merged.loc["2026-10-27 17:00", "adj_factor"]) == 0.97,
          f"{merged.loc['2026-10-27 17:00', 'adj_factor']}")
    check("实时②新 bar 因子为 1",
          float(merged.loc["2026-10-27 18:00", "adj_factor"]) == 1.0)
    check("实时②新 bar 有交易日", "trading_day" in merged.columns
          and pd.notna(merged.loc["2026-10-27 18:00", "trading_day"]))
    check("实时②拼接后索引递增", bool(merged.index.is_monotonic_increasing))

    empty = L.merge_live(pd.DataFrame(), bars)
    check("实时③没有本地历史时整段当实时",
          len(empty) == 2 and bool((empty["adj_factor"] == 1.0).all()))

    # 账户配置：实盘缺参数必须报错，不能静默退化成模拟
    bad = L.AccountSpec(name="x", kind="account", user="u", password="p")
    try:
        bad.build()
        check("实时④实盘缺参数报错", False, "没有抛异常")
    except ValueError as exc:
        check("实时④实盘缺参数报错", "缺少参数" in str(exc), str(exc))

    old = os.environ.get("LOCALAPPDATA")
    tmpdir = tempfile.mkdtemp(prefix="tqacct_")
    os.environ["LOCALAPPDATA"] = tmpdir
    try:
        specs = [L.AccountSpec(name="模拟", kind="kq", user="u1", password="p1"),
                 L.AccountSpec(name="实盘", kind="account", user="u2", password="p2",
                               broker_id="B", account_id="A", account_password="C")]
        L.save_accounts(specs)
        back = L.load_accounts()
        check("实时⑤账户列表存取一致",
              [s.name for s in back] == ["模拟", "实盘"] and back[1].broker_id == "B",
              f"{[s.name for s in back]}")
    finally:
        if old is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = old
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def check_cli_entry() -> None:
    """命令行入口能真正跑到「列出品种」这一步。

    `--stats` 和 `--help` 都在 argparse 阶段就返回了，不会走 `pick_pairs()` ——
    重构时漏掉一个 import，这两个命令照样绿，但一开跑就 NameError。
    """
    try:
        import download_tq as D
    except Exception as exc:  # noqa: BLE001
        skip("命令行入口", f"无法导入 download_tq：{exc}")
        return
    try:
        pairs = D.pick_pairs(None, True)
    except Exception as exc:  # noqa: BLE001
        check("命令行入口能列出品种", False, f"{type(exc).__name__}: {exc}")
        return
    check("命令行入口能列出品种", len(pairs) > 0, f"{len(pairs)} 个品种")
    check("命令行入口品种格式正确",
          all(isinstance(a, str) and isinstance(b, str) for a, b in pairs))


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


def _leftover(tmpdir: str) -> str:
    """残留的临时文件里还有内容的那些。槽位文件本身会留下，但必须是 0 字节。"""
    bad = []
    for fn in sorted(os.listdir(tmpdir)):
        if not fn.startswith("_dl_"):
            continue
        p = os.path.join(tmpdir, fn)
        if os.path.getsize(p) > 0:
            bad.append(f"{fn}({os.path.getsize(p)}B)")
    return "、".join(bad) or "无"


def _no_leftover(tmpdir: str) -> bool:
    return _leftover(tmpdir) == "无"


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

        # ---- 临时文件：正常结束与异常中断都不能留下数据 ----
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
            check("临时文件：正常结束不留数据", _no_leftover(tmpdir), _leftover(tmpdir))
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
            check("临时文件：断线中断也要清干净", _no_leftover(tmpdir), _leftover(tmpdir))
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


def check_factor_refresh() -> None:
    """复权因子是派生数据：换月点要跟序列对齐，因子要按切换表重算。

    钉住三件在真实数据上踩过的事：

    1. **首条换月之前的 bar 不能是 1.0** —— 那一段的因子应该是全部比率的连乘。
       漏了它 CJ 的复权价在首条换月日一根 bar 涨 280%。
    2. **切换日要挪到序列真正跳空的那天** —— 天勤的每日主力映射和主连 K 线偶尔
       差一个交易日。顺带覆盖 `UPDATE` 后必须消费结果，否则下一条 execute 报
       `unfinished result`（下载跑到第一个品种就崩）。
    3. **挪完要收敛** —— 再跑一遍不能继续往前挪。
    """
    import tempfile

    try:
        from ttrade import tqfetch as D
    except Exception as exc:  # noqa: BLE001
        skip("复权因子重算", f"无法导入 tqfetch：{exc}")
        return

    old_env = os.environ.get("TRENDDESK_TQ_DB")
    tmpdir = tempfile.mkdtemp(prefix="tqtest_")
    os.environ["TRENDDESK_TQ_DB"] = os.path.join(tmpdir, "t.duckdb")
    try:
        days = pd.bdate_range("2026-01-05", periods=5)
        closes = [100.0, 101.0, 102.0, 204.0, 205.0]   # 第 4 天翻倍
        bars = pd.DataFrame({
            "dt": days + pd.Timedelta(hours=9),
            "trading_day": days,
            "open": closes, "high": closes, "low": closes, "close": closes,
            "volume": 1, "open_oi": 1, "close_oi": 1,
        })
        # 切换日故意填早一天：序列是第 4 天才跳的
        rec = pd.DataFrame([{
            "switch_date": days[2], "old_tq": "SHFE.rb2601", "new_tq": "SHFE.rb2605",
            "old_ts": "RB2601.SHFE", "new_ts": "RB2605.SHFE", "ratio": 2.0,
        }])
        with S.connect() as con:
            S.upsert_bars(con, "RB", "SHFE", "1h", bars)
            S.upsert_switch(con, "RB", "SHFE", rec)
            n = D.realign_switch(con, "RB", "SHFE")
            check("切换日按序列跳空校正", n == 1, f"挪了 {n} 条")
            # 只能在同一个连接上查：DuckDB 的文件锁是排他的，再开只读连接会撞上
            got = con.execute(
                "SELECT switch_date FROM switch WHERE symbol='RB' AND exchange='SHFE'"
            ).fetchall()
            check("校正后的切换日", pd.Timestamp(got[0][0]) == days[3], str(got[0][0]))

            D.refresh_factors(con, "RB", "SHFE", "1h")
            df = con.execute(
                "SELECT close, adj_factor FROM bars WHERE symbol='RB' AND exchange='SHFE'"
                " AND period='1h' ORDER BY dt"
            ).df()
            f = df["adj_factor"].to_numpy(float)
            check("首条换月之前的因子是全部比率连乘", abs(f[0] - 2.0) < 1e-9, f"{f[0]}")
            check("换月之后的因子为 1", abs(f[-1] - 1.0) < 1e-9, f"{f[-1]}")
            adj = df["close"].to_numpy(float) * f
            check("复权后序列连续",
                  bool(np.allclose(adj[1:] / adj[:-1], [1.01, 1.009901, 1.0, 1.004902],
                                   atol=2e-3)),
                  f"{adj[-1] / adj[0]:.4f}")

            check("重复校正收敛", D.realign_switch(con, "RB", "SHFE") == 0)

            # 同一个连接上接着处理下一个品种。set_adj_factor 曾经用 fetchone 读计数，
            # 结果集没关掉，连接一直处于「有未完成结果」，下一个品种一 register 就报
            # ResourceInUseException —— 下载跑到第二个品种必崩，而单品种测试看不出来。
            try:
                S.upsert_bars(con, "HC", "SHFE", "1h", bars)
                D.refresh_factors(con, "HC", "SHFE", "1h")
                reused = True
            except Exception:  # noqa: BLE001
                reused = False
            check("重算因子后连接仍可复用", reused)
    except Exception as exc:  # noqa: BLE001
        check("复权因子重算", False, f"{type(exc).__name__}: {exc}")
    finally:
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
    check_live_helpers()
    check_cli_entry()
    check_factor_refresh()
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
