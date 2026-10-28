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

    # state() 必须先刷新 quote 再读 underlying：underlying 是在 _quote_snapshot()
    # 里跟着主连 quote 更新的，先构造 LiveState 会拿到上一轮的值 —— 订阅后的第一次
    # 必然为空，而自动下单正是靠它拿真实合约，空一次就丢一次信号。
    class _Q:
        instrument_id = "KQ.m@SHFE.rb"
        instrument_name = "螺纹主连"
        underlying_symbol = "SHFE.rb2701"
        datetime = "2026-10-28 09:00:00"
        last_price = 3047.0
        bid_price1 = 3046.0
        ask_price1 = 3047.0

    hub = L.LiveHub(L.AccountSpec())
    hub._quote = _Q()
    st = hub.state()
    check("实时⑥underlying 不能慢一拍", st.underlying == "SHFE.rb2701", repr(st.underlying))
    check("实时⑥quote 快照与 underlying 同源",
          st.quote.get("underlying") == "SHFE.rb2701", repr(st.quote.get("underlying")))

    # 持仓快照的合约名：`Position` 上没有 `symbol` 属性，合约代码是 get_position()
    # 返回的 dict 的 key（或 instrument_id）。取错属性会让持仓表的「合约」列整列空白，
    # 而委托/成交用的是 instrument_id 所以看着正常。
    from types import SimpleNamespace

    def _pos(**kw):
        base = dict(pos=-1, pos_long=0, pos_short=1, pos_long_today=0, pos_short_today=1,
                    open_price_long=float("nan"), open_price_short=109360.0,
                    float_profit=950.0, position_profit=0.0, margin=59752.0,
                    last_price=108860.0)
        base.update(kw)
        return SimpleNamespace(**base)

    hub._positions = {"SHFE.cu2612": _pos(instrument_id="cu2612")}
    rows = hub._positions_snapshot()
    check("实时⑦持仓合约名取 instrument_id", rows and rows[0]["symbol"] == "cu2612",
          repr(rows[0]["symbol"]) if rows else "空")
    hub._positions = {"SHFE.cu2612": _pos()}
    rows = hub._positions_snapshot()
    check("实时⑦没有 instrument_id 时回退到 dict 的 key",
          rows and rows[0]["symbol"] == "SHFE.cu2612",
          repr(rows[0]["symbol"]) if rows else "空")
    check("实时⑦空头持仓的 open_price_long 是 NaN（界面要按方向取）",
          rows and rows[0]["open_price_long"] != rows[0]["open_price_long"]
          and rows[0]["open_price_short"] == 109360.0,
          f"{rows[0]['open_price_long']}")

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


def check_live_worker() -> None:
    """登录账户不能被补历史卡住。

    交易页点「连接」的语义是登录账户，用户要看持仓/资金/委托 —— 这些在
    `hub.start()` 之后就有了。补历史一个品种十年 5m 要几十秒到几分钟，撞上
    天勤运维窗口（19:00-19:30）还会一直等到 19:31。所以：

    ① `need_history=False` 时完全不碰 `update_main`
    ② `need_history=True` 但补历史抛异常时，仍然把账户连上
    """
    try:
        from gui import live as LV
        import ttrade.tqfetch as TF
    except Exception as exc:  # noqa: BLE001
        skip("实时 worker", f"无法导入 gui.live：{exc}")
        return

    state = {"update": 0, "start": 0, "subscribe": 0, "closed": 0}
    holder: dict = {}

    class FakeHub:
        """只记调用，不连天勤。`wait()` 里让 worker 停，循环跑一轮就退出。"""

        def __init__(self, spec):
            self.spec = spec
            self.underlying = ""

        def start(self):
            state["start"] += 1

        def subscribe(self, symbol, exchange, period):
            state["subscribe"] += 1

        def wait(self, timeout=0.8):
            holder["w"].stop()
            return False

        def bars(self):
            return pd.DataFrame()

        def state(self):
            return None

        def close(self):
            state["closed"] += 1

    def run_once(need, update_impl):
        for k in state:
            state[k] = 0
        LV.LiveHub = FakeHub
        TF.update_main = update_impl
        w = LV.LiveWorker(None, "RB", "SHFE", "5m", need_history=need)
        holder["w"] = w
        logs: list[str] = []
        fails: list[str] = []
        w.log.connect(logs.append)
        w.failed.connect(fails.append)
        w.run()
        return logs, fails

    old_hub, old_upd = LV.LiveHub, TF.update_main
    try:
        # ① 交易页连接：不补历史
        logs, fails = run_once(
            False, lambda *a, **k: state.__setitem__("update", state["update"] + 1) or 0
        )
        check("实时⑧登录账户不补历史（不调 update_main）", state["update"] == 0,
              f"调用 {state['update']} 次")
        check("实时⑧登录账户仍然连上并订阅",
              state["start"] == 1 and state["subscribe"] == 1,
              f"start={state['start']} subscribe={state['subscribe']}")
        check("实时⑧登录过程没有报错", not fails, repr(fails))
        check("实时⑧日志里不提补历史",
              not any("补历史" in s or "补齐" in s for s in logs), repr(logs))

        # ② 研究页：要补历史，但补失败不能拖累连接
        def boom(*a, **k):
            state["update"] += 1
            raise RuntimeError("天勤运维窗口")

        logs, fails = run_once(True, boom)
        check("实时⑨补历史失败仍然连上",
              state["start"] == 1 and state["subscribe"] == 1,
              f"start={state['start']} subscribe={state['subscribe']}")
        check("实时⑨补历史失败不算连接失败", not fails, repr(fails))
        check("实时⑨日志说明补历史失败并继续",
              any("补历史失败" in s and "继续连接" in s for s in logs), repr(logs))

        # ③ 研究页正常补历史
        logs, _ = run_once(
            True, lambda *a, **k: state.__setitem__("update", state["update"] + 1) or 123
        )
        check("实时⑩补历史成功会写日志",
              any("本地历史已补齐" in s and "123" in s for s in logs), repr(logs))
    finally:
        LV.LiveHub = old_hub
        TF.update_main = old_upd


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

        def fake_dl(api, jobs, period, out_dir, tag, max_parallel=8, errors=None):
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

        # ---- 失败不能被当成「没数据」----
        # 上一轮真实下载里 54 个品种的合约全被静默吞掉，日志只写「新下 0 个合约」，
        # 看着像正常跳过。现在：失败一轮就重试，两轮都失败要计入 failed 并写进 status。
        rounds = {"n": 0}

        def flaky_dl(api, jobs, period, out_dir, tag, max_parallel=8, errors=None):
            rounds["n"] += 1
            out = {}
            for sym, _s, _e in jobs:
                if rounds["n"] == 1:
                    if errors is not None:
                        errors[sym] = "UnboundLocalError: cannot access local variable 'writer'"
                    continue
                out[sym] = _fake_frame(today - pd.Timedelta(hours=1))
            return out

        D.download_many = flaky_dl
        r4 = D.write_contracts(_FakeApi(["SHFE.ag2609", "SHFE.ag2612"]), "AG", "SHFE", "5m", 1)
        check("下载失败：重试一轮后成功", r4["failed"] == 0 and r4["contracts"] == 2,
              f"failed={r4['failed']} contracts={r4['contracts']} rounds={rounds['n']}")

        def always_fail(api, jobs, period, out_dir, tag, max_parallel=8, errors=None):
            if errors is not None:
                for sym, _s, _e in jobs:
                    errors[sym] = "ConnectionError: boom"
            return {}

        D.download_many = always_fail
        r5 = D.write_contracts(_FakeApi(["SHFE.au2609", "SHFE.au2612"]), "AU", "SHFE", "5m", 1)
        check("下载失败：两轮都失败要报出来",
              r5["failed"] == 2 and "失败" in r5["status"],
              f"failed={r5['failed']} status={r5['status']}")
        check("下载失败：失败合约不能被记成空合约",
              r5["blank"] == 0, f"blank={r5['blank']}")

        # ---- 临时文件：正常结束与异常中断都不能留下数据 ----
        # 下面的用例要测真实的 download_many，先把前面替换掉的假下载器还回去。
        D.download_many = real_dl
        import sys as _sys
        import types

        class _Task:
            def __init__(self, api, symbol_list, dur_sec, start_dt, end_dt, csv_file_name):
                self.sym = symbol_list
                self.path = csv_file_name
                # DataDownloader 写出的表头是 `合约.datetime`，不是 `dt` —— 这里必须
                # 照真实格式造，否则 _read_downloaded 认不出时间列，测试会假通过。
                df = _fake_frame(pd.Timestamp("2026-01-05 15:00")).rename(
                    columns={"dt": "datetime"}
                )
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


def check_main_backfill() -> None:
    """库里只有近几年数据时，增量必须把前面那段也补上。

    增量原来只看 `last_dt`：库里最后一根在哪就从哪往前推 3 天，**从不看第一根**。
    早期版本用两年窗口跑过一轮的品种，之后每次只补最新几根，那几年永远补不上，
    而日志显示「+14 行」看着完全正常。RB 的 1h 实测停在两年前（3369 根），
    5m 同期是 164544 根。
    """
    import tempfile

    try:
        from ttrade import tqfetch as D
    except Exception as exc:  # noqa: BLE001
        skip("主连补历史缺口", f"无法导入 tqfetch：{exc}")
        return

    old_env = os.environ.get("TRENDDESK_TQ_DB")
    tmpdir = tempfile.mkdtemp(prefix="tqmain_")
    os.environ["TRENDDESK_TQ_DB"] = os.path.join(tmpdir, "t.duckdb")
    real_dl = D.download_many
    try:
        today = pd.Timestamp.today().normalize()
        have = pd.DataFrame({
            "dt": pd.date_range(today - pd.Timedelta(days=700), today, freq="h"),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 1, "open_oi": 1, "close_oi": 1,
        })
        have["adj_factor"] = 1.0
        with S.connect() as con:
            S.upsert_bars(con, "RB", "SHFE", "1h", have)

        spans: list[tuple] = []

        def fake_dl(api, jobs, period, out_dir, tag, max_parallel=8, errors=None):
            spans.extend([(j[1], j[2]) for j in jobs])
            out = {}
            for sym, s, e in jobs:
                n = max(2, (e - s).days * 4)
                out[sym] = pd.DataFrame({
                    "dt": pd.date_range(pd.Timestamp(s), periods=n, freq="6h"),
                    "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                    "volume": 1, "open_oi": 1, "close_oi": 1,
                })
            return out

        D.download_many = fake_dl
        D.write_main_many(None, [("RB", "SHFE")], "1h", 10)

        hard = (today - pd.Timedelta(days=3650)).date()
        first = have["dt"].iloc[0]
        last = have["dt"].iloc[-1]
        check("主连补历史：库里起点偏晚时要拆成两段", len(spans) == 2, f"{len(spans)} 段")
        if len(spans) == 2:
            (h0, h1), (t0, _t1) = spans
            check("主连补历史：头段从回溯区间开始", h0 <= hard + pd.Timedelta(days=2),
                  f"{h0} vs {hard}")
            check("主连补历史：头段止于库里第一根之前", h1 < first.date(), f"{h1}")
            check("主连补历史：尾段从最后一根往前推",
                  t0 <= last.date() - pd.Timedelta(days=2), f"{t0}")

        # 库里数据已经完整覆盖回溯区间时，不能多下一段
        spans.clear()
        with S.connect() as con:
            full = pd.DataFrame({
                "dt": pd.date_range(today - pd.Timedelta(days=3800), today, freq="h"),
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                "volume": 1, "open_oi": 1, "close_oi": 1,
            })
            full["adj_factor"] = 1.0
            S.upsert_bars(con, "HC", "SHFE", "1h", full)
        D.write_main_many(None, [("HC", "SHFE")], "1h", 10)
        check("主连补历史：覆盖完整时不重复补", len(spans) == 1, f"{len(spans)} 段")
    except Exception as exc:  # noqa: BLE001
        check("主连补历史缺口", False, f"{type(exc).__name__}: {exc}")
    finally:
        D.download_many = real_dl
        if old_env is None:
            os.environ.pop("TRENDDESK_TQ_DB", None)
        else:
            os.environ["TRENDDESK_TQ_DB"] = old_env
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def check_strategy_store() -> None:
    """策略配置 / 策略库 / 组合 / 追单设置。

    重点盯两件容易出人命的事：**sid 在存盘时就定下来**（不然每次重启换身份，
    按 sid 改、按 sid 删全部落空），以及**库和组合互不影响**（模板改了不能
    悄悄改变正在跑的策略）。
    """
    import shutil

    from ttrade import strategy_store as St

    tmpdir = tempfile.mkdtemp(prefix="td_store_")
    old_dir = St.user_dir
    St.user_dir = lambda: tmpdir
    try:
        a = St.StrategyConfig(
            symbol="RB", exchange="SHFE", period="1d",
            rule_key="ma_cross", rule_params={"fast": 5}, volume=2,
        )
        check("策略配置：构造时 sid 为空", a.sid == "", a.sid)
        St.save_portfolio([a])
        check("策略配置：存盘时补上 sid", bool(a.sid), a.sid)
        got = St.load_portfolio()
        check("策略配置：读回的 sid 与存盘一致",
              bool(got) and got[0].sid == a.sid,
              f"{got[0].sid if got else '-'} vs {a.sid}")

        b = got[0].clone(volume=7)
        b.sid = got[0].sid
        St.update_portfolio(b)
        check("组合：按 sid 改手数生效", St.load_portfolio()[0].volume == 7,
              str(St.load_portfolio()[0].volume))

        # 手改 JSON 复制粘贴会造出两条同 sid 的记录，删一条会把另一条也删掉
        St.save_portfolio([a, a])
        two = St.load_portfolio()
        check("组合：重复 sid 自动去重", len({c.sid for c in two}) == 2,
              str([c.sid for c in two]))

        St.save_library([])
        St.save_portfolio([])
        lib = St.add_to_library(a)
        item = St.add_to_portfolio(lib)
        item.volume = 9
        St.update_portfolio(item)
        check("组合与库解耦：改组合不影响库",
              St.load_library()[0].volume == a.volume,
              f"{St.load_library()[0].volume} vs {a.volume}")
        check("组合与库解耦：sid 不同", item.sid != lib.sid, f"{item.sid} / {lib.sid}")

        # 「保存配置 + 加入组合」连点两下不能变成两条，否则跑起来是双倍仓位
        check("重复入库：同一套参数认得出", lib.same_shape(lib.clone()), "未认出")
        check("重复入库：手数不同不算重复",
              not lib.same_shape(lib.clone(volume=lib.volume + 1)), "误判为重复")
        check("重复入库：账户不同不算重复",
              not lib.same_shape(lib.clone(account="另一个")), "误判为重复")

        St.save_chase(St.ChaseConfig(enabled=True, wait_sec=3, retries=2, ticks=5))
        c = St.load_chase()
        check("追单设置：往返一致", (c.wait_sec, c.retries, c.ticks) == (3, 2, 5),
              str(c.to_dict()))
        St._save(St.CHASE_FILE, {"wait_sec": 0, "retries": -3, "ticks": -1})
        c = St.load_chase()
        check("追单设置：非法值收敛",
              c.wait_sec >= 1 and c.retries >= 0 and c.ticks >= 0, str(c.to_dict()))

        # 配置文件坏了不能连带程序起不来
        with open(os.path.join(tmpdir, St.PORT_FILE), "w", encoding="utf-8") as fh:
            fh.write("{坏掉的")
        check("组合：坏文件读成空表", St.load_portfolio() == [])
    except Exception as exc:  # noqa: BLE001
        check("策略配置与组合", False, f"{type(exc).__name__}: {exc}")
    finally:
        St.user_dir = old_dir
        shutil.rmtree(tmpdir, ignore_errors=True)


def check_strategy_runner() -> None:
    """多策略引擎：平今判断、理论/实际比对、对齐只补托管合约。"""
    from ttrade import strategy_runner as R
    from ttrade import strategy_store as St
    from ttrade.strategy_runner import PortfolioRunner
    from ttrade.tqlive import AccountSpec

    class Pos:
        def __init__(self, long=0, short=0, long_today=0, short_today=0) -> None:
            self.pos_long = long
            self.pos_short = short
            self.pos_long_today = long_today
            self.pos_short_today = short_today

    class Quote:
        price_tick = 1.0
        ask_price1 = 3050.0
        bid_price1 = 3049.0
        last_price = 3049.5

    class Api:
        def __init__(self) -> None:
            self.orders: list[dict] = []

        def get_quote(self, _sym):
            return Quote()

        def insert_order(self, **kw):
            self.orders.append(kw)
            return "oid"

    try:
        r = PortfolioRunner(AccountSpec(name="t", kind="sim"), St.ChaseConfig())

        r._pos = {"SHFE.rb2701": Pos(long=3, long_today=2)}
        check("平今：今仓够就平今", r._offset_for("SHFE.rb2701", "SELL", 2) == "CLOSETODAY")
        check("平今：今仓不够就平昨", r._offset_for("SHFE.rb2701", "SELL", 3) == "CLOSE")
        check("平今：空仓时开仓", r._offset_for("SHFE.cu2612", "BUY", 1) == "OPEN")
        r._pos = {"CFFEX.IF2612": Pos(long=2, long_today=2)}
        check("平今：中金所不分平今",
              r._offset_for("CFFEX.IF2612", "SELL", 2) == "CLOSE")

        cfg = St.StrategyConfig(sid="s1", symbol="RB", exchange="SHFE",
                                period="1d", rule_key="ma_cross", volume=2)
        st = r.add(cfg)
        st.running = True
        st.underlying = "SHFE.rb2701"
        st.signal = 1
        st.target = 2
        r._pos = {"SHFE.rb2701": Pos(), "SHFE.cu2612": Pos(short=1)}

        rows = {x["symbol"]: x for x in r.mismatch()}
        check("比对：理论仓与账户仓不一致能列出来", set(rows) == {"SHFE.rb2701", "SHFE.cu2612"},
              str(sorted(rows)))
        check("比对：有策略在跑的合约标为托管", rows["SHFE.rb2701"]["managed"] is True)
        check("比对：没策略在跑的合约不算托管", rows["SHFE.cu2612"]["managed"] is False)
        check("比对：差额方向正确", rows["SHFE.rb2701"]["delta"] == 2,
              str(rows["SHFE.rb2701"]["delta"]))

        # 一键对齐不能顺手把「策略已停但仓位还在」的合约平掉
        r.api = Api()
        n = r.align()
        check("对齐：只补托管合约", n == 1 and len(r.api.orders) == 1, f"{n} 笔")
        if r.api.orders:
            o = r.api.orders[0]
            check("对齐：方向与手数按差额",
                  o["symbol"] == "SHFE.rb2701" and o["direction"] == "BUY"
                  and o["volume"] == 2 and o["offset"] == "OPEN",
                  str(o))

        # 改手数要让目标仓位立刻跟上，否则得等下一根 bar
        cfg2 = cfg.clone(volume=5)
        cfg2.sid = cfg.sid
        r.update(cfg2)
        check("改手数：目标仓位立刻重算", st.target == 5, str(st.target))
        check("改手数：槽位配置已换新", st.cfg.volume == 5, str(st.cfg.volume))

        # 换月：老合约上的仓位不会自己消失，要记下来按 0 目标清掉
        st.underlying = "SHFE.rb2612"
        st.retired = ["SHFE.rb2611"]
        r._pos = {"SHFE.rb2611": Pos(long=3), "SHFE.rb2612": Pos()}
        st.target = 2
        theo = r.theoretical()
        check("换月：老合约目标是 0", theo.get("SHFE.rb2611") == 0, str(theo))
        check("换月：新合约照常计入", theo.get("SHFE.rb2612") == 2, str(theo))
        rows2 = {x["symbol"]: x for x in r.mismatch()}
        check("换月：老合约仍算托管", rows2["SHFE.rb2611"]["managed"] is True,
              str(rows2.get("SHFE.rb2611")))
        r.api = Api()
        n2 = r.align()
        check("换月：对齐会把老合约平掉", n2 == 2 and len(r.api.orders) == 2, f"{n2} 笔")
        old = [o for o in r.api.orders if o["symbol"] == "SHFE.rb2611"]
        check("换月：老合约下的是卖平 3 手",
              bool(old) and old[0]["direction"] == "SELL" and old[0]["volume"] == 3
              and old[0]["offset"] == "CLOSE", str(old))

        # 停掉的策略不参与理论持仓汇总
        st.running = False
        st.retired = []
        check("理论持仓：只算运行中的策略", r.theoretical() == {}, str(r.theoretical()))

        check("实时周期：覆盖分钟级与小时级",
              all(k in R.RT_SEC for k in ("5m", "15m", "30m", "1h", "2h", "4h", "1d")),
              str(sorted(R.RT_SEC)))

        # 天勤的 kline 是 DataFrame，`kl["datetime"][-1]` 是按**标签**取值，
        # 会抛 KeyError: -1 —— 曾经让每条策略在第一个 tick 就停机。这里用一份
        # 结构相同的 DataFrame 把整条「K 线 → 信号 → 目标仓位」走一遍。
        import pandas as pd

        n = 80
        base = pd.Timestamp("2026-01-05 09:00").value
        bars = pd.DataFrame({
            "datetime": [base + i * 300 * 10**9 for i in range(n)],
            "open": [3000.0 + i for i in range(n)],
            "high": [3002.0 + i for i in range(n)],
            "low": [2998.0 + i for i in range(n)],
            "close": [3000.0 + i for i in range(n)],
            "volume": [100.0] * n,
            "open_oi": [1000.0] * n,
            "close_oi": [1000.0] * n,
        })
        c5 = St.StrategyConfig(sid="s5", symbol="RB", exchange="SHFE",
                               period="5m", rule_key="ma_cross", volume=2,
                               rule_params={"fast": 5, "slow": 20})
        st5 = r.add(c5)
        r._klines[c5.key()] = bars
        r._quotes[c5.key()] = None
        st5.running = True
        r._refresh_signal(st5)
        check("信号：能从天勤 K 线算出目标仓位",
              st5.error == "" and st5.last_bar == int(bars["datetime"].iloc[-1]),
              f"last_bar={st5.last_bar} err={st5.error}")
        check("信号：单边上涨给多头、手数乘上", st5.signal == 1 and st5.target == 2,
              f"signal={st5.signal} target={st5.target}")
        # 同一根 bar 里不能重复计算
        st5.signal = 0
        r._refresh_signal(st5)
        check("信号：同一根 bar 不重算", st5.signal == 0, str(st5.signal))

        # 周期不认识时必须在 enable 就拒掉，而不是等订阅时抛 KeyError 把策略打死
        bad = St.StrategyConfig(sid="s9", symbol="RB", exchange="SHFE",
                                period="7m", rule_key="ma_cross", volume=1)
        st_bad = r.add(bad)
        r.enable("s9", True)
        check("启停：不支持的周期被拒绝",
              not st_bad.running and "周期" in st_bad.error, st_bad.error)
    except Exception as exc:  # noqa: BLE001
        check("多策略引擎", False, f"{type(exc).__name__}: {exc}")


def check_library_docs() -> None:
    """因子库 / 策略库的说明文档必须覆盖注册表，不能漏、不能多。"""
    from ttrade import library_docs as L
    from ttrade.factors import FACTORS
    from ttrade.strategies import STRATEGIES

    try:
        # 用户注册的因子不进 `FACTOR_DOCS`（公式和源码存在 `FactorSpec` 上），
        # 所以「缺文档」只对内置因子成立：内置因子没有 spec.formula 兜底
        missing = [k for k in FACTORS if k not in L.FACTOR_DOCS and not FACTORS[k].formula]
        extra = [k for k in L.FACTOR_DOCS if k not in FACTORS]
        check("因子文档：覆盖全部注册因子", not missing, f"缺 {missing}")
        check("因子文档：没有多余条目", not extra, f"多 {extra}")

        no_formula = [k for k in FACTORS if "未收录" in (L.factor_doc(k).formula or "")]
        check("因子文档：每条都有公式", not no_formula, str(no_formula))

        no_code = [k for k in FACTORS if not L.factor_doc(k).code.strip()]
        check("因子文档：能取到实现代码", not no_code, f"{len(no_code)} 条取不到")

        # 同因子：用户注册的策略不在 STRATEGY_DOCS 里，靠 spec.code 兜底
        smissing = [k for k in STRATEGIES if k not in L.STRATEGY_DOCS
                    and not STRATEGIES[k].code]
        check("策略文档：覆盖全部注册策略", not smissing, f"缺 {smissing}")
        thin = [
            k for k in STRATEGIES
            if not all(getattr(L.strategy_doc(k), f) for f in ("logic", "edge", "fits", "risk"))
        ]
        check("策略文档：逻辑/优势/适应/风险齐全", not thin, str(thin))

        brief = L.factor_brief()
        check("因子清单：一行一个、含公式",
              brief.count("\n") + 1 == len(FACTORS) and "｜" in brief,
              f"{brief.count(chr(10)) + 1} 行")
        check("策略清单：含默认参数",
              all(k in L.strategy_brief() for k in STRATEGIES), "缺键")
    except Exception as exc:  # noqa: BLE001
        check("因子库与策略库文档", False, f"{type(exc).__name__}: {exc}")


def check_agent() -> None:
    """大模型接入层：配置、动作解析、提示词拼装。不联网。"""
    import shutil

    from ttrade import agent as A

    tmpdir = tempfile.mkdtemp(prefix="td_agent_")
    old_dir = A.user_dir
    A.user_dir = lambda: tmpdir
    try:
        cfg = A.AgentConfig()
        ok, why = cfg.ready()
        check("模型配置：没填 Key 时不能发请求", not ok and "Key" in why, why)

        cfg.api_key = "sk-test"
        cfg.model = "deepseek-chat"
        A.save_config(cfg)
        back = A.load_config()
        check("模型配置：往返一致",
              back.api_key == "sk-test" and back.model == "deepseek-chat",
              back.model)

        check("模型配置：地址拼得对",
              back.endpoint("/chat/completions")
              == "https://api.deepseek.com/v1/chat/completions",
              back.endpoint("/chat/completions"))

        # 推理型模型不能带 temperature，各家会直接 400
        rc = A.AgentConfig(model="deepseek-reasoner", api_key="k")
        body = A.LLMClient(rc)._body([], True)
        check("推理型模型不带温度", "temperature" not in body, str(body))
        body2 = A.LLMClient(A.AgentConfig(model="gpt-4o", api_key="k"))._body([], True)
        check("普通模型带温度", "temperature" in body2, str(body2))

        # 动作解析
        txt = ('先看螺纹。\n```json\n'
               '{"action":"trade","mode":"auto","symbol":"RB","exchange":"SHFE",'
               '"period":"1d","strategy":"ma_cross","params":{"fast":50},"volume":2}\n```')
        a = A.parse_action(txt)
        check("动作：按策略交易能解析", a.ok and a.payload["strategy"] == "ma_cross",
              a.error)
        check("动作：说明里有品种和手数",
              "RB_SHFE" in a.describe() and "2 手" in a.describe(), a.describe())

        m = A.parse_action('```json\n{"action":"trade","mode":"manual","symbol":"CU",'
                           '"exchange":"SHFE","side":"SELL","offset":"CLOSETODAY",'
                           '"volume":1}\n```')
        check("动作：主观卖出平今", m.ok and m.describe() == "卖出平今 CU_SHFE 1 手",
              m.describe())

        bad = A.parse_action('```json\n{"action":"trade","mode":"manual","symbol":"CU"}\n```')
        check("动作：主观交易缺 side 要报错", not bad.ok and "side" in bad.error, bad.error)
        check("动作：没有动作块就是空", A.parse_action("聊两句").kind == "")
        check("动作：坏 JSON 不崩", A.parse_action("```json\n{坏的}\n```").error != "")

        # 提示词
        msgs = A.build_messages([{"role": "user", "content": "旧问题"}], "新问题")
        check("提示词：系统提示在最前",
              msgs[0]["role"] == "system" and "趋势跟踪研究台" in msgs[0]["content"])
        check("提示词：历史在前、本轮在后",
              msgs[-1]["content"] == "新问题" and len(msgs) == 3, f"{len(msgs)} 条")
        check("历史：超长会截断", len(A.trim_history([{"role": "user", "content": "x"}] * 100)) == 40)

        # 提示词里的示例必须是**合法 JSON**。模型会照着抄，示例里塞真换行等于
        # 教它输出 `parse_action` 解析不了的块 —— 而且错得很隐蔽：模型看着像
        # 在正常工作，实际每个动作都静默失败
        import json as _json
        import re as _re

        blocks = _re.findall(r"```json\s*(\{.*?\})\s*```", A.SYSTEM_PROMPT, _re.S)
        broken = []
        for blk in blocks:
            try:
                _json.loads(blk)
            except ValueError:
                broken.append(blk[:30])
        check("提示词：示例动作块都是合法 JSON", bool(blocks) and not broken, str(broken))

        kinds = {_json.loads(b).get("action") for b in blocks if b not in broken}
        check("提示词：四种动作都有示例",
              {"trade", "backtest", "factor", "strategy"} <= kinds, str(kinds))
    except Exception as exc:  # noqa: BLE001
        check("大模型接入层", False, f"{type(exc).__name__}: {exc}")
    finally:
        A.user_dir = old_dir
        shutil.rmtree(tmpdir, ignore_errors=True)


def check_agent_stream() -> None:
    """流式解析。起一个本地假接口，验证 SSE 拼接、非流式回退、HTTP 错误、中途停止。

    SSE 是最容易「静默失效」的一段：解析错了不会抛异常，只会一个字都出不来，
    看起来像模型没回话。
    """
    import json as _json
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from ttrade import agent as A

    mode = {"v": "sse"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):   # 别把访问日志打到测试输出里
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            self.rfile.read(n)
            if mode["v"] == "401":
                body = b'{"error":{"message":"invalid api key"}}'
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if mode["v"] == "plain":
                body = _json.dumps(
                    {"choices": [{"message": {"content": "纯 JSON 回复"}}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if mode["v"] == "slow":
                # 推一段就长时间不出声：模拟模型思考时 socket 上没有字节，
                # `should_stop` 检查不到，只有 abort() 关掉 response 才停得下来
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk = {"choices": [{"delta": {"content": "半"}}]}
                self.wfile.write(
                    f"data: {_json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
                )
                self.wfile.flush()
                time.sleep(3.0)
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionAbortedError):
                    pass
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            pieces = ["螺纹", "在均线", "上方。"] if mode["v"] == "sse" else ["x"] * 200
            try:
                for piece in pieces:
                    chunk = {"choices": [{"delta": {"content": piece}}]}
                    self.wfile.write(
                        f"data: {_json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
                    )
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError):
                pass      # 客户端中途停止，正常

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    cfg = A.AgentConfig(provider="custom", base_url=f"http://127.0.0.1:{port}/v1",
                        model="m", api_key="k")
    client = A.LLMClient(cfg)
    msgs = [{"role": "user", "content": "hi"}]
    try:
        got: list[str] = []
        text = client.chat(msgs, on_delta=got.append)
        check("流式：多段能拼成完整回答", text == "螺纹在均线上方。", repr(text))
        check("流式：每段都回调了", len(got) == 3, f"{len(got)} 段")

        mode["v"] = "plain"
        check("流式：网关不支持时回退到整段 JSON",
              client.chat(msgs) == "纯 JSON 回复", "回退失败")

        mode["v"] = "401"
        try:
            client.chat(msgs)
            check("流式：401 要抛 AgentError", False, "没抛")
        except A.AgentError as exc:
            check("流式：401 要抛 AgentError", "401" in str(exc), str(exc)[:60])

        mode["v"] = "sse"
        seen = {"n": 0}

        def stop():
            seen["n"] += 1
            return seen["n"] > 1

        check("流式：能中途停止", client.chat(msgs, should_stop=stop) == "螺纹", "停不下来")

        # abort() 是另一条路：模型思考时 socket 上几十秒没有字节，`should_stop`
        # 根本没机会被检查，线程就卡在 iter_lines 的阻塞读上。关掉 response 才能
        # 让它立刻抛出来 —— 否则界面上的「停止」要等 HTTP 超时（默认 120 秒）。
        mode["v"] = "slow"
        box: dict = {}

        def run_chat() -> None:
            try:
                box["text"] = client.chat(msgs)
            except Exception as exc:  # noqa: BLE001
                box["err"] = exc

        th = threading.Thread(target=run_chat, daemon=True)
        th.start()
        time.sleep(0.4)          # 等它进到阻塞读
        client.abort()
        th.join(1.5)             # 服务器要 3 秒后才推 DONE：1.5 秒内退出就说明是 abort 生效
        check("流式：abort 能掐断阻塞中的读", not th.is_alive(),
              f"1.5 秒后还卡着（服务器 3 秒后才收尾）")
        # 掐断后可能抛异常，也可能读到空行后正常返回已收到的部分 —— 两种都算对，
        # 关键是不能把「半截」变成「整段」
        check("流式：abort 后保留已收到的部分",
              box.get("text") == "半" or "err" in box, repr(box))
        check("流式：abort 之后 response 已清空", client._resp is None,
              repr(client._resp))
    except Exception as exc:  # noqa: BLE001
        check("流式解析", False, f"{type(exc).__name__}: {exc}")
    finally:
        srv.shutdown()


def check_agent_thread() -> None:
    """Agent 页的线程必须能退出，否则「发送按钮点一次就灰掉」。

    `thread.started` 连的是普通函数（`worker.run`），它返回后线程的事件循环还在
    跑 —— 不显式 `quit()` 就永远不发 `finished`，`_on_thread_done` 也就永远不跑，
    `_busy` 卡在 True。用户看到的是：第一轮答完，输入框里还能打字，但发送按钮
    一直灰着；点「停止」也没用，因为停止同样靠 `_on_thread_done` 收尾。
    """
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication

        from ttrade import agent as A
        from gui import agent_page as G
    except Exception as exc:  # noqa: BLE001
        skip("Agent 线程", f"无法导入：{exc}")
        return

    app = QApplication.instance() or QApplication([])
    calls: list[int] = []

    def fake_chat(self, messages, on_delta=None, should_stop=None):
        calls.append(1)
        if on_delta is not None:
            on_delta("答")
        return "答"

    old = A.LLMClient.chat
    A.LLMClient.chat = fake_chat
    pg = None
    try:
        pg = G.AgentPage()
        pg.cfg.api_key = "k"
        pg.cfg.provider = "deepseek"
        pg.cfg.base_url = "https://api.deepseek.com/v1"
        pg.cfg.model = "deepseek-chat"
        pg.cfg.confirm_trade = False

        def pump(n: int = 80) -> None:
            for _ in range(n):
                app.processEvents()
                time.sleep(0.01)

        pg.ed_input.setPlainText("一")
        pg._send()
        pump()
        check("Agent 线程：第一轮结束后发送按钮恢复",
              not pg._busy and pg.btn_send.isEnabled(),
              f"busy={pg._busy} enabled={pg.btn_send.isEnabled()}")
        check("Agent 线程：第一轮结束状态回到就绪",
              pg.lbl_state.text() == "就绪", repr(pg.lbl_state.text()))

        pg.ed_input.setPlainText("二")
        pg._send()
        pump()
        check("Agent 线程：能连发第二轮", len(calls) == 2, f"{len(calls)} 次")
        check("Agent 线程：第二轮后仍然就绪",
              not pg._busy and pg.lbl_state.text() == "就绪",
              f"busy={pg._busy} state={pg.lbl_state.text()!r}")
    except Exception as exc:  # noqa: BLE001
        check("Agent 线程", False, f"{type(exc).__name__}: {exc}")
    finally:
        A.LLMClient.chat = old
        if pg is not None:
            try:
                pg.shutdown()
            except Exception:  # noqa: BLE001
                pass


def check_user_factors() -> None:
    """Agent 注册的自定义因子：执行沙箱、试算校验、落盘往返、删除。

    这条链最容易出的问题是「注册成功了但选不到」：因子进了 `FACTORS`，却不在
    `CATEGORIES` 里，而下拉是按 `CATEGORIES` 分组的 —— 库里有、界面上没有。
    """
    from ttrade import factors as F
    from ttrade import library_docs as L
    from ttrade import userfactors as UF

    tmpdir = tempfile.mkdtemp(prefix="td_uf_")
    old_path = UF.path
    UF.path = lambda: os.path.join(tmpdir, "user_factors.json")
    key = "zz_smoke_test"
    try:
        # 1) `import os` 必须失败。白名单 builtins 里没有 `__import__`，
        #    这一条挡的是模型「顺手读个文件」，不是蓄意攻击
        ok, msg = UF.register(UF.UserFactor(
            key=key, label="试", func="f_x",
            code="import os\ndef f_x(df):\n    return df['close']"))
        check("用户因子：白名单挡住 import", not ok and "跑不起来" in msg, msg)

        # 2) 返回值不是 Series
        ok, msg = UF.register(UF.UserFactor(
            key=key, label="试", func="f_x",
            code="def f_x(df):\n    return np.zeros(len(df))"))
        check("用户因子：返回 ndarray 被拒", not ok and "Series" in msg, msg)

        # 3) 长度不对
        ok, msg = UF.register(UF.UserFactor(
            key=key, label="试", func="f_x",
            code="def f_x(df):\n    return _col(df, 'close').iloc[:-5]"))
        check("用户因子：长度不一致被拒", not ok and "长度" in msg, msg)

        # 4) 索引不对：dropna + reset_index 是模型最常犯的错
        ok, msg = UF.register(UF.UserFactor(
            key=key, label="试", func="f_x",
            code="def f_x(df):\n    return _col(df, 'close').reset_index(drop=True)"))
        check("用户因子：索引不对齐被拒", not ok and "索引" in msg, msg)

        # 5) 全 NaN 只警告不拦：缺衍生列的因子在别的标的上是能用的
        ok, msg = UF.register(UF.UserFactor(
            key=key, label="试", func="f_x",
            code="def f_x(df):\n    return _nan_series(df)"))
        check("用户因子：全 NaN 只给警告", ok and "NaN" in msg, msg)
        UF.remove(key)

        # 6) 正常因子
        good = UF.UserFactor(
            key=key, label="冒烟测试因子", category="冒烟",
            desc="5 日均值比 20 日均值", formula="MA(close,5)/MA(close,20)",
            hint_dir=1, func="f_zz",
            code=("def f_zz(df):\n"
                  "    c = _col(df, 'close')\n"
                  "    return _safe_div(c.rolling(5).mean(), c.rolling(20).mean())"))
        ok, msg = UF.register(good)
        check("用户因子：正常因子注册成功", ok, msg)
        check("用户因子：进了 FACTORS", key in F.FACTORS, key)
        check("用户因子：新类别补进 CATEGORIES", "冒烟" in F.CATEGORIES,
              str(F.CATEGORIES[-3:]))

        doc = L.factor_doc(key)
        check("用户因子：文档带公式与源码",
              doc is not None and doc.formula == good.formula
              and "def f_zz" in doc.code,
              "" if doc is None else f"formula={doc.formula!r}")

        # 7) 重名要拒绝，不能悄悄覆盖内置因子
        ok2, msg2 = UF.register(good)
        check("用户因子：重名被拒", not ok2 and "已经有一个" in msg2, msg2)
        ok3, msg3 = UF.register(UF.UserFactor(
            key="mom_60", label="撞名", func="f_y",
            code="def f_y(df):\n    return _col(df, 'close')"))
        check("用户因子：不能覆盖内置因子", not ok3, msg3)

        # 8) 代号格式
        ok4, msg4 = UF.register(UF.UserFactor(
            key="Zz-Bad", label="坏代号", func="f_z",
            code="def f_z(df):\n    return _col(df, 'close')"))
        check("用户因子：代号格式校验", not ok4 and "代号" in msg4, msg4)

        # 9) 重启后能挂回来
        box: dict = {}
        n = UF.register_all_into(box)
        check("用户因子：重启后从盘里挂回来", n == 1 and key in box, f"n={n}")
        check("用户因子：挂回来的函数能跑",
              key in box and len(box[key].func(UF.sample_df())) == len(UF.sample_df()),
              "")

        # 10) 删除
        check("用户因子：删除后 FACTORS 里没了", UF.remove(key) and key not in F.FACTORS, "")
        check("用户因子：删完盘里也没了", UF.register_all_into({}) == 0, "")
        check("用户因子：内置因子删不掉", not UF.remove("mom_60"), "")
    except Exception as exc:  # noqa: BLE001
        check("用户自定义因子", False, f"{type(exc).__name__}: {exc}")
    finally:
        UF.path = old_path
        F.FACTORS.pop(key, None)
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def check_user_strategies() -> None:
    """Agent 注册的自定义策略：沙箱、试算校验、落盘往返、删除。

    比因子多一条「不能全程空仓」—— signal 全 0 的策略在界面上看起来是「策略
    有效但从不持仓」，用户会以为是数据问题，其实是条件永远不成立。
    """
    from ttrade import library_docs as L
    from ttrade import strategies as S
    from ttrade import userstrategies as US

    tmpdir = tempfile.mkdtemp(prefix="td_us_")
    old_path = US.path
    US.path = lambda: os.path.join(tmpdir, "user_strategies.json")
    key = "zz_smoke_strat"
    try:
        ok, msg = US.register(US.UserStrategy(
            key=key, label="试", func="sig_x",
            code="import os\ndef sig_x(df):\n    return df"))
        check("用户策略：白名单挡住 import", not ok and "跑不起来" in msg, msg)

        ok, msg = US.register(US.UserStrategy(
            key=key, label="试", func="sig_x",
            code="def sig_x(df):\n    return df"))
        check("用户策略：缺 signal 列被拒", not ok and "signal" in msg, msg)

        ok, msg = US.register(US.UserStrategy(
            key=key, label="试", func="sig_x",
            code="def sig_x(df):\n    out = df.copy()\n    out['signal'] = 2\n    return out"))
        check("用户策略：signal 越界被拒", not ok and "只能" in msg, msg)

        ok, msg = US.register(US.UserStrategy(
            key=key, label="试", func="sig_x",
            code="def sig_x(df):\n    out = df.copy()\n    out['signal'] = 0\n    return out"))
        check("用户策略：全程空仓被拒", not ok and "空仓" in msg, msg)

        ok, msg = US.register(US.UserStrategy(
            key=key, label="试", func="sig_x",
            code=("def sig_x(df):\n    out = df.copy()\n    out['signal'] = 1\n"
                  "    return out.iloc[:-5]")))
        check("用户策略：长度不一致被拒", not ok and "长度" in msg, msg)

        good = US.UserStrategy(
            key=key, label="冒烟策略", desc="20 日突破",
            params={"entry": 240, "exit": 720},
            func="sig_zz",
            code=("def sig_zz(df, entry=240, exit=720):\n"
                  "    out = df.copy()\n"
                  "    hi = out['high'].rolling(entry).max().shift(1)\n"
                  "    lo = out['low'].rolling(exit).min().shift(1)\n"
                  "    out['signal'] = _position_state(\n"
                  "        long_in=out['close'] > hi, long_out=out['close'] < lo,\n"
                  "        short_in=out['close'] < lo, short_out=out['close'] > hi)\n"
                  "    return out"))
        ok, msg = US.register(good)
        check("用户策略：正常策略注册成功", ok, msg)
        check("用户策略：进了 STRATEGIES", key in S.STRATEGIES, key)

        spec = S.STRATEGIES.get(key)
        # 没给 param_meta 时要按默认值推范围，不能套 (1,100) 把 720 夹成 100
        meta = (spec.param_meta or {}) if spec else {}
        check("用户策略：参数范围按默认值推导",
              meta.get("exit", (0, 0, 0))[1] >= 720,
              str(meta))

        df = US.sample_df()
        out = S.run_strategy(df, key, {"entry": 120})
        check("用户策略：能按参数跑出信号",
              len(out) == len(df) and int((out["signal"] != 0).sum()) > 0,
              f"{len(out)} 行")

        doc = L.strategy_doc(key)
        check("用户策略：文档带源码",
              doc is not None and "def sig_zz" in doc.code, "")

        ok2, msg2 = US.register(good)
        check("用户策略：重名被拒", not ok2 and "已经有一个" in msg2, msg2)
        ok3, msg3 = US.register(US.UserStrategy(
            key="turtle", label="撞名", func="sig_y",
            code="def sig_y(df):\n    out = df.copy()\n    out['signal'] = 1\n    return out"))
        check("用户策略：不能覆盖内置策略", not ok3, msg3)
        ok4, msg4 = US.register(US.UserStrategy(
            key="Zz-Bad", label="坏代号", func="sig_z",
            code="def sig_z(df):\n    out = df.copy()\n    out['signal'] = 1\n    return out"))
        check("用户策略：代号格式校验", not ok4 and "代号" in msg4, msg4)

        box: dict = {}
        n = US.register_all_into(box)
        check("用户策略：重启后从盘里挂回来", n == 1 and key in box, f"n={n}")
        check("用户策略：挂回来的函数能跑",
              key in box and "signal" in box[key].func(df).columns, "")

        check("用户策略：删除后 STRATEGIES 里没了",
              US.remove(key) and key not in S.STRATEGIES, "")
        check("用户策略：删完盘里也没了", US.register_all_into({}) == 0, "")
        check("用户策略：内置策略删不掉", not US.remove("turtle"), "")
    except Exception as exc:  # noqa: BLE001
        check("用户自定义策略", False, f"{type(exc).__name__}: {exc}")
    finally:
        US.path = old_path
        S.STRATEGIES.pop(key, None)
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def check_agent_memory() -> None:
    """长期记忆：追加去重、按行摊平、封顶、注入 system prompt。"""
    from ttrade import agent as A
    from ttrade import agent_memory as Mem

    tmpdir = tempfile.mkdtemp(prefix="td_mem_")
    old_path = Mem.path
    Mem.path = lambda: os.path.join(tmpdir, "agent_memory.json")
    try:
        check("记忆：初始为空", Mem.count() == 0 and Mem.brief() == "", "")

        n = Mem.append([Mem.MemoryItem(text="- 螺纹只做多，1 手\n- 止损 2×ATR")])
        check("记忆：追加成功", n == 1 and Mem.count() == 1, f"n={n}")

        n2 = Mem.append([Mem.MemoryItem(text="- 螺纹只做多，1 手\n- 止损 2×ATR")])
        check("记忆：整条重复被去重", n2 == 0 and Mem.count() == 1, f"n={n2}")

        # 另一条记忆里也提到同一句：存储上不算重复，但注入时不能重复出现
        Mem.append([Mem.MemoryItem(text="螺纹只做多，1 手")])
        b = Mem.brief()
        check("记忆：同一句不重复注入", b.count("螺纹只做多") == 1, repr(b))
        check("记忆：列表符号不叠加", "- -" not in b, b)
        check("记忆：一条记忆里的多行摊平成多行",
              b.count("\n") == 1 and b.startswith("- "), repr(b))
        Mem.clear()
        Mem.append([Mem.MemoryItem(text="- 螺纹只做多，1 手\n- 止损 2×ATR")])

        # 封顶：超过 MAX_ITEMS 只留最新的
        Mem.append([Mem.MemoryItem(text=f"第 {i} 条") for i in range(Mem.MAX_ITEMS + 5)])
        check("记忆：条数封顶", Mem.count() == Mem.MAX_ITEMS, str(Mem.count()))
        check("记忆：留下的是最新的",
              "第 64 条" in Mem.brief() and "第 0 条" not in Mem.brief(), "")

        check("记忆：注入到 system 消息",
              [m["role"] for m in A.build_messages([], "问", memory="- 螺纹只做多")]
              == ["system", "system", "user"], "")
        check("记忆：没记忆时不加空 system 消息",
              len(A.build_messages([], "问")) == 2, "")

        msg = A.memory_messages([
            {"role": "user", "content": "帮我注册"},
            {"role": "assistant", "content": "好\n```json\n{\"a\": 1}\n```\n完成"},
        ])[0]["content"]
        check("记忆：压缩请求剔掉代码块",
              "```" not in msg and "已省略" in msg, msg[-40:])

        Mem.clear()
        check("记忆：清空后归零", Mem.count() == 0 and Mem.brief() == "", "")
    except Exception as exc:  # noqa: BLE001
        check("长期记忆", False, f"{type(exc).__name__}: {exc}")
    finally:
        Mem.path = old_path
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
    check_live_worker()
    check_cli_entry()
    check_factor_refresh()
    check_main_backfill()
    check_strategy_store()
    check_strategy_runner()
    check_library_docs()
    check_agent()
    check_agent_stream()
    check_agent_thread()
    check_user_factors()
    check_user_strategies()
    check_agent_memory()
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
