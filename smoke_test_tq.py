"""天勤数据链路的离线自测：切换表、复权因子、多周期合并。

不联网、不依赖天勤账户，只读 `data_tq/` 里已经下好的文件。有数据就全量校验，
没数据就跳过（打包后的环境本来就没有这份数据，不该因此报错）。

    python smoke_test_tq.py            # 校验全部已下载品种
    python smoke_test_tq.py RB_SHFE    # 只校验一个
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
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


def check_intraday_aux() -> None:
    """分钟级的衍生列：展期 / 期权因子在 1h、5m 上必须有数值。

    这些列只存在于 `data_futures_aux/`（由真实合约聚合而来），日线分支一直在
    加载时挂上去，分钟分支漏了 —— 结果是「下了全品种 5m/1h 真实合约，展期因子
    却整列空着」。这里同时盯住两件事：列挂上了，且用的是**上一交易日**的快照
    （当天 15:00 收盘才算得出来的价差不能灌进上午的 bar）。
    """
    import os

    from ttrade import auxdata as A
    from ttrade import factors as F
    from ttrade import tqdata as T

    try:
        check("周期：15m/30m 归到 5m",
              [T.source_period(p) for p in ("15m", "30m")] == ["5m", "5m"])
        check("周期：2h/4h 归到 1h",
              [T.source_period(p) for p in ("2h", "4h")] == ["1h", "1h"])
        check("周期：5m/1h/1d 原样",
              [T.source_period(p) for p in ("5m", "1h", "1d")] == ["5m", "1h", "1d"])
        check("周期：天勤日线 1d_tq 落到 1d",
              T.source_period("1d_tq") == "1d")
        check("周期：日线判定同时认 1d 和 1d_tq",
              T.is_daily("1d") and T.is_daily("1d_tq") and not T.is_daily("1h"))
        check("周期：天勤日线的展示名是「日线」",
              T.PERIOD_LABEL.get("1d_tq") == "日线")

        aux_dir = A._default_dir()
        key = ""
        for k in keys():
            sym, ex = k.rsplit("_", 1)
            if os.path.exists(A.curve_path(aux_dir, sym, ex)) and T.has_period(sym, ex, "1h"):
                key = k
                break
        if not key:
            skip("分钟级衍生列", "没有同时具备 1h 行情和衍生数据的品种")
            return
        sym, ex = key.rsplit("_", 1)

        df = T.load_main(sym, ex, "1h", adjust="qfq").tail(2000)
        out = A.attach_intraday(df, sym, ex)
        check(f"{key} 1h：挂上了期限结构列", "c1_close" in out.columns and "c2_close" in out.columns)
        check(f"{key} 1h：期限结构没有空洞",
              float(out["c1_close"].notna().mean()) > 0.99,
              f"{out['c1_close'].notna().mean():.1%}")
        check(f"{key} 1h：行数不变", len(out) == len(df), f"{len(out)} vs {len(df)}")

        # 第一根 bar 用的是「它所在交易日之前」最近一个快照。
        # 交易日取 `trading_day` 列而不是 index.normalize()：数据第一根就是夜盘
        # 21:00 的话，自然日是前一天，会去比错一天
        curve = A.load_curve(aux_dir, sym, ex)
        day = (pd.Timestamp(out["trading_day"].iloc[0]) if "trading_day" in out.columns
               else pd.Timestamp(out.index[0]).normalize())
        prev = curve.loc[curve.index < day]
        if len(prev):
            check(f"{key} 1h：用上一交易日的快照（不提前用当天收盘）",
                  abs(float(out["c1_close"].iloc[0]) - float(prev["c1_close"].iloc[-1])) < 1e-9,
                  f"{out['c1_close'].iloc[0]} vs {prev['c1_close'].iloc[-1]}")

        ry = F.f_roll_yield(out)
        check(f"{key} 1h：展期收益率算得出", float(ry.notna().mean()) > 0.99,
              f"{ry.notna().mean():.1%}")
        # 夜盘 bar 的时间戳是前一天 21:00，但属于**下一个**交易日，必须和那天
        # 的日盘读到同一份快照。按 index.normalize() 对齐的话夜盘会取到 D-2，
        # 同一个交易日里冒出两个值 —— 日内因子和 ML 特征里凭空多一跳
        if "trading_day" in out.columns:
            td = pd.to_datetime(out["trading_day"])
            hh = out.index.strftime("%H:%M")
            night = out[hh >= "21:00"]
            if len(night):
                d0 = td.loc[night.index[0]]
                same = out.loc[td == d0, "c1_close"]
                check(f"{key} 夜盘与次日日盘用同一份快照",
                      same.nunique() == 1,
                      f"交易日 {str(d0)[:10]} 有 {same.nunique()} 个不同值")

        # 衍生列一天一个快照，同一天里所有 bar 的「20 日变化」必须一模一样。
        # 直接 `.diff(20)` 的话，5m 上算的是 100 分钟前的差，一天内根根不同
        ch = F.f_roll_change(out).dropna()
        grp = ch.groupby(pd.to_datetime(out["trading_day"]) if "trading_day" in out.columns
                         else ch.index.normalize())
        sizes = grp.size()
        full = [d for d in sizes.index if sizes[d] >= 3]
        same_day = grp.get_group(full[-1]) if full else ch.iloc[:0]
        check(f"{key} 1h：展期变化按交易日算（同一天内取值一致）",
              len(same_day) >= 3 and same_day.nunique() == 1,
              f"{len(same_day)} 根，{same_day.nunique()} 个不同值")

        # 日线分支不能被改坏：还是按同一天 join
        plain = pd.DataFrame({"close": [1.0] * 5}, index=curve.index[:5])
        d1 = A.attach(plain, sym, ex, aux_dir)
        check(f"{key} 日线：attach 按日期并入且不改行数",
              len(d1) == 5 and "c1_close" in d1.columns
              and float(d1["c1_close"].iloc[0]) == float(curve["c1_close"].iloc[0]))
    except Exception as exc:  # noqa: BLE001
        check("分钟级衍生列", False, f"{type(exc).__name__}: {exc}")


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

    # 列不能被砍。本地表带着展期 / 期权那几十列（c1_close / iv_near / pcr …），
    # 实时推送里没有这些。早先按**交集**取列，合并结果里这几列整列消失，
    # `factors.available()` 一看「一根有值的都没有」就把对应因子踢出合成 ——
    # 表现是「一开实时，19 个因子变 12 个、反转数也跟着变，策略像自己换了一套」。
    aux = local.copy()
    aux["pcr"] = [0.85]
    aux["c1_close"] = [101000.0]
    merged2 = L.merge_live(aux, bars)
    check("实时⑤合并后列一个不少",
          list(merged2.columns) == list(aux.columns),
          f"少了：{[c for c in aux.columns if c not in merged2.columns]}")
    check("实时⑤新 bar 缺的列留空而不是 0",
          pd.isna(merged2.loc["2026-10-27 18:00", "pcr"])
          and pd.isna(merged2.loc["2026-10-27 18:00", "c1_close"]),
          str(merged2.loc["2026-10-27 18:00", ["pcr", "c1_close"]].to_dict()))
    check("实时⑤历史段辅助列还在",
          float(merged2.loc["2026-10-27 17:00", "pcr"]) == 0.85,
          f"{merged2.loc['2026-10-27 17:00', 'pcr']}")

    # 同一根：本地库里已经有这根（日线最典型 —— 下载时把当天那根半成品写进去了），
    # 实时这份要**覆盖价格**，否则图上最后一根永远停在下载那一刻：顶部报价在跳、
    # K 线一动不动。因子仍用本地那份（实时段一律 1.0）。
    tick = bars.copy()
    tick.loc["2026-10-27 17:00", ["close", "high"]] = [105.0, 106.0]
    same = L.merge_live(local, tick[tick.index == local.index[-1]])
    check("实时④同一根用实时价格覆盖",
          len(same) == 1 and float(same["close"].iloc[-1]) == 105.0
          and float(same["high"].iloc[-1]) == 106.0,
          str(same[["close", "high"]].iloc[-1].to_dict()))
    check("实时④覆盖后因子仍是本地的",
          float(same["adj_factor"].iloc[-1]) == 0.97,
          f"{same['adj_factor'].iloc[-1]}")
    check("实时④不改动传进来的本地表",
          float(local["close"].iloc[0]) == 102.0,
          f"{local['close'].iloc[0]}")

    empty = L.merge_live(pd.DataFrame(), bars)
    check("实时③没有本地历史时整段当实时",
          len(empty) == 2 and bool((empty["adj_factor"] == 1.0).all()))
    # 15m/30m/2h/4h 的实时订阅：订**源周期**（5m/1h）再本地合并，和本地历史
    # 同一口径。直接订原生 15m 会和 5m×3 合并的历史错位（:00/:15 vs :10/:25），
    # 拼起来就是碎的 —— 和 1h 那个坑同源。
    class _SubApi:
        def __init__(self):
            self.secs = []

        def get_quote(self, _sym):
            return None

        def get_kline_serial(self, _sym, sec, data_length=0):
            self.secs.append(sec)
            return None

    for per, sec, nn in (("5m", 300, 1), ("15m", 300, 3), ("30m", 300, 6),
                         ("1h", 3600, 1), ("2h", 3600, 2), ("4h", 3600, 4)):
        h = L.LiveHub(L.AccountSpec())
        h.api = _SubApi()
        h.subscribe("AG", "SHFE", per)
        check(f"实时⑧{per} 订源周期 {sec}s", h.api.secs == [sec], str(h.api.secs))
        check(f"实时⑧{per} 合并根数 {nn}", h._n == nn, str(h._n))

    # 合成规则：3 根 5m → 1 根 15m（open 取首、close 取尾、量累加）
    from ttrade import tqdata as TQ
    five = pd.DataFrame({
        "open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5],
        "low": [0.5, 1.5, 2.5], "close": [1.2, 2.2, 3.2],
        "volume": [1.0, 2.0, 3.0], "open_oi": [5.0, 6.0, 7.0],
        "close_oi": [5.0, 6.0, 7.0],
    }, index=pd.DatetimeIndex(
        ["2026-01-05 09:05", "2026-01-05 09:10", "2026-01-05 09:15"],
        name="datetime"))
    q = TQ.resample(five, 3)
    check("实时⑨resample 3 根合 1 根",
          len(q) == 1 and float(q["open"].iloc[0]) == 1.0
          and float(q["close"].iloc[0]) == 3.2 and float(q["volume"].iloc[0]) == 6.0,
          f"{len(q)} 根")


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
    #
    # 多账户之后快照一律走**账户对象**（`obj.get_position()`）：多账户模式下
    # `api.get_position()` 必须带 account 参数，而账户对象自己那套在单账户、多账户
    # 两种模式下都能用 —— 所以测试里只造账户对象，不碰 api。
    from types import SimpleNamespace

    def _pos(**kw):
        base = dict(pos=-1, pos_long=0, pos_short=1, pos_long_today=0, pos_short_today=1,
                    open_price_long=float("nan"), open_price_short=109360.0,
                    float_profit=950.0, position_profit=0.0, margin=59752.0,
                    last_price=108860.0)
        base.update(kw)
        return SimpleNamespace(**base)

    class _Acct:
        """假账户对象：只实现四个 get_*，够快照用。"""

        def __init__(self, pos=None, orders=None, trades=None, acc=None):
            self.pos = pos or {}
            self.orders = orders or {}
            self.trades = trades or {}
            self.acc = acc or {}

        def get_position(self):
            return self.pos

        def get_order(self):
            return self.orders

        def get_trade(self):
            return self.trades

        def get_account(self):
            return SimpleNamespace(**self.acc) if self.acc else None

    obj_a = _Acct({"SHFE.cu2612": _pos(instrument_id="cu2612")})
    hub._accs = [(L.AccountSpec(name="模拟A"), obj_a)]
    rows = hub._positions_snapshot(obj_a)
    check("实时⑦持仓合约名取 instrument_id", rows and rows[0]["symbol"] == "cu2612",
          repr(rows[0]["symbol"]) if rows else "空")
    obj_a.pos = {"SHFE.cu2612": _pos()}
    rows = hub._positions_snapshot(obj_a)
    check("实时⑦没有 instrument_id 时回退到 dict 的 key",
          rows and rows[0]["symbol"] == "SHFE.cu2612",
          repr(rows[0]["symbol"]) if rows else "空")
    check("实时⑦空头持仓的 open_price_long 是 NaN（界面要按方向取）",
          rows and rows[0]["open_price_long"] != rows[0]["open_price_long"]
          and rows[0]["open_price_short"] == 109360.0,
          f"{rows[0]['open_price_long']}")
    check("实时⑦账户对象为空时不炸", hub._positions_snapshot(None) == [])

    # 委托 / 成交快照只留最近 N 条。一天被拒的废单能有几千条，全塞进
    # QTableWidget 之后，切到「账户透视和交易」第一帧就要摆几万个单元格，
    # 界面直接卡死（用户报的「一切到这个页就卡住」）
    def _ord(i, status="FINISHED"):
        return SimpleNamespace(
            instrument_id="ag2612", direction="SELL", offset="CLOSE",
            volume_orign=2, volume_left=0, limit_price=13112.0, status=status,
            last_msg="平昨手数超过持仓仓量", insert_date_time=1e18 - i,
        )

    obj_a.orders = {f"o{i}": _ord(i) for i in range(L.SNAP_ROWS * 3)}
    rows = hub._orders_snapshot(obj_a)
    check("实时⑧委托快照只留最近 N 条", len(rows) == L.SNAP_ROWS, f"{len(rows)} 条")
    check("实时⑧委托按时间倒序",
          rows[0]["insert_date_time"] > rows[-1]["insert_date_time"])
    check("实时⑧委托带北京时间列", len(rows[0].get("time", "")) == 14,
          repr(rows[0].get("time")))

    obj_a.orders = {**{f"o{i}": _ord(i) for i in range(L.SNAP_ROWS * 3)},
                    "alive": _ord(0, status="ALIVE")}
    rows = hub._orders_snapshot(obj_a)
    check("实时⑧未成交委托不会被截掉", rows[0]["status"] == "ALIVE", rows[0]["status"])

    def _trd(i):
        return SimpleNamespace(
            order_id=f"o{i}", instrument_id="ag2612", direction="SELL",
            offset="CLOSE", price=13112.0, volume=2, trade_date_time=1e18 - i,
        )

    obj_a.trades = {f"t{i}": _trd(i) for i in range(L.SNAP_ROWS * 3)}
    rows = hub._trades_snapshot(obj_a)
    check("实时⑧成交快照只留最近 N 条", len(rows) == L.SNAP_ROWS, f"{len(rows)} 条")
    check("实时⑧成交带北京时间列", len(rows[0].get("time", "")) == 14,
          repr(rows[0].get("time")))

    # 多账户：一条连接、每账户一份明细，合并那份给首屏资金卡用。
    obj_a.acc = {"balance": 600000.0, "pre_balance": 550000.0, "margin": 120000.0,
                 "available": 480000.0, "commission": 30.0}
    obj_a.pos = {"SHFE.cu2612": _pos()}
    obj_a.orders = {}
    obj_a.trades = {}
    obj_b = _Acct(acc={"balance": 100000.0, "pre_balance": 90000.0, "margin": 20000.0,
                       "available": 80000.0, "commission": 12.5})
    hub._accs = [(L.AccountSpec(name="模拟A"), obj_a),
                 (L.AccountSpec(name="模拟B"), obj_b)]
    st = hub.state()
    check("实时⑩账户列表一行一个账户", len(st.accounts) == 2,
          f"{[r['name'] for r in st.accounts]}")
    check("实时⑩合并权益是求和",
          abs(st.account.get("balance", 0) - 700000.0) < 1e-6,
          f"{st.account.get('balance')}")
    check("实时⑩合并风险度用合计保证金/合计权益重算",
          abs(st.account.get("risk_ratio", 0) - 140000.0 / 700000.0) < 1e-9,
          f"{st.account.get('risk_ratio')}")
    check("实时⑩当日盈亏逐账户相加",
          abs(st.account.get("day_profit", 0) - 60000.0) < 1e-6,
          f"{st.account.get('day_profit')}")
    check("实时⑩持仓行带账户名",
          st.positions and st.positions[0].get("account") == "模拟A",
          repr(st.positions[0].get("account")) if st.positions else "空")
    check("实时⑩明细按账户分开存", set(st.by_account) == {"模拟A", "模拟B"},
          str(list(st.by_account)))

    try:
        hub._acc_obj("")
        check("实时⑩多账户不指定账户就报错", False, "没有抛异常")
    except RuntimeError:
        check("实时⑩多账户不指定账户就报错", True)
    check("实时⑩按名字取到对应账户", hub._acc_obj("模拟B") is obj_b)
    try:
        hub._acc_obj("不存在的账户")
        check("实时⑩账户名写错报错", False, "没有抛异常")
    except ValueError:
        check("实时⑩账户名写错报错", True)

    # 纳秒时间戳 → 北京时间。0 / 非法值给空串，界面才不会显示 1970 年。
    check("实时⑨时间戳 0 给空串", L._stamp(0) == "" and L._stamp("x") == "",
          repr(L._stamp(0)))
    check("实时⑨时间戳转北京时间",
          L._stamp(1772000000 * 10 ** 9) == "02-25 14:13:20",
          repr(L._stamp(1772000000 * 10 ** 9)))
    check("实时⑨合约乘数按品种查得到", L._mult_of("SHFE.rb2610") > 1,
          str(L._mult_of("SHFE.rb2610")))

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

    def wait_update(self, deadline=None) -> None:
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

        # ---- 取消 ----
        # 界面上点「取消」后不能等这一批 24 个合约全飞完才停。批次之间检查一次，
        # 停的时候要明确回报 cancelled，否则界面会把它算成「这个品种下好了」。
        seen.clear()

        def ok_dl(api, jobs, period, out_dir, tag, max_parallel=8, errors=None):
            seen.append([s for s, _, _ in jobs])
            return {s: _fake_frame(today - pd.Timedelta(hours=1)) for s, _, _ in jobs}

        D.download_many = ok_dl
        rc = D.write_contracts(_FakeApi(["SHFE.ni2609", "SHFE.ni2612"]),
                               "NI", "SHFE", "5m", 1, cancel=lambda: True)
        check("取消：一批都没起就返回",
              rc.get("cancelled") is True and not seen and rc["contracts"] == 0,
              f"cancelled={rc.get('cancelled')} 请求={seen}")
        check("取消：状态写成「已取消」，别冒充 ok",
              rc["status"] == "已取消", rc["status"])

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
                def wait_update(self, deadline=None) -> None:
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

            # 卡死保护：连接看着还在（wait_update 正常返回）、但任务永远不结束。
            # 没有看门狗的话这里会一直转下去 —— 实盘补数据时就是这么僵了 20 分钟。
            class _StuckTask(_Task):
                class _T:
                    @staticmethod
                    def cancel() -> None:
                        pass

                _task = _T()

                def is_finished(self) -> bool:
                    return False

            class _StuckApi:
                def query_quotes(self, **_kw):
                    return []

                def wait_update(self, deadline=None) -> None:
                    pass

            fake_tools.DataDownloader = _StuckTask
            old_stall = D._STALL_SEC
            D._STALL_SEC = 0.05
            try:
                errs: dict = {}
                _t0 = time.time()
                got = D.download_many(_StuckApi(), jobs, "5m", tmpdir, "T",
                                      max_parallel=2, errors=errs)
                _cost = time.time() - _t0
            finally:
                D._STALL_SEC = old_stall
            check("卡死保护：卡住后能自己退出", _cost < 20, f"{_cost:.2f}s")
            check("卡死保护：卡住的合约记进 errors，不当成没数据",
                  set(errs) == {"SHFE.rb2601", "SHFE.rb2605"}
                  and all("超时" in v for v in errs.values()), f"{errs}")
            check("卡死保护：不返回半截数据", got == {}, f"{list(got)}")
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

        # 重名限制：库里出现三条「白银动量60-1h」之后，名字就没法指代任何一条了
        named = a.clone(name="白银动量60-1h")
        St.add_to_library(named)
        dup = a.clone(name="白银动量60-1h")
        try:
            St.add_to_library(dup)
            check("策略库：重名被拒绝", False, "居然存进去了")
        except ValueError as exc:
            check("策略库：重名被拒绝", "已经有" in str(exc), str(exc))
        check("策略库：重名之后库里还是两条",
              len([c for c in St.load_library()
                   if St.norm_name(c.title()) == St.norm_name("白银动量60-1h")]) == 1,
              str([c.title() for c in St.load_library()]))
        # 没起名的配置不该被挡：同品种同策略、参数不同是正常用法，自动加序号
        u1 = a.clone()
        u2 = a.clone(rule_params={"fast": 9})
        St.add_to_library(u1)
        St.add_to_library(u2)
        check("策略库：没起名时自动加序号",
              u1.name and u2.name and u1.name != u2.name,
              f"{u1.name} / {u2.name}")
        # 成本档位要能存能读 —— 同一份策略两套成本结论不同，得能复现
        cp = a.clone(name="成本口径")
        cp.cost_profile = "stress"
        St.add_to_library(cp)
        back = next(c for c in St.load_library() if c.sid == cp.sid)
        check("策略配置：成本档位往返一致", back.cost_profile == "stress",
              back.cost_profile)

        # 截面字段要能存能读：品种池丢了就等于把策略变成了单品种
        cx = St.StrategyConfig(
            name="截面测试", symbol="CU", exchange="SHFE", period="5m",
            family="cross", cross_symbols=["CU_SHFE", "RB_SHFE", "M_DCE"],
            cross_mode="select", cross_rebalance=48, cross_top_pct=25.0,
            cross_bot_pct=20.0, cross_weight="score", factor_mode="ml",
            factor_keys=["mom_20"],
            timing_family="factor", timing_rule_key="ma_cross",
            timing_factor_keys=["mom_60"], cost_profile="neutral",
        )
        St.add_to_library(cx)
        cb = next(c for c in St.load_library() if c.sid == cx.sid)
        check("策略配置：截面字段往返一致",
              (cb.cross_symbols, cb.cross_mode, cb.cross_rebalance,
               round(cb.cross_top_pct, 1), round(cb.cross_bot_pct, 1),
               cb.cross_weight, cb.factor_mode,
               cb.timing_family, cb.timing_factor_keys)
              == (["CU_SHFE", "RB_SHFE", "M_DCE"], "select", 48, 25.0, 20.0,
                  "score", "ml", "factor", ["mom_60"]),
              str((cb.cross_symbols, cb.cross_mode, cb.cross_rebalance,
                   cb.cross_top_pct, cb.cross_bot_pct, cb.cross_weight,
                   cb.factor_mode, cb.timing_family)))
        check("策略配置：截面的标题不写单个品种",
              "CU_SHFE" not in cb.title() and "3 个品种" in cb.strategy_label(),
              cb.title())
        check("策略配置：截面参数摘要含调仓与多空比例",
              "调仓" in cb.params_text() and "多25%/空20%" in cb.params_text(),
              cb.params_text())
        # 多空两条腿的百分比各存各的：多头 25% / 空头 20% 是一种常见配置
        check("策略配置：截面多空百分比分开记",
              "多25%/空20%" in cb.params_text(), cb.params_text())
        # 资金分配默认风险平价：等权会让波动最大的那个品种主导整条净值曲线
        check("策略配置：截面资金分配默认风险平价",
              St.StrategyConfig(family="cross").cross_weight == "risk",
              St.StrategyConfig(family="cross").cross_weight)
        wt = a.clone(name="权重口径", family="cross", cross_weight="score")
        St.add_to_library(wt)
        back_w = next(c for c in St.load_library() if c.sid == wt.sid)
        check("策略配置：截面权重口径往返一致",
              back_w.cross_weight == "score" and "按分数" in back_w.params_text(),
              f"{back_w.cross_weight} / {back_w.params_text()}")

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


def check_costs() -> None:
    """成本档位与品种信息。

    盯三件事：期货成本是「交易所手续费 + N 跳滑点」而不是一个 bp；三档的滑点
    确实单调（压力 ≥ 中性 ≥ 乐观）；按手收的手续费真的从权益里扣掉了。
    """
    from ttrade import backtest as B
    from ttrade import costs, symbols_info

    check("品种信息：tick 表命中",
          symbols_info.TICKS.get("RB") == 1.0
          and symbols_info.TICKS.get("CU") == 10.0
          and symbols_info.TICKS.get("LC") == 50.0, "tick 表不对")
    check("品种信息：手续费口径按交易所",
          symbols_info.defaults("TA", "CZCE")["fee_mode"] == "lot"
          and symbols_info.defaults("RB", "SHFE")["fee_mode"] == "rate",
          "手续费口径不对")
    check("品种信息：交易时段按品种区分",
          "09:30" in symbols_info.session_of("IF", "CFFEX")
          and "21:00" in symbols_info.session_of("RB", "SHFE"),
          symbols_info.session_of("IF", "CFFEX"))
    # 反推 tick：价格都落在 0.5 的整数倍上，最大公约数就该是 0.5
    check("品种信息：从价格反推 tick",
          symbols_info.infer_tick([100.0, 100.5, 101.0, 101.5, 102.0]) == 0.5,
          str(symbols_info.infer_tick([100.0, 100.5, 101.0, 101.5, 102.0])))

    p = {k: costs.profile(k) for k in ("optimistic", "neutral", "stress")}
    check("成本档位：滑点单调递增",
          p["optimistic"]["slip_ticks"] < p["neutral"]["slip_ticks"]
          < p["stress"]["slip_ticks"],
          str({k: v["slip_ticks"] for k, v in p.items()}))

    bt = B.BacktestConfig()
    costs.apply(bt, "RB", "SHFE", "optimistic")
    check("成本档位：期货按 tick 算滑点",
          bt.tick_size == 1.0 and bt.slippage_ticks == 1.0 and bt.fee_rate > 0,
          f"tick={bt.tick_size} ticks={bt.slippage_ticks} fee={bt.fee_rate}")
    check("成本档位：一手滑点 = 跳数 × tick",
          abs(bt.slip_amount(3000.0) - 1.0) < 1e-9, str(bt.slip_amount(3000.0)))

    us = B.BacktestConfig()
    costs.apply(us, "AAPL", "", None, market="us")
    check("成本档位：美股口径不变",
          us.fee_rate == costs.US_FEE and us.slippage == costs.US_SLIP
          and us.fee_per_lot == 0.0, f"{us.fee_rate}/{us.slippage}")

    # 按手收的手续费要真扣：同一段行情，每手手续费调大，期末权益必须变小
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    px = pd.Series(np.linspace(100, 120, 40), index=idx)
    df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                       "close": px, "signal": [1] * 40})

    def _eq(fee_lot: float) -> float:
        c = B.BacktestConfig(initial_cash=1_000_000, sizing="fixed", volume=1,
                             multiplier=10.0, fee_per_lot=fee_lot, fee_rate=0.0,
                             slippage=0.0)
        return float(B.run_backtest(df, c).metrics["final_equity"])

    check("回测：按手手续费真的扣钱", _eq(0) > _eq(2000),
          f"{_eq(0):.0f} vs {_eq(2000):.0f}")
    check("成本说明：写清这一档收多少", "跳" in costs.describe("CU", "SHFE"),
          costs.describe("CU", "SHFE"))


def check_cross() -> None:
    """截面策略：横截面 Z-score + 多强空弱。

    造一批「强弱恒定」的品种：A 每天涨 0.2%、F 每天跌 0.2%。动量因子在截面上
    永远把 A 排第一、F 排最后，所以「多 A 空 F」必须赚钱 —— 这条链上任何一环
    （因子对齐、横截面标准化、排序、权重、成本）错了，收益就会变成 0 或负数。
    """
    from ttrade import cross
    from ttrade.backtest import BacktestConfig as BtConfig

    idx = pd.date_range("2024-01-01", periods=240, freq="B")
    drifts = [0.002, 0.0012, 0.0004, -0.0004, -0.0012, -0.002]
    frames = {}
    for i, d in enumerate(drifts):
        px = pd.Series(100.0 * (1 + d) ** np.arange(len(idx)), index=idx)
        frames[f"S{i}_X"] = pd.DataFrame({
            "open": px, "high": px * 1.002, "low": px * 0.998,
            "close": px, "volume": 1000.0,
        })

    cfg = cross.CrossConfig(mode="hedge", rebalance=5, top_pct=10,
                            factor_mode="multi", factor_keys=["mom_20"])
    check("截面：一边持几个（10% × 6 个 ≈ 1 个）", cfg.n_legs(6) == 1, str(cfg.n_legs(6)))
    check("截面：品种不够时至少 1 个", cross.CrossConfig(top_pct=50).n_legs(3) == 1,
          str(cross.CrossConfig(top_pct=50).n_legs(3)))

    # 多空两条腿各算各的：多头 10% → 1 个，空头 33% → 2 个。共用一个百分比的话
    # 「多最强 1 个、空最弱 3 个」这种不对称配置就表达不出来
    split = cross.CrossConfig(mode="hedge", top_pct=10, bot_pct=33.3)
    check("截面：多空腿数分开算",
          (split.n_legs(6), split.n_short_legs(6)) == (1, 2),
          f"{split.n_legs(6)} 多 / {split.n_short_legs(6)} 空")
    asym = cross.run(frames, cross.CrossConfig(
        mode="hedge", rebalance=5, top_pct=16.7, bot_pct=33.3,
        factor_keys=["mom_20"]), BtConfig())
    check("截面：不对称多空仍然净敞口为 0",
          asym.ok and abs(float(asym.weights.iloc[-1].sum())) < 1e-9,
          "" if not asym.ok else str(float(asym.weights.iloc[-1].sum())))
    # 最后一根可能刚好是空仓（分数全空），取最近一根有仓位的看腿数
    held = (asym.weights[asym.weights.abs().sum(axis=1) > 0]
            if asym.ok else pd.DataFrame())
    n_l = int(held.iloc[-1].gt(0).sum()) if len(held) else 0
    n_s = int(held.iloc[-1].lt(0).sum()) if len(held) else 0
    check("截面：空头腿确实比多头多", (n_l, n_s) == (1, 2), f"{n_l} 多 / {n_s} 空")

    score = cross.score_panel(frames, ["mom_20"])
    last = score.iloc[-1].dropna()
    check("截面：分数把最强的排第一", last.idxmax() == "S0_X", str(last.round(2).to_dict()))
    check("截面：分数把最弱的排最后", last.idxmin() == "S5_X", str(last.round(2).to_dict()))
    check("截面：横截面 Z-score 均值约 0", abs(float(last.mean())) < 1e-9,
          str(float(last.mean())))

    # 三种因子模式都要能出分数 —— 单因子 / 多因子 / 机器学习，界面上是同一个
    # 下拉，后端走的是三条不同的路
    for m in ("single", "multi"):
        sc = cross.score_panel(frames, ["mom_20", "mom_60"], mode=m)
        check(f"截面：{m} 模式出分数", not sc.empty and sc.notna().any().any(),
              str(sc.shape))
    # 训练窗口要按样本量缩：默认 500 根起训，这 240 根日线直接判「样本太短」
    from ttrade import factor_strategy as fstrat

    small = fstrat.FactorConfig(min_train=60, train_window=60, step=40, horizon=1)
    ml = cross.score_panel(frames, ["mom_20", "mom_60", "ma_gap_20"], mode="ml",
                           fcfg=small)
    check("截面：机器学习模式出分数", not ml.empty and ml.notna().any().any(),
          str(ml.shape))
    try:
        cross.score_panel(frames, ["mom_20"], mode="ml", fcfg=small)
        one_factor_ml = "居然跑通了"
    except ValueError as exc:
        one_factor_ml = "" if "至少" in str(exc) else str(exc)
    check("截面：机器学习至少要 2 个因子", not one_factor_ml, one_factor_ml)

    # 全空因子不能把合成分数毒掉：NaN + x 还是 NaN，十几个因子里混进一个
    # 展期/期权因子（日线库没有这些列），整张分数面板就会变成空的
    one = cross.score_panel(frames, ["mom_20"], mode="multi")
    both = cross.score_panel(frames, ["roll_curv", "mom_20"], mode="multi")
    check("截面：全空因子不毒掉合成",
          both.shape == one.shape
          and np.allclose(both.fillna(-9).to_numpy(), one.fillna(-9).to_numpy()),
          f"{both.shape} vs {one.shape}")

    # 风险平价：波动率越大分到的钱越少，每条腿的风险贡献才拉得平。造一批
    # 「漂移相同、波动率 0.2%→2% 逐级放大」的品种，低波动的必须拿到大头
    vframes = {}
    for i, vol in enumerate([0.002, 0.004, 0.008, 0.012, 0.016, 0.02]):
        r = 0.001 + vol * np.random.default_rng(11 + i).normal(size=len(idx))
        px = pd.Series(100 * np.cumprod(1 + r), index=idx)
        vframes[f"V{i}_X"] = pd.DataFrame({"open": px, "high": px, "low": px,
                                           "close": px, "volume": 1000.0})
    rp_cfg = cross.CrossConfig(mode="hedge", rebalance=5, top_pct=33.3, bot_pct=33.3,
                               weight="risk", factor_keys=["mom_20"])
    check("截面：风险平价是默认权重", cross.CrossConfig().weight == "risk",
          cross.CrossConfig().weight)
    rp = cross.run(vframes, rp_cfg, BtConfig())
    check("截面：风险平价能跑出净值", rp.ok, rp.error)
    if rp.ok:
        held = rp.weights[rp.weights.abs().sum(axis=1) > 0]
        row = held.iloc[-1]
        longs = row[row > 0]
        vols = {k: float(df["close"].pct_change().rolling(60, min_periods=20)
                         .std().iloc[-1]) for k, df in vframes.items()}
        # 多头腿里不一定有 V0（动量排序看的是涨得最猛的那个），所以按「这一腿里
        # 谁波动最小」比，而不是按名字比
        calm = min(longs.index, key=lambda k: vols[k])
        check("截面：风险平价给低波动更多钱",
              longs.idxmax() == calm, f"最稳的 {calm} / 拿最多 {longs.idxmax()}")
        check("截面：风险平价多头合计 0.5",
              abs(float(longs.sum()) - 0.5) < 1e-9, str(float(longs.sum())))
        check("截面：风险平价空头合计 -0.5",
              abs(float(row[row < 0].sum()) + 0.5) < 1e-9,
              str(float(row[row < 0].sum())))
        check("截面：风险平价净敞口为 0", abs(float(row.sum())) < 1e-9,
              str(float(row.sum())))
        # 风险贡献 = 权重 × 波动率，同一条腿里应该差不多（差一个数量级就是分错了）
        risk = [abs(v) * vols[k] for k, v in longs.items()]
        check("截面：风险平价的风险贡献接近相等",
              max(risk) / max(min(risk), 1e-12) < 2.0,
              str([round(x, 6) for x in risk]))
    eqw = cross.run(vframes, cross.CrossConfig(
        mode="hedge", rebalance=5, top_pct=33.3, bot_pct=33.3,
        weight="equal", factor_keys=["mom_20"]), BtConfig())
    check("截面：等权与风险平价的净值不同",
          eqw.ok and not np.allclose(eqw.equity.to_numpy()[-200:],
                                     rp.equity.to_numpy()[-200:]),
          "" if not (eqw.ok and rp.ok) else "居然一模一样")

    res = cross.run(frames, cfg, BtConfig())
    check("截面：回测能跑出净值", res.ok, res.error)
    # 净值从「第一个算得出截面的 bar」起步：动量 20 要 20 根预热，前面那段没有
    # 分数就没有仓位，不该凭空多出 20 个 0 收益的点
    check("截面：净值长度对得上分数面板", len(res.equity) == len(score),
          f"{len(res.equity)} vs {len(score)}")
    check("截面：多强空弱是赚钱的", res.metrics["total_return"] > 0.1,
          f"{res.metrics['total_return']:.3f}")
    check("截面：净敞口接近 0",
          abs(float(res.weights.iloc[-1].sum())) < 1e-9,
          str(float(res.weights.iloc[-1].sum())))

    # 只做多时不该有空头腿
    lon = cross.run(frames, cross.CrossConfig(
        mode="hedge", rebalance=5, top_pct=10, long_only=True,
        factor_keys=["mom_20"]), BtConfig())
    check("截面：只做多没有负权重", lon.ok and float(lon.weights.min().min()) >= 0,
          "" if not lon.ok else str(float(lon.weights.min().min())))

    # 品种不够要报错，不能算出一条假的曲线
    bad = cross.run({"S0_X": frames["S0_X"], "S1_X": frames["S1_X"]}, cfg, BtConfig())
    check("截面：品种不足直接报错", not bad.ok and "至少要" in bad.error, bad.error)

    # 选品种 + 时序：先用截面挑最强，再让双均线决定多空
    sel = cross.run(frames, cross.CrossConfig(
        mode="select", rebalance=5, top_pct=50, factor_keys=["mom_20"],
        timing_family="rule", timing_rule_key="ma_cross",
        timing_rule_params={"fast": 5, "slow": 20}), BtConfig())
    check("截面选品种+时序：能跑出净值", sel.ok, sel.error)
    check("截面选品种+时序：只持有选中的品种",
          float(sel.weights.abs().sum(axis=1).max()) <= 1.0 + 1e-9,
          str(float(sel.weights.abs().sum(axis=1).max())))

    # 包成 `Result` 之后绩效页要能直接吃：缺一个键就是 KeyError，而界面上
    # 表现成「点了截面策略整页崩掉」
    from ttrade.backtest import _METRIC_KEYS

    r = cross.to_result(res, label="截面对冲", step=cfg.step(), bpy=252.0)
    miss = [k for k in _METRIC_KEYS if k not in r.metrics]
    check("截面：转 Result 后指标一个不缺", not miss, "缺：" + "、".join(miss))
    check("截面：转 Result 后带期末资金",
          "final_equity" in r.metrics and r.metrics["final_equity"] > 0,
          str(r.metrics.get("final_equity")))
    check("截面：调仓次数按区间数算", r.metrics["n_trades"] == len(range(0, len(score), 5)),
          f"{r.metrics['n_trades']} vs {len(range(0, len(score), 5))}")

    # 交易记录：绩效页说「调仓 406 次」而交易记录一片空白，是因为 `to_result`
    # 把 trades 写死成了 []。截面没有一手一手的成交，只能按「某品种一段连续
    # 同向持仓」切 —— 权重变大变小不算新的一笔
    check("截面：交易记录不是空的", len(res.trades) > 0, str(len(res.trades)))
    check("截面：转 Result 后交易记录透传", len(r.trades) == len(res.trades),
          f"{len(r.trades)} vs {len(res.trades)}")
    check("截面：每笔交易都带品种名", all(t.symbol for t in res.trades),
          str([t.symbol for t in res.trades[:3]]))
    check("截面：每笔交易方向非 0",
          all(t.direction in (-1, 1) for t in res.trades),
          str(sorted({t.direction for t in res.trades})))
    # 强弱恒定，S0 从头到尾都是多头：一笔，不该被每次调仓切成几十笔
    s0 = [t for t in res.trades if t.symbol == "S0_X"]
    check("截面：最强品种全程只有一笔多单",
          len(s0) == 1 and s0[0].direction == 1 and s0[0].ret > 0,
          str([(str(t.entry_date.date()), t.direction, round(t.ret, 3)) for t in s0]))
    s5 = [t for t in res.trades if t.symbol == "S5_X"]
    check("截面：最弱品种全程只有一笔空单",
          len(s5) == 1 and s5[0].direction == -1,
          str([(str(t.entry_date.date()), t.direction) for t in s5]))
    # 同一品种的两笔不能重叠（平仓当天可以同时开反向）
    by_sym: dict[str, list] = {}
    for t in res.trades:
        by_sym.setdefault(t.symbol, []).append(t)
    overlapped = []
    for k, ts in by_sym.items():
        ts.sort(key=lambda t: t.entry_date)
        overlapped += [(k, str(a.exit_date), str(b.entry_date))
                       for a, b in zip(ts, ts[1:]) if a.exit_date > b.entry_date]
    check("截面：同一品种的交易段不重叠", not overlapped, str(overlapped[:3]))

    # 截面因子分析：同一批品种、同一根 bar 上「因子排的名次」和「下一段相对
    # 涨跌的名次」对不对得上。强弱恒定，动量必须 IC 接近 1
    rep = cross.factor_report(frames, ["mom_20", "mom_60"], res.score,
                              fcfg={"horizon": 5})
    check("截面因子：能出报告", rep.ok, rep.error)
    check("截面因子：前向期数从 dict 配置读出来", rep.horizon == 5, str(rep.horizon))
    check("截面因子：表格含合成分数行", "__score__" in rep.table.index,
          str(list(rep.table.index)))
    check("截面因子：动量在截面上排得准（IC>0）",
          float(rep.table.loc["mom_20", "ic"]) > 0.9,
          f"{float(rep.table.loc['mom_20', 'ic']):+.3f}")
    q = rep.quantile.dropna()
    check("截面因子：分位收益单调递增", len(q) >= 4 and q.is_monotonic_increasing,
          str(q.round(4).to_dict()))
    check("截面因子：多空价差赚钱",
          len(rep.spread.dropna()) > 2 and float(rep.spread.iloc[-1]) > 1.0,
          str(float(rep.spread.iloc[-1])) if len(rep.spread) else "空")
    # 没有分数就直说，别抛异常 —— 界面上抛异常就是整页崩
    empty = cross.factor_report(frames, ["mom_20"], pd.DataFrame())
    check("截面因子：没有分数时给出错误而不是崩溃",
          not empty.ok and bool(empty.error), empty.error)

    # 向量化的横截面 Z-score 必须和逐行那版一模一样：算错了强弱排序就会翻，
    # 但净值曲线照样画得出来，肉眼看不出来
    rng = np.random.default_rng(7)
    panel = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("ABCD"))
    panel.iloc[::5, 1] = np.nan
    panel.iloc[3] = 7.0
    a = panel.apply(cross._row_z, axis=1)
    b = cross._cross_z(panel)
    check("截面：向量化 Z-score 与逐行一致",
          np.allclose(a.fillna(-9).to_numpy(), b.fillna(-9).to_numpy()),
          str(float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy())))))


def check_strategy_runner() -> None:
    """多策略引擎：平今判断、理论/实际比对、对齐只补托管合约。"""
    from ttrade import strategy_runner as R
    from ttrade import strategy_store as St
    from ttrade.strategy_runner import PortfolioRunner
    from ttrade.tqlive import AccountSpec

    class Pos:
        """天勤的持仓对象。两套字段都填：模拟盘填 `pos_*`，实盘 CTP 填 `volume_*`。"""

        def __init__(self, long=0, short=0, long_today=0, short_today=0,
                     long_his=None, short_his=None) -> None:
            self.pos_long = long
            self.pos_short = short
            self.pos_long_today = long_today
            self.pos_short_today = short_today
            self.pos_long_his = (long - long_today) if long_his is None else long_his
            self.pos_short_his = (short - short_today) if short_his is None else short_his
            self.volume_long = long
            self.volume_short = short
            self.volume_long_today = long_today
            self.volume_short_today = short_today
            self.volume_long_his = self.pos_long_his
            self.volume_short_his = self.pos_short_his

    class Quote:
        price_tick = 1.0
        ask_price1 = 3050.0
        bid_price1 = 3049.0
        last_price = 3049.5
        margin = 4000.0          # 每手保证金，定仓用
        volume_multiple = 10     # 合约乘数，按止损金额定仓用

    class Order(dict):
        """tqsdk 3.x 的 `insert_order()` 返回的是这个：dict 子类，不可哈希。

        早先代码把返回值直接当订单号存起来，下一轮 `dict.get(Order)` 就抛
        `TypeError: unhashable type: 'Order'`，策略在第一个 tick 之后全部停机。
        """

    class Api:
        def __init__(self) -> None:
            self.orders: list[dict] = []
            self.cancelled: list = []
            self._seq = 0

        def get_quote(self, _sym):
            return Quote()

        def insert_order(self, **kw):
            self.orders.append(kw)
            self._seq += 1
            return Order(order_id=f"oid{self._seq}")

        def cancel_order(self, order):
            self.cancelled.append(order)

    # 持仓账本会落盘。整段切到临时目录 —— 不切的话跑一次冒烟就在真实用户目录
    # 留下一份 live_positions.json，真程序下次启动会以为「账本早就有了」而跳过
    # 首次认领，升级上来的仓位就会被再开一遍。
    from ttrade import positionbook as _PB

    _pb_old = _PB.user_dir
    _PB.user_dir = lambda: tempfile.mkdtemp(prefix="tdpos_")

    try:
        r = PortfolioRunner(AccountSpec(name="t", kind="sim"), St.ChaseConfig())
        # 下面这些用例都是「账户口径」的（`_legs` / 对齐 / 拒单），别让「首次
        # 启用账本时认领账户仓位」插进来改变 delta —— 那一条单独测
        r._fresh_book = False

        # 平仓必须按实际今昨仓拆开。整笔当 CLOSE 发出去，超过昨仓的部分会被
        # 交易所整单打回（截图上「平昨手数超过持仓仓量」刷屏就是这个）
        r._pos = {"SHFE.rb2701": Pos(long=3, long_today=2)}
        check("平仓：先平昨、再平今",
              r._legs("SHFE.rb2701", -3) == [("CLOSE", 1), ("CLOSETODAY", 2)],
              str(r._legs("SHFE.rb2701", -3)))
        check("平仓：只平一部分时昨仓优先",
              r._legs("SHFE.rb2701", -2) == [("CLOSE", 1), ("CLOSETODAY", 1)],
              str(r._legs("SHFE.rb2701", -2)))
        check("平仓：空仓时直接开仓",
              r._legs("SHFE.cu2612", 1) == [("OPEN", 1)],
              str(r._legs("SHFE.cu2612", 1)))
        # 反手：+3 手多 → -2 手空，delta = -5，3 手平完还剩 2 手要开
        check("平仓：反手先平后开",
              r._legs("SHFE.rb2701", -5) == [("CLOSE", 1), ("CLOSETODAY", 2), ("OPEN", 2)],
              str(r._legs("SHFE.rb2701", -5)))
        r._pos = {"CFFEX.IF2612": Pos(long=2, long_today=2)}
        check("平仓：中金所不分平今",
              r._legs("CFFEX.IF2612", -2) == [("CLOSE", 2)],
              str(r._legs("CFFEX.IF2612", -2)))

        # **只有上期所 / 能源中心分平今**。大商所、郑商所一律 CLOSE。
        # 早先写 `symbol.startswith("CFFEX")`，而 symbol 形如 `SHFE.rb2601`，
        # 这个条件永远不成立 —— 于是给大商所也发了 CLOSETODAY，整单被打回
        r._pos = {"DCE.i2601": Pos(long=3, long_today=2)}
        check("平仓：大商所不分平今，一律 CLOSE",
              r._legs("DCE.i2601", -3) == [("CLOSE", 3)],
              str(r._legs("DCE.i2601", -3)))
        r._pos = {"CZCE.TA601": Pos(long=2, long_today=2)}
        check("平仓：郑商所不分平今，一律 CLOSE",
              r._legs("CZCE.TA601", -2) == [("CLOSE", 2)],
              str(r._legs("CZCE.TA601", -2)))
        r._pos = {"INE.sc2612": Pos(long=2, long_today=2)}
        check("平仓：能源中心分平今",
              r._legs("INE.sc2612", -2) == [("CLOSETODAY", 2)],
              str(r._legs("INE.sc2612", -2)))

        # 今昨明细滞后（刚成交那几百毫秒）时不能瞎猜。猜成昨仓 → 发 CLOSE →
        # 交易所回「平昨手数不能超过持仓仓量」→ 追单重报 → 一秒刷十几条
        p_lag = Pos(long=3, long_his=0, long_today=0)
        r._pos = {"SHFE.rb2701": p_lag}
        check("平仓：今昨明细没就绪时返回 None，不瞎猜",
              r._legs("SHFE.rb2701", -3) is None,
              str(r._legs("SHFE.rb2701", -3)))

        # 实盘 CTP 只填 volume_* 那一套，pos_* 全是 0。只读 pos_* 会把今仓当昨仓
        p_ctp = Pos()
        p_ctp.volume_short = 4
        p_ctp.volume_short_today = 4
        r._pos = {"SHFE.ag2612": p_ctp}
        check("平仓：实盘只填 volume_* 也能认出今仓",
              r._legs("SHFE.ag2612", 4) == [("CLOSETODAY", 4)],
              str(r._legs("SHFE.ag2612", 4)))

        cfg = St.StrategyConfig(sid="s1", symbol="RB", exchange="SHFE",
                                period="1d", rule_key="ma_cross", volume=2)
        st = r.add(cfg)
        st.running = True
        st.underlying = "SHFE.rb2701"
        st.signal = 1
        st.target = 2
        r._pos = {"SHFE.rb2701": Pos(), "SHFE.cu2612": Pos(short=1)}

        rows = {x["symbol"]: x for x in r.mismatch(fresh=True)}
        check("比对：理论仓与账户仓不一致能列出来", set(rows) == {"SHFE.rb2701", "SHFE.cu2612"},
              str(sorted(rows)))
        check("比对：有策略在跑的合约标为托管", rows["SHFE.rb2701"]["managed"] is True)
        check("比对：没策略在跑的合约不算托管", rows["SHFE.cu2612"]["managed"] is False)
        check("比对：差额方向正确", rows["SHFE.rb2701"]["delta"] == 2,
              str(rows["SHFE.rb2701"]["delta"]))

        # 一键对齐的语义是「账户和策略列表完全一致」，但平别人的仓得能关掉
        r.api = Api()
        n = r.align(close_orphans=False)
        check("对齐：只补托管合约（显式关掉平未托管）",
              n == 1 and len(r.api.orders) == 1, f"{n} 笔")
        if r.api.orders:
            o = r.api.orders[0]
            check("对齐：方向与手数按差额",
                  o["symbol"] == "SHFE.rb2701" and o["direction"] == "BUY"
                  and o["volume"] == 2 and o["offset"] == "OPEN",
                  str(o))

        # **对齐发的单必须记进在途列表**。不记的话 `_rebalance` 下一 tick 看到
        # 仓位还没变就又发一遍 —— 一秒十几次 tick，一笔对齐变成十几笔委托，
        # 成交之后就是「平了又开、开了又平」。
        check("对齐：发出的单记进在途列表", len(r._align_slot.pending) == 1,
              f"{len(r._align_slot.pending)} 笔")

        # 上一轮还没落地时再点，不能再发一轮
        r.api = Api()
        again = r.align(close_orphans=False)
        check("对齐：在途时再点不重复发单",
              again == 0 and not r.api.orders, f"{again} 笔 / {len(r.api.orders)}")

        # 在途期间策略也不许动手，否则「策略平、对齐开」互相打架
        r.api = Api()
        st.target = 5
        st.held = {}
        r._rebalance(st)
        check("对齐：在途时策略不调仓", not r.api.orders, f"{len(r.api.orders)} 笔")

        # 追单把这一轮追完（成交）之后就收工
        r._orders = {"oid1": type("O", (), {
            "status": "FINISHED", "volume_left": 0, "last_msg": "",
        })()}
        r._chase(r._align_slot)
        check("对齐：成交后清空在途列表", not r._align_slot.pending,
              f"{len(r._align_slot.pending)} 笔")
        r._orders = None

        # 挂在盘口外的对齐单永远不成交。没有超时兜底的话，在途列表永远不清空，
        # `_rebalance` 就一直让路 —— 整个账户看着像卡死
        r.api = Api()
        was = [(sl, sl.running) for sl in r.slots()]
        for sl in r.slots():
            sl.running = False
        r._align_slot.pending = [R.PendingOrder(
            order_id="oidT", symbol="SHFE.rb2701", direction="BUY",
            offset="OPEN", volume=1, price=3000.0, ts=time.time() - 999,
        )]
        r._orders = {"oidT": type("O", (), {
            "status": "ALIVE", "volume_left": 1, "last_msg": "",
        })()}
        r.tick()
        check("对齐：挂太久自动撤单", len(r.api.cancelled) == 1,
              f"撤 {len(r.api.cancelled)} 笔")
        check("对齐：超时后清空在途，策略恢复", not r._align_slot.pending,
              f"{len(r._align_slot.pending)} 笔")
        for sl, on in was:
            sl.running = on
        r._orders = None

        # 连点两下：5 秒冷却期内不能再下发一轮（用户手动点也是这个节奏）
        r.api = Api()
        r._align_ts = time.time()
        blocked = r.align(close_orphans=False)
        check("对齐：5 秒内连点不重复下发",
              blocked == 0 and not r.api.orders, f"{blocked} 笔 / {len(r.api.orders)}")

        r.api = Api()
        r._align_ts = 0.0                      # 模拟「过了一会儿再点」
        n = r.align()
        check("对齐：默认连未托管仓位一起平", n == 2 and len(r.api.orders) == 2, f"{n} 笔")
        cu = [o for o in r.api.orders if o["symbol"] == "SHFE.cu2612"]
        check("对齐：未托管合约反向平掉",
              bool(cu) and cu[0]["direction"] == "BUY" and cu[0]["volume"] == 1
              and cu[0]["offset"] == "CLOSE", str(cu))
        r._align_slot.pending.clear()      # 这一轮假设已经成交，继续往下测

        # insert_order 返回 Order 对象。把它当订单号存，下一轮查委托就炸
        r.api = Api()
        st.pending.clear()
        r._send(st, "SHFE.rb2701", 1)
        check("下单：订单号取 Order.order_id 而不是 Order 本身",
              bool(st.pending) and isinstance(st.pending[0].order_id, str)
              and st.pending[0].order_id == "oid1",
              str(st.pending[0].order_id) if st.pending else "没有委托")
        # ---- 追单状态机：撤 → 等死 → 重报 ----
        def _chase_once(status="ALIVE", left=1, phase="live", retries=0,
                        enabled=True, max_retries=3):
            api = Api()
            r.api = api
            r.chase = St.ChaseConfig(enabled=enabled, wait_sec=1,
                                     retries=max_retries, ticks=2)
            st.pending = [R.PendingOrder(
                order_id="oid1", symbol="SHFE.rb2701", direction="BUY",
                offset="OPEN", volume=1, price=3000.0, ts=0.0,
                retries=retries, phase=phase,
            )]
            r._orders = {"oid1": type("O", (), {
                "status": status, "volume_left": left, "last_msg": "资金不足",
            })()}
            r._chase(st)
            return api, list(st.pending)

        api, left_over = _chase_once(status="FINISHED", left=0)
        check("追单：成交后从在途列表里去掉", not left_over and not api.orders)

        api, left_over = _chase_once(status="ALIVE", left=0)
        check("追单：成交吃满但状态没翻，也当结束", not left_over)

        api, left_over = _chase_once(enabled=False)
        check("追单：关掉之后不撤不追，单子继续挂着",
              not api.cancelled and len(left_over) == 1 and left_over[0].phase == "live")

        api, left_over = _chase_once(max_retries=0)
        check("追单：次数设 0 时只挂着不撤",
              not api.cancelled and len(left_over) == 1)

        api, left_over = _chase_once()
        check("追单：超时先撤单，等确认死再重报",
              len(api.cancelled) == 1 and len(left_over) == 1
              and left_over[0].phase == "cancelling" and not api.orders,
              f"撤 {len(api.cancelled)} 次，phase={left_over[0].phase if left_over else '—'}")

        # 这一条是那个「只撤不追」的 bug：撤单确认（DEAD）之后必须重报
        api, left_over = _chase_once(status="DEAD", phase="cancelling", retries=0)
        check("追单：撤单确认后重报",
              len(api.orders) == 1 and len(left_over) == 1
              and left_over[0].retries == 1 and left_over[0].phase == "live"
              and left_over[0].order_id == "oid1",
              f"{len(api.orders)} 笔重报，retries={left_over[0].retries if left_over else '—'}")

        api, left_over = _chase_once(status="DEAD", phase="cancelling", retries=3)
        check("追单：追满次数就放弃", not left_over and not api.orders)

        api, left_over = _chase_once(status="DEAD", phase="live")
        check("追单：交易所拒单直接丢掉，不重报",
              not left_over and not api.orders)

        st.pending.clear()
        r._orders = None

        # 改手数要让目标仓位立刻跟上，否则得等下一根 bar。
        # 按资金/止损金额定仓时手数依赖价格，只能等下一根 bar 重算 —— 这里测的是
        # 固定手数模式：手数就写在配置里，必须立刻生效
        cfg2 = cfg.clone(volume=5)
        cfg2.sizing = "fixed"
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
        rows2 = {x["symbol"]: x for x in r.mismatch(fresh=True)}
        check("换月：老合约仍算托管", rows2["SHFE.rb2611"]["managed"] is True,
              str(rows2.get("SHFE.rb2611")))
        r.api = Api()
        r._align_ts = 0.0
        n2 = r.align()
        check("换月：对齐会把老合约平掉", n2 == 2 and len(r.api.orders) == 2, f"{n2} 笔")
        r._align_slot.pending.clear()
        old = [o for o in r.api.orders if o["symbol"] == "SHFE.rb2611"]
        check("换月：老合约下的是卖平 3 手",
              bool(old) and old[0]["direction"] == "SELL" and old[0]["volume"] == 3
              and old[0]["offset"] == "CLOSE", str(old))

        # 停掉的策略不参与理论持仓汇总
        st.running = False
        st.retired = []
        check("理论持仓：只算运行中的策略", r.theoretical() == {}, str(r.theoretical()))

        # ---- 单策略只跟自己的账本比，不盯账户净仓 ----
        # 账户是**一个**、策略是**一群**：同一个品种被两条策略同时交易时，账户
        # 净仓是它们的和，单看账户分不出哪几手是谁的。早先按账户调仓，两条策略
        # 会互相把对方的仓当成自己的偏差，来回对打；手工开的仓也会被策略当成
        # 「少开了」补回去。现在每条策略按**自己委托的成交**记账（positionbook）。
        from ttrade import positionbook as PB

        pb_tmp = tempfile.mkdtemp(prefix="tdpos")
        pb_old = PB.user_dir
        PB.user_dir = lambda: pb_tmp
        try:
            r2 = PortfolioRunner(AccountSpec(name="仓位账本测试", kind="sim"),
                                 St.ChaseConfig())
            acct = r2.spec.label()

            def _slot(sid: str, vol: int = 5):
                s = r2.add(St.StrategyConfig(
                    sid=sid, symbol="RB", exchange="SHFE", period="1d",
                    rule_key="ma_cross", sizing="fixed", volume=vol))
                s.running = True
                s.underlying = "SHFE.rb2701"
                return s

            a = _slot("a")
            b = _slot("b")
            a.signal = b.signal = -1
            a.target = b.target = -5
            a.held = {"SHFE.rb2701": -5}     # a 的 5 手空已经在账户里了
            b.held = {}
            r2.api = Api()
            r2._pos = {"SHFE.rb2701": Pos(short=5)}
            r2._rebalance(a)
            check("账本：自己已经到位就不再发单", not r2.api.orders,
                  f"{len(r2.api.orders)} 笔")
            r2._rebalance(b)
            check("账本：另一条策略照自己的账本开 5 手",
                  len(r2.api.orders) == 1 and r2.api.orders[0]["volume"] == 5,
                  str(r2.api.orders))

            # 账户里还有 7 手多 —— 手工开的。两条策略目标都是 0，谁都不许动它
            a.pending.clear()
            b.pending.clear()
            r2.api = Api()
            r2._pos = {"SHFE.rb2701": Pos(long=7)}
            a.target = b.target = 0
            a.held = {}
            b.held = {}
            r2._rebalance(a)
            r2._rebalance(b)
            check("账本：账户里的手工仓不被策略平掉", not r2.api.orders,
                  f"{len(r2.api.orders)} 笔")

            # 成交要记进自己的账本，并且落盘 —— 重启不读回来的话，目标 -3 会
            # 从头再开一遍，账户直接变 -6
            c = _slot("c", 3)
            c.target = -3
            c.held = {}
            r2.api = Api()
            r2._pos = {"SHFE.rb2701": Pos()}
            r2._rebalance(c)
            oid = c.pending[0].order_id if c.pending else ""
            r2._orders = {oid: type("O", (), {
                "status": "FINISHED", "volume_left": 0, "last_msg": "",
            })()}
            r2._chase(c)
            check("账本：成交记进自己的持仓", c.held.get("SHFE.rb2701") == -3, str(c.held))
            check("账本：落盘后重启能读回来",
                  PB.load(acct).get("c", {}).get("SHFE.rb2701") == -3,
                  str(PB.load(acct)))
            c.pending.clear()

            # 部分成交后被撤：重报只能补剩下的。不减 `volume` 的话重报会把成交
            # 过的部分再发一遍，5 手变 8 手
            d = _slot("d", 5)
            d.target = -5
            d.held = {}
            r2.api = Api()
            r2._pos = {"SHFE.rb2701": Pos()}
            r2._rebalance(d)
            part = d.pending[0]
            r2._orders = {part.order_id: type("O", (), {
                "status": "DEAD", "volume_left": 2, "last_msg": "",
            })()}
            part.phase = "cancelling"
            part.retries = 0
            r2.chase = St.ChaseConfig(enabled=True, wait_sec=1, retries=3, ticks=1)
            r2._chase(d)
            check("账本：部分成交只补剩下的（5 手成交 3 手，重报 2 手）",
                  d.held.get("SHFE.rb2701") == -3
                  and bool(r2.api.orders) and r2.api.orders[-1]["volume"] == 2,
                  f"held={d.held} 重报={[o['volume'] for o in r2.api.orders]}")
            d.pending.clear()

            # 账户被手工平掉之后：账本 -5、账户 0，目标回到 0 时想「买 5 手平仓」。
            # 不能真的去买 —— 那是反手开 5 手多。只能把账本按账户收窄
            a.underlying = "SHFE.rb2701"
            a.target = 0
            a.held = {"SHFE.rb2701": -5}
            a.book_ts = 0.0
            r2.api = Api()
            r2._pos = {"SHFE.rb2701": Pos()}
            r2._rebalance(a)
            check("账本：账户里没这些仓了就不反手开多",
                  not r2.api.orders and not a.held,
                  f"{len(r2.api.orders)} 笔 / held={a.held}")

            # 换月的老合约也按自己的账本平，不看账户（账户里可能有别人的仓）
            a.retired = ["SHFE.rb2611"]
            a.held = {"SHFE.rb2611": 3}
            a.book_ts = 0.0
            r2.api = Api()
            r2._pos = {"SHFE.rb2611": Pos(long=10, long_his=10)}   # 其中 7 手是别人的
            r2._rebalance(a)
            old = r2.api.orders[0] if r2.api.orders else {}
            check("账本：换月老合约只平自己那 3 手",
                  old.get("direction") == "SELL" and old.get("volume") == 3
                  and old.get("offset") == "CLOSE", str(old))

            # 一键对齐之后账本要跟着摆：对齐的目标就是 sum(target)，发完之后
            # 每条运行中的策略账本正好是自己的 target
            a.retired = []
            a.held = {"SHFE.rb2701": -9}
            a.target = -5
            a.book_ts = 0.0
            b.held = {"SHFE.rb2701": 4}
            b.target = 0
            b.book_ts = 0.0
            r2.api = Api()
            r2._pos = {"SHFE.rb2701": Pos()}
            r2._align_ts = 0.0
            r2._align_slot.pending.clear()
            r2.align(close_orphans=False)
            check("账本：一键对齐后按 target 重新记账",
                  a.held.get("SHFE.rb2701") == -5 and not b.held,
                  f"a={a.held} b={b.held}")
            r2._align_slot.pending.clear()

            # 升级上来的第一次启动：账本文件还不存在，账户里已经有 -5 手。
            # 不认领的话目标 -5 会再开 5 手，账户直接变 -10
            r3 = PortfolioRunner(AccountSpec(name="认领测试", kind="sim"),
                                 St.ChaseConfig())
            r3._fresh_book = True
            r3.api = Api()
            e = r3.add(St.StrategyConfig(sid="e", symbol="RB", exchange="SHFE",
                                         period="1d", rule_key="ma_cross",
                                         sizing="fixed", volume=5))
            e.running = True
            e.underlying = "SHFE.rb2701"
            e.target = -5
            r3._pos = {"SHFE.rb2701": Pos(short=5)}
            r3._rebalance(e)
            check("账本：首次启用时认领账户里已有的仓位，不重复开",
                  e.held.get("SHFE.rb2701") == -5 and not r3.api.orders,
                  f"held={e.held} 下单={len(r3.api.orders)} 笔")
        finally:
            PB.user_dir = pb_old

        # ---- 仓位分配：不再固定 1 手 ----
        r.api = Api()
        r._pos = {}
        cap = St.StrategyConfig(sid="s-cap", symbol="RB", exchange="SHFE",
                                period="1d", rule_key="ma_cross",
                                sizing="capital", principal=1_000_000.0,
                                capital_pct=50.0)
        st_cap = r.add(cap)
        st_cap.underlying = "SHFE.rb2701"
        check("仓位：50 万 ÷ 每手保证金 4000 = 125 手",
              r._lots(st_cap, 3049.5, 0.0) == 125, str(r._lots(st_cap, 3049.5, 0.0)))

        # 本金是**乘数**：比例不变、本金翻倍，手数跟着翻倍。存百分比的全部意义
        # 就在这一条 —— 改本金不用挨个回去重算
        st_cap.cfg.principal = 2_000_000.0
        check("仓位：本金翻倍 → 手数翻倍",
              r._lots(st_cap, 3049.5, 0.0) == 250, str(r._lots(st_cap, 3049.5, 0.0)))
        st_cap.cfg.principal = 1_000_000.0
        st_cap.cfg.capital_pct = 30.0
        check("仓位：本金 100 万 × 30% = 30 万 → 75 手",
              r._lots(st_cap, 3049.5, 0.0) == 75, str(r._lots(st_cap, 3049.5, 0.0)))
        st_cap.cfg.capital_pct = 50.0

        rk = St.StrategyConfig(sid="s-risk", symbol="RB", exchange="SHFE",
                               period="1d", rule_key="sbbox_mini",
                               sizing="risk", principal=1_000_000.0, risk_pct=2.0,
                               stop_mult=5.0, stop_atr_n=14)
        st_rk = r.add(rk)
        st_rk.underlying = "SHFE.rb2701"
        # 止损距离 5×20 = 100 点，乘数 10 → 每手亏 1000，2 万 ÷ 1000 = 20 手
        check("仓位：按止损金额反推（2 万 / 5×ATR×乘数）",
              r._lots(st_rk, 3049.5, 20.0) == 20, str(r._lots(st_rk, 3049.5, 20.0)))

        fx = St.StrategyConfig(sid="s-fix", symbol="RB", exchange="SHFE",
                               period="1d", rule_key="ma_cross",
                               sizing="fixed", volume=7)
        st_fx = r.add(fx)
        st_fx.underlying = "SHFE.rb2701"
        check("仓位：固定手数仍然可用", r._lots(st_fx, 3049.5, 20.0) == 7)

        # 老配置没有 sizing 字段：按止损金额定仓的策略要自动走 risk
        old_risk = St.StrategyConfig.from_dict(
            {"sid": "s-old", "symbol": "RB", "exchange": "SHFE",
             "rule_key": "sbbox_mini", "rule_params": {"stop_mult": 5.0}}
        )
        check("仓位：老配置里 sbbox_mini 自动按止损金额",
              old_risk.sizing == "risk", old_risk.sizing)
        old_cap = St.StrategyConfig.from_dict(
            {"sid": "s-old2", "symbol": "RB", "exchange": "SHFE", "rule_key": "ma_cross"}
        )
        check("仓位：老配置默认按资金 50 万",
              old_cap.sizing == "capital" and old_cap.capital == 500_000.0,
              f"{old_cap.sizing} {old_cap.capital}")
        # 老配置的绝对金额要折成本金百分比，而且**折完还是原来那笔钱**。
        # 直接套新默认 30% 的话，用户升级当天手数就变成 0.6 倍
        old_cap2 = St.StrategyConfig.from_dict(
            {"sid": "s-old3", "symbol": "RB", "exchange": "SHFE", "rule_key": "ma_cross",
             "capital": 500_000.0, "risk_amount": 20_000.0}
        )
        check("仓位：老配置的金额折成本金百分比",
              (old_cap2.principal, old_cap2.capital_pct, old_cap2.capital,
               old_cap2.risk_pct, old_cap2.risk_amount)
              == (1_000_000.0, 50.0, 500_000.0, 2.0, 20_000.0),
              f"{old_cap2.capital_pct}% {old_cap2.capital}")

        # 保证金拿不到时按品种兜底：国债 3%（约 33 倍）、股指 12%（约 8 倍）
        check("仓位：国债保证金率 3%", abs(R.default_margin_rate("T") - 0.03) < 1e-9)
        check("仓位：股指保证金率 12%", abs(R.default_margin_rate("IF") - 0.12) < 1e-9)
        check("仓位：商品保证金率 10%", abs(R.default_margin_rate("RB") - 0.10) < 1e-9)

        for sid in ("s-cap", "s-risk", "s-fix"):
            r.remove(sid)

        # ---- 拒单不能只记一笔 ----
        # `tick()` 跟着行情推送跑，一秒能来十几次。没有冷却 + 计数的话，
        # 一笔注定失败的委托一秒能重报十几遍，日志刷屏、主线程直接卡死
        # （截图上「平昨手数不能超过持仓仓量」刷屏就是这个）。
        rej = St.StrategyConfig(sid="s-rej", symbol="RB", exchange="SHFE",
                                period="1d", rule_key="ma_cross", volume=1)
        st_rej = r.add(rej)
        st_rej.underlying = "SHFE.rb2701"
        st_rej.running = True
        st_rej.target = 1
        r._pos = {}
        r.chase = St.ChaseConfig(enabled=True, wait_sec=1, retries=2, ticks=1)

        def _rejected() -> None:
            r._orders = {"oidR": type("O", (), {
                "status": "DEAD", "volume_left": 1,
                "last_msg": "平昨手数(0)不能超过持仓仓量(0)",
            })()}
            st_rej.pending = [R.PendingOrder(
                order_id="oidR", symbol="SHFE.rb2701", direction="BUY",
                offset="OPEN", volume=1, price=3000.0, ts=0.0,
            )]
            r._chase(st_rej)
            st_rej.pending.clear()

        _rejected()
        check("拒单：记下次数与原因",
              st_rej.rejects == 1 and "不能超过持仓仓量" in st_rej.reject_msg,
              f"{st_rej.rejects} / {st_rej.reject_msg}")

        r.api = Api()
        st_rej.reject_ts = time.time()
        r._rebalance(st_rej)
        check("拒单：冷却期内不重报", not r.api.orders, f"{len(r.api.orders)} 笔")

        st_rej.reject_ts = time.time() - 60
        r._rebalance(st_rej)
        check("拒单：冷却结束才重报", len(r.api.orders) == 1, f"{len(r.api.orders)} 笔")
        st_rej.pending.clear()

        r.api = Api()
        st_rej.rejects = 0
        st_rej.running = True
        st_rej.error = ""
        for _ in range(3):
            _rejected()
        check("拒单：连续 3 次自动停机",
              st_rej.running is False and st_rej.rejects == 3,
              f"running={st_rej.running} rejects={st_rej.rejects}")
        check("拒单：停机原因写进 error", "已停机" in st_rej.error, st_rej.error)

        r.api = Api()
        st_rej.target = 2
        r._rebalance(st_rej)
        check("拒单：停机后不再发单", not r.api.orders, f"{len(r.api.orders)} 笔")

        # 重新启动要能清掉旧的拒单记录，否则一启动就是「已停机」
        r.enable("s-rej", False)
        r.enable("s-rej", True)
        check("拒单：重新启动清空计数",
              st_rej.rejects == 0 and st_rej.running is True,
              f"rejects={st_rej.rejects} running={st_rej.running}")
        r.remove("s-rej")

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
        # 固定 2 手：这一段测的是「K 线 → 信号 → 目标仓位」，手数怎么来的
        # 由上面「仓位分配」那几条单独覆盖
        c5 = St.StrategyConfig(sid="s5", symbol="RB", exchange="SHFE",
                               period="5m", rule_key="ma_cross",
                               sizing="fixed", volume=2,
                               rule_params={"fast": 5, "slow": 20})
        st5 = r.add(c5)
        # 这一段测的是「K 线 → 信号」的纯逻辑，用人工序列；置空 qfq 缓存让
        # `_signal_frame` 退回实时价，否则会去读真库把人工序列覆盖掉
        r._qfq_base[r._skey(c5)] = pd.DataFrame()
        r._klines[r._skey(c5)] = bars
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

        # 最后一根是正在走的活 bar。拿它算，开盘那一瞬 close==open，突破类条件
        # 会在开盘成立、收盘又消失 —— 实盘比回测早一根进场。这里追加一根「开盘
        # 就砸到 1000」的活 bar：算进去的话快线会掉到慢线下方（信号变 0），
        # 只喂已收盘的 bar 则应当维持上一根的多头信号
        live = pd.concat([bars, pd.DataFrame({
            "datetime": [base + n * 300 * 10**9],
            "open": [1000.0], "high": [1000.0], "low": [1000.0],
            "close": [1000.0], "volume": [100.0],
            "open_oi": [1000.0], "close_oi": [1000.0],
        })]).reset_index(drop=True)
        r._klines[r._skey(c5)] = live
        r._refresh_signal(st5)
        check("信号：新 bar 到了会用刚收盘的那根重算",
              st5.last_bar == int(live["datetime"].iloc[-1]), str(st5.last_bar))
        check("信号：正在走的活 bar 不参与计算（开盘假信号）",
              st5.signal == 1 and st5.target == 2,
              f"signal={st5.signal} target={st5.target}")

        # **触发是按 bar 的，不是按 tick 的**。同一根 bar 里价格怎么跳都不该
        # 重算信号 —— 重算就会在盘中反复翻仓，和回测「T 收盘出信号、T+1 开盘
        # 成交」的口径彻底分叉。这里把活 bar 的收盘价从 1000 推到 9000，
        # 信号和 last_bar 都必须一动不动
        before = (st5.signal, st5.target, st5.last_bar)
        for i in range(5):
            live.iloc[-1, live.columns.get_loc("close")] = 1000.0 + i * 2000
            r._refresh_signal(st5)
        check("触发：同一根 bar 内推价不重算信号",
              (st5.signal, st5.target, st5.last_bar) == before,
              f"{before} → {(st5.signal, st5.target, st5.last_bar)}")

        # 实盘算信号必须用「前复权历史 + 实时 bar」，和回测同一口径 —— 否则换月
        # 那一下的复权差会把信号推到阈值另一侧（回测空仓、实盘反向持仓）。这里
        # 不碰真库：手动给一份 qfq 底，验证只接更晚的实时 bar、重叠保留本地那份
        rq = PortfolioRunner(AccountSpec(name="口径测试", kind="sim"), St.ChaseConfig())
        cq = St.StrategyConfig(sid="sq", symbol="ZZ", exchange="SHFE", period="5m",
                               rule_key="ma_cross", sizing="fixed", volume=1)
        stq = rq.add(cq)
        bi = pd.date_range("2026-01-05 09:00", periods=40, freq="5min")
        baseq = pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 1.0, "open_oi": 1.0, "close_oi": 1.0, "adj_factor": 1.1,
        }, index=bi)
        baseq.index.name = "datetime"
        rq._qfq_base[rq._skey(cq)] = baseq
        lv = pd.DataFrame({
            "datetime": [(bi[0] + pd.Timedelta(minutes=5 * (30 + k)))
                         .tz_localize("Asia/Shanghai").tz_convert("UTC")
                         .as_unit("ns").value for k in range(15)],
            "open": 200.0, "high": 200.0, "low": 200.0, "close": 200.0,
            "volume": 1.0, "open_oi": 1.0, "close_oi": 1.0,
        })
        fr = rq._signal_frame(stq, lv)
        check("实盘口径：前复权底 + 只接更晚的实时 bar",
              len(fr) == 45
              and abs(float(fr["close"].iloc[39]) - 100.0) < 1e-9
              and abs(float(fr["close"].iloc[-1]) - 200.0) < 1e-9,
              f"len={len(fr)} c39={float(fr['close'].iloc[39]):.2f} "
              f"last={float(fr['close'].iloc[-1]):.2f}")

        # 周期不认识时必须在 enable 就拒掉，而不是等订阅时抛 KeyError 把策略打死
        bad = St.StrategyConfig(sid="s9", symbol="RB", exchange="SHFE",
                                period="7m", rule_key="ma_cross", volume=1)
        st_bad = r.add(bad)
        r.enable("s9", True)
        check("启停：不支持的周期被拒绝",
              not st_bad.running and "周期" in st_bad.error, st_bad.error)
        # ---- 比对结果多久刷新一次 ----
        # 快照 0.4 秒推一次。每推一次都把「全部策略理论仓 × 全部持仓」对一遍，
        # 账户里合约一多就是白烧 CPU，而仓位是秒级才变一次的东西。缓存 5 秒；
        # 真正要下单（align）时走 fresh=True，拿旧数补仓会补错手数。
        r._slots.clear()
        r._align_slot.pending.clear()
        r._mm_ts = 0.0
        st_mm = r.add(St.StrategyConfig(sid="s-mm", symbol="RB", exchange="SHFE",
                                        period="1d", rule_key="ma_cross", volume=2))
        st_mm.running = True
        st_mm.underlying = "SHFE.rb2701"
        st_mm.target = 2
        r._pos = {"SHFE.rb2701": Pos()}
        first = r.mismatch(fresh=True)
        check("比对：算出来是差 2 手多",
              len(first) == 1 and first[0]["delta"] == 2, str(first))
        first[0]["delta"] = 999
        check("比对：返回副本，改不到缓存",
              r._mm_cache and r._mm_cache[0]["delta"] == 2, str(r._mm_cache))
        r._pos = {"SHFE.rb2701": Pos(long=2)}
        check("比对：5 秒内不重算（仍报差 2 手）",
              [x["delta"] for x in r.mismatch()] == [2], str(r.mismatch()))
        check("比对：fresh=True 绕过缓存（此刻已对齐）",
              r.mismatch(fresh=True) == [], str(r.mismatch(fresh=True)))

        # ---- 界面看 K 线：引擎得把订阅着的实时序列交出去 ----
        # 组合页原来只画本地库里的历史，打开一看停在几天前。引擎订的本来就是
        # 实时行情，缺的只是一个通道；而且是「点名才给」，不是每份快照都带。
        kl = pd.DataFrame({
            "datetime": [1_700_000_000 * 10**9, 1_700_000_300 * 10**9],
            "open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5],
            "close": [1.2, 2.2], "volume": [10.0, 20.0],
            "open_oi": [0.0, 0.0], "close_oi": [0.0, 0.0],
        })
        r._klines[r._skey(st_mm.cfg)] = kl
        got = r.bars_of(st_mm.cfg.sid)
        check("实时 K 线：能取到订阅着的序列", got is not None and len(got) == 2,
              f"{None if got is None else len(got)} 根")
        check("实时 K 线：策略不存在返回 None", r.bars_of("没有这个") is None)
        r.want_bars([st_mm.cfg.sid])
        check("实时 K 线：点名的策略进了下一份快照",
              st_mm.cfg.sid in (r.snapshot().get("bars") or {}),
              str(list(r.snapshot().get("bars") or {})))
        check("实时 K 线：取完就清空，不每份快照都带",
              r.snapshot().get("bars") == {}, str(r.snapshot().get("bars")))

        # ---- 退出 / 重启时的遗留委托 ----
        # 程序被强杀（或直接关机）时 `close()` 跑不到，限价单一直挂在交易所。
        # 下次连上账户，用户什么都没点，委托却在成交 —— 看着就像「不点按钮
        # 它自己在下单」。所以每发一笔就记下 order_id，下次连上先撤掉。
        import shutil

        from ttrade import daylimit as DL2
        from ttrade import orderbook as OB

        _ob_dir, _dl_dir = OB.user_dir, DL2.user_dir
        _tmp = tempfile.mkdtemp(prefix="td_ob_")
        OB.user_dir = lambda: _tmp
        DL2.user_dir = lambda: _tmp
        try:
            check("遗留委托：一开始是空的", OB.load("acct") == [], str(OB.load("acct")))
            OB.remember("acct", "oidA")
            OB.remember("acct", "oidB")
            check("遗留委托：按顺序读回", OB.load("acct") == ["oidA", "oidB"],
                  str(OB.load("acct")))
            OB.remember("acct", "oidA")
            check("遗留委托：重复记不翻倍", OB.load("acct") == ["oidA", "oidB"],
                  str(OB.load("acct")))
            OB.forget("acct", "oidA")
            check("遗留委托：成交后删掉", OB.load("acct") == ["oidB"],
                  str(OB.load("acct")))
            check("遗留委托：账户之间各记各的", OB.load("acct2") == [],
                  str(OB.load("acct2")))

            # 跨交易日的旧名单必须失效。那些单交易所早就作废了，拿旧 id 去撤，
            # 万一撞上今天复用同一个号的单，撤掉的就是别人正在挂的单
            _td = OB.trading_day
            OB.trading_day = lambda: "2000-01-01"
            OB.remember("stale", "oidOld")
            OB.trading_day = _td
            check("遗留委托：跨交易日自动失效", OB.load("stale") == [],
                  str(OB.load("stale")))

            class _Ord:
                def __init__(self, status: str) -> None:
                    self.status = status

            r.api = Api()
            r._orders = {"oidAlive": _Ord("ALIVE"), "oidDone": _Ord("FINISHED")}
            OB.remember(r.spec.label(), "oidAlive")
            OB.remember(r.spec.label(), "oidDone")
            r._sweep_orders()
            check("重启：撤掉上次遗留的活单",
                  r.api.cancelled == [r._orders["oidAlive"]],
                  str(len(r.api.cancelled)))
            check("重启：已经结束的单不碰",
                  r._orders["oidDone"] not in r.api.cancelled)
            check("重启：名单撤完就清空", OB.load(r.spec.label()) == [],
                  str(OB.load(r.spec.label())))

            # 退出时要把在途单撤掉。程序都退出了还挂着限价单，等于「没人管的
            # 仓位」：下次打开账户，用户没点任何按钮，委托却照样成交
            r._orders = {"oidP": _Ord("ALIVE")}
            r.api = Api()
            api_ref = r.api
            st_mm.pending = [R.PendingOrder(order_id="oidP", symbol="SHFE.rb2701",
                                            direction="BUY", offset="OPEN",
                                            volume=1, price=3000.0, ts=time.time())]
            r.close()
            check("退出：撤掉在途单", len(api_ref.cancelled) == 1,
                  str(len(api_ref.cancelled)))
            check("退出：在途列表清空", not st_mm.pending, str(len(st_mm.pending)))
        finally:
            OB.user_dir, DL2.user_dir = _ob_dir, _dl_dir
            shutil.rmtree(_tmp, ignore_errors=True)

    except Exception as exc:  # noqa: BLE001
        check("多策略引擎", False, f"{type(exc).__name__}: {exc}")
    finally:
        _PB.user_dir = _pb_old


def check_sizing() -> None:
    """回测的仓位口径。三条规则最容易悄悄改坏：

    - 按资金分配是**单利**：赚了不加大仓、亏了不缩仓
    - 期货一手是「价格 × 合约乘数」，忘了乘数就是 10 倍杠杆
    - 「名义敞口 = 权益」不能把本金全押成保证金
    """
    from ttrade.backtest import BacktestConfig, buy_and_hold, run_backtest, size_lots

    base = dict(multiplier=10.0, margin_rate=0.10, capital=500_000.0)
    cap = BacktestConfig(sizing="capital", **base)
    # 价格 3000、乘数 10、保证金 10% → 每手 3000，50 万正好 166 手
    check("仓位：50 万 ÷ (3000×10×10%) = 166 手",
          size_lots(cap, 3000.0, 500_000.0) == 166,
          str(size_lots(cap, 3000.0, 500_000.0)))
    # 单利：权益翻倍 / 腰斩都不影响手数
    check("仓位：按资金分配是单利（赚了不加仓）",
          size_lots(cap, 3000.0, 1_000_000.0) == 166,
          str(size_lots(cap, 3000.0, 1_000_000.0)))
    check("仓位：按资金分配是单利（亏了不减仓）",
          size_lots(cap, 3000.0, 200_000.0) == 166,
          str(size_lots(cap, 3000.0, 200_000.0)))

    risk = BacktestConfig(sizing="risk", **base)
    # 2 万 ÷ (5 × ATR 20 × 乘数 10) = 20 手
    check("仓位：止损 2 万 ÷ (5×20×10) = 20 手",
          size_lots(risk, 3000.0, 500_000.0, 20.0) == 20,
          str(size_lots(risk, 3000.0, 500_000.0, 20.0)))

    unit = BacktestConfig(sizing="unit", **base)
    # 名义敞口 = 权益：50 万 ÷ (3000×10) = 16.7 手，不是 166 手
    check("仓位：本金全额口径不乘杠杆",
          abs(size_lots(unit, 3000.0, 500_000.0) - 16.6667) < 0.01,
          str(size_lots(unit, 3000.0, 500_000.0)))

    # 真跑一遍。手数会跟着开仓价变（价高就少开几手），但**名义敞口**必须
    # 一直钉在「分配资金 × 杠杆」上 —— 那才是单利；跟着权益走就是复利
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    px = pd.Series(3000.0 + np.sin(np.arange(n) / 9.0) * 10.0, index=idx)
    df = pd.DataFrame({"open": px, "high": px * 1.001, "low": px * 0.999,
                       "close": px, "signal": (np.arange(n) // 40) % 2})
    res = run_backtest(df, BacktestConfig(sizing="capital", **base))
    notional = [t.lots * 10.0 * t.entry_price for t in res.trades]
    check("仓位：单利下名义敞口固定在 500 万",
          bool(notional) and max(abs(x - 5_000_000) for x in notional) < 32_000,
          f"{min(notional or [0]):,.0f}~{max(notional or [0]):,.0f}")

    # 基准必须和策略同口径，否则「策略 vs 买入持有」是在比杠杆
    bh = buy_and_hold(df, BacktestConfig(sizing="capital", **base))
    check("仓位：买入持有与策略同口径",
          abs(bh.trades[0].lots - 166) < 0.01, str(bh.trades[0].lots))


def check_no_blowup() -> None:
    """回测**不做爆仓强平**。

    权益亏穿（净值转负）也要照跑到最后一根：回测是虚拟测试，杠杆按保证金率算，
    比实盘敢开的大得多，强平掉只会把后半段历史抹掉。守三件事：
    亏穿之后持仓还在、净值能继续算（不是被钉在 0）、绩效里没有 NaN。
    """
    from ttrade.backtest import BacktestConfig, run_backtest

    try:
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        px = np.linspace(100.0, 400.0, n)      # 一路涨，空头必然亏穿
        df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                           "close": px, "signal": -1}, index=idx)
        cfg = BacktestConfig(initial_cash=100_000, sizing="fixed", volume=100,
                             multiplier=10.0, fee_rate=0.0, slippage=0.0)
        res = run_backtest(df, cfg)
        eq = res.equity

        check("回测：不强平（净值能转负）", float(eq.min()) < 0,
              f"最低权益 {eq.min():,.0f}")
        check("回测：亏穿之后仍持仓到最后",
              int((res.position != 0).sum()) == n - 1,
              f"持仓 {int((res.position != 0).sum())}/{n - 1} 根")
        # 原来亏穿那根会被写成 0 并从此停摆，曲线变成一条平线
        check("回测：净值不是被钉在 0", float(eq.iloc[-1]) < 0,
              f"期末 {eq.iloc[-1]:,.0f}")
        # 比例型指标（夏普 / 年化波动 / 卡玛）在亏穿后没有意义，标 NaN；
        # 绝对值指标（累计收益 / 回撤 / 期末资金）照常给，不能变成 NaN
        check("回测：亏穿后夏普/波动/卡玛标 NaN",
              res.metrics["sharpe"] != res.metrics["sharpe"]
              and res.metrics["ann_vol"] != res.metrics["ann_vol"]
              and res.metrics["calmar"] != res.metrics["calmar"],
              f"{res.metrics['sharpe']}/{res.metrics['ann_vol']}")
        bad = {k: v for k, v in res.metrics.items()
               if k not in ("sharpe", "ann_vol", "calmar")
               and isinstance(v, float) and v != v}
        check("回测：绝对值指标没有 NaN", not bad, str(bad))
        check("回测：累计收益可以小于 -100%", res.metrics["total_return"] < -1.0,
              f"{res.metrics['total_return']:.2f}")
        check("回测：已经没有爆仓这个指标", "n_blowup" not in res.metrics,
              str(sorted(res.metrics)[:4]))
    except Exception as exc:  # noqa: BLE001
        check("回测不强平", False, f"{type(exc).__name__}: {exc}")


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
        # 只管内置策略：用户注册的策略没有手写文档，四个字段里只有 logic 有值
        # （兜底成 spec.desc），拿同一把尺子量会把「用 Agent 存了一条策略」判成错
        thin = [
            k for k in STRATEGIES if k in L.STRATEGY_DOCS
            and not all(getattr(L.strategy_doc(k), f) for f in ("logic", "edge", "fits", "risk"))
        ]
        check("策略文档：逻辑/优势/适应/风险齐全", not thin, str(thin))

        # 2026-10 从「我的」转成内置的三条。内置的标志是 `code` 为空 —— 列表里
        # 显示「内置」、删除按钮也不给删；要是哪天又被写回 user_*.json，
        # `_load_user_*` 会带着 code 覆盖掉内置版本，这三条就该红了
        for k in ("sbbox_mini", "im_orb"):
            check(f"内置策略：{k} 不是用户策略", not STRATEGIES[k].code, "")
        check("内置因子：roll_main_oi 不是用户因子",
              not FACTORS["roll_main_oi"].code, "")

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
              msgs[-1]["content"] == "新问题" and msgs[-2]["content"] == "旧问题",
              f"{[m['role'] for m in msgs]}")
        check("提示词：策略清单夹在基础提示词和对话之间",
              msgs[0]["role"] == "system"
              and "当前平台上的策略" in msgs[1]["content"]
              and msgs[2]["content"] == "旧问题",
              f"{[m['role'] for m in msgs]}")
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
              any(m["role"] == "system" and "螺纹只做多" in m["content"]
                  for m in A.build_messages([], "问", memory="- 螺纹只做多")), "")
        check("记忆：没记忆时不加空 system 消息",
              not any("## 长期记忆" in m["content"] for m in A.build_messages([], "问")), "")

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


def check_agent_fs() -> None:
    """升级权限的文件读写：路径不能跑出根目录、写是原子的、换行跟着原文件走。"""
    from ttrade import agent as A
    from ttrade import agent_fs as FS

    tmpdir = tempfile.mkdtemp(prefix="td_fs_")
    old_root = FS.root
    FS.root = lambda: tmpdir
    try:
        check("文件权限：根目录可取", os.path.isdir(FS.root()), FS.root())

        # 路径闸门。这层挡不住蓄意破坏，但必须挡住 `../` 和绝对路径 ——
        # 否则「只能改 TrendDesk 目录」这句话就是假的
        ok_path, err = FS.resolve("ttrade/factors.py")
        check("文件权限：正常相对路径能落", bool(ok_path) and not err, err)
        for bad in ("../outside.txt", "a/../../b.txt", "/etc/passwd", "C:/Windows/win.ini"):
            _p, e = FS.resolve(bad)
            check(f"文件权限：拒绝 {bad}", bool(e), e)

        # 写 → 读
        ok, msg = FS.write_text("notes/a.txt", "第一行\n第二行")
        check("文件权限：写入成功", ok, msg)
        check("文件权限：目录自动建出来",
              os.path.isfile(os.path.join(tmpdir, "notes", "a.txt")), "")
        text, err = FS.read_text("notes/a.txt")
        check("文件权限：读回一致", text == "第一行\n第二行" and not err, err)
        check("文件权限：覆盖已有文件不报错", FS.write_text("notes/a.txt", "改了")[0], "")

        # 换行跟着原文件走：仓库是 CRLF 的，模型只给 \n，直接写会让整个文件变红
        with open(os.path.join(tmpdir, "crlf.py"), "wb") as fh:
            fh.write(b"a = 1\r\nb = 2\r\n")
        FS.write_text("crlf.py", "a = 9\nb = 2\n")
        with open(os.path.join(tmpdir, "crlf.py"), "rb") as fh:
            raw = fh.read()
        check("文件权限：保留原文件的 CRLF", raw == b"a = 9\r\nb = 2\r\n", repr(raw))

        # 二进制不读：parquet/图片塞进上下文只会烧 token
        with open(os.path.join(tmpdir, "x.bin"), "wb") as fh:
            fh.write(b"\x00\x01\x02")
        _t, e = FS.read_text("x.bin")
        check("文件权限：二进制拒绝读取", bool(e) and "二进制" in e, e)

        # 目录列表：跳过 .git 这类，且标明是不是目录
        os.makedirs(os.path.join(tmpdir, ".git"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "sub"), exist_ok=True)
        listing, e = FS.list_dir(".")
        check("文件权限：列目录跳过 .git",
              not e and "sub/" in listing and ".git" not in listing, e or listing[:60])

        # 大小上限
        big = "x" * (FS.MAX_WRITE + 1)
        ok, msg = FS.write_text("big.txt", big)
        check("文件权限：超长写入被拒", not ok and "太长" in msg, msg)

        # 动作解析
        r = A.parse_action('```json\n{"action":"fs","op":"read","path":"ttrade/agent.py"}\n```')
        check("动作：读文件能解析", r.ok and r.describe() == "读取 ttrade/agent.py", r.error)
        w = A.parse_action('```json\n{"action":"fs","op":"write","path":"a.txt",'
                           '"content":"hi"}\n```')
        check("动作：写文件能解析", w.ok and "写入 a.txt" in w.describe(), w.describe())
        check("动作：写入缺 content 要报错",
              not A.parse_action('```json\n{"action":"fs","op":"write","path":"a.txt"}\n```').ok, "")
        check("动作：op 不认识要报错",
              not A.parse_action('```json\n{"action":"fs","op":"rm","path":"a.txt"}\n```').ok, "")
        check("动作：path 带 .. 要报错",
              not A.parse_action('```json\n{"action":"fs","op":"read","path":"../a"}\n```').ok, "")
        check("动作：绝对路径要报错",
              not A.parse_action('```json\n{"action":"fs","op":"read","path":"C:/a"}\n```').ok, "")

        # 提示词：不开权限时不提这件事，开了才追加那段
        off = A.build_messages([], "问")[0]["content"]
        on = A.build_messages([], "问", allow_fs=True)[0]["content"]
        check("提示词：默认不提文件动作", '"action": "fs"' not in off, "")
        check("提示词：开权限后追加文件段",
              '"action": "fs"' in on and "升级权限" in on and len(on) > len(off), "")
        check("提示词：文件段里写了真实根目录",
              FS.root().replace("\\", "/") in on, FS.root())
        import json as _json
        import re as _re

        blocks = _re.findall(r"```json\s*(\{.*?\})\s*```", on, _re.S)
        broken = []
        for blk in blocks:
            try:
                _json.loads(blk)
            except ValueError:
                broken.append(blk[:30])
        check("提示词：文件段示例是合法 JSON", bool(blocks) and not broken, str(broken))
    except Exception as exc:  # noqa: BLE001
        check("文件读写权限", False, f"{type(exc).__name__}: {exc}")
    finally:
        FS.root = old_root
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def check_factor_manual() -> None:
    """平台手册：覆盖全部因子、代码不是空壳、原语有源码、doc 动作能解析。

    这份东西是给模型当教材的，错了比没有更糟 —— 模型会照着错的列名写因子，
    然后在另一种数据上整列 NaN，而且不报错。
    """
    from ttrade import agent as A
    from ttrade import factor_manual as M
    from ttrade import library_docs as L
    from ttrade.factors import FACTORS

    try:
        # 每个因子都要有条目，且代码块不是空的
        missing, no_code = [], []
        for k in FACTORS:
            sec = M.factor_section(k)
            if "###" not in sec:
                missing.append(k)
            elif "```python" not in sec:
                no_code.append(k)
        check("手册：覆盖全部因子", not missing, str(missing[:5]))
        check("手册：每个因子都有实现代码", not no_code, str(no_code[:5]))

        # 内置因子的源码必须是真源码，不能是「（未收录公式）」这种占位
        placeholders = [k for k in FACTORS
                        if "未收录" in (L.factor_doc(k).formula if L.factor_doc(k) else "")]
        check("手册：没有未收录公式的因子", not placeholders, str(placeholders[:5]))

        # 原语段必须有源码 —— 这是「代码展示不全」的正解：
        # `f_mom_20` 只有一行 `_ret(...)`，不给 `_ret` 的定义等于什么都没说
        prims = M.primitive_section()
        for name in ("_ret", "_safe_div", "_col", "_oi"):
            check(f"手册：原语 {name} 有源码", f"def {name}(" in prims, "")

        # 手册和沙箱必须一致。手册是模型唯一的教材：写着一个沙箱里没有的名字，
        # 模型照抄就是 NameError，而它只会把「注册失败」当成自己写错了
        from ttrade import userfactors as UF
        from ttrade import userstrategies as US

        ns = UF._namespace()
        missing = [n for n in M._FACTOR_PRIMITIVES if n not in ns]
        check("手册：因子原语都能在沙箱里用", not missing, str(missing))
        sns = US._namespace()
        missing_s = [n for n in M._STRATEGY_PRIMITIVES if n not in sns]
        check("手册：策略原语都能在沙箱里用", not missing_s, str(missing_s))
        # 反过来也查一遍：沙箱里多给的东西，手册里也得写着
        extra = [
            n for n in ns
            if n.startswith("_") and not n.startswith("__")
            and n not in M._FACTOR_PRIMITIVES
        ]
        check("手册：沙箱里没有手册之外的原语", not extra, str(extra))

        data = M.DATA_GUIDE
        # 列名两套这件事必须写清楚：这是最容易踩且不报错的坑
        check("手册：写明两套持仓量列名",
              "open_interest" in data and "open_oi" in data and "_oi(df)" in data, "")
        check("手册：写明复权口径", "qfq" in data and "adj_factor" in data, "")
        check("手册：写明衍生列来源", "data_futures_aux" in data, "")

        # topic 解析
        for t, kw in (("data", "行情数据"), ("primitives", "原语"),
                      ("contract", "契约"), ("factors", "索引"),
                      ("cat:趋势", "趋势因子"), ("factor:mom_20", "动量 20 日")):
            title, body = M.resolve(t)
            check(f"手册：topic {t} 解析", kw in (title + body), title)
        title, body = M.resolve("这不是个 topic")
        check("手册：不认识的 topic 回索引", "索引" in title or "不认识" in body, title)

        # 整份手册能拼出来，长度在合理区间（太短说明哪段没拼上）
        full = M.manual()
        check("手册：全文长度合理", 20000 < len(full) < 200000, f"{len(full)} 字符")
        check("手册：全文含全部章节",
              all(s in full for s in ("## 一、", "## 二、", "## 三、", "## 四、")), "")

        # 动作解析
        a = A.parse_action('```json\n{"action":"doc","topic":"factors"}\n```')
        check("动作：查手册能解析", a.ok and "factors" in a.describe(), a.error)
        check("动作：查手册缺 topic 要报错",
              not A.parse_action('```json\n{"action":"doc"}\n```').ok, "")
        # doc 是只读动作，不该被当成交易动作发给主窗口
        check("动作：doc 不是交易动作", a.kind == "doc", a.kind)

        # 提示词里得告诉模型有这么个东西，否则它不会去查
        prompt = A.build_messages([], "问")[0]["content"]
        check("提示词：提到平台手册", '"action": "doc"' in prompt, "")
        check("提示词：列了常用 topic",
              all(t in prompt for t in ("data", "primitives", "contract", "factors")), "")
        # 提示词里原来那句「df 含 open_oi/close_oi」是错的（日线叫 open_interest），
        # 模型照着写会在日线上全 NaN
        check("提示词：不再教错列名",
              "open/high/low/close/volume/open_oi" not in prompt, "")
    except Exception as exc:  # noqa: BLE001
        check("平台手册", False, f"{type(exc).__name__}: {exc}")


def check_agent_attachments() -> None:
    """附件 + 运行输出：文档能读、图片能缩、结果能喂回模型。

    这条链最容易出的问题是「附件加进去了，但发出去的是空消息」：图片编码失败
    会静默给一个空 b64，模型收到一个 data URI 里什么都没有的部件，接口直接 400，
    而用户只看到「出错了」。
    """
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ttrade import agent as A
    from ttrade import attachments as ATT

    try:
        tmpdir = tempfile.mkdtemp(prefix="td_att_")

        # 1) 图片：大图要缩，透明底不能变黑，b64 不能是空的
        png = os.path.join(tmpdir, "big.png")
        a = None
        try:
            from PIL import Image

            Image.new("RGBA", (3000, 1800), (10, 20, 30, 0)).save(png)
            a = ATT.load(png)
            check("附件：大图缩到 1600 以内", a.ok and "缩到" in a.note, a.note)
            check("附件：图片有 base64", len(a.b64) > 1000, f"{len(a.b64)} 字符")
            check("附件：图片按 jpeg 传", a.mime == "image/jpeg", a.mime)
        except ImportError:
            skip("附件：图片缩放", "没装 Pillow")

        # 2) 文本：utf-8 与 gbk 都要读得出来。中文 Windows 导出的 csv 十有八九
        #    是 gbk，按 utf-8 硬解会满屏问号，模型读到的数字是错的还不自知
        utf = os.path.join(tmpdir, "a.md")
        open(utf, "w", encoding="utf-8").write("# 需求\n用 20 日动量")
        b = ATT.load(utf)
        check("附件：utf-8 文本能读", b.ok and "20 日动量" in b.text, b.text[:20])

        gbk = os.path.join(tmpdir, "b.csv")
        open(gbk, "wb").write("日期,收盘\n2024-01-02,100\n".encode("gbk"))
        c = ATT.load(gbk)
        check("附件：gbk 文本能读", c.ok and "收盘" in c.text, c.text[:20])

        # 3) docx 自己解压读，不依赖 python-docx
        import zipfile

        docx = os.path.join(tmpdir, "n.docx")
        with zipfile.ZipFile(docx, "w") as z:
            z.writestr(
                "word/document.xml",
                "<w:document><w:body><w:p><w:r><w:t>第一段</w:t></w:r></w:p>"
                "<w:p><w:r><w:t>第二段</w:t></w:r></w:p></w:body></w:document>",
            )
        d = ATT.load(docx)
        check("附件：docx 抠出文字且分段", d.ok and d.text.splitlines() == ["第一段", "第二段"],
              repr(d.text))

        # 4) 认不出来的格式要**说清楚怎么办**，不能默默当成文本读
        bad = os.path.join(tmpdir, "x.xlsx")
        open(bad, "w").write("x")
        e = ATT.load(bad)
        check("附件：不支持的格式给出提示",
              not e.ok and "不支持" in e.note and "截图" in e.note, e.note)

        # 5) 拼消息：没图必须是纯字符串（老网关只吃字符串），有图才是 parts
        check("附件：纯文档拼成字符串",
              isinstance(ATT.content("看这个", [b, c]), str), "")
        parts = ATT.content("看这个", [a, b])
        check("附件：带图拼成 parts",
              isinstance(parts, list) and parts[0]["type"] == "text"
              and parts[-1]["type"] == "image_url", "")
        check("附件：图片走 data URI",
              parts[-1]["image_url"]["url"].startswith("data:image/jpeg;base64,"), "")

        # 6) 带图消息的 content 是列表，`len()` 数出来是部件数不是字数
        check("附件：text_of 摊平 parts",
              "看这个" in A.text_of(parts) and "一张图片" in A.text_of(parts),
              A.text_of(parts)[:30])

        # 7) 历史里只留最近两条带图消息的图 —— 四十条全留就是十几兆请求体
        hist = [{"role": "user", "content": ATT.content(f"第{i}张", [a])} for i in range(4)]
        kept = A._prune_images(hist, keep=2)
        imgs = sum(1 for m in kept
                   if any(p.get("type") == "image_url" for p in m["content"]))
        check("附件：老图不再重复传", imgs == 2, f"{imgs} 条还带着图")
        msgs = A.build_messages(hist, "还有吗")
        check("附件：build_messages 保留多模态",
              isinstance(msgs[-2]["content"], list), type(msgs[-2]["content"]).__name__)

        # 8) 提示词要教它怎么看附件、报错后先查手册
        prompt = A.SYSTEM_PROMPT
        check("提示词：说明附件怎么读", "附件" in prompt and "不要问「附件里是什么」" in prompt, "")
        check("提示词：报错后先查手册", "注册失败" in prompt and "先查手册" in prompt, "")

        # 9) 界面：输出框、附件按钮、回测结果自动喂回
        from PySide6.QtWidgets import QApplication

        from gui import agent_page as G

        app = QApplication.instance() or QApplication([])
        sent: list = []

        def fake_chat(self, messages, on_delta=None, should_stop=None):
            sent.append(messages)
            return "收到"

        old_chat = A.LLMClient.chat
        A.LLMClient.chat = fake_chat
        pg = None
        try:
            pg = G.AgentPage()
            pg.cfg.api_key = "k"
            pg.cfg.provider = "deepseek"
            pg.cfg.base_url = "https://api.deepseek.com/v1"
            pg.cfg.model = "deepseek-chat"
            pg.cfg.confirm_trade = False

            def pump(n: int = 60) -> None:
                for _ in range(n):
                    app.processEvents()
                    time.sleep(0.01)

            check("输出框：控件都在",
                  hasattr(pg, "out") and hasattr(pg, "btn_out_feed")
                  and hasattr(pg, "btn_attach"), "")

            pg.push_output("第一次 A")
            pg.push_output("第二次 B")
            check("输出框：逐条追加", pg.out.toPlainText().count("次 ") == 2,
                  pg.out.toPlainText()[:40])
            check("输出框：有内容才能发回", pg.btn_out_feed.isEnabled(), "")

            # 拖滑杆会连着来一串，只该留下最后一条
            pg._clear_output()
            pg.push_output("拖 1", True)
            pg.push_output("拖 2", True)
            check("输出框：连续调整覆盖上一条",
                  pg.out.toPlainText().count("拖") == 1, pg.out.toPlainText()[-12:])

            # 灌配置时「中间态 + 收尾」会各算一次，同一份结果不能记两条
            pg._clear_output()
            pg.push_output("同一份结果")
            pg.push_output("同一份结果")
            check("输出框：同一份结果不记两条",
                  pg.out.toPlainText().count("同一份结果") == 1,
                  pg.out.toPlainText())

            # 模型发起的回测：结果要自动喂回去，不然它只能问用户
            pg._clear_output()
            sent.clear()
            pg.begin_backtest_feedback()
            pg.push_output("年化 +12.3%")
            pump()
            fed = bool(sent) and "12.3" in str(sent[-1][-1]["content"])
            check("输出框：Agent 发起的回测自动回喂", fed, f"{len(sent)} 轮")
            check("输出框：喂过一次就清标记", pg._expect_bt is False, "")

            # 用户手动回测 / 拖参数不该触发一轮对话
            pg._clear_output()
            sent.clear()
            pump()
            pg.push_output("手动跑的结果")
            pump()
            check("输出框：手动回测不打扰模型", not sent, f"{len(sent)} 轮")

            # 注册失败：报错和契约必须一起回到模型手里。以前失败只写在界面上，
            # 模型以为自己成功了，下一轮接着按错的假设聊
            from ttrade import userfactors as UF

            tmp2 = tempfile.mkdtemp(prefix="td_uf2_")
            old_path = UF.path
            UF.path = lambda: os.path.join(tmp2, "uf.json")
            try:
                sent.clear()
                pg._register_factor({
                    "key": "zz_smoke_bad", "label": "坏", "func": "f_x",
                    "code": "def f_x(df):\n    return _col(df, 'close').reset_index(drop=True)",
                })
                pump()
                note = str(sent[-1][-1]["content"]) if sent else ""
                check("注册失败：报错回喂模型", "注册失败" in note, f"{len(sent)} 轮")
                check("注册失败：契约随报错一起给", "写因子的契约" in note, f"{len(note)} 字符")
            finally:
                UF.path = old_path

            # 附件条：加一个真文件，chip 要出现，点掉要消失
            pg._atts = [b]
            pg._refresh_atts()
            check("附件条：有附件才显示", not pg.att_box.isHidden(), "")
            pg._drop_att(0)
            check("附件条：删掉后隐藏", pg.att_box.isHidden() and not pg._atts, "")

            # 端到端：带图发一条，发出去的必须是 parts，发完附件条要清空
            sent.clear()
            pg._atts = [a] if (a is not None and a.ok) else []
            if pg._atts:
                pg.ed_input.setPlainText("照这张图写个因子")
                pg._send()
                pump()
                body = sent[-1][-1]["content"] if sent else ""
                check("附件：发送时走多模态",
                      isinstance(body, list) and body[-1]["type"] == "image_url",
                      type(body).__name__)
                check("附件：发完清空附件条", not pg._atts and pg.att_box.isHidden(), "")
                check("附件：正文带上文件名", "big.png" in A.text_of(body), "")
        finally:
            A.LLMClient.chat = old_chat
            if pg is not None:
                try:
                    pg.shutdown()
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        check("附件 / 输出框", False, f"{type(exc).__name__}: {exc}")


def check_chat_render() -> None:
    """模型回复的渲染 + 输入框粘贴。

    这条链以前是断的：`_on_delta` 用 `insertPlainText` 逐字追加，`_on_done`
    也不重排，所以 `**加粗**`、`| 表格 |`、三反引号代码块全是原文照贴。
    这里盯四件事：markdown 真的变成标签、流式那遍纯文本被抹干净、表格列宽
    按内容分、粘贴的三条路径（图 / 文件 / 网页表格）都接得住。
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from gui import chat_render as CR

    try:
        # 1) 解析：每种块都要认出来
        md = (
            "## 结论\n\n螺纹**不适合**追多，`mom_20` 掉到 0.31。\n\n"
            "- 第一条\n- 第二条\n\n"
            "| 参数 | 值 | 说明 |\n|---|---|---|\n| fast | 50 | 短均线 |\n| ic | -0.031 | 滚动 |\n\n"
            "> 注意方向\n\n```python\ndef f_x(df):\n    return 1\n```\n\n---\n\n"
            "[手册](https://example.com/m)\n"
        )
        kinds = [b["type"] for b in CR.parse(md)]
        check("渲染：块类型齐全",
              kinds == ["head", "text", "list", "table", "quote", "code", "hr", "text"],
              str(kinds))

        # 2) 表格：列数对得上，数字列右对齐
        tb = [b for b in CR.parse(md) if b["type"] == "table"][0]
        check("渲染：表格行数", len(tb["rows"]) == 2, str(tb["rows"]))
        check("渲染：数字列右对齐", tb["align"] == ["l", "r", "l"], str(tb["align"]))

        # 3) 渲染：markdown 记号不能留在正文里
        html = CR.render(CR.parse(md))
        plain = re.sub(r"<[^>]+>", "", html)
        check("渲染：没有残留 ** 记号", "**" not in plain, plain[:30])
        check("渲染：粗体成了标签", "font-weight:700" in html, "")
        check("渲染：表格是 <table>", "<table" in html, "")
        check("渲染：代码块是 <pre>", "<pre" in html, "")
        check("渲染：链接保留 href", 'href="https://example.com/m"' in html, "")
        check("渲染：转义了尖括号", "&lt;" in CR.inline("a < b") and "<b>" not in CR.inline("a < b"), "")

        # 4) 列宽：必须写在每个 td 上 —— 只写 th 的话 Qt 不认（踩过）
        cell = CR.block_table(["参数", "值"], [["fast", "50"], ["slow", "200"]])
        h = CR.render([cell])
        check("渲染：列宽写在 td 上", h.count('width="') >= 4, f"{h.count('width=')} 处")

        # 5) 色条：Qt 只认 td 的 border-left，所以必须拼成窄 td
        w = CR.render([CR.block_warning("风控", ["买 1 手"], "bad")])
        check("渲染：告警块用色条单元格",
              'bgcolor="#b91c1c"' in w and 'bgcolor="#fdeaea"' in w, "")
        check("渲染：告警块没用 border-left",
              "border-left" not in w, "")

        # 6) 指标卡片：好坏染色
        m = CR.render([CR.block_metrics("回测", [("收益", "+38.2%", "good"),
                                                ("回撤", "-21.4%", "bad")])])
        check("渲染：指标好坏染色", "#166534" in m and "#b91c1c" in m, "")

        # 7) 认不出的块不能整页空白
        check("渲染：未知块跳过不崩",
              CR.render([{"type": "什么鬼"}, CR.block_text("还在")]).count("还在") == 1, "")

        # 8) 剪贴板里的表格（从网页 / Excel / Word 框选复制）转 markdown
        from gui import agent_page as G

        src = ("<table><tr><th>因子</th><th>IC</th></tr>"
               "<tr><td>mom_20</td><td>0.031</td></tr></table>")
        md_tbl = G.html_table_to_md(src)
        check("粘贴：网页表格转 markdown",
              md_tbl.startswith("| 因子 | IC |") and "| mom_20 | 0.031 |" in md_tbl, md_tbl[:24])
        check("粘贴：普通 HTML 不抢", G.html_table_to_md("<p>一段话</p>") == "", "")

        # 9) 界面端到端：流式纯文本 → 收尾换富文本
        from PySide6.QtWidgets import QApplication

        from ttrade import attachments as ATT

        app = QApplication.instance() or QApplication([])
        pg = G.AgentPage()
        try:
            reply = ("## 结论\n\n螺纹**不适合追多**。\n\n"
                     "| 参数 | 值 |\n|---|---|\n| fast | 50 |\n")
            pg.log.clear()
            pg._stream_buf = ""
            pg._msg_start = None
            for i in range(0, len(reply), 7):
                pg._on_delta(reply[i:i + 7])
            check("流式：中途是纯文本（不重排，省 CPU）",
                  "**不适合追多**" in pg.log.toPlainText(), "")
            pg._on_done(reply)
            app.processEvents()
            out = pg.log.toHtml()
            text = pg.log.toPlainText()
            check("流式：收尾后没有 ** 记号", "**" not in text, text[:24])
            check("流式：收尾后有真表格", "<table" in out, "")
            check("流式：只渲染一份（流式文本被抹掉）", text.count("结论") == 1,
                  f"结论 x{text.count('结论')}")

            # 10) 追加时必须断开块：`insertHtml` 在段末会把片段首个块并进上一段，
            #     于是「你」「助手」会粘在上一句后面，`## 结论` 也缩成行内
            pg.log.clear()
            pg._append("<p>第一段</p>")
            pg._append("<p>第二段</p>")
            check("追加：两段各占一行",
                  pg.log.toPlainText().splitlines()[:2] == ["第一段", "第二段"],
                  repr(pg.log.toPlainText()[:20]))

            # 11) 出错也要留下半截 + 红块
            pg2 = G.AgentPage()
            pg2._stream_buf = ""
            pg2._msg_start = None
            pg2._on_delta("说到一半")
            pg2._on_failed("连接超时")
            app.processEvents()
            h2 = pg2.log.toHtml()
            check("流式：出错保留半截并标红",
                  "说到一半" in pg2.log.toPlainText() and "#fdeaea" in h2, "")
            pg2.shutdown()

            # 11) 输入框：粘贴截图 / 文件 / 网页表格三条路径
            check("输入框：是 ChatInput", isinstance(pg.ed_input, G.ChatInput), "")

            from PySide6.QtCore import QMimeData, QUrl
            from PySide6.QtGui import QColor, QImage

            mi = QMimeData()
            img = QImage(320, 200, QImage.Format_ARGB32)
            img.fill(QColor("#3366cc"))
            mi.setImageData(img)
            pg.ed_input.insertFromMimeData(mi)
            app.processEvents()
            check("输入框：粘贴截图进附件条",
                  len(pg._atts) == 1 and pg._atts[0].kind == "image",
                  f"{len(pg._atts)} 个")

            mf = QMimeData()
            here = os.path.abspath("requirements.txt")
            mf.setUrls([QUrl.fromLocalFile(here)])
            pg.ed_input.insertFromMimeData(mf)
            app.processEvents()
            check("输入框：粘贴文件进附件条",
                  len(pg._atts) == 2 and pg._atts[1].kind == "text",
                  f"{len(pg._atts)} 个")
            check("输入框：附件条显示缩略图",
                  not pg.att_box.isHidden() and pg.att_lay.count() >= 2, "")

            pg3 = G.AgentPage()
            mt = QMimeData()
            mt.setHtml(src)
            mt.setText("因子\tIC\nmom_20\t0.031")
            pg3.ed_input.insertFromMimeData(mt)
            check("输入框：粘贴网页表格转表格文本",
                  "| mom_20 | 0.031 |" in pg3.ed_input.toPlainText(), "")
            pg3.shutdown()

            # 12) QImage 直接变附件（不落盘）
            qa = ATT.from_qimage(img, "截图 1.png")
            check("附件：QImage 直接转", qa.ok and len(qa.b64) > 200, qa.note)
        finally:
            try:
                pg.shutdown()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        check("渲染 / 粘贴", False, f"{type(exc).__name__}: {exc}")


def check_portfolio_params() -> None:
    """组合页的设置参数、回测区间，以及因子策略的实时信号。"""
    import os

    import numpy as np
    import pandas as pd

    from ttrade import factor_strategy as fstrat
    from ttrade import replay
    from ttrade import strategy_store as St
    from ttrade.strategy_runner import PortfolioRunner
    from ttrade.tqlive import AccountSpec

    try:
        idx = pd.bdate_range("2020-01-01", periods=1200)
        df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                           "close": 100.0, "volume": 1.0}, index=idx)
        w = replay.window(df, 1.0)
        check("区间：近 1 年只切一年多的数据",
              len(w) < len(df) and w.index[-1] == df.index[-1]
              and len(w) <= 262 + replay.WARMUP_BARS + 2, f"{len(w)} 根")
        check("区间：前面留了预热段", len(w) > 253, f"{len(w)} 根")
        check("区间：全部历史原样返回", replay.window(df, 0) is df)
        check("区间：数据比窗口短时原样返回",
              len(replay.window(df.iloc[:100], 10)) == 100)

        c = St.StrategyConfig(symbol="RB", exchange="SHFE", rule_key="ma_cross",
                              range_years=3.0)
        c2 = St.StrategyConfig.from_dict(c.to_dict())
        check("区间：配置能存能读", abs(c2.range_years - 3.0) < 1e-9, str(c2.range_years))
        # 区间只影响看多少年，不改变交易行为，不该被当成「另一套配置」
        other = St.StrategyConfig.from_dict({**c.to_dict(), "range_years": 0})
        check("区间：不影响重复判定", c.same_shape(other) is True)

        # 因子策略的实时信号必须和研究页同一套算法：两边各写一份，
        # 「回测赚钱、实盘亏钱」就永远说不清是策略不行还是实现不一致
        rng = np.random.default_rng(3)
        n = 800
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        live = pd.DataFrame(
            {"open": px, "high": px * 1.01, "low": px * 0.99,
             "close": px, "volume": 1.0},
            index=pd.bdate_range("2022-01-03", periods=n),
        )
        fcfg = {"window": 60, "horizon": 5, "entry_z": 1.0, "exit_z": 0.3,
                "orient": "long_high"}
        cfgf = St.StrategyConfig(sid="f1", symbol="RB", exchange="SHFE", period="1h",
                                 family="factor", factor_mode="single",
                                 factor_keys=["mom_20"], factor_cfg=fcfg)
        r = PortfolioRunner(AccountSpec(name="t", kind="sim"), St.ChaseConfig())
        stf = r.add(cfgf)
        got = r._signal(stf, live)
        out = fstrat.run_factor_strategy(live, "single", ["mom_20"],
                                         replay.factor_config(fcfg))
        want = out.df["signal"].iloc[-1]
        check("因子实时：信号与回测同一套算法",
              got == (0 if pd.isna(want) else int(want)), f"{got} vs {want}")

        r.enable(cfgf.sid, True)
        check("因子实时：单因子策略可以启动",
              stf.running and not stf.error, stf.error)

        cfgml = cfgf.clone(factor_mode="ml")
        stml = r.add(cfgml)
        r.enable(cfgml.sid, True)
        check("因子实时：机器学习被拒绝并说明原因",
              not stml.running and "机器学习" in stml.error, stml.error)

        # 历史不够时不抛异常，先空仓等数据
        check("因子实时：历史不够时给 0 而不是崩",
              r._signal(stf, live.iloc[:100]) == 0)
    except Exception as exc:  # noqa: BLE001
        check("设置参数 / 回测区间 / 因子实时", False, f"{type(exc).__name__}: {exc}")
        return

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication

        from gui.portfolio_page import EditDialog

        app = QApplication.instance() or QApplication([])   # noqa: F841
        cfg = St.StrategyConfig(symbol="RB", exchange="SHFE", rule_key="ma_cross",
                                rule_params={"fast": 24, "slow": 101},
                                range_years=2.0)
        dlg = EditDialog(cfg, ["默认"])
        check("设置参数：按策略定义生成控件",
              set(dlg.sp_params) == {"fast", "slow"}, str(sorted(dlg.sp_params)))
        check("设置参数：回填当前参数", dlg.sp_params["fast"].value() == 24)
        check("设置参数：回填回测区间", dlg.cb_range.currentData() == 2.0)
        dlg.sp_params["fast"].setValue(30)
        dlg.cb_range.setCurrentIndex(3)          # 近 5 年
        dlg.apply_to(cfg)
        check("设置参数：改动写回配置",
              cfg.rule_params == {"fast": 30, "slow": 101} and cfg.range_years == 3.0,
              f"{cfg.rule_params} / {cfg.range_years}")
        dlg.deleteLater()
    except Exception as exc:  # noqa: BLE001
        check("设置参数对话框", False, f"{type(exc).__name__}: {exc}")


def check_result_view() -> None:
    """结果视图：阶段分析的口径、组合合成、离线回放。

    这些数字会同时出现在页面上（年度柱状 + 月度矩阵 + 绩效页），对不上就是
    用户眼里的 bug —— 尤其「12 个月连乘 ≠ 全年」，一眼就能看出来。
    """
    from ttrade import replay
    from ttrade import strategy_store as SStore

    def _intraday(days: pd.DatetimeIndex, px) -> pd.DataFrame:
        """把日线收盘摊成每天 8 根 30 分钟线，用来测混合周期的组合。

        价格在当天和次日收盘之间线性走 —— 不需要真实形态，只要「同一天有多个
        时间戳、且和日线策略的日期重合但 bar 数差几十倍」这个结构。
        """
        ts, p = [], []
        for i, d in enumerate(days):
            a = float(px[i])
            b = float(px[min(i + 1, len(px) - 1)])
            for k in range(8):
                ts.append(d + pd.Timedelta(hours=9 + k))
                p.append(a + (b - a) * k / 8.0)
        return pd.DataFrame(
            {"open": p, "high": [v * 1.002 for v in p],
             "low": [v * 0.998 for v in p], "close": p, "volume": 1000.0},
            index=pd.DatetimeIndex(ts),
        )
    from ttrade.backtest import (
        _to_daily,
        combine,
        combine_trades,
        month_matrix,
        stage_table,
    )

    try:
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2021-01-04", periods=700)
        # 造一段有涨有跌的行情：单调上涨的话月度矩阵全是正数，测不出符号错误
        drift = np.sin(np.arange(len(idx)) / 60.0) * 0.004 + 0.0004
        close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.008, len(idx))))
        df = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.001, len(idx))),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1000.0,
            },
            index=idx,
        )

        eq = pd.Series(close, index=idx)
        st = stage_table(eq, "YE")
        total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        prod = float((1 + st["ret"]).prod() - 1.0)
        check("阶段：年度收益连乘等于总收益", abs(prod - total) < 1e-9,
              f"{prod:.6f} vs {total:.6f}")

        mm = month_matrix(eq)
        check("阶段：月度矩阵按年成行",
              list(mm.index) == sorted({d.year for d in idx}), str(list(mm.index)))
        # 每一年的 12 个月连乘必须等于那一年的收益。段收益取「上段末→本段末」
        # 就是为了这条 —— 用段内第一根当基准会把月初的跳空吞掉
        ok_year = True
        for year in mm.index:
            row = mm.loc[year].dropna()
            got = float((1 + row).prod() - 1.0)
            want = float(st.loc[st["period"].dt.year == year, "ret"].iloc[0])
            if abs(got - want) > 1e-9:
                ok_year = False
                break
        check("阶段：每年 12 个月连乘等于年度收益", ok_year)

        # 回放：规则策略能出结果，且带基准；不带基准时 bh 必须是 None
        c1 = SStore.StrategyConfig(symbol="TEST", period="1d", rule_key="ma_cross",
                                   rule_params={"fast": 5, "slow": 20})
        c2 = SStore.StrategyConfig(symbol="TEST", period="1d", rule_key="donchian",
                                   rule_params={"entry": 20, "exit": 10})
        r1 = replay.run(c1, df)
        r2 = replay.run(c2, df)
        check("回放：规则策略出结果", r1.ok and r1.res is not None, r1.error)
        check("回放：带买入持有基准", r1.bh is not None)
        check("回放：含 signal 列", r1.df is not None and "signal" in r1.df.columns)
        check("回放：可以不要基准",
              replay.run(c1, df, with_bh=False).bh is None)

        # 出错要落在 error 里，不能抛 —— 调用方是「双击一行看个图」
        bad = replay.run(SStore.StrategyConfig(symbol="TEST", rule_key="不存在的策略"), df)
        check("回放：未知策略不抛异常", (not bad.ok) and "没有这个策略" in bad.error, bad.error)
        check("回放：没数据也不抛", replay.run(c1, None).ok is False)

        # 配置里留着旧版本删掉的字段时不能 TypeError
        fcfg = replay.factor_config({"window": 120, "已经删掉的字段": 1})
        check("回放：旧配置的多余字段被忽略", fcfg.window == 120, str(fcfg.window))

        # 组合：曲线先各自压成「每个交易日一个点」再等权平均。按时间戳并集
        # 直接对齐的话，混合周期（30 分钟 + 日线）的组合每天有几十个点只有分钟
        # 级参与、一个点两条都参与，均值在两者之间来回跳 —— 画出来是一条上下
        # 翻飞的带子，看着像净值乱跳，其实是成分每天都在变
        combo = combine([r1.res, r2.res], label="组合")
        check("组合：曲线落在日频上",
              all(t.hour == 0 and t.minute == 0 for t in combo.equity.index))
        check("组合：一天一个点",
              combo.equity.index.is_unique
              and len(combo.equity) == len(combo.equity.index.normalize().unique()),
              f"{len(combo.equity)} 点")
        check("组合：首日净值等于初始资金",
              abs(float(combo.equity.iloc[0]) - 100_000) < 1e-6,
              f"{combo.equity.iloc[0]:.2f}")
        # 两条日线策略的日期完全重合，等权就是归一化后的均值
        norm = pd.concat([r1.res.equity / r1.res.equity.iloc[0],
                          r2.res.equity / r2.res.equity.iloc[0]], axis=1).mean(axis=1)
        check("组合：日线等权等于归一化均值",
              bool(np.allclose(combo.equity.values, (norm * 100_000).values)))
        check("组合：交易记录是两条之和",
              len(combo.trades) == len(r1.res.trades) + len(r2.res.trades),
              f"{len(combo.trades)} 笔")
        check("组合：指标齐全", "cagr" in combo.metrics and "max_dd" in combo.metrics)

        # 混合周期：30 分钟 + 日线。压完的点数必须等于「并集里的交易日数」，
        # 而不是「分钟级 bar 数」
        c3 = SStore.StrategyConfig(symbol="TEST", period="30m", rule_key="ma_cross",
                                   rule_params={"fast": 5, "slow": 20})
        r3 = replay.run(c3, _intraday(idx[:160], close[:160]), with_bh=False)
        check("混合：30 分钟回放能出结果", r3.ok, r3.error)
        check("混合：原始曲线确实是分钟级",
              any(t.hour != 0 for t in r3.res.equity.index))
        daily = _to_daily(r3.res.equity)
        check("混合：压成日频后一天一个点",
              len(daily) == len(daily.index.normalize().unique())
              and all(t.hour == 0 for t in daily.index),
              f"{len(daily)} 点")
        mixed = combine([r3.res, r1.res], label="组合")
        check("混合：组合曲线也落在日频上",
              all(t.hour == 0 and t.minute == 0 for t in mixed.equity.index))
        check("混合：点数不超过并集交易日数",
              len(mixed.equity) <= len(set(r1.res.equity.index.normalize())
                                      | set(r3.res.equity.index.normalize())),
              f"{len(mixed.equity)} 点")

        # 一条策略先没数据（r3 只到第 160 天）：它那一份钱还在，只是不动了。
        # 从均值里摘掉的话，组合从「两条平均」掉成「另一条自己」—— 末端凭空
        # 跳水一大截，而交易记录上那天还在赚
        end_day = r3.res.equity.dropna().index[-1].normalize()
        after = mixed.equity.index[mixed.equity.index > end_day]
        jump = abs(float(mixed.equity.loc[after[0]]
                         / mixed.equity.shift(1).loc[after[0]] - 1.0))
        check("混合：成分结束时曲线不跳", jump < 0.05, f"{jump:.1%}")
        a_end = float(r1.res.equity.iloc[-1] / r1.res.equity.iloc[0])
        b_end = float(daily.iloc[-1] / daily.iloc[0])
        check("混合：末值等于两条归一净值的平均",
              abs(float(mixed.equity.iloc[-1]) - (a_end + b_end) / 2 * 100_000) < 1e-6,
              f"{mixed.equity.iloc[-1]:.0f}")

        # 交易记录按开仓时间混排。分组拼接的话，表格是「A 从 2021 一路做到
        # 2023，然后 B 又从 2021 开始」—— 读起来像先后跑了两遍
        pairs = combine_trades([("日线", r1.res), ("30 分钟", r3.res)])
        check("混排：条数等于两条之和",
              len(pairs) == len(r1.res.trades) + len(r3.res.trades),
              f"{len(pairs)} 笔")
        check("混排：按开仓时间单调",
              all(pairs[i][1].entry_date <= pairs[i + 1][1].entry_date
                  for i in range(len(pairs) - 1)))
        names = [n for n, _t in pairs]
        check("混排：两个策略交替出现",
              "日线" in names and "30 分钟" in names
              and names.index("30 分钟") < len(names) - 1 - names[::-1].index("日线"),
              str(names[:8]))

        # 结论文本：组合没有基准，不能写出「基准 / 超额」两行
        text_one = replay.summary(r1, "TEST", "日线")
        text_combo = replay.summary(
            replay.Replay(res=combo, bh=None, period="1d"), "组合（2 条策略）", "等权")
        check("结论：单策略写基准", "基准" in text_one and "超额" in text_one)
        check("结论：组合不写基准", "基准" not in text_combo and "超额" not in text_combo,
              text_combo[:40])
        check("结论：组合带条数", "组合（2 条策略）" in text_combo)
        check("结论：日线持仓写「天」", "天" in text_one and "根（约" not in text_one)
    except Exception as exc:  # noqa: BLE001
        check("阶段 / 组合 / 回放", False, f"{type(exc).__name__}: {exc}")
        return

    # 页数由 kind 决定：单策略五页，组合三页（没有 K 线和因子 —— 跨品种加权
    # 没有单一对照物，也没有「这个组合的因子」这种说法）
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication

        from gui.result_view import ResultView
        from ttrade.plotstyle import setup_matplotlib_fonts

        # 不设字体的话标题里的中文全是方块，而且只在 warning 里出现一次，
        # 页面上看起来就是「图里没字」——主程序启动时做过同样的事
        setup_matplotlib_fonts()
        app = QApplication.instance() or QApplication([])
        one = ResultView("single")
        many = ResultView("portfolio")
        titles_one = [t for t, _ in one.pages()]
        titles_many = [t for t, _ in many.pages()]
        check("视图：单策略五页", len(titles_one) == 5, str(titles_one))
        check("视图：组合四页",
              titles_many == ["绩效", "截面因子分析", "交易记录", "阶段分析"],
              str(titles_many))
        check("视图：组合绩效表只有两列", many.table.columnCount() == 2,
              str(many.table.columnCount()))

        # 无基准路径必须真的能画出来：组合页的整条链都走这里，
        # 画不出来就是一片空白，而界面上不会报错
        many.show_result(None, None, combo, None, many.ctx)
        check("视图：组合无基准也能渲染", many.res is not None and len(many.res.equity) > 0)
        check("视图：组合交易表带策略列", many.tbl_trades.columnCount() == 11,
              str(many.tbl_trades.columnCount()))
        check("视图：单策略交易表不带策略列", one.tbl_trades.columnCount() == 10,
              str(one.tbl_trades.columnCount()))

        # K 线页：信号必须画在「下一根开盘」上。回测是 T 收盘出信号、T+1 开盘
        # 成交，画在信号那根的收盘价上会让人以为按收盘价成交，和回测对不上
        from gui.result_view import ViewCtx

        one.show_result(r1.df, r1.df, r1.res, r1.bh,
                        ViewCtx(symbol="TEST", period="1d", market="cn", label="MA"))
        sig = r1.df["signal"].to_numpy(int)
        t0 = r1.res.trades[0]
        pos = list(r1.df.index).index(t0.entry_date)
        check("K线：开仓落在信号后的那一根",
              sig[pos - 1] == 1 and sig[pos - 2] != 1,
              f"第 {pos} 根，前一根 signal={sig[pos - 1]}")
        check("K线：成交价是那一根的开盘价",
              abs(t0.entry_price / float(r1.df["open"].iloc[pos]) - 1) < 0.01,
              f"{t0.entry_price:.2f} vs {float(r1.df['open'].iloc[pos]):.2f}")

        # 十字光标：双击开启后，信息框要给出时间 + 开高低收，遇到信号根还要
        # 带上开平方向、成交价、手数
        one._k_cross = True
        one._show_cross(0)
        tip = one._bar_text(0)
        check("K线：十字光标给出开高低收",
              all(k in tip for k in ("开 ", "高 ", "低 ", "收 ")), tip.replace("\n", " / "))
        check("K线：十字光标可见", one._k_tip.get_visible())
        note = one._k_notes.get(t0.entry_date) or []
        check("K线：信号根带开平信息", bool(note) and "手" in note[0], str(note))
        one._hide_cross()
        check("K线：十字光标可关闭", not one._k_tip.get_visible())

        # 交易表不能截断。组合报告里三条策略的起点差七年，890 笔里最早的 390 笔
        # 曾经被「只显示最近 500 笔」砍掉，表格第一行落在 2023 年，而资金曲线是
        # 从 2016 年开始的 —— 看起来就像早年的交易丢了
        from ttrade.backtest import Trade as _T

        many.trades = [
            ("策略", _T(
                entry_date=pd.Timestamp("2016-01-04") + pd.Timedelta(days=i),
                exit_date=pd.Timestamp("2016-01-05") + pd.Timedelta(days=i),
                entry_price=100.0, exit_price=101.0, ret=0.01, bars=1,
                lots=3,
            ))
            for i in reversed(range(620))
        ]
        many._refresh_trades()
        check("视图：交易表全部列出（不截断到 500 笔）",
              many.tbl_trades.rowCount() == 620, str(many.tbl_trades.rowCount()))
        check("视图：交易表按开仓时间重排",
              many.tbl_trades.item(0, 3).text() < many.tbl_trades.item(619, 3).text(),
              f"{many.tbl_trades.item(0, 3).text()} → {many.tbl_trades.item(619, 3).text()}")
        # 手数列：回测按手数下单之后，逐笔明细要能对着看「这笔下了几手」
        check("视图：交易表有手数列", many.tbl_trades.horizontalHeaderItem(7).text() == "手数",
              many.tbl_trades.horizontalHeaderItem(7).text())
        check("视图：手数按笔显示", many.tbl_trades.item(0, 7).text() == "3",
              many.tbl_trades.item(0, 7).text())
        many.clear("清空")
        check("视图：清空后没有残留结果", many.res is None)
        app.processEvents()
    except Exception as exc:  # noqa: BLE001
        check("结果视图", False, f"{type(exc).__name__}: {exc}")


def check_day_limit() -> None:
    """日内报单 / 撤单额度。

    交易所对每个账户的日内报单笔数、撤单笔数都有硬上限，超了轻则当天限制开仓、
    重则被列入重点监控。这里守三件事：到额度就停手、跨交易日归零、重启不清零。
    """
    from datetime import datetime

    from ttrade import daylimit as DL

    try:
        check("额度：周二白天算周二",
              DL.trading_day(datetime(2026, 10, 27, 10, 0)) == "2026-10-27",
              DL.trading_day(datetime(2026, 10, 27, 10, 0)))
        check("额度：夜盘归下一交易日",
              DL.trading_day(datetime(2026, 10, 27, 21, 30)) == "2026-10-28",
              DL.trading_day(datetime(2026, 10, 27, 21, 30)))
        check("额度：周五夜盘归下周一",
              DL.trading_day(datetime(2026, 10, 23, 21, 30)) == "2026-10-26",
              DL.trading_day(datetime(2026, 10, 23, 21, 30)))
        check("额度：周六凌晨仍算周一",
              DL.trading_day(datetime(2026, 10, 24, 1, 0)) == "2026-10-26",
              DL.trading_day(datetime(2026, 10, 24, 1, 0)))
        check("额度：周一凌晨算周一",
              DL.trading_day(datetime(2026, 10, 26, 1, 0)) == "2026-10-26",
              DL.trading_day(datetime(2026, 10, 26, 1, 0)))
    except Exception as exc:  # noqa: BLE001
        check("额度：交易日映射", False, f"{type(exc).__name__}: {exc}")

    tmpdir = tempfile.mkdtemp(prefix="tqlimit_")
    old_dir = DL.user_dir
    DL.user_dir = lambda: tmpdir
    try:
        check("额度：报单上限 1000", DL.DAY_ORDER_LIMIT == 1000, str(DL.DAY_ORDER_LIMIT))
        lim = DL.DayLimit("测试账户")
        for _ in range(DL.DAY_ORDER_LIMIT):
            lim.order()
        check("额度：报单满额度后拒绝", lim.order() is False, str(lim.order_left()))
        check("额度：剩余次数归零", lim.order_left() == 0, str(lim.order_left()))

        # 重启程序不能清零 —— 交易所记的是账户当天已经发出去多少笔
        again = DL.DayLimit("测试账户")
        check("额度：重启后计数还在", again.order_left() == 0, str(again.order_left()))
        check("额度：换账户各算各的",
              DL.DayLimit("另一个").order_left() == DL.DAY_ORDER_LIMIT)

        # 撤单和报单是两条线，互不占额度
        other = DL.DayLimit("撤单测试")
        for _ in range(DL.DAY_CANCEL_LIMIT):
            other.cancel()
        check("额度：撤单满额度后拒绝", other.cancel() is False)
        check("额度：撤单用满不影响报单", other.order() is True)

        # 跨交易日归零
        path = os.path.join(tmpdir, DL.LIMIT_FILE)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["测试账户"]["day"] = "2000-01-01"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        check("额度：跨交易日归零",
              DL.DayLimit("测试账户").order_left() == DL.DAY_ORDER_LIMIT,
              str(DL.DayLimit("测试账户").order_left()))

        # 组合引擎：额度用尽既不能发单，也要停机，不能一直重试
        from ttrade import strategy_store as St
        from ttrade.strategy_runner import PortfolioRunner
        from ttrade.tqlive import AccountSpec

        class _Quote:
            price_tick = 1.0
            ask_price1 = 3050.0
            bid_price1 = 3049.0
            last_price = 3049.5

        class _Api:
            def __init__(self):
                self.n = 0

            def get_quote(self, _sym):
                return _Quote()

            def insert_order(self, **kw):
                self.n += 1
                return type("O", (), {"order_id": f"o{self.n}"})()

        # 计数按 `spec.label()` 走（"名字 · 类型"），不是光看名字
        spec = AccountSpec(name="测试账户", kind="kq")
        full = DL.DayLimit(spec.label())
        for _ in range(DL.DAY_ORDER_LIMIT):
            full.order()
        runner = PortfolioRunner(spec, St.ChaseConfig())
        runner.api = _Api()
        cfg = St.StrategyConfig(sid="s-limit", symbol="RB", exchange="SHFE",
                                period="1d", rule_key="ma_cross", volume=1)
        slot = runner.add(cfg)
        slot.running = True
        slot.underlying = "SHFE.rb2701"
        runner._pos = {}
        runner._send(slot, "SHFE.rb2701", 1)
        check("额度：用尽后不发单", runner.api.n == 0, f"{runner.api.n} 笔")
        check("额度：用尽后停机", slot.running is False and "上限" in slot.error,
              slot.error)
        check("额度：快照带上今日用量",
              "今日报单" in runner.snapshot()["limits"],
              runner.snapshot()["limits"])

        # 账户页的手动下单走另一条链路，也要拦
        from ttrade import tqlive as L

        hub = L.LiveHub(spec)
        hub.api = object()
        try:
            hub.place("SHFE.rb2701", "BUY", "OPEN", 1, 3000.0)
            check("额度：手动下单也会被拦", False, "没有抛异常")
        except RuntimeError as exc:
            check("额度：手动下单也会被拦", "上限" in str(exc), str(exc))
    except Exception as exc:  # noqa: BLE001
        check("额度：整体", False, f"{type(exc).__name__}: {exc}")
    finally:
        DL.user_dir = old_dir
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def check_topup() -> None:
    """策略启动时把本地历史补到最新。

    引擎手里的 K 线本来就是实时的，但本地库停在几天前 —— 组合页的 K 线图、
    回测都拿旧数据。这里守四件事：只落源周期、只写库里没有的、正在走的那根
    不写、同一品种只补一次。
    """
    from ttrade import strategy_store as St
    from ttrade import topup
    from ttrade import tqstore
    from ttrade.strategy_runner import PortfolioRunner
    from ttrade.tqlive import AccountSpec

    def mk(ts: list, px: list) -> pd.DataFrame:
        """按天勤的返回格式造一份序列：UTC 纳秒 + OHLCV。"""
        ns = [int((t - pd.Timedelta(hours=8)).value) for t in ts]
        return pd.DataFrame({
            "datetime": ns,
            "open": px, "high": [p + 1 for p in px],
            "low": [p - 1 for p in px], "close": [p + 0.5 for p in px],
            "volume": [100.0] * len(px),
            "open_oi": [0.0] * len(px), "close_oi": [0.0] * len(px),
        })

    tmpdir = tempfile.mkdtemp(prefix="tqtopup_")
    old_env = os.environ.get("TRENDDESK_TQ_DB")
    os.environ["TRENDDESK_TQ_DB"] = os.path.join(tmpdir, "t.duckdb")
    real_save = topup.save
    try:
        now = pd.Timestamp.now().floor("min")
        past = [now - pd.Timedelta(minutes=5 * (3 - i)) for i in range(3)]
        live = now + pd.Timedelta(minutes=5)      # 还在走的那根
        ser = mk(past + [live], [100.0, 101.0, 102.0, 103.0])

        r = topup.save(ser, "RB", "SHFE", "5m")
        check("补历史：写进源周期", r["added"] == 3 and r["src"] == "5m", str(r))
        check("补历史：库里确实多了 3 根",
              tqstore.bar_count("RB", "SHFE", "5m") == 3,
              str(tqstore.bar_count("RB", "SHFE", "5m")))
        # 活 bar 写进去就是一根永久停在半路的「假收盘」，回测拿它算信号
        check("补历史：正在走的那根不写",
              tqstore.last_dt("RB", "SHFE", "5m") == past[-1],
              str(tqstore.last_dt("RB", "SHFE", "5m")))

        r2 = topup.save(ser, "RB", "SHFE", "5m")
        check("补历史：重叠的不重写", r2["added"] == 0 and r2["ok"], str(r2))

        # 15m / 30m 在库里就是 5m 合并出来的，落盘必须落到 5m
        r3 = topup.save(ser, "RB", "SHFE", "15m")
        check("补历史：15m 落到 5m 源周期", r3["src"] == "5m", str(r3))
        check("补历史：日线不走这条链路",
              topup.save(ser, "RB", "SHFE", "1d")["error"] != "", "1d 应当被拒")
        check("补历史：序列还没填充时不置 ok",
              topup.save(pd.DataFrame(), "RB", "SHFE", "5m")["ok"] is False)

        # 待换月：切换表里已经有「将来某天换月」，实时补进来的这一段就该乘上那次
        # 换月的比率。写死 1.0 会让复权价在换月处留一个假跳空（EC 实测 18%），
        # 而且要到下一次全量下载才自愈 —— 那之前所有回测和信号都是错的。
        with tqstore.connect() as con:
            tqstore.upsert_switch(con, "ZN", "SHFE", pd.DataFrame([{
                "switch_date": (now + pd.Timedelta(days=30)).normalize(),
                "old_tq": "SHFE.zn2612", "new_tq": "SHFE.zn2703",
                "old_ts": "ZN2612.SHFE", "new_ts": "ZN2703.SHFE", "ratio": 1.25,
            }]))
        topup.save(ser, "ZN", "SHFE", "5m")
        with tqstore.connect() as con:
            zf = con.execute(
                "SELECT adj_factor FROM bars WHERE symbol='ZN' ORDER BY dt"
            ).df()["adj_factor"].to_numpy(float)
        check("补历史：待换月时因子取换月比率",
              len(zf) == 3 and bool(np.allclose(zf, 1.25)), str(zf))

        # 同一个品种被几十条策略共享时只能补一次，每次都写库是自找锁冲突
        calls: list = []

        def fake(*a, **k):
            calls.append(a)
            return {"added": 0, "src": "5m", "last": None, "error": "", "ok": True}

        topup.save = fake
        runner = PortfolioRunner(AccountSpec(name="补历史测试", kind="sim"),
                                 St.ChaseConfig())
        runner.api = object()
        st = runner.add(St.StrategyConfig(sid="t1", symbol="RB", exchange="SHFE",
                                          period="5m", rule_key="ma_cross",
                                          volume=1))
        runner._klines[runner._skey(st.cfg)] = ser
        runner._top_up(st)
        check("补历史：第一次就写一次", len(calls) == 1, f"{len(calls)} 次")
        # tick() 一秒十几次。指望「下一 tick 再看一眼」会让 3000 行每 tick
        # 转一遍 DataFrame —— 没成功要隔 TOP_RETRY 秒再试
        runner._top_up(st)
        check("补历史：2 秒内不重复尝试", len(calls) == 1, f"{len(calls)} 次")
        runner._top_try.clear()      # 假装过了 2 秒
        runner._top_up(st)
        check("补历史：同一品种只补一次", len(calls) == 1, f"{len(calls)} 次")

        # 写库失败多半是 DuckDB 被别的连接锁着（下载器在跑、主线程正在读），
        # 那是临时的。一次失败就永久放弃，会变成「启动那一刻刚好在读数据，
        # 之后就再也不补了」
        topup.save = lambda *a, **k: {"added": 0, "src": "5m", "last": None,
                                      "error": "写库失败：Conflicting lock",
                                      "ok": False}
        runner2 = PortfolioRunner(AccountSpec(name="补历史测试3", kind="sim"),
                                  St.ChaseConfig())
        runner2.api = object()
        st3 = runner2.add(St.StrategyConfig(sid="t3", symbol="RB", exchange="SHFE",
                                            period="5m", rule_key="ma_cross",
                                            volume=1))
        runner2._klines[runner2._skey(st3.cfg)] = ser
        runner2._top_up(st3)
        check("补历史：写库失败不列进已补", not runner2._topped, str(runner2._topped))
        runner2._top_try.clear()
        logs: list = []
        runner2._log_fn = logs.append
        runner2._top_up(st3)
        check("补历史：同样的失败不重复刷日志", logs == [], str(logs))

        # 15m 策略落库也落 5m —— 库里没有 15m 这张表。`_subscribe` 订的就已经
        # 是 5m 了，所以这里必须**复用**那条序列，不能为补库再订一份（重复订阅）
        asked: list = []

        class Api:
            def get_kline_serial(self, tq, sec, data_length=0):
                asked.append((tq, sec))
                return ser

        runner2 = PortfolioRunner(AccountSpec(name="补历史测试2", kind="sim"),
                                  St.ChaseConfig())
        runner2.api = Api()
        st2 = runner2.add(St.StrategyConfig(sid="t2", symbol="RB", exchange="SHFE",
                                            period="15m", rule_key="ma_cross",
                                            volume=1))
        runner2._klines[runner2._skey(st2.cfg)] = ser
        got: list = []

        def keep(serial, *a, **k):
            got.append(serial)
            return {"added": 0, "src": "5m", "last": None, "error": "", "ok": True}

        topup.save = keep
        runner2._top_up(st2)
        check("补历史：15m 策略复用已订的 5m 序列，不再额外订阅",
              asked == [] and len(got) == 1 and got[0] is ser,
              f"额外订阅 {asked} / 落库 {len(got)} 次")

        # 同一个品种的 5m 和 1h 是两条序列。共用一个订阅槽的话，后订阅的
        # 策略直接拿到前一条 —— 1h 策略按 5m 信号下单，补库还会把 5m 的 bar
        # 写进 1h 表，图上最后一截变成五分钟一根
        class SubApi:
            def __init__(self) -> None:
                self.secs: list = []

            def get_kline_serial(self, tq, sec, data_length=0):
                self.secs.append(sec)
                return ser

            def get_quote(self, tq):
                return None

        r3 = PortfolioRunner(AccountSpec(name="订阅测试", kind="sim"),
                             St.ChaseConfig())
        r3.api = SubApi()
        a = r3.add(St.StrategyConfig(sid="a", symbol="AG", exchange="SHFE",
                                     period="5m", rule_key="ma_cross", volume=1))
        b = r3.add(St.StrategyConfig(sid="b", symbol="AG", exchange="SHFE",
                                     period="1h", rule_key="ma_cross", volume=1))
        r3._subscribe(a)
        r3._subscribe(b)
        check("订阅：同品种不同周期各订各的",
              sorted(r3.api.secs) == [300, 3600], str(r3.api.secs))

        # 各自拿到的也必须是自己的那条
        check("订阅：各自读到自己的序列",
              r3._klines.get(r3._skey(a.cfg)) is ser
              and r3._klines.get(r3._skey(b.cfg)) is ser
              and len(r3._klines) == 2, str(list(r3._klines)))
    except Exception as exc:  # noqa: BLE001
        check("补历史", False, f"{type(exc).__name__}: {exc}")
    finally:
        topup.save = real_save
        if old_env is None:
            os.environ.pop("TRENDDESK_TQ_DB", None)
        else:
            os.environ["TRENDDESK_TQ_DB"] = old_env
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def check_ranges() -> None:
    """下载区间的档位：Tushare 与天勤各有一套，键不能撞车。"""
    from ttrade import updater as U

    keys_ = [k for k, _l, _d in U.RANGES]
    check("区间：Tushare 有「近 1 天 / 近 1 周」", "1d" in keys_ and "1w" in keys_, str(keys_))
    check("区间：Tushare 有 3 / 5 / 10 年",
          all(k in keys_ for k in ("3y", "5y", "10y")), str(keys_))
    check("区间：天数随档位递增",
          [d for _k, _l, d in U.RANGES][:-1] == sorted([d for _k, _l, d in U.RANGES][:-1]),
          str([d for _k, _l, d in U.RANGES]))

    from gui.data_page import TQ_RANGES

    days = [d for d, _l in TQ_RANGES]
    check("区间：天勤六档且递增", days == sorted(days) and len(days) == 6, str(days))
    check("区间：天勤「近 20 年」是最后一档", days[-1] >= 7000, str(days[-1]))

    from ttrade import tqfetch as F

    check("区间：天数优先于年数",
          F.window_start(10, 7) == F.window_start(0, 7)
          and F.window_start(1, 0) > F.window_start(2, 0),
          "")


def check_tq_curve() -> None:
    """天勤真实合约 → 期限结构 → aux 读取的回退链。"""
    from ttrade import auxdata as A
    from ttrade import tqcurve as Q
    from ttrade import tqstore as S

    if not S.exists():
        skip("天勤期限结构", "没有 DuckDB")
        return
    with S.connect(read_only=True) as con:
        # 轻量库可能连 contract_bars 都没有（make_lite_db.py --no-contracts）
        if not S._has_table(con, "contract_bars"):
            skip("天勤期限结构", "这个库里没有真实合约")
            return
        row = con.execute(
            "SELECT symbol, exchange, period, count(*) n FROM contract_bars"
            " GROUP BY 1,2,3 ORDER BY n DESC LIMIT 1"
        ).fetchone()
    if not row:
        skip("天勤期限结构", "库里没有真实合约（--contracts 没跑过）")
        return
    sym, ex, period, _n = row

    df, note = Q.build(sym, ex, period)
    check(f"期限结构：{sym}_{ex} {period} 能算出来", not df.empty, note)
    check("期限结构：说明里写明实际用的源周期", "源 " in note, note)
    if df.empty:
        return
    check("期限结构：aux 要读的列一个不少",
          all(c in df.columns for c in A.CURVE_COLS),
          f"{len(df.columns)} 列，缺 {[c for c in A.CURVE_COLS if c not in df.columns]}")
    check("期限结构：每天至少两腿",
          df["c1_close"].notna().all() and df["c2_close"].notna().any(), "")
    # 近月比次近月先到期
    both = df.dropna(subset=["c1_dte", "c2_dte"])
    check("期限结构：c1 比 c2 先到期",
          bool((both["c1_dte"] <= both["c2_dte"]).all()), "")

    # 一个品种一个文件，按 1d → 1h → 5m 挑精度最高的那份：库里只要有日线真实
    # 合约，就该用日线算，而不是被调用方传进来的 5m 带跑
    with S.connect(read_only=True) as con:
        has_1d = bool(con.execute(
            "SELECT count(*) FROM contract_bars WHERE symbol=? AND exchange=?"
            " AND period='1d'", [sym, ex]).fetchone()[0])
    if has_1d:
        _d, note_1d = Q.build(sym, ex, "5m")
        check("期限结构：有日线时优先用日线（哪怕传的是 5m）", "源 1d" in note_1d, note_1d)

    # 尾部要跟得上行情：只查 expired=True 时在交易的合约一个都下不到，曲线会
    # 停在最后一个摘牌日（2026-10-29 实测停在了 09-10，之后六周展期因子全空）
    with S.connect(read_only=True) as con:
        bmax = con.execute(
            "SELECT max(trading_day) FROM bars WHERE symbol=? AND exchange=? AND period='1d'",
            [sym, ex],
        ).fetchone()[0]
    if bmax is not None:
        gap = (pd.Timestamp(bmax) - pd.Timestamp(df["date"].max())).days
        check("期限结构尾部跟得上行情", gap <= 10, f"差 {gap} 天（行情 {bmax}）")

    got = Q.build_all([(sym, ex)], period=period)
    path = Q.curve_path(sym, ex)
    check("期限结构：落盘到 data_tq/curve", os.path.exists(path) and got[f"{sym}_{ex}"] > 0,
          path)

    # aux 目录里没有这个品种的文件 → 必须回退到天勤那份
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        back = A.load_curve(tmp, sym, ex)
        check("期限结构：aux 缺文件时回退到天勤那份", not back.empty and "c1_close" in back.columns,
              f"{len(back)} 行")


def check_new_factors() -> None:
    """日内 / 时段 / 季节性三类新因子：口径、留空规则、PIT。"""
    from datetime import date, timedelta

    import pandas as pd

    from ttrade import factors as F
    from ttrade import tqdata

    # 只钉**内置**的数量。用户自定义因子（Agent 生成的）会挂在同一个 FACTORS 里，
    # 拿总数去断言的话，用户每加一个因子这条就红一次 —— 那不是代码坏了。
    n_builtin = len(F.FACTORS) - F.USER_FACTOR_COUNT
    check(f"因子库：内置 {n_builtin} 个 / 13 类", n_builtin == 66 and len(F.CATEGORIES) == 13,
          f"{n_builtin} 个内置 + {F.USER_FACTOR_COUNT} 个自定义 / {len(F.CATEGORIES)} 类")
    cats = {k: F.FACTORS[k].category for k in
            ("vwap_gap", "trend_window", "season_avg", "season_cal", "roll_intraday")}
    check("新因子归到了对的类别",
          cats == {"vwap_gap": "日内", "trend_window": "时段", "season_avg": "季节性",
                   "season_cal": "品种季节", "roll_intraday": "展期"}, str(cats))

    daily = tqdata.load_main("RB", "SHFE", "1d_tq")
    intra = ("vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap",
             "close_drive", "trend_window", "quiet_window")
    check("日线上日内/时段因子整列为空",
          all(F.raw(daily, k).isna().all() for k in intra),
          f"{len(intra)} 个")

    start = date.today() - timedelta(days=45)
    m5 = tqdata.load_main("RB", "SHFE", "5m", start=start)
    # 数「覆盖了几天」而不是「有多少根非空」：窗口类因子走完才给值，一天里只有
    # 收盘前那几根有数（close_drive 一天两根），按根数卡会误判
    ndays = m5.index.normalize().nunique()
    cover = [F.raw(m5, k).notna().groupby(m5.index.normalize()).any().sum() for k in intra]
    check("5m 上日内因子每个交易日都算得出来", min(cover) >= ndays - 1,
          f"最少 {min(cover)} / {ndays} 个交易日")
    h1 = tqdata.load_main("RB", "SHFE", "1h", start=start)
    check("30 分钟窗口在 1h 上留空（切不出这么细）",
          F.raw(h1, "trend_window").isna().all() and F.raw(h1, "quiet_window").isna().all(), "")

    # 季节性 PIT：2024 年 3 月的值必须只由 2023 年及以前的三月算出
    g = pd.to_datetime(daily["trading_day"])
    dc = daily["close"].groupby(g).last()
    r = dc.pct_change()
    idx = pd.DatetimeIndex(dc.index)
    tab = r.groupby([idx.year, idx.month]).sum().unstack()
    want = float(tab.loc[tab.index < 2024, 3].mean())
    got = F.raw(daily, "season_avg")
    sel = (g.dt.year == 2024) & (g.dt.month == 3)
    val = float(got[sel].iloc[0]) if sel.any() else float("nan")
    check("季节均值严格只用往年同月", abs(val - want) < 1e-9, f"{val:.6f} vs {want:.6f}")

    cal = F.raw(daily, "season_cal")
    mar = float(cal[(g.dt.month == 3)].iloc[0])
    aug = float(cal[(g.dt.month == 8)].iloc[0])
    check("品种季节：螺纹 3 月旺季 +1、8 月无窗口 0", mar == 1.0 and aug == 0.0,
          f"3 月 {mar:+.0f} / 8 月 {aug:+.0f}")
    m_daily = tqdata.load_main("M", "DCE", "1d_tq")
    gm = pd.to_datetime(m_daily["trading_day"])
    cm = F.raw(m_daily, "season_cal")
    sep = float(cm[gm.dt.month == 9].iloc[0])
    check("品种季节：豆粕 9 月收割压力 -1", sep == -1.0, f"{sep:+.0f}")

    # 表里没有的品种（尿素）必须整列为空，不能填 0 冒充「有值但没窗口」
    ur = F.raw(tqdata.load_main("UR", "CZCE", "1d_tq"), "season_cal")
    check("没有日历的品种整列为空", ur.isna().all(), f"非空 {int(ur.notna().sum())}")

    # ---- PIT：小周期不能用大周期同时刻的信息 ----
    # 2026-10-29 查出来的真 bug：窗口收益原来铺满全天，09:05 那根 bar 读到了
    # 10:15 的收盘价，日内因子 IC 被这条未来信息吹起来。验法是「截断数据看前值
    # 变不变」—— 截在 09:55，前 12 个值必须和完整数据算出来的一模一样。
    hh5 = m5.index.strftime("%H:%M")
    pos = [i for i, t in enumerate(hh5) if t == "09:55"]
    if pos:
        k = pos[-1] + 1
        moved = []
        for key in F.FACTORS:          # 全部 67 个，不只是那几个窗口因子
            a = F.raw(m5, key).iloc[max(0, k - 12):k].to_numpy(float)
            b = F.raw(m5.iloc[:k], key).iloc[-len(a):].to_numpy(float)
            bad = 0
            for x, y in zip(a, b):
                if np.isnan(x) and np.isnan(y):
                    continue
                if np.isnan(x) or np.isnan(y):
                    bad += 1
                elif abs(x - y) > 1e-9 * max(1.0, abs(x)):
                    bad += 1
            if bad:
                moved.append(f"{key}({bad})")
        check("PIT：截断在 09:55 后，窗口因子前值不变", not moved, f"变了：{moved}")
        v = F.raw(m5, "trend_window")[hh5 == "09:10"]
        check("PIT：09:10 拿不到 09:45-10:15 的收益",
              int(v.notna().sum()) == 0, f"非空 {int(v.notna().sum())} 个")

def main() -> int:
    ks = keys()
    if not ks:
        print("仓库里还没有切换表，先跑 download_tq.py --switch --all")
        return 1
    if len(sys.argv) > 1:
        ks = [k for k in ks if k == sys.argv[1]] or ks

    check_trading_day()
    check_ranges()
    check_tq_curve()
    check_new_factors()
    check_contract_incremental()
    check_live_helpers()
    check_live_worker()
    check_cli_entry()
    check_factor_refresh()
    check_main_backfill()
    check_strategy_store()
    check_costs()
    check_cross()
    check_strategy_runner()
    check_day_limit()
    check_topup()
    check_sizing()
    check_no_blowup()
    check_intraday_aux()
    check_library_docs()
    check_agent()
    check_agent_stream()
    check_agent_thread()
    check_user_factors()
    check_user_strategies()
    check_agent_memory()
    check_agent_fs()
    check_factor_manual()
    check_agent_attachments()
    check_chat_render()
    check_portfolio_params()
    check_result_view()
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
