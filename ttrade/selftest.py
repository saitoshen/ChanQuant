"""打包后的自检：不打开窗口，验证资源、数据、回测、绘图、Qt 插件是否都正常。

    ChanQuant.exe --selftest     # Windows
    ChanQuant.app/Contents/MacOS/ChanQuant --selftest   # macOS

打包成窗口程序后没有控制台，`print` 看不到，所以结果同时写入
`<用户目录>/logs/selftest.log`，并用退出码表示成败（0 成功 / 1 失败）。
CI 里跑一遍这个就能确认「打出来的包真的能用」，而不用去点界面。
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback


def _run() -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    _cache: dict = {}

    # 全程离屏，自检不应该在用户屏幕上闪一个窗口
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    def check(name: str, fn) -> None:
        nonlocal ok
        try:
            detail = fn()
            lines.append(f"[ OK ] {name}" + (f"  {detail}" if detail else ""))
        except Exception as exc:  # noqa: BLE001 - 自检要吞掉异常继续跑完
            ok = False
            lines.append(f"[FAIL] {name}  {type(exc).__name__}: {exc}")
            lines.append(traceback.format_exc())

    from ttrade.paths import bundle_root, cache_dir, data_dir, is_frozen, log_dir, user_dir

    lines.append(f"frozen={is_frozen()}  python={sys.version.split()[0]}  platform={sys.platform}")
    lines.append(f"bundle_root = {bundle_root()}")
    lines.append(f"data(us)    = {data_dir('us')}")
    lines.append(f"data(cn)    = {data_dir('cn')}")
    lines.append(f"user_dir    = {user_dir()}")
    lines.append(f"log_dir     = {log_dir()}")
    lines.append("")

    def _count(market: str) -> str:
        d = data_dir(market)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"目录不存在：{d}")
        n = len([f for f in os.listdir(d) if f.endswith(".csv")])
        if n == 0:
            raise FileNotFoundError(f"目录里没有 CSV：{d}")
        return f"{n} 个 CSV"

    check("美股数据目录", lambda: _count("us"))
    check("期货数据目录", lambda: _count("cn"))

    def _load_us():
        import pandas as pd

        from ttrade.eastmoney import load_local

        path = os.path.join(data_dir("us"), "AAPL.csv")
        sym = "AAPL" if os.path.exists(path) else None
        if sym is None:
            cands = [f[:-4].replace("_", ".") for f in os.listdir(data_dir("us"))
                     if f.endswith(".csv") and not f.startswith("_")]
            if not cands:
                raise FileNotFoundError("没有可用的美股 CSV")
            sym = sorted(cands)[0]
        df = load_local(sym, data_dir("us"))
        assert isinstance(df, pd.DataFrame) and len(df) > 100
        return f"{sym} {len(df)} 行 {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}"

    check("读取美股行情", _load_us)

    def _load_cn():
        from ttrade.futures import load_local as load_fut

        for sym, ex in (("RB", "SHFE"), ("CU", "SHFE"), ("M", "DCE")):
            if os.path.exists(os.path.join(data_dir("cn"), f"{sym}_{ex}.csv")):
                df = load_fut(sym, ex, data_dir("cn"))
                assert len(df) > 100
                return (f"{sym}_{ex} {len(df)} 行，最新 {df['close'].iloc[-1]:.2f}"
                        f"，复权因子列={'adj_factor' in df.columns}")
        raise FileNotFoundError("没有可用的期货 CSV")

    check("读取期货行情（含复权）", _load_cn)

    def _opt_free_path():
        """免费账户（无 tq_dl）的期权下载**不能被跳过**。

        回归点：数据管理页原来一看没有 `tq_dl` 就把期权整段跳过。实测免费账号
        对已摘牌合约照样能 `get_kline_serial`，所以要降级到逐合约路径。

        全程离线：假 api + 临时 DuckDB（`TRENDDESK_OPT_DB`），不碰真库。
        """
        import shutil
        import tempfile

        import pandas as pd

        from ttrade import optfetch as OF
        from ttrade import optstore as OS

        tmp = tempfile.mkdtemp(prefix="optfree_")
        old_env = os.environ.get("TRENDDESK_OPT_DB")
        old_csv = OF.export_csv
        os.environ["TRENDDESK_OPT_DB"] = os.path.join(tmp, "opt.duckdb")
        # export_csv 会往真实 data 目录写，自检里禁掉
        OF.export_csv = lambda *a, **k: ""

        und = "DCE.m2401"
        calls = ["DCE.m2401-C-3000", "DCE.m2401-P-3000"]
        # 到期日 = 2024-01-18：最后一根 01-17，正好落在「到期前 1 天」之内，
        # 第二次跑才会被判成「已是最新」
        expire_s = pd.Timestamp("2024-01-18 15:00:00", tz="Asia/Shanghai").value / 1e9

        def _kl(base: float, nan: bool = False) -> pd.DataFrame:
            # 天勤的 `datetime` 是 **UTC 纳秒整数**。别用
            # `DatetimeIndex.astype("int64")` —— 带时区的索引会按 pandas 的
            # 解析精度给成微秒，日期直接跑到 1970 年，结果被区间过滤成空表
            t0 = pd.Timestamp("2024-01-15 00:00:00", tz="Asia/Shanghai").value
            ns = [t0 + i * 86_400 * 10 ** 9 for i in range(3)]
            px = [float("nan")] * 3 if nan else [base, base + 1, base + 2]
            return pd.DataFrame({
                "datetime": ns,
                "open": px, "high": px, "low": px, "close": px,
                "volume": 5, "open_oi": 7, "close_oi": 8,
            })

        class _Q:
            expire_datetime = expire_s
            underlying_symbol = und

        class _Api:
            def __init__(self, pro: bool = False) -> None:
                self.pro = pro
                self.asked: list[int] = []

            class _Auth:
                def __init__(self, pro: bool) -> None:
                    self.pro = pro

                def _has_feature(self, name: str) -> bool:
                    # 免费账户：tq_dl 没有；其他功能位也一律没有
                    return self.pro

            @property
            def _auth(self):
                return _Api._Auth(self.pro)

            def query_quotes(self, ins_class=None, exchange_id=None, expired=None):
                return list(calls) if exchange_id == ["DCE"] else []

            def get_quote(self, sym):
                return _Q()

            def get_kline_serial(self, sym, dur, data_length=0):
                self.asked.append(int(data_length))
                if sym.endswith("-P-3000"):        # 深度虚值，从没成交
                    return _kl(0, nan=True)
                return _kl(1.0)

        class _Sess:
            def __init__(self, api) -> None:
                self.api = api

            def get(self):
                return self.api

            def reset(self) -> None:
                pass

        try:
            api = _Api(pro=False)
            # 第一条腿：`update` 自己认出免费账户并降级（不是调用方硬传 free=True）
            st = OF.update(_Sess(api), periods=("1d",), incremental=True)
            one = st["1d"]
            assert one.get("free") is True, f"没走免费路径：{one}"
            assert one["rows"] == 3, f"写入 {one['rows']} 行，应为 3"
            assert max(api.asked) <= 10000, f"请求了 {max(api.asked)} 根"

            with OS.connect(read_only=True) as con:
                n_meta = con.execute("SELECT count(*) FROM option_meta").fetchone()[0]
                n_bar = con.execute("SELECT count(*) FROM option_bars").fetchone()[0]
                n_und = con.execute(
                    "SELECT count(*) FROM option_underlying").fetchone()[0]
            assert n_meta == 2, f"合约 {n_meta} 个，应为 2（含没成交的那个）"
            assert n_bar == 3, f"K 线 {n_bar} 行"
            assert n_und == 3, f"标的 {n_und} 行"

            # 第二条腿：断点续传 —— 已下到到期日前的链要跳过
            st2 = OF.update(_Sess(api), periods=("1d",), incremental=True)
            assert st2["1d"]["skipped"] == 1, f"没跳过：{st2['1d']}"
            assert st2["1d"]["rows"] == 0, f"重复写入 {st2['1d']['rows']} 行"

            # 第三条腿：专业账户不能走免费路径（否则白降级）
            from ttrade import tqfetch as TQF
            assert TQF.can_download(_Api(pro=False)) is False
            assert TQF.can_download(_Api(pro=True)) is True
            return (f"免费账户降级写入 {one['rows']} 行、{n_meta} 个合约；"
                    f"重跑跳过 {st2['1d']['skipped']} 条链")
        finally:
            OF.export_csv = old_csv
            if old_env is None:
                os.environ.pop("TRENDDESK_OPT_DB", None)
            else:
                os.environ["TRENDDESK_OPT_DB"] = old_env
            shutil.rmtree(tmp, ignore_errors=True)

    check("期权免费路径（无 tq_dl 也照下）", _opt_free_path)

    def _tq_data():
        """天勤分钟级仓库：能读、复权因子与切换表一致。

        因子是**派生数据**，只在写 bar 那一刻算一次会过期（切换表补全、每来一次新
        换月都会让它变）。所以这里现场拿切换表重算一遍逐根比对 —— 这是唯一能发现
        「库里因子过期」的检查，而那种错误不会报错，只会让复权序列在换月处出现假跳空。
        """
        from ttrade import tqdata as T
        from ttrade import tqstore as S
        from ttrade.tqfetch import adj_factor_series, factor_at

        import pandas as pd

        if not S.exists():
            return "未下载（可选：download_tq.py --main 1h --all）"
        try:
            st = S.stats()
        except Exception as exc:  # noqa: BLE001
            # 下载脚本正持着 DuckDB 的排他锁。这是预期状态，不该算自检失败。
            return f"行情库读不了（可能正在下载）：{type(exc).__name__}"
        per = st.get("periods") or {}
        if not per:
            return "仓库为空（可选）"
        cover = "，".join(f"{p} {v['symbols']} 品种 / {v['rows']} 行" for p, v in per.items())

        picked = None
        for sym, ex in S.switch_symbols():
            for period in ("1h", "5m"):
                if period in per and S.bar_count(sym, ex, period) > 100:
                    picked = (sym, ex, period)
                    break
            if picked:
                break
        if picked is None:
            return f"{cover}（没有可校验的品种）"

        sym, ex, period = picked
        df = T.load_main(sym, ex, period, adjust="none")
        sw = S.load_switch(sym, ex)
        got = df["adj_factor"].to_numpy(float)
        want = factor_at(adj_factor_series(sw), pd.Series(df["trading_day"]))
        bad = int((abs(got - want) > 1e-12).sum())
        if bad:
            raise RuntimeError(
                f"{sym}_{ex} {period} 有 {bad} 根 bar 的复权因子与切换表不一致"
                f"（最大偏差 {float(abs(got - want).max()):.3g}），跑 --refresh-factors 重算"
            )
        return f"{cover}；{sym}_{ex} {period} 因子与切换表逐根一致"

    check("天勤分钟级行情", _tq_data)

    def _tq_source():
        """轻量库 / 完整库的判定与容错。

        安装包只带约 270MB 的轻量库（主连三周期 + 真实合约日线），完整库 11GB 要
        用户自己下。判定错了两个方向都出事：下完 11GB 还在读轻量库（等于白下），
        或者轻量库缺 `contract_bars` 的分钟明细时 `stats()` 直接抛 CatalogException
        —— 那是数据管理页首屏要调的，抛了整页打不开。
        """
        from ttrade import tqstore as S

        kind = S.source()
        if kind == "none":
            return "未安装天勤库（可选）"
        st = S.stats()
        if st.get("kind") != kind:
            raise RuntimeError(f"source()={kind} 与 stats().kind={st.get('kind')} 不一致")
        # 不存在的品种必须返回空表，不能抛（轻量库里 5m 的真实合约根本没有）
        if len(S.load_contracts("__none__", "__none__", "5m")):
            raise RuntimeError("load_contracts 对不存在的品种返回了数据")
        per = "，".join(sorted(st.get("periods") or {}))
        cb = (st.get("contracts") or {}).get("contracts", 0)
        return f"{kind} 库 {st.get('size_mb')} MB，周期 {per}，真实合约 {cb} 个"

    check("天勤库类型（完整 / 轻量）", _tq_source)

    def _tq_aux():
        """分钟级也要算得出展期因子。

        `data_futures_aux/` 是日频的，分钟 bar 的时间戳里根本没有 00:00 这个点，
        按日期 join 一根都命中不上 —— 1h / 5m 的展期、期权因子曾长期整列 NaN，
        而界面上只表现为「算不出来」，不会报错。这里直接量「加载之后有值」。
        """
        import os

        from ttrade import auxdata as A
        from ttrade import factors as F
        from ttrade import tqdata as T
        from ttrade import tqstore as S

        if not S.exists():
            return "未下载（可选）"
        aux_dir = A._default_dir()
        for sym, ex in S.switch_symbols():
            if not os.path.exists(A.curve_path(aux_dir, sym, ex)):
                continue
            for period in ("1h", "5m"):
                if not T.has_period(sym, ex, period):
                    continue
                df = T.load_main(sym, ex, period).tail(500)
                out = A.attach_intraday(df, sym, ex)
                if "c1_close" not in out.columns:
                    raise RuntimeError(f"{sym}_{ex} {period} 没挂上期限结构列")
                ry = F.f_roll_yield(out)
                if float(ry.notna().mean()) < 0.99:
                    raise RuntimeError(
                        f"{sym}_{ex} {period} 展期收益率只有 {ry.notna().mean():.0%} 有值"
                    )
                return f"{sym}_{ex} {period} 展期收益率 {ry.notna().mean():.0%} 有值"
        return "没有同时具备分钟行情和衍生数据的品种（可选）"

    check("分钟级衍生列", _tq_aux)

    def _fund_attrs():
        """期货日线要挂品种标记，否则基本面因子整列「无数据」。

        `factors_ext._symbol()` 只认 `df.attrs["symbol"]`，认不出来就返回空串，
        仓单 / 库存 / 现货 / 基差 / 交割 / 板块这一整片因子全退化成 NaN。两条
        路径都会丢这个标记：GUI 的日线走 `load_futures_csv()`，它压根没挂；
        `auxdata.attach()` 用 `join` 并衍生列，pandas 的 merge 系列不传播
        attrs，挂过的也会被抹掉。批量回测走 `futures.load()`，它在 attach 之后
        才挂，所以一直是好的 —— 于是同一个品种「回测有值、看图没值」。
        """
        import os

        from ttrade import auxdata as A
        from ttrade import edb as E
        from ttrade import factors as F
        from ttrade.paths import data_dir as _dd

        try:
            from gui.app import load_futures_csv
        except Exception as exc:  # noqa: BLE001
            return f"跳过（GUI 导入失败：{exc}）"
        try:
            m = E.meta()
        except Exception:  # noqa: BLE001
            return "未下载 EDB（可选）"
        if m is None or len(m) == 0:
            return "未下载 EDB（可选）"
        wh = {str(v).lower() for v in m[m["category"] == "warehouse"]["variety"]}
        d = _dd("cn")
        # 先把候选收齐再逐条验：这一项的靶子是 `attrs["symbol"]` 别丢，
        # 不是「仓单数值必须已经下全」。以前取到第一个候选就 return，碰上
        # 只同步了指标清单、没下数值的品种（免费账户下不了非价量数据）就整项
        # 报红，把真正的红也淹掉了
        cands: list[tuple[str, str, str, object]] = []
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".csv") or fn.startswith("_"):
                continue
            key = fn[:-4]
            sym, _, ex = key.rpartition("_")
            if sym.lower() not in wh:
                continue
            df = load_futures_csv(key, d)
            if df is None or len(df) < 300:
                continue
            cands.append((key, sym, ex, df))
        if not cands:
            return "没有同时具备 EDB 仓单和日线 CSV 的品种（可选）"

        best_key, best_r = "", -1.0
        for key, sym, ex, df in cands:
            if df.attrs.get("symbol") != sym:
                raise RuntimeError(
                    f"{key} 加载后 attrs['symbol'] 为空 —— 基本面因子会整列 NaN"
                )
            # attach 之后再验一次：join 抹 attrs 的话这里就红
            out = A.attach(df, sym, ex)
            if out.attrs.get("symbol") != sym:
                raise RuntimeError(
                    f"{key} 并衍生列之后 attrs['symbol'] 被抹掉了（join 丢 attrs）"
                )
            r = float(F.raw(df, "war_oi").notna().mean())
            if r > best_r:
                best_key, best_r = key, r
            if r >= 0.5:
                return f"{key} 品种标记 {sym}，仓单持仓比 {r:.0%} 有值"
        # 一个品种都没仓单数值 ⇒ 无从验证覆盖率，但 attrs 那条已经全部验过
        return (f"跳过仓单覆盖率（{len(cands)} 个候选的 EDB 仓单只有清单没数值，"
                f"最好 {best_key} {best_r:.0%}）；品种标记 {len(cands)} 个都已挂上")

    check("期货日线品种标记（基本面因子）", _fund_attrs)

    def _qfq_tail():
        """前复权之后，最后一根必须等于真实盘面价。

        后复权会让最新价停在「按历史换月累计放大」的水平上（螺纹 3424 vs 盘面
        3047），图上看着没事，但实盘按价格算保证金、算手数时就是另一套数 ——
        回测 161 手、实盘 145 手，差在这里。所以这里逐品种比一次「前复权末值
        vs 不复权末值」，两者不等就说明复权基准没归一到最新一根。
        """
        import os

        from ttrade import futures
        from ttrade import tqdata as T
        from ttrade import tqstore as S

        picked, bad = 0, []
        if S.exists():
            for sym, ex in S.switch_symbols():
                for period in ("1h", "5m"):
                    if not T.has_period(sym, ex, period):
                        continue
                    q = T.load_main(sym, ex, period, adjust="qfq")
                    r = T.load_main(sym, ex, period, adjust="none")
                    if not len(q) or not len(r):
                        continue
                    if q.index[-1] != r.index[-1]:
                        continue
                    picked += 1
                    a, b = float(q["close"].iloc[-1]), float(r["close"].iloc[-1])
                    if abs(a - b) > max(1e-6, abs(b) * 1e-9):
                        bad.append(f"{sym}_{ex} {period} {a:.2f}≠{b:.2f}")
                    break
                if picked >= 3:
                    break
        for f in sorted(os.listdir(data_dir("cn"))):
            if not f.endswith(".csv") or f.startswith("_"):
                continue
            key = f[:-4]
            sym, _, ex = key.rpartition("_")
            if not sym or not ex:
                continue
            try:
                q = futures.load_local(sym, ex, data_dir("cn"), adjust="qfq")
                r = futures.load_local(sym, ex, data_dir("cn"), adjust="none")
            except Exception:  # noqa: BLE001
                continue
            picked += 1
            a, b = float(q["close"].iloc[-1]), float(r["close"].iloc[-1])
            if abs(a - b) > max(1e-6, abs(b) * 1e-9):
                bad.append(f"{key} 日线 {a:.2f}≠{b:.2f}")
            break
        if bad:
            raise RuntimeError("前复权末值不等于盘面价：" + "；".join(bad[:4]))
        if not picked:
            return "未下载（可选）"
        return f"{picked} 个品种的前复权末值都等于真实盘面价"

    check("前复权末值=盘面价", _qfq_tail)

    def _multiday():
        """2 日 / 周线：由日线合成，不落盘。

        这里验的不是「能跑」，而是几条**错了不会报错、只会让数字看起来还行**
        的性质：

        1. 根数对得上（2 日 = ceil(日线/2)，周线 ≈ 日线/5）；
        2. OHLC 是组内聚合 —— 开取第一根、收取最后一根、高低取极值，量额累加
           后总量守恒；
        3. 换一个起始日期切，**同一根 K 线不能变**。这条最容易踩：先切再合，
           配对就跟着起点走，同一根历史周线在两份报告里长得不一样；
        4. 年化根数按数据自己推（2 日 ≈ 日线一半），不能写死 252 —— 写死的话
           年化波动会被高估约 √2 倍，夏普看着更好，纯属假的。
        """
        import numpy as np

        from ttrade import tqdata as T
        from ttrade.backtest import bars_per_year

        # 找一个日线 CSV 在的品种，不依赖天勤库（CI 上不一定有）
        import os

        from ttrade import futures

        sym = ex = ""
        for f in sorted(os.listdir(data_dir("cn"))):
            if f.endswith(".csv") and not f.startswith("_"):
                s, _, e = f[:-4].rpartition("_")
                if s and e:
                    sym, ex = s, e
                    break
        if not sym:
            return "没有期货日线 CSV（可选）"

        # 底料就是这份 CSV —— 不是 `load_main("1d")`（那是天勤日线，只有 2016
        # 年以后），比对基准必须和合成用的同一份
        daily = futures.load_local(sym, ex, data_dir("cn"), adjust="qfq")
        if not len(daily):
            return "日线为空（可选）"
        out = []
        for p in T.MULTIDAY:
            m = T.load_main(sym, ex, p)
            if not len(m):
                raise RuntimeError(f"{p} 合成出来是空的")
            want = (len(daily) + 1) // 2 if p == "2d" else None
            if want is not None and len(m) != want:
                raise RuntimeError(f"2 日线根数 {len(m)} ≠ ceil({len(daily)}/2)={want}")
            if p == "1w" and not (len(daily) / 5.6 < len(m) < len(daily) / 4.2):
                raise RuntimeError(f"周线根数 {len(m)} 和日线 {len(daily)} 不成比例")
            # OHLC 聚合：极值不能越出日线区间，量额守恒
            if float(m["high"].max()) > float(daily["high"].max()) + 1e-6:
                raise RuntimeError(f"{p} 的最高价超过了日线的最高价")
            if float(m["low"].min()) < float(daily["low"].min()) - 1e-6:
                raise RuntimeError(f"{p} 的最低价低于了日线的最低价")
            if abs(float(m["volume"].sum()) - float(daily["volume"].sum())) > 1e-6:
                raise RuntimeError(f"{p} 的成交量合计和日线对不上（合成时漏根了）")
            # 同一根不能因为起点不同而变
            cut = m.index[len(m) // 2]
            part = T.load_main(sym, ex, p, start=cut)
            row_a, row_b = m.loc[cut], part.iloc[0]
            if abs(float(row_a["open"]) - float(row_b["open"])) > 1e-9 \
                    or abs(float(row_a["close"]) - float(row_b["close"])) > 1e-9:
                raise RuntimeError(f"{p} 换了起始日期之后同一根 K 线变了（先切再合了？）")
            bpy = bars_per_year(m.index)
            d_bpy = bars_per_year(daily.index)
            if p == "2d" and not (0.35 * d_bpy < bpy < 0.65 * d_bpy):
                raise RuntimeError(f"2 日线年化根数 {bpy:.0f} 和日线 {d_bpy:.0f} 对不上")
            if p == "1w" and not (30 < bpy < 60):
                raise RuntimeError(f"周线年化根数 {bpy:.0f} 不在 30~60 之间")
            out.append(f"{T.PERIOD_LABEL[p]} {len(m)} 根 / 年化 {bpy:.0f} 根")
            # 合成周期不该有自己的一份落盘数据
            if os.path.exists(T.main_path(sym, ex, p)):
                raise RuntimeError(f"{p} 居然有独立落盘文件，应该是现合成的")
        return f"{sym}_{ex} " + "、".join(out)

    check("2 日 / 周线由日线合成", _multiday)

    def _backtest():
        from ttrade.backtest import BacktestConfig, run_backtest
        from ttrade.eastmoney import load_local
        from ttrade.strategies import run_strategy

        cands = sorted(f[:-4].replace("_", ".") for f in os.listdir(data_dir("us"))
                       if f.endswith(".csv") and not f.startswith("_"))
        df = load_local(cands[0], data_dir("us"))
        sig = run_strategy(df, "donchian")
        res = run_backtest(sig, BacktestConfig())
        m = res.metrics
        return f"cagr={m['cagr'] * 100:.1f}% sharpe={m['sharpe']:.2f} trades={m['n_trades']}"

    check("策略回测", _backtest)

    def _trade_notes():
        """交易记录的备注：进场策略 / 出场原因，外加累计盈亏的口径。

        三条都不难错、错了又都不好发现，所以单列一项：
        - 引擎的兜底原因（策略名 / 信号离场 / 信号反向）
        - 策略写进 df 的逐根原因，必须按**决策根**取（差一根就把原因安到
          另一根上，界面上看不出错，只会觉得「这次平得莫名其妙」）
        - 累计盈亏是逐笔金额累加（不是百分比连乘，也不是百分比相加）
        """
        from ttrade.backtest import BacktestConfig, run_backtest
        import numpy as np
        import pandas as pd

        idx = pd.date_range("2021-01-01", periods=40, freq="D")
        px = np.r_[np.full(10, 100.0), np.linspace(100, 120, 15),
                   np.linspace(120, 105, 15)]
        base = pd.DataFrame({"open": px, "high": px + 1, "low": px - 1,
                             "close": px,
                             "signal": np.r_[np.zeros(5, int), np.ones(30, int),
                                             np.zeros(5, int)]}, index=idx)
        r = run_backtest(base, BacktestConfig(), label="唐奇安")
        if not r.trades:
            raise RuntimeError("样本没产生交易，这一项失去意义")
        t0 = r.trades[0]
        if t0.entry_note != "唐奇安":
            raise RuntimeError(f"进场原因没兜底成策略名：{t0.entry_note!r}")
        if t0.exit_note != "信号离场":
            raise RuntimeError(f"出场原因没兜底：{t0.exit_note!r}")

        # 反向：信号从 +1 直接到 -1，出场原因该是「信号反向」而不是「信号离场」
        flip = base.copy()
        flip["signal"] = np.r_[np.ones(20, int), -np.ones(20, int)]
        rf = run_backtest(flip, BacktestConfig(), label="双均线")
        if not rf.trades or rf.trades[0].exit_note != "信号反向":
            raise RuntimeError(
                f"反向时出场原因不对：{rf.trades[0].exit_note if rf.trades else '没有交易'}")

        # 策略自己写的原因要按决策根取：信号在第 6 根（下标 5）由 0 变 1，
        # 进场发生在第 7 根开盘，原因必须来自下标 5 那根（决策根）；
        # 平仓同理，信号在下标 35 由 1 变 0，平仓在下标 36 开盘
        noted = base.copy()
        noted["exit_note"] = [""] * len(base)
        noted["entry_note"] = [""] * len(base)
        noted.iloc[5, noted.columns.get_loc("entry_note")] = "第 6 根决策进场"
        noted.iloc[35, noted.columns.get_loc("exit_note")] = "第 36 根决策离场"
        noted.iloc[36, noted.columns.get_loc("exit_note")] = "错位了"
        rn = run_backtest(noted, BacktestConfig(), label="x")
        if not rn.trades:
            raise RuntimeError("带备注列的样本没产生交易")
        tn = rn.trades[0]
        if tn.entry_note != "第 6 根决策进场" or tn.exit_note != "第 36 根决策离场":
            raise RuntimeError(
                f"逐根原因没按决策根取：进场 {tn.entry_note!r} / 出场 {tn.exit_note!r}")

        # 因子策略：两列都要有，出场原因要写清是哪条口径
        from ttrade import factor_strategy as FS
        from ttrade import factors as F

        df_f = _us_df()
        key = "mom_20" if "mom_20" in F.FACTORS else next(iter(F.FACTORS))
        cfg_f = FS.FactorConfig(window=60, entry_z=1.0, exit_z=0.3,
                                exit_mode="tier",
                                exit_tiers=[[2.0, 2.0], [4.0, 1.2], [6.0, 0.8]],
                                exit_sl=True, exit_sl_atr=2.5)
        out_f = FS.run_factor_strategy(df_f, "single", [key], cfg_f)
        d_f = out_f.df
        if "entry_note" not in d_f.columns or "exit_note" not in d_f.columns:
            raise RuntimeError("因子策略没往 df 里写 entry_note / exit_note")
        ent = d_f.loc[d_f["entry_note"].astype(bool), "entry_note"]
        ext = d_f.loc[d_f["exit_note"].astype(bool), "exit_note"]
        if ent.empty:
            raise RuntimeError("因子策略一条进场原因都没写")
        if not str(ent.iloc[0]).startswith("因子分"):
            raise RuntimeError(f"进场原因没写分数：{ent.iloc[0]!r}")
        if ext.empty:
            raise RuntimeError("开了分档回撤却一条出场原因都没写")
        if "分档浮盈回撤" not in str(ext.iloc[0]):
            raise RuntimeError(f"出场原因没写清口径：{ext.iloc[0]!r}")

        # 累计盈亏：逐笔金额累加（不是百分比连乘、也不是百分比相加）
        class _T:
            def __init__(self, pnl):
                self.pnl = pnl

        from gui.result_view import cumulative_pnl

        cc = cumulative_pnl([_T(120.0), _T(-60.0)])
        if abs(cc[0] - 120.0) > 1e-9 or abs(cc[1] - 60.0) > 1e-9:
            raise RuntimeError(f"累计盈亏不是逐笔金额累加：{cc}")
        return (f"兜底 {t0.entry_note}/{t0.exit_note}、反向 {rf.trades[0].exit_note}、"
                f"决策根对齐、因子写入口径、累计按金额累加")

    def _us_df():
        """自检里反复要用的一份行情，按需加载一次就缓存。"""
        nonlocal _cache
        if "us" not in _cache:
            from ttrade.eastmoney import load_local

            cands = sorted(f[:-4].replace("_", ".") for f in os.listdir(data_dir("us"))
                           if f.endswith(".csv") and not f.startswith("_"))
            _cache["us"] = load_local(cands[0], data_dir("us"))
        return _cache["us"]

    # 放在 `_us_df` 之后：`check()` 是立即执行的，定义在前会拿不到这个闭包
    check("交易记录备注与累计盈亏", _trade_notes)

    def _a_screen():
        """A 股选股策略（`a_vol_breakout`）的条件口径。

        这类「一堆阈值」的策略最怕的不是算错，而是**某一条悄悄没生效**：
        ST 过滤没挂上、涨停按错板块的幅度判、PE 把负值也算成「低估值」——
        都不会报错，结果只会变好看。所以每条边界都钉一遍。
        """
        import numpy as np
        import pandas as pd

        from ttrade.strategies import _round2, run_strategy

        def make(pct, vol, pe, amt, close, board=0.0, is_st=False):
            n = len(pct)
            px = np.asarray(close, dtype=float)
            return pd.DataFrame({
                "open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
                "volume": np.asarray(vol, dtype=float),
                "amount": np.asarray(amt, dtype=float),          # 千元
                "pct_chg": np.asarray(pct, dtype=float),
                "pe_ttm": np.asarray(pe, dtype=float),
                "board_code": np.full(n, board),
                "is_st": np.full(n, is_st),
                "signal": np.zeros(n, dtype=int),
            }, index=pd.date_range("2024-01-01", periods=n, freq="B"))

        # ① 涨停价必须四舍五入。`np.round` 是银行家舍入（3.005 → 3.00），
        #    交易所是四舍五入（3.01），差一分钱就能把涨停判成没涨停
        if abs(_round2(np.array([3.005]))[0] - 3.01) > 1e-9:
            raise RuntimeError("_round2 没做四舍五入（银行家舍入会把涨停判错）")

        pct = [0, 0, 0, 0, 5.0, 1, 1, 1, 1, 1]
        vol = [100, 100, 100, 100, 500, 100, 100, 100, 100, 100]
        amt = [10000] * 4 + [60000] + [10000] * 5      # 千元，5000 万 = 50000
        base = make(pct, vol, [15] * 10, amt, [50] * 10)
        sig = run_strategy(base, "a_vol_breakout", {"hold_days": 5})["signal"].to_numpy()
        if sig[4] != 1 or sig[4:9].sum() != 5 or sig[9] != 0:
            raise RuntimeError(f"基本触发/持有根数不对：{sig.tolist()}")

        # ② ST、北交所：一律不买
        if run_strategy(make(pct, vol, [15] * 10, amt, [50] * 10, is_st=True),
                        "a_vol_breakout", {})["signal"].sum() != 0:
            raise RuntimeError("ST 股不该有信号")
        if run_strategy(make(pct, vol, [15] * 10, amt, [50] * 10, board=3.0),
                        "a_vol_breakout", {})["signal"].sum() != 0:
            raise RuntimeError("北交所不该有信号")

        # ③ 昨日涨停 ⇒ 今天不买。主板 10%、创业板 20% 要分开判：
        #    创业板涨 20% 用主板口径看只是「涨得多」，会被误买
        for board, chg, px in ((0.0, 10.0, 55.0), (1.0, 20.0, 12.0)):
            f = make([0, chg, 5.0, 0, 0, 0, 0, 0, 0, 0], [100] * 10, [15] * 10,
                     [60000] * 10, [px] * 10, board=board)
            f.iloc[0, f.columns.get_loc("close")] = px / (1 + chg / 100)
            f.iloc[1, [f.columns.get_loc(c) for c in ("open", "high", "low", "close")]] = px
            if run_strategy(f, "a_vol_breakout", {})["signal"].iloc[1] != 0:
                raise RuntimeError(f"board={board} 昨日涨停，今天不该买")

        # ④ PE：上限含等号，负 PE 不算低估值；25 / 30 两档要能切换
        for pe, want in ((25.0, True), (25.01, False), (0.0, False), (-8.0, False)):
            f = make(pct, vol, [pe] * 10, amt, [50] * 10)
            got = run_strategy(f, "a_vol_breakout", {"pe_max": 25})["signal"].sum() > 0
            if bool(got) != want:
                raise RuntimeError(f"PE={pe} 判定错：{got} != {want}")
        f27 = make(pct, vol, [27.0] * 10, amt, [50] * 10)
        if run_strategy(f27, "a_vol_breakout", {"pe_max": 30})["signal"].sum() == 0:
            raise RuntimeError("PE 上限调到 30，27 倍该放进来")

        # ⑤ 量比 / 成交额 / 股价：边界都是「≥ 或 ≤」，差一点点就该翻面
        for volv, want in ((299, False), (300, True)):
            f = make(pct, [100] * 4 + [volv] + [100] * 5, [15] * 10, amt, [50] * 10)
            if (run_strategy(f, "a_vol_breakout", {"vol_mult": 3})["signal"].sum() > 0) != want:
                raise RuntimeError(f"量比 {volv/100:.2f} 倍判定错")
        for a, want in ((49999, False), (50000, True)):
            f = make(pct, vol, [15] * 10, [10000] * 4 + [a] + [10000] * 5, [50] * 10)
            if (run_strategy(f, "a_vol_breakout", {"amt_min": 5000})["signal"].sum() > 0) != want:
                raise RuntimeError(f"成交额 {a} 千元判定错")
        for px, want in ((80.01, False), (79.99, True)):
            f = make(pct, vol, [15] * 10, amt, [px] * 10)
            if (run_strategy(f, "a_vol_breakout", {"max_price": 80})["signal"].sum() > 0) != want:
                raise RuntimeError(f"股价 {px} 判定错")

        return ("四舍五入涨停价、5 根持有、ST/北交所排除、"
                "主板 10% 与创业板 20% 涨停分别判、PE 25/30 两档、"
                "量比/成交额/股价三条边界")

    check("A股选股策略（放量启动）的条件口径", _a_screen)

    def _a_rule():
        """全市场组合回测（`stockrule`）的两个**量纲/编码**坑。

        这两个错都不会报错，只会让回测结果系统性偏松：

        ① `stock_meta.board` 存的是中文（主板/创业板/…），`_limit_pct` 要的是
           0/1/2/3。「非北交所」按白名单 `{0,1,2}` 判 ⇒ 没做映射时整列都是
           NaN，全部被剔掉，回测直接变成空仓；或者映射漏了沪市 600/601/603，
           整个沪市主板静默消失。
        ② Tushare `daily.amount` 是**千元**，而配置里的 `amt_min` 按万元给。
           不除 10 就变成「成交额 ≥ 500 万」也能过，等于这条条件宽松了 10 倍。
        """
        import numpy as np
        import pandas as pd

        from ttrade import stockrule as SR

        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        codes = ["SSE.600000", "SZSE.300001", "BSE.920285"]

        def panel(amount_last):
            return {
                "close": pd.DataFrame(50.0, index=idx, columns=codes),
                "open": pd.DataFrame(50.0, index=idx, columns=codes),
                "volume": pd.DataFrame([[100, 100, 100]] * 2 + [[600, 600, 600]],
                                       index=idx, columns=codes),
                "amount": pd.DataFrame([[1e5, 1e5, 1e5]] * 2
                                       + [amount_last], index=idx, columns=codes),
                "pct_chg": pd.DataFrame([[0.0, 0.0, 0.0]] * 2 + [[5.0, 5.0, 5.0]],
                                        index=idx, columns=codes),
                "pe_ttm": pd.DataFrame(15.0, index=idx, columns=codes),
                "board": pd.Series([0.0, 1.0, 3.0], index=codes),      # 沪主板/创业板/北交所
                "is_st": pd.DataFrame(False, index=idx, columns=codes),
            }

        # 成交额 49999 千元 = 4999.9 万元 ⇒ 不够；50000 千元 = 5000 万 ⇒ 刚好够
        low = SR.screen(panel([49999.0, 49999.0, 49999.0]),
                        SR.ScreenConfig(amt_min=5000.0)).to_numpy()[-1]
        high = SR.screen(panel([50000.0, 50000.0, 50000.0]),
                         SR.ScreenConfig(amt_min=5000.0)).to_numpy()[-1]
        if low.any():
            raise RuntimeError("4999.9 万元的成交额不该通过 5000 万的门槛（量纲差 10 倍）")
        if not (high[0] and high[1]):
            raise RuntimeError("沪市主板 600xxx / 创业板 300xxx 都应该通过「非北交所」")
        if high[2]:
            raise RuntimeError("北交所（board=3）不该通过")

        return "成交额千元→万元换算、板块编码 0/1/2/3 与「非北交所」白名单"

    check("A股全市场组合回测的量纲与板块编码", _a_rule)

    def _short_default():
        """内置策略「允许做空」的默认勾选：期货勾上，A 股 / 美股不勾。

        这是容易悄悄回归的一处：判据原来散在三个地方（研究页读市场表、策略库
        读 `note == "short"`、内置模板在 `to_config` 里直接把 note 清空），
        结果 34 条期货内置模板全部默认**只做多** —— 而双均线 / 唐奇安 / MACD
        的说明本身就是「上穿做多、下穿做空」，不勾做空只剩半个策略。

        这一项三条腿都要钉住：① 判据函数本身；② 34 条期货模板的 `note`；
        ③ A 股模板必须**不**带 short（融券做空对散户不成立）。
        """
        from ttrade import replay as RP
        from ttrade import strategy_templates as T
        from ttrade.strategy_store import StrategyConfig

        cases = [
            (StrategyConfig(symbol="RB", exchange="SHFE", period="1d"), True, "期货"),
            (StrategyConfig(symbol="600000", exchange="SSE", period="1d"), False, "A股"),
            (StrategyConfig(symbol="AAPL", exchange="", period="1d"), False, "美股"),
        ]
        for cfg, want, name in cases:
            got = RP.default_allow_short(cfg)
            if bool(got) != want:
                raise RuntimeError(
                    f"{name}的「允许做空」默认值错：{got} != {want}")

        fut = []
        stock = []
        for k in T.keys():
            c = T.to_config(k)
            if c is None:
                continue
            if c.market == "cn_stock":
                if c.note == "short":
                    raise RuntimeError(f"{k} 是 A 股模板，不该默认允许做空")
                stock.append(k)
            else:
                if c.note != "short":
                    raise RuntimeError(
                        f"期货模板 {k} 没默认勾「允许做空」（note={c.note!r}）")
                fut.append(k)
        if len(fut) < 30:
            raise RuntimeError(f"期货模板只有 {len(fut)} 条，样本不对")
        if not stock:
            raise RuntimeError("没有 A 股内置模板")

        # A 股策略必须标了市场，否则会出现在期货的策略下拉里、点一次报一次错
        from ttrade.strategies import STRATEGIES

        spec = STRATEGIES.get("a_vol_breakout")
        if spec is None:
            raise RuntimeError("a_vol_breakout 没注册成内置策略")
        if tuple(spec.markets) != ("cn_stock",):
            raise RuntimeError(f"a_vol_breakout 的 markets 不对：{spec.markets}")
        return (f"期货 {len(fut)} 条模板默认勾做空、A 股 {len(stock)} 条不勾；"
                f"a_vol_breakout 限定 {spec.markets}")

    check("内置策略「允许做空」的默认勾选", _short_default)

    def _no_long_flag():
        """「禁止做多」：只留空头，界面默认不勾。

        它和「允许做空」是一对：两个都勾等于一笔不做，所以界面把「不允许做空
        + 禁止做多」这种组合直接弹掉（见 `MainWindow._sync_dir_flags`）。
        引擎这一侧要保证的是——勾上之后**一根多头都没有**，而不是「大部分是空头」。

        单品种和截面两条路都要验：截面不是把 signal 取反，而是**根本不建多头腿**，
        所以看的是权重里有没有正数。
        """
        import numpy as np
        import pandas as pd

        from ttrade import cross as CX
        from ttrade.backtest import BacktestConfig, run_backtest

        idx = pd.date_range("2022-01-03", periods=200, freq="B")
        px = 100 * np.exp(np.cumsum(np.r_[np.full(60, 0.002), np.full(60, -0.002),
                                          np.full(80, 0.001)]))
        sig = np.where(np.arange(200) % 80 < 40, 1, -1)
        df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                           "close": px, "signal": sig}, index=idx)

        def _run(**kw):
            cfg = BacktestConfig(initial_cash=100_000.0, sizing="fixed",
                                 volume=10, **kw)
            return run_backtest(df, cfg).metrics

        both = _run()
        long_only = _run(long_only=True)
        short_only = _run(short_only=True)
        if both["long_exposure"] <= 0 or both["short_exposure"] <= 0:
            raise RuntimeError("默认（双向）本该多空都有")
        if long_only["short_exposure"] > 0 or long_only["n_short"]:
            raise RuntimeError("只做多里还有空头")
        if short_only["long_exposure"] > 0 or short_only["n_long"]:
            raise RuntimeError(
                f"只做空里还有多头：long_exposure={short_only['long_exposure']:.3f}、"
                f"n_long={short_only['n_long']}")

        # ---- 截面：只做空时权重里不能有正数
        rng = np.random.default_rng(4)
        cidx = pd.date_range("2021-01-04", periods=300, freq="B")
        frames = {}
        for k in range(6):
            p = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(cidx))))
            frames[f"S{k}_SHFE"] = pd.DataFrame(
                {"open": p, "high": p * 1.01, "low": p * 0.99, "close": p,
                 "volume": 1e4, "open_interest": 1e4}, index=cidx)
        bt = BacktestConfig(initial_cash=500_000.0, fee_rate=0.0003, slippage=0.0)

        def _cross(**kw):
            cfg = CX.CrossConfig(mode="hedge", factor_mode="single",
                                 factor_keys=["mom_20"], top_pct=30.0, bot_pct=30.0,
                                 weight="equal", rebalance=5, **kw)
            out = CX.run(frames, cfg, bt)
            if not out.ok:
                raise RuntimeError(f"截面回测没跑通：{out.error}")
            return out.weights

        w_both = _cross()
        w_long = _cross(long_only=True)
        w_short = _cross(short_only=True)
        if float(w_both.clip(lower=0).sum(axis=1).max()) <= 0:
            raise RuntimeError("默认对冲本该有多头腿")
        if float(w_long.clip(upper=0).abs().sum(axis=1).max()) > 1e-12:
            raise RuntimeError("截面「只做多」里还有空头腿")
        if float(w_short.clip(lower=0).sum(axis=1).max()) > 1e-12:
            raise RuntimeError("截面「只做空」里还有多头腿")
        if float(w_short.clip(upper=0).abs().sum(axis=1).max()) <= 0:
            raise RuntimeError("截面「只做空」连空头腿都没了")

        # ---- 收编模板：默认「允许做空、不禁止做多」
        from ttrade import strategy_templates as T

        libs = [k for k in T.keys() if k.startswith("tpl_lib_")]
        if len(libs) < 50:
            raise RuntimeError(f"收编进去的模板只有 {len(libs)} 条")
        for k in libs:
            c = T.to_config(k)
            if c is None:
                raise RuntimeError(f"{k} 还原不出来")
            if getattr(c, "short_only", False):
                raise RuntimeError(f"{k} 默认勾了「禁止做多」，应该默认不勾")
            if c.market != "cn_stock" and c.note != "short":
                raise RuntimeError(f"{k} 默认没勾「允许做空」")
        names = T.names()
        if len(set(names)) != len(names):
            raise RuntimeError("内置模板有重名")
        return (f"单品种 long={long_only['long_exposure']:.2f}/short={short_only['short_exposure']:.2f}、"
                f"截面多头腿 {float(w_short.clip(lower=0).sum(axis=1).max()):.0f}、"
                f"收编 {len(libs)} 条模板默认允许做空且不禁止做多")

    check("禁止做多（只做空）", _no_long_flag)

    def _chart_from_trades():
        """K 线与信号页的持仓/标记必须**与交易记录同源**。

        原来图上是从 `signal` 列推的，两个坑都让图和交易记录对不上：
        ① 只做多时 `signal` 却是多空双向算出来的，图上 `max(sig,0)` 裁掉空头，
           空头那段变成「空仓」，而回测里本来能开多 —— 碳酸锂实测记录 52 笔、
           图上 24 段、开仓点重叠 0 个；
        ② K 线内策略是当根成交，图上统一平移一根（T+1 口径）⇒ 标记全晚一根。

        现在改成从 `self.trades` 还原。这一项逐条钉住那个映射：
        区间、开平仓那根、方向、手数，以及「开仓在区间之前」别整段丢掉。
        """
        import numpy as np
        import pandas as pd

        from ttrade.backtest import Trade

        from gui.result_view import ResultView

        idx = pd.date_range("2024-03-01", periods=20, freq="D")
        t1 = Trade(entry_date=idx[2], exit_date=idx[8], entry_price=100.0,
                   exit_price=110.0, ret=0.1, bars=6, direction=1, lots=3.0)
        t2 = Trade(entry_date=idx[10], exit_date=idx[-1], entry_price=120.0,
                   exit_price=115.0, ret=0.04, bars=9, direction=-1,
                   open_position=True, lots=5.0)
        fake = type("R", (), {"trades": [("X", t1), ("X", t2)]})()

        pos, ent, ext, lines = ResultView._k_pos_from_trades(fake, idx)
        want = np.zeros(20, dtype=int)
        want[2:8] = 1          # 平仓那根（idx[8]）开盘就平了 ⇒ 不含
        want[10:] = -1         # 未平仓 ⇒ 一直到最后一根
        if not (pos == want).all():
            raise RuntimeError(f"持仓区间不对：{pos.tolist()}")
        if sorted(ent) != [2, 10]:
            raise RuntimeError(f"开仓标记位置不对：{sorted(ent)}")
        if sorted(ext) != [8]:
            raise RuntimeError(f"平仓标记位置不对：{sorted(ext)}")
        if abs(ent[2][0][1] - 3.0) > 1e-9 or abs(ent[10][0][1] - 5.0) > 1e-9:
            raise RuntimeError("手数没带到图上（开仓标记要能手数）")
        if len(lines) != 2 or lines[1][1] != len(idx) - 1:
            raise RuntimeError(f"持仓连线不对：{lines}")

        # 切「近 1 年」：开仓在区间之前、还没平的那笔，必须从第一根就显示持仓，
        # 否则整段凭空消失，看着像空仓
        sub = idx[5:]
        pos2, ent2, ext2, _ = ResultView._k_pos_from_trades(fake, sub)
        if not (pos2[:3] == 1).all():
            raise RuntimeError(f"区间前开仓的那笔没延续进来：{pos2[:5].tolist()}")
        if 0 in ent2:
            raise RuntimeError("区间前的开仓不该被画成「第 1 根开仓」")
        if sorted(ext2) != [3]:        # idx[8] 在子区间里是第 3 根
            raise RuntimeError(f"子区间的平仓位置不对：{sorted(ext2)}")

        return (f"同源映射：{int((pos != 0).sum())} 根持仓、"
                f"开仓 {len(ent)} 个、平仓 {len(ext)} 个、"
                f"手数 3/5 手带到图上、区间前开仓延续、连线到右边缘")

    # 注意：`check()` 是立即执行的，这一项 import 了 Qt，
    # 要等下面的 `_window` 把 QApplication 建起来之前再验（见文件末尾的 check）

    def _intrabar():
        """K 线内进出场（`run_intrabar_backtest`）。

        这条分支容易出的错都是「不报错、只是数字悄悄变好看」那种，所以五件事
        必须逐条验：入口成交在**触发价**（不是下一根开盘）、止损成交在
        **止损价**、拿掉那几列之后老路径**一个数都不许变**、**进场当根不出场**
        （逐笔 bars>0，且只有诊断开关能造出 bars==0）、显式打开
        `intrabar_samebar_stop` 才在当根扫。
        """
        import numpy as np
        import pandas as pd

        from ttrade.backtest import BacktestConfig, run_backtest

        cfg = BacktestConfig(initial_cash=100_000.0, sizing="fixed", volume=1,
                             multiplier=1.0, fee_per_lot=0.0, tick_size=0.0,
                             slippage=0.0)

        # 10 根平价，第 7 根（下标 6）一根大阳线扫过触发价 105
        px = np.array([100.0] * 6 + [104.0, 110.0, 112.0, 111.0], dtype=float)
        hi = np.array([101.0] * 6 + [110.0, 111.0, 113.0, 112.0], dtype=float)
        lo = np.array([99.0] * 6 + [103.0, 108.0, 110.0, 109.0], dtype=float)
        idx = pd.date_range("2021-01-01", periods=len(px), freq="D")
        d = pd.DataFrame({"open": px, "high": hi, "low": lo, "close": px,
                          "signal": np.zeros(len(px), dtype=int)}, index=idx)
        d["entry_long"] = np.nan
        d["entry_short"] = np.nan
        d["exit_long"] = 0.0
        d["exit_short"] = 0.0
        d["stop_dist"] = 4.0
        d.iloc[6, d.columns.get_loc("entry_long")] = 105.0

        r = run_backtest(d, cfg, label="K线内")
        if r.metrics.get("intrabar") != 1.0:
            raise RuntimeError("带了 entry_long 却没走 K 线内分支")
        if len(r.trades) != 1:
            raise RuntimeError(f"应该只有 1 笔交易，实得 {len(r.trades)} 笔")
        t = r.trades[0]
        # 开盘 104 低于触发价 105 ⇒ 按 105 成交；滑点为 0
        if abs(t.entry_price - 105.0) > 1e-9:
            raise RuntimeError(f"入场没按触发价成交：{t.entry_price}")
        # 止损 105 − 4 = 101，当根最低 103 没打到 ⇒ 应该还在持仓
        if not t.open_position:
            raise RuntimeError("当根最低 103 没到止损 101，不该平仓")

        # ---- 同根止损：**默认关**，进场当根不判止损 ----
        #
        # 最低价压到 100（止损 101），但这根 bar 的 low 完全可能是在进场之前
        # 做出来的 —— 进场要求价格摸到 105，而 100 在开盘 104 那一刻就在下面
        # 了。默认口径认定「这个 low 早于建仓」，所以这一笔不该被扫掉。
        d2 = d.copy()
        d2.iloc[6, d2.columns.get_loc("low")] = 100.0
        r2 = run_backtest(d2, cfg, label="K线内")
        if len(r2.trades) != 1 or not r2.trades[0].open_position:
            raise RuntimeError("默认关同根止损，这一笔不该被当根扫掉")
        if abs(r2.trades[0].entry_price - 105.0) > 1e-9:
            raise RuntimeError(f"进场价被同根止损带歪：{r2.trades[0].entry_price}")

        # 显式打开同根止损（悲观上界）⇒ 当根就该按 101 止损
        cfg2 = BacktestConfig(initial_cash=100_000.0, sizing="fixed", volume=1,
                              multiplier=1.0, fee_per_lot=0.0, tick_size=0.0,
                              slippage=0.0, intrabar_samebar_stop=True)
        r3 = run_backtest(d2, cfg2, label="K线内")
        if len(r3.trades) != 1 or r3.trades[0].open_position:
            raise RuntimeError("开了同根止损，打穿却没平仓")
        if abs(r3.trades[0].exit_price - 101.0) > 1e-9:
            raise RuntimeError(f"止损没按止损价成交：{r3.trades[0].exit_price}")

        # 默认口径下，**下一根**才判止损：把下根最低价压到 100 ⇒ 按 101 平
        d3 = d2.copy()
        d3.iloc[7, d3.columns.get_loc("low")] = 100.0
        r4 = run_backtest(d3, cfg, label="K线内")
        if len(r4.trades) != 1 or r4.trades[0].open_position:
            raise RuntimeError("下一根打穿止损却没平仓")
        if abs(r4.trades[0].exit_price - 101.0) > 1e-9:
            raise RuntimeError(f"下一根止损价不对：{r4.trades[0].exit_price}")

        # 拿掉那几列 ⇒ 走老路径，且结果必须和「本来就没有这几列」完全一致
        plain = d[["open", "high", "low", "close", "signal"]].copy()
        rp = run_backtest(plain, cfg, label="老路")
        if rp.metrics.get("intrabar") is not None:
            raise RuntimeError("没有 entry_* 列却走了 K 线内分支")
        if rp.metrics["n_trades"] != 0:
            raise RuntimeError("signal 全 0 却产生了交易")

        # 出场：上一根收盘给标志，下一根**开盘**成交
        d4 = plain.copy()
        d4["signal"] = np.r_[np.ones(7, dtype=int), np.zeros(3, dtype=int)]
        a = run_backtest(d4, cfg, label="老路")
        d5 = d.copy()
        d5["signal"] = d4["signal"]
        d5["exit_long"] = 0.0
        d5.iloc[7, d5.columns.get_loc("exit_long")] = 1.0
        b = run_backtest(d5, cfg, label="K线内")
        if not a.trades or not b.trades:
            raise RuntimeError("离场对比样本没成交")
        if abs(a.trades[0].exit_price - b.trades[0].exit_price) > 1e-9:
            raise RuntimeError(
                f"反向突破离场没按开盘价成交：老路 {a.trades[0].exit_price} "
                f"vs K 线内 {b.trades[0].exit_price}")

        # ---- 不反手：平仓当根不许再开新仓，下一根才重新看触发价 ----
        #
        # 持有多单时看不到空头触发价；多单出场（ATR 止损 / 反向突破开盘平）
        # 的**那一根**即使空头触发价被打穿，也只能空仓等着。否则一根 K 线里
        # 「先止损平多、再跌破开空」会被记成反手，而这两件事的先后盘口上分不
        # 出来 —— 挑对自己有利的那个就是白送 alpha
        def _flip_frame(exit_at_open: bool):
            px_ = np.array([100.0] * 10, dtype=float)
            hi_ = np.array([101.0] * 6 + [106.0, 101.0, 99.0, 99.0], dtype=float)
            lo_ = np.array([99.0] * 6 + [102.0, 95.0, 90.0, 99.0], dtype=float)
            f = pd.DataFrame({"open": px_, "high": hi_, "low": lo_, "close": px_,
                              "signal": np.zeros(10, dtype=int)},
                             index=pd.date_range("2022-01-01", periods=10, freq="D"))
            f["entry_long"] = np.nan
            f["entry_short"] = np.nan
            f["exit_long"] = 0.0
            f["exit_short"] = 0.0
            f["stop_dist"] = 4.0
            f.iloc[6, f.columns.get_loc("entry_long")] = 105.0   # 第 7 根做多
            f.iloc[6, f.columns.get_loc("exit_long")] = 1.0 if exit_at_open else 0.0
            # 第 8 根：多单被打止损（101，开盘 100 更低 ⇒ 按 100 成交），
            # 同一根的空头触发价 96 也被打穿（最低 95）
            f.iloc[7, f.columns.get_loc("entry_short")] = 96.0
            # 第 9 根：继续往下，空头在这一根才该开出来
            f.iloc[8, f.columns.get_loc("entry_short")] = 96.0
            return f

        for tag, ex_open in (("ATR 止损", False), ("反向突破离场", True)):
            f = _flip_frame(ex_open)
            rr = run_backtest(f, cfg, label="不反手")
            if len(rr.trades) != 2:
                raise RuntimeError(
                    f"{tag} 那一根不该开新仓：预期 2 笔（多单 + 下一根的空单），"
                    f"实得 {len(rr.trades)} 笔")
            if int(rr.position.iloc[7]) != 0:
                raise RuntimeError(
                    f"{tag} 当根应该空仓，实得持仓 {int(rr.position.iloc[7])}")
            t2 = rr.trades[1]
            if t2.direction != -1:
                raise RuntimeError(f"{tag} 之后那一笔应该做空，实得方向 {t2.direction}")
            if rr.equity.index.get_loc(t2.entry_date) != 8:
                raise RuntimeError(
                    f"{tag} 之后的空单应该在第 9 根开，实际开在 "
                    f"{rr.equity.index.get_loc(t2.entry_date) + 1} 根")
            # 对照：允许平仓当根再开仓（乐观口径）⇒ 同一根里就反手了
            cfg_opt = BacktestConfig(initial_cash=100_000.0, sizing="fixed",
                                     volume=1, multiplier=1.0, fee_per_lot=0.0,
                                     tick_size=0.0, slippage=0.0,
                                     intrabar_reentry_samebar=True)
            ro = run_backtest(f, cfg_opt, label="反手口径")
            if len(ro.trades) < 2:
                raise RuntimeError("开了 intrabar_reentry_samebar 却没反手")
            if ro.equity.index.get_loc(ro.trades[1].entry_date) != 7:
                raise RuntimeError(
                    "开了 intrabar_reentry_samebar，空单应该在同一根就开出来")

        # 反方向也要一样：持空单时看不到多头触发价，空单出场之后下一根才做多
        op_s = np.array([100.0] * 6 + [98.0, 96.0, 99.0, 100.0], dtype=float)
        hi_s = np.array([101.0] * 6 + [98.5, 100.0, 101.0, 101.0], dtype=float)
        lo_s = np.array([99.0] * 6 + [94.0, 95.0, 97.0, 98.0], dtype=float)
        cl_s = np.array([100.0] * 6 + [96.0, 97.0, 100.0, 100.0], dtype=float)
        fs = pd.DataFrame({"open": op_s, "high": hi_s, "low": lo_s, "close": cl_s,
                           "signal": np.zeros(10, dtype=int)},
                          index=pd.date_range("2023-01-01", periods=10, freq="D"))
        fs["entry_long"] = np.nan
        fs["entry_short"] = np.nan
        fs["exit_long"] = 0.0
        fs["exit_short"] = 0.0
        fs["stop_dist"] = 4.0
        fs.iloc[6, fs.columns.get_loc("entry_short")] = 95.0   # 第 7 根做空
        fs.iloc[7, fs.columns.get_loc("entry_long")] = 99.0    # 同根多头触发价也被打穿
        fs.iloc[8, fs.columns.get_loc("entry_long")] = 99.0    # 下一根才该开多
        rs = run_backtest(fs, cfg, label="不反手·空")
        if len(rs.trades) != 2 or rs.trades[1].direction != 1:
            raise RuntimeError(
                f"空单出场之后应该只留下一笔下一根的多单：{[(t.direction, str(t.entry_date)) for t in rs.trades]}")
        if int(rs.position.iloc[7]) != 0:
            raise RuntimeError(f"空单止损当根应该空仓，实得 {int(rs.position.iloc[7])}")
        if rs.equity.index.get_loc(rs.trades[1].entry_date) != 8:
            raise RuntimeError("空单出场之后的做多应该在第 9 根，实得 "
                               f"第 {rs.equity.index.get_loc(rs.trades[1].entry_date) + 1} 根")
        rs_opt = run_backtest(fs, cfg_opt, label="反手口径·空")
        if len(rs_opt.trades) < 2 or rs_opt.equity.index.get_loc(
                rs_opt.trades[1].entry_date) != 7:
            raise RuntimeError("开了 intrabar_reentry_samebar，多单应该在同一根就开出来")

        # ---- 进场是「且」不是「或」：成交价必须同时越过两条线 ----
        #
        # 这两个条件都是「价格向上越过某个价位」，引擎取 max（做空取 min）作为
        # 触发价 —— 越过 max 就同时越过了两条。看着像「二选一」的写法，其实
        # 恰是「两个都成立」的盘中判据。所以每笔成交价减两条线都必须 > 0，
        # 有一条 ≤ 0 就说明它只破了其中一条线。
        from ttrade.strategies import run_strategy as _rs

        # 先在**同一份被过滤过的行情**上算，再拿它跑引擎：`run_backtest` 会丢掉
        # 开/收非正的行，用未过滤的数组按下标去对，会整体错位一格以上
        u = _us_df()
        u = u.dropna(subset=["open", "close"])
        u = u[(u["open"] > 0) & (u["close"] > 0)]
        o_ = _rs(u, "sbbox_ib")
        dkey = (pd.DatetimeIndex(u.index) + pd.Timedelta(hours=6)).normalize()
        bh = u["high"].rolling(20, min_periods=20).max().shift(1).to_numpy(float)
        bl = u["low"].rolling(20, min_periods=20).min().shift(1).to_numpy(float)
        dop = u["open"].groupby(dkey).transform("first").to_numpy(float)
        pd_ = pd.DataFrame({"h": u["high"].to_numpy(float),
                            "l": u["low"].to_numpy(float)}, index=dkey)
        rng_ = (pd_["h"].groupby(level=0).max()
                - pd_["l"].groupby(level=0).min()).shift(1)
        ydy = pd.Series(dkey).map(rng_).to_numpy(float)
        ru_ = run_backtest(o_, cfg, label="且/或")
        if not ru_.trades:
            raise RuntimeError("且/或自检样本没成交")
        pos_of = {ts: i for i, ts in enumerate(ru_.equity.index)}
        bad = []
        for tr in ru_.trades:
            i = pos_of[tr.entry_date]
            px = tr.entry_price
            # 用 `>=` 不是 `>`：触发价取的是两条线里更高的那条，成交价正好**落在**
            # 那条线上（滑点为 0 时是精确相等）。「碰到就算突破」是策略自己的口径
            # （`hi >= up`），这里跟着它
            eps = 1e-9
            if tr.direction > 0:
                ok = px >= bh[i] - eps and px >= dop[i] + 0.4 * ydy[i] - eps
            else:
                ok = px <= bl[i] + eps and px <= dop[i] - 0.4 * ydy[i] + eps
            if not ok:
                bad.append((str(tr.entry_date), tr.direction, round(px, 4),
                            round(bh[i], 4), round(dop[i] + 0.4 * ydy[i], 4)))
        if bad:
            raise RuntimeError(
                f"{len(bad)}/{len(ru_.trades)} 笔进场只满足了其中一条线，"
                f"例如 {bad[0]}")

        # ---- 🔴 铁律：进场当根不出场 ----
        #
        # 两根一起验，缺一不可：
        #   ① 真实策略 + 真实行情，**逐笔**都不许出现 bars == 0（当根出场）；
        #   ② 合成样本上证明「只有诊断开关 `intrabar_samebar_stop=True` 能
        #      造出 bars == 0」—— 否则说明还有别的路能当根出场，守卫漏了。
        #
        # `bars = 出场那根 - 进场那根`，当根出场就是 0。未平仓那笔不算（它的
        # `bars` 是「到样本末尾还剩几根」，进场正好落在最后一根时也是 0）。
        bad_same = [tr for tr in ru_.trades
                    if not tr.open_position and tr.bars == 0]
        if bad_same:
            tr = bad_same[0]
            raise RuntimeError(
                f"进场当根出场：{len(bad_same)}/{len(ru_.trades)} 笔在 "
                f"{tr.entry_date} 那根就平了（{tr.exit_note}）—— 进场那一根的 "
                f"high/low 分不出建仓前还是建仓后，不能用来判出场")
        if [tr for tr in r2.trades if not tr.open_position and tr.bars == 0]:
            raise RuntimeError("默认口径下仍出现当根出场")
        if r4.trades[0].bars != 1:
            raise RuntimeError(
                f"下一根止损那笔的持仓根数应是 1，实得 {r4.trades[0].bars}")
        if not [tr for tr in r3.trades if tr.bars == 0]:
            raise RuntimeError(
                "开了诊断开关却没造出当根出场 —— 说明它已经不起作用了，"
                "「进场当根不出场」这条断言也就没法被证伪")

        # ---- 按止损金额定仓：手数必须用这一笔的止损距离算 ----
        # 手数 = risk_amount ÷ (stop_dist × 乘数)。走「先除 stop_mult 还原 ATR
        # 再乘回去」那条老路的话，cfg.stop_mult 和策略参数不一致就会按另一个
        # 风险下单，而且不报错
        cfg_risk = BacktestConfig(initial_cash=1_000_000.0, sizing="risk",
                                  risk_amount=10_000.0, multiplier=10.0,
                                  fee_per_lot=0.0, tick_size=0.0, slippage=0.0,
                                  stop_mult=1.5)
        rr_risk = run_backtest(d, cfg_risk, label="按止损金额")
        want_lots = int(10_000.0 / (4.0 * 10.0))     # stop_dist=4、乘数 10
        if not rr_risk.trades or int(rr_risk.trades[0].lots) != want_lots:
            raise RuntimeError(
                f"按止损金额定仓的手数不对：{rr_risk.trades[0].lots if rr_risk.trades else 0}"
                f"，应为 {want_lots}")
        cfg_risk2 = BacktestConfig(initial_cash=1_000_000.0, sizing="risk",
                                   risk_amount=10_000.0, multiplier=10.0,
                                   fee_per_lot=0.0, tick_size=0.0, slippage=0.0,
                                   stop_mult=99.0)   # 改 stop_mult 不该影响手数
        rr_risk2 = run_backtest(d, cfg_risk2, label="按止损金额·改倍数")
        if int(rr_risk2.trades[0].lots) != want_lots:
            raise RuntimeError(
                f"手数跟着 cfg.stop_mult 变了（{rr_risk2.trades[0].lots} vs {want_lots}）"
                "—— 说明还在按 ATR 反推，没直接用 stop_dist")

        return (f"触发价成交 {t.entry_price:g}、同根止损默认关（当根不扫、"
                f"下一根按 {r4.trades[0].exit_price:g} 扫）、显式打开按 "
                f"{r3.trades[0].exit_price:g} 当根扫、"
                f"进场当根不出场（{len(ru_.trades)} 笔逐笔 bars>0）、"
                f"平仓当根不反手（多/空两侧都验，下一根才开）、"
                f"进场两条线都是「且」（{len(ru_.trades)} 笔逐笔验）、"
                f"按止损金额定仓 {want_lots} 手、无列时走老路径")

    check("K 线内进出场（触发价 / 进场当根不出场 / 老路径不变）", _intrabar)

    def _td_key():
        """交易日归日：sbbox 必须读 `trading_day` 列，不能自己写 `index + 6h`。

        周五 21:00 的夜盘归下周一，这条靠时间推算必错：周五 21:00 + 6h 落到
        **周六**。切错之后不报错，只是周一的「当日开盘」取成周一 09:00 的开盘
        （丢掉隔夜跳空）、「昨日全天振幅」取成周五夜盘那一小段，开仓门槛悄悄
        松掉 —— RB/SA 实测 13.2% / 12.8% 的 bar 受影响。所以这里造三帧：

          a) 无 `trading_day` 列            → 走 +6h 兜底（已知会错）
          b) 列 = +6h 的结果                → 必须与 a 逐值一致（兜底没被改坏）
          c) 列 = 真值（周五夜盘归周一）     → 必须与 a **不同**，且周一那根的
                                              触发价要等于独立重算的值

        最后一条是关键：只验 `_trading_days` 返回对没用，得验策略真的在用它。
        """
        import numpy as np
        import pandas as pd

        from ttrade.strategies import _trading_days, run_strategy

        spec = [("2026-10-23", (9, 10, 11, 13, 14, 21, 22, 23)),   # 周五 + 夜盘
                ("2026-10-26", (9, 10, 11, 13, 14)),               # 周一
                ("2026-10-27", (9, 10, 11, 13, 14, 21, 22, 23))]   # 周二 + 夜盘
        idx, px = [], []
        v = 100.0
        for d, hours in spec:
            for hh in hours:
                idx.append(pd.Timestamp(f"{d} {hh:02d}:00:00"))
                px.append(v)
                v += 0.5
        base = pd.DataFrame({"open": px, "high": [x + 1.0 for x in px],
                             "low": [x - 1.0 for x in px],
                             "close": [x + 0.2 for x in px],
                             "volume": 100}, index=pd.DatetimeIndex(idx))
        plus6 = pd.DatetimeIndex(base.index + pd.Timedelta(hours=6)).normalize()
        fri_night = [i for i, t in enumerate(base.index)
                     if pd.Timestamp("2026-10-23 21:00") <= t
                     <= pd.Timestamp("2026-10-23 23:00")]
        i_mon = list(base.index).index(pd.Timestamp("2026-10-26 09:00"))
        if len(fri_night) != 3:
            raise RuntimeError(f"造的数据不对，周五夜盘应有 3 根，实得 {len(fri_night)}")

        # c 帧：把 +6h 给出的那个周六改回周一（这就是交易所口径）
        truth = [pd.Timestamp("2026-10-26") if t == pd.Timestamp("2026-10-24") else t
                 for t in plus6]
        a = base.copy()
        b = base.copy(); b["trading_day"] = plus6
        c = base.copy(); c["trading_day"] = pd.DatetimeIndex(truth)

        # ① 规则：c 帧下周五夜盘归周一；a/b 帧下落到周六（证明兜底确实是错的）
        if any(pd.Timestamp(_trading_days(c)[i]) != pd.Timestamp("2026-10-26")
               for i in fri_night):
            raise RuntimeError("带 trading_day 列时，周五夜盘没归到周一")
        if any(pd.Timestamp(_trading_days(a)[i]) != pd.Timestamp("2026-10-24")
               for i in fri_night):
            raise RuntimeError("无 trading_day 列时的 +6h 兜底不再落到周六，"
                               "说明兜底被改动了（本测试的前提失效）")

        # ② 兜底没被改坏：列里装 +6h 的结果，输出应与无列完全一致
        kw = {"entry_n": 2, "exit_n": 2, "atr_n": 2, "stop_mult": 1.5}
        oa = run_strategy(a, "sbbox_ib", kw)
        ob = run_strategy(b, "sbbox_ib", kw)
        oc = run_strategy(c, "sbbox_ib", kw)
        fa = oa["entry_long"].fillna(-9e18).to_numpy(float)
        fb = ob["entry_long"].fillna(-9e18).to_numpy(float)
        if not np.allclose(fa, fb):
            raise RuntimeError("列里装的是 +6h 结果，输出却与无列不同 —— 兜底路径变了")

        # ③ 策略真的读了这一列
        fc = oc["entry_long"].fillna(-9e18).to_numpy(float)
        if np.allclose(fa, fc):
            raise RuntimeError("sbbox_ib 没读 trading_day 列（输出与 +6h 一模一样）")

        # ④ 周一 09:00 那根的触发价 = max(箱体上沿, 周五 21:00 开盘 + 0.4×周五全天振幅)
        # 🔴 「周五全天」是**周五这个交易日**（日盘 + 周四夜盘），**不含周五夜盘** ——
        # 周五夜盘属于下周一。把夜盘算进来的话会得出「周五振幅」偏大，正是这个
        # 口径错误的另一种写法。
        box_hi = float(base["high"].iloc[i_mon - 2:i_mon].max())
        do_c = float(base["open"].iloc[fri_night[0]])
        fri_day = [i for i in range(len(base)) if truth[i] == pd.Timestamp("2026-10-23")]
        rng_fri = float(base["high"].iloc[fri_day].max()
                        - base["low"].iloc[fri_day].min())
        want = max(box_hi, do_c + 0.4 * rng_fri)
        got = float(oc["entry_long"].iloc[i_mon])
        if abs(got - want) > 1e-9:
            raise RuntimeError(
                f"周一触发价 {got:.4f} ≠ 独立重算 {want:.4f}"
                f"（箱体 {box_hi:.2f}、周五开盘 {do_c:.2f}、周五振幅 {rng_fri:.2f}）")
        got_a = float(oa["entry_long"].iloc[i_mon])
        return (f"+6h 兜底仍落周六、trading_day 列周五夜盘归周一、"
                f"周一触发价 {got:.2f}（+6h 口径会算成 {got_a:.2f}）")

    check("交易日归日（sbbox 读 trading_day 列、不自己写 +6h）", _td_key)

    def _trade_cost():
        """成本必须**双边**收，逐笔成本加起来要等于绩效页的「总成本」。

        「单边成本」是**每次成交**的口径（一买、一卖各收一次），不是「一开一平
        只收一次」。真收成一次的话，净值会凭空好看一半，而且不报错 —— 只有
        对着手算的数字才发现。所以这里把一笔完整交易的成本拆开对一遍：

            cost = 开仓手续费 + 开仓滑点 + 平仓手续费 + 平仓滑点
            pnl  = 毛盈亏 − cost
            Σpnl = 期末资金 − 初始资金

        最后一条是总账：逐笔金额必须能一路加回账户变化，否则「收益」那一列
        只是好看的数字，对不上钱。
        """
        import numpy as np
        import pandas as pd

        from ttrade.backtest import BacktestConfig, run_backtest

        idx = pd.date_range("2021-01-01", periods=60, freq="D")
        # 两段持仓：先涨后跌。必须**有赚有亏**，否则胜率是 100%，盈利因子和
        # 平均盈亏比都退化成哨兵值，那条恒等式就验不到了
        px = np.r_[np.full(6, 100.0), np.linspace(100, 120, 20), np.full(5, 120.0),
                   np.linspace(120, 95, 20), np.full(9, 95.0)]
        sig = np.zeros(60, dtype=int)
        sig[6:26] = 1
        sig[31:51] = 1
        df = pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px,
                           "signal": sig}, index=idx)
        # 滑点 0、只留比例手续费：这样两侧成本都是「手数 × 乘数 × 价 × 费率」，
        # 能拿价格和手数直接手算
        cfg = BacktestConfig(fee_rate=0.001, slippage=0.0)
        r = run_backtest(df, cfg)
        if not r.trades:
            raise RuntimeError("样本没产生交易，这一项失去意义")
        t = r.trades[0]
        fee_in = t.lots * t.entry_price * 0.001
        fee_out = t.lots * t.exit_price * 0.001
        if abs(t.cost - (fee_in + fee_out)) > 1e-6:
            raise RuntimeError(
                f"逐笔成本不是双边：{t.cost:.4f}，手算 {fee_in + fee_out:.4f}")
        if abs(t.pnl - (t.gross - t.cost)) > 1e-9:
            raise RuntimeError(f"净盈亏不是「毛 − 成本」：{t.pnl:.4f}")
        total = sum(x.cost for x in r.trades)
        if abs(total - r.metrics["total_cost"]) > 1e-6:
            raise RuntimeError(
                f"逐笔成本合计 {total:.4f} ≠ 绩效页总成本 {r.metrics['total_cost']:.4f}")
        if abs(sum(x.pnl for x in r.trades)
               - (r.metrics["final_equity"] - cfg.initial_cash)) > 1e-6:
            raise RuntimeError("逐笔金额加不回账户变化（Σpnl ≠ 期末 − 初始）")

        # 带滑点再跑一遍。这一遍专门抓「毛盈亏用成交价（已含滑点）算」的错 ——
        # 那样滑点会被扣两遍，Σpnl 会比账户变化少「总滑点」那么多，而且不报错。
        # 上面那一遍滑点是 0，正好漏掉这个错
        cfg_s = BacktestConfig(fee_rate=0.001, slippage=0.002)
        rs = run_backtest(df, cfg_s)
        if not rs.trades:
            raise RuntimeError("带滑点的样本没产生交易")
        slip_gap = abs(sum(x.pnl for x in rs.trades)
                       - (rs.metrics["final_equity"] - cfg_s.initial_cash))
        if slip_gap > 1e-6:
            raise RuntimeError(
                f"带滑点时逐笔金额对不上账户变化，差 {slip_gap:.4f}"
                f"（总成本 {rs.metrics['total_cost']:.4f}）—— 滑点被算了两遍？")
        if abs(sum(x.cost for x in rs.trades) - rs.metrics["total_cost"]) > 1e-6:
            raise RuntimeError("带滑点时逐笔成本合计 ≠ 总成本")

        # 盈利因子 = 胜率/(1−胜率) × 平均盈亏比。两条公式必须自洽，否则界面上
        # 那两个数摆在一起会互相矛盾
        m = r.metrics
        if m["n_trades"] and 0 < m["win_rate"] < 1 and 0 < m["payoff"] < 999:
            lhs = m["profit_factor"]
            rhs = (m["win_rate"] / (1 - m["win_rate"])) * m["payoff"]
            if abs(lhs - rhs) > 1e-6 * max(1.0, abs(lhs)):
                raise RuntimeError(
                    f"盈利因子 {lhs:.4f} ≠ 胜率/(1−胜率)×平均盈亏比 {rhs:.4f}")
        return (f"双边 {t.cost:.2f}（手算 {fee_in + fee_out:.2f}）、"
                f"Σ成本=总成本、Σ净额=账户变化（含滑点也对得上）、盈利因子自洽")

    check("交易成本双边与金额口径", _trade_cost)

    def _return_basis():
        """业绩统计全套单利：本金固定，所有比例都以本金为分母。

        框架是**单利**框架 —— 本金和使用资金固定不变，赚到的钱不会自动变成
        下一笔的仓位。所以：

            累计收益 = (期末权益 − 本金) ÷ 本金
            年化收益 = 累计收益 ÷ 年数            （不取 1/年数 次方）
            逐根收益 = 当日盈亏 ÷ 本金
            最大回撤 = (权益 − 历史峰值) ÷ 本金
            卡玛     = 单利年化 ÷ |最大回撤|

        分母全是本金 —— 回撤要是写成「相对峰值」，赚了一半之后同样的回落只
        会写出 −13%（而不是 −20%），回撤会随着盈利越变越好看，卡玛的分子
        分母也就不是一个基准了。

        外加阶段分析：月度/年度收益也是「段盈亏 ÷ 本金」，所以**十二个月相加
        等于年度、各年相加等于累计**。这条最容易被改回复利（除以段初权益），
        一旦回去，月表上的数字和年度柱状图就对不上了。
        """
        import numpy as np
        import pandas as pd

        from gui.result_view import cumulative_pnl
        from ttrade.backtest import (BacktestConfig, month_matrix, run_backtest,
                                     stage_table)

        # 三年、两段上涨夹一段下跌：既有多笔交易，又有明显的回撤段
        n = 780
        idx = pd.date_range("2019-01-01", periods=n, freq="D")
        drift = np.r_[np.full(240, 0.0009), np.full(180, -0.0006),
                      np.full(360, 0.0005)]
        px = 100 * np.exp(np.cumsum(drift))
        # 信号滞后 60 根：进场晚、离场也晚，正好让它**扛着一段下跌**，
        # 权益上才有看得见的回撤可验（不滞后的话下跌段全程空仓，回撤接近 0）
        sig = pd.Series((drift > 0).astype(int), index=idx).shift(60).fillna(0).astype(int)
        df = pd.DataFrame({"open": px, "high": px + 1, "low": px - 1,
                           "close": px, "signal": sig}, index=idx)
        cfg = BacktestConfig(fee_rate=0.0005, slippage=0.0)
        r = run_backtest(df, cfg)
        m = r.metrics
        eq = r.equity.dropna()
        if len(r.trades) < 2:
            raise RuntimeError("样本交易太少，这一项验不出东西")

        # 本金：绩效表第一行读它，也是下面所有比例的分母
        if abs(float(m.get("initial_cash", 0.0)) - cfg.initial_cash) > 1e-9:
            raise RuntimeError(f"本金不是 initial_cash：{m.get('initial_cash')}")

        base = float(m["initial_cash"])
        total = float(m["total_return"])
        if abs(total - (float(eq.iloc[-1]) - base) / base) > 1e-9:
            raise RuntimeError("累计收益不是「(期末 − 本金) ÷ 本金」")
        # 金额口径：逐笔净额相加 ÷ 本金。和上面那个必须是同一个数，
        # 否则交易记录和绩效页会各说一套
        by_money = sum(t.pnl for t in r.trades) / base
        if abs(total - by_money) > 1e-9:
            raise RuntimeError(
                f"累计收益 {total:.6f} ≠ 逐笔净额合计 ÷ 本金 {by_money:.6f}")
        cums = cumulative_pnl(r.trades)
        if abs(cums[-1] - (m["final_equity"] - base)) > 1e-6:
            raise RuntimeError("交易记录的累计盈亏 ≠ 期末 − 初始")

        # 单利年化：累计 ÷ 年数。写成 (1+累计)^(1/年数)−1 就是复利，直接判死
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        if abs(m["cagr"] - total / years) > 1e-9:
            raise RuntimeError(
                f"年化 {m['cagr']:.6f} ≠ 累计 ÷ 年数 {total / years:.6f}（又用复利折年了？）")

        # 波动 / 夏普走「当日盈亏 ÷ 本金」，和年化收益同一个分母
        step = eq.diff().fillna(0.0) / base
        want_vol = float(step.std(ddof=1) * np.sqrt(cfg.bars_per_year))
        if abs(m["ann_vol"] - want_vol) > 1e-9:
            raise RuntimeError(
                f"年化波动 {m['ann_vol']:.6f} ≠ 逐根收益率年化 {want_vol:.6f}")
        want_sharpe = float(step.mean() * cfg.bars_per_year / want_vol)
        if abs(m["sharpe"] - want_sharpe) > 1e-9:
            raise RuntimeError(f"夏普 {m['sharpe']:.4f} ≠ {want_sharpe:.4f}")

        want_dd = float(((eq - eq.cummax()) / base).min())
        if abs(m["max_dd"] - want_dd) > 1e-9:
            raise RuntimeError(
                f"最大回撤 {m['max_dd']:.6f} ≠ (权益−峰值)/本金 {want_dd:.6f}")
        if abs(m["calmar"] - m["cagr"] / abs(m["max_dd"])) > 1e-9:
            raise RuntimeError("卡玛不是「单利年化 ÷ |最大回撤|」")

        # 阶段分析：各年相加 = 累计；每年的十二个月相加 = 该年
        yrs = stage_table(eq, "YE", base=base)
        if yrs.empty:
            raise RuntimeError("阶段表是空的")
        if abs(float(yrs["ret"].sum()) - total) > 1e-9:
            raise RuntimeError(
                f"各年收益相加 {yrs['ret'].sum():.6f} ≠ 累计收益 {total:.6f}")
        months = month_matrix(eq, base=base)
        for _, row in yrs.iterrows():
            y = row["period"].year
            if y not in months.index:
                continue
            got = float(months.loc[y].sum(skipna=True))
            if abs(got - float(row["ret"])) > 1e-9:
                raise RuntimeError(
                    f"{y} 年十二个月相加 {got:.6f} ≠ 年度收益 {row['ret']:.6f}")

        # 防回归：累计收益**不是**逐笔收益率相加。这个样本有赚有亏，两种算法
        # 必然分家；一旦哪天改回相加，这一条会先炸
        by_pct = sum(t.ret for t in r.trades)
        if abs(total - by_pct) < 1e-6:
            raise RuntimeError(
                f"累计收益等于逐笔收益率相加（{by_pct:.4f}）—— 口径退回百分比相加了？")
        return (f"本金 {base:,.0f}、累计 {total*100:+.1f}%、年化 {m['cagr']*100:+.2f}%"
                f"（累计÷{years:.2f}年）、回撤 {m['max_dd']*100:.1f}%、"
                f"月度相加=年度、年度相加=累计")

    def _cross_trade_money():
        """截面策略的逐笔金额也要能加回净值变化。

        截面没有「一手」的概念，仓位是按权重挂着的，所以金额只能从
        「上根净值 × 上根权重 × 这根收益」累出来，成本按换手拆。这里最容易
        错的是换手怎么分：反手那一根 `|Δw|` 是两倍持仓，一半平旧仓、一半开
        新仓 —— 用 `0.5 × |Δw|` 图省事的话，空仓开仓 / 平回空仓时对面没有段，
        逐笔成本合计会只剩总成本的一半，而且没有任何地方会报错。
        """
        import numpy as np
        import pandas as pd

        from ttrade import cross as CX
        from ttrade.backtest import BacktestConfig

        rng = np.random.default_rng(7)
        idx = pd.date_range("2023-01-02", periods=260, freq="B")
        frames = {}
        for k in range(6):
            r = rng.normal(0.0002, 0.012, len(idx))
            px = 100 * np.exp(np.cumsum(r))
            frames[f"S{k}_SHFE"] = pd.DataFrame(
                {"open": px * (1 + rng.normal(0, 0.001, len(idx))),
                 "high": px * 1.01, "low": px * 0.99, "close": px,
                 "volume": rng.integers(1e4, 1e5, len(idx)).astype(float),
                 "open_interest": rng.integers(1e4, 1e5, len(idx)).astype(float)},
                index=idx)
        cfg = CX.CrossConfig(mode="hedge", factor_mode="single",
                             factor_keys=["mom_20"], top_pct=30.0, bot_pct=30.0,
                             weight="equal", rebalance=5)
        bt = BacktestConfig(initial_cash=500_000.0, fee_rate=0.0003, slippage=0.0)
        out = CX.run(frames, cfg, bt)
        if not out.ok:
            raise RuntimeError(f"截面回测没跑通：{out.error}")
        res = CX.to_result(out, label="截面", step=out.step, bpy=252.0,
                           initial_cash=bt.initial_cash)
        if not res.trades:
            raise RuntimeError("截面回测没产生交易")
        cost_gap = abs(sum(t.cost for t in res.trades) - res.metrics["total_cost"])
        pnl_gap = abs(sum(t.pnl for t in res.trades)
                      - (res.metrics["final_equity"] - bt.initial_cash))
        if cost_gap > 1e-3:
            raise RuntimeError(
                f"截面逐笔成本合计对不上总成本，差 {cost_gap:.4f}")
        if pnl_gap > 1e-3:
            raise RuntimeError(
                f"截面逐笔净额对不上净值变化，差 {pnl_gap:.4f}")

        # ---- 单利 + 逐品种
        #
        # 组合净值必须是 `1 + Σ各腿净收益`（单利），不是 `(1+net)` 连乘 ——
        # 复利下本金涨上去，同样的盈亏会被算成更小的百分比，期末资金、成本、
        # 夏普、卡玛跟着全偏。而且各腿相加必须**正好回到组合**，否则绩效页
        # 切到单品种看到的是另一套数
        m = res.metrics
        leg_sum = sum(r.metrics["total_return"] for r in res.legs.values())
        if abs(leg_sum - m["total_return"]) > 1e-9:
            raise RuntimeError(
                f"各腿累计相加 {leg_sum:.6f} ≠ 组合累计 {m['total_return']:.6f}")
        leg_cost = sum(r.metrics["total_cost"] for r in res.legs.values())
        if abs(leg_cost - m["total_cost"]) > 1e-3:
            raise RuntimeError(
                f"各腿成本相加 {leg_cost:.2f} ≠ 组合总成本 {m['total_cost']:.2f}")
        for sym, lr in res.legs.items():
            lm = lr.metrics
            if abs(float(lm["initial_cash"]) - bt.initial_cash) > 1e-6:
                raise RuntimeError(f"{sym} 的本金不是 {bt.initial_cash}")
            want = (1.0 + float(lm["total_return"])) * bt.initial_cash
            if abs(float(lm["final_equity"]) - want) > 1e-3:
                raise RuntimeError(
                    f"{sym} 期末资金 {lm['final_equity']:.2f} ≠ 本金×(1+累计) {want:.2f}")
        # 年化必须是单利：累计 ÷ 年数
        yrs = (res.equity.index[-1] - res.equity.index[0]).days / 365.25
        if abs(m["cagr"] - m["total_return"] / yrs) > 1e-9:
            raise RuntimeError(
                f"截面的年化 {m['cagr']:.6f} ≠ 累计 ÷ 年数 {m['total_return']/yrs:.6f}")
        return (f"{len(res.trades)} 笔，Σ成本=总成本、Σ净额=净值变化、"
                f"{len(res.legs)} 个品种各腿相加=组合、年化走单利")

    check("截面逐笔金额口径", _cross_trade_money)

    check("收益与年化口径一致", _return_basis)

    def _lib_name_rules():
        """策略库存入时的两种重名。

        规则不一样，混在一起就一定会有一边做错：

        - 撞**内置模板**的名字 → 不许存。左侧策略库把内置和「我的」分两组列，
          同名之后谁也指代不清
        - 撞**自己库里**的一条 → 可以存，覆盖它。覆盖要**原地替换**（位置、
          创建时间、sid 都不变），换个新 sid 的话组合和 Agent 那边就指向一个
          不存在的 id 了
        """
        import tempfile

        from ttrade import strategy_store as S

        tmp = tempfile.mkdtemp(prefix="cq_lib_")
        old = S.LIB_FILE
        S.LIB_FILE = os.path.join(tmp, "lib.json")
        try:
            def mk(name):
                return S.StrategyConfig(symbol="RB_SHFE", name=name)

            first = S.add_to_library(mk("压测用·白银动量60-1h"))
            try:
                S.add_to_library(mk("压测用·白银动量60-1h"))
                raise RuntimeError("同名自定义策略没被拦下来")
            except S.NameConflict as exc:
                if exc.builtin or exc.item is None:
                    raise RuntimeError(f"自定义重名被判成了内置：{exc}")
            again = S.add_to_library(mk("压测用·白银动量60-1h"), overwrite=True)
            lib = S.load_library()
            if len(lib) != 1:
                raise RuntimeError(f"覆盖之后库里变成了 {len(lib)} 条")
            if lib[0].sid != first.sid or again.sid != first.sid:
                raise RuntimeError("覆盖换了 sid（组合和 Agent 那边会指向不存在的 id）")

            # 改名撞车：把这一条改叫别人的名字。要拦，而且**不能**给 item ——
            # 给了界面就会弹「要覆盖吗」，点了覆盖又回到同一条分支，转一圈
            # 还是存不进去
            S.add_to_library(mk("压测用·碳酸锂动量60-5m"))
            first.name = "压测用·碳酸锂动量60-5m"
            try:
                S.add_to_library(first)
                raise RuntimeError("改名撞车没被拦下来")
            except S.NameConflict as exc:
                if exc.item is not None or exc.builtin:
                    raise RuntimeError("改名撞车不该让界面去问「覆盖吗」")
            first.name = "压测用·白银动量60-1h"

            tpl_name = S.TPL.names()[0]
            try:
                S.add_to_library(mk(tpl_name))
                raise RuntimeError("和内置策略重名却存进去了")
            except S.NameConflict as exc:
                if not exc.builtin:
                    raise RuntimeError(f"内置重名没标 builtin：{exc}")

            # 从模板另存是「另存到我的」，不该被自己那条内置名挡住，自动编号
            t = S.TPL.to_config(S.TPL.keys()[0])
            t.symbol = "RB_SHFE"
            copy = S.add_to_library(t)
            if S.norm_name(copy.title()) in S.builtin_names():
                raise RuntimeError(f"模板副本的名字和内置撞上了：{copy.title()}")
            return (f"自定义重名要确认、覆盖原地替换（{len(lib)} 条、sid 不变）、"
                    f"内置重名拦住（{tpl_name}）、模板副本自动编号为「{copy.title()}」")
        finally:
            S.LIB_FILE = old

    check("策略库存入的重名规则", _lib_name_rules)

    def _factor_lib():
        from ttrade import factors as F

        df = _us_df()
        # 美股没有持仓量，oi_change 这类因子本来就不适用，不算失败
        avail = F.filter_available(df, list(F.FACTORS))
        bad = [k for k in avail if F.raw(df, k).notna().sum() < 50]
        if bad:
            raise RuntimeError(f"这些因子在 {len(df)} 行样本上算不出值：{bad}")
        missing = [k for k in F.FACTORS if k not in avail]
        extra = f"（{missing} 该市场无数据）" if missing else ""
        return f"{len(avail)}/{len(F.FACTORS)} 个因子可算{extra}，样本 {len(df)} 行"

    check("因子库", _factor_lib)

    def _aux_data():
        """衍生数据：期限结构 / 期权摘要能否挂到行情上，展期因子算不算得对。"""
        import pandas as pd

        from ttrade import factors as F
        from ttrade.auxdata import summary as aux_summary
        from ttrade.futures import load_local as load_fut

        info = aux_summary(data_dir("aux"))
        if not info["curve"]:
            # 衍生数据是可选的：不下载也能用，只是展期/期权因子没有值
            return "未下载（download_aux.py 可生成），展期/期权因子不可用"

        sym, ex = "CU", "SHFE"
        if not os.path.exists(os.path.join(data_dir("cn"), f"{sym}_{ex}.csv")):
            cands = [f[:-4] for f in os.listdir(data_dir("cn"))
                     if f.endswith(".csv") and not f.startswith("_") and "_" in f]
            if not cands:
                raise FileNotFoundError("没有可用的期货 CSV")
            sym, ex = sorted(cands)[0].rsplit("_", 1)

        df = load_fut(sym, ex, data_dir("cn"))
        miss = [c for c in ("c1_close", "c2_close", "c2_dte") if c not in df.columns]
        if miss:
            raise RuntimeError(f"{sym}_{ex} 没挂上衍生列（缺 {miss}）")

        ry = F.raw(df, "roll_yield")
        if ry.notna().sum() < 20:
            raise RuntimeError(f"{sym}_{ex} 的展期收益率只有 {ry.notna().sum()} 个有效值")

        # 年化 = 收益率 × 365 / 两个合约的到期间隔。这条恒等式只要错，说明
        # 期限结构的列错位了（比如把 c1 当成了 c2），比「有值」更能说明问题。
        ann = F.raw(df, "roll_annual")
        gap = (pd.to_numeric(df["c2_dte"], errors="coerce")
               - pd.to_numeric(df["c1_dte"], errors="coerce")).clip(lower=1.0)
        err = (ann - ry * 365.0 / gap).abs().max()
        if pd.notna(err) and err > 1e-9:
            raise RuntimeError(f"展期年化与收益率不成比例，最大偏差 {err:.2e}")

        # 三套口径的结构性检查。这里查的不是「有没有值」，而是口径本身对不对：
        # 同一套曲线里三腿的到期日必须递增，且筛过持仓量的近腿不可能比 c1 更近
        # （c1 是全品种最近的那个合约，筛活跃只会在候选里挑更远的）。
        stale = info.get("stale", 0)
        n_o3 = F.raw(df, "roll_yield_o3").notna().sum()
        if not n_o3:
            raise RuntimeError(
                f"{sym}_{ex} 没有 OI 口径的期限结构（{stale} 个 aux 文件是旧版本），"
                "跑 rebuild_aux.py 或重新 download_aux.py"
            )
        for pre in ("c", "o3_", "o6_"):
            d = [pd.to_numeric(df[f"{pre}{i}_dte"], errors="coerce") for i in (1, 2, 3)]
            for i in range(2):
                bad = (d[i] > d[i + 1]).sum()
                if bad:
                    raise RuntimeError(f"{pre} 曲线的到期日没有递增，{bad} 行异常")
        # 筛活跃只会把近腿推远，所以 o6_1 一定不比 c1 更近，o3_1 也不比 o6_1 更近
        c1d = pd.to_numeric(df["c1_dte"], errors="coerce")
        o3d = pd.to_numeric(df["o3_1_dte"], errors="coerce")
        o6d = pd.to_numeric(df["o6_1_dte"], errors="coerce")
        if (o3d < c1d).any() or (o6d < c1d).any():
            raise RuntimeError("OI 口径的近腿比 c1 还近，选合约的逻辑有问题")
        if (o3d < o6d).any():
            raise RuntimeError("OI3 的近腿比 OI6 还近：候选集越小只会更远")

        opt = f"，期权 {info['opt']} 个品种" if info["opt"] else "，期权尚未下载"
        stale_note = f"，{stale} 个 aux 文件待重建" if stale else ""
        return (f"{info['curve']} 个品种{opt}，{info['dates']} 个交易日"
                f"（{info['first']}~{info['last']}）；{sym}_{ex} 展期 {ry.notna().sum()} 个值"
                f"，均值 {ry.mean() * 100:+.2f}%；三套口径 "
                f"{ry.notna().sum()}/{n_o3}/{F.raw(df, 'roll_yield_o6').notna().sum()}"
                f"{stale_note}")

    check("衍生数据（展期 / 期权因子）", _aux_data)

    def _factor_metrics():
        from ttrade import factor_metrics as M
        from ttrade import factors as F

        df = _us_df()
        rep = M.factor_report(df, list(F.FACTORS), horizon=5, window=252)
        if len(rep) != len(F.FACTORS):
            raise RuntimeError(f"评估表只有 {len(rep)} 行，应为 {len(F.FACTORS)}")
        if rep["ic"].isna().all():
            raise RuntimeError("所有因子的 IC 都是 NaN")
        return f"{len(rep)} 个因子，|IC| 最大 {rep['ic'].abs().max():.3f}"

    check("因子评估（IC/ICIR/Newey-West）", _factor_metrics)

    def _factor_profile():
        """横截面口径的几个纯函数 + 手册 HTML。

        **不跑整库体检**：那要十几秒、还要 18 个品种的日线，放进自检会让
        每次启动慢一大截。这里只验最容易写错的三处 —— 按行的秩相关、半衰期
        插值、速度分档，以及手册能不能拼出 HTML（少一个 key 会直接 KeyError）。
        """
        import numpy as np
        import pandas as pd

        from ttrade import factor_pdf as PDF
        from ttrade import factor_profile as FP

        # 至少 5 列：横截面 IC 要求一天有 MIN_SYMBOLS 个品种才排名
        idx = pd.bdate_range("2024-01-01", periods=60)
        base = np.arange(60.0)
        cols = {f"S{i}": base * k for i, k in enumerate((1, 2, 3, 4, 5, 6))}
        a = pd.DataFrame(cols, index=idx)
        b = pd.DataFrame(cols, index=idx)
        ic = FP._row_spearman(a, b).dropna()
        if not (ic > 0.99).all():
            raise RuntimeError(f"完全同序的三列秩相关应为 1，实得 {ic.iloc[0]:.3f}")

        # 半衰期：IC 从 0.04 线性掉到 0.02 在 h=5 → 半衰期正好 5
        hl = FP._half_life({1: 0.04, 2: 0.035, 5: 0.02, 10: 0.01})
        if abs(hl - 5.0) > 0.01:
            raise RuntimeError(f"半衰期应为 5.0，实得 {hl:.2f}")
        if FP._speed_class(2.0, 0.03)[0] != "快":
            raise RuntimeError("半衰期 2 天应判为「快」")
        if FP._speed_class(2.0, 0.001)[0] != "信号弱":
            raise RuntimeError("IC 只有 0.001 时不该给速度结论")

        res = FP.ProfileResult(
            meta={"generated": "2024-01-01 00:00", "symbols": ["RB_SHFE"],
                  "n_factors": 1, "span": ["2020-01-01", "2024-01-01"]},
            factors=[{"key": "mom_60", "label": "动量 60 日", "category": "趋势",
                      "desc": "测试", "formula": "ret(60)", "ic_mean": 0.02,
                      "icir": 0.05, "t": 2.5, "pos": 0.55, "half_life": 12.0,
                      "speed": "中", "period": "日线", "turnover": 0.05,
                      "ts_agree": 0.6, "year_ratio": 0.7, "robust": 60.0,
                      "best_symbols": ["RB_SHFE"], "mates": [], "win_ic": {},
                      "no_daily": False}],
            combos=[], redundant=[])
        html = PDF.build_html(res)
        if "因子库深度分析手册" not in html or "动量 60 日" not in html:
            raise RuntimeError("手册 HTML 里没有标题或因子名")
        return f"秩相关 {ic.iloc[0]:.2f}，半衰期 {hl:.1f} 日，手册 {len(html):,} 字符"

    check("因子体检口径（横截面 / 半衰期 / 手册）", _factor_profile)

    def _factor_strategy():
        from ttrade import factor_strategy as FS
        from ttrade import factors as F

        df = _us_df()
        cfg = FS.FactorConfig(window=252, horizon=5)
        out = FS.run_factor_strategy(df, "single", [F.DEFAULT_SINGLE], cfg)
        held = int((out.df["signal"] != 0).sum())
        if held == 0:
            raise RuntimeError("单因子策略没有产生任何持仓")
        if "factor_score" not in out.df.columns:
            raise RuntimeError("策略输出缺少 factor_score 列")
        return f"单因子 · {F.FACTORS[F.DEFAULT_SINGLE].label}，{held} 根持仓"

    check("因子策略（单因子）", _factor_strategy)

    def _factor_exit_z():
        """离场阈值取负：放宽的是离场、拉长的是持仓，不能顺手把进场也改了。"""
        import numpy as np

        from ttrade import factor_strategy as FS

        # 1. 负值不能被归一化夹回 0。界面上拖到 -1、跑出来还是 0，是那种
        #    「改了参数但结果一模一样」的静默失败，最难查
        if abs(FS.FactorConfig(exit_z=-1.0).normalized().exit_z + 1.0) > 1e-9:
            raise RuntimeError("离场阈值 -1.0 被归一化改掉了")
        if abs(FS.FactorConfig(exit_z=-5.0).normalized().exit_z + 2.0) > 1e-9:
            raise RuntimeError("离场阈值下限应为 -2")
        # 2. 上限仍压在进场阈值以内：倒挂之后分数刚越过进场线就满足出场条件，
        #    信号每根都在开平之间抖
        if abs(FS.FactorConfig(entry_z=0.5, exit_z=2.0).normalized().exit_z - 0.5) > 1e-9:
            raise RuntimeError("离场阈值没有压在进场阈值以内")

        # 3. 多单：分数 +2 站住，回落到 -0.5，再跌到 -1.5。
        #    exit=0.3 在回落那根就平；exit=-1.0 要等到 -1.5。
        #    看「最后一根多头」而不是「第一根空仓」：分数直接跌到 -1.5 时
        #    同一根就反手做空了，pos 是 -1 不是 0
        s = np.r_[np.full(10, np.nan), np.full(6, 2.0),
                  np.full(6, -0.5), np.full(4, -1.5)]
        tight = FS.hysteresis(s, 1.0, 0.3)
        loose = FS.hysteresis(s, 1.0, -1.0)
        k_tight = int(np.max(np.flatnonzero(tight == 1)))
        k_loose = int(np.max(np.flatnonzero(loose == 1)))
        if (k_tight, k_loose) != (15, 21):
            raise RuntimeError(
                f"负阈值没拉长持仓：exit=0.3 拿到第 {k_tight} 根、"
                f"exit=-1 拿到第 {k_loose} 根（应为 15 / 21）")

        # 4. 空头要对称：分数 -2 站住，回升到 +0.5，再涨到 +1.5
        s2 = np.r_[np.full(10, np.nan), np.full(6, -2.0),
                   np.full(6, 0.5), np.full(4, 1.5)]
        a2 = FS.hysteresis(s2, 1.0, 0.3)
        b2 = FS.hysteresis(s2, 1.0, -1.0)
        if (int(np.max(np.flatnonzero(a2 == -1))),
                int(np.max(np.flatnonzero(b2 == -1)))) != (15, 21):
            raise RuntimeError("空头的负离场阈值不对称")

        # 5. 实盘逐根推进的那份状态机必须和回测同一套判据。两处各写一遍、
        #    只有一处放开了负值的话，回测拿着、实盘平掉，没有任何线索可查
        import types

        from ttrade import ml_artifact as MA

        fake = types.SimpleNamespace(
            cfg=FS.FactorConfig(entry_z=1.0, exit_z=-1.0).normalized(),
            state=types.SimpleNamespace(cur=0),
        )
        step = types.MethodType(MA.LivePredictor._hysteresis_step, fake)
        live = np.asarray([step(float(v)) for v in s], dtype=int)
        if not np.array_equal(live, loose):
            raise RuntimeError("实盘状态机和回测 hysteresis 在负阈值下不一致")
        return (f"归一化保留 −1、下限 −2、上限压进场；多空都拿到第 21 根"
                f"（exit=0.3 只到第 15 根）；实盘同步")

    check("离场阈值取负（持仓更长）", _factor_exit_z)

    def _factor_no_score():
        """禁止分数出场：离场阈值不参与，但反向信号和 ATR 规则照旧。"""
        import numpy as np
        import pandas as pd

        from ttrade import factor_strategy as FS

        # 1. 归一化。一条出场规则都没有的组合要自动关掉 —— 否则仓位会一直
        #    拿到回测结束，而界面上那个勾还亮着
        if FS.FactorConfig(exit_mode="score", exit_no_score=True).normalized().exit_no_score:
            raise RuntimeError("没有任何 ATR 规则时「禁止分数出场」应该被关掉")
        if not FS.FactorConfig(exit_mode="score", exit_sl=True,
                               exit_no_score=True).normalized().exit_no_score:
            raise RuntimeError("只开止损时「禁止分数出场」是有效配置，不该被关掉")
        if not FS.FactorConfig(exit_mode="tier",
                               exit_no_score=True).normalized().exit_no_score:
            raise RuntimeError("选了分档回撤时「禁止分数出场」不该被关掉")

        # 2. 分数回落到阈值内：默认口径平仓，勾了「禁止分数出场」就不平
        n = 40
        px = np.r_[np.full(10, 100.0), 100 + np.arange(1, 11), np.full(20, 110.0)]
        df = pd.DataFrame({"open": px, "high": px + 0.5, "low": px - 0.5,
                           "close": px},
                          index=pd.date_range("2021-01-01", periods=n, freq="D"))
        sig = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int),
                    np.zeros(20, dtype=int)]
        cut, _ = FS.apply_exit_rules(sig, df, FS.FactorConfig(exit_mode="tier"))
        if (cut[20:] != 0).any():
            raise RuntimeError("没勾「禁止分数出场」时应该照常按离场阈值平仓")
        hold, desc = FS.apply_exit_rules(
            sig, df, FS.FactorConfig(exit_mode="tier", exit_no_score=True))
        if (hold[20:] != 1).any():
            k = int(np.argmax(hold[20:] != 1)) + 20
            raise RuntimeError(f"勾了「禁止分数出场」还在按离场阈值平仓：第 {k} 根")
        if "不按分数出场" not in desc:
            raise RuntimeError(f"摘要没写「不按分数出场」：{desc!r}")

        # 3. 分数**反向**越过进场阈值仍然平仓：那是反向信号，不是离场阈值
        rev = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int),
                    np.full(20, -1, dtype=int)]
        out_rev, _ = FS.apply_exit_rules(
            rev, df, FS.FactorConfig(exit_mode="tier", exit_no_score=True))
        if (out_rev[20:] != -1).any():
            raise RuntimeError("分数反向时仍然要平仓（拿着反方向的仓等 ATR 太危险）")

        # 4. 只勾止损时，止损就是唯一一条出场线 —— 信号全程 +1 也照样被平
        px2 = np.r_[np.full(10, 100.0), np.full(10, 100.0), np.full(20, 97.0)]
        df2 = pd.DataFrame({"open": px2, "high": px2 + 0.5, "low": px2 - 0.5,
                            "close": px2},
                           index=pd.date_range("2021-01-01", periods=n, freq="D"))
        out_sl, _ = FS.apply_exit_rules(
            np.ones(n, dtype=int), df2,
            FS.FactorConfig(exit_mode="score", exit_sl=True, exit_sl_atr=2.0,
                            exit_no_score=True))
        if (out_sl[20:] != 0).any():
            raise RuntimeError("禁止分数出场后，止损这条线没有生效")
        return (f"归一化拦掉「无出场规则」；回落不平（第 20 根起仍持有）；"
                f"反向照平；止损 2ATR 独立生效")

    check("禁止分数出场", _factor_no_score)

    def _factor_no_flip():
        """反向也不平仓：分数翻向不换仓，只认 ATR。"""
        import numpy as np
        import pandas as pd

        from ttrade import factor_strategy as FS

        # 1. 归一化。反向都不平，回落自然也不平；没有 ATR 规则时两条都关掉
        c = FS.FactorConfig(exit_mode="tier", exit_no_flip=True).normalized()
        if not (c.exit_no_flip and c.exit_no_score):
            raise RuntimeError("勾了「反向也不平」应同时打开「不按分数出场」")
        c2 = FS.FactorConfig(exit_mode="score", exit_no_flip=True).normalized()
        if c2.exit_no_flip or c2.exit_no_score:
            raise RuntimeError("没有任何 ATR 规则时，两条都该被关掉")

        n = 40
        idx = pd.date_range("2021-01-01", periods=n, freq="D")

        def mk(px):
            return pd.DataFrame({"open": px, "high": px + 0.5, "low": px - 0.5,
                                 "close": px}, index=idx)

        # 2. 分数从 +1 直接翻到 -1（中间不经过 0）
        px = np.r_[np.full(10, 100.0), np.linspace(100, 106, 10), np.full(20, 106.0)]
        df = mk(px)
        sig = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int),
                    np.full(20, -1, dtype=int)]
        # 只拦回落：反向照平（翻成 -1）
        out_a, _ = FS.apply_exit_rules(
            sig, df, FS.FactorConfig(exit_mode="tier", exit_no_score=True))
        if (out_a[20:] != -1).any():
            raise RuntimeError("只勾「不使用离场阈值」时，反向仍然要平仓")
        # 反向也不平：仓位一直是 +1
        out_b, desc = FS.apply_exit_rules(
            sig, df, FS.FactorConfig(exit_mode="tier", exit_no_flip=True))
        if (out_b[20:] != 1).any():
            k = int(np.argmax(out_b[20:] != 1)) + 20
            raise RuntimeError(f"勾了「反向也不平仓」还在换仓：第 {k} 根")
        if "反向也不平仓" not in desc:
            raise RuntimeError(f"摘要没写「反向也不平仓」：{desc!r}")

        # 3. ATR 平掉之后，分数翻向要能开新仓 —— 否则 no_flip 下分数一路
        #    反向不回头，仓位就永久空着（止损一次、策略作废）
        px2 = np.r_[np.full(10, 100.0), np.linspace(100, 112, 12),
                    np.linspace(112, 100, 12), np.full(6, 100.0)]
        sig2 = np.r_[np.zeros(10, dtype=int), np.ones(12, dtype=int),
                     np.full(18, -1, dtype=int)]
        out_c, _ = FS.apply_exit_rules(
            sig2, mk(px2), FS.FactorConfig(exit_mode="tier", exit_no_flip=True))
        if not (out_c[22:] == -1).any():
            raise RuntimeError(
                "ATR 平掉之后分数翻向没能开新仓（no_flip 下仓位会永久空着）")
        return ("归一化：反向不平隐含不按分数出场、无 ATR 规则时全关；"
                "回落与反向都不平（第 20 根起仍持有 +1）；"
                "ATR 平掉后反向信号能开新仓")

    check("反向也不平仓", _factor_no_flip)

    def _factor_cross_period():
        """跨周期因子：大周期的 bar 只在收完之后才许被小周期看到。

        这一段测的是**未来数据**，不是「能不能算出来」。做法是拿真行情跑一遍，
        然后把某个时刻之后的大周期数据全改成垃圾，再看那个时刻之前的对齐值
        有没有变 —— 变了就说明它偷看了未来。朴素做法（直接把周线的时间戳
        当可用时刻、或者按 `reindex(ffill)` 贴）在这里一定会挂。
        """
        import numpy as np
        import pandas as pd
        from ttrade import factor_strategy as FS
        from ttrade import factors as F
        from ttrade import tqdata

        # 基准用 5 分钟（小周期调大周期，最容易踩未来数据），源用周线
        base = tqdata.load_main("RB", "SHFE", "5m", adjust="qfq").tail(6000)
        week = tqdata.load_main("RB", "SHFE", "1w", adjust="qfq")
        if len(base) < 1000 or len(week) < 300:
            raise RuntimeError("行情不够，跨周期自检跑不了")

        z = F.standardize(F.raw(week, "mom_20"), window=252, method="z")
        got = FS.align_to_base(z, base.index, "5m", "1w")

        # 1. 未来扰动：T 之后改成 99，T 之前一个数都不许动
        T = base.index[-300]
        z2 = z.copy()
        z2.loc[z2.index > T] = 99.0
        got2 = FS.align_to_base(z2, base.index, "5m", "1w")
        mask = base.index <= T
        a = got[mask].fillna(-9.0).to_numpy()
        b = got2[mask].fillna(-9.0).to_numpy()
        if not np.allclose(a, b):
            raise RuntimeError(
                f"改动未来数据后历史对齐值变了 {int((a != b).sum())} 根 —— 有未来数据")

        # 2. 逐根核对：对齐值 = 决策时刻不晚于基准的**最后一根**周线
        dt_src = FS._decision_time(z.index, "1w")
        dt_base = FS._decision_time(base.index, "5m")
        vals = z.to_numpy(dtype=float)
        # 最后一根周线可能还没走完，按实现约定丢掉
        keep = len(vals) - 1
        pos = dt_src[:keep].searchsorted(dt_base, side="right") - 1
        exp = np.where(pos >= 0, vals[np.clip(pos, 0, keep - 1)], np.nan)
        both = np.isfinite(exp) & np.isfinite(got.to_numpy())
        if not np.allclose(exp[both], got.to_numpy()[both]):
            raise RuntimeError("对齐结果和「最后一根已收完的周线」对不上")

        # 3. 周内不该换值：同一 ISO 周里的对齐值只允许在周五收盘后换一次
        s = got.dropna()
        wk = pd.Series(s.index).dt.to_period("W").to_numpy()
        multi = int((s.groupby(wk).nunique() > 2).sum())
        if multi:
            raise RuntimeError(f"有 {multi} 个自然周里周线因子变了两次以上")

        # 4. 周线基准 + 日线因子：跨周期要真的接上，而且比单周期多一列说明
        wbase = tqdata.load_main("RB", "SHFE", "1w", adjust="qfq")
        day = tqdata.load_main("RB", "SHFE", "1d_tq", adjust="qfq")
        cfg = FS.FactorConfig(window=252, horizon=5,
                              factor_periods={"mom_20": "1d_tq"})
        extra = FS.period_frames(cfg, ["mom_20", "mom_60"], "1w", lambda p: day)
        if list(extra) != ["1d_tq"]:
            raise RuntimeError(f"period_frames 没认出要读哪个周期：{list(extra)}")
        pack = FS.build_score(wbase, "multi", ["mom_20", "mom_60"], cfg,
                              extra=extra, base_period="1w")
        if not any("跨周期因子" in n for n in pack.notes):
            raise RuntimeError("结果说明里没写跨周期")
        # 同一天：跨周期的 mom_20 用的就是那根日线的值
        al = FS.align_to_base(F.standardize(F.raw(day, "mom_20"), window=252,
                                           method="z"),
                              wbase.index, "1w", "1d_tq")
        if not np.allclose(pack.fmat["mom_20"].fillna(-9).to_numpy(),
                           al.fillna(-9).to_numpy()):
            raise RuntimeError("跨周期那一列和手工对齐的结果不一致")

        # 5. 缺周期要**报错**，不能悄悄把因子剔掉（剔掉只会让分数变了，
        #    界面上看不出来）
        try:
            FS.build_score(wbase, "multi", ["mom_20"], cfg, extra={},
                           base_period="1w")
        except ValueError as exc:
            if "跨周期因子缺行情" not in str(exc):
                raise RuntimeError(f"缺周期的报错看不懂：{exc}") from exc
        else:
            raise RuntimeError("跨周期行情缺失时没有报错")

        # 6. 单周期（不配跨周期）必须和以前逐位一致
        plain = FS.FactorConfig(window=252, horizon=5)
        p0 = FS.build_score(wbase, "multi", ["mom_20", "mom_60"], plain)
        ref = F.factor_frame(wbase, ["mom_20", "mom_60"], window=252, method="z")
        if not np.allclose(p0.fmat.fillna(-9).to_numpy(), ref.fillna(-9).to_numpy()):
            raise RuntimeError("没配跨周期时结果变了（老策略不该被影响）")

        return (f"未来扰动不影响历史；对齐=最后一根已收完的周线；"
                f"{len(s)} 根覆盖里周内不换值；周线+日线因子接上；缺周期报错")

    check("跨周期因子（截断 / 对齐）", _factor_cross_period)


    def _factor_multi_period():
        """同一个因子同时用两个周期：各算一列，一起进合成。

        以前 `factor_periods` 是「因子 → 周期」的一对一映射，同一个键只能有
        一个周期。现在周期编在**实例 id** 里（`mom_20@1d_tq`），同一个因子出现
        两次就是两列。要验的四件事：两列真的不一样、老写法仍然等价、去重、
        权重 / 反转能按实例单独给。
        """
        import numpy as np

        from ttrade import factor_strategy as FS
        from ttrade import factors as F
        from ttrade import tqdata

        wbase = tqdata.load_main("RB", "SHFE", "1w", adjust="qfq")
        day = tqdata.load_main("RB", "SHFE", "1d_tq", adjust="qfq")
        plain = FS.FactorConfig(window=252, horizon=5)

        # 1. 同一个因子两档周期 → 两列，且不相等
        keys = ["mom_20", "mom_20@1d_tq"]
        extra = FS.period_frames(plain, keys, "1w", lambda p: day)
        if list(extra) != ["1d_tq"]:
            raise RuntimeError(f"period_frames 没认出实例里的周期：{list(extra)}")
        pack = FS.build_score(wbase, "multi", keys, plain,
                              extra=extra, base_period="1w")
        if list(pack.fmat.columns) != keys:
            raise RuntimeError(f"列名不是两个实例：{list(pack.fmat.columns)}")
        a = pack.fmat["mom_20"].fillna(-9).to_numpy()
        b = pack.fmat["mom_20@1d_tq"].fillna(-9).to_numpy()
        if np.allclose(a, b):
            raise RuntimeError("两档周期算出来一模一样 —— 周期没生效")
        if not np.isfinite(b).any():
            raise RuntimeError("跨周期那一列全空")

        # 2. 老写法（纯键 + factor_periods）必须和实例写法逐位相同
        old = FS.FactorConfig(window=252, horizon=5,
                              factor_periods={"mom_20": "1d_tq"})
        pack2 = FS.build_score(wbase, "multi", ["mom_20", "mom_60"], old,
                               extra=FS.period_frames(
                                   old, ["mom_20", "mom_60"], "1w", lambda p: day),
                               base_period="1w")
        if not np.allclose(pack2.fmat["mom_20"].fillna(-9).to_numpy(), b):
            raise RuntimeError("老配置（factor_periods 映射）读出来的结果变了")

        # 3. 同一个实例重复出现只算一列
        dup = FS.build_score(wbase, "multi", ["mom_20@1d_tq", "mom_20@1d_tq"],
                             plain, extra=extra, base_period="1w")
        if list(dup.fmat.columns) != ["mom_20@1d_tq"]:
            raise RuntimeError(f"重复的实例没去重：{list(dup.fmat.columns)}")

        # 4. 权重 / 反转 / 标签都认实例
        c = FS.FactorConfig(weights={"mom_20@1d_tq": 2.0},
                            reverse={"mom_20": True})
        if abs(c.weight_of("mom_20@1d_tq") - 2.0) > 1e-9:
            raise RuntimeError("按实例给的权重没生效")
        if abs(c.weight_of("mom_20") - 1.0) > 1e-9:
            raise RuntimeError("按纯键给的权重不该影响另一个实例")
        if not (c.is_reversed("mom_20@1d_tq") and c.is_reversed("mom_20")):
            raise RuntimeError("按纯键勾的反转没传到实例上")
        lab = F.spec_label("mom_20@1d_tq")
        if "日线" not in lab or lab == F.spec_label("mom_20"):
            raise RuntimeError(f"实例的可读名没带上周期：{lab}")

        return (f"同一因子两档周期 → {list(pack.fmat.columns)}；"
                f"老写法逐位相同；重复实例去重；权重/反转按实例生效")

    check("同一因子多周期（因子实例）", _factor_multi_period)


    def _factor_filter():
        from ttrade import factor_strategy as FS
        from ttrade import factors as F
        from ttrade.strategies import run_strategy

        df = _us_df()
        cfg = FS.FactorConfig(window=252, horizon=5)
        sig = run_strategy(df, "donchian")["signal"].to_numpy(int)
        out = FS.run_rule_with_filter(df, sig, "multi", F.DEFAULT_MULTI, cfg)
        if "rule_signal" not in out.df.columns:
            raise RuntimeError("过滤结果缺少 rule_signal 列")
        base = int((sig != 0).sum())
        kept = int((out.df["signal"] != 0).sum())

        # K 线内那一族的策略走「规则 + 因子过滤」时，`entry_*` / `stop_dist` 必须
        # 被扔掉：留着的话 `run_backtest` 会分流到 intrabar 分支，而那条分支**根本
        # 不读 `signal`** —— 「因子过滤」就变成一个点了没用、也不报错的按钮。
        # 这三件事一起钉住：过滤跑得通、那几列真的没了、回测没走 intrabar 分支
        ib = run_strategy(df, "sbbox_ib")
        if "entry_long" not in ib.columns:
            raise RuntimeError("sbbox_ib 没出 entry_long —— intrabar 通道没接上")
        out2 = FS.run_rule_with_filter(ib, ib["signal"].to_numpy(int), "multi",
                                       F.DEFAULT_MULTI, cfg)
        left = [c for c in ("entry_long", "entry_short", "exit_long",
                            "exit_short", "stop_dist") if c in out2.df.columns]
        if left:
            raise RuntimeError(f"规则 + 因子过滤没扔掉 K 线内那几列：{left}")
        from ttrade.backtest import BacktestConfig, run_backtest

        rb = run_backtest(out2.df, BacktestConfig(initial_cash=100_000.0,
                                                  sizing="fixed", volume=1))
        if rb.metrics.get("intrabar") is not None:
            raise RuntimeError("过滤之后还是走了 K 线内分支（signal 被无视）")

        return (f"规则 {base} 根持仓 → 过滤后 {kept} 根；"
                f"K 线内策略过滤后不再走 intrabar 分支")

    check("规则 + 因子过滤", _factor_filter)

    def _ml_multi():
        """机器学习多品种：每个品种各训一遍，折级进度要报上来，还得能在折间叫停。

        以前多品种这条路直接把机器学习拦下来（"每个品种都要单独重训一遍"），
        拦掉等于「这套参数换一批品种还灵不灵」永远问不了。放开之后有两个新
        要求：进度必须细到折（否则界面几分钟不动，看着像卡死），停止必须能在
        折之间生效（否则点完还得等整个品种训完）。
        """
        import numpy as np
        import pandas as pd

        from gui import app as GUI
        from ttrade import factor_strategy as FS
        from ttrade.backtest import BacktestConfig

        n = 700
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        r = (np.sin(np.arange(n) / 25.0) * 0.01
             + np.random.default_rng(3).normal(0, 0.004, n))
        close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        # 开盘价必须和收盘价不同：前向收益是「次日开盘 / 当日收盘」，两者取同一
        # 个数的话标签恒等于 0，模型学出来的权重全是 0，整条策略空仓 —— 而
        # 「跑通了、只是没交易」看起来和正常结果一模一样
        open_ = close * (1 + 0.001 * np.sin(np.arange(n) / 3.0))
        df = pd.DataFrame({
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close, "volume": 1e5,
        })
        fcfg = FS.FactorConfig(model="linear", window=60, horizon=1,
                               train_window=200, step=60, min_train=120)
        job = GUI.BacktestJob(
            key="X", label="X", df=df, cfg=BacktestConfig(initial_cash=100_000),
            family=GUI.FAMILY_FACTOR, rule_key="", rule_params={},
            factor_mode="ml", factor_keys=("mom_20", "mom_60", "ma_gap_20"),
            factor_cfg=fcfg, filtered=False,
        )
        seen: list[str] = []
        res = GUI.run_backtest_job(job, progress=lambda d, t, m: seen.append(m))
        if res is None:
            raise RuntimeError("机器学习多品种任务没有结果")
        if not seen:
            raise RuntimeError("没有报出折级进度，多品种跑 ML 时界面会像卡死")
        if not res.metrics["n_trades"]:
            raise RuntimeError("机器学习跑完一笔没做，等于这条链没验到")
        try:
            GUI.run_backtest_job(job, cancel=lambda: True)
        except FS.Cancelled:
            pass
        else:
            raise RuntimeError("叫停没生效，得等整个品种训完才停")
        return (f"{len(seen)} 次折级进度，停止即时生效，"
                f"{res.metrics['n_trades']} 笔交易")

    check("机器学习多品种任务", _ml_multi)

    def _multi_pooled():
        """组合页的多品种回放要认「合并训练」。

        研究页勾了「合并训练」是「一个模型训完，再各自回测」；组合页走的是
        `replay.run_multi`，以前它根本不看 `pooled`，一律逐品种各训一遍。
        后果有两个：同一份配置在两个页面跑出**两条不同的净值**，而且 N 个品种
        要多训 N-1 遍（实测 65 因子 / 5m 十年 57 秒/品种，19 个品种就是 18 分钟
        对 1 分半）。

        这里不打桩看「有没有调那个函数」，而是直接比两条曲线：合并训练与逐品种
        必须**不一样**（真按 pooled 走了），且合并训练那条要和手工调
        `run_pooled_strategy` 算出来的一致。
        """
        import numpy as np
        import pandas as pd

        from ttrade import factor_strategy as FS
        from ttrade import replay as RP
        from ttrade import strategy_store as SStore
        from ttrade.backtest import combine, run_backtest

        def _mk(seed):
            n = 700
            idx = pd.date_range("2020-01-01", periods=n, freq="B")
            r = (np.sin(np.arange(n) / 25.0) * 0.01
                 + np.random.default_rng(seed).normal(0, 0.004, n))
            close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
            open_ = close * (1 + 0.001 * np.sin(np.arange(n) / 3.0))
            return pd.DataFrame({
                "open": open_,
                "high": np.maximum(open_, close) * 1.005,
                "low": np.minimum(open_, close) * 0.995,
                "close": close, "volume": 1e5,
            })

        frames = {"A_SHFE": _mk(1), "B_SHFE": _mk(2)}
        keys = ["mom_20", "mom_60", "ma_gap_20"]
        fc = FS.FactorConfig(model="linear", window=60, horizon=1,
                             train_window=200, step=60, min_train=120,
                             pooled=True)
        cfg = SStore.StrategyConfig(
            symbol="A", exchange="SHFE", period="1d", family="factor",
            rule_key="", factor_mode="ml", factor_keys=keys,
            factor_cfg=dict(vars(fc)), sizing="capital",
            principal=100_000.0, capital_pct=100.0,
            multi_symbols=list(frames),
        )

        # 打桩数一下有没有走合并训练那条路
        hits: list[int] = []
        orig = FS.run_pooled_strategy

        def _spy(*a, **kw):
            hits.append(1)
            return orig(*a, **kw)

        FS.run_pooled_strategy = _spy
        try:
            rp_pool = RP.run_multi(cfg, frames)
        finally:
            FS.run_pooled_strategy = orig
        if not rp_pool.ok:
            raise RuntimeError(f"合并训练回放失败：{rp_pool.error}")
        if not hits:
            raise RuntimeError("配置里 pooled=True，组合页回放却没走合并训练")

        # 逐品种那条路：曲线必须不一样，否则说明 pooled 只是个摆设
        off = cfg.clone()
        off.factor_cfg = dict(off.factor_cfg, pooled=False)
        rp_one = RP.run_multi(off, frames)
        if not rp_one.ok:
            raise RuntimeError(f"逐品种回放失败：{rp_one.error}")
        a, b = rp_pool.res.equity, rp_one.res.equity
        j = a.index.intersection(b.index)
        if len(j) < 10:
            raise RuntimeError("合并训练与逐品种没有共同交易日，没法比对")
        na = a.loc[j] / a.loc[j].iloc[0]
        nb = b.loc[j] / b.loc[j].iloc[0]
        gap = float((na - nb).abs().max())
        if gap < 1e-9:
            raise RuntimeError("合并训练和逐品种跑出一模一样的曲线：pooled 没生效")

        # 和手工调的合并训练对齐（同一条路，逐位一致）
        outs = orig(frames, keys, fc)
        manual = []
        for key, out in outs.items():
            one = cfg.clone(symbol=key.split("_")[0], exchange="SHFE",
                            multi_symbols=[], cross_symbols=[])
            bt = RP.make_bt("cn", False, one)
            bt.bars_per_year = RP.bars_per_year(out.df.index)
            manual.append(run_backtest(out.df, bt, label=out.label))
        res = combine(manual, label="多品种等权")
        m = res.equity
        j2 = a.index.intersection(m.index)
        dev = float((a.loc[j2] / a.loc[j2].iloc[0]
                     - m.loc[j2] / m.loc[j2].iloc[0]).abs().max())
        if dev > 1e-9:
            raise RuntimeError(f"组合页的合并训练与手工调的结果不一致（{dev:.2e}）")
        return (f"pooled 走了合并训练，与逐品种最大偏差 {gap:.4f}，"
                f"与手工口径偏差 {dev:.1e}")

    check("多品种回放认「合并训练」", _multi_pooled)

    def _ml_artifact():
        """训练产物：每一折落一个 .pkl，实盘只预测的结果和回测**一致**。

        这一条是「实盘滚动调用这些 .pkl」的验收。三个要点：
        1. 每一折真的落成独立文件（不是只存最后一个）
        2. 实盘加载产物后算出的信号 / 分数和回测一样
        3. 产物目录在**用户可写**目录里 —— 打包后 `bundle_root()` 是只读的，
           落那儿等于写不进去
        """
        import numpy as np
        import pandas as pd

        from ttrade import factor_strategy as FS
        from ttrade import ml_artifact as MLA

        n = 700
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        r = (np.sin(np.arange(n) / 25.0) * 0.01
             + np.random.default_rng(3).normal(0, 0.004, n))
        close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        open_ = close * (1 + 0.001 * np.sin(np.arange(n) / 3.0))
        df = pd.DataFrame({
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close, "volume": 1e5,
        })
        cfg = FS.FactorConfig(model="linear", window=60, horizon=1,
                              train_window=200, step=60, min_train=120)
        keys = ["mom_20", "mom_60", "ma_gap_20"]
        sid = "__selftest_ml#RB_SHFE"
        try:
            art, fset = MLA.train_fold_set(df, sid, cfg, keys,
                                           symbol="RB", exchange="SHFE")
            if len(fset.folds) < 2:
                raise RuntimeError(f"只训出 {len(fset.folds)} 折，没法验「滚动调用」")
            for fa in fset.folds:
                if not os.path.exists(MLA.fold_path(sid, fa.index)):
                    raise RuntimeError(f"第 {fa.index} 折没落成独立文件")
            # 产物目录必须可写（打包后 bundle_root 是只读的）
            adir = MLA.artifact_dir()
            if not os.path.isdir(adir) or not os.access(adir, os.W_OK):
                raise RuntimeError(f"产物目录不可写：{adir}")

            pred, why = MLA.load_for_live(sid, cfg, keys)
            if pred is None:
                raise RuntimeError(f"产物加载不了：{why}")
            got = pred.predict(df)
            if not got.get("ok"):
                raise RuntimeError(f"实盘只预测算不出来：{got.get('error')}")

            out = FS.run_factor_strategy(df, "ml", keys, cfg)
            want_sig = int(out.df["signal"].iloc[-1])
            want_score = float(out.df["factor_score"].iloc[-1])
            if int(got["signal"]) != want_sig:
                raise RuntimeError(
                    f"实盘信号 {got['signal']} 和回测 {want_sig} 不一样")
            if not np.isfinite(want_score) or \
                    abs(float(got["score"]) - want_score) > 1e-6:
                raise RuntimeError(
                    f"实盘分数 {got['score']} 和回测 {want_score} 不一样")
            return (f"{len(fset.folds)} 折各一个 .pkl，"
                    f"信号 {int(got['signal'])} / 分数 {float(got['score']):.6f} 与回测一致")
        finally:
            try:
                MLA.FoldSet.load(sid).remove()
            except Exception:  # noqa: BLE001
                pass

    check("机器学习训练产物（.pkl 滚动调用）", _ml_artifact)

    def _ml_trigger():
        """触发模式（稀疏采样）的决策口径：观点 ffill、进场在触发点、出场看强度。

        稀疏采样把「在哪建样本」和「在哪做决策」绑在一起之后，出场就成了问题：
        分数在触发点之间是阶梯常值，拿它判「跌破 0.3」永远等不到，仓位会一路拖到
        下一次触发。所以出场改成看**触发强度**（密集、每根都有值），进场仍看模型
        分数，方向由触发给。
        """
        import numpy as np
        import pandas as pd

        from ttrade import factor_strategy as FS

        n = 900
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        r = (np.sin(np.arange(n) / 25.0) * 0.01
             + np.random.default_rng(1).normal(0, 0.004, n))
        close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        # 开盘价 ≠ 收盘价，否则前向收益恒为 0、模型权重全 0（同 `_ml_multi`）
        open_ = close * (1 + 0.001 * np.sin(np.arange(n) / 3.0))
        df = pd.DataFrame({
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close, "volume": 1e5,
        })
        cfg = FS.FactorConfig(model="linear", window=60, horizon=1,
                              train_window=200, step=60, min_train=120,
                              entry_z=1.0, exit_z=0.3, sample_mode="trigger",
                              trigger_keys=["mom_20"], trigger_z=1.0,
                              trigger_tiers=0, trigger_cool=3)
        out = FS.run_factor_strategy(df, "ml", ["mom_20", "mom_60", "ma_gap_20"], cfg)
        plan = out.pack.trigger
        if plan is None or not plan.mask.any():
            raise RuntimeError("触发模式下没有决策点")
        sig = out.df["signal"].to_numpy()
        sc = out.pack.score.to_numpy()
        chg = np.flatnonzero(np.diff(np.r_[0, sig]) != 0)

        # 1. 观点只在决策点刷新，之间是常值
        prev, drift = np.nan, 0
        for i in range(len(sc)):
            if plan.mask[i]:
                prev = sc[i]
            elif not np.isclose(sc[i], prev, equal_nan=True):
                drift += 1
        if drift:
            raise RuntimeError(f"{drift} 根没沿用上一次的观点（ffill 没生效）")

        # 2. 进场只发生在决策点
        enter = [i for i in chg if sig[i] != 0]
        if not enter:
            raise RuntimeError("一笔没开，这条链没验到")
        stray = [i for i in enter if not plan.mask[i]]
        if stray:
            raise RuntimeError(f"{len(stray)} 次开仓不在决策点：{stray[:3]}")

        # 3. 因子通道的仓位按强度出场：平仓那根，强度已回落到离场阈值以内
        bad = []
        for i in chg:
            if sig[i] != 0:
                continue
            j = i - 1
            while j >= 0 and sig[j] == 0:
                j -= 1
            if j < 0:
                continue
            s = plan.strength[i]
            if not (np.isfinite(s) and (s < 0.3 if sig[j] == 1 else s > -0.3)):
                bad.append(i)
        if bad:
            raise RuntimeError(f"{len(bad)} 根平仓不是按触发强度：{bad[:3]}")

        # 4. 不勾稀疏采样时没有触发计划，走的还是老状态机
        import dataclasses

        plain = FS.run_factor_strategy(
            df, "ml", ["mom_20", "mom_60", "ma_gap_20"],
            dataclasses.replace(cfg, sample_mode="all"))
        if plain.pack.trigger is not None:
            raise RuntimeError("没开稀疏采样却生成了触发计划")

        # 5. 反转因子：强度严格取负（样本方向、进场方向、出场判据一起翻）
        flip = FS.run_factor_strategy(
            df, "ml", ["mom_20", "mom_60", "ma_gap_20"],
            dataclasses.replace(cfg, trigger_flip=True))
        s_a = np.nan_to_num(plan.strength, nan=0.0)
        s_b = np.nan_to_num(flip.pack.trigger.strength, nan=0.0)
        if not np.allclose(s_a, -s_b, atol=1e-12):
            raise RuntimeError("反转因子后触发强度不是原强度的相反数")
        if not flip.pack.trigger.mask.any():
            raise RuntimeError("反转因子后一个决策点都没有")

        # 6. 只用规则：两个触发因子都留空，方向全由规则给、出场跟规则 signal
        only = FS.run_factor_strategy(
            df, "ml", ["mom_20", "mom_60", "ma_gap_20"],
            dataclasses.replace(cfg, trigger_keys=[], trigger_rule="ma_cross",
                                trigger_params={"fast": 3, "slow": 8}))
        op = only.pack.trigger
        if op is None or op.factor_channel or not op.rule_channel:
            raise RuntimeError("只用规则时通道判定不对")
        osig = only.df["signal"].to_numpy()
        if not (osig != 0).any():
            raise RuntimeError("只用规则一笔都没开")
        # 规则通道的仓位必须始终和规则自己的方向一致（含出场）
        bad = int(np.sum((osig != 0) & (osig != op.rule_sig)))
        if bad:
            raise RuntimeError(f"{bad} 根仓位和规则方向不一致")

        # 7. 规则反转：rule_sig 取负，armed 是同一批
        rflip = FS.run_factor_strategy(
            df, "ml", ["mom_20", "mom_60", "ma_gap_20"],
            dataclasses.replace(cfg, trigger_keys=[], trigger_rule="ma_cross",
                                trigger_params={"fast": 3, "slow": 8},
                                trigger_rule_flip=True))
        rp = rflip.pack.trigger
        if not np.allclose(op.rule_sig, -rp.rule_sig):
            raise RuntimeError("反转规则后 rule_sig 不是相反数")
        if not np.array_equal(op.armed, rp.armed):
            raise RuntimeError("反转规则改变了「刚开仓」的那几根")

        # 8. 一个触发源都没有：必须在源头拦住，不能跑出一条空仓净值
        try:
            FS.run_factor_strategy(
                df, "ml", ["mom_20", "mom_60", "ma_gap_20"],
                dataclasses.replace(cfg, trigger_keys=[], trigger_rule=""))
        except ValueError as exc:
            if "触发源" not in str(exc):
                raise RuntimeError(f"报错文案没讲清原因：{exc}") from exc
        else:
            raise RuntimeError("没有任何触发源却跑通了")

        return (f"{int(plan.mask.sum())} 个决策点 / {len(enter)} 次开仓，"
                f"{len(chg) - len(enter)} 次按强度平仓；"
                f"只用规则 {int((osig != 0).sum())} 根持仓")

    check("机器学习触发模式（稀疏采样）", _ml_trigger)

    def _ml_degenerate():
        """稀疏样本训出「常数模型」的兜底：不能出常数分数，也不能丢掉最后一折。

        真实故障（SA / JM / PD / PT / SN / LH / BR 七个品种）：触发模式下
        折内只剩 40~85 个样本，树模型劈不开 → 只训出一棵单叶树 → 预测恒定
        → 滚动 z 的标准差是 0 → z 分数永远 NaN → 实盘报「z 分数算不出来
        （历史预测不够）」，而且状态永远显示「机器学习训练中」。

        真正的原因是 **L1 正则和标签量纲不匹配**：标签是前向收益（标准差
        ~5e-3），而 `reg_alpha=0.1` 的绝对惩罚比任何分裂的收益都大，
        LightGBM 于是放弃分裂。这里验三件事：

        1. 小标签量纲 + 小样本，训出来的模型**不是**常数
        2. walk-forward 不会因为常数模型把整折丢掉（最后一折丢了，
           实盘那段时间就没有预测 → 还是算不出 z 分数）
        3. 分数有方差（滚动 z 才算得出来）
        """
        import numpy as np
        import pandas as pd

        from ttrade import factor_strategy as FS
        from ttrade import mlmodels as ML

        # 1. 直接验模型层：小样本 + 极小量纲的标签，训出来必须有分裂
        rng = np.random.default_rng(7)
        Xs = rng.normal(size=(54, 42))
        ys = Xs[:, 0] * 0.001 + rng.normal(scale=0.005, size=54)
        mdl = ML.build("lightgbm")
        mdl.fit(Xs, ys)
        pr = np.asarray(mdl.predict(Xs), dtype=float)
        if len(np.unique(np.round(pr, 12))) < 2:
            raise RuntimeError(
                f"小样本训出了常数模型（{len(np.unique(np.round(pr, 12)))} 个取值）")
        if FS._is_degenerate_model(mdl, Xs[:64]):
            raise RuntimeError("小样本模型被判成退化模型")

        # 2. 走一遍 walk-forward：稀疏采样 + 小量纲标签，最后一折不能丢
        n = 1200
        idx = pd.date_range("2020-01-01", periods=n, freq="5min")
        r = (np.sin(np.arange(n) / 25.0) * 0.01
             + np.random.default_rng(11).normal(0, 0.004, n))
        close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        open_ = close * (1 + 0.001 * np.sin(np.arange(n) / 3.0))
        df = pd.DataFrame({
            "open": open_, "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close, "volume": 1e5,
        })
        cfg = FS.FactorConfig(model="lightgbm", window=60, horizon=1,
                              train_window=300, step=120, min_train=200,
                              entry_z=1.0, exit_z=0.3, sample_mode="trigger",
                              trigger_keys=["mom_20"], trigger_z=1.0,
                              trigger_tiers=0, trigger_cool=3)
        keys = ["mom_20", "mom_60", "ma_gap_20"]
        stats: dict = {}
        models: list = []
        out = FS.run_factor_strategy(df, "ml", keys, cfg, collect_models=models)
        pred = out.pack.pred.to_numpy(dtype=float)
        fin = np.isfinite(pred)
        if not fin.any():
            raise RuntimeError("一个预测都没出来")
        # 最后一折必须覆盖到最后（实盘就是在这段上跑）
        gap = len(pred) - 1 - int(np.flatnonzero(fin)[-1])
        if gap > 0:
            raise RuntimeError(f"最后 {gap} 根没有预测（最后一折被丢了）")
        # 尾部 252 根的预测不能恒定 —— 那正是实盘 z 缓冲的来源
        tail = pred[-252:]
        tailf = tail[np.isfinite(tail)]
        if len(tailf) < 2 or len(np.unique(np.round(tailf, 12))) < 2:
            raise RuntimeError("尾部预测是常数 —— 滚动 z 的标准差会是 0")
        # 分数必须有方差
        sc = out.pack.score.to_numpy(dtype=float)
        scf = sc[np.isfinite(sc)]
        if len(scf) < 2 or float(scf.std()) < 1e-6:
            raise RuntimeError(f"分数没有方差（std={scf.std() if len(scf) else 0}）")

        return (f"小样本模型 {len(np.unique(np.round(pr, 12)))} 个取值；"
                f"{len(models)} 折，尾部预测覆盖完整，分数 std={scf.std():.3f}")

    check("机器学习常数模型兜底（小样本不丢折）", _ml_degenerate)

    def _ml_live_poll():
        """机器学习实盘：信号在后台算完的那一刻，没有新 bar 也要能读到。

        实盘信号走后台线程，`ml_ready` 之前 `tick()` 不动仓。而「同一根 bar 里
        不重算」这条短路如果连**读缓存**也一起挡掉，收盘前最后一次计算就永远
        读不到 —— 界面一直停在「计算中」，直到下一个交易时段。
        """
        import time as _t

        import pandas as pd

        from ttrade import strategy_runner as R
        from ttrade import strategy_store as SStore

        class _Ml:
            def __init__(self):
                self.out: dict = {}
                self.err: dict = {}
                self.submits: list = []

            def submit(self, key, df, cfg, keys, predictor=None, extra=None):
                self.submits.append((key, predictor is not None))

            def signal(self, key):
                hit = self.out.get(key)
                return (None, 0.0) if not hit else (int(hit), 1.0)

            def error(self, key):
                return str(self.err.get(key) or "")

        idx = pd.date_range("2026-11-03 20:00", periods=320, freq="5min")
        base = pd.Series(range(320), index=idx, dtype=float) + 3000.0
        frame = pd.DataFrame({"open": base, "high": base + 5, "low": base - 5,
                              "close": base, "volume": 100.0,
                              "open_oi": 1.0, "close_oi": 1.0}, index=idx)

        class _H(R.PortfolioRunner):
            def __init__(self):
                self._slots, self._klines, self._quotes = {}, {}, {}
                self._qfq_base, self.api, self.fast = {}, None, True
                self._ml = _Ml()
                self.frame = frame

            def _subscribe(self, st): pass
            def _skey(self, cfg): return cfg.key()
            def _signal_frame(self, st, kl): return self.frame
            def _predictor_for(self, st):
                return object() if self.fast else None
            def _with_aux(self, st, df): return df
            def _mark_price(self, st, df): return 3000.0
            def _log(self, msg): pass

        def _slot() -> R.SlotState:
            cfg = SStore.StrategyConfig(
                sid="sid-poll", symbol="RB", exchange="SHFE", period="5m",
                family="factor", factor_mode="ml", factor_keys=["mom_20"],
                sizing="fixed", volume=1, note="short",
            )
            st = R.SlotState(cfg=cfg)
            st.running, st.ml_ready = True, False
            return st

        def _kl(dt):
            return pd.DataFrame({"datetime": [int(dt.value)]})

        # ---- 同一根 bar 里读到后台算好的信号 ----
        h, st = _H(), _slot()
        h._klines[st.cfg.key()] = _kl(pd.Timestamp("2026-11-03 23:05"))
        for _ in range(5):
            h._refresh_signal(st)               # 5 个 tick 都还在同一根 bar 里
        if st.ml_ready:
            raise RuntimeError("信号还没算出来就 ready 了")
        if st.ml_why != "" or st.last_action != "机器学习信号计算中，暂不动仓":
            raise RuntimeError(f"「计算中」这一态没走对：{st.last_action!r}")
        if len(h._ml.submits) != 1:
            raise RuntimeError(f"同一根 bar 里重复提交了 {len(h._ml.submits)} 次")
        h._ml.out[st.cfg.key()] = -1            # 后台线程算完了
        h._refresh_signal(st)
        if not st.ml_ready or st.signal != -1 or st.target != -st.lots:
            raise RuntimeError(
                f"同一根 bar 里没读到信号：ready={st.ml_ready} "
                f"signal={st.signal} target={st.target}")

        # ---- 没有训练产物 → 「重训中」，不能混成「计算中」----
        h2, st2 = _H(), _slot()
        h2.fast = False
        h2._klines[st2.cfg.key()] = _kl(pd.Timestamp("2026-11-03 23:05"))
        h2._refresh_signal(st2)
        if st2.ml_why != "retrain":
            raise RuntimeError(f"慢路径应标 retrain，实际 {st2.ml_why!r}")

        # ---- 永远算不出来 → 把真实原因顶到界面 ----
        h3, st3 = _H(), _slot()
        h3._klines[st3.cfg.key()] = _kl(pd.Timestamp("2026-11-03 23:05"))
        h3._ml.err[st3.cfg.key()] = "z 分数算不出来（历史预测不够）"
        h3._refresh_signal(st3)
        if st3.ml_why != "z 分数算不出来（历史预测不够）":
            raise RuntimeError(f"真实原因没顶上来：{st3.ml_why!r}")

        # ---- 行情活性：收盘之后不许再发单 ----
        now = _t.time()
        st4 = _slot()
        for bar, want in ((now, True), (now - 3600, False), (0, False),
                          (None, False), (now + 300, True)):
            st4.last_bar = int(bar * 1e9) if bar else (0 if bar == 0 else None)
            if h._market_alive(st4) is not want:
                raise RuntimeError(f"行情活性判据错：last_bar={bar} 应为 {want}")

        # ---- 周期秒数：15m 是「5m × 3」= 900 秒，不是 180 ----
        sec = R.PortfolioRunner._bar_seconds
        for period, want in (("5m", 300.0), ("15m", 900.0), ("30m", 1800.0),
                             ("1h", 3600.0), ("2h", 7200.0), ("4h", 14400.0)):
            if sec(period) != want:
                raise RuntimeError(f"{period} 应为 {want} 秒，实际 {sec(period)}")

        return "同一根 bar 内读到后台信号 / 三态分开 / 收盘不发单 / 周期秒数对"

    check("机器学习实盘取信号（没有新 bar 也要能读到）", _ml_live_poll)

    def _live_rule_filter():
        """实盘信号必须和回测走同一条「规则 + 因子过滤」。

        `_signal` 原来只跑规则、**完全不看 `factor_mode`**：回测里被因子否决掉
        的那些开仓，实盘照开 —— 账户里真会多出那些仓位，而且两条曲线从第一天
        起就不是同一条。这里挑一个「规则说要开、因子说不行」的时点，要求实盘
        信号跟着因子走；顺带核一遍「没开过滤时实盘信号就是规则信号」。
        """
        from ttrade import replay as RP
        from ttrade import strategy_runner as R
        from ttrade import strategy_store as SStore
        from ttrade.factor_strategy import run_rule_with_filter
        from ttrade.strategies import run_strategy

        df = _us_df()
        base = SStore.StrategyConfig(symbol="X", family="rule", rule_key="donchian",
                                     period="1d", sizing="unit", principal=100000.0)
        filt = base.clone(factor_mode="multi", factor_keys=["mom_20", "mom_60"],
                          factor_cfg={"window": 252, "entry_z": 1.0,
                                      "exit_z": 0.3, "filter_mode": "hold"})
        fc = RP.factor_config(filt.factor_cfg)
        keys = list(filt.factor_keys)

        class _H(R.PortfolioRunner):
            """只借 `_signal` 这一条路：辅助列和跨周期行情都按「没有」处理。"""

            def __init__(self):
                self._aux_warned = set()
                self._qfq_base = {}
                self._log_fn = None

            def _with_aux(self, st, d):
                return d

            def _period_extra(self, st, f):
                return {}

        h = _H()

        class _St:
            symbol = "X"
            exchange = ""
            underlying = ""
            cfg = None

        st = _St()

        def _sig(cfg, d):
            st.cfg = cfg
            return h._signal(st, d)

        rule_sig = run_strategy(df, "donchian", {})["signal"].to_numpy(int)
        want_f = int(run_rule_with_filter(
            df, rule_sig, "multi", keys, fc).df["signal"].iloc[-1])
        got_f = _sig(filt, df)
        got_p = _sig(base, df)
        if got_f != want_f:
            raise RuntimeError(
                f"实盘过滤信号和回测口径不一致：实盘 {got_f} / 回测 {want_f}")
        if got_p != int(rule_sig[-1]):
            raise RuntimeError(f"没开过滤时实盘信号不该变：{got_p} / {int(rule_sig[-1])}")

        # 找一个两者分歧的时点。找不到（数据换了）也不算失败 —— 上面那两条
        # 已经证明实盘走的是同一条过滤口径，这一条只是把它钉到具体某根 bar 上
        div = None
        for cut in range(len(df) - 5, max(320, len(df) - 120), -1):
            sub = df.iloc[:cut]
            rs = run_strategy(sub, "donchian", {})["signal"].to_numpy(int)
            a = int(rs[-1])
            b = int(run_rule_with_filter(sub, rs, "multi", keys, fc).df["signal"].iloc[-1])
            if a != b:
                div = (cut, a, b)
                break
        if div is not None:
            cut, a, b = div
            sub = df.iloc[:cut]
            if _sig(filt, sub) != b:
                raise RuntimeError(
                    f"分歧点上实盘没跟着因子走：{_sig(filt, sub)} / 过滤 {b}")
            if _sig(base, sub) != a:
                raise RuntimeError(f"分歧点上不过滤的实盘信号错了：{_sig(base, sub)} / {a}")
            tail = f"分歧点 第 {cut} 根：规则 {a} → 过滤 {b}"
        else:
            tail = "样本里没有分歧点，只核了口径一致"
        return f"实盘=回测同一套过滤，{tail}"

    check("实盘信号走因子过滤", _live_rule_filter)

    def _align_lock():
        """锁仓（多空双开）必须被看见，而且能被一键对齐平掉。

        净仓口径会把「多 4 空 4」压成 0，和空仓长得一模一样。2026-11-11 的
        实盘就是这样：ag2612 多 4 空 4、cu2612 多 1 空 1、lc2701 多 1 空 1，
        合计 47 万保证金占着，监控器报「一致」，一键对齐一笔都不发。
        """
        from ttrade import strategy_store as SStore
        from ttrade.strategy_runner import PortfolioRunner, SlotState

        class P:
            def __init__(self, lo, sh):
                self.pos_long, self.pos_short = lo, sh
                for s, v in (("long", lo), ("short", sh)):
                    setattr(self, f"pos_{s}_his", v)
                    setattr(self, f"pos_{s}_today", 0)
                    setattr(self, f"volume_{s}_his", v)
                    setattr(self, f"volume_{s}_today", 0)

        # 绕过 __init__：它要建日期额度台账、持仓账本目录，自检不该留这些垃圾
        r = PortfolioRunner.__new__(PortfolioRunner)
        r._log_fn = None
        r._mm_cache, r._mm_ts = [], 0.0
        r._align_ts = 0.0
        r.api = object()
        r._slots = {}
        st = SlotState(cfg=SStore.StrategyConfig(sid="t", name="jd", period="1h"))
        st.running, st.underlying, st.target = True, "DCE.jd2612", -24
        r._slots["t"] = st
        r._align_slot = SlotState(cfg=SStore.StrategyConfig(
            sid="__align__", name="一键对齐", period="1d"))
        r._pos = {"SHFE.ag2612": P(4, 4), "DCE.jd2612": P(0, 24)}

        rows = r.mismatch(fresh=True)
        syms = [x["symbol"] for x in rows]
        if syms != ["SHFE.ag2612"]:
            raise RuntimeError(f"锁仓没露出来：{syms}（jd 净额一致，不该报）")
        row = rows[0]
        if not row["locked"] or (row["a_long"], row["a_short"]) != (4, 4):
            raise RuntimeError(f"锁仓标记不对：{row}")
        if row["target"] != 0 or row["actual"] != 0 or row["delta"] != 0:
            raise RuntimeError(f"净额口径该是 0/0/0：{row}")

        # 对齐要发两笔（卖平多 + 买平空），不能因为净差额 0 就一笔不发
        sent = []
        r._send = lambda s, sym, delta, closeable=None: (
            sent.append((sym, delta)) or s.pending.append("x") or 1)
        r._sync_after_align = lambda sym: None
        n = r.align(close_orphans=True)
        if sorted(sent) != [("SHFE.ag2612", -4), ("SHFE.ag2612", 4)] or n != 2:
            raise RuntimeError(f"锁仓该拆成两笔平仓，实发 {sent}")

        # 平完之后必须判一致
        r._pos = {"SHFE.ag2612": P(0, 0), "DCE.jd2612": P(0, 24)}
        r._mm_ts = 0.0
        if r.mismatch(fresh=True):
            raise RuntimeError("平完还报不一致")
        return "净 0 的锁仓能看见，对齐拆成卖平多 + 买平空两笔"

    check("一键对齐看得见锁仓（多空双开）", _align_lock)

    def _add_or_update():
        """新加进组合的策略要能被「启动 / 停止」接上。

        2026-11-11 用户报：组合页新加一条策略，点「启动 / 停止」没反应。根因是
        加进组合只写了磁盘、从没推给引擎，而 `PortfolioRunner.enable()` 对不认识
        的 sid 是 `self._slots.get(sid) is None` 然后**静默 return** —— 不报错、
        不变化。`add()` 又不能拿来补挂：它无条件重建 SlotState，正在跑的槽位会
        被抹掉 running / 目标仓 / 挂单。
        """
        from ttrade import strategy_store as SStore
        from ttrade.strategy_runner import PortfolioRunner, SlotState

        class _Spec:
            def label(self):
                return "__selftest__"       # positionbook.load 只读，不落台账

        r = PortfolioRunner.__new__(PortfolioRunner)
        r._slots, r._groups, r._parents, r._child_of = {}, {}, {}, {}
        r._log = lambda *_a, **_k: None
        r._cancel_all = lambda st: None
        r.spec = _Spec()

        # ① 已经在跑的槽位：必须原对象原状态，不能被重建
        old = SStore.StrategyConfig(sid="old", name="旧", period="1h")
        st = SlotState(cfg=old)
        # `signal × lots` 要自洽：`update()` 按 `target = signal * max(1, lots)` 重算
        # 目标仓（改手数立刻生效，这是对的），夹具写矛盾值就成了假失败
        st.running, st.signal, st.lots, st.target = True, -1, 3, -3
        r._slots["old"] = st
        r.add_or_update(old)
        if r._slots.get("old") is not st or not st.running or st.target != -3:
            raise RuntimeError("add_or_update 重建了正在跑的槽位，running/target 丢了")

        # ② 新加的策略：补挂之后 `enable()` 才不会提前 return
        new = SStore.StrategyConfig(sid="new", name="新", period="1h")
        if r.get("new") is not None:
            raise RuntimeError("新策略不该已经在引擎里（前提就不成立）")
        r.add_or_update(new)
        if r.get("new") is None:
            raise RuntimeError("add_or_update 没把新策略接进来，enable 还是会静默返回")

        # ③ 反证：`add()` 就会把状态抹掉 —— 所以补挂只能走 add_or_update
        r.add(old)
        if r._slots.get("old") is st or r._slots["old"].running:
            raise RuntimeError("add() 本该重建槽位，这条自检没有意义了")
        return "跑着的槽位原样保留（running/target 不动），新策略能补挂；add() 会重建（已反证）"

    check("新加的策略能被启停接上（add_or_update）", _add_or_update)

    def _want_on_rule():
        """「启动 / 停止」的方向要按真实运行状态判，不能按存盘的 enabled 开关。

        2026-11-11 用户报：停止全部之后选一条点按钮，提示「标记为停止，启动后生效」，
        怎么点都起不来。原因是 `停止全部` 只断连接、不清 `enabled`，按 `not cfg.enabled`
        取反，第一下算出来是「关」—— 得连点两下才变「开」，而且那一下本来也没启动
        任何东西（当时 `ctl.running` 已是 False，只写了标记）。
        """
        from gui.portfolio_page import PortfolioPage

        for slot, want, why in [
            ({}, True, "没有槽位（停止全部之后就是这样）"),
            ({"running": False}, True, "槽位在但没跑"),
            ({"running": True}, False, "正在跑"),
        ]:
            got = PortfolioPage._want_on(slot)
            if got != want:
                raise RuntimeError(f"{why}：应为 {want}，实得 {got}")

        # 反证：旧口径在同一种状态（enabled=True 但什么都没跑）会判成「停止」
        enabled_after_stop_all = True
        old_rule = not enabled_after_stop_all
        new_rule = PortfolioPage._want_on({})
        if old_rule == new_rule:
            raise RuntimeError(f"旧口径 {old_rule} 该与新口径 {new_rule} 相反")
        return "无槽位/未跑→启动、在跑→停止；旧口径（not enabled）会判成停止"

    check("启停按钮按真实运行状态判方向", _want_on_rule)

    def _grid_levels():
        """网格交易：signal 是**档位号**，回测走独立的分级仓位分支。

        2026-11-11 新增。三条必须成立：
        ① 网格要真的跑出 |signal|>1 的档位（否则它就是一条普通的高低卖）；
        ② 档位信号**不能**进老循环 —— 老循环里 `pos = want` 之后到处按
           `pos == 1` 判多空，-3 会被当成「非多头」走进空头分支，算错也不报错；
        ③ 逐笔 `Σ(gross − cost)` 必须等于账户变化（项目里的老判据，分级分支
           每条腿各记一笔，最容易在这里对不上）。
        """
        import numpy as np
        import pandas as pd

        from ttrade.backtest import BacktestConfig, run_backtest
        from ttrade.strategies import run_strategy

        idx = pd.date_range("2024-01-01", periods=40, freq="D")
        px = [100.0] * 20 + [99, 98, 97, 96, 97, 98, 99, 100, 101, 102]
        px += [100.0] * (40 - len(px))
        d = pd.DataFrame({
            "open": px, "high": [p * 1.002 for p in px],
            "low": [p * 0.998 for p in px], "close": px, "volume": [1000] * 40,
        }, index=idx)

        # 🔴基线显式固定：`grid_mode`/`hold`/`atr_mult` 的默认值是「-1 = 按品种」，
        #    而这个夹具没有 `attrs["symbol"]`、只有 40 根 bar（ATR300 预热不了）
        #    ⇒ 必须显式指定走百分比口径，否则整条信号会是 0。
        out = run_strategy(d, "grid_trade",
                           {"ma_n": 10, "grid_n": 8, "grid_pct": 1.0,
                            "grid_mode": 0, "hold": 0})
        sig = out["signal"].to_numpy(int)
        if int(np.abs(sig).max()) <= 1:
            raise RuntimeError(f"网格没有产生档位，还是 ±1：{sorted(set(sig.tolist()))}")

        cfg = BacktestConfig(initial_cash=100000, multiplier=10, margin_rate=0.1,
                             sizing="fixed", volume=1, fee_rate=0.0001)
        r = run_backtest(out, cfg, label="网格")
        if "max_level" not in r.metrics:
            raise RuntimeError("档位信号没走分级分支（进了老循环）")
        pnl = sum(t.gross - t.cost for t in r.trades)
        chg = float(r.equity.iloc[-1]) - cfg.initial_cash
        if abs(pnl - chg) > 1e-6:
            raise RuntimeError(f"Σ(gross−cost)={pnl:.4f} 与账户变化 {chg:.4f} 对不上")

        # 老策略必须仍然走老路径，否则全部历史回测都换口径了
        d2 = d.copy()
        d2["signal"] = 1
        if "max_level" in run_backtest(d2, cfg).metrics:
            raise RuntimeError("±1 的信号也进了分级分支（老路径被改到了）")
        return (f"档位 max={int(np.abs(sig).max())}、分级回测逐笔自洽"
                f"（Σ净额 {pnl:.2f}）、±1 仍走老路径")

    check("网格交易（档位仓位）的回测口径", _grid_levels)

    def _grid_intraday():
        """网格日内版：14:50 之后不许再有信号、14:55 起必须空仓、不隔夜。

        2026-11-11 新增。这一条同时守住两个刚踩过的坑：

        ① **节假日前一天漏网**：「下一根已经过了平仓时刻」是靠分钟数比的，
           而节前没有夜盘，下一根直接跳到节后 09:00（540 < 895），14:55 那根
           就打不掉了 ⇒ 等于给节后开盘挂一张隔夜单。判据改成「下一根的时钟
           比这一根更早」＝换场/换日。下面专门造一个「日盘后直接跳到次日」
           的样本，就是当初漏掉的那个形状。
        ② **持仓曲线回不到 0**：`pos` 原来写在 `if want_dir != 0` 里面，清仓
           那根不进这个分支 ⇒ 持仓一直沿用上一个方向，看图上「14:55 是否
           空仓」永远显示有仓。所以这里断言的是**持仓序列**，不是权益。
        """
        import numpy as np
        import pandas as pd

        from ttrade.backtest import BacktestConfig, run_backtest
        from ttrade.strategies import run_strategy

        day_min = list(range(540, 691, 5)) + list(range(810, 896, 5))   # 09:00~14:55
        night_min = list(range(1260, 1376, 5))                          # 21:00~22:55
        stamps, prices = [], []
        for k in range(3):
            base = pd.Timestamp("2024-01-02") + pd.Timedelta(days=k)
            for j, m in enumerate(day_min + night_min):
                # 第 1 天砸 5 格、第 2 天抽回 5 格，第 0 天打底
                if k == 1:
                    px = 100.0 - min(j, 5) * 1.0
                elif k == 2:
                    px = 95.0 + min(j, 5) * 1.0
                else:
                    px = 100.0
                stamps.append(base + pd.Timedelta(minutes=m))
                prices.append(px)

        idx = pd.DatetimeIndex(stamps)
        d = pd.DataFrame({"open": prices, "high": prices, "low": prices,
                          "close": prices, "volume": [1000] * len(idx)}, index=idx)

        out = run_strategy(d, "grid_trade",
                           {"ma_n": 10, "grid_n": 8, "grid_pct": 1.0,
                            "flat_pct": 0.0, "flat_min": 895, "night": 0,
                            "grid_mode": 0, "hold": 0})
        sig = out["signal"].to_numpy(int)
        ms = (idx.hour * 60 + idx.minute).to_numpy(int)
        is_day = (ms >= 480) & (ms < 960)
        if int(np.abs(sig).max()) <= 1:
            raise RuntimeError(f"样本没跑出档位：{sorted(set(sig.tolist()))}")

        late = is_day & (ms >= 890)                     # 14:50 及以后
        if np.any(sig[late] != 0):
            bad = [str(idx[i]) for i in np.flatnonzero(late & (sig != 0))][:3]
            raise RuntimeError(f"收盘前还有信号（会给次日挂隔夜单）：{bad}")
        if np.any(sig[~is_day] != 0):
            raise RuntimeError("night=0 时夜盘还有信号")

        cfg = BacktestConfig(initial_cash=100000, multiplier=10, margin_rate=0.1,
                             sizing="fixed", volume=1, fee_rate=0.0001)
        r = run_backtest(out, cfg, label="网格")
        pos = r.position.to_numpy(int)
        if np.any(pos[is_day & (ms >= 895)] != 0):
            raise RuntimeError("14:55 那根还带着仓（日内强平没生效）")
        if np.any(pos[~is_day] != 0):
            raise RuntimeError("夜盘带着仓")
        days = idx.normalize().to_numpy()
        first = np.r_[True, days[1:] != days[:-1]]
        if np.any(pos[first] != 0):
            raise RuntimeError("每个交易日第一根还带着仓（隔夜了）")
        # 反过来也要保证这一刻真的交易过，否则上面几条是「一直空仓」的假通过
        if not np.any(pos != 0):
            raise RuntimeError("样本从头到尾没持仓，日内约束是假通过")
        return (f"档位 max={int(np.abs(sig).max())}、14:50 后零信号、"
                f"14:55/夜盘/次日开盘均零仓")

    check("网格交易（日内强平 + 不隔夜）", _grid_intraday)

    def _grid_no_lookahead():
        """网格挂单模型不许用当根收盘定方向（2026-11-11 修掉的前视）。

        做法：把第 k 根的**收盘价**随意改掉，第 k 根**及之前**已经完全平掉的
        那批交易必须一模一样。之前 `want = sign(sig[i])` 用的是当根收盘算出的
        档位号，却在本根内成交 —— 等于用收盘知道该做多、再回头在开盘挂单。
        实测这一条值多少钱：中证 1000 顺势版 +923% → +98%。
        """
        import numpy as np
        import pandas as pd

        from ttrade.backtest import BacktestConfig, run_backtest
        from ttrade.strategies import run_strategy

        rng = np.random.default_rng(7)
        day_min = list(range(540, 691, 5)) + list(range(810, 896, 5))
        stamps = []
        for k in range(6):
            base = pd.Timestamp("2024-03-04") + pd.Timedelta(days=k)
            stamps += [base + pd.Timedelta(minutes=m) for m in day_min]
        idx = pd.DatetimeIndex(stamps)
        n = len(idx)
        px = 100.0 + np.cumsum(rng.normal(0, 0.12, n))
        d = pd.DataFrame({"open": px, "high": px + 0.15, "low": px - 0.15,
                          "close": px, "volume": [1000] * n}, index=idx)

        prm = {"ma_n": 12, "grid_n": 3, "grid_pct": 0.3,
               "flat_pct": 0.0, "flat_min": 895, "night": 0, "follow": 1,
               "grid_mode": 0, "hold": 0}
        cfg = BacktestConfig(initial_cash=100000, sizing="fixed", volume=1,
                             multiplier=10, margin_rate=0.1, fee_rate=0.0001)

        a = run_backtest(run_strategy(d, "grid_trade", prm), cfg)
        k = n // 2
        d2 = d.copy()
        d2.iloc[k, d2.columns.get_loc("close")] = px[k] * 1.05   # 只改这一根收盘
        b = run_backtest(run_strategy(d2, "grid_trade", prm), cfg)

        cut = idx[k]
        # 比 **进场**：第 k 根当根进场的价格如果受「这一根收盘」影响，就是前视。
        # 出场价不比 —— 它本来就晚于 k，受后续行情影响是应该的。
        ta = [(t.entry_date, round(t.entry_price, 6)) for t in a.trades
              if t.entry_date <= cut]
        tb = [(t.entry_date, round(t.entry_price, 6)) for t in b.trades
              if t.entry_date <= cut]
        if ta != tb:
            raise RuntimeError(f"改了第 {k} 根的收盘，这一根及之前的进场价却变了 —— "
                               f"有前视（{len(ta)} 笔 vs {len(tb)} 笔）")
        if not ta:
            raise RuntimeError("样本太短，前半段没成交，测不出前视")
        return f"扰动第 {k} 根收盘，当根及之前 {len(ta)} 笔进场价一分不动"

    check("网格挂单模型：不许用当根收盘定方向（前视）", _grid_no_lookahead)

    def _grid_side_modes():
        """三种方向结构：只做多头 / 只做空头 / 多空都做（中间带同时挂两张网）。

        判据全是**结构性不变量**，不看收益：
        ① `grid_side=0` 只出多单、`=1` 只出空单；
        ② `grid_side=2, mid_band=0` 必须退化成原单边网 —— 多空**交替**出现但
           **从不同时持有**（`mixed_bars=0`）、`n_sides=1`（手数不摊）；
        ③ `grid_side=2, mid_band` 很大时中间带里两个方向要**真的同时挂着**
           （`mixed_bars>0`）、`n_sides=2`；
        ④ `sizing=fixed` 的手数不受摊薄影响（每格固定 N 手是定义，不是预算）。
        """
        import numpy as np
        import pandas as pd

        from ttrade.backtest import BacktestConfig, run_backtest
        from ttrade.strategies import run_strategy

        rng = np.random.default_rng(11)
        day_min = list(range(540, 691, 5)) + list(range(810, 896, 5))
        stamps = []
        for k in range(14):
            base = pd.Timestamp("2024-03-04") + pd.Timedelta(days=k)
            stamps += [base + pd.Timedelta(minutes=m) for m in day_min]
        idx = pd.DatetimeIndex(stamps)
        n = len(idx)
        # 明显来回摆：中间带内上下都要被扫到，否则「同时双向」测不出来
        px = 100.0 + 6.0 * np.sin(np.arange(n) / 45.0) + np.cumsum(rng.normal(0, 0.04, n))
        d = pd.DataFrame({"open": px, "high": px + 0.30, "low": px - 0.30,
                          "close": px, "volume": [1000] * n}, index=idx)
        # 逆势（跌买涨卖）：多空分居中枢两侧，最方便判「有没有同时持有」
        base_prm = {"ma_n": 24, "grid_n": 3, "grid_pct": 0.3, "flat_pct": 0.0,
                    "flat_min": 895, "night": 0, "follow": 0,
                    "grid_mode": 0, "hold": 0, "net_stop": 0.0}
        cfg = BacktestConfig(initial_cash=1_000_000, sizing="fixed", volume=1,
                             multiplier=10, margin_rate=0.1, fee_rate=0.0001)

        def run(sid, mb):
            p = dict(base_prm)
            p["grid_side"] = sid
            p["mid_band"] = mb
            return run_backtest(run_strategy(d, "grid_trade", p), cfg)

        lo, sh = run(0, 0.0), run(1, 0.0)
        if any(t.direction < 0 for t in lo.trades):
            raise RuntimeError("grid_side=0 出了空单 —— 只做多头没生效")
        if any(t.direction > 0 for t in sh.trades):
            raise RuntimeError("grid_side=1 出了多单 —— 只做空头没生效")
        if not lo.trades or not sh.trades:
            raise RuntimeError("只多/只空样本里一笔没成交，夹具太短")

        one = run(2, 0.0)
        if int(one.metrics["mixed_bars"]) != 0:
            raise RuntimeError(f"mid_band=0 竟然多空同时持有 "
                               f"{one.metrics['mixed_bars']} 根 —— 没退化成单边网")
        if float(one.metrics["n_sides"]) != 1.0:
            raise RuntimeError("mid_band=0 手数却被摊成两半了")
        dirs = {t.direction for t in one.trades}
        if dirs != {1, -1}:
            raise RuntimeError(f"单边网只出了 {dirs} 一个方向 —— 夹具没覆盖到另一侧")

        both = run(2, 99.0)
        if int(both.metrics["mixed_bars"]) == 0:
            raise RuntimeError("mid_band=99 多空从来没有同时持有 —— 双向没生效")
        if float(both.metrics["n_sides"]) != 2.0:
            raise RuntimeError("双向模式下预算没摊成两份")
        if {t.direction for t in both.trades} != {1, -1}:
            raise RuntimeError("双向模式只出了一个方向")

        # ④ 摊薄只对「按钱定手数」的口径生效：`fixed` 是「每格固定 N 手」，
        #    双向时也不能变成半手/两倍手。
        if any(abs(float(t.lots) - 1.0) > 1e-9 for t in both.trades if t.lots):
            raise RuntimeError("fixed 口径（每格 1 手）被协议摊薄改掉了")
        return (f"只多 {len(lo.trades)} 笔 / 只空 {len(sh.trades)} 笔；"
                f"带=0 同时持有 0 根(n_sides=1)；带=∞ 同时持有 "
                f"{both.metrics['mixed_bars']} 根(n_sides=2)")

    check("网格三种方向结构（只多/只空/多空都做）", _grid_side_modes)

    def _grid_symbol_defaults():
        """分品种默认值：表外品种兜底 = **逆势 + 不做震荡过滤**，表内定稿一字不改。

        这是回归保险，盯的是两件曾经踩过的事：
        ① 兜底方向曾经是「顺势」，等于把股指口径套到 72 个非股指品种上 —— 全市场
           78 个品种只有 7 个正收益、中位累计 −43.9%（换逆势后 −16.5%）。
        ② `flat_pct` 默认为 **0 = 不做过滤**（用户定的）：过滤能把爆仓压到 0、回撤
           收到 −22%，但中位收益反而略降、还拖坏 13 个品种，所以只做成可选开关。
           判据用「默认的零信号率必须与显式 0 逐位相同，且严格低于开 1% 时」。
        """
        import numpy as np
        import pandas as pd

        from ttrade.strategies import run_strategy

        rng = np.random.default_rng(3)
        stamps = []
        for k in range(14):
            base = pd.Timestamp("2024-03-04") + pd.Timedelta(days=k)
            stamps += [base + pd.Timedelta(minutes=m)
                       for m in list(range(540, 691, 5)) + list(range(810, 896, 5))]
        idx = pd.DatetimeIndex(stamps)
        nb = len(idx)
        px = 100.0 + 6.0 * np.sin(np.arange(nb) / 40.0) + np.cumsum(rng.normal(0, 0.05, nb))
        d = pd.DataFrame({"open": px, "high": px + 0.3, "low": px - 0.3,
                          "close": px, "volume": [1000] * nb}, index=idx)

        def zrate(o):
            return float((pd.to_numeric(o["signal"], errors="coerce") == 0).mean())

        d.attrs["symbol"] = "RB"
        o_def = run_strategy(d, "grid_trade", {})
        o_off = run_strategy(d, "grid_trade", {"flat_pct": 0.0})
        o_on = run_strategy(d, "grid_trade", {"flat_pct": 1.0})
        f_rb = float(pd.to_numeric(o_def["grid_follow"], errors="coerce").iloc[0])
        d.attrs["symbol"] = "IM"
        f_im = float(pd.to_numeric(run_strategy(d, "grid_trade", {})["grid_follow"],
                                   errors="coerce").iloc[0])

        if f_rb != 0.0:
            raise RuntimeError(f"表外品种兜底方向不是逆势（拿到 {f_rb}）——"
                               f"整批非股指品种会被股指口径拖死")
        if f_im != 1.0:
            raise RuntimeError(f"股指定稿的顺势被兜底改掉了（拿到 {f_im}）")
        z_def, z_off, z_on = zrate(o_def), zrate(o_off), zrate(o_on)
        if abs(z_def - z_off) > 1e-12:
            raise RuntimeError(f"震荡过滤不是默认关闭（默认零信号 {z_def:.1%} "
                               f"vs 显式 0 的 {z_off:.1%}）")
        if not z_on > z_off:
            raise RuntimeError(f"震荡过滤开关失效（开 1% 零信号 {z_on:.1%} "
                               f"没有多于关掉 {z_off:.1%}）")

        # hold：表外品种兜底 **持过夜**（`grid_flat` 全 False = 没有任何一根被
        # 时段规则强制平仓）；股指还是日内强平（必须有被强平的那几根）。
        d.attrs["symbol"] = "RB"
        flat_rb = int(pd.to_numeric(run_strategy(d, "grid_trade", {})["grid_flat"],
                                    errors="coerce").fillna(0).sum())
        d.attrs["symbol"] = "IM"
        flat_im = int(pd.to_numeric(run_strategy(d, "grid_trade", {})["grid_flat"],
                                    errors="coerce").fillna(0).sum())
        if flat_rb != 0:
            raise RuntimeError(f"表外品种兜底不是过夜（有 {flat_rb} 根被时段强平）")
        if flat_im == 0:
            raise RuntimeError("股指定稿的日内强平丢了（一根强平都没有）")
        return (f"RB 兜底 逆势+过夜+不过滤（零信号 {z_def:.0%}，开 1% 是 {z_on:.0%}，"
                f"强平 0 根）；IM 仍 顺势+日内（强平 {flat_im} 根）")

    check("网格分品种默认值（表外逆势 / 过滤默认不做 / 表内定稿不动）", _grid_symbol_defaults)




    def _factor_reverse():
        """逐因子反转：分数严格取负，仓位严格对调。"""
        import numpy as np

        from ttrade import factor_strategy as FS
        from ttrade import factors as F

        df = _us_df()
        key = F.DEFAULT_SINGLE
        cfg = FS.FactorConfig(window=252, horizon=5)
        rev = FS.FactorConfig(window=252, horizon=5, reverse={key: True})

        a = FS.run_factor_strategy(df, "single", [key], cfg)
        b = FS.run_factor_strategy(df, "single", [key], rev)
        # 阈值状态机左右对称，所以分数取负后仓位必须**严格**是相反数
        if not np.allclose(b.pack.score.fillna(0.0), -a.pack.score.fillna(0.0),
                           atol=1e-12):
            raise RuntimeError("反转后分数不是原分数的相反数")
        if not np.array_equal(b.df["signal"].to_numpy(), -a.df["signal"].to_numpy()):
            raise RuntimeError("反转后仓位不是原仓位的相反数")

        # 多因子只反一个：必须只影响那一列，其它因子照旧
        keys = list(F.DEFAULT_MULTI)
        m0 = FS.build_score(df, "multi", keys, cfg)
        one = keys[-1]
        m1 = FS.build_score(
            df, "multi", keys, FS.FactorConfig(window=252, horizon=5, reverse={one: True})
        )
        signed = (m0.fmat * m0.orient).copy()
        signed[one] = -signed[one]
        manual = signed.mean(axis=1, skipna=True)
        if not np.allclose(m1.score.fillna(0.0), manual.fillna(0.0), atol=1e-9):
            raise RuntimeError("多因子只反一个因子时，结果与手算不一致")

        n_long = int((a.df["signal"] > 0).sum())
        n_short = int((a.df["signal"] < 0).sum())
        return f"{F.FACTORS[key].label} 反转后多空对调（{n_long}/{n_short} 根）"

    check("因子反转（逐因子）", _factor_reverse)

    def _factor_orient():
        """方向口径三档：固定方向下「反转」必须是绝对的，跨品种可比。"""
        import numpy as np

        from ttrade import factor_strategy as FS
        from ttrade import factors as F

        df = _us_df()
        key = F.DEFAULT_SINGLE
        up = FS.build_score(df, "single", [key], FS.FactorConfig(window=252, horizon=5,
                                                                orient="long_high"))
        dn = FS.build_score(df, "single", [key], FS.FactorConfig(window=252, horizon=5,
                                                                orient="long_low"))
        # 固定高分做多 = 直接用因子值；固定高分做空 = 取负
        if not np.allclose(up.score.fillna(0.0), up.fmat[key].fillna(0.0), atol=1e-12):
            raise RuntimeError("long_high 下分数应等于因子值")
        if not np.allclose(dn.score.fillna(0.0), -dn.fmat[key].fillna(0.0), atol=1e-12):
            raise RuntimeError("long_low 下分数应等于因子值取负")

        # 固定做多 + 反转 ≡ 固定做空（这才是跨品种一致的绝对方向）
        flip = FS.build_score(df, "single", [key], FS.FactorConfig(
            window=252, horizon=5, orient="long_high", reverse={key: True}))
        if not np.allclose(flip.score.fillna(0.0), dn.score.fillna(0.0), atol=1e-12):
            raise RuntimeError("固定做多 + 反转 应严格等于 固定做空")

        # 自动模式下要能报出末期方向，否则用户看不出反转最后落在哪
        auto = FS.build_score(df, "single", [key], FS.FactorConfig(window=252, horizon=5))
        if not any("末期方向" in n for n in auto.notes):
            raise RuntimeError("自动定向缺少「末期方向」说明")
        return "固定多/固定空/自动三档正常，末期方向已报出"

    check("因子方向口径", _factor_orient)

    def _rt_periods():
        """每个周期都要能进实时 —— 除了 2 日 / 周线，它们必须被明确挡住。

        天勤日线的周期键是 `1d_tq`，而 `RT_SEC` 里只有源周期 `1d`。门禁要是按
        原键判断，日线策略一启用就报「实时不支持 1d_tq」直接停机 —— 界面上是
        一条红的「错误」，看不出是周期映射的问题。所以这里逐个周期真调一遍
        `enable()`，而不是只断言常量表包含关系。

        2 日 / 周线由日线合成，**不该**放行：2 日线按「连续两根日线成对」合成，
        每来一根新日线配对就整体错位一格，实时段和历史段会算出不同的 K 线；
        放行的话回测和实盘两边分数永远对不上，还不报错。所以这两档要求「有
        明确的错误提示」而不是「能跑」。
        """
        from ttrade import strategy_runner as R
        from ttrade import strategy_store as SStore
        from ttrade import tqdata
        from ttrade.tqlive import AccountSpec

        spec = AccountSpec(name="__selftest__", kind="sim")
        run = R.PortfolioRunner(spec)
        bad = []
        n_live = 0
        for p in tqdata.PERIOD_ORDER:
            sid = f"__rt_{p}"
            run.add(SStore.StrategyConfig(
                sid=sid, name=f"自检 {p}", symbol="RB", exchange="SHFE",
                period=p, family="factor", factor_mode="single",
                factor_keys=["rev_5"], account=spec.name))
            run.enable(sid, True)
            st = run._slots[sid]
            if tqdata.is_multiday(p):
                # 必须挡住，且要说清原因（不是「不支持这个周期」这种含糊话）
                if st.running or "由日线合成" not in (st.error or ""):
                    bad.append(f"{p}：合成周期没被挡住（{st.error or '居然启动了'}）")
            else:
                n_live += 1
                if not st.running or st.error:
                    bad.append(f"{p}：{st.error or '未启动'}")
        if bad:
            raise RuntimeError("；".join(bad))
        return f"{n_live} 个周期可实时，2 日 / 周线明确挡在门外"

    check("实时周期门禁", _rt_periods)

    def _ml_deps():
        from ttrade import mlmodels as ML

        why: dict = {}
        avail = ML.available(why=why)
        missing = [k for k in ML.MODELS if k not in avail]
        # 四个模型都在 requirements.txt 里，少一个就是打包漏了，不能算通过。
        # 之前这里只判「至少有一个」，于是 macOS x64 少了 XGBoost 也照样打勾，
        # 真正的失败点被推迟到下面的窗口检查才暴露，日志里看不出原因。
        if missing:
            raise RuntimeError(
                "缺少 "
                + "、".join(ML.MODELS[k]["label"] for k in missing)
                + " —— "
                + "；".join(f"{ML.MODELS[k]['label']}: {why.get(k, '?')}" for k in missing)
                + (
                    "｜"
                    + ML._xgboost_lib_hint()
                    if "xgboost" in missing
                    else ""
                )
            )
        return "、".join(ML.MODELS[k]["label"] for k in avail)

    check("机器学习模型依赖", _ml_deps)

    def _tushare():
        """Tushare 是期货日线 / 美股 / A 股日线的唯一来源。

        唯一的 import 在 `ttrade/futures.py:get_pro()` 里，而且是**函数内**的
        （惰性建 pro 客户端）。打包漏了它的表现是「一点更新日线就 ImportError，
        源码运行却完全正常」—— 用户只会看到一句 ModuleNotFoundError，看不出
        是打包问题，所以必须由自检来守。

        依赖也要一起验：tushare 自己 import 时会拉 lxml / tqdm / simplejson /
        bs4 / six，缺任何一个都是同样的报错形状。

        🔴 `six` 是 v2.6.19 实际翻车的那一个：tushare 声明的依赖里没有它，但
        `tushare/futures/intlfutures.py` 顶层 `import six`，而 `import tushare`
        会连带加载该子模块。CI 环境没装 six ⇒ 打包版一点「更新国内期货」就
        ModuleNotFoundError。所以 `tushare.futures.intlfutures` 也单列出来验。
        """
        import importlib

        need = ["tushare", "tushare.futures.intlfutures", "tushare.pro",
                "tushare.pro.client", "six", "lxml", "tqdm", "simplejson", "bs4"]
        miss, why = [], {}
        for m in need:
            try:
                importlib.import_module(m)
            except Exception as exc:      # noqa: BLE001
                miss.append(m)
                why[m] = f"{type(exc).__name__}: {exc}"
        if miss:
            raise RuntimeError(
                "缺 " + "、".join(miss) + "｜"
                + "；".join(f"{k}: {v}" for k, v in why.items())
                + "（requirements.txt 与 packaging/ChanQuant.spec 的 HIDDEN 里"
                  "都应当已列上 six / tushare）"
            )
        ver = getattr(importlib.import_module("tushare"), "__version__", "?")
        from ttrade import futures as FU

        try:
            tok = FU.read_token()
        except Exception:      # noqa: BLE001
            # 没配 token 不算失败：这是配置问题，不是包装错了
            return f"tushare {ver}、依赖齐全；token 未配置（日线 / 美股 / A 股下不了）"
        # 再真建一次 pro 客户端：这一步会把整条 import 链走完（含 futures /
        # intlfutures → six），但**不联网** —— DataApi 构造只做字符串拼接。
        # 只 import 模块名是不够的，漏的往往是链条中间那一段。
        try:
            FU.get_pro()
        except Exception as exc:          # noqa: BLE001
            raise RuntimeError(
                f"依赖都在但建 pro 客户端失败：{type(exc).__name__}: {exc}"
            ) from exc
        return f"tushare {ver}、依赖齐全、pro 客户端可建；token {len(tok)} 字符"

    check("Tushare 客户端（期货/美股/A股日线）", _tushare)

    def _matplotlib():
        os.environ.setdefault("MPLCONFIGDIR", cache_dir("mpl"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from ttrade.plotstyle import available_cjk_fonts, check_cjk_render, setup_matplotlib_fonts

        setup_matplotlib_fonts()
        fonts = available_cjk_fonts()
        if not fonts:
            raise RuntimeError("系统中没有任何中文字体，图表中文会显示为方框")

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(np.arange(10), np.arange(10) ** 2)
        ax.set_title("自检 · 中文标题")  # 缺字会在这里被 check_cjk_render 抓住
        out = os.path.join(cache_dir("mpl"), "selftest.png")
        check_cjk_render(fig, out)
        plt.close(fig)
        size = os.path.getsize(out)
        assert size > 1000, f"图片太小：{size}B"
        return f"字体 {fonts[0]}，渲染 {size} B"

    check("matplotlib 绘图（含中文字体）", _matplotlib)

    def _qt():
        from PySide6.QtCore import qVersion
        from PySide6.QtWidgets import QApplication, QLabel

        app = QApplication.instance() or QApplication([])
        w = QLabel("自检")
        w.resize(120, 40)
        w.show()
        app.processEvents()
        w.close()
        return f"Qt {qVersion()} 离屏渲染正常"

    check("Qt 平台插件", _qt)

    def _ctp():
        """CTP 客户端：装没装都要能跑，装了就得能构造出正确的报单参数。

        `openctp-ctp` 只有 Windows / Linux 的 wheel，macOS 上没有 —— 所以这条
        自检**不能因为没装就红**，否则 macOS 三平台构建全挂。真正要验的是
        「没装时优雅降级」和「装了时交易所解析 / 开平映射是对的」。
        """
        from ttrade import ctp_client as C

        cfg = C.CtpConfig(preset="simnow_7x24").fill_preset()
        assert cfg.md_front.startswith("tcp://"), f"预设没填上：{cfg.md_front!r}"
        assert cfg.td_front.startswith("tcp://"), f"预设没填上：{cfg.td_front!r}"
        # 自定义前置不能被预设覆盖
        mine = C.CtpConfig(preset="simnow_7x24", md_front="tcp://1.2.3.4:1000").fill_preset()
        assert mine.md_front == "tcp://1.2.3.4:1000", "手填的前置地址被预设覆盖了"

        tr = C.TraderClient(cfg)
        cases = {"rb2601": "SHFE", "TA701": "CZCE", "IF2412": "CFFEX",
                 "sc2412": "INE", "m2501": "DCE", "si2411": "GFEX",
                 "cu2601": "SHFE", "SR701": "CZCE"}
        bad = {k: tr._exchange_of(k) for k, v in cases.items() if tr._exchange_of(k) != v}
        assert not bad, f"交易所解析错了：{bad}"

        # 下单参数校验：没登录就该直接报错，而不是静默返回一个空单号
        try:
            tr.place("rb2601", "BUY", "OPEN", 3000.0, 1)
        except C.CtpError:
            pass
        else:
            raise RuntimeError("没登录时 place() 竟然没报错")

        if not C.HAS_CTP:
            return f"openctp-ctp 未安装（CTP 不可用，其余功能不受影响）：{C.CTP_IMPORT_ERROR[:60]}"
        return f"openctp-ctp 已装，{len(cases)} 个合约的交易所解析正确"

    check("CTP 客户端（可选依赖）", _ctp)

    def _ctp_live():
        """CTP 会话层：合约归一 / 主力解析 / tick 聚合 / 快照映射。

        这一层**不需要连柜台**就能验 —— 合约写法、主力表、tick 切 bar、资金
        持仓的字段映射全是纯函数，真正联网的那部分在 `TraderClient` 里。
        """
        from ttrade import ctp_live as CL
        from ttrade.tqlive import AccountSpec

        # 合约写法：Tushare 的 `.SHF` / `.ZCE` / `.CFX` 后缀、天勤的 `SHFE.rb`、
        # 以及纯 `RB2601` 都要收敛到 CTP 的 InstrumentID
        got = {
            "RB2601": CL.ctp_instrument("RB2601"),
            "RB2601.SHF": CL.ctp_instrument("RB2601.SHF"),
            "SHFE.rb2601": CL.ctp_instrument("SHFE.rb2601"),
            "TA2611.ZCE": CL.ctp_instrument("TA2611.ZCE", "CZCE"),
            "IF2611.CFX": CL.ctp_instrument("IF2611.CFX"),
            "SC2612.INE": CL.ctp_instrument("SC2612.INE"),
        }
        want = {"RB2601": "rb2601", "RB2601.SHF": "rb2601", "SHFE.rb2601": "rb2601",
                "TA2611.ZCE": "TA611", "IF2611.CFX": "IF2611", "SC2612.INE": "sc2612"}
        bad = {k: v for k, v in got.items() if v != want[k]}
        assert not bad, f"CTP 合约代码归一错了：{bad}（期望 {want}）"

        # 主力合约：从本地 aux 表取，查不到就空串（不是抛异常）
        main = CL.main_contract("RB", "SHFE")
        assert main.startswith("rb") and main[2:].isdigit(), f"螺纹主力解析错了：{main!r}"
        assert CL.main_contract("NOPE", "SHFE") == "", "不存在的品种应该返回空串"

        # tick → bar：累计量必须差分，不然成交量是一条单调上升的斜线
        b = CL._BarBuilder(period="1m")
        for stamp, vol, px in (("20261101 14:35:01", 100, 3000),
                               ("20261101 14:35:31", 120, 3010),
                               ("20261101 14:36:05", 150, 3020)):
            b.push({"datetime": stamp, "last": px, "volume": vol, "oi": 1})
        df = b.frame()
        assert len(df) == 2, f"1 分钟应该切出 2 根 bar，得到 {len(df)}"
        assert float(df["close"].iloc[0]) == 3010, f"第一根收盘错了：{df['close'].iloc[0]}"
        assert float(df["volume"].iloc[0]) == 20, (
            f"成交量要按累计量差分，第一根应该是 20 手，得到 {df['volume'].iloc[0]}")
        assert float(df["volume"].iloc[1]) == 30, "第二根成交量错了"

        # 快照映射：字段名对齐 `LiveState`，交易页直接拿去填表
        hub = CL.CtpHub([AccountSpec(name="ctp1", kind="ctp", user="123")])

        class _FakeTrader:
            logged_in = type("E", (), {"is_set": lambda s: True})()
            instruments = {"rb2611": {"tick": 1.0, "multiple": 10.0, "exchange": "SHFE"}}

            def snapshot(self):
                return {
                    "account": {"balance": 100000, "available": 80000, "margin": 20000,
                                "float_profit": 500, "position_profit": 500,
                                "close_profit": 0, "commission": 10, "pre_balance": 99000},
                    "positions": {"rb2611": {"instrument": "rb2611", "long": 2, "short": 0,
                                             "long_yd": 1, "long_today": 1, "short_yd": 0,
                                             "short_today": 0, "price": 60000.0,
                                             "profit": 500.0, "margin": 20000.0}},
                    "orders": {"1": {"order_ref": "1", "order_sys_id": "", "instrument": "rb2611",
                                     "direction": "0", "offset": "0", "price": 3000,
                                     "volume": 1, "traded": 0, "status": "未成交", "msg": ""}},
                    "trades": {},
                }

        hub.trader = _FakeTrader()
        st = hub.state()
        assert st.connected and st.accounts, "CTP 快照没生成账户行"
        row = st.accounts[0]
        assert row["day_profit"] == 1000.0, f"当日盈亏要减昨结算：{row['day_profit']}"
        assert row["platform"] == "CTP", row["platform"]
        pos = st.positions[0]
        assert (pos["pos_long"], pos["pos_long_today"]) == (2, 1), f"持仓今昨仓错了：{pos}"
        # 开仓均价 = OpenCost / (手数 × 乘数)。乘数取 10，漏了乘数会差一个数量级
        assert abs(pos["open_price_long"] - 3000.0) < 1e-6, (
            f"开仓均价错了（乘数没乘上？）：{pos['open_price_long']}")
        assert st.orders[0]["status"] == "ALIVE", "未成交的委托要标成 ALIVE，撤单才找得到"
        assert st.orders[0]["direction"] == "BUY" and st.orders[0]["offset"] == "OPEN"

        # 没连柜台时下单要报错，而不是返回一个空单号
        try:
            hub.place("rb2611", "BUY", "OPEN", 1, None)
        except Exception as exc:  # noqa: BLE001
            assert "限价" in str(exc) or "还没连接" in str(exc), f"报错信息不对：{exc}"
        else:
            raise RuntimeError("没连柜台时 place() 竟然没报错")

        # 委托状态表：这张表错一格，界面上活单会显示成「已撤单」、成交单显示成
        # 「未成交」。CTP 的 '3' 是「未成交还在队列中」，是活单 —— SimNow 实测踩过
        from ttrade.ctp_client import ALIVE_STATUS, _STATUS
        assert _STATUS["0"] == "全部成交" and _STATUS["5"] == "已撤单", _STATUS
        assert _STATUS["3"] == "未成交" and _STATUS["3"] in ALIVE_STATUS, (
            f"OrderStatus '3' 是活单，映射错了：{_STATUS['3']}")

        # 订阅行情：第二个参数是**个数**，传 0 柜台照样回「成功」但一条 tick 不推
        from ttrade.ctp_client import MdClient

        class _FakeApi:
            def __init__(self):
                self.calls = []

            def SubscribeMarketData(self, items, n):
                self.calls.append((list(items), n))

        fake = MdClient.__new__(MdClient)
        fake.api, fake.last_error = _FakeApi(), ""
        fake.subscribe(["rb2611", "cu2612"])
        items, n = fake.api.calls[0]
        assert n == len(items) == 2, f"订阅个数要传实际数量，传 {n} 会静默无 tick"

        # 查合约：传合约号要带上 InstrumentID（单条秒回），不传才是全量。
        # 这段要构造 CTP 的请求对象，所以只能在装了 openctp-ctp 的平台上跑 ——
        # macOS 没有这个包（只有 Win/Linux wheel），跳过而不是判失败
        from ttrade.ctp_client import HAS_CTP, TraderClient

        if HAS_CTP:
            got_req = []

            class _FakeTdApi:
                def ReqQryInstrument(self, req, rid):
                    got_req.append(req.InstrumentID)

            import threading

            td = TraderClient.__new__(TraderClient)
            td.api, td._req = _FakeTdApi(), 0
            td.instruments_done = threading.Event()
            td.query_instruments("rb2611")
            td.query_instruments()
            assert got_req == ["rb2611", ""], f"单合约查询没带 InstrumentID：{got_req}"

        return "合约归一 / 主力解析 / tick 切 bar / 资金持仓映射 / 状态表 全对"

    check("CTP 会话层（交易页后端）", _ctp_live)

    def _window():
        """真正把主窗口搭起来，验证 UI 代码在打包后也能跑通。"""
        from PySide6.QtWidgets import QApplication

        from gui.app import (
            COL_PERIOD,
            FAMILY_RULE,
            STYLE_SHEET,
            MainWindow,
            setup_light_palette,
        )

        app = QApplication.instance() or QApplication([])
        setup_light_palette(app)
        app.setStyleSheet(STYLE_SHEET)

        win = MainWindow()
        # 启动**绝不能**连天勤：研究页只看本地库，账户连接归「账户透视和交易」页。
        # 早先这里会「上膛」并在 1.5 秒后自动接一次，于是每看一次回测、每换一个
        # 品种都可能登录一遍账户。构造完立刻判，再跑几轮 processEvents 之后也要
        # 还是没连 —— 那样才说明没有任何延时定时器偷偷把连接拉起来。
        live_off = win.live.running is False
        win.resize(1280, 820)
        win.show()
        app.processEvents()
        # 首屏该是 ChanAgent。这一条要**在切页之前**取 —— 后面几项都要看研究页
        # 那张图，而停在不显示研究页的时候 K 线是故意不画的（只记一笔，切回来
        # 再补），所以下面会切过去
        first_page_agent = win.pages.currentWidget() is win.agent_page
        win._switch_page(0)
        app.processEvents()
        # 启动时首次回测是延后 220ms 排队的（先让窗口画出来再算）。自检里没有
        # 真的事件循环，这里直接把它跑完 —— 否则它会挑在后面测滚轮 / 组合的
        # 时候插进来重画一次，窗口被重置，滚轮那项就时好时坏。
        if getattr(win, "_pending", False):
            win._deferred_run()
            app.processEvents()
        # 恢复的 `last_config` 可能是「因子 + 机器学习」—— 那条首屏走后台线程，
        # 画布要等它跑完才有东西。自检要看的是「一条规则策略的 K 线 + 绩效」，
        # 先把在飞的任务停掉、切回规则族同步算一遍；不然这一项的结果取决于用户
        # 上次存了什么配置（写死初始状态的教训见下面 is_us 那段）
        if getattr(win, "res", None) is None:
            win._cancel_factor_worker()
            # 恢复的配置可能是多品种：启动时它**故意**不自动回测（N 条曲线要跑
            # 几分钟），而且这一项要验的是「一条规则策略的 K 线 + 绩效」。先退回
            # 单品种再算，别让这一项的结果取决于用户上次存了几个品种
            win.sym_pick.clear()
            win.cb_family.setCurrentIndex(win.cb_family.findData(FAMILY_RULE))
            win.run_analysis()
            app.processEvents()
        n = len(win.store.symbols)
        has_res = getattr(win, "res", None) is not None
        # 跑过几轮事件循环之后再看一眼：延时定时器如果有，这会儿该触发了
        live_off = live_off and (win.live.running is False)
        # 数据管理页：五个页面按钮一一对应，补数据的控件在数据页里。
        # 衍生数据的两个勾选框只在期货市场出现 —— 美股没有真实合约和期权，
        # 摆在那里只会让人以为「美股也能下期限结构」。
        # **按当前市场判断，不能写死 hidden**：窗口启动时会恢复上次的配置
        # （`last_config.json`，可能是期货），写死的话用户上次用期货、这次跑
        # 自检就必红 —— 而那不是代码坏了，是自检假设了初始状态。
        is_us = win.market == "us"
        dp = getattr(win, "data_page", None)
        from ttrade import updater as _upd

        src_tq = dp is not None and dp.cb_source.currentData() == "tq"
        tq_vis = is_us or not src_tq

        def _form_order_ok(d) -> bool:
            """「开始补充」必须紧跟在下载项之后，成本和定时不许混进来。

            以前它压在「每日自动补充」和「交易成本」两段下面，看着像「把定时
            和成本一起提交」，点下去却只补数据。这条断言防的就是它被搬回去。
            """
            lay = d.form_card.layout()
            idx = {}
            for i in range(lay.count()):
                w = lay.itemAt(i).widget()
                if w is not None:
                    idx[w] = i
            if d.btn_update not in idx:
                return False
            i_btn = idx[d.btn_update]
            # 期权勾选必须在按钮前面（同一张卡里）
            if idx.get(d.chk_upd_options, 10 ** 6) > i_btn:
                return False
            # 成本 / 定时一个都不许留在数据源卡里
            for w in (d.chk_auto, d.chk_daily, d.cb_cost, d.lbl_cost):
                if w in idx:
                    return False
            # 凭据横条也不许回到数据源卡里 —— A 股那页数据源卡是藏着的，
            # 藏进去就配不了 Tushare token，而 A 股股票库恰恰只能靠它
            if d.btn_token in idx or d.btn_tq_acct in idx:
                return False
            # 它们得在参数卡里，而且「开始补充」不在那张卡里
            clay = d.cfg_card.layout()
            for w in (d.chk_auto, d.chk_daily, d.cb_cost):
                if clay.indexOf(w) < 0:
                    return False
            return clay.indexOf(d.btn_update) < 0

        def _sched_ok(d) -> bool:
            """两条每日补充日程要和**存盘的那份**一致。

            早先这里写死「默认不勾 / 15:30 / 17:00」—— 用户把每日补充打开之后
            自检就必红，可那是用户设置，不是代码坏了。没存过盘才该是默认值。
            """
            try:
                with open(d._auto_path(), encoding="utf-8") as fh:
                    saved = json.load(fh)
            except Exception:  # noqa: BLE001
                saved = {}
            if "aux" not in saved and ("enabled" in saved or "time" in saved):
                saved = {"aux": saved}        # 老版本只有一条，存的是扁平结构
            for key, chk, tm, rng in (
                ("aux", d.chk_auto, d.time_auto, d.cb_auto_range),
                ("daily", d.chk_daily, d.time_daily, d.cb_daily_range),
            ):
                one = saved.get(key) or {}
                if bool(one.get("enabled", False)) != chk.isChecked():
                    return False
                if one.get("time") and tm.time().toString("HH:mm") != str(one["time"]):
                    return False
                if one.get("range") and rng.currentData() != one["range"]:
                    return False
            return True

        has_upd = (
            dp is not None
            and win.pages.count() == 6
            and len(win._page_btns) == 6
            and win.pages.widget(5) is dp
            # 策略验证页排在数据页之前，序号要和按钮条对得上
            and win.pages.widget(4) is getattr(win, "verify_page", None)
            and dp.market == win.market
            and not dp.busy()
            and dp.cb_upd_range.count() == len(_upd.RANGES)
            and dp.cb_source.count() == 2
            # 市场单选归数据页自己管，启动时跟研究页一致
            and len(dp._market_btns) == 3
            and dp._market_btns[dp.market].isChecked()
            # A 股股票库面板：五个下载项 + 天勤分钟线开关 + 库现状
            and hasattr(dp, "stock_card")
            and len(dp.stock_checks) == 5
            and hasattr(dp, "chk_stock_tq")
            and dp.stock_card.isHidden() == (dp.market != "cn_stock")
            and dp.form_card.isHidden() == (dp.market == "cn_stock")
            # 周期 / 回溯区间 / 真实合约只属于天勤那一路
            and dp.cb_period.isHidden() == tq_vis
            and dp.cb_tq_range.isHidden() == tq_vis
            and dp.chk_tq_contracts.isHidden() == tq_vis
            and dp.chk_upd_contracts.isHidden() == (is_us or src_tq)
            and dp.chk_upd_options.isHidden() == (is_us or src_tq)
            and not dp.chk_upd_contracts.isChecked()
            and not dp.chk_upd_options.isChecked()
            # 每日定时补充：两条日程（15:30 衍生 / 17:00 日线）的控件都在、
            # 默认不勾（不能让用户装完就被自动跑一次几十分钟的下载），
            # 区间都是「近 1 天 / 近 1 周」两档
            and hasattr(dp, "chk_auto")
            and hasattr(dp, "time_auto")
            and hasattr(dp, "cb_auto_range")
            and hasattr(dp, "chk_daily")
            and hasattr(dp, "time_daily")
            and hasattr(dp, "cb_daily_range")
            and dp.cb_auto_range.count() == 2
            and dp.cb_daily_range.count() == 2
            and _sched_ok(dp)
            # 三栏并列：数据源 / 参数（成本 + 定时）/ 日志。成本和定时不许
            # 再回到数据源卡里 —— 回到一起，「开始补充」就又夹在它们中间了
            and hasattr(dp, "cfg_card")
            and hasattr(dp, "log_card")
            and _form_order_ok(dp)
            # 两条通道各自的凭据入口（Tushare token / 天勤账号）
            and hasattr(dp, "btn_token")
            and hasattr(dp, "btn_tq_acct")
            and hasattr(dp, "lbl_token")
            and hasattr(dp, "lbl_tq_acct")
            and "token" in dp.lbl_token.text()
        )
        # 账户透视页：多账户列表 + 三张表都要带「资金账户」列，委托/成交还要
        # 带「时间」。这些列是逐列摆出来的，漏一列不会报错、只会让人看不出来
        # 某一笔是哪个户做的，所以这里按表头文字验一遍。
        tp = getattr(win, "trade_page", None)

        def _heads(tbl):
            return [tbl.horizontalHeaderItem(i).text()
                    for i in range(tbl.columnCount())]

        trade_ok = (
            tp is not None
            and _heads(tp.tbl_accts) == [
                "账户", "账户平台", "资金账户", "最新权益", "当日盈亏", "净杠杆",
                "总杠杆", "可用资金", "成交金额", "持仓保证金", "手续费"]
            and _heads(tp.tbl_pos)[:2] == ["资金账户", "合约"]
            and _heads(tp.tbl_orders)[:2] == ["资金账户", "时间"]
            and _heads(tp.tbl_trades)[:2] == ["资金账户", "时间"]
            # 默认全勾：以前是「下拉框选中谁就连谁」，现在多个账户并存
            and tp._checked == {s.name for s in tp._accounts}
        )
        trade_info = (f"{tp.tbl_accts.columnCount()} 列账户表 / "
                      f"{len(tp._accounts)} 个账户" if tp is not None else "无交易页")
        # ChanAgent 页：删除按钮在没选中东西时必须是灰的，权限开关默认关。
        # 升级权限的文件模块是**函数里 import** 的，打包漏了只会在用户勾上开关
        # 之后才炸 —— 所以在这里 import 一次，把它变成打包期就能发现的问题。
        ap = getattr(win, "agent_page", None)
        fs_ok = True
        fs_err = ""
        try:
            from ttrade import agent_fs  # noqa: F401

            root = agent_fs.root()
            fs_ok = bool(root) and os.path.isdir(root)
        except Exception as exc:  # noqa: BLE001
            fs_ok, fs_err = False, f"{type(exc).__name__}: {exc}"
        # 平台手册同样是函数里 import 的。而且它靠 inspect 取源码 —— 打包后
        # 如果 .py 没打进去，手册会退化成一堆空代码块，界面上看不出来，
        # 模型照着学就是一片空白。这里要求每段都有代码。
        manual_ok = True
        manual_info = ""
        try:
            from ttrade import factor_manual as M
            from ttrade.factors import FACTORS

            sec = M.factor_section("mom_20")
            manual_ok = "def f_mom_20(" in sec and "def _ret(" in M.primitive_section()
            manual_info = f"{len(M.manual())} 字符 / {len(FACTORS)} 个因子"
        except Exception as exc:  # noqa: BLE001
            manual_ok, manual_info = False, f"{type(exc).__name__}: {exc}"
        # 回测结论要推到 Agent 页右下角的输出框。以前这些数字只活在绩效卡上，
        # 模型问「结果是多少」时用户根本没有一处能复制 —— 这条链断了就是老毛病复发
        out_ok = False
        out_info = ""
        try:
            text = win._result_text()
            out_ok = bool(text) and bool(getattr(ap, "_last_output", ""))
            out_info = f"{len(text)} 字符"
        except Exception as exc:  # noqa: BLE001
            out_info = f"{type(exc).__name__}: {exc}"
        # 对话区渲染：模型回的 markdown 必须变成标签，流式那遍纯文本要被抹干净。
        # 这条以前是断的 —— 表格、粗体、代码块全按原文符号显示，用户看到的
        # 是一屏 `**` 和竖线。这里直接往 `ap` 上灌一段回答走完整条流式路径。
        chat_ok = False
        chat_info = ""
        try:
            reply = ("## 结论\n\n螺纹**不适合追多**。\n\n"
                     "| 参数 | 值 |\n|---|---|\n| fast | 50 |\n")
            ap.log.clear()
            ap._stream_buf = ""
            ap._msg_start = None
            for i in range(0, len(reply), 9):
                ap._on_delta(reply[i:i + 9])
            ap._on_done(reply)
            app.processEvents()
            plain = ap.log.toPlainText()
            html = ap.log.toHtml()
            chat_ok = (
                "**" not in plain
                and "<table" in html
                and "font-weight:700" in html
                and plain.count("结论") == 1          # 流式文本确实被替换掉了
                and hasattr(ap.ed_input, "files_dropped")   # 输入框是 ChatInput
            )
            chat_info = f"{len(plain)} 字符 / {'表格' if '<table' in html else '无表格'}"
        except Exception as exc:  # noqa: BLE001
            chat_info = f"{type(exc).__name__}: {exc}"

        # 能力包注册表 + 附件资源化。打包时 `ttrade.skills.*` 是靠 spec 的
        # hiddenimports 收进去的，漏一个包不会报错，只会表现为「模型给了动作块、
        # 平台回没有这个动作」。这里造一份临时 xlsx 走完整条取数路径。
        skill_ok = False
        skill_info = ""
        try:
            import tempfile as _tf

            import pandas as pd

            from ttrade import session_files as _SF
            from ttrade import skillpacks as _SP
            from ttrade import skills as _SK
            from ttrade import mcp_server as _MC
            from ttrade.skills import monitor as _MON

            n_tool = len(_SK.REGISTRY)
            tmpd = _tf.mkdtemp(prefix="td_self_")
            try:
                book = os.path.join(tmpd, "t.xlsx")
                pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_excel(book, index=False)
                sess = _SF.SessionFiles()
                item = sess.add_path(book)
                res = _SK.dispatch(_SK.SkillContext(session=sess),
                                   "read_excel", {"file_id": "f1"})
            finally:
                import shutil as _sh

                _sh.rmtree(tmpd, ignore_errors=True)
            # 盯盘扫描：造一组「前面 20 根都在 100、最后一根 120」的假数据
            _hit = _MON._check(
                pd.DataFrame({"close": [100.0] * 21 + [120.0],
                              "high": [101.0] * 21 + [121.0],
                              "low": [99.0] * 22,
                              "volume": [100.0] * 22}),
                {"rule": "breakout_high", "window": 20},
            )
            scan_ok = _hit is not None and _hit["dist"] > 0
            # MCP 导出的两条硬要求：schema 是标准 JSON Schema、下单永不暴露
            mcp_ok = (
                _SK.get("query_bars").spec()["inputSchema"]["properties"]["rows"]["type"]
                == "integer"
                and "trade" not in {t.name for t in _MC.exposed(True)}
            )
            # 技能（SKILL.md）：索引要非空、正文要读得出来。用户自己加的
            # 技能在打包版里也应该能扫到 —— 目录走 `paths.user_dir()`
            _sp = _SP.usable_packs()
            sk_ok = bool(_sp) and "`" in _SP.index_text() and bool(
                _SP.read(_sp[0].name)[0])
            skill_ok = (
                n_tool >= 17
                and item is not None
                and res.ok
                and any(b["type"] == "table" for b in res.blocks)
                and hasattr(ap, "session")
                and hasattr(ap, "_ctx")
                and hasattr(ap, "store")
                and scan_ok
                and mcp_ok
                and sk_ok
            )
            skill_info = (f"{n_tool} 个动作，技能 {len(_sp)} 份"
                          f"（{'可读' if sk_ok else '读不出'}），"
                          f"附件读表 {'通' if res.ok else '断'}，"
                          f"扫描 {'通' if scan_ok else '断'}，"
                          f"MCP {'通' if mcp_ok else '断'}")
        except Exception as exc:  # noqa: BLE001
            skill_info = f"{type(exc).__name__}: {exc}"

        agent_ok = (
            ap is not None
            and hasattr(ap, "btn_del_factor")
            and hasattr(ap, "btn_del_strat")
            and hasattr(ap, "btn_manual")
            # 附件与运行输出也是函数里 import / 新建的控件，缺一个功能就没了
            and hasattr(ap, "btn_attach")
            # Skills 管理挂在附件左边，少了它技能就只能靠改文件
            and hasattr(ap, "btn_skills")
            and hasattr(ap, "out")
            and hasattr(ap, "btn_out_feed")
            # 因子审核 / 深度分析 / 手册：三个按钮，前两个没选中因子时必须灰着
            and hasattr(ap, "btn_audit")
            and hasattr(ap, "btn_deep")
            and hasattr(ap, "btn_handbook")
            and not ap.btn_audit.isEnabled()
            and not ap.btn_deep.isEnabled()
            and not ap.btn_del_factor.isEnabled()
            and not ap.btn_del_strat.isEnabled()
            and ap.cfg.allow_fs is False
            and fs_ok
            and manual_ok
            and out_ok
            and chat_ok
            and skill_ok
        )
        # K 线可视窗口：初始应停在最新一段，且滚动条范围与窗口自洽。
        # 这些状态现在归 `ResultView` 管（研究页和组合页共用一份实现），
        # 所以要从 `win.view` 上读，而不是主窗口自己。
        rv = getattr(win, "view", None)
        k_ok = (
            rv is not None
            and hasattr(rv, "sb_kline")
            and rv._k_total > 0
            and 0 < rv._k_win <= rv._k_total
            and rv._k_start == max(0, rv._k_total - rv._k_win)
        )
        k_info = f"窗口 {rv._k_win}/{rv._k_total} 根" if rv is not None else "无结果视图"

        # 研究页的「启动实时」：挂在 K 线与信号页的标题行上，默认没点；换品种 /
        # 换周期必须能把它复位（这里不真的连天勤 —— 自检不许联网）。
        chart_page = rv.pages()[0][1] if rv is not None else None
        live_btn_ok = (
            hasattr(win, "btn_live")
            and chart_page is not None
            and chart_page.isAncestorOf(win.btn_live)
            and not win.btn_live.isChecked()
            and win.btn_live.text() == "启动实时"
        )
        # 手动置成「已点」再走一次换品种的复位路径，必须回到未点状态
        win.btn_live.blockSignals(True)
        win.btn_live.setChecked(True)
        win.btn_live.setText("■ 实时中")
        win.btn_live.blockSignals(False)
        win._stop_study_live()
        live_btn_ok = (live_btn_ok and not win.btn_live.isChecked()
                       and win.btn_live.text() == "启动实时"
                       and win.study_live.running is False)

        # 因子面板：三个策略族、三种因子模式、47 个因子、四类模型都得在
        fams = {win.cb_family.itemData(i) for i in range(win.cb_family.count())}
        modes = {win.cb_fmode.itemData(i) for i in range(win.cb_fmode.count())}
        filters = {win.cb_filter_when.itemData(i) for i in range(win.cb_filter_when.count())}
        orients = {win.cb_forient.itemData(i) for i in range(win.cb_forient.count())}
        n_factors = win.cb_fsingle.count()
        n_models = win.cb_fmodel.count()
        n_rows_f = win.tbl_fmulti.rowCount()
        n_cols_f = win.tbl_fmulti.columnCount()
        # 「周期」列必须**点得开**：`InstantPopup` 只负责「点一下弹菜单」，菜单
        # 本身要 `setMenu()` 挂上去。漏了那一句，按钮长得完全正常、点下去一点
        # 反应都没有，而别的自动检查全是绿的 —— v2.6.11 就是这么漏出去的。
        # 这里只验「菜单挂着 + 勾选能连勾两档」，不碰窗口里的真实勾选（会排一次
        # 回测，把后面几项读数带偏）
        period_ok = False
        period_info = ""
        try:
            _btn = win.tbl_fmulti.cellWidget(0, COL_PERIOD)
            _menu = _btn.menu()
            _n_act = len(_menu.actions())
            _tmp = type(_btn)()          # 另建一个，不动窗口里的状态
            _t = _tmp.menu().actions()
            _t[1].trigger()
            _t[2].trigger()
            _two = len(_tmp.periods())
            _t[1].trigger()              # 再点一次 = 取消勾选
            _one = len(_tmp.periods())
            period_ok = (_menu is not None
                         and _n_act == win.cb_fperiod.count()
                         and all(a.isCheckable() for a in _menu.actions())
                         and _two == 3 and _one == 2)
            period_info = (f"{type(_btn).__name__} 菜单 {_n_act} 项，"
                           f"连勾两档 {_two} 档 / 取消后 {_one} 档")
        except Exception as exc:  # noqa: BLE001
            period_info = f"{type(exc).__name__}: {exc}"

        factor_ok = (
            fams == {"rule", "factor", "cross"}
            and modes == {"single", "multi", "ml"}
            and filters == {"entry", "delay", "hold"}
            # 方向口径三档必须都在，否则「反转」只能相对自动定向，跨品种不可比
            and orients == {"auto", "long_high", "long_low"}
            and n_factors >= 47
            and n_models >= 4
            # 多因子表要 47 行 × 5 列（因子 / 周期 / 用 / 反 / 权重），少了「反」
            # 列就没法逐因子反转，少了「权重」列「不等权」就是空话，少了「周期」
            # 列就没法跨周期引用
            and n_rows_f >= 47
            and n_cols_f == 5
            and period_ok
            and win.chk_frev is not None
            # 权重 / 合成口径五档（等权 / 因子与 / 因子或 / 不等权 / 滚动 IC）
            # + 打分范围说明 + 出场规则入口。因子与、因子或是「合成口径」而不是
            # 权重，两档都不读 weights，但必须和另外三档并列在同一个下拉里。
            and {win.cb_fweight.itemData(i) for i in range(win.cb_fweight.count())}
            == {"equal", "custom", "ic", "and", "or"}
            # 标准化口径两档（z-score / 分位）—— 两档都映射到 ±4，
            # 少一档就只能用 z，极端值敏感的那批品种没得换
            and {win.cb_fmethod.itemData(i) for i in range(win.cb_fmethod.count())}
            == {"z", "rank"}
            and "±4" in win.lbl_frange.text()
            and win.btn_fexit is not None
            # 多品种勾选 / 截面参数 / 成本档位 / 两套结果视图
            and win.sym_pick is not None
            and win.cb_cross_mode.count() >= 2
            and win.cb_cross_w.count() >= 2
            and win.cb_timing.count() >= 2
            and win.view_stack.count() == 2
            and win.cb_cost.count() >= 3
            # 回测触发：一个显式按钮 + 一个「改参数自动回测」开关。开关**出厂是
            # 关的** —— 勾上要先弹框讲清楚「以后每动一次界面都重跑」，不然改个
            # 滑块等三分钟会以为死机
            and win.btn_run.text() == "策略回测"
            and not win.chk_auto.isChecked()
            and win.chk_auto.isEnabled()
            # 多品种回测在后台跑，只有跑着的时候才露出「停止回测」
            and win.btn_stop.isHidden()
            # 首屏是 ChanAgent，不是研究页
            and first_page_agent
            # 资金分配默认风险平价（第一项）
            and win.cb_cross_w.itemData(0) == "risk"
        )

        # 勾选面板：点一个品种**不能把面板关掉**（`QMenu` 的 checkable action 就会），
        # 勾选过程中也不能触发重算 —— 一次截面回测几秒，勾 6 个重算 6 遍没法用。
        # 早先那版就是栽在这两条上，用户的原话是「点完一个这个页面就推出去」。
        pick_ok = False
        pick_info = ""
        try:
            from gui.pickers import SymbolPicker

            pk = SymbolPicker("自检", win)
            pk.move(10, 10)
            pk.show()
            app.processEvents()
            pk.set_items([
                {"key": "A_SHFE", "label": "A", "exchange": "SHFE"},
                {"key": "B_SHFE", "label": "B", "exchange": "SHFE"},
                {"key": "C_DCE", "label": "C", "exchange": "DCE"},
            ], keep=["A_SHFE"])
            hits: list[int] = []
            pk.changed.connect(lambda: hits.append(1))
            pk._toggle()
            app.processEvents()
            opened = pk._panel.isVisible()
            pk._panel._boxes["B_SHFE"].setChecked(True)
            app.processEvents()
            still_open = pk._panel.isVisible()
            picked = pk.selected()
            during = len(hits)
            pk._panel.hide()
            app.processEvents()
            after = len(hits)
            pk._panel._toggle_group([{"key": "C_DCE", "label": "C", "exchange": "DCE"}])
            group_sel = pk.selected()
            pk._panel._set_all(True)
            all_n = len(pk.selected())
            pk._panel._set_all(False)
            pick_ok = (opened and still_open and during == 0 and after == 1
                       and picked == ["A_SHFE", "B_SHFE"]
                       and group_sel == ["A_SHFE", "B_SHFE", "C_DCE"]
                       and all_n == 3 and not pk.selected())
            pick_info = (f"打开={opened} 勾选中不关={still_open} "
                         f"勾选中触发={during} 关闭后触发={after}")
            pk.deleteLater()
        except Exception as exc:  # noqa: BLE001
            pick_info = f"{type(exc).__name__}: {exc}"
        factor_ok = factor_ok and pick_ok
        factor_info = (f"{len(fams)} 族 / {len(modes)} 模式 / {n_factors} 因子 / "
                       f"{n_models} 模型 / 勾选 {pick_info}")

        # 因子分析页：表格应铺满全因子，四格图应建出 4 个轴
        win.tabs.setCurrentIndex(2)
        app.processEvents()
        n_rows = rv.tbl_factor.rowCount()
        n_axes = len(rv.canvas_f.fig.axes)
        win.tabs.setCurrentIndex(0)
        app.processEvents()

        # K 线滚轮缩放：走真实的事件派发路径，不直接调 _on_kline_wheel。
        # 必须验这一条，因为 matplotlib 的回调注册表挂在 Figure 上
        # （canvas.callbacks 是 figure._canvas_callbacks 的代理），MplCanvas.fresh()
        # 换 Figure 就会让之前 mpl_connect 的连接全部静默失效 —— 滚轮缩放
        # 正是这么失灵过一次的，而且界面不会报任何错。
        from matplotlib.backend_bases import MouseEvent

        def _wheel(step: int) -> tuple[int, int]:
            if not rv.canvas_k.fig.axes:
                # 别让 IndexError 冒出去 —— 那只会告诉人「list index out of
                # range」，看不出是「首屏根本没画 K 线」。这条自检真栽过：
                # 全新用户目录下没跑首屏回测，图上连轴都没有
                raise RuntimeError(
                    f"K 线画布没有轴（窗口 {rv._k_win}/{rv._k_total} 根，"
                    f"待补画 {len(rv._pending_draw)} 张）：首屏没跑回测或画布被推迟"
                )
            ax = rv.canvas_k.fig.axes[0]
            bb = ax.get_window_extent()
            before = rv._k_win
            MouseEvent(
                "scroll_event", rv.canvas_k,
                (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2, step=step,
            )._process()
            app.processEvents()
            return before, rv._k_win

        w0, w1 = _wheel(+1)   # 向上滚 = 放大 = 窗口变窄
        w2, w3 = _wheel(-1)   # 向下滚 = 缩小 = 窗口变宽
        wheel_ok = w1 < w0 and w3 > w2
        wheel_info = f"{w0}→{w1}→{w3} 根"

        # 波幅突破三条（从 future_quant 移植，日盘 / 夜盘拆成两条）：研究页要能
        # 正常出参数滑杆，
        # 参数个数和 `param_meta` 对不上就会少一根 —— 那种错不报异常，
        # 只是参数悄悄调不了
        brk_ok = False
        brk_info = ""
        try:
            keep = win.cb_strategy.currentIndex()
            counts = {}
            for key in ("ria_break_day", "ria_break_night", "cta_break"):
                i = win.cb_strategy.findData(key)
                if i < 0:
                    continue
                win.cb_strategy.setCurrentIndex(i)
                app.processEvents()
                counts[key] = len(win.param_widgets)
            win.cb_strategy.setCurrentIndex(keep)
            app.processEvents()
            brk_ok = (counts.get("ria_break_day") == 6
                      and counts.get("ria_break_night") == 6
                      and counts.get("cta_break") == 6
                      and win.sidebar.width() >= 340)
            brk_info = f"{counts} / 侧边栏 {win.sidebar.width()}px"
        except Exception as exc:  # noqa: BLE001
            brk_info = f"{type(exc).__name__}: {exc}"
        if not brk_ok:
            raise RuntimeError(
                f"波幅突破策略没挂上（{brk_info}）：研究页的参数滑杆数量不对，"
                "或者侧边栏宽度没生效"
            )

        # 策略验证页：取研究页当前策略 → 跑一遍前视检测 → 检查点表要有行，
        # 而且结论必须是「一致」（内置策略不该有前视偏差）。这条同时验了后台
        # 线程收尾、表格填充和重放画图 —— 三处都只在真的跑起来时才可能出错
        verify_ok = False
        verify_info = ""
        try:
            vp = getattr(win, "verify_page", None)
            win._switch_page(4)
            app.processEvents()
            vp._take_from_study()
            app.processEvents()
            vp._start_check()
            t0 = time.time()
            while vp._thread is not None and time.time() - t0 < 120:
                app.processEvents()
                time.sleep(0.05)
            rep = vp.report
            verify_ok = (
                rep is not None
                and rep.ok
                and len(rep.checks) >= 4
                and vp.tbl_check.rowCount() == len(rep.checks)
                and len(rep.df) > 0
            )
            verify_info = (f"{len(rep.checks)} 个检查点全一致" if rep is not None
                           and rep.ok else (vp.lbl_verdict.text()[:80] or "没跑出结果"))
        except Exception as exc:  # noqa: BLE001
            verify_info = f"{type(exc).__name__}: {exc}"
        finally:
            win._switch_page(0)
            app.processEvents()

        # 组合报告：两条策略等权。一次验三件事 —— 组合页能不能复用同一套
        # 结果视图（不能就是一片空白）、组合有没有三页（绩效/交易/阶段）、
        # 绩效表是不是只有两列（多出「买入持有」那列说明还在按单策略画）
        combo_ok = False
        combo_info = ""
        try:
            from ttrade import strategy_store as SStore
            from ttrade.strategies import STRATEGIES

            key = win.cb_symbol.currentData() or ""
            if win.market == "cn":
                sym, _, ex = key.rpartition("_")
            else:
                sym, ex = key, ""
            note = "short" if win._short_enabled() else ""
            cfgs = [
                SStore.StrategyConfig(
                    symbol=sym, exchange=ex, period=win._period(), rule_key=k,
                    rule_params=dict(STRATEGIES[k].params), note=note,
                )
                for k in ("ma_cross", "donchian")
            ]
            text = win.portfolio_page.show_combo(cfgs)
            tabs, pv = win.portfolio_page._views.get("portfolio", (None, None))
            combo_ok = (
                tabs is not None
                and pv is not None
                and tabs.count() == 4
                and pv.table.columnCount() == 2
                and pv.res is not None
                and "组合（2 条策略）" in text
                # 「补数据」按钮：期权 / 展期数据过期时，这一页要能就地补
                and hasattr(win.portfolio_page, "btn_data")
            )
            combo_info = (f"{tabs.count()} 页 / {pv.table.columnCount()} 列"
                          if tabs is not None else "未建出视图")
        except Exception as exc:  # noqa: BLE001
            combo_info = f"{type(exc).__name__}: {exc}"

        # 截面策略跑完：交易记录必须有货（用户报过「绩效说调仓 406 次、交易
        # 记录一片空白」），并且要长出「截面因子分析」页
        cross_ok = False
        cross_info = ""
        try:
            import numpy as _np
            import pandas as _pd

            from ttrade import cross as _cross
            from ttrade.backtest import BacktestConfig as _Bt

            cidx = _pd.date_range("2024-01-01", periods=200, freq="B")
            cframes = {}
            for i, d in enumerate([0.002, 0.0012, 0.0004,
                                   -0.0004, -0.0012, -0.002]):
                px = _pd.Series(100.0 * (1 + d) ** _np.arange(len(cidx)), index=cidx)
                cframes[f"C{i}_X"] = _pd.DataFrame({
                    "open": px, "high": px, "low": px, "close": px,
                    "volume": 1000.0,
                })
            ccfg = _cross.CrossConfig(mode="hedge", rebalance=5, top_pct=10,
                                      bot_pct=10, factor_keys=["mom_20"])
            cbt = _Bt()
            # **走真正的后台线程**：截面回测现在整条路都在 `CrossWorker` 里跑。
            # 以前只有机器学习那条路异步，单因子 / 多因子同步算，几十个品种的
            # 5 分钟历史一跑窗口就假死
            win._start_cross_worker(cframes, ccfg, cbt, "自检截面对冲",
                                    win._period(), [])
            started = win._c_thread is not None
            t0 = time.time()
            while time.time() - t0 < 60 and win._c_thread is not None:
                app.processEvents()
                time.sleep(0.02)
            cout = win.res
            cross_async_ok = started and win._c_thread is None and cout is not None
            app.processEvents()
            pv = win.view_port
            tabs = win.tabs_port
            # 「截面因子分析」和「因子分析」一样是**露面才算**：切到那一页才
            # 算 IC、才画图，省得每次回测都白烧一秒
            for i in range(tabs.count()):
                if tabs.tabText(i) == "截面因子分析":
                    tabs.setCurrentIndex(i)
                    break
            app.processEvents()
            visible = [tabs.tabText(i) for i in range(tabs.count())
                       if tabs.isTabVisible(i)]
            cross_ok = (
                cross_async_ok
                and "截面因子分析" in visible
                and pv.tbl_cfactor.rowCount() >= 2
                and len(pv.canvas_cf.fig.axes) == 4
                and pv.tbl_trades.rowCount() > 0
                and pv.tbl_trades.horizontalHeaderItem(1).text() == "品种"
            )
            cross_info = (f"后台线程 {cross_async_ok} / "
                          f"交易 {pv.tbl_trades.rowCount()} 笔 / "
                          f"因子 {pv.tbl_cfactor.rowCount()} 行 / "
                          f"页签 {'、'.join(visible)}")
        except Exception as exc:  # noqa: BLE001
            cross_info = f"{type(exc).__name__}: {exc}"

        # 回测区间：切片要真的生效。不生效的话 5 分钟十年照样从头算，
        # 「近 1 年」只是个显示开关，用户以为快了其实没快
        #
        # **先钉回「单品种 + 规则策略」**：`df_raw` 只有单品种那条路才有，截面和
        # 多品种都是「一批品种 → 一条组合净值」，`_finish_cross` 会把它清成 None。
        # 用户上次停在截面策略的话这里会必红 —— 可那不是代码坏了，是自检假设了
        # 初始状态（和上面 `is_us` 一个道理）。
        range_ok = False
        range_info = ""
        try:
            win.family = FAMILY_RULE
            if hasattr(win, "sym_pick"):
                win.sym_pick.clear()          # 多品种勾选也清掉，否则走多品种那条路
            win._sync_family_panels()
            app.processEvents()
            win.cb_range.setCurrentIndex(0)      # 近 1 年
            win.run_analysis()
            n_1y = len(win.df_raw)
            win.cb_range.setCurrentIndex(5)      # 全部历史
            win.run_analysis()
            n_all = len(win.df_raw)
            range_ok = 0 < n_1y < n_all and hasattr(win, "btn_load_lib")
            range_info = f"近 1 年 {n_1y} 根 / 全部 {n_all} 根"
        except Exception as exc:  # noqa: BLE001
            range_info = f"{type(exc).__name__}: {exc}"

        # 2 日 / 周线：下拉里要有，选了要真的换成更粗的 K 线。
        # 只看「有没有这个选项」不够 —— 选项在、`DataStore` 没接上，界面上
        # 会安静地继续画日线，用户以为自己在看周线
        md_ok = False
        md_info = ""
        try:
            def _pick(period: str) -> int:
                i = win.cb_period.findData(period)
                if i < 0:
                    raise RuntimeError(f"周期下拉里没有 {period}")
                win.cb_period.blockSignals(True)
                win.cb_period.setCurrentIndex(i)
                win.cb_period.blockSignals(False)
                win.cb_range.setCurrentIndex(5)      # 全部历史
                win.run_analysis()
                return len(win.df_raw)

            n_1d = _pick("1d")
            n_2d = _pick("2d")
            n_1w = _pick("1w")
            if n_1d <= 0 or n_2d != (n_1d + 1) // 2 or not (n_1d / 5.6 < n_1w < n_1d / 4.2):
                raise RuntimeError(
                    f"合成周期根数不对：日线 {n_1d} / 2 日 {n_2d} / 周线 {n_1w}")
            md_ok = True
            md_info = f"日线 {n_1d} 根 → 2 日 {n_2d} 根、周线 {n_1w} 根"
        except Exception as exc:  # noqa: BLE001
            md_info = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                win.cb_period.blockSignals(True)
                win.cb_period.setCurrentIndex(win.cb_period.findData("1d_tq"))
                win.cb_period.blockSignals(False)
                win.cb_range.setCurrentIndex(0)
                win.run_analysis()
            except Exception:  # noqa: BLE001
                pass

        win.close()
        app.processEvents()

        if n == 0:
            raise RuntimeError("主窗口没有加载到任何标的，数据目录可能没打进包")
        if not has_res:
            raise RuntimeError("主窗口未完成首次回测")
        if not has_upd:
            raise RuntimeError("数据管理页缺失或初始状态异常")
        if not trade_ok:
            raise RuntimeError(
                f"账户透视页异常（{trade_info}）：账户列表 / 资金账户列 / 时间列缺失"
            )
        if not agent_ok:
            raise RuntimeError(
                "ChanAgent 页异常：删除按钮初始状态不对，或升级权限的文件模块 / "
                "平台手册取不到 / 回测结果没推到输出框"
                + (f"（{fs_err}）" if fs_err else "")
                + (f"（手册：{manual_info}）" if not manual_ok else "")
                + (f"（输出：{out_info}）" if not out_ok else "")
            )
        if not wheel_ok:
            raise RuntimeError(
                f"K 线滚轮缩放失效（窗口 {wheel_info}）：scroll_event 回调没生效，"
                "多半是 MplCanvas.fresh() 换 Figure 之后没重挂事件连接"
            )
        if not k_ok:
            raise RuntimeError(
                f"K 线窗口状态异常：win={rv._k_win} start={rv._k_start} total={rv._k_total}"
            )
        if not live_btn_ok:
            raise RuntimeError(
                "「启动实时」按钮异常：不在 K 线与信号页上，或换品种后没复位"
            )
        if not combo_ok:
            raise RuntimeError(f"组合报告异常：{combo_info}")
        if not cross_ok:
            raise RuntimeError(
                f"截面结果视图异常（{cross_info}）：交易记录为空，或者没有长出"
                "「截面因子分析」页"
            )
        if not range_ok:
            raise RuntimeError(
                f"回测区间没生效（{range_info}）：cb_range 只是改了显示窗口，"
                "策略仍然在全历史上算"
            )
        if not md_ok:
            raise RuntimeError(f"2 日 / 周线在界面上没接上（{md_info}）")
        if not factor_ok:
            raise RuntimeError(f"因子面板控件缺失：{factor_info}"
                               + (f"（周期列：{period_info}）" if not period_ok else ""))
        if not live_off:
            raise RuntimeError(
                "启动时自动接了实时（live 已在跑）：研究页不该连账户，"
                "账户连接只在「账户透视和交易」页"
            )
        # 因子分析页只列**当前市场能用**的因子（A 股专属的盈利收益率 / 换手率 /
        # 概念板块在期货标的上永远算不出来，列着就是一片「无数据」）。所以行数
        # 应该正好等于该市场的因子数，也就是单因子下拉的条目数 —— 两边都走
        # `factors.filter_by_market()`，对不上说明有一边漏了过滤。
        if n_rows != n_factors:
            raise RuntimeError(
                f"因子分析页有 {n_rows} 行，当前市场应有 {n_factors} 个因子")
        if n_axes != 4:
            raise RuntimeError(f"因子分析页图轴数为 {n_axes}，应为 4")
        if not verify_ok:
            raise RuntimeError(
                f"策略验证页异常（{verify_info}）：取不到当前策略、前视检测跑不出"
                "检查点，或者检查点表没填上"
            )

        # 截面策略的两步配置：参数面板要能借进弹窗、切走能还回侧边栏且顺序不乱。
        # 这一步放在最后 —— 它会切策略族，`_schedule` 会顺手排一次回测，前面那些
        # 断言都已经取完数了，不会被它带偏
        cross_dlg_ok = False
        cross_dlg_info = ""
        # ---- 配置往返：多品种勾选必须存得下来、也恢复得回去
        #
        # 以前只存 `symbol` / `exchange`（当前在看的那个），规则 / 因子族的多品种
        # 清单存盘后直接丢了 —— 「存一条跑 78 个品种的策略」再打开只剩 1 个品种。
        # 走真正的 `_current_config()` / `apply_config()`，丢字段的 bug 恰恰出在
        # 这两个函数的衔接上。放在最后：`apply_config` 会重排一次回测
        multi_rt_ok = False
        multi_rt_info = ""
        try:
            pool = [it["key"] for it in win.store.items][:3]
            win.sym_pick.set_selected(pool)
            app.processEvents()
            before = list(win.sym_pick.selected())
            cfg_rt = win._current_config()
            win.sym_pick.clear()
            app.processEvents()
            cleared = list(win.sym_pick.selected())
            win.apply_config(cfg_rt)
            app.processEvents()
            after_sel = list(win.sym_pick.selected())
            # 标题：用户起过名就用名字，没起名才拼「N 个品种 · 策略」。
            # 把名字清掉再验，否则验的是用户上次起的名字
            t = cfg_rt.clone(name="") if hasattr(cfg_rt, "clone") else cfg_rt
            title_txt = t.title()
            multi_rt_ok = (len(before) >= 2 and not cleared
                           and after_sel == before
                           and list(cfg_rt.multi_symbols) == before
                           and title_txt.startswith(f"{len(before)} 个品种"))
            multi_rt_info = (f"存 {len(before)} 个 / 清空 {len(cleared)} 个 / "
                             f"恢复 {len(after_sel)} 个 / 标题 {title_txt[:16]}")
            win.sym_pick.clear()
            app.processEvents()
        except Exception as exc:  # noqa: BLE001
            multi_rt_info = f"{type(exc).__name__}: {exc}"
        if not multi_rt_ok:
            raise RuntimeError(
                f"多品种配置往返异常（{multi_rt_info}）：存盘只留了单品种，"
                "或者勾选恢复不回去"
            )

        # ---- 组合页的「设置参数」也要能改品种 ----
        #
        # 研究页能勾一批品种，进了组合就再也改不了 —— 存盘里 78 个品种，列表上
        # 显示的却是第一个。这条走真正的 `EditDialog` + 页面的品种清单：
        # 勾选要回填得回来、改完要写进 `multi_symbols`、列表要写「N 个品种」
        pdlg_ok = False
        pdlg_info = ""
        try:
            from ttrade import strategy_store as _St
            from ttrade.strategies import STRATEGIES as _S

            from gui.portfolio_page import EditDialog as _ED
            from gui.portfolio_page import _sym_text as _stext

            pool = [it["key"] for it in win.store.items][:3]
            if len(pool) < 2:
                raise RuntimeError("本地库品种不足 2 个")
            k0 = pool[0]
            if win.market == "cn":
                sym0, _, ex0 = k0.rpartition("_")
            elif win.market == "cn_stock":
                sym0, ex0 = k0, k0.split(".", 1)[0]
            else:
                sym0, ex0 = k0, ""
            pc = _St.StrategyConfig(
                symbol=sym0, exchange=ex0, period=win._period(),
                rule_key="ma_cross", rule_params=dict(_S["ma_cross"].params),
                multi_symbols=list(pool),
            )
            pdlg = _ED(pc, [], universe=win.portfolio_page._universe_of(pc))
            filled = sorted(pdlg.pick.selected())
            pdlg.pick.set_selected(pool[:2])
            pdlg.apply_to(pc)
            pdlg_ok = (filled == sorted(pool)
                       and sorted(pc.multi_symbols) == sorted(pool[:2])
                       and _stext(pc).startswith("2 个品种"))
            pdlg_info = (f"回填 {len(filled)} 个 / 存 {len(pc.multi_symbols)} 个 / "
                         f"列表写「{_stext(pc)}」")
            pdlg.deleteLater()
        except Exception as exc:  # noqa: BLE001
            pdlg_info = f"{type(exc).__name__}: {exc}"
        if not pdlg_ok:
            raise RuntimeError(f"组合页设置框改品种异常（{pdlg_info}）")
        try:
            side_lay = win._panel_slots[win.box_factor][0]

            def _order(lay):
                return [lay.itemAt(i).widget() for i in range(lay.count())
                        if lay.itemAt(i).widget() is not None]

            before = _order(side_lay)
            win.cb_family.setCurrentIndex(win.cb_family.findData("cross"))
            app.processEvents()
            # 上次存的配置可能停在选品种模式，先钉回对冲 —— 下面那几条断言是
            # 按「对冲 → 选品种」的顺序写的
            win.cb_cross_mode.setCurrentIndex(win.cb_cross_mode.findData("hedge"))
            app.processEvents()
            # 调仓间隔的单位跟着模式走：对冲按根，选品种按天 / 月
            reb_unit = win.cb_cross_reb_unit
            hedge_units = [reb_unit.itemData(i) for i in range(reb_unit.count())]
            hedge_ok = (
                win.box_fsel.parentWidget() is win.cross_factor_host
                and win.box_factor.parentWidget() is win.cross_factor_host
                and win.grp_timing.isHidden()
                # 多空腿数只在第二步、只在截面对冲模式下出现
                and not win.grp_legs.isHidden()
                and win.grp_legs.parentWidget() is win.cross_step2
                and win.cross_dlg.step1_lay.count() >= 2
                and win.cross_dlg.step2_lay.count() >= 2
                and bool(win.lbl_cross_sum.text().strip())
                and "bar" in hedge_units
            )
            win.cb_cross_mode.setCurrentIndex(win.cb_cross_mode.findData("select"))
            app.processEvents()
            win.cb_timing.setCurrentIndex(win.cb_timing.findData("rule"))
            app.processEvents()
            select_units = [reb_unit.itemData(i) for i in range(reb_unit.count())]
            # 第一步的因子周期：默认「跟随策略周期」，其余周期可选项里必须有日线
            # （5 分钟策略按日线成交额筛池是主场景）。和策略周期相同的那一项要灰掉
            # —— 选了等于没选，不灰就是骗人
            p1cb = win.cb_step1_period
            p1_items = [(p1cb.itemData(i), p1cb.model().item(i).isEnabled()
                         if p1cb.model() and p1cb.model().item(i) is not None else True)
                        for i in range(p1cb.count())]
            p1_data = [d for d, _ in p1_items]
            cur_p = win._period()
            p1_ok = (p1_data and p1_data[0] in ("", None)
                     and "1d_tq" in p1_data
                     and not dict(p1_items).get(cur_p, False))
            win.cb_step1_period.setCurrentIndex(p1cb.findData("1d_tq"))
            app.processEvents()
            p1_sum = win._cross_summary()
            p1_pick = (win._step1_period() == "1d_tq" if cur_p != "1d_tq" else True)
            p1_note = ("第一步因子：日线" in p1_sum) if cur_p != "1d_tq" else True
            win.cb_step1_period.setCurrentIndex(0)
            app.processEvents()
            p1_back = win._step1_period() == "" and "第一步因子" not in win._cross_summary()
            # 名义敞口：截面要跟「仓位」那一组控件走，和多品种回测同一个口径。
            # 以前截面恒按 1 倍算，仓位面板在截面模式下等于摆设
            lev = win._cross_leverage()
            lev_ok = (lev is not None and abs(float(lev) - 1.0) > 0.02
                      and "名义敞口" in win._cross_summary()
                      ) if (win.cb_size.currentData() == "capital") else (
                lev is not None and abs(float(lev) - 1.0) <= 0.02)
            select_ok = (
                not win.grp_timing.isHidden()
                and win.box_rule.parentWidget() is win.cross_timing_host
                # 选品种模式不需要腿数：方向由时序信号逐品种给
                and win.grp_legs.isHidden()
                # 选品种不能按「根」调仓（5 分钟上 5 根=25 分钟换一次），
                # 只给交易日 / 个月
                and "bar" not in select_units
                and "day" in select_units and "month" in select_units
                and "交易日" in win.lbl_cross_sum.text()
                and p1_ok and p1_pick and p1_note and p1_back and lev_ok
            )
            win.cb_family.setCurrentIndex(win.cb_family.findData("rule"))
            app.processEvents()
            win.cross_dlg.close()
            back_ok = _order(side_lay) == before
            cross_dlg_ok = hedge_ok and select_ok and back_ok
            cross_dlg_info = (f"对冲 {hedge_ok} / 选品种 {select_ok} / "
                              f"还回侧边栏 {back_ok} / 第一步周期 "
                              f"{p1_ok},{p1_pick},{p1_note},{p1_back} / 杠杆 {lev_ok}")
        except Exception as exc:  # noqa: BLE001
            cross_dlg_info = f"{type(exc).__name__}: {exc}"
        if not cross_dlg_ok:
            raise RuntimeError(
                f"截面两步配置窗异常（{cross_dlg_info}）：参数面板借进弹窗、切走还回"
                "侧边栏，或者侧边栏顺序被搬乱了，或者「第一步因子周期」选不动"
            )

        # ---- 组合页「查看回测」必须在后台线程跑 ----
        #
        # 多品种一次是 N 遍策略计算，机器学习还要各训几十折（19 品种 / 5m 十年
        # 实测十几分钟）。同步跑的话整个窗口「未响应」，连「停止」都点不动。
        # 这里验的是「点下去立刻返回」：`_start_view` 之后线程已经在跑，主线程
        # 还在，等它自己算完再出图。
        view_ok, view_info = False, ""
        try:
            from ttrade import strategy_store as _St2
            from ttrade.strategies import STRATEGIES as _S2

            pp = win.portfolio_page
            key0 = win.store.items[0]["key"]
            if win.market == "cn":
                _sym, _, _ex = key0.rpartition("_")
            elif win.market == "cn_stock":
                _sym, _ex = key0, key0.split(".", 1)[0]
            else:
                _sym, _ex = key0, ""
            # 规则策略最便宜：这条验的是「跑在哪个线程」，不是「算得对不对」
            vcfg = _St2.StrategyConfig(
                symbol=_sym, exchange=_ex, period="1d", family="rule",
                rule_key="ma_cross", rule_params=dict(_S2["ma_cross"].params),
                sizing="capital", principal=1_000_000.0, capital_pct=30.0,
            )
            pp._start_view([vcfg], "single")
            threaded = pp._view_thread is not None
            deadline = time.time() + 60
            while time.time() < deadline and pp._view_thread is not None:
                app.processEvents()
                time.sleep(0.02)
            done = pp._view_thread is None
            on_view = pp.stack.currentIndex() == 1

            # 实时 K 线的「底料」必须是**行情本身**，不能是 `_prepare` 那个
            # `(行情, 跨周期行情)` 两元组。落到 `_live_base` 里还留着元组的话，
            # `merge_live` 的 `local.empty` 会抛
            # `AttributeError: 'tuple' object has no attribute 'empty'`，
            # 界面顶部红字「实时 K 线刷新失败」，5 秒一刷，图永远不往前动。
            # 拆包只有 `run_replay_job` 做了，`_present_single` 曾经漏了两处。
            base = pp._live_base
            live_ok = isinstance(base, pd.DataFrame)
            if live_ok:
                from ttrade.tqlive import merge_live as _merge_live
                _merge_live(base, base.tail(1).copy())   # 拼得上才算通
            view_ok = threaded and done and on_view and live_ok
            view_info = (f"起线程={threaded} 算完={done} 切到结果页={on_view} "
                         f"实时底料={type(base).__name__}")
            pp.shutdown()
        except Exception as exc:  # noqa: BLE001
            view_info = f"{type(exc).__name__}: {exc}"
        if not view_ok:
            raise RuntimeError(
                f"组合页「查看回测」没跑在后台（{view_info}）：点下去应该立刻返回、"
                "线程在后台算，算完自己切到结果页；实时底料还必须是 DataFrame"
            )

        # ---- 侧边栏宽度：控件不该被挤到右边看不见
        #
        # 横向滚动条现在是「按需」的，但那只是兜底 —— 正常状态下不该出现。这一块
        # 出过两次事：机器学习「稀疏采样」里两个因子下拉并排，最小宽度各 290 px，
        # 把整张卡片撑到 640 px，右边直接被裁掉；用户看到的是「很多框跑到右边
        # 看不见了」。
        #
        # 量的是**面板自己**的最小宽度，不读 `content.minimumSizeHint()`：布局的
        # 最小尺寸是逐层缓存的，研究页不在前台时，控件显隐传不回父布局，读出来会
        # 永远停在上一个状态 —— 那就正好把要抓的 bug 漏掉
        from PySide6.QtWidgets import QScrollArea

        scroll = win.sidebar.findChild(QScrollArea, "SideScroll")
        side_info = "侧边栏不可测"
        if scroll is not None:
            # 面板外面还有三层留白：内容右 6 + 卡片左右 12×2 + 子卡片左右 10×2
            PAD = 50
            panels = [(name, getattr(win, name, None)) for name in
                      ("box_sparse", "box_fsel", "box_rule", "box_filter", "box_factor")]
            panels = [(n, w) for n, w in panels if w is not None]

            # 量的时候别让「改参数自动回测」插进来跑一整套（ML 要几十折）
            was_auto = win.chk_auto.isChecked()
            win.chk_auto.blockSignals(True)
            win.chk_auto.setChecked(False)
            worst: list[tuple[str, int]] = []
            try:
                for fam, modes in (("rule", ("",)), ("factor", ("single", "multi", "ml")),
                                   ("cross", ("",))):
                    win.cb_family.setCurrentIndex(win.cb_family.findData(fam))
                    app.processEvents()
                    for mode in modes:
                        if mode:
                            win.cb_fmode.setCurrentIndex(win.cb_fmode.findData(mode))
                        # 下拉没变的时候不触发 `currentIndexChanged`，面板显隐不会
                        # 跟着刷新（上一轮可能刚被截面模式藏起来）。直接同步一次
                        win._sync_factor_panel()
                        app.processEvents()
                        if mode == "ml":
                            # 先关再开：配置恢复时是 blockSignals 后 setChecked 的，
                            # 直接 setChecked(True) 可能不触发 toggled，参数区不展开
                            # 就等于没量到这一块（而它正是撑宽侧边栏的元凶）
                            win.chk_sparse.setChecked(False)
                            win.chk_sparse.setChecked(True)
                            app.processEvents()
                        for name, w in panels:
                            if w.isHidden():
                                continue
                            worst.append((f"{fam}/{mode or '-'}·{name}",
                                          int(w.minimumSizeHint().width())))
                        if mode == "ml":
                            win.chk_sparse.setChecked(False)
                win.cb_family.setCurrentIndex(win.cb_family.findData("rule"))
                app.processEvents()
            finally:
                win.chk_auto.setChecked(was_auto)
                win.chk_auto.blockSignals(False)
            vp = scroll.viewport().width()
            if not worst:
                raise RuntimeError("侧边栏面板一个都没量到")
            top = max(worst, key=lambda x: x[1])
            side_info = f"侧边栏 {vp}px / 最宽面板 {top[1]}px（{top[0]}）"
            over = [f"{tag} {w}px" for tag, w in worst if w > vp - PAD]
            if over:
                raise RuntimeError(f"侧边栏放不下（可视 {vp}px，留白 {PAD}px）："
                                   + "、".join(over))

        # ---- 触发面板：两个「不用」选项 + 两个反转开关
        #
        # 「不用触发因子」必须在下拉自己的列表里：切市场会 clear() 重填，早先
        # cb_trig2 那一项是在外面 insertItem 补的，切一次市场就没了 —— 用户会
        # 莫名其妙发现「不用第二个因子」这个选项消失
        trig_ok, trig_info = True, ""
        try:
            win.cb_family.setCurrentIndex(win.cb_family.findData("factor"))
            win.cb_fmode.setCurrentIndex(win.cb_fmode.findData("ml"))
            win._sync_factor_panel()
            win.chk_sparse.setChecked(True)
            app.processEvents()
            if win.cb_trig1.itemData(0) != "" or win.cb_trig2.itemData(0) != "":
                raise RuntimeError("两个触发因子下拉的第一项不是「不用」")
            # 两个都选「不用」= 只用规则：因子反转该灰、规则反转该亮
            win.cb_trig1.setCurrentIndex(0)
            win.cb_trig2.setCurrentIndex(0)
            # 挑一条**有参数**的规则：没有参数的规则，「参数」按钮本来就该是灰的
            _ri = win.cb_trig_rule.findData("ma_cross")
            win.cb_trig_rule.setCurrentIndex(_ri if _ri > 0 else 1)
            app.processEvents()
            if win.chk_trig_flip.isEnabled():
                raise RuntimeError("没有触发因子时「反转因子」还能勾")
            if not win.chk_trig_rule_flip.isEnabled():
                raise RuntimeError("选了触发规则，「反转规则」却是灰的")
            if win._trigger_keys():
                raise RuntimeError("两个「不用」仍然算出了触发因子")
            # 规则触发参数：按钮跟着有没有规则灰 / 亮，改过的参数要进 FactorConfig
            rule_key = win.cb_trig_rule.currentData()
            if not win.btn_trig_params.isEnabled():
                raise RuntimeError("选了触发规则，「参数」按钮却是灰的")
            win.trig_rule_params[rule_key] = {"fast": 5, "slow": 20}
            win._sync_trigger_src()
            got = win._factor_cfg().trigger_params
            if got != {"fast": 5, "slow": 20}:
                raise RuntimeError(f"规则触发参数没进配置：{got}")
            if "快线周期 5" not in win.btn_trig_params.toolTip():
                raise RuntimeError("按钮提示里看不到改过的参数")
            # 换一条规则：参数不能串到别一条上
            other_rule = next((win.cb_trig_rule.itemData(i)
                               for i in range(1, win.cb_trig_rule.count())
                               if win.cb_trig_rule.itemData(i) != rule_key), "")
            if other_rule:
                win.cb_trig_rule.setCurrentIndex(
                    win.cb_trig_rule.findData(other_rule))
                app.processEvents()
                if win._factor_cfg().trigger_params:
                    raise RuntimeError("换了触发规则，上一条的参数跟过来了")
                win.cb_trig_rule.setCurrentIndex(win.cb_trig_rule.findData(rule_key))
                app.processEvents()
            # 参数窗本身：值能取回、恢复默认能回默认
            from gui.info_dialogs import RuleParamsDialog

            dlg = RuleParamsDialog("测试规则", [
                ("fast", "快线周期", 2, 200, 1, 10, 25),
                ("vol_ratio", "波幅比例", 0.1, 3.0, 0.1, 0.4, 1.5),
            ], win)
            vals = dlg.values()
            if vals != {"fast": 25, "vol_ratio": 1.5}:
                raise RuntimeError(f"参数窗取值不对：{vals}")
            dlg._reset()
            if dlg.values() != {"fast": 10, "vol_ratio": 0.4}:
                raise RuntimeError(f"恢复默认没生效：{dlg.values()}")
            dlg.deleteLater()
            win.trig_rule_params.pop(rule_key, None)
            # 切市场再切回来：「不用」不能丢
            mk = win.cb_market.currentData()
            others = [win.cb_market.itemData(i) for i in range(win.cb_market.count())]
            others = [m for m in others if m and m != mk]
            if others:
                win.cb_market.setCurrentIndex(win.cb_market.findData(others[0]))
                app.processEvents()
                if win.cb_trig1.itemData(0) != "" or win.cb_trig2.itemData(0) != "":
                    raise RuntimeError("切市场后「不用」这个选项丢了")
                win.cb_market.setCurrentIndex(win.cb_market.findData(mk))
                app.processEvents()

            # ---- 稀疏采样要能存进策略库、也要能从库里读回来 ----
            #
            # `_factor_cfg()` 一直把 `sample_mode` / `trigger_*` 写进配置，但
            # `_apply_factor_cfg()` 一个都没恢复：存进库再载入，勾是空的、档位
            # 回到默认，用户看到的是「策略库里没有这一项」。这里走真正的
            # `_current_config()` / `apply_config()`，先把界面拨到「全不一样」
            # 再载回来，逐个字段比对。
            win.chk_sparse.setChecked(True)
            app.processEvents()
            fkey = win.cb_trig1.itemData(1) or ""
            win.cb_trig1.setCurrentIndex(1 if fkey else 0)
            win.cb_trig_side.setCurrentIndex(win.cb_trig_side.findData("up"))
            win.sp_trig_z.setValue(1.8)
            win.sp_trig_tiers.setValue(4)
            win.sp_trig_span.setValue(0.8)
            win.sp_trig_cool.setValue(23)
            win.chk_trig_flip.setChecked(True)
            app.processEvents()
            want = win._factor_cfg()
            cfg_sp = win._current_config()
            if str((cfg_sp.factor_cfg or {}).get("sample_mode") or "") != "trigger":
                raise RuntimeError("稀疏采样没写进 StrategyConfig.factor_cfg")
            # 拨到「全不一样」：勾去掉、档位清空、方向回默认、反转取消
            win.sp_trig_z.setValue(1.0)
            win.sp_trig_tiers.setValue(0)
            win.sp_trig_span.setValue(0.3)
            win.sp_trig_cool.setValue(0)
            win.cb_trig_side.setCurrentIndex(0)
            win.chk_trig_flip.setChecked(False)
            win.cb_trig1.setCurrentIndex(0)
            win.chk_sparse.setChecked(False)
            app.processEvents()
            win.apply_config(cfg_sp)
            app.processEvents()
            got = win._factor_cfg()
            if not (win.chk_sparse.isChecked() and not win.sparse_body.isHidden()
                    and got.sample_mode == "trigger"
                    and list(got.trigger_keys) == list(want.trigger_keys)
                    and got.trigger_rule == want.trigger_rule
                    and got.trigger_side == "up"
                    and abs(got.trigger_z - 1.8) < 1e-9
                    and got.trigger_tiers == 4
                    and abs(got.trigger_span - 0.8) < 1e-9
                    and got.trigger_cool == 23
                    and got.trigger_flip):
                raise RuntimeError(
                    "稀疏采样没能从策略库里载回来（勾="
                    f"{win.chk_sparse.isChecked()} / {got.sample_mode} / "
                    f"档 {got.trigger_tiers}·{got.trigger_span}σ·冷却 "
                    f"{got.trigger_cool} / 阈值 {got.trigger_z} / "
                    f"方向 {got.trigger_side} / 触发 {got.trigger_keys} / "
                    f"反转 {got.trigger_flip}）")
            win.chk_sparse.setChecked(False)

            # ---- 多因子组合的权重口径 + 出场规则 ----
            #
            # 这两项都是「界面能设、配置能存、后端真的用上」三件事，任何一环断了
            # 都只表现成「结果和上次一样」。所以既走一遍配置往返，也在合成数据上
            # 验一下规则真的改变了平仓时点。
            import numpy as np
            import pandas as pd
            from PySide6.QtCore import Qt

            win.cb_family.setCurrentIndex(win.cb_family.findData("factor"))
            win.cb_fmode.setCurrentIndex(win.cb_fmode.findData("multi"))
            app.processEvents()
            if win.cb_fweight.isHidden():
                raise RuntimeError("多因子组合下没露出权重口径下拉")
            win.cb_fweight.setCurrentIndex(win.cb_fweight.findData("custom"))
            app.processEvents()
            if win.tbl_fmulti.isColumnHidden(4):
                raise RuntimeError("选了「不等权」但权重列还是藏着的")
            # 勾上前两行：默认勾选（`DEFAULT_MULTI`）不一定落在这个市场的因子集里，
            # 上一段稀疏采样的操作也可能把勾去掉
            for _r in range(min(2, win.tbl_fmulti.rowCount())):
                _cell = win.tbl_fmulti.item(_r, 2)
                if _cell is not None:
                    _cell.setCheckState(Qt.Checked)
            app.processEvents()
            rows = [r for r in range(win.tbl_fmulti.rowCount())
                    if win.tbl_fmulti.item(r, 2) is not None
                    and win.tbl_fmulti.item(r, 2).checkState() == Qt.Checked]
            if len(rows) < 2:
                raise RuntimeError("自检环境里勾中的因子不足 2 个，权重往返测不了")
            win.tbl_fmulti.item(rows[0], 4).setText("3")
            win.tbl_fmulti.item(rows[1], 4).setText("0.5")
            app.processEvents()
            want_w = win._factor_cfg()
            cfg_w = win._current_config()
            if not (want_w.weight_mode == "custom"
                    and abs(float(want_w.weights.get(
                        win.tbl_fmulti.item(rows[0], 0).data(Qt.UserRole), 0)) - 3.0) < 1e-9):
                raise RuntimeError(f"权重没进配置：{want_w.weight_mode} {want_w.weights}")
            win.cb_fweight.setCurrentIndex(win.cb_fweight.findData("equal"))
            win.tbl_fmulti.item(rows[0], 4).setText("1")
            win.tbl_fmulti.item(rows[1], 4).setText("1")
            app.processEvents()
            win.apply_config(cfg_w)
            app.processEvents()
            got_w = win._factor_cfg()
            if not (win.cb_fweight.currentData() == "custom"
                    and abs(float(got_w.weights.get(
                        win.tbl_fmulti.item(rows[0], 0).data(Qt.UserRole), 0)) - 3.0) < 1e-9):
                raise RuntimeError(
                    f"权重口径没从策略库载回来（{win.cb_fweight.currentData()} / "
                    f"{got_w.weights}）")

            # 权重按比例起作用：3 : 0.5 和 6 : 1 是同一件事
            from ttrade import factor_strategy as _FS

            _idx = pd.date_range("2015-01-01", periods=420, freq="B")
            _rng = np.random.default_rng(7)
            _px = 100 * np.exp(np.cumsum(_rng.normal(0, 0.012, len(_idx))))
            _syn = pd.DataFrame(
                {"open": _px, "high": _px * 1.008, "low": _px * 0.992,
                 "close": _px, "volume": 1e5}, index=_idx)
            _syn.attrs["symbol"] = "AG"
            _syn.attrs["exchange"] = "SHFE"
            _keys = ["mom_20", "mom_60", "ma_gap_20"]
            _a = _FS.build_score(_syn, "multi", _keys, _FS.FactorConfig(
                weight_mode="custom", weights={_keys[0]: 3.0, _keys[1]: 0.5,
                                               _keys[2]: 1.0}))
            _b = _FS.build_score(_syn, "multi", _keys, _FS.FactorConfig(
                weight_mode="custom", weights={_keys[0]: 6.0, _keys[1]: 1.0,
                                               _keys[2]: 2.0}))
            _c = _FS.build_score(_syn, "multi", _keys, _FS.FactorConfig())
            _wa = _a.score.dropna()
            _diff_ratio = float((_a.score - _b.score).abs().max(skipna=True))
            if not (_diff_ratio < 1e-9 and len(_wa) > 50):
                raise RuntimeError(f"权重按比例缩放后分数变了（最大差 {_diff_ratio:.2e}）")
            if float((_a.score - _c.score).abs().max(skipna=True)) < 1e-6:
                raise RuntimeError("自定权重和等权算出来一模一样，权重没生效")
            _ic = _FS.build_score(_syn, "multi", _keys,
                                  _FS.FactorConfig(weight_mode="ic", ic_rebal=40))
            if float((_ic.score - _c.score).abs().max(skipna=True)) < 1e-6:
                raise RuntimeError("滚动 IC 加权和等权算出来一模一样，权重没生效")

            # ---- 因子与 / 因子或：必须和「逐因子比阈值再全部 / 任一」逐根一致 ----
            #
            # 这两档不走加权平均，合成分数只是一个能骗过阈值状态机的等价量
            # （与 = 取最弱那个，或 = 取最响那个）。既然是个等价量，就必须拿
            # 原始因子分数按同一套阈值自己判一遍来对，否则「取最弱」写错成
            # 「取最强」在图上完全看不出来，只是信号变多变少。
            _e = 1.0
            for _wm in ("and", "or"):
                _p = _FS.build_score(_syn, "multi", _keys,
                                     _FS.FactorConfig(weight_mode=_wm,
                                                      entry_z=_e, exit_z=0.3))
                _m = _p.fmat * _p.orient        # 逐因子 PIT 取号后的分数
                if _wm == "and":
                    # 「与」没有歧义：每个因子都得过线。缺数当「没确认」
                    _want = {1.0: (_m > _e).all(axis=1),
                             -1.0: (_m < -_e).all(axis=1)}
                else:
                    # 「或」必须在**方向冲突**上做规定：一根里 A 说多、B 说空，
                    # 「任一满足」两边都成立，而仓位不可能同时多空。口径定的是
                    # 「谁 |z| 大听谁的」，所以基准得挑最响的因子看它站哪边，
                    # 写成简单的 `any` 会在冲突行上判成两个方向同时成立
                    _arr = _m.to_numpy(dtype=float)
                    _ab = np.where(np.isnan(_arr), -np.inf, np.abs(_arr))
                    _ref = _arr[np.arange(len(_arr)), _ab.argmax(axis=1)]
                    _ref = pd.Series(
                        np.where(np.isnan(_arr).all(axis=1), np.nan, _ref),
                        index=_m.index)
                    _want = {1.0: (_ref > _e), -1.0: (_ref < -_e)}
                for _side in (1.0, -1.0):
                    _have = ((_p.score > _e) if _side > 0
                             else (_p.score < -_e)).fillna(False)
                    _w = _want[_side].fillna(False)
                    _bad = int((_w.to_numpy() != _have.to_numpy()).sum())
                    if _bad:
                        raise RuntimeError(
                            f"「{_wm}」合成口径与逐因子判定不一致：{_bad} 根对不上"
                            f"（方向 {'多' if _side > 0 else '空'}）")
                if _wm == "and":
                    # 与的信号必须不严于等权：最靠近零的那个因子的幅度，不会
                    # 超过这几个因子的平均幅度（正负混杂时它直接记 0）
                    _z = pd.concat([_p.score.abs().rename("a"),
                                    _c.score.abs().rename("b")], axis=1).dropna()
                    if len(_z) and bool((_z.a > _z.b + 1e-9).any()):
                        raise RuntimeError("「因子与」的分数幅度跑到了等权之上，取错极端了")
                if _wm == "or":
                    _z = pd.concat([_p.score.abs().rename("a"),
                                    _c.score.abs().rename("b")], axis=1).dropna()
                    if len(_z) and bool((_z.a < _z.b - 1e-9).any()):
                        raise RuntimeError("「因子或」的分数幅度落到了等权之下，取错极端了")

            # ---- 标准化口径：z-score / 分位 ----
            #
            # 分位这一档以前映射到 ±0.5，而阈值滑杆最小 0.3、最大 3.0 —— 阈值一
            # 调到 0.5 就再也开不出仓，界面上看起来是「这个策略从来不交易」。
            # 现在两档都落在 ±4，所以既验范围，也验配置能存能载。
            from ttrade import factors as _F

            _raw = pd.Series(
                np.r_[np.linspace(100.0, 120.0, 300), 400.0,
                      np.linspace(120.0, 118.0, 100)],
                index=pd.date_range("2018-01-01", periods=401, freq="B"))
            _rng_rank = _F.standardize(_raw, window=252, method="rank").dropna()
            _rng_z = _F.standardize(_raw, window=252, method="z").dropna()
            if _rng_rank.empty or _rng_z.empty:
                raise RuntimeError("两种口径都算不出分数，窗口给错了")
            if float(_rng_rank.abs().max()) > 4.0 + 1e-9:
                raise RuntimeError(
                    f"分位口径的分数越过了 ±4：{float(_rng_rank.abs().max()):.3f}")
            if float(_rng_z.abs().max()) > 4.0 + 1e-9:
                raise RuntimeError(
                    f"z 口径的分数越过了 ±4：{float(_rng_z.abs().max()):.3f}")
            # 分位对极端值免疫：那根 400 的毛刺在 z 下被截到 4（说明它确实是个
            # 巨幅偏离），分位这边它只能顶到 +4，且不因毛刺变大而变
            _spike = _rng_rank.idxmax()
            if abs(float(_rng_rank.loc[_spike]) - 4.0) > 1e-9:
                raise RuntimeError(f"分位口径下最高分的毛刺不是 +4："
                                   f"{float(_rng_rank.loc[_spike]):.3f}")
            _raw2 = _raw.copy()
            _raw2.loc[_spike] = 4000.0
            _rng_rank2 = _F.standardize(_raw2, window=252, method="rank")
            if abs(float(_rng_rank2.loc[_spike]) - 4.0) > 1e-9:
                raise RuntimeError("毛刺放大十倍后分位分数变了，分位没免疫极端值")

            # 配置往返：分位存进策略库再载回来
            win.cb_fmethod.setCurrentIndex(win.cb_fmethod.findData("rank"))
            app.processEvents()
            cfg_m = win._current_config()
            if win._factor_cfg().method != "rank":
                raise RuntimeError("分位口径没写进 factor_cfg")
            win.cb_fmethod.setCurrentIndex(win.cb_fmethod.findData("z"))
            app.processEvents()
            win.apply_config(cfg_m)
            app.processEvents()
            if not (win.cb_fmethod.currentData() == "rank"
                    and win._factor_cfg().method == "rank"):
                raise RuntimeError(
                    f"标准化口径没从策略库载回来（{win.cb_fmethod.currentData()} / "
                    f"{win._factor_cfg().method}）")
            win.cb_fmethod.setCurrentIndex(win.cb_fmethod.findData("z"))
            app.processEvents()

            # 出场规则：分档（浮盈越多回撤越少）必须比绝对回撤**更早**平仓
            _n = 20 + 10 + 12
            _close = np.r_[np.full(20, 100.0), 100 + np.arange(1, 11),
                           110 - np.arange(1, 13) * 1.0]
            _ex = pd.DataFrame({"open": _close, "high": _close + 0.5,
                                "low": _close - 0.5, "close": _close},
                               index=pd.date_range("2020-01-01", periods=_n, freq="D"))
            _sig = np.ones(_n, dtype=int)
            _tier, _ = _FS.apply_exit_rules(
                _sig, _ex, _FS.FactorConfig(exit_mode="tier"))
            _abs, _ = _FS.apply_exit_rules(
                _sig, _ex, _FS.FactorConfig(exit_mode="abs", exit_abs_atr=2.0))
            _box, _ = _FS.apply_exit_rules(
                _sig, _ex, _FS.FactorConfig(exit_mode="box", exit_box_bars=10))
            _score, _ = _FS.apply_exit_rules(
                _sig, _ex, _FS.FactorConfig(exit_mode="score"))
            _k_tier = int(np.argmax(_tier == 0)) if (_tier == 0).any() else -1
            _k_abs = int(np.argmax(_abs == 0)) if (_abs == 0).any() else -1
            if not (_k_tier > 0 and _k_abs > _k_tier):
                raise RuntimeError(
                    f"分档回撤没比绝对回撤更早平仓（分档 {_k_tier} / 绝对 {_k_abs}）")
            if not (_box == 0).any():
                raise RuntimeError("反向箱体突破在单边上涨后回落时没平仓")
            if not np.array_equal(_score, _sig):
                raise RuntimeError("「分数出场」不该改动信号")

            # ---- 独立止损 ----
            #
            # 三件事必须同时成立，缺一条就会表现成「勾了止损但没反应」：
            # 1. 口径还是「分数出场」时它也生效（这是它和四种口径最大的区别）；
            # 2. 它和别的口径**叠加**，不是替换 —— 分档回撤开着也照样止损；
            # 3. 止损线钉在进场价上，不跟着浮盈抬（否则它就是「绝对回撤」了）。
            _n2 = 30
            _dn = np.r_[np.full(10, 100.0), 100 - np.arange(1, 11) * 1.0,
                        np.full(10, 89.0)]
            _df2 = pd.DataFrame({"open": _dn, "high": _dn + 0.4,
                                 "low": _dn - 0.4, "close": _dn},
                                index=pd.date_range("2021-01-01", periods=_n2, freq="D"))
            _sig2 = np.ones(_n2, dtype=int)
            _sl, _ = _FS.apply_exit_rules(
                _sig2, _df2,
                _FS.FactorConfig(exit_mode="score", exit_sl=True, exit_sl_atr=2.0))
            _k_sl = int(np.argmax(_sl == 0)) if (_sl == 0).any() else -1
            if _k_sl <= 10:
                raise RuntimeError(
                    f"「分数出场」下止损没生效（第一根平仓在 {_k_sl}，应该 10 根之后）")
            # 止损关掉时同一条路径必须原样返回
            _off, _ = _FS.apply_exit_rules(
                _sig2, _df2, _FS.FactorConfig(exit_mode="score", exit_sl=False))
            if not np.array_equal(_off, _sig2):
                raise RuntimeError("没勾止损却改动了信号")
            # 叠加：分档回撤 + 宽止损 ≈ 只有分档；分档 + 紧止损要更早平
            _tier_only, _ = _FS.apply_exit_rules(
                _sig2, _df2, _FS.FactorConfig(exit_mode="tier"))
            _tier_wide, _ = _FS.apply_exit_rules(
                _sig2, _df2,
                _FS.FactorConfig(exit_mode="tier", exit_sl=True, exit_sl_atr=50.0))
            _tier_tight, _ = _FS.apply_exit_rules(
                _sig2, _df2,
                _FS.FactorConfig(exit_mode="tier", exit_sl=True, exit_sl_atr=0.5))
            if not np.array_equal(_tier_only, _tier_wide):
                raise RuntimeError("宽止损（50ATR）不该改变分档回撤的结果")
            _k_t1 = int(np.argmax(_tier_only == 0)) if (_tier_only == 0).any() else 999
            _k_t2 = int(np.argmax(_tier_tight == 0)) if (_tier_tight == 0).any() else 999
            if not _k_t2 <= _k_t1:
                raise RuntimeError(
                    f"叠加止损之后没有更早平仓（分档 {_k_t1} / 分档+紧止损 {_k_t2}）")
            # 止损线不跟着浮盈抬：先涨到 +5ATR 再回到进场价，止损设在 2ATR 时
            # **不该**在回到进场价时平仓（那是绝对回撤干的事）
            _n3 = 30
            _up = np.r_[np.full(5, 100.0), 100 + np.arange(1, 16) * 1.0,
                        115 - np.arange(1, 11) * 1.0]
            _df3 = pd.DataFrame({"open": _up, "high": _up + 0.4,
                                 "low": _up - 0.4, "close": _up},
                                index=pd.date_range("2022-01-01", periods=_n3, freq="D"))
            _sl3, _ = _FS.apply_exit_rules(
                np.ones(_n3, dtype=int), _df3,
                _FS.FactorConfig(exit_mode="score", exit_sl=True, exit_sl_atr=2.0))
            _abs3, _ = _FS.apply_exit_rules(
                np.ones(_n3, dtype=int), _df3,
                _FS.FactorConfig(exit_mode="abs", exit_abs_atr=2.0))
            if not ((_abs3 == 0).any() and not (_sl3 == 0).any()):
                raise RuntimeError(
                    "止损和绝对回撤被做成了同一件事：涨上去再回落时，"
                    f"绝对回撤该平（{'平' if (_abs3 == 0).any() else '没平'}）、"
                    f"止损不该平（{'平了' if (_sl3 == 0).any() else '没平'}）")

            # 出场规则的配置往返
            win.exit_cfg = {"exit_mode": "tier",
                            "exit_tiers": [[1.5, 2.5], [3.0, 1.0], [5.0, 0.5]],
                            "exit_abs_atr": 2.0, "exit_box_bars": 20,
                            "exit_box_buf": 0.0, "exit_atr_n": 14,
                            "exit_sl": True, "exit_sl_atr": 1.5}
            win._sync_exit_summary()
            cfg_x = win._current_config()
            win.exit_cfg = {}
            win._sync_exit_summary()
            win.apply_config(cfg_x)
            app.processEvents()
            got_x = win._factor_cfg()
            if not (got_x.exit_mode == "tier"
                    and [[round(float(a), 4), round(float(b), 4)]
                         for a, b in got_x.exit_tiers]
                    == [[1.5, 2.5], [3.0, 1.0], [5.0, 0.5]]):
                raise RuntimeError(f"出场规则没从策略库载回来：{got_x.exit_mode} "
                                   f"{got_x.exit_tiers}")
            # 止损单独往返一次：只勾止损、口径仍是「分数出场」时，老写法会把
            # 整份 exit_cfg 清空（判据只看 exit_mode），止损就悄悄丢了
            win.exit_cfg = {"exit_mode": "score", "exit_sl": True,
                            "exit_sl_atr": 1.5}
            win._sync_exit_summary()
            cfg_y = win._current_config()
            win.exit_cfg = {}
            win._sync_exit_summary()
            win.apply_config(cfg_y)
            app.processEvents()
            got_y = win._factor_cfg()
            if not (got_y.exit_sl and abs(float(got_y.exit_sl_atr) - 1.5) < 1e-9
                    and got_y.exit_mode == "score"):
                raise RuntimeError(
                    f"只勾止损时配置没往返：exit_sl={got_y.exit_sl} "
                    f"{got_y.exit_sl_atr} / mode={got_y.exit_mode}")
            # 「禁止分数出场」单独往返一次：同一个坑 —— `_apply_factor_cfg`
            # 的清空判据只看 exit_mode 的话，勾了它载回来就没了
            win.exit_cfg = {"exit_mode": "score", "exit_sl": True,
                            "exit_sl_atr": 2.0, "exit_no_score": True}
            win._sync_exit_summary()
            cfg_n = win._current_config()
            win.exit_cfg = {}
            win._sync_exit_summary()
            win.apply_config(cfg_n)
            app.processEvents()
            got_n = win._factor_cfg()
            if not got_n.exit_no_score:
                raise RuntimeError("「禁止分数出场」没往返（载入后被清空了）")
            win.exit_cfg = {}
            win._sync_exit_summary()

            # ---- 规则 + 因子过滤：存进策略库再载回来，过滤条件必须还在 ----
            #
            # 这一段曾经整条丢掉：`_current_config` 只在「因子策略 / 截面」时才
            # 把 `factor_mode` / `factor_keys` 写进配置，于是「规则 + 因子过滤」
            # 存进去再载回来就变成一条**没有过滤**的规则策略 —— 回测笔数直接变
            # 一个人，界面上还什么都看不出来（勾选框是空的，看着像「库里没存」）。
            from ttrade import strategy_store as _SStore

            win.cb_family.setCurrentIndex(win.cb_family.findData("rule"))
            app.processEvents()
            win.chk_filter.setChecked(True)
            app.processEvents()
            win.cb_filter_mode.setCurrentIndex(win.cb_filter_mode.findData("multi"))
            win.cb_filter_when.setCurrentIndex(win.cb_filter_when.findData("hold"))
            app.processEvents()
            _cf = win._current_config()
            if not (_cf.factor_mode == "multi" and _cf.factor_keys):
                raise RuntimeError(
                    f"因子过滤没进配置：mode={_cf.factor_mode!r} "
                    f"keys={len(_cf.factor_keys or [])}")
            if str((_cf.factor_cfg or {}).get("filter_mode") or "") != "hold":
                raise RuntimeError("过滤时机（转差就离场）没进 factor_cfg")
            if "因子过滤" not in _cf.strategy_label():
                raise RuntimeError(f"策略库列表看不出这条带过滤：{_cf.strategy_label()}")
            if "过滤" not in _cf.params_text():
                raise RuntimeError(f"参数摘要里没有过滤条件：{_cf.params_text()}")
            _n_keys = len(_cf.factor_keys)
            # 清空界面，再从「库里读回来」的那份配置灌回去
            win.chk_filter.setChecked(False)
            app.processEvents()
            win.apply_config(_SStore.StrategyConfig.from_dict(_cf.to_dict()))
            app.processEvents()
            if not (win.chk_filter.isChecked()
                    and win.cb_filter_mode.currentData() == "multi"
                    and win.cb_filter_when.currentData() == "hold"
                    and win._filter_active()):
                raise RuntimeError(
                    f"过滤条件没载回来：勾={win.chk_filter.isChecked()} "
                    f"mode={win.cb_filter_mode.currentData()} "
                    f"when={win.cb_filter_when.currentData()}")
            _cf2 = win._current_config()
            if (_cf2.factor_mode, len(_cf2.factor_keys or [])) != ("multi", _n_keys):
                raise RuntimeError(
                    f"过滤条件载回来之后再存一次就变了：{_cf2.factor_mode} "
                    f"{len(_cf2.factor_keys or [])} / 原来 {_n_keys}")
            win.chk_filter.setChecked(False)
            app.processEvents()

            # 弹窗本身：三档乱序要排好、回撤幅度写反要理顺，不然「浮盈越多
            # 回撤越小」这句话在界面上是假的，而规则会照错的跑
            from gui.exit_dialog import ExitRuleDialog, summarize

            _d = ExitRuleDialog({"exit_mode": "tier",
                                 "exit_tiers": [[6.0, 0.8], [2.0, 2.0], [4.0, 1.2]]})
            _v = _d.values()
            if _v["exit_tiers"] != [[2.0, 2.0], [4.0, 1.2], [6.0, 0.8]]:
                raise RuntimeError(f"分档参数没排好：{_v['exit_tiers']}")
            _d2 = ExitRuleDialog({"exit_mode": "tier",
                                  "exit_tiers": [[2.0, 0.5], [4.0, 3.0], [6.0, 2.0]]})
            _v2 = _d2.values()
            _tr = [b for _, b in _v2["exit_tiers"]]
            if _tr != sorted(_tr, reverse=True):
                raise RuntimeError(f"回撤幅度没有随浮盈递减：{_v2['exit_tiers']}")
            if "分数出场" not in summarize({}):
                raise RuntimeError("默认摘要没写「分数出场」")
            _d.deleteLater()
            _d2.deleteLater()

            # 载入一份 ML 配置会顺手起后台线程（因子分析页重算 IC）。自检到这里
            # 就结束了，线程还在跑的话窗口析构时 Qt 报
            # `QThread: Destroyed while thread is still running`，进程以非零码
            # 退出 —— CI 上表现成「30 条检查全 OK，job 却红」，特别费解。
            for _cancel in (win._cancel_factor_worker, win._cancel_multi_worker,
                            win._cancel_cross_worker):
                try:
                    _cancel()
                except Exception:      # noqa: BLE001 - 收尾失败不该影响断言
                    pass
            trig_info = ("「不用」两项都在（切市场后仍在），反转开关跟着触发源灰 / 亮，"
                         "规则触发参数能进配置且换规则不串，"
                         "稀疏采样存得进策略库也载得回来，"
                         "五档合成口径（等权/自定/IC/因子与/因子或）都真起作用且能往返，"
                         "「与」「或」与逐因子判定逐根一致，"
                         "分位口径也映射到 ±4、对毛刺免疫且能往返，"
                         "分档回撤比绝对回撤更早平仓、独立止损与各口径叠加且不跟浮盈抬、"
                         "只勾止损也能往返、出场规则能往返、"
                         "规则+因子过滤能往返")
        except Exception as exc:      # noqa: BLE001
            trig_ok, trig_info = False, f"{type(exc).__name__}: {exc}"
        if not trig_ok:
            raise RuntimeError(f"触发面板异常：{trig_info}")

        return (f"{n} 个标的，K 线/绩效/因子页已渲染，{k_info}，"
                f"{factor_info}，滚轮缩放 {wheel_info}，策略验证 {verify_info}，"
                f"波幅突破 {brk_info}，"
                f"组合 {combo_info}，"
                f"截面 {cross_info}，两步配置 {cross_dlg_info}，"
                f"多品种往返 {multi_rt_info}，"
                f"组合页回测 {view_info}，"
                f"{side_info}，触发 {trig_info}，"
                f"回测区间 {range_info}，合成周期 {md_info}，实时开关已复位，"
                f"数据管理页就绪，手册 {manual_info}，输出 {out_info}，"
                f"对话渲染 {chat_info}，能力包 {skill_info}")

    check("构建主窗口（端到端）", _window)

    # K 线与信号页的持仓映射：这一项 import 了 Qt，放在窗口之后跑
    check("K线与信号：持仓与交易记录同源（含手数）", _chart_from_trades)

    return ok, lines


def run() -> int:
    """执行自检并返回退出码。"""
    from ttrade.paths import log_dir

    try:
        ok, lines = _run()
    except Exception:  # noqa: BLE001 - 连路径都取不到时的兜底
        ok = False
        lines = ["自检初始化失败：", traceback.format_exc()]

    text = "\n".join(lines)
    try:
        path = os.path.join(log_dir(), "selftest.log")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        text += f"\n\n结果已写入 {path}"
    except Exception:
        pass

    # 打包成窗口程序时 sys.stdout 是 None（print 会静默丢弃），CI/命令行下才看得到。
    # 英文版 Windows 控制台默认 cp1252/cp936，直接 print 中文会乱码或抛
    # UnicodeEncodeError，所以先把流切成 utf-8。
    if sys.stdout is not None:
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            print(text)
        except Exception:
            pass
    return 0 if ok else 1
