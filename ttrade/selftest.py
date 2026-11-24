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
    # 当前正在跑哪一项。崩溃（段错误）时进程直接死，`lines` 来不及落盘，
    # faulthandler 也只能给出已经损坏的 C 层栈 —— 2026-11-16 的 macOS 构建
    # 就只落了 "selftest.py line 7841 in run" 两帧，等于没说是哪一项死的。
    # 这里每进一项就把名字刷到文件里（不缓冲），崩了至少知道死在谁身上。
    _prog = None
    # 日志**边跑边落盘**，理由同上但更狠：原来 `run()` 只在最后把整个 `lines`
    # 写一次文件，段错误一来进程直接死 ⇒ `selftest.log` **不存在**。CI 那几轮
    # 只能拿到「日志文件不存在（自检没跑完就没机会落盘）」，一条线索都没有
    # （v3.9.6 与 v3.9.15 的 macOS arm64 都是这么白跑的）。
    _log_fh = None

    def _emit(text: str) -> None:
        """记一行，并立刻刷进 `selftest.log`。

        `flush()` 就够：进程被信号打死时数据已经在**内核**缓冲里，只有断电才
        需要 `fsync`，而 135 项每项 fsync 一次在 Windows 上要多花好几秒。
        """
        lines.append(text)
        if _log_fh is not None:
            try:
                _log_fh.write(text + "\n")
                _log_fh.flush()
            except Exception:  # noqa: BLE001 - 写不进就少一层线索，不影响自检
                pass

    # 全程离屏，自检不应该在用户屏幕上闪一个窗口
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    def check(name: str, fn) -> None:
        nonlocal ok
        if _prog is not None:
            try:
                _prog.write(name + "\n")
                _prog.flush()
            except Exception:  # noqa: BLE001 - 写不进就少一层线索，不影响自检
                pass
        try:
            detail = fn()
            _emit(f"[ OK ] {name}" + (f"  {detail}" if detail else ""))
        except Exception as exc:  # noqa: BLE001 - 自检要吞掉异常继续跑完
            ok = False
            _emit(f"[FAIL] {name}  {type(exc).__name__}: {exc}")
            _emit(traceback.format_exc())

    from ttrade.paths import bundle_root, cache_dir, data_dir, is_frozen, log_dir, user_dir

    try:
        _prog = open(os.path.join(log_dir(), "selftest_progress.txt"), "w",
                     encoding="utf-8")
    except Exception:  # noqa: BLE001
        _prog = None
    try:
        # buffering=1 = 行缓冲，配合 `_emit` 里的显式 flush；段错误时文件已经有
        # 到死点为止的全文，CI 那边 `cat` 出来就能看到是**哪一项**挂的
        _log_fh = open(os.path.join(log_dir(), "selftest.log"), "w",
                       encoding="utf-8", buffering=1)
    except Exception:  # noqa: BLE001
        _log_fh = None

    _emit(
        f"frozen={is_frozen()}  python={sys.version.split()[0]}  platform={sys.platform}")
    _emit(f"bundle_root = {bundle_root()}")
    _emit(f"data(us)    = {data_dir('us')}")
    _emit(f"data(cn)    = {data_dir('cn')}")
    _emit(f"user_dir    = {user_dir()}")
    _emit(f"log_dir     = {log_dir()}")
    _emit("")

    def _store_isolated():
        """自检绝不能读写用户的组合 / 策略库。

        这一条是补一次真实事故：自检会搭真窗口，里面有切页签、加策略这些动作，
        它们在生产里就该落盘 —— 结果跑一轮自检就把用户的 `portfolio.json` 删掉
        一条、二十几条本金重排了。`run()` 现在把整个存储目录重定向到临时目录，
        这里钉死它确实生效。
        """
        from ttrade import strategy_store as S2

        cur = S2.store_dir()
        if os.path.abspath(cur) == os.path.abspath(user_dir()):
            raise RuntimeError(
                f"存储目录还是用户目录（{cur}）—— 自检会改坏用户数据")
        if not os.path.isdir(cur):
            raise RuntimeError(f"重定向到的目录不存在：{cur}")
        # 写一次再读回来，确认落的是临时目录
        probe = "selftest_probe.json"
        S2._save(probe, {"x": 1})                      # noqa: SLF001
        hit = os.path.join(cur, probe)
        ok = os.path.exists(hit)
        try:
            os.remove(hit)
        except OSError:
            pass
        if not ok:
            raise RuntimeError(f"写盘没落在重定向目录里：{hit}")
        return f"存储目录 → {cur}（不是用户目录）"

    check("自检不碰用户数据（存储目录已重定向）", _store_isolated)

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
        # 「只做空」默认是**不该勾**的（勾错了等于半个策略不干活），但它也可以是
        # 策略定义本身：`sbbox多品种波段1h 基本面空` 就是照着「只做空头」调的，
        # 收编时把它掰回双向，载入后跑出来的曲线和用户库里那条对不上。
        # 所以白名单放行，条件是说**说明里必须写明「只做空」** —— 否则以后照着
        # 这条改品种的人不会知道自己在改一条单边策略。
        only_short_ok = {"tpl_lib_rule_26"}
        for k in libs:
            c = T.to_config(k)
            if c is None:
                raise RuntimeError(f"{k} 还原不出来")
            if getattr(c, "short_only", False):
                if k not in only_short_ok:
                    raise RuntimeError(f"{k} 默认勾了「禁止做多」，应该默认不勾")
                if "只做空" not in T.teach(k):
                    raise RuntimeError(f"{k} 是只做空策略，说明里却没写「只做空」")
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

    def _ws_bar_two_rows():
        """组合页的工作区工具条必须是**两行**：工作区一行、资金分配一行。

        挤一行不会报错，只会悄悄缩水：实测总宽 1714px 撞上 1058px 的卡片，
        Qt 就按比例压每一个控件 —— 页签从 237px 压到 123px（冒出左右滚动箭头，
        第二个工作区看不见），「＋」从 42px 压到 30px（文字被内边距吃光，
        渲染成一个空框）。所以用**结构**钉住，而不是凭眼睛看截图。
        """
        from PySide6.QtWidgets import QHBoxLayout

        from gui.workspace_bar import WorkspaceBar

        bar = WorkspaceBar()
        bar.set_workspaces(["默认组合", "短线策略"], 0)
        root = bar.layout()
        if root is None or root.count() != 2:
            raise RuntimeError(f"工具条不是两行：layout 有 {root.count() if root else 0} 项")
        rows = []
        for i in range(2):
            it = root.itemAt(i)
            lay = it.layout() if it is not None else None
            if not isinstance(lay, QHBoxLayout):
                raise RuntimeError(f"第 {i + 1} 行不是横排布局")
            rows.append(lay)
        if rows[0].indexOf(bar.tabs) < 0:
            raise RuntimeError("工作区页签不在第一行")
        for w, name in ((bar.sp_total, "总资金"), (bar.cb_mode, "分配方式"),
                        (bar.cb_basis, "校准"), (bar.cb_knob, "调整"),
                        (bar.btn_measure, "量一下占用"), (bar.btn_preview, "分配预览")):
            if rows[1].indexOf(w) < 0:
                raise RuntimeError(f"「{name}」没有落在第二行")
            if rows[0].indexOf(w) >= 0:
                raise RuntimeError(f"「{name}」还挤在工作区那一行")
        # 页签宽度要够：被压窄时 QTabBar 会冒滚动箭头，这一条是那个症状的判据
        hint = bar.tabs.sizeHint().width()
        if hint < 150:
            raise RuntimeError(f"页签 sizeHint 只有 {hint}px，工作区名一长就看不见")
        # 「新建」不能让固定宽度把文字吃掉（原来 `setFixedWidth(30)`，sizeHint 却是
        # 42 —— 全局按钮内边距 2×14px，内容只剩 2px，渲染出来是个空框）
        from PySide6.QtWidgets import QPushButton

        add = next((b for b in bar.findChildren(QPushButton)
                    if b.text() == "新建"), None)
        if add is None:
            raise RuntimeError("找不到「新建」按钮")
        if add.maximumWidth() < add.sizeHint().width():
            raise RuntimeError(
                f"「新建」被固定成 {add.maximumWidth()}px，比文字宽度 "
                f"{add.sizeHint().width()}px 还窄（会渲染成空框）")
        # 🔴 长提示不许把这一行「想要的宽度」顶大：提示标签自己按 stretch 吃剩余
        # 空间、用 elidedText 截断；一旦它的 sizeHint 按全文算，第一行需求就会
        # 超过卡片宽，Qt 又按比例压窄所有控件，页签立刻冒滚动箭头（实测 237→123）。
        tab_before = bar.tabs.sizeHint().width()
        bar.hint("7 条 · 本金合计 143.4 万 · = 总资金的 1.04 倍 · ⚠ 没量占用，"
                 "校准会退回按分配额")
        if bar.lbl_hint.sizeHint().width() > 80:
            raise RuntimeError(
                f"提示标签的 sizeHint 是 {bar.lbl_hint.sizeHint().width()}px，"
                "按全文算宽了 —— 会把第一行的页签压窄")
        if bar.tabs.sizeHint().width() != tab_before:
            raise RuntimeError("加了长提示之后页签宽度变了，说明提示在抢这一行的宽度")
        return (f"两行：工作区（页签 sizeHint {hint}px + 新建/改名/删除工作区）｜"
                "资金分配（总资金/分配方式/校准/调整/量一下占用/分配预览）")

    def _perf_yscale():
        """资金曲线纵轴的「线性 / 对数」开关（绩效页页首那个下拉框）。

        默认必须是**线性**：等权组合的净值就在 1 附近走，对数轴把 0.9~1.1
        这一段压成平线、把后面的涨幅拉开，看着像「前半段根本没波动」。对数
        留给要读等比例涨跌的曲线（复权净值、跨数量级）。

        四件事钉住：① 默认线性、ylabel 不写「对数」；② 切过去真是 log 轴；
        ③ 净值里有 ≤ 0 的点（亏穿）时对数画不出来 —— matplotlib 会把那一段
        直接丢掉、看着像「亏光那天就停了」，所以必须退回线性、把选项置灰、
        并在图上写明原因；④ 用户的选择要跟着重画和换腿走，不能自己掉回线性。
        """
        import numpy as np
        import pandas as pd

        from ttrade.backtest import _METRIC_KEYS, Result

        from gui.result_view import ResultView, ViewCtx

        def mk(eq, label="X", legs=None):
            m = {k: 0.0 for k in _METRIC_KEYS}
            m.update({"initial_cash": 100_000.0, "final_equity": 120_000.0,
                      "cagr": 0.2, "max_dd": -0.1, "sharpe": 1.0,
                      "n_trades": 5})
            return Result(equity=eq, position=pd.Series(0.0, index=eq.index),
                          trades=[], metrics=m, label=label, legs=legs or {})

        idx = pd.date_range("2022-01-03", periods=300, freq="B")
        good = pd.Series(np.linspace(1.0, 1.8, len(idx)) + 0.02, index=idx)
        bad = pd.Series(np.linspace(1.0, -0.4, len(idx)), index=idx)

        def ax_of(v):
            return v.canvas_p.figure.axes[0]

        v = ResultView(kind="portfolio")
        v.show_result(None, None, mk(good), None, ViewCtx(label="组合"))
        cb, ax = v.cb_yscale, ax_of(v)
        if cb is None:
            raise RuntimeError("绩效页没有纵轴下拉框（单策略也必须能切）")
        if cb.currentData() != 0 or ax.get_yscale() != "linear":
            raise RuntimeError(
                f"默认不是线性：选中 {cb.currentData()}、轴 {ax.get_yscale()}")
        if "对数" in ax.get_ylabel():
            raise RuntimeError(f"线性轴上还写着对数：{ax.get_ylabel()!r}")
        if not cb.model().item(1).isEnabled():
            raise RuntimeError("全正净值时「对数」却被置灰了")

        cb.setCurrentIndex(1)
        if ax_of(v).get_yscale() != "log" or "对数" not in ax_of(v).get_ylabel():
            raise RuntimeError("切到对数没生效")
        v._draw_perf()                     # 重画一次，状态不能丢
        if ax_of(v).get_yscale() != "log" or v.cb_yscale.currentData() != 1:
            raise RuntimeError("重画之后纵轴自己掉回线性了")

        # 亏穿：对数画不出来，必须退回线性并说明
        v.show_result(None, None, mk(bad), None, ViewCtx(label="亏穿"))
        ax = ax_of(v)
        if cb.model().item(1).isEnabled():
            raise RuntimeError("净值有负值时「对数」还能选（点了会以为没生效）")
        if ax.get_yscale() != "linear":
            raise RuntimeError(f"亏穿时还用了对数轴：{ax.get_yscale()}")
        if not any("对数轴画不出来" in t.get_text() for t in ax.texts):
            raise RuntimeError("亏穿时退回线性却没写原因，用户只会以为点了没反应")

        # 换腿：全组合正、某条腿亏穿 —— 各判各的，用户选的「对数」别被偷偷改掉
        v2 = ResultView(kind="portfolio")
        v2.show_result(None, None, mk(good, legs={"亏穿腿": mk(bad)}), None,
                       ViewCtx(label="组合"))
        v2.cb_yscale.setCurrentIndex(1)
        keys = [v2.cb_leg.itemData(i) for i in range(v2.cb_leg.count())]
        v2.cb_leg.setCurrentIndex(keys.index("亏穿腿"))
        if ax_of(v2).get_yscale() != "linear":
            raise RuntimeError("切到亏穿那条腿没退回线性")
        if v2.cb_yscale.currentData() != 1:
            raise RuntimeError("换腿把用户选的「对数」改成了线性")
        v2.cb_leg.setCurrentIndex(0)
        if ax_of(v2).get_yscale() != "log":
            raise RuntimeError("切回全组合没恢复对数")

        return ("默认线性 / 切对数真变 log / 亏穿退回线性并置灰＋说明 / "
                "重画与换腿都保住选择")

    # 注意：`check()` 是立即执行的，这一项 import 了 Qt，
    # 要等下面的 `_window` 把 QApplication 建起来之前再验（见文件末尾的 check）

    def _intrabar():
        """K 线内进出场（`run_intrabar_backtest`）。

        这条分支容易出的错都是「不报错、只是数字悄悄变好看」那种，所以必须
        逐条验：入口成交在**触发价**（不是下一根开盘）、止损成交在
        **止损价**、拿掉那几列之后老路径**一个数都不许变**、**进场当根不出场**
        （逐笔 bars>0，且只有诊断开关能造出 bars==0）、显式打开
        `intrabar_samebar_stop` 才在当根扫、**「只做多 / 禁止做多」两个方向
        开关都真的起作用**（`short_only` 曾经漏在这条分支里，勾了照样开多）。
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

        # ---- 方向开关：`long_only` 和 `short_only` 两条都必须真起作用 ----
        #
        # 用户 2026-11-23：「箱体波幅突破.k线内策略，选择禁止做多，依然还开出了
        # 多单，说明这个禁止做多选项对这个策略没有起到作用」。`long_only` 一直
        # 有（抹 `entry_short` 那一行），`short_only` **漏了** —— 老路径靠裁
        # signal 实现，K 线内没有 signal，于是这个开关对 `sbbox_ib` / `im_orb`
        # 这一族全是摆设，勾了照样开多。
        from dataclasses import replace as _replace

        r3 = run_backtest(d, _replace(cfg, short_only=True), label="K线内")
        if len(r3.trades):
            raise RuntimeError(
                f"勾了「禁止做多」还开出 {len(r3.trades)} 笔 "
                f"（方向 {[t.direction for t in r3.trades]}）—— "
                "做多触发价没被抹掉")
        # 对照：同一份数据不勾就该开得出来，否则上面那条断言是假绿
        if not len(run_backtest(d, cfg, label="K线内").trades):
            raise RuntimeError("对照组一笔都没开，这条断言测不到东西")

        # 反方向同理：`long_only` 必须把 `entry_short` 抹掉
        d_s = d.copy()
        d_s["entry_long"] = np.nan
        d_s.iloc[6, d_s.columns.get_loc("entry_short")] = 95.0
        d_s.iloc[6, d_s.columns.get_loc("low")] = 94.0
        r4 = run_backtest(d_s, _replace(cfg, long_only=True), label="K线内")
        if len(r4.trades):
            raise RuntimeError(
                f"勾了「只做多」还开出 {len(r4.trades)} 笔空单 —— "
                "做空触发价没被抹掉")
        if not len(run_backtest(d_s, cfg, label="K线内").trades):
            raise RuntimeError("只做空那份样本本身不触发，测不到东西")

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

    def _ib_theo():
        """K 线内策略的「理论持仓」= signal × lots，**不是** target。

        用户 2026-11-23：「波段策略工作区加载的 sbbox 策略，显示理论持仓是 0，
        但是查看回测里头，交易记录有挺多品种有持仓」。

        真因是 `_refresh_signal` 里 K 线内那一支写了 `st.signal = 0`：signal 一清，
        「理论」列和两个盈亏列（都按 signal × lots / target 算）对 K 线内策略
        永远是 +0。实测那条 `sbbox多品种波段1h` 的 34 个品种里，
        `run_strategy("sbbox_ib")` 最后一根上有 **29 个非零**，正好等于回测里
        「持仓中 29 笔」—— 状态机从来没说空仓，是引擎把信号擦了。

        反方向的坑也要钉住：**理论仓 ≠ 下单目标**。拿 signal 直接下单等于用一根
        已经收盘的 bar 去追一个过去的价位，所以改手数那处（`_update_one`）也不许
        按 signal 反推 K 线内策略的 target。
        """
        import inspect

        from ttrade import strategy_runner as SR
        from ttrade import strategy_store as SStore

        class _Fake:
            """只借三个方法：`_ib_on` / `_theo_qty` / `_update_one`。"""

            _ib_on = staticmethod(SR.PortfolioRunner._ib_on)
            _theo_lots_qty = SR.PortfolioRunner._theo_lots_qty
            _theo_qty = SR.PortfolioRunner._theo_qty
            _update_one = SR.PortfolioRunner._update_one
            _slots: dict = {}

        def mk(sid: str, rule_key: str, **kw):
            return SStore.StrategyConfig(
                sid=sid, name=sid, symbol="RB", exchange="SHFE", period="1h",
                family="rule", rule_key=rule_key, **kw)

        f = _Fake()

        # ① K 线内：理论仓 = signal × lots；target 仍是 0（盘中还没碰到触发价）
        st = SR.SlotState(cfg=mk("t-ib", "sbbox_ib", sizing="fixed", volume=1))
        st.running, st.signal, st.lots, st.target = True, -1, 32, 0
        got = f._theo_qty(st)
        if got != -32:
            raise RuntimeError(
                f"K 线内策略的理论仓该是 signal×lots = -32，实际 {got} —— "
                "「理论持仓」列又会显示成 0（2026-11-23 用户报的 sbbox）")

        # ② 普通策略：理论仓就是 target
        st2 = SR.SlotState(cfg=mk("t-plain", "ma_cross", sizing="fixed",
                                  volume=1))
        st2.running, st2.signal, st2.lots, st2.target = True, -1, 1, -1
        if f._theo_qty(st2) != -1:
            raise RuntimeError("普通策略的理论仓该直接读 target")

        # ③ 改手数：K 线内的 target 不许被 signal 反推（那会凭空发一笔单）。
        # 用 `sizing="fixed"` 才真会走到「手数立刻变」那条分支 —— 默认的
        # capital 模式手数要等下一根 bar 按价格重算，改 volume 本来就不动。
        st.target = 0
        f._slots = {"t-ib": st}
        f._update_one(mk("t-ib", "sbbox_ib", sizing="fixed", volume=3))
        if int(st.lots) != 3:
            raise RuntimeError(
                f"固定手数模式下改 volume 没立刻生效（lots={st.lots}）—— "
                "下面的断言就测不到东西了")
        if int(st.target) != 0:
            raise RuntimeError(
                f"改手数把 K 线内策略的 target 设成了 {st.target} —— "
                "等于拿一根已收盘的 bar 去追过去的价位，会凭空发一笔单")
        # 普通策略照旧要跟着手数走，别把这条也一起关了
        st2.target = -1
        f._slots = {"t-plain": st2}
        f._update_one(mk("t-plain", "ma_cross", sizing="fixed", volume=3))
        if int(st2.target) != -3:
            raise RuntimeError(
                f"普通策略改手数之后目标仓该变成 -3，实际 {st2.target}")

        # ④ 快照必须推 `theo`，界面「理论」列读它
        src = inspect.getsource(SR.PortfolioRunner.snapshot)
        if '"theo": self._theo_qty(st)' not in src:
            raise RuntimeError(
                "快照里没推 `theo` —— 界面只能退回 target，K 线内策略又显示成 0")
        import gui.portfolio_page as PP

        page = inspect.getsource(PP)
        if 'slot.get("theo"' not in page:
            raise RuntimeError(
                "界面「理论」列还在读 target —— K 线内策略会显示 +0，"
                "而回测里同一根 bar 上是持仓的")
        if "_theo_qty" not in inspect.getsource(SR.PortfolioRunner._slot_pnl):
            raise RuntimeError(
                "盈亏还在按 target 算 —— K 线内策略的当日/持仓盈亏会恒等于 +0")

        return ("K 线内策略的理论仓按「方向 × 开仓手数」算（不是 target，"
                "回测同一个数）；改手数不会反推它的下单目标；"
                "快照推 `theo`、界面与两个盈亏列都读它")

    check("K 线内策略的理论持仓（sbbox 显示 0 那个）", _ib_theo)

    def _theo_entry_lots():
        """理论手数取「**开仓那一根**算出来的那个数」，绝不按当前 K 线重算。

        用户 2026-11-23：「交易区理论列，需要按照开仓的手数来算，不能根据当前
        k 线来算，实际上应该就从开仓那边获取，而不是自己擅自计算，这样是很严重
        的 bug」。

        真 bug：`_theo_qty` 以前用 `st.lots`，而 `st.lots` **每根 bar 都按当根
        ATR 重算**。risk 口径下 `手数 = 止损额 ÷ (ATR × 倍数 × 乘数)`，ATR 从
        12 掉到 4.7，同一笔仓的理论手数就从 39 手漂到 106 手 —— 界面和回测交易
        记录里那一笔永远对不上。回测是「开仓那一根算一次、固定到平仓」，实盘
        必须同一个口径。

        这里用真数据（RB 日线）跑一遍：`_note_theo_entry` 倒推到信号波段起点，
        算出来的手数要**逐位等于回测同一笔的 `Trade.lots`**；再往后接 6 根波幅
        翻倍的 K 线，手数必须一动不动。
        """
        import inspect

        import pandas as pd

        from ttrade import backtest as BT
        from ttrade import strategy_runner as SR
        from ttrade import strategy_store as SStore
        from ttrade import tqdata
        from ttrade.strategies import run_strategy

        class _Fake:
            # 只借「算手数 / 记理论值」这几个方法，不碰行情、不碰柜台
            _terms = staticmethod(SR.PortfolioRunner._terms)
            _size = SR.PortfolioRunner._size
            _volume_multiple = SR.PortfolioRunner._volume_multiple
            _entry_lots_at = SR.PortfolioRunner._entry_lots_at
            _note_theo_entry = SR.PortfolioRunner._note_theo_entry
            _theo_qty = SR.PortfolioRunner._theo_qty
            _theo_lots_qty = SR.PortfolioRunner._theo_lots_qty
            _ib_on = staticmethod(SR.PortfolioRunner._ib_on)
            _ib_open_trade = SR.PortfolioRunner._ib_open_trade
            _latch_theo_lots = SR.PortfolioRunner._latch_theo_lots
            _latch_to_order = SR.PortfolioRunner._latch_to_order
            _theo_rank = staticmethod(SR.PortfolioRunner._theo_rank)
            _side_cost_px = SR.PortfolioRunner._side_cost_px
            _put_theo = SR.PortfolioRunner._put_theo

            def _log(self, *a, **k):
                pass

        try:
            df = tqdata.load_main("RB", "SHFE", "1d")
        except Exception as exc:              # noqa: BLE001
            return f"跳过：读不到 RB 日线（{type(exc).__name__}: {exc}）"
        df = df[[c for c in ("open", "high", "low", "close") if c in df.columns]]
        df = df.dropna().tail(1200)
        if len(df) < 200:
            return f"跳过：RB 日线只有 {len(df)} 根"

        sig = run_strategy(df, "ma_cross", {})["signal"]
        arr = sig.to_numpy()
        sig0 = int(arr[-1])
        if not sig0:
            return "跳过：RB 日线最后那根信号是 0，测不到「持仓中的那笔」"
        need = 6

        f = _Fake()
        st = SR.SlotState(cfg=SStore.StrategyConfig(
            sid="t-lots", name="t-lots", symbol="RB", exchange="SHFE",
            period="1d", family="rule", rule_key="ma_cross", sizing="risk",
            principal=1_000_000.0, risk_pct=2.0, stop_atr_n=14, stop_mult=2.0))
        f._note_theo_entry(st, df, sig)
        if st.theo_lots <= 0:
            raise RuntimeError(
                "「开仓手数」没算出来（theo_lots=0）—— 理论列会退回按当前 K 线"
                "重算，就是用户 11-23 报的那个 bug")
        if st.theo_lots_src != SR.THEO_SIG:
            raise RuntimeError(f"来源该是 sig，实际 {st.theo_lots_src!r}")
        st.signal = sig0
        if f._theo_lots_qty(st) != sig0 * st.theo_lots:
            raise RuntimeError("理论仓 ≠ 方向 × 开仓手数")
        st.target = sig0 * st.theo_lots
        if f._theo_qty(st) != st.target:
            raise RuntimeError("普通策略的理论仓该等于 target（两者同源）")

        # ① 与回测最后一笔逐位相等（用户就是拿这两个数对账的）
        bcfg = BT.BacktestConfig(
            initial_cash=1_000_000.0, sizing="risk", risk_amount=20_000.0,
            stop_atr_n=14, stop_mult=2.0, multiplier=10.0, margin_rate=0.10,
            fee_rate=0.0, fee_per_lot=0.0, tick_size=0.0, slippage_ticks=0.0,
            slippage=0.0, bars_per_year=244.0)
        trades = BT.run_backtest(df.assign(signal=sig), bcfg, label="t").trades
        if not trades:
            return "跳过：这段数据没跑出交易，对不上账"
        last = trades[-1]
        if int(last.lots) != st.theo_lots:
            raise RuntimeError(
                f"实盘口径 {st.theo_lots} 手 ≠ 回测这一笔 {int(last.lots)} 手 ——"
                "两边手数的输入（价 / ATR / 乘数 / 止损额）又分叉了")
        entry_lots = int(st.theo_lots)     # ← 就是与回测对得上的那个数

        # ② 后面接 6 根波幅翻倍的 K 线：手数必须一动不动
        rows, idx6 = [], []
        base = df.iloc[-1]
        for m in range(need):
            row = base.copy()
            row["open"] = base["close"]
            row["high"] = base["close"] * 1.25
            row["low"] = base["close"] * 0.75
            row["close"] = base["close"]
            rows.append(row)
            idx6.append(df.index[-1] + pd.Timedelta(days=m + 1))
        d2 = pd.concat([df, pd.DataFrame(rows, index=idx6)])
        s2 = pd.concat([sig, pd.Series([sig0] * need, index=d2.index[-need:])])
        if len(s2) != len(d2):
            raise RuntimeError(f"造数据失败 {len(s2)} ≠ {len(d2)}")
        st.signal = sig0
        f._note_theo_entry(st, d2, s2)
        if st.theo_lots != int(last.lots):
            raise RuntimeError(
                f"波动一变理论手数就跟着变（{int(last.lots)} → {st.theo_lots}）"
                " —— 又回到「按当前 K 线算」了")
        # 对照：按当前这根重算会是另一个数，否则这条测试证明不了什么
        drifted = f._size(st, float(d2["close"].iloc[-1]), SR.atr(d2, 14))[0]
        if abs(int(drifted) - st.theo_lots) < 1:
            raise RuntimeError(
                f"对照组造不出差异（{drifted} 手 vs {st.theo_lots} 手）—— "
                "这条断言测不到漂移，得把波幅放大")

        # ③ 翻方向必须重算
        flip = pd.concat([sig, pd.Series([-sig0] * need, index=d2.index[-need:])])
        st.signal = -sig0
        f._note_theo_entry(st, d2, flip)
        if st.theo_lots_sig != -sig0:
            raise RuntimeError("翻方向之后手数没有重算 —— 会拿着上一波的手数开新仓")

        # ④ K 线内真触发：手数用那一刻**实际下单**的那份，且此后推断不许盖掉它
        st.lots = 7
        f._latch_to_order(st, -sig0)
        if st.theo_lots_src != SR.THEO_IB:
            raise RuntimeError("K 线内触发之后来源没升到 ib 档")
        if st.theo_lots != 7:
            raise RuntimeError(f"触发时实际下单 7 手，记成了 {st.theo_lots} 手")
        st.lots = 12                      # 收盘那根重算出来是 12 手（波动变了）
        f._note_theo_entry(st, d2, flip)  # 同一波，推断不许覆盖
        if st.theo_lots != 7:
            raise RuntimeError(
                f"实际下单是 7 手，推断却把它改成了 {st.theo_lots} 手 ——"
                "「从开仓那边获取」这一条没守住")

        # ⑤ 源码级：目标仓必须走 `_theo_lots_qty`，不许再是 `signal × lots`
        body = inspect.getsource(SR.PortfolioRunner._refresh_signal)
        if "st.target = st.signal * st.lots" in body:
            raise RuntimeError(
                "`_refresh_signal` 又按当前 K 线的 lots 算目标仓了 —— "
                "同一笔持仓会被 ATR 的漂移一路加仓 / 减仓")
        if "st.target = self._theo_lots_qty(st)" not in body:
            raise RuntimeError("`_refresh_signal` 没用 `_theo_lots_qty()` 定目标仓")

        # ⑥ K 线内那一族（用户最早报的 sbbox）：开仓那根要由**回测那台状态机**
        #    给出来，不许再从「signal 段起点」倒推 —— 实测 AG 的段起点比回测开仓
        #    还晚、RB 的段起点比回测开仓晚六周（用户 11-23：「signal 段起点如果
        #    是 10-22，那确实会很多错误的」）。这一段的回测配置也必须走
        #    `replay.make_bt()`，和 `_ib_open_trade()` 完全同一个出口。
        note6 = ""
        t6 = None                             # 给下面 ⑨ 段用，别留 NameError
        try:
            d1h = tqdata.load_main("RB", "SHFE", "1h").tail(1500)
            r6 = run_strategy(d1h, "sbbox_ib", {})
        except Exception as exc:              # noqa: BLE001
            d1h, r6 = None, None
            note6 = f"；K 线内那半段跳过（读不到 1h：{type(exc).__name__}）"
        if r6 is not None and "stop_dist" in r6.columns:
            s6 = r6["signal"]
            sg6 = int(s6.iloc[-1])
            cfg6 = SStore.StrategyConfig(
                sid="t-ib-lots", name="t-ib-lots", symbol="RB",
                exchange="SHFE", period="1h", family="rule",
                rule_key="sbbox_ib", sizing="risk", principal=1_000_000.0,
                risk_pct=1.0, stop_atr_n=14, stop_mult=2.0)
            st6 = SR.SlotState(cfg=cfg6)
            if sg6:
                f._note_theo_entry(st6, d1h, s6, r6["stop_dist"],
                                   r6["entry_long"], r6["entry_short"],
                                   r6["exit_long"], r6["exit_short"])
                bt6 = SR.replay.make_bt("cn", SR.replay.resolve_allow_short(cfg6),
                                        cfg6)
                res6 = BT.run_backtest(d1h.assign(
                    signal=s6, stop_dist=r6["stop_dist"],
                    entry_long=r6["entry_long"], entry_short=r6["entry_short"],
                    exit_long=r6["exit_long"], exit_short=r6["exit_short"]), bt6)
                t6 = res6.trades[-1] if res6.trades else None
                if t6 is not None and t6.open_position and t6.direction == sg6:
                    if int(t6.lots) != st6.theo_lots:
                        raise RuntimeError(
                            f"K 线内：理论 {st6.theo_lots} 手 ≠ 回测这笔 "
                            f"{int(t6.lots)} 手 —— 开仓那根没找对"
                            "（要按回测状态机，不能拿 signal 段起点）")
                    note6 = f"；K 线内实测 {st6.theo_lots} 手也等于回测 Trade.lots"
                else:
                    note6 = "；K 线内那笔已平仓，对拍跳过"
            else:
                note6 = "；K 线内末根信号是 0，对拍跳过"

        # ⑦ 起点对齐：`replay.window()` 的切点必须落在**整日**上，否则回测末根
        #    15:00 收盘、实盘末根盘中，同一天切出相邻两根当起点，整条状态机
        #    错开一根（用户 11-23：「实盘引擎的开始时间要和回测的一致」）。
        d = pd.date_range("2024-01-01", periods=900, freq="h")
        px = pd.Series(100.0, index=d)
        frame = pd.DataFrame({"open": px, "high": px, "low": px, "close": px})
        w_close = SR.replay.window(frame, 0.05)
        w_intra = SR.replay.window(frame.iloc[:-1], 0.05)
        if w_close.index[0] != w_intra.index[0]:
            raise RuntimeError(
                f"回测（末根 {frame.index[-1]}）切到 {w_close.index[0]}，"
                f"实盘（末根 {frame.index[-2]}）切到 {w_intra.index[0]} —— "
                "起点没按整日锚定，实盘和回测会错开一根")
        note7 = f"；起点按整日锚定（同日 15:00 / 盘中两种末根都切到 {w_close.index[0]}）"

        # ⑧ 降级帧不许写理论值：信号照算，但「理论」列只认同源前复权帧。
        #    2026-11-23 白银基本面分位组合回测最后一笔 11-18 就平了（该 0），
        #    界面却显示 −2 —— 原因就是那一轮用的是「实时原始价 2643 根」。
        st8 = SR.SlotState(cfg=st.cfg)
        st8.frame_ok = True
        f._note_theo_entry(st8, df, sig)
        keep_lots, keep_px = st8.theo_lots, st8.theo_px
        if keep_lots <= 0:
            raise RuntimeError("⑧ 前置条件不成立：降级帧断言需要先有一个非 0 的手数")
        st8.frame_ok = False
        f._note_theo_entry(st8, df, pd.Series([-sig0] * len(df), index=df.index))
        if (st8.theo_lots, st8.theo_px) != (keep_lots, keep_px):
            raise RuntimeError(
                f"⑧ 降级帧把理论值改掉了（{keep_lots}→{st8.theo_lots}）—— "
                "原始价帧算出来的信号和回测不是一回事，不许写进理论列")
        note8 = "；降级帧不写理论值（冻结上一份）"

        # ⑨ 状态机赢过 `ib` 档：`ib` 记的是「触发那一刻 st.lots」，而 st.lots 是
        #    按当根 ATR 重算的数；状态机只要有笔没平的开仓，它就是回测口径。
        #    2026-11-23 AG 理论 −1 / 回测 −2 就是 `ib` 档把正确的 2 盖成了 1。
        note9 = "；K 线内那半段未跑，⑨ 跳过"
        if r6 is not None and "stop_dist" in r6.columns and sg6 and t6 is not None \
                and t6.open_position:
            st9 = SR.SlotState(cfg=cfg6)
            st9.frame_ok = True
            st9.lots = 9
            f._latch_to_order(st9, sg6)          # 假装盘中触发、当时 9 手
            if st9.theo_lots != 9 or st9.theo_lots_src != SR.THEO_IB:
                raise RuntimeError("⑨ 前置条件不成立：ib 档没锁上")
            f._note_theo_entry(st9, d1h, s6, r6["stop_dist"],
                               r6["entry_long"], r6["entry_short"],
                               r6["exit_long"], r6["exit_short"])
            if st9.theo_lots_src != SR.THEO_SIG:
                raise RuntimeError(
                    "⑨ 状态机报得出开仓，来源却没降回 sig 档 —— "
                    "`ib` 档会把「触发那一刻按当根 ATR 算的手数」永久钉住")
            if int(st9.theo_lots) != int(t6.lots):
                raise RuntimeError(
                    f"⑨ 状态机这笔是 {int(t6.lots)} 手，理论列写成 {st9.theo_lots} 手")
            note9 = f"；ib 档（9 手）被状态机的 {st9.theo_lots} 手盖回来"

        return (f"手数在「开仓那一根」定死：RB 日线实测 {entry_lots} 手，"
                f"与回测同一笔的 Trade.lots 逐位相等；"
                f"波幅翻倍后仍是 {entry_lots} 手不漂"
                f"（按当前 K 线重算会变成 {drifted} 手）；翻方向才重算；"
                "K 线内触发后用实际下单手数，推断不覆盖" + note6 + note7
                + note8 + note9)

    check("理论手数取开仓那一根（不按当前 K 线重算）", _theo_entry_lots)

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

    def _cross_lots() -> str:
        """截面交易记录必须有手数（2026-11-23 用户：「截面对冲策略的交易记录没有手数」）。

        以前 `trades_from_weights` 把 `lots` 写死 0.0，理由是「截面是权重不是
        手数」。权重**能**折成手数，只是得给它一个本金：

            名义额 = 本金 × |权重|          ← 截面引擎里 `gross` 就是按这个口径累的
            手数   = 名义额 ÷ (开仓价 × 合约乘数)

        两个坑必须一起钉：

        ① **手数是「每 1 元本金」的比例**（和 `gross` / `cost` 同一套），由
           `to_result` 乘回本金。不乘的话界面上是 0.0000029 手，看着和「没有手数」
           一样；乘两遍则是天文数字。
        ② **乘数必须查 `symbols_info`**，不能用 `BacktestConfig().multiplier`
           （默认 1）—— 沥青乘数 10，拿 1 去算就是 10 倍手数。前复权价还要按
           `price_scale` 换回真实盘面价（`P_adj = P_raw × scale`），否则历史那
           段的手数会偏。
        """
        import numpy as np
        import pandas as pd

        from ttrade import cross as CX
        from ttrade.backtest import BacktestConfig

        rng = np.random.default_rng(11)
        idx = pd.date_range("2023-01-02", periods=260, freq="B")
        frames = {}
        scales = {}
        for k in range(6):
            r = rng.normal(0.0002, 0.012, len(idx))
            px = 100 * np.exp(np.cumsum(r))
            # 前复权因子：最后一根 = 1（和 `futures.load(adjust="qfq")` 同口径）
            sc = np.linspace(0.9, 1.0, len(idx))
            frames[f"S{k}_SHFE"] = pd.DataFrame(
                {"open": px * (1 + rng.normal(0, 0.001, len(idx))),
                 "high": px * 1.01, "low": px * 0.99, "close": px,
                 "price_scale": sc,
                 "volume": rng.integers(1e4, 1e5, len(idx)).astype(float),
                 "open_interest": rng.integers(1e4, 1e5, len(idx)).astype(float)},
                index=idx)
            scales[f"S{k}_SHFE"] = sc

        cfg = CX.CrossConfig(mode="hedge", factor_mode="single",
                             factor_keys=["mom_20"], top_pct=30.0, bot_pct=30.0,
                             weight="equal", rebalance=5)
        cash = 1_000_000.0
        bt = BacktestConfig(initial_cash=cash, fee_rate=0.0003, slippage=0.0)
        out = CX.run(frames, cfg, bt)
        if not out.ok:
            raise RuntimeError(f"截面回测没跑通：{out.error}")
        res = CX.to_result(out, label="截面", step=out.step, bpy=252.0,
                           initial_cash=cash)
        if not res.trades:
            raise RuntimeError("截面回测没产生交易")
        if any(float(t.lots) <= 0 for t in res.trades):
            n = sum(1 for t in res.trades if float(t.lots) <= 0)
            raise RuntimeError(f"{n}/{len(res.trades)} 笔的手数是 0 —— 界面上会显示「—」")

        # 恒等式：手数 × 乘数 × 真实盘面价 == 本金 × |开仓那根的权重|
        mults = CX.mult_vector(frames)
        w = out.weights
        for t in res.trades:
            wv = float(w.at[t.entry_date, t.symbol])
            if abs(wv) <= 0:
                continue
            sc = float(frames[t.symbol]["price_scale"].loc[t.entry_date])
            notional = float(t.lots) * float(mults[t.symbol]) * (t.entry_price / sc)
            want = cash * abs(wv)
            if abs(notional - want) > 1e-6 * max(want, 1.0):
                raise RuntimeError(
                    f"{t.symbol} {t.entry_date:%Y-%m-%d} 名义额 {notional:.2f} "
                    f"≠ 本金×权重 {want:.2f}")

        # 乘数必须是真乘数：拿 1 去算的话沥青会差 10 倍
        real = {"BU_SHFE": 10.0, "RB_SHFE": 10.0, "AG_SHFE": 15.0,
                "AL_SHFE": 5.0, "AU_SHFE": 1000.0}
        got = CX.mult_vector({k: frames["S0_SHFE"] for k in real})
        for k, v in real.items():
            if abs(float(got.get(k, 0.0)) - v) > 1e-9:
                raise RuntimeError(f"{k} 的合约乘数拿到 {got.get(k)}，应该是 {v}")

        # 复权因子方向：前复权价被缩小过，换回真实盘面价要**除** scale。
        # 直接调一次构造函数，把 scale 减半 ⇒ 手数应该也减半（价格按比例变大）
        one = {k: frames[k] for k in ("S0_SHFE",)}
        w1 = pd.DataFrame({"S0_SHFE": [0.1, 0.1, 0.0]},
                          index=idx[:3])
        c1 = pd.DataFrame({"S0_SHFE": [100.0, 100.0, 100.0]}, index=idx[:3])
        base_kw = dict(rets=None, cvec=None, eq_prev=None)
        t_no = CX.trades_from_weights(w1, c1, mults=pd.Series({"S0_SHFE": 10.0}),
                                      **base_kw)[0]
        half = pd.DataFrame({"S0_SHFE": [0.5, 0.5, 0.5]}, index=idx[:3])
        t_half = CX.trades_from_weights(
            w1, c1, mults=pd.Series({"S0_SHFE": 10.0}), scales=half, **base_kw)[0]
        if abs(float(t_no.lots) - 0.1 / (100.0 * 10.0)) > 1e-12:
            raise RuntimeError(f"不复权口径的手数算错：{t_no.lots}")
        if abs(float(t_half.lots) - float(t_no.lots) * 0.5) > 1e-15:
            raise RuntimeError(
                f"复权因子方向错了：scale=0.5 时手数 {t_half.lots}，"
                f"应该是 {float(t_no.lots) * 0.5}")
        return (f"{len(res.trades)} 笔全部有手数；手数×乘数×真实价 = 本金×权重"
                "（逐笔验过）；乘数取真值（BU 10 / AG 15 / AU 1000）；"
                "前复权因子按 P_adj=P_raw×scale 换回真实盘面价")

    check("截面交易记录的手数（权重折手数）", _cross_lots)

    def _pool_reb_cycle():
        """筛品种频率：每 N 个交易日 / 每 N 个月重排一次池，间隔内沿用上一份。

        用户在 2026-11-23 要求「筛品种池下面增加一个筛品种的频率，单位是交易日
        或者月」。语义必须先钉死，否则界面上给了个旋钮、回测里却还在逐 bar 重排
        （看起来完全正常，只是换手悄悄变高）：

        ① `reb=1/day` 就是原来那套（逐 bar 判池），一行都不能变；
        ② `reb=3/day` 每 3 个交易日才采一次，块内**完全相同**；
        ③ 月份口径按**自然月**切，不是按「30 天」；
        ④ 中间那一段真的会和逐 bar 版不同（否则等于这个参数没生效）；
        ⑤ 存读往返：3 个月 ⇒ 3/"month"；脏数据（0 / "bar" / 999）要被夹回。
        """
        import numpy as np
        import pandas as pd

        from ttrade import cross as CX
        from ttrade import strategy_store as SStore

        idx = pd.bdate_range("2026-01-05", periods=24)      # 跨 1、2 两个月
        cols = ["A", "B", "C", "D"]
        # 池分数每天轮换一次名次：这样「重排」一定改变入选名单
        pool = pd.DataFrame(
            [[float((i + j) % 4) for j in range(4)] for i in range(len(idx))],
            index=idx, columns=cols)
        score = pd.DataFrame(1.0, index=idx, columns=cols)

        base = CX._apply_pool(score, pool, 50.0).notna()    # reb 缺省 = 逐 bar

        def _same_rows(a, b) -> bool:
            return bool((a.fillna(False) == b.fillna(False)).all().all())

        # ② 每 3 个交易日
        d3 = CX._apply_pool(score, pool, 50.0, reb=3, reb_unit="day").notna()
        for a in range(0, len(idx), 3):
            blk = d3.iloc[a:a + 3]
            if blk.nunique().max() > 1:
                raise RuntimeError(f"每 3 个交易日的块内池子变了（第 {a} 行起）")
            if not _same_rows(blk.iloc[[0]], base.iloc[[a]]):
                raise RuntimeError(f"第 {a} 行不是按那一根判的池")
        # ④ 必须真的和逐 bar 不同，否则参数等于没生效
        if _same_rows(d3, base):
            raise RuntimeError("每 3 个交易日重排的结果和逐 bar 完全一样 —— 频率没落地")

        # ③ 自然月
        dm = CX._apply_pool(score, pool, 50.0, reb=1, reb_unit="month").notna()
        months = pd.DatetimeIndex(idx).to_period("M")
        for m in months.unique():
            blk = dm[months == m]
            if blk.nunique().max() > 1:
                raise RuntimeError(f"{m} 这个月内池子变了 —— 没有按自然月切")
        # 夏令时 / 月末长度都不用管：月份是日历口径，不是 30 天
        if _same_rows(dm, base):
            raise RuntimeError("按月重排和逐 bar 完全一样 —— 月份口径没落地")

        # ⑤ 存读往返 + 脏数据
        good = SStore.StrategyConfig.from_dict(
            {"symbol": "RB", "exchange": "SHFE", "family": "cross",
             "cross_pool_reb": 3, "cross_pool_reb_unit": "month"})
        if (int(good.cross_pool_reb), good.cross_pool_reb_unit) != (3, "month"):
            raise RuntimeError(
                f"筛池频率存读往返丢了：{good.cross_pool_reb}/{good.cross_pool_reb_unit}")
        bad = SStore.StrategyConfig.from_dict(
            {"symbol": "RB", "exchange": "SHFE", "family": "cross",
             "cross_pool_reb": 0, "cross_pool_reb_unit": "bar"})
        if (int(bad.cross_pool_reb), bad.cross_pool_reb_unit) != (1, "day"):
            raise RuntimeError(
                f"脏数据没夹回来：{bad.cross_pool_reb}/{bad.cross_pool_reb_unit}")
        big = SStore.StrategyConfig.from_dict(
            {"symbol": "RB", "exchange": "SHFE", "family": "cross",
             "cross_pool_reb": 999, "cross_pool_reb_unit": "month"})
        if int(big.cross_pool_reb) != 240:
            raise RuntimeError(f"超大值没夹住：{big.cross_pool_reb}")
        return ("逐 bar = 踩点采样（块内一致）；3 个交易日 / 自然月都真的换了池；"
                "存读往返 3 个月、脏数据夹回 1 个交易日、999 夹到 240")

    check("筛品种频率（交易日 / 月）", _pool_reb_cycle)

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

    def _index_opt_map():
        """股指期货 ↔ 股指期权的品种名映射（IF→IO / IH→HO / IM→MO）。

        期货按**合约**命名、期权按**标的指数**命名，两边永远对不上：
        Tushare 那条 `opt_daily + settle_map` 链路也就建不出
        `opt_IF_*.csv`，股指期货的 5 个 IV 因子一直是空的。
        """
        from ttrade import auxdata as A
        from ttrade import factors as F
        from ttrade import optstore as OS
        from ttrade import tqdata as T
        from ttrade.skills.option import _match_variety

        import pandas as pd

        # 归一化：带交易所后缀、裸代码、小写都要认（`"IF".rpartition("_")`
        # 那支坑过一次，会把品种切空）
        want = {"IF": "IO", "IF_CFFEX": "IO", "if": "IO",
                "IH": "HO", "IM": "MO",
                "IC": "IC",           # 中证500 没有场内期权，不能乱映射
                "CU": "CU", "RB_SHFE": "RB", "CF": "CF"}
        for k, v in want.items():
            got = OS.opt_variety(k)
            if got != v:
                raise RuntimeError(f"{k} 的期权品种应当是 {v}，实际 {got!r}")

        if not OS.exists():
            return "期权库未下载，只验了映射"

        # 走 aux 回退：股指三家必须拿到 IV，且列要和 aux 那份同构
        need = list(OS.IV_COLS)
        got_rows = {}
        for sym in ("IF", "IH", "IM"):
            df = A.load_opt(data_dir("aux"), sym, "CFFEX")
            if df.empty:
                raise RuntimeError(f"{sym} 取不到期权 IV（映射没生效）")
            miss = [c for c in need if c not in df.columns]
            if miss:
                raise RuntimeError(f"{sym} 的 IV 表缺列 {miss}")
            if not isinstance(df.index, pd.DatetimeIndex):
                raise RuntimeError(f"{sym} 的 IV 表索引不是日期")
            got_rows[sym] = len(df)
        # 商品那条路不能被改道：CU 有 aux 文件，必须还是读 aux
        cu = A.load_opt(data_dir("aux"), "CU", "SHFE")
        if cu.empty:
            raise RuntimeError("CU 的期权 IV 丢了（回退把 aux 顶掉了）")

        # 因子层：股指期货日线上 IV 因子真算得出值
        bars = T.load_main("IF", "CFFEX", "1d", adjust="hfq")
        n_iv = 0
        if not bars.empty:
            out = A.attach(bars, "IF", "CFFEX")
            n_iv = int(pd.Series(F.FACTORS["iv_atm"].func(out)).notna().sum())
            if n_iv < 200:
                raise RuntimeError(f"IF 的 iv_atm 只有 {n_iv} 个有效值")

        if _match_variety("IF", ["IO", "cu", "rb"]) != "IO":
            raise RuntimeError("技能层的品种匹配没走映射")
        if _match_variety("IC", ["IO", "cu", "rb"]) is not None:
            raise RuntimeError("IC 不该匹配到任何期权品种")
        return (f"IF→IO / IH→HO / IM→MO；IV 行数 "
                + "、".join(f"{k} {v}" for k, v in got_rows.items())
                + f"；IF 日线 iv_atm {n_iv} 个值")

    check("股指期货 ↔ 期权品种映射（IF→IO）", _index_opt_map)

    def _opt_iv_period():
        """分钟级期权 IV：按 bar 周期取档 + 无前视 + `vol_20` 年化按密度。

        这三处都是**不报错但结果全错**的类型：取错档只让日内台阶变粗、asof
        允许精确匹配只让 IC 好看、年化系数写死 252 只让 `iv_rv` 恒正 —— 图上看
        都像正常因子。所以必须逐条断言，不能只看「有没有数」。
        """
        import numpy as np
        import pandas as pd

        from ttrade import auxdata as A
        from ttrade import factors as F
        from ttrade import optstore as OS

        # ① bar 周期识别：六档 + 日线。用相邻间隔的中位数，别用「每天几根」
        base = pd.Timestamp("2026-01-05 09:00")
        for step, want in ((5, "5m"), (15, "15m"), (30, "30m"),
                           (60, "1h"), (120, "2h"), (240, "4h")):
            idx = pd.DatetimeIndex([base + pd.Timedelta(minutes=step * i)
                                    for i in range(40)])
            df = pd.DataFrame({"close": np.arange(40.0)}, index=idx)
            got = A.bar_period(df)
            if got != want:
                raise RuntimeError(f"{step} 分钟的 bar 认成 {got}，应为 {want}")
        didx = pd.DatetimeIndex(pd.bdate_range("2026-01-05", periods=40))
        got = A.bar_period(pd.DataFrame({"close": np.arange(40.0)}, index=didx))
        if got != "1d":
            raise RuntimeError(f"日线认成 {got}，应为 1d")

        # ② 无前视：IV 快照与 bar 同刻，**同刻那条不能出现在当根 bar 上**
        grid = pd.DatetimeIndex([base, base + pd.Timedelta(hours=1),
                                 base + pd.Timedelta(hours=2)])
        iv = pd.DataFrame({"iv_near": [1.0, 2.0, 3.0]}, index=grid)
        bars = pd.DataFrame(
            {"close": [10.0, 11.0, 12.0, 13.0]},
            index=grid.append(pd.DatetimeIndex([base + pd.Timedelta(days=63)])),
        )
        vals = A._join_bar_asof(bars, iv)["iv_near"].to_numpy(dtype=float)
        if not np.isnan(vals[0]):
            raise RuntimeError(f"第一根 bar 不该有 IV，实际 {vals[0]}")
        if not (vals[1] == 1.0 and vals[2] == 2.0):
            raise RuntimeError(f"同刻快照被提前用了：{list(vals[:3])}，应为 nan/1/2")
        if not np.isnan(vals[3]):
            raise RuntimeError(f"覆盖窗口外仍在沿用最后一条快照：{vals[3]}")

        # ②b 退档（1h 快照挂 5m bar）：**快照所属的 1h bar 必须已经收盘**。
        # 库里 1h 的 dt 是起始时间，收盘在 dt+1h —— 用同一合约收盘价反推证实过
        # （dt=09:00 的 close 等于 5m 里 09:55 那根的 close）。只判 `dt < t` 的话
        # 09:30 这根 5m bar 会拿到 09:00 那条（含 09:30~10:00 的成交），前视半小时。
        s5 = pd.DatetimeIndex([base + pd.Timedelta(minutes=30 * i)
                               for i in range(6)])
        bars5 = pd.DataFrame({"close": np.arange(6.0)}, index=s5)
        h1 = pd.DataFrame(
            {"iv_near": [0.5, 1.0, 2.0]},
            index=pd.DatetimeIndex([base - pd.Timedelta(hours=1), base,
                                    base + pd.Timedelta(hours=1)]),
        )
        got_c = A._join_bar_asof(bars5, h1,
                                 coarse=A.period_gap("1h"))["iv_near"]
        got_c = got_c.to_numpy(dtype=float)
        # dt: 08:00→0.5 09:00→1.0 10:00→2.0，各自在 +1h 之后才可见
        want_c = np.array([0.5, 0.5, 1.0, 1.0, 2.0, 2.0])
        if not np.allclose(got_c, want_c, equal_nan=True):
            raise RuntimeError(
                f"退档用了未收盘的 1h 快照：{list(got_c)}，应为 {list(want_c)}"
                "（09:30 只能拿到 08:00 那条；09:00 那条含 09:00~10:00 的成交，"
                "要到 10:00 才可见）")
        # 同档（5m 快照挂 5m bar）：后移 5 分钟后与「dt 严格早于 bar」逐位等价
        grid5 = pd.DatetimeIndex([base + pd.Timedelta(minutes=5 * i)
                                  for i in range(30)])
        iv5 = pd.DataFrame({"iv_near": np.arange(30.0) / 10.0}, index=grid5)
        same = A._join_bar_asof(bars5, iv5, coarse=A.period_gap("5m"))
        old = A._join_bar_asof(bars5, iv5)          # coarse=None → 原语义
        if not np.allclose(same["iv_near"].to_numpy(dtype=float),
                           old["iv_near"].to_numpy(dtype=float),
                           equal_nan=True, rtol=0, atol=0):
            raise RuntimeError("同档退档改动把 5m→5m 的结果改了（IH 会掉数）")

        # ③ 档位选择：不能挑到停更的档（IO 的 5m 只到 2020-11）
        picked = "IF 取档未验（期权库未下载）"
        if OS.exists():
            dspan = OS.iv_span("IF", "1d")
            if dspan is None:
                raise RuntimeError("IF 在期权库里没有日频 IV")
            i5 = pd.DatetimeIndex([base + pd.Timedelta(minutes=5 * i)
                                   for i in range(120)])
            p = A._iv_period_for(pd.DataFrame({"close": np.arange(120.0)},
                                              index=i5), "IF")
            if p == "1d":
                raise RuntimeError("IF 有分钟档，5m bar 不该退到日频")
            sp = OS.iv_span("IF", p)
            if sp is None:
                raise RuntimeError(f"IF 挑到了没有数据的档 {p}")
            if sp[2] < dspan[2] - pd.Timedelta(days=30):
                raise RuntimeError(
                    f"IF 挑到停更的档 {p}（到 {sp[2].date()}，日频到 {dspan[2].date()}）")
            if sp[3] < dspan[3] * 0.5:
                raise RuntimeError(
                    f"IF 挑到太浅的档 {p}（只有 {sp[3]} 天，日频 {dspan[3]} 天）")
            # 有 aux 文件的商品必须锁死日频，否则换周期就换了数据源
            cu = A._use_iv_period(data_dir("aux"), "CU", "SHFE",
                                  pd.DataFrame({"close": np.arange(120.0)}, index=i5))
            if cu != "1d":
                raise RuntimeError(f"CU 有 aux 文件却用了日内档 {cu}")
            picked = f"IF 5m bar → {p}（{sp[0]} 行 / {sp[3]} 天）"

        # ⑤ 取档闸门本身（纯函数，用构造的覆盖数据，不依赖库里的现状）
        day = (1000, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-11-01"), 900)
        deep_new = (5000, pd.Timestamp("2024-01-02"), pd.Timestamp("2026-10-30"), 800)
        stale = (7000, pd.Timestamp("2020-03-23"), pd.Timestamp("2020-11-16"), 155)
        thin = (400, pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-30"), 40)
        gate = [
            ("够新够深 → 选中", {"5m": deep_new}, ["5m", "1h"], "5m"),
            ("停更五年 → 跳过", {"5m": stale, "1h": deep_new}, ["5m", "1h"], "1h"),
            ("只回补两个月 → 跳过（深度闸门）",
             {"5m": thin, "1h": deep_new}, ["5m", "1h"], "1h"),
            ("全都不合格 → 回日频", {"5m": thin, "1h": stale}, ["5m", "1h"], "1d"),
            ("没有这档 → 往下退", {"1h": deep_new}, ["5m", "1h"], "1h"),
        ]
        for why, spans, cands, want_p in gate:
            got_p = A._pick_iv_tier(cands, spans, day)
            if got_p != want_p:
                raise RuntimeError(f"取档闸门「{why}」判成 {got_p}，应为 {want_p}")

        # ④ vol_20 年化：日线必须逐位不变（历史基线和 IC 都按 252 算过）
        rng = np.random.default_rng(7)
        d = pd.DataFrame(
            {"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))},
            index=pd.bdate_range("2025-01-06", periods=300),
        )
        want = (d["close"].pct_change().rolling(20, min_periods=20).std(ddof=1)
                * np.sqrt(F.TRADING_DAYS)).to_numpy()
        got = F.FACTORS["vol_20"].func(d).to_numpy()
        if not np.allclose(got, want, equal_nan=True, rtol=0, atol=1e-15):
            raise RuntimeError("日线 vol_20 被改了口径（必须仍是 sqrt(252)）")
        # 分钟线按「一年多少根」放缩：10 天 × 6 根 → sqrt(252 × 6)
        hrs = pd.DatetimeIndex([pd.Timestamp("2026-01-05") + pd.Timedelta(days=i)
                                + pd.Timedelta(hours=h)
                                for i in range(10) for h in (9, 10, 11, 13, 14, 15)])
        hi = pd.DataFrame(
            {"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(hrs))))},
            index=hrs,
        )
        sc = F._ann_scale(hi)
        want_sc = float(np.sqrt(F.TRADING_DAYS * 6))
        if abs(sc - want_sc) > 1e-9:
            raise RuntimeError(f"1h 的年化系数 {sc:.4f}，应为 {want_sc:.4f}")
        if abs(F._ann_scale(d) - float(np.sqrt(F.TRADING_DAYS))) > 1e-12:
            raise RuntimeError("日线年化系数被分钟逻辑带偏了")
        return f"6 档周期识别 + 无前视；{picked}；1h 年化 ×{sc / np.sqrt(F.TRADING_DAYS):.2f}"

    check("分钟级期权 IV（取档 / 无前视 / 年化）", _opt_iv_period)

    def _factor_window_bars():
        """标准化窗口的单位是 **bar 根数**，不做周期换算。

        曾经按交易日换算过一版（5m 上 252 = 12000 根），后来改回来了。要钉住的
        是三件事：① 同一个 252 在任何周期上都是 252 根；② 日线逐位不变（历史
        IC 基线都按 252 根算的）；③ 5m 的第一个有效值落在第 63 根，不是第
        3024 根 —— 后者会让回测窗口前两个月因子全空，过滤条件形同不存在。
        """
        import numpy as np
        import pandas as pd

        from ttrade import factors as F

        days = pd.bdate_range("2024-01-01", periods=70)
        d1 = pd.DatetimeIndex(days)
        h1 = pd.DatetimeIndex([d + pd.Timedelta(hours=h)
                               for d in days for h in (9, 10, 11, 13, 14, 15)])
        m5 = pd.DatetimeIndex([d + pd.Timedelta(minutes=5 * i)
                               for d in days for i in range(48)])
        wk = pd.DatetimeIndex([days[0] + pd.Timedelta(weeks=i) for i in range(80)])

        # ① 各档都是 252 根：窗口参数不跟着周期变长度
        for name, idx in (("日线", d1), ("1h", h1), ("5m", m5), ("周线", wk)):
            got = F.win_bars(252)
            if got != 252:
                raise RuntimeError(f"{name} 的窗口算成 {got} 根，应为 252 根")
            if F.win_bars(0) != 1:
                raise RuntimeError("窗口 0 应钳到 1 根（rolling(0) 会抛异常）")

        # ② 日线必须**逐位不变**：历史 IC 基线和所有日线结论都按 252 根算的
        rng = np.random.default_rng(23)
        s = pd.Series(rng.normal(size=400).cumsum(),
                      index=pd.bdate_range("2023-01-02", periods=400))
        want = ((s - s.rolling(252, min_periods=63).mean())
                / s.rolling(252, min_periods=63).std(ddof=1)).clip(-4, 4)
        got = F.standardize(s, window=252, method="z")
        if not np.allclose(got.to_numpy(), want.to_numpy(), equal_nan=True,
                           rtol=0, atol=1e-12):
            raise RuntimeError("日线标准化被「按天」改动带偏了（必须逐位不变）")

        # ③ 5m 的预热是 63 根（252 的 1/4），和日线一模一样
        s5 = pd.Series(np.arange(len(m5), dtype=float), index=m5)
        z5 = F.standardize(s5, window=252, method="z")
        first = int(z5.notna().to_numpy().argmax())      # 0-based
        nb = F.win_bars(252)
        if first != nb // 4 - 1:
            raise RuntimeError(
                f"5m 的首个有效值在第 {first + 1} 根，应为第 {nb // 4} 根")
        if first > 200:
            raise RuntimeError(
                f"5m 的窗口又被按交易日换算了（首值第 {first + 1} 根，"
                "按天算会是 3024 根）")

        # ④ IC / 分桶窗口同一个口径，否则「标准化按根、IC 按天」会打架
        from ttrade import factor_metrics as M
        raw5 = pd.Series(rng.normal(size=len(m5)).cumsum(), index=m5)
        fwd5 = raw5.pct_change().shift(-1)
        w60 = F.win_bars(60)
        if w60 != 60:
            raise RuntimeError(f"窗口 60 算成 {w60} 根，应为 60 根")
        ic5 = M.ic_series(raw5, fwd5, 60)
        first_ic = int(ic5.notna().to_numpy().argmax())      # 0-based
        if int(ic5.notna().sum()) == 0:
            raise RuntimeError("IC 序列整列为空：滚动窗口没跟上")
        if first_ic > 200:
            raise RuntimeError(
                f"IC 第一个有效值在第 {first_ic + 1} 根，窗口被按天算过"
                f"（按根算第 {w60 // 2} 根就该出值）")
        q = M.quantile_returns(raw5, fwd5, 5, 60)
        if len(q) != 5:
            raise RuntimeError(f"分位收益只有 {len(q)} 桶，应为 5")

        return (f"各周期都是 252 根（日线逐位不变）/ 5m 首值第 {first + 1} 根 / "
                f"IC 首值第 {first_ic + 1} 根；分位同口径")

    check("因子窗口按根（不是交易日）", _factor_window_bars)

    def _portfolio_plan():
        """组合工作区的资金分配：ERC / 三种口径 / 缩放。

        这一层全是「不报错但结果错」的类型：ERC 用错迭代会把权重推到角点
        （看着解出来了，其实只留两条策略）；`fixed` 口径被放大而手数根本不动；
        `usage` 口径忘了按实测占用反推，本金就只是被均分了。所以逐条钉死。
        """
        import numpy as np
        import pandas as pd

        from ttrade import portfolio_plan as PL

        # ① ERC：风险贡献份额必须**完全相等**
        rng = np.random.default_rng(31)
        for n in (1, 2, 5, 12):
            a = rng.normal(size=(n, n))
            cov = a @ a.T / n + np.eye(n) * 0.2
            d = np.sqrt(np.diag(cov))
            cov = cov / np.outer(d, d)
            w = PL.erc_weights(cov)
            if abs(float(w.sum()) - 1.0) > 1e-12:
                raise RuntimeError(f"n={n} 的 ERC 权重和不等于 1（{w.sum()}）")
            rc = w * (cov @ w)
            share = rc / rc.sum()
            if float(np.max(np.abs(share - 1.0 / n))) > 1e-6:
                raise RuntimeError(
                    f"n={n} 的 ERC 风险贡献不相等：{np.round(share, 6)}，"
                    f"应全为 {1.0 / n:.6f}")
            # 完美共线也不能炸
            one = np.ones((n, n))
            w2 = PL.erc_weights(one)
            if not np.all(np.isfinite(w2)) or abs(float(w2.sum()) - 1.0) > 1e-9:
                raise RuntimeError("完全共线时 ERC 权重不是有限值/和为 1")

        # ② 造三条策略：占用率故意拉开，才能看出「按占用」和「按本金」的区别
        def mk(sid, principal, avg, peak, vol, sizing="capital",
               pf=0.0, span=0.0, blown=False):
            """pf / span 默认 0 = 「不知道」，风险平价里按中性处理（不奖不罚）。"""
            return PL.Usage(sid=sid, name=sid, sizing=sizing, principal=principal,
                            capital=principal * 0.3, avg_margin=avg,
                            peak_margin=peak, avg_notional=avg * 10,
                            peak_notional=peak * 10, util=avg / principal,
                            vol=vol, ret=0.5, max_dd=0.2, bars=1000,
                            profit_factor=pf, span_years=span, blown=blown)

        us = [mk("A", 100_000, 1_000, 20_000, 0.20),
              mk("B", 100_000, 5_000, 25_000, 0.40),
              mk("C", 100_000, 500, 5_000, 0.10, sizing="fixed")]
        total = 300_000.0

        # ②a 按分配额：本金正好被均分，且合计 = 总资金
        sh = PL.allocate(us, total, "equal", basis="principal")
        if abs(sum(s.new_principal for s in sh) - total) > 1e-6:
            raise RuntimeError("按分配额的本金合计不等于总资金")
        if any(abs(s.weight - 1 / 3) > 1e-12 for s in sh):
            raise RuntimeError("资金等权的权重不是 1/3")

        # ②b 按平均占用：缩放后预计占用 == 目标（固定手数那条除外）
        sh = PL.allocate(us, total, "equal", basis="usage")
        for s in sh:
            if s.sid == "C":
                if abs(s.new_principal - 100_000) > 1e-6:
                    raise RuntimeError("固定手数那条被改了本金 —— 手数根本不会动")
                if "固定手数" not in s.note:
                    raise RuntimeError("固定手数那条没有备注")
                continue
            if abs(s.est_avg - s.target) > 1e-6:
                raise RuntimeError(
                    f"{s.sid} 按平均占用校准后预计占用 {s.est_avg:.2f} ≠ 目标 {s.target:.2f}")

        # ②c 按峰值占用：对齐的是**峰值**。`est_avg` 现在是「同口径预计值」
        #     （按峰值校准就是预计峰值），所以它必须等于目标；而平均占用
        #     本身不该被拉到目标（那才是按平均口径做的事）
        sh = PL.allocate(us, total, "equal", basis="peak")
        if abs(sh[0].est_avg - sh[0].target) > 1e-6:
            raise RuntimeError(
                f"按峰值口径的预计占用 {sh[0].est_avg:.2f} ≠ 目标 {sh[0].target:.2f}")
        if abs(sh[0].scale - (sh[0].target / 20_000.0)) > 1e-9:
            raise RuntimeError("按峰值的缩放系数不对")
        if sh[0].avg_margin >= sh[0].target:
            raise RuntimeError("按峰值校准不该把平均占用也拉到目标额")

        # ②d 三个旋钮给出**同一个占用**，区别只在本金 / 占用百分比怎么分
        got = {}
        for knob in ("principal", "pct", "split"):
            got[knob] = PL.allocate(us, total, "equal", basis="usage", knob=knob)
            for s in got[knob]:
                if s.sid == "C":
                    continue
                if abs(s.est_avg - s.target) > 1e-6:
                    raise RuntimeError(
                        f"knob={knob} 的 {s.sid} 预计占用 {s.est_avg:.2f} ≠ 目标 {s.target:.2f}")
        # 只调本金：本金膨胀、百分比不动
        if abs(got["principal"][1].new_pct - got["principal"][1].old_pct) > 1e-12:
            raise RuntimeError("knob=principal 不该改占用百分比")
        if got["principal"][1].new_principal <= got["principal"][1].old_principal:
            raise RuntimeError("knob=principal 该把本金放大")
        # 只调百分比：百分比没顶到上限时本金必须一动不动（顶到了就必然外溢到
        # 本金 —— 否则占用根本调不到位，那才是真的错）
        if (abs(got["pct"][1].new_pct - PL.PCT_CAP) > 1e-9
                and abs(got["pct"][1].new_principal - got["pct"][1].old_principal) > 1e-6):
            raise RuntimeError("knob=pct 在百分比没顶到上限时改了本金")
        if got["pct"][1].new_pct <= got["pct"][1].old_pct:
            raise RuntimeError("knob=pct 该把占用百分比放大")
        # split：先由百分比承担（本金往目标额上归位），本金膨胀必须**小于**
        # 纯本金口径。百分比顶到上限时本金仍会超目标 —— 那是数学上躲不掉的，
        # 因为「占用 = 分配额」等于要求保证金用满
        sp = [s for s in got["split"] if s.sid != "C"]
        for s, p in zip(sp, [s for s in got["principal"] if s.sid != "C"]):
            if s.new_principal > p.new_principal + 1e-6:
                raise RuntimeError(
                    f"{s.sid} 的 split 本金 {s.new_principal:,.0f} 比纯本金口径 "
                    f"{p.new_principal:,.0f} 还大 —— 拆分方向反了")
        # 百分比顶到上限之后，余下的必须落回本金，否则占用准不了
        cap_us = [mk("D", 100_000, 500, 40_000, 0.2)]
        cap_us[0].capital_pct = 5.0
        sc = PL.allocate(cap_us, 300_000, "equal", basis="usage", knob="pct",
                         pct_cap=20.0)[0]
        if sc.new_pct > 20.0 + 1e-9:
            raise RuntimeError(f"占用百分比超过了上限：{sc.new_pct}")
        if abs(sc.est_avg - sc.target) > 1e-6:
            raise RuntimeError("百分比顶到上限后没有把余下的差异转到本金")

        # ②e 非 capital 口径：百分比不该被动（手数压根不看它）
        for sz in ("risk", "unit"):
            u = mk("E", 100_000, 1_000, 20_000, 0.2, sizing=sz)
            u.capital_pct = 30.0
            e = PL.allocate([u], 300_000, "equal", basis="usage", knob="pct")[0]
            if abs(e.new_pct - 30.0) > 1e-12:
                raise RuntimeError(f"sizing={sz} 时不该改占用百分比")
            if abs(e.est_avg - e.target) > 1e-6:
                raise RuntimeError(f"sizing={sz} 时本金没把占用校准到位")

        # ③ 风险平价：权重 ∝ 1/(净值波动 ÷ 资金利用率 × 回测时长系数)
        #
        # 🔴 这条口径 3.6.1 改过：以前权重只看净值波动，于是「用 10% 的钱做出
        #    20% 波动」和「用 90% 的钱做出 20% 波动」拿一样多 —— 前者每块钱
        #    承担 200% 波动、后者只有 22%，明显该少给。所以下面分两组验：
        #    利用率一致时看波动，波动一致时看利用率。
        same_u = [mk("A", 100_000, 50_000, 60_000, 0.20),
                  mk("B", 100_000, 50_000, 60_000, 0.40),
                  mk("C", 100_000, 50_000, 60_000, 0.10)]
        w = PL.weights("risk", same_u)
        if abs(float(w.sum()) - 1.0) > 1e-12:
            raise RuntimeError("风险平价权重和不为 1")
        if not (w[1] < w[0] < w[2]):
            raise RuntimeError(
                f"利用率相同时波动率越大权重该越小，实际 {np.round(w, 4)}")
        # 波动一致、利用率 10% vs 90%：前者单位资金的风险是后者的 9 倍 ⇒ 少给
        diff_u = [mk("L", 100_000, 10_000, 12_000, 0.20),
                  mk("H", 100_000, 90_000, 99_000, 0.20)]
        wu = PL.weights("risk", diff_u)
        if not (wu[0] * 9.0 < wu[1] * 1.001 and wu[1] > wu[0]):
            raise RuntimeError(
                f"同样波动下利用率低的该拿更少（util 10% vs 90%），"
                f"实际 {np.round(wu, 4)}")

        # ④ 手动：按金额占比归一
        w = PL.weights("manual", us, manual={"A": 1, "B": 2, "C": 1})
        if not np.allclose(w, [0.25, 0.5, 0.25]):
            raise RuntimeError(f"手动设置没按比例归一：{np.round(w, 4)}")

        # ⑤ 顺序一致：权重的下标要和 usages 对齐（错位会把钱分给别的策略）
        sh = PL.allocate(us, total, "manual", manual={"A": 1, "B": 0, "C": 0},
                         basis="principal")
        if abs(sh[0].new_principal - total) > 1e-6 or sh[1].new_principal > 1e-6:
            raise RuntimeError("手动分配的顺序错位了")

        # ⑥ 亏穿：波动率按**整条曲线**的真值参与配权（3.6.2 用户要求）。
        #
        #    🔴 以前 `_metrics` 在 `equity.min() <= 0` 时把 `ann_vol` 标 NaN，
        #    亏穿的策略配权只能按「组合典型风险」估一个数 —— 用户看到的正是
        #    「波动明明很大，风险分数却不高」。现在两条都要钉住：
        #      ⑥a `blown` 标记**不改变**权重（它只用于说明，不是评分项）
        #      ⑥b 亏穿那条波动大 ⇒ 权重**确实更小**（真值参与排序，没被顶替）
        #    另外 NaN 那条防线仍要留（老记录 / 取不到行情），否则 `1/vol`
        #    会把整张表的权重全变成 nan。
        calm = mk("K", 100_000, 50_000, 50_000, 0.20)
        hot = mk("H", 100_000, 50_000, 50_000, 0.80)
        calm_b = mk("K", 100_000, 50_000, 50_000, 0.20, blown=True)
        w_plain = PL.risk_parity([calm, hot])
        w_mark = PL.risk_parity([calm_b, hot])
        if float(np.max(np.abs(w_plain - w_mark))) > 1e-12:
            raise RuntimeError(
                f"「曾经亏穿」不该改变权重，实际 {np.round(w_plain, 4)} vs "
                f"{np.round(w_mark, 4)}")
        # ⑥b 真亏穿的那条（波动 120%）权重必须小于温和的那条（20%）
        blew = mk("B", 100_000, 50_000, 50_000, 1.20, blown=True)
        w2 = PL.risk_parity([blew, calm])
        if not w2[0] < w2[1]:
            raise RuntimeError(
                f"亏穿且波动大的策略权重该更小，实际 {np.round(w2, 4)} —— "
                f"波动率被「典型风险」顶替了")
        # 说明里必须点名，否则用户不知道为什么它权重小
        _sc, _whys = PL.risk_scores([blew, calm])
        if "亏穿" not in _whys[0]:
            raise RuntimeError(f"亏穿没在说明里点名：{_whys[0]!r}")

        # ⑥c NaN 防线：混进一条 NaN 不能把全体带成 nan
        bad = us + [mk("X", 200_000, 100, 50_000, float("nan"))]
        w = PL.risk_parity(bad)
        if not np.all(np.isfinite(w)):
            raise RuntimeError(f"NaN 波动率把权重全带成 nan 了：{np.round(w, 4)}")
        if abs(float(w.sum()) - 1.0) > 1e-12:
            raise RuntimeError("混入 NaN 之后权重和不为 1")
        sh = PL.allocate(bad, total, "risk", basis="usage")
        if not all(np.isfinite([s.weight, s.target, s.scale, s.new_principal,
                                s.est_avg]).all() for s in sh):
            raise RuntimeError("NaN 顺着 allocate 传下去了")

        # ⑦ 🔴 vol=0 比 NaN 更危险：它不报错，直接吃掉全部资金。
        #    没量过占用的策略走默认 `vol=0`，`max(vol,1e-9)` 让 1/vol=1e9，
        #    实测 23 条正常 + 4 条没数据时，没数据的 4 条合计拿到 100%。
        mix = ([mk(f"N{i}", 100_000, 100, 40_000, 0.30) for i in range(6)]
               + [mk(f"Z{i}", 100_000, 100, 1_000, 0.0) for i in range(2)])
        w = PL.risk_parity(mix)
        if not np.all(np.isfinite(w)) or abs(float(w.sum()) - 1.0) > 1e-12:
            raise RuntimeError(f"缺波动率的场景权重异常：{np.round(w, 4)}")
        tail = float(w[6:].sum())
        if tail > 0.5:
            raise RuntimeError(
                f"没有波动率数据的策略拿走了 {tail:.1%} 权重（应按典型风险计）")
        if abs(w[6] - w[0]) > 1e-9:
            raise RuntimeError("缺数据那条该按组合典型风险配权")
        # 全部 vol=0（刚开工作区、一条都没量过）→ 退回等权
        zero = PL.risk_parity([mk(f"Q{i}", 100_000, 100, 1_000, 0.0) for i in range(4)])
        if not np.allclose(zero, 0.25):
            raise RuntimeError(f"全都缺波动率时该等权，实际 {np.round(zero, 4)}")

        # ⑧ 🔴 「应用分配」之后预览表还显示应用前的「本金(前)」——
        #    用户的原话是「点了应用，参数没保存，下次又恢复成原来的」。
        #    根因：`_alloc_preview` 传给 `allocate_workspace` 的是「量占用」那一刻
        #    的 `_usages` 快照，里面的 `principal` / `capital_pct` 是应用前的。
        #    不折算的话 `knob=split` 的 `kprin = 目标 ÷ 本金(前)` 也拿旧值当分母。
        u0 = mk("A", 100_000, 60_000, 87_000, 0.20)
        u0.meas_principal, u0.meas_pct = 100_000, 30.0
        # 应用了一次：本金 10 万 → 33.3 万、百分比 30% → 50%（占用 ×5.56）
        u1 = PL.rebase(u0, 333_333.33, 50.0)
        if abs(u1.principal - 333_333.33) > 1e-6 or abs(u1.capital_pct - 50.0) > 1e-9:
            raise RuntimeError("rebase 没把本金/占用百分比更新成当前配置")
        if abs(u1.avg_margin - 333_333.33) > 1.0:
            raise RuntimeError(
                f"实测占用没按配置折算：{u1.avg_margin:,.0f}，应 ≈ 333,333")
        # 幂等：再折一次不能又乘一遍
        u2 = PL.rebase(u1, 333_333.33, 50.0)
        if abs(u2.avg_margin - u1.avg_margin) > 1e-6:
            raise RuntimeError("rebase 不幂等：同一配置折算两遍值又变了")
        # 而且折算之后 `allocate` 必须说「不用改」（缩放 1.00x）
        sh = PL.allocate([u1], 333_333.33, "equal", basis="usage", knob="split")
        if abs(sh[0].scale - 1.0) > 1e-6:
            raise RuntimeError(
                f"折算到当前配置后缩放应为 1.00x，实际 {sh[0].scale:.4f}x")
        if abs(sh[0].old_principal - 333_333.33) > 1e-6:
            raise RuntimeError("预览表的「本金(前)」不是刚应用后的值")
        # 固定手数不许折：手数不跟本金走，硬乘等于凭空加杠杆
        fx = mk("F", 100_000, 60_000, 87_000, 0.20, sizing="fixed")
        fx.meas_principal = 100_000
        if abs(PL.rebase(fx, 500_000, 30.0).avg_margin - 60_000) > 1e-6:
            raise RuntimeError("固定手数那条被 rebase 折了 —— 手数根本不会变")

        # ⑨ 风险平价的三把尺子：利用率 / 盈利因子 / 回测时长
        # ⑨a 盈利因子低于门槛（`PF_FLOOR`，现在 1.1）⇒ 最低档（= 组合里风险最高
        #     那一档的权重）。不判 0：判 0 等于静默把策略从组合里删掉，那是替
        #     用户做决定
        pfu = [mk("OK", 100_000, 50_000, 60_000, 0.10, pf=2.0),
               mk("HI", 100_000, 50_000, 60_000, 0.40, pf=2.0),
               mk("LO", 100_000, 50_000, 60_000, 0.05, pf=1.05)]
        w = PL.risk_parity(pfu)
        if abs(w[2] - w[1]) > 1e-12 or w[2] <= 0:
            raise RuntimeError(
                f"盈利因子 1.05 < {PL.PF_FLOOR:g} 该落到最低档"
                f"（= 波动最大那条的权重），实际 {np.round(w, 4)}")
        # 反过来：门槛以上不罚。用 1.15 —— 它低于旧的 1.3、高于现在的 1.1，
        # 门槛一旦被改回去这条就会红
        ok = [mk("A", 100_000, 50_000, 60_000, 0.10, pf=1.15),
              mk("B", 100_000, 50_000, 60_000, 0.20, pf=5.0)]
        w = PL.risk_parity(ok)
        if not (w[1] < w[0] < 0.75):
            raise RuntimeError(
                f"盈利因子 1.15 ≥ {PL.PF_FLOOR:g} 不该被打最低档，"
                f"实际 {np.round(w, 4)}")
        # ⑨b 回测时长分档：≥3 年 > 2~3 年 > 1~2 年 > 半年~1 年，且幅度 ≤ 1.40
        sp = [mk("Y4", 100_000, 50_000, 60_000, 0.20, span=4.0),
              mk("Y2", 100_000, 50_000, 60_000, 0.20, span=2.5),
              mk("Y1", 100_000, 50_000, 60_000, 0.20, span=1.2),
              mk("Y0", 100_000, 50_000, 60_000, 0.20, span=0.8)]
        w = PL.risk_parity(sp)
        if not (w[0] > w[1] > w[2] > w[3]):
            raise RuntimeError(f"回测越长权重该越大，实际 {np.round(w, 4)}")
        if w[0] / w[3] > 1.40 + 1e-9:
            raise RuntimeError(
                f"时长的影响 {w[0] / w[3]:.2f}x 盖过了波动率本身（上限 1.40x）")
        # ⑨c 不足半年 ⇒ 最低档
        new = [mk("OLD", 100_000, 50_000, 60_000, 0.30, span=3.0),
               mk("NEW", 100_000, 50_000, 60_000, 0.05, span=0.3)]
        w = PL.risk_parity(new)
        if abs(w[0] - w[1]) > 1e-12:
            raise RuntimeError(
                f"回测不足半年该落到最低档（和最高风险那条同权），"
                f"实际 {np.round(w, 4)}")
        # ⑨d 「不知道」不奖不罚：pf=0 / span=0 的分数必须和「3 年 + 1.8」一致
        unk = [mk("A", 100_000, 50_000, 60_000, 0.20),
               mk("B", 100_000, 50_000, 60_000, 0.20, pf=1.8, span=3.0)]
        w = PL.risk_parity(unk)
        if abs(w[0] - w[1]) > 1e-12:
            raise RuntimeError(
                f"没量到盈利因子/时长时该按中性处理，实际 {np.round(w, 4)}")
        # ⑨e 档位说明必须能传到预览表上（否则用户不知道权重为什么是 1.x%）
        sh = PL.allocate(pfu, 300_000, "risk", basis="usage")
        if "最低档" not in (sh[2].tier or ""):
            raise RuntimeError(f"最低档没在预览表点名：{sh[2].tier!r}")
        if abs(sh[2].pf - 1.05) > 1e-9:
            raise RuntimeError("预览表里的盈利因子没带到 Share 上")

        # ⑩ 一手保底：分到的钱开不出 1 手就抬到够 1 手，**不动别人**，
        #    差额加进总本金（用户要求「不挤压其余品种的权重分配，只是增加
        #    总本金」）。
        #
        #    `capital` 是唯一会「钱不够一手 ⇒ 0 手 ⇒ 整段空跑」的口径；
        #    `budget` / `risk` 在 `size_lots` 里自带 `max(1, ...)`，`fixed`
        #    手数写死，这三种只写说明、不抬。
        #
        #    🔴 门槛是**解析**的：`P_req = L_req ÷ k`（k = 每 1 元本金动用几成）。
        #    拿「实测平均手数 ÷ 本金」反推杠杆是**按成交时点加权**的 —— 成交
        #    集中在便宜的年份（那几年 1/L 特别大）就会高估杠杆、把门槛估低。
        #    实测集运被抬到 37,014 后，最贵那一手只有 0.48 手，照样开不出。
        #    下面第 ① 条专门反证「门槛不许跟着成交手数走」。
        #
        #    🔴 判定的本金必须是**分到的**那份：判定的是「分到的钱够不够」，
        #    拿量测时的本金去判，分配把它砍半了也看不出来（工作区 23 条里
        #    有 4 条正是这种）。
        def mk_lot(sid, principal, vol, lot_avg, lot_peak, lots=1.0,
                   sizing="capital", pct=100.0, mr=0.1):
            u = mk(sid, principal, 1_000, 9_000, vol, sizing=sizing,
                   pf=2.0, span=4.0)
            u.capital_pct = pct
            u.capital = principal * pct / 100.0
            u.lot_margin_avg = lot_avg
            u.lot_margin_peak = lot_peak
            u.avg_lots = lots
            u.margin_rate = mr
            return u

        # 一手均价 10 万、峰值 20 万 ⇒ 目标 15 万；pct=100% ⇒ 门槛就是 15 万
        s_small = mk_lot("S", 100_000, 0.30, 100_000, 200_000, 1.0)
        s_rich = mk_lot("R", 100_000, 0.20, 10_000, 10_000, 8.0)
        lot_us = [s_small, s_rich]
        need, _why = PL.lot_floor(s_small, 0.5, principal=100_000.0)
        if abs(need - 150_000.0) > 1.0:
            raise RuntimeError(f"一手门槛算错了：{need:,.0f} ≠ 150,000")
        # ① 解析口径：门槛只跟价格和 k 有关，**不许跟着成交手数变**
        for _lots in (0.2, 1.0, 9.9):
            s_small.avg_lots = _lots
            _n, _ = PL.lot_floor(s_small, 0.5, principal=100_000.0)
            if abs(_n - 150_000.0) > 1e-6:
                raise RuntimeError(
                    f"门槛跟着成交手数变了（avg_lots={_lots} → {_n:,.0f}）——"
                    "还在拿实测杠杆反推")
        s_small.avg_lots = 1.0
        # ② 动用比例 k 真的参与：pct=50% ⇒ 同样一手要两倍本金
        s_half = mk_lot("P", 100_000, 0.30, 100_000, 200_000, pct=50.0)
        need_half, _ = PL.lot_floor(s_half, 0.5, principal=100_000.0)
        if abs(need_half - 300_000.0) > 1.0:
            raise RuntimeError(f"pct=50% 的门槛该翻倍：{need_half:,.0f} ≠ 300,000")
        # ③ unit 口径的 k = 保证金率：10% ⇒ 15 万 ÷ 10% = 150 万
        s_unit = mk_lot("U", 100_000, 0.30, 100_000, 200_000, sizing="unit", mr=0.1)
        need_unit, _ = PL.lot_floor(s_unit, 0.5, principal=100_000.0)
        if abs(need_unit - 1_500_000.0) > 10.0:
            raise RuntimeError(f"unit 口径门槛该是 15 万 ÷ 10%：{need_unit:,.0f}")
        # ④ ratio=0.5 的不变量：最贵那一根也 ≥ 0.5 手（四舍五入成 1 手）。
        #    ratio 被调小就保不住「任何时点都开得出」，这条钉住那个配比
        for _la, _lp in ((100_000, 200_000), (29_724, 77_584), (1_000, 50_000)):
            _lr = _la + PL.LOT_LIFT_RATIO * (_lp - _la)
            if _lr / _lp < 0.5 - 1e-12:
                raise RuntimeError(
                    f"最贵那一根只够 {_lr / _lp:.2f} 手（<0.5 就是 0 手）")
        # ⑤ 等权 20 万 ⇒ 各 10 万；S 只够 10万 ÷ 15万 = 0.67 手 ⇒ 抬到 15 万
        sh10 = PL.allocate(lot_us, 200_000.0, "equal", basis="principal")
        if abs(sh10[0].new_principal - 150_000.0) > 1.0:
            raise RuntimeError(
                f"开不出 1 手的没抬到 15 万：{sh10[0].new_principal:,.0f}")
        if not sh10[0].lifted or abs(sh10[0].lift_extra - 50_000.0) > 1.0:
            raise RuntimeError(f"上调额记错了：{sh10[0].lift_extra!r}")
        # ⑥ 别人一分不让：权重不变，本金还是原来的
        if abs(sh10[1].new_principal - 100_000.0) > 1e-6:
            raise RuntimeError(
                f"一手保底不该动别人的本金：{sh10[1].new_principal:,.0f}")
        if abs(sh10[0].weight - sh10[1].weight) > 1e-12:
            raise RuntimeError("一手保底把权重也改了 —— 它只该加本金")
        if abs(sum(s.new_principal for s in sh10) - 250_000.0) > 1.0:
            raise RuntimeError("本金合计该是 20 万 + 5 万")
        # ⑦ 已经够一手的条目不许抬（否则等于凭空给它加钱）
        if sh10[1].lifted or sh10[1].lot_need > 0:
            raise RuntimeError("够一手的条目被抬了")
        # ⑧ 上限：单条**上调额**不超过总资金的 LOT_LIFT_CAP_SHARE，到顶就停并写明
        s_huge = mk_lot("H", 100_000, 0.30, 10_000_000, 10_000_000, 0.01)
        sh11 = PL.allocate([s_huge, mk_lot("Q", 100_000, 0.20, 10_000, 10_000, 8.0)],
                           300_000.0, "equal", basis="principal")
        room = 300_000.0 * PL.LOT_LIFT_CAP_SHARE
        if abs(sh11[0].lift_extra - room) > 1.0:
            raise RuntimeError(
                f"上调额没顶在 {room:,.0f}：{sh11[0].lift_extra:,.0f}")
        if "仍不足一手" not in (sh11[0].floor_note or ""):
            raise RuntimeError(
                f"顶到上限却没说「仍不足一手」：{sh11[0].floor_note!r}")
        # ⑨ 自带 1 手保底的口径（budget / risk）：门槛是**峰值占用**，不是「够
        #    一手」。这两个口径手数被 `max(1, …)` 兜住 —— 本金砍到 1 万它照样
        #    下 1 手、照样冻 3.9 万保证金（实测 lc-sbbox 是 327% 本金），所以
        #    本金低于峰值占用 = 隐性超杠杆，实盘还会被拒单。
        b_risk = mk_lot("K", 100_000, 0.30, 40_000, 60_000, 0.0, sizing="risk")
        b_risk.peak_margin = 38_911.0
        b_risk.meas_peak_margin = 38_911.0     # 量测时的峰值占用
        # ⑨a 分到的钱 < 峰值占用 ⇒ 抬到峰值占用，差额加总本金
        sh_risk = PL.allocate([b_risk, mk_lot("Q2", 100_000, 0.20, 10_000, 10_000, 8.0)],
                              60_000.0, "equal", basis="principal")
        if abs(sh_risk[0].new_principal - 38_911.0) > 1.0:
            raise RuntimeError(
                f"risk 口径没抬到峰值占用：{sh_risk[0].new_principal:,.0f}")
        if abs(sh_risk[0].lift_extra - 8_911.0) > 1.0:
            raise RuntimeError(f"risk 上调额记错了：{sh_risk[0].lift_extra!r}")
        if sh_risk[0].lift_kind != "peak":
            raise RuntimeError(f"门槛种类记错了：{sh_risk[0].lift_kind!r}")
        if abs(sh_risk[1].new_principal - 30_000.0) > 1e-6:
            raise RuntimeError("risk 保底把钱挪给别人了")
        # ⑨b 本金已覆盖峰值占用 ⇒ 不抬，但说明里要写清它自带 1 手保底
        b_rich = mk_lot("K2", 100_000, 0.30, 40_000, 60_000, 0.0, sizing="risk")
        b_rich.peak_margin = 9_000.0
        b_rich.meas_peak_margin = 9_000.0
        sh_r2 = PL.allocate([b_rich, mk_lot("Q3", 100_000, 0.20, 10_000, 10_000, 8.0)],
                            60_000.0, "equal", basis="principal")
        if sh_r2[0].lifted:
            raise RuntimeError("本金已覆盖峰值占用，不该再抬")
        if "保底" not in (sh_r2[0].floor_note or ""):
            raise RuntimeError(f"没写清为什么没抬：{sh_r2[0].floor_note!r}")
        # ⑨c 🔴 rebase 不许折 `meas_peak_margin`：折算是按「占用 ∝ 本金」算的，
        #     而这两口径在「小到只剩 1 手」那段占用是**常数**。折一下门槛就没了
        #     （11,898 那条会从 38,911 折成 4,883，判定够了却仍是 3 倍杠杆）
        shrunk = PL.rebase(b_risk, 12_000.0, 10.0, capital=1_200.0, sizing="risk")
        if abs(shrunk.peak_margin - 38_911.0 * 0.12) > 1.0:
            raise RuntimeError("rebase 该把 peak_margin 折到当前本金口径")
        if abs(shrunk.meas_peak_margin - 38_911.0) > 1e-6:
            raise RuntimeError(
                f"rebase 把量测峰值占了折掉了：{shrunk.meas_peak_margin:,.0f}"
                " —— 门槛会被折没")
        _n2, _ = PL.lot_floor(shrunk, 0.5, principal=12_000.0)
        if abs(_n2 - 38_911.0) > 1.0:
            raise RuntimeError(f"rebase 之后门槛跟着缩了：{_n2:,.0f}")
        # ⑩ 关掉开关就一条都不抬（给「我就是要按权重硬分」的人留后路）
        off = PL.allocate(lot_us, 200_000.0, "equal", basis="principal", lift=False)
        if any(s.lifted for s in off):
            raise RuntimeError("lift=False 还在抬")
        if abs(off[0].new_principal - 100_000.0) > 1e-6:
            raise RuntimeError("lift=False 的本金被改了")

        return ("ERC 风险贡献等权（n=1/2/5/12，误差 <1e-6）；"
                "按本金/按平均占用/按峰值三种口径；固定手数不缩放；手动按比例；"
                "风险平价挡住 NaN（老记录/取不到行情不连累全体）与 vol=0（不吞掉全部资金）；"
                "标准化风险＝波动÷利用率×时长系数（利用率高/回测长/"
                f"盈利因子≥{PL.PF_FLOOR:g} 拿更多，低于 {PL.PF_FLOOR:g} 或不足半年"
                "落最低档但不清零；亏穿不再额外降档 —— 波动率含翻负段按真值配权、"
                "杠杆由利用率体现）；"
                f"一手保底（开不出 1 手的抬到「均价~峰值」中间那一手，差额只加总本金、"
                f"不动别人；单条上调额上限 {PL.LOT_LIFT_CAP_SHARE:.0%} 总资金，到顶写明仍不足；"
                "budget/risk 自带保底 1 手不抬）；"
                "实测占用按当前配置折算（应用分配后本金(前)是真值、再折不自乘）")

    check("组合工作区资金分配（ERC / 三种口径）", _portfolio_plan)

    def _lots_same_source():
        """实盘和回测的手数必须**同一个式子、同一张表**。

        以前实盘读天勤合约上的 `quote.margin` / `quote.volume_multiple`，回测读
        「数据管理 → 品种信息」。两边天生对不上，而实盘那一侧还有一串不确定性：
        主连（`KQ.m@`）压根没有 `margin`（恒 0）、模拟/快照环境不推这个字段、
        真合约的 `margin` 按当日结算价天天漂。任一条命中就掉进「每手保证金 = 0
        → 保底 1 手」，**不报错**。实测同一条 T 策略一天之内手数在 1~4 手之间跳，
        回测却始终 4~5 手 —— 用户没法解释也没法对账。

        所以这里钉三件事：
        ① 同一张表 + 同一个价格 ⇒ 两边手数一模一样（含 `budget` 口径）；
        ② 表里改保证金率，两边**同步**变（不是只变一边）；
        ③ 算手数时**不许再去问天勤行情**（用会抛异常的子类反证）。
        """
        from types import SimpleNamespace as NS

        from ttrade import backtest as BT
        from ttrade import replay, symbols_info as SI
        from ttrade import strategy_runner as SR

        real_info = SI.info
        table = {"T": {"mult": 10000.0, "margin": 0.02, "tick": 0.005,
                       "fee_mode": "lot", "fee_value": 3.0}}
        try:
            SI.info = lambda sym, ex="": dict(
                table.get(str(sym).upper()) or {"mult": 1.0, "margin": 0.03,
                                                "tick": 1.0, "fee_mode": "rate",
                                                "fee_value": 0.0001})

            class NoTq(SR.PortfolioRunner):
                """算手数时一旦回头去问天勤行情就炸 —— 这正是要禁掉的那条路。"""

                def _contract_quote(self, *a, **k):   # noqa: D102
                    raise AssertionError("算手数时又去问天勤 quotes 了")

            inst = NoTq.__new__(NoTq)

            def mk(**kw):
                base = dict(symbol="T", exchange="CFFEX", sizing="capital",
                            capital=105_033.4, principal=104_407.0,
                            capital_pct=100.6, volume=1, stop_mult=5.0,
                            risk_amount=20_000.0, note="", cost_profile=None,
                            short_only=False, factor_cfg={})
                base.update(kw)
                return NS(**base)

            px = 110.065

            def live_n(cfg, price=px):
                return NoTq._size(inst, NS(cfg=cfg, lots=0, underlying="CFFEX.T2603"),
                                  price, 0.0)

            # ① 同一张表 ⇒ 两边一样（取整都走 lots_round：四舍五入）
            cfg = mk()
            bt = replay.make_bt("cn", False, cfg)
            raw = 105_033.4 / (px * 10_000.0 * 0.02)
            want = BT.lots_round(raw)          # 4.77 手 → 5 手
            n_bt = BT.size_lots(bt, px, bt.initial_cash)
            n_live, why = live_n(cfg)
            if abs(n_bt - want) > 1e-9 or n_live != want:
                raise RuntimeError(
                    f"同表不同步：回测 {n_bt} 手、实盘 {n_live} 手，按表应为 {want} 手")
            if "品种信息表" not in why:
                raise RuntimeError(f"实盘手数没说明来源：{why}")

            # ② 表里改保证金率，两边必须同步变（只变一边就是又分叉了）
            table["T"]["margin"] = 0.05
            bt5 = replay.make_bt("cn", False, cfg)
            n_bt5 = BT.size_lots(bt5, px, bt5.initial_cash)
            n_live5, _ = live_n(cfg)
            if n_bt5 != n_live5 or n_live5 == n_live:
                raise RuntimeError(
                    f"改表后两边没同步：回测 {n_bt5} 手、实盘 {n_live5} 手"
                    f"（原来是 {n_live} 手）")

            # ③ 表里没填 → 两边都用同一个兜底率，并且说明里点名
            table["T"]["margin"] = 0.0
            bt0 = replay.make_bt("cn", False, cfg)
            n_bt0 = BT.size_lots(bt0, px, bt0.initial_cash)
            n_live0, why0 = live_n(cfg)
            if n_bt0 != n_live0:
                raise RuntimeError(
                    f"保证金率缺失时兜底不一致：回测 {n_bt0} 手、实盘 {n_live0} 手")
            if "兜底" not in why0:
                raise RuntimeError(f"兜底没有在日志里点名：{why0}")

            # ④ 乘数没维护：实盘**保底 1 手**（宁可少开，不能凭空放几千手），
            #    并且明确写出来；回测那边按 1 算出来的数不可信，不该被抄过来
            table["T"]["mult"] = 0.0
            n_live_m, why_m = live_n(cfg)
            if n_live_m != 1 or "乘数没维护" not in why_m:
                raise RuntimeError(f"乘数缺失时实盘没守住 1 手：{n_live_m} 手（{why_m}）")

            # ⑤ `budget` 名义口径也同源
            table["T"].update({"mult": 10000.0, "margin": 0.02})
            cfg_b = mk(sizing="budget", capital=1_000_000.0)
            bt_b = replay.make_bt("cn", False, cfg_b)
            n_bt_b = BT.size_lots(bt_b, px, bt_b.initial_cash)
            n_live_b, _ = live_n(cfg_b)
            if abs(n_bt_b - n_live_b) > 1e-9:
                raise RuntimeError(
                    f"budget 口径两边不一致：回测 {n_bt_b} 手、实盘 {n_live_b} 手")

            # ⑥ 取整口径必须是**四舍五入**。三个对照组都不能错：
            #    截断会系统性少配半手（4.77→4，只用掉 77% 预算，且永远偏少不抵消）；
            #    Python 的 `round()` 是银行家舍入（2.5→2、3.5→4），在 .5 上
            #    有时进有时不进，用户拿「四舍五入」四个字没法预测
            for x, exp in ((4.77, 5), (4.4, 4), (2.5, 3), (3.5, 4), (0.49, 0)):
                got = BT.lots_round(x)
                if got != exp:
                    raise RuntimeError(
                        f"手数取整 {x} → {got}，应为 {exp}（四舍五入，不是截断/"
                        f"银行家舍入）")
            per = px * 10_000.0 * 0.02
            # (手数理论值, 回测应为, 实盘应为)。
            # 🔴 0.6 那一组**两边必须一样给 1 手**：`lots_round(0.6)==1`，
            # 0.5 就是四舍五入的分界。以前回测在 `raw < 1` 时直接判 0 手
            # （外面多了一层 `if raw >= 1` 的门槛），把分界线挪到了 1.0 手，
            # 于是「够 0.6 手」的策略在回测里整段空仓、实盘却会下 1 手 ——
            # 用户当场追问「够 0.5 手为什么不算 1 手」，就是这条。
            # 真正会分叉的只有 <0.5 手：回测 0 手（四舍五入到 0），
            # 实盘保底 1 手（0 手等于策略静默不动）。
            for raw_n, exp_bt, exp_live in ((4.77, 5, 5), (1.4, 1, 1),
                                            (0.6, 1, 1), (0.45, 0, 1)):
                c = mk(capital=per * raw_n)
                b = replay.make_bt("cn", False, c)
                g_bt = BT.size_lots(b, px, b.initial_cash)
                g_live, _ = live_n(c)
                if abs(g_bt - exp_bt) > 1e-9 or g_live != exp_live:
                    raise RuntimeError(
                        f"预算 {raw_n:.2f} 手（{per * raw_n:,.0f} 元）：回测给 "
                        f"{g_bt} 手（应 {exp_bt}）、实盘给 {g_live} 手（应 {exp_live}）")

            # ⑦ **公式只有一份**：`size_lots` 是 `size_plan` 的薄包装，实盘 `_size`
            #    也调 `size_plan(integer=True)`。上面 ①~⑥ 全是「两份实现、对拍数字」，
            #    改一处忘另一处只会在某个口径、某个价位上分叉（0.45 手那条线就是），
            #    连对拍都可能覆盖不到。这里直接钉住实现：源码里不许再出现取整公式。
            import inspect

            for name in ("_size", "_entry_lots_at", "_ib_arm"):
                src = inspect.getsource(getattr(SR.PortfolioRunner, name))
                body = "\n".join(ln for ln in src.splitlines()
                                 if not ln.lstrip().startswith("#"))
                if "lots_round(" in body:
                    raise RuntimeError(
                        f"实盘 `{name}` 又自己算手数了（源码里出现 lots_round）—— "
                        "手数公式只许有 `backtest.size_plan` 一份")
            for m in ("capital", "budget", "risk", "unit", "fixed"):
                for atr_v in (0.0, 1.7, 12.3):
                    c = mk(sizing=m, capital=per * 4.77)
                    b = replay.make_bt("cn", False, c)
                    n_i = int(BT.size_plan(b, px, 0.0, atr_v, integer=True)[0])
                    g_i, _w = live_n(c)
                    if g_i != n_i:
                        raise RuntimeError(
                            f"{m} 口径：实盘 {g_i} 手 ≠ size_plan(integer=True) {n_i} 手")
                    if abs(BT.size_lots(b, px, b.initial_cash, atr_v)
                           - BT.size_plan(b, px, b.initial_cash, atr_v)[0]) > 1e-9:
                        raise RuntimeError("`size_lots` 不再是 `size_plan` 的薄包装")
        finally:
            SI.info = real_info
        return ("**手数公式只有一份**（`backtest.size_plan`）：`size_lots` 是它的"
                "薄包装，实盘 `_size` / `_entry_lots_at` / `_ib_arm` 都调它"
                "（`integer=True`），源码里不许再出现取整公式；"
                "同一张品种信息表 ⇒ 同输入同手数（capital / budget）；"
                "改保证金率两边同步变；表里缺失时兜底一致；乘数缺失实盘守 1 手；"
                "算手数全程不碰天勤行情；"
                "取整一律四舍五入（4.77→5、2.5→3，不是截断也不是银行家舍入），"
                "0.6 手两边都给 1 手（分界点是 0.5，不是 1.0）；"
                "只有 <0.5 手才分叉：回测 0 手、实盘保底 1 手（`integer=True`）")

    check("实盘与回测手数同源（品种信息表）", _lots_same_source)

    def _occupancy_diag():
        """绩效页的「资金占用」三列：平均开仓占用 / 峰值占用 / 利用率。

        这一层也是「不报错但数错」的类型：
        - 占用忘了乘合约乘数 / 忘了乘保证金率
        - **平均用逐根 bar 算而不是逐笔开仓算** —— 这是最隐蔽的一个，算出来的
          数「看起来合理」（本金的百分之几），但量的根本不是「出手时用多少钱」，
          而且会随空仓时间长短变化（同一套仓位，交易越稀疏这个数越小）
        - 组合层把各成分峰值直接相加当成真实峰值

        全部用**能手算的假数据**钉死。
        """
        import numpy as np
        import pandas as pd

        from ttrade import backtest as BT

        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        close = pd.Series([100.0, 200.0, 300.0, 400.0], index=idx)
        pos = pd.Series([0.0, 2.0, -1.0, 0.0], index=idx)
        eq = pd.Series([100_000.0, 100_500.0, 100_800.0, 101_000.0], index=idx)

        def tr(lots, px):
            return BT.Trade(entry_date=idx[0], exit_date=idx[1],
                            entry_price=float(px), exit_price=float(px),
                            ret=0.0, bars=1, lots=float(lots))

        # ① 平均 = **逐笔开仓**占用的均值，乘数 10、保证金率 0.1
        #    2 手 @200 → 2×200×10×0.1 = 400
        #    1 手 @300 → 300
        #    0.5 手 @400 → 200
        #    平均 = 300、平均名义 = 3000
        trades = [tr(2, 200), tr(1, 300), tr(0.5, 400)]
        m = BT._metrics(eq, pos, trades, base=100_000.0,
                        price=close, mult=10.0, margin_rate=0.1)
        for k, want in (("avg_margin", 300.0), ("avg_notional", 3000.0)):
            if abs(float(m[k]) - want) > 1e-9:
                raise RuntimeError(f"{k} 应为 {want}，实际 {m[k]}")
        if abs(float(m["util"]) - 300.0 / 100_000.0) > 1e-15:
            raise RuntimeError(f"利用率应为 300/100000，实际 {m['util']}")

        # ①b **不能是逐根均值**：逐根算出来是 175（[0,4000,3000,0]×0.1 的均值），
        #     和逐笔的 300 差得远。这条断言就是防回退。
        if abs(float(m["avg_margin"]) - 175.0) < 1e-9:
            raise RuntimeError("平均占用退化成了逐根 bar 的均值")

        # ①c 交易越稠密不该改变「出手时用多少」：再插 3 笔同样大小的开仓，
        #     平均还是 300 —— 逐根均值会跟着空仓比例漂
        dense = trades + [tr(2, 200), tr(2, 200), tr(2, 200)]
        m2 = BT._metrics(eq, pos, dense, base=100_000.0,
                         price=close, mult=10.0, margin_rate=0.1)
        want = (400 * 4 + 300 + 200) / 6.0
        if abs(float(m2["avg_margin"]) - want) > 1e-9:
            raise RuntimeError(f"加了笔数后平均值应为 {want}，实际 {m2['avg_margin']}")

        # ② 峰值走**逐根**，而且必须按**真实手数**算。
        #
        #    🔴 `position` 只是方向（±1），不是手数。以前的实现拿 `|pos| × 价格 ×
        #    乘数` 当峰值名义，等于按「1 手」算 —— 手数大于 1 时峰值被系统性
        #    低估，还会出现「平均占用 > 峰值占用」。这条用例就是钉死这个口径：
        #    5 手拿着，峰值名义必须是 5×400×10=20000，不是 4000。
        pos_dir = pd.Series([0.0, 1.0, 1.0, 0.0], index=idx)
        big = [BT.Trade(entry_date=idx[0], exit_date=idx[3], entry_price=200.0,
                        exit_price=300.0, ret=0.0, bars=3, lots=5.0)]
        mb = BT._metrics(eq, pos_dir, big, base=100_000.0,
                         price=close, mult=10.0, margin_rate=0.1)
        for k, want in (("peak_notional", 20_000.0), ("peak_margin", 2_000.0)):
            if abs(float(mb[k]) - want) > 1e-9:
                raise RuntimeError(
                    f"{k} 应为 {want}，实际 {mb[k]} —— 峰值没按真实手数算")
        if float(mb["avg_margin"]) >= float(mb["peak_margin"]):
            raise RuntimeError("平均开仓占用不该大于等于峰值占用")

        # ②b 多笔重叠时手数要**相加**（多空两边都占保证金），不是取最大
        two = [BT.Trade(entry_date=idx[0], exit_date=idx[3], entry_price=200.0,
                        exit_price=300.0, ret=0.0, bars=3, lots=3.0,
                        direction=1),
               BT.Trade(entry_date=idx[0], exit_date=idx[3], entry_price=200.0,
                        exit_price=300.0, ret=0.0, bars=3, lots=2.0,
                        direction=-1)]
        mt = BT._metrics(eq, pos_dir, two, base=100_000.0,
                         price=close, mult=10.0, margin_rate=0.1)
        if abs(float(mt["peak_notional"]) - 20_000.0) > 1e-9:
            raise RuntimeError(
                f"两条腿重叠时峰值名义应为 (3+2)×400×10=20000，实际 {mt['peak_notional']}")

        # ②c 交易记录不带手数（老结果 / 外部构造）时退回方向序列，不能变成 0
        bare = [BT.Trade(entry_date=idx[0], exit_date=idx[1], entry_price=200.0,
                         exit_price=200.0, ret=0.0, bars=1)]
        mbr = BT._metrics(eq, pos_dir, bare, base=100_000.0,
                          price=close, mult=10.0, margin_rate=0.1)
        if float(mbr["peak_notional"]) <= 0:
            raise RuntimeError("没有手数信息时峰值占用不该归零")

        # ③ 平均**不依赖行情序列**（只要交易记录），峰值才依赖价格。
        #    组合层 / 老结果没传 price 时，平均值该照样给出来
        m0 = BT._metrics(eq, pos, trades, base=100_000.0, mult=10.0, margin_rate=0.1)
        if abs(float(m0["avg_margin"]) - 300.0) > 1e-9:
            raise RuntimeError("没传价格时平均开仓占用被清零了 —— 它只需要交易记录")
        if float(m0["peak_margin"]) != 0.0:
            raise RuntimeError("没传价格时峰值占用必须是 0，不能凭空给数")

        # ④ 没开过仓 / 保证金率为 0 → 必须是 0，不能凭空给数（除零陷阱）
        for kw in ({"price": close, "mult": 10.0, "margin_rate": 0.1},
                   {"price": close, "mult": 10.0, "margin_rate": 0.0}):
            mz = BT._metrics(eq, pos, [], base=100_000.0, **kw)
            if any(float(mz[k]) != 0.0 for k in
                   ("avg_notional", "avg_margin", "util")):
                raise RuntimeError("没开过仓 / 保证金率为 0 时占用必须是 0")

        # ⑤ 键必须进 `_METRIC_KEYS`：老报告缺这一项时按 0 补齐，
        #    否则绩效页 `m.get(...)` 拿到 None 会写成「—」而不是 0
        for k in ("avg_notional", "peak_notional", "avg_margin", "peak_margin", "util",
                  "blown"):
            if k not in BT._METRIC_KEYS:
                raise RuntimeError(f"{k} 没进 _METRIC_KEYS")

        # ⑤b 权益亏穿时**波动率照算**（3.6.2 按用户要求改）。
        #     用户原话：「波动率要覆盖整个过程资金曲线，亏穿了要继续跑完得到
        #     完整数据」「杠杆比率低，也亏不穿，这就是为什么要用资金利用率」。
        #     所以：vol 必须是整条曲线（含翻负段）上的真值；夏普 / 卡玛仍标
        #     NaN（净值翻号之后的「反弹」不是收益）；`blown` 单独留痕。
        #     🔴 以前三个一起标 NaN，亏穿的策略在配权里拿不到自己的波动率，
        #     只能被「组合典型风险」顶替 —— 表现就是「波动很大、分数却不高」。
        neg = pd.Series([100_000.0, 60_000.0, -30_000.0, 20_000.0],
                        index=idx, dtype=float)
        mn = BT._metrics(neg, pos, [], base=100_000.0)
        if not np.isfinite(float(mn["ann_vol"])) or float(mn["ann_vol"]) <= 0:
            raise RuntimeError(f"亏穿时波动率被抹掉了：{mn['ann_vol']}")
        # 手算：逐根盈亏 ÷ 本金 = [0, −0.40, −0.90, +0.50]
        want_v = float(np.std([0.0, -0.40, -0.90, 0.50], ddof=1)
                       * np.sqrt(float(BT.TRADING_DAYS)))
        if abs(float(mn["ann_vol"]) - want_v) > 1e-9:
            raise RuntimeError(
                f"亏穿时波动率不是整条曲线的真值：{mn['ann_vol']:.6f} ≠ {want_v:.6f}")
        if not (np.isnan(float(mn["sharpe"])) and np.isnan(float(mn["calmar"]))):
            raise RuntimeError("亏穿时夏普 / 卡玛必须仍是 NaN")
        if float(mn["blown"]) != 1.0:
            raise RuntimeError("亏穿时 `blown` 标记没置上")
        if float(mt["blown"]) != 0.0:
            raise RuntimeError("没亏穿的策略 `blown` 必须是 0")

        # ⑤ 组合层：平均占用**可加**（精确），峰值只能给**上限**
        #    成分 a：本金 100、平均占用 10、峰值 50
        #    成分 b：本金 200、平均占用 30、峰值 70
        #    组合 300 万等权 → 平均 = (10/100 + 30/200)/2 × 300 = 37.5
        #                        峰值上限 = (50/100 + 70/200)/2 × 300 = 127.5
        idx2 = pd.date_range("2024-01-01", periods=4, freq="D")
        flat = pd.Series(0.0, index=idx2)

        def mk(series, am, pm):
            return BT.Result(equity=pd.Series(series, index=idx2, dtype=float),
                             position=flat, trades=[],
                             metrics={"avg_margin": am, "peak_margin": pm,
                                      "total_cost": 0.0, "notional": 0.0},
                             label="x")

        r1 = mk([100.0, 101, 102, 103], 10.0, 50.0)
        r2 = mk([200.0, 202, 204, 206], 30.0, 70.0)
        pc = BT.combine([r1, r2], initial_cash=300.0)
        want = (10.0 / 100.0 + 30.0 / 200.0) / 2 * 300.0
        want_pk = (50.0 / 100.0 + 70.0 / 200.0) / 2 * 300.0
        if abs(float(pc.metrics["avg_margin"]) - want) > 1e-9:
            raise RuntimeError(
                f"组合平均占用应为 {want}，实际 {pc.metrics['avg_margin']}")
        if abs(float(pc.metrics["peak_margin"]) - want_pk) > 1e-9:
            raise RuntimeError(
                f"组合峰值占用应为 {want_pk}，实际 {pc.metrics['peak_margin']}")
        if abs(float(pc.metrics["util"]) - want / 300.0) > 1e-12:
            raise RuntimeError("组合利用率的分母不是组合总资金")
        if float(pc.metrics.get("peak_is_bound") or 0) != 1.0:
            raise RuntimeError("组合峰值没有标成「上限」，界面会把它当精确值")
        # 上限性质：峰值之和必须 ≥ 各成分峰值（真峰值在两者之间）
        if float(pc.metrics["peak_margin"]) < max(50.0, 70.0):
            raise RuntimeError("组合峰值上限低于单个成分的峰值，不可能")

    check("资金占用诊断（平均/峰值/利用率）", _occupancy_diag)

    def _workspace_resolve():
        """工作区解析：本金用工作区的、sid 不变、原对象不被改、多账户展开。

        这几条都是「不报错但结果错」：`clone()` 会换 sid（写回策略池时就写到
        另一条上了）、忘了还原就是把工作区的钱写进策略池的默认值、`accounts`
        写成字符串会被按字符遍历成三个账户。
        """
        from ttrade import strategy_store as SStore
        from ttrade import workspace_store as WS

        def cfg(sid, principal, acc=""):
            c = SStore.StrategyConfig(sid=sid, name=sid, symbol="RB", period="1d",
                                      rule_key="ma_cross", principal=principal)
            c.account = acc
            if acc:
                c.accounts = [acc]
            return c

        pool = [cfg("a", 100_000.0, "sim"), cfg("b", 1_000_000.0)]
        w = WS.new_workspace("t", total=0.0)
        w.add("a", 400_000.0)
        w.add("zzz", 50_000.0)              # 池子里没有 → 应当报 missing

        cfgs, missing = WS.resolve(w, pool)
        if [c.sid for c in cfgs] != ["a"]:
            raise RuntimeError(f"解析出的 sid 不对：{[c.sid for c in cfgs]}")
        if missing != ["zzz"]:
            raise RuntimeError(f"找不到的 sid 没报出来：{missing}")
        if abs(float(cfgs[0].principal) - 400_000.0) > 1e-9:
            raise RuntimeError(
                f"本金用的是策略池的 {cfgs[0].principal}，不是工作区的 400000")
        if float(pool[0].principal) != 100_000.0:
            raise RuntimeError("解析把策略池里那份本金改了 —— 必须只改副本")
        if cfgs[0] is pool[0]:
            raise RuntimeError("解析返回的是原对象，不是副本")

        # 成员本金为 0（没分配过）时退回策略自己那份，不能变成 0 本金
        w2 = WS.new_workspace("t2")
        w2.add("a", 0.0)
        c2, _m = WS.resolve(w2, pool)
        if abs(float(c2[0].principal) - 100_000.0) > 1e-9:
            raise RuntimeError("成员本金为 0 时没有退回策略自己的本金")

        # ---- 多账户 ----
        m = cfg("m", 100_000.0)
        m.accounts = ["sim", "real", "sim", ""]        # 重复 + 空名要消毒
        m = SStore.StrategyConfig.from_dict(m.to_dict())
        if m.accounts != ["sim", "real"]:
            raise RuntimeError(f"accounts 没去重/去空：{m.accounts}")
        if m.account != "sim":
            raise RuntimeError(f"account 没跟 accounts[0] 同步：{m.account}")
        if m.account_list("默认") != ["sim", "real"]:
            raise RuntimeError(f"account_list 展开不对：{m.account_list('默认')}")
        # 空 = 跟随默认账户
        m2 = SStore.StrategyConfig.from_dict({"sid": "n", "symbol": "RB"})
        m2.account, m2.accounts = "", []
        if m2.account_list("默认") != ["默认"]:
            raise RuntimeError("没绑账户时应当跟随默认账户")
        # 手写 JSON 把 accounts 写成字符串：不能按字符拆成 s/i/m
        m3 = SStore.StrategyConfig.from_dict({"sid": "s", "symbol": "RB",
                                              "accounts": "sim"})
        if m3.accounts != ["sim"]:
            raise RuntimeError(f"字符串 accounts 被拆成了 {m3.accounts}")

    check("组合工作区解析（本金/sid/多账户）", _workspace_resolve)

    def _macro_pit():
        """宏观因子的「只用 T-1 及更早」这条（5m / 1h 不前视）。

        ## 为什么单列一条自检

        宏观序列是**日频**、按**发布日**入库的。5m 一根 bar 是 5 分钟，
        如果不做滞后，当天盘的每一根都能看到当天盘后才知道的数字 —— 这类
        前视不会报错、不会让回测崩，只会把结果抬得好看，且**只在分钟周期
        上出现**（日线上看不出来）。所以必须有一个人守着的断言。

        验证分三半：
          ① `_asof(lag_days=1)` 把 D 的值推到 D+1 才可见
          ② 反证：同一份数据用 `lag_days=0` 时 D 当天就能看到 —— 说明这个
             断言真的能抓到前视，而不是恒真
          ③ 扩张分位（`_pct_series`）只用「≤t」的值：把序列砍到第 200 天，
             前 200 天的分位必须一模一样；反证「全样本分位」在这种砍法下一定
             不等 —— 否则这条也是恒真的空断言
        """
        import numpy as np
        import pandas as pd

        from ttrade import factors_macro as FM

        D = pd.Timestamp("2026-10-21")
        s = pd.Series([42.0], index=pd.DatetimeIndex([D]))
        bars = pd.date_range("2026-10-20 09:00", "2026-10-22 15:00", freq="5min")
        idx = pd.DatetimeIndex(bars)

        got = FM._asof(s, idx, lag_days=1)
        by_day = pd.DataFrame({"d": idx.normalize(), "v": got.to_numpy()}) \
            .groupby("d")["v"].last()
        if not pd.isna(by_day.loc[D]):
            raise RuntimeError(
                f"发布日当天就用到了当天的宏观值（前视）：{by_day.loc[D]}")
        if float(by_day.loc[pd.Timestamp("2026-10-22")]) != 42.0:
            raise RuntimeError("D 的值在 D+1 没露出来 —— 滞后过头或对齐错了")

        # 反证：lag_days=0 时 D 当天就该看到（否则上一条恒真，是假断言）
        raw = FM._asof(s, idx, lag_days=0)
        by_day0 = pd.DataFrame({"d": idx.normalize(), "v": raw.to_numpy()}) \
            .groupby("d")["v"].last()
        if pd.isna(by_day0.loc[D]):
            raise RuntimeError("lag_days=0 在当天也看不到 —— 这条断言抓不到前视")

        # 日内所有 bar 必须同值（宏观是日频，不该在一天里变）
        if pd.Series(got.to_numpy()[~np.isnan(got.to_numpy())]).nunique() > 1:
            raise RuntimeError("同一个发布值在日内出现了多个取值")

        n_codes = len(FM._LABEL)
        if n_codes < 20:
            raise RuntimeError(f"注册的宏观因子只有 {n_codes} 个，序列表被改小了")

        # ③ 扩张分位不能吃未来：砍掉后面 100 天，前 200 天的分位必须一模一样。
        #    这是宏观这条链路上**唯一**一处会碰到完整序列的地方 —— `_asof` 只管
        #    「哪天的值可见」，管不了「分位是怎么算的」。要是哪天有人把
        #    `expanding().rank()` 改成 `rank(pct=True)`（全样本分位），`_asof`
        #    那两条照样全绿，而因子已经把未来三年的分布信息灌进今天了。
        from ttrade import macro as M

        n = 300
        dd = pd.bdate_range("2024-01-01", periods=n)
        vals = pd.Series(np.linspace(10.0, 40.0, n)
                         + np.sin(np.arange(n) / 7.0), index=dd)
        orig = M.series
        try:
            M.series = lambda code: vals                         # noqa: ARG005
            FM.clear_cache()
            f_full = FM._pct_series("usdcnh")
            M.series = lambda code: vals[vals.index <= dd[199]]   # noqa: ARG005
            FM.clear_cache()
            f_half = FM._pct_series("usdcnh")
        finally:
            M.series = orig
            FM.clear_cache()
        a = f_full.reindex(dd[:200]).to_numpy(dtype=float)
        b = f_half.reindex(dd[:200]).to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True):
            raise RuntimeError(
                "砍掉第 200 天之后的宏观序列，前 200 天的分位就变了 —— "
                "扩张分位用到了未来数据")
        # 反证：全样本分位在同样的砍法下一定不等，否则上面那条是空断言
        fs = vals.rank(pct=True).reindex(dd[:200]).to_numpy(dtype=float)
        if np.allclose(a, fs, equal_nan=True):
            raise RuntimeError("扩张分位等于全样本分位 —— 这条断言抓不到东西")

        return (f"{n_codes} 个宏观因子；发布日当天零可见、D+1 可见；"
                f"反证 lag=0 立刻可见；扩张分位砍掉后 100 天后前 200 天不变"
                f"（且 ≠ 全样本分位）")

    check("宏观因子的发布日对齐（5m/1h 不前视）", _macro_pit)

    def _macro_avail_8():
        """日更宏观在凌晨只能看到 D-2：汇率两项 + 美债实际利率。

        用户 2026-11-19 钉死：Tushare 的日频汇率 **D+1 08:00 之后才更新**。
        商品夜盘是 21:00 ~ 次日 02:30，00:00 之后那一段如果还按「自然日 − 1」
        取数，拿到的是**当时还没发布**的那一行 —— 真正的未来函数，而且只在凌晨
        两三个小时里发作，日盘看不出来、日线上更看不出来。

        同日追加：`us_real10` 也要套。它的时区链比汇率更长 —— 纽约是
        **UTC−5（EST）/ UTC−4（EDT）**，财政部按美国交易日 D 在美东
        15:30~18:00 收完盘才发布 ⇒ 北京时间 **D+1 04:30~07:00**。
        所以 bar 次日 02:00 用它就是偷未来。

        验三件：
          ① 边界精确：D 07:59 仍是 D-2，到 D 08:00 才切到 D-1；
          ② 夜盘各时点：D 02:00 取 D-2、D 21:00 取 D-1，日盘取 D-1；
          ③ `_make()` 真的把参数接上了（三个 code 逐个验，不是只有 `_asof`
             支持、调用方却没传）。
        """
        import numpy as np
        import pandas as pd

        from ttrade import factors_macro as FM
        from ttrade import macro as M

        # 🔴 不写死名单（AVAIL_08 会随新序列扩，写死一次就得跟着改一次），
        #    改成两条结构约束 + 一条反证：
        #      ① 里面每个 code 都得真注册过，否则这条判据在管空气；
        #      ② 「收盘落在纽约时段」的一条都不能漏 —— 漏了凌晨就取到当天才收的行。
        unknown = set(FM.AVAIL_08) - set(FM._LABEL)
        if unknown:
            raise RuntimeError(f"AVAIL_08 里有没注册的 code：{sorted(unknown)}")
        for code in ("dxy", "usdcnh", "us_real10", "spx", "ixic", "spx_vol20"):
            if code not in FM.AVAIL_08:
                raise RuntimeError(
                    f"{code} 的收盘在纽约时段（北京次日 04:00~06:00 才成立），"
                    f"不在 AVAIL_08 里就会在夜盘凌晨用上当天才收的那一行")
        if any(FM._avail_hour(c) != 8 for c in FM.AVAIL_08):
            raise RuntimeError("AVAIL_08 里的 code 有 `_avail_hour` 不等于 8 的")
        # 反证：不在名单里的必须是 0，否则「加进名单」这个动作什么也没改变
        if FM._avail_hour("shibor_on") != 0:
            raise RuntimeError("不在 AVAIL_08 的序列也被当成了 08:00 才可用")

        d0, d1, d2 = (pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-20"),
                      pd.Timestamp("2026-10-21"))
        s = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex([d0, d1, d2]))

        def at(ts, avail):
            i = pd.DatetimeIndex([ts])
            return float(FM._asof(s, i, lag_days=1, avail_hour=avail).iloc[0])

        for ts, want, why in (
            (d2 + pd.Timedelta(hours=2), 1.0, "凌晨 02:00 必须取 D-2"),
            (d2 + pd.Timedelta(hours=7, minutes=59), 1.0, "07:59 仍是 D-2"),
            (d2 + pd.Timedelta(hours=8), 2.0, "08:00 必须切到 D-1"),
            (d2 + pd.Timedelta(hours=9, minutes=30), 2.0, "日盘取 D-1"),
            (d2 + pd.Timedelta(hours=21), 2.0, "夜盘 21:00 取 D-1"),
        ):
            got = at(ts, 8)
            if got != want:
                raise RuntimeError(f"{ts} 取到 {got}，应为 {want}（{why}）")

        # 反证：不传 8 这条，凌晨就会取到 D-1 —— 说明上面那几条不是恒真
        if at(d2 + pd.Timedelta(hours=2), 0) != 2.0:
            raise RuntimeError("avail_hour=0 时凌晨也取 D-2 —— 这条断言抓不到东西")

        # ③ `_make` 必须把这个参数传下去（参数加了、调用方不传，等于没改）
        #    三个 code 都要验：us_real10 是 2026-11-19 追加的（美债按美东交易日发布，
        #    UTC−5/−4 ⇒ 北京 D+1 04:30~07:00 才有，凌晨绝拿不到当天那行）。
        n = 300
        dd = pd.date_range("2025-01-01", periods=n, freq="D")
        vals = pd.Series(np.sin(np.arange(n) / 9.0) * 10 + 20, index=dd)
        orig = M.series
        got_by_code = {}
        try:
            M.series = lambda code: vals                          # noqa: ARG005
            FM.clear_cache()
            pct = FM._pct_series("usdcnh")
            probe = pd.DatetimeIndex([dd[250] + pd.Timedelta(hours=2),
                                      dd[250] + pd.Timedelta(hours=10)])
            for code in ("usdcnh", "dxy", "us_real10", "usdcnh_em", "dxy_em"):
                got_by_code[code] = FM._make(code)(
                    pd.DataFrame(index=probe)).to_numpy(float)
            # 期望值按**每个 code 自己的**滞后天数算：FXCM 外汇（usdcnh/dxy）标签
            # 比真实交易日早一天，总滞后是 2；美债走财政部，是 1（见 `_LABEL_LAG`）
            w8 = FM._asof(pct, probe, lag_days=FM._lag_days("usdcnh"),
                          avail_hour=8).to_numpy(float)
            w0 = FM._asof(pct, probe, lag_days=FM._lag_days("usdcnh"),
                          avail_hour=0).to_numpy(float)
        finally:
            M.series = orig
            FM.clear_cache()
        got = got_by_code["usdcnh"]
        if np.isnan(got).any():
            raise RuntimeError("探针落在分位的最小观测窗口里，这条断言测了个空")
        if not np.allclose(got, w8, equal_nan=True):
            raise RuntimeError("_make('usdcnh') 没把 avail_hour=8 / lag_days 传下去")
        if np.allclose(got, w0, equal_nan=True):
            raise RuntimeError("_make('usdcnh') 的结果与 avail_hour=0 相同 —— 参数没生效")
        for code in ("dxy", "us_real10", "usdcnh_em", "dxy_em"):
            g = got_by_code[code]
            want = FM._asof(pct, probe, lag_days=FM._lag_days(code),
                            avail_hour=8).to_numpy(float)
            if not np.allclose(g, want, equal_nan=True):
                raise RuntimeError(f"_make('{code}') 没把 avail_hour/lag_days 传下去")
        # 东财两条不带标签错位，滞后期必须比 Tushare 那两条少一天 ——
        # 这不是「差不多」，是它们存在的意义：同一时点能拿到更新鲜的一行。
        if FM._lag_days("usdcnh_em") >= FM._lag_days("usdcnh"):
            raise RuntimeError("东财口径的滞后没有比 Tushare 短，说明 _LABEL_LAG 传染过去了")

        return ("07:59→D-2 / 08:00→D-1 边界精确；凌晨 02:00 取 D-2、日盘与夜盘 "
                "21:00 取 D-1；反证 avail=0 会取到 D-1；_make 对 "
                "usdcnh/dxy/us_real10/usdcnh_em/dxy_em 五条都按各自的 lag_days 接通")

    check("汇率因子 08:00 前用自然日-2（夜盘凌晨不前视）", _macro_avail_8)

    def _macro_no_same_day():
        """**每根 bar 只能用到「严格早于它自然日」的那一行**，用真实行情扫一遍。

        2026-11-20 用户报：IM 的「美元兑离岸人民币」在 9-19 09:30 用到了 9-19
        当天的日频值。这种指控没法靠看图回答 —— 日频因子日内恒定，图上是一条
        台阶，看不出台阶是从哪一行搬来的。所以需要一条能直接输出「第 T 根取的
        是哪一行」的判据。

        对拍**不用 `_asof`**（那是被测实现）：换成 pandas 自己的 `Series.asof()`，
        按「时刻 − 8 小时（发布）− 1 天」独立算一遍期望值。两套实现给出的行
        不一样就是 bug —— 同源对拍（两边都调 merge_asof）会让错的地方一起错，
        这条项目里已经踩过（#145）。

        探针**必须带夜盘**：日盘 09:30 的 bar 在 avail_hour=0/8 两种口径下取到的
        都是 D−1（相差不到一天），只有 00:00~08:00 那段才分得开。CU 有夜盘，
        拿它当样本；顺带覆盖 IM 那种没有夜盘的品种。
        """
        from ttrade import factors_macro as FM
        from ttrade import tqdata as T
        import numpy as _np
        import pandas as _pd

        pd, np = _pd, _np                  # 本函数自己带依赖，别指望外层作用域

        pct = FM._pct_series("usdcnh")
        if pct is None or len(pct) < 200:
            # 🔴 宏观库（`edb.duckdb`）和天勤轻量库一样**不在 git 里**（几百 MB，
            # 由 `macro.py` 的定时任务从天勤 / Tushare / 东财抓）。CI 与全新安装
            # 本来就没有这份数据 —— 那是环境状态，不是代码缺陷。
            # v3.8.1 就是栽在这里：本地有库、扫得很漂亮，CI 上直接
            # 「usdcnh 序列读不到」，三个平台的自检全红，Release 发出一个**空壳**
            # （零资产，见 v3.8.1 的 release）。
            # 判据：库里**一条宏观数据都没有** ⇒ 跳过（没同步过）；库里有数据但
            # 唯独没有 usdcnh ⇒ 红（真出了事，比如序列名改了）。
            from ttrade import macro as _M
            st = _M.stats()
            if not st.get("exists") or not st.get("rows"):
                return ("跳过：宏观库 edb.duckdb 没有数据（它不在 git 里，CI / 全新"
                        "安装本来就没有）。本地跑过宏观同步之后这条才会真扫")
            raise RuntimeError(
                f"usdcnh 序列读不到，但库里有 {st.get('rows')} 行宏观数据"
                f"（{st.get('codes')} 个指标）—— 序列名或同步链路出问题了")
        idx = pd.DatetimeIndex(pct.index)

        def expect(t):
            """独立口径：发布时刻之后、再退 `_lag_days` 个自然日，取 ≤ 的那一行。"""
            return float(pct.asof(t - pd.Timedelta(hours=8)
                                  - pd.Timedelta(days=FM._lag_days("usdcnh"))))

        def used_day(t):
            pos = int(idx.searchsorted(pd.Timestamp(t) - pd.Timedelta(hours=8)
                                       - pd.Timedelta(days=FM._lag_days("usdcnh")),
                                       side="right")) - 1
            return None if pos < 0 else idx[pos]

        bad: list[str] = []
        n_probe = 0
        locked = ""
        for sym, ex in (("CU", "SHFE"), ("IM", "CFFEX")):
            try:
                df = T.load_main(sym, ex, "5m")
            except Exception as exc:  # noqa: BLE001 - 库没同步到就没有样本
                # DuckDB 单文件库是排他锁：界面开着的时候这里会打不开。
                # 那是环境状态，不是代码缺陷 —— 记下原因，别报成一条红
                locked = f"{type(exc).__name__}: {exc}"
                continue
            if df is None or len(df) < 500:
                continue
            tail = df.tail(4000)
            got = FM._make("usdcnh")(tail)
            night = tail.index[tail.index.hour < 8][:60]
            day = tail.index[(tail.index.hour == 9) & (tail.index.minute == 30)][:60]
            probes = list(night) + list(day)
            for t in probes:
                n_probe += 1
                a, b = float(got.loc[t]), expect(t)
                if np.isnan(a) and np.isnan(b):
                    continue
                if a != b:
                    bad.append(f"{sym} {t:%Y-%m-%d %H:%M} 取 {a:.6f}，应为 {b:.6f}")
                    continue
                # 命中的行必须比 bar 的自然日**至少老 `_lag_days` 天**：FXCM 那张表
                # 的行标签比真实交易日早一天，纽约日 D 的收盘写进 D−1 行、要到
                # 北京 D+1 06:00 才成立 ⇒ 标签加 2 天才是可用时刻。差一天就是前视
                u = used_day(t)
                need = pd.Timestamp(t).normalize() - pd.Timedelta(
                    days=FM._lag_days("usdcnh"))
                if u is not None and u > need:
                    bad.append(f"{sym} {t:%Y-%m-%d %H:%M} 命中行 {u:%Y-%m-%d} "
                               f"晚于可用边界 {need:%Y-%m-%d}")
        if n_probe == 0:
            if locked:
                return f"跳过：行情库被占用（{locked[:60]}），关掉界面再跑才是真扫"
            raise RuntimeError("没有可用的 5m 行情做探针，这条判据测了个空")
        if bad:
            raise RuntimeError("日频汇率因子前视：" + "；".join(bad[:4]))

        # 界面十字光标显示的是 `asof_row` 给的日期，实际取数是 `_asof`。
        # 两者必须是同一行 —— 否则界面会一本正经地报错日期，比不显示更糟
        probe_ts = [t for t in probes[:20]]
        rows = FM.asof_rows("usdcnh", probe_ts,
                            lag_days=FM._lag_days("usdcnh"),
                            avail_hour=FM._avail_hour("usdcnh"))
        ok_row = [r for r in rows if not pd.isna(r)]
        if len(ok_row) < len(probe_ts):
            raise RuntimeError("asof_rows 给出了 NaT —— 光标会显示成没有来源")
        if not np.allclose(pct.reindex(pd.DatetimeIndex(ok_row)).to_numpy(float),
                           got.reindex(probe_ts).to_numpy(float), equal_nan=True):
            raise RuntimeError("asof_rows 与 _asof 取到的行不一致 —— 界面会报错日期")

        lag = FM._lag_days("usdcnh")
        return (f"{n_probe} 个探针（含夜盘 00:00~08:00）全部只用到 D−{lag} 及更早的行，"
                "与 pandas asof 独立口径逐位一致；光标显示的取数日与实际取数同源")

    check("日频汇率因子不碰当日行（真实行情扫描）", _macro_no_same_day)

    def _macro_label_offset():
        """反推 Tushare 外汇表的**标签偏移**，必须等于代码里写的那一个。

        `_LABEL_LAG` 写的是一条经验事实：FXCM 日线的 `trade_date` 比这根 K 线
        真实的交易日早一天（纽约日 D 的收盘写进 D−1 那一行）。这条事实是用
        `data_duka` 的 5m 汇率独立算出来的，所以只能用同一份数据反推 —— 一旦
        Tushare 改了标签、或者我们换了数据源，这条会红，而不是让前视悄悄回来。

        做法：两边各算日收益率（收益率平移不变、不受报价源的点差偏移影响），
        在 k ∈ [−2, 2] 里找相关性最高的那个 k。日频汇率太平滑，**不能拿价位
        去配**：价位差会被「源不同 + 序列断点」淹掉，之前就这么误判过一次。
        """
        from ttrade import factors_macro as FM
        from ttrade import macro as M
        import os
        import pandas as pd

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data_duka", "USDCNH_5m.parquet")
        if not os.path.exists(path):
            return "跳过：没有 data_duka/USDCNH_5m.parquet，无法反推标签偏移"
        bars = pd.read_parquet(path)
        if bars is None or bars.empty:
            return "跳过：5m 汇率是空文件"
        # 纽约日边界 = 北京 06:00（与 `factors_fx5.DAY_CUT_HOUR` 同一口径）
        ny_day = (bars.index - pd.Timedelta(hours=6)).normalize()
        ny_close = bars["close"].groupby(ny_day).last()
        ny_ret = ny_close.pct_change()

        raw = M.series("usdcnh")
        if raw is None or len(raw) < 200:
            # 同上：宏观库不在 git 里，没同步过就是「跳过」，不是「代码坏了」。
            st = M.stats()
            if not st.get("exists") or not st.get("rows"):
                return "跳过：宏观库 edb.duckdb 没有数据，反推不了标签偏移"
            raise RuntimeError(
                f"usdcnh 序列读不到，但库里有 {st.get('rows')} 行宏观数据")
        mac_ret = raw.pct_change()

        lo, hi = pd.Timestamp("2026-01-01"), ny_close.index[-1]
        r1 = mac_ret.loc[lo:hi].dropna()
        r2 = ny_ret.loc[lo:hi].dropna()
        scores = {}
        for k in (-2, -1, 0, 1, 2):
            sh = r2.copy()
            sh.index = sh.index - pd.Timedelta(days=k)
            j = pd.concat([r1.rename("m"), sh.rename("n")], axis=1, sort=True).dropna()
            if len(j) >= 40:
                scores[k] = (float(j["m"].corr(j["n"])), len(j))
        if 0 not in scores:
            raise RuntimeError("样本不足，反推不出偏移")
        best_k, (best_c, n) = max(scores.items(), key=lambda kv: abs(kv[1][0]))
        want = int(FM._LABEL_LAG.get("usdcnh", 0))
        if best_c < 0.8:
            raise RuntimeError(f"相关性太低（{best_c:+.3f}），反推不可信")
        if abs(best_c) - abs(scores[0][0]) < 0.3:
            raise RuntimeError(f"最优偏移不显著：k={best_k} {best_c:+.3f} vs k=0 "
                               f"{scores[0][0]:+.3f}")
        if best_k != want:
            raise RuntimeError(f"标签偏移变了：实测 k={best_k:+d}（corr {best_c:+.3f}），"
                               f"代码里 _LABEL_LAG['usdcnh']={want} —— 前视会回来")
        return (f"实测宏观行 D 与纽约日 D{best_k:+d} 的收益相关 {best_c:+.3f}"
                f"（n={n}，k=0 只有 {scores[0][0]:+.3f}），与 _LABEL_LAG 写的一致")

    check("Tushare 外汇表的标签偏移（用 5m 汇率反推）", _macro_label_offset)

    def _macro_index_offset():
        """四个外盘指数的**标签反推**：行 D 必须真的是 D 那天的收盘。

        用户 2026-11-20 要求「spx / ixic / hsi / n225 也补一条反推判据」。
        手法同上（日收益率、k ∈ [−2, 2] 找峰值），裁判换成东财日线 —— 东财的
        标签就是真实交易日（`usdcnh` 那条已用 dukascopy 5m 验过 0.988@k=0）。

        抓的不是「价位对不对」，而是「行 D 装的是不是 D 那天的收盘」。汇率那两条
        实测 k=+1（红过一次，就是这次修的那个前视）；指数这四条实测 k=0、相关
        1.000。任何一边改了标签，这里会红。

        取不到就往回退成「跳过」而不是红：东财接口、宏观库锁都会让这条断网式
        失败，那是环境问题，不是判据失效 —— 判据红了才说明数据真的变了。
        """
        import pandas as pd
        from ttrade import macro as M

        rows = []
        for code, secid in M._EM_INDEX.items():
            try:
                em = M._em_closed(secid)
            except Exception as e:                       # noqa: BLE001
                return f"跳过：东财 {code} 取不到（{type(e).__name__}: {e}）"
            if em is None or len(em) < 250:
                return f"跳过：东财 {code} 只有 {0 if em is None else len(em)} 行"
            try:
                ts = M.series(code)
            except Exception as e:                       # noqa: BLE001
                return f"跳过：宏观库读不到（{type(e).__name__}: {e}）"
            if ts is None or len(ts) < 250:
                return f"跳过：库里 {code} 只有 {0 if ts is None else len(ts)} 行"

            lo = max(em.index[0], ts.index[0], pd.Timestamp("2016-01-01"))
            hi = min(em.index[-1], ts.index[-1])
            r1 = ts.pct_change().loc[lo:hi].dropna()
            r2 = em.pct_change().loc[lo:hi].dropna()
            scores = {}
            for k in (-2, -1, 0, 1, 2):
                sh = r2.copy()
                sh.index = sh.index - pd.Timedelta(days=k)
                j = pd.concat([r1.rename("m"), sh.rename("n")],
                              axis=1, sort=True).dropna()
                if len(j) >= 100:
                    scores[k] = float(j["m"].corr(j["n"]))
            if 0 not in scores:
                raise RuntimeError(f"{code} 样本不足，反推不出偏移")
            best_k = max(scores, key=lambda k: abs(scores[k]))
            best_c = scores[best_k]
            if best_k != 0:
                raise RuntimeError(
                    f"{code} 的标签偏移变了：实测 k={best_k:+d}（corr {best_c:+.3f}），"
                    f"k=0 只有 {scores[0]:+.3f} —— 行 D 装的不再是 D 那天的收盘")
            if best_c < 0.99:
                raise RuntimeError(
                    f"{code} 与东财相关只有 {best_c:+.3f}，已经不像同一份数据了")
            rows.append((code, best_c))

        return "四条外盘指数全部 k=+0、与东财收益相关 " + " / ".join(
            f"{c} {v:+.3f}" for c, v in rows)

    check("外盘指数的标签反推（对东财日线）", _macro_index_offset)

    def _em_factor_closed_only():
        """东财日线只许收**已经收完**的那一根。

        东财盘中会返回今天那根没走完的 bar（实测 2026-11-20 11:38 拿到
        `11-20 6.6732`，那只是上午的价）。08:45 的定时任务照收，库里就把半个
        交易日的价当成日收盘存了下来 —— 而且看起来完全正常，从数据上查不出来，
        只能在取数这一处挡住。
        """
        import pandas as pd

        from ttrade import macro as M

        today = pd.Timestamp.today().normalize()
        idx = pd.DatetimeIndex([today - pd.Timedelta(days=3),
                                today - pd.Timedelta(days=1),
                                today, today + pd.Timedelta(days=1)])
        fake = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        orig = M._em_kline
        try:
            M._em_kline = lambda secid, start: fake          # noqa: ARG005
            got = M._em_fx("usdcnh_em")
            got_idx = M._em_closed("100.SPX")
        finally:
            M._em_kline = orig
        want = today - pd.Timedelta(days=1)
        for name, s in (("_em_fx", got), ("_em_closed", got_idx)):
            if len(s) != 2 or s.index[-1] != want:
                raise RuntimeError(
                    f"{name} 没有滤掉今天/未来那根，留下 {[d.date() for d in s.index]}"
                    f"（应只剩 ≤ {want.date()} 的 2 根）")
        # 反证：不滤的话会有 4 根 —— 说明上面那条断言不是恒真
        if len(fake) == len(got):
            raise RuntimeError("滤前滤后行数一样，这条断言测了个空")
        return f"今天({today.date()})与未来那一根都被滤掉，`_em_fx` / `_em_closed` 都只留 ≤ 昨天"

    check("东财日线只收已收完的那一根（盘中不给半根）", _em_factor_closed_only)

    def _contract_terms():
        """「从天勤刷新品种参数」这条通路：取到的真值要能写进表、回测要读到。

        ## 为什么必须验

        回测的每手保证金 = 价格 × 乘数 × 保证金率，而那张表里的默认值是
        按类别拍的（商品 10% / 股指 12% / 国债 3%）。实测同一笔分配资金下，
        白银交易所 22.7% ⇒ 回测手数是实盘的 2.3 倍；白糖 6.1% ⇒ 0.6 倍。
        公式是同一条，差在参数来源，所以必须有一条「把真实值刷进表」的通路，
        而且刷完回测要立刻读到 —— 否则组合页的「查看回测」永远和实盘对不上。

        用 `ZZTEST` 这种不存在的品种做，跑完删掉 —— 绝不去碰用户真实配置。
        """
        from ttrade import replay
        from ttrade import strategy_store as SStore
        from ttrade import symbols_info as SI
        from ttrade import symsync as SYNC

        sym, ex = "ZZTEST", "TEST"

        def row(mult, rate):
            return [{"symbol": sym, "exchange": ex, "contract": "ZZTEST2601",
                     "mult": mult, "margin": rate * 100.0 * mult,
                     "margin_rate": rate, "price": 100.0}]

        try:
            SI.reset_override(sym, ex)
            base = SI.info(sym, ex)
            # ① 差异小于 2% 不写（保证金随结算价天天微调，全表重写没意义）
            st = SYNC.apply(row(1.0, float(base["margin"]) * 1.01))
            if st["changed"]:
                raise RuntimeError("1% 的差异不该触发刷新")
            # ② 差异大就写，且标成 live
            st = SYNC.apply(row(1.0, 0.227))
            if st["changed"] != 1:
                raise RuntimeError(f"22.7% 没被写进表：{st}")
            if abs(float(SI.info(sym, ex)["margin"]) - 0.227) > 1e-12:
                raise RuntimeError("刷新之后 info() 读到的还是旧值")
            if str(SI.info(sym, ex)["margin_src"]) != "live":
                raise RuntimeError("刷新没标成 live，界面上分不出是谁填的")
            # ③ 回测侧要真的跟着变：make_bt 的 margin_rate 来自 symbols_info
            cfg = SStore.StrategyConfig(symbol=sym, exchange=ex, sizing="capital",
                                        principal=200_000.0, capital_pct=30.0)
            bt = replay.make_bt("cn", True, cfg)
            if abs(float(bt.margin_rate) - 0.227) > 1e-12:
                raise RuntimeError(
                    f"回测没读到刷新后的保证金率：{bt.margin_rate}（应为 0.227）")
            # ④ 显式刷新**允许覆盖手填值**（这是用户主动点的，不是后台偷偷改）
            SI.set_override(sym, ex, margin=0.30)
            if str(SI.info(sym, ex)["margin_src"]) != "user":
                raise RuntimeError("set_override 默认没有标成 user")
            if not SYNC.apply(row(1.0, 0.12))["changed"]:
                raise RuntimeError("手填值挡住了主动刷新 —— 这条通路就失效了")
            if abs(float(SI.info(sym, ex)["margin"]) - 0.12) > 1e-12:
                raise RuntimeError("刷新没有把表改成真实值")
            # ⑤ 非法值（0 / NaN）一律不写，绝不能把配置写成 0
            if SYNC.apply(row(0.0, 0.0))["changed"]:
                raise RuntimeError("0 值不该被写进表")
            # ⑥ 乘数必须**精确**比：9999 → 10000 只差 0.01%，套 2% 容差会永远刷不掉。
            #    `specs.multiplier()` 从成交额反推时就会留下这种零头（真实 T 是
            #    10000 反推成 9999，AU 1000 → 999，SC 1000 → 999），不精确比
            #    的结果是「点了刷新、表里还是 999」，看起来像功能没生效。
            SI.reset_override(sym, ex)
            SI.set_override(sym, ex, mult=999.0, tick=1.0)
            st = SYNC.apply([{"symbol": sym, "exchange": ex, "mult": 1000.0,
                              "margin_rate": 0.12, "tick": 0.02, "price": 100.0}])
            if st["changed"] != 1:
                raise RuntimeError(f"999→1000 被当成「没变化」跳过了：{st}")
            got = SI.info(sym, ex)
            if abs(float(got["mult"]) - 1000.0) > 1e-9:
                raise RuntimeError(f"乘数没刷成 1000：{got['mult']}")
            if abs(float(got["tick"]) - 0.02) > 1e-12:
                raise RuntimeError(f"最小跳动没写进表：{got['tick']}")
            # ⑦ 再刷一次必须变成「无变化」—— 否则说明判等永远不成立，每次都重写盘
            st = SYNC.apply([{"symbol": sym, "exchange": ex, "mult": 1000.0,
                              "margin_rate": 0.12, "tick": 0.02, "price": 100.0}])
            if st["changed"]:
                raise RuntimeError("同样的值刷第二遍还在写盘，判等有问题")
            # ⑧ 手续费「按手」口径：规整的 5 分档 + 现有就是 lot ⇒ 写元/手
            #    （au 20 / SR 2 / m 1.5 这类，交易所就是按手报的）
            SI.set_override(sym, ex, fee_mode="lot", fee_value=3.0)
            SYNC.apply([{"symbol": sym, "exchange": ex, "mult": 1000.0,
                         "margin_rate": 0.12, "tick": 0.02, "price": 100.0,
                         "commission": 10.0}])
            got = SI.info(sym, ex)
            if str(got["fee_mode"]) != "lot" or abs(float(got["fee_value"]) - 10.0) > 1e-12:
                raise RuntimeError(f"按手口径没写对：{got['fee_mode']} {got['fee_value']}")
            # ⑨ 手续费「按比例」口径：带零头的值只可能是「价 × 乘数 × 比例」
            #    折算出来的（cu 26.285 / AG 9.3712 / IM 29.9018），必须反推成比例，
            #    否则价格翻倍后成本还是那个死数 = 低估成本 = 策略虚假变好
            st = SYNC.apply([{"symbol": sym, "exchange": ex, "mult": 1000.0,
                              "margin_rate": 0.12, "tick": 0.02, "price": 100.0,
                              "commission": 9.3712}])
            got = SI.info(sym, ex)
            want = 9.3712 / (100.0 * 1000.0)
            if str(got["fee_mode"]) != "rate":
                raise RuntimeError(f"带零头的手续费没切到 rate：{got['fee_mode']}")
            if abs(float(got["fee_value"]) - want) > 1e-12:
                raise RuntimeError(f"反推比例不对：{got['fee_value']} vs {want}")
            if not st["fee_hint"]:
                raise RuntimeError("口径 lot→rate 变了却没提示，用户看不出成本规律变了")
            # ⑩ 同样的手续费再刷一遍不能重复写（比例是反推的，必须留容差）
            if SYNC.apply([{"symbol": sym, "exchange": ex, "mult": 1000.0,
                            "margin_rate": 0.12, "tick": 0.02, "price": 100.0,
                            "commission": 9.3712}])["changed"]:
                raise RuntimeError("手续费判等不稳，每次刷新都在重写盘")
        finally:
            SI.reset_override(sym, ex)
        return ("差异 2% 内跳过、超出写并标 live；make_bt 立刻读到新值；"
                "主动刷新覆盖手填；0 值被拒；乘数/跳动精确比（999→1000 必刷）；"
                "手续费按整 1 角档判按手、带零头判比例并反推；口径翻转会提示")

    check("从天勤刷新品种参数（乘数/保证金率 → 回测对齐）", _contract_terms)

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
        # 基准是分钟级：它的决策时刻就是 bar 自己的时间戳（`source=False`）
        dt_base = FS._decision_time(base.index, "5m", source=False)
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

    def _cross_period_minute():
        """分钟级跨周期：1h 的因子在 5m 策略上必须晚一根。

        这是补一次真实事故：5m 策略挂 1h 的 `mom_20`，卡玛翻了几倍。原因不是
        因子算错，而是**分钟级 bar 的时间戳是开始时刻**（天勤约定，本地 1h 库也
        是天勤下来的）——标 10:00 的那根覆盖 10:00–11:00，11:00 才收完，而
        `_decision_time` 当时对分钟级原样返回，等于 10:05 就用上了 10:00–11:00
        的收盘价。

        上一项（周线）测不出这个：周线走的是日线级那条分支（`normalize + 15:00`），
        本来就是对的；而且它拿 `_decision_time` 自己算期望值，实现错了也自洽。
        所以这里**用固定数值硬编码期望**，不准再调被测函数。
        """
        import numpy as np
        import pandas as pd

        from ttrade import factor_strategy as FS

        # 1. 时间戳 → 收完时刻的偏移量。**等于源周期时长**，不是目标周期：
        #    合成周期的索引取「组内最后一根源 bar 的开始时刻」
        #    （15m 首根 09:10 = 5m 的第 3 根，收完在 09:15）
        for p, want in (("5m", 300), ("15m", 300), ("30m", 300), ("1h", 3600),
                        ("2h", 3600), ("4h", 3600), ("1d_tq", 0), ("1w", 0)):
            got = FS._close_lag(p)
            if got != want:
                raise RuntimeError(f"_close_lag({p!r})：应 {want}，实际 {got}")

        # 2. 人造 1h 源：09:00/10:00/11:00/13:00/14:00 五根，值 1..5
        src_idx = pd.to_datetime(["2026-11-18 09:00", "2026-11-18 10:00",
                                  "2026-11-18 11:00", "2026-11-18 13:00",
                                  "2026-11-18 14:00"])
        src = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=src_idx)
        base = pd.date_range("2026-11-18 09:00", "2026-11-18 14:55", freq="5min")
        got = FS.align_to_base(src, base, "5m", "1h")

        def at(ts: str) -> float:
            return float(got.loc[pd.Timestamp("2026-11-18 " + ts)])

        # 09:00 那根 1h 也要 10:00 才收完 ⇒ 上午 10 点之前一根都用不上
        if np.isfinite(at("09:05")) or np.isfinite(at("09:55")):
            raise RuntimeError("09:00 那根 1h 在 10:00 之前就被用了（未来数据）")
        # 10:00 那根覆盖 10:00–11:00 ⇒ 11:00 之前只能用 09:00 那根（值 1）
        for ts in ("10:00", "10:05", "10:30", "10:55"):
            if at(ts) != 1.0:
                raise RuntimeError(
                    f"{ts} 拿到的 1h 因子是 {at(ts)}，应该是 09:00 那根（1.0）"
                    "—— 用上了还没收完的 10:00 那根就是未来数据")
        # 11:00 收完 ⇒ 11:00 起换 10:00 那根（值 2）；13:00 那根到 14:00 才可用
        for ts, want in (("11:00", 2.0), ("11:55", 2.0), ("13:05", 3.0),
                         ("14:05", 4.0), ("14:55", 4.0)):
            if at(ts) != want:
                raise RuntimeError(f"{ts} 应拿到 {want}，实际 {at(ts)}")

        # 3. 未来扰动：改「10:00 那根」的值，10:55 之前一根都不许动
        src2 = src.copy()
        src2.loc[pd.Timestamp("2026-11-18 10:00")] = 99.0
        got2 = FS.align_to_base(src2, base, "5m", "1h")
        mask = base <= pd.Timestamp("2026-11-18 10:55")
        a = got[mask].fillna(-9.0).to_numpy()
        b = got2[mask].fillna(-9.0).to_numpy()
        if not np.allclose(a, b):
            raise RuntimeError("改动 10:00 那根之后 10:55 之前的值变了 —— 有未来数据")
        # 但它必须在 11:00 之后**真的**被用上（否则上面那条靠「整列没用」也能过）
        if float(got2.loc[pd.Timestamp("2026-11-18 11:00")]) != 99.0:
            raise RuntimeError("10:00 那根在 11:00 之后也没被用上（对齐断了）")

        # 4. 日线级口径不许被这次改动带偏：15:00 才是当天的可用时刻
        day_idx = pd.to_datetime(["2026-11-17", "2026-11-18"])
        day = pd.Series([7.0, 8.0], index=day_idx)
        dbase = pd.date_range("2026-11-18 09:00", "2026-11-19 14:55", freq="5min")
        dgot = FS.align_to_base(day, dbase, "5m", "1d_tq")
        if float(dgot.loc[pd.Timestamp("2026-11-18 09:05")]) != 7.0:
            raise RuntimeError("日线在当天开盘就用上了（应只有 15:00 之后才可用）")
        if float(dgot.loc[pd.Timestamp("2026-11-18 14:55")]) != 7.0:
            raise RuntimeError(
                "15:00 之前就用上了当天日线（14:55 那根收盘价在 15:00，日线同刻才收完）")
        if float(dgot.loc[pd.Timestamp("2026-11-19 09:05")]) != 8.0:
            raise RuntimeError("前一天的日线到第二天开盘还没生效")
        # 2 日 / 周线照旧丢最后一根（会变形的 bar）
        if len(FS._decision_time(pd.DatetimeIndex(day_idx), "1w")) != 2:
            raise RuntimeError("周线的决策时刻算坏了")

        # 5. **全组合矩阵**：基准 × 源 逐对做未来扰动。
        #    只钉 1h→5m 不够 —— 用户踩到的就是这个坑，其它组合同样可能漏。
        #    `LAG` 是测试自己认的物理事实（源 bar 从时间戳到收完的秒数），
        #    故意不复用 `_close_lag`，否则实现错了这里也自洽。
        FREQ = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h",
                "2h": "2h", "4h": "4h", "1d_tq": "D", "2d": "2D", "1w": "7D"}
        LAG = {"5m": 300, "15m": 300, "30m": 300, "1h": 3600, "2h": 3600,
               "4h": 3600, "1d_tq": None, "2d": None, "1w": None}  # None = 15:00 收
        SPAN = pd.Timedelta(days=28)
        t0 = pd.Timestamp("2026-11-02 09:00")
        bases = ["5m", "15m", "30m", "1h", "1d_tq"]
        sources = ["5m", "15m", "30m", "1h", "2h", "4h", "1d_tq", "2d", "1w"]

        def done_at(st: pd.Timestamp, p: str) -> pd.Timestamp:
            lag = LAG[p]
            return st + pd.Timedelta(seconds=lag) if lag else st.normalize() + FS.DAY_CLOSE

        checked = 0
        for b in bases:
            b_idx = pd.date_range(t0, t0 + SPAN, freq=FREQ[b])
            # 基准的**决策时刻**：分钟级就是它的时间戳，日线级是当天 15:00
            dt_base = (b_idx if LAG[b] is not None
                       else b_idx.normalize() + FS.DAY_CLOSE)
            for s in sources:
                if s == b:
                    continue
                s_idx = pd.date_range(t0, t0 + SPAN, freq=FREQ[s])
                if len(s_idx) < 5 or len(b_idx) < 5:
                    continue
                src = pd.Series(np.arange(len(s_idx), dtype=float), index=s_idx)
                got = FS.align_to_base(src, b_idx, b, s)
                # 两个截断点都要测，缺一个就漏一半：
                #   · 开始时刻：这根 bar 还没走完，基准此刻不能用它（抓「提前用半根」，
                #     1h/15m 这类就是这么漏的）；
                #   · 收完时刻：它之后的 bar 更不许用（抓「跨 bar 泄漏」）。
                k = len(s_idx) // 2
                for T in (s_idx[k], done_at(s_idx[k], s)):
                    m = np.asarray(dt_base <= T)
                    if not m.any():
                        continue
                    src2 = src.copy()
                    bad = [i for i, t in enumerate(s_idx) if done_at(t, s) > T]
                    if not bad:
                        continue
                    src2.iloc[bad] = 999.0
                    got2 = FS.align_to_base(src2, b_idx, b, s)
                    a = got[m].fillna(-9.0).to_numpy()
                    c = got2[m].fillna(-9.0).to_numpy()
                    if not np.allclose(a, c):
                        n_bad = int((a != c).sum())
                        raise RuntimeError(
                            f"{b} 基准 / {s} 源：把 T={T} 时还没收完的 {len(bad)} 根源"
                            f"bar 改成 999 之后，决策时刻不晚于 T 的 {n_bad} 个对齐值"
                            "变了 —— 有未来数据")
                    # 扰动必须真的生效，否则上面那条靠「整列没用」也能过
                    later = got2[~m]
                    if not (later.fillna(-9.0) == 999.0).any():
                        raise RuntimeError(
                            f"{b}/{s}：T 之后的扰动值没有被用上（对齐断了）")
                    checked += 1
        if checked < 60:
            raise RuntimeError(f"组合矩阵只跑了 {checked} 对，覆盖不够")

        return ("1h 的 10:00 那根到 11:00 才可用（10:00/10:55 仍是 09:00 那根 1.0，"
                "11:00 起 2.0、13:05 是 3.0）；改 10:00 那根不影响 10:55 之前、"
                f"却在 11:00 生效；日线仍 15:00 可用；{checked} 个「基准×源」组合"
                "在各自截断点上未来扰动全部无影响（含 15m/30m/2h/4h 合成周期与 1d/2d/1w）")

    check("跨周期因子：分钟级（1h 因子在 5m 上必须晚一根）", _cross_period_minute)


    def _factor_daily_lag():
        """日频外部因子铺到分钟线上：按**自然日**差分，且只吃 T-1 的读数。

        EDB 的仓单 / 库存 / 利润 / 交割量一天只有一个数，`_asof(lag_days=1)`
        把它铺到分钟线上 —— 同一个**自然日**里所有 bar 都是同一个常数。这时候
        因子里的 `s.shift(5)` / `s.pct_change(5)` 量到的是「5 根 bar 前」（5m
        上就是 25 分钟），同一天里恒为 0，只有每天头 n 根才是真值。方向和前视
        相反，但一样隐蔽：因子有数、不报错、IC 还挺像样，实际是另一个东西。

        🔴🔴 **分组键必须是自然日，不能是交易日。** `_asof` 是按自然日 as-of
        的（EDB 的读数是「某一天收盘后才有」，统一推后一个日历日），所以「哪根
        bar 能看到哪一天的读数」这条边界画在自然日上。用 `trading_day` 分组会
        把夜盘和次日日盘并成一格：21:00 那段的交易日是**次日**，`.last()` 取到
        次日日盘的读数再搬回夜盘 —— 等于 21:00 就用上「当日的日频数据」。
        这个构造特意带了**夜盘**（挂在上一交易日的日历日、交易日归次日），
        不带夜盘的话两种分组键一模一样，测了也白测。

        验五件事：
          1. 同一个**自然日**内恒定（按交易日看会出现两个值，那正是错的）
          2. 值 = 「自然日 X 能看到的读数 / 5 个交易日前的读数」，不是「n 根 bar 前」
          3. 分位窗口的最后一格是「X 能看到的读数」，不是「X 当天或次日」的
          4. 改未来日期的数据，过去那些天一动不动
          5. 日线帧上 `_day_shift` / `_day_diff` 退化成普通 `shift` / `diff`
        """
        import numpy as np
        import pandas as pd

        from ttrade import factors_ext as FE

        # --- 人造 5m 骨架：45 个交易日，每天「日盘 10 根 + 夜盘 6 根」。
        #     夜盘挂在**上一交易日**的日历日 21:00–22:25，交易日归次日 ——
        #     和真实夜盘一致。日盘根数 10 > 5，老实现会在日盘后 5 根退化成 0。
        days = pd.bdate_range("2024-01-02", periods=45)
        stamps, tdays = [], []
        for i, d in enumerate(days):
            for k in range(10):                     # 日盘 09:00–09:45
                stamps.append(d + pd.Timedelta(minutes=5 * k))
                tdays.append(d)
            if i + 1 < len(days):                   # 夜盘 21:00–22:25，归次日
                for k in range(6):
                    stamps.append(d + pd.Timedelta(hours=21, minutes=5 * k))
                    tdays.append(days[i + 1])
        idx = pd.DatetimeIndex(stamps)
        n_bar = len(idx)
        # 收盘价逐 bar 递增一点点：`spot_basis_chg_5` 的分子要随 bar 变
        close = 100.0 + 0.01 * np.arange(n_bar, dtype=float)

        def _make():
            df = pd.DataFrame({
                "open": close, "high": close + 1.0, "low": close - 1.0,
                "close": close, "volume": 10.0, "open_interest": 1000.0,
                "trading_day": pd.to_datetime(tdays),
            }, index=idx)
            df.attrs["symbol"] = "RB"
            return df

        # --- 人造 EDB：从 2023-12-25 起每个工作日一个值，值 = 1,2,3,...
        #     （顺手留出前置区间，免得 2024-01-02 那天 as-of 落空）
        raw = pd.Series(np.arange(1, 61, dtype=float),
                        index=pd.bdate_range("2023-12-25", periods=60))
        raw_dates = list(raw.index)
        raw_vals = [float(v) for v in raw.to_numpy()]

        def manual_asof(ts, series_dates=raw_dates, series_vals=raw_vals,
                        lag_days=1):
            """手写一遍 as-of 取数，不复用 `_asof` —— 否则就是自洽测试。"""
            t = pd.Timestamp(ts).normalize() - pd.Timedelta(days=int(lag_days))
            pick = np.nan
            for d, v in zip(series_dates, series_vals):
                if d <= t:
                    pick = v
                else:
                    break
            return pick

        orig_pick = FE._edb_pick
        try:
            FE._edb_pick = lambda sym: {
                "warehouse": raw, "inventory": raw, "profit": raw,
                "delivery": raw, "spot_price": raw,
            }
            df = _make()
            war_val = FE.f_war_val(df)
            war_chg1 = FE.f_war_chg_1(df)
            war_chg5 = FE.f_war_chg_5(df)
            inv_chg5 = FE.f_inv_chg_5(df)
            pro_chg5 = FE.f_profit_chg_5(df)
            del_chg30 = FE.f_deliv_chg_30(df)
            bas_chg5 = FE.f_spot_basis_chg_5(df)

            # 1. 同一个**自然日**内恒定
            adate = pd.Series(idx.normalize(), index=idx)
            for nm, s in (("war_val", war_val), ("war_chg_5", war_chg5),
                          ("inv_chg_5", inv_chg5), ("pro_chg_5", pro_chg5),
                          ("deliv_chg_30", del_chg30)):
                nc = s.groupby(adate).nunique(dropna=True)
                if (nc > 1).any():
                    bad = nc[nc > 1].index[0]
                    raise AssertionError(
                        f"{nm} 在 {pd.Timestamp(bad).date()} 一个自然日内有 "
                        f"{int(nc.loc[bad])} 个不同值 —— 要么还在按 bar 差分，"
                        f"要么分组键用了交易日（夜盘段会拿到次日读数）")
            # 反证：这个骨架必须真的把「夜盘段」和「日盘段」分成两个交易日
            # 口径的值，否则按交易日分组的错法在这里测不出来（测试就变空了）
            tkey = pd.Series([d for d in tdays], index=idx)
            nc_t = war_chg5.groupby(tkey).nunique(dropna=True)
            if not (nc_t > 1).any():
                raise AssertionError(
                    "骨架没造出「夜盘归次日」的效果 —— 按交易日分组也不会错，"
                    "这条自检等于没测到东西")

            # 2. 值 = 两个交易日的序列值之比
            for i, d in enumerate(days):
                here = manual_asof(d)
                if abs(war_val.loc[war_val.index.normalize() == d].iloc[0] - here) > 1e-9:
                    raise AssertionError(f"war_val 在 {d.date()} 不等于 T-1 的序列值")
                if i >= 1:
                    # war_chg_1：今天那根用的值 比 上一个交易日那根用的值
                    exp1 = here / manual_asof(days[i - 1]) - 1.0
                    got1 = war_chg1.loc[(war_chg1.index.normalize() == d)].iloc[0]
                    if np.isfinite(exp1) and abs(got1 - exp1) > 1e-9:
                        raise AssertionError(
                            f"war_chg_1 在 {d.date()} 是 {got1:.6f}，"
                            f"按交易日应是 {exp1:.6f}")
                if i >= 5:
                    exp5 = here / manual_asof(days[i - 5]) - 1.0
                    got5 = war_chg5.loc[war_chg5.index.normalize() == d].iloc[0]
                    if np.isfinite(exp5) and abs(got5 - exp5) > 1e-9:
                        raise AssertionError(
                            f"war_chg_5 在 {d.date()} 是 {got5:.6f}，"
                            f"按交易日应是 {exp5:.6f}（老实现会算成 25 分钟）")

            # 老实现会在日盘后 5 根上退化成 0；按天算则一个 0 都不该有
            if (np.abs(war_chg5.dropna()) < 1e-12).any():
                raise AssertionError("war_chg_5 出现了 0 值 —— 仍是按 bar 差分的")

            # 2b. 分位窗口的最后一格必须是「这个自然日能看到的读数」。
            #     用交易日分组会让夜盘段拿到次日日盘的读数 —— 那正是
            #     「21:00 就用上当日的日频数据」。
            fund = FE._fund(df, "warehouse")
            rk = FE._pct_rank(df, fund, n=10, minp=3)
            exp_avail = pd.Series([manual_asof(d) for d in days], index=days)
            exp_rk = exp_avail.rolling(10, min_periods=3).rank(pct=True)
            got_rk = rk.groupby(pd.DatetimeIndex(rk.index).normalize()).first()
            for d in days[2:]:
                a, b = float(got_rk.loc[d]), float(exp_rk.loc[d])
                if abs(a - b) > 1e-12:
                    raise AssertionError(
                        f"war_pct 的分位窗口在 {d.date()} 是 {a:.6f}，"
                        f"按「只看得到 <= 前一天」应是 {b:.6f}")

            # 3. 未来扰动不影响过去
            bumped = raw.copy()
            bumped.loc[bumped.index >= days[-2]] *= 100.0
            if not (bumped.to_numpy() != raw.to_numpy()).any():
                raise AssertionError("扰动日的构造错了：一个值都没改到")
            FE._edb_pick = lambda sym: {"warehouse": bumped}
            war_chg5b = FE.f_war_chg_5(_make())
            cut = war_chg5.index.normalize() <= days[-3]
            a = war_chg5[cut].to_numpy(dtype=float)
            b = war_chg5b[cut].to_numpy(dtype=float)
            if not np.allclose(a, b, equal_nan=True):
                raise AssertionError(
                    f"改了 {days[-2].date()} 之后的数据，{days[-3].date()} 及之前的 "
                    "war_chg_5 变了 —— 有前视")

            # 4. 日线帧上退化成普通 shift / diff
            dclose = pd.Series(100.0 + np.arange(60, dtype=float),
                               index=pd.bdate_range("2024-01-02", periods=60))
            ddf = pd.DataFrame({"close": dclose, "trading_day": dclose.index},
                               index=dclose.index)
            ddf.attrs["symbol"] = "RB"
            if not np.allclose(FE._day_shift(ddf, dclose, 5).to_numpy(),
                               dclose.shift(5).to_numpy(), equal_nan=True):
                raise AssertionError("日线帧上 `_day_shift` 没退化成 shift(5)")
            if not np.allclose(FE._day_diff(ddf, dclose, 5).to_numpy(),
                               dclose.diff(5).to_numpy(), equal_nan=True):
                raise AssertionError("日线帧上 `_day_diff` 没退化成 diff(5)")

            # 5. 基差变化率的**分子**仍跟当根 bar（不是常数），减数走交易日
            bidx = bas_chg5.index
            ulast = {d: bidx[bidx.normalize() == d][-1] for d in days}
            for i in (25, 30, 35):
                d, d5 = days[i], days[i - 5]
                spot = float(manual_asof(d))
                spot5 = float(manual_asof(d5))
                t5 = ulast[d5]
                bas_old = (spot5 - df.at[t5, "close"]) / df.at[t5, "close"]
                for pos in (0, 3, 7):
                    t = bidx[bidx.normalize() == d][pos]
                    bas_now = (spot - df.at[t, "close"]) / df.at[t, "close"]
                    exp = bas_now - bas_old
                    got = float(bas_chg5.loc[t])
                    if abs(got - exp) > 1e-9:
                        raise AssertionError(
                            f"spot_basis_chg_5 在 {t} 是 {got:.8f}，应是 {exp:.8f}")

        finally:
            FE._edb_pick = orig_pick
            FE.clear_cache()

        return ("仓单/库存/利润/交割量/现货在 5m 上按**自然日**差分（同日内常数、"
                "等于「X 能看到的读数」对 5 个交易日前，不是 25 分钟前）；"
                "夜盘段不会被搬用次日日盘的读数；改未来数据不影响过去；"
                "日线帧上退化成普通 shift/diff")

    check("日频因子在分钟线上按自然日算 + 只吃 T-1（含夜盘段）", _factor_daily_lag)


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

        # 口径锁：多品种是「一个账户的钱按品种数均分」。每条腿只拿 1/N，
        # 组合曲线仍从**全额**本金起步。2026-11-19 之前这里每条腿都拿满一份，
        # 回测的手数是实盘的 N 倍
        legs_ic = [float(r.metrics.get("initial_cash") or 0)
                   for r in rp_pool.res.legs.values()]
        want = float(cfg.principal) / len(frames)
        if not legs_ic or any(abs(x - want) > 1 for x in legs_ic):
            raise RuntimeError(f"每条腿的本金不是「总本金 ÷ 品种数」：{legs_ic}")
        comb_ic = float(rp_pool.res.metrics.get("initial_cash") or 0)
        if abs(comb_ic - float(cfg.principal)) > 1:
            raise RuntimeError(f"组合本金不是全额：{comb_ic}，配置 {cfg.principal}")

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
        # 手工口径也要按「总本金 ÷ 品种数」拆，和 `run_multi` 一致（2026-11-19）。
        # 不拆的话每腿多拿一倍本金，`lots_round` 取整会差一手，两条曲线对不上
        per_sym = cfg.per_symbol_principal(len(frames))
        for key, out in outs.items():
            one = cfg.clone(symbol=key.split("_")[0], exchange="SHFE",
                            multi_symbols=[], cross_symbols=[])
            one.principal = per_sym
            # 手工口径的「允许做空」必须和 `run_multi` 走同一个判据 ——
            # 这里以前硬编码 False，于是批量扫描的批次标签把策略裁成只做多时
            # 两边还「对得上」，掩盖了 #145
            bt = RP.make_bt("cn", RP.resolve_allow_short(one), one)
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
        #
        # 基准是**日线**：这里的参数（训练窗口 300）本来就是「交易日」，
        # 换成 24 小时连续的 5min 假数据会让「60 个交易日」变成 17280 根，
        # 1200 根的样本一根都出不了信号 —— 那不是策略的问题，是假数据
        # 没有真实的交易时段（真 5m 一天只有 48~68 根）。
        n = 1200
        idx = pd.bdate_range("2020-01-01", periods=n)
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
                # ML 快/慢路径会往诊断日志里写 `[TGT]`。真构造函数里 `_diag`
                # 一定有（`LiveLog()`），这里绕过了 `super().__init__()`，得自己补
                self._diag = None

            def _subscribe(self, st): pass
            def _skey(self, cfg): return cfg.key()
            # 这一项测的是「同一根 bar 里读后台信号」，不是交易时段门禁 ——
            # 把时段闸门关掉，否则它会随「自检跑在几点」而红（`_session_gate`
            # 那一项才是测门禁本身，用的是固定时间戳）
            def _open_now(self, cfg): return True
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
                self._diag = None        # 同上：绕过了真构造函数

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

    def _live_aux_frame():
        """实盘跨周期帧必须挂上衍生列，否则展期 / 期权因子被**静默剔除**。

        2026-11-13 的实盘就是这样：`IM定时突破+展期美元指数过滤` 的策略在
        `tqdata.load_main` 拿到的日线只有 10 列（OHLCV + 持仓量），
        `roll_main_oi` 整列 NaN（研究页 78% 有值、实盘 0%），`factor_matrix`
        把它剔掉，实盘跑的是「只有一个美元因子」的另一条策略 —— 日志里一个字
        都没有。日线走 `attach`、分钟走 `attach_intraday`，别搞反：分钟 bar
        用当天的期限结构就是未来信息。
        """
        import pandas as pd

        from ttrade import auxdata
        from ttrade.strategy_runner import PortfolioRunner as R

        seen = []
        orig = (auxdata.attach, auxdata.attach_intraday)

        def _fake_daily(df, sym, ex, aux_dir=None):
            seen.append("daily")
            out = df.copy()
            out["c1_oi"] = 1.0
            return out

        def _fake_intra(df, sym, ex, aux_dir=None):
            seen.append("intra")
            out = df.copy()
            out["c1_oi"] = 2.0
            return out

        auxdata.attach, auxdata.attach_intraday = _fake_daily, _fake_intra
        try:
            day = pd.DataFrame({"close": [1.0] * 40},
                               index=pd.bdate_range("2026-01-05", periods=40))
            m5 = pd.DataFrame(
                {"close": [1.0] * 200},
                index=pd.date_range("2026-01-05 09:00", periods=200, freq="5min"))
            a = R._aux_frame(day, "RB", "SHFE")
            b = R._aux_frame(m5, "RB", "SHFE")
            c = R._aux_frame(None, "RB", "SHFE")
            # 空表也要安全返回（跨周期行情没读到是常事）
            e = R._aux_frame(day.iloc[:0], "RB", "SHFE")
        finally:
            auxdata.attach, auxdata.attach_intraday = orig

        if seen != ["daily", "intra"]:
            raise RuntimeError(f"挂列的次序/选择不对：{seen}（应 daily → intra）")
        if "c1_oi" not in a.columns or "c1_oi" not in b.columns:
            raise RuntimeError("衍生列没挂上")
        if c is not None or len(e) != 0:
            raise RuntimeError("空行情应原样返回（None / 空表）")
        return f"日线走 attach、5m 走 attach_intraday（{len(a.columns)}/{len(b.columns)} 列）"

    check("实盘跨周期帧挂衍生列", _live_aux_frame)

    def _live_dropped_note():
        """因子被剔除必须吭声，且只吭一次。

        静默剔除是最难查的一类：策略照跑、仓照开，只是过滤条件比回测少几票，
        曲线上看不出异常。上面那条 `_live_aux_frame` 修的是「为什么会缺」，
        这条钉的是「缺了要说话」。
        """
        import pandas as pd

        from ttrade.strategy_runner import PortfolioRunner as R

        class _Cfg:
            sid = "s1"

        class _St:
            cfg = _Cfg()

            def title(self):
                return "某策略"

        class _Out:
            class pack:                      # noqa: N801 - 只做壳
                fmat = pd.DataFrame(columns=["macro_usdcnh"])

        r = object.__new__(R)
        r._factor_warned = set()
        logs = []
        r._log_fn = logs.append

        keys = ["roll_main_oi@1d_tq", "macro_usdcnh"]
        r._note_dropped(_St(), keys, _Out())      # 少了 1 个 → 记一条
        r._note_dropped(_St(), keys, _Out())      # 同一槽位不再重复

        if len(logs) != 1:
            raise RuntimeError(f"剔除提示应恰好 1 条，实际 {len(logs)} 条：{logs}")
        if "1/2 个因子参与合成" not in logs[0]:
            raise RuntimeError(f"提示没说清「实际参与几个」：{logs[0]}")
        if "roll_main_oi@1d_tq" not in logs[0]:
            raise RuntimeError(f"提示没指出缺哪个因子：{logs[0]}")

        # 一个都没少时**也要记一条**（"2/2 个因子参与合成"），但不该说「缺」。
        # 集运欧线那次就是反过来的情形：实盘比回测**多**几票，日志里既没有剔除
        # 也没有任何线索 —— 只有「实际参与几个」能让人回头去数回测那边。
        class _Full:
            class pack:                      # noqa: N801
                fmat = pd.DataFrame(columns=keys)

        r._factor_warned.clear()
        seen = len(logs)
        r._note_dropped(_St(), keys, _Full())
        if len(logs) != seen + 1:
            raise RuntimeError(f"一个都没剔也要记一条（否则没法比对两边列数）：{logs[seen:]}")
        if "2/2 个因子参与合成" not in logs[-1] or "缺" in logs[-1]:
            raise RuntimeError(f"全在时的提示不对：{logs[-1]}")
        return f"剔除 1 个时报 1 条并去重、全在时记参与数：{logs[0][:30]}"

    check("实盘因子被剔除要提示", _live_dropped_note)

    def _live_error_log():
        """槽位错误要落盘，且同一条只写一次。

        GUI 的日志控件在内存里，关窗就没了；表格「错误」那格只有 tooltip 一句
        话。2026-11-13 用户报「宏观美元策略实时报错」时，盘上没有任何记录，
        只能靠猜 —— `live.log` 以前只有机器学习在写。
        """
        from ttrade.strategy_runner import PortfolioRunner as R

        class _Cfg:
            def __init__(self, sid):
                self.sid = sid

        class _St:
            def __init__(self, sid, err):
                self.cfg = _Cfg(sid)
                self.error = err
                self.running = False

            def title(self):
                return f"策略{self.cfg.sid}"

        class _Diag:
            def __init__(self):
                self.lines = []

            def write(self, tag, msg):
                self.lines.append((tag, msg))

        r = object.__new__(R)
        r._err_logged = set()
        r._diag = _Diag()
        r._slots = {"a": _St("a", "ValueError: 跨周期因子缺行情"), "b": _St("b", "")}
        r._log_errors()
        r._log_errors()                       # 再调一次不许重复写
        if len(r._diag.lines) != 1:
            raise RuntimeError(f"应只写 1 行，实际 {r._diag.lines}")
        tag, msg = r._diag.lines[0]
        if tag != "LIVE" or "跨周期因子缺行情" not in msg:
            raise RuntimeError(f"写的内容不对：{r._diag.lines[0]}")

        # 错误变了要再写一条（换了一个故障点不是同一件事）
        r._slots["a"].error = "TypeError: 另一件事"
        r._log_errors()
        if len(r._diag.lines) != 2:
            raise RuntimeError(f"换错误应再写一条，实际 {len(r._diag.lines)} 条")
        return f"错误落盘 {len(r._diag.lines)} 条，同一条不重复"

    check("实盘错误落盘（去重）", _live_error_log)

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

    def _align_intrabar_theo():
        """K 线内那族的**理论仓**必须进监控器，也必须能被一键对齐真的建出来。

        2026-11-24 用户报「一键对齐没有读出来这个多品种模型的持仓」。融航账户上
        `sbbox多品种波段1h 基本面空` 7 条腿，策略表「理论」列写着 -6 / -8 / -10 /
        -44，点一键对齐却只下发了 `CFFEX.IM2612 / CFFEX.TL2612` 两笔 —— 这 7 条腿
        一条都不在名单里（`live.log` 里只有 09:18 那笔 `SHFE.al2701 卖 OPEN 6 手`，
        其余 6 条只有 `[LOTS] 开仓手数 0 → 8/10/44`）。

        根因：`theoretical()` 原来用 `st.target`，而 K 线内那族（`sbbox_ib`）的
        `target` 是**下单目标** —— 只有盘中价格真碰到触发价（`_ib_arm`）才有值，
        冷启动时恒 0。`target=0` 且账户无仓 ⇒ `mismatch()` 里
        `(0,0) == (0,0)` 直接 `continue`，一行都不产出。

        第二处同源错误在 `_sync_after_align()`：它也用 `target` 算，而且拿
        `if sym not in st.held: continue` 当闸门 —— 即使对齐把仓开出来了也不记账本，
        下一轮 `mismatch()` 把它当「未托管」，再点一次对齐就平掉（开了又平）。

        两处都必须用 `_theo_qty()`（= 方向 × 锁存的开仓手数，与回测 `position`
        同一列），**下单路径不动**（仍按 `target` 等触发价）。
        """
        from ttrade import strategy_store as SStore
        from ttrade.strategy_runner import (STRATEGIES, PortfolioRunner,
                                            SlotState)

        spec = STRATEGIES.get("sbbox_ib")
        if not getattr(spec, "intrabar", False):
            raise RuntimeError("前提不成立：sbbox_ib 不在策略表里 / 不是 intrabar")

        class P:
            def __init__(self, lo, sh):
                self.pos_long, self.pos_short = lo, sh
                for s, v in (("long", lo), ("short", sh)):
                    setattr(self, f"pos_{s}_his", v)
                    setattr(self, f"pos_{s}_today", 0)
                    setattr(self, f"volume_{s}_his", v)
                    setattr(self, f"volume_{s}_today", 0)

        r = PortfolioRunner.__new__(PortfolioRunner)
        r._log_fn = None
        r._diag = None
        r._mm_cache, r._mm_ts = [], 0.0
        r._align_ts = 0.0
        r.api = object()                      # 没有 positions_ready ⇒ 按「已读回」
        r._save_book = lambda s: None          # 不落盘，自检不留垃圾
        # 台账「加载首轮不交易」那一套（`_fresh_book` / `_adopted`）不在本用例
        # 的验算范围里，单独桩掉；要验的是 `_rebalance` 会不会反手平仓
        r._adopt_if_needed = lambda s: None
        st = SlotState(cfg=SStore.StrategyConfig(
            sid="t", name="sbbox多品种波段1h 基本面空", period="1h",
            family="rule", rule_key="sbbox_ib", note="short", short_only=True))
        st.running, st.underlying = True, "SHFE.cu2701"
        # 策略口径：做空、开仓 8 手（`[LOTS] 开仓手数 0 → 8`）；
        # 但这一轮从没触发入场 ⇒ 下单目标 `target` 还是 0
        st.signal, st.theo_lots, st.lots = -1, 8, 8
        st.target = 0
        r._slots = {"t": st}
        r._align_slot = SlotState(cfg=SStore.StrategyConfig(
            sid="__align__", name="一键对齐", period="1d"))
        r._pos = {}                            # 账户里确实没有这个仓

        if not r._ib_on(st):
            raise RuntimeError("前提不成立：这条策略没走 K 线内口径")
        if r._theo_qty(st) != -8:
            raise RuntimeError(f"理论仓该是 -8（信号×锁存手数），实际 {r._theo_qty(st)}")

        # ① 理论仓要读得出来
        theo = r.theoretical()
        if theo.get("SHFE.cu2701") != -8:
            raise RuntimeError(f"theoretical() 漏了 K 线内那族：{theo}")

        # ② 监控器/对齐要看得见这条腿，并且标注「回测口径、实盘没触发过」
        rows = r.mismatch(fresh=True)
        hit = [x for x in rows if x["symbol"] == "SHFE.cu2701"]
        if len(hit) != 1:
            raise RuntimeError(f"监控器里没有这条腿：{[x['symbol'] for x in rows]}")
        row = hit[0]
        if (row["target"], row["actual"], row["delta"]) != (-8, 0, -8):
            raise RuntimeError(f"差额算错了：{row}")
        if not row["theo_only"]:
            raise RuntimeError("没标出「回测口径 / 实盘没触发过」—— 用户会以为是调仓")

        # ③ 对齐要真发一笔（8 手空）
        sent = []
        r._send = lambda s, sym, delta, closeable=None: (
            sent.append((sym, delta)) or s.pending.append("x") or 1)
        n = r.align(close_orphans=True)
        if sent != [("SHFE.cu2701", -8)] or n != 1:
            raise RuntimeError(f"对齐该补开 8 手空，实发 {sent}")
        # ④ 对齐之后账本里必须有这条 —— 否则下一轮被当「未托管」再平掉
        if int(st.held.get("SHFE.cu2701") or 0) != -8:
            raise RuntimeError(f"对齐开了仓但账本没建条目：{st.held}")
        # ④b `target` 必须跟着一起摆。只写 `held` 的话，对齐之后是
        #     `target=0 / held=-8` —— 正好命中 `_rebalance` 里「目标没变 + 减仓」
        #     那条放行分支 ⇒ 下一轮 `delta=+8` 把刚补开的 8 手按市价买平。
        #     账户先摆成「已成交」，否则 `_rebalance` 里的「账户持仓还没跟上」
        #     那道闸会把它挡掉，这条断言就抓不到东西（下面的反证证明它不是恒真）。
        r._pos = {"SHFE.cu2701": P(0, 8)}
        if int(st.target or 0) != -8:
            raise RuntimeError(f"对齐后 target 没跟着摆（{st.target}）—— 会被自己平掉")
        r._align_slot.pending.clear()
        sent.clear()
        r._rebalance(st)
        if sent:
            raise RuntimeError(f"对齐补开的仓被策略自己平掉了：{sent}")

        # ④c 反证：把 target 掰回 0（= 只写 held 的旧写法），上面那条必须能变红
        keep = (st.target, st.based_target)
        st.target, st.based_target = 0, 0
        sent.clear()
        r._rebalance(st)
        st.target, st.based_target = keep
        if sent != [("SHFE.cu2701", 8)]:
            raise RuntimeError(
                f"反证不成立（旧写法竟然不发单，说明 ④b 抓不到东西）：{sent}")

        # ⑤ 账户跟上之后必须判一致（不能反复报差）
        r._mm_ts = 0.0
        r._align_slot.pending.clear()
        left = [x["symbol"] for x in r.mismatch(fresh=True)]
        if left:
            raise RuntimeError(f"账户补上之后还在报不一致：{left}")

        # ⑤b 策略自己的委托还在飞时，对齐不许动这个合约 —— 账户一秒后就变，
        #     按「现在没有」补差额就是把同一笔仓开两遍。这是「理论仓改走
        #     `_theo_qty`」带来的新敞口：从没触发过的腿也可能是刚报出去的单。
        st.pending.append("order-1")
        r._pos = {}
        r._mm_ts = 0.0
        r._align_ts = 0.0
        r._align_slot.pending.clear()
        sent.clear()
        if r.align(close_orphans=True) != 0 or sent:
            raise RuntimeError(f"策略还有在途单时对齐照发了：{sent}")
        st.pending.clear()

        # ⑥ 反证一：真触发过（`target` 有值）就不该再标「回测口径」
        st.target = -8
        r._mm_ts = 0.0
        r._pos = {}
        if r._theo_pending():
            raise RuntimeError("target 有值时不该进「回测口径」那一批")
        if r.mismatch(fresh=True)[0]["theo_only"]:
            raise RuntimeError("target 有值时 theo_only 该是假 —— 上面那几条可能是恒真")

        # ⑦ 反证二：普通策略（非 K 线内）空仓时理论仓就是 0，不许被这次改动带进来
        st2 = SlotState(cfg=SStore.StrategyConfig(
            sid="n", name="普通空仓策略", period="1h", family="rule",
            rule_key="ma_cross"))
        st2.running, st2.underlying = True, "DCE.jd2701"
        st2.signal, st2.theo_lots, st2.lots, st2.target = -1, 9, 9, 0
        r._slots = {"n": st2}
        if r.theoretical().get("DCE.jd2701"):
            raise RuntimeError("普通策略空仓也被算成有理论仓了 —— `_theo_qty` 分叉错了")
        r._mm_ts = 0.0
        if [x["symbol"] for x in r.mismatch(fresh=True)]:
            raise RuntimeError("普通策略空仓不该出现在监控器里")

        return ("K 线内那族的理论仓能读出来（-8）、标注「回测口径」、对齐补开 8 手、"
                "账本与 target 一起摆好（下一轮不会被自己平掉，反证能变红）、"
                "在途单时不重复开仓；账户跟上后判一致；真触发过/普通空仓两条反证"
                "都不误报")

    check("一键对齐读得出 K 线内多品种策略的理论仓", _align_intrabar_theo)

    def _align_no_flip():
        """一键对齐**不许**把「平仓」变成「反方向开仓」，也不许动没确认过的合约。

        2026-11-20 那次实盘的形状：确认框写的是「CFFEX.T2612 −6 手 → −2（平空4）」，
        发出去的却是 `SELL OPEN 2`（连发 3 笔），另外还平了一个确认框里根本
        没出现过的合约。三个独立缺陷，各验一条：

          ① 持仓快照在「算差额」和「发单」之间变了 ⇒ 可平量查不到时**一枪不放**，
             不能退化成 OPEN（`_legs` 的老行为）；
          ② `align(symbols=...)` 只动传进来的那几个合约 —— 没同意过的仓不碰；
          ③ `_legs(closeable=)` 的可平量上限在上期所「平今平昨」那一段也要夹住，
             否则「平自己 2 手」会变成「平账户 9 手」，把别人的仓顺手平了。
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

        r = PortfolioRunner.__new__(PortfolioRunner)
        r._log_fn = None
        r._diag = None
        r._mm_cache, r._mm_ts = [], 0.0
        r._align_ts = 0.0
        r.api = object()
        r._align_slot = SlotState(cfg=SStore.StrategyConfig(
            sid="__align__", name="一键对齐", period="1d"))
        r._slots = {}
        sent = []
        r._send = lambda s, sym, delta, closeable=None: (
            sent.append((sym, delta, closeable)) or s.pending.append("x") or 1)
        r._sync_after_align = lambda sym: None

        # ① 差额已经算出来了（该平 4 手空），但账户此刻读不到空头
        r._pos = {}
        row = {"symbol": "CFFEX.T2612", "target": -2, "actual": -6, "delta": 4,
               "managed": True, "t_long": 0, "t_short": 2,
               "a_long": 0, "a_short": 6, "locked": False}
        r._align_slot.pending.clear()
        if r._send_row(r._align_slot, row) != 0 or sent:
            raise RuntimeError(f"可平量查不到还是发了单：{sent}")
        if "不发反向开仓" not in (r._align_slot.last_action or ""):
            raise RuntimeError(f"没说明为什么没发：{r._align_slot.last_action!r}")

        # 账户真有这 6 手空时，应当平 4 手（BUY），而不是开空
        r._pos = {"CFFEX.T2612": P(0, 6)}
        r._align_slot.pending.clear()
        sent.clear()
        if r._send_row(r._align_slot, row) != 1:
            raise RuntimeError("账户有仓时一笔都没发")
        if sent != [("CFFEX.T2612", 4, 4)]:
            raise RuntimeError(f"该买平 4 手（closeable=4），实发 {sent}")

        # ② 只动确认过的合约；持仓基数变了就不发
        st = SlotState(cfg=SStore.StrategyConfig(sid="t", name="T", period="1h"))
        st.running, st.underlying, st.target = True, "CFFEX.T2612", -2
        r._slots = {"t": st}
        r._pos = {"CFFEX.T2612": P(0, 6), "DCE.jd2701": P(0, 6)}
        r._mm_ts = 0.0
        r._align_ts = 0.0
        r._align_slot.pending.clear()
        r._align_slot.rejects, r._align_slot.reject_msg = 0, ""
        r._align_slot.last_action = ""
        sent.clear()
        r.align(rows=[{"symbol": "CFFEX.T2612", "a_long": 0, "a_short": 6}])
        touched = {s for s, _d, _c in sent}
        if touched != {"CFFEX.T2612"}:
            raise RuntimeError(f"动了没确认过的合约：{sorted(touched)}")

        # 确认时写的是「空 6」，可账户此刻读到空 0 —— 换个基数会把「平空4」
        # 算成「开空2」，所以必须整笔跳过（而不是照新基数发）
        r._pos = {"CFFEX.T2612": P(0, 0), "DCE.jd2701": P(0, 6)}
        r._mm_ts = 0.0
        r._align_ts = 0.0
        r._align_slot.pending.clear()
        r._align_slot.last_action = ""
        sent.clear()
        r.align(rows=[{"symbol": "CFFEX.T2612", "a_long": 0, "a_short": 6}])
        if sent:
            raise RuntimeError(f"持仓基数变了还照发（会变成反向开仓）：{sent}")

        # ③ 上期所的平今平昨分支：可平量不许被「账户口径」抬上去
        r._pos = {"SHFE.rb2701": P(9, 0)}
        legs = r._legs("SHFE.rb2701", -5, closeable=2)
        if legs != [("CLOSE", 2), ("OPEN", 3)]:
            raise RuntimeError(f"可平量没夹住（会平掉别人的仓）：{legs}")
        # 反证：不给 closeable 时仍是账户口径 —— 说明上面那条不是恒真
        if r._legs("SHFE.rb2701", -5, closeable=None) != [("CLOSE", 5)]:
            raise RuntimeError("closeable=None 时应按账户口径平 5 手")

        return ("可平量查不到时一枪不放（写明原因）；只动确认过的合约；"
                "上期所分支的可平量上限夹住了（平 2 + 开 3，不是平 5）")

    check("一键对齐不许把平仓变成反向开仓", _align_no_flip)

    def _align_logged_in():
        """账户没登录的，一键对齐不许动它的仓（用户 2026-11-20 要求）。

        为什么必须挡：账户登出后 `api is None`，`align()` 一句「账户没有接进来」
        就返回 0 —— 可界面上那些合约照样列在确认框里，用户点完看不到任何反应，
        还以为平完了。更隐蔽的是「半登录」：`_pos` 还留着登出前的旧数，
        `mismatch()` 照常算出差额，看起来一切正常。

        所以判据用**账户登录**（`api is not None` ⇒ `account_ready`），不是行情
        （`alive` 看的是 feed，账户登出策略照样在跑），也不是 `pos_ready`。
        """
        from gui.portfolio_page import split_alignable

        snaps = {
            "已登录账户": {"account_ready": True,
                      "mismatch": [{"symbol": "CFFEX.T2612",
                                    "a_long": 0, "a_short": 6}]},
            "未登录账户": {"account_ready": False,
                      "mismatch": [{"symbol": "SHFE.rb2701",
                                    "a_long": 3, "a_short": 0},
                                   {"symbol": "DCE.jd2701",
                                    "a_long": 0, "a_short": 6}]},
            "没差额账户": {"account_ready": True, "mismatch": []},
        }
        ready, blocked = split_alignable(snaps)
        if list(ready) != ["已登录账户"]:
            raise RuntimeError(f"可对齐的账户里混进了未登录的：{list(ready)}")
        if len(ready["已登录账户"]) != 1:
            raise RuntimeError("可对齐账户的合约数被改了")
        if len(blocked) != 1 or "未登录账户" not in blocked[0] or "2 个合约" not in blocked[0]:
            raise RuntimeError(f"未登录账户没被点名：{blocked}")
        if any("没差额" in b for b in blocked):
            raise RuntimeError("没有差额的账户不该进跳过名单（每次点都刷一排）")

        # 反证：把 account_ready 翻成 True 后它就该进可对齐那一批 ——
        # 说明上面那条不是「因为没写 mismatch 才没进去」的恒真断言
        snaps["未登录账户"]["account_ready"] = True
        ready2, blocked2 = split_alignable(snaps)
        if "未登录账户" not in ready2 or blocked2:
            raise RuntimeError("account_ready=True 时没放行 —— 上面那条断言抓不到东西")

        # 老快照没有这个字段时按「已登录」处理：升级那一刻所有账户都不会
        # 突然变成「没登录」而被全部跳过
        ready3, blocked3 = split_alignable({"老账户": {"mismatch": [{"symbol": "X"}]}})
        if list(ready3) != ["老账户"] or blocked3:
            raise RuntimeError(f"缺 account_ready 字段时应默认已登录：{ready3} {blocked3}")

        if split_alignable(None) != ({}, []):
            raise RuntimeError("snaps 为空时应返回空结果")

        # 引擎这一层也必须有同一道闸：`api is None` ⇒ 一笔不发、并且说话
        from ttrade.strategy_runner import PortfolioRunner as _PR

        rr = _PR.__new__(_PR)
        rr.api = None
        rr._log_fn = None
        rr._diag = None
        if rr.align(rows=[]) != 0:
            raise RuntimeError("账户没登录时 align() 居然发单了")

        return ("未登录账户的 2 个合约被挡下并点名；翻成已登录立刻放行；"
                "缺字段按已登录处理（升级不会全被跳过）；align() 在 api=None 时返回 0")

    check("账户没登录就不做一键对齐", _align_logged_in)

    def _align_timeout_resync():
        """对齐「发出去但没成交」那条路：超时窗口要够追单，撤单后账本要回摆。

        2026-11-23 用户报「发了好几次一键对齐，但发出的委托单很少」，日志里是
        这样：09:30、09:31、09:37、09:39 各点了一次，每次都发出 5~7 笔，然后
        「对齐委托 30 秒未成交，已撤单」，而监控器上的差额**一次都没变**。
        两个缺陷：

        ① **超时窗口写死 30 秒，追单还没做完就整批撤掉**。追单是「等 wait_sec
           没成交就撤掉重报，最多 retries 次」，默认 5×3 就要 20 秒，再加上
           首轮、撤单确认的往返，30 秒根本不够 —— 用户把 wait_sec 调到 10 秒时
           更是必然撕在半路。

        ② 🔴 **`_sync_after_align` 是乐观的**：委托一发出去就把各策略的账本写成
           `target`。可限价单会被撤掉，账户原地没动，账本却说「到位了」。之后
           策略一翻信号就按假账本发「平仓 2 手」，柜台打回「平仓手数超过持仓量」
           —— 截图里 T2612 / IC2612 的那两笔拒单就是这么来的。撤单之后必须把
           账本按**账户实际**摆回来。

        另外钉住两件观测上的事：每笔落盘日志都要带账户名（两个账户写同一个
        live.log，不带就分不清谁是谁），以及某个合约没发出去时**逐个**记一笔
        「没发出去 + 原因」—— 以前只在整轮一笔都没发时才 echo，于是「10 个合约
        只动了 5 个」在日志里查不出另外 5 个为什么不动。
        """
        from ttrade import strategy_store as SStore
        from ttrade.strategy_runner import (ALIGN_TIMEOUT, PortfolioRunner,
                                            SlotState)

        ChaseConfig = SStore.ChaseConfig

        class P:
            def __init__(self, lo, sh):
                self.pos_long, self.pos_short = lo, sh
                for side, v in (("long", lo), ("short", sh)):
                    setattr(self, f"pos_{side}_his", v)
                    setattr(self, f"pos_{side}_today", 0)
                    setattr(self, f"volume_{side}_his", v)
                    setattr(self, f"volume_{side}_today", 0)

        class Spec:
            def label(self):
                return "自检户"

        class Diag:
            def __init__(self):
                self.rows = []

            def write(self, tag, msg):
                self.rows.append(f"[{tag}] {msg}")

        r = PortfolioRunner.__new__(PortfolioRunner)
        r._log_fn = None
        r._diag = Diag()
        r.spec = Spec()
        r._mm_cache, r._mm_ts = [], 0.0
        r._align_ts = 0.0
        r._align_touched = set()
        r.api = object()
        r._pos = {}
        r._save_book = lambda st: None
        r._align_slot = SlotState(cfg=SStore.StrategyConfig(
            sid="__align__", name="一键对齐", period="1d"))

        # ---- ① 超时窗口跟着追单参数走
        r.chase = ChaseConfig()                      # wait 5 × 3 次
        if abs(r._align_timeout() - ALIGN_TIMEOUT) > 1e-6:
            raise RuntimeError(
                f"默认追单下超时就该是 {ALIGN_TIMEOUT} 秒，"
                f"实际 {r._align_timeout()}")
        r.chase = ChaseConfig(wait_sec=10, retries=3)
        if abs(r._align_timeout() - 50.0) > 1e-6:
            raise RuntimeError(
                f"wait_sec=10 × (3+2) 应该给 50 秒，实际 {r._align_timeout()}"
                "（写死 30 秒会在追单做到一半时把整批撤掉）")
        r.chase = ChaseConfig(enabled=False)
        if abs(r._align_timeout() - ALIGN_TIMEOUT) > 1e-6:
            raise RuntimeError("关掉追单时不该放大超时窗口")

        # ---- ② 撤单后账本按账户实际回摆
        st = SlotState(cfg=SStore.StrategyConfig(
            sid="t", name="T", period="1h"))
        st.running, st.underlying, st.target = True, "CFFEX.T2612", -2
        # 乐观写法留下的假账本：账本 -2，账户其实平的
        st.held = {"CFFEX.T2612": -2}
        r._slots = {"t": st}
        r._pos = {}
        moved = r._resync_books()
        if st.held:
            raise RuntimeError(
                f"撤单后账本还留着 {st.held}（账户是平的，之后会发"
                "「平仓手数超过持仓量」被柜台打回）")
        if not moved or "t CFFEX.T2612 -2" not in moved[0]:
            raise RuntimeError(f"回摆清单没记下改了什么：{moved}")

        # 账户还有 3 手空时，账本要摆成 -3（不是 0、也不是原来的 -2）
        r._pos = {"CFFEX.T2612": P(0, 3)}
        st.held = {"CFFEX.T2612": -2}
        r._resync_books()
        if st.held.get("CFFEX.T2612") != -3:
            raise RuntimeError(f"账本没按账户实际（-3）摆：{st.held}")
        # 反证：不回摆的话它就一直是 -2 —— 上面那条不是恒真断言
        if int(r._net_pos("CFFEX.T2612")) == -2:
            raise RuntimeError("账户净仓本身就是 -2，这条断言抓不到东西")
        # `based` / `based_target` 不许被这次回摆改动：对齐没成功就不该
        # 把当前目标当成新基准，否则「目标不变不发单」会把策略永久冻住
        if getattr(st, "based", False) is not False:
            raise RuntimeError("回摆账本时把 based 也改了 —— 策略会被永久冻住")

        # ---- ③ 每个合约没发出去都要落一行原因
        r._diag.rows.clear()
        r._log_fn = lambda m: None
        r._mm_ts = 0.0
        r._align_ts = 0.0
        r._align_slot.pending.clear()
        r._align_slot.rejects, r._align_slot.reject_msg = 0, ""
        r._align_slot.last_action = ""

        def _no_send(_st, row):
            _st.last_action = "等盘口（这个合约还没收到行情）"
            return 0

        r._send_row = _no_send
        r.mismatch = lambda fresh=False: [{
            "symbol": "CFFEX.T2612", "target": -2, "actual": 0, "delta": -2,
            "managed": True, "t_long": 0, "t_short": 2,
            "a_long": 0, "a_short": 0, "locked": False}]
        if r.align(rows=[{"symbol": "CFFEX.T2612", "a_long": 0, "a_short": 0}]) != 0:
            raise RuntimeError("一个没成交的合约却让 align() 报了下发")
        joined = " ".join(r._diag.rows)
        if "没发出去" not in joined or "等盘口" not in joined:
            raise RuntimeError(f"没发出去的原因没落日志：{r._diag.rows}")
        if "[自检户]" not in joined:
            raise RuntimeError(
                f"落盘日志没带账户名（两个账户写同一个 live.log 会分不清）："
                f"{r._diag.rows[:2]}")

        # ---- ④ 持仓引用要定期重挂，比对缓存要跟着失效
        #
        # 用户 2026-11-23 报「账户已经平仓和开仓对齐了，这边还是显示没有对齐」：
        # 账户早就把 T2612 平了、交易页的持仓表里连这个合约都没有，监控器还写着
        # 「账户 -2、平空2」。`_pos` 只在 `_connect_account()` 里取一次，靠的是
        # 账户接口返回「活引用」，而活引用断了**不报错**。
        from ttrade.strategy_runner import MM_LOG_SEC, POS_ANCHOR_SEC

        class _Api:
            def __init__(self):
                self.pos = {"CFFEX.T2612": P(0, 2)}
                self.ord = {}
                self.n = 0

            def get_position(self):
                self.n += 1
                return self.pos

            def get_order(self):
                return self.ord

        r.api = _Api()
        r._pos = {"CFFEX.T2612": P(0, 2)}
        r._orders = {}
        r._pos_ts = 0.0
        r._mm_ts = time.time()          # 模拟「刚刚算过一次，还在缓存期内」
        r._reanchor()
        if r._pos is not r.api.pos:
            raise RuntimeError(
                "重挂之后 _pos 还是那个旧对象 —— 活引用断了就永远读一份冻住的数")
        if r._mm_ts != 0.0:
            raise RuntimeError(
                "重挂之后没清比对缓存（MISMATCH_GAP=5 秒），界面会慢一拍才显示已对齐")
        n0 = r.api.n
        r._reanchor()
        if r.api.n != n0:
            raise RuntimeError(
                f"POS_ANCHOR_SEC={POS_ANCHOR_SEC} 秒内重挂了两次（tick 是 0.8 秒一轮）")

        # `_mm_note`：只有**一直**不一致才落一行，而且要带账户接口的原始读数
        r._diag.rows.clear()
        r._mm_since, r._mm_logged = {}, set()
        row = {"symbol": "CFFEX.T2612", "target": 0, "actual": -2, "delta": 2,
               "managed": True, "t_long": 0, "t_short": 0, "a_long": 0,
               "a_short": 2, "locked": False}
        r._mm_note([row])
        if r._diag.rows:
            raise RuntimeError("刚出现的不一致就写日志了，live.log 会被刷满")
        r._mm_since["CFFEX.T2612"] = time.time() - MM_LOG_SEC - 1
        r._mm_note([row])
        if not r._diag.rows:
            raise RuntimeError(
                f"同一条不一致挂了 {MM_LOG_SEC:.0f} 秒以上，却没落 [MM] 诊断")
        last = r._diag.rows[-1]
        if "pos_long" not in last or "[自检户]" not in last:
            raise RuntimeError(f"[MM] 没带账户接口的原始读数 / 账户名：{last}")
        r._mm_note([])                  # 恢复一致
        if r._mm_since or r._mm_logged:
            raise RuntimeError("合约恢复一致后没清掉 [MM] 的记录，下次不会再写")

        # 真跑一遍 `mismatch()`：账户平 + 目标 0 必须是「一致」，账户 -2 必须报
        # （上面 ③ 把 `mismatch` 整个换成了替身，先撤掉）
        r.__dict__.pop("mismatch", None)
        r.__dict__.pop("_send_row", None)
        r._mm_since, r._mm_logged = {}, set()
        st2 = SlotState(cfg=SStore.StrategyConfig(sid="t2", name="T2", period="1h"))
        st2.running, st2.underlying, st2.target = True, "CFFEX.T2612", 0
        r._slots = {"t2": st2}
        r._pos, r._mm_ts = {}, 0.0
        if r.mismatch(fresh=True):
            raise RuntimeError("账户是平的、目标也是 0，却报不一致")
        r._pos = {"CFFEX.T2612": P(0, 2)}
        r._mm_ts = 0.0
        rows = r.mismatch(fresh=True)
        if len(rows) != 1 or int(rows[0].get("a_short") or 0) != 2:
            raise RuntimeError(f"账户空 2 手、目标 0 却没报出来：{rows}")
        raw = r._pos_raw("CFFEX.T2612")
        if "pos_short=2" not in raw:
            raise RuntimeError(f"`_pos_raw` 没把账户接口的原始读数打全：{raw}")

        # ---- ⑤ 柜台一条持仓都没回来：不许报「一致」，也不许对齐
        #
        # 用户 2026-11-23 报「融航的账户实际持仓有很多，策略理论持仓都为 0，
        # 可一键对齐这边显示都是对齐的」。真因不是比对算错：组合页那条连接
        # **一条持仓回调都没收到**，`_pos` 是空字典 —— 而 `mismatch()` 的合约
        # 集合 = 理论持仓 ∪ `_pos` 里有的合约，两边都空 ⇒「一个不一致都没有」。
        #
        # 空字典有两种含义，早先被混成了一种：**账户真没仓** / **柜台没回**。
        # 前者可以报「一致」，后者只能说「还没读到」，更不能拿它算差额
        # （会以为账户是空的，把整个理论仓位重新开一遍）。
        from ttrade.strategy_runner import POS_LOG_SEC

        class _BlindApi:
            """账户挂着，但持仓一次都没读回来（`positions_ready()` 为假）。"""

            last_error = "查持仓失败：[17] 查询未就绪"

            def positions_ready(self):
                return False

            def get_position(self):
                return {}

            def get_order(self):
                return {}

        class _OkApi(_BlindApi):
            def positions_ready(self):
                return True

        r._diag.rows.clear()
        r.api, r._pos = _BlindApi(), {}
        if r._pos_live():
            raise RuntimeError(
                "柜台一条持仓都没回来，`_pos_live()` 却是真 —— 界面会报「一致」")
        r.api = _OkApi()
        if not r._pos_live():
            raise RuntimeError("柜台说读到了、账户确实空仓，却仍被当成「没读到」")
        r.api = object()                 # 天勤：没有 `positions_ready` 这一说
        if not r._pos_live():
            raise RuntimeError("天勤账户被当成了「持仓没读到」")
        r.api, r._pos = _OkApi(), None   # 连持仓对象都没拿到（登录中）
        if r._pos_live():
            raise RuntimeError("`_pos is None` 时 `_pos_live()` 应该是假")

        # 比对为空但**没读到**时，`align()` 必须拒
        r.api, r._pos = _BlindApi(), {}
        r._align_ts = 0.0
        r._align_slot.pending.clear()
        r._align_slot.last_action = ""
        if r.align() != 0:
            raise RuntimeError("持仓一条都没读到，却让 align() 发出去了")
        if "读回来" not in " ".join(r._diag.rows):
            raise RuntimeError(f"拒绝对齐时没说清原因：{r._diag.rows}")

        # `[POS]` 诊断：等够 POS_LOG_SEC 才写一次，且把柜台的原始情况带上
        r._diag.rows.clear()
        r._pos_blind_at, r._pos_warned = 0.0, 0.0
        r._pos_note()
        if r._diag.rows:
            raise RuntimeError("刚连上就写 [POS]，live.log 会被刷满")
        r._pos_blind_at = time.time() - POS_LOG_SEC - 1
        r._pos_note()
        last = (r._diag.rows or [""])[-1]
        for need in ("[POS]", "positions_loaded", "查询未就绪", "[自检户]"):
            if need not in last:
                raise RuntimeError(f"[POS] 诊断缺 {need}：{last}")
        r._pos_note()                    # 刚抱怨过，不许连着写
        if len(r._diag.rows) != 1:
            raise RuntimeError("同一个「没读到」连着写了两行 [POS]")
        r.api = _OkApi()                 # 读回来了：计时清掉，下次真断了才算新的一段
        r._pos_note()
        if r._pos_blind_at or r._pos_warned:
            raise RuntimeError("持仓读回来之后没清 [POS] 的计时")

        # ---- ⑥ 界面「刷新」：重挂持仓引用 + 清掉比对缓存（2026-11-23 用户要求）
        #
        # 比对缓存 5 秒、持仓引用 2 秒才重挂一次。成交之后想马上确认「齐没齐」，
        # 等这两个节拍太难受，所以监控器上加了刷新按钮。它必须**绕过两重节流**，
        # 否则连点两下第二下什么都不做，用户会以为按钮坏了。
        r.api = _Api()
        r.api.pos = {"CFFEX.T2612": P(0, 2)}
        r._pos = {}
        r._pos_ts = time.time()          # 刚重挂过：`_reanchor()` 自己会跳过
        r._mm_ts = time.time()           # 缓存也在有效期内
        r._mm_cache = [{"symbol": "OLD"}]   # 缓存里是上一轮的旧结论
        r._slots = {"t2": st2}
        st2.target, st2.running = 0, True
        rows = r.refresh_now()
        if r._pos is not r.api.pos:
            raise RuntimeError("刷新没有重挂持仓引用（`_pos_ts` 没清零），"
                               "账户的活引用断了就永远刷不出新数")
        # 缓存被**重新算过**了：里面不再是那条 OLD。只清时间戳不清内容的话，
        # 下一次 `mismatch()` 仍会把旧结论端出来
        if not r._mm_cache or r._mm_cache[0].get("symbol") != "CFFEX.T2612":
            raise RuntimeError(f"刷新之后缓存还是旧的：{r._mm_cache}")
        if len(rows) != 1 or int(rows[0].get("a_short") or 0) != 2:
            raise RuntimeError(f"刷新返回的不是当下的不一致：{rows}")

        return ("超时窗口跟追单参数走（默认 30 秒 / wait 10 秒→50 秒 / 关掉追单不放大）；"
                "撤单后账本按账户实际回摆（平→0、有 3 手空→-3），based 不动；"
                "每个没发出去的合约都落一行原因，且日志带账户名；"
                "持仓引用每 2 秒重挂并清掉比对缓存，长期不一致落 [MM] 原始读数；"
                "柜台一条持仓都没读到时不报「一致」、不放开对齐，并落 [POS] 诊断；"
                "「刷新」能绕过两重节流立刻重读持仓并重算差额")

    check("一键对齐：超时窗口 / 撤单后账本回摆 / 没发出去要说原因",
          _align_timeout_resync)

    def _mon_blind():
        """监控器：持仓没从柜台读回来时，「账户实际 / 差额」必须是「—」不是 0。

        2026-11-23 融航实测：账户里明明有 `CZCE.SA701` 空 1 手，监控器那一行写
        「账户实际 0 / 差额 -1」。真因是柜台**一条持仓回包都没给**
        （`pump_queries()` 第一句在融航上抛 `AttributeError` 被吞），`_pos` 是空
        字典 —— 但界面把「空字典」读成了「账户没仓」。

        所以两件事必须分开判：

        - `account_ready`：账户接口挂上了（界面同时拿它判「在不在线」）
        - `pos_synced`：持仓**真读回来过一整轮**（`positions_loaded`，空仓空包
          也算读完、柜台拒答不算）

        没 `pos_synced` 的时候写 0 就是在撒谎：必须是「—」，差额也不能标红 ——
        红字的意思是「这一行真的差这么多」，读不到时没有这个结论。
        """
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])      # noqa: F841
        import gui.portfolio_page as PP

        page = PP.PortfolioPage()
        try:
            page._watch = set()          # 空名单 = 全部监控
            page._items = []
            page._rows = []
            page._slot_multi = {}
            page._slot_view = {}
            page._snaps = {
                # 读回来的账户：照常写数，不能被上面那条规则连坐
                "快期模拟": {
                    "account_ready": True, "pos_synced": True,
                    "mismatch": [{"symbol": "SHFE.pb2701", "target": 9,
                                  "actual": 9, "delta": 0, "managed": True,
                                  "a_long": 0, "a_short": 0}],
                },
                # 没读回来的：`actual` 是「空字典里查不到」的 0，不是真 0
                "融航zxqhmn01": {
                    "account_ready": True, "pos_synced": False,
                    "mismatch": [{"symbol": "CZCE.SA701", "target": -1,
                                  "actual": 0, "delta": -1, "managed": True,
                                  "a_long": 0, "a_short": 1}],
                },
            }
            page._mon_sig = None
            page._render()
            m = page.tbl_mon
            idx = {m.item(r, 0).text(): r for r in range(m.rowCount())}
            if set(idx) != {"快期模拟", "融航zxqhmn01"}:
                raise RuntimeError(f"监控器该有两行：{list(idx)}")

            # ① 读回来的那一行：数字照旧
            i = idx["快期模拟"]
            if (m.item(i, 3).text(), m.item(i, 4).text()) != ("9", "0"):
                raise RuntimeError(
                    f"读回来的账户数字被连坐了：{m.item(i, 3).text()} / "
                    f"{m.item(i, 4).text()}")

            # ② 没读回来的那一行：账户实际 / 差额都是「—」，且说明里写清不是 0
            i = idx["融航zxqhmn01"]
            got = (m.item(i, 2).text(), m.item(i, 3).text(), m.item(i, 4).text())
            if got != ("-1", "—", "—"):
                raise RuntimeError(
                    f"持仓没读回来时不能画 0：理论/实际/差额 = {got}"
                    "（应 -1 / — / —）")
            note = m.item(i, 5).text()
            for need in ("不是 0", "还不知道"):
                if need not in note:
                    raise RuntimeError(f"说明里要写清「{need}」：{note}")
            if m.item(i, 4).foreground().color().name() != PP.C_MUTED.lower():
                raise RuntimeError(
                    "差额是「—」的时候不能标红 —— 红字等于断言「真差了这么多」")

            # ③ 读回来之后必须**当场重绘**。「账户实际」的画法是 `pos_synced`
            # 决定的，它不进 `_mon_sig` 的话表格内容没变就不重建，用户得等下一次
            # 别的字段变化才看得到数 —— 表现就是「刷新了还是 0」
            page._snaps["融航zxqhmn01"]["pos_synced"] = True
            page._render()
            m = page.tbl_mon
            i = {m.item(r, 0).text(): r for r in range(m.rowCount())}["融航zxqhmn01"]
            got = (m.item(i, 3).text(), m.item(i, 4).text())
            if got != ("0", "平空1"):
                raise RuntimeError(
                    f"持仓读回来之后没重绘（`pos_synced` 没进 `_mon_sig`？）：{got}")
            if m.item(i, 4).foreground().color().name() == PP.C_MUTED.lower():
                raise RuntimeError("读回来之后差额不该还是灰的")

            # ④ 一条不一致都列不出来、又确实没读到 ⇒ 整条拦住说「还不知道」，
            # 绝不能落到下面那句「账户仓位与策略理论持仓一致」
            page.ctl._workers = {"融航zxqhmn01": object()}     # 假装在跑
            page._snaps = {"融航zxqhmn01": {"account_ready": True,
                                            "pos_synced": False,
                                            "mismatch": []}}
            page._mon_sig = None
            page._render()
            txt = page.lbl_mon.text()
            if "一致" in txt.split("（")[0] or "不作数" not in txt:
                raise RuntimeError(f"没读到持仓时说了「一致」：{txt}")
            if "融航zxqhmn01" not in txt:
                raise RuntimeError(f"要指名是哪个账户没读到：{txt}")
        finally:
            page.deleteLater()

        return ("持仓没从柜台读回来（`pos_synced` 假）时「账户实际 / 差额」写「—」"
                "并在说明里写「不是 0，是还不知道」、差额不标红；读回来的账户不受"
                "影响；`pos_synced` 进比对签名，读回来当场重绘成 0 / 平空1；"
                "一条都列不出来又没读到时不报「一致」而是「不作数」并指名账户")

    check("监控器：持仓没读回来时不许画 0", _mon_blind)

    def _order_ack():
        """报单发出去之后「柜台到底有没有这笔单」必须能看出来。

        2026-11-23 融航：一键对齐前后一共发了四笔 `CFFEX.TL2612`（13:00 的策略单、
        13:05 / 13:06 / 13:08 的对齐单），日志每笔都是「剩2 ALIVE」挂满 30 秒被撤，
        而交易页委托框里一笔都没有 —— 用户只能问「到底发出去没有」。

        三个判据，缺一个就回到「说不清」：

        1. `ReqOrderInsert` / `ReqOrderAction` 返回非 0 ⇒ **请求没进发送队列**，
           必须抛出来。以前四条路径（CTP 报单 / CTP 撤单 / 融航报单 / 融航撤单）
           谁都不看返回值；
        2. 委托行还是「乐观」的（柜台从没回过包去覆盖它）时，日志**不许写
           ALIVE** —— 那个词是「挂着等成交」，和「柜台没这笔单」是两件事；
        3. 8 秒还没回就要主动落一行，不能等 30 秒撤单时才发现
           （`ORD_ACK_SEC` 必须小于 `ALIGN_TIMEOUT`，否则两句判词撞车，
           30 秒那边先说话，真相永远看不到）。
        """
        import inspect
        import time as _t
        import types

        from ttrade import ctp_client as C
        from ttrade import rohon_client as R
        from ttrade import strategy_runner as S

        # ---- ① 四个发送口都必须看返回值
        for fn, label, key in (
            (C.TraderClient.place, "CTP 报单", "ReqOrderInsert"),
            (C.TraderClient.cancel, "CTP 撤单", "ReqOrderAction"),
            (R.RohonTraderClient.place, "融航报单", "reqOrderInsert"),
            (R.RohonTraderClient.cancel, "融航撤单", "reqOrderAction"),
        ):
            src = inspect.getsource(fn)
            # 注释行要剔掉：这些函数里都写着「为什么必须看返回值」，注释里
            # 就带着 `ReqOrderInsert` 几个字，直接搜会搜到那句注释
            lines = [ln.strip() for ln in src.splitlines()
                     if not ln.strip().startswith("#")]
            hits = [i for i, ln in enumerate(lines) if key in ln]
            if not hits:
                raise RuntimeError(f"{label} 里找不到 {key}")
            window = " ".join(lines[hits[0]:hits[0] + 4])
            if "raise" not in window and "last_error" not in window:
                raise RuntimeError(
                    f"{label} 发完 {key} 不看返回值 —— 没发出去会被当成发出去了")
        if "超限" not in C._req_why(-3) or "网络" not in C._req_why(-1):
            raise RuntimeError(f"`_req_why` 没翻出来：{C._req_why(-3)}")

        # ---- ②/③ 在途委托：乐观行不许写 ALIVE，超时要主动说
        if not (S.ORD_ACK_SEC < S.ALIGN_TIMEOUT):
            raise RuntimeError(
                f"ORD_ACK_SEC({S.ORD_ACK_SEC}) 必须小于 ALIGN_TIMEOUT"
                f"({S.ALIGN_TIMEOUT})，否则「柜台没回报」还没说出来就先撤单了")

        r = S.PortfolioRunner.__new__(S.PortfolioRunner)
        logs: list[str] = []
        r._log = lambda m: logs.append(str(m))
        r._diagw = lambda tag, m: logs.append(f"[{tag}] {m}")
        r._align_log = lambda m: logs.append(f"[ALIGN] {m}")
        r._ack_warned = set()
        r._align_slot = object()

        o = S.PendingOrder(order_id="12", symbol="CFFEX.TL2612", direction="SELL",
                           offset="OPEN", volume=2, price=118.05, ts=_t.time())
        r._orders = {"12": types.SimpleNamespace(
            status="ALIVE", volume_left=2, optimistic=True, ts=_t.time())}
        st = types.SimpleNamespace(pending=[o], title=lambda: "测试")

        detail = r._pending_detail(st)
        if "ALIVE" in detail[0]:
            raise RuntimeError(f"柜台没回过的委托被写成 ALIVE 了：{detail}")
        if "没回" not in detail[0]:
            raise RuntimeError(f"在途明细要写明「柜台没回」：{detail}")

        r._ack_note(st, o)                      # 刚发出去，还不该说话
        if logs:
            raise RuntimeError(f"{S.ORD_ACK_SEC:.0f} 秒内就报警了：{logs}")
        # 一句进界面日志、一句进 live.log（`_log` + `_diagw`），所以数的是
        # 「同一笔委托被提了几次」，不是列表长度
        r._orders["12"].ts = _t.time() - S.ORD_ACK_SEC - 1.0
        r._ack_note(st, o)
        said = [m for m in logs if "柜台一条委托回报都没有" in m]
        if len(said) != 2:
            raise RuntimeError(f"超时没落日志（或落得不对）：{logs}")
        r._ack_note(st, o)                      # 同一笔只报一次
        said = [m for m in logs if "柜台一条委托回报都没有" in m]
        if len(said) != 2:
            raise RuntimeError(f"同一笔委托报了两遍：{logs}")

        # 柜台真回了包（`optimistic` 被 `_fill_orders` 覆盖成 False）就不该再提
        r._orders["12"] = types.SimpleNamespace(
            status="ALIVE", volume_left=2, optimistic=False, ts=0.0)
        r._ack_warned.clear()
        logs.clear()
        r._ack_note(st, o)
        if logs:
            raise RuntimeError(f"柜台已经回了包还在喊「没回报」：{logs}")
        if "ALIVE" not in r._pending_detail(st)[0]:
            raise RuntimeError("有柜台回报的委托应当正常显示状态")

        return ("四个发送口（CTP/融航 的报单与撤单）都检查 `Req*` 返回值并翻成人话；"
                "柜台没回过的委托写「柜台 N 秒没回委托」而不是 ALIVE；"
                f"超过 {S.ORD_ACK_SEC:.0f} 秒主动落一行且每单只落一次；"
                "柜台回过包之后不再误报")

    check("报单回报：柜台没回这笔单时必须说出来", _order_ack)

    def _refresh_relogin():
        """「刷新」在账户没接上时必须真的去接一次；用户手动登出的不许偷偷接回。

        2026-11-23：用户在交易页把融航登出再登录，交易页显示已登录、持仓表也有
        数据，组合页监控器却一直写「账户未登录」，连点刷新都没用。

        真因是「没有账户」被混成了一种：

          - 用户自己点「登录 / 登出账户」登出 ⇒ 要尊重，不能偷偷登回；
          - 这条 runner 压根没接上（启动时接入失败 / 账户在交易页重新登录过）
            ⇒ `refresh()` 只是把空引用再挂一遍，**什么都没变**，按钮看着像坏的。

        所以：`user_detached` 分开这两种，刷新只对后者真的调 `attach()`。
        另外接 / 脱账户必须落一行 `[ACC]` 到 live.log —— `_log()` 只进界面那块
        文本框，关窗就没了，那天 live.log 里一条接/脱记录都没有，只能猜。
        """
        import types

        from ttrade import strategy_runner as S

        # ---- ① runner 侧：两个标记 + 落盘
        logs: list[tuple] = []
        r = S.PortfolioRunner.__new__(S.PortfolioRunner)
        r.api, r._pos, r._orders = object(), {}, {}
        r._slots = {}
        r._align_slot = types.SimpleNamespace(pending=[])
        r._detached_by_user = False
        r._log = lambda m: None
        r._diagw = lambda tag, m: logs.append((tag, m))
        r.detach_account()
        if not r._detached_by_user:
            raise RuntimeError("手动登出没打上 `user_detached` —— 刷新会把它偷偷登回去")
        if r.api is not None or r._pos is not None:
            raise RuntimeError("登出之后 api / _pos 没清干净")
        if not any(t == "ACC" and "登出" in m for t, m in logs):
            raise RuntimeError(f"接/脱账户没落盘（live.log 里看不到）：{logs}")

        # ---- ② 界面侧：只有「没接上」才真的 attach
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])      # noqa: F841
        import gui.portfolio_page as PP

        class _Ctl:
            def __init__(self):
                self.calls: list[tuple[str, str]] = []

            @property
            def running(self) -> bool:
                return True

            def keys(self):
                return ["融航zxqhmn01"]

            def attach(self, k, sp):
                self.calls.append(("attach", k))
                return True

            def refresh(self, k):
                self.calls.append(("refresh", k))
                return True

        page = PP.PortfolioPage()
        try:
            page.ctl = _Ctl()
            page._watch = set()
            page._spec = lambda k: object()        # 磁盘上还有这个账户
            page._items, page._rows = [], []
            page._snaps = {"融航zxqhmn01": {"account_ready": False,
                                            "user_detached": False,
                                            "mismatch": []}}
            page._refresh_monitor()
            if not page.ctl.calls or page.ctl.calls[0][0] != "attach":
                raise RuntimeError(
                    f"没接上的账户，刷新应该真去接一次：{page.ctl.calls}")

            page.ctl.calls.clear()
            page._snaps["融航zxqhmn01"]["user_detached"] = True
            page._refresh_monitor()
            if any(c[0] == "attach" for c in page.ctl.calls):
                raise RuntimeError(
                    f"用户手动登出的账户被「刷新」偷偷登回去了：{page.ctl.calls}")

            # 接上了就只重读，不再 attach
            page.ctl.calls.clear()
            page._snaps["融航zxqhmn01"].update(account_ready=True,
                                               user_detached=False)
            page._refresh_monitor()
            if page.ctl.calls != [("refresh", "融航zxqhmn01")]:
                raise RuntimeError(f"已经接上的账户不该再 attach：{page.ctl.calls}")
        finally:
            page.deleteLater()

        return ("接/脱账户落 `[ACC]` 到 live.log；手动登出打 `user_detached`，"
                "「刷新」不会把它偷偷登回去；没接上的账户点「刷新」会真的 attach，"
                "已经接上的只重读持仓不重连")

    check("刷新：账户没接上时要真的去接一次", _refresh_relogin)

    def _pos_round():
        """融航持仓查询：**按 RequestID 归轮 + 开轮清一次**，不许被空包刷成空表。

        2026-11-23 下午的真事：交易页持仓表有 7 个合约（IM2612 多1、TL2612 空2、
        SA701 空2、sn2612 空2…），组合页监控器却写「账户实际 0」，于是以为
        对齐没生效、反复发单。真因是融航那条连接的 `positions` 被刷成了空表，
        而且 `positions_loaded=True` —— 连「还没读到」都不说。

        🔴 2026-11-23 用户报「融航账户有时候显示正确的持仓，有时会少一些持仓
        品种」（那次少的是 TL2612，占用保证金总额两张截图一模一样 ⇒ 柜台没少
        给仓，是这张表**在收的过程中被读走了**）。

        真因：早先是「开轮就地 `clear()`、之后原地累加」。从清表到最后一包之间，
        任何一次 `snapshot()` 读到的都是**半份持仓** —— 5 个合约掉成 4 个，掉的
        那个恰好是还没到的那一包。窗口长短取决于柜台逐包推送的节奏，所以表现成
        「有时候对、有时候少」；交易页和组合页各自采样，于是两边结论还不一致。
        更糟的是 `positions_loaded` 一直是真 ⇒ 引擎拿半份表去 `mismatch()`，
        缺的合约被判成「账户没有」⇒ 一键对齐再发一笔单。

        现在的判据：`_pos_stage` 收，收满（`last`）才**整份替换** `positions`。
        清表只和替换同一刻发生，外界永远读不到半份 —— 本块第 ② 条专门钉这个。

        另外四条：
          - 按 RequestID 归轮：上一轮迟到的尾巴不许污染新一轮；
          - 同一轮只提交一次：多余的空终止包不许把表刷成空（老写法就死在这）；
          - 有开头没结尾的轮（柜台丢包）只许计数（`_pos_dropped`），不许动表；
          - 上期所同一方向会推今仓 / 昨仓**两行**，六个数必须 `+=`，赋值会把
            前一行盖掉（持仓手数少一半）。
        """
        import ast
        import inspect
        import textwrap
        import threading
        import time
        import types

        from ttrade import rohon_client as R

        # 从 `_make_td` 的源码里把回包函数抽出来（本机没装 vnpy_rohon，
        # 造不出真的 `_TdApi`，但函数体本身和厂商库无关）
        tree = ast.parse(textwrap.dedent(inspect.getsource(R._make_td)))
        node = None
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == "onRspQryInvestorPosition":
                node = n
                break
        if node is None:
            raise RuntimeError("`_make_td` 里找不到 onRspQryInvestorPosition")

        class _Outer:
            pass

        outer = _Outer()
        outer._lock = threading.Lock()
        outer._req_lock = threading.Lock()
        outer.positions = {}
        outer._pos_stage = None
        outer._pos_stage_req = -1
        outer._pos_stage_rows = 0
        outer._pos_done_req = -1
        outer._pos_dropped = 0
        outer._pos_req = 0
        outer._pos_rows = 0
        outer.pos_at = 0.0
        outer.positions_loaded = False
        outer.last_error = ""
        outer.orders = {}
        outer.trades = {}
        outer.account = {}
        outer.logged_in = threading.Event()
        outer.logged_in.set()
        outer.settled = threading.Event()
        outer._req = 0
        outer._next_req = lambda: setattr(outer, "_req", outer._req + 1) or outer._req
        outer.cfg = types.SimpleNamespace(broker_id="9999", user_id="tester")
        outer._qry_note = lambda what, n: None

        sent: list[int] = []

        class _Api:
            def reqQryInvestorPosition(self, req, rid):
                sent.append(rid)
                return 0

        outer.api = _Api()
        start_round = R.RohonTraderClient.query_positions

        ns = dict(vars(R))
        ns["outer"] = outer
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],
                                                          type_ignores=[])),
                     "rohon_client.py", "exec"), ns)
        _raw = ns["onRspQryInvestorPosition"]
        # 它原本是 `_Td` 上的方法（源码在 `class _Td` 里），抽出来之后第一个
        # 参数还是 `self` —— 这里直接绑个假 self
        cb = lambda data, error, reqid, last: _raw(None, data, error, reqid, last)  # noqa: E731

        def row(ins, side, n, yd=0, last=False, rid=None, cost=0.0):
            cb({"InstrumentID": ins, "PosiDirection": side, "Position": n,
                "YdPosition": yd, "TodayPosition": n - yd, "OpenCost": cost,
                "PositionProfit": 0.0, "UseMargin": 0.0},
               None, rid if rid is not None else outer._pos_req, last)

        # ---- ① 一轮正常：两个合约，多空各一行
        start_round(outer)
        rid1 = outer._pos_req
        if sent != [rid1] or outer._pos_stage is not None:
            raise RuntimeError(f"开轮没记 reqid / 没开 stage：{sent} {outer._pos_stage}")
        row("IM2612", "2", 1, last=False)
        if outer.positions:
            raise RuntimeError("第一条回包到了就往外写了半份持仓（旧写法就是这样）")
        row("TL2612", "3", 2, last=True)
        if not outer.positions_loaded:
            raise RuntimeError("`bIsLast` 到了却没记「读完了」")
        if (outer.positions["IM2612"]["long"], outer.positions["TL2612"]["short"]) != (1, 2):
            raise RuntimeError(f"一轮持仓没累加对：{outer.positions}")
        if outer._pos_rows != 2:
            raise RuntimeError(f"回包行数记错了：{outer._pos_rows}")
        if not outer.pos_at:
            raise RuntimeError("整轮收满了却没记 `pos_at`（`POS_STALE_SEC` 判不了新旧）")
        if outer._pos_stage is not None:
            raise RuntimeError("提交之后 stage 没释放，下一轮会往旧 stage 里继续加")

        # ---- ② 🔴 半份持仓不许外泄（2026-11-23「有时 5 行、有时 4 行」）
        # 新一轮只喂第一条，对外那份必须还是**上一轮收全的整份**
        start_round(outer)
        row("SA701", "3", 3, last=False)
        snap = R.RohonTraderClient.snapshot(outer)["positions"]
        if set(snap) != {"IM2612", "TL2612"}:
            raise RuntimeError(
                f"新一轮收到一半就把表换掉了 —— 界面这一瞬间会少一个合约：{list(snap)}")
        if snap["TL2612"]["short"] != 2:
            raise RuntimeError(f"半份窗口里把上一轮的合约改坏了：{snap}")
        # 收满才换
        row("sn2612", "3", 1, last=True)
        if set(outer.positions) != {"SA701", "sn2612"}:
            raise RuntimeError(f"整轮收满没换上新的：{list(outer.positions)}")

        # ---- ③ 多余的 `last` 空终止包：**不许**把表刷成空
        cb(None, None, rid1, True)
        if not outer.positions:
            raise RuntimeError("多一个空 `last` 包就把持仓整表清空了（旧写法就是这个）")
        if not outer.positions_loaded:
            raise RuntimeError("空包之后「读完了」被清掉了")

        # ---- ④ 上一轮迟到的回包：不许污染新一轮
        start_round(outer)
        rid_prev = outer._pos_req
        row("SA701", "3", 3, last=False)
        row("TL2612", "3", 2, last=True)                 # 这一轮收满
        start_round(outer)                               # 下一轮开跑
        row("cu2701", "2", 1, last=False)
        row("TL2612", "3", 2, last=True, rid=rid_prev)   # 上一轮的尾巴
        snap = R.RohonTraderClient.snapshot(outer)["positions"]
        if set(snap) != {"SA701", "TL2612"}:
            raise RuntimeError(f"迟到的旧回包污染了新一轮：{list(snap)}")
        row("cu2701", "2", 1, last=True)                 # 本轮收满
        if set(outer.positions) != {"cu2701"}:
            raise RuntimeError(f"新一轮收满后应该是 cu2701：{list(outer.positions)}")

        # ---- ⑤ 有开头没结尾的轮（柜台丢包）：只计数，不许动表
        start_round(outer)
        row("SA701", "3", 3, last=False)                 # 只来一条，没有 `last`
        good = dict(outer.positions)
        drop0 = outer._pos_dropped
        start_round(outer)                               # 下一拍接着发
        if outer._pos_dropped != drop0 + 1:
            raise RuntimeError(f"没收全的轮没计数：{outer._pos_dropped} vs {drop0}")
        snap = R.RohonTraderClient.snapshot(outer)["positions"]
        if snap != good:
            raise RuntimeError(
                f"没收全的那轮把对外那份改了（会少合约）：{list(snap)}")

        # ---- ⑥ 真没仓：一条空包 + last ⇒ 空表且「读完了」
        start_round(outer)
        cb(None, None, outer._pos_req, True)
        if outer.positions or not outer.positions_loaded:
            raise RuntimeError(f"空仓这一轮判错了：{outer.positions} / "
                               f"loaded={outer.positions_loaded}")

        # ---- ⑦ 上期所：同一方向推今仓 + 昨仓两行，必须相加
        start_round(outer)
        row("sn2612", "2", 1, yd=0, last=False)        # 今仓 1
        row("sn2612", "2", 2, yd=2, last=True)         # 昨仓 2
        got = outer.positions["sn2612"]
        if (got["long"], got["long_today"], got["long_yd"]) != (3, 1, 2):
            raise RuntimeError(f"上期所今昨两行没累加（赋值会把前一行盖掉）：{got}")

        # ---- ⑧ 柜台报错：不许当成「查完了」
        start_round(outer)
        before = outer.positions_loaded
        outer.positions_loaded = False
        cb(None, {"ErrorID": 3, "ErrorMsg": "每秒查询超限"}, outer._pos_req, True)
        if outer.positions_loaded:
            raise RuntimeError("查持仓报错却记成「读完了」")
        if "每秒查询超限" not in outer.last_error:
            raise RuntimeError(f"查持仓失败的原因没记下来：{outer.last_error!r}")
        outer.positions_loaded = before

        return ("持仓整轮收满才提交（新一轮收到一半时对外那份仍是上一轮的整份，"
                "界面不会少一个合约）；按 RequestID 归轮，迟到的旧回包不污染；"
                "多余的空 `last` 包不会把表刷成空；没收全的轮只计数不动表；"
                "真没仓时记「读完了」；上期所今昨两行累加（1+2=3）；"
                "柜台报错不冒充「查完了」")

    check("融航持仓：整轮收满才提交，半份不许外泄", _pos_round)

    def _load_no_trade():
        """加载策略时**不许**按台账自动交易（2026-11-17 用户要求）。

        用户原话：「还是先回滚恢复到原来的台账的机制吧，只是加一个策略加载的时候
        不要自动去根据台账去做交易。」

        老痛点：`_add_one` 读回账本 → `_rebalance` 拿它跟 `target` 比 → 差额当场
        变委托。用户刚在界面上看到「仓位异常」，单子已经发出去了。

        现在的判据：**加载后目标没变过，就一手都不发**（账本/账户差多少都不管，
        交给一键对齐）；只有信号真的把 `target` 改了，才按台账调仓。
        换月不走这条 —— 老合约该平还得平。

        2026-11-19 补两个例外（用户连续两天报「平仓/开仓没发出去」）：
        目标没变但要**减仓**时照发；账本与账户都空时要**建仓**也照发 ——
        这两种都没有「重复开一笔」的风险，挡着等于把当天的信号吞掉。

        顺带钉住台账的另一半：账本比账户多时要按账户收窄（只减不增）。
        """
        from ttrade import strategy_store as SStore
        from ttrade.strategy_runner import PortfolioRunner, SlotState

        class _Spec:
            def label(self):
                return "__selftest__"

        class _P:
            def __init__(self, lo=0, sh=0):
                self.pos_long, self.pos_short = lo, sh
                for side, v in (("long", lo), ("short", sh)):
                    setattr(self, "pos_%s_his" % side, v)
                    setattr(self, "pos_%s_today" % side, 0)
                    setattr(self, "volume_%s_his" % side, v)
                    setattr(self, "volume_%s_today" % side, 0)

        def _runner():
            r = PortfolioRunner.__new__(PortfolioRunner)
            r._slots, r._groups, r._parents = {}, {}, {}
            r._mm_cache, r._mm_ts = [], 0.0
            r._align_ts = 0.0
            r._pos = {}
            r.api = object()
            r.spec = _Spec()
            r._fresh_book = False       # 不认领账户里已有的仓
            r._adopted = set()
            r._gap_note = {}
            r._log = lambda *_a, **_k: None
            r._align_slot = SlotState(cfg=SStore.StrategyConfig(
                sid="__align__", name="一键对齐", period="1d"))
            r.book = []

            def _send(st, sym, delta, closeable=None):
                r.book.append((sym, int(delta), closeable))
                st.pending.append("x")
                return 1

            r._send = _send
            return r

        def _slot(r, sid="a", under="SHFE.rb2701", target=5):
            st = SlotState(cfg=SStore.StrategyConfig(
                sid=sid, name=sid, symbol="RB", exchange="SHFE", period="1d"))
            st.running, st.underlying, st.target = True, under, int(target)
            r._slots[sid] = st
            return st

        # ① 加载首轮：目标 +5、账本空、账户上站着 10 手（多半是别人的）
        #    ⇒ 一手都不许发，只把「这时的目标」记成基准
        r = _runner()
        st = _slot(r, target=5)
        st.held = {}
        r._pos = {"SHFE.rb2701": _P(10, 0)}
        r._rebalance(st)
        if r.book:
            raise RuntimeError(f"加载首轮就按台账发单了：{r.book}")
        if not st.based or st.based_target != 5:
            raise RuntimeError(f"加载基准没记上：based={st.based} "
                               f"based_target={st.based_target}")

        # ② 目标没变：第二轮、第三轮仍然一手不发（这正是老行为发单的地方）
        st.pending.clear()
        r.book.clear()
        r._rebalance(st)
        r._rebalance(st)
        if r.book:
            raise RuntimeError(f"目标没变还在发单：{r.book}")

        # ③ 信号把目标改成 -3：这时才按台账调仓（账本 0 ⇒ 卖 3 手）
        st.target = -3
        r._rebalance(st)
        if not r.book or r.book[0][1] != -3:
            raise RuntimeError(f"目标变了该按台账卖 3 手：{r.book}")

        # ④ 换月**不受这条限制**：加载首轮也要把老合约平掉
        r2 = _runner()
        st2 = _slot(r2, target=2)
        st2.held = {"SHFE.rb2701": 3}
        st2.retired = ["SHFE.rb2701"]
        st2.underlying = "SHFE.rb2705"
        r2._pos = {"SHFE.rb2701": _P(3, 0)}
        r2._rebalance(st2)
        if not r2.book or r2.book[0][0] != "SHFE.rb2701" or r2.book[0][1] != -3:
            raise RuntimeError(f"换月平老合约被加载门挡住了：{r2.book}")

        # ⑤ 台账的「只减不增」：账本 4 手、账户里只剩 2 手（被别人平掉了）
        #    ⇒ 只能按 2 手发单。照着账本平 4 手会被交易所打回，`_legs` 还会
        #    把平不掉的量当成开仓，反手开出一笔没道理的反向仓
        r3 = _runner()
        st3 = _slot(r3, target=0)
        st3.based = True
        st3.based_target = 5
        st3.held = {"SHFE.rb2701": 4}
        r3._pos = {"SHFE.rb2701": _P(2, 0)}
        r3._rebalance(st3)
        if not r3.book or r3.book[0][1] != -2:
            raise RuntimeError(f"账本比账户多时要收窄到可平量：{r3.book}")

        # ⑥ 目标没变、但账本有仓、信号要平 ⇒ **必须发单**（2026-11-19 用户报的）
        #    昨天最后一根 bar 出的平仓信号，执行时点就是今天开盘：加载时算出来的
        #    `target` 已经是 0，正好等于 `based_target`。老写法在这里 return，
        #    这笔平仓永远发不出去，账户里的仓一直挂着，只有一键对齐能平
        r4 = _runner()
        st4 = _slot(r4, target=0)
        st4.based = True
        st4.based_target = 0
        st4.held = {"SHFE.rb2701": -2}
        r4._pos = {"SHFE.rb2701": _P(0, 2)}
        r4._rebalance(st4)
        if not r4.book or r4.book[0][1] != 2:
            raise RuntimeError(f"目标没变但该平仓时没发单：{r4.book}")

        # ⑦a 目标没变、账本空、账户里也没有这个合约 ⇒ **纯建仓，必须发**
        #     2026-11-19 用户截图那一行：目标 -1、账本 0，工具提示叫他「点一键
        #     对齐」。这不是历史差额，是今天的信号，没有仓可以重复开
        r5 = _runner()
        st5 = _slot(r5, target=-2)
        st5.based = True
        st5.based_target = -2
        st5.held = {}
        r5._pos = {}
        r5._rebalance(st5)
        if not r5.book or r5.book[0][1] != -2:
            raise RuntimeError(f"账本空、账户也空时该建仓却没发：{r5.book}")

        # ⑦b 但账户上多出 1 手没人认领（重启丢了账本）⇒ 仍然挡住，不许再开一笔
        r6 = _runner()
        st6 = _slot(r6, target=-2)
        st6.based = True
        st6.based_target = -2
        st6.held = {}
        r6._pos = {"SHFE.rb2701": _P(0, 1)}
        r6._rebalance(st6)
        if r6.book:
            raise RuntimeError(f"账户上有无主的仓时仍按目标开仓了：{r6.book}")

        # ⑦c 账户持仓还没拉回来（_pos 为 None）⇒ 一律不发，不能凭空的账本开仓
        r7 = _runner()
        st7 = _slot(r7, target=-2)
        st7.based = True
        st7.based_target = -2
        st7.held = {}
        r7._pos = None
        r7._rebalance(st7)
        if r7.book:
            raise RuntimeError(f"持仓还没就绪就发单了：{r7.book}")

        return ("加载首轮不发单、目标不变不加仓、目标改了才按台账调；"
                "目标不变但该减仓时照发；账本与账户都空时纯建仓照发；"
                "换月不受限；账本比账户多时按可平量收窄")

    check("加载策略时不按台账自动交易", _load_no_trade)

    def _data_gap_gate():
        """K 线缺交易日 / 补库静默成功 —— 2026-11-17 的现场。

        用户报三件事：IF 的 5m 历史还停在 11-10；同一个策略在两个账户上一个
        1 手、一个 0 手；回测里却还持仓。

        根因不在策略：本地库缺了 11-11~11-17，`merge_live` 把实时段直接贴在
        11-10 后面 —— 索引看着连续，中间五个交易日凭空消失。指标（60 根动量、
        上一交易日波幅、ATR）跨着断口算，信号就和回测对不上。柜台账户更容易撞：
        它没有回填历史的能力，序列全靠本地库里种进去的 300 根。
        """
        import inspect

        import pandas as pd

        from ttrade import strategy_runner as R
        from ttrade import topup, tqstore

        cal = sorted(tqstore.calendar())
        if len(cal) < 10:
            raise RuntimeError("库里没有日线日历，这条自检跑不了")

        def mk(i):
            return pd.DataFrame({"close": [1.0] * len(i)}, index=i)

        def at(days):
            return pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta(hours=10)
                                     for d in days])

        # ① 完整的一段不能报缺口；抽掉一天必须报出来
        tail = cal[-6:]
        if R.hole_days(mk(at(tail))):
            raise RuntimeError(f"完整的 K 线被判成缺口：{R.hole_days(mk(at(tail)))}")
        miss = R.hole_days(mk(at(tail[:2] + tail[3:])))
        if miss != [tail[2]]:
            raise RuntimeError(f"缺一天没判出来：{miss} vs {[tail[2]]}")

        # ② 空表 / 只有索引的帧：空表返回空（没得判），只有索引也要照判
        if R.hole_days(pd.DataFrame()):
            raise RuntimeError("空表被判成缺口")
        if not R.hole_days(pd.DataFrame(index=at(tail[:2] + tail[3:]))):
            raise RuntimeError("只有索引的帧漏判了（`DataFrame.empty` 对 0 列也是 True）")

        # ③ 🔴 长假不能误判：日历里间隔最大的那两天之间没有任何交易日
        #    （春节 / 国庆），拿自然日判会把它们当成缺口，一年两次把策略停掉
        best = max(range(len(cal) - 1), key=lambda i: (cal[i + 1] - cal[i]).days)
        d0, d1 = cal[best], cal[best + 1]
        if (d1 - d0).days < 3:
            raise RuntimeError("日历里没有长假段，这条反证没意义")
        if R.hole_days(mk(at([d0, d1]))):
            raise RuntimeError(f"长假被当成缺口（{d0} → {d1} 共 {(d1 - d0).days} 天）")

        # ④ 补库：整段落在交易时段外（午休 / 收盘快照）时**不能当成功**。
        #    当成功就把这个品种永久钉在旧数据上，一句话都不说
        serial = pd.DataFrame({
            # 天勤给的是 UTC 纳秒，北京 11:45（CFFEX 午休）＝ UTC 03:45
            "datetime": [pd.Timestamp("2026-11-17 03:45").value],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1.0], "open_oi": [1.0], "close_oi": [1.0],
        })
        r = topup.save(serial, "IF", "CFFEX", "5m")
        if r["ok"] or not r["error"]:
            raise RuntimeError(f"全是时段外的 bar 竟被当成补库成功：{r}")

        # ⑤ 停机槽位也要继续补库（源码判据：这里没有真连接，只能看结构）
        src = inspect.getsource(R.PortfolioRunner.tick)
        i = src.find("if not st.running:")
        if i < 0 or "_top_up" not in src[i:i + 500]:
            raise RuntimeError("停机后不再补库：库会一直停在旧数据上（IF 就是这样停在 11-10）")

        # ⑥ bar 数不足要写出来，不许静默 return（否则界面上只剩「理论 0 手」）
        src2 = inspect.getsource(R.PortfolioRunner._refresh_signal)
        j = src2.find("MIN_BARS")
        if j < 0 or "last_action" not in src2[j:j + 500]:
            raise RuntimeError("bar 数不足时又静默 return 了，界面上看不出原因")

        # ⑦ 🔴 增量窗口的起点就一条：**现有数据到哪一刻，就从那儿补到此刻**。
        #    老写法 `max(hard_start, last-3d)` 会被「回溯区间」夹住 —— 定时任务
        #    固定 days=1（hard_start = 昨天），库尾停在 11-10 时算出来是「从
        #    11-16 开始补」，中间那截永远补不上，而日志照样报「+14 行」。
        #    这里把下载函数换掉，只看它算出来的窗口。
        from datetime import date, timedelta

        from ttrade import tqfetch as TQ

        seen: list = []
        keep = (TQ.download_many, TQ.can_download, tqstore.last_dt,
                tqstore.first_dt, tqstore.load_switch, tqstore.miss_days)
        d_last = cal[-4]

        def fake_dl(api, reqs, period, root, label, max_parallel=8):
            seen.extend(reqs)
            return {}

        try:
            TQ.download_many = fake_dl
            TQ.can_download = lambda api: True
            tqstore.last_dt = lambda s, e, p: pd.Timestamp(d_last) + pd.Timedelta(hours=15)
            tqstore.first_dt = lambda s, e, p: pd.Timestamp(cal[0])
            tqstore.load_switch = lambda s, e: pd.DataFrame()
            tqstore.miss_days = lambda s, e, p, c=None: []
            TQ.write_main_many(None, [("IF", "CFFEX")], "5m", 0, days=1)
        finally:
            (TQ.download_many, TQ.can_download, tqstore.last_dt,
             tqstore.first_dt, tqstore.load_switch, tqstore.miss_days) = keep
        if not seen:
            raise RuntimeError("增量窗口一个都没算出来")
        got = seen[0][1]
        want = d_last - timedelta(days=3)
        if got != want:
            raise RuntimeError(
                f"增量起点被回溯区间夹住了：算成 {got}，应当是库尾往前 3 天的 {want}"
                f"（今天 {date.today()}，days=1 ⇒ 老写法会算成 {date.today() - timedelta(days=1)}）")

        # ⑧ 库尾之前中间缺的那几天：从**最早缺的那天**补起，一次覆盖洞和尾巴
        seen.clear()
        d_hole = cal[-12]
        try:
            TQ.download_many = fake_dl
            TQ.can_download = lambda api: True
            tqstore.last_dt = lambda s, e, p: pd.Timestamp(d_last) + pd.Timedelta(hours=15)
            tqstore.first_dt = lambda s, e, p: pd.Timestamp(cal[0])
            tqstore.load_switch = lambda s, e: pd.DataFrame()
            tqstore.miss_days = lambda s, e, p, c=None: [d_hole]
            TQ.write_main_many(None, [("IF", "CFFEX")], "5m", 0, days=1)
        finally:
            (TQ.download_many, TQ.can_download, tqstore.last_dt,
             tqstore.first_dt, tqstore.load_switch, tqstore.miss_days) = keep
        if not seen or seen[0][1] != d_hole:
            raise RuntimeError(
                f"中间有洞时没从洞的起点补起：算出 {seen[0][1] if seen else None}，"
                f"应当是 {d_hole}")

        return ("缺口按交易日历判（长假不误报、0 列的帧也照判）；"
                "补库不再静默成功；停机仍补库；bar 不足会写明原因；"
                "增量起点 = 库尾/最早的洞 → 此刻，不受回溯区间夹取")

    check("K 线缺交易日 / 补库静默成功", _data_gap_gate)

    def _stale_tail_gate():
        """帧**尾**整体落后 —— 2026-11-19 的现场（cu 报 -1 手，回测是 0）。

        `hole_days` 只查帧**内部**有没有断口。本地 5m 库两天没更新、天勤序列又
        刚往回补（3001 根里只有 1 根比库新）时，合并出来的帧是「看着连续的旧数据」：
        末根 11-17 14:55，和实时段最后一根同一根，一根洞都没有。信号就在一份两天前
        的数据上算出来了。

        判据改成跟「按交易时段本该走完的最后一根」比。这里验算期望值本身。
        """
        import pandas as pd

        from ttrade import strategy_runner as R

        def exp(sym, ex, per, ts):
            v = R.expected_last_bar(sym, ex, per, ts)
            return None if v is None else pd.Timestamp(v)

        # ① 收盘之后：期望值是当天收盘前最后一根的起点（cu 日盘 15:00 收，
        #    5m 最后一根 14:55）。这条正是漏掉的那一档
        got = exp("CU", "SHFE", "5m", "2026-11-19 17:40")
        if got != pd.Timestamp("2026-11-19 14:55"):
            raise RuntimeError(f"收盘后期望帧尾算错：{got}")

        # ② 午休：期望值是上午收盘那根（11:25），不是 14:55
        got = exp("CU", "SHFE", "5m", "2026-11-19 11:31")
        if got != pd.Timestamp("2026-11-19 11:25"):
            raise RuntimeError(f"午休期望帧尾算错：{got}")

        # ③ 没有夜盘的品种（IC）：收盘后同上，跨零点后仍是当天 14:55
        got = exp("IC", "CFFEX", "5m", "2026-11-20 01:10")
        if got != pd.Timestamp("2026-11-19 14:55"):
            raise RuntimeError(f"无夜盘品种跨零点算错：{got}")

        # ④ 日线 / 解析不出时段的品种：返回 None（调用方不判，宁可不判别误判）
        if R.expected_last_bar("CU", "SHFE", "1d_tq", "2026-11-19 17:40") is not None:
            raise RuntimeError("日线周期不该按分钟判帧尾")
        if R.expected_last_bar("600519", "SSE", "5m", "2026-11-19 17:40") is not None:
            raise RuntimeError("A 股没有期货时段表，不该判")

        # ④b **夜盘中**（21:00→02:30 的段里）：期望值是刚走完的那根，不是凌晨
        #     02:25。老写法对跨零点的段把 end 当成「今天 02:30」，22:00 那一刻
        #     两个候选都不满足 `last <= now`，退回去拿昨天的 02:25 —— 守卫在
        #     整段夜盘里形同虚设，而这正是 CTA 最活跃的一段
        got = exp("CU", "SHFE", "5m", "2026-11-19 22:00")
        if got != pd.Timestamp("2026-11-19 21:55"):
            raise RuntimeError(f"夜盘中期望帧尾算错：{got}（应当 21:55）")
        got = exp("CU", "SHFE", "5m", "2026-11-20 01:02")
        if got != pd.Timestamp("2026-11-20 00:55"):
            raise RuntimeError(f"夜盘跨零点后期望帧尾算错：{got}（应当 00:55）")

        # ⑤ 判据本身：停两天要拦得住，只落后一根（当天 14:55）不能拦
        e = exp("CU", "SHFE", "5m", "2026-11-19 17:40")
        step = pd.Timedelta(seconds=R.RT_SEC["5m"])
        stale = pd.Timestamp("2026-11-17 14:55")
        if not (stale < e - step):
            raise RuntimeError("帧停在前两天却没被判成落后")
        if pd.Timestamp("2026-11-19 14:55") < e - step:
            raise RuntimeError("当根就被判成落后，会误伤正常信号")

        return ("帧尾按交易时段判落后：收盘后要 14:55、午休要 11:25、"
                "跨零点/无夜盘都对；日线与 A 股不判；停两天拦住、当根不误伤")

    check("行情帧尾部落后（本地库没补上）", _stale_tail_gate)

    def _half_bar_three_places():
        """半根 bar：补库入口、实时合并、算信号切尾**三处同一条口径**。

        2026-11-19 在库里实测到 4 根 cu 的残缺 bar（11-03 21:55 / 22:05 / 23:05、
        11-04 10:55）：close 与天勤差 10~70 元、成交量只有一半。成因是补库的判据
        写成 `index[-1] > now` —— 天勤的时间戳是 bar 的**开始**时间，这个式子恒为
        假，正在走的那根每次都写进去了。

        危害不是「多一根」：它卡在中间，之后增量只追加更晚的，谁也修不好；
        `iloc[:-1]` 也只切最后一行，切不到它。回测（干净库）和实盘（带着半根）
        于是长期差一根的 high/low/close。
        """
        import pandas as pd

        from ttrade import topup
        from ttrade.tqlive import merge_live

        idx = pd.date_range("2026-11-19 14:30", periods=6, freq="5min")
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 10,
             "open_oi": 0.0, "close_oi": 0.0}, index=idx)

        # ① 盘中 14:52：最后一根 14:50 还在走 → 丢掉；14:45 及更早保留
        got = topup.drop_live_tail(df, "CU", "SHFE", "5m", ts="2026-11-19 14:52")
        if got.index[-1] != pd.Timestamp("2026-11-19 14:45"):
            raise RuntimeError(
                f"盘中没丢掉正在走的那根：留下 {got.index[-1]}（应当 14:45）")
        # ② 收盘后 17:40：14:55 那根已经走完，不能丢（老写法把它丢了）
        got = topup.drop_live_tail(df, "CU", "SHFE", "5m", ts="2026-11-19 17:40")
        if len(got) != len(df):
            raise RuntimeError(f"收盘后把走完的 bar 丢了：{len(got)}/{len(df)}")
        # ③ 日线：11-19 收盘后那根是完整的；盘中 10:00 那根还在走
        d = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                          "volume": 10, "open_oi": 0.0, "close_oi": 0.0},
                         index=pd.DatetimeIndex(["2026-11-17", "2026-11-18",
                                                 "2026-11-19"]))
        got = topup.drop_live_tail(d, "CU", "SHFE", "1d", ts="2026-11-19 17:40")
        if got.index[-1] != pd.Timestamp("2026-11-19"):
            raise RuntimeError(
                f"日线收盘后仍在用前一天：{got.index[-1]}（应当 11-19）")
        got = topup.drop_live_tail(d, "CU", "SHFE", "1d", ts="2026-11-19 10:00")
        if got.index[-1] != pd.Timestamp("2026-11-18"):
            raise RuntimeError(
                f"日线盘中该丢掉今天那根：{got.index[-1]}（应当 11-18）")

        # ④ 合并：库里最后一根是半根（close 停在 100），实时同一根已经走到 110。
        #    必须覆盖成 110，并且乘上本地的复权因子（2.0 → 220），否则接缝会跳
        local = pd.DataFrame(
            {"open": [200.0, 200.0], "high": [200.0, 200.0], "low": [200.0, 200.0],
             "close": [200.0, 200.0], "volume": [10.0, 10.0],
             "open_oi": [0.0, 0.0], "close_oi": [0.0, 0.0],
             "adj_factor": [2.0, 2.0]},
            index=pd.DatetimeIndex(["2026-11-19 14:45", "2026-11-19 14:50"]))
        live = pd.DataFrame(
            {"open": [110.0, 120.0], "high": [110.0, 120.0], "low": [110.0, 120.0],
             "close": [110.0, 120.0], "volume": [50.0, 60.0],
             "open_oi": [0.0, 0.0], "close_oi": [0.0, 0.0]},
            index=pd.DatetimeIndex(["2026-11-19 14:50", "2026-11-19 14:55"]))
        out = merge_live(local, live)
        if abs(float(out.loc["2026-11-19 14:50", "close"]) - 220.0) > 1e-9:
            raise RuntimeError(
                "库尾半根没有被实时值覆盖（或没乘复权因子）："
                f"{float(out.loc['2026-11-19 14:50', 'close'])}（应当 220）")
        if abs(float(out.loc["2026-11-19 14:50", "volume"]) - 50.0) > 1e-9:
            raise RuntimeError("半根的成交量没被覆盖成实时值")
        if abs(float(out.loc["2026-11-19 14:55", "close"]) - 120.0) > 1e-9:
            raise RuntimeError("新 bar 没接上（或新 bar 的复权因子不是 1.0）")
        return ("盘中丢活 bar / 收盘后保留、日线盘中丢今天收盘后保留、"
                "库尾半根被实时值覆盖且乘回复权因子")

    check("半根 bar：补库 / 合并 / 切尾同一口径", _half_bar_three_places)

    def _merge_live_attrs():
        """`merge_live` 之后 `attrs["symbol"]` 必须还在。

        2026-11-20 现场：白银（AG 1d_tq）实盘理论 −2 手、研究页「空仓」，
        回测也对不上。根因不在台账、也不在信号规则，而在 `merge_live` 结尾

            return pd.concat([out, add]).sort_index()

        —— `pd.concat` **不传播 `df.attrs`**，而 `factors_ext._symbol()` 正是靠
        `attrs["symbol"]` 认品种。丢了以后 `_fund()` 整段返回 NaN，
        `war_chg_5` / `war_val` / `inv_*` / `profit_*` / `spot_basis` 全列没值，
        `factor_matrix` → `filter_available(min_obs=60)` 把它们**静默剔除**，
        `mean(skipna=True)` 少一票，分数从 +0.4582 掉到 −0.5568，跨过
        `exit_z=-0.3` ⇒ 信号 0 翻成 −1。

        实测（AG 1d_tq，库尾 11-19 + 实时 11-20）：修复前 `sig=-1`，
        修复后 `sig=0`，与回测一致。`war_chg_5` 有效值 2438 → 0 → 2439。

        研究页（`DataStore.get` → `load_main`，attrs 完好）与实盘（`merge_live`
        之后，attrs 丢了）从此走的必须是同一份「因子可用集合」。
        """
        import pandas as pd

        from ttrade.tqlive import merge_live

        cols = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                "volume": 10.0, "open_oi": 0.0, "close_oi": 0.0,
                "adj_factor": 1.0}
        local = pd.DataFrame(cols, index=pd.DatetimeIndex(
            ["2026-11-18", "2026-11-19"]))
        local.attrs["symbol"] = "AG"
        local.attrs["exchange"] = "SHFE"
        live = pd.DataFrame(cols, index=pd.DatetimeIndex(
            ["2026-11-20"]))
        live.attrs["symbol"] = "AG"

        # ① 正常拼接（走 pd.concat 那条路）
        got = merge_live(local, live)
        sym = (got.attrs or {}).get("symbol")
        if sym != "AG":
            raise RuntimeError(
                f"合并后 attrs['symbol'] 丢了（得到 {sym!r}）—— 基本面 / 仓单 / "
                "基差因子会整列 NaN 并被静默剔除，实盘与回测从此各算一套")
        if (got.attrs or {}).get("exchange") != "SHFE":
            raise RuntimeError("合并后 attrs['exchange'] 也丢了")

        # ② 没有新 bar（`add.empty` 早退）也要带
        got = merge_live(local, live.iloc[:0])
        if (got.attrs or {}).get("symbol") != "AG":
            raise RuntimeError("没有新 bar 时 attrs 丢了")

        # ③ 本地为空（拿实时那份起步）
        got = merge_live(None, live)
        if (got.attrs or {}).get("symbol") != "AG":
            raise RuntimeError("本地为空时 attrs 丢了")

        # 反证：把 attrs 摘掉以后 `_symbol()` 必须真的读不到 —— 否则上面
        # 三条断言是假绿（比如 `_symbol` 改成了读 symbol 列）
        from ttrade import factors_ext as _FX
        bare = got.copy()
        bare.attrs = {}
        if _FX._symbol(bare):
            raise RuntimeError(
                f"`_symbol()` 在 attrs 为空时仍返回 {_FX._symbol(bare)!r} —— "
                "这组断言测不到 attrs，需要改成别的手段再测一遍")
        return "merge_live 的四个出口都保留 attrs（symbol / exchange），且 _symbol 确实读它"

    check("实时合并保留 attrs（品种不丢）", _merge_live_attrs)

    def _ts_precision_ns_us():
        """时间戳精度：实盘帧是 ns、库内帧是 us —— `merge_asof` 会直接抛。

        2026-11-20 现场（白银 / 铅 1d_tq「研究页空仓、实盘还有空单」），两条
        互相独立的原因叠在一起，这是第二条：

        * DuckDB 读出来 `datetime64[us]`（研究页 / 回测走这条）；
        * 天勤推上来 `datetime64[ns]`（`tqfetch.local_time`）；
        * `pd.concat` 取更精的 ⇒ 实盘帧整帧变 **ns**；
        * `factors_ext._asof` 里 `merge_asof(左 ns, 右 us)` ⇒
          `MergeError: incompatible merge keys` ⇒ 基本面因子被静默剔除 ⇒
          少一票 ⇒ 分数从 +0.46 掉到 −0.55 ⇒ sig 0 变 −1。

        三件都要测：① 反证「混精度确实会炸」；② `merge_live` 的输出精度跟
        库内那份一致；③ 在 ns 帧上算基本面因子与 us 帧逐值相等。
        """
        import numpy as np
        import pandas as pd

        from ttrade import factors as F
        from ttrade.tqlive import merge_live

        # ① 反证：左 ns、右 us 的 merge_asof 必须炸 —— 否则这条自检是假绿
        left = pd.DataFrame({"t": pd.to_datetime(["2026-11-18", "2026-11-19"])
                             .astype("datetime64[ns]"), "v": 0.0})
        right = pd.DataFrame({"t": pd.to_datetime(["2026-11-17", "2026-11-18"])
                              .astype("datetime64[us]"), "w": 1.0})
        try:
            pd.merge_asof(left.sort_values("t"), right.sort_values("t"), on="t",
                          direction="backward")
        except Exception:  # noqa: BLE001
            pass
        else:
            raise RuntimeError(
                "ns/us 混精度的 merge_asof 没有抛异常 —— 本机 pandas 行为已变，"
                "这条自检的前提不成立，需要重新确认（别直接删断言）")

        # ② `merge_live` 必须把结果拉回库内那份的精度
        cols = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                "volume": 10.0, "open_oi": 0.0, "close_oi": 0.0,
                "adj_factor": 1.0}
        local = pd.DataFrame(cols, index=pd.DatetimeIndex(
            ["2026-11-18", "2026-11-19"]).astype("datetime64[us]"))
        local.attrs["symbol"] = "AG"
        live = pd.DataFrame(cols, index=pd.DatetimeIndex(
            ["2026-11-20"]).astype("datetime64[ns]"))
        got = merge_live(local, live)
        if got.index.dtype != local.index.dtype:
            raise RuntimeError(
                f"合并后索引精度是 {got.index.dtype}，库内是 {local.index.dtype}"
                " —— 基本面 / 宏观因子的 merge_asof 会抛 MergeError")

        # ③ 拿真实品种比一次：ns 帧与 us 帧的仓单因子必须逐值相等
        from ttrade import tqdata
        try:
            df = tqdata.load_main("AG", "SHFE", "1d_tq", adjust="qfq")
        except Exception:  # noqa: BLE001
            return "反证 + 精度对齐通过（库里没有 AG，跳过逐值对比）"
        if df is None or len(df) < 120:
            return "反证 + 精度对齐通过（AG 行数不足，跳过逐值对比）"
        df = df.tail(400).copy()
        df.attrs["symbol"] = "AG"
        us = df.copy()
        ns = df.copy()
        ns.index = pd.DatetimeIndex(ns.index).astype("datetime64[ns]")
        # 走公开入口，`_asof` / 日频差分 / 分位这几条路一起覆盖
        a = F.raw(us, "war_chg_5")
        b = F.raw(ns, "war_chg_5")
        if a.notna().sum() == 0:
            return "反证 + 精度对齐通过（AG 没有仓单数据，跳过逐值对比）"
        if not np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                           equal_nan=True):
            raise RuntimeError("ns 帧与 us 帧的仓单因子不一致 —— 精度归一没生效")
        if b.notna().sum() == 0:
            raise RuntimeError("ns 帧上 war_chg_5 一个有效值都没有（因子被静默剔除）")
        return (f"混精度必炸（反证通过）、merge_live 精度跟库内、"
                f"ns 帧仓单因子逐值等于 us 帧（{int(b.notna().sum())} 个有效值）")

    check("时间戳精度（ns/us）与 merge_asof", _ts_precision_ns_us)

    def _db_lock_kind():
        """DuckDB 写连接被**读**挡住要重试，被**另一个写**挡住不能重试。

        2026-11-19 现场：GUI 开着时补 5m，写到第 9 个品种就报
        「Cannot open file ... 另一个程序正在使用此文件」—— 界面是按查询
        open/close 的，那一瞬正好有个读连接。前 8 个品种已入库、后面全没写，
        看着像「补了一半」，其实只差几秒重试。
        写写冲突（第二个下载器）必须原样抛，不能拿重试掩盖。
        """
        from ttrade import tqstore as S

        ok = [
            Exception('IO Error: Cannot open file "x.duckdb": 另一个程序正在使用此文件，'
                      '进程无法访问。'),
            Exception("IO Error: Cannot open file \"x.duckdb\": The process cannot "
                      "access the file because it is being used by another process."),
        ]
        for exc in ok:
            if not S._blocked_by_reader(exc):
                raise RuntimeError(f"读连接挡住没认出来：{exc}")
        no = [
            Exception("TransactionContext Error: Failed to commit: Conflicting lock "
                      "is present on database"),
            Exception("Binder Error: no such table"),
        ]
        for exc in no:
            if S._blocked_by_reader(exc):
                raise RuntimeError(f"写写冲突被当成读挡住了（会被重试掩盖）：{exc}")

        return "被界面读连接短暂挡住 ⇒ 重试（最多 5.5s）；写写冲突照抛"

    check("DuckDB 写连接被读挡住会重试", _db_lock_kind)

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

    def _feed_serial_keyed_by_period():
        """行情源的 K 线缓存必须按 **(合约, 周期秒数)** 建键。

        2026-11-23 用户报「棕榈油回测空仓、实盘理论 −2 手」。真因不是策略：
        组合里另有一条 **1h** 的板块级策略，品种池含 `P_DCE`，它先订成功占住了
        `_serials[tq]`；随后 `1d_tq` 的棕榈油槽位去取同一个 tq，**拿回的是那条
        1h 序列**。`bars_from_serial` 于是给出小时线，帧尾多 5 根、末根从
        `00:00` 变成 `14:00` —— 因子在小时线上重算，signal 从 0 翻成 −1。

        只按合约建键的写法在界面上**完全看不出来**：不报错、不冲突、策略照跑。
        这里钉三件事：
        ① 同一合约的两个周期是不同的键；
        ② 两条序列各自取得回来，互不覆盖；
        ③ `bars_from_serial` 拿错序列时末根确实会变成小时线（反向证明这条
           污染链是真的，而不是理论猜测）。
        """
        import pandas as pd
        import threading

        from ttrade import market_feed as MF
        from ttrade.tqlive import bars_from_serial

        # 缓存挂在**每条会话**（`_Chan`）上，不是 `MarketFeed` 那层门面上
        kk = MF._Chan._kk
        if kk("DCE.p2601", 300) == kk("DCE.p2601", 86400):
            raise RuntimeError("同一合约的 5m 和 1d 是同一个缓存键 —— 会互相串号")
        if kk("DCE.p2601", 300) != kk("DCE.p2601", 300):
            raise RuntimeError("同一个 (合约, 周期) 两次算出的键不一样")

        ch = MF._Chan.__new__(MF._Chan)   # 只测键和取，不连天勤
        ch._cond = threading.RLock()
        ch._serials = {}
        ch._quotes = {}
        ch._api = None
        s5 = object()
        s1d = object()
        ch._serials[kk("DCE.p2601", 300)] = s5
        ch._serials[kk("DCE.p2601", 86400)] = s1d
        if ch.peek("DCE.p2601", 300) is not s5:
            raise RuntimeError("按 5 分钟取不到自己的序列")
        if ch.peek("DCE.p2601", 86400) is not s1d:
            raise RuntimeError("按日线取到了别人的序列（串号没修好）")
        if ch.peek("DCE.p2601", 3600) is not None:
            raise RuntimeError("没订过的周期不该返回任何序列")

        # 门面那层必须把周期原样转下去 —— 少传一个参数就退化成「0 秒」那一档，
        # 键对上了也取不到东西（这种 bug 只有真实订阅时才暴露，最难查）
        class _FakeChan:
            def __init__(self):
                self.seen = []

            def request(self, tq, dur, length, kind="k"):
                self.seen.append(("request", tq, dur, length, kind))

            def peek(self, tq, dur=0, kind="k"):
                self.seen.append(("peek", tq, dur, kind))
                return "SER"

            def read_kline(self, tq, copy=True, duration_sec=0):
                self.seen.append(("read", tq, copy, duration_sec))
                return None

        fake = _FakeChan()
        mf = MF.MarketFeed.__new__(MF.MarketFeed)
        mf.chan = lambda source=None, _f=fake: _f     # type: ignore[method-assign]
        mf.request_kline("DCE.p2601", 86400, 100)
        if mf.peek_kline("DCE.p2601", 86400) != "SER":
            raise RuntimeError("peek_kline 没把序列透传出来")
        mf.read_kline("DCE.p2601", duration_sec=86400)
        want = [("request", "DCE.p2601", 86400, 100, "k"),
                ("peek", "DCE.p2601", 86400, "k"),
                ("read", "DCE.p2601", True, 86400)]
        if fake.seen != want:
            raise RuntimeError(f"门面没把周期转下去：{fake.seen}（应 {want}）")

        # ③ 反向证明：把小时线当成日线帧的尾巴，末根就是 14:00
        t0 = pd.Timestamp("2026-11-23 02:00", tz="UTC")   # +8h = 北京 10:00
        ser = pd.DataFrame({
            "datetime": [int((t0 + pd.Timedelta(hours=k)).value) for k in range(5)],
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
            "volume": 1.0, "open_oi": 1.0, "close_oi": 1.0,
        })
        got = bars_from_serial(ser, "1h")
        last = got.index[-1] if len(got) else None
        if last != pd.Timestamp("2026-11-23 14:00"):
            raise RuntimeError(
                f"小时线末根应为 14:00，实际 {last} —— 反证链不成立，"
                "说明这个测试没复现出实盘那种帧")
        return ("K 线缓存按 (合约, 周期秒数) 建键：5m 与 1d 不互相覆盖、"
                "没订过的周期取回 None；串源时末根会从 00:00 变成 14:00"
                "（日线策略按小时线算信号的那条链）")

    check("行情源不串号（同合约多周期各订各的）", _feed_serial_keyed_by_period)

    def _bar_clock_alias():
        """`1d_tq` 必须和 `1d` 一样能判「半根」—— 否则日线策略拿未收盘的日线算信号。

        2026-11-20 的现场：白银 11-19 收盘信号是 0（空单已出场），实盘那一步
        拿到的是 **11-20 还没收盘的日线**（当天 AG 从 10805 砸到 10026），分数
        从 +0.28 拉到 −0.56，空头信号翻回 −1，于是「图上已出场、实盘还挂着 −2
        手、并且又去开空」。根因只有一个：`RT_SEC` 表里只有源周期 `1d`，
        `RT_SEC.get("1d_tq")` 返回 0 ⇒ `is_unfinished` 一律 False ⇒ 半根从不被切。

        这里不用写死的名单，全部按周期表推：
        ① 每个非合成周期都要查得到秒数，且 `别名 = 源周期 × 合并根数`；
        ② `2d` / `1w` 必须查不到（合成周期实时门禁本来就挡着）；
        ③ 同一个时刻，`1d_tq` 与 `1d` 对「今天这根走完没有」的判断必须**逐点相同**
           （这是「同一语义只许有一个判据」的结构断言，不写死 True/False）；
        ④ 日线必须在**当日 15:00** 就走完 —— DCE / CZCE / GFEX 的夜盘收在 23:00
           但不跨零点，旧口径会把它们算成 23:00，等于日线策略永远晚一天进场。
        """
        from ttrade import barclock as BC
        from ttrade import symbols_info as SI
        from ttrade import tqdata as TD

        bad = []
        n = 0
        for p in TD.PERIOD_ORDER:
            sec = BC.rt_sec(p)
            if TD.is_multiday(p):
                if sec:
                    bad.append(f"{p} 是由日线合成的，不该有实时秒数（{sec}）")
                continue
            if sec <= 0:
                bad.append(f"{p} 查不到一根多少秒")
                continue
            n += 1
            src, cnt = TD.PERIODS[p]
            want = BC.rt_sec(src) * max(1, int(cnt))
            if sec != want:
                bad.append(f"{p}={sec}s，按源周期 {src}×{cnt} 应是 {want}s")

        # ③ 逐点比对：盘中 / 收盘后 / 夜盘 三个时刻，两个键必须同答案
        for ts in ("2026-11-20 09:05", "2026-11-20 14:20",
                   "2026-11-20 15:01", "2026-11-20 21:30", "2026-11-21 02:10"):
            for sym, ex in (("AG", "SHFE"), ("P", "DCE"), ("IF", "CFFEX")):
                a = BC.is_unfinished("2026-11-20", sym, ex, "1d_tq", ts)
                b = BC.is_unfinished("2026-11-20", sym, ex, "1d", ts)
                if a != b:
                    bad.append(f"{ts} {sym}：1d_tq={a} 与 1d={b} 不一致")
        # 反证：换成 11-21 那根，同一个时刻必须是「没走完」（否则上面那条白测）
        if not BC.is_unfinished("2026-11-21", "AG", "SHFE", "1d_tq", "2026-11-20 21:30"):
            bad.append("11-20 夜盘时 11-21 那根应当是活的，判成了走完")

        # ④ 所有交易所的日线都在 15:00 收，不再出现 23:00
        for sym, ex in (("P", "DCE"), ("SR", "CZCE"), ("lc", "GFEX"),
                        ("AG", "SHFE"), ("IF", "CFFEX")):
            end = BC.day_close(sym, ex, "2026-11-20")
            if (end.hour, end.minute) != (15, 0) or end.day != 20:
                bad.append(f"{sym}.{ex} 日线收盘判成 {end:%m-%d %H:%M}，应是 11-20 15:00")
            if not SI.session_ranges(sym, ex):
                bad.append(f"{sym}.{ex} 交易时段解析不出来，上面那条是空转")
        if bad:
            raise RuntimeError("；".join(bad))
        return (f"{n} 个实时周期长度自洽（别名=源周期×根数），"
                "1d_tq 与 1d 逐点一致，日线统一 15:00 收盘")

    check("日线周期别名与半根判定", _bar_clock_alias)

    def _aux_period_caliber():
        """研究页和实盘挂衍生列必须是**同一种对齐口径**。

        2026-11-20 棕榈油：回测显示「当前状态：空仓」，组合里理论还是 −2 手。
        根因就是两处各挑了一种口径 —— 研究页日线走 `attach()`（同一天的期限
        结构），实盘 `_with_aux` 不分周期一律 `attach_intraday()`（整体 shift 1）。
        展期因子差一天，分数从 −1.2259 变成 −1.3808，正好跨过 `exit_z`。

        所以这里钉三件事：
        ① `attach_for_period` 按周期分派 —— 日线（含 1d_tq / 2d / 1w）与
           `attach()` 逐值相等，分钟与 `attach_intraday()` 逐值相等；
        ② 反证：两种口径在尾部的 `c1_close` 必须**不一样**，否则 ① 是空转；
        ③ 结构：`strategy_runner` / `gui/app.py` 的源码里不许直接调那两个
           底层函数，只能经过 `attach_for_period`（谁再手写一遍就红）。
        """
        import inspect

        import numpy as np
        import pandas as pd

        from ttrade import auxdata as AD
        from ttrade import strategy_runner as R
        from ttrade import tqdata as TD

        bad = []
        daily = TD.load_main("P", "DCE", "1d_tq", adjust="qfq")
        if daily is None or daily.empty:
            return "本地没有 P 日线，跳过（不影响其它检查）"
        minute = TD.load_main("P", "DCE", "1h", adjust="qfq")
        minute = minute.iloc[-200:] if minute is not None and len(minute) else minute

        same = AD.attach(daily, "P", "DCE")
        lag = AD.attach_intraday(daily, "P", "DCE")
        col = next((c for c in ("c1_close", "c2_close", "iv_near")
                    if c in same.columns and c in lag.columns), "")
        if not col:
            return "没有衍生数据（展期/期权快照），跳过口径比对"
        got_d = AD.attach_for_period(daily, "P", "DCE", "1d_tq")
        got_w = AD.attach_for_period(daily, "P", "DCE", "2d")
        for tag, got, want in (("1d_tq", got_d, same), ("2d", got_w, same)):
            a = pd.to_numeric(got[col], errors="coerce").to_numpy(float)
            b = pd.to_numeric(want[col], errors="coerce").to_numpy(float)
            if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.allclose(
                    a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-9):
                bad.append(f"{tag} 没走日线口径（同日）")
        if minute is not None and len(minute):
            got_m = AD.attach_for_period(minute, "P", "DCE", "1h")
            want_m = AD.attach_intraday(minute, "P", "DCE")
            a = pd.to_numeric(got_m[col], errors="coerce").to_numpy(float)
            b = pd.to_numeric(want_m[col], errors="coerce").to_numpy(float)
            if not np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-9):
                bad.append("1h 没走分钟口径（隔一个交易日）")

        # ② 反证：两种口径在尾部必须真的有差，否则上面两条谁都能过
        a = pd.to_numeric(same[col], errors="coerce")
        b = pd.to_numeric(lag[col], errors="coerce")
        tail = (a.tail(5).to_numpy(float), b.tail(5).to_numpy(float))
        if np.allclose(np.nan_to_num(tail[0]), np.nan_to_num(tail[1]), atol=1e-9):
            bad.append("同日与隔日在尾部完全相同，口径比对是空转（衍生数据可能停更）")

        # ③ 结构：别处不许再手写一遍选择逻辑
        for mod, name in ((R, "ttrade/strategy_runner.py"),
                          (__import__("gui.app", fromlist=["app"]), "gui/app.py")):
            src = inspect.getsource(mod)
            for fn in ("attach_intraday(", "attach_daily("):
                if fn in src:
                    bad.append(f"{name} 里还有 {fn} —— 必须走 attach_for_period")
        if bad:
            raise RuntimeError("；".join(bad))
        return (f"日线（1d_tq/2d）走同日、分钟走隔一日，"
                f"尾部 {col} 两种口径确实不同；两个入口都只经 attach_for_period")

    check("衍生列（展期/期权）的周期口径", _aux_period_caliber)

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

        以前 `openctp-ctp` 只有 Windows / Linux 的 wheel，macOS 上装不了，
        所以这条自检**不能因为没装就红**。2026-11-23 起 macOS 也有了
        （6.7.0.9 那份发了 macosx 轮子，CI 在 macOS 上钉的就是它）—— 于是
        「macOS 上加载不了」从「正常」变成了「打包出错」，见下面的分支。
        Linux 仍然按可选处理。
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

        # `HAS_CTP` 现在是惰性的（通道用到时才绑定），这里先绑一次再判断，
        # 否则会误判成「没装」而跳过后面所有实质检查
        try:
            C.bind_channel("ctp")
        except Exception:  # noqa: BLE001 - 自测里不该因通道冲突中断
            pass
        if not C.HAS_CTP:
            # macOS 包从 3.8.9 起**承诺**带 CTP（openctp-ctp 6.7.0.9 有 macosx
            # 轮子，CI 会装）。这里要是也不可用，只有两种可能：构建机没装上，
            # 或者 PyInstaller 把 `.so` 和 `libthosttraderapi_se.dylib` 拆到了
            # 不同目录（`@loader_path` 解析不到）。两种都得让构建红 ——
            # 静默跳过的话，用户拿到的是「macOS 上点 CTP 账户说未安装」的包。
            if sys.platform == "darwin":
                raise RuntimeError(
                    "macOS 发布包必须带 CTP，但它加载不了："
                    f"{C.CTP_IMPORT_ERROR}")
            return f"openctp-ctp 未安装（CTP 不可用，其余功能不受影响）：{C.CTP_IMPORT_ERROR[:60]}"

        # ---- 历史委托 / 成交补查（2026-11-23 融航「今空有仓、委托框空」）
        #
        # 柜台只在登录**之后**推 OnRtnOrder / OnRtnTrade。持仓靠
        # `query_positions` 补齐了，委托和成交不补就是两张空表。这里钉住三件事：
        # 发的是哪两个请求、回调落进哪张表、查完（含「空表」和「柜台拒绝」）
        # 有没有记对「查完了」。
        sent = []

        class _FakeTd:
            def ReqQryOrder(self, req, rid):
                sent.append(("order", req.BrokerID, req.InvestorID))

            def ReqQryTrade(self, req, rid):
                sent.append(("trade", req.BrokerID, req.InvestorID))

        import threading as _thq

        q = C.TraderClient.__new__(C.TraderClient)
        q.api, q._req, q.cfg = _FakeTd(), 0, cfg
        q._req_lock = _thq.Lock()          # `__new__` 跳过了 __init__
        q.query_orders()
        q.query_trades()
        want = [("order", cfg.broker_id, cfg.user_id),
                ("trade", cfg.broker_id, cfg.user_id)]
        assert sent == want, f"历史查询没带经纪商/投资者账号：{sent}"

        import types as _types

        h = C.TraderClient(cfg)
        spi = h._make_spi()
        one = _types.SimpleNamespace(
            OrderRef="12", OrderSysID="12345", InstrumentID="T2612",
            Direction="1", CombOffsetFlag="0", LimitPrice=110.03,
            VolumeTotalOriginal=2, VolumeTraded=2, OrderStatus="0",
            StatusMsg="", InsertTime="09:57:48", TradingDay="20261123")
        spi.OnRspQryOrder(one, None, 1, True)
        assert h.orders_loaded, "历史委托查完（bIsLast）没记「查完了」"
        row = h.orders.get("12345") or {}
        for key, val in (("instrument", "T2612"), ("direction", "1"),
                         ("offset", "0"), ("status", "全部成交"),
                         ("volume", 2), ("traded", 2),
                         ("insert_time", "09:57:48"), ("trading_day", "20261123")):
            assert row.get(key) == val, \
                f"历史委托的 {key} 映射错了：{row.get(key)!r}（应为 {val!r}）"
        # 本地单号也要能查到：`_chase` / 撤单都是按 OrderRef 找单的
        assert h.orders.get("12"), "历史委托没按 OrderRef 再存一份，撤单会找不到"

        tr = _types.SimpleNamespace(
            TradeID="T1", OrderRef="12", InstrumentID="T2612",
            Direction="1", OffsetFlag="0", Price=110.02, Volume=2,
            TradeTime="09:57:48")
        spi.OnRspQryTrade(tr, None, 2, True)
        assert h.trades_loaded, "历史成交查完（bIsLast）没记「查完了」"
        trow = h.trades.get("T1") or {}
        assert trow.get("price") == 110.02 and trow.get("volume") == 2, \
            f"历史成交的价格/手数映射错了：{trow}"

        # ---- 持仓查询：「查完了」和「一条都没有」必须分得开（2026-11-23 融航）
        #
        # 组合页靠 `positions_loaded` 判断「比对结果作不作数」。没有它的时候，
        # `positions` 为空分不清「账户真没仓」和「柜台一条回调都没来」，
        # 后者会拿空表去报「账户与策略理论持仓一致」—— 那正是用户看到的假象。
        p = C.TraderClient(cfg)
        assert not p.positions_loaded, "刚建的客户端就自称持仓读完了"
        p._make_spi().OnRspQryInvestorPosition(None, None, 1, True)
        assert p.positions_loaded, (
            "空仓（柜台只回一条空包 + bIsLast）没记成「查完了」——"
            "界面会永远停在「还没读到」，真有仓了也不敢说一致")
        assert p.snapshot().get("positions_loaded") is True, \
            "`snapshot()` 没把「持仓查完了」透给上层，组合页只能靠空表猜"

        # 只来了一条（`bIsLast=False`）不算读完：半份快照拿去算差额会补错量
        half = C.TraderClient(cfg)
        hspi = half._make_spi()
        hspi.OnRspQryInvestorPosition(_types.SimpleNamespace(
            InstrumentID="T2612", PosiDirection="3", Position=2, YdPosition=0,
            TodayPosition=2, OpenCost=220.0, PositionProfit=0.0,
            UseMargin=0.0, TradingDay="20261123"), None, 1, False)
        assert not half.positions_loaded, "只来了一条（bIsLast=False）就算读完了"
        hspi.OnRspQryInvestorPosition(None, None, 1, True)
        assert half.positions_loaded, "整轮收尾了还没记成「查完了」"
        assert int(half.positions["T2612"]["short"]) == 2, \
            f"收尾那一拍把已经攒到的持仓弄丢了：{half.positions}"

        # 柜台拒绝（中台不放行查询）：不能当读到了，原因要留下
        deny = C.TraderClient(cfg)
        deny._make_spi().OnRspQryInvestorPosition(
            None, _types.SimpleNamespace(ErrorID=17, ErrorMsg="查询未就绪"), 2, True)
        assert not deny.positions_loaded, \
            "柜台拒了却当成「持仓读到了」—— 界面会拿空表报「一致」"
        assert "持仓" in (deny.last_error or ""), \
            f"查持仓被拒时没留下原因：{deny.last_error!r}"

        # 当天一笔都没有：柜台回一条空包 + bIsLast=True，也必须算「查完了」，
        # 否则每 1 秒重查一次，直到 HIST_TRIES 用光都在打空枪
        e = C.TraderClient(cfg)
        espi = e._make_spi()
        espi.OnRspQryOrder(None, None, 3, True)
        espi.OnRspQryTrade(None, None, 4, True)
        assert e.orders_loaded and e.trades_loaded, \
            "空表（无当日委托/成交）没记成「查完了」，会一直重查"

        # 柜台拒绝（有的中台不放行 ReqQryOrder）：不能当查完，原因要留下
        bad = C.TraderClient(cfg)
        bspi = bad._make_spi()
        err = _types.SimpleNamespace(ErrorID=17, ErrorMsg="查询未就绪")
        bspi.OnRspQryOrder(None, err, 5, True)
        assert not bad.orders_loaded, "柜台拒了却当成「查完了」，历史委托永远不会重试"
        assert "历史委托" in (bad.last_error or ""), \
            f"柜台拒绝时没留下原因：{bad.last_error!r}"

        # ---- 结算确认：登录后必须自动做，失败能重试，跨交易日要重做
        #
        # 没做这一步，柜台把**每一笔**报单都打回「未确认结算结果」，而界面
        # 一切正常（已连接、策略在跑）—— 只有单子全失败。
        import threading as _th

        class _SettleApi:
            def __init__(self):
                self.n = 0
                self.last = None

            def ReqSettlementInfoConfirm(self, req, rid):
                self.n += 1
                self.last = (req.BrokerID, req.InvestorID)

        s = C.TraderClient(cfg)
        s.api = _SettleApi()
        s.logged_in = _th.Event()
        assert s.confirm_settlement() is False and s.api.n == 0, \
            "还没登录就发了结算确认"
        s.logged_in.set()
        assert s.confirm_settlement() is True, "登录后第一次结算确认没发出去"
        assert s.api.n == 1 and s.api.last == (cfg.broker_id, cfg.user_id), \
            f"结算确认没带经纪商/投资者账号：{s.api.last}"
        assert s.confirm_settlement() is False, "3 秒内重复发了结算确认（柜台有限流）"
        s.settled.set()                 # 确认回包到了
        s._settle_at = 0.0
        assert s.confirm_settlement() is False, "已经确认过了还在重发"

        # 跨交易日（夜盘 21:00 起算下一交易日）：标志要清掉并强制重确认
        s.trading_day = "20261123"
        s._note_trading_day("20261124")
        assert not s.settled.is_set() and s.api.n == 2, (
            f"跨交易日没重新确认（settled={s.settled.is_set()} n={s.api.n}）"
            " —— 夜盘第一笔报单会被打回「未确认结算结果」")
        s.settled.set()
        s._note_trading_day("20261124")
        assert s.settled.is_set() and s.api.n == 2, "同一天被当成切日，每轮都在重确认"
        s._note_trading_day("")
        assert s.settled.is_set(), "柜台没给日期时空串被当成了切日"
        # 基准还是空的时候只记下来、不重确认（登录时刚确认过）；不记的话跨日
        # 探测永远醒不过来
        first = C.TraderClient(cfg)
        first.api = _SettleApi()
        first.logged_in.set()
        first._note_trading_day("20261125")
        assert first.trading_day == "20261125" and first.api.n == 0, (
            f"第一次见到交易日就重确认了（{first.trading_day} / {first.api.n}）")

        # 登录回包 → 自动发结算确认 → 回包成功才置 settled（才能下单）
        chain = C.TraderClient(cfg)
        capi = _SettleApi()
        chain.api = capi
        cspi = chain._make_spi()
        cspi.OnRspUserLogin(_types.SimpleNamespace(
            FrontID=1, SessionID=2, TradingDay="20261123"), None, 1, True)
        assert chain.logged_in.is_set(), "登录回包没置 logged_in"
        assert capi.n == 1, "登录成功之后**没有自动做结算确认**"
        assert chain.trading_day == "20261123", chain.trading_day
        cspi.OnRspSettlementInfoConfirm(None, None, 2, True)
        assert chain.settled.is_set(), "结算确认成功却没置 settled（永远下不了单）"

        bad2 = C.TraderClient(cfg)
        bad2.api = _SettleApi()
        b2 = bad2._make_spi()
        b2.OnRspSettlementInfoConfirm(None, _types.SimpleNamespace(
            ErrorID=26, ErrorMsg="结算结果未就绪"), 3, True)
        assert not bad2.settled.is_set(), "柜台拒了却置了 settled，后面报单必然全被拒"
        assert "结算" in (bad2.last_error or ""), bad2.last_error

        # ---- 市价单：AnyPrice + 价格 0 + TimeCondition=IOC（2026-11-23）
        #
        # 🔴 三个字段必须**一起**对：AnyPrice 配 GFD 会被交易所打回「该合约不支持
        # 此时间条件」；价格不填 0 会被当成「0 元限价单」拒掉。错一个都是「委托
        # 被拒」，界面上看不出是哪个字段的锅
        class _FakeTd:
            def __init__(self):
                self.reqs = []

            def ReqOrderInsert(self, req, _nid):
                self.reqs.append(req)

        rob = C.TraderClient(cfg)
        rob.api = _FakeTd()
        rob.logged_in.set()
        rob.settled.set()
        rob.place("rb2601", "BUY", "OPEN", 0.0, 1, market=True)
        r1 = rob.api.reqs[-1]
        if (r1.OrderPriceType != C.tdapi.THOST_FTDC_OPT_AnyPrice
                or float(r1.LimitPrice) != 0.0
                or r1.TimeCondition != C.tdapi.THOST_FTDC_TC_IOC):
            raise RuntimeError(
                f"市价单字段错了：price_type={r1.OrderPriceType!r} "
                f"limit={r1.LimitPrice!r} time={r1.TimeCondition!r}"
                "（要 AnyPrice + 0 + IOC）")
        # NaN 也当市价（天勤那边就是这么表示市价的）
        rob.place("rb2601", "SELL", "CLOSE", float("nan"), 1)
        r2 = rob.api.reqs[-1]
        if r2.OrderPriceType != C.tdapi.THOST_FTDC_OPT_AnyPrice:
            raise RuntimeError("NaN 价格没按市价发")
        # 给了价格就还是限价单 + GFD
        rob.place("rb2601", "BUY", "OPEN", 3000.0, 1)
        r3 = rob.api.reqs[-1]
        if (r3.OrderPriceType != C.tdapi.THOST_FTDC_OPT_LimitPrice
                or r3.TimeCondition != C.tdapi.THOST_FTDC_TC_GFD):
            raise RuntimeError("限价单被改成了市价单的字段")

        return (f"openctp-ctp 已装，{len(cases)} 个合约的交易所解析正确；"
                "历史委托/成交补查带经纪商+投资者账号，字段映射与增量回报一致，"
                "空表和拒单都记对「查完了」；"
                "市价单是 AnyPrice + 0 价 + IOC（NaN 也发市价，限价单不受影响）；"
                "登录后自动确认结算、失败可重试、跨交易日会重做")

    check("CTP 客户端（可选依赖）", _ctp)

    def _opt_runtime(kind: str, what: str) -> str:
        """可选运行时缺了怎么办：Windows 上必须红，其它平台跳过。

        macOS 上 `openctp-ctp` 根本没有 wheel（CI 第 6 步在 macOS 是 skipped），
        融航的 vendor 又只有 Windows 的 .pyd —— 这两样在 macOS 上一定拿不到。
        2026-11-23 加「两套 dll 同进程」那条自检时忘了分平台，macOS x64 直接红在
        `No module named 'openctp_ctp'`，整个 run 判失败，三个平台的包全被 Release
        扣住（Windows 已经打好了也发不出去）。但在 Windows 上缺了就是真事故
        ——发布包本该带它，静默跳过等于把护栏拆了。
        """
        from ttrade import ctp_channel

        ok, why = ctp_channel.available(kind)
        if ok:
            return ""
        if sys.platform.startswith("win"):
            raise RuntimeError(f"{what}缺失（Windows 上发布包必须带它）：{why}")
        return f"跳过：{what}在当前平台拿不到（{why}）"

    def _rohon():
        """融航资管中台：和 CTP 一样是可选依赖，没装必须优雅降级。

        `vnpy_rohon` 只有 Windows 的 cp310 / cp313 wheel，macOS / Linux / 本机
        的 3.14 base 全都没有 —— 所以这条自检**不能因为没装就红**。要验的是：
        「没装时报的是人话、带装法」和「装了时方向/开平映射是对的」。
        """
        import types

        from ttrade import ctp_channel
        from ttrade.ctp_live import RohonHub, _tcp_addr

        assert RohonHub.PLATFORM == "融航" and RohonHub.KIND == "rohon", \
            "融航柜台的平台名/类型标识被改了（界面按这两个值分组）"
        # 融航的行情前置就是标准 CTP md 服务（实测能登录能订阅），所以有行情；
        # 但「有没有行情」最终由账户里填没填 `md_front` 决定
        assert RohonHub.SUPPORTS_MD is True, \
            "融航的行情前置是标准 CTP md 服务，SUPPORTS_MD 应为 True"

        # 前置地址：账户表里填的是裸 host:port，RegisterFront 只认带协议的，
        # 而且填错**不报错**、只是静默超时
        assert _tcp_addr("1.2.3.4:1000") == "tcp://1.2.3.4:1000"
        assert _tcp_addr("tcp://1.2.3.4:1000") == "tcp://1.2.3.4:1000"
        assert _tcp_addr("  ") == "", "空地址被补成了 tcp://"

        skip = _opt_runtime("rohon", "融航运行时")
        if skip:
            return skip

        # 配置构造：**不能走 fill_preset**。走了的话，交易前置漏填会被
        # SimNow 预设悄悄补成 182.254.243.31，融航 dll 一 connect 就段错误
        hub = RohonHub([])
        cfg = hub.make_config(types.SimpleNamespace(
            name="t", td_front="1.2.3.4:1000", broker_id="B", user="u",
            password="p", app_id="a", auth_code="c"))
        assert cfg.channel == "rohon", f"通道没标成 rohon（会是 {cfg.channel}）"
        assert cfg.td_front == "tcp://1.2.3.4:1000", f"交易前置没归一：{cfg.td_front}"
        assert cfg.md_front == "", "账户没填 md_front，就不该有行情地址"

        # 账户填了行情前置 -> 归一成 tcp://，交给 MdClient
        cfg3 = hub.make_config(types.SimpleNamespace(
            name="t3", td_front="1.2.3.4:1000", md_front="5.6.7.8:2000",
            broker_id="B", user="u", password="p", app_id="a", auth_code="c"))
        assert cfg3.md_front == "tcp://5.6.7.8:2000", \
            f"行情前置没归一：{cfg3.md_front}"

        cfg2 = hub.make_config(types.SimpleNamespace(
            name="t2", td_front="", broker_id="B", user="u",
            password="p", app_id="a", auth_code="c"))
        assert cfg2.td_front == "", \
            "空交易前置被预设补上了 —— 那会连到 SimNow，融航 dll 直接崩"

        # 🔴 两套 dll 能不能同进程 —— 2026-11-23 改判成「能」，见下面的
        # `_dual_channel` 那一块（这里不重复加载，免得这一块的耗时翻倍）

        # ---- 市价单（2026-11-23 用户：「融航也要支持市价单」）
        #
        # 只查源码不真报单（这一块不连柜台，报单路径没有真会话可打）。
        # 要钉住的是「字段跟 CTP 一个规矩」——AnyPrice + 价格 0 + IOC，
        # 以及那句「只收限价单」的老闸门真的删掉了
        import inspect

        from ttrade import rohon_client as RC

        src = inspect.getsource(RC.RohonTraderClient.place)
        # 只看 `raise` 那几行：文档里还会提到「以前只收限价单」，那是解释不是闸门
        for line in src.splitlines():
            s = line.strip()
            if s.startswith("raise") and "只收限价单" in s:
                raise RuntimeError(f"融航 place() 还留着「只收限价单」的闸门：{s}")
        for need in ("THOST_FTDC_OPT_AnyPrice", "THOST_FTDC_TC_IOC", "market"):
            if need not in src:
                raise RuntimeError(f"融航 place() 没处理市价单：缺 {need}")
        if '"LimitPrice": 0.0 if market else px' not in src:
            raise RuntimeError("融航的市价单没把 LimitPrice 填 0（会被当成 0 元限价单）")

        # ---- 持仓回包：按「一轮」攒、收尾整份换掉
        #
        # 🔴 融航的持仓**只能**靠这个查询刷新（中台不推持仓），而柜台每秒最多
        # 一个查询 ⇒ 同一个合约会被反复问。早先 `profit` / `margin` 是 `+=`
        # 且从不重置，浮盈和保证金每问一遍就翻一倍；平掉的仓也永远删不掉。
        tr = RC.RohonTraderClient(types.SimpleNamespace(user_id="u", broker_id="b"))
        spi = RC._make_td(tr)
        one = {"InstrumentID": "rb2601", "PosiDirection": "2", "Position": 2,
               "YdPosition": 1, "TodayPosition": 1, "OpenCost": 60000.0,
               "PositionProfit": 100.0, "UseMargin": 20000.0}
        two = dict(one, PosiDirection="3", Position=1, YdPosition=0,
                   TodayPosition=1, OpenCost=30000.0, PositionProfit=50.0,
                   UseMargin=10000.0)
        if tr.positions_loaded:
            raise RuntimeError("新建的客户端不该自称「已读过持仓」")
        spi.onRspQryInvestorPosition(one, None, 1, False)
        spi.onRspQryInvestorPosition(two, None, 1, True)
        pos = tr.positions["rb2601"]
        if (pos["long"], pos["short"]) != (2, 1) or \
                (pos["cost_long"], pos["cost_short"]) != (60000.0, 30000.0):
            raise RuntimeError(f"多空持仓/成本没分开记：{pos}")
        if pos["profit"] != 150.0 or pos["margin"] != 30000.0:
            raise RuntimeError(f"浮盈/保证金累加错了：{pos}")
        if not tr.positions_loaded:
            raise RuntimeError(
                "收尾包到了却没置 positions_loaded —— 对齐那边就分不清"
                "「账户真没仓」和「柜台没回」")
        # 第二轮：柜台又给了一份。浮盈不许翻倍、已平掉的空头要消失。
        # 🔴 契约是「**整轮收满才提交**」：`_pos_stage` 收，`bIsLast` 那一包
        # 才整份替换 `positions`。早先是「开轮就地清一次、之后原地累加」——
        # 从清表到最后一包之间读到的都是**半份持仓**，界面就会「少一个合约」
        # （2026-11-23 用户报的融航那个现象），引擎还会拿半份表去判「账户没有」
        tr._pos_req = 3
        spi.onRspQryInvestorPosition(one, None, 3, True)
        pos = tr.positions["rb2601"]
        if pos["profit"] != 100.0 or pos["short"] != 0:
            raise RuntimeError(f"第二轮没清掉上一轮（浮盈翻倍/旧仓没清）：{pos}")
        # 半份窗口：新一轮只到了一条，对外那份必须还是**上一轮整份**
        tr._pos_req = 4
        spi.onRspQryInvestorPosition(two, None, 4, False)
        if tr.positions["rb2601"]["short"] != 0:
            raise RuntimeError(
                "新一轮收到一半就把表换掉了 —— 界面这一瞬间会少一个合约")
        spi.onRspQryInvestorPosition(None, None, 4, True)     # 本轮收尾
        if tr.positions["rb2601"]["short"] != 1:
            raise RuntimeError(f"整轮收满没换上新表：{tr.positions['rb2601']}")
        # 同一轮里再补一个空终止包：**不许**把表刷成空
        spi.onRspQryInvestorPosition(None, None, 4, True)
        if not tr.positions:
            raise RuntimeError("多一个空 `last` 包就把持仓清空了（旧写法就是这个）")
        # 上一轮迟到的回包：不许污染当前这一轮
        spi.onRspQryInvestorPosition(one, None, 3, True)
        if tr.positions["rb2601"]["long"] != 0:
            raise RuntimeError("上一轮迟到的回包又累进当前这一轮了（手数会翻倍）")
        # 真的没仓：空包也要算「读完了」
        tr2 = RC.RohonTraderClient(types.SimpleNamespace(user_id="u", broker_id="b"))
        RC._make_td(tr2).onRspQryInvestorPosition(None, None, 1, True)
        if tr2.positions or not tr2.positions_loaded:
            raise RuntimeError("空仓的空包没置 positions_loaded（会被当成「没读到」）")

        return ("融航走 CTP 兼容 dll（6.7.2.8），行情/交易地址归一正确；"
                "市价单字段与 CTP 一致（AnyPrice + 0 价 + IOC）；"
                "持仓回包整轮收满才提交（半份不外泄 ⇒ 不会少一个合约）、"
                "浮盈不翻倍、平掉的仓会消失、多余空包不刷表、迟到的旧回包不污染、"
                "空包也算读完")

    check("融航客户端（可选依赖）", _rohon)

    def _dual_channel() -> str:
        """标准 CTP 与融航两套 dll **可以同时在同一个进程里**（2026-11-23 改判）。

        用户问：「现在 CTP 和融航不能同时登录，说设计独立进程可以支持同时登录，
        评估下可行吗」。评估的第一步不是设计进程，而是确认这条限制还成不成立。

        原来的依据是「同进程建两套 API 必 exit 139」，据此做了**进程级互斥**：
        一绑一个通道，再要另一个就抛 `ChannelError`，界面只能二选一。

        2026-11-23 用 `_probe_dll.py` 分四阶段重测：只用标准 / 只用融航 /
        两套同时建对象 / 两套**同时 Init**（四条会话线程同跑 3 秒再全 Release）
        —— 全部正常，无段错误。原因在目录里看得见：融航那份 dll 是**改名后的**
        （`thosttraderapi_se-5906db0c….dll`，delvewheel 的哈希后缀就是为此存在），
        且以独立顶层包名 `rohon_openctp_ctp` 加载，两套不共享模块也不共享符号。

        所以真正要钉的是两件事：
          ① `load()` 不许再因为「已经绑了另一个」抛异常（互斥锁已拆）；
          ② 🔴 请求结构体必须**按客户端各取各的**：`mdapi`/`tdapi` 原先挂在模块上，
             后绑的会覆盖先绑的 ⇒ 先建的那个客户端按**另一套 dll 的结构体布局**
             造请求（价格/手数变垃圾值，甚至崩）。现在走 `TraderClient._td`。
        """
        from ttrade import ctp_channel
        from ttrade import ctp_client as C

        # 🔴 两家都得真装齐才验得了。macOS 上 `openctp-ctp` 没有 wheel、
        # 融航 vendor 只有 Windows 的 .pyd —— 少一家就不是「同进程共存」的场景，
        # 直接跳过（`_opt_runtime` 在 Windows 上照样会红）
        for kind, what in (("rohon", "融航运行时"),
                           ("ctp", "标准 CTP 运行时（openctp-ctp）")):
            skip = _opt_runtime(kind, what)
            if skip:
                return skip

        # ① 两家都要能拿到，而且是**不同**的模块对象
        md_c, td_c = ctp_channel.load("ctp")
        md_r, td_r = ctp_channel.load("rohon")
        if td_c is td_r or md_c is md_r:
            raise RuntimeError(
                "两个通道返回了同一套模块 —— 结构体必然串（价格/手数会变垃圾值），"
                "说明 `load()` 的缓存按错了键")
        if td_c.CThostFtdcInputOrderField is td_r.CThostFtdcInputOrderField:
            raise RuntimeError("两家的报单结构体是同一个类 —— 根本没加载到两套 dll")
        # 反过来再取一次：先绑的那家不许被后绑的挤掉（缓存要按通道分开存）
        if ctp_channel.load("ctp")[1] is not td_c:
            raise RuntimeError("先绑 CTP、再绑融航之后，CTP 那套被换掉了")

        # ② 两个客户端同时活着，各自的请求必须来自各自的 dll
        def mk(kind: str, who: str):
            return C.TraderClient(C.CtpConfig(
                name=who, preset="", channel=kind,
                td_front="tcp://127.0.0.1:1",      # 只建对象，不 start()
                broker_id="9999", user_id=who, password="p"))

        c1 = mk("ctp", "自检CTP")
        c2 = mk("rohon", "自检融航")
        if c1._td is not td_c or c2._td is not td_r:
            raise RuntimeError(
                "客户端的结构体模块没按各自通道取 —— "
                f"ctp={getattr(c1._td, '__name__', c1._td)}，"
                f"rohon={getattr(c2._td, '__name__', c2._td)}")
        r1 = c1._td.CThostFtdcInputOrderField()
        r2 = c2._td.CThostFtdcInputOrderField()
        if type(r1) is type(r2):
            raise RuntimeError("两个客户端造出了同一种报单结构体 —— 一定是共用了一套 dll")
        # 后建的那个不许污染先建的（这就是「模块级全局」那个坑的正面证据）
        c3 = mk("ctp", "自检CTP2")
        if type(c3._td.CThostFtdcInputOrderField()) is not type(r1):
            raise RuntimeError("又建了一个 CTP 客户端之后，先前那个的结构体变了")

        # ③ 🔴 SPI 回调里造请求，**不能**用 SPI 自己的 `self`
        # 用户 2026-11-23 报的「融航认证请求失败：'_Spi' object has no attribute
        # '_td'」就是这个：批量改 `tdapi.` → `self._td.` 时，内层 `class _Spi`
        # 里的 `self` 是 SPI 对象。SPI 的回调自己 try/except 掉了异常（只写
        # `last_error`），所以这里必须**两件事都查**：没记错误 + 请求真发出去了
        class _FakeApi:
            def __init__(self):
                self.calls = []

            def __getattr__(self, name):
                if name.startswith("Req"):
                    def _rec(*a, **k):
                        self.calls.append(name)
                        return 0

                    return _rec
                raise AttributeError(name)

        for who, c in (("标准", c1), ("融航", c2)):
            api = _FakeApi()
            c.api = api
            c._spi = c._make_spi()      # 必须被持有：CTP 侧拿的是裸指针
            c._spi.OnFrontConnected()   # 认证请求就在这一步造
            if c.last_error:
                raise RuntimeError(
                    f"[{who}] 前置连上就报错（SPI 回调里造不出请求）：{c.last_error}")
            if "ReqAuthenticate" not in api.calls:
                raise RuntimeError(f"[{who}] 认证请求没发出去：{api.calls}")
            c._spi.OnRspAuthenticate(None, None, 1, True)   # 认证过 → 发登录
            if c.last_error or "ReqUserLogin" not in api.calls:
                raise RuntimeError(
                    f"[{who}] 认证之后没发登录请求：{c.last_error!r} {api.calls}")
            c.api = None

        return ("两套 dll 同进程共存（load 两家都成、互不挤掉）；"
                f"请求结构体各取各的（{type(r1).__module__} / {type(r2).__module__}）；"
                "SPI 回调里用 `outer._td` 造请求（不会把 SPI 对象当成客户端）；"
                "后建的客户端不影响先建的 —— 所以 CTP 户和融航户可以同时登录，"
                "不需要独立进程")

    check("CTP + 融航同时登录（同进程两套 dll）", _dual_channel)

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
        bare = CL.CtpHub([])
        try:
            bare.place("rb2611", "BUY", "OPEN", 1, None)
        except Exception as exc:  # noqa: BLE001
            assert "还没连接" in str(exc), f"报错信息不对：{exc}"
        else:
            raise RuntimeError("没连柜台时 place() 竟然没报错")

        # 市价单：价格留空 / 0 ⇒ 交给 trader 时是「价格 0 + market=True」。
        # 🔴 以前这里抛「只收限价单」，2026-11-23 起全部交易所都支持市价单了
        seen = []

        def _rec(ins, d, o, px, vol, market=False):
            seen.append((ins, d, o, px, vol, market))
            return "9001"

        hub.trader.place = _rec
        assert hub.place("rb2611", "BUY", "CLOSE", 2, None) == "9001"
        assert seen[-1] == ("rb2611", "BUY", "CLOSE", 0.0, 2, True), seen[-1]
        hub.place("rb2611", "SELL", "OPEN", 1, 0.0)
        assert seen[-1][3] == 0.0 and seen[-1][5] is True, seen[-1]
        # 给了价格就还是限价单，并且按最小变动价位对齐（rb 一跳 1 元）
        hub.place("rb2611", "SELL", "OPEN", 1, 3000.4)
        assert (seen[-1][3], seen[-1][5]) == (3000.0, False), seen[-1]

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
        from ttrade import ctp_client as _cc
        from ttrade.ctp_client import TraderClient

        try:
            _cc.bind_channel("ctp")     # 惰性绑定：不先绑，`HAS_CTP` 一律是 False
        except Exception:  # noqa: BLE001
            pass
        HAS_CTP = _cc.HAS_CTP

        if HAS_CTP:
            got_req = []

            class _FakeTdApi:
                def ReqQryInstrument(self, req, rid):
                    got_req.append(req.InstrumentID)

            import threading

            td = TraderClient.__new__(TraderClient)
            td.api, td._req = _FakeTdApi(), 0
            td._req_lock = threading.Lock()   # `__new__` 跳过了 __init__
            td.instruments_done = threading.Event()
            td.query_instruments("rb2611")
            td.query_instruments()
            assert got_req == ["rb2611", ""], f"单合约查询没带 InstrumentID：{got_req}"

        # ---- 登录后补查历史委托 / 成交的排片
        #
        # 「今空 5 手、委托框和成交框却是空的」就是这么来的：柜台只推登录之后
        # 的回报。补查要挤进「每秒一个查询」的轮转里，补完必须停，柜台不放行
        # 时要放弃并记原因（不然每 2 秒打一枪，白耗柜台的流控）。
        from ttrade.ctp_live import HIST_TRIES

        class _T:
            orders_loaded = False
            trades_loaded = False

            def __init__(self):
                self.calls = []
                self.stuck = False

            def query_orders(self):
                self.calls.append("orders")
                if not self.stuck:
                    self.orders_loaded = True

            def query_trades(self):
                self.calls.append("trades")
                self.trades_loaded = True

        hub = CL.CtpHub([])
        hub.trader = _T()
        hub._hist_at = 0.0
        assert hub._query_hist() is True and hub.trader.calls == ["orders"], \
            f"第一拍该先补委托：{hub.trader.calls}"
        hub._hist_at = 0.0
        assert hub._query_hist() is True and hub.trader.calls == ["orders", "trades"], \
            f"第二拍该补成交：{hub.trader.calls}"
        assert hub._query_hist() is False, "补完之后还在发查询（白占柜台的每秒额度）"

        # 柜台只允许每秒一个查询：1 秒内第二次必须自己咽回去，否则两笔查询
        # 里有一笔会被柜台**静默**丢掉，看起来就是「查了但没结果」
        hub3 = CL.CtpHub([])
        hub3.trader = _T()
        hub3._hist_at = 0.0
        hub3._query_hist()
        assert hub3._query_hist() is False, "1 秒内连发了两个查询"

        stuck = _T()
        stuck.stuck = True
        hub2 = CL.CtpHub([])
        hub2.trader = stuck
        for _ in range(HIST_TRIES):
            hub2._hist_at = 0.0
            hub2._query_hist()
        assert hub2._query_hist() is False, f"试了 {HIST_TRIES} 拍还没放弃"
        assert "历史委托" in (hub2.last_error or ""), \
            f"放弃时没把原因说出来：{hub2.last_error!r}"

        # 结算没确认时，查询额度要**先给结算**（没它一笔都下不出去）
        class _Pri:
            orders_loaded = trades_loaded = True

            def __init__(self):
                self.calls = []

            def confirm_settlement(self):
                self.calls.append("settle")
                return True

            def query_account(self):
                self.calls.append("acct")

            def query_positions(self):
                self.calls.append("pos")

        hub4 = CL.CtpHub([])
        hub4.trader = _Pri()
        hub4._last_query = 0.0
        hub4.wait(0.05)
        assert hub4.trader.calls[:1] == ["settle"], \
            f"结算没确认，却先去查资金/持仓了：{hub4.trader.calls}"

        # ---- 查询额度只能有一份轮转（2026-11-23 融航「组合页读不到持仓」）
        #
        # 交易页走 `hub.wait()`，组合页走 `BrokerApi._refresh()`。早先两边各写
        # 一套节流，组合页那套在补历史委托时用的是**另一个**闸门，同一毫秒里
        # 既发「查历史委托」又发「查持仓」；柜台限制每秒一个，后发那笔被静默
        # 丢弃 —— 交易页持仓表好好的，组合页那条连接 `positions` 永远是空的。
        # 现在两边都只调 `pump_queries()`，所以这里钉住它自己的行为。
        class _Rot:
            orders_loaded = trades_loaded = True     # 历史补查不掺进来

            def __init__(self):
                self.calls = []

            def confirm_settlement(self):
                return False                          # 已确认过，让出额度

            def query_account(self):
                self.calls.append("acct")

            def query_positions(self):
                self.calls.append("pos")

        rot = CL.CtpHub([])
        rot.trader = _Rot()
        rot._last_query = 0.0
        assert rot.pump_queries() == "positions", \
            "登录后第一拍该先要持仓（界面和比对都等它）"
        assert rot._pos_try == 1, f"发过几拍持仓查询没记账：{rot._pos_try}"
        assert rot.pump_queries() == "", "1 秒内又发了一个查询（柜台只给每秒一个）"
        rot._last_query = 0.0
        assert rot.pump_queries() == "account", "持仓之后这一拍该轮到资金"
        assert rot.trader.calls == ["pos", "acct"], rot.trader.calls
        # `force=True` 放开 1 秒闸门（`BrokerApi._refresh(force=True)` 用）
        assert rot.pump_queries(force=True) == "positions", "force 没放开节流"

        return ("合约归一 / 主力解析 / tick 切 bar / 资金持仓映射 / 状态表 全对；"
                "登录后补查历史委托+成交（先委托后成交、每秒一个、补完即停、"
                "试够次数放弃并记原因）；结算没确认时查询额度优先给它；"
                "查询额度的轮转只有 `pump_queries()` 一份（资金/持仓交替、"
                "1 秒内不重发，force 才放开）")

    check("CTP 会话层（交易页后端）", _ctp_live)

    def _broker_api() -> str:
        """柜台执行层：把 CTP / 融航 装成 `TqApi` 的形状（`ttrade/broker_api.py`）。

        这一层的三个 bug 都是「静默的」，所以必须离线跑一遍真流程：

        1. **时间戳单位**：pandas 2.x 起 `DatetimeIndex` 的单位可能是微秒，
           `asi8` 给的是本单位整数。当纳秒用，整条序列会落到 1970 年，
           `bars_from_serial` 只是把它整段剔掉 —— 表现为「300 行序列、0 根 bar」；
        2. **正在走的那根 bar 要就地更新**：只接「新增」的话，一根 bar 在它诞生
           那一刻就定格了，K 线内触发（拿最新价跟触发价比）当根永远不触发；
        3. **没有柜台回报的委托要判死**：一直 ALIVE 会让引擎永远等它、永远不重发
           —— 看着像策略死了。
        """
        import pandas as pd

        from ttrade.broker_api import OPTIMISTIC_TTL, BrokerApi
        from ttrade.tqlive import AccountSpec, bars_from_serial

        class _Md:
            def __init__(self):
                self.ticks = {}
                self.subs = []

            def subscribe(self, items):
                self.subs.extend(items)

            def last(self, ins):
                return dict(self.ticks.get(ins) or {})

            def push(self, ins, stamp, price, vol):
                self.ticks[ins] = {"instrument": ins, "datetime": stamp,
                                   "last": price, "bid": price - 1, "ask": price + 1,
                                   "volume": vol, "oi": 1000}

        class _Td:
            def __init__(self):
                self.instruments = {"rb2701": {"tick": 1.0, "exchange": "SHFE"}}
                self.positions, self.orders, self.trades = {}, {}, {}
                self.account, self.placed = {}, []
                # 柜台「持仓读到过一整轮」的标志，见 `BrokerApi.positions_ready()`
                self.positions_loaded = True

            def query_account(self):
                pass

            def query_positions(self):
                pass

            def query_instruments(self, ins=""):
                self.instruments.setdefault(ins, {"tick": 1.0, "exchange": "SHFE"})

            def snapshot(self):
                return {"account": self.account, "positions": self.positions,
                        "orders": self.orders, "trades": self.trades,
                        "positions_loaded": self.positions_loaded}

            def place(self, ins, direction, offset, price, volume, market=False):
                ref = str(9000 + len(self.placed))
                # 第 7 项是 market：市价单要靠「价格 0 + market=True」一起判，
                # 只看价格会把「0 价的限价单」也认成市价
                self.placed.append((ins, direction, offset, price, volume, ref,
                                    bool(market)))
                return ref

        class _Hub:
            def __init__(self):
                self.md, self.trader, self.last_error = _Md(), _Td(), ""
                self.cancelled = []
                self.pumps = []

            def start(self):
                pass

            def close(self):
                pass

            def pump_queries(self, force=False):
                # 账户查询的唯一入口（交易页和组合页共用一份轮转）。
                # 组合页早先自己另写一套，和补历史委托撞在同一毫秒里，
                # 柜台只收第一个 ⇒ `positions` 永远是空的
                self.pumps.append(bool(force))
                self.trader.query_account()
                return "account"

            def place(self, symbol, direction, offset, volume, limit_price=None):
                try:
                    px = float(limit_price)
                except (TypeError, ValueError):
                    px = float("nan")
                return self.trader.place(symbol, direction, offset, px,
                                         int(volume), market=not (px > 0))

            def cancel(self, order_id, account=""):
                self.cancelled.append(str(order_id))

        def _hist(code, ex, period):
            end = pd.Timestamp.now().normalize() + pd.Timedelta(hours=9)
            idx = pd.DatetimeIndex([end - pd.Timedelta(minutes=5 * (300 - i))
                                    for i in range(300)])
            n = len(idx)
            return pd.DataFrame({
                "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
                "close": [100.5] * n, "volume": [10.0] * n,
                "open_oi": [1000.0] * n, "close_oi": [1000.0] * n}, index=idx)

        spec = AccountSpec(name="自检CTP", kind="ctp", user="u", password="p")
        hub = _Hub()
        api = BrokerApi(spec, log=lambda _m: None, hub=hub, history=_hist)
        api._main_of = lambda code, ex: "rb2701"
        api._ensure_sub = lambda ins: None
        api.start()

        kl = api.get_kline_serial("KQ.m@SHFE.rb", 300, 300)
        live = bars_from_serial(kl, "5m")
        assert len(live) == 300, f"历史灌进去要能转成行情表，实际 {len(live)} 根"

        q = api.get_quote("KQ.m@SHFE.rb")
        assert q.underlying_symbol == "SHFE.rb2701", q.underlying_symbol

        base = pd.Timestamp.now().normalize() + pd.Timedelta(hours=9)
        for mins, px, vol in ((1, 3551.0, 110), (2, 3552.0, 120), (6, 3553.0, 140)):
            ts = base + pd.Timedelta(minutes=mins)
            hub.md.push("rb2701", f"{ts:%Y%m%d %H:%M:%S}", px, vol)
            api.wait_update(deadline=time.time() + 0.02)
        live = bars_from_serial(api.get_kline_serial("KQ.m@SHFE.rb", 300, 300), "5m")
        assert len(live) == 302, f"tick 要拼成新 bar，实际 {len(live)} 根"
        assert abs(live["close"].iloc[-2] - 3552.0) < 1e-9, (
            f"同一根 bar 里来了新 tick 要就地更新，实际 {live['close'].iloc[-2]}")
        assert abs(live["volume"].iloc[-1] - 20.0) < 1e-9, live["volume"].iloc[-1]

        order = api.insert_order(symbol="SHFE.rb2701", direction="BUY",
                                 offset="OPEN", volume=2, limit_price=3550.0)
        assert api.get_order()[order.order_id].status == "ALIVE"
        hub.trader.orders["1000"] = {"order_ref": "1000", "instrument": "rb2701",
                                     "direction": "0", "offset": "0", "price": 3550.0,
                                     "volume": 2, "traded": 2, "status": "全部成交"}
        hub.trader.orders["1001"] = {"order_ref": "1001", "instrument": "rb2701",
                                     "direction": "0", "offset": "0", "price": 3550.0,
                                     "volume": 2, "traded": 0, "status": "被拒",
                                     "msg": "价格不对"}
        hub.trader.positions["rb2701"] = {"long": 5, "short": 1, "long_yd": 2,
                                          "short_yd": 0, "price": 3500.0}
        api._refresh(force=True)
        assert api.get_order()["1000"].status == "FINISHED"
        assert api.get_order()["1001"].status == "DEAD"
        assert api.get_order()["1001"].last_msg == "价格不对"
        pos = api.get_position()["SHFE.rb2701"]
        assert (pos.pos_long, pos.pos_short, pos.pos_long_his,
                pos.pos_long_today) == (5, 1, 2, 3), pos

        # 柜台一直不回报：超时判死，否则引擎永远等它
        api._ord[order.order_id].ts -= OPTIMISTIC_TTL * 2
        api._refresh(force=True)
        assert api.get_order()[order.order_id].status == "DEAD", "没回报的委托要判死"

        # 市价（NaN / 空价）要发**市价单**：价格填 0 + market=True
        # （2026-11-23 用户：「所有交易所都支持市价单了，上期所和能源所 7 月 6 日
        #   也开放了」；以前这里抛「只收限价单」，现在不能再抛）
        o_nan = api.insert_order(symbol="SHFE.rb2701", direction="BUY",
                                 offset="OPEN", volume=1,
                                 limit_price=float("nan"))
        ins_, _d, _o, px_, vol_, _ref, mkt_ = hub.trader.placed[-1]
        assert (px_, mkt_) == (0.0, True), hub.trader.placed[-1]
        assert o_nan.limit_price == 0.0, o_nan.limit_price
        o_none = api.insert_order(symbol="SHFE.rb2701", direction="SELL",
                                  offset="CLOSE", volume=1, limit_price=None)
        ins_, _d, _o, px_, vol_, _ref, mkt_ = hub.trader.placed[-1]
        assert (px_, mkt_) == (0.0, True), hub.trader.placed[-1]
        assert o_none.limit_price == 0.0, o_none.limit_price
        # 给了价格就是限价单，market 必须是 False（否则 0 价单会被当成市价）
        api.insert_order(symbol="SHFE.rb2701", direction="BUY", offset="OPEN",
                         volume=1, limit_price=3500.0)
        ins_, _d, _o, px_, vol_, _ref, mkt_ = hub.trader.placed[-1]
        assert (px_, mkt_) == (3500.0, False), hub.trader.placed[-1]

        # 结算 / 登录没就绪时必须**拦住**，而且给的是「等一下就好」的人话 ——
        # 抛异常会被 `tick()` 当成策略出错停机，跨交易日那几秒会把所有策略一起停
        class _Ev:
            def __init__(self, on=False):
                self.on = on

            def is_set(self):
                return self.on

        hub.trader.logged_in, hub.trader.settled = _Ev(True), _Ev(False)
        ok, why = api.ready_to_trade()
        assert not ok and "结算" in why, f"结算没确认却放行了：{ok} {why}"
        hub.trader.settled = _Ev(True)
        assert api.ready_to_trade() == (True, ""), "结算确认好了还不放行"
        hub.trader.logged_in = _Ev(False)
        ok, why = api.ready_to_trade()
        assert not ok and "登录" in why, (ok, why)

        # 查询额度只走 `hub.pump_queries()`：组合页自己另写一套会和补历史委托
        # 撞在同一毫秒，柜台只收第一个 —— 持仓查询被静默丢弃（2026-11-23 融航）
        assert hub.pumps, "`_refresh` 没走 `hub.pump_queries()`，又在自己抢柜台额度"

        # 「持仓读到了没有」：`get_position()` 是空字典分不清「真没仓」和
        # 「柜台没回」，必须由这一条判 —— 界面靠它决定报「一致」还是「还不知道」
        assert api.positions_ready() is True, "柜台说读完了，却报「没读到」"
        hub.trader.positions_loaded = False
        assert api.positions_ready() is False, (
            "柜台一条持仓都没回来，`positions_ready()` 却是真 ——"
            "组合页会拿空表报「账户与策略理论持仓一致」")
        hub.trader.positions_loaded = True
        # 自相矛盾的快照：持仓空表，资金那边占用保证金却有数。有保证金必有持仓
        hub.trader.account = {"margin": 123456.0}
        api._pos.clear()
        assert api.positions_ready() is False, (
            "占用保证金有数、持仓空表，这种自相矛盾的快照被当成「已同步」了 ——"
            "界面会报「一致」，而账户里其实有仓")
        hub.trader.account = {}
        assert api.positions_ready() is True, "保证金回到 0 之后该放行"

        api.close()
        return ("序列 300 根 + 拼 bar + 委托三态 + 今昨仓 + 无回报判死 全对；"
                "结算/登录未就绪时闸门拦住并给人话；"
                "账户查询只走 `pump_queries()` 一份轮转；"
                "`positions_ready()` 能分清「账户真没仓」和「柜台没回」")

    check("柜台执行层（策略页跑 CTP / 融航）", _broker_api)

    def _today_split():
        """上期所平今 / 平昨：今仓从 `TodayPosition` 认，别把 YdPosition 当昨仓。

        2026-11-19 用户报「一键对齐：委托被拒（第 1 次）[51] CTP:平昨仓位不足」，
        而账户里 sn2612 明明是今仓 1 手（持仓表也写着「今多 1」）。根因是引擎
        拿 YdPosition 当昨仓 → 发 `CLOSE` 去平一手今天才开的仓，上期所整单打回。
        同一份快照两处两种说法，是因为持仓表（`ctp_live`）和引擎
        （`broker_api._fill_positions`）各取了一套字段。

        这一项钉三处：快照口径 / 拆单指令 / 被拒后自动换一边重报。
        """
        import types

        from ttrade import strategy_runner as R
        from ttrade.ctp_live import split_today_his
        from ttrade.tqlive import AccountSpec

        # ① 快照口径。分条推送累加后 his+today 会大于总仓，必须收敛回总仓，
        #    且以今仓为准（今仓不可能超过总仓，昨仓才是「剩下的」）。
        cases = [
            ("今仓 1、YdPosition 报 0", {"long": 1, "long_yd": 0, "long_today": 1}, 1, (0, 1)),
            ("今昨都报 1、总仓 1（累加后不自洽）",
             {"long": 1, "long_yd": 1, "long_today": 1}, 1, (0, 1)),
            ("纯昨仓 2", {"long": 2, "long_yd": 2, "long_today": 0}, 2, (2, 0)),
            ("柜台没给 TodayPosition，退回「总 − 昨」",
             {"long": 3, "long_yd": 1}, 3, (1, 2)),
            ("空仓", {"long": 0}, 0, (0, 0)),
        ]
        for note, row, total, want in cases:
            got = split_today_his(row, "long", total)
            assert got == want, f"{note}：要 {want}，得到 {got}"

        # ② 拆单指令：今仓必须发 CLOSETODAY
        spec = AccountSpec(name="__selftest__", kind="sim")
        run = R.PortfolioRunner(spec)
        run._pos = {"SHFE.sn2612": types.SimpleNamespace(
            pos_long=1, pos_short=0, volume_long=1, volume_short=0,
            pos_long_his=0, pos_long_today=1, pos_short_his=0, pos_short_today=0,
            volume_long_his=0, volume_long_today=1,
            volume_short_his=0, volume_short_today=0)}
        legs = run._legs("SHFE.sn2612", -1)
        assert legs == [("CLOSETODAY", 1)], f"今仓要发平今，得到 {legs}"
        # 昨仓反过来必须是 CLOSE —— 给上期所发 CLOSETODAY 会被打回「平今仓位不足」
        run._pos["SHFE.sn2612"].pos_long_today = 0
        run._pos["SHFE.sn2612"].volume_long_today = 0
        run._pos["SHFE.sn2612"].pos_long_his = 1
        run._pos["SHFE.sn2612"].volume_long_his = 1
        assert run._legs("SHFE.sn2612", -1) == [("CLOSE", 1)], "昨仓要发平昨"
        # 非上期所 / 能源中心不分今昨：一律 CLOSE（发 CLOSETODAY 会被整单打回）
        run._pos["DCE.i2701"] = types.SimpleNamespace(
            pos_long=1, pos_short=0, volume_long=1, volume_short=0,
            pos_long_his=1, pos_long_today=0,
            pos_short_his=0, pos_short_today=0,
            volume_long_his=1, volume_long_today=0,
            volume_short_his=0, volume_short_today=0)
        assert run._legs("DCE.i2701", -1) == [("CLOSE", 1)], "大商所不能用 CLOSETODAY"

        # ③ 被拒后自动换一边重报，且只翻一次
        class _Api:
            def __init__(self):
                self.calls: list[dict] = []

            def get_quote(self, sym):
                return types.SimpleNamespace(
                    price_tick=10.0, ask_price1=325810.0, bid_price1=325800.0,
                    last_price=325800.0)

            def insert_order(self, **kw):
                self.calls.append(kw)
                return {"order_id": f"o{len(self.calls)}"}

        run.api = _Api()
        st = run._align_slot
        o = R.PendingOrder(order_id="x", symbol="SHFE.sn2612", direction="SELL",
                           offset="CLOSE", volume=1, price=325800.0, ts=0.0)
        ok = run._flip_offset(st, o, "[51] CTP:平昨仓位不足")
        assert ok, "平昨仓位不足没触发换边重报"
        assert o.offset == "CLOSETODAY", f"该换成平今，得到 {o.offset}"
        assert run.api.calls and run.api.calls[-1]["offset"] == "CLOSETODAY", run.api.calls
        assert run.api.calls[-1]["volume"] == 1, "手数不能变"
        assert not run._flip_offset(st, o, "[51] CTP:平今仓位不足"), "只该翻一次"
        # 与仓位无关的拒单（资金不足）不换边 —— 换了也还是失败，只会白耗额度
        o2 = R.PendingOrder(order_id="y", symbol="SHFE.sn2612", direction="SELL",
                            offset="CLOSE", volume=1, price=325800.0, ts=0.0)
        assert not run._flip_offset(st, o2, "[31] CTP:资金不足"), "资金不足不该换边"
        # 大商所不换边
        o3 = R.PendingOrder(order_id="z", symbol="DCE.i2701", direction="SELL",
                            offset="CLOSE", volume=1, price=800.0, ts=0.0)
        assert not run._flip_offset(st, o3, "平昨仓位不足"), "大商所不分今昨"
        return "今仓优先取 TodayPosition、上期所拆平今、被拒自动换边一次 全对"

    check("平今平昨：今仓优先 + 被拒自动换边", _today_split)

    def _many_failed(n: int = 40) -> list[str]:
        """造一批「同一原因、不同品种」的失败清单。

        多品种回测的真实输出就是这形状：每个失败品种各一条
        「AD 铸造铝合金　有色：选中的因子在该标的上都没有足够数据」。
        """
        one = "选中的因子在该标的上都没有足够数据（如美股无持仓量）"
        codes = (list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") * 2)[:n]
        return [f"{c} 品种{i}　化工：{one}" for i, c in enumerate(codes)]

    def _status_brief():
        """多品种失败清单的压缩口径（纯函数部分）。

        `brief_failed` 只负责「总数 / 归类 / 取前几个」，高度在
        「构建主窗口」那项里用真实 `Hint` 样式量。
        """
        from gui.app import brief_failed, split_failed

        brief, full = brief_failed(_many_failed())
        if "缺 40 个" not in brief:
            raise RuntimeError(f"没报总数：{brief}")
        if "因子无数据" not in brief:
            raise RuntimeError(f"原因没归类：{brief}")
        if len(brief) > 60:
            raise RuntimeError(f"短句还是太长（{len(brief)} 字）：{brief}")
        if len(full.splitlines()) != 40:
            raise RuntimeError(f"tooltip 没保留逐条原文：{len(full.splitlines())} 行")

        # 原因自己带全角冒号：只在**第一个**冒号处切，品种名不能被切走
        name, reason = split_failed(
            "RB 螺纹钢　黑色：跨周期因子缺行情：日线（mom_20）")
        if name != "RB 螺纹钢　黑色" or not reason.startswith("跨周期因子缺行情："):
            raise RuntimeError(f"切分错了：{name!r} / {reason!r}")

        # 原因不止一种：要写出「几种原因」，每种给一个代表品种
        mixed = ["AD 铸造铝合金　有色：" + "选中的因子在该标的上都没有足够数据",
                 "FU 燃料油　能源：没有产生信号",
                 "P 棕榈油　农产品：这段行情只有 60 根，太短了算不出因子"]
        b2, f2 = brief_failed(mixed)
        if "3 种原因" not in b2 or "无信号" not in b2 or "AD" not in b2:
            raise RuntimeError(f"混原因没归好：{b2}")
        if len(f2.splitlines()) != 3:
            raise RuntimeError("混原因的 tooltip 行数不对")

        if brief_failed([]) != ("", ""):
            raise RuntimeError("空清单应当返回两个空串")

        # 因子分析页底部那条「已剔除 N 个…：a、b、c…」同样是长清单
        from gui.result_view import brief_note

        long_note = ("已剔除 40 个在该标的上没有足够数据的因子："
                     + "、".join(f"因子{i}'" for i in range(40)))
        cut = brief_note(long_note)
        if len(cut) > 121 or not cut.endswith("…"):
            raise RuntimeError(f"长说明没压住：{len(cut)} 字 / {cut[-12:]!r}")
        if "已剔除 40 个" not in cut:
            raise RuntimeError(f"压缩把「几个」丢了：{cut[:20]!r}")
        if brief_note("短说明") != "短说明":
            raise RuntimeError("短说明被改动了")
        # 截断点要落在顿号后：切在因子名中间比少给一个名字更难读
        if not cut.endswith("、…") and not cut.endswith("…"):
            raise RuntimeError(f"截断点不在顿号后：{cut[-6:]!r}")
        return (f"40 条压成 {len(brief)} 字「{brief}」，混原因「{b2}」，"
                f"长说明 {len(long_note)}→{len(cut)} 字")

    def _window():
        """真正把主窗口搭起来，验证 UI 代码在打包后也能跑通。"""
        from PySide6.QtWidgets import QApplication, QLabel

        from gui.app import (
            COL_METHOD,
            COL_NAME,
            COL_PERIOD,
            COL_USE,
            COL_W,
            FAMILY_RULE,
            STYLE_SHEET,
            MainWindow,
            brief_failed,
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
        # 2026-11-20：单因子 / 多因子也走后台线程了，`run_analysis()` 返回时
        # `res` 可能还没算出来。这里把事件循环转几圈等那条线程收工，再判
        # 「首屏有没有结果」—— 不等的话这一项会随机器快慢时红时绿
        for _ in range(300):
            app.processEvents()
            _th = getattr(win, "_f_thread", None)
            if _th is None or not _th.isRunning():
                break
            time.sleep(0.02)
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

        # 回溯区间必须真的传到期权那条线（`TqWorker.range_start`）。
        # 这两个值错了的后果很不对称：
        #   · 永远取全量 → 点了「近 1 天」还是要跑 27 小时，两千多条已摘牌老链
        #     一条不落地扫（这就是用户报的那个现象）；
        #   · 反过来把「近 20 年」也算成一天 → 首轮全量回填被悄悄掐成一天，
        #     而且不报错，只是库里永远只有最近几天的 IV。
        opt_range_info = ""
        opt_range_ok = True
        try:
            from datetime import date as _d, timedelta as _td

            from gui.data_page import TqWorker as _TqW
            from ttrade import optfetch as _OPT

            _near = _TqW("1d", 1).range_start()
            _all = _TqW("1d", 7300).range_start()
            opt_range_ok = (_near == _d.today() - _td(days=1)
                            and _all == _OPT.DEFAULT_START)
            opt_range_info = f"近1天→{_near} / 近20年→{_all}"
        except Exception as _exc:      # noqa: BLE001
            opt_range_ok = False
            opt_range_info = f"{type(_exc).__name__}: {_exc}"

        # 定时日程必须还是五条，且「宏观 / 基本面 / 天勤」这三条**不是补 CSV** 的
        # 要各自带着 `macro` / `edb` / `tq` 标、`range` 置 None。
        # 漏掉或并回市场队列的后果是静默的：界面上勾着「每天补基本面」，到点
        # `_start_auto_next()` 却拿 `UpdateWorker` 去补期货 CSV，edb.duckdb
        # 一行不涨、也不报错。这条断言就是防它被谁合并回去。
        auto_jobs_info = ""
        auto_jobs_ok = True
        try:
            _jobs = dp._auto_jobs() if dp is not None else []
            _keys = [j["key"] for j in _jobs]
            _bad = []
            for _j in _jobs:
                _flags = [f for f in ("macro", "edb", "tq") if _j.get(f)]
                if _j["key"] == "macro":
                    # 宏观 + 基本面**一条日程**（用户 11-17 17:01 要求合并），
                    # 两个时刻都要在，串行链也要在 —— 少了 `chain` 就只跑宏观，
                    # EDB 静默不更新，日志里看不出来
                    if sorted(_flags) != ["edb", "macro"]:
                        _bad.append(f"macro 标记是 {_flags}")
                    if len(_j.get("times") or []) != 2:
                        _bad.append("宏观日程不是两个时刻")
                    if "edb" not in (_j.get("chain") or []):
                        _bad.append("宏观日程没有 edb 串行链")
                elif _j["key"] == "tq":
                    if _flags != ["tq"]:
                        _bad.append(f"tq 标记是 {_flags}")
                    if len(_j.get("times") or []) != 1:
                        _bad.append("天勤日程时刻数不对")
                else:
                    if _flags:
                        _bad.append(f"{_j['key']} 多带了标记")
                    if len(_j.get("times") or []) != 1:
                        _bad.append(f"{_j['key']} 时刻数不对")
                if _j["key"] in ("macro", "tq") and _j.get("range") is not None:
                    _bad.append(f"{_j['key']} 不该有区间")
            auto_jobs_ok = _keys == ["aux", "daily", "macro", "tq"] and not _bad
            auto_jobs_info = " / ".join(_keys) + ("；" + "、".join(_bad) if _bad else "")
        except Exception as _exc:      # noqa: BLE001
            auto_jobs_ok = False
            auto_jobs_info = f"{type(_exc).__name__}: {_exc}"

        # 切换表必须是**从天勤下载刷新**的，不能退回「表空的才补」。
        # 这条防的是一个特别隐蔽的退化：`ensure_switch()` 只给空表补一张，
        # 已经有表的品种永远不再更新 —— 于是每次换月都没 ratio，前复权序列在
        # 换月处留一根假跳空（JD 2026-11-17 从 jd2612 换 jd2701 差 −10%），
        # 而界面上一切正常、不报任何错。
        switch_src_info = ""
        switch_src_ok = True
        try:
            import inspect as _inspect

            from gui import data_page as _dp
            from ttrade import tqfetch as _TQ

            _run = _inspect.getsource(_dp.TqWorker.run)
            _bad = []
            if "update_switch" not in _run:
                _bad.append("补充流程没调 update_switch")
            if "ensure_switch" in _run:
                _bad.append("还在用 ensure_switch（只补空表）")
            for _fn in ("update_switch", "apply_switch", "switch_fresh"):
                if not callable(getattr(_TQ, _fn, None)):
                    _bad.append(f"tqfetch 少了 {_fn}()")
            if "force_switch" not in _inspect.getsource(_dp.TqWorker.__init__):
                _bad.append("TqWorker 没有 force_switch 开关")
            switch_src_ok = not _bad
            switch_src_info = "、".join(_bad)
        except Exception as _exc:      # noqa: BLE001
            switch_src_ok = False
            switch_src_info = f"{type(_exc).__name__}: {_exc}"

        # 定时补天勤必须**三个源周期全补、日线排第一**；期权**只补日线**
        # （用户 11-17 16:15 / 16:17 定的口径）。少一条的后果全是静默的：
        #   · 少 1d → 1d 表停更，而交易日历是从 1d 推的 ⇒ 洞检测和策略端缺口闸
        #     一起失灵，界面天天报「补完了」；
        #   · 少 5m/1h → 策略跑的周期没人续，库停在某一天（11-17 的 IF）；
        #   · 期权跟着界面选到 5m → 只有中金所 IO/MO/HO 有数据，别的品种一条都
        #     补不到，还看不出是漏了。
        tq_periods_info = ""
        tq_periods_ok = True
        try:
            from gui import data_page as _dp2
            from ttrade import tqfetch as _TQ2

            _bad = []
            if list(_dp2.TQ_SOURCE_PERIODS) != ["1d", "1h", "5m"]:
                _bad.append(f"源周期是 {_dp2.TQ_SOURCE_PERIODS}")
            elif "1d" != _dp2.TQ_SOURCE_PERIODS[0]:
                _bad.append("日线没排第一（日历要先刷新）")
            if list(_dp2.TQ_OPT_PERIODS) != ["1d"]:
                _bad.append(f"期权周期是 {_dp2.TQ_OPT_PERIODS}")
            _st = _inspect.getsource(_dp2.DataPage._start_tq)
            if _st.count("TQ_SOURCE_PERIODS") < 2:
                _bad.append("定时/手动没有都按三周期走")
            if "force_switch=True" not in _st:
                _bad.append("切换表没做到每轮真刷（同一天手动补过就跳过）")
            _run2 = _inspect.getsource(_dp2.TqWorker.run)
            if "TQ_OPT_PERIODS" not in _run2:
                _bad.append("期权没固定用 TQ_OPT_PERIODS")
            if "periods=(self.period," in _run2:
                _bad.append("期权还在跟界面周期")
            if "calendar(refresh=True)" not in _inspect.getsource(_TQ2.write_main_many):
                _bad.append("补完 1d 没作废日历缓存")
            tq_periods_ok = not _bad
            tq_periods_info = "、".join(_bad) or " / ".join(_dp2.TQ_SOURCE_PERIODS)
        except Exception as _exc:      # noqa: BLE001
            tq_periods_ok = False
            tq_periods_info = f"{type(_exc).__name__}: {_exc}"

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
            for key, chk, tms, rng in (
                ("aux", d.chk_auto, [d.time_auto], d.cb_auto_range),
                ("daily", d.chk_daily, [d.time_daily], d.cb_daily_range),
                # 宏观 + 基本面合成了一条，**两个时刻**（08:45 / 20:45），存盘里
                # 是 `time` 和 `time2` 两个键
                ("macro", d.chk_macro_auto,
                 [d.time_macro_auto, d.time_macro_auto2], None),
                # 天勤那条没有区间下拉（`range` 存的是 null），只比勾选和时刻
                ("tq", d.chk_tq_auto, [d.time_tq_auto], None),
            ):
                one = saved.get(key) or {}
                if bool(one.get("enabled", False)) != chk.isChecked():
                    return False
                for edit, field in zip(tms, ("time", "time2")):
                    if one.get(field) and edit.time().toString("HH:mm") != str(one[field]):
                        return False
                if one.get("range") and (rng is None
                                         or rng.currentData() != one["range"]):
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
            # 每日定时补充：几条日程（衍生 / 日线 / 宏观 / 基本面 / 天勤）的控件
            # 都在、默认不勾（不能让用户装完就被自动跑一次几十分钟的下载），
            # 市场那两条的区间都是「近 1 天 / 近 1 周」两档
            and hasattr(dp, "chk_auto")
            and hasattr(dp, "time_auto")
            and hasattr(dp, "cb_auto_range")
            and hasattr(dp, "chk_daily")
            and hasattr(dp, "time_daily")
            and hasattr(dp, "cb_daily_range")
            # 天勤那条（切换表 → 主连 → 复权）也要有开关和时刻。
            # **不许断言它「默认没勾」**：用户打开定时之后自检就必红 —— 那是
            # 设置，不是代码坏了。真正的判据是「界面上和存盘的一致」，在
            # `_sched_ok()` 里比。
            and hasattr(dp, "chk_tq_auto")
            and hasattr(dp, "time_tq_auto")
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
                "账户", "连接", "账户平台", "资金账户", "最新权益", "当日盈亏",
                "净杠杆", "总杠杆", "可用资金", "成交金额", "持仓保证金", "手续费"]
            and _heads(tp.tbl_pos)[:2] == ["资金账户", "合约"]
            and _heads(tp.tbl_orders)[:2] == ["资金账户", "时间"]
            and _heads(tp.tbl_trades)[:2] == ["资金账户", "时间"]
            # 默认全勾：以前是「下拉框选中谁就连谁」，现在多个账户并存
            and tp._checked == {s.name for s in tp._accounts}
            # 「连接」列每行一个按钮：逐账户登录 / 登出的入口，少一个就只能整批
            and tp.tbl_accts.rowCount() == len(tp._accounts)
            and len(tp._acct_btns) == len(tp._accounts)
            and all(tp.tbl_accts.cellWidget(i, 1) is not None
                    for i in range(tp.tbl_accts.rowCount()))
        )
        trade_info = (f"{tp.tbl_accts.columnCount()} 列账户表 / "
                      f"{len(tp._accounts)} 个账户" if tp is not None else "无交易页")

        # 多柜台同时登录：分组 / 合并 / 行情源 / 融航口径。
        # 「天勤和 CTP 一起登」是靠**按柜台分组建多条连接**实现的，不是把它们塞进
        # 同一条会话；这条断言盯的就是那个分组函数和合并函数 —— 一旦有人把融航
        # 或 CTP 又算成天勤，界面会静默退化成「只连上天勤那几个户」，账户表看着
        # 有行、数字全是 —。
        acct_ok = True
        acct_info = ""
        try:
            from gui.live import group_specs
            from ttrade.ctp_live import RohonHub
            from ttrade.tqlive import (ACCOUNT_KINDS as _KINDS,
                                       BACKENDS as _BACKENDS,
                                       AccountSpec as _AS,
                                       LiveState as _LS,
                                       merge_states)

            _tq = _AS(name="自检天勤", kind="kq", user="u", password="p")
            _ctp = _AS(name="自检CTP", kind="ctp", user="u", password="p")
            _roh = _AS(name="自检融航", kind="rohon", user="u", password="p")
            _g = group_specs([_tq, _ctp, _roh])
            acct_ok = (list(_g) == ["tq", "ctp", "rohon"]
                       and [len(_g[k]) for k in _g] == [1, 1, 1])

            def _fake(tag, bal):
                acc = {"balance": bal, "available": bal, "margin": 0.0}
                return _LS(connected=True, key=f"{tag}_K", period="5m",
                           underlying=f"{tag}_rb2601",
                           quote={"last": float(bal)},
                           accounts=[{"name": tag, "account": acc}],
                           by_account={tag: {"spec": {"name": tag},
                                             "account": acc, "positions": [],
                                             "orders": [], "trades": []}},
                           account=acc)

            _m = merge_states([("tq", _fake("天勤户", 100.0)),
                               ("ctp", _fake("CTP户", 50.0))], "ctp")
            acct_ok = (acct_ok
                       and _m.quote.get("last") == 50.0
                       and _m.underlying == "CTP户_rb2601"
                       and set(_m.by_account) == {"天勤户", "CTP户"}
                       and round(_m.account.get("balance") or 0, 2) == 150.0)
            # 融航：账户类型要在表里、backend 要走 rohon、Hub 的柜台名要对。
            # 现在融航走 CTP 兼容 dll（vendor/rohon_ctp），**不再依赖 vnpy_rohon**，
            # 所以这里不再要求报错里出现包名；缺运行时的话，文案要能指到 vendor
            acct_ok = (acct_ok and "rohon" in dict(_KINDS)
                       and _roh.backend == "rohon"
                       and RohonHub.PLATFORM == "融航"
                       and tp is not None and hasattr(tp, "cb_feed"))
            try:
                RohonHub([]).check_deps()
            except Exception as _exc:      # noqa: BLE001
                _msg = str(_exc)
                acct_ok = acct_ok and "融航" in _msg and "vendor" in _msg

            # ---- 「策略交易和组合」跑柜台账户（CTP / 融航）
            # 以前这一页只认天勤账户，CTP / 融航 配进来要等到 `start()` 最后一步
            # 才炸（前面还先补了几分钟数据）。现在柜台账户能跑：历史行情读本地
            # 天勤数据、实时行情和下单走柜台（`ttrade/broker_api.py`）。这条断言
            # 盯四件事：
            #   1. 体检挪到补数据和连柜台**之前**（看源码里的先后顺序）；
            #   2. 真正跑不了的只有「柜台 + 股票」，而且报错说清为什么；
            #   3. `start()` 真的按 `backend` 分叉到 BrokerApi；
            #   4. 界面上账户名字标出「历史行情读本地天勤」，别让人以为柜台给历史。
            from gui.portfolio_page import BROKER_HINT, account_label, auto_ok
            from ttrade.strategy_runner import PortfolioRunner, require_live_account

            import inspect as _inspect

            _msg_stock = ""
            try:
                require_live_account(_AS(name="自检CTP", kind="ctp",
                                         user="u", password="p",
                                         account_id="1"))
                # 柜台 + 股票：`kind` 换成股票模拟再套 ctp 后端走不通，只能直接
                # 造一个「backend=ctp 且 is_stock」的组合，走同一条判据
                class _BrokerStock(_AS):
                    kind = "ctp"

                    @property
                    def is_stock(self):        # noqa: D102 - 自检桩
                        return True
                require_live_account(_BrokerStock(name="自检股票CTP", kind="ctp",
                                                 user="u", password="p"))
            except ValueError as _exc:
                _msg_stock = str(_exc)
            _msg_pwd = ""
            try:
                require_live_account(_AS(name="自检空密码", kind="ctp", user="u"))
            except ValueError as _exc:
                _msg_pwd = str(_exc)
            _src = _inspect.getsource(PortfolioRunner.start)
            _conn = _inspect.getsource(PortfolioRunner._connect_account)
            _auxsrc = _inspect.getsource(PortfolioRunner._aux_bg)
            _i_guard = _conn.find("require_live_account(self.spec)")
            # 衍生数据改后台（`_aux_bg` 线程）：同步跑 `_ensure_aux()` 实测 15 秒
            # （2026-11-20「界面卡住」），所以这里查的是「start 里有没有把它甩到
            # 后台」，以及那个后台函数里确实还在调 `_ensure_aux()`
            _i_aux = _src.find("self._aux_bg")
            if _auxsrc.find("self._ensure_aux()") < 0:
                _i_aux = -1
            # 🔴 2026-11-24：天勤那一支不再自己 `TqApi(...)`，改成到 `hubreg`
            # 借会话（一个天勤账户在整个系统只能有一条会话）。文案跟着换 ——
            # 这里盯的是「体检在前、连账户在后、天勤分支在柜台分支之前」，
            # 不是某个具体的构造调用
            _i_api = _conn.find('hubreg.acquire("tq"')
            _i_broker = _conn.find("self._broker_api(backend)")
            # 2026-11-20 起账户只是**最后一步**：行情先起，衍生数据次之，账户最后。
            # 顺序错了就退回「账户连不上 ⇒ 策略不跑」的老样子
            _i_feed = _src.find("self.feed = MarketFeed.get()")
            _i_acct = _src.find("self._connect_account()")
            _order = (_i_guard >= 0 and _i_aux > 0 and _i_api > 0
                      and _i_feed > 0 and _i_acct > _i_feed
                      and _i_aux < _i_acct and _i_guard < _i_api)
            acct_ok = (acct_ok
                       and require_live_account(_tq) is None
                       and require_live_account(_ctp) is None
                       and require_live_account(_roh) is None
                       and bool(_msg_stock) and "期货" in _msg_stock
                       and "自检股票CTP" in _msg_stock
                       and bool(_msg_pwd) and "缺少" in _msg_pwd
                       and _order and _i_broker > _i_api
                       and auto_ok(_tq) and auto_ok(_ctp)
                       and BROKER_HINT in account_label(_ctp)
                       and BROKER_HINT not in account_label(_tq))

            # 执行层本身：接口形状 + 时间戳单位 + 委托三态。
            # `_ns` 那条是**实测踩出来的**：pandas 2.x 起 `DatetimeIndex` 的单位
            # 可能是微秒，`asi8` 给的是本单位的整数，当纳秒用整条序列会落到
            # 1970-01-22，而且 `bars_from_serial` 只是把它整段剔掉 —— 表现是
            # 「序列有 300 行、一根 bar 都转不出来」，没有任何报错。
            from ttrade import broker_api as _BA

            import pandas as _pd

            _api_keys = ("get_kline_serial", "get_quote", "get_position", "get_order",
                         "insert_order", "cancel_order", "wait_update", "close",
                         "start")
            _ns_v = _BA._ns(_pd.DatetimeIndex(["2026-11-16 09:00:00"]))[0]
            _expect = int(_pd.Timestamp("2026-11-16 01:00:00Z").value)
            acct_ok = (acct_ok
                       and all(hasattr(_BA.BrokerApi, k) for k in _api_keys)
                       and int(_ns_v) == _expect
                       and _BA._STATUS_MAP.get("全部成交") == "FINISHED"
                       and _BA._STATUS_MAP.get("已撤单") == "DEAD"
                       and "部分成交" not in _BA._STATUS_MAP)
            acct_info = ("分组 " + "/".join(_g)
                         + f"，合并 {len(_m.by_account)} 户 / 权益 "
                         + f"{_m.account.get('balance'):,.0f}"
                         + f"，策略页柜台执行层 {'OK' if _order else '位置不对'}"
                         + ("" if _i_broker > _i_api else "（没接 broker_api）"))
        except Exception as _exc:          # noqa: BLE001
            acct_ok = False
            acct_info = f"{type(_exc).__name__}: {_exc}"
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

        # 「口径」列：每行一个三态下拉（跟 / z / 分），**默认必须是「跟」**。
        # 出厂就替用户选好「分位」有两个坏处：他自己不知道被动过，而且以后
        # 改全局默认口径时这一格会不跟着动（显式覆盖 ≠ 跟随）。
        method_ok = False
        method_info = ""
        try:
            _mc = win.tbl_fmulti.cellWidget(0, COL_METHOD)
            _opts = {_mc.itemData(i) for i in range(_mc.count())}
            _n_combos = sum(
                1 for r in range(win.tbl_fmulti.rowCount())
                if type(win.tbl_fmulti.cellWidget(r, COL_METHOD)).__name__ == "QComboBox"
            )
            method_ok = (type(_mc).__name__ == "QComboBox"
                         and _opts == {"", "z", "rank"}
                         and (_mc.currentData() or "") == ""
                         and _n_combos == win.tbl_fmulti.rowCount())
            method_info = (f"{_n_combos}/{win.tbl_fmulti.rowCount()} 行有下拉，"
                           f"选项 {sorted(str(o) for o in _opts)}，"
                           f"默认 {(_mc.currentData() or '跟')!r}")
        except Exception as exc:  # noqa: BLE001
            method_info = f"{type(exc).__name__}: {exc}"

        factor_ok = (
            fams == {"rule", "factor", "cross"}
            and modes == {"single", "multi", "ml"}
            and filters == {"entry", "delay", "hold"}
            # 方向口径三档必须都在，否则「反转」只能相对自动定向，跨品种不可比
            and orients == {"auto", "long_high", "long_low"}
            and n_factors >= 47
            and n_models >= 4
            # 多因子表要 47 行 × 6 列（因子 / 周期 / 用 / 反 / 口径 / 权重），
            # 少了「反」列就没法逐因子反转，少了「权重」列「不等权」就是空话，
            # 少了「周期」列就没法跨周期引用，少了「口径」列就只能在整条策略上
            # 统一用一种标准化方法（仓单要分位、动量要 z 就做不到）
            and n_rows_f >= 47
            and n_cols_f == 6
            and period_ok
            and method_ok
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
                       f"{n_models} 模型 / 勾选 {pick_info} / "
                       f"表 {n_rows_f}行×{n_cols_f}列 / 周期 {period_info} / "
                       f"口径 {method_info}")

        # 逐因子口径的**端到端往返**：界面上把一个因子改成「分」→ `_factor_cfg()`
        # 必须把它收进 `method_of` → `_apply_factor_keys()` 读回来还是「分」。
        #
        # 为什么非要走窗口：`FactorConfig.method_of` 本身在别处已验过（第 80 项），
        # 但界面这一层最容易「存了没读 / 读了没存」—— #135 就是 `_norm` 白名单
        # 漏字段，配置读回后静默变成另一个口径，而笔数对拍完全看不出来。
        gauge_ok = False
        gauge_info = ""
        try:
            from PySide6.QtCore import Qt as _Qt

            _st = win._multi_state()
            _pers = win._factor_periods()      # 改动前抓，往返要用原样带回去
            _k0 = next(iter(_st))
            _mc0 = win.tbl_fmulti.cellWidget(0, COL_METHOD)
            _backup = _mc0.currentData()
            _mc0.blockSignals(True)
            _mc0.setCurrentIndex(_mc0.findData("rank"))
            _mc0.blockSignals(False)
            _cfg1 = win._factor_cfg()
            _got = dict(_cfg1.method_of or {})
            # 再走一遍「载入策略」的路径：应该把这一格恢复成「分」，
            # 其余没配的落回「跟」
            # `_factor_cfg()` 给的是 `FactorConfig`（只有因子配置，没有 keys），
            # 键清单得从表格读；周期也用改动前的快照带回去，否则这趟往返会把
            # 跨周期设置清成「本周期」，后面几项检查会看到莫名其妙的变化
            _used = [k for k, v in _st.items() if v[0]]
            _rev = {k: True for k, v in _st.items() if v[1]}
            win._apply_factor_keys(_used, _rev, _pers, _got)
            _after = (_mc0.currentData() or "")
            _others = {str(win.tbl_fmulti.cellWidget(r, COL_METHOD).currentData() or "")
                       for r in range(win.tbl_fmulti.rowCount())
                       if win.tbl_fmulti.cellWidget(r, COL_METHOD) is not None
                       and win.tbl_fmulti.item(r, COL_NAME) is not None
                       and str(win.tbl_fmulti.item(r, COL_NAME).data(_Qt.UserRole))
                       != _k0}
            gauge_ok = (_got == {_k0: "rank"} and _after == "rank"
                        and _others <= {""})
            gauge_info = (f"{_k0} 收集 {_got} / 读回 {_after!r} / "
                          f"其余格 {sorted(str(o) for o in _others)}")
            # 还原：口径回原值，勾选按备份恢复（`_apply_factor_keys` 会把
            # 不在 keys 里的行取消勾选，不还原会影响后面的检查）
            _mc0.blockSignals(True)
            _mi = _mc0.findData(_backup or "")
            _mc0.setCurrentIndex(_mi if _mi >= 0 else 0)
            _mc0.blockSignals(False)
            win._apply_factor_keys(
                _used, _rev, _pers, {k: v[4] for k, v in _st.items() if v[4]})
        except Exception as exc:  # noqa: BLE001
            gauge_info = f"{type(exc).__name__}: {exc}"

        if not gauge_ok:
            raise RuntimeError(f"逐因子口径界面往返失败：{gauge_info}")

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

        # 2026-11-20：单因子 / 多因子也改成**后台线程**了（`_start_factor_worker`
        # 不再只给机器学习开）—— 22 万根 5m × 十几个因子实测几秒，同步跑在主线程
        # 就是用户报的「策略研究页卡住」。代价是首屏结果不再在窗口构造返回前就位，
        # 所以这里先等那条线程收工再断言画布；不等的话断的是「还没算完」，
        # 不是「画布坏了」，会变成随机器快慢时红时绿的假失败。
        for _ in range(300):
            app.processEvents()
            _th = getattr(win, "_f_thread", None)
            if _th is None or not _th.isRunning():
                break
            time.sleep(0.02)
        app.processEvents()

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
            for key in ("ria_break_day", "ria_break_night", "cta_break",
                        "cta_break2"):
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
                      and counts.get("cta_break2") == 6
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
            # 组合的本金必须是**各策略之和**，不是 `combine` 的默认 10 万：
            # 两条本金各 10 万的策略凑成组合，「本金」那行得写 20 万。本金是
            # 单利口径下所有比例的分母，写错它整张绩效表都是错的
            legs = list((getattr(pv.res, "legs", None) or {}).values())
            sum_base = sum(float(r.metrics.get("initial_cash") or 0) for r in legs)
            combo_base = float((getattr(pv.res, "metrics", {}) or {})
                               .get("initial_cash") or 0)
            combo_ok = (
                tabs is not None
                and pv is not None
                and tabs.count() == 4
                and pv.table.columnCount() == 2
                and pv.res is not None
                and "组合（2 条策略）" in text
                # 「补数据」按钮：期权 / 展期数据过期时，这一页要能就地补
                and hasattr(win.portfolio_page, "btn_data")
                # 分策略曲线：绩效页的下拉框得能切到每一条
                and len(legs) == 2
                and pv.cb_leg is not None and pv.cb_leg.isEnabled()
                and pv.cb_leg.count() == 3
                and pv.lbl_leg is not None and pv.lbl_leg.text() == "策略"
                # 本金 = 各策略之和（这里两条各 10 万 ⇒ 20 万，不是 10 万）
                and sum_base > 0
                and abs(combo_base - sum_base) < 1e-6
                and abs(combo_base - 100_000.0) > 1.0
            )
            combo_info = (f"{tabs.count()} 页 / {pv.table.columnCount()} 列"
                          if tabs is not None else "未建出视图")
            combo_info += (f" / 本金 {combo_base:,.0f}（{len(legs)} 条策略，"
                           f"合计 {sum_base:,.0f}）")
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
        if not opt_range_ok:
            raise RuntimeError(
                f"回溯区间没接到期权（{opt_range_info}）：选「近 1 天」期权也该只补"
                "最近一天，选「近 20 年」该等于全量回填"
            )
        if not auto_jobs_ok:
            raise RuntimeError(
                f"每日定时日程不对（{auto_jobs_info}）：应该是 aux / daily / macro / "
                "edb / tq 五条，宏观、基本面、天勤各自带 macro / edb / tq 标且没有"
                "区间 —— 少了标记它们就会被当市场队列，到点补的是 CSV，"
                "基本面一行不涨还不报错"
            )
        if not switch_src_ok:
            raise RuntimeError(
                f"主力切换表没有真正从天勤更新（{switch_src_info}）：必须走 "
                "update_switch()，只补空表的 ensure_switch() 会让换月后新增的换月"
                "记录永远进不来，前复权序列在换月处留假跳空"
            )
        if not tq_periods_ok:
            raise RuntimeError(
                f"天勤补数的周期口径不对（{tq_periods_info}）：主连和真实合约都要 "
                "1d + 1h + 5m（日线打头，它兼作交易日历来源），期权只补日线 —— "
                "少一条都是静默的：库停在某一天、洞检测失灵、展期族整列无数据"
            )
        if not trade_ok:
            raise RuntimeError(
                f"账户透视页异常（{trade_info}）：账户列表 / 资金账户列 / 时间列缺失"
            )
        if not acct_ok:
            raise RuntimeError(
                f"多柜台（天勤 / CTP / 融航）口径不对（{acct_info}）：勾选跨柜台时"
                "要按柜台分成多条连接，账户合并、行情只取选定的那一路；融航账户"
                "必须是 rohon 后端，不能落进天勤分支；另外「策略交易和组合」只跑"
                "天勤账户，CTP / 融航 必须在补数据与连 tqsdk 之前就被挡下来，"
                "并在账户名字上标出「本页不支持」"
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

            # ---- 板块级（多品种）配置：`symbol` / `exchange` 是空的 ----
            #
            # 上面那条往返走的是 `_current_config()`，它一定会带一个真实 symbol，
            # 所以从没覆盖到「板块级」这种形态 —— 用户报的「产业板块的多品种策略
            # 注册完再加载，品种勾选没了」就是这么溜过去的。原形是两条判据只看单
            # 品种字段：`market_of()` 见 exchange 空就判 us（美股，整个品种池被换
            # 掉）、`key()` 返回空串（下拉框定位落空）。这里把代表品种抹掉再灌一
            # 次，勾选必须原样回来。
            bare_rt = cfg_rt.clone(symbol="", exchange="")
            win.sym_pick.clear()
            app.processEvents()
            win.apply_config(bare_rt)
            app.processEvents()
            bare_sel = list(win.sym_pick.selected())
            if bare_sel != before:
                raise RuntimeError(
                    f"板块级（无 symbol/exchange）配置灌回界面丢了品种："
                    f"{len(bare_sel)} 个 vs {len(before)} 个"
                    f"（market={bare_rt.market}、key={bare_rt.key()!r}）")
            multi_rt_info += f" / 板块级 {len(bare_sel)} 个也回来"
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
                # 因子参数整块在**第二步**（第一步只负责「选哪些品种」）
                and win.box_fparams.parentWidget() is win.cross_fparam_host
                # 截面对冲没有单腿进场，阈值一个都不该出现
                and "entry_z" not in win.fparam_widgets
                and "exit_z" not in win.fparam_widgets
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
            # 时序腿换成「因子」：阈值必须回来 —— 那条腿就是靠分数过阈值开平的。
            # 截面对冲拿掉阈值是因为**没有这条腿**，不是「截面不用阈值」。
            win.cb_timing.setCurrentIndex(win.cb_timing.findData("factor"))
            app.processEvents()
            tf_ok = ("entry_z" in win.fparam_widgets
                     and "exit_z" in win.fparam_widgets
                     and win.box_fparams.parentWidget() is win.cross_fparam_host)
            win.cb_timing.setCurrentIndex(win.cb_timing.findData("rule"))
            app.processEvents()
            stage_ok = tf_ok
            win.cb_family.setCurrentIndex(win.cb_family.findData("rule"))
            app.processEvents()
            win.cross_dlg.close()
            back_ok = _order(side_lay) == before
            cross_dlg_ok = hedge_ok and select_ok and stage_ok and back_ok
            cross_dlg_info = (f"对冲 {hedge_ok} / 选品种 {select_ok} / "
                              f"时序因子阈值回来 {stage_ok} / "
                              f"还回侧边栏 {back_ok} / 第一步周期 "
                              f"{p1_ok},{p1_pick},{p1_note},{p1_back} / 杠杆 {lev_ok}")
        except Exception as exc:  # noqa: BLE001
            cross_dlg_info = f"{type(exc).__name__}: {exc}"
        if not cross_dlg_ok:
            raise RuntimeError(
                f"截面两步配置窗异常（{cross_dlg_info}）：参数面板借进弹窗、切走还回"
                "侧边栏，或者侧边栏顺序被搬乱了，或者「第一步因子周期」选不动"
            )

        # ---- 两步窗的**归属**：左边只筛品种，右边选多空 ----
        #
        # 用户 2026-11-23 两次要求：先把「因子模式 / 截面因子 / 等权」挪到右边、
        # 再把「调仓频率」也挪过去，并明确「左边的选项都是用于第一步筛选品种的」。
        # 这里钉的是归属而不是长相：换了布局、但某个控件还留在错的栏里，界面上
        # 几乎看不出来（两栏都能滚），只有这条断言会红。
        place_ok, place_info = False, ""
        _pool_was = win.chk_pool.isChecked()
        try:
            win.cb_family.setCurrentIndex(win.cb_family.findData("cross"))
            app.processEvents()
            s1, s2 = win.cross_step1, win.cross_step2

            def _in(w, host):
                return w is host or host.isAncestorOf(w)

            left = (_in(win.grp_pool, s1) and _in(win.sp_pool_reb, s1)
                    and _in(win.cb_pool_reb_unit, s1)
                    and _in(win.cb_step1_period, s1)
                    and _in(win.cb_cross_mode, s1))
            right = (_in(win.cross_factor_host, s2) and _in(win.box_fsel, s2)
                     and _in(win.cross_fparam_host, s2)
                     and _in(win.sp_cross_reb, s2)
                     and _in(win.cb_cross_reb_unit, s2)
                     and _in(win.grp_legs, s2))
            wrong = [nm for nm, w in (("截面因子", win.cross_factor_host),
                                      ("调仓频率", win.sp_cross_reb),
                                      ("多空腿数", win.grp_legs))
                     if _in(w, s1)]
            units = [win.cb_pool_reb_unit.itemData(i)
                     for i in range(win.cb_pool_reb_unit.count())]
            unit_ok = units == ["day", "month"]      # 只要交易日 / 月，不要「根」
            # 频率控件跟着「先筛品种池」的开关走：没开筛池就没有频率可谈。
            # 🔴 这里**屏蔽信号**再手动同步显隐：`setChecked` 会走
            # `_on_cross_param_changed` → `_schedule()` → 后台起一次截面回测，
            # 而窗口马上要被销毁，线程还在跑 ⇒ 收尾时偶发段错误（实测 exit=139）。
            # 这条验的是「谁跟谁一起显示」，不需要真算一遍。
            win.chk_pool.blockSignals(True)
            win.chk_pool.setChecked(False)
            win.chk_pool.blockSignals(False)
            win._sync_cross_controls()
            app.processEvents()
            hide_ok = not win.row_pool_reb.isVisible()
            win.chk_pool.blockSignals(True)
            win.chk_pool.setChecked(True)
            win.chk_pool.blockSignals(False)
            win._sync_cross_controls()
            app.processEvents()
            show_ok = win.row_pool_reb.isVisible()
            # 侧边栏摘要里要写出频率：不开窗也得看得见「多久重排一次」
            sum_ok = "重排" in win.lbl_cross_sum.text()
            place_ok = (left and right and not wrong and unit_ok
                        and hide_ok and show_ok and sum_ok)
            place_info = (f"左筛品种 {left} / 右选多空 {right} / 留在左边的 {wrong} / "
                          f"单位 {units} / 藏-显 {hide_ok}-{show_ok} / 摘要 {sum_ok}")
        except Exception as exc:  # noqa: BLE001
            place_info = f"{type(exc).__name__}: {exc}"
        finally:
            win.chk_pool.blockSignals(True)
            win.chk_pool.setChecked(_pool_was)
            win.chk_pool.blockSignals(False)
            win._sync_cross_controls()
            app.processEvents()
        if not place_ok:
            raise RuntimeError(f"截面两步窗归属不对（{place_info}）")

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
                _cell = win.tbl_fmulti.item(_r, COL_USE)
                if _cell is not None:
                    _cell.setCheckState(Qt.Checked)
            app.processEvents()
            rows = [r for r in range(win.tbl_fmulti.rowCount())
                    if win.tbl_fmulti.item(r, COL_USE) is not None
                    and win.tbl_fmulti.item(r, COL_USE).checkState() == Qt.Checked]
            if len(rows) < 2:
                raise RuntimeError("自检环境里勾中的因子不足 2 个，权重往返测不了")
            win.tbl_fmulti.item(rows[0], COL_W).setText("3")
            win.tbl_fmulti.item(rows[1], COL_W).setText("0.5")
            app.processEvents()
            want_w = win._factor_cfg()
            cfg_w = win._current_config()
            if not (want_w.weight_mode == "custom"
                    and abs(float(want_w.weights.get(
                        win.tbl_fmulti.item(rows[0], 0).data(Qt.UserRole), 0)) - 3.0) < 1e-9):
                raise RuntimeError(f"权重没进配置：{want_w.weight_mode} {want_w.weights}")
            win.cb_fweight.setCurrentIndex(win.cb_fweight.findData("equal"))
            win.tbl_fmulti.item(rows[0], COL_W).setText("1")
            win.tbl_fmulti.item(rows[1], COL_W).setText("1")
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

        # ---- 状态栏不能被几十条失败清单撑爆 ----
        # 多品种回测里「一个因子在 N 个品种上都没数据」是逐品种写一条的，
        # 四十条原文会把 360 px 的侧边栏撑满整栏、把右边的图挤成一条缝。
        # 这里用真实 `Hint` 样式量高度：压缩后必须不到原文的 1/3，且不超过
        # 三行左右 —— 文案对不对上面那项纯函数检查管，这项只管「会不会挤」。
        _bs, _full = brief_failed(_many_failed())
        _probe = QLabel()
        _probe.setObjectName("Hint")
        _probe.setWordWrap(True)
        _probe.setFixedWidth(360)
        _probe.setText("　".join(_many_failed()))
        _h_raw = _probe.sizeHint().height()
        _probe.setText(_bs)
        _h_new = _probe.sizeHint().height()
        _probe.deleteLater()
        if not (_h_new * 3 <= _h_raw and _h_new <= 90):
            raise RuntimeError(
                f"状态栏没压住：原文 {_h_raw}px → 压缩后 {_h_new}px（{_bs}）")
        status_info = f"多品种失败清单 {_h_raw}→{_h_new}px"

        return (f"{n} 个标的，K 线/绩效/因子页已渲染，{k_info}，"
                f"{factor_info}，滚轮缩放 {wheel_info}，策略验证 {verify_info}，"
                f"波幅突破 {brk_info}，"
                f"组合 {combo_info}，"
                f"截面 {cross_info}，多柜台 {acct_info}，两步配置 {cross_dlg_info}，"
                f"多品种往返 {multi_rt_info}，"
                f"组合页回测 {view_info}，"
                f"{side_info}，触发 {trig_info}，"
                f"回测区间 {range_info}，合成周期 {md_info}，实时开关已复位，"
                f"{status_info}，"
                f"数据管理页就绪（期权回溯区间 {opt_range_info}；定时 {auto_jobs_info}；"
                f"天勤周期 {tq_periods_info}；"
                f"切换表{'真下载' if switch_src_ok else switch_src_info}），手册 {manual_info}，输出 {out_info}，"
                f"对话渲染 {chat_info}，能力包 {skill_info}")

    check("长提示的压缩口径（状态栏 / 因子页脚注）", _status_brief)
    check("构建主窗口（端到端）", _window)

    # K 线与信号页的持仓映射：这一项 import 了 Qt，放在窗口之后跑
    def _strategy_save_action():
        """「存入策略库」动作：能存进去、重名会拒、参数错能当场说清。

        这个动作补的是一个真实断点：`strategy` 动作注册的是规则策略的**代码**，
        而「策略库」里带 sid 的整份配置以前没有任何动作能写 —— Agent 扫完一个
        板块想落成条目，只能停下来让人点界面。

        反证比正例重要：这条路径一旦悄悄存错（比如多品种只存了 `MA` 没有交易所、
        或者重名把别人的条目覆盖掉），界面上看不出任何异常。
        """
        from ttrade import strategy_store as S
        from ttrade.skills.base import REGISTRY

        t = REGISTRY["strategy_save"]
        p, err = t.prepare({
            "name": "自检用·动作存库 2h", "symbol": "MA", "exchange": "CZCE",
            "period": "2h", "factor_keys": ["war_pct", "roll_main_share"],
            "factor_cfg": {"window": 983, "horizon": 20, "entry_z": 1.0,
                           "orient": "long_low"},
        })
        if err:
            raise RuntimeError(f"合法参数被拒：{err}")
        r = t.handler(None, p)
        if not r.ok:
            raise RuntimeError(f"存库失败：{r.error}")
        one = [c for c in S.load_library() if c.name == p["name"]]
        if not one:
            raise RuntimeError("存完在库里找不到")
        got = one[0]

        # 重名必须拒（`overwrite` 没开时）—— 静默覆盖别人比存不进去更糟
        p2, _ = t.prepare(dict(p))
        r2 = t.handler(None, p2)
        if r2.ok:
            raise RuntimeError("重名没被拒，会覆盖已有条目")

        # 多品种：`symbols` 必须带交易所，存完回读要对得上
        pm, err = t.prepare({
            "name": "自检用·动作存库 多品种", "symbols": ["MA_CZCE", "TA_CZCE"],
            "period": "2h", "factor_keys": ["sector_mom_10"],
            "factor_cfg": {"window": 983, "horizon": 20},
        })
        if err:
            raise RuntimeError(f"多品种合法参数被拒：{err}")
        rm = t.handler(None, pm)
        if not rm.ok:
            raise RuntimeError(f"多品种存库失败：{rm.error}")
        multi = [c for c in S.load_library() if c.name == pm["name"]][0]
        if multi.multi_symbols != ["MA_CZCE", "TA_CZCE"]:
            raise RuntimeError(f"多品种回读不对：{multi.multi_symbols}")
        # 🔴 2026-11-16：板块级策略只有 `multi_symbols`、没有 symbol/exchange，
        # 空着会让 `market_of()` 判成 us（美股）、`key()` 返回空串 ⇒ 界面加载时
        # 先切到美股市场、再按空 key 定位品种，两步全落空，多品种勾选被清空
        # （用户报的「注册时没存多品种信息」）。340 条板块级条目全中过。
        # 这里钉死三件事：存盘时补代表品种、市场判据退到品种池、key 不为空
        if (multi.symbol, multi.exchange) != ("MA", "CZCE"):
            raise RuntimeError(
                f"多品种代表品种没补上：{multi.symbol!r}/{multi.exchange!r}")
        if multi.market != "cn":
            raise RuntimeError(f"多品种市场判错了：{multi.market}（应为 cn）")
        if multi.key() != "MA_CZCE":
            raise RuntimeError(f"多品种 key() 不对：{multi.key()!r}")
        bare = multi.clone(symbol="", exchange="")
        if bare.market != "cn" or bare.key() != "MA_CZCE":
            raise RuntimeError(
                f"空 symbol 时兜底失效：market={bare.market} key={bare.key()!r}")

        bad = []
        for args, want in (
            ({}, "name"),
            ({"name": "x", "symbols": ["MA"], "factor_keys": ["war_pct"]}, "交易所"),
            ({"name": "x", "symbol": "MA", "exchange": "CZCE",
              "factor_keys": ["war_pct"], "period": "7m"}, "period"),
            ({"name": "x", "symbol": "MA", "exchange": "CZCE"}, "factor_keys"),
        ):
            _pp, e = t.prepare(args)
            if not e or want not in e:
                bad.append(f"{args} → {e!r}")
        if bad:
            raise RuntimeError("参数校验漏了：" + "；".join(bad))

        return ("存/拒/校验都对：%s 期 %s、多品种 2 个键、4 条非法参数全挡住"
                % (got.period, got.symbol))

    def _sizing_roundtrip():
        """仓位口径存→读往返：五种 `sizing` 一个都不能被改写。

        🔴 这条是补一个**真实发生过**的静默故障：`_norm` 里的白名单只有
        `capital / fixed / risk`，而 `backtest.size_lots` 认五种。存进去的
        `budget`（按分配资金定手数）读回来变成 `capital`（按保证金定手数），
        json 里明明写着 budget —— 同一条策略点开跑出另一条曲线，
        实测 MA 1d 全史累计 +42.4% 变成 +615%，而界面上没有任何提示。

        必须用 `load_library()`（走磁盘 + `_norm`）验，不能用 `add_to_library`
        的返回值：返回值是内存对象、没走归一化，这个洞在返回值上完全看不出来。
        """
        from ttrade import strategy_store as S

        base = dict(symbol="MA", exchange="CZCE", period="1d_tq",
                    family="factor", factor_mode="multi",
                    factor_keys=["close_pos"],
                    factor_cfg={"window": 252, "horizon": 5, "entry_z": 1.0,
                                "exit_z": 0.3, "weight_mode": "equal",
                                "ic_rebal": 60, "method": "z",
                                "orient": "long_low"},
                    principal=1_000_000, capital_pct=50.0)
        made, bad = [], []
        try:
            for s in S.SIZING_OK:
                c = S.StrategyConfig(name="自检用·口径·" + s, sizing=s, **base)
                got = S.add_to_library(c, overwrite=True)
                made.append(got.sid)
            disk = {x.sid: x for x in S.load_library()}
            for s in S.SIZING_OK:
                sid = made[S.SIZING_OK.index(s)]
                back = disk.get(sid)
                if back is None:
                    bad.append(f"{s} 存丢了")
                elif back.sizing != s:
                    bad.append(f"{s} 读回来变成 {back.sizing}")
        finally:
            for sid in made:
                try:
                    S.remove_from_library(sid)
                except Exception:  # noqa: BLE001
                    pass
        if bad:
            raise RuntimeError("仓位口径存不住：" + "；".join(bad))
        if len(S.SIZING_OK) != 5:
            raise RuntimeError("size_lots 的口径数变了，这条检查要跟着改")
        return "5 种口径全部原样返回：" + " / ".join(S.SIZING_OK)

    check("仓位口径存→读往返（budget/unit 不能被改写成 capital）", _sizing_roundtrip)

    def _allow_short_three_state():
        """允许做空只有**一个**判据，且 `note` 的批次标签不能改变方向（#145）。

        原状：回测读 `note == "short"`、界面读 `note == "short" or 市场默认`。
        批量扫描把 `note` 写成批次标签（`sector-pack-v3` / `-q33` / `-m5b` …）
        ⇒ 930 条策略在界面上「允许做空」是勾着的，回测里 `long_only=True`，
        静默只剩多头。417 条抽样实测：86% 空头笔数=0、空头仅占 7%。
        """
        from ttrade import replay as _R
        from ttrade.strategy_store import StrategyConfig as _S

        fut = _S(symbol="MA", exchange="CZCE", period="1h", family="rule",
                 rule_key="ma_cross")
        stk = _S(symbol="600000", exchange="SSE", period="1d", family="rule",
                 rule_key="ma_cross")
        cases = [
            ("期货 note=short", fut.clone(note="short"), True),
            ("期货 note=long", fut.clone(note="long"), False),
            ("期货 批次标签", fut.clone(note="sector-pack-v3"), True),
            ("期货 note 空", fut.clone(note=""), True),
            ("A股 批次标签", stk.clone(note="sector-pack-v3"), False),
            ("A股 note=short", stk.clone(note="short"), True),
            ("只做空 note=long", fut.clone(note="long", short_only=True), True),
        ]
        bad = []
        for name, c, want in cases:
            got = _R.resolve_allow_short(c)
            if got != want:
                bad.append(f"{name}：{got} != {want}")
            # 判据必须真的落到回测配置上，否则「判对但没接线」。
            # A 股/美股由市场自身的 `allow_short=False` 兜住，`long_only` 必为 True，
            # 所以只在市场允许做空时才要求两者相反
            bt = _R.make_bt(c.market, got, c)
            market_ok = bool(_R.MARKET_COST.get(c.market, {}).get("allow_short"))
            if market_ok and bt.long_only == got:
                bad.append(f"{name}：make_bt 的 long_only 没跟着判据走")
            if not market_ok and not bt.long_only:
                bad.append(f"{name}：市场本不允许做空，long_only 却是 False")
        if bad:
            raise RuntimeError("允许做空判据不对：" + "；".join(bad))
        if stk.market != "cn_stock":
            raise RuntimeError(f"A 股市场判错了：{stk.market}")
        return "三态 + 市场默认 7 组全对，批次标签不再裁成只做多"

    check("允许做空判据（note 三态 / 批次标签不外溢）", _allow_short_three_state)

    def _break_caliber():
        """波幅突破的两条口径：检查点时刻、因子基准价（#146）。

        1. 检查点有**三档口径**（`caliber`），2026-11-17 并入一个参数：
           `orig` / `window` / `strict`。这里同时钉住三件事 ——
           a) **默认必须是 `orig`**（用户决定：先让策略库存量曲线可复现）；
           b) `window` / `strict` 下 `cta_break2` 的检查点只落在 **10:05 那根**
              —— 交易日从夜盘 21:00 起算，21:00 那根不能把当天的 `checked` 用掉；
           c) 夜盘腿是**独立注册**的 `cta_break_night`，进场时刻只能是 21:30。
           🔴 只测 `window` 是不够的：默认值一改，这条测试就该红，否则「默认口径
           被谁悄悄改了」永远发现不了。
        2. 用户因子 `open_range_break` 的基准是**当日开盘**、分母是**上一交易日**
           波幅。原来写的是「当根 open / 上一根 high−low」，日线上碰巧对，5m 上
           全错（同一根 bar 旧版 −0.83 / 新版 +0.63，连符号都反）。
        """
        import inspect

        import numpy as np
        import pandas as pd

        from ttrade import factors as F
        from ttrade import strategies as S

        rows = []
        days = ["2024-01-02", "2024-01-03", "2024-01-04"]
        # 日盘 09:00~15:00 + 夜盘 21:00~23:00。夜盘归**下一个**交易日（交易所口径），
        # 这正是「交易日从 21:00 起算」的来源 —— 不造夜盘就测不出检查点的时段限制
        for k, d in enumerate(days):
            t = pd.date_range(d + " 09:00", d + " 15:00", freq="5min")
            drift = 0.0 if k == 0 else (0.006 if k == 1 else -0.006)
            px = 100.0 * np.exp(drift * np.arange(len(t)))
            for i, ts in enumerate(t):
                rows.append((ts, px[i] * 0.999, px[i] * 1.002,
                             px[i] * 0.998, px[i], 1000.0, pd.Timestamp(d)))
            if k + 1 < len(days):
                tn = pd.date_range(d + " 21:00", d + " 23:00", freq="5min")
                pn = px[-1] * np.exp(drift * np.arange(len(tn)))
                for i, ts in enumerate(tn):
                    rows.append((ts, pn[i] * 0.999, pn[i] * 1.002,
                                 pn[i] * 0.998, pn[i], 1000.0,
                                 pd.Timestamp(days[k + 1])))
        df = pd.DataFrame(rows, columns=["dt", "open", "high", "low", "close",
                                         "volume", "trading_day"])
        df = df.set_index("dt").sort_index()
        # 再补一段**跳空夜盘**（01-04 夜 → 交易日 01-05）：夜盘档位要有大幅偏离
        # 才可能触发。前一日波幅被顺势段放大到 35 点，所以跳空得开到 +20%（0.4×35
        # ≈ 14 点）才够得着阈值
        tn = pd.date_range("2024-01-04 21:00", "2024-01-04 23:00", freq="5min")
        jump = np.r_[100.0, np.full(len(tn) - 1, 122.0)]
        extra = pd.DataFrame({"dt": tn, "open": jump * 0.999,
                              "high": jump * 1.002, "low": jump * 0.998,
                              "close": jump, "volume": 1000.0,
                              "trading_day": pd.Timestamp("2024-01-05")})
        df = pd.concat([df, extra.set_index("dt")]).sort_index()

        # 夜盘档位单独测：上面 `df` 里白天的仓会一直持到夜盘（次日 09:05 才平），
        # 把 21:30 那个检查点挡住 —— 那是另一件事，不该混进这条断言。这里造一份
        # 「前一日横盘 + 夜盘跳空」，保证进夜盘时是空仓
        n_rows = [(ts, 99.9, 100.2, 99.8, 100.0, 1000.0,
                   pd.Timestamp("2024-01-08"))
                  for ts in pd.date_range("2024-01-08 09:00",
                                          "2024-01-08 15:00", freq="5min")]
        for i, ts in enumerate(pd.date_range("2024-01-08 21:00",
                                             "2024-01-08 23:00", freq="5min")):
            px = 100.0 if i == 0 else 122.0
            n_rows.append((ts, px * 0.999, px * 1.002, px * 0.998, px,
                           1000.0, pd.Timestamp("2024-01-09")))
        df_n = pd.DataFrame(n_rows, columns=["dt", "open", "high", "low", "close",
                                             "volume", "trading_day"])
        df_n = df_n.set_index("dt").sort_index()

        ms = (df.index.hour * 60 + df.index.minute).to_numpy()
        sp = S.STRATEGIES["cta_break2"]

        def _sig(data, spec, **over):
            p = dict(spec.params)
            p.update(over)
            return spec.func(data, **p)["signal"].to_numpy()

        def _entries(sig):
            return (sig != 0) & (sig != np.r_[0, sig[:-1]])

        # ① 默认口径必须是 orig —— 签在函数签名上，不靠记忆
        dflt = inspect.signature(S.strat_cta_break).parameters["caliber"].default
        if dflt != "orig":
            raise RuntimeError(
                "strat_cta_break 的默认 caliber 应为 orig（存量曲线口径，"
                f"用户 2026-11-17 决定），实际 {dflt!r}")

        # ② window / strict：日盘检查点只落在 10:05 —— 21:00 那根不能把当天的
        #    `checked` 用掉，这正是 #146 修的那个 bug
        win = _sig(df, sp, caliber="window")
        ent = _entries(win)
        if not ent.any():
            raise RuntimeError("合成行情在 caliber=window 下一笔都没开")
        bad = sorted(set(ms[ent]) - {605})
        if bad:
            raise RuntimeError(
                f"cta_break2（caliber=window）不在 10:05 进场，多出时刻 {bad[:4]}")
        strict = _sig(df, sp, caliber="strict")
        if not np.array_equal(win, strict):
            raise RuntimeError("5m 上 window 与 strict 应该逐根等价，实际不同")

        # ③ orig 在同样的行情上结果必须不同（21:00 那根先把 checked 用掉了）；
        #    相同就说明默认值没真的生效
        if np.array_equal(_sig(df, sp, caliber="orig"), win):
            raise RuntimeError("caliber=orig 与 window 结果相同 —— 默认口径没生效")

        # ④ 夜盘腿：独立注册的 cta_break_night 只能 21:30 进场（固定 window 口径）。
        #    用干净行情，避免被白天的持仓挡住
        sn = S.STRATEGIES["cta_break_night"]
        if (sn.params.get("check_min"), sn.params.get("exit_min")) != (1290, 545):
            raise RuntimeError(
                f"cta_break_night 的默认时刻应是 21:30 进 / 09:05 平，"
                f"实际 {sn.params.get('check_min')} / {sn.params.get('exit_min')}")
        s2 = _sig(df_n, sn)
        e2 = _entries(s2)
        ms_n = (df_n.index.hour * 60 + df_n.index.minute).to_numpy()
        if not e2.any() or set(ms_n[e2]) != {1290}:
            raise RuntimeError(
                "cta_break_night 不在 21:30 进场："
                f"{sorted(set(ms_n[e2])) if e2.any() else '一笔没开'}")

        # 因子：当日开盘 + 昨日波幅 + 1.0 = 40%
        g = df.groupby("trading_day")
        o0 = g["open"].transform("first")
        rng = (g["high"].max() - g["low"].min()).shift(1)
        want = (df["close"] - o0) / df["trading_day"].map(rng) / 0.4
        got = F.FACTORS["open_range_break"].func(df)
        dev = float(np.nanmax(np.abs(got.to_numpy() - want.to_numpy())))
        if dev > 1e-9:
            raise RuntimeError(f"波幅突破因子没按当日开盘算：偏差 {dev:.2e}")
        # 反证：拿「当根 open」当基准必须对不上，否则这条检查是摆设
        wrong = (df["close"] - df["open"]) / df["trading_day"].map(rng) / 0.4
        if float(np.nanmax(np.abs(wrong.to_numpy() - got.to_numpy()))) < 1e-6:
            raise RuntimeError("合成行情造得太规整，证明不了当日开盘口径")
        # 必须是**内置 Python 因子**，不是公式因子：公式因子各自在独立命名空间里
        # 编译，既调不到公共 helper，也没法在模块里互相复用（clip 版就得抄一遍）
        from ttrade import userfactors as _UF
        if {f.key for f in _UF.load_all()} & {"open_range_break",
                                              "open_range_break_clip"}:
            raise RuntimeError("波幅突破又变回用户（公式）因子了")
        for _k in ("open_range_break", "open_range_break_clip"):
            _mod = getattr(F.FACTORS[_k].func, "__module__", "")
            if _mod != "ttrade.factors_intraday":
                raise RuntimeError(f"{_k} 不是内置因子（func 来自 {_mod!r}）")

        return (f"默认 orig（与 window 结果不同）；window/strict 只落在 10:05"
                f"（{int(ent.sum())} 笔）且 5m 上逐根等价；夜盘腿 21:30 进场"
                f"（{int(e2.sum())} 笔）；因子按当日开盘，偏差 {dev:.0e}；"
                f"两个因子都是内置 Python 实现")

    check("波幅突破：检查点时刻 + 当日开盘口径", _break_caliber)

    def _session_gate():
        """交易时段门禁：行情（天勤 / CTP / 融航）与策略引擎共用**一条**判据。

        为什么要有这一条：收盘之后 CTP 还会把最后一次快照一直往外吐、天勤的
        模拟账户还会继续往序列里长 bar —— 行情看着在动，策略就跟着触发、下单，
        实盘和「真的能成交」完全脱节。修法不是各处补一句 `9 <= hour <= 15`
        （那正是 #5 那类 bug 的温床），而是所有地方都问
        `symbols_info.in_session()`。

        这里钉住三件事：
        1. 时段判据本身 —— 夜盘跨零点、**周五夜盘跑到周六凌晨**、周日夜盘算
           下周一、午休空档、股指没夜盘、国债到 15:15；
        2. CTP 攒 bar 时会**丢掉**时段外的 tick（含收盘后快照、周末）；
        3. 策略引擎的 `_open_now()` 与 `in_session()` 逐品种一致（单一判据），
           板块级按「任一成员开市」，非期货（A 股）不受这张表管。
        """
        from datetime import datetime as _dt
        import inspect

        from ttrade import ctp_live as CL
        from ttrade import strategy_runner as SR
        from ttrade import symbols_info as SI

        def _t(s):
            d, _, hm = s.partition(" ")
            y, mo, dd = (int(x) for x in d.split("-"))
            h, mi = (int(x) for x in hm.split(":"))
            return _dt(y, mo, dd, h, mi)

        cases = (
            ("SC", "INE", "2026-11-17 14:35", True, "周二日盘"),
            ("SC", "INE", "2026-11-17 15:00", False, "收盘时刻本身判关"),
            ("SC", "INE", "2026-11-17 13:00", False, "午休空档"),
            ("SC", "INE", "2026-11-17 21:30", True, "夜盘"),
            ("SC", "INE", "2026-11-18 01:30", True, "夜盘跨零点（到 02:30）"),
            ("SC", "INE", "2026-11-18 03:00", False, "夜盘已收"),
            ("SC", "INE", "2026-11-21 01:00", True, "周六凌晨 = 周五夜盘延续"),
            ("SC", "INE", "2026-11-21 21:30", False, "周六晚上没有盘"),
            ("SC", "INE", "2026-11-22 01:00", False, "周日凌晨没有盘"),
            ("SC", "INE", "2026-11-22 21:30", True, "周日夜盘算下周一"),
            ("RB", "SHFE", "2026-11-17 23:00", False, "黑色 23:00 收"),
            ("IM", "CFFEX", "2026-11-17 09:05", False, "股指 09:30 开"),
            ("IM", "CFFEX", "2026-11-17 21:30", False, "股指没有夜盘"),
            ("T", "CFFEX", "2026-11-17 15:10", True, "国债交易到 15:15"),
            ("T", "CFFEX", "2026-11-17 15:15", False, "国债 15:15 收"),
            ("T", "CFFEX", "2026-11-17 09:20", False, "国债 09:30 开（2020-07 起）"),
            # 下面这几个是 2026-11-17 用近 100 天真实 5m 逐根验表才查出来的配错：
            # 郑商所 SA/PF/PR/PX/SH **有** 21:00-23:00 夜盘，被错列进「无夜盘」；
            # 上期所 AO/AD 夜盘到 01:00，被当成 23:00。写进自检，防止再抄错一遍。
            ("SA", "CZCE", "2026-11-17 21:30", True, "纯碱有夜盘"),
            ("PX", "CZCE", "2026-11-17 21:30", True, "对二甲苯有夜盘"),
            ("SH", "CZCE", "2026-11-17 21:30", True, "烧碱有夜盘"),
            ("AO", "SHFE", "2026-11-18 00:30", True, "氧化铝夜盘到 01:00"),
            ("AD", "SHFE", "2026-11-18 00:30", True, "铸造铝合金夜盘到 01:00"),
            ("UR", "CZCE", "2026-11-17 21:30", False, "尿素真的没有夜盘"),
            ("AP", "CZCE", "2026-11-17 21:30", False, "苹果没有夜盘"),
        )
        bad = [f"{s} {t}（{why}）期望 {w} 实得 {SI.in_session(s, e, _t(t))}"
               for s, e, t, w, why in cases
               if SI.in_session(s, e, _t(t)) != w]
        if bad:
            raise RuntimeError("交易时段判据错了：" + "；".join(bad[:3]))

        # CTP：时段外的 tick 不能攒成 bar（收盘后快照 / 周末都要挡）
        b = CL._BarBuilder(period="1m", symbol="SC", exchange="INE")
        for t, vol in (("20261117 14:35:00", 10), ("20261117 15:05:00", 11),
                       ("20261117 21:05:00", 12), ("20261121 10:00:00", 13)):
            b.push({"datetime": t, "last": 500.0, "volume": vol})
        kept = [f"{ts:%H:%M}" for ts in b.frame().index]
        if kept != ["14:35", "21:05"]:
            raise RuntimeError(f"CTP 没挡住时段外的 tick：留下 {kept}")

        class _Cfg:
            def __init__(self, sym, ex, ms=None):
                self.symbol, self.exchange = sym, ex
                self.multi_symbols = list(ms or [])

        for sym, ex in (("IM", "CFFEX"), ("SC", "INE"), ("RB", "SHFE")):
            want = SI.in_session(sym, ex)
            got = SR.PortfolioRunner._open_now(None, _Cfg(sym, ex))
            if got != want:
                raise RuntimeError(
                    f"_open_now 与 in_session 不一致（{sym}）：{want} vs {got}")
        # 🔴 「任一成员开市」不能拿**当前时刻**去验：15:00 一过 IM 和 RB 都
        # 收盘，这条断言就必红（2026-11-17 15:03 实测红过）。判据换成确定的
        # 假函数，自检才能全天候跑。
        _is = SI.in_session
        try:
            SI.in_session = lambda s, e, t=None: str(s).startswith("RB")
            if not SR.PortfolioRunner._open_now(
                    None, _Cfg("", "SHFE", ["IM_CFFEX", "RB_SHFE"])):
                raise RuntimeError("板块级策略没有按「任一成员开市」判定")
            SI.in_session = lambda s, e, t=None: False
            if SR.PortfolioRunner._open_now(
                    None, _Cfg("", "SHFE", ["IM_CFFEX", "RB_SHFE"])):
                raise RuntimeError("板块级策略在全部成员闭市时仍然开跑")
        finally:
            SI.in_session = _is
        # A 股在品种信息表里没有时段 ⇒ 一律判「开着」，三个时刻都试一遍
        for _tick in ("2026-11-17 03:00", "2026-11-17 10:00", "2026-11-17 15:30"):
            if not SI.in_session("600519", "SSE", _t(_tick)):
                raise RuntimeError("A 股被期货交易时段表误伤了（%s）" % _tick)

        # 🔴 非交易时段**照常算信号，只暂停下单**。老写法在 `_refresh_signal`
        # 开头就 return，结果面板上的「理论」一直是 0，用户两次误判成
        # 「台账没了所以理论也没了」（2026-11-17 15:55）。
        src_r = inspect.getsource(SR.PortfolioRunner._refresh_signal)
        j = src_r.find("open_now = self._open_now")
        k = src_r.find("key = self._skey(st.cfg)")
        if j < 0 or k < 0 or k < j:
            raise RuntimeError("非交易时段在算信号之前就 return 了，理论仓位会一直是 0")
        if "return" in src_r[j:k]:
            raise RuntimeError("算信号之前还有一条 return：理论仓位会一直是 0")
        src_t = inspect.getsource(SR.PortfolioRunner.tick)
        if "if not st.last_action:" not in src_t:
            raise RuntimeError("tick 的非交易时段分支会盖掉信号那边写的提示")
        return (f"{len(cases)} 个时刻判对（含夜盘跨零点 / 周五夜盘跑到周六凌晨 / "
                f"午休 / 股指无夜盘）；CTP 挡掉时段外 tick（留 {','.join(kept)}）；"
                f"策略门禁与 in_session 逐品种一致，板块级任一开市即跑，A 股不受管")

    check("交易时段门禁（行情与策略共用一条判据）", _session_gate)

    def _sector_maps():
        """板块两层（L1 9 个 / L2 16 个）必须严格对齐。

        这条检查存在的理由：L2 是手写映射，漏一个品种它就会从那层的扫描里
        **静默消失**（照样跑完，只是少算了几个品种，界面上看不出来）；一个品种
        被两个 L2 认领则会在两张表里各算一次。两种错都不会报异常。
        """
        from ttrade import futures as F

        bad = F.sector_maps_ok()
        if bad:
            raise RuntimeError("板块映射不对齐：" + "；".join(bad))
        if len(F.SECTOR_MAP) != 9:
            raise RuntimeError(f"大类数变了（{len(F.SECTOR_MAP)}），产业板块划分要跟着改")
        if len(F.SECTOR_L2) != 16:
            raise RuntimeError(f"产业板块数变了（{len(F.SECTOR_L2)}），文档/提示词要跟着改")
        # 每个大类都要能列出自己的产业板块，且合起来正好是全部 16 个
        got = [s for l1 in F.SECTOR_MAP for s in F.l2_names(l1)]
        if sorted(got) != sorted(F.SECTOR_L2):
            raise RuntimeError("大类展开的产业板块集合和 SECTOR_L2 不一致：" +
                               "、".join(sorted(set(F.SECTOR_L2) - set(got))) or "重复")
        # 品种 → L2 → 父级 必须回到品种的 L1
        for code, l1 in {s: k for k, v in F.SECTOR_MAP.items() for s in v.split()}.items():
            l2 = F.sector_l2_of(code)
            if not l2 or l2 == "其他":
                raise RuntimeError(f"{code} 没有产业板块")
            if F.l1_of_l2(l2) != l1:
                raise RuntimeError(f"{code} 的大类是 {l1}，但产业板块 {l2} 挂在 "
                                   f"{F.l1_of_l2(l2)} 下")
        return (f"{len(F.SECTOR_MAP)} 大类 / {len(F.SECTOR_L2)} 产业板块，"
                f"92 个品种一一对应（化工拆 4、农产品拆 4、金融拆 2）")

    check("板块两层映射（L1 9 / L2 16 严格对齐）", _sector_maps)

    def _factor_method_of():
        """逐因子标准化口径（`FactorConfig.method_of`）。

        「仓单走分位、动量走 z」就是靠这张表表达的（依据见
        `_tools/sector_sweep/s6_norm.py`）。它失效的方式**不报错** ——
        `method_for` 认不出键就静默退回全局 `method`，策略照跑、数字照出，
        只是所有因子都用了同一个口径。所以这里用两条反证：指定 rank 的列
        必须与单独用 rank 算的逐位相等，同时必须与 z 算的**不相等**。
        只查前者会被「全走 z」蒙过去。
        """
        import numpy as np
        import pandas as pd

        from ttrade import factors as F
        from ttrade.factor_strategy import FactorConfig, factor_matrix

        cfg = FactorConfig(window=252, method="z",
                           method_of={"mom_20": "rank"}).normalized()
        if cfg.method_for("mom_20") != "rank":
            raise RuntimeError("指定 rank 的因子没走分位")
        if cfg.method_for("vol_20") != "z":
            raise RuntimeError("没指定的因子没跟全局 z")
        if cfg.method_for("mom_20@1h") != "rank":
            raise RuntimeError("带周期的实例 id 没退回纯键")
        for alias in ("pct", "percentile", "quantile"):
            if FactorConfig(method_of={"x": alias}).method_for("x") != "rank":
                raise RuntimeError(f"别名 {alias} 没被认成分位")
        # 全局 rank 时显式写 z 必须能覆盖（最容易被「只留 rank」写坏的一条）
        c2 = FactorConfig(method="rank", method_of={"vol_20": "z"}).normalized()
        if c2.method_for("vol_20") != "z" or c2.method_for("mom_20") != "rank":
            raise RuntimeError("全局 rank + 逐因子 z 的覆盖失效")
        # 归一化：认不出的词 → z 且**保留键**（全局 rank 时它是有效覆盖），
        # 空串才丢键
        c3 = FactorConfig(method_of={"a": "pct", "b": "zscore", "c": ""}).normalized()
        if c3.method_of != {"a": "rank", "b": "z"}:
            raise RuntimeError(f"归一化结果不对：{c3.method_of}")

        # 真实入口：两个因子混口径不能串味。只用 OHLCV 因子 —— 仓单类要读外部
        # EDB 库，自检环境不一定有数据，拿不到会静默变成整列 NaN 被滤掉
        n = 900
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        rng = np.random.default_rng(7)
        px = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
        df = pd.DataFrame({"open": px, "high": px * 1.005, "low": px * 0.995,
                           "close": px, "volume": 1000.0}, index=idx)
        keys = ["mom_20", "vol_20"]
        fmat, _raw, kept, _gone, _miss = factor_matrix(df, keys, cfg)
        if set(kept) != set(keys):
            raise RuntimeError(f"因子没算全：kept={kept}（合成数据缺列？）")
        for k in keys:
            m = cfg.method_for(k)
            want = F.standardize(F.raw(df, k), window=cfg.window, method=m)
            other = F.standardize(F.raw(df, k), window=cfg.window,
                                  method="z" if m == "rank" else "rank")
            if not np.allclose(fmat[k].values, want.values, equal_nan=True):
                raise RuntimeError(f"{k} 的分数不等于「按自己的口径单独算」—— 串味了")
            if np.allclose(want.values, other.values, equal_nan=True):
                raise RuntimeError(f"{k} 两种口径算出完全相同的结果（口径没生效）")
        return ("逐因子口径生效：mom_20 走分位、vol_20 跟全局 z，两列各自逐位相等且"
                "与另一口径不等；带周期 id 退回纯键；全局 rank + 逐因子 z 可覆盖")

    def _account_timer():
        """账户定时登录 / 登出：四个时刻、跨零点、去重、配置往返。

        判据是「往回看 10 分钟」而不是「今天几点过了没」——夜盘的登出是**次日**
        02:35，按日期判会永远不触发（或者一开机就触发）。这条把四个边角都钉住：
        到点前不动、到点动、过了宽限不动、同一时刻只动一次。
        """
        import inspect

        from datetime import datetime as _dt

        from gui.trade_page import (
            ACC_DEFAULTS,
            TradePage,
            acc_due,
            acc_transitions,
            load_acc_timer,
        )

        if (ACC_DEFAULTS["night"] != ["20:55", "02:35"]
                or ACC_DEFAULTS["day"] != ["08:55", "15:20"]):
            raise RuntimeError(
                f"默认时刻不对：夜盘 {ACC_DEFAULTS['night']}、日盘 {ACC_DEFAULTS['day']}"
                "（用户定的是 20:55/02:35 与 08:55/15:20）")
        cfg = dict(ACC_DEFAULTS)
        cfg["enabled"] = True
        order = [(k, h) for k, h, _lab in acc_transitions(cfg)]
        if order != [("login", "08:55"), ("logout", "15:20"),
                     ("login", "20:55"), ("logout", "02:35")]:
            raise RuntimeError(f"四个时刻的顺序/取值不对：{order}")

        for s, want in (
            ("2026-11-17 20:54", None),
            ("2026-11-17 20:55", ("login", "夜盘")),
            ("2026-11-17 20:58", ("login", "夜盘")),
            ("2026-11-17 21:10", None),          # 超过 10 分钟宽限
            ("2026-11-18 02:35", ("logout", "夜盘")),   # 跨零点
            ("2026-11-18 02:30", None),
            ("2026-11-18 08:55", ("login", "日盘")),
            ("2026-11-18 15:20", ("logout", "日盘")),
            ("2026-11-18 15:31", None),
        ):
            got = acc_due(cfg, _dt.strptime(s, "%Y-%m-%d %H:%M"))
            got = None if got is None else (got[0], got[1])
            if got != want:
                raise RuntimeError(f"{s} 应判 {want}，实际 {got}")
        # 同一时刻只执行一次：心跳 20 秒一轮，不去重会连着发十几个登录
        d = _dt(2026, 11, 18, 2, 35)
        first = acc_due(cfg, d)
        if first is None or acc_due(cfg, d, first[2]) is not None:
            raise RuntimeError("同一时刻没有去重，会重复登录 / 登出")
        if acc_due(dict(ACC_DEFAULTS), _dt(2026, 11, 17, 21, 0)) is not None:
            raise RuntimeError("没启用时仍然触发了动作")
        if not isinstance(load_acc_timer().get("night"), list):
            raise RuntimeError("账户定时配置读回来不是预期结构")

        src = inspect.getsource(TradePage._acc_tick)
        for must in ("connect_requested.emit", "disconnect_requested.emit",
                     "self.live.running"):
            if must not in src:
                raise RuntimeError(f"`_acc_tick` 少了 {must}（不会真连/真断）")
        if "QTimer" not in inspect.getsource(TradePage.__init__):
            raise RuntimeError("TradePage 没起定时器")
        return ("四个时刻判对（含夜盘 02:35 跨零点与 10 分钟宽限），同一时刻只动一次，"
                "未启用不动作，登录走 connect_requested、登出走 disconnect_requested")

    def _per_account_sessions():
        """逐账户登录 / 登出：一个账户一条会话。

        这条判据错了**不会报错**：点了某个账户的登录，另一个户被悄悄掐掉，界面
        上只是那个户的持仓和资金突然不动了。所以逐条钉住：已连着的账户重复
        `start` 不许重开、`stop_one` 不许连带别人、纯行情会话不算账户、合并快照
        得把几条会话的账户都带上。
        """
        import inspect

        from gui.live import MARKET_KEY, LiveController, _session_key
        from gui.trade_page import TradePage
        from ttrade.tqlive import AccountSpec, LiveState

        ctl = LiveController()
        starts: list[str] = []

        class _S:
            def __init__(self, name, spec, sym, per):
                self.name = name
                self.spec = spec
                self.backend = getattr(spec, "backend", "tq")
                self.symbol, self.period = sym, per
                self.state = None
                self.worker = None
                self.thread = None

        def _fake_start(spec, name, sym, ex, per, hist):
            starts.append(name)
            ctl._sessions[name] = _S(name, spec, sym, per)

        ctl._start_one = _fake_start
        A = AccountSpec(name="自检A", kind="kq", user="u", password="p")
        B = AccountSpec(name="自检B", kind="ctp", user="u", password="p")
        ctl.start([A, B], "rb", "SHFE", "5m", need_history=False, feed="tq")
        if starts != ["自检A", "自检B"] or ctl.running_names != {"自检A", "自检B"}:
            raise RuntimeError(f"一次登录两个账户没建两条会话：{starts}")
        # 再点一次「连接」（只勾了 A）：A 不能被掐掉重开
        starts.clear()
        ctl.start([A], "rb", "SHFE", "5m", need_history=False)
        if starts:
            raise RuntimeError(f"已经连着的账户被重开了：{starts}")
        # 换了合约要重订
        ctl.start([A], "m", "DCE", "5m", need_history=False)
        if starts != ["自检A"]:
            raise RuntimeError(f"换了合约却没重订：{starts}")
        # 单点登出：别人还在
        ctl._sessions.pop("自检A")
        if ctl.running_names != {"自检B"}:
            raise RuntimeError(f"登出一个账户把别的也带走了：{ctl.running_names}")
        if ctl.stop_one("根本没这个户"):
            raise RuntimeError("stop_one 对不存在的账户返回了 True")
        # 纯行情会话不算账户
        if _session_key(None) != MARKET_KEY or MARKET_KEY in ctl.running_names:
            raise RuntimeError("纯行情会话被算成了账户")

        # 合并快照：两个会话的账户都要在，权益相加
        def _st(tag, bal):
            return LiveState(
                connected=True, quote={"last": bal},
                accounts=[{"name": tag, "account": {"balance": bal}}],
                by_account={tag: {"spec": {"name": tag},
                                  "account": {"balance": bal},
                                  "positions": [], "orders": [], "trades": []}},
                account={"balance": bal})

        ctl._sessions.clear()
        for tag, bal in (("自检A", 100.0), ("自检B", 50.0)):
            sp = A if tag == "自检A" else B
            s = _S(tag, sp, "rb", "5m")
            s.state = _st(tag, bal)
            ctl._sessions[tag] = s
        m = ctl._merged()
        if set(m.by_account) != {"自检A", "自检B"}:
            raise RuntimeError(f"合并快照里少了账户：{sorted(m.by_account)}")
        if abs(float(m.account.get("balance") or 0) - 150.0) > 1e-6:
            raise RuntimeError(f"合并权益不对：{m.account.get('balance')}")

        # 界面侧：每行一个按钮、点它只连这一个户；顶部按钮按勾选的连没连判方向
        src_tp = inspect.getsource(TradePage)
        for must in ("setCellWidget(i, 1, btn)", "disconnect_one.emit",
                     "def _edit_account", "def _toggle_account",
                     "def _sync_conn_buttons"):
            if must not in src_tp:
                raise RuntimeError(f"交易页少了 {must}")
        if "✎" not in inspect.getsource(TradePage._wrap):
            raise RuntimeError("账户表的「编辑」按钮不见了")
        if "[spec]" not in inspect.getsource(TradePage._toggle_account):
            raise RuntimeError("单点登录没有只发一个 spec（会连带别的户一起登）")
        if "is_running" not in inspect.getsource(TradePage._toggle_connect):
            raise RuntimeError("顶部按钮还在用「有没有一条会话」判方向，"
                               "单独登一个户就会显示成「断开」")
        from gui.app import MainWindow

        if "stop_one" not in inspect.getsource(MainWindow._on_live_stop_one):
            raise RuntimeError("主窗口没有把单户登出接到 stop_one")
        return ("已连着的账户重复 start 不重开（换合约才重订）、单户登出不带走别人、"
                "纯行情会话不算账户、合并快照两户权益 100+50=150；"
                "账户表每行一个按钮、单点只连这一个户、顶部按勾选判方向")

    def _contract_input():
        """手输合约 → 天勤写法。下单面板只收裸合约代码（`rb2601`）。

        这条错了的后果是**下到一个不存在的合约**（柜台拒单，还能看出来）或者
        更糟：下到一个同名的、另一个交易所的合约上。所以两种写法都得认，猜不到
        交易所时必须返回空串而不是硬猜一个。
        """
        import inspect

        from gui.trade_page import TradePage
        from ttrade.ctp_live import ctp_instrument
        from ttrade.tqsdk_symbols import contract_from_input, to_ts

        for args, want in (
            (("rb2601", "SHFE"), ("SHFE.rb2601", "SHFE")),
            (("SHFE.rb2601", ""), ("SHFE.rb2601", "SHFE")),
            (("shfe.rb2601", ""), ("SHFE.rb2601", "SHFE")),
            (("RB2601.SHF", ""), ("SHFE.rb2601", "SHFE")),
            (("CU2612.SHF", ""), ("SHFE.cu2612", "SHFE")),
            (("TA2706.ZCE", ""), ("CZCE.TA706", "CZCE")),
            # 光秃秃的合约 + 没有交易所：不猜，返回空让界面报「认不出」
            (("rb2601", ""), ("", "")),
            (("TA2706", ""), ("", "")),
            (("", ""), ("", "")),
            (("垃圾", "SHFE"), ("", "")),
        ):
            got = contract_from_input(*args)
            if got != want:
                raise RuntimeError(f"contract_from_input{args}：应 {want}，实际 {got}")
        # 天勤写法要能直接喂给 CTP（柜台那边自己剥前缀），否则「同一条下单链路
        # 服务两种柜台」就不成立
        if ctp_instrument("SHFE.rb2601") != "rb2601":
            raise RuntimeError("天勤写法喂给 CTP 没有收敛成 InstrumentID")
        if to_ts("SHFE.rb2601", "2026-11-17") != "RB2601.SHF":
            raise RuntimeError("天勤写法转不回 Tushare 合约（本地取价要用它）")
        # 界面侧：合约框要走解析，不能把裸代码原样扔给柜台
        src = inspect.getsource(TradePage._place)
        if "_resolve_contract" not in src:
            raise RuntimeError("下单没有解析合约，裸代码会原样发到柜台")
        if "quote_of" not in inspect.getsource(TradePage._fill_price):
            raise RuntimeError("选中合约后没有去取最新价")
        if "setValue" not in inspect.getsource(TradePage._on_quote):
            raise RuntimeError("取价回执没有填进价格框")
        return ("四种写法都归一：rb2601+SHFE / SHFE.rb2601 / shfe.rb2601 / "
                "RB2601.SHF；郑商所 4 位收成 3 位（TA2706→TA706）；"
                "没交易所时不猜（返回空）；归一后天勤能下单、CTP 也能用")

    def _auto_two_slots():
        """「一条日程两个时刻」的判据（宏观 08:45 / 20:45，以及 EDB 串行链）。

        这条判据错了不会有任何报错：少跑一场只是数据旧半天。所以逐条钉住：
        未到点不动、到点跑、同一天第二场照跑、晚开机只补一场、老格式不重复补。
        """
        from gui.data_page import due_slot

        T = ["08:45", "20:45"]
        for now, last, want in (
            ("08:44", "", None),
            ("08:45", "", "2026-11-17@08:45"),
            ("12:00", "2026-11-17@08:45", None),
            ("20:45", "2026-11-17@08:45", "2026-11-17@20:45"),
            ("23:30", "2026-11-17@08:45", "2026-11-17@20:45"),
            ("23:30", "2026-11-17@20:45", None),
            # 晚开机：两场都过了，只补最晚那一场，不连跑两遍
            ("23:30", "", "2026-11-17@20:45"),
            # 老格式（只有日期）：升级当天当整天都跑过，别重复补
            ("23:30", "2026-11-17", None),
            # 单时刻的日程（aux / daily / tq）行为不变
            ("14:00", "", None),
            ("15:30", "", "2026-11-17@15:30"),
        ):
            times = T if now not in ("14:00", "15:30") else ["15:30"]
            got = due_slot(times, now, last, "2026-11-17")
            if got != want:
                raise RuntimeError(f"{times} {now} last={last!r}：应判 {want}，实际 {got}")
        return ("未到点不动 / 到点跑 / 同一天第二场照跑 / 晚开机只补最晚一场 / "
                "老日期格式不重复补；单时刻日程行为不变")

    check("逐因子标准化口径（method_of：仓单分位 / 动量 z 可混用）", _factor_method_of)

    check("每日定时：一条日程两个时刻（08:45 / 20:45）", _auto_two_slots)

    check("账户定时登录 / 登出（夜盘 20:55→02:35、日盘 08:55→15:20）", _account_timer)

    check("逐账户登录 / 登出（一条会话一个账户）", _per_account_sessions)

    def _tl0_naming():
        """品种键 `品种_交易所` → CTP 合约。30 年期国债踩过的那个坑。

        `TL0_CFFEX` 是策略配置里的品种键（下划线分隔）。以前只认点号写法，
        整串被当成合约名：字母串拼成 `TLCFFEX`、数字只剩那个 `0`，最后发出
        `tlcffex0`；柜台认不出交易所，回 `[148] 无效的 ExchangeID 字段` ——
        而这条错**完全不提合约名**，只能靠人肉反查（2026-11-19）。

        顺便钉住正名：数据源把品种代码写成 `TL0`（`0` 是连续合约标记），
        本地文件、库、策略配置一律用 `TL`。
        """
        import json
        import os

        from ttrade.broker_api import _split
        from ttrade.ctp_live import (ctp_instrument, exchange_of_contract,
                                     looks_contract, main_contract,
                                     variety_exchange)
        from ttrade.paths import data_dir

        bad = []
        for src, want_ins in (("TL0_CFFEX", "TL"), ("TL_CFFEX", "TL"),
                              ("CFFEX.TL2612", "TL2612"),
                              ("TL2612.CFFEX", "TL2612"),
                              ("CFFEX.TL0", "TL"), ("TL0.CFX", "TL"),
                              ("TL0", "TL"), ("TL", "TL")):
            ex, code = _split(src)
            ins = ctp_instrument(code, ex)
            if ins != want_ins:
                bad.append(f"{src} → {ins!r}（应 {want_ins!r}）")
            if "cffex" in ins.lower():
                bad.append(f"{src} → {ins!r} 里混进了交易所名")
        if bad:
            raise RuntimeError("品种键归一错：" + "；".join(bad))
        if exchange_of_contract("TL0_CFFEX") != "CFFEX":
            raise RuntimeError("TL0_CFFEX 的交易所没解析出来，报单会被柜台回 [148]")
        if variety_exchange("TL") != "CFFEX":
            raise RuntimeError("TL 查不到交易所（data_futures 里的文件正名了吗）")
        if looks_contract("TL0") or looks_contract("TL"):
            raise RuntimeError("TL0 / TL 被当成了具体合约")
        if not looks_contract("TL2612"):
            raise RuntimeError("TL2612 没被当成具体合约")
        if main_contract("TL", "CFFEX") != "TL2612":
            raise RuntimeError("TL 的主力合约查不到")

        # 数据文件要真的是正名后的（库里/配置里的迁移由 _tools/rename_tl0.py 做）
        d = data_dir("cn")
        if os.path.exists(os.path.join(d, "TL0_CFFEX.csv")):
            raise RuntimeError("data_futures 里还有 TL0_CFFEX.csv，没正名")
        if not os.path.exists(os.path.join(d, "TL_CFFEX.csv")):
            raise RuntimeError("data_futures 里没有 TL_CFFEX.csv")
        mp = os.path.join(d, "_manifest.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as fh:
                for it in json.load(fh):
                    if str(it.get("ts_code", "")).startswith("TL0"):
                        if it.get("symbol") != "TL" or it.get("file") != "TL_CFFEX.csv":
                            raise RuntimeError(f"manifest 里 TL 条目没正名：{it}")
        return ("品种键四种写法都归一（TL0_CFFEX / TL_CFFEX / CFFEX.TL2612 / "
                "TL2612.CFFEX），交易所解析到 CFFEX、主力 TL2612；"
                "TL0 不再被当合约；数据文件与 manifest 已正名 TL_CFFEX")

    check("合约手输 → 天勤写法（不带交易所前缀）", _contract_input)
    check("品种键 TL0_CFFEX → CTP 合约（[148] 无效 ExchangeID 回归）", _tl0_naming)

    check("存入策略库的动作（strategy_save）", _strategy_save_action)
    check("K线与信号：持仓与交易记录同源（含手数）", _chart_from_trades)
    check("资金曲线纵轴（线性 / 对数 / 亏穿退回）", _perf_yscale)

    def _trade_pnl_quote():
        """交易页四处：持仓「当日盈亏」、成交「平仓盈亏」、CTP 委托时间、「报价」框。

        这四条**都不会报错，只会显示一个看着没问题的错东西** —— 所以逐条钉住：

        * CTP 的委托「时间」列整列显示的是拒单理由（`_orders` 里把 `msg` 当
          时间用，而 `ctp_client` 当年根本没读 `InsertTime`）；
        * 平仓盈亏如果拿不到开仓均价就写 0，会被读成「这一笔刚好打平」；
        * 「资金账户」列吃 Stretch 会把整列拉到比表还宽，该宽的合约/说明只剩 90px；
        * 报价框双击要真把合约和价格写进下单行，否则「双击没反应」只能靠人眼发现。
        """
        import inspect

        from PySide6.QtWidgets import QApplication, QHeaderView

        from gui.live import LiveController
        from gui.trade_page import TradePage
        from ttrade import ctp_client
        from ttrade.ctp_live import CtpHub, _ctp_stamp, _ctp_ts
        from ttrade.tqlive import (LiveState, est_close_pnl, merge_states,
                                   open_avg_of, update_open_avg)

        # ---- 1. 成交的平仓盈亏：多空各用自己那侧的均价，开仓单 0，拿不到均价 None
        cache: dict = {}
        update_open_avg(cache, [{"symbol": "SHFE.rb2601",
                                 "open_price_long": 3500.0,
                                 "open_price_short": 3600.0}], now=1000.0)
        if open_avg_of(cache, "SHFE.rb2601", now=1000.0) != (3500.0, 3600.0):
            raise RuntimeError("开仓均价没按多空分开记")
        trades = [
            {"symbol": "SHFE.rb2601", "direction": "SELL", "offset": "CLOSE",
             "price": 3550.0, "volume": 2},
            {"symbol": "SHFE.rb2601", "direction": "BUY", "offset": "CLOSETODAY",
             "price": 3500.0, "volume": 1},
            {"symbol": "SHFE.rb2601", "direction": "BUY", "offset": "OPEN",
             "price": 3500.0, "volume": 1},
            {"symbol": "SHFE.rb9999", "direction": "SELL", "offset": "CLOSE",
             "price": 3550.0, "volume": 1},
        ]
        est_close_pnl(trades, cache, lambda _s: 10.0, now=1000.0)
        want = [1000.0, 1000.0, 0.0, None]
        got = [t["close_profit"] for t in trades]
        if got != want:
            raise RuntimeError(f"平仓盈亏估错：应 {want}，实际 {got}"
                               "（卖出平多/买入平空各用自己那侧均价；开仓 0；"
                               "没均价必须是 None 不是 0）")
        if any(open_avg_of(cache, "SHFE.rb2601", now=1000.0 + 7 * 3600)):
            raise RuntimeError("开仓均价过了保鲜期还在用（隔夜会凭空造盈亏）")

        # ---- 2. CTP 委托时间：MM-DD HH:MM:SS，**不许是委托说明**
        if _ctp_stamp("20261118", "14:35:02") != "11-18 14:35:02":
            raise RuntimeError(f"CTP 委托时间拼错：{_ctp_stamp('20261118', '14:35:02')!r}")
        if _ctp_stamp("", "14:35:02") != "14:35:02":
            raise RuntimeError("只有 InsertTime 时不该丢掉时间")
        if _ctp_ts("20261118", "14:35:02") <= _ctp_ts("20261118", "09:01:00"):
            raise RuntimeError("排序键没按时间递增")
        hub = CtpHub([])
        rows = hub._orders({
            "ref1": {"order_sys_id": "sys1", "order_ref": "ref1",
                     "instrument": "rb2601", "direction": "0", "offset": "0",
                     "price": 3500.0, "volume": 2, "traded": 0,
                     "status": "未成交", "msg": "已提交",
                     "insert_time": "14:35:02", "trading_day": "20261118"},
            "ref2": {"order_sys_id": "sys2", "order_ref": "ref2",
                     "instrument": "rb2601", "direction": "1", "offset": "1",
                     "price": 3400.0, "volume": 1, "traded": 1,
                     "status": "已撤单",
                     "msg": "50:已撤单报单被拒绝SHFE:价格跌破跌停板",
                     "insert_time": "09:01:00", "trading_day": "20261118"},
        })
        if not rows or rows[0]["status"] != "ALIVE":
            raise RuntimeError("未成交的委托没有排在最前面（撤单要点它们）")
        if rows[0]["time"] != "11-18 14:35:02":
            raise RuntimeError(f"未成交单的时间列是 {rows[0]['time']!r}，不是时间")
        if "拒绝" in "".join(r["time"] for r in rows):
            raise RuntimeError("委托「时间」列又显示成委托说明了")
        if rows[0]["last_msg"] != "已提交":
            raise RuntimeError("委托说明被挪没了（拒单理由要靠它看）")
        spi_src = inspect.getsource(ctp_client)
        if "InsertTime" not in spi_src or "TradingDay" not in spi_src:
            raise RuntimeError("ctp_client 没把 InsertTime / TradingDay 落下来")

        # ---- 3. 报价：只取行情源那一路
        st = merge_states(
            [("tq", LiveState(quotes=[{"symbol": "SHFE.rb2601", "last": 1.0}])),
             ("ctp", LiveState(quotes=[{"symbol": "DCE.m2601", "last": 2.0}]))],
            "ctp")
        if [q["symbol"] for q in st.quotes] != ["DCE.m2601"]:
            raise RuntimeError("报价没按行情源取一路（两个柜台的价会来回跳）")

        # ---- 4. 界面：列、颜色、列宽、双击
        app = QApplication.instance() or QApplication([])  # noqa: F841
        page = TradePage(LiveController())
        try:
            pos_hdr = [page.tbl_pos.horizontalHeaderItem(i).text()
                       for i in range(page.tbl_pos.columnCount())]
            if "当日盈亏" not in pos_hdr:
                raise RuntimeError(f"持仓表没有「当日盈亏」列：{pos_hdr}")
            page._fill_positions([{
                "account": "甲", "symbol": "SHFE.rb2601", "pos": 2,
                "pos_long": 2, "pos_short": 0,
                "pos_long_today": 2, "pos_short_today": 0,
                "open_price_long": 3500.0, "open_price_short": 0.0,
                "float_profit": 120.0, "position_profit": -80.0,
                "margin": 7000.0,
            }])
            j = pos_hdr.index("当日盈亏")
            if page.tbl_pos.item(0, j).text() != "-80.00":
                raise RuntimeError(
                    f"当日盈亏填成了 {page.tbl_pos.item(0, j).text()!r}"
                    "（要柜台给的 position_profit，不是浮盈）")

            trd_hdr = [page.tbl_trades.horizontalHeaderItem(i).text()
                       for i in range(page.tbl_trades.columnCount())]
            if "平仓盈亏" not in trd_hdr:
                raise RuntimeError(f"成交表没有「平仓盈亏」列：{trd_hdr}")
            page._fill_trades([
                {"account": "甲", "time": "11-18 14:00:00",
                 "symbol": "SHFE.rb2601", "direction": "SELL", "offset": "CLOSE",
                 "price": 3550.0, "volume": 2, "close_profit": 250.0},
                {"account": "甲", "time": "11-18 14:01:00",
                 "symbol": "SHFE.rb2601", "direction": "BUY", "offset": "CLOSE",
                 "price": 3550.0, "volume": 1, "close_profit": None},
            ])
            k = trd_hdr.index("平仓盈亏")
            if page.tbl_trades.item(0, k).text() != "+250.00":
                raise RuntimeError(f"平仓盈亏显示成 {page.tbl_trades.item(0, k).text()!r}")
            if page.tbl_trades.item(1, k).text() != "—":
                raise RuntimeError("拿不到均价的平仓盈亏要显示「—」，不是 0")

            # ---- 4b 手续费列：柜台给的直接显示、估算的带 ≈、算不出来的画「—」
            from ttrade import costs as _costs
            if "手续费" not in trd_hdr:
                raise RuntimeError(f"成交表没有「手续费」列：{trd_hdr}")
            # 逐笔口径（costs.commission）：按手 / 按成交额两种都要对。
            #
            # 🔴 断言必须**与用户覆盖表无关**。品种费率有两层：`defaults()` 的
            # 出厂值 + `symbols_info.json` 的用户覆盖（本机 74 条，豆粕被改成
            # 1.5 元/手）。原先这里直接写死 3.0 / 0.71 —— 本机对得上是因为覆盖表
            # 正好那样填，CI 上没有覆盖表，豆粕走 DCE 的默认「万分之一」算出 6.4，
            # 这条自检当场红（v3.8.2 第一次 CI 就是栽在这里，三个平台一起）。
            # 所以先把覆盖表清空，拿**出厂值**当基准：
            #     郑商所默认按手 3 元/张（TA 乘数 5）；上期所默认按成交额万分之一。
            from ttrade import symbols_info as _SI
            _ov = _SI.load_overrides
            try:
                _SI.load_overrides = lambda: {}
                lot_fee = _costs.commission("TA", "CZCE", 5900.0, 2)
                if lot_fee is None or abs(lot_fee - 6.0) > 1e-6:
                    raise RuntimeError(
                        f"按手收的手续费不对：{lot_fee}（3 元/手 × 2 手 = 6.0）")
                rb_fee = _costs.commission("RB", "SHFE", 3550.0, 2)
                if rb_fee is None or abs(rb_fee - 7.1) > 1e-3:
                    raise RuntimeError(
                        f"按成交额收的手续费不对：{rb_fee}"
                        "（0.0001 × 3550 × 2 手 × 乘数 10 = 7.1）")
                # `rb2601.SHF` 这种写法必须折成 SHFE，否则乘数退化成 1.0
                if abs(_costs.commission("RB", "SHF", 3550.0, 2) - rb_fee) > 1e-6:
                    raise RuntimeError(
                        "交易所缩写 SHF 没折成 SHFE —— 乘数会退化成 1.0，手续费差 10 倍")
                if _costs.commission("RB", "SHFE", 0.0, 2) is not None:
                    raise RuntimeError(
                        "没有价格时手续费要返回 None，不能返回 0（界面要画「—」）")
                if _costs.commission("ZZZZ", "", 3550.0, 2) is not None:
                    raise RuntimeError(
                        "品种表里没有的合约要返回 None —— specs 对未知品种会悄悄"
                        "退回乘数 1.0，按它算出来的手续费差一个数量级")
            finally:
                _SI.load_overrides = _ov

            fee_j = trd_hdr.index("手续费")
            page._fill_trades([
                {"account": "甲", "time": "11-18 14:00:00",
                 "symbol": "SHFE.rb2601", "direction": "SELL", "offset": "CLOSE",
                 "price": 3550.0, "volume": 2, "close_profit": 250.0,
                 "commission": 5.0, "commission_est": False},
                {"account": "甲", "time": "11-18 14:01:00",
                 "symbol": "DCE.m2601", "direction": "BUY", "offset": "CLOSE",
                 "price": 3200.0, "volume": 1, "close_profit": 120.0,
                 "commission": 3.5, "commission_est": True},
                {"account": "甲", "time": "11-18 14:02:00",
                 "symbol": "DCE.m2601", "direction": "BUY", "offset": "OPEN",
                 "price": 3210.0, "volume": 1, "close_profit": 0.0,
                 "commission": None, "commission_est": False},
            ])
            for i, want_fee in enumerate(("5.00", "≈3.50", "—")):
                got = page.tbl_trades.item(i, fee_j).text()
                if got != want_fee:
                    raise RuntimeError(
                        f"成交表第 {i} 行手续费是 {got!r}，应 {want_fee!r}"
                        "（柜台给的直接写；估算的带 ≈；算不出来的画「—」不是 0.00）")

            # ---- 4c 「按品种统计」：勾上翻到汇总表，按品种聚合，末行合计
            sum_hdr = [page.tbl_trade_sum.horizontalHeaderItem(i).text()
                       for i in range(page.tbl_trade_sum.columnCount())]
            if sum_hdr != ["品种", "笔数", "手数", "手续费", "平仓盈亏"]:
                raise RuntimeError(f"汇总表列不对：{sum_hdr}")
            if page.stack_trades.currentIndex() != 0:
                raise RuntimeError("默认应该看逐笔成交（明细页）")
            page.chk_trade_sum.setChecked(True)
            if page.stack_trades.currentIndex() != 1:
                raise RuntimeError("勾上「按品种统计」没切到汇总表")
            st_sum = page.tbl_trade_sum
            # 品种 笔数 手数 手续费 平仓盈亏
            want = [
                ("M", "2", "2", "≈3.50*", "+120.00"),
                ("RB", "1", "2", "5.00", "+250.00"),
                ("合计", "3", "4", "≈8.50*", "+370.00"),
            ]
            for i, row in enumerate(want):
                got = tuple(st_sum.item(i, j).text() for j in range(5))
                if got != row:
                    raise RuntimeError(f"汇总表第 {i} 行是 {got}，应 {row}")
            if not st_sum.item(2, 0).font().bold():
                raise RuntimeError("汇总表最后一行「合计」没有加粗")
            page.chk_trade_sum.setChecked(False)
            if page.stack_trades.currentIndex() != 0:
                raise RuntimeError("取消「按品种统计」没切回明细表")
            page._fill_trades([])
            if page.tbl_trade_sum.rowCount() != 0:
                raise RuntimeError("没有成交时汇总表不该留着一行空的「合计」")

            # 「资金账户」列不许吃 Stretch；该宽的那列才 Stretch
            for tbl, wide in ((page.tbl_pos, 1), (page.tbl_orders, 9),
                              (page.tbl_trades, 2)):
                h = tbl.horizontalHeader()
                if h.sectionResizeMode(0) == QHeaderView.Stretch:
                    raise RuntimeError("「资金账户」列又在吃剩余宽度（整列会被拉爆）")
                if h.sectionResizeMode(wide) != QHeaderView.Stretch:
                    raise RuntimeError(f"第 {wide} 列应该是吃剩余宽度的那一列")

            # 报价框：列、涨跌、双击带入下单行
            q_hdr = [page.tbl_quotes.horizontalHeaderItem(i).text()
                     for i in range(page.tbl_quotes.columnCount())]
            if "最新" not in q_hdr or "买价" not in q_hdr:
                raise RuntimeError(f"报价框列不对：{q_hdr}")
            page._fill_quotes([{
                "symbol": "SHFE.rb2601", "instrument": "rb2601",
                "last": 3550.0, "bid": 3549.0, "ask": 3551.0,
                "pre_close": 3500.0, "datetime": "2026-11-18 14:35:02",
            }])
            if page.tbl_quotes.item(0, 0).text() != "rb2601":
                raise RuntimeError("报价框合约显示的不是裸代码")
            # 最新价 / 涨跌的小数位跟着合约的最小变动价位走：RB 一跳 1 ⇒ 3550，
            # 不是 3550.00（用户 2026-11-20 要求，和品种价格列同一套）
            if page.tbl_quotes.item(0, 1).text() != "3,550":
                raise RuntimeError(
                    f"报价框最新价没填对：{page.tbl_quotes.item(0, 1).text()!r}")
            if page.tbl_quotes.item(0, 2).text() != "+50":
                raise RuntimeError(
                    f"涨跌算错：{page.tbl_quotes.item(0, 2).text()!r}（要 3550−3500）")
            # 天勤的 datetime 带微秒，`[-8:]` 会截出 `2.000000`
            page._fill_quotes([{
                "symbol": "SHFE.rb2601", "instrument": "rb2601", "last": 3550.0,
                "bid": 3549.0, "ask": 3551.0, "pre_close": 3500.0,
                "datetime": "2026-11-18 14:35:02.000000",
            }])
            if page.tbl_quotes.item(0, 5).text() != "14:35:02":
                raise RuntimeError(
                    f"行情时间截错：{page.tbl_quotes.item(0, 5).text()!r}"
                    "（天勤带微秒，不能直接 [-8:]）")
            page._fill_quotes([{
                "symbol": "SHFE.rb2601", "instrument": "rb2601",
                "last": 3550.0, "bid": 3549.0, "ask": 3551.0,
                "pre_close": 3500.0, "datetime": "2026-11-18 14:35:02",
            }])
            page._on_quote_dbl(page.tbl_quotes.item(0, 0))
            if page.cb_contract.currentText() != "rb2601":
                raise RuntimeError("双击报价行没把合约带进下单行")
            if abs(page.sp_price.value() - 3550.0) > 1e-6:
                raise RuntimeError(
                    f"双击报价行没把价格带进下单行：{page.sp_price.value()}")

            # 盯哪些合约：持仓 + 下单框 + 手动加的，去重并归一成天勤写法
            page.cb_contract.setCurrentText("rb2601")
            page._watch_extra = ["SHFE.rb2601"]
            got = page._watch_targets([{"symbol": "SHFE.rb2601"}])
            if got != ["SHFE.rb2601"]:
                raise RuntimeError(f"报价框合约没去重 / 没归一：{got}")
        finally:
            page.deleteLater()
        return ("持仓当日盈亏走 position_profit（-80.00）、成交平仓盈亏按多空各侧均价"
                "（+250.00 / 拿不到均价画「—」）、CTP 委托时间=11-18 14:35:02 而不是"
                "委托说明、三表资金账户列不再吃 Stretch、报价框双击带入 "
                "rb2601@3550 且合约去重归一；手续费逐笔显示（柜台 5.00 / 估算≈3.50 / "
                "算不出来「—」）、按品种汇总 M 2 笔 2 手 ≈3.50*、RB 1 笔 2 手 5.00、"
                "合计 3 笔 4 手 ≈8.50*/+370.00 且加粗，开关翻页、空成交不留空合计行")

    check("组合页工作区工具条：工作区一行、资金分配一行", _ws_bar_two_rows)

    def _strat_pnl_three_cols():
        """策略列表的「品种价格 / 当日盈亏 / 持仓盈亏」三列。

        🔴 口径（2026-11-20 用户两次明确）：「持仓盈亏和当日盈亏都是策略理论上
        的，和实际账户交易无关」「不需要和实际交易和台账的情况拿来算」。

        所以这两个数只能由 ①理论仓 `target`（signal × lots）②理论建仓价
        `theo_px`（信号序列倒推）③最新价 决定。台账 `held` / 均价 `cost` /
        已落袋 `real_day` **一律不参与** —— 这条自检就是钉住这一点：故意把台账
        写成另一套完全不同的数（持仓、均价、已落袋都给个夸张值），两列必须
        **一个子儿都不动**。

        还要钉住用户报的那个 bug：**当天开的仓，当日盈亏必须等于持仓盈亏**。
        以前当日盈亏拿合约的昨结算当成本，IC 11-20 实测 +55,400 vs 持仓 +8,080。

        另外两件事一起钉：均价按成交逐笔累计（同向加权 / 减仓不动 / 反手重设，
        那是**下单**用的账本，仍在维护）、表头三列在「状态」右边。
        """
        from ttrade import strategy_runner as R
        from ttrade import strategy_store as SStore
        from ttrade.tqlive import AccountSpec

        spec = AccountSpec(name="__selftest_pnl__", kind="sim")
        run = R.PortfolioRunner(spec)
        run.add(SStore.StrategyConfig(
            sid="__pnl_a", name="自检盈亏", symbol="RB", exchange="SHFE",
            period="5m", family="factor", factor_mode="single",
            factor_keys=["rev_5"], account=spec.name))
        st = run._slots["__pnl_a"]
        st.underlying = "SHFE.rb2701"
        st.held.clear()
        st.cost.clear()
        st.real_day = 0.0
        st.real_day_of = ""

        class _Q:
            last_price = 1200.0
            pre_close = 1000.0

        # 不连行情：直接把盘口换成固定值。乘数走真实品种信息表（和下单同源）
        run._contract_quote = lambda sym: _Q()      # noqa: ARG005
        mult = run._volume_multiple(st)
        if mult <= 0:
            raise RuntimeError("RB 的乘数读不到，品种信息表没维护")

        def feed(direction, offset, vol, price):
            o = R.PendingOrder(order_id=f"o{feed.n}", symbol="SHFE.rb2701",
                               direction=direction, offset=offset, volume=vol,
                               price=price, ts=0.0)
            feed.n += 1
            run._book(st, o, vol)
        feed.n = 0

        # 台账照旧按成交记账（下单要用）：开 2 手 @1000、加 3 手 @1100 → 均价 1060
        feed("BUY", "OPEN", 2, 1000.0)
        feed("BUY", "OPEN", 3, 1100.0)
        if st.held.get("SHFE.rb2701") != 5:
            raise RuntimeError(f"账本手数不对：{st.held}")
        if abs(st.cost["SHFE.rb2701"] - 1060.0) > 1e-6:
            raise RuntimeError(f"加仓均价要按手数加权到 1060：{st.cost['SHFE.rb2701']}")
        feed("SELL", "CLOSE", 2, 1200.0)
        if abs(st.cost["SHFE.rb2701"] - 1060.0) > 1e-6:
            raise RuntimeError("减仓不该改均价（卖剩下的还是原来那批）")
        real = (1200.0 - 1060.0) * 2 * mult
        if abs(st.real_day - real) > 1e-6:
            raise RuntimeError(f"当日已实现盈亏算错：{st.real_day} ≠ {real}")

        # ---- 理论口径：只认 target / theo_px / 最新价 ----
        from ttrade import daylimit as _dl

        st.target = 3                       # 理论 3 手多
        st.theo_px = 1180.0
        st.theo_day = _dl.trading_day()     # 今天建的 —— 两个数必须相等
        st.theo_sig = 1
        px, day, pos, _note = run._slot_pnl(st)
        if abs(px - 1200.0) > 1e-9:
            raise RuntimeError(f"品种价格要取最新价：{px}")
        pos_want = (1200.0 - 1180.0) * 3 * mult
        if abs(pos - pos_want) > 1e-6:
            raise RuntimeError(f"持仓盈亏要按理论建仓价算：{pos} ≠ {pos_want}")
        if abs(day - pos_want) > 1e-6:
            raise RuntimeError(
                f"当天开的理论上仓，当日盈亏必须等于持仓盈亏：{day} ≠ {pos_want}")
        # 🔴 台账那三个数随便怎么歪，两列都不许动一格
        st.cost["SHFE.rb2701"] = 500.0
        st.real_day = 98765.0
        _px2, day2, pos2, _n2 = run._slot_pnl(st)
        if (abs(day2 - day) > 1e-9) or (abs(pos2 - pos) > 1e-9):
            raise RuntimeError(
                f"改了台账（均价 500、已落袋 98765）两列就变了 —— "
                f"又去读台账了：day {day2} pos {pos2}")
        # 隔夜仓：当日基准换成昨结算（1000），持仓仍按建仓价 1180
        st.theo_day = "2000-01-03"
        _px3, day3, pos3, _n3 = run._slot_pnl(st)
        if abs(pos3 - pos_want) > 1e-6:
            raise RuntimeError(f"隔夜仓的持仓盈亏仍按建仓价：{pos3} ≠ {pos_want}")
        day3_want = (1200.0 - 1000.0) * 3 * mult
        if abs(day3 - day3_want) > 1e-6:
            raise RuntimeError(f"隔夜仓的当日盈亏要按昨结算：{day3} ≠ {day3_want}")
        # 理论空仓：两列写 0（不是「—」，也不是拿台账顶上）
        st.target = 0
        _px4, day4, pos4, _n4 = run._slot_pnl(st)
        if pos4 != 0.0 or day4 != 0.0:
            raise RuntimeError(f"理论空仓时两列要写 0：day {day4} pos {pos4}")
        # 反手：均价必须重设成新方向的价（这是下单用的账本，仍在维护）
        st.target = 3
        feed("SELL", "OPEN", 4, 900.0)
        if st.held.get("SHFE.rb2701") != -1:
            raise RuntimeError(f"反手后要剩 -1 手：{st.held}")
        if abs(st.cost["SHFE.rb2701"] - 900.0) > 1e-6:
            raise RuntimeError(
                f"反手后均价要重设成 900（不能沿用旧仓的 1060）："
                f"{st.cost['SHFE.rb2701']}")

        # ---- 理论建仓价怎么倒推：信号最后一波翻向的下一根 ----
        import pandas as _pd
        # 每边成本必须是正数（成本档位 / 品种信息表读到了）。量完先把它抹成 0，
        # 后面几段只测「取哪根 bar / 复权折算」，免得数字里混着成本看不清。
        # 🔴 必须在**和断言同一个价格**上量：手续费那一路是「按成交额」收的
        # （`fee = fee_val × px`），在 100 上量、在 10 上比，会差 10 倍
        # ——2026-11-20 就是这么红过一次（11.0001 vs 11.001）。
        _c = run._side_cost_px(st, 10.0)
        if not (_c > 0):
            raise RuntimeError(
                "每边成本算出来是 0（成本档位或品种信息表没读到）——"
                "「开仓价 = 开盘价 + 成本」就退化成开盘价了")
        run._side_cost_px = lambda _st, _px: 0.0
        idx = _pd.date_range("2026-11-16", periods=6, freq="D")
        frame = _pd.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
                              index=idx)
        sig = _pd.Series([0, 0, 1, 1, 1, 1], index=idx)
        st2 = run._slots["__pnl_a"]
        st2.theo_px, st2.theo_day, st2.theo_sig, st2.theo_src = 0.0, "", 0, ""
        run._note_theo_entry(st2, frame, sig)
        # 信号在 11-18 那根翻成 1 → 11-19 成交 → 建仓价取 11-19 的 close=13
        if abs(st2.theo_px - 13.0) > 1e-9:
            raise RuntimeError(
                f"理论建仓价要取「翻向那根的下一根」收盘：{st2.theo_px}（应 13）")
        if st2.theo_day != "2026-11-19":
            raise RuntimeError(f"理论建仓交易日算错：{st2.theo_day}（应 2026-11-19）")
        # 幂等：同一帧再算一次结果必须一样（**不是**「跳过重算」—— 那个缓存会
        # 把降级帧算出的错值锁死，见 2026-11-20 16:49 那次「没有本地历史」）
        run._note_theo_entry(st2, frame, sig)
        if abs(st2.theo_px - 13.0) > 1e-9 or st2.theo_day != "2026-11-19":
            raise RuntimeError(
                f"重算后建仓价/日变了（应当幂等）：{st2.theo_px} @ {st2.theo_day}")
        # 🔴 降级帧算出的错值必须能**自愈**：换一段「一直是多」的信号，建仓价要
        # 跟着挪到最早那根之后 —— 服务过 16:49 那种「没有本地历史」的帧以后，
        # 库内前复权回来了就得纠正过来，不能一直挂着错价。
        frame2 = _pd.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
                               index=idx)
        run._note_theo_entry(st2, frame2,
                             _pd.Series([1, 1, 1, 1, 1, 1], index=idx))
        if abs(st2.theo_px - 11.0) > 1e-9 or st2.theo_day != "2026-11-17":
            raise RuntimeError(
                f"旧建仓价没跟着新帧更新（会锁死错值）："
                f"{st2.theo_px} @ {st2.theo_day}（应 11.0 @ 2026-11-17）")
        # 前复权帧 → 建仓价必须折回**原始价**（和实时报价同一口径）
        frame3 = _pd.DataFrame({"close": [10.0, 11.0, 20.0, 20.0, 20.0, 20.0],
                                "adj_factor": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]},
                               index=idx)
        run._note_theo_entry(st2, frame3, _pd.Series([0, 0, 1, 1, 1, 1], index=idx))
        if abs(st2.theo_px - 10.0) > 1e-9:
            raise RuntimeError(
                f"前复权价没折回原始价：{st2.theo_px}（20 / adj 2.0 应当 10）")
        # 🔴 成交价取**下一根的开盘**（不是收盘），且加上每边成本。
        # 用户 2026-11-20：「开仓价格是开盘价格+成本，收盘价格比这个低，应该有
        # 亏损，不是 0」—— 取收盘价时「今天开的仓」理论价正好等于今天的收盘价，
        # 两列永远显示 0。
        del run._side_cost_px                       # 还原真实现
        frame4 = _pd.DataFrame({"open": [10.0] * 6, "close": [10.0] * 6,
                                "adj_factor": [1.0] * 6}, index=idx)
        run._note_theo_entry(st2, frame4,
                             _pd.Series([0, 1, 1, 1, 1, 1], index=idx))
        if abs(st2.theo_px - (10.0 + _c)) > 1e-9:
            raise RuntimeError(
                f"多头开仓价要取「翻向下一根的开盘 + 成本」：{st2.theo_px}"
                f"（应 {10.0 + _c}）")
        run._note_theo_entry(st2, frame4,
                             _pd.Series([0, -1, -1, -1, -1, -1], index=idx))
        if abs(st2.theo_px - (10.0 - _c)) > 1e-9:
            raise RuntimeError(
                f"空头开仓价要「开盘 − 成本」（卖贱）：{st2.theo_px}"
                f"（应 {10.0 - _c}）")
        # 信号归零 → 清掉建仓价
        run._note_theo_entry(st2, frame, _pd.Series([0, 0, 0, 0, 0, 0], index=idx))
        if st2.theo_px or st2.theo_sig:
            raise RuntimeError("信号归零后要清掉理论建仓价")

        # 🔴 **主口径：理论建仓价由策略自己写**（用户 2026-11-20：
        # 「开仓价格应该就是根据策略来，而不是自己设定一个规则」
        # 「理论开仓价格，应该是从实盘策略中直接获取，不然如果 k 线中开仓，
        #  又算不对了」）。优先级 下单价 > 触发价 > 推断，低档不许盖高档。
        st2.theo_px, st2.theo_day, st2.theo_sig, st2.theo_src = 0.0, "", 0, ""
        st2.target = 3
        run._note_order_entry(st2, 3, 5000.0)
        if abs(st2.theo_px - 5000.0) > 1e-9 or st2.theo_src != R.THEO_ORDER:
            raise RuntimeError(
                f"策略的下单价没记成理论建仓价：{st2.theo_px} / {st2.theo_src}")
        if st2.theo_day != _dl.trading_day():
            raise RuntimeError(f"下单价要落在今天的交易日上：{st2.theo_day}")
        # 同向再来一笔（加仓）不许改建仓价：改了持仓盈亏会凭空涨一截
        run._note_order_entry(st2, 2, 5100.0)
        if abs(st2.theo_px - 5000.0) > 1e-9:
            raise RuntimeError(f"同向加仓把建仓价刷成 {st2.theo_px} 了")
        # 平仓单更不许写：减仓的价不是建仓价
        run._note_order_entry(st2, -3, 3000.0)
        if abs(st2.theo_px - 5000.0) > 1e-9:
            raise RuntimeError(f"平仓单把建仓价刷成 {st2.theo_px} 了")
        # 显示层的推断（下一根开盘 + 成本）不许盖真实下单价
        run._note_theo_entry(st2, frame4, _pd.Series([0, 1, 1, 1, 1, 1], index=idx))
        if abs(st2.theo_px - 5000.0) > 1e-9:
            raise RuntimeError(
                f"「下一根开盘」的推断裂把真实下单价盖掉了：{st2.theo_px}")
        # K 线内触发价：它就是策略给的价，叠一层每边成本（触发价不含滑点）
        st2.theo_px, st2.theo_day, st2.theo_sig, st2.theo_src = 0.0, "", 0, ""
        _c900 = run._side_cost_px(st2, 900.0)
        run._note_ib_entry(st2, 900.0, 1)
        if abs(st2.theo_px - (900.0 + _c900)) > 1e-9 or st2.theo_src != R.THEO_IB:
            raise RuntimeError(
                f"K 线内触发价没记成建仓价：{st2.theo_px}"
                f"（应 {900.0 + _c900}）/ {st2.theo_src}")
        # 紧跟着的那一笔真实委托要能把它升成「下单价」
        run._note_order_entry(st2, 3, 905.0)
        if abs(st2.theo_px - 905.0) > 1e-9 or st2.theo_src != R.THEO_ORDER:
            raise RuntimeError(
                f"触发价没被真实下单价升上去：{st2.theo_px} / {st2.theo_src}")
        # 方向翻了 ⇒ 新的一仓，无条件覆盖
        st2.target = -2
        run._note_order_entry(st2, -2, 880.0)
        if abs(st2.theo_px - 880.0) > 1e-9 or st2.theo_sig != -1:
            raise RuntimeError(
                f"翻向后建仓价没重设：{st2.theo_px} / sig {st2.theo_sig}")
        # 方向对不上（手上是上一波的价）⇒ 两列必须写「—」，不能拿旧价算
        # （拿旧价算会把盈亏符号直接反掉）
        st2.theo_px, st2.theo_day, st2.theo_sig, st2.theo_src = (
            1000.0, _dl.trading_day(), 1, R.THEO_ORDER)
        st2.target = -2
        _p5, _d5, _q5, _n5 = run._slot_pnl(st2)
        if not (_q5 != _q5):
            raise RuntimeError(
                f"理论仓方向和建仓价对不上时不能出数：pos {_q5} note {_n5}")
        st2.target = 0

        # 清空账本，免得写进真实的持仓文件
        st.held.clear()
        st.cost.clear()
        R.positionbook.forget(spec.label(), "__pnl_a")

        # ---- 界面三列 ----
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])      # noqa: F841
        from gui.portfolio_page import PortfolioPage

        page = PortfolioPage()
        try:
            page._items = []
            t = page.tbl_strat
            hdr = [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())]
            want = ["品种价格", "当日盈亏", "持仓盈亏"]
            for name in want:
                if name not in hdr:
                    raise RuntimeError(f"策略表没有「{name}」列：{hdr}")
            if hdr.index("状态") > hdr.index("品种价格"):
                raise RuntimeError(f"这三列要在「状态」右边：{hdr}")
            t.setRowCount(1)
            page._render_pnl(t, 0, {"price": 1200.0, "day_pnl": 1234.0,
                                    "pos_pnl": -567.0, "underlying": "SHFE.rb2701"})
            got = [t.item(0, j).text() for j in (7, 8, 9)]
            # 品种价格的小数位跟合约的最小变动价位走（RB 一跳 1 ⇒ 写 1,200，
            # 不是 1,200.00）—— 用户 2026-11-20 要求
            if got != ["1,200", "+1,234", "-567"]:
                raise RuntimeError(f"三列填出来的字不对：{got}")
            # 不知道的数写「—」，不能写 0（0 会被读成「不赚不亏」）
            page._render_pnl(t, 0, {"price": None, "day_pnl": None,
                                    "pos_pnl": None, "multi": 27})
            got = [t.item(0, j).text() for j in (7, 8, 9)]
            if got != ["—", "—", "—"]:
                raise RuntimeError(f"算不出来的格子要写「—」：{got}")
        finally:
            page.deleteLater()
        return ("理论口径：台账改成 500/98765 两列一格不动、当天开的仓当日≡持仓、"
                "隔夜仓按昨结算、空仓写 0；建仓价由策略自己写（下单价 > 触发价 >"
                "推断，加仓/平仓/推断都不许盖下单价）；方向对不上写「—」；"
                "均价加权 1060/减仓不动/反手重设 900；"
                "三列在「状态」右侧且不知道时写「—」")

    check("策略列表：品种价格 / 当日盈亏 / 持仓盈亏", _strat_pnl_three_cols)

    def _strat_total_row():
        """策略表最后一行「合计」—— 用户 2026-11-20 要的统计行。

        要钉住三件事：

        - 合计 = **各条策略的当日盈亏 / 持仓盈亏相加**，口径和上面每一行一样；
        - 只加**父行**：展开出来的品种行是一条策略内部的拆分，一起加就把它
          算了两遍（这是这类汇总行最容易犯的错）；
        - 有策略算不出来（`None`）时，合计只加算得出来的那部分，并标一个 `*`
          —— 写「—」会让整行失去意义，按 0 算又是在撒谎。
        """
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])      # noqa: F841
        from gui.portfolio_page import PortfolioPage
        from ttrade import strategy_store as SStore

        def _cfg(sid, name):
            return SStore.StrategyConfig(
                sid=sid, name=name, symbol="RB", exchange="SHFE", period="5m",
                family="factor", factor_mode="single", factor_keys=["rev_5"],
                account="__selftest_total__")

        c1, c2, c3 = _cfg("__tot_a", "甲"), _cfg("__tot_b", "乙"), _cfg("__tot_c", "丙")
        page = PortfolioPage()
        try:
            page._items = [c1, c2, c3]
            page._slot_multi = {}
            page._slot_view = {
                "__tot_a": {"price": 1200.0, "day_pnl": 1234.0, "pos_pnl": -567.0,
                            "target": 3, "lots": 3, "multi": 0, "running": True,
                            "underlying": "SHFE.rb2701"},
                "__tot_b": {"price": 900.0, "day_pnl": -234.0, "pos_pnl": 100.0,
                            "target": -1, "lots": 1, "multi": 0, "running": True,
                            "underlying": "SHFE.rb2701"},
                # 丙：算不出来（没行情 / 乘数没维护）
                "__tot_c": {"price": None, "day_pnl": None, "pos_pnl": None,
                            "target": 0, "lots": 0, "multi": 0, "running": False,
                            "underlying": ""},
            }
            page._rows = []
            page._fill_strategies()
            page._render()
            t = page.tbl_strat
            last = t.rowCount() - 1
            if last != len(page._rows):
                raise RuntimeError(
                    f"合计行要正好在最后一行：row {last} / 数据 {len(page._rows)} 行")
            if not t.item(last, 0).text().startswith("合计"):
                raise RuntimeError(f"最后一行的名字不对：{t.item(last, 0).text()}")
            # 甲 + 乙；丙算不出来 ⇒ 只加两个，并打 *
            got_day = t.item(last, 8).text()
            got_pos = t.item(last, 9).text()
            if got_day != "+1,000*" or got_pos != "-467*":
                raise RuntimeError(
                    f"合计行数字不对：day {got_day} pos {got_pos}"
                    f"（应 +1,000* / -467*）")
            # 单行还是各自的数，没有被合计行污染
            if t.item(0, 8).text() != "+1,234" or t.item(0, 9).text() != "-567":
                raise RuntimeError(
                    f"第一条策略那一行被改了：{t.item(0, 8).text()} / "
                    f"{t.item(0, 9).text()}")
            # 三条都能算 ⇒ 没有星号
            page._slot_view["__tot_c"] = dict(
                page._slot_view["__tot_b"], day_pnl=0.0, pos_pnl=0.0, price=900.0)
            page._render()
            if t.item(last, 8).text() != "+1,000" or t.item(last, 9).text() != "-467":
                raise RuntimeError(
                    f"三条都算得出来时不该带 *：{t.item(last, 8).text()} / "
                    f"{t.item(last, 9).text()}")
            # 🔴 展开多品种：子行不许被加进合计（不然那一条算两遍）
            c4 = SStore.StrategyConfig(
                sid="__tot_d", name="丁", symbol="RB", exchange="SHFE",
                period="5m", family="factor", factor_mode="single",
                factor_keys=["rev_5"], account="__selftest_total__",
                multi_symbols=["RB", "HC"])
            page._items = [c4]
            page._expanded = {"__tot_d"}
            page._slot_multi = {
                "__tot_d": {
                    "": {"price": 1200.0, "day_pnl": 50.0, "pos_pnl": -20.0,
                         "target": 1, "lots": 1, "multi": 0, "running": True,
                         "underlying": "SHFE.rb2701"},
                },
            }
            page._slot_view = {}
            page._fill_strategies()
            page._render()
            t = page.tbl_strat
            if t.rowCount() != 1 + 2 + 1:
                raise RuntimeError(
                    f"展开后该是「父 1 + 子 2 + 合计 1」四行：{t.rowCount()}")
            last = t.rowCount() - 1
            if t.item(last, 8).text() != "+50" or t.item(last, 9).text() != "-20":
                raise RuntimeError(
                    f"子行被算进合计了（应当只加父行）："
                    f"{t.item(last, 8).text()} / {t.item(last, 9).text()}")
            if not t.item(last, 0).text().endswith("1 条策略）"):
                raise RuntimeError(f"合计行数错了：{t.item(last, 0).text()}")
        finally:
            page.deleteLater()
        return ("合计行在最后一行：甲 +1,234/−567 与 乙 −234/+100 相加得 "
                "+1,000/−467；有策略算不出来时标 *；单行数字不受影响；"
                "多品种展开后子行不加进合计（只有 1 条策略的量）")

    check("策略列表：最后一行「合计」", _strat_total_row)

    def _lib_scroll_and_tick():
        """策略库对话框记住滚动位置 + 价格框步长 = 合约的最小变动价位。

        两条都是用户 2026-11-20 提的：

        - 「加载策略的时候，记住滚动滑轮的位置，下次打开继续在这个位置，方便
          找策略」—— 记的是**那一行的 sid**，库里增删过后仍能找回同一条；
          sid 没了退回行号；两者都没有就停在顶部。
        - 「价格框里头，上下调节价格的按钮，按照合约的最小跳动来自动调整幅度
          为最小跳动」—— 铜一跳 10、螺纹 1、还有 0.5 一档的品种，固定步长 1
          会「按一下跳两跳」或「按十下才动一格」。
        """
        from PySide6.QtWidgets import (
            QAbstractItemView, QApplication, QDoubleSpinBox)

        app = QApplication.instance() or QApplication([])      # noqa: F841
        from gui.portfolio_page import LibraryDialog
        from gui.trade_page import TradePage, _tick_decimals
        from ttrade import strategy_store as SStore

        items = [SStore.StrategyConfig(
            sid=f"__scr_{i:02d}", name=f"滚动{i:02d}", symbol="RB",
            exchange="SHFE", period="5m", family="rule", rule_key="ma_cross",
            account="__selftest__") for i in range(40)]

        def top_row(dlg) -> int:
            return dlg.lst.indexAt(dlg.lst.viewport().rect().topLeft()).row()

        # ---- ① 滚到第 12 行再关，重开要回到 12 ----
        d1 = LibraryDialog(items, title="加载策略")
        d1.resize(720, 260)
        d1.show()
        app.processEvents()
        app.processEvents()
        d1.lst.scrollToItem(d1.lst.item(12), QAbstractItemView.PositionAtTop)
        app.processEvents()
        if top_row(d1) != 12:
            raise RuntimeError(f"策略库列表滚不动，视口顶行是 {top_row(d1)}")
        d1.close()
        app.processEvents()
        got = SStore.ui_state(LibraryDialog.SCROLL_KEY, {})
        if not isinstance(got, dict) or got.get("sid") != "__scr_12":
            raise RuntimeError(f"滚动位置没落盘：{got}")

        d2 = LibraryDialog(items, title="加载策略")
        d2.resize(720, 260)
        d2.show()
        app.processEvents()
        app.processEvents()
        if top_row(d2) != 12:
            raise RuntimeError(
                f"重开没回到上次那一行（视口顶行 {top_row(d2)}，应 12）")
        d2.close()
        app.processEvents()

        # 库里那条被删了 ⇒ 退回行号，不许崩、不许跳回顶部
        SStore.put_ui_state(LibraryDialog.SCROLL_KEY,
                            {"sid": "__gone__", "row": 9})
        d3 = LibraryDialog(items, title="加载策略")
        d3.resize(720, 260)
        d3.show()
        app.processEvents()
        app.processEvents()
        if top_row(d3) != 9:
            raise RuntimeError(
                f"sid 找不到时要退回行号 9：{top_row(d3)}")
        d3.close()
        app.processEvents()

        # ---- ② 价格框步长 = 最小变动价位 ----
        class _Stub:
            def __init__(self, stock=False):
                self._stock_mode = stock
                self.sp_price = QDoubleSpinBox()
                self.sp_price.setDecimals(4)          # 出厂就是 4 位
                self.sp_price.setRange(0, 9_999_999)
                self.sp_price.setSingleStep(1)        # 和出厂默认一致

        def apply(sym, value, tick=0.0, stock=False):
            o = _Stub(stock)
            o.sp_price.setValue(value)
            got_tick = TradePage._apply_tick(o, sym, tick)
            return (got_tick, o.sp_price.singleStep(),
                    o.sp_price.decimals(), o.sp_price.value())

        # 铜一跳 10：步长 10，且不再显示 8240.0000
        t, step, dec, val = apply("SHFE.cu2701", 8240.0)
        if abs(t - 10) > 1e-9 or abs(step - 10) > 1e-9 or dec != 0:
            raise RuntimeError(f"铜的步长不对：tick {t} 步长 {step} 位数 {dec}")
        if abs(val - 8240.0) > 1e-9:
            raise RuntimeError(f"对齐位数时把价格改掉了：{val}")
        # 螺纹一跳 1
        t, step, dec, _v = apply("SHFE.rb2701", 3550.0)
        if abs(t - 1) > 1e-9 or abs(step - 1) > 1e-9 or dec != 0:
            raise RuntimeError(f"螺纹的步长不对：tick {t} 步长 {step} 位数 {dec}")
        # 🔴 调用方传进来的 tick（可能还是**上一个合约**的报价）不许盖掉品种表。
        # 2026-11-23 用户报「价格框上下键的跳动不对，很多都跑到 10 了，连 SA、IM
        # 也是」—— 10 就是之前看过的铜。报价是异步的，改完合约那一刻
        # `st.quote` 往往还是旧合约的，按它设步长就停在上一个品种上。
        t, step, dec, _v = apply("CZCE.SA2701", 1400.0, tick=10.0)
        if abs(step - 1) > 1e-9:
            raise RuntimeError(f"SA 的步长被传进来的 10 带跑了：{step}")
        t, step, dec, _v = apply("CFFEX.IM2612", 5479.0, tick=10.0)
        if abs(step - 0.2) > 1e-9 or dec != 1:
            raise RuntimeError(f"IM 的步长被传进来的 10 带跑了：{step} 位数 {dec}")
        # 品种表里没有的（新上市 / 拼错的品种）：退化用传进来的 tick，别固定在 1
        _t, step, _d, _v = apply("ZZZZ2701", 100.0, tick=2.0)
        if abs(step - 2) > 1e-9:
            raise RuntimeError(f"表里没有的品种没用兜底 tick：{step}")
        # 0.5 一档：步长 0.5、留 1 位
        _t, step, dec, val = apply("", 3550.5, tick=0.5)
        if abs(step - 0.5) > 1e-9 or dec != 1 or abs(val - 3550.5) > 1e-9:
            raise RuntimeError(f"0.5 一档不对：步长 {step} 位数 {dec} 值 {val}")
        # 现值在目标位数下表示不出来（tick 和价不是一套）⇒ 不许四舍五入
        _t, _s, dec, val = apply("", 3550.55, tick=1.0)
        if abs(val - 3550.55) > 1e-9:
            raise RuntimeError(f"不该把 3550.55 抹成 {val}（tick=1 时）")
        # 0.01 → 10 一档，小数位要能收回来
        if (_tick_decimals(0.01), _tick_decimals(0.2),
                _tick_decimals(10.0), _tick_decimals(0.005)) != (2, 1, 0, 3):
            raise RuntimeError("最小变动价位 → 小数位换算错了")
        # A 股不在品种表里，但一档就是 0.01 元
        t, step, dec, val = apply("SSE.600000", 12.34, stock=True)
        if abs(step - 0.01) > 1e-9 or dec != 2 or abs(val - 12.34) > 1e-9:
            raise RuntimeError(f"A 股步长不对：{t} 步长 {step} 位数 {dec} 值 {val}")

        # ---- ③ 显示位数：品种价格列 / 报价框都按 tick 收 ----
        from ttrade import symbols_info as SI

        if SI.contract_product("SHFE.cu2701") != ("CU", "SHFE"):
            raise RuntimeError("交易所.合约 拆品种错了")
        if SI.contract_product("rb2601") != ("RB", ""):
            raise RuntimeError("裸合约代码拆错了")
        for sym, raw, want in (
                ("GFEX.lc2701", 90040.0, "90,040"),      # 一跳 20
                ("ZF.CF2601", 5056.0, "5,056"),          # 一跳 5
                ("CFFEX.T2612", 118.115, "118.115"),     # 一跳 0.005
                ("SHFE.rb2701", 5942.8, "5,942.8"),      # 表示不出来就多留一位
        ):
            got = SI.price_text(raw, sym)
            if got != want:
                raise RuntimeError(f"{sym} 的价格位数不对：{got}（应 {want}）")
        # 涨跌带符号；不知 tick 退回 2 位；None 写「—」
        if SI.price_text(-1460.0, "GFEX.lc2701", signed=True) != "-1,460":
            raise RuntimeError("涨跌的符号/位数不对")
        if SI.price_text(1.2, "") != "1.20":
            raise RuntimeError("不知道 tick 时要退回 2 位小数")
        if SI.price_text(None, "GFEX.lc2701") != "—":
            raise RuntimeError("没有价格要写「—」")

        SStore.put_ui_state(LibraryDialog.SCROLL_KEY, {})     # 别留给下一次用例
        return ("策略库对话框：滚到第 12 行→关掉重开仍是 12、sid 失效退回行号 9；"
                "价格框步长跟着 tick 走（铜 10 / 螺纹 1 / 0.5 一档），"
                "小数位顺带对齐但不改价；品种价格与报价框的显示位数也按 tick 收"
                "（lc 90,040 / CF 5,056 / T 118.115），且不抹掉真实价位")

    check("策略库对话框滚动位置 / 价格框步长", _lib_scroll_and_tick)

    check("交易页：持仓当日盈亏 / 成交手续费与按品种统计 / CTP 委托时间 / 报价框",
          _trade_pnl_quote)

    def _close_pnl_replay():
        """重新登录后「平仓盈亏」不许整列变 0/—（2026-11-23 用户）。

        `est_close_pnl` 靠持仓的开仓均价倒推，而那份缓存只在内存里：程序一重启
        就空，**已经平掉的仓**在持仓表里也没有均价 ⇒ 那一列全变 `None`（画「—」），
        界面看着像「平仓盈亏归 0」。真正的数据在当日成交流里（CTP 走
        `ReqQryTrade`），按时间 FIFO 就能还原。

        这些断言全是「算错了也不会报错、只会显示一个看着合理的数」的：
        方向符号、部分平仓的加权成本、跨零点排序、平昨的盯市基准。
        """
        from ttrade.ctp_live import CtpHub
        from ttrade.tqlive import (_last_daily_close, _trade_time_key,
                                   replay_close_pnl)

        def mult(_s):
            return 10.0

        # ---- ① 同日「开 → 平」：精确值（重启后 `close_profit` 是 None，要盖掉）
        trs = [
            {"trade_id": "1", "symbol": "SHFE.rb2601", "direction": "BUY",
             "offset": "OPEN", "price": 3500.0, "volume": 2,
             "time": "09:01:00", "close_profit": None},
            {"trade_id": "2", "symbol": "SHFE.rb2601", "direction": "SELL",
             "offset": "CLOSETODAY", "price": 3550.0, "volume": 2,
             "time": "10:30:00", "close_profit": None},
        ]
        if replay_close_pnl(trs, mult) != 1:
            raise RuntimeError("同日开→平没算出来（重启后平仓盈亏就该靠它）")
        if abs(trs[1]["close_profit"] - 1000.0) > 1e-6:
            raise RuntimeError(f"卖出平多算错：{trs[1]['close_profit']}"
                               "（应 (3550−3500)×2×10 = 1000）")
        if trs[0]["close_profit"] is not None:
            raise RuntimeError("开仓单被写了平仓盈亏（那一列只该有平仓的）")

        # ---- ② 部分平仓：成本按 FIFO 加权，不是拿最新一笔开仓价
        trs = [
            {"trade_id": "1", "symbol": "SHFE.rb2601", "direction": "BUY",
             "offset": "OPEN", "price": 3500.0, "volume": 2, "time": "09:01:00"},
            {"trade_id": "2", "symbol": "SHFE.rb2601", "direction": "BUY",
             "offset": "OPEN", "price": 3520.0, "volume": 2, "time": "09:05:00"},
            {"trade_id": "3", "symbol": "SHFE.rb2601", "direction": "SELL",
             "offset": "CLOSETODAY", "price": 3560.0, "volume": 3, "time": "10:00:00"},
        ]
        replay_close_pnl(trs, mult)
        # 成本 = (3500×2 + 3520×1) / 3 = 3506.667 ⇒ 盈亏 = 53.333×3×10
        if abs(trs[2]["close_profit"] - (3560.0 - 10520.0 / 3.0) * 3 * 10) > 1e-6:
            raise RuntimeError(f"分批开仓的 FIFO 加权成本算错：{trs[2]['close_profit']}")

        # ---- ③ 平空方向符号：卖空 3600、买平 3500 ⇒ 赚（不是亏）
        trs = [
            {"trade_id": "1", "symbol": "SHFE.rb2601", "direction": "SELL",
             "offset": "OPEN", "price": 3600.0, "volume": 1, "time": "09:01:00"},
            {"trade_id": "2", "symbol": "SHFE.rb2601", "direction": "BUY",
             "offset": "CLOSETODAY", "price": 3500.0, "volume": 1, "time": "10:00:00"},
        ]
        replay_close_pnl(trs, mult)
        if abs(trs[1]["close_profit"] - 1000.0) > 1e-6:
            raise RuntimeError(f"买入平空符号错：{trs[1]['close_profit']}（应为 +1000）")

        # ---- ④ 缺昨仓开仓记录、又没给昨结算价 ⇒ **不许编**，保持 None
        trs = [{"trade_id": "1", "symbol": "SHFE.rb2601", "direction": "SELL",
                "offset": "CLOSE", "price": 3550.0, "volume": 1,
                "time": "10:00:00", "close_profit": None}]
        if replay_close_pnl(trs, mult) != 0 or trs[0]["close_profit"] is not None:
            raise RuntimeError("缺昨仓开仓记录时硬编了一个平仓盈亏")

        # ---- ⑤ 给了昨结算价 ⇒ 按逐日盯市算（平仓价 − 昨结算价）
        trs = [{"trade_id": "1", "symbol": "SHFE.rb2601", "direction": "SELL",
                "offset": "CLOSEYESTERDAY", "price": 3550.0, "volume": 1,
                "time": "10:00:00", "close_profit": None}]
        replay_close_pnl(trs, mult, pre_settle_of=lambda _s: 3500.0)
        if abs(trs[0]["close_profit"] - 500.0) > 1e-6:
            raise RuntimeError(f"平昨的盯市口径算错：{trs[0]['close_profit']}"
                               "（应 (3550−3500)×1×10 = 500）")

        # ---- ⑥ 成交流凑不齐（只开到 1 手却平 2 手）⇒ 缺的那手按基准价补，
        #         凑到的那手仍按 FIFO。用户 2026-11-23：「平仓盈亏可以根据昨日
        #         结算价或者收盘价格来计算，如果获取不到历史成交的话」
        trs = [{"trade_id": "1", "symbol": "SHFE.rb2601", "direction": "SELL",
                "offset": "CLOSETODAY", "price": 3550.0, "volume": 2,
                "time": "10:00:00", "close_profit": None},
               {"trade_id": "2", "symbol": "SHFE.rb2601", "direction": "BUY",
                "offset": "OPEN", "price": 3400.0, "volume": 1,
                "time": "09:01:00", "close_profit": 0.0}]
        replay_close_pnl(trs, mult, pre_settle_of=lambda _s: 3500.0)
        # 成本 = 3400×1（FIFO）+ 3500×1（基准）= 6900 ⇒ (3550 − 3450)×2×10
        if trs[0]["close_profit"] is None or abs(trs[0]["close_profit"] - 2000.0) > 1e-6:
            raise RuntimeError(f"成交流缺一块时没按基准价补齐：{trs[0]['close_profit']}"
                               "（应 3400 那手按 FIFO、缺的手按 3500，得 2000）")

        # ---- ⑦ `est_close_pnl` 已经估出来的值**不许被覆盖**（持仓开仓均价
        #         比昨结算更贴这笔交易，只有在还有仓时才拿得到）
        trs = [{"trade_id": "1", "symbol": "SHFE.rb2601", "direction": "SELL",
                "offset": "CLOSE", "price": 3550.0, "volume": 2,
                "time": "10:00:00", "close_profit": 250.0}]
        replay_close_pnl(trs, mult, pre_settle_of=lambda _s: 3500.0)
        if abs(trs[0]["close_profit"] - 250.0) > 1e-6:
            raise RuntimeError(f"把 est 的估算值盖掉了：{trs[0]['close_profit']}")

        # ---- ⑧ 连基准都没有 ⇒ 宁可留「—」，不许按 0 价成本编一个
        trs = [{"trade_id": "1", "symbol": "SHFE.rb2601", "direction": "SELL",
                "offset": "CLOSE", "price": 3550.0, "volume": 1,
                "time": "10:00:00", "close_profit": None}]
        replay_close_pnl(trs, mult, pre_settle_of=lambda _s: 0.0)
        if trs[0]["close_profit"] is not None:
            raise RuntimeError("基准价是 0 时还硬算出一笔平仓盈亏")

        # ---- ⑨ 夜盘跨零点：21:05 开的仓必须排在 01:30 的平仓**之前**
        if not _trade_time_key({"time": "01:30:00"}) > _trade_time_key({"time": "23:05:00"}):
            raise RuntimeError("成交排序键跨零点错了（夜盘 01:30 被排到 23:05 前面）")
        trs = [
            {"trade_id": "2", "symbol": "SHFE.au2602", "direction": "BUY",
             "offset": "OPEN", "price": 3000.0, "volume": 1, "time": "21:05:00"},
            {"trade_id": "1", "symbol": "SHFE.au2602", "direction": "BUY",
             "offset": "OPEN", "price": 3500.0, "volume": 1, "time": "23:05:00"},
            {"trade_id": "3", "symbol": "SHFE.au2602", "direction": "SELL",
             "offset": "CLOSETODAY", "price": 3100.0, "volume": 1, "time": "01:30:00"},
        ]
        replay_close_pnl(trs, mult)
        # 先进先出 ⇒ 平掉的是 21:05 那笔 3000 的仓：(3100−3000)×1×10
        if abs(trs[2]["close_profit"] - 1000.0) > 1e-6:
            raise RuntimeError(f"夜盘跨零点 FIFO 错位：{trs[2]['close_profit']}"
                               "（应平掉最早那笔 3000 的仓 = 1000）")

        # ---- ⑩ 基准价的最后一级：本地日线最后一根收盘（行情/成交流都没有时）
        lc = _last_daily_close("SHFE.rb2601")
        if lc > 0:
            # 🔴 这里**不许写死价格**。这些 CSV 每天都在长：2026-11-23 收盘后
            # rb 的最后一根从 2999 变成 3004，写死的断言自己就红了（跟代码改动
            # 无关，纯粹是「把行情数据钉死在测试里」）。要验的是「从文件尾部
            # 倒着取到的是**最后一行**的 close」，所以拿整份解析的最后一根来比
            # —— 两条路径不同（seek 尾部 vs 全量 read_csv），取错行立刻能发现。
            import pandas as _pd

            from ttrade import symbols_info as _SI
            from ttrade.tqlive import daily_csv_path as _dcp

            _vari, _ex = _SI.contract_product("SHFE.rb2601")
            _full = _pd.read_csv(_dcp(_vari.lower(), _ex))
            _want = float(_full["close"].iloc[-1])
            if abs(lc - _want) > 1e-6:
                raise RuntimeError(
                    f"本地日线昨收读错了：{lc}（库里最后一根是 {_want}）")
        if _last_daily_close("SHFE.zzzz9999") != 0.0:
            raise RuntimeError("没有日线文件的品种该返回 0，不许瞎猜")

        # ---- ⑧ CTP 的成交表**不许**在 replay 之前截断到 200 笔
        src = {}
        for i in range(205):
            src[f"t{i}"] = {"trade_id": f"t{i}", "instrument": "rb2601",
                            "direction": "0" if i % 2 == 0 else "1",
                            "offset": "0", "price": 3500.0, "volume": 1,
                            "time": "09:00:%02d" % (i % 60)}
        got = CtpHub([])._trades(src)
        if len(got) != 205:
            raise RuntimeError(f"CTP 成交在倒推前就没截断了：{len(got)} 行"
                               "（只留 200 笔的话早上的开仓记录会丢，"
                               "平仓一笔都算不出来）")
        # 平昨（'4'）必须映射成 CLOSE 系列，别当成开仓
        one = CtpHub([])._trades({"x": {"trade_id": "x", "instrument": "rb2601",
                                        "direction": "1", "offset": "4",
                                        "price": 3500.0, "volume": 1,
                                        "time": "09:00:00"}})
        if not one or one[0]["offset"] != "CLOSEYESTERDAY":
            raise RuntimeError(f"CTP 平昨（offset=4）没映射成 CLOSE 系列："
                               f"{one[0]['offset'] if one else '—'}")

        return ("平仓盈亏倒推：同日开→平 1000 元、分批开仓按 FIFO 加权、平空符号 +、"
                "缺开仓记录按基准价补齐（3400/FIFO + 3500/基准 ⇒ 2000）、平昨盯市 500、"
                "est 的估算值不被覆盖、没基准宁可留「—」、本地日线昨收兜底、"
                "夜盘跨零点 21:05 的仓先平（1000），CTP 成交倒推前不截断")

    check("重新登录后的平仓盈亏（成交流 FIFO + 昨结算/昨收兜底）", _close_pnl_replay)

    def _one_session() -> str:
        """一个账户一条会话（2026-11-23 融航「两块功能脱节」的彻底解法）。

        组合页（`BrokerApi`）和交易页（`gui.live.LiveWorker`）早先各自
        `CtpHub([spec]).start()` —— 同一个账户在柜台那边挂着**两条 CTP 会话**。
        症状正好是用户说的「脱节」，三件事叠在一起：

        - **委托回报按会话推送**：交易页「一键对齐」发出去的单，回报只进交易页
          那条会话，组合页的 `_ord` 一直空 ⇒「对齐的委托在账户透视里查不到」；
        - **查询额度各抢一份**：柜台「每秒一个查询」按会话算，两套闸门撞车时被
          丢的那笔**不回包**，于是「一边刷得快、一边刷得慢」；
        - **持仓两份快照** ⇒「交易页有 7 个合约，监控器全说不一致」。

        现在两边都到 `ttrade/hubreg.py` 认领**同一个对象**。这个块钉四件事：
        1. 同一账户两次 `acquire()` 拿到同一对象、refs=2；换了前置就是另一条；
        2. 归属权：只有最后一个使用者放手才真的 `close()`（否则交易页一收尾，
           组合页的持仓和回报当场断，还不报错）；
        3. `CtpHub.start()` 幂等 —— 后到的一方不会再登一次（二次登录会把先到
           那方的 `trader` 换掉，它手里的持仓/回报静默失效）；
        4. `pump_queries()` 的闸门原子 —— 两个页面同时调，闸门被占着时放过这一
           拍，而不是两个都通过判断、同一毫秒发两笔。
        """
        import inspect

        from ttrade import ctp_live as CL
        from ttrade import hubreg
        from ttrade.tqlive import AccountSpec

        hubreg.clear()

        class _H:
            """假柜台：只数 start / close 被调了几次。"""

            def __init__(self, specs=None):
                self.specs = list(specs or [])
                self.starts = 0
                self.closed = 0

            def start(self):
                self.starts += 1

            def close(self):
                self.closed += 1

        sp = AccountSpec(name="融航户A", kind="rohon", user="u1", td_front="t1")
        a = hubreg.acquire("rohon", [sp], lambda: _H([sp]))
        b = hubreg.acquire("rohon", [sp], lambda: _H([sp]))
        assert a is b, ("同一个账户拿到了两条会话 —— 委托回报互不相通、"
                        "查询额度各抢一份，就是用户说的「脱节」")
        assert hubreg.refs(a) == 2, f"引用数不对：{hubreg.refs(a)}"

        # 换了交易前置（SimNow 改融航、换服务器）就是另一条会话：拿旧连接去发
        # 新配置的单，等于把单打到另一个柜台上
        other = hubreg.acquire(
            "rohon", [AccountSpec(name="融航户A", kind="rohon", user="u1",
                                  td_front="t2")], lambda: _H())
        assert other is not a, "换了交易前置还是同一条会话（旧连接被复用）"
        assert hubreg.release(other) is True, "这条会话只有它自己用，该放手"
        assert hubreg.size() == 1, f"注册表里多出僵尸：{hubreg.describe()}"
        other.close()

        # 归属权：交易页先收尾，组合页还在用
        assert hubreg.release(a) is False, "还有人握着引用，却把它判成该关了"
        assert a.closed == 0, "组合页还在靠这条会话读持仓，被交易页关掉了"
        assert hubreg.release(a) is True, "最后一个使用者放手却没资格关"
        assert hubreg.size() == 0, f"注册表没清干净：{hubreg.describe()}"

        # 不在表里的连接（纯行情会话 / 单测注入的假柜台）照旧由调用方关。
        # 🔴 天勤的**账户**会话 2026-11-24 起进表了（见下面 `_tq_one_session`），
        # 这里说的「不在表里」只剩纯行情那种（`LiveHub(specs=[])` 不建键）
        assert hubreg.release(_H()) is True, "不在注册表里的连接被当成了共享会话"

        # ---- `start()` 幂等，但**断线之后必须重登**
        #
        # 幂等是为了「后到的一方不再登一次」；而「断线重连」正好相反 —— 那时
        # `_started` 还立着、`trader` 还指着对象，只有 `logged_in` 被清了。拿
        # `_started` 当判据，重连就变成「直接返回一条死会话」：界面写已连接，
        # 行情/持仓/下单一样都没有。
        import threading as _th

        class _Cfg:
            name = "测试户"
            user_id = "u1"
            password = "p1"
            td_front = "tcp://127.0.0.1:1"
            md_front = ""

        class _FakeTd:
            last_error = ""

            def __init__(self):
                self.logged_in = _th.Event()
                self.stopped = 0
                self.queries = []

            def start(self):
                self.logged_in.set()

            def wait_ready(self, timeout=0.0):
                return True

            def confirm_settlement(self):
                return False

            def query_positions(self):
                self.queries.append("pos")

            def query_account(self):
                self.queries.append("acct")

            def stop(self):
                self.stopped += 1
                self.logged_in.clear()

        class _Hub(CL.CtpHub):
            made: list = []

            def check_deps(self):
                pass

            def make_config(self, spec):
                return _Cfg()

            def make_trader(self, cfg):
                t = _FakeTd()
                type(self).made.append(t)
                return t

            def make_md(self, cfg):
                raise AssertionError("账户没填 md_front，不该去连行情")

        _Hub.made = []
        hb = _Hub([object()])          # 内容无所谓，`make_config` 被架空了
        hb.start()
        hb.start()
        hb.start()
        assert len(_Hub.made) == 1, \
            f"`start()` 不幂等，后到的一方又登了一次（共 {len(_Hub.made)} 次）"
        assert hb._started is True, "登上了却没打记号，下次还会再登一遍"
        assert _Hub.made[0].queries == ["pos"], \
            f"登录后的第一拍只该发持仓查询（柜台每秒一个）：{_Hub.made[0].queries}"

        # 连接掉了（`logged_in` 清掉，但 `_started` 还在）：必须重登，且先停旧客户端
        hb.trader.logged_in.clear()
        hb.start()
        assert len(_Hub.made) == 2, ("断线后 `start()` 直接短路返回了一条死会话 —— "
                                     "界面写已连接，持仓和下单一样都没有")
        assert _Hub.made[0].stopped >= 1, \
            "重登前没停掉上一条客户端（两条 TCP 会话抢同一个账户，先登的从此收不到回报）"
        hb.close()
        assert hb._started is False, "关掉之后还留着「已登录」的记号"
        hb.start()
        assert len(_Hub.made) == 3, "关掉之后重新 start 该真的重登，不该短路"
        hb.close()

        # ---- 查询闸门：两个页面同时调，一秒内也只能放一笔
        class _R:
            orders_loaded = trades_loaded = True

            def __init__(self):
                self.calls = []

            def query_account(self):
                self.calls.append("acct")

            def query_positions(self):
                self.calls.append("pos")

        rot = CL.CtpHub([])
        rot.trader = _R()
        rot._last_query = 0.0
        assert rot._q_lock.acquire(blocking=False), "闸门锁拿不到"
        try:
            assert rot.pump_queries(force=True) == "", \
                "闸门被另一个页面占着时还在发查询（同一毫秒两笔，柜台丢一笔不回包）"
            assert rot.trader.calls == [], rot.trader.calls
        finally:
            rot._q_lock.release()
        assert rot.pump_queries(force=True) == "positions", "闸门放开后该能发出去"

        # ---- 接线：两个页面都必须走注册表，不能自己再 new 一条
        from gui.live import make_hub

        h1 = make_hub("rohon", [sp])
        h2 = make_hub("rohon", [sp])
        assert h1 is h2, "交易页和组合页拿的不是同一条会话"
        assert hubreg.refs(h1) == 2, f"两边各借一次，refs 该是 2：{hubreg.refs(h1)}"
        hubreg.release(h1)
        assert hubreg.release(h1) is True
        h1.close()

        from ttrade import broker_api as BA

        for fn, needle, why in (
            (BA.BrokerApi.start, "hubreg.acquire",
             "`BrokerApi.start()` 没走注册表 —— 组合页会为同一个账户再开一条会话"),
            (BA.BrokerApi.close, "hubreg.release",
             "`BrokerApi.close()` 直接 close —— 会把交易页那条会话一起掐掉"),
        ):
            src = "\n".join(ln for ln in inspect.getsource(fn).splitlines()
                            if not ln.strip().startswith("#"))
            assert needle in src, why

        from gui import live as GL

        src = inspect.getsource(GL.HubSet.close)
        assert "hubreg.release" in src, \
            "`HubSet.close()` 直接关连接 —— 组合页的会话会被交易页收尾带走"

        hubreg.clear()
        return ("同一账户两次 acquire 拿到同一对象（refs=2，换前置才换会话）；"
                "只有最后一个使用者放手才 close（交易页先收尾不掐组合页）；"
                "start() 幂等、关掉后能重登；查询闸门是原子的（被占时放过这一拍）；"
                "make_hub / BrokerApi / HubSet 三处接线都走 hubreg")

    check("一个账户一条会话（交易页与组合页共用同一条连接）", _one_session)

    def _tq_one_session() -> str:
        """天勤也要一条会话（2026-11-24 用户要求）。

        用户原话：「天勤的账户也要和 CTP 和融航账户一样，同一个账户在整个系统
        共享同一个会话，这样一键对齐和账户持仓和委托成交的会话才能实时响应。」

        天勤原来被 `hubreg` 明确排除，理由是「`TqApi` 单线程语义，两个页面共用
        一个对象会写坏订阅状态」。结论没错，但办法应该是**让它有一条自己的
        线程**，而不是让它登两次 —— 天勤按会话推持仓 / 委托 / 成交，两条会话
        互不相通，用户看到的就永远是「下单了，监控器不动」。

        这个块钉五件事：
        1. `hubreg` 认天勤，同一账户两次 `acquire()` 是同一个对象；
        2. 键里带账户类型 —— 同一个天勤账号的 `kq`（模拟）和 `account`（实盘）
           必须是两条会话；
        3. `wait_update()` **只**在泵线程里被调（以前是调用方自己推，共用就会
           两条线程一起推）；
        4. `close()` 带引用计数（交易页先收尾不能把组合页的持仓掐断）；
        5. 组合页要的 `TqApi` 同名方法（`get_position` / `get_order` /
           `insert_order` / `cancel_order`）都转出去了，`positions_ready()` 用
           `Account.balance` 的 nan 区分「还没读到」和「真的是 0」。
        """
        import inspect
        import threading as _th

        from ttrade import hubreg
        from ttrade import strategy_runner as SR
        from ttrade import strategy_store as SStore
        from ttrade.tqlive import AccountSpec, LiveHub

        hubreg.clear()

        class _Pos:
            def __init__(self):
                self.instrument_id = "SHFE.cu2701"
                self.pos_long, self.pos_short = 0, 8
                self.pos = -8
                self.pos_long_today = self.pos_short_today = 0
                self.open_price_long = self.open_price_short = 0.0
                self.float_profit = self.margin = 0.0

        class _Ord:
            def __init__(self, oid):
                self.order_id = oid
                self.status = "ALIVE"

        class _Api:
            """假 `TqApi`：只实现 `LiveHub` 用到的几个方法，并记下谁在调。"""

            def __init__(self):
                self.threads = set()
                self.calls = []
                self.updates = 0
                self.orders = []

            def wait_update(self, deadline=None):
                self.threads.add(_th.current_thread().name)
                self.updates += 1
                return True

            def get_position(self, *a, **k):
                self.threads.add(_th.current_thread().name)
                return {"SHFE.cu2701": _Pos()}

            def get_order(self, *a, **k):
                self.threads.add(_th.current_thread().name)
                return {o.order_id: o for o in self.orders}

            def insert_order(self, **kw):
                self.threads.add(_th.current_thread().name)
                self.calls.append(("insert_order", dict(kw)))
                o = _Ord(f"o{len(self.orders) + 1}")
                self.orders.append(o)
                return o

            def cancel_order(self, order=None, **kw):
                self.threads.add(_th.current_thread().name)
                self.calls.append(("cancel_order", order))
                return True

            def close(self):
                self.calls.append(("close", None))

        class _Acc:
            def __init__(self, balance):
                self.balance = balance

            def get_account(self):
                return self

        spec = AccountSpec(name="快期A", kind="kq", user="u1", password="p1")
        made = []
        h1 = hubreg.acquire("tq", [spec],
                            lambda: made.append(1) or LiveHub([spec]))
        h2 = hubreg.acquire("tq", [spec],
                            lambda: made.append(1) or LiveHub([spec]))
        assert "tq" in hubreg.SHARED, "天勤还是没进注册表 —— 会登两条会话"
        assert h1 is h2, "同一个天勤账户拿到了两条会话（持仓/委托互不相通）"
        assert len(made) == 1, f"同一个账户被构造了 {len(made)} 次"
        assert hubreg.refs(h1) == 2, f"引用数不对：{hubreg.refs(h1)}"

        # 键里带账户类型：同一个天勤账号既能挂模拟也能挂实盘
        real = AccountSpec(name="快期A", kind="account", user="u1", password="p1")
        h3 = hubreg.acquire("tq", [real], lambda: LiveHub([real]))
        assert h3 is not h1, ("同一个天勤账号的模拟户和实盘户串成了一条会话 —— "
                              "下的单会全落在模拟盘里")
        assert hubreg.hub_key("tq", real)[-1] == "account", "键里没带账户类型"

        # ---- 泵线程独占 wait_update ----
        hub = h1
        api = _Api()
        hub.api = api
        hub._start_pump()
        assert hub.wait(1.0) is True, "泵没把数据推过来，wait() 一直超时"
        time.sleep(0.15)
        assert api.threads == {"tq-pump"}, (
            f"wait_update 被别的线程也调了：{api.threads} —— 两条线程驱动同一个 "
            "TqApi 会写坏内部订阅状态")

        # ---- 组合页要的 tqsdk 同名接口 ----
        assert sorted(hub.get_position()) == ["SHFE.cu2701"], "get_position 没转出去"
        assert hub.get_order() == {}, "get_order 没转出去"
        o = hub.insert_order(symbol="SHFE.cu2701", direction="BUY",
                             offset="CLOSE", volume=8, limit_price=70000.0)
        assert getattr(o, "order_id", ""), "insert_order 没转出去"
        assert api.calls[0][1]["volume"] == 8, "下单参数没原样传下去"
        assert hub.cancel_order(o) is True, "cancel_order 没转出去"

        # ---- positions_ready：空字典 ≠ 没仓 ----
        hub._accs = [(spec, _Acc(float("nan")))]
        assert hub.positions_ready() is False, \
            "第一帧还没到就说「读到了」—— 一键对齐会拿「账户是空的」重开一遍"
        hub._accs = [(spec, _Acc(1_000_000.0))]
        assert hub.positions_ready() is True, "账户数据到了还说没读到"

        # ---- 组合页的引擎直接吃这条会话（不用改一行）----
        hub._accs = [(spec, _Acc(1_000_000.0))]
        run = SR.PortfolioRunner.__new__(SR.PortfolioRunner)
        run._log_fn = run._diag = None
        run._mm_cache, run._mm_ts = [], 0.0
        run._align_ts = 0.0
        run.api = hub
        run._save_book = lambda s: None
        run._adopt_if_needed = lambda s: None
        run._pos = hub.get_position()          # tqsdk 形状的活引用
        run._orders = hub.get_order()
        run._slots = {}
        st0 = SR.SlotState(cfg=SStore.StrategyConfig(
            sid="t", name="cu", period="1h"))
        st0.running, st0.underlying, st0.target = True, "SHFE.cu2701", -8
        run._slots["t"] = st0
        run._align_slot = SR.SlotState(cfg=SStore.StrategyConfig(
            sid="__align__", name="一键对齐", period="1d"))
        assert run._side_pair("SHFE.cu2701") == (0, 8), \
            "组合页的引擎读不懂 LiveHub 转出来的持仓（tqsdk 字段对不上）"
        assert run._net_pos("SHFE.cu2701") == -8, "净仓算错了"
        assert run._pos_live() is True, "positions_ready 没接上 —— 对齐会被永久挡住"
        assert [x["symbol"] for x in run.mismatch(fresh=True)] == [], \
            "理论与账户明明一致却报不一致"

        # ---- close() 按引用计数 ----
        hub.close()                        # 交易页先收尾
        assert hub.api is not None, "组合页还在用这条会话，被交易页关掉了"
        assert hubreg.refs(hub) == 1, f"放手一次之后 refs={hubreg.refs(hub)}"
        hub.close()                        # 组合页收尾
        assert hub.api is None, "最后一个使用者放手却没关连接"
        assert ("close", None) in api.calls, "api.close() 没被调"
        assert hubreg.refs(hub) == 0, f"注册表留下僵尸：{hubreg.describe()}"

        # ---- 接线：组合页那条路必须借会话，不能自己登 ----
        src = "\n".join(ln for ln in inspect.getsource(
            SR.PortfolioRunner._connect_account).splitlines()
            if not ln.strip().startswith("#"))
        assert 'hubreg.acquire("tq"' in src, (
            "`PortfolioRunner._connect_account()` 还在自己 `TqApi(...)` —— "
            "同一个天勤账户会登两条会话")

        from gui import live as GL

        gsrc = "\n".join(ln for ln in inspect.getsource(GL.make_hub).splitlines()
                         if not ln.strip().startswith("#"))
        assert 'hubreg.acquire("tq"' in gsrc, \
            "交易页建天勤连接没走注册表 —— 反过来也不会复用组合页那条"

        h3.close()
        hubreg.clear()
        return ("天勤进注册表（同账户一条会话、键带 kq/实盘类型）；wait_update 只在 "
                "tq-pump 里被调；close 按引用计数；get_position/insert_order/"
                "cancel_order/positions_ready 都转出去了；组合页与交易页两处接线"
                "都走 hubreg")

    check("天勤也一个账户一条会话（交易页与组合页共用 TqApi）", _tq_one_session)

    def _hub_fail_cooldown() -> str:
        """登录失败后不许反复 `Release()` + `Init()`（2026-11-23 融航连崩三次）。

        现场：融航 `[-1] 交易核心未初始化` 一直登不上，用户每次点「启动全部 /
        刷新」，`_start_locked()` 就把老客户端 `Release()` 掉、再建一条新的
        `Init()`，失败时那条新的也在半路被 `Release()`。事件日志：
        `vendor\\rohon_ctp\\openctp_ctp.libs\\MSVCR100.dll` 0xc0000005 ——
        19:28:01 / 19:28:31 / 19:29:12 连崩三次，都是**原生的访问违例**，
        Python 层一条 traceback 都留不下来。

        判据只有一条：失败之后再来问，**一次也不许碰底层 API**。
        """
        import inspect
        import threading

        from ttrade import ctp_live as CL
        from ttrade import ctp_client as CC

        h = CL.CtpHub.__new__(CL.CtpHub)
        h._start_lock = threading.Lock()
        h._started = False
        h._fail_at = 0.0
        h._fail_msg = ""
        h._alive = lambda: False
        calls = {"n": 0}

        def _boom():
            calls["n"] += 1
            raise CL.CtpError("[-1] 交易核心未初始化")

        h._start_locked = _boom
        msgs = []
        for _ in range(3):
            try:
                h.start()
            except Exception as exc:     # noqa: BLE001
                msgs.append(str(exc))
        if calls["n"] != 1:
            raise RuntimeError(
                f"登录失败后还在反复连柜台（{calls['n']} 次）—— 每次都会走一遍 "
                "Release+Init，融航那条 dll 会在这种拆建下段错误")
        if "秒后可再试" not in msgs[-1]:
            raise RuntimeError(f"冷却期内的报错没说清楚什么时候能再试：{msgs[-1]!r}")
        # 冷却过了要能重试（别把按钮永久焊死）
        h._fail_at = 0.0
        try:
            h.start()
        except Exception:                # noqa: BLE001
            pass
        if calls["n"] != 2:
            raise RuntimeError("冷却期过去之后没再试 —— 「刷新」按钮会变成摆设")

        # Release 之前必须先摘 SPI，否则正在执行的回调会踩到已回收的对象
        src = inspect.getsource(CC._release_api)
        if src.index("api.RegisterSpi(None)") > src.index("api.Release()"):
            raise RuntimeError("`_release_api` 又变成先 Release 再摘 SPI 了")
        if "_release_api(api)" not in inspect.getsource(CC.TraderClient.stop):
            raise RuntimeError("`TraderClient.stop()` 绕过了 `_release_api`")
        return (f"失败 {calls['n'] - 1} 次只碰了一次底层 API（冷却 "
                f"{CL.CtpHub.FAIL_COOLDOWN:.0f} 秒，过后能重试）；"
                "Release 之前一定先 RegisterSpi(None)")

    check("登录失败后不反复 Release+Init（融航段错误）", _hub_fail_cooldown)

    def _no_dangling_api() -> str:
        """会话收掉之后，任何 `Req*` 都不许再碰 C++（2026-11-23 两次 0xc0000005）。

        现场：事件日志 20:32:57 与 21:09:26 两条一模一样 —— `pythonw.exe`、
        错误模块 `openctp_ctp.libs\\thosttraderapi_se-….dll`、`0xc0000005`、
        偏移量都是 `0x25d9e0`。崩溃前日志一切正常（全是 `[ALIGN]` / `[ORD]`
        下单），融航那边整天在刷 `[-1] 交易核心未初始化`。

        两个真根因，都在「句柄的生命周期」上：

        1. **流目录撞车**。`_flow_dir` 原来是 `td_{int(time.time())}` —— 秒级。
           而「启动全部」是把账户**并行**连的（融航 + CTP 几个户同一秒），两条
           `CThostFtdcTraderApi` 于是落在同一个目录，往同一组 `td_*.con` 里写。
        2. **`Req*` 与 `Release()` 之间没有锁**。每个调用点都是「先看一眼
           `self.api is None`、再用它发」两步，`stop()` 挤在中间就是拿悬空指针
           调进 C++ —— 这是访问违例，Python 层拦不到，进程直接没了。

        钉住四条：流目录唯一；关掉之后报单要**抛**、查询要**静默**；
        `Release()` 必须在**锁外**调用（持锁 Release 会和回调线程互等死锁）；
        `start()` 重复调用要先把老会话收掉。
        """
        import inspect as _ins
        import os as _os
        import threading as _th

        from ttrade import ctp_client as CC

        # ① 同一秒里连开两条也不许撞目录
        a = CC._new_flow_dir("td")
        b = CC._new_flow_dir("td")
        if a == b:
            raise RuntimeError(f"两个会话拿到同一个流目录（{a}）—— 必崩")
        for d in (a, b):
            if not _os.path.isdir(d):
                raise RuntimeError(f"流目录没建出来：{d}")

        # ② 句柄为 None：报单 / 撤单必须抛（调用方要知道没发出去），查询不许抛
        t = CC.TraderClient.__new__(CC.TraderClient)     # `__new__`：不碰 dll
        t.api = None
        t._api_lock = _th.RLock()
        if t._send("查询", lambda ap: 0, quiet=True) is not None:
            raise RuntimeError("会话已关，查询路径还在往下走（应该返回 None）")
        try:
            t._send("报单", lambda ap: 0)
        except CC.CtpError:
            pass
        else:
            raise RuntimeError("会话已关，报单路径没报错 —— 调用方会以为单子发出去了")

        # ③ `stop()`：锁内摘句柄、**锁外** Release
        holder = {}

        class _Api:
            def __init__(self):
                self.released = 0
                self.lock_free = False

            def RegisterSpi(self, _spi):
                pass

            def Release(self):
                self.released += 1
                # 这条最关键：Release 时锁必须是空的。CTP 的 `Release()` 会等
                # 自己的回调线程退出，而那条线程上的回调也要这把锁 ⇒ 持锁
                # `Release()` 就是互等，界面直接卡死（比崩溃更难查）
                cli = holder["cli"]
                got = cli._api_lock.acquire(blocking=False)
                self.lock_free = bool(got)
                if got:
                    cli._api_lock.release()

        api = _Api()
        t2 = CC.TraderClient.__new__(CC.TraderClient)
        t2.api = api
        t2._api_lock = _th.RLock()
        t2.connected, t2.logged_in, t2.settled = _th.Event(), _th.Event(), _th.Event()
        for ev in (t2.connected, t2.logged_in, t2.settled):
            ev.set()
        holder["cli"] = t2
        t2.stop()
        if t2.api is not None:
            raise RuntimeError("`stop()` 之后 `api` 没置空 —— 下一句还会用它")
        if api.released != 1:
            raise RuntimeError(f"`stop()` 没 Release 掉底层会话（{api.released} 次）")
        if not api.lock_free:
            raise RuntimeError("`Release()` 是在锁里调的 —— 会和回调线程互等死锁")
        if any(ev.is_set() for ev in (t2.connected, t2.logged_in, t2.settled)):
            raise RuntimeError("`stop()` 没清连接 / 登录 / 结算标志")
        try:
            t2._send("报单", lambda ap: 0)
        except CC.CtpError:
            pass
        else:
            raise RuntimeError("`stop()` 之后还能发单 —— 悬空句柄就是这么来的")
        # 重复 stop 不许再 Release 一次（第二次 Release 同一对象也是原生崩溃）
        t2.stop()
        if api.released != 1:
            raise RuntimeError("`stop()` 调两次会 Release 两次 —— CTP 侧会崩")

        # ④ 重复 `start()` 要先把老会话收掉（同一个账户两条 TCP 会话，柜台按
        #    后登的为准，先登那条从此收不到回报；老 api 不 Release 还漏线程）
        src = _ins.getsource(CC.TraderClient.start)
        if src.find("self.stop()") < 0 or \
                src.index("self.stop()") > src.index("CreateFtdcTraderApi"):
            raise RuntimeError("`start()` 重复调用没收掉老会话")
        # 建 → Init → 挂句柄必须在同一把锁里：`Init()` 是异步的，
        # `OnFrontConnected` 可能先到，那时 `self.api` 还是 None
        if src.index("with self._guard()") > src.index("api.Init()"):
            raise RuntimeError("`start()` 建会话的整段没在锁里（回调会先到）")

        # ⑤ 所有 `Req*` 都必须走 `_send`（不许再有裸 `self.api.Req*`）。
        #    🔴 用 AST 而不是字符串匹配：注释和 docstring 里写着
        #    「原来那个调用点是 `self.api.Req*Xxx()`」这种说明，字符串匹配会假红
        import ast as _ast

        bad = []
        for node in _ast.walk(_ast.parse(_ins.getsource(CC))):
            if not isinstance(node, _ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, _ast.Attribute):
                continue
            base = fn.value
            if (isinstance(base, _ast.Attribute) and base.attr == "api"
                    and isinstance(base.value, _ast.Name)
                    and base.value.id in ("self", "outer")):
                bad.append(f"{base.value.id}.api.{fn.attr}")
        if bad:
            raise RuntimeError(f"还有裸的句柄调用（那一步没有锁）：{sorted(set(bad))}")
        return ("流目录按「进程+秒+序号」唯一；会话关掉后报单抛错、查询静默返回"
                " None；`Release()` 在锁外且只调一次；`start()` 重复调用先收老会话；"
                "全部 `Req*` 走 `_send`（读句柄和用它是一个原子动作）")

    check("会话收掉后不许再碰 C++ 句柄（0xc0000005）", _no_dangling_api)


    def _acct_pump() -> str:
        """组合页每轮必须**同时**推行情和账户（2026-11-23 融航「监控器不动」）。

        用户报「一键对齐之后委托和成交都出来了，监控器这边还是没有及时更新账户
        的持仓」。日志里的决定性数字：`[MM]` 写着「已经不一致 **122 秒**」，而
        账户是 122 秒前刚连上的 —— 也就是说 `_pos` 从**连接那一刻**起就没变过。

        真根因：`PortfolioRunner.wait()` 只推全局行情 `MarketFeed`，柜台的
        `BrokerApi.wait_update()`（也就是 `_refresh()`）除了 `start()` 里那一次
        `force` 之后再没被调用过。于是持仓 / 委托 / 资金全冻住：
        `[POS]` 里那句「发过 **0 拍**持仓查询」就是这个 —— 一拍都没人发。
        对齐单的柜台回报其实早就到了，只是没人去读 `snapshot()`，界面只能说
        「柜台 N 秒没回委托」。

        另外一个必须一起钉的：`positions_loaded` 只说「读到过」，不说「刚才读过」。
        所以 `positions_ready()` 还得看 `pos_at`（最近一轮回包的时刻），超过
        `POS_STALE_SEC` 就改口说「还没同步」，不能让旧数冒充现值。
        """
        import time as _t
        import types

        from ttrade.broker_api import POS_STALE_SEC, BrokerApi
        from ttrade.strategy_runner import PortfolioRunner

        class _Feed:
            generation = 1

            def __init__(self):
                self.waits = 0

            def wait(self, timeout=0.0):
                self.waits += 1
                return True

        class _Api:
            def __init__(self):
                self.calls = 0

            def refresh_account(self):
                self.calls += 1

        r = PortfolioRunner.__new__(PortfolioRunner)
        r.feed = _Feed()
        r._feed_gen = 1                 # 已经对齐，别走「切源」分支
        r._klines, r._quotes = {}, {}
        r._log = lambda _m: None
        r.last_error = ""
        r.api = _Api()
        r.wait(0.01)
        r.wait(0.01)
        assert r.api.calls == 2, (
            "`PortfolioRunner.wait()` 没推账户连接 —— 柜台的 `_refresh()` 只在"
            f"连上那一刻跑过一次，持仓/委托从此冻住（推了 {r.api.calls} 次）")
        assert r.feed.waits == 2, "行情那一步被账户挤掉了"

        # 天勤那种接口没有 `refresh_account`，不能因此抛异常（策略照跑）
        r.api = object()
        r.wait(0.01)
        r.api = None
        r.wait(0.01)

        # ---- 持仓「读到过」≠「刚才读过」
        class _T:
            positions_loaded = True

            def __init__(self, at):
                self.pos_at = at
                self.account = {"margin": 0.0}

        api = BrokerApi.__new__(BrokerApi)
        api._pos = {"SHFE.sn2612": object()}
        api.last_error = ""
        api.hub = types.SimpleNamespace(
            trader=_T(_t.time() - POS_STALE_SEC - 5))
        assert api.positions_ready() is False, \
            "持仓几十秒没刷新过还算「读到了」，界面会拿旧数冒充现值"
        assert "没从柜台刷新" in api.last_error, api.last_error
        api.hub.trader.pos_at = _t.time()
        assert api.positions_ready() is True, "刚刷过的持仓被判成没读到"
        # 老后端（没有 `pos_at`）不受影响：不能因为加了新判据就把它们拦住
        api.hub.trader = _T(0.0)
        assert api.positions_ready() is True

        return ("组合页每轮同时推行情和账户（`refresh_account` 被调用，"
                "没有这个方法的接口不受影响）；持仓「读到过」还必须「刚才读过」"
                f"（超过 {POS_STALE_SEC:.0f} 秒没刷新 ⇒ 判未同步并写明原因）")

    check("组合页每轮同时推账户（否则持仓/委托冻在连接那一刻）", _acct_pump)

    def _monitor_cfg() -> str:
        """「监控设置」：账户重读周期 —— 默认 5 秒、钳到 1~60、改完**真的生效**。

        用户原话（2026-11-23）：「一键对齐的刷新时间改为每 5 秒刷新一次，增加一个
        监控设置按钮，可以设置一键对齐的刷新时间，目前 1-2 秒太少。」

        🔴 为什么下限是 1 秒而不是「随便填」：柜台同一会话「每秒最多**一个**查询，
        而且查询和报单共用这个名额」。周期压到 1~2 秒时账户一多就会和报单抢，
        抢输的常常是报单（表现成「点了没反应 / 单子晚到」）。

        三个判据：
          ① 默认 5 秒；文件缺失 / 内容坏掉都退回 5（不能因为一个坏 json 就起不来）
          ② 范围钳在 1~60
          ③ **改完立刻生效** —— `setattr(runner, "monitor", cfg)` 必须同时换掉节流
             用的 `_acct_sec`。只写文件的话用户点完「确定」界面毫无变化，只能重启
             策略才生效（`set_chase` 当年就是只改了 worker 属性，白改一场）
        """
        import inspect

        from ttrade import strategy_store as SS
        from ttrade.strategy_runner import PortfolioRunner

        # ---- ① 默认值 / ② 钳位
        assert SS.MonitorConfig().refresh_sec == 5, "默认不是 5 秒"
        # 0 / None 当「没设过」⇒ 退回默认 5（`cfg.refresh_sec or 5`），
        # 负数 / 超上限才走钳位
        cases = [(0, 5), (None, 5), (-5, 1), (1, 1), (5, 5), (60, 60), (999, 60),
                 ("7", 7)]
        for raw, want in cases:
            got = SS.MonitorConfig.from_dict({"refresh_sec": raw}).refresh_sec
            assert got == want, f"周期 {raw!r} 应该钳到 {want}，实际 {got}"
        # 多出来的键不许炸（老版本写过的字段留在文件里很正常）
        assert SS.MonitorConfig.from_dict(
            {"refresh_sec": 9, "what": 1}).refresh_sec == 9

        # 文件坏掉 → 退回默认，而不是让引擎起不来
        old_load = SS._load
        try:
            for junk in ({"refresh_sec": "abc"}, {}, None, {"refresh_sec": None}):
                SS._load = lambda _n, _j=junk: _j
                assert SS.load_monitor().refresh_sec == 5, \
                    f"monitor.json 里是 {junk!r} 时没退回默认 5 秒"
        finally:
            SS._load = old_load

        # ---- ③ 改完立刻生效：setter 必须同换 `_acct_sec`
        class _Feed:
            generation = 1

            def __init__(self):
                self.waits = 0

            def wait(self, timeout=0.0):
                self.waits += 1
                return True

        class _Api:
            def __init__(self):
                self.calls = 0

            def refresh_account(self):
                self.calls += 1

        r = PortfolioRunner.__new__(PortfolioRunner)
        r.feed, r._feed_gen = _Feed(), 1
        r._klines, r._quotes = {}, {}
        r._log = lambda _m: None
        r.last_error = ""
        r.api = _Api()
        r._acct_at = 0.0
        r.monitor = SS.MonitorConfig(refresh_sec=2)
        assert r._acct_sec == 2.0, (
            f"下发了新的重读周期，节流用的 `_acct_sec` 没跟着变（还是 {r._acct_sec}）"
            " —— 用户点完「确定」什么都不会改")
        # 节流真的按 2 秒走：连推两拍只查一次，过了 2 秒才再查
        r.wait(0.01)
        r.wait(0.01)
        assert r.api.calls == 1, f"2 秒的周期里连查了 {r.api.calls} 次"
        r._acct_at -= 3.0
        r.wait(0.01)
        assert r.api.calls == 2, "过了周期还不去查，等于设置被无视"
        # 越界的值也钳住（界面上手输 / 老文件都可能超）
        r.monitor = SS.MonitorConfig(refresh_sec=-3)
        assert r._acct_sec == 1.0, f"负数没钳到 1：{r._acct_sec}"
        r.monitor = SS.MonitorConfig(refresh_sec=0)
        assert r._acct_sec == 5.0, f"0（= 没设过）没退回 5：{r._acct_sec}"

        # ---- ④ 界面这条链：按钮 → set_monitor → 每条 runner
        from gui.portfolio_page import PortfolioController, RunnerWorker

        ctl = PortfolioController.__new__(PortfolioController)
        ctl._workers = {"甲": object(), "乙": object()}
        got: list[tuple[str, float]] = []

        class _R:
            def __init__(self, key):
                self._key = key

            @property
            def monitor(self):
                return None

            @monitor.setter
            def monitor(self, cfg):
                got.append((self._key, float(cfg.refresh_sec)))

        recs = {k: _R(k) for k in ctl._workers}
        ctl._submit = lambda key, fn: fn(recs[key])
        ctl.set_monitor(SS.MonitorConfig(refresh_sec=8))
        assert sorted(got) == [("乙", 8.0), ("甲", 8.0)], \
            f"「监控设置」没下发到每条在跑的 runner：{got}"
        # 起新线程时也要带上这份配置，否则下次启动又退回文件里的（而不是界面上的）
        src = inspect.getsource(RunnerWorker.run)
        assert "PortfolioRunner(self.spec, self.chase, self.monitor)" in src, \
            "新起的 runner 没把监控周期传进去"
        src_c = inspect.getsource(PortfolioController.start)
        assert "monitor=None" in src_c and "RunnerWorker(key, spec, cfgs, chase, monitor)" in src_c, \
            "`PortfolioController.start` 没把 monitor 透给 worker"

        return ("监控周期默认 5 秒、钳在 1~60（0/空当「没设过」⇒ 退回 5，负数→1，"
                "999→60，字符串数字能转）、文件坏掉退回默认；下发后 `_acct_sec` "
                "当场换掉，节流真的按新周期走（2 秒内只查一拍）；"
                "按钮 → `set_monitor` → 每条在跑的 runner，新起的 runner 也带着这份配置")

    check("监控设置：账户重读周期（默认 5 秒，改完立刻生效）", _monitor_cfg)

    def _window_unit():
        """标准化窗口按交易日 + 老存档按根 + 预测期数放到 100。"""
        import inspect

        import pandas as pd

        from ttrade import factor_strategy as FS
        from ttrade import replay as RP

        # ---- ① 交易日 → 根数，按**这份数据自己的**密度
        idx_d = pd.date_range("2024-01-01", periods=400, freq="B")
        idx_h = pd.date_range("2024-01-01", periods=400, freq="h")
        assert FS.bars_per_day(idx_d) == 1.0, "日线的密度不是 1"
        bpd_h = FS.bars_per_day(idx_h)
        assert bpd_h > 1.0, f"小时线的密度算成 {bpd_h}，等于没折"
        assert FS.window_bars(252, 1.0) == 252, "日线 252 天应当还是 252 根"
        assert FS.window_bars(252, 7.0) == 1764, "252 天 × 每天 7 根 ≠ 1764"
        # 下限 20 根：0 / 负数不能让 `rolling(0)` 抛异常
        assert FS.window_bars(0, 1.0) == 20 and FS.window_bars(-5, 1.0) == 20

        c = FS.bind_window(FS.FactorConfig(window=252), idx_h)
        assert c.window == FS.window_bars(252, bpd_h), \
            f"小时线上 252 天没折成根数（实际 {c.window}）"
        assert c.window > 252, "小时线上窗口还是 252 根 ⇒ 折算是假的"
        assert c._window_resolved is True

        # 幂等：嵌套调用（build_score → factor_matrix）不许换两遍
        assert FS.bind_window(c, idx_h).window == c.window, "折了两遍，窗口翻倍"
        assert FS.bind_window(c.normalized(), idx_h).window == c.window, \
            "过了 `normalized()` 又折了一遍"
        # 日线不动：252 天 = 252 根，老基线（所有历史 IC 锚点）不许移位
        assert FS.bind_window(FS.FactorConfig(window=252), idx_d).window == 252

        # ---- ② 老存档按根：window 原样，一个数字都不动
        old = RP.factor_config({"window": 1764, "entry_z": 1.0})
        assert old.window_unit == "bar", \
            f"老配置（没有 window_unit）被当成 {old.window_unit} 了"
        assert FS.bind_window(old, idx_h).window == 1764, (
            "老策略的 1764 根被按交易日折了 —— 1h 上会变成七年窗口、因子整条全空")
        new = RP.factor_config({"window": 252, "window_unit": "day"})
        assert new.window_unit == "day"
        assert FS.bind_window(new, idx_h).window == FS.window_bars(252, bpd_h)
        # 两种口径的窗口下限各不相同，谁都不许被另一条顶掉
        assert FS.FactorConfig(window=30, window_unit="bar").normalized().window == 60
        assert FS.FactorConfig(window=5).normalized().window == 20
        # 单位写错一个词 ⇒ 归一到交易日，不许静默按根跑
        assert FS.FactorConfig(window_unit="days").normalized().window_unit == "day"

        # ---- ③ 预测期数上限 100
        assert FS.FactorConfig(horizon=100).normalized().horizon == 100, \
            "预测期数被钳回去了"
        from gui import app as GA

        _lb, _lo, _hi, _st, _df = GA.FACTOR_PARAMS["horizon"]
        assert _hi == 100 and _df == 5, f"预测期数滑杆上限还是 {_hi}"
        _lb, _lo, _hi, _st, _df = GA.FACTOR_PARAMS["window"]
        assert "交易日" in _lb and (_lo, _hi, _df) == (20, 1000, 252), \
            f"窗口滑杆没改成交易日口径：{_lb} {_lo}~{_hi} 默认 {_df}"

        # 模型摘要必须带单位：老 252 根和新 252 天不是同一个窗口
        from ttrade import ml_artifact as MA

        assert MA._cfg_digest(FS.FactorConfig(window=252)).get("window_unit") == "day"
        assert MA._cfg_digest(
            FS.FactorConfig(window=252, window_unit="bar")).get("window_unit") == "bar"

        # ---- ④ 界面：标签跟着单位变，保存时把单位写回去
        class _Lbl:
            def __init__(self):
                self.text = ""
                self.tip = ""

            def setText(self, t):
                self.text = t

            def setToolTip(self, t):
                self.tip = t

        class _Sld:
            def __init__(self):
                self._label = _Lbl()
                self._v = 252

            def maximum(self):
                return 1000

            def setMaximum(self, v):
                pass

            def setValue(self, v):
                self._v = int(v)

            def value(self):
                return self._v

        stub = GA.MainWindow.__new__(GA.MainWindow)
        stub.fparam_widgets = {"window": _Sld()}
        stub.fparam_scales = {"window": 1}
        # 这三处 `_apply_factor_cfg` 会直接读（不是 `hasattr` 兜的），给 None 就跳过；
        # 另外两处「同步」在这个精简桩上没有意义，挡掉
        stub.cb_fmodel = stub.cb_forient = stub.cb_filter_when = None
        stub._sync_trigger_src = lambda: None
        stub._sync_exit_summary = lambda: None

        stub._apply_factor_cfg({"window": 1764})
        assert stub._fwin_unit == "bar", "老配置（无 window_unit）没认成「根」"
        assert "根" in stub.fparam_widgets["window"]._label.text, \
            f"老口径的标签没说实话：{stub.fparam_widgets['window']._label.text}"
        assert stub.fparam_widgets["window"]._v == 1764, "载入老配置把窗口改掉了"

        stub._apply_factor_cfg({"window": 252, "window_unit": "day"})
        assert stub._fwin_unit == "day"
        assert "交易日" in stub.fparam_widgets["window"]._label.text, \
            "新口径的标签没写「交易日」"
        # 保存时必须把当前单位写回去，否则载入一次就悄悄换成另一种含义
        assert "window_unit=" in inspect.getsource(GA.MainWindow._factor_cfg), \
            "`_factor_cfg()` 没把 window_unit 写进配置"
        # 重建参数面板后标签要重新同步（否则载入老策略后切一下模式就变回「交易日」）
        assert "_sync_window_unit_label()" in inspect.getsource(
            GA.MainWindow._rebuild_fparams), "重建参数面板后没同步窗口单位标签"

        # ---- ⑤ 截面资金分配：多空合在一起分
        from ttrade.cross import CrossConfig, _hedge_weights

        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        cols = ["A_SHFE", "B_SHFE", "C_SHFE", "D_DCE", "E_DCE", "F_CZCE"]
        # 一次调仓、强弱分明：多头 A/B，空头 F
        score = pd.DataFrame([[3.0, 2.0, 1.0, 0.0, -0.5, -1.0]] * 3,
                             index=idx, columns=cols)
        cfg = CrossConfig(top_pct=40.0, bot_pct=20.0, weight="equal")
        w = _hedge_weights(score, cfg, step=1)
        row = w.iloc[0]
        longs, shorts = row[row > 0], row[row < 0]
        assert len(longs) == 2 and len(shorts) == 1, \
            f"腿数不对：多 {len(longs)} 空 {len(shorts)}"
        assert abs(row.abs().sum() - 1.0) < 1e-9, \
            f"合并后 Σ|w| 应当还是 1（×杠杆），实际 {row.abs().sum()}"
        # 🔴 关键：两边**不必等值**（老口径是多头合计 0.5 / 空头合计 −0.5）
        assert abs(longs.sum() - 2.0 / 3.0) < 1e-9, f"多头合计 {longs.sum()}"
        assert abs(shorts.sum() + 1.0 / 3.0) < 1e-9, f"空头合计 {shorts.sum()}"
        assert abs(longs.sum() + shorts.sum()) > 0.1, "净敞口又被钉回 0 了"

        # 按分数加权：w ∝ |分数|（3:2:1 ⇒ 0.5 / 1/3 / −1/6）
        w2 = _hedge_weights(score, CrossConfig(top_pct=40.0, bot_pct=20.0,
                                              weight="score"), step=1).iloc[0]
        assert abs(w2["A_SHFE"] - 0.5) < 1e-9 and abs(w2["F_CZCE"] + 1.0 / 6.0) < 1e-9, \
            f"按分数分钱不对：{dict(w2.round(4))}"
        assert abs(w2.abs().sum() - 1.0) < 1e-9

        # 分档：按 |分数| 排**同一个队列**，空头最弱那只进第一档
        # （3 个选中 / 3 档 ⇒ 每档一只，份数 4:2:1）
        w3 = _hedge_weights(score, CrossConfig(top_pct=40.0, bot_pct=20.0,
                                              weight="tier", tiers=3),
                            step=1).iloc[0]
        assert abs(w3.abs().sum() - 1.0) < 1e-9, "分档之后 Σ|w| 不等于 1"
        assert abs(w3["A_SHFE"]) > abs(w3["B_SHFE"]) > abs(w3["F_CZCE"]), \
            f"分档没按 |分数| 排（应当是 A > B > F）：{dict(w3.round(4))}"
        assert abs(abs(w3["A_SHFE"]) - 4.0 / 7.0) < 1e-9, \
            f"第一档没拿 4/7：{abs(w3['A_SHFE']):.4f}"
        # 风险平价：波动越小给越多（这里只验 Σ|w| 和方向）
        vol = pd.DataFrame([[0.1, 0.2, 0.2, 0.2, 0.2, 0.4]] * 3,
                           index=idx, columns=cols)
        w4 = _hedge_weights(score, CrossConfig(top_pct=40.0, bot_pct=20.0,
                                               weight="risk"), vol=vol, step=1).iloc[0]
        assert abs(w4.abs().sum() - 1.0) < 1e-9
        assert w4["A_SHFE"] > w4["B_SHFE"] > 0 > w4["F_CZCE"], \
            f"风险平价没按 1/σ 分：{dict(w4.round(4))}"

        # 界面上不许再写「腿内」
        hint_src = inspect.getsource(GA.MainWindow._sync_cross_controls)
        for fn in (GA.MainWindow._sync_cross_controls, GA.MainWindow._cross_summary):
            assert "腿内" not in inspect.getsource(fn), \
                f"{fn.__name__} 里还留着「腿内」的说法"
        assert "合在一起" in hint_src, "对冲模式的提示没写「多空合在一起分钱」"

        return ("窗口=交易日（日线 252→252 根、1h 252→按密度折，幂等不翻倍）；"
                "老存档 window_unit 缺失 ⇒ 按根、1764 原样不动、下限 60 根；"
                "预测期数上限 100、窗口滑杆 20~1000 默认 252；"
                "标签按单位写「交易日 / 根・旧」且重建后仍同步；"
                "截面 Σ|w|=1 但多空各自合计不必相等（2/3 vs −1/3），"
                "等权/分数/分档|分数|/风险平价四种都在合并队列上分")

    check("标准化窗口按交易日（老存档按根）+ 预测期数 100 + 资金分配合并", _window_unit)

    def _cross_two_periods():
        """截面的第一步 / 第二步各有自己的因子周期，互不影响。"""
        import inspect

        import pandas as pd

        from ttrade import cross as CR
        from ttrade import strategy_store as SS

        # ---- ① 两个字段都在、都落盘
        names = {f.name for f in __import__("dataclasses").fields(SS.StrategyConfig)}
        assert "cross_step1_period" in names and "cross_step2_period" in names, \
            f"策略库里少了因子周期字段：{sorted(n for n in names if 'cross_step' in n)}"
        c = SS.StrategyConfig(cross_step1_period="1d", cross_step2_period="1h")
        d = __import__("dataclasses").asdict(c)
        assert d["cross_step1_period"] == "1d" and d["cross_step2_period"] == "1h"
        # 老配置（没有这个键）读回来是空串 = 跟随策略周期，和原来一致
        assert SS.StrategyConfig().cross_step2_period == ""

        # ---- ② 引擎：排名到底在哪份行情上算
        #
        # 用 spy 拦在 `score_panel` 上：它拿到的那份 `frames` 就是「排名用的行情」。
        # 直接跑完整引擎要几十个真实品种，这里只验「拿的是哪一份」
        idx_h = pd.date_range("2024-01-01 09:00", periods=5, freq="h")
        idx_d = pd.date_range("2024-01-01", periods=3, freq="D")
        frames = {f"S{i}_SHFE": pd.DataFrame({"close": [1.0, 2, 3, 4, 5]}, index=idx_h)
                  for i in range(3)}
        daily = {f"S{i}_SHFE": pd.DataFrame({"close": [1.0, 2, 3]}, index=idx_d)
                 for i in range(3)}
        bt = __import__("ttrade.backtest", fromlist=["BacktestConfig"]).BacktestConfig(
            initial_cash=1_000_000.0)
        seen = []
        orig = CR.score_panel

        def spy(fr, keys, *a, **kw):
            seen.append(pd.DatetimeIndex(next(iter(fr.values())).index))
            raise RuntimeError("STOP-HERE")

        CR.score_panel = spy
        try:
            cfg = CR.CrossConfig(mode="hedge", factor_mode="multi",
                                 factor_keys=["mom_20"], period="1h",
                                 step2_period="1d")
            out = CR.run(frames, cfg, bt, step2_frames=daily)
            assert "STOP-HERE" in str(out.error), out.error
            assert len(seen) == 1
            assert seen[0].equals(idx_d), (
                "配了第二步周期（日线），排名还是拿策略周期（小时）在算")

            # 只配第一步、第二步跟随策略周期 ⇒ 排名用**策略周期**，
            # 不能像以前那样跟着第一步走
            seen.clear()
            cfg1 = CR.CrossConfig(mode="hedge", factor_mode="multi",
                                  factor_keys=["mom_20"], period="1h",
                                  step1_period="1d")
            CR.run(frames, cfg1, bt, step1_frames=daily)
            assert len(seen) == 1 and seen[0].equals(idx_h), (
                "第一步换了周期，第二步的排名被带着一起换了（两条腿又共用一个 src）")
        finally:
            CR.score_panel = orig

        # ---- ③ 界面：右边那个下拉真的接上了
        src = inspect.getsource(__import__("gui.app", fromlist=["MainWindow"]).MainWindow._run_cross)
        assert "step2_frames" in src and "step2_period=p2" in src, \
            "`_run_cross` 没把第二步周期传给引擎"
        src2 = inspect.getsource(
            __import__("gui.app", fromlist=["MainWindow"]).MainWindow._build_cross_panels)
        assert "cb_step2_period" in src2, "第二步面板里没有「第二步因子周期」下拉"

        return ("第一步（筛池）/ 第二步（排名）各有独立因子周期：配了第二步周期就"
                "拿那份行情排名，只配第一步时排名仍按策略周期（不再跟着第一步走）；"
                "`cross_step2_period` 存得进策略库、老配置落到「跟随策略周期」")

    check("截面两步各有独立的因子周期（第一步筛池 / 第二步排名）", _cross_two_periods)

    def _hl_point():
        """HL point 复现：新因子注册齐全、两种资金分配范围、策略库往返。"""
        import dataclasses
        import inspect

        import numpy as np
        import pandas as pd

        from ttrade import factors as FF
        from ttrade import strategy_store as SS
        from ttrade.cross import CrossConfig, _hedge_weights

        # ---- ① 42 个新因子都注册了，而且算得出值（不是整列空）
        new_keys = [k for k, s in FF.FACTORS.items()
                    if s.category == "MACD 风洞"] + [
            "mom_5", "mom_accel", "slope_20", "slope_60", "ma_align",
            "pos_60", "new_high_10", "bb_width_20", "bb_width_60", "dev_60",
            "high_ma_gap_20", "vol_5", "vol_surge_5", "vol_shrink",
            "atr_ratio", "range_ratio", "vol_price_diverge",
            "contraction_score", "oi_change_10", "oi_vol_ratio",
            "oi_price_corr_5", "intraday_strength", "gap_fill_ratio",
            "spot_mom_5", "spot_mom_20", "warehouse_chg_20", "inventory_yoy",
            "profit_pct", "deliv_pct",
        ]
        missing = [k for k in new_keys if k not in FF.FACTORS]
        assert not missing, f"HL point 的因子没注册：{missing}"
        assert "MACD 风洞" in FF.CATEGORIES, "新类别没进 CATEGORIES（界面上选不到）"

        rng = pd.date_range("2021-01-04", periods=900, freq="B")
        rnd = np.random.RandomState(3)
        c = 3000 * np.exp(np.cumsum(rnd.normal(0.0002, 0.012, len(rng))))
        o = c * (1 + rnd.normal(0, 0.002, len(rng)))
        df = pd.DataFrame(
            {"open": o,
             "high": np.maximum(o, c) * (1 + np.abs(rnd.normal(0, 0.004, len(rng)))),
             "low": np.minimum(o, c) * (1 - np.abs(rnd.normal(0, 0.004, len(rng)))),
             "close": c,
             "volume": np.abs(rnd.normal(1e5, 4e4, len(rng))),
             "open_interest": np.abs(rnd.normal(5e5, 5e4, len(rng)))},
            index=rng)
        df.attrs["symbol"] = "RB"
        dead = []
        for k in new_keys:
            s = FF.raw(df, k).replace([np.inf, -np.inf], np.nan)
            if int(s.notna().sum()) < 100:
                dead.append((k, int(s.notna().sum())))
        assert not dead, f"这些新因子在 900 根日线上几乎全空：{dead}"

        # 🔴 mom_accel 与 mom_diverge 数学恒等（(a−1)−(b−1) ≡ a−b），
        # 原项目当两列用，这边只注册一列 —— 注册两列等于给模型喂重复特征
        from ttrade import factors_hl as FH

        assert "mom_diverge" not in FF.FACTORS, "mom_diverge 不该注册（和 mom_accel 恒等）"
        a = FF.raw(df, "mom_accel")
        b = FH.f_mom_diverge(df)
        bad = (a - b).abs()
        bad = bad[bad.notna()]
        assert len(bad) and float(bad.max()) < 1e-12, \
            f"mom_accel 和 mom_diverge 不再恒等了（最大差 {float(bad.max())}）—— 可以当两列用了"

        # ---- ② 资金分配范围：legs = 多空各半仓（净敞口恒 0）
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        cols = ["A_SHFE", "B_SHFE", "C_SHFE", "D_DCE", "E_DCE", "F_CZCE"]
        score = pd.DataFrame([[3.0, 2.0, 1.0, -1.0, -2.0, -3.0]] * 3,
                             index=idx, columns=cols)
        flat = pd.DataFrame(0.01, index=idx, columns=cols)
        # 两边波动完全一样时，两种口径必须给出同一组权重
        w_p = _hedge_weights(score, CrossConfig(top_pct=50.0, bot_pct=50.0,
                                               weight="risk",
                                               weight_scope="pooled"),
                             vol=flat, step=1).iloc[0]
        w_l = _hedge_weights(score, CrossConfig(top_pct=50.0, bot_pct=50.0,
                                               weight="risk",
                                               weight_scope="legs"),
                             vol=flat, step=1).iloc[0]
        assert (w_p - w_l).abs().max() < 1e-12, "同波动下两种口径不该有差别"
        assert abs(w_l.abs().sum() - 1.0) < 1e-9

        # 多头低波动、空头高波动：pooled 会把钱全压到多头（净敞口跑偏），
        # legs 仍然两边各半。这就是复现 HL point 必须切 legs 的原因
        skew = pd.DataFrame({"A_SHFE": 0.005, "B_SHFE": 0.005, "C_SHFE": 0.005,
                             "D_DCE": 0.05, "E_DCE": 0.05, "F_CZCE": 0.05},
                            index=idx, dtype=float)
        r_p = _hedge_weights(score, CrossConfig(top_pct=50.0, bot_pct=50.0,
                                               weight="risk",
                                               weight_scope="pooled"),
                            vol=skew, step=1).iloc[0]
        r_l = _hedge_weights(score, CrossConfig(top_pct=50.0, bot_pct=50.0,
                                               weight="risk",
                                               weight_scope="legs"),
                            vol=skew, step=1).iloc[0]
        assert abs(r_p.sum()) > 0.5, \
            f"合并口径在波动不对称时应当明显偏多头，实际净敞口 {r_p.sum():.3f}"
        assert abs(r_l.sum()) < 1e-12, f"legs 口径净敞口必须是 0，实际 {r_l.sum():.3f}"
        assert abs(r_l.abs().sum() - 1.0) < 1e-9
        # 只有一边有仓：不应该凭空变成半仓
        one = _hedge_weights(score, CrossConfig(top_pct=50.0, bot_pct=0.0,
                                              weight="equal",
                                              weight_scope="legs"),
                             vol=flat, step=1).iloc[0]
        assert abs(one.abs().sum() - 1.0) < 1e-9, \
            f"legs 口径下「只做多」被砍成半仓了：{one.abs().sum():.3f}"

        # ---- ③ 落盘往返 + 默认值
        names = {f.name for f in dataclasses.fields(SS.StrategyConfig)}
        assert "cross_weight_scope" in names, "策略库少了 cross_weight_scope"
        assert SS.StrategyConfig().cross_weight_scope == "pooled", \
            "默认口径必须是 pooled（2026-11-23 之后的行为），不能是 legs"
        rec = SS.StrategyConfig.from_dict({"cross_weight_scope": "legs"})
        assert rec.cross_weight_scope == "legs"
        rec2 = SS.StrategyConfig.from_dict({"cross_weight_scope": "随便写的"})
        assert rec2.cross_weight_scope == "pooled", "手改 JSON 写错应当落到默认口径"
        # 摘要里得写出来，否则两条曲线差在哪看不出来。
        # 注意 `params_text()` 按 `family` 分支，不写 family="cross" 会走规则那条
        assert "多空各半仓" in SS.StrategyConfig(
            family="cross", cross_weight_scope="legs").params_text(), \
            "legs 口径没进策略摘要"
        assert "多空各半仓" not in SS.StrategyConfig(family="cross").params_text(), \
            "默认口径不该在摘要里多一行噪音"
        # replay 必须把口径透传下去（漏了就悄悄按 pooled 回测）
        from ttrade import replay as _RP

        assert "weight_scope=getattr(cfg" in inspect.getsource(_RP.run_cross), \
            "replay.run_cross 没把 cross_weight_scope 透传给 CrossConfig"

        return ("HL point 的 42 个因子全部注册且非空（MACD 风洞 13 个自成一类）；"
                "mom_accel ≡ mom_diverge 只注册一列；"
                "资金分配范围 legs = 多空各半仓（净敞口恒 0、只做多不砍半仓）、"
                "pooled 在波动不对称时净敞口会跑偏 —— 复现严格中性策略必须切 legs")

    check("HL point 复现：新因子 + 资金分配范围（多空各半仓 / 合并）", _hl_point)

    # 跑完就把进度文件收掉：留着会让下一轮的诊断以为"上次崩过"。
    # 崩溃时走不到这里，文件正好留下最后一项的名字。
    if _prog is not None:
        try:
            _prog.close()
            os.remove(os.path.join(log_dir(), "selftest_progress.txt"))
        except Exception:  # noqa: BLE001
            pass
    # 关掉增量日志：`run()` 之后还会把 `lines` 整份重写一遍（内容相同），
    # 同一个文件不能同时开着两个写句柄
    if _log_fh is not None:
        try:
            _log_fh.close()
        except Exception:  # noqa: BLE001
            pass

    return ok, lines


def run() -> int:
    """执行自检并返回退出码。"""
    from ttrade.paths import log_dir

    # 自检**不许连真行情**：`MainWindow` 一建就会调 `MarketFeed.warm_up()`，
    # 那会拿用户的账号去登录一次天勤 —— 离线跑必红、联网跑又要等几秒，还会让
    #「构建主窗口」那一项随网络时红时绿。开关只有 `market_feed.AUTO_CONNECT` 一处。
    from ttrade import market_feed as _mf

    _mf.AUTO_CONNECT = False

    # 🔴 段错误时进程直接死，`lines` 来不及落盘，`try/except Exception` 也抓不到。
    # faulthandler 能把崩溃瞬间的栈写下来，但它**不能常开**：它会把 Qt / 原生库
    # 内部那些「先探一下地址再恢复」的段错误变成致命错误。实测 macOS x64 上
    # 开了它自检反而从 exit 0 变成 exit 139（同一份代码、同一个包），关掉就恢复。
    # 所以做成开关，默认关；要查崩溃栈时显式打开：
    #     CHANQUANT_SELFTEST_FAULTHANDLER=1 ChanQuant --selftest
    # 日常定位「死在哪一项」靠下面的 selftest_progress.txt —— 那个只是文件写，
    # 不改任何信号行为，永远不会自己把进程弄崩。
    crash_path = os.path.join(log_dir(), "selftest_crash.txt")
    _crash_fh = None
    if os.environ.get("CHANQUANT_SELFTEST_FAULTHANDLER") == "1":
        try:
            import faulthandler

            _crash_fh = open(crash_path, "w", encoding="utf-8")
            faulthandler.enable(file=_crash_fh, all_threads=True)
        except Exception:  # noqa: BLE001 - 拿不到就不启用，不影响自检本身
            _crash_fh = None

    # 🔴 自检会把**真实主窗口**搭起来，里面有切页签、加策略、改本金这些模拟
    # 动作。它们在生产里本来就该落盘，但跑自检时绝不能改用户的组合 —— 实测
    # 跑一轮就把 `portfolio.json` 里一条策略删没了、二十几条本金被重排。
    #
    # 所以整个存储目录重定向到临时目录（先把现有 json 复制一份当快照）。
    # 日志不在这个目录里，`selftest.log` 的位置不变，CI 仍然找得到。
    tmp = ""
    try:
        import tempfile

        from ttrade import strategy_store as _SStore

        tmp = tempfile.mkdtemp(prefix="cq_selftest_")
        _SStore.set_store_dir(tmp, seed=True)
    except Exception:  # noqa: BLE001 - 重定向失败也要把自检跑完
        tmp = ""

    try:
        ok, lines = _run()
    except Exception:  # noqa: BLE001 - 连路径都取不到时的兜底
        ok = False
        lines = ["自检初始化失败：", traceback.format_exc()]
    finally:
        if tmp:
            try:
                from ttrade import strategy_store as _SStore

                _SStore.set_store_dir("")       # 恢复默认目录
            except Exception:  # noqa: BLE001
                pass
            # 全绿就把临时目录删掉；有红的话留着，好让人去里面翻现场
            if ok:
                try:
                    import shutil

                    shutil.rmtree(tmp, ignore_errors=True)
                except Exception:  # noqa: BLE001
                    pass

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
    # 没崩就把空的崩溃文件删掉：留着会让下一轮的诊断误以为"上次崩过"
    if _crash_fh is not None:
        try:
            import faulthandler

            faulthandler.disable()
            _crash_fh.close()
            if ok:
                os.remove(crash_path)
        except Exception:  # noqa: BLE001
            pass
    return 0 if ok else 1
