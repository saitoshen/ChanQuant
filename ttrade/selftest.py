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
            if r < 0.5:
                raise RuntimeError(f"{key} 仓单持仓比只有 {r:.0%} 有值")
            return f"{key} 品种标记 {sym}，仓单持仓比 {r:.0%} 有值"
        return "没有同时具备 EDB 仓单和日线 CSV 的品种（可选）"

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

    def _us_df():
        """自检里反复要用的一份行情，按需加载一次就缓存。"""
        nonlocal _cache
        if "us" not in _cache:
            from ttrade.eastmoney import load_local

            cands = sorted(f[:-4].replace("_", ".") for f in os.listdir(data_dir("us"))
                           if f.endswith(".csv") and not f.startswith("_"))
            _cache["us"] = load_local(cands[0], data_dir("us"))
        return _cache["us"]

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
        return f"规则 {base} 根持仓 → 过滤后 {kept} 根"

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

            def submit(self, key, df, cfg, keys, predictor=None):
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
        """每个周期都要能进实时。

        天勤日线的周期键是 `1d_tq`，而 `RT_SEC` 里只有源周期 `1d`。门禁要是按
        原键判断，日线策略一启用就报「实时不支持 1d_tq」直接停机 —— 界面上是
        一条红的「错误」，看不出是周期映射的问题。所以这里逐个周期真调一遍
        `enable()`，而不是只断言常量表包含关系。
        """
        from ttrade import strategy_runner as R
        from ttrade import strategy_store as SStore
        from ttrade import tqdata
        from ttrade.tqlive import AccountSpec

        spec = AccountSpec(name="__selftest__", kind="sim")
        run = R.PortfolioRunner(spec)
        bad = []
        for p in tqdata.PERIOD_ORDER:
            sid = f"__rt_{p}"
            run.add(SStore.StrategyConfig(
                sid=sid, name=f"自检 {p}", symbol="RB", exchange="SHFE",
                period=p, family="factor", factor_mode="single",
                factor_keys=["rev_5"], account=spec.name))
            run.enable(sid, True)
            st = run._slots[sid]
            if not st.running or st.error:
                bad.append(f"{p}：{st.error or '未启动'}")
        if bad:
            raise RuntimeError("；".join(bad))
        return f"{len(tqdata.PERIOD_ORDER)} 个周期全部可实时"

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
        factor_ok = (
            fams == {"rule", "factor", "cross"}
            and modes == {"single", "multi", "ml"}
            and filters == {"entry", "delay", "hold"}
            # 方向口径三档必须都在，否则「反转」只能相对自动定向，跨品种不可比
            and orients == {"auto", "long_high", "long_low"}
            and n_factors >= 47
            and n_models >= 4
            # 多因子表要 47 行 × 3 列（因子 / 用 / 反），少了「反」列就没法逐因子反转
            and n_rows_f >= 47
            and n_cols_f == 3
            and win.chk_frev is not None
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
        if not factor_ok:
            raise RuntimeError(f"因子面板控件缺失：{factor_info}")
        if not live_off:
            raise RuntimeError(
                "启动时自动接了实时（live 已在跑）：研究页不该连账户，"
                "账户连接只在「账户透视和交易」页"
            )
        # 因子分析页列出**全部**因子（算不出的显示「无数据」），所以行数应等于
        # 因子库大小。用 `n_factors` 而不是写死数字：加因子时这里不用改。
        if n_rows < n_factors:
            raise RuntimeError(f"因子分析页只有 {n_rows} 行，应为 {n_factors}")
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
            win.chk_sparse.setChecked(False)
            trig_info = ("「不用」两项都在（切市场后仍在），反转开关跟着触发源灰 / 亮，"
                         "规则触发参数能进配置且换规则不串")
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
                f"{side_info}，触发 {trig_info}，"
                f"回测区间 {range_info}，实时开关已复位，"
                f"数据管理页就绪，手册 {manual_info}，输出 {out_info}，"
                f"对话渲染 {chat_info}，能力包 {skill_info}")

    check("构建主窗口（端到端）", _window)

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
