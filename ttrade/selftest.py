"""打包后的自检：不打开窗口，验证资源、数据、回测、绘图、Qt 插件是否都正常。

    TrendDesk.exe --selftest     # Windows
    TrendDesk.app/Contents/MacOS/TrendDesk --selftest   # macOS

打包成窗口程序后没有控制台，`print` 看不到，所以结果同时写入
`<用户目录>/logs/selftest.log`，并用退出码表示成败（0 成功 / 1 失败）。
CI 里跑一遍这个就能确认「打出来的包真的能用」，而不用去点界面。
"""

from __future__ import annotations

import os
import sys
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

    def _window():
        """真正把主窗口搭起来，验证 UI 代码在打包后也能跑通。"""
        from PySide6.QtWidgets import QApplication

        from gui.app import STYLE_SHEET, MainWindow, setup_light_palette

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
        # 启动时首次回测是延后 220ms 排队的（先让窗口画出来再算）。自检里没有
        # 真的事件循环，这里直接把它跑完 —— 否则它会挑在后面测滚轮 / 组合的
        # 时候插进来重画一次，窗口被重置，滚轮那项就时好时坏。
        if getattr(win, "_pending", False):
            win._deferred_run()
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
        has_upd = (
            dp is not None
            and win.pages.count() == 5
            and len(win._page_btns) == 5
            and win.pages.widget(4) is dp
            and dp.market == win.market
            and not dp.busy()
            and dp.cb_upd_range.count() == len(_upd.RANGES)
            and dp.cb_source.count() == 2
            # 市场单选归数据页自己管，启动时跟研究页一致
            and len(dp._market_btns) == 2
            and dp._market_btns[dp.market].isChecked()
            # 周期 / 回溯区间 / 真实合约只属于天勤那一路
            and dp.cb_period.isHidden() == tq_vis
            and dp.cb_tq_range.isHidden() == tq_vis
            and dp.chk_tq_contracts.isHidden() == tq_vis
            and dp.chk_upd_contracts.isHidden() == (is_us or src_tq)
            and dp.chk_upd_options.isHidden() == (is_us or src_tq)
            and not dp.chk_upd_contracts.isChecked()
            and not dp.chk_upd_options.isChecked()
            # 每日定时补充：控件在、只对期货显示、默认不勾（不能让用户
            # 装完就被自动跑一次几十分钟的下载）
            and hasattr(dp, "chk_auto")
            and hasattr(dp, "time_auto")
            and hasattr(dp, "cb_auto_range")
            and dp.chk_auto.isHidden() == is_us
            and not dp.chk_auto.isChecked()
            and dp.cb_auto_range.count() == 2
        )
        # 交易Agent 页：删除按钮在没选中东西时必须是灰的，权限开关默认关。
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
        agent_ok = (
            ap is not None
            and hasattr(ap, "btn_del_factor")
            and hasattr(ap, "btn_del_strat")
            and hasattr(ap, "btn_manual")
            # 附件与运行输出也是函数里 import / 新建的控件，缺一个功能就没了
            and hasattr(ap, "btn_attach")
            and hasattr(ap, "out")
            and hasattr(ap, "btn_out_feed")
            and not ap.btn_del_factor.isEnabled()
            and not ap.btn_del_strat.isEnabled()
            and ap.cfg.allow_fs is False
            and fs_ok
            and manual_ok
            and out_ok
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

        # 因子面板：两个策略族、三种因子模式、47 个因子、四类模型都得在
        fams = {win.cb_family.itemData(i) for i in range(win.cb_family.count())}
        modes = {win.cb_fmode.itemData(i) for i in range(win.cb_fmode.count())}
        filters = {win.cb_filter_when.itemData(i) for i in range(win.cb_filter_when.count())}
        orients = {win.cb_forient.itemData(i) for i in range(win.cb_forient.count())}
        n_factors = win.cb_fsingle.count()
        n_models = win.cb_fmodel.count()
        n_rows_f = win.tbl_fmulti.rowCount()
        n_cols_f = win.tbl_fmulti.columnCount()
        factor_ok = (
            fams == {"rule", "factor"}
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
        )
        factor_info = f"{len(fams)} 族 / {len(modes)} 模式 / {n_factors} 因子 / {n_models} 模型"

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
                and tabs.count() == 3
                and pv.table.columnCount() == 2
                and pv.res is not None
                and "组合（2 条策略）" in text
            )
            combo_info = (f"{tabs.count()} 页 / {pv.table.columnCount()} 列"
                          if tabs is not None else "未建出视图")
        except Exception as exc:  # noqa: BLE001
            combo_info = f"{type(exc).__name__}: {exc}"

        # 回测区间：切片要真的生效。不生效的话 5 分钟十年照样从头算，
        # 「近 1 年」只是个显示开关，用户以为快了其实没快
        range_ok = False
        range_info = ""
        try:
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
        if not agent_ok:
            raise RuntimeError(
                "交易Agent 页异常：删除按钮初始状态不对，或升级权限的文件模块 / "
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
        if not combo_ok:
            raise RuntimeError(f"组合报告异常：{combo_info}")
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
        return (f"{n} 个标的，K 线/绩效/因子页已渲染，{k_info}，"
                f"{factor_info}，滚轮缩放 {wheel_info}，组合 {combo_info}，"
                f"回测区间 {range_info}，"
                f"数据管理页就绪，手册 {manual_info}，输出 {out_info}")

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
