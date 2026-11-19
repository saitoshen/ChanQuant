# -*- coding: utf-8 -*-
"""Q17：window=252 为什么夏普最高？①窗口内分位有多稳 ②加扰动后还剩多少"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import symbols_info, tqdata

PRIN = 1_000_000.0
_bases = {}

def make_noisy(sigma, seed):
    def f(df):
        k = id(df)
        if k not in _bases:
            _bases[k] = F.FACTORS["macro_usdcnh"].func(df)
        base = _bases[k]
        if sigma == 0:
            return base
        day = pd.DatetimeIndex(df.index).normalize().asi8
        uniq, inv = np.unique(day, return_inverse=True)
        rng = np.random.default_rng(seed)
        n = rng.normal(0.0, sigma, len(uniq))[inv]
        v = np.clip(base.to_numpy(dtype=float) + n, 0.0, 1.0)
        out = pd.Series(v, index=df.index)
        out[base.isna()] = np.nan
        return out
    return f

SIG = [0.0, 0.0065, 0.013]
SEEDS = [11, 23, 37, 41, 59]
for s in SIG:
    for sd in (SEEDS if s > 0 else [0]):
        k = "w2_%g_%d" % (s * 10000, sd)
        if k not in F.FACTORS:
            F.FACTORS[k] = F.FactorSpec(k, "扰动%.4f" % s, make_noisy(s, sd), "诊断", "-", formula="-")

def bt(df, key, si, sym, exch, win):
    cfg = FS.FactorConfig(window=win, method="z", entry_z=1.0, exit_z=0.3,
                          orient="long_low", horizon=12)
    out = FS.run_factor_strategy(df, "single", [key], cfg)
    c = BT.BacktestConfig(initial_cash=PRIN)
    c.sizing = "capital"; c.capital = PRIN * float(si["margin"])
    c.multiplier = float(si["mult"]); c.margin_rate = float(si["margin"])
    c.bars_per_year = BT.bars_per_year(out.df.index)
    costs.apply(df.pipe(lambda d: d) is not None and c, sym, exch) if False else costs.apply(c, sym, exch)
    r = BT.run_backtest(out.df, c)
    m = r.metrics; del out
    return float(m["sharpe"] or 0), int(m["n_trades"] or 0)

SYMS = [("AU","SHFE"),("AG","SHFE"),("IM","CFFEX")]
frames = {}
for sym, exch in SYMS:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    if not len(df): continue
    frames[sym] = df
    si = symbols_info.info(sym, exch)
    a = F.FACTORS["macro_usdcnh"].func(df)
    print("\n" + "=" * 74)
    print("%s 每天 %.0f 根" % (sym, float(df.groupby(df.index.normalize()).size().median())))
    for win in (252, 545, 1440):
        s = a.rolling(win, min_periods=max(60, win // 4)).std()
        print("  window %4d (%.1f 天): 分位窗口 std 中位 %.5f ⇒ 数据源差 0.0130 = %.1f 个 σ"
              % (win, win / float(df.groupby(df.index.normalize()).size().median()),
                 float(s.median()), 0.0130 / max(float(s.median()), 1e-9)))

print("\n" + "=" * 90)
print("加扰动后的夏普（5 个随机种子）")
for sym in frames:
    df = frames[sym]
    si = symbols_info.info(tqdata and __import__("ttrade.symbols_info", fromlist=["x"]).__name__, None) if False else None
for sym, exch in SYMS:
    df = frames[sym]
    si = symbols_info.info(sym, exch)
    print("\n  %s" % sym)
    print("    %-9s %8s %8s %8s %8s %8s" % ("window", "原值", "σ=0.0065", "σ=0.0130", "笔数", "σ13笔数"))
    for win in (252, 545, 1440):
        cells = []
        for s in SIG:
            vals, ns = [], []
            for sd in (SEEDS if s > 0 else [0]):
                k = "w2_%g_%d" % (s * 10000, sd)
                sh, n = bt(df, k, si, sym, exch, win)
                vals.append(sh); ns.append(n)
                gc.collect()
            cells.append((np.median(vals), float(np.min(vals)), float(np.max(vals)), int(np.median(ns))))
        b = cells[0][0]
        print("    %-9d %8.2f %8.2f %8.2f %8d %8d" %
              (win, b, cells[1][0], cells[2][0], cells[1][3], cells[2][3]))
        print("    %-9s %8s %8s %8s" % ("", "(min/max)", "%.2f/%.2f" % (cells[1][1], cells[1][2]),
                                        "%.2f/%.2f" % (cells[2][1], cells[2][2])))
