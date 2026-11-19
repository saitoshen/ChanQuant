# -*- coding: utf-8 -*-
"""Q11：给 macro_usdcnh 按**自然日**加微扰（修正上轮按 bar 加的错误）。

σ=0.013 就是 Duka 与 Tushare 的实测分位差。如果这个量级的扰动就能让夏普
在 ±1 之间乱跳，那 +1.18 不是 alpha，是参数化的产物。
"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import symbols_info, tqdata

PRIN = 1_000_000.0
_cache_base = {}

def make_noisy(sigma, seed):
    def f(df):
        key = id(df)
        if key not in _cache_base:
            _cache_base[key] = F.FACTORS["macro_usdcnh"].func(df)
        base = _cache_base[key]
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

SIGMAS = [0.0, 0.003, 0.0065, 0.013]
SEEDS = [11, 23, 37, 41, 59]
_keys = {}
for s in SIGMAS:
    for sd in (SEEDS if s > 0 else [0]):
        k = "rn_%g_%d" % (s * 10000, sd)
        _keys[(s, sd)] = k
        if k not in F.FACTORS:
            F.FACTORS[k] = F.FactorSpec(k, "扰动σ=%.4f s%d" % (s, sd),
                                        make_noisy(s, sd), "诊断", "-", formula="-")

def bt(df, key, si, sym, exch, win):
    cfg = FS.FactorConfig(window=win, method="z", entry_z=1.0, exit_z=0.3,
                          orient="long_low", horizon=12)
    out = FS.run_factor_strategy(df, "single", [key], cfg)
    c = BT.BacktestConfig(initial_cash=PRIN)
    c.sizing = "capital"; c.capital = PRIN * float(si["margin"])
    c.multiplier = float(si["mult"]); c.margin_rate = float(si["margin"])
    c.bars_per_year = BT.bars_per_year(out.df.index)
    costs.apply(c, sym, exch)
    r = BT.run_backtest(out.df, c)
    m = r.metrics; del out
    return float(m["sharpe"] or 0), int(m["n_trades"] or 0)

rows = []
for sym, exch in [("AU","SHFE"), ("IM","CFFEX")]:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    if not len(df): continue
    si = symbols_info.info(sym, exch)
    for win in (545, 2880):
        for s in SIGMAS:
            for sd in (SEEDS if s > 0 else [0]):
                sh, n = bt(df, _keys[(s, sd)], si, sym, exch, win)
                rows.append(dict(sym=sym, win=win, sigma=s, seed=sd, 夏普=sh, 笔数=n))
                gc.collect()
    del df; _cache_base.clear(); gc.collect()

d = pd.DataFrame(rows)
print("=" * 88)
print("按自然日加扰动后，同一个因子的夏普分布（5 个随机种子）")
for sym in d.sym.unique():
    for win in (545, 2880):
        print("\n  %s  window=%d 根（%.1f 天）" % (sym, win, win/288.0))
        print("    %-10s %8s %8s %8s %8s %8s" %
              ("σ", "原值", "min", "中位", "max", "笔数中位"))
        for s in SIGMAS:
            sub = d[(d.sym==sym)&(d.win==win)&(d.sigma==s)]
            base = sub[sub.seed==0]["夏普"]
            sh = sub[sub.seed>0]["夏普"] if s > 0 else sub["夏普"]
            b = "%.2f" % base.iloc[0] if len(base) else "-"
            print("    %-10.4f %8s %8.2f %8.2f %8.2f %8d" %
                  (s, b, sh.min(), sh.median(), sh.max(),
                   int(sub["笔数"].median())))
print("\n  σ=0.013 就是 Duka 与 Tushare 的实测分位差；σ=0.003 只有它的 1/4。")
