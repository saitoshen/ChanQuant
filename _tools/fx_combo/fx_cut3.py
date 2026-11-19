# -*- coding: utf-8 -*-
"""Q15：改完默认切分点后，fx5d 的实测夏普（与改之前的 A 口径对照）"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import factors_fx5 as X, symbols_info, tqdata

PRIN = 1_000_000.0

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
    m = r.metrics; eq = r.equity / PRIN - 1.0; del out
    return float(m["sharpe"] or 0), int(m["n_trades"] or 0), eq

rows, curves = [], {}
SYMS = [("AU","SHFE"),("AG","SHFE"),("CU","SHFE"),("SN","SHFE"),("IM","CFFEX")]
for sym, exch in SYMS:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    if not len(df): continue
    si = symbols_info.info(sym, exch)
    for win in (545, 2880):
        sh, n, eq = bt(df, "fx5d_usdcnh_pct", si, sym, exch, win)
        curves.setdefault(win, []).append(eq.rename(sym))
        rows.append(dict(sym=sym, win=win, 夏普=round(sh,2), 笔数=n))
        gc.collect()
    del df; gc.collect()

t = pd.DataFrame(rows)
for win in (545, 2880):
    s = t[t.win==win].set_index("sym")
    com = pd.concat(curves[win], axis=1).ffill().fillna(0.0)
    nav = 1.0 + com.mean(axis=1); ret = nav.pct_change().dropna()
    sh = float(ret.mean()/ret.std()*np.sqrt(252*48)) if ret.std() else np.nan
    print("\nwindow=%d 根（%.1f 天）" % (win, win/288.0))
    print(s.to_string())
    print("  组合夏普 %+5.2f  总收益 %+6.1f%%  回撤 %5.1f%%" %
          (sh, (nav.iloc[-1]-1)*100, float((nav/nav.cummax()-1).min()*100)))
