# -*- coding: utf-8 -*-
"""Q16：window 252 / 545 / 1440 / 2880 / 5760 扫一遍（新 06:00 切分的 fx5d vs macro）

天数按**品种实际每天 bar 数**折算（AU/AG 111 根、CU/SN 93 根、IM 48 根），
不能用 288（那是 24 小时连续市场的密度）。
"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import symbols_info, tqdata

PRIN = 1_000_000.0
WINS = [252, 545, 1440, 2880, 5760]
KEYS = [("macro_usdcnh", "Tushare"), ("fx5d_usdcnh_pct", "5m聚合")]

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
    m = r.metrics; sig = out.df["signal"].to_numpy(); eq = r.equity / PRIN - 1.0
    del out
    return float(m["sharpe"] or 0), int(m["n_trades"] or 0), eq, sig

rows, curves, bpd = [], {}, {}
SYMS = [("AU","SHFE"),("AG","SHFE"),("CU","SHFE"),("SN","SHFE"),("IM","CFFEX")]
for sym, exch in SYMS:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    if not len(df): continue
    si = symbols_info.info(sym, exch)
    bpd[sym] = float(df.groupby(df.index.normalize()).size().median())
    for win in WINS:
        sigs = {}
        for key, lab in KEYS:
            sh, n, eq, sig = bt(df, key, si, sym, exch, win)
            sigs[lab] = sig
            curves.setdefault((win, lab), []).append(eq.rename(sym))
            rows.append(dict(sym=sym, win=win, 因子=lab, 夏普=round(sh, 2), 笔数=n))
            gc.collect()
        rows.append(dict(sym=sym, win=win, 因子="信号差%",
                         夏普=round(float((sigs["Tushare"] != sigs["5m聚合"]).mean()*100), 1), 笔数=0))
    del df; gc.collect()

t = pd.DataFrame(rows)
print("每天 bar 数（中位）:", {k: int(v) for k, v in bpd.items()})
print("\n" + "=" * 92)
for win in WINS:
    s = t[(t.win == win) & (t.因子 != "信号差%")]
    dd = t[(t.win == win) & (t.因子 == "信号差%")].set_index("sym")["夏普"]
    print("\nwindow = %d 根（AU %.1f 天 / IM %.1f 天）" %
          (win, win / bpd.get("AU", 111), win / bpd.get("IM", 48)))
    pv = s.pivot(index="sym", columns="因子", values="夏普")
    pv["差距"] = (pv["Tushare"] - pv["5m聚合"]).round(2)
    pv["笔数_T"] = s[s.因子=="Tushare"].set_index("sym")["笔数"]
    pv["笔数_D"] = s[s.因子=="5m聚合"].set_index("sym")["笔数"]
    print(pv.to_string())
    print("  两因子信号不一致%%: %s" % ", ".join("%s %.0f" % (k, v) for k, v in dd.items()))
    for kidx, lab in enumerate(["Tushare", "5m聚合"]):
        com = pd.concat(curves[(win, lab)], axis=1).ffill().fillna(0.0)
        nav = 1.0 + com.mean(axis=1); ret = nav.pct_change().dropna()
        sh = float(ret.mean()/ret.std()*np.sqrt(252*10)) if ret.std() else np.nan
        print("   组合 %-8s 夏普 %+5.2f  总收益 %+7.1f%%  回撤 %5.1f%%" %
              (lab, sh, (nav.iloc[-1]-1)*100, float((nav/nav.cummax()-1).min()*100)))
