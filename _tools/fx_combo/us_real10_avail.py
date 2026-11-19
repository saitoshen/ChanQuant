# -*- coding: utf-8 -*-
"""us_real10 加 avail_hour=8 前后：因子值差多少、策略差多少"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import factors_macro as FM, symbols_info, tqdata, macro as M

PRIN = 1_000_000.0
_pct = None

def get_pct():
    global _pct
    if _pct is None:
        s = M.series("us_real10").astype(float).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        _pct = s.expanding(min_periods=FM.MIN_OBS).rank(pct=True)
    return _pct

def mk(h):
    def f(df):
        return FM._asof(get_pct(), df.index, lag_days=1, avail_hour=h)
    return f

for h, k in ((0, "ur_h0"), (8, "ur_h8")):
    if k not in F.FACTORS:
        F.FACTORS[k] = F.FactorSpec(k, "美债实际利率·%dh" % h, mk(h), "诊断", "-", formula="-")

def bt(df, key, si, sym, exch, win=1440):
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

print("=" * 80)
print("① 因子值差异（按小时段统计）")
df = tqdata.load_main("AU", "SHFE", "5m", adjust="qfq", start="2020-01-01")
a = F.FACTORS["ur_h0"].func(df)
b = F.FACTORS["ur_h8"].func(df)
j = pd.concat([a.rename("h0"), b.rename("h8")], axis=1).dropna()
print("  两列都有值 %d/%d 根" % (len(j), len(df)))
diff = (j.h0 - j.h8).abs()
print("  逐位差：均值 %.4f  中位 %.4f  不等占比 %.1f%%"
      % (diff.mean(), diff.median(), (diff > 1e-9).mean() * 100))
hh = pd.Series(j.index.hour, index=j.index)
print("\n  按北京时间小时看「不一致占比」：")
for h in range(24):
    m = hh == h
    if m.sum() == 0: continue
    print("    %02d 点  %5.1f%%  （%d 根）" % (h, (diff[m] > 1e-9).mean() * 100, m.sum()))

print("\n" + "=" * 80)
print("② 策略层（window=1440、e1.0/x0.3、long_low）")
rows = []
for sym, exch in [("AU","SHFE"),("AG","SHFE"),("IF","CFFEX"),("IM","CFFEX")]:
    d = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    if not len(d): continue
    si = symbols_info.info(sym, exch)
    sh0, n0, eq0, s0 = bt(d, "ur_h0", si, sym, exch)
    sh8, n8, eq8, s8 = bt(d, "ur_h8", si, sym, exch)
    print("  %-4s  原(0h) 夏普 %+5.2f %4d 笔   加 8h 夏普 %+5.2f %4d 笔   信号不一致 %.1f%%"
          % (sym, sh0, n0, sh8, n8, (s0 != s8).mean() * 100))
    rows.append((sym, sh0, sh8))
    del d; gc.collect()
