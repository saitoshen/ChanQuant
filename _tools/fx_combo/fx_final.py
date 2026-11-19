# -*- coding: utf-8 -*-
"""Q8：①545 根窗口里分位到底有多稳（z 的分母）②提前 8 小时在合理窗口下值不值"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import factors_macro as FM, symbols_info, tqdata, macro as M
from ttrade import factors_fx5 as F5

PRIN = 1_000_000.0

# ── ① z 的分母 ─────────────────────────────────────────────────────────────
df = tqdata.load_main("AU", "SHFE", "5m", adjust="qfq", start="2020-01-01")
a = F.FACTORS["macro_usdcnh"].func(df)
print("=" * 80)
print("① macro_usdcnh 在 5m bar 上的滚动窗口统计")
for w in (545, 2880, 14400):
    m = a.rolling(w, min_periods=max(60, w // 4)).mean()
    s = a.rolling(w, min_periods=max(60, w // 4)).std()
    z = ((a - m) / s.replace(0, np.nan)).dropna()
    print("  window %6d 根 (%.1f 天): 窗口 std 中位 %.5f | z 的 std %.2f | |z|>1 占 %.1f%% | |z|>4 占 %.2f%%"
          % (w, w / 288.0, float(s.median()), float(z.std()),
             float((z.abs() > 1).mean() * 100), float((z.abs() > 4).mean() * 100)))
print("  ⇒ 545 根窗口的分位 std 只有 %.5f，两个数据源 0.0130 的分位差就是 %.1f 个 σ"
      % (float(a.rolling(545, min_periods=136).std().median()),
         0.0130 / max(float(a.rolling(545, min_periods=136).std().median()), 1e-9)))
del df; gc.collect()

# ── ② 提前 8 小时的价值（合理窗口下） ──────────────────────────────────────
def _ts(code):
    s = M.series(code).astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s

def asof_h(h):
    def f_series(series, index):
        t = pd.DatetimeIndex(index).astype("datetime64[ns]") - pd.Timedelta(days=1)
        if h: t = t - pd.Timedelta(hours=h)
        right = pd.DataFrame({"t": pd.DatetimeIndex(series.index).astype("datetime64[ns]"),
                              "v": series.to_numpy(dtype=float)}).dropna().sort_values("t")
        right = right.groupby("t", as_index=False).last()
        left = pd.DataFrame({"t": t}); left["_i"] = np.arange(len(left))
        o = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward")
        return pd.Series(o.sort_values("_i")["v"].to_numpy(dtype=float), index=index)
    return f_series

def mk(h):
    af = asof_h(h)
    def f(df):
        s = _ts("usdcnh"); s = s[s.index >= pd.Timestamp("2020-01-01")]
        return af(s.expanding(min_periods=60).rank(pct=True), df.index)
    return f

for h, k in ((8, "dgh_08"), (0, "dgh_00")):
    if k not in F.FACTORS:
        F.FACTORS[k] = F.FactorSpec(k, "Tushare·D+1 %02d:00" % h, mk(h), "诊断", "-", formula="-")

def backtest(df, key, si, sym, exch, win):
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

SYMS = [("AU","SHFE"),("AG","SHFE"),("CU","SHFE"),("SN","SHFE"),("IM","CFFEX")]
print("\n" + "=" * 80)
print("② 「早 8 小时到手」值不值（Tushare 同一份数据，只改可用时刻）")
for win in (2880, 14400):
    rows, curves = [], {}
    for sym, exch in SYMS:
        d = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
        if not len(d): continue
        si = symbols_info.info(sym, exch)
        for k, lab in (("dgh_08", "08:00"), ("dgh_00", "00:00")):
            sh, n, eq = backtest(d, k, si, sym, exch, win)
            curves.setdefault(lab, []).append(eq.rename(sym))
            rows.append(dict(sym=sym, 可用时刻=lab, 夏普=round(sh, 2), 笔数=n))
            gc.collect()
        del d; gc.collect()
    t = pd.DataFrame(rows)
    pv = t.pivot(index="sym", columns="可用时刻", values="夏普")
    print("\n  window=%d 根（%.0f 天）" % (win, win / 288.0))
    print(pv.to_string())
    for lab in ("08:00", "00:00"):
        com = pd.concat(curves[lab], axis=1).ffill().fillna(0.0)
        nav = 1.0 + com.mean(axis=1); ret = nav.pct_change().dropna()
        sh = float(ret.mean()/ret.std()*np.sqrt(252*48)) if ret.std() else np.nan
        print("    组合 %s 夏普 %+5.2f  总收益 %+7.1f%%  回撤 %6.1f%%"
              % (lab, sh, (nav.iloc[-1]-1)*100, float((nav/nav.cummax()-1).min()*100)))
