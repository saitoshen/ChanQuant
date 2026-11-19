# -*- coding: utf-8 -*-
"""Q10：把两边都压到周一到周五，看「周末行」是不是两数据源差异的来源"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import factors_macro as FM, symbols_info, tqdata, macro as M

PRIN = 1_000_000.0
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ts_s(code):
    s = M.series(code).astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s

def duka_d(code):
    df = pd.read_parquet(os.path.join(ROOT, "data_duka",
        {"usdcnh":"USDCNH_5m.parquet","dxy":"DOLLARIDX_5m.parquet"}[code]))
    df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]")
    return df["close"].astype(float).resample("1D").last().dropna()

def asof00(series, index):
    t = pd.DatetimeIndex(index).astype("datetime64[ns]") - pd.Timedelta(days=1)
    r = pd.DataFrame({"t": pd.DatetimeIndex(series.index).astype("datetime64[ns]"),
                      "v": series.to_numpy(dtype=float)}).dropna().sort_values("t")
    r = r.groupby("t", as_index=False).last()
    l = pd.DataFrame({"t": t}); l["_i"] = np.arange(len(l))
    o = pd.merge_asof(l.sort_values("t"), r, on="t", direction="backward")
    return pd.Series(o.sort_values("_i")["v"].to_numpy(dtype=float), index=index)

def mk(src, wd_only):
    def f(df):
        s = (ts_s("usdcnh") if src == "ts" else duka_d("usdcnh"))
        s = s[s.index >= pd.Timestamp("2020-01-01")]
        if wd_only:
            s = s[s.index.dayofweek < 5]
        return asof00(s.expanding(min_periods=60).rank(pct=True), df.index)
    return f

NAMES = [("wk_ts_full","Tush全史含周末","ts",False), ("wk_ts_wd","Tush只周一~周五","ts",True),
         ("wk_dk_full","Duka含周末","dk",False), ("wk_dk_wd","Duka只周一~周五","dk",True)]
for k, lab, src, wd in NAMES:
    if k not in F.FACTORS:
        F.FACTORS[k] = F.FactorSpec(k, lab, mk(src, wd), "诊断", lab, formula="-")

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
    m = r.metrics; sig = out.df["signal"].to_numpy(); del out
    return float(m["sharpe"] or 0), int(m["n_trades"] or 0), sig

print("=" * 84)
print("PART A｜分位差：含周末 vs 只周一~周五")
for label, wd in (("含周末(现状)", False), ("只周一~周五", True)):
    a = F.FACTORS["wk_ts_full" if not wd else "wk_ts_wd"].func(
        tqdata.load_main("AU","SHFE","5m",adjust="qfq",start="2020-01-01"))
    b = F.FACTORS["wk_dk_full" if not wd else "wk_dk_wd"].func(
        tqdata.load_main("AU","SHFE","5m",adjust="qfq",start="2020-01-01"))
    j = pd.concat([a.rename("t"), b.rename("d")], axis=1).dropna()
    df_ = (j.t - j.d).abs()
    print("  %-14s 逐位差均值 %.4f 中位 %.4f >0.2 占 %.1f%% 相关 %.4f"
          % (label, df_.mean(), df_.median(), (df_>0.2).mean()*100, j.t.corr(j.d)))

print("\n" + "=" * 84)
print("PART B｜夏普（window 545 / 2880）")
SYMS = [("AU","SHFE"),("AG","SHFE"),("IM","CFFEX")]
rows = []
for sym, exch in SYMS:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    if not len(df): continue
    si = symbols_info.info(sym, exch)
    for win in (545, 2880):
        sigs = {}
        for k, lab, _, _ in NAMES:
            sh, n, sg = bt(df, k, si, sym, exch, win)
            sigs[lab] = sg
            rows.append(dict(sym=sym, win=win, 口径=lab, 夏普=round(sh,2), 笔数=n))
            gc.collect()
        for x, y, tag in (("Tush全史含周末","Duka含周末","互相差(含周末)"),
                          ("Tush只周一~周五","Duka只周一~周五","互相差(只工作日)")):
            rows.append(dict(sym=sym, win=win, 口径=tag,
                             夏普=round(float((sigs[x]!=sigs[y]).mean()*100),1), 笔数=0))
    del df; gc.collect()

t = pd.DataFrame(rows)
for win in (545, 2880):
    s = t[t.win==win]
    print("\n  --- window %d 根 ---" % win)
    print(s.pivot(index="sym", columns="口径", values="夏普").to_string())
