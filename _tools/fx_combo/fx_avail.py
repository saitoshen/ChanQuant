# -*- coding: utf-8 -*-
"""Q12：「提前可用」到底值多少 —— 必须把三种口径分开

  Tushare·08:00   合法（Tushare 的 D 行 D+1 08:00 才发布）
  Tushare·00:00   🔴非法：把 D 行提前 8 小时用，那时它还没发布（偷未来）
  Duka 5m·00:00   合法：D 日收盘在 D+1 00:00 已能从行情读到
  Duka 5m·08:00   合法但保守（假装要等 8 小时）—— 与上一行之差 = 5m 的真实提前价值
"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import symbols_info, tqdata, macro as M
from ttrade import factors_fx5 as F5

PRIN = 1_000_000.0
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ts_series(code):
    s = M.series(code).astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s

def duka_series(code):
    df = pd.read_parquet(os.path.join(ROOT, "data_duka",
        {"usdcnh":"USDCNH_5m.parquet","dxy":"DOLLARIDX_5m.parquet"}[code]))
    df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]")
    d = df["close"].astype(float).resample("1D").last().dropna()
    return d.expanding(min_periods=60).rank(pct=True)

def asof(series, index, hours):
    t = pd.DatetimeIndex(index).astype("datetime64[ns]") - pd.Timedelta(days=1)
    if hours: t = t - pd.Timedelta(hours=hours)
    r = pd.DataFrame({"t": pd.DatetimeIndex(series.index).astype("datetime64[ns]"),
                      "v": series.to_numpy(dtype=float)}).dropna().sort_values("t")
    r = r.groupby("t", as_index=False).last()
    l = pd.DataFrame({"t": t}); l["_i"] = np.arange(len(l))
    o = pd.merge_asof(l.sort_values("t"), r, on="t", direction="backward")
    return pd.Series(o.sort_values("_i")["v"].to_numpy(dtype=float), index=index)

def mkk(src, hours):
    def f(df):
        s = ts_series("usdcnh") if src == "ts" else duka_series("usdcnh")
        if src == "ts":
            s = s.expanding(min_periods=60).rank(pct=True)
        return asof(s, df.index, hours)
    return f

CASES = [("av_ts08","Tushare·08:00(合法)",   "ts", 8),
         ("av_ts00","Tushare·00:00(🔴偷未来)","ts", 0),
         ("av_dk08","Duka5m·08:00(保守)",    "dk", 8),
         ("av_dk00","Duka5m·00:00(合法)",    "dk", 0)]
for k, lab, src, h in CASES:
    if k not in F.FACTORS:
        F.FACTORS[k] = F.FactorSpec(k, lab, mkk(src, h), "诊断", lab, formula="-")

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
for sym, exch in [("AU","SHFE"),("AG","SHFE"),("IM","CFFEX")]:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    if not len(df): continue
    si = symbols_info.info(sym, exch)
    for win in (545, 2880):
        for k, lab, _, _ in CASES:
            sh, n, eq = bt(df, k, si, sym, exch, win)
            curves.setdefault((win, lab), []).append(eq.rename(sym))
            rows.append(dict(sym=sym, win=win, 口径=lab, 夏普=round(sh,2), 笔数=n))
            gc.collect()
    del df; gc.collect()

t = pd.DataFrame(rows)
for win in (545, 2880):
    print("\n" + "=" * 84)
    print("window = %d 根（%.1f 天）" % (win, win/288.0))
    print(t[t.win==win].pivot(index="sym", columns="口径", values="夏普").to_string())
    for lab in [c[1] for c in CASES]:
        com = pd.concat(curves[(win, lab)], axis=1).ffill().fillna(0.0)
        nav = 1.0 + com.mean(axis=1); ret = nav.pct_change().dropna()
        sh = float(ret.mean()/ret.std()*np.sqrt(252*48)) if ret.std() else np.nan
        print("   组合 %-24s 夏普 %+5.2f  总收益 %+7.1f%%" % (lab, sh, (nav.iloc[-1]-1)*100))
