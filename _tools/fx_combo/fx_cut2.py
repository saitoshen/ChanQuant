# -*- coding: utf-8 -*-
"""Q14：四种日线切分点，策略层夏普差多少（可用时刻 = 该组最后一根 bar + 5min）"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import symbols_info, tqdata

PRIN = 1_000_000.0
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BAR = pd.Timedelta("5min")
_cache = {}

def bars_usdcnh():
    if "b" not in _cache:
        df = pd.read_parquet(os.path.join(ROOT, "data_duka", "USDCNH_5m.parquet"))
        df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]")
        _cache["b"] = df.sort_index()
    return _cache["b"]

def pct_at_cut(cut_h):
    if ("p", cut_h) in _cache:
        return _cache[("p", cut_h)]
    b = bars_usdcnh()
    shifted = b.index - pd.Timedelta(hours=cut_h)
    day = shifted.normalize()
    px = b["close"].astype(float)
    close = px.groupby(day).last()
    avail = b.index.to_series().groupby(day).last() + BAR
    pct = close.expanding(min_periods=60).rank(pct=True)
    out = (pct, avail.reindex(pct.index))
    _cache[("p", cut_h)] = out
    return out

def asof_by_avail(pct, avail, index):
    r = pd.DataFrame({"t": pd.DatetimeIndex(avail.to_numpy()).astype("datetime64[ns]"),
                      "v": pct.to_numpy(dtype=float)}).dropna().sort_values("t")
    l = pd.DataFrame({"t": pd.DatetimeIndex(index).astype("datetime64[ns]")})
    l["_i"] = np.arange(len(l))
    o = pd.merge_asof(l.sort_values("t"), r, on="t", direction="backward")
    return pd.Series(o.sort_values("_i")["v"].to_numpy(dtype=float), index=index)

CUTS = [("ct_a00","A 北京00:00(现状)",0), ("ct_b08","B 北京08:00(UTC日)",8),
        ("ct_c06","C 北京06:00(纽约冬)",6), ("ct_d05","D 北京05:00(纽约夏)",5)]
for k, lab, h in CUTS:
    def mk(h=h):
        def f(df):
            pct, avail = pct_at_cut(h)
            return asof_by_avail(pct, avail, df.index)
        return f
    if k not in F.FACTORS:
        F.FACTORS[k] = F.FactorSpec(k, lab, mk(), "诊断", lab, formula="-")

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
SYMS = [("AU","SHFE"),("AG","SHFE"),("IM","CFFEX"),("CU","SHFE"),("SN","SHFE")]
for sym, exch in SYMS:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    if not len(df): continue
    si = symbols_info.info(sym, exch)
    for win in (545,):
        for k, lab, _ in CUTS:
            sh, n, eq = bt(df, k, si, sym, exch, win)
            curves.setdefault((win, lab), []).append(eq.rename(sym))
            rows.append(dict(sym=sym, win=win, 口径=lab, 夏普=round(sh,2), 笔数=n))
            gc.collect()
    del df; gc.collect()

t = pd.DataFrame(rows)
for win in (545, 2880):
    print("\n" + "=" * 88)
    print("window = %d 根（%.1f 天）" % (win, win/288.0))
    print(t[t.win==win].pivot(index="sym", columns="口径", values="夏普").to_string())
    print("  笔数：")
    print(t[t.win==win].pivot(index="sym", columns="口径", values="笔数").to_string())
    for k, lab, _ in CUTS:
        com = pd.concat(curves[(win, lab)], axis=1).ffill().fillna(0.0)
        nav = 1.0 + com.mean(axis=1); ret = nav.pct_change().dropna()
        sh = float(ret.mean()/ret.std()*np.sqrt(252*48)) if ret.std() else np.nan
        print("   组合 %-22s 夏普 %+5.2f  总收益 %+7.1f%%" % (lab, sh, (nav.iloc[-1]-1)*100))
