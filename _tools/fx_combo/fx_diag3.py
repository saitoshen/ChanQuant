# -*- coding: utf-8 -*-
"""Q3：分位几乎一样（差 0.0120），为什么夏普从 +1.18 掉到 +0.05？

把 macro_usdcnh 和 fx5d_usdcnh_pct 的差异拆成两维，各自单独翻转：
  维1 分位起点   Tushare 全史(2012)  vs  只从 2020 起
  维2 可用时刻   D+1 08:00           vs  D+1 00:00

四组对照 = 2×2 全因子，一眼看出是哪一维在起作用。
"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import factors_macro as FM, symbols_info, tqdata, macro as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRIN = 1_000_000.0

# ── 三种分位序列 ───────────────────────────────────────────────────────────
def _ts_series(code):
    s = M.series(code).astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s

_S = {}

def pct_full(code):                      # Tushare 全史扩张分位（系统现用）
    k = (code, "full")
    if k not in _S:
        _S[k] = _ts_series(code).expanding(min_periods=60).rank(pct=True)
    return _S[k]

def pct_2020(code):                      # Tushare 但只从 2020 起算
    k = (code, "2020")
    if k not in _S:
        s = _ts_series(code)
        s = s[s.index >= pd.Timestamp("2020-01-01")]
        _S[k] = s.expanding(min_periods=60).rank(pct=True)
    return _S[k]

# ── 两个可用时刻 ───────────────────────────────────────────────────────────
def asof08(series, index):               # D+1 08:00（macro 现用）
    return FM._asof(series, index, lag_days=1, avail_hour=8)

def asof00(series, index):               # D+1 00:00（5m 聚合用）
    if series is None or not len(series):
        return pd.Series(np.nan, index=index, dtype=float)
    t = pd.DatetimeIndex(index).astype("datetime64[ns]") - pd.Timedelta(days=1)
    right = pd.DataFrame({"t": pd.DatetimeIndex(series.index).astype("datetime64[ns]"),
                          "v": series.to_numpy(dtype=float)}).dropna().sort_values("t")
    right = right.groupby("t", as_index=False).last()
    left = pd.DataFrame({"t": t}); left["_i"] = np.arange(len(left))
    o = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward")
    return pd.Series(o.sort_values("_i")["v"].to_numpy(dtype=float), index=index)

# ── 注册 4 个诊断因子 ──────────────────────────────────────────────────────
CASES = [
    ("full_08", "全史分位·D+1 08:00", pct_full, asof08),
    ("full_00", "全史分位·D+1 00:00", pct_full, asof00),
    ("2020_08", "2020起分位·D+1 08:00", pct_2020, asof08),
    ("2020_00", "2020起分位·D+1 00:00", pct_2020, asof00),
]
for tag, label, pf, af in CASES:
    key = "dg_%s" % tag
    def mk(pf=pf, af=af):
        def f(df):
            return af(pf("usdcnh"), df.index)
        return f
    if key not in F.FACTORS:
        F.FACTORS[key] = F.FactorSpec(key, label, mk(), "诊断", label, formula="-")

# ── 回测 ───────────────────────────────────────────────────────────────────
def backtest(df, key, si, sym, exch):
    cfg = FS.FactorConfig(window=545, method="z", entry_z=1.0, exit_z=0.3,
                          orient="long_low", horizon=12)
    out = FS.run_factor_strategy(df, "single", [key], cfg)
    c = BT.BacktestConfig(initial_cash=PRIN)
    c.sizing = "capital"
    c.capital = PRIN * float(si["margin"])
    c.multiplier = float(si["mult"]); c.margin_rate = float(si["margin"])
    c.bars_per_year = BT.bars_per_year(out.df.index)
    costs.apply(c, sym, exch)
    r = BT.run_backtest(out.df, c)
    sig = (out.df["signal"].to_numpy() != 0).mean()
    eq = r.equity / PRIN - 1.0
    del out
    return eq, r.metrics, float(sig)

SYMS = [("AU","SHFE"),("AG","SHFE"),("CU","SHFE"),("SN","SHFE"),("AL","SHFE"),
        ("BC","INE"),("IH","CFFEX"),("IF","CFFEX"),("IC","CFFEX"),("IM","CFFEX")]
START = "2020-01-01"
rows, curves = [], {}
print("样本自 %s  e3.0/x1.0/w545  orient=long_low（固定，不用 auto）\n" % START)
for sym, exch in SYMS:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start=START)
    if not len(df):
        print("  %s 无数据" % sym); continue
    si = symbols_info.info(sym, exch)
    for key, label, _, _ in [("macro_usdcnh", "Tushare现用(全史·08)", None, None)] + \
                             [("dg_%s" % t, l, None, None) for t, l, _, _ in CASES]:
        eq, m, occ = backtest(df, key, si, sym, exch)
        curves.setdefault(key, []).append(eq.rename(sym))
        rows.append(dict(sym=sym, key=key, 口径=label,
                         夏普=round(float(m["sharpe"] or 0), 2),
                         年化=round(float(m["cagr"] or 0)*100, 1),
                         回撤=round(float(m["max_dd"] or 0)*100, 1),
                         笔数=int(m["n_trades"] or 0),
                         占用=round(occ*100, 1)))
        gc.collect()
    del df; gc.collect()

d = pd.DataFrame(rows)
print("=" * 88)
print("【单品种夏普】")
pv = d.pivot(index="sym", columns="口径", values="夏普")
order = ["Tushare现用(全史·08)", "全史分位·D+1 08:00", "全史分位·D+1 00:00",
         "2020起分位·D+1 08:00", "2020起分位·D+1 00:00"]
print(pv.reindex(columns=[c for c in order if c in pv.columns]).to_string())
print("\n【笔数】")
pv2 = d.pivot(index="sym", columns="口径", values="笔数")
print(pv2.reindex(columns=[c for c in order if c in pv.columns]).to_string())
print("\n【占用%】")
pv3 = d.pivot(index="sym", columns="口径", values="占用")
print(pv3.reindex(columns=[c for c in order if c in pv.columns]).to_string())

print("\n" + "=" * 88)
print("【组合（各品种等本金）】")
for key in ["macro_usdcnh"] + ["dg_%s" % t for t, _, _, _ in CASES]:
    legs = curves.get(key, [])
    if not legs: continue
    com = pd.concat(legs, axis=1).ffill().fillna(0.0)
    nav = 1.0 + com.mean(axis=1)
    ret = nav.pct_change().dropna()
    sh = float(ret.mean()/ret.std()*np.sqrt(252*48)) if ret.std() else np.nan
    dd = float((nav/nav.cummax()-1).min()*100)
    lab = dict((k, l) for k, l, _, _ in CASES)[key.replace("dg_", "")] if key.startswith("dg_") else "Tushare现用(全史·08)"
    print("  %-24s 夏普 %+5.2f   年化 %+6.1f%%   回撤 %6.1f%%" %
          (lab, sh, (nav.iloc[-1]-1)*100/((nav.index[-1]-nav.index[0]).days/365.25), dd))
