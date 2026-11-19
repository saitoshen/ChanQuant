# -*- coding: utf-8 -*-
"""Q6（决定性）：给 macro_usdcnh 只加一点点分位噪声，夏普会不会崩？

实测：Duka 与 Tushare 的分位逐位差只有 0.0130（相关 0.9981），但策略信号
不一致 30.6%。如果给原生因子加 0.013 的噪声就把夏普打下来，那这套参数
（z545 + 滞回 1.0/0.3）本身就把噪声放大成了结论 —— 不是数据源的问题。
"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import factors_macro as FM, symbols_info, tqdata, macro as M

PRIN = 1_000_000.0
_rng = np.random.default_rng(20261119)

def make_noisy(sigma):
    def f(df):
        base = F.FACTORS["macro_usdcnh"].func(df)
        v = base.to_numpy(dtype=float)
        n = _rng.normal(0, sigma, len(v)) if sigma else np.zeros(len(v))
        out = pd.Series(np.clip(v + n, 0.0, 1.0), index=df.index)
        out[base.isna()] = np.nan
        return out
    return f

for s in (0.005, 0.013, 0.03, 0.06):
    k = "rb_s%04d" % int(s * 1000)
    if k not in F.FACTORS:
        F.FACTORS[k] = F.FactorSpec(k, "扰动%.3f" % s, make_noisy(s), "诊断",
                                    "-", formula="-")

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
    sig = out.df["signal"].to_numpy()
    eq = r.equity / PRIN - 1.0
    del out
    return eq, r.metrics, sig

KEYS = ["macro_usdcnh"] + ["rb_s%04d" % int(s*1000) for s in (0.005, 0.013, 0.03, 0.06)]
SYMS = [("AU","SHFE"),("AG","SHFE"),("CU","SHFE"),("SN","SHFE"),("AL","SHFE"),
        ("BC","INE"),("IH","CFFEX"),("IF","CFFEX"),("IC","CFFEX"),("IM","CFFEX")]
rows, curves = [], {}
for sym, exch in SYMS:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    if not len(df): continue
    si = symbols_info.info(sym, exch)
    base_sig = None
    for key in KEYS:
        eq, m, sig = backtest(df, key, si, sym, exch)
        if key == "macro_usdcnh":
            base_sig = sig
        curves.setdefault(key, []).append(eq.rename(sym))
        rows.append(dict(sym=sym, key=key, sigma=0.0 if key == "macro_usdcnh"
                         else float(key.replace("rb_s",""))/1000,
                         夏普=round(float(m["sharpe"] or 0), 2),
                         年化=round(float(m["cagr"] or 0)*100, 1),
                         回撤=round(float(m["max_dd"] or 0)*100, 1),
                         笔数=int(m["n_trades"] or 0),
                         信号差=round(float((sig != base_sig).mean()*100), 1) if base_sig is not None else 0.0))
        gc.collect()
    del df; gc.collect()

d = pd.DataFrame(rows)
print("=" * 92)
print("【单品种夏普：给 macro_usdcnh 加分位噪声】")
pv = d.pivot(index="sym", columns="sigma", values="夏普")
print(pv.to_string())
print("\n【信号不一致率 %】")
print(d.pivot(index="sym", columns="sigma", values="信号差").to_string())
print("\n【笔数】")
print(d.pivot(index="sym", columns="sigma", values="笔数").to_string())

print("\n" + "=" * 92)
print("【组合夏普（等本金）】")
for key in KEYS:
    com = pd.concat(curves[key], axis=1).ffill().fillna(0.0)
    nav = 1.0 + com.mean(axis=1)
    ret = nav.pct_change().dropna()
    sh = float(ret.mean()/ret.std()*np.sqrt(252*48)) if ret.std() else np.nan
    dd = float((nav/nav.cummax()-1).min()*100)
    lab = "原始" if key == "macro_usdcnh" else "噪声 σ=%.3f" % (float(key.replace("rb_s",""))/1000)
    print("  %-14s 夏普 %+5.2f   总收益 %+7.1f%%   回撤 %6.1f%%" %
          (lab, sh, (nav.iloc[-1]-1)*100, dd))
