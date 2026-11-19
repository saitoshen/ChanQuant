# -*- coding: utf-8 -*-
"""Q7：window=545 根在 5m 上只有 1.9 个交易日 —— 日频因子被 z 标准化退化了？

分位是慢变量，1.9 天内几乎不动 ⇒ 窗口 std 极小 ⇒ z = 差/极小 std 把 0.013 的
输入差放大成几个标准差 ⇒ 两个几乎同样的因子给出 30% 不同的信号。

扫 window：545(1.9天) / 2880(10天) / 14400(50天)，看两边是否收敛。
"""
import gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import backtest as BT, costs, factor_strategy as FS, factors as F
from ttrade import factors_macro as FM, symbols_info, tqdata, macro as M

PRIN = 1_000_000.0

def _ts(code):
    s = M.series(code).astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s

def asof00(series, index):
    t = pd.DatetimeIndex(index).astype("datetime64[ns]") - pd.Timedelta(days=1)
    right = pd.DataFrame({"t": pd.DatetimeIndex(series.index).astype("datetime64[ns]"),
                          "v": series.to_numpy(dtype=float)}).dropna().sort_values("t")
    right = right.groupby("t", as_index=False).last()
    left = pd.DataFrame({"t": t}); left["_i"] = np.arange(len(left))
    o = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward")
    return pd.Series(o.sort_values("_i")["v"].to_numpy(dtype=float), index=index)

def dg2020(df):
    s = _ts("usdcnh"); s = s[s.index >= pd.Timestamp("2020-01-01")]
    return asof00(s.expanding(min_periods=60).rank(pct=True), df.index)

KEY = "dgw_tush2020"
if KEY not in F.FACTORS:
    F.FACTORS[KEY] = F.FactorSpec(KEY, "Tushare2020起·00可用", dg2020, "诊断", "-", formula="-")

CASES = [("macro_usdcnh", "Tushare全史"), (KEY, "Tushare2020起"), ("fx5d_usdcnh_pct", "Duka5m聚合")]

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
    m = r.metrics; sig = out.df["signal"].to_numpy()
    del out
    return float(m["sharpe"] or 0), float(m["cagr"] or 0)*100, int(m["n_trades"] or 0), sig

SYMS = [("AU","SHFE"),("AG","SHFE"),("CU","SHFE"),("SN","SHFE"),("IM","CFFEX")]
WINS = [545, 2880, 14400]
rows = []
for sym, exch in SYMS:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    if not len(df): continue
    si = symbols_info.info(sym, exch)
    for win in WINS:
        sigs = {}
        for key, lab in CASES:
            sh, cagr, n, sig = backtest(df, key, si, sym, exch, win)
            sigs[lab] = sig
            rows.append(dict(sym=sym, win=win, 因子=lab, 夏普=round(sh, 2),
                             年化=round(cagr, 1), 笔数=n))
            gc.collect()
        a, b = sigs["Tushare2020起"], sigs["Duka5m聚合"]
        rows.append(dict(sym=sym, win=win, 因子="信号不一致%",
                         夏普=round(float((a != b).mean()*100), 1), 年化=np.nan, 笔数=0))
    del df; gc.collect()

d = pd.DataFrame(rows)
for win in WINS:
    sub = d[(d.win == win) & (d.因子 != "信号不一致%")]
    print("\n" + "=" * 78)
    print("window = %d 根（5m 上约 %.1f 个交易日）" % (win, win / 288.0))
    print("【夏普】")
    print(sub.pivot(index="sym", columns="因子", values="夏普").to_string())
    print("【笔数】")
    print(sub.pivot(index="sym", columns="因子", values="笔数").to_string())
    dd = d[(d.win == win) & (d.因子 == "信号不一致%")].set_index("sym")["夏普"]
    print("【Tushare2020起 vs Duka 信号不一致%%】 %s" %
          ", ".join("%s %.1f" % (k, v) for k, v in dd.items()))
