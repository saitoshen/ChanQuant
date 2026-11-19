"""汇率 5m 因子（Dukascopy）效果测试 + 时间校正对照。

用法:
    python _tools/fx_combo/fx5_test.py [START] [cmd|idx|both]

两个问题一起回答：
  1. 这个 5m 因子相对日频 macro_usdcnh 有没有用；
  2. 把「可用时刻 = bar 结束时刻」这条校正去掉（`AVAIL_SHIFT=0`）会虚增多少。
"""

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from ttrade import backtest as BT
from ttrade import costs
from ttrade import factor_strategy as FS
from ttrade import factors as F
from ttrade import factors_fx5 as X
from ttrade import symbols_info, tqdata

PRIN = 1_000_000.0
KEYS5 = ["fx5_usdcnh_chg_6", "fx5_usdcnh_chg_12", "fx5_usdcnh_chg_48",
         "fx5_usdcnh_z_48", "fx5_usdcnh_rv_48", "fx5_usdcnh_pct"]
SHORT = {"fx5_usdcnh_chg_6": "5m·30分", "fx5_usdcnh_chg_12": "5m·1时",
         "fx5_usdcnh_chg_48": "5m·4时", "fx5_usdcnh_z_48": "5m·z48",
         "fx5_usdcnh_rv_48": "5m·rv48", "fx5_usdcnh_pct": "5m·分位",
         "macro_usdcnh": "日频"}


def backtest(df, key, si, sym, exch):
    cfg = FS.FactorConfig(window=545, method="z", entry_z=1.0, exit_z=0.3,
                          orient="long_low", horizon=12)
    out = FS.run_factor_strategy(df, "single", [key], cfg)
    c = BT.BacktestConfig(initial_cash=PRIN)
    c.sizing = "capital"
    c.capital = PRIN * float(si["margin"])
    c.multiplier = float(si["mult"])
    c.margin_rate = float(si["margin"])
    c.bars_per_year = BT.bars_per_year(out.df.index)
    costs.apply(c, sym, exch)
    r = BT.run_backtest(out.df, c)
    sig = out.df["signal"].to_numpy()
    eq = r.equity / PRIN - 1.0
    del out
    return eq, r.metrics, float((sig != 0).mean()), sig


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2020-01-01"
    grp = sys.argv[2] if len(sys.argv) > 2 else "both"
    syms = []
    if grp in ("cmd", "both"):
        syms += [("AU", "SHFE"), ("AG", "SHFE"), ("SN", "SHFE"),
                 ("AL", "SHFE"), ("CU", "SHFE"), ("BC", "INE")]
    if grp in ("idx", "both"):
        syms += [("IH", "CFFEX"), ("IF", "CFFEX"), ("IC", "CFFEX"), ("IM", "CFFEX")]

    print(f"样本自 {start}（fx5 数据 2020-01-02 起）  品种 {len(syms)}")
    print("参数：window 545 / entry 1.0 / exit 0.3 / orient long_low / horizon 12\n")

    rows = []
    lags = [("lag1", pd.Timedelta("5min")), ("lag0", pd.Timedelta(0))]
    curves = {}
    for sym, exch in syms:
        df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start=start)
        if not len(df):
            print(f"  {sym} 无数据，跳过")
            continue
        si = symbols_info.info(sym, exch)
        sigs = {}
        for lag_name, shift in lags:
            X.AVAIL_SHIFT = shift
            for key in KEYS5 + ["macro_usdcnh"]:
                # lag 只影响 fx5 系列
                if lag_name == "lag0" and not key.startswith("fx5_"):
                    continue
                eq, m, occ, sig = backtest(df, key, si, sym, exch)
                curves.setdefault((key, lag_name), []).append(eq.rename(sym))
                sigs[(key, lag_name)] = sig
                rows.append(dict(sym=sym, key=key, 口径=SHORT[key],
                                 lag=lag_name,
                                 夏普=round(float(m["sharpe"] or 0), 2),
                                 年化=round(float(m["cagr"] or 0) * 100, 1),
                                 回撤=round(float(m["max_dd"] or 0) * 100, 1),
                                 卡玛=round(float(m["calmar"] or 0), 2),
                                 笔数=int(m["n_trades"] or 0),
                                 占用=round(occ * 100, 1)))
                gc.collect()
        # lag 对因子值/信号的影响
        for key in KEYS5:
            X.AVAIL_SHIFT = pd.Timedelta("5min")
            a = F.FACTORS[key].func(df)
            X.AVAIL_SHIFT = pd.Timedelta(0)
            b = F.FACTORS[key].func(df)
            same = (a.dropna() == b.reindex(a.dropna().index)).mean()
            rows_sig = (sigs.get((key, "lag0")), sigs.get((key, "lag1")))
            diff = float((rows_sig[0] != rows_sig[1]).mean()) if rows_sig[0] is not None else np.nan
            print(f"  {sym} {SHORT[key]:<8} 因子值相同 {same*100:5.1f}%  信号差异 {diff*100:5.1f}%")
        X.AVAIL_SHIFT = pd.Timedelta("5min")
        del df
        gc.collect()

    d = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    for grp_name, members in (("商品", [s for s, e in syms if (s, e) in
                                        [("AU", "SHFE"), ("AG", "SHFE"), ("SN", "SHFE"),
                                         ("AL", "SHFE"), ("CU", "SHFE"), ("BC", "INE")]]),
                              ("股指", [s for s, e in syms if (s, e) in
                                        [("IH", "CFFEX"), ("IF", "CFFEX"),
                                         ("IC", "CFFEX"), ("IM", "CFFEX")]])):
        sub = d[d.sym.isin(members)]
        if not len(sub):
            continue
        print(f"\n【{grp_name}】单品种夏普")
        pv = sub[sub.lag == "lag1"].pivot(index="sym", columns="口径", values="夏普")
        order = [SHORT[k] for k in KEYS5] + ["日频"]
        print(pv.reindex(columns=[c for c in order if c in pv.columns]).to_string())
        print(f"\n【{grp_name}】lag0 − lag1 的夏普差（越界虚增）")
        a = sub[sub.lag == "lag1"].set_index(["sym", "口径"])["夏普"]
        b = sub[sub.lag == "lag0"].set_index(["sym", "口径"])["夏普"]
        diff = (b - a).dropna()
        if len(diff):
            print(diff.unstack().round(2).to_string())

        print(f"\n【{grp_name}】组合层面（各品种等权）")
        print(f"  {'口径':<10} {'lag':<5} {'累计':>8} {'年化':>7} {'日波动':>7} "
              f"{'夏普':>6} {'回撤':>8} {'卡玛':>6}")
        for key in KEYS5 + ["macro_usdcnh"]:
            for lag_name in ("lag1", "lag0"):
                if lag_name == "lag0" and not key.startswith("fx5_"):
                    continue
                cs = []
                for (k, lg), lst in curves.items():
                    if k == key and lg == lag_name:
                        cs += [c for c in lst if c.name in members]
                if not cs:
                    continue
                mat = pd.concat(cs, axis=1).sort_index().ffill().fillna(0.0)
                port = mat.mean(axis=1)
                r = port.resample("1D").last().dropna().diff().dropna()
                dd = float((port - port.cummax()).min())
                yrs = max((port.index[-1] - port.index[0]).days / 365.25, 1e-9)
                print(f"  {SHORT[key]:<10} {lag_name:<5} {port.iloc[-1]*100:+7.1f}% "
                      f"{port.iloc[-1]/yrs*100:+6.1f}% "
                      f"{r.std(ddof=1)*np.sqrt(252)*100:6.1f}% "
                      f"{r.mean()/r.std(ddof=1)*np.sqrt(252):+6.2f} "
                      f"{dd*100:7.1f}% {port.iloc[-1]/yrs/abs(dd):+6.2f}")


if __name__ == "__main__":
    main()
