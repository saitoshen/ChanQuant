# -*- coding: utf-8 -*-
"""Q5：同一份 bar 上，fx5d_usdcnh_pct 与「Tushare 2020起分位·D+1 00:00」的
因子值 / 信号 / IC 差多少 —— 分位只差 0.0120 的话，信号不该差出一条街。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import factors as F, factors_macro as FM, factor_strategy as FS
from ttrade import symbols_info, tqdata, macro as M

def _ts_series(code):
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
    s = _ts_series("usdcnh")
    s = s[s.index >= pd.Timestamp("2020-01-01")]
    return asof00(s.expanding(min_periods=60).rank(pct=True), df.index)

for sym, exch in [("AU","SHFE"), ("AG","SHFE"), ("IM","CFFEX")]:
    df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
    a = F.FACTORS["fx5d_usdcnh_pct"].func(df)
    b = dg2020(df)
    j = pd.concat([a.rename("duka"), b.rename("tush")], axis=1).dropna()
    print("\n" + "=" * 74)
    print("%s  bar %d 根，两列都有值 %d 根" % (sym, len(df), len(j)))
    print("  Duka 分位: mean %.4f std %.4f min %.4f max %.4f  unique %d" %
          (j.duka.mean(), j.duka.std(), j.duka.min(), j.duka.max(), j.duka.nunique()))
    print("  Tush 分位: mean %.4f std %.4f min %.4f max %.4f  unique %d" %
          (j.tush.mean(), j.tush.std(), j.tush.min(), j.tush.max(), j.tush.nunique()))
    diff = (j.duka - j.tush).abs()
    print("  逐位差: 均值 %.4f 中位 %.4f >0.1 占 %.1f%% >0.2 占 %.1f%%" %
          (diff.mean(), diff.median(), (diff > 0.1).mean()*100, (diff > 0.2).mean()*100))
    print("  相关: %.4f" % j.duka.corr(j.tush))
    # 前向收益 IC
    fwd = (df["close"].shift(-12) / df["close"] - 1.0)
    f2 = fwd.reindex(j.index)
    print("  IC(h=12): Duka %+.4f   Tush %+.4f" %
          (j.duka.corr(f2, method="spearman"), j.tush.corr(f2, method="spearman")))
    # 信号
    cfg = FS.FactorConfig(window=545, method="z", entry_z=1.0, exit_z=0.3,
                          orient="long_low", horizon=12)
    k = "dgq5_tush"
    if k not in F.FACTORS:
        F.FACTORS[k] = F.FactorSpec(k, "诊断", dg2020, "诊断", "-", formula="-")
    o1 = FS.run_factor_strategy(df, "single", ["fx5d_usdcnh_pct"], cfg)
    o2 = FS.run_factor_strategy(df, "single", [k], cfg)
    s1 = o1.df["signal"].to_numpy(); s2 = o2.df["signal"].to_numpy()
    print("  信号: 不一致 %.1f%%   Duka 非零 %.1f%%  Tush 非零 %.1f%%" %
          ((s1 != s2).mean()*100, (s1 != 0).mean()*100, (s2 != 0).mean()*100))
