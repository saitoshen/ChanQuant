# -*- coding: utf-8 -*-
"""Q9：Tushare 与 Duka 各自的「一周几天」——周末到底有没有行"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import macro as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WD = ["周一","周二","周三","周四","周五","周六","周日"]

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

for code in ("usdcnh", "dxy"):
    print("=" * 76)
    t = ts_s(code); d = duka_d(code)
    t20 = t[t.index >= pd.Timestamp("2020-01-01")]
    print("### %s" % code)
    print("%-10s %6s %s" % ("", "总行数", "  ".join("%5s" % w for w in WD)))
    for nm, s in (("Tushare全史", t), ("Tushare2020起", t20), ("Duka日线", d)):
        c = s.index.dayofweek.value_counts().reindex(range(7), fill_value=0)
        print("%-12s %6d %s" % (nm, len(s), "  ".join("%5d" % c[i] for i in range(7))))
    # 每周条数
    wk = lambda s: s.groupby([s.index.isocalendar().year, s.index.isocalendar().week]).size()
    print("  每周条数 众数: Tushare2020起 %s | Duka %s" %
          (wk(t20).mode().tolist(), wk(d).mode().tolist()))
    # 周末行的值从哪来
    we = t20[t20.index.dayofweek >= 5]
    if len(we):
        print("  Tushare 周末行 %d 条，样例：" % len(we))
        print(we.head(6).to_string())
    print()
