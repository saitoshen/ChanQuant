# -*- coding: utf-8 -*-
"""Q4：fx5d_usdcnh_pct 和「2020起分位·D+1 00:00」口径完全相同，为什么一个 -0.30 一个 +1.52？
直接在同一份 bar 上比两个因子的值。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import factors as F, factors_fx5 as F5, symbols_info, tqdata, macro as M

df = tqdata.load_main("AU", "SHFE", "5m", adjust="qfq", start="2020-01-01")
a = F.FACTORS["fx5d_usdcnh_pct"].func(df)
print("fx5d_usdcnh_pct: 非空 %d/%d  unique %d" % (a.notna().sum(), len(a), a.nunique()))
print(a.dropna().head(3).to_string())
print("...")
print(a.dropna().tail(3).to_string())

# 手工复算一遍，逐层打印
b = F5._load("usdcnh")
print("\nbars: %d 根  %s → %s" % (len(b), b.index[0], b.index[-1]))
d = b["close"].astype(float).resample("1D").last().dropna()
print("日线 %d 条  %s → %s" % (len(d), d.index[0], d.index[-1]))
wk = d.index.dayofweek
print("日线里各星期几的条数:", dict(zip(*np.unique(wk, return_counts=True))))
pct = d.expanding(min_periods=60).rank(pct=True)
print("分位 %d 条，前 5：" % len(pct))
print(pct.head(5).to_string())
print("\n周日那几条（前 5）：")
sun = pct[pct.index.dayofweek == 6]
print(sun.head(5).to_string())
