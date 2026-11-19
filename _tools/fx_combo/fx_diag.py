# -*- coding: utf-8 -*-
"""诊断：5m 聚合的日线为什么和 Tushare 日频对不上（取样时点？数据源？）

三个问题一次答：
  Q1 Tushare 的「日收盘」对应北京几点？拿 5m 在不同时点切片，与 Tushare 对拍。
  Q2 分位差是均匀的还是有结构（按年 / 按日变动大小）？
  Q3 去掉分位、直接用日变动，还崩不崩？
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

from ttrade import macro as M
from ttrade import factors_macro as FM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def bars(code):
    fn = {"usdcnh": "USDCNH_5m.parquet", "dxy": "DOLLARIDX_5m.parquet"}[code]
    p = os.path.join(ROOT, "data_duka", fn)
    df = pd.read_parquet(p)
    if "close" not in df and df.shape[1] >= 1:
        pass
    df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]")
    return df

def tushare_series(code):
    raw = M.series(code)
    s = raw.astype(float).sort_index()
    return s[~s.index.duplicated(keep="last")]

print("=" * 78)
for code in ("usdcnh", "dxy"):
    ts = tushare_series(code)
    b = bars(code)
    print("%s  Tushare: %d 行  %s → %s" % (code, len(ts), ts.index[0].date(), ts.index[-1].date()))
    print("%s  Duka 5m: %d 根  %s → %s" % (code, len(b), b.index[0], b.index[-1]))
    print("%s  Duka 覆盖日: %d 个自然日" % (code, len(set(b.index.normalize()))))
    # 收盘时刻分布（每天最后一根的时刻）
    last = b.groupby(b.index.normalize()).apply(lambda x: x.index[-1])
    print("   每日最后一根的时刻 众数(前5):")
    print(last.dt.strftime("%H:%M").value_counts().head(5).to_string())
    print("-" * 78)
