# -*- coding: utf-8 -*-
"""Q1 取样时点敏感性 + Q2 分位差的结构"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from ttrade import macro as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FN = {"usdcnh": "USDCNH_5m.parquet", "dxy": "DOLLARIDX_5m.parquet"}

def load(code):
    df = pd.read_parquet(os.path.join(ROOT, "data_duka", FN[code]))
    df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]")
    return df

def ts_close(code):
    s = M.series(code).astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s

def slice_at(b, hh, mm):
    """每个自然日取 <= 当天 hh:mm 的最后一根 close；取不到则 NaN"""
    cutoff = b.index.normalize() + pd.Timedelta(hours=hh, minutes=mm)
    # 每个自然日的最后一根
    idx = pd.Series(b.index, index=b.index).groupby(b.index.normalize()).max()
    out = {}
    for day, last in idx.items():
        seg = b.loc[day:cutoff[0] if False else (day + pd.Timedelta(hours=hh, minutes=mm))]
        out[day] = float(seg["close"].iloc[-1]) if len(seg) else np.nan
    return pd.Series(out).sort_index()

def daily_last(b, day_shift=0):
    s = b["close"].astype(float).groupby(b.index.normalize() + pd.Timedelta(days=day_shift)).last()
    s.index = pd.DatetimeIndex(s.index).astype("datetime64[ns]")
    return s

print("=" * 80)
for code in ("usdcnh", "dxy"):
    b = load(code)
    ts = ts_close(code)
    print("\n### %s" % code)
    # 候选口径
    cands = {}
    for hh, mm, nm in [(0,0,'当日00:00'), (15,0,'当日15:00'), (21,0,'当日21:00'),
                       (23,0,'当日23:00'), (23,55,'当日23:55(现用)'),
                       (5,55,'次日05:55(纽约收)'), (8,0,'次日08:00')]:
        if nm.startswith("次日"):
            s = daily_last(b, 1)
            # 次日 05:55 = 当日 23:59 之后再取到次日 05:55：直接用切片法
            continue
        cands[nm] = slice_at(b, hh, mm)
    # 次日口径：用「当天 hh:mm 切片，索引 +1 天」
    for hh, mm, nm in [(5,55,'次日05:55(纽约收)'), (8,0,'次日08:00')]:
        s = slice_at(b, hh, mm)
        s.index = s.index + pd.Timedelta(days=1)
        cands[nm] = s
    cands['当日23:55(现用)'] = daily_last(b, 0)

    print("%-18s %6s %10s %10s %10s %10s %8s" %
          ("口径", "样本", "水平相关", "差中位bp", "分位相关", "分位差均值", ">0.2占比"))
    for nm, s in cands.items():
        j = pd.concat([s.rename("a"), ts.rename("b")], axis=1).dropna()
        if len(j) < 100:
            print("%-18s 样本不足 %d" % (nm, len(j))); continue
        # 只保留重叠区间（Duka 覆盖段）
        corr = j["a"].corr(j["b"])
        diff_bp = ((j["a"] - j["b"]) / j["b"] * 1e4)
        pa = j["a"].expanding(min_periods=60).rank(pct=True)
        pb = j["b"].expand(min_periods=60).rank(pct=True) if False else j["b"].expanding(min_periods=60).rank(pct=True)
        pc = pa.corr(pb)
        pd_ = (pa - pb).abs()
        print("%-18s %6d %10.5f %10.2f %10.4f %10.4f %7.1f%%" %
              (nm, len(j), corr, diff_bp.median(), pc, pd_.mean(), (pd_ > 0.2).mean()*100))
