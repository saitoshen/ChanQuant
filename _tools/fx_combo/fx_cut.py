# -*- coding: utf-8 -*-
"""Q13：日线切分点该定在几点？

  A 北京 00:00 切（现状）：北京周一 00:00~04:55 归周一、周六 00:00~04:55 自成一条
  B 北京 08:00 切（=UTC 日，用户提议）：北京周一 00:00~08:00 归上一组
  C 北京 06:00 切（纽约日，冬令时正确）
  D 北京 05:00 切（纽约日，夏令时正确）

每个口径都按「该组最后一根 bar 的时刻 + 5min」算可用时刻，不额外加滞后。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import macro as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WD = ["一","二","三","四","五","六","日"]
BAR = pd.Timedelta("5min")

def duka_bars(code):
    df = pd.read_parquet(os.path.join(ROOT, "data_duka",
        {"usdcnh":"USDCNH_5m.parquet","dxy":"DOLLARIDX_5m.parquet"}[code]))
    df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]")
    return df.sort_index()

def ts_close(code):
    s = M.series(code).astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s

def by_cut(bars, cut_h):
    """按「北京时刻 − cut_h」的自然日分组，返回 (收盘, 可用时刻, 组内根数)"""
    shifted = bars.index - pd.Timedelta(hours=cut_h)
    day = shifted.normalize()
    px = bars["close"].astype(float)
    close = px.groupby(day).last()
    avail = bars.index.to_series().groupby(day).last() + BAR
    n = px.groupby(day).size()
    return close, avail, n

b = duka_bars("usdcnh")
ts = ts_close("usdcnh")
ts20 = ts[ts.index >= pd.Timestamp("2020-01-01")]
print("Duka 5m %d 根  Tushare(2020起) %d 行" % (len(b), len(ts20)))

CUTS = [("A 北京00:00(现状)", 0), ("B 北京08:00(UTC日)", 8),
        ("C 北京06:00(纽约冬)", 6), ("D 北京05:00(纽约夏)", 5)]
res = {}
print("\n%-22s %6s %8s %6s %-22s %s" %
      ("口径", "天数", "每周条数", "短条", "星期分布", "收盘时刻众数"))
for lab, h in CUTS:
    close, avail, n = by_cut(b, h)
    wk = 7 * 24 * 12   # 每天满 288 根
    short = int((n < 100).sum())
    c = close.index.dayofweek.value_counts().reindex(range(7), fill_value=0)
    mode_t = avail.dt.strftime("%H:%M").value_counts().head(3)
    res[lab] = (close, avail)
    print("%-22s %6d %8s %6d %-22s %s" %
          (lab, len(close), str(wk and pd.Series(
              close.groupby([close.index.isocalendar().year,
                             close.index.isocalendar().week]).size()).mode().tolist()),
           short,
           " ".join("%d" % c[i] for i in range(7)),
           " ".join("%s×%d" % (k, v) for k, v in mode_t.items())))

print("\n" + "=" * 90)
print("与 Tushare 对拍（重叠段；同样从 2020 起各算扩张分位）")
ts20pct = ts20.expanding(min_periods=60).rank(pct=True)
ts_full_pct = ts.expanding(min_periods=60).rank(pct=True)
print("%-22s %6s %10s %10s %10s %10s" %
      ("口径", "样本", "水平相关", "差中位bp", "分位差均值", "分位相关"))
for lab, h in CUTS:
    close, _, _ = by_cut(b, h)
    for nm, ref in ((" vs Tushare 同日", ts20), (" vs Tushare 次日(+1)", ts20.shift(-1))):
        j = pd.concat([close.rename("a"), ref.rename("b")], axis=1, sort=True).dropna()
        if len(j) < 200:
            print("%-22s %-20s 样本不足 %d" % (lab, nm, len(j))); continue
        pa = j["a"].expanding(min_periods=60).rank(pct=True)
        pb = j["b"].expanding(min_periods=60).rank(pct=True)
        print("%-22s %-20s %6d %10.5f %10.2f %10.4f %10.4f" %
              (lab, nm, len(j), j.a.corr(j.b),
               ((j.a - j.b) / j.b * 1e4).median(),
               (pa - pb).abs().mean(), pa.corr(pb)))
