import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from ttrade import macro as M
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WD = ["一","二","三","四","五","六","日"]
t = M.series("usdcnh").astype(float).sort_index()
t = t[~t.index.duplicated(keep="last")]
t.index = pd.DatetimeIndex(t.index).normalize()
d = pd.read_parquet(os.path.join(ROOT,"data_duka","USDCNH_5m.parquet"))
d.index = pd.DatetimeIndex(d.index).astype("datetime64[ns]")
dd = d["close"].astype(float).resample("1D").last().dropna()

print("--- Tushare 2020-01-01 ~ 2020-01-20 ---")
for dt, v in t[(t.index>="2020-01-01")&(t.index<="2020-01-20")].items():
    print("  %s 周%s  %.6f" % (dt.date(), WD[dt.dayofweek], v))
print("--- Duka 同一段 ---")
for dt, v in dd[(dd.index>="2020-01-01")&(dd.index<="2020-01-20")].items():
    print("  %s 周%s  %.6f" % (dt.date(), WD[dt.dayofweek], v))

print("\n--- 某周 Tushare 逐日（2026-06-01~06-14）---")
for dt, v in t[(t.index>="2026-06-01")&(t.index<="2026-06-14")].items():
    print("  %s 周%s  %.6f" % (dt.date(), WD[dt.dayofweek], v))
print("--- Duka 同一段 ---")
for dt, v in dd[(dd.index>="2026-06-01")&(dd.index<="2026-06-14")].items():
    print("  %s 周%s  %.6f" % (dt.date(), WD[dt.dayofweek], v))
