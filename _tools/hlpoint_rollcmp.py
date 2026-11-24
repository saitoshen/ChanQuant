# -*- coding: utf-8 -*-
"""展期因子对拍：原项目 `roll_yield_panel_all87.csv` vs 库里 `roll_main_oi`。

IC 符号整体反了，但两边的**选腿逻辑和公式都一致**（都是「OI 最大 = 近腿，
其后 OI 最大 = 远腿，roll=(near-far)/far」）。那就只剩两种可能：
  ① 因子值本身对不上（数据源 / 平滑 / 滞后）；
  ② 对得上，但 IC 差异是品种池 / 样本期的噪声。
这里直接比数值，给出相关系数，一句话定性。
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "D:/ChanQuant-Dev")
sys.path.insert(0, "D:/ChanQuant-Dev/_tools")

from ttrade import auxdata, factors as F, tqdata  # noqa: E402

ORIG = "D:/CozeTask/future_quant/data/roll_yield_panel_all87.csv"
SYMS = [("RB", "SHFE"), ("CU", "SHFE"), ("M", "DCE"), ("TA", "CZCE"),
        ("AL", "SHFE"), ("JM", "DCE")]

p = pd.read_csv(ORIG, parse_dates=["trade_date"])
p["variety"] = p["variety"].astype(str).str.upper()

for sym, ex in SYMS:
    df = tqdata.load_main(sym, ex, "1h", adjust="qfq")
    df = auxdata.attach_for_period(df, sym, ex, "1h")
    idx = df.index
    hour = idx.hour.to_numpy()
    day = idx.normalize()
    keep = (hour >= 8) & (hour < 16)
    pos = np.flatnonzero(keep)
    d = day[pos].to_numpy()
    last = np.append(d[1:] != d[:-1], True)
    pos = pos[last]
    sday = day[pos]
    mine = pd.Series(
        pd.to_numeric(F.FACTORS["roll_main_oi"].func(df), errors="coerce").to_numpy(float)[pos],
        index=sday,
    )
    sub = p[p["variety"] == sym.upper()].set_index("trade_date")
    both = pd.DataFrame({"orig": sub["roll"], "mine": mine}).dropna()
    both = both[both.index >= "2017-01-01"]
    if len(both) < 50:
        print(f"{sym}: 重叠 {len(both)} 天，跳过")
        continue
    c = both.corr().iloc[0, 1]
    sp = both.corr(method="spearman").iloc[0, 1]
    # 滞后一天后的相关性（原项目 shift(1)，库当日）
    c1 = both["orig"].corr(both["mine"].shift(1))
    print(f"{sym:<4} n={len(both):>5}  Pearson {c:+.3f}  Spearman {sp:+.3f}  "
          f"上日错位 {c1:+.3f} | 原均值 {both['orig'].mean():+.4f} 库均值 {both['mine'].mean():+.4f}")
    print(f"     前 3 行 orig={both['orig'].head(3).round(4).tolist()} "
          f"mine={both['mine'].head(3).round(4).tolist()}")
