# -*- coding: utf-8 -*-
"""核实 HL-Point 36 个名字在 ChanQuant 因子库里的命中情况（不带任何预设）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttrade import factors as F  # noqa: E402

ALL36 = [
    "dev_z_20", "dev_z_60", "dev_z_120", "dev_z_agg", "dev_spread",
    "up_sh", "dn_sh", "up_sh_atr", "dn_sh_atr", "up_sh_int", "dn_sh_int",
    "shadow_high_z", "shadow_low_z", "slope_atr", "r2", "slope_chg_atr",
    "dist_hi", "dist_lo", "ext_hi", "ext_lo", "ch_slope_atr",
    "hit_high", "hit_low", "clus_high", "clus_low", "level_high", "level_low",
    "oi_chg_5", "oi_chg_20", "oi_z_60", "oi_ratio_60", "oi_px_sync_5",
    "roll", "roll_ann", "curv", "roll_chg_20",
]

have = set(F.FACTORS)
hit = [k for k in ALL36 if k in have]
miss = [k for k in ALL36 if k not in have]
print("库内总数:", len(have))
print("命中:", len(hit), hit)
print("缺:", len(miss))
print()

# 逐类看库内现有分类，找可用现成件
from collections import Counter  # noqa: E402

cats = Counter(s.category for s in F.FACTORS.values())
print("分类:", dict(cats))
print()
for c in sorted(cats):
    ks = [k for k, s in F.FACTORS.items() if s.category == c]
    print(f"  {c}: {ks}")
