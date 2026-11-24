# -*- coding: utf-8 -*-
"""复权口径探针：qfq vs 不复权，同一批因子、同一份 IC 管线。

2×2 归因已经证明「库的因子函数 ≈ 原项目」（r +0.86、35/36 同号、逐条差 <0.001），
差异只能来自数据侧。最可疑的一条是 **复权**：

- 原项目用 **不复权** 的 888 主连（换月当场跳空，跳空的幅度**正好等于展期价差**）；
- 库里 `tqdata.load_main(..., adjust="qfq")` 把换月跳空拉平了。

如果 hold 期内跨过一次换月，不复权口径下的收益就**被展期结构机械污染**：
carry 越正（back），换月时价格越是往下跳 ⇒ 收益与 carry 负相关。
这足以把 `roll` 的 IC 从 +0.03 翻成 −0.03。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_adj_probe.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/d/ChanQuant-Dev")
sys.path.insert(0, "D:/ChanQuant-Dev")
sys.path.insert(0, "D:/ChanQuant-Dev/_tools")

from ttrade import auxdata, factors as F, tqdata  # noqa: E402

from hlpoint_ic_cmp import (HS, MINW, START, END, snap_positions,  # noqa: E402
                            xs_ic_table)
from hlpoint_full import all_symbols  # noqa: E402

ONLY = ["roll_main_oi", "roll_main_annual", "roll_main_curv", "roll_chg_20",
        "dist_hi", "dev_z_60", "shadow_high_z"]
NSYM = 24


def build(adjust) -> tuple[dict, int]:
    store: dict = {}
    n = 0
    for sym, ex, _ in all_symbols()[:NSYM]:
        try:
            df = tqdata.load_main(sym, ex, "1h", adjust=adjust)
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 500:
            continue
        df = auxdata.attach_for_period(df, sym, ex, "1h")
        pos = snap_positions(df.index)
        if len(pos) < 50:
            continue
        sday = df.index.normalize().to_numpy()[pos]
        m = (sday >= START.to_datetime64()) & (sday <= END.to_datetime64())
        if m.sum() < 50:
            continue
        pos, sday = pos[m], pd.DatetimeIndex(sday[m])
        for k in ONLY:
            v = pd.to_numeric(F.FACTORS[k].func(df), errors="coerce").to_numpy(float)
            store.setdefault(k, {})[sym] = pd.Series(v[pos], index=sday)
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
        for h in HS:
            p2 = pos + h
            ok = p2 < len(close)
            fwd = np.full(len(pos), np.nan)
            fwd[ok] = close[p2[ok]] / close[pos[ok]] - 1.0
            store.setdefault(f"__y{h}__", {})[sym] = pd.Series(fwd, index=sday)
        n += 1
    return store, n


def main() -> int:
    res = {}
    for adj in ("qfq", None):
        store, n = build(adj)
        print(f"adjust={adj!r}: {n} 个品种")
        res[str(adj)] = {h: xs_ic_table(store, h) for h in HS}

    print(f"\n{'因子':<18}{'qfq H46':>10}{'t':>7}{'不复权 H46':>11}{'t':>7}"
          f"{'qfq H30':>10}{'不复权 H30':>11}")
    print("-" * 76)
    for k in ONLY:
        r1, r2 = res["qfq"][46], res["None"][46]
        r3, r4 = res["qfq"][30], res["None"][30]
        if k not in r1.index:
            continue
        print(f"{k:<18}{r1.loc[k,'ic']:>+10.4f}{r1.loc[k,'t']:>7.2f}"
              f"{r2.loc[k,'ic']:>+11.4f}{r2.loc[k,'t']:>7.2f}"
              f"{r3.loc[k,'ic']:>+10.4f}{r4.loc[k,'ic']:>+11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
