# -*- coding: utf-8 -*-
"""原项目的 `adj='v2'` 后复权，到底把换月跳空消掉了多少？

用户 11-24 的追问把这里逼出来了：原项目回测 **不是** 用不复权 888，而是
`data_feed.load_bars(..., adj='v2')`。那么「不复权 IC ≈ 原报表」就不能当因果。
真问题是：**v2 复权因子有没有真的把 888 换月跳空抹平？**

做法：同一品种、同一段，取两次 bar（`adj='v2'` 与 `adj=None`），比较
  · `r_v2`   = v2 复权后的 h 根收益
  · `r_raw`  = 原始收益
  · `d      = r_raw − r_v2`   ← v2 **没有**消掉的那部分跳空
若 v2 干净，`d` 应该恒为 0；若 v2 完全没起作用，`d` 就等于全额跳空。

同时算 IC(roll, r_v2) / IC(roll, r_raw) / IC(roll, d)，与 24 品种那套
（`hlpoint_revadj_probe.py`）对照。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_v2adj_probe.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "D:/CozeTask/future_quant")

from scipy.stats import spearmanr  # noqa: E402
from strategy.hl_point import data_feed as D  # noqa: E402

SYMS = ["RB888", "CU888", "M888", "TA888", "I888", "AU888"]
START, END = "2019-01-01", "2026-10-14"
H = 30           # 原项目写死 BPD=6 → 30 根


def snap(idx: pd.DatetimeIndex) -> np.ndarray:
    hour = idx.hour.to_numpy()
    keep = (hour >= 8) & (hour < 16)
    pos = np.flatnonzero(keep)
    d = idx.normalize().to_numpy()[pos]
    return pos[np.append(d[1:] != d[:-1], True)]


def main() -> int:
    rows = []
    for s in SYMS:
        try:
            bv = D.load_bars(s, "1h", START, END, adj="v2", use_cache=True)
            br = D.load_bars(s, "1h", START, END, adj=None, use_cache=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{s}: 取数失败 {type(exc).__name__}: {exc}")
            continue
        if bv is None or br is None:
            print(f"{s}: 空")
            continue
        n = min(len(bv), len(br))
        cv = pd.to_numeric(bv["close"], errors="coerce").to_numpy(float)[:n]
        cr = pd.to_numeric(br["close"], errors="coerce").to_numpy(float)[:n]
        pos = snap(bv.index[:n])
        if len(pos) < 100:
            print(f"{s}: 快照太少 {len(pos)}")
            continue
        p2 = pos + H
        ok = p2 < n
        rv = np.full(len(pos), np.nan)
        rr = np.full(len(pos), np.nan)
        rv[ok] = cv[p2[ok]] / cv[pos[ok]] - 1.0
        rr[ok] = cr[p2[ok]] / cr[pos[ok]] - 1.0
        d = rr - rv
        # 复权因子在两口径下的差异本身（比例）
        fv = pd.to_numeric(bv.get("adj_factor"), errors="coerce").to_numpy(float)[:n] \
            if "adj_factor" in bv.columns else np.full(n, np.nan)
        rows.append(dict(
            sym=s, n_bars=n, n_snap=int(ok.sum()),
            ratio_cv_cr=float(np.nanmedian(cv / np.where(cr == 0, np.nan, cr))),
            adj_moves=int(np.nansum(np.abs(np.diff(fv)) > 0)) if np.isfinite(fv).any() else -1,
            gap_nonzero=float(np.nanmean(np.abs(d) > 1e-9)),
            gap_abs_mean=float(np.nanmean(np.abs(d))),
            gap_ic=float(spearmanr(rv[np.isfinite(d)], d[np.isfinite(d)]).statistic)
            if np.isfinite(d).sum() > 30 else np.nan,
        ))
        print(f"{s}: bars={n} 快照={int(ok.sum())} "
              f"|d|均值={np.nanmean(np.abs(d)) * 100:.3f}% "
              f"非零={100 * np.nanmean(np.abs(d) > 1e-9):.1f}% "
              f"复权因子跳变={rows[-1]['adj_moves']}")

    if not rows:
        print("一个都没取到")
        return 1
    det = pd.DataFrame(rows)
    det.to_csv("D:/ChanQuant-Dev/reports/hlpoint_v2adj.csv", index=False,
               encoding="utf-8-sig")
    print("\n" + det.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
