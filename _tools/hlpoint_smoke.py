# -*- coding: utf-8 -*-
"""HL-Point（1h / 36 因子 / 十分位多空）在 ChanQuant 上的冒烟回测。

拉一批商品的 1h 主连，挂上限期结构 aux，跑 ``replay.run_cross``，打印
收益 / 夏普 / 调仓次数 / 每期腿数 / 因子有效数。

用法：
    C:/ProgramData/anaconda3/python.exe _tools/hlpoint_smoke.py
    C:/ProgramData/anaconda3/python.exe _tools/hlpoint_smoke.py --years 3 --n 24
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttrade import auxdata, replay, tqdata  # noqa: E402

#: 跨板块代表池（原项目 DEFAULT_POOL 的扩大版，保证每个时点截面 >= 10 个品种）
POOL = [
    ("RB", "SHFE"), ("I", "DCE"), ("HC", "SHFE"), ("J", "DCE"), ("JM", "DCE"),
    ("CU", "SHFE"), ("AL", "SHFE"), ("ZN", "SHFE"), ("NI", "SHFE"), ("SN", "SHFE"),
    ("AU", "SHFE"), ("AG", "SHFE"),
    ("SC", "INE"), ("FU", "SHFE"), ("BU", "SHFE"), ("PG", "DCE"),
    ("M", "DCE"), ("Y", "DCE"), ("P", "DCE"), ("OI", "CZCE"), ("RM", "CZCE"),
    ("C", "DCE"), ("CS", "DCE"), ("SR", "CZCE"), ("CF", "CZCE"), ("AP", "CZCE"),
    ("TA", "CZCE"), ("MA", "CZCE"), ("EG", "DCE"), ("PP", "DCE"), ("L", "DCE"),
    ("V", "DCE"), ("RU", "SHFE"), ("SA", "CZCE"), ("FG", "CZCE"), ("UR", "CZCE"),
]


def load_frames(years: float, n: int) -> dict:
    out = {}
    t0 = time.time()
    for sym, ex in POOL[:n]:
        try:
            df = tqdata.load_main(sym, ex, "1h", adjust="qfq")
        except Exception as e:  # noqa: BLE001
            print(f"  {sym} 读取失败: {e}")
            continue
        if df is None or len(df) < 500:
            print(f"  {sym} 样本不足（{0 if df is None else len(df)}）")
            continue
        df = auxdata.attach_for_period(df, sym, ex, "1h")
        if years:
            cut = df.index[-1] - pd.Timedelta(days=int(365 * years))
            df = df[df.index >= cut]
        out[f"{sym}_{ex}"] = df
    print(f"载入 {len(out)} 个品种，耗时 {time.time() - t0:.1f}s")
    return out


def main() -> int:
    years = 2.0
    n = 99
    for i, a in enumerate(sys.argv):
        if a == "--years":
            years = float(sys.argv[i + 1])
        if a == "--n":
            n = int(sys.argv[i + 1])

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "_tools"))
    from hl_point_register import build  # noqa: E402

    cfg, syms, keys = build(False)
    frames = load_frames(years, n)
    if len(frames) < 10:
        print("品种太少，十分位分不出来")
        return 1

    # 只保留真正有数据的品种
    cfg = cfg.clone(cross_symbols=list(frames), multi_symbols=list(frames))

    lens = [len(v) for v in frames.values()]
    print(f"每品种 bar 数：中位 {int(np.median(lens))}、最少 {min(lens)}、最多 {max(lens)}")
    print(f"因子 {len(keys)} 个，其中带衍生数据的："
          f"{sum(1 for k in ('roll_main_oi','roll_main_annual','roll_main_curv','roll_chg_20') if k in keys)}")

    t0 = time.time()
    res = replay.run_cross(cfg, frames)
    dt = time.time() - t0
    if getattr(res, "error", ""):
        print("🔴 回测失败:", res.error)
        return 1

    eq = res.res.equity.dropna() if res.res is not None else pd.Series(dtype=float)
    if eq.empty:
        print("🔴 净值是空的:", res.error or "未知原因")
        return 1
    m = (res.res.metrics if res.res is not None else {}) or {}
    print(f"\n回测耗时 {dt:.1f}s")
    print(f"净值：{eq.iloc[0]:.4f} → {eq.iloc[-1]:.4f}   总收益 {eq.iloc[-1] / eq.iloc[0] - 1:+.2%}")
    for k, v in list(m.items())[:16]:
        print(f"  {k:<16}{v}")
    cr = res.cross
    print(f"调仓 {getattr(cr, 'n_rebalance', 0)} 次，"
          f"每 {getattr(cr, 'step', 0)} 根，最近一次腿数 {getattr(cr, 'legs', 0)}")
    print(f"交易记录 {len(getattr(cr, 'trades', []))} 条")

    w = getattr(cr, "weights", None)
    if w is not None and len(w):
        nz = (w.abs() > 0).sum(axis=1)
        print(f"每期持仓品种数：中位 {int(nz.median())}、最大 {int(nz.max())}")
        net = w.sum(axis=1)
        print(f"净敞口 |Σw|：中位 {float(net.abs().median()):.4f}、最大 {float(net.abs().max()):.4f}"
              "   ← legs 口径应≈0")
        print(f"Σ|w|：中位 {float(w.abs().sum(axis=1).median()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
