# -*- coding: utf-8 -*-
"""直接调 cross.run，检查权重 / Σ|w| / 逐期收益，定位净值爆炸的原因。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_diag.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import auxdata, cross, replay, tqdata  # noqa: E402
from hl_point_register import build  # noqa: E402
from hlpoint_smoke import POOL  # noqa: E402


def load_frames(n: int, years: float) -> dict:
    out = {}
    for sym, ex in POOL[:n]:
        try:
            df = tqdata.load_main(sym, ex, "1h", adjust="qfq")
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 500:
            continue
        df = auxdata.attach_for_period(df, sym, ex, "1h")
        if years:
            cut = df.index[-1] - pd.Timedelta(days=int(365 * years))
            df = df[df.index >= cut]
        out[f"{sym}_{ex}"] = df
    return out


def main() -> int:
    cfg0, _s, keys = build(False)
    frames = load_frames(20, 2.0)
    cfg = cfg0.clone(cross_symbols=list(frames), multi_symbols=list(frames))
    panel = cross.close_panel(frames)
    bt = replay.make_bt(replay.market_of(cfg), True, cfg)

    ccfg = cross.CrossConfig(
        mode=cfg.cross_mode, rebalance=cfg.cross_rebalance,
        rebalance_unit=cfg.cross_rebalance_unit,
        top_pct=cfg.cross_top_pct, bot_pct=cfg.cross_bot_pct,
        weight=cfg.cross_weight, weight_scope=cfg.cross_weight_scope,
        tiers=cfg.cross_tiers, pool_mode=cfg.cross_pool_mode,
        factor_mode=cfg.factor_mode, factor_keys=list(keys),
        factor_cfg=dict(cfg.factor_cfg), label_mode=cfg.cross_label_mode,
        period=cfg.period,
    )
    print("bt: long_only=", bt.long_only, "short_only=", bt.short_only,
          "lev=", getattr(bt, "leverage", None), "capital_pct=",
          getattr(bt, "capital_pct", None))

    out = cross.run(frames, ccfg, bt)
    if not out.ok:
        print("🔴", out.error)
        return 1
    print(f"状态: ok  调仓 {out.n_rebalance} 次  每 {out.step} 根  legs={out.legs}"
          f"  bars_per_day={out.bars_per_day}")
    print("metrics:", {k: round(v, 6) if isinstance(v, float) else v
                       for k, v in (out.metrics or {}).items()})

    w = out.weights
    nz = (w.abs() > 1e-12).sum(axis=1)
    sabs = w.abs().sum(axis=1)
    net = w.sum(axis=1)
    print(f"\n权重形状 {w.shape}")
    print(f"每行非零品种数：中位 {nz.median():.0f} 最大 {nz.max()} 最小 {nz.min()}")
    print(f"Σ|w|：中位 {sabs.median():.4f} 最大 {sabs.max():.4f}")
    print(f"净敞口 Σw：中位 {net.median():+.4f} 最大 {net.abs().max():.4f}")
    print("\n最近 6 次调仓的权重（非零）：")
    rows = w[(w.abs() > 1e-12).any(axis=1)].tail(6)
    for ts, r in rows.iterrows():
        nzr = r[r.abs() > 1e-12]
        print(f"  {ts}  " + "  ".join(f"{k}={v:+.4f}" for k, v in nzr.items()))

    eq = out.equity.dropna()
    ret = eq.pct_change().dropna()
    print(f"\n净值 {eq.iloc[0]:.4f} → {eq.iloc[-1]:.4f} ({eq.iloc[-1] / eq.iloc[0] - 1:+.1%})")
    print(f"逐根收益：均值 {ret.mean():+.6f} 标准差 {ret.std():.6f} "
          f"最大 {ret.max():+.4f} 最小 {ret.min():+.4f}")
    worst = ret.abs().nlargest(5)
    print("5 个最大单根跳动：")
    for ts, v in worst.items():
        print(f"  {ts}  {v:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
