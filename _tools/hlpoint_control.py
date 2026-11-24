# -*- coding: utf-8 -*-
"""HL-Point 复现的「泄漏对照」：把每期截面分数随机打乱，看绩效会不会塌掉。

原理：截面策略的全部信息就是「同一时刻谁强谁弱」。若把每一行的分数重新洗牌
（**保留**「这一期有没有持仓、持几个」，只摧毁排序），真实 alpha 归零。这时
如果净值还在涨，说明涨的不是模型而是管线里的某种前视。

同时给出：
  · 原版 / 打乱版 / 反向版（分数取负）三组对照；
  · 按**调仓期**统计的期均收益与 t 值（不依赖 ChanQuant 的年化约定，
    和 HL 报告里的「400 期 +0.899%(t 4.84)」可直接比）。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_control.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import cross as X  # noqa: E402
from hl_point_register import build  # noqa: E402
from hlpoint_diag import load_frames  # noqa: E402


def make_ccfg(cfg, keys):
    return X.CrossConfig(
        mode=cfg.cross_mode, rebalance=cfg.cross_rebalance,
        rebalance_unit=cfg.cross_rebalance_unit,
        top_pct=cfg.cross_top_pct, bot_pct=cfg.cross_bot_pct,
        weight=cfg.cross_weight, weight_scope=cfg.cross_weight_scope,
        tiers=cfg.cross_tiers, pool_mode=cfg.cross_pool_mode,
        factor_mode=cfg.factor_mode, factor_keys=list(keys),
        factor_cfg=dict(cfg.factor_cfg), label_mode=cfg.cross_label_mode,
        period=cfg.period,
    )


def period_stats(eq: pd.Series, step: int) -> tuple:
    """按调仓期切净值，返回 (期数, 期均收益, t 值, 期胜率)。"""
    ret = eq.pct_change().fillna(0.0)
    seg = []
    for i in range(0, len(ret), step):
        chunk = ret.iloc[i:i + step]
        if len(chunk) > 1:
            seg.append(float((1.0 + chunk).prod() - 1.0))
    if len(seg) < 3:
        return 0, float("nan"), float("nan"), float("nan")
    a = np.asarray(seg)
    mu, sd = a.mean(), a.std(ddof=1)
    t = mu / (sd / np.sqrt(len(a))) if sd > 0 else float("nan")
    return len(a), float(mu), float(t), float((a > 0).mean())


def run_one(frames, ccfg, bt, shuf: str = ""):
    """跑一次；``shuf`` = "" / "shuffle" / "flip"。打分函数临时替换。"""
    orig = X.score_panel
    if shuf:
        def patched(*a, **k):
            sc = orig(*a, **k)
            v = sc.to_numpy(dtype=float).copy()
            if shuf == "flip":
                v = -v
            else:
                # 每一行（每个时间戳）独立打乱：保留「谁有值」，摧毁「谁强谁弱」
                rng = np.random.default_rng(20261123)
                for i in range(v.shape[0]):
                    rng.shuffle(v[i])
            return pd.DataFrame(v, index=sc.index, columns=sc.columns)
        X.score_panel = patched
    try:
        return X.run(frames, ccfg, bt)
    finally:
        X.score_panel = orig


def main() -> int:
    from ttrade import replay

    cfg0, _s, keys = build(False)
    ntest = int(os.environ.get("HP_N", "20"))
    years = float(os.environ.get("HP_YEARS", "2"))
    frames = load_frames(ntest, years)
    cfg = cfg0.clone(cross_symbols=list(frames), multi_symbols=list(frames))
    bt = replay.make_bt(replay.market_of(cfg), True, cfg)
    ccfg = make_ccfg(cfg, keys)

    print(f"品种 {len(frames)}，因子 {len(keys)}，每 {cfg.cross_rebalance} 根调仓")
    print()
    hdr = (f"{'变体':<10}{'总收益':>10}{'期数':>6}{'期均':>9}{'t 值':>8}"
           f"{'期胜率':>8}{'最大回撤':>10}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for tag, mode in (("原始", ""), ("打乱", "shuffle"), ("反向", "flip")):
        t0 = time.time()
        out = run_one(frames, ccfg, bt, mode)
        if not out.ok:
            print(f"{tag:<10} 失败: {out.error}")
            return 1
        eq = out.equity.dropna()
        n, mu, t, wr = period_stats(eq, max(1, out.step))
        dd = float((eq / eq.cummax() - 1.0).min())
        print(f"{tag:<10}{eq.iloc[-1] / eq.iloc[0] - 1:>+9.1%}{n:>6}"
              f"{mu:>+8.3%}{t:>8.2f}{wr:>8.1%}{dd:>10.1%}"
              f"   ({time.time() - t0:.0f}s)")
        rows.append((tag, eq))
    print()
    print("读法：打乱版若仍有明显正收益 ⇒ 管线里有前视（真实 alpha 已被摧毁，")
    print("      收益却还在）；反向版应显著为负 —— 三者方向一致才算干净。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
