# -*- coding: utf-8 -*-
"""独立复算：绕开 cross.run 的组合引擎，直接用分数面板 + 前向收益算十分位多空。

为什么要这一步：`cross.run` 那一整条路径（权重矩阵 / shift / 成本 / 单利累计）
任何一处错了，都会让绩效看起来「很好」而打乱对照却照样干净（打乱只验证
「排序有没有用」，不验证「算得对不对」）。这里用**最朴素的方式**重算一遍：

    每个调仓时点 t：按分数排序，取前 k 做多、后 k 做空（等权、各占 50%），
    持有 H 根，收益 = mean(top 的未来 H 根收益) × 0.5 − mean(bot 的) × 0.5

和引擎的期均收益对比，差异应在一两个 bp 量级（引擎多扣了成本、权重有
边界处理）。顺带给出**逐期横截面 rank IC**（和 HL 报告的 0.025 量级对照）。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_repl.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.stats import spearmanr  # noqa: E402

from ttrade import cross as X  # noqa: E402
from hl_point_register import build  # noqa: E402
from hlpoint_control import make_ccfg  # noqa: E402
from hlpoint_diag import load_frames  # noqa: E402


def main() -> int:
    ntest = int(os.environ.get("HP_N", "36"))
    cfg0, _s, keys = build(False)
    frames = load_frames(ntest, 2.0)
    cfg = cfg0.clone(cross_symbols=list(frames), multi_symbols=list(frames))
    ccfg = make_ccfg(cfg, keys)

    close = X.close_panel(frames)
    score = X.score_panel(frames, list(keys), {}, mode="ml",
                          fcfg=X._factor_cfg(ccfg),
                          label_mode=ccfg.label_mode, base_period=ccfg.period)
    score = score.dropna(how="all")
    close = close.reindex(score.index).ffill()
    # 🔴 必须按**列名**对齐成同一个顺序：`close_panel` 的列序是 `frames` 的插入序，
    # `score.columns` 是因子算完之后重建的顺序，两者不一定相同。直接各自
    # `.to_numpy()` 再相乘就是按位置配品种，会把 A 的收益记到 B 头上 ——
    # 静默给出一个看起来像「没有 alpha」的假结论。
    close = close.reindex(columns=score.columns)
    print(f"分数面板 {score.shape}  {score.index[0]} ~ {score.index[-1]}")
    assert list(close.columns) == list(score.columns)

    H = int(cfg.cross_rebalance)
    bpd = X.fstrat.bars_per_day(score.index) if hasattr(X, "fstrat") else 9.28
    step = ccfg.step(bpd)
    print(f"持有期 H={H} 根，调仓步长 step={step} 根，密度 {bpd:.2f} 根/天")

    px = close.to_numpy(float)
    fwd = np.full_like(px, np.nan)
    fwd[:-H] = px[H:] / px[:-H] - 1.0           # t 处 = t → t+H 的收益

    S = score.to_numpy(float)
    rows = range(0, len(score), step)

    ics, spreads, legs_n = [], [], []
    for i in rows:
        s, f = S[i], fwd[i]
        m = np.isfinite(s) & np.isfinite(f)
        if m.sum() < 10:
            continue
        k = m.sum() // 10
        if k < 1:
            continue
        idx = np.flatnonzero(m)
        order = idx[np.argsort(s[idx])]
        bot, top = order[:k], order[-k:]
        ic = spearmanr(s[idx], f[idx]).statistic
        if np.isfinite(ic):
            ics.append(float(ic))
        # 原策略口径：1.0×top − 1.0×bot（gross 200%）。ChanQuant legs 是各 0.5
        spreads.append(0.5 * float(f[top].mean()) - 0.5 * float(f[bot].mean()))
        legs_n.append(k)

    a = np.asarray(spreads, dtype=float)
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
    print()
    print(f"独立复算（绕开组合引擎）：")
    print(f"  期数 {len(a)}   每腿 {int(np.median(legs_n))} 个   覆盖率 "
          f"{len(a) / len(list(rows)):.0%}（前段无预测被跳过）")
    print(f"  期均收益 {a.mean():+.3%}   期标准差 {a.std(ddof=1):.3%}   t 值 {t:.2f}"
          f"   胜率 {(a > 0).mean():.1%}")
    print(f"  逐期横截面 rank IC：均值 {np.mean(ics):+.4f}、"
          f"标准差 {np.std(ics):.4f}、t 值 {np.mean(ics) / (np.std(ics) / np.sqrt(len(ics))):.2f}")
    print(f"  累计（单利求和）{a.sum():+.1%}   —— 引擎应略低（多扣成本）")

    # 和引擎对比
    from ttrade import replay
    bt = replay.make_bt(replay.market_of(cfg), True, cfg)
    out = X.run(frames, ccfg, bt)
    if out.ok:
        eq = out.equity.dropna()
        ret = eq.pct_change().fillna(0.0)
        seg = [float((1.0 + ret.iloc[i:i + out.step]).prod() - 1.0)
               for i in range(0, len(ret), out.step) if len(ret.iloc[i:i + out.step]) > 1]
        seg = np.asarray(seg, dtype=float)
        print()
        print(f"引擎（cross.run）：期数 {len(seg)}  期均 {seg.mean():+.3%}  "
              f"t {seg.mean() / (seg.std(ddof=1) / np.sqrt(len(seg))):.2f}  "
              f"累计 {eq.iloc[-1] / eq.iloc[0] - 1:+.1%}  成本 {out.metrics.get('cost_pct', 0):.2%}")
        print(f"差异（引擎 − 独立）{(seg.mean() - a.mean()):+.4%}/期")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
