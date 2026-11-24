# -*- coding: utf-8 -*-
"""同一批调仓期上，三种算法摆在一起对：净值到底从哪儿来。

(a) 逐根 Σ_t w[t-1]·ret[t]        ← 引擎实际用的（应等于净值）
(b) 逐期 Σ_i w[i]·fwd[i→i+H]      ← 权重恒定假设下的等价写法
(c) 逐期 0.5·top.mean − 0.5·bot.mean ← 原策略 decile_ls 的字面公式

三个必须在 1% 以内一致。任何一个大出一截，就是那一处的时钟/权重错位。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_three.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import cross as X  # noqa: E402
from hl_point_register import build  # noqa: E402
from hlpoint_control import make_ccfg  # noqa: E402
from hlpoint_diag import load_frames  # noqa: E402


def main() -> int:
    from ttrade import replay

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
    bt = replay.make_bt(replay.market_of(cfg), True, cfg)
    out = X.run(frames, ccfg, bt)

    w = out.weights.reindex(columns=score.columns).fillna(0.0)
    step = int(out.step)
    px = close.to_numpy(float)
    W = w.to_numpy(float)
    S = score.to_numpy(float)
    n, m = px.shape
    print(f"面板 {n}×{m}  步长 {step}  净值累计 {out.equity.iloc[-1] - 1:+.4f}")
    print(f"close 面板 NaN 占比 {np.mean(~np.isfinite(px)):.2%}")

    # (a) 逐根
    rets = np.full((n, m), np.nan)
    rets[1:] = px[1:] / px[:-1] - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)
    per_bar = (np.vstack([np.zeros((1, m)), W[:-1]]) * rets).sum(axis=1)
    print(f"(a) 逐根 Σ w[t-1]·ret[t]      累计 {per_bar.sum():+.4f}")

    starts = list(range(0, n - step, step))
    fwd = np.full((n, m), np.nan)
    fwd[:n - step] = px[step:] / px[:n - step] - 1.0

    a_blocks, b_blocks, c_blocks = [], [], []
    drop = []
    for i in starts:
        a_blocks.append(float(per_bar[i + 1:i + 1 + step].sum()))
        b_blocks.append(float(np.nansum(W[i] * fwd[i])))
        sel = W[i] != 0
        if sel.any():
            drop.append(float(np.sum(np.abs(W[i][sel]) * ~np.isfinite(fwd[i][sel]))
                              / np.sum(np.abs(W[i][sel]))))
        s = S[i]
        ok = np.isfinite(s)
        if ok.sum() >= 10:
            idx = np.flatnonzero(ok)
            order = idx[np.argsort(s[idx])]
            k = max(1, int(round(ok.sum() * 0.10)))
            f = fwd[i][idx]
            top, bot = order[-k:], order[:k]
            c_blocks.append(0.5 * np.nanmean(fwd[i][top]) - 0.5 * np.nanmean(fwd[i][bot]))
        else:
            c_blocks.append(np.nan)

    a = np.asarray(a_blocks)
    b = np.asarray(b_blocks)
    c = np.asarray(c_blocks, dtype=float)
    print(f"(a) 逐期块累加              累计 {a.sum():+.4f}  期均 {a.mean():+.3%}")
    print(f"(b) Σ w[i]·fwd[i]           累计 {b.sum():+.4f}  期均 {b.mean():+.3%}")
    print(f"(c) 0.5(top−bot) 字面公式    累计 {np.nansum(c):+.4f}  期均 {np.nanmean(c):+.3%}")
    print(f"每期因 fwd 为 NaN 丢掉的权重占比：中位 {np.median(drop):.1%} "
          f"最大 {np.max(drop):.1%}")

    print()
    print("同批期逐期对照（前 12 期，单位 bp）")
    print(f"{'期':>4}{'i':>6}{'(a)bp':>10}{'(b)bp':>10}{'(c)bp':>10}")
    for j in range(min(12, len(starts))):
        print(f"{j:>4}{starts[j]:>6}{a[j]*1e4:>10.1f}{b[j]*1e4:>10.1f}{c[j]*1e4:>10.1f}")

    # ---- 引擎净值到底怎么来的：pandas 一条、numpy 一条，并列 ----
    print()
    nan_cells = int(np.sum(~np.isfinite(px)))
    print(f"close 面板 NaN 单元 {nan_cells} 个 / {n * m}")
    pr_pd = (w.shift(1) * close.pct_change().fillna(0.0)).sum(axis=1)
    pr_np = (np.vstack([np.zeros((1, m)), W[:-1]]) * rets).sum(axis=1)
    print(f"pandas: Σ (w.shift(1)·pct_change)    累计 {float(pr_pd.sum()):+.4f}")
    print(f"numpy : Σ (W[:-1]·rets)            累计 {float(pr_np.sum()):+.4f}")
    print(f"引擎 equity 累计                   {out.equity.iloc[-1] - 1:+.4f}")
    print(f"out.weights 形状 {out.weights.shape}  index 与 score 相同? "
          f"{out.weights.index.equals(score.index)}")
    print(f"out.equity 形状 {out.equity.shape}  首尾 {out.equity.iloc[0]:.4f} "
          f"{out.equity.iloc[-1]:.4f}")

    # 关键判据
    print()
    if abs(b.sum() - a.sum()) / max(abs(a.sum()), 1e-9) < 0.05:
        print("✅ (a)≈(b) —— 权重恒定假设成立，引擎逐根累计没问题")
    else:
        print("🔴 (a)≠(b) —— 引擎逐根累计与「权重×持有期收益」不等价")
    print(f"   (a) 与 equity 的差额 {(a.sum() - (out.equity.iloc[-1] - 1)):+.4f}")
    print(f"   (c) 字面 decile 公式 vs (b) 引擎权重：{np.nanmean(c):+.4%} vs {b.mean():+.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
