# -*- coding: utf-8 -*-
"""逐期比对：引擎选出来的多空腿 vs 按分数自己排的 top/bot。

独立复算说「分数没有 alpha（IC 0.006）」，引擎却说「每期 +1.4%」。要么
①引擎选的品种不是我排的那批，要么②引擎算收益的时钟错位。这里把两者的
每一期持仓摆在一起看。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_legs.py
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

    from ttrade import replay
    bt = replay.make_bt(replay.market_of(cfg), True, cfg)
    out = X.run(frames, ccfg, bt)
    if not out.ok:
        print("🔴", out.error)
        return 1
    w = out.weights.reindex(columns=score.columns).fillna(0.0)
    print(f"分数面板 {score.shape}  权重矩阵 {w.shape}  步长 {out.step}")

    S = score.to_numpy(float)
    W = w.to_numpy(float)
    cols = np.asarray(score.columns)
    # 只看权重矩阵里真的发生过变化的那一行（调仓行）
    chg = np.flatnonzero((np.abs(np.diff(W, axis=0)).sum(axis=1) > 1e-12)) + 1
    print(f"权重变化行 {len(chg)} 个，前 5 个位置 {chg[:5]}")

    hit = 0
    tot = 0
    for i in chg[:400]:
        s = S[i]
        m = np.isfinite(s)
        if m.sum() < 10:
            continue
        idx = np.flatnonzero(m)
        order = idx[np.argsort(s[idx])]
        k = m.sum() // 10
        my_top, my_bot = set(order[-k:]), set(order[:k])
        eng_top = set(np.flatnonzero(W[i] > 1e-12))
        eng_bot = set(np.flatnonzero(W[i] < -1e-12))
        tot += 1
        if my_top == eng_top and my_bot == eng_bot:
            hit += 1
        elif tot <= 5:
            print(f"  行 {i}  {score.index[i]}")
            print(f"    我的多 {[cols[j] for j in sorted(my_top)]}")
            print(f"    引擎多 {[cols[j] for j in sorted(eng_top)]}")
            print(f"    我的空 {[cols[j] for j in sorted(my_bot)]}")
            print(f"    引擎空 {[cols[j] for j in sorted(eng_bot)]}")
    print(f"\n持仓完全一致：{hit}/{tot}")

    # 关键：看这一行的前向收益 vs 该行收益
    H = int(out.step)
    px = close.to_numpy(float)
    fwd = np.full_like(px, np.nan)
    fwd[:-H] = px[H:] / px[:-H] - 1.0
    rets = np.vstack([np.full((1, px.shape[1]), np.nan), px[1:] / px[:-1] - 1.0])

    print("\n逐行核对（取 3 个调仓行）")
    for i in chg[40:43]:
        if i + H >= len(px):
            continue
        wo = W[i - 1] if i > 0 else W[i]
        held_from_i = W[i]
        r_fwd = np.nansum(fwd[i] * held_from_i)          # 从 i 持有到 i+H
        r_eng = np.nansum(rets[i] * wo)                  # 引擎：i-1 的权重吃 i 的收益
        r_fwd_shift = np.nansum(fwd[i] * wo)
        print(f"  行 {i} {score.index[i]}")
        print(f"    Σ(w[i]·fwd[i→i+H])        = {r_fwd:+.4f}")
        print(f"    Σ(w[i-1]·ret[i])  （引擎） = {r_eng:+.4f}")
        print(f"    Σ(w[i-1]·fwd[i→i+H])      = {r_fwd_shift:+.4f}")

    # 引擎总收益 vs 用权重矩阵自己重算
    rets_df = close.pct_change().fillna(0.0)
    my_ret = (w.shift(1) * rets_df).sum(axis=1)
    print(f"\n引擎净值累计 {out.equity.iloc[-1] - 1:+.4f}")
    print(f"用 out.weights 重算 (w.shift(1)·ret) 累计 {my_ret.sum():+.4f}")
    print(f"  ⇒ 权重矩阵本身能吃出这么多？{abs(my_ret.sum() - (out.equity.iloc[-1] - 1)) < 0.05}")

    # ★ 决定性：每一期 w[i]·fwd[i→i+H] 累加，应≈净值累计（权重在期内恒定）
    step = int(out.step)
    per = []
    for i in range(0, len(px), step):
        if i + step >= len(px):
            break
        per.append(float(np.nansum(W[i] * fwd[i])))
    per = np.asarray(per, dtype=float)
    print(f"\n★ 逐期 Σ(w[i]·fwd[i→i+H])：{len(per)} 期，累加 {per.sum():+.4f}，"
          f"期均 {np.nanmean(per):+.4%}")
    print(f"  引擎单利累计 {out.equity.iloc[-1] - 1:+.4f}   "
          f"差额 {per.sum() - (out.equity.iloc[-1] - 1):+.4f}")

    # ★ 同样的期，但把权重整体滞后一期（w[i-1]），泄漏会在这里塌掉
    per_lag = []
    for i in range(0, len(px) - step, step):
        j = max(1, i)
        per_lag.append(float(np.nansum(W[j - 1] * fwd[j])))
    per_lag = np.asarray(per_lag, dtype=float)
    print(f"★ 换成 w[i-1] 选股：期均 {np.nanmean(per_lag):+.4%}，累加 {per_lag.sum():+.4f}")
    for s in (1, 2, 5, 10, 46):
        k = []
        for i in range(0, len(px) - step, step):
            j = min(len(px) - 1, i + s)
            k.append(float(np.nansum(W[i] * fwd[j])))
        print(f"   fwd 起点后移 {s:>2} 根：期均 {np.nanmean(k):+.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
