# -*- coding: utf-8 -*-
"""配置变体扫描：找出「为什么复现版没有 alpha」。

在**全部品种 × 全历史**上逐项改一个参数，看期均 / t / IC 怎么动：

  1. ``orient``        auto（按滚动 IC 定方向）/ long_high（固定「分高做多」）
  2. ``horizon``       46 根（实测密度 5 天）/ 30 根（原项目写死 BPD=6）
  3. ``label_mode``    excess（收益减截面均值）/ volret（再除波动）

🔴 本机内存只有 4GB，68 品种 × 2.4 万根跑一次 `score_panel` 就顶到上限。
所以：
  · 帧只保留因子真正要用的列（`KEEP_COLS`）；
  · **一次进程只跑一个变体**（`HP_VARIANT=i`），由外面循环调用，
    避免六个变体的内存碎片叠在一起。

用法：
    for i in 0 1 2 3 4 5; do HP_VARIANT=$i python -u _tools/hlpoint_sweep.py; done
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import cross as X, replay  # noqa: E402
from hl_point_register import build  # noqa: E402
from hlpoint_control import make_ccfg, period_stats  # noqa: E402
from hlpoint_full import load_all  # noqa: E402

VARIANTS = [
    ("long_high / H=46", {"orient": "long_high"}, 46, "excess"),
    ("long_high / H=30", {"orient": "long_high"}, 30, "excess"),
    ("long_high / volret", {"orient": "long_high"}, 46, "volret"),
    ("auto / H=46（仅对照）", {"orient": "auto"}, 46, "excess"),
]

#: 34 个因子只碰这些列（多留几个无害，主要是砍掉用不上的衍生列省内存）
KEEP = ("open", "high", "low", "close", "volume", "open_interest", "open_oi",
        "trading_day", "price_scale", "adj_factor")
KEEP_PRE = ("o6_", "o3_", "c1_", "c2_", "c3_")


def trim(frames: dict) -> dict:
    out = {}
    for k, df in frames.items():
        cols = [c for c in df.columns
                if c in KEEP or any(c.startswith(p) for p in KEEP_PRE)]
        out[k] = df[cols].copy()
    return out


def xs_ic(panel: pd.DataFrame, fwd: pd.DataFrame, every: int = 10) -> tuple:
    from scipy.stats import spearmanr
    ics = []
    for i in range(0, len(panel), every):
        s = panel.iloc[i].to_numpy(float)
        f = fwd.iloc[i].to_numpy(float)
        m = np.isfinite(s) & np.isfinite(f)
        if m.sum() < 10:
            continue
        ic = spearmanr(s[m], f[m]).statistic
        if np.isfinite(ic):
            ics.append(float(ic))
    if len(ics) < 5:
        return float("nan"), float("nan"), 0
    a = np.asarray(ics)
    return float(a.mean()), float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), len(a)


def main() -> int:
    idx = int(os.environ.get("HP_VARIANT", "0"))
    tag, patch, hz, lab = VARIANTS[idx]
    maxn = int(os.environ.get("HP_MAXN", "0") or 0)

    cfg0, _s, keys = build(False)
    frames = trim(load_all(maxn))
    cfg = cfg0.clone(cross_symbols=list(frames), multi_symbols=list(frames),
                     cross_rebalance=hz, cross_label_mode=lab)
    fcfg = dict(cfg.factor_cfg)
    fcfg.update(patch)
    fcfg["horizon"] = hz
    cfg = cfg.clone(factor_cfg=fcfg)
    ccfg = make_ccfg(cfg, keys)
    bt = replay.make_bt(replay.market_of(cfg), True, cfg)

    t0 = time.time()
    # 🔴 只算一次分数面板：`replay.run_cross` 会把 `(frames, ccfg, score)` 一起
    # 塞在 `Replay.cross` 里。再单独调一次 `score_panel` 等于把 24523×68×34 的
    # 高瘦矩阵 build 两遍 —— 本机可用内存只有 2.5GB，第一次就是这么 OOM 的。
    rep = replay.run_cross(cfg, frames)
    if not rep.ok:
        print(f"{tag}\t失败: {rep.error}")
        return 1
    out = rep.res
    eq = out.equity.dropna()
    # `Replay.res` 是 `backtest.Result`，它没有 `step` 字段；`cross.to_result`
    # 把引擎算好的步长写进了 `metrics["avg_bars"]`（绩效页的「平均持仓根数」
    # 读的就是它），这里取回来。别自己再调 `cfg.step(bpd)` —— 拿不到数据密度。
    stp = int((out.metrics or {}).get("avg_bars") or hz)
    n, mu, t, wr = period_stats(eq, max(1, stp))
    tot = float(eq.iloc[-1] / eq.iloc[0] - 1)
    sc = None
    if rep.cross is not None and len(rep.cross) >= 3:
        sc = rep.cross[2]
    if sc is not None and len(sc):
        sc = sc.dropna(how="all")
        close = X.close_panel(frames).reindex(index=sc.index, columns=sc.columns).ffill()
        fwd = close.shift(-hz) / close - 1.0
        ic, ict, nsec = xs_ic(sc, fwd)
        del close, fwd
    else:
        ic, ict, nsec = float("nan"), float("nan"), 0
    del sc, out, rep, eq
    gc.collect()

    print(f"{tag:<22}{tot:>+9.1%}{n:>6}{mu:>+8.3%}{t:>7.2f}{wr:>7.1%}"
          f"{ic:>+9.4f}{ict:>7.2f}{nsec:>6}{time.time() - t0:>6.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
