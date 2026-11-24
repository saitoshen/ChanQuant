# -*- coding: utf-8 -*-
"""定位泄漏层级：先看**单个因子**的横截面 rank IC，再看模型分数。

因子层干净（IC 在 0.05 以内）而模型分数 IC 高得离谱 ⇒ 泄漏在 `_ml_scores`
（标签对齐 / 训练集切分）；因子层就已经很高 ⇒ 泄漏在因子或 `standardize`。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_leak.py
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
from ttrade import factor_metrics as FM  # noqa: E402
from hl_point_register import build  # noqa: E402
from hlpoint_control import make_ccfg  # noqa: E402
from hlpoint_diag import load_frames  # noqa: E402


def xs_ic(panel: pd.DataFrame, fwd: pd.DataFrame, every: int = 5) -> tuple:
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
    if not ics:
        return float("nan"), float("nan"), 0
    a = np.asarray(ics)
    return float(a.mean()), float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), len(a)


def main() -> int:
    ntest = int(os.environ.get("HP_N", "36"))
    cfg0, _s, keys = build(False)
    frames = load_frames(ntest, 2.0)
    cfg = cfg0.clone(cross_symbols=list(frames), multi_symbols=list(frames))
    ccfg = make_ccfg(cfg, keys)
    H = int(cfg.cross_rebalance)

    close = X.close_panel(frames)
    close = close.sort_index()
    fwd = close.shift(-H) / close - 1.0

    print(f"前向 {H} 根收益面板 {fwd.shape}")
    print()
    print(f"{'因子':<18}{'rank IC':>10}{'t 值':>8}{'期数':>6}")
    print("-" * 44)
    worst = []
    for k in keys:
        raw = X.score_panel(frames, [k], {}, mode="multi",
                            fcfg=X._factor_cfg(ccfg),
                            label_mode=ccfg.label_mode, base_period=ccfg.period)
        if raw.empty:
            print(f"{k:<18}{'空':>10}")
            continue
        mu, t, n = xs_ic(raw.reindex(index=fwd.index, columns=fwd.columns), fwd)
        print(f"{k:<18}{mu:>+10.4f}{t:>8.2f}{n:>6}")
        worst.append((k, abs(mu)))

    worst.sort(key=lambda x: -x[1])
    print()
    print("按 |IC| 排序前 5：", [(k, round(v, 4)) for k, v in worst[:5]])

    # 模型分数
    sc = X.score_panel(frames, list(keys), {}, mode="ml",
                       fcfg=X._factor_cfg(ccfg),
                       label_mode=ccfg.label_mode, base_period=ccfg.period)
    if not sc.empty:
        mu, t, n = xs_ic(sc.reindex(index=fwd.index, columns=fwd.columns), fwd)
        print(f"\n模型分数 rank IC {mu:+.4f}  t {t:.2f}  期数 {n}")

    # 顺手核对：标签本身是不是被当成特征了
    print("\n标签定义核对：", FM.forward_return.__doc__.splitlines()[0]
          if FM.forward_return.__doc__ else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
