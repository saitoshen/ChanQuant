# -*- coding: utf-8 -*-
"""HL-Point 复现：**全部品种 × 全历史** 的三组对照。

品种 = 天勤 1h 库里所有非金融品种（股指/国债剔除），历史 = 库里最长（2016 起）。

输出四张表：
  1. 三组对照（原始 / 截面对照打乱 / 分数取负）：总收益、期数、期均、t、胜率、回撤
  2. 每个调仓期的多空腿数与截面宽度（十分位要求截面 ≥10 个品种）
  3. 分数面板的横截面 rank IC（OOS）
  4. 与 HL-Point 原文报告的锚点并排（期均 0.899%、t 4.84、IC 0.025 量级）

用法：
    C:/ProgramData/anaconda3/python.exe _tools/hlpoint_full.py
    HP_MAXN=20 ... _tools/hlpoint_full.py     # 只跑前 20 个（调试）
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.stats import spearmanr  # noqa: E402

from ttrade import auxdata, cross as X, replay, tqstore, tqdata  # noqa: E402
from hl_point_register import build, FINANCIAL  # noqa: E402
from hlpoint_control import make_ccfg, run_one  # noqa: E402


def all_symbols() -> list:
    """1h 库里有数据的全部非金融商品（按 1h bar 数降序，多的优先）。"""
    cov = tqstore.coverage("1h")
    rows = []
    for (sym, ex), (_a, _b, n) in cov.items():
        if sym.upper() in FINANCIAL:
            continue
        rows.append((sym, ex, int(n)))
    rows.sort(key=lambda r: -r[2])
    return rows


def load_all(maxn: int = 0) -> dict:
    rows = all_symbols()
    if maxn:
        rows = rows[:maxn]
    out, t0 = {}, time.time()
    for sym, ex, _n in rows:
        try:
            df = tqdata.load_main(sym, ex, "1h", adjust="qfq")
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}_{ex} 读取失败: {e}")
            continue
        if df is None or len(df) < 2000:
            continue
        df = auxdata.attach_for_period(df, sym, ex, "1h")
        out[f"{sym}_{ex}"] = df
    print(f"载入 {len(out)} 个品种（{len(rows)} 个候选），耗时 {time.time() - t0:.1f}s")
    lens = [len(v) for v in out.values()]
    print(f"每品种 bar 数：中位 {int(np.median(lens))}、最少 {min(lens)}、最多 {max(lens)}")
    return out


def period_stats(eq: pd.Series, step: int) -> tuple:
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
    return len(a), float(mu), float(mu / (sd / np.sqrt(len(a)))), float((a > 0).mean())


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
    if len(ics) < 5:
        return float("nan"), float("nan"), 0
    a = np.asarray(ics)
    return float(a.mean()), float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), len(a)


def main() -> int:
    maxn = int(os.environ.get("HP_MAXN", "0") or 0)
    cfg0, _s, keys = build(False)
    frames = load_all(maxn)
    if len(frames) < 10:
        print("品种不足 10 个")
        return 1
    cfg = cfg0.clone(cross_symbols=list(frames), multi_symbols=list(frames))
    ccfg = make_ccfg(cfg, keys)
    bt = replay.make_bt(replay.market_of(cfg), True, cfg)
    H = int(cfg.cross_rebalance)
    print(f"因子 {len(keys)} 个，每 {H} 根调仓，多空各 {cfg.cross_top_pct:.0f}%")
    print()

    # ---- 1. 三组对照 ----
    hdr = (f"{'变体':<10}{'总收益':>10}{'期数':>6}{'期均':>9}{'t 值':>8}"
           f"{'期胜率':>8}{'最大回撤':>10}{'耗时':>7}")
    print(hdr)
    print("-" * len(hdr))
    eqs = {}
    for tag, mode in (("原始", ""), ("打乱", "shuffle"), ("反向", "flip")):
        t0 = time.time()
        out = run_one(frames, ccfg, bt, mode)
        if not out.ok:
            print(f"{tag:<10} 失败: {out.error}")
            return 1
        eq = out.equity.dropna()
        n, mu, t, wr = period_stats(eq, max(1, out.step))
        dd = float((eq / eq.cummax() - 1.0).min())
        print(f"{tag:<10}{eq.iloc[-1] / eq.iloc[0] - 1:>+9.1%}{n:>6}{mu:>+8.3%}"
              f"{t:>8.2f}{wr:>8.1%}{dd:>10.1%}{time.time() - t0:>6.0f}s")
        eqs[tag] = out
    print()

    # ---- 2. 截面宽度 + 分数 IC ----
    out = eqs["原始"]
    w = out.weights.reindex(columns=sorted(out.weights.columns))
    nz = (w.abs() > 1e-12).sum(axis=1)
    print(f"每期持仓品种数：中位 {int(nz.median())}、最大 {int(nz.max())}")
    print(f"Σ|w| 中位 {float(w.abs().sum(axis=1).median()):.4f}，"
          f"净敞口 |Σw| 最大 {float(w.sum(axis=1).abs().max()):.4f}（legs 应≈0）")

    close = X.close_panel(frames)
    score = X.score_panel(frames, list(keys), {}, mode="ml",
                          fcfg=X._factor_cfg(ccfg),
                          label_mode=ccfg.label_mode, base_period=ccfg.period)
    score = score.dropna(how="all")
    # 🔴 必须按列名对齐（`close_panel` 的列序和 `score.columns` 不一定相同）
    close = close.reindex(index=score.index, columns=score.columns).ffill()
    fwd = close.shift(-H) / close - 1.0
    mu, t, n = xs_ic(score, fwd)
    print(f"\n分数面板 {score.shape}  {score.index[0]} ~ {score.index[-1]}")
    print(f"OOS 横截面 rank IC：均值 {mu:+.4f}、t {t:.2f}、截面数 {n}")
    print()
    print("HL-Point 原文锚点（87 品种 / 2021-2026 / 400 期）：")
    print("  期均 +0.899%   t 4.84   最强单因子 IC dev_spread +0.0253 / curv −0.0283")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
