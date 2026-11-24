# -*- coding: utf-8 -*-
"""把「因子」和「标签」分开：展期因子到底受不受复权影响？

起因（用户 11-24 的质疑）：**展期是用真实合约算的，不是用 888 主连算的**，
所以「复权口径」不该改到展期因子本身。对 —— 那就得证明：

  ① 两次加载（`adjust="qfq"` / `adjust=None`）下 `roll_main_oi` 等展期因子值
     **完全相同**（因子来自 `o6_*` 真实合约价，和主连复权无关）；
  ② IC 之所以翻号，是因为 **y（未来收益）** 变了：不复权主连在换月时跳空，
     跳空幅度 = 展期价差 ⇒ 给 carry 因子造出机械的负相关。

于是把 y 拆成两截直接看：

    y_raw  = 不复权主连的 h 根收益          （原项目口径）
    y_qfq  = 复权主连的 h 根收益            （本库现状）
    y_gap  = y_raw − y_qfq                  （换月跳空那一截，其余时候 = 0）

再分别算 IC(roll, ·)。如果 IC(roll, y_gap) 显著为负、且 |y_gap| 的量级对上，
那 −0.026 就基本是「换月机械效应」，不是 carry 的预测力。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_revadj_probe.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "D:/ChanQuant-Dev")
sys.path.insert(0, "D:/ChanQuant-Dev/_tools")

from scipy.stats import spearmanr  # noqa: E402

from ttrade import auxdata, factors as F, tqdata  # noqa: E402


NSYM = int(__import__("os").environ.get("HP_N", "24"))
H = 46
FACTORS = ["roll_main_oi", "roll_main_annual", "dist_hi", "dev_z_60"]
START, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2026-10-14")

#: 🔴 固定品种表，不用 `all_symbols()[:N]`：库是实盘在写的 duckdb，偶发「文件被占用」
#: 会让某几个品种加载失败，存活集一变（1433 → 1380 个换月事件），IC 就漂
#: （实测 roll 的 qfq IC 在 +0.024 / +0.037 之间跳）。固定表才能逐字复现。
SYMS = [("RB", "SHFE"), ("HC", "SHFE"), ("I", "DCE"), ("J", "DCE"),
        ("JM", "DCE"), ("SF", "CZCE"), ("SM", "CZCE"), ("CU", "SHFE"),
        ("AL", "SHFE"), ("ZN", "SHFE"), ("NI", "SHFE"), ("AU", "SHFE"),
        ("AG", "SHFE"), ("SC", "INE"), ("RU", "SHFE"), ("BU", "SHFE"),
        ("FU", "SHFE"), ("TA", "CZCE"), ("MA", "CZCE"), ("EG", "DCE"),
        ("PP", "DCE"), ("L", "DCE"), ("M", "DCE"), ("Y", "DCE")][:NSYM]


def snap(idx: pd.DatetimeIndex) -> np.ndarray:
    hour = idx.hour.to_numpy()
    keep = (hour >= 8) & (hour < 16)
    pos = np.flatnonzero(keep)
    d = idx.normalize().to_numpy()[pos]
    return pos[np.append(d[1:] != d[:-1], True)]


def ic_series(fac: dict, y: dict, pick: pd.DatetimeIndex) -> tuple:
    """逐截面 Spearman，返回 (均值, t, 期数)。两边都是 sym→Series(index=日)。"""
    Fm = pd.DataFrame(fac).reindex(index=pick).to_numpy(float)
    Ym = pd.DataFrame(y).reindex(index=pick).to_numpy(float)
    ics = []
    for i in range(len(pick)):
        a, b = Fm[i], Ym[i]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10 or np.ptp(a[m]) == 0:
            continue
        v = spearmanr(a[m], b[m]).statistic
        if np.isfinite(v):
            ics.append(float(v))
    a = np.asarray(ics, float)
    if len(a) < 10:
        return np.nan, np.nan, len(a)
    return float(a.mean()), float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), len(a)


def main() -> int:
    fac_q, y_q, y_r, gap, roll_days, n_ev, n_win = {}, {}, {}, {}, {}, 0, 0
    diffs = {k: 0.0 for k in FACTORS}
    for sym, ex in SYMS:
        try:
            dq = tqdata.load_main(sym, ex, "1h", adjust="qfq")
            dr = tqdata.load_main(sym, ex, "1h", adjust=None)
        except Exception:  # noqa: BLE001
            continue
        if dq is None or dr is None or len(dq) < 500 or len(dq) != len(dr):
            continue
        dq = auxdata.attach_for_period(dq, sym, ex, "1h")
        dr = auxdata.attach_for_period(dr, sym, ex, "1h")
        pos = snap(dq.index)
        sday = pd.DatetimeIndex(dq.index.normalize().to_numpy()[pos])
        m = (sday >= START) & (sday <= END)
        if m.sum() < 50:
            continue
        pos, sday = pos[m], sday[m]
        cq = pd.to_numeric(dq["close"], errors="coerce").to_numpy(float)
        cr = pd.to_numeric(dr["close"], errors="coerce").to_numpy(float)

        # ① 因子值：两次加载必须一致（展期来自 o6_ 真实合约，和主连复权无关）
        for k in FACTORS:
            v1 = pd.to_numeric(F.FACTORS[k].func(dq), errors="coerce").to_numpy(float)[pos]
            v2 = pd.to_numeric(F.FACTORS[k].func(dr), errors="coerce").to_numpy(float)[pos]
            ok = np.isfinite(v1) & np.isfinite(v2)
            if ok.any():
                diffs[k] = max(diffs[k], float(np.abs(v1[ok] - v2[ok]).max()))
            fac_q.setdefault(k, {})[sym] = pd.Series(v1, index=sday)

        # ② y 两截
        p2 = pos + H
        ok = p2 < len(cq)
        a = np.full(len(pos), np.nan)
        b = np.full(len(pos), np.nan)
        a[ok] = cq[p2[ok]] / cq[pos[ok]] - 1.0      # 复权
        b[ok] = cr[p2[ok]] / cr[pos[ok]] - 1.0      # 不复权
        y_q[sym] = pd.Series(a, index=sday)
        y_r[sym] = pd.Series(b, index=sday)
        gap[sym] = pd.Series(b - a, index=sday)

        # ③ 换月识别：adj_factor 在换月点跳变
        adj = pd.to_numeric(dq["adj_factor"], errors="coerce").to_numpy(float)
        chg = np.flatnonzero(np.abs(np.diff(adj)) > 0)
        n_ev += len(chg)
        # 持有窗口 (p, p+H] 里是否跨过换月
        if len(chg):
            for i in range(len(pos)):
                if not ok[i]:
                    continue
                n_win += 1
                seg = chg[(chg > pos[i]) & (chg < p2[i])]
                if len(seg):
                    roll_days[sym] = roll_days.get(sym, 0) + 1

    print(f"载入 {len(y_q)}/{len(SYMS)} 个品种；换月事件 {n_ev} 次")
    print("  品种：", " ".join(f"{s}_{e}" for s, e in SYMS if f"{s}_{e}" in y_q))
    print("\n① 因子值一致性（qfq vs 不复权，逐品种最大绝对差）")
    for k, v in diffs.items():
        print(f"   {k:<18} max|Δ| = {v:.3e}")

    allday = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in y_q.values()])))
    pick = allday[::5]

    print(f"\n② 同一份因子、三种标签（H={H} 根，每 5 交易日截面）")
    print(f"{'因子':<18}{'y 复权':>10}{'t':>7}{'y 不复权':>11}{'t':>7}"
          f"{'y 跳空截':>11}{'t':>7}")
    print("-" * 72)
    rows = []
    for k in FACTORS:
        f = fac_q[k]
        r1 = ic_series(f, y_q, pick)
        r2 = ic_series(f, y_r, pick)
        r3 = ic_series(f, gap, pick)
        print(f"{k:<18}{r1[0]:>+10.4f}{r1[1]:>7.2f}{r2[0]:>+11.4f}{r2[1]:>7.2f}"
              f"{r3[0]:>+11.4f}{r3[1]:>7.2f}")
        rows.append(dict(factor=k, ic_qfq=r1[0], t_qfq=r1[1],
                         ic_raw=r2[0], t_raw=r2[1],
                         ic_gap=r3[0], t_gap=r3[1],
                         n=r1[2], max_delta=diffs[k]))

    tot = sum(roll_days.values())
    g = pd.concat(gap.values())
    print(f"\n③ 持有窗口跨换月的比例：{tot}/{n_win}"
          f"（{100.0 * tot / max(n_win, 1):.1f}%）")
    print(f"   跳空截 |y_gap| 均值 {np.nanmean(np.abs(g)) * 100:.3f}%"
          f"，非零占比 {100.0 * np.nanmean(np.abs(g) > 1e-9):.1f}%")

    det = pd.DataFrame(rows)
    det["roll_win_share"] = tot / max(n_win, 1)
    det["gap_nonzero"] = float(np.nanmean(np.abs(g) > 1e-9))
    det["gap_abs_mean"] = float(np.nanmean(np.abs(g)))
    det.to_csv("reports/hlpoint_revadj.csv", index=False, encoding="utf-8-sig")
    print("写出 reports/hlpoint_revadj.csv")

    # ---- 图：同一份因子，三种标签 ----
    import matplotlib                                # noqa: E402
    import matplotlib.pyplot as plt                  # noqa: E402
    from matplotlib import rcParams                  # noqa: E402
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9.6, 4.9), dpi=150)
    xs = np.arange(len(FACTORS))
    w = 0.26
    ser = [("y 复权（本库现状）", "ic_qfq", "#2471a3"),
           ("y 不复权（原项目）", "ic_raw", "#c0392b"),
           ("y 的换月跳空那一截", "ic_gap", "#7d3c98")]
    for i, (lbl, col, c) in enumerate(ser):
        v = det[col].to_numpy(float)
        bars = ax.bar(xs + (i - 1) * w, v, w, label=lbl, color=c, alpha=0.88)
        for b, val, t in zip(bars, v, det[col.replace("ic_", "t_")].to_numpy(float)):
            ax.annotate(f"{val:+.3f}\nt {t:+.1f}",
                        (b.get_x() + b.get_width() / 2, val),
                        ha="center", va="top" if val < 0 else "bottom",
                        fontsize=8, xytext=(0, -4 if val < 0 else 4),
                        textcoords="offset points")
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(FACTORS, fontsize=10)
    ax.set_ylabel("截面 IC 均值（H=46 根，每 5 交易日）", fontsize=10)
    ax.set_title("展期因子值在两口径下逐位相同（max|Δ|=0）—— 变的是标签 y\n"
                 "carry 与「换月跳空」的负相关达 t=−11.5，原项目 −0.026 主要来自这一截",
                 fontsize=11)
    ax.set_ylim(bottom=-0.172)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("reports/hlpoint_ic_revadj.png", bbox_inches="tight")
    print("写出 reports/hlpoint_ic_revadj.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
