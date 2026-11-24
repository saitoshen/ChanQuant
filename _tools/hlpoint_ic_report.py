# -*- coding: utf-8 -*-
"""出图 + 汇总表：ChanQuant 逐因子 IC vs 原项目。

读 `reports/hlpoint_ic_cmp.csv`（库函数）与 `reports/hlpoint_ic_origfn.csv`
（原因数×本机数据），画两栏散点并写合并表。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "D:/ChanQuant-Dev")

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import rcParams                      # noqa: E402

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

KEYS_ORDER = None


def main() -> int:
    cq = pd.read_csv("reports/hlpoint_ic_cmp.csv", index_col=0)
    fn = pd.read_csv("reports/hlpoint_ic_origfn.csv", index_col=0)
    keys = [k for k in cq.index if k in fn.index]
    t = pd.DataFrame(index=keys)
    t["分组"] = cq.loc[keys, "group"]
    t["因子"] = cq.loc[keys, "name"]
    t["原项目IC"] = fn.loc[keys, "原IC"]
    t["原项目t"] = fn.loc[keys, "原t"]
    t["CQ_IC_H46"] = cq.loc[keys, "CQ46_IC"]
    t["CQ_t_H46"] = cq.loc[keys, "CQ46_t"]
    t["原函数本机IC_H46"] = fn.loc[keys, "原fn46"]
    t["原函数本机t"] = fn.loc[keys, "原fn46_t"]
    t["CQ_IC_H30"] = cq.loc[keys, "CQ30_IC"]
    t["ΔIC_原vsCQ"] = t["CQ_IC_H46"] - t["原项目IC"]
    t["ΔIC_原vs原函数"] = t["原函数本机IC_H46"] - t["原项目IC"]
    t["ΔIC_原函数vsCQ"] = t["CQ_IC_H46"] - t["原函数本机IC_H46"]
    t.index.name = "key"
    t.to_csv("reports/hlpoint_ic_cmp_merged.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.8), dpi=150)
    panels = [
        (axes[0], t["原项目IC"], t["CQ_IC_H46"],
         "① 原报表  vs  ChanQuant 库函数\n（换了数据、也换了函数）"),
        (axes[1], t["原项目IC"], t["原函数本机IC_H46"],
         "② 原报表  vs  原因数 × 本机数据\n（只换数据）"),
        (axes[2], t["原函数本机IC_H46"], t["CQ_IC_H46"],
         "③ 原因数  vs  库函数（同一份本机数据）\n（只换函数）"),
    ]
    lim = 0.045
    rollset = {"roll_main_oi", "roll_main_annual", "roll_main_curv", "roll_chg_20"}
    for ax, xa, ya, title in panels:
        m = np.isfinite(xa) & np.isfinite(ya)
        ax.axhline(0, color="#999", lw=0.8, zorder=1)
        ax.axvline(0, color="#999", lw=0.8, zorder=1)
        ax.plot([-lim, lim], [-lim, lim], color="#7f8c8d", lw=1.0, ls="--",
                zorder=2, label="45 度线（完全一致）")
        pos = (xa[m] * ya[m]) > 0
        ax.scatter(xa[m][pos], ya[m][pos], s=34, c="#c0392b", alpha=0.85,
                   edgecolor="white", linewidth=0.6, zorder=3, label="同号")
        ax.scatter(xa[m][~pos], ya[m][~pos], s=34, c="#2471a3", alpha=0.85,
                   edgecolor="white", linewidth=0.6, zorder=3, label="反号")
        for k in m[m].index:
            if k in rollset:
                ax.annotate("展期", (xa[k], ya[k]), fontsize=7.5, color="#7d3c98",
                            xytext=(4, 3), textcoords="offset points")
        from scipy.stats import spearmanr
        r = np.corrcoef(xa[m], ya[m])[0, 1]
        rho = spearmanr(xa[m], ya[m]).statistic
        ax.set_title(f"{title}\nr = {r:+.3f}   ρ = {rho:+.3f}", fontsize=10.5)
        ax.set_xlabel("横轴 IC", fontsize=10)
        ax.set_ylabel("纵轴 IC（同一 IC 管线：日盘快照·每 5 交易日）", fontsize=10)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.25, ls=":")
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("HL-Point 36 因子逐因子 IC：换函数几乎逐位重合（③），换数据全散开（②）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = "reports/hlpoint_ic_cmp.png"
    fig.savefig(out, bbox_inches="tight")
    print("写出", out)
    print(t.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
