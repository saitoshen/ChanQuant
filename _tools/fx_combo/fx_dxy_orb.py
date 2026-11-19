"""波幅突破 + 美元指数 二因子等价对比（2016 起）。

用法：
  /c/ProgramData/anaconda3/python.exe _tools/fx_combo/fx_dxy_orb.py [YYYY-MM-DD]

口径（全部 orient=long_low + 波幅突破 reverse 翻成顺突破）：
  dxy单 / ORB单 / usdcnh+ORB / dxy+ORB / usdcnh+dxy / 三因子等权
"""
from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from ttrade import backtest as BT          # noqa: E402
from ttrade import costs                   # noqa: E402
from ttrade import factor_strategy as FS   # noqa: E402
from ttrade import symbols_info            # noqa: E402
from ttrade import tqdata                  # noqa: E402

PRIN = 1_000_000.0
WINDOW, HORIZON = 545, 12
ENTRY, EXIT = 1.0, 0.3
SYMS = [("AU", "SHFE"), ("AG", "SHFE"), ("SN", "SHFE"),
        ("AL", "SHFE"), ("CU", "SHFE"), ("BC", "INE")]
CNH, DXY, ORB = "macro_usdcnh", "macro_dxy", "open_range_break"
REV = {ORB: True}          # 顺突破

CASES = [
    ("dxy单",        [DXY],             {}),
    ("ORB单",        [ORB],             REV),
    ("usdcnh+ORB",   [CNH, ORB],        REV),
    ("dxy+ORB",      [DXY, ORB],        REV),
    ("usdcnh+dxy",   [CNH, DXY],        {}),
    ("三因子等权",   [CNH, DXY, ORB],   REV),
]
WIN = ("2021-03-05", "2021-12-27")     # FX 等权的最大回撤窗口


def main(start: str) -> None:
    rows: list[dict] = []
    curves: dict[str, list] = {n: [] for n, *_ in CASES}
    occ: dict[str, list] = {n: [] for n, *_ in CASES}

    for sym, exch in SYMS:
        df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start=start)
        si = symbols_info.info(sym, exch)
        for name, keys, rev in CASES:
            cfg = FS.FactorConfig(window=WINDOW, method="z", entry_z=ENTRY,
                                  exit_z=EXIT, orient="long_low", horizon=HORIZON,
                                  weight_mode="equal", reverse=dict(rev))
            mode = "single" if len(keys) == 1 else "multi"
            out = FS.run_factor_strategy(df, mode, keys, cfg)
            c = BT.BacktestConfig(initial_cash=PRIN)
            c.sizing, c.capital = "capital", PRIN * float(si["margin"])
            c.multiplier = float(si["mult"])
            c.margin_rate = float(si["margin"])
            c.bars_per_year = BT.bars_per_year(out.df.index)
            costs.apply(c, sym, exch)
            res = BT.run_backtest(out.df, c)
            m = res.metrics
            s = out.df["signal"].to_numpy()
            rows.append(dict(sym=sym, 口径=name,
                             夏普=round(float(m["sharpe"] or 0), 2),
                             年化=round(float(m["cagr"] or 0) * 100, 1),
                             波动=round(float(m["ann_vol"] or 0) * 100, 1),
                             回撤=round(float(m["max_dd"] or 0) * 100, 1),
                             卡玛=round(float(m["calmar"] or 0), 2),
                             笔数=int(m["n_trades"] or 0),
                             占用=round((s != 0).mean() * 100, 1)))
            curves[name].append((res.equity / PRIN - 1.0).rename(sym))
            occ[name].append((s != 0).mean())
            del out
        del df
        gc.collect()

    d = pd.DataFrame(rows)
    order = [c[0] for c in CASES]
    for col in ("夏普", "卡玛", "回撤", "年化", "波动", "占用", "笔数"):
        print(f"\n{col}：")
        print(d.pivot(index="sym", columns="口径", values=col)
               .reindex(columns=order).to_string())

    port = {}
    print("\n" + "=" * 88 + "\n组合层面（各腿 100 万、等权合成）")
    for name in order:
        mat = pd.concat(curves[name], axis=1).sort_index().ffill().fillna(0.0)
        p = mat.mean(axis=1)
        port[name] = p
        r = p.resample("1D").last().dropna().diff().dropna()
        dd = float((p - p.cummax()).min())
        yrs = max((p.index[-1] - p.index[0]).days / 365.25, 1e-9)
        cagr = float(p.iloc[-1]) / yrs
        sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else 0.0
        print(f"{name:<12} 累计 {p.iloc[-1]*100:+7.1f}%  年化 {cagr*100:+6.1f}%  "
              f"日波动 {r.std(ddof=1)*np.sqrt(252)*100:5.1f}%  夏普 {sh:+.2f}  "
              f"回撤 {dd*100:6.1f}%  卡玛 {cagr/abs(dd) if dd<0 else 0:+.2f}  "
              f"占用 {np.mean(occ[name])*100:4.1f}%")

    print("\n分段（组合夏普 / 回撤 / 累计）：")
    for name in order:
        p = port[name].resample("1D").last().dropna()
        r = p.diff()
        line = []
        for a, b, lab in ((None, "2022-01-01", "≤2021"), ("2022-01-01", None, "2022+")):
            sub = p.loc[a:b]
            rr = r.loc[a:b].dropna()
            sh = rr.mean() / rr.std(ddof=1) * np.sqrt(252) if rr.std(ddof=1) > 0 else 0.0
            dd = float((sub - sub.cummax()).min())
            line.append(f"{lab} {sh:+.2f} / {dd*100:6.1f}% / {(sub.iloc[-1]-sub.iloc[0])*100:+6.0f}%")
        print(f"  {name:<12} " + " ｜ ".join(line))

    print(f"\nFX 最坏窗口 {WIN[0]} → {WIN[1]}：")
    for name in order:
        p = port[name]
        seg = p.loc[WIN[0]:WIN[1]]
        print(f"  {name:<12} 窗口内 {(seg.iloc[-1]-seg.iloc[0])*100:+6.1f}%  "
              f"区间最低 {(seg-seg.iloc[0]).min()*100:+6.1f}%")

    print("\n分年（组合口径）：")
    for name in order:
        p = port[name].resample("1D").last().dropna()
        yr = p.resample("YE").last()
        prev, seg = 0.0, []
        for t, v in yr.items():
            seg.append(f"{t.year} {(v-prev)*100:+.0f}%")
            prev = v
        print(f"  {name:<12} " + "  ".join(seg))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2016-01-01")
