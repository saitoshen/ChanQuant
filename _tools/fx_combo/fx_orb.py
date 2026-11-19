"""汇率双因子 + 波幅突破：2016 起等权组合对比。

用法：
  /c/ProgramData/anaconda3/python.exe _tools/fx_combo/fx_orb.py [YYYY-MM-DD]

口径：
  FX等权         macro_usdcnh + macro_dxy 等权（orient=long_low，反向用）
  ORB顺/ORB反    open_range_break 单因子，两个方向都跑一遍看符号
  FX+ORB等权     三因子等权；波幅突破按 long_high（reverse 打开）
  FX+ORB等权(反) 三因子等权；波幅突破也按 long_low
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
from ttrade import factor_metrics as M     # noqa: E402
from ttrade import factor_strategy as FS   # noqa: E402
from ttrade import symbols_info            # noqa: E402
from ttrade import tqdata                  # noqa: E402

PRIN = 1_000_000.0
PERIOD = "5m"
WINDOW, HORIZON = 545, 12
ENTRY, EXIT = 1.0, 0.3
SYMS = [("AU", "SHFE"), ("AG", "SHFE"), ("SN", "SHFE"),
        ("AL", "SHFE"), ("CU", "SHFE"), ("BC", "INE")]
FX = ["macro_usdcnh", "macro_dxy"]
ORB = "open_range_break"

# (名字, mode, keys, weight_mode, orient, reverse)
CASES = [
    ("FX等权",        "multi",  FX,            "equal", "long_low",  {}),
    ("ORB顺",         "single", [ORB],         None,    "long_high", {}),
    ("ORB反",         "single", [ORB],         None,    "long_low",  {}),
    ("FX+ORB等权",    "multi",  FX + [ORB],    "equal", "long_low",  {ORB: True}),
    ("FX+ORB等权(反)", "multi", FX + [ORB],    "equal", "long_low",  {}),
]


def main(start: str) -> None:
    rows: list[dict] = []
    curves: dict[str, list[pd.Series]] = {c[0]: [] for c in CASES}
    ic_rows: dict[str, pd.Series] = {}

    for sym, exch in SYMS:
        df = tqdata.load_main(sym, exch, PERIOD, adjust="qfq", start=start)
        si = symbols_info.info(sym, exch)
        mult, mg = float(si["mult"]), float(si["margin"])

        # 波幅突破单因子的逐年 IC（原始值 vs 12 根前向，Spearman）
        cfg0 = FS.FactorConfig(window=WINDOW, method="z", entry_z=ENTRY,
                               exit_z=EXIT, orient="long_high", horizon=HORIZON)
        o0 = FS.run_factor_strategy(df, "single", [ORB], cfg0)
        fwd = M.forward_return(df, HORIZON)
        g = pd.DataFrame({"s": o0.df[f"f_{ORB}"], "f": fwd}).dropna()
        ic_rows[sym] = g.groupby(g.index.year).apply(
            lambda x: x["s"].corr(x["f"], method="spearman"))
        del o0

        for name, mode, keys, wmode, orient, rev in CASES:
            kw = dict(window=WINDOW, method="z", entry_z=ENTRY, exit_z=EXIT,
                      orient=orient, horizon=HORIZON, reverse=dict(rev))
            if wmode:
                kw["weight_mode"] = wmode
            out = FS.run_factor_strategy(df, mode, keys, FS.FactorConfig(**kw))
            c = BT.BacktestConfig(initial_cash=PRIN)
            c.sizing, c.capital = "capital", PRIN * mg
            c.multiplier, c.margin_rate = mult, mg
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
                             胜率=round(float(m["win_rate"] or 0) * 100, 1),
                             占用=round((s != 0).mean() * 100, 1)))
            curves[name].append((res.equity / PRIN - 1.0).rename(sym))
            del out
        del df
        gc.collect()

    d = pd.DataFrame(rows)
    order = [c[0] for c in CASES]
    for col in ("夏普", "卡玛", "回撤", "年化", "波动", "占用", "笔数", "胜率"):
        print(f"\n{col}：")
        print(d.pivot(index="sym", columns="口径", values=col)
               .reindex(columns=order).to_string())

    print("\n" + "=" * 84 + "\n组合层面（各腿 100 万、等权合成）")
    for name in order:
        mat = pd.concat(curves[name], axis=1).sort_index().ffill().fillna(0.0)
        port = mat.mean(axis=1)
        r = port.resample("1D").last().dropna().diff().dropna()
        dd = float((port - port.cummax()).min())
        yrs = max((port.index[-1] - port.index[0]).days / 365.25, 1e-9)
        cagr = float(port.iloc[-1]) / yrs
        sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else 0.0
        print(f"{name:<12} 累计 {port.iloc[-1]*100:+7.1f}%  年化 {cagr*100:+6.1f}%  "
              f"日波动 {r.std(ddof=1)*np.sqrt(252)*100:5.1f}%  夏普 {sh:+.2f}  "
              f"回撤 {dd*100:6.1f}%  卡玛 {cagr/abs(dd) if dd<0 else 0:+.2f}")

    print("\n分段（组合夏普 / 累计）：")
    for name in order:
        mat = pd.concat(curves[name], axis=1).sort_index().ffill().fillna(0.0)
        port = mat.mean(axis=1).resample("1D").last().dropna()
        r = port.diff()
        line = []
        for a, b, lab in ((None, "2022-01-01", "≤2021"), ("2022-01-01", None, "2022+")):
            sub = port.loc[a:b]
            rr = r.loc[a:b].dropna()
            sh = (rr.mean() / rr.std(ddof=1) * np.sqrt(252)
                  if rr.std(ddof=1) > 0 else 0.0)
            dd = float((sub - sub.cummax()).min())
            line.append(f"{lab} 夏普 {sh:+.2f} 回撤 {dd*100:6.1f}% "
                        f"累计 {(sub.iloc[-1]-sub.iloc[0])*100:+6.0f}%")
        print(f"  {name:<12} " + " ｜ ".join(line))

    print("\nopen_range_break 逐年 IC（原始值 vs 12 根前向，期望为正=顺突破用）")
    ic = pd.DataFrame(ic_rows)
    ic["均值"] = ic.mean(axis=1)
    print(ic.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2016-01-01")
