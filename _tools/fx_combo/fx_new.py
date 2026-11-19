"""新品种：贵金属 PT/PD + 股指 IH/IF/IC/IM 的汇率/波幅突破组合测试。

用法：
  /c/ProgramData/anaconda3/python.exe _tools/fx_combo/fx_new.py [YYYY-MM-DD]

口径：usdcnh单 / dxy单 / ORB单 / usdcnh+dxy 等权 / 三因子等权（ORB 顺突破）。
各品种数据起点不同（PT/PD 2025-11、IM 2022-07），逐条打印起止，别跨品种硬比。
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
CNH, DXY, ORB = "macro_usdcnh", "macro_dxy", "open_range_break"
REV = {ORB: True}

SYMS = [("PT", "GFEX"), ("PD", "GFEX"),
        ("IH", "CFFEX"), ("IF", "CFFEX"), ("IC", "CFFEX"), ("IM", "CFFEX")]
CASES = [
    ("usdcnh单",  [CNH],           {}),
    ("dxy单",     [DXY],           {}),
    ("ORB单",     [ORB],           REV),
    ("usdcnh+dxy", [CNH, DXY],     {}),
    ("三因子等权", [CNH, DXY, ORB], REV),
]


def main(start: str) -> None:
    rows: list[dict] = []
    curves: dict[str, dict[str, pd.Series]] = {n: {} for n, *_ in CASES}

    for sym, exch in SYMS:
        df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start=start)
        if len(df) < 2000:
            print(f"!! {sym}_{exch} 只有 {len(df)} 根，跳过")
            continue
        si = symbols_info.info(sym, exch)
        span = f"{df.index[0].date()}~{df.index[-1].date()}"
        for name, keys, rev in CASES:
            cfg = FS.FactorConfig(window=WINDOW, method="z", entry_z=ENTRY,
                                  exit_z=EXIT, orient="long_low", horizon=HORIZON,
                                  weight_mode="equal", reverse=dict(rev))
            out = FS.run_factor_strategy(
                df, "single" if len(keys) == 1 else "multi", keys, cfg)
            c = BT.BacktestConfig(initial_cash=PRIN)
            c.sizing, c.capital = "capital", PRIN * float(si["margin"])
            c.multiplier = float(si["mult"])
            c.margin_rate = float(si["margin"])
            c.bars_per_year = BT.bars_per_year(out.df.index)
            costs.apply(c, sym, exch)
            res = BT.run_backtest(out.df, c)
            m = res.metrics
            s = out.df["signal"].to_numpy()
            rows.append(dict(sym=sym, 区间=span, 口径=name,
                             夏普=round(float(m["sharpe"] or 0), 2),
                             年化=round(float(m["cagr"] or 0) * 100, 1),
                             回撤=round(float(m["max_dd"] or 0) * 100, 1),
                             卡玛=round(float(m["calmar"] or 0), 2),
                             笔数=int(m["n_trades"] or 0),
                             胜率=round(float(m["win_rate"] or 0) * 100, 1),
                             占用=round((s != 0).mean() * 100, 1)))
            curves[name][sym] = (res.equity / PRIN - 1.0)
            del out
        del df
        gc.collect()

    d = pd.DataFrame(rows)
    order = [c[0] for c in CASES]
    print("\n各品种区间：")
    print(d.drop_duplicates("sym")[["sym", "区间"]].to_string(index=False))
    for col in ("夏普", "卡玛", "回撤", "年化", "占用", "笔数", "胜率"):
        print(f"\n{col}：")
        print(d.pivot(index="sym", columns="口径", values=col)
               .reindex(columns=order).to_string())

    print("\n组合层面（同一品种组内等权合成）")
    for grp, names in (("贵金属 PT/PD", ["PT", "PD"]),
                       ("股指 IH/IF/IC/IM", ["IH", "IF", "IC", "IM"])):
        print(f"\n  [{grp}]")
        for name in order:
            sub = {k: v for k, v in curves[name].items() if k in names}
            if not sub:
                continue
            mat = pd.concat(sub, axis=1).sort_index().ffill().fillna(0.0)
            p = mat.mean(axis=1)
            r = p.resample("1D").last().dropna().diff().dropna()
            dd = float((p - p.cummax()).min())
            yrs = max((p.index[-1] - p.index[0]).days / 365.25, 1e-9)
            cagr = float(p.iloc[-1]) / yrs
            sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else 0.0
            print(f"    {name:<10} 累计 {p.iloc[-1]*100:+7.1f}%  年化 {cagr*100:+6.1f}%  "
                  f"日波动 {r.std(ddof=1)*np.sqrt(252)*100:5.1f}%  夏普 {sh:+.2f}  "
                  f"回撤 {dd*100:6.1f}%  卡玛 {cagr/abs(dd) if dd<0 else 0:+.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2016-01-01")
