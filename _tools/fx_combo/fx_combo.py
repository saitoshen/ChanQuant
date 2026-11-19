"""汇率双因子合成口径对比（单品种 + 组合），样本期可指定。

用法：
  /c/ProgramData/anaconda3/python.exe _tools/fx_combo/fx_combo.py [YYYY-MM-DD]

口径：usdcnh 单 / dxy 单 / 等权 / 因子与 / 因子或，其余参数完全一致。
每个品种只 load 一次行情，四种口径共用（5m 全史 26 万根，重复 load 很费）。
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
PERIOD = "5m"
WINDOW, HORIZON = 545, 12
ENTRY, EXIT = 1.0, 0.3
SYMS = [("AU", "SHFE"), ("AG", "SHFE"), ("SN", "SHFE"),
        ("AL", "SHFE"), ("CU", "SHFE"), ("BC", "INE")]
KEYS2 = ["macro_usdcnh", "macro_dxy"]
CASES = [
    ("usdcnh", "single", ["macro_usdcnh"], None),
    ("dxy",    "single", ["macro_dxy"],    None),
    ("等权",   "multi",  KEYS2,            "equal"),
    ("因子与", "multi",  KEYS2,            "and"),
    ("因子或", "multi",  KEYS2,            "or"),
]


def cfg_of(wmode: str | None) -> FS.FactorConfig:
    kw = dict(window=WINDOW, method="z", entry_z=ENTRY, exit_z=EXIT,
              orient="long_low", horizon=HORIZON)
    if wmode:
        kw["weight_mode"] = wmode
    return FS.FactorConfig(**kw)


def bt_of(sym: str, exch: str, index, si: dict) -> BT.BacktestConfig:
    mult, mg = float(si["mult"]), float(si["margin"])
    c = BT.BacktestConfig(initial_cash=PRIN)
    c.sizing, c.capital = "capital", PRIN * mg   # 名义敞口 ≈ 本金
    c.multiplier, c.margin_rate = mult, mg
    c.bars_per_year = BT.bars_per_year(index)
    costs.apply(c, sym, exch)
    return c


def main(start: str) -> None:
    rows: list[dict] = []
    curves: dict[str, list[pd.Series]] = {n: [] for n, *_ in CASES}

    for sym, exch in SYMS:
        df = tqdata.load_main(sym, exch, PERIOD, adjust="qfq", start=start)
        si = symbols_info.info(sym, exch)
        first = df.index[0]
        for name, mode, keys, wmode in CASES:
            out = FS.run_factor_strategy(df, mode, keys, cfg_of(wmode))
            cfg = bt_of(sym, exch, out.df.index, si)
            res = BT.run_backtest(out.df, cfg)
            m = res.metrics
            s = out.df["signal"].to_numpy()
            rows.append(dict(sym=sym, 口径=name, 起=str(first.date()), bars=len(out.df),
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

    print("\n" + "=" * 78 + "\n组合层面（各腿 100 万、等权合成）")
    for name in order:
        mat = pd.concat(curves[name], axis=1).sort_index().ffill().fillna(0.0)
        port = mat.mean(axis=1)
        r = port.resample("1D").last().dropna().diff().dropna()
        dd = float((port - port.cummax()).min())
        yrs = max((port.index[-1] - port.index[0]).days / 365.25, 1e-9)
        cagr = float(port.iloc[-1]) / yrs
        sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else 0.0
        corr = mat.resample("1D").last().diff().dropna().corr()
        print(f"{name:<6} 累计 {port.iloc[-1]*100:+7.1f}%  年化 {cagr*100:+6.1f}%  "
              f"日波动 {r.std(ddof=1)*np.sqrt(252)*100:5.1f}%  夏普 {sh:+.2f}  "
              f"回撤 {dd*100:6.1f}%  卡玛 {cagr/abs(dd) if dd<0 else 0:+.2f}  "
              f"腿间相关 {corr.values[np.triu_indices(corr.shape[0],1)].mean():.2f}")
        print("       各腿：" + "  ".join(f"{c} {mat[c].iloc[-1]*100:+.0f}%" for c in mat))

    print("\n分年（组合口径）：")
    for name in order:
        mat = pd.concat(curves[name], axis=1).sort_index().ffill().fillna(0.0)
        port = mat.mean(axis=1).resample("1D").last().dropna()
        yr = port.resample("YE").last()
        prev = 0.0
        seg = []
        for t, v in yr.items():
            seg.append(f"{t.year} {(v-prev)*100:+.0f}%")
            prev = v
        print(f"  {name:<4} " + "  ".join(seg))

    print("\n分段（组合口径，夏普 / 回撤）：")
    for name in order:
        mat = pd.concat(curves[name], axis=1).sort_index().ffill().fillna(0.0)
        port = mat.mean(axis=1).resample("1D").last().dropna()
        r = port.diff()
        line = []
        for a, b, lab in ((None, "2022-01-01", "≤2021"),
                          ("2022-01-01", None, "2022+")):
            sub = port.loc[a:b]
            rr = r.loc[a:b].dropna()
            sh = (rr.mean() / rr.std(ddof=1) * np.sqrt(252)
                  if rr.std(ddof=1) > 0 else 0.0)
            dd = float((sub - sub.cummax()).min())
            line.append(f"{lab} 夏普 {sh:+.2f} 回撤 {dd*100:6.1f}% "
                        f"累计 {(sub.iloc[-1]-sub.iloc[0])*100:+6.0f}%")
        print(f"  {name:<4} " + " ｜ ".join(line))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2016-01-01")
