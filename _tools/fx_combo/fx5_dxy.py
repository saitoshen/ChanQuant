"""dxy 5m 因子 vs 日频：用现有 Dukascopy 覆盖（2020 + 2021H1 + 2026）做策略对照。

⚠️ 样本不连续（2022~2025 缺 dxy 5m），只能看方向，不能当结论。

用法: python _tools/fx_combo/fx5_dxy.py
"""

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from ttrade import backtest as BT
from ttrade import costs
from ttrade import factor_strategy as FS
from ttrade import symbols_info, tqdata

PRIN = 1_000_000.0
CMD = [("AU", "SHFE"), ("AG", "SHFE"), ("SN", "SHFE"),
       ("AL", "SHFE"), ("CU", "SHFE"), ("BC", "INE")]
IDX = [("IH", "CFFEX"), ("IF", "CFFEX"), ("IC", "CFFEX"), ("IM", "CFFEX")]
CASES = [
    ("fx5_dxy_chg_48", "5m.dxy.chg48", "long_high"),
    ("fx5_dxy_chg_48", "5m.dxy.chg48", "long_low"),
    ("fx5_dxy_z_48", "5m.dxy.z48", "long_high"),
    ("fx5_dxy_z_48", "5m.dxy.z48", "long_low"),
    ("macro_dxy", "日频dxy", "long_high"),
    ("macro_dxy", "日频dxy", "long_low"),
    ("macro_usdcnh", "日频usdcnh", "long_low"),
]


def one(df, key, orient, si, sym, exch):
    cfg = FS.FactorConfig(window=545, method="z", entry_z=3.0, exit_z=1.0,
                          orient=orient, horizon=12)
    out = FS.run_factor_strategy(df, "single", [key], cfg)
    c = BT.BacktestConfig(initial_cash=PRIN)
    c.sizing = "capital"
    c.capital = PRIN * float(si["margin"])
    c.multiplier = float(si["mult"])
    c.margin_rate = float(si["margin"])
    c.bars_per_year = BT.bars_per_year(out.df.index)
    costs.apply(c, sym, exch)
    r = BT.run_backtest(out.df, c)
    eq = r.equity / PRIN - 1.0
    m = r.metrics
    sig = out.df["signal"].to_numpy()
    occ = float((sig != 0).mean())
    del out
    return eq, m, occ


def main():
    curves = {}
    for grp, members in (("商品", CMD), ("股指", IDX)):
        for sym, exch in members:
            df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
            si = symbols_info.info(sym, exch)
            for key, label, orient in CASES:
                eq, m, occ = one(df, key, orient, si, sym, exch)
                curves.setdefault((grp, label, orient), []).append(eq.rename(sym))
            del df
            gc.collect()

    for grp in ("商品", "股指"):
        print(f"\n【{grp}】e3.0/x1.0（2020 + 2021H1 + 2026，dxy 5m 仅覆盖这 800 天）")
        print(f"  {'口径':<16} {'方向':<10} {'累计':>8} {'日波动':>7} {'夏普':>6} "
              f"{'回撤':>8} {'卡玛':>6} {'笔数':>6}")
        for key, label, orient in CASES:
            lst = curves.get((grp, label, orient))
            if not lst:
                continue
            mat = pd.concat(lst, axis=1).sort_index().ffill().fillna(0.0)
            port = mat.mean(axis=1)
            r = port.resample("1D").last().dropna().diff().dropna()
            dd = float((port - port.cummax()).min())
            yrs = max((port.index[-1] - port.index[0]).days / 365.25, 1e-9)
            n = 0
            print(f"  {label:<16} {orient:<10} {port.iloc[-1]*100:+7.1f}% "
                  f"{r.std(ddof=1)*np.sqrt(252)*100:6.1f}% "
                  f"{r.mean()/r.std(ddof=1)*np.sqrt(252):+6.2f} "
                  f"{dd*100:7.1f}% {port.iloc[-1]/yrs/abs(dd):+6.2f}")


if __name__ == "__main__":
    main()
