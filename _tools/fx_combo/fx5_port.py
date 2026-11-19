"""5m 汇率因子 vs 日频：组合层面最终对照（lag 已校正）。

用法: python _tools/fx_combo/fx5_port.py [START]
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
CASES = [("fx5_usdcnh_chg_48", "long_high"), ("fx5_usdcnh_chg_48", "long_low"),
         ("fx5_usdcnh_z_48", "long_high"), ("fx5_usdcnh_z_48", "long_low"),
         ("macro_usdcnh", "long_high"), ("macro_usdcnh", "long_low")]


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
    eq = (r.equity / PRIN - 1.0)
    m = r.metrics
    del out
    return eq, m


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2020-01-01"
    curves = {}
    for grp_name, members in (("商品", CMD), ("股指", IDX)):
        for sym, exch in members:
            df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start=start)
            si = symbols_info.info(sym, exch)
            for key, orient in CASES:
                eq, m = one(df, key, orient, si, sym, exch)
                curves.setdefault((grp_name, key, orient), []).append(eq.rename(sym))
            del df
            gc.collect()

    for grp_name in ("商品", "股指"):
        print(f"\n【{grp_name}】e3.0/x1.0，各品种等权合成")
        print(f"  {'因子':<22} {'方向':<10} {'累计':>8} {'年化':>7} {'日波动':>7} "
              f"{'夏普':>6} {'回撤':>8} {'卡玛':>6}")
        for key, orient in CASES:
            lst = curves.get((grp_name, key, orient))
            if not lst:
                continue
            mat = pd.concat(lst, axis=1).sort_index().ffill().fillna(0.0)
            port = mat.mean(axis=1)
            r = port.resample("1D").last().dropna().diff().dropna()
            dd = float((port - port.cummax()).min())
            yrs = max((port.index[-1] - port.index[0]).days / 365.25, 1e-9)
            print(f"  {key.replace('fx5_usdcnh_','5m.'):<22} {orient:<10} "
                  f"{port.iloc[-1]*100:+7.1f}% {port.iloc[-1]/yrs*100:+6.1f}% "
                  f"{r.std(ddof=1)*np.sqrt(252)*100:6.1f}% "
                  f"{r.mean()/r.std(ddof=1)*np.sqrt(252):+6.2f} "
                  f"{dd*100:7.1f}% {port.iloc[-1]/yrs/abs(dd):+6.2f}")


if __name__ == "__main__":
    main()
