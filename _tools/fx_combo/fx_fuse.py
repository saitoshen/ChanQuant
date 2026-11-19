"""日频 + 5m 融合：寻找「低换手 + 信息及时」的组合方式。

用法: python _tools/fx_combo/fx_fuse.py [SYM...]

方案：
  1 日频单（baseline）
  2 5m 单（反向，与日频同向）
  3 custom 加权 0.8/0.2、0.7/0.3、0.5/0.5
  4 5m 降频：每天只在 09:00 / 13:30 / 21:00 各更新一次（3 次/日）
  5 日频 + 降频 5m 加权
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
from ttrade import factors as F
from ttrade import symbols_info, tqdata

PRIN = 1_000_000.0
D5 = "fx5_usdcnh_chg_48"
SLOW = "fx5slow_usdcnh_chg48"


def _slow(df: pd.DataFrame) -> pd.Series:
    """5m 因子降频：只在 09:00 / 13:30 / 21:00 取新值，其余 ffill。"""
    s = F.FACTORS[D5].func(df)
    hh = df.index.hour
    mm = df.index.minute
    mask = ((hh == 9) & (mm == 0)) | ((hh == 13) & (mm == 30)) | ((hh == 21) & (mm == 0))
    return s.where(mask).ffill()


def register_slow():
    if SLOW in F.FACTORS:
        return
    F.FACTORS[SLOW] = F.FactorSpec(
        SLOW, "汇率5m·降频3次日", _slow, "汇率5m",
        "5m 汇率因子降频到每天 3 个采样点（09:00/13:30/21:00），其余 ffill。",
        formula="chg48 每 5min；采样点 = {09:00,13:30,21:00}，其余 ffill",
    )


CASES = [
    ("日频单", "single", ["macro_usdcnh"], {}),
    ("5m单(反)", "single", [D5], {"reverse": {D5: True}}),
    ("5m降频", "single", [SLOW], {"reverse": {SLOW: True}}),
    ("加权8:2", "multi", ["macro_usdcnh", D5],
     {"weight_mode": "custom", "weights": {"macro_usdcnh": 0.8, D5: 0.2},
      "reverse": {D5: True}}),
    ("加权7:3", "multi", ["macro_usdcnh", D5],
     {"weight_mode": "custom", "weights": {"macro_usdcnh": 0.7, D5: 0.3},
      "reverse": {D5: True}}),
    ("加权5:5", "multi", ["macro_usdcnh", D5],
     {"weight_mode": "custom", "weights": {"macro_usdcnh": 0.5, D5: 0.5},
      "reverse": {D5: True}}),
    ("日频+降频7:3", "multi", ["macro_usdcnh", SLOW],
     {"weight_mode": "custom", "weights": {"macro_usdcnh": 0.7, SLOW: 0.3},
      "reverse": {SLOW: True}}),
    ("日频+降频5:5", "multi", ["macro_usdcnh", SLOW],
     {"weight_mode": "custom", "weights": {"macro_usdcnh": 0.5, SLOW: 0.5},
      "reverse": {SLOW: True}}),
]


def one(df, si, sym, exch, mode, keys, extra):
    cfg = FS.FactorConfig(window=545, method="z", entry_z=3.0, exit_z=1.0,
                          orient="long_low", horizon=12, **extra)
    out = FS.run_factor_strategy(df, mode, keys, cfg)
    c = BT.BacktestConfig(initial_cash=PRIN)
    c.sizing = "capital"
    c.capital = PRIN * float(si["margin"])
    c.multiplier = float(si["mult"])
    c.margin_rate = float(si["margin"])
    c.bars_per_year = BT.bars_per_year(out.df.index)
    costs.apply(c, sym, exch)
    r = BT.run_backtest(out.df, c)
    m = r.metrics
    tot = float(r.equity.iloc[-1]) / PRIN - 1.0
    n = int(m["n_trades"] or 0)
    occ = float((out.df["signal"].to_numpy() != 0).mean())
    del out
    return m["sharpe"], tot, n, (tot / n if n else 0.0), m["max_dd"], occ


def main():
    register_slow()
    syms = []
    for a in sys.argv[1:]:
        s, e = a.split(":")
        syms.append((s, e))
    if not syms:
        syms = [("AU", "SHFE"), ("AG", "SHFE"), ("CU", "SHFE"), ("IM", "CFFEX")]

    for sym, exch in syms:
        df = tqdata.load_main(sym, exch, "5m", adjust="qfq", start="2020-01-01")
        si = symbols_info.info(sym, exch)
        print(f"\n== {sym}  e3.0/x1.0  orient=long_low ==")
        print(f"  {'方案':<14} {'夏普':>6} {'累计':>8} {'笔数':>5} {'每笔':>8} "
              f"{'回撤':>8} {'占用':>6}")
        for nm, mode, keys, extra in CASES:
            sh, tot, n, per, dd, occ = one(df, si, sym, exch, mode, keys, extra)
            print(f"  {nm:<14} {sh:+6.2f} {tot*100:+7.1f}% {n:5d} "
                  f"{per*100:+7.3f}% {dd*100:7.1f}% {occ*100:5.1f}%")
        del df
        gc.collect()


if __name__ == "__main__":
    main()
