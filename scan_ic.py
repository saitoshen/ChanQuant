"""全样本因子 IC 扫描：逐品种算时序 Spearman IC，再跨品种取均值。

README 里「94 个品种的全样本」那两张表就是它跑出来的。用法：

    python scan_ic.py                        # 默认两组：展期 vs 动量、期权六因子
    python scan_ic.py --keys roll_yield,mom_60 --horizons 5,20,60
    python scan_ic.py --min-rows 1000        # 只要历史够长的品种

**为什么是「逐品种 IC 再平均」而不是「把所有品种拼成一条序列算一个 IC」**：
后者会把「品种之间价格水平不同」当成信号。沪铜 70000、螺纹 3000，拼起来排序时
沪铜永远排在最上面，算出来的 IC 只反映品种间的常数差异，与因子无关。时序 IC
每个品种各算一次，只比较同一品种在不同时点的相对位置，才是单资产择时的正确问法。

**为什么要求 `len(df) >= min_rows`**：新上市品种只有几百根，60 日动量加 5 日前向
收益之后剩下的样本不足以让秩相关稳定，留着只会把均值往噪声方向拉。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import factors as F  # noqa: E402
from ttrade import futures as FU  # noqa: E402
from ttrade.factor_metrics import forward_return, spearman  # noqa: E402

# 默认两组：第一组验证「展期 vs 动量互补」，第二组验证期权因子里哪类有信息
DEFAULT_GROUPS = [
    ("展期 vs 动量", ["roll_yield", "mom_60"], [5, 20, 60]),
    ("期权六因子", ["iv_term", "pcr", "iv_rv", "iv_atm", "iv_change", "iv_skew"], [5, 20]),
]


def one_ic(df: pd.DataFrame, key: str, horizon: int, min_obs: int) -> float | None:
    """单个品种单个 horizon 的时序 IC。算不出就返回 None（不写 0）。"""
    try:
        f = F.standardize(F.raw(df, key))
    except Exception:
        # 展期/期权因子在没有 aux 数据的品种上会缺列，属于正常情况
        return None
    r = forward_return(df, horizon)
    d = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
    if len(d) < min_obs:
        return None
    v = spearman(d["f"], d["r"])
    return float(v) if np.isfinite(v) else None


def scan(files: list[str], keys: list[str], horizons: list[int],
         min_rows: int, min_obs: int) -> dict:
    res = {k: {h: [] for h in horizons} for k in keys}
    for i, path in enumerate(files, 1):
        sym, _, ex = os.path.basename(path)[:-4].partition("_")
        try:
            df = FU.load_local(sym, ex, "data_futures")
        except Exception as exc:
            print(f"  [!] {sym}_{ex} 读不了：{exc}", flush=True)
            continue
        if df is None or len(df) < min_rows:
            continue
        for k in keys:
            for h in horizons:
                v = one_ic(df, k, h, min_obs)
                if v is not None:
                    res[k][h].append(v)
        if i % 25 == 0:
            print(f"  ...{i}/{len(files)}", flush=True)
    return res


def show(title: str, res: dict, keys: list[str], horizons: list[int]) -> None:
    print(f"\n### {title}")
    print("| 因子 | " + " | ".join(f"h={h}" for h in horizons) + " |")
    print("|---" * (len(horizons) + 1) + "|")
    for k in keys:
        cells = []
        for h in horizons:
            vs = res[k][h]
            if not vs:
                cells.append("-")
                continue
            m = float(np.mean(vs))
            sd = float(np.std(vs, ddof=1)) if len(vs) > 1 else float("nan")
            pos = 100.0 * sum(1 for v in vs if v > 0) / len(vs)
            cells.append(f"{m:+.3f} / {sd:.3f}（{pos:.0f}% 为正，{len(vs)} 个）")
        print(f"| {F.FACTORS[k].label} | " + " | ".join(cells) + " |")


def main() -> int:
    ap = argparse.ArgumentParser(description="全样本因子 IC 扫描")
    ap.add_argument("--keys", default="", help="逗号分隔的因子 key，留空则跑默认两组")
    ap.add_argument("--horizons", default="", help="逗号分隔的持有期，留空用默认")
    ap.add_argument("--min-rows", type=int, default=500, help="品种最少 K 线数")
    ap.add_argument("--min-obs", type=int, default=250, help="单品种最少有效样本数")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join("data_futures", "*.csv")))
    if not files:
        print("data_futures 下没有 CSV，先跑 download.py")
        return 1
    print(f"日线品种 {len(files)} 个", flush=True)

    if args.keys:
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
        bad = [k for k in keys if k not in F.FACTORS]
        if bad:
            print(f"未知因子：{bad}")
            return 1
        horizons = [int(h) for h in args.horizons.split(",") if h.strip()] or [5, 20, 60]
        show(f"{len(keys)} 个因子", scan(files, keys, horizons, args.min_rows, args.min_obs),
             keys, horizons)
        return 0

    for title, keys, horizons in DEFAULT_GROUPS:
        res = scan(files, keys, horizons, args.min_rows, args.min_obs)
        show(title, res, keys, horizons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
