"""因子 PIT 审计：在交易日中途截断数据，因子值不该变。

**为什么要这个检查。** 因子在 t 那根 bar 上的值，只能用到 t 收盘及之前的信息。
最常见的错法是「小周期用了大周期同时刻的信息」：把「当日 09:45-10:15 的收益」
直接铺到当天每一根 bar 上，于是 09:05 那根就拿到了 10:15 的收盘价 —— 回测里这个
信号等于提前知道当天后面怎么走，IC 漂亮、实盘一上就露馅。

**怎么查。** 把行情截断在交易日中途，重算因子，截断点之前的值必须和完整数据算
出来的一模一样。只在收盘截断是不够的：那时窗口早走完了，查不出「铺满全天」这种
问题，所以截断点要落在每个时段窗口的**中间**。

用法：

    python audit_pit.py          # 5m / 1h / 1d 各跑一遍
    python audit_pit.py 5m       # 只跑一个周期
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ttrade import auxdata
from ttrade import factors as F
from ttrade import futures
from ttrade import tqdata
from ttrade.paths import data_dir

SYMBOL, EXCHANGE = "RB", "SHFE"
TAIL = {"5m": 4000, "1h": 6000, "1d": 0}   # 0 = 全量


def load(period: str) -> pd.DataFrame:
    if period == "1d":
        df = futures.load_local(SYMBOL, EXCHANGE, data_dir("cn"), adjust="hfq")
        return auxdata.attach(df, SYMBOL, EXCHANGE)
    df = tqdata.load_main(SYMBOL, EXCHANGE, period, adjust="hfq")
    # 展期 / 期权是日频快照，分钟线上必须先 shift(1) 才能挂（attach_intraday 干这个）
    df = auxdata.attach_intraday(df, SYMBOL, EXCHANGE)
    n = TAIL.get(period, 0)
    return df.tail(n) if n else df


def same(a: float, b: float) -> bool:
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def audit(period: str, n_tail: int = 12) -> int:
    """返回有问题的因子个数。"""
    df = load(period)
    if df.empty:
        print(f"{period}: 没有数据")
        return 0
    idx = pd.DatetimeIndex(df.index)
    days = idx.normalize()
    uniq = sorted(set(days))
    day = uniq[-4] if len(uniq) > 4 else uniq[-2]
    hh = idx.strftime("%H:%M")

    pos_day = np.where(days == day)[0]
    if len(pos_day) < 4:
        # 日线一天只有一根，「当天中途」不存在；能查的是「截断到第 k 天，前面各天
        # 的值会不会变」—— rolling 里错用未来数据就是这么露馅的
        cuts = [(str(idx[i])[:10], i + 1) for i in range(len(df) - 12, len(df) - 1)]
    else:
        # 覆盖每个时段窗口的中间。1h 一天就 9 根，全取；5m 一天四十多根，每 3 根
        # 取一个，够覆盖 09:45-10:15 / 13:30-14:00 / 14:00-15:00 这几个窗口
        step = 1 if len(pos_day) <= 12 else 3
        cuts = [(hh[i], int(i) + 1) for i in pos_day[2::step]]

    print(f"{period}: {len(df)} 根 {idx[0]} ~ {idx[-1]}，截断日 {day.date()}，"
          f"{len(cuts)} 个截断点")

    bad: list[tuple[str, str, str, str]] = []
    for key, spec in F.FACTORS.items():
        try:
            full = spec.func(df)
        except Exception as exc:  # noqa: BLE001
            bad.append((key, spec.category, "ERR", str(exc)[:40]))
            continue
        worst, at, shift = 0, "", ""
        for label, k in cuts:
            try:
                cut = spec.func(df.iloc[:k])
            except Exception as exc:  # noqa: BLE001
                worst, at = 99, f"{label} {type(exc).__name__}"
                break
            a = full.iloc[max(0, k - n_tail):k]
            b = cut.iloc[-len(a):]
            if len(b) != len(a):
                worst, at = 99, f"{label} 长度不一致"
                break
            n_bad = sum(0 if same(x, y) else 1
                        for x, y in zip(a.to_numpy(float), b.to_numpy(float)))
            if n_bad > worst:
                worst, at = n_bad, label
                shift = f"NaN {int(a.isna().sum())}→{int(b.isna().sum())}"
        if worst:
            bad.append((key, spec.category, f"前视 {worst}/{n_tail}", f"@{at} {shift}".strip()))

    if not bad:
        print(f"  {len(F.FACTORS)} 个因子全部 PIT 正确")
        return 0
    print(f"  {len(F.FACTORS)} 个因子里有 {len(bad)} 个有问题：")
    w = max(len(r[0]) for r in bad) + 2
    for key, cat, verdict, note in bad:
        print(f"    {key:<{w}} {cat:<6} {verdict:<12} {note}")
    return len(bad)


if __name__ == "__main__":
    periods = sys.argv[1:] or ["5m", "1h", "1d"]
    total = 0
    for p in periods:
        total += audit(p)
        print()
    print("全部通过" if total == 0 else f"共 {total} 个因子有前视")
    raise SystemExit(0 if total == 0 else 1)
