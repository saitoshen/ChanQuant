"""天勤数据链路的离线自测：切换表、复权因子、多周期合并。

不联网、不依赖天勤账户，只读 `data_tq/` 里已经下好的文件。有数据就全量校验，
没数据就跳过（打包后的环境本来就没有这份数据，不该因此报错）。

    python smoke_test_tq.py            # 校验全部已下载品种
    python smoke_test_tq.py RB_SHFE    # 只校验一个
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import tqdata as T  # noqa: E402
from ttrade import tqstore as S  # noqa: E402

_ok = _fail = _skip = 0
_msgs: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _ok, _fail
    if cond:
        _ok += 1
    else:
        _fail += 1
        _msgs.append(f"[FAIL] {name}  {detail}")


def skip(name: str, why: str) -> None:
    global _skip
    _skip += 1
    _msgs.append(f"[SKIP] {name}  {why}")


def keys() -> list[str]:
    return [f"{s}_{e}" for s, e in S.switch_symbols()]


def check_one(key: str) -> None:
    sym, ex = key.rsplit("_", 1)
    sw = S.load_switch(sym, ex)
    check(f"{key} 切换表非空", len(sw) > 0, f"{len(sw)} 条")
    if not len(sw):
        return
    r = pd.to_numeric(sw["ratio"], errors="coerce")
    check(f"{key} ratio 落在合理区间", bool(((r > 0.5) & (r < 2.0)).all()),
          f"min={r.min():.4f} max={r.max():.4f}")
    check(f"{key} ratio 无缺失", bool(r.notna().all()), f"缺 {int(r.isna().sum())} 条")
    # 天勤主力只会向远月滚，切换后不应回到更早的月份
    later = sw["new_ts"].str.extract(r"(\d{4})\.")[0].astype(int)
    earlier = sw["old_ts"].str.extract(r"(\d{4})\.")[0].astype(int)
    back = int((later <= earlier).sum())
    check(f"{key} 换月方向向前", back == 0, f"{back} 次回切")

    for period in ("5m", "1h"):
        if not T.has_period(sym, ex, period):
            continue
        df = T.load_main(sym, ex, period, adjust="none")
        check(f"{key} {period} 有数据", len(df) > 100, f"{len(df)} 行")
        if len(df) < 100:
            continue
        check(f"{key} {period} 价格为正",
              bool((df[["open", "high", "low", "close"]] > 0).all().all()))
        check(f"{key} {period} high≥low",
              bool((df["high"] >= df["low"]).all()))
        check(f"{key} {period} 时间戳唯一", bool(df.index.is_unique))

        # 后复权因子必须单调不增：越早的段落乘得越大
        f = df["adj_factor"].to_numpy(float)
        check(f"{key} {period} 复权因子单调不增", bool((np.diff(f) <= 1e-12).all()),
              f"最大反弹 {np.diff(f).max():.2e}")
        check(f"{key} {period} 最新段因子为 1", abs(f[-1] - 1.0) < 1e-9, f"{f[-1]}")

        # 复权后，换月 bar 的跳空不应显著大于平常 bar
        adj = df["close"] * df["adj_factor"]
        ret = adj.pct_change().abs()
        swd = set(pd.to_datetime(sw["switch_date"]).dt.normalize())
        is_sw = pd.Series(df["trading_day"]).isin(swd).to_numpy()
        if is_sw.sum() >= 3:
            m_sw = float(ret[is_sw].mean())
            m_norm = float(ret[~is_sw].mean())
            check(f"{key} {period} 换月跳空已消除",
                  m_sw < max(m_norm * 3, 0.002),
                  f"换月 {m_sw:.5f} vs 平常 {m_norm:.5f}")


def check_resample(key: str) -> None:
    sym, ex = key.rsplit("_", 1)
    if not T.has_period(sym, ex, "5m"):
        return
    d5 = T.load_main(sym, ex, "5m")
    for period, n in (("15m", 3), ("30m", 6)):
        if not T.has_period(sym, ex, period):
            continue
        dn = T.load_main(sym, ex, period)
        check(f"{key} {period} 成交量守恒",
              abs(float(dn["volume"].sum()) - float(d5["volume"].sum())) < 1e-6,
              f"{dn['volume'].sum():.0f} vs {d5['volume'].sum():.0f}")
        check(f"{key} {period} 最高价不低于 5m",
              float(dn["high"].max()) >= float(d5["high"].max()) - 1e-9)
        check(f"{key} {period} 行数约为 5m 的 1/{n}",
              abs(len(dn) - len(d5) / n) <= len(d5) * 0.02,
              f"{len(dn)} vs {len(d5) / n:.0f}")


def check_trading_day() -> None:
    """夜盘 bar 必须归到下一个有日盘的交易日。"""
    idx = pd.DatetimeIndex(
        ["2026-09-01 21:00", "2026-09-01 22:00", "2026-09-02 09:00", "2026-09-02 14:00"]
    )
    td = T.trading_days(pd.Series(idx)).dt.normalize().tolist()
    want = [pd.Timestamp("2026-09-02")] * 4
    check("交易日映射（夜盘归次日）", td == want, f"{[str(x.date()) for x in td]}")
    # 周五夜盘要归到下周一（用真实的 2026-10-23 周五 / 10-26 周一）
    idx2 = pd.DatetimeIndex(["2026-10-23 21:00", "2026-10-26 09:00"])
    td2 = T.trading_days(pd.Series(idx2)).dt.normalize().tolist()
    check("交易日映射（周五夜盘归下周一）",
          td2[0] == pd.Timestamp("2026-10-26") and td2[1] == pd.Timestamp("2026-10-26"),
          f"{[str(x.date()) for x in td2]}")


def main() -> int:
    ks = keys()
    if not ks:
        print("仓库里还没有切换表，先跑 download_tq.py --switch --all")
        return 1
    if len(sys.argv) > 1:
        ks = [k for k in ks if k == sys.argv[1]] or ks

    check_trading_day()
    for k in ks:
        try:
            check_one(k)
            check_resample(k)
        except Exception as exc:
            check(f"{k} 读取", False, f"{type(exc).__name__}: {exc}")

    print("\n".join(_msgs))
    print("=" * 60)
    print(f"通过 {_ok} 项，失败 {_fail} 项，跳过 {_skip} 项")
    print("=" * 60)
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
