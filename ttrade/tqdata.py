"""天勤 5m / 1h 行情的读取层：多周期加载、复权、resample。

数据由 `download_tq.py` 落在 `data_tq/`，目录结构与读取约定：

    data_tq/main/5m/RB_SHFE.csv     主连 5m，列含原始 OHLC + adj_factor
    data_tq/main/1h/RB_SHFE.csv     主连 1h
    data_tq/switch/RB_SHFE.csv      主力切换表（天勤口径）
    data_tq/contracts/5m/RB_SHFE.csv 真实合约 5m（不复权，单合约本身无换月）

**周期是「源周期 + 合并根数」而不是独立下载**。天勤只下 5m 和 1h 两种，其余周期
在本地按「同一个交易日内的连续 N 根」合并：

    15m = 5m × 3     30m = 5m × 6     2h = 1h × 2     4h = 1h × 4

不用 `df.resample("2h")` 这种按钟点对齐的写法：国内期货一天分四段
（21:00-23:00 夜盘、09:00-11:30、13:30-15:00），按钟点切会把夜盘和日盘拼进同一个
bar，或者切出 20:00-22:00 这种根本不存在的区间。按「日内第几根」分组，
跨节假日的夜盘归属也能自动对齐到交易日（见 `trading_days`）。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import tqstore
from .paths import data_dir

# 周期名 → (源周期, 合并根数)。源周期就是实际落盘的那个。
PERIODS: dict[str, tuple[str, int]] = {
    "5m": ("5m", 1),
    "15m": ("5m", 3),
    "30m": ("5m", 6),
    "1h": ("1h", 1),
    "2h": ("1h", 2),
    "4h": ("1h", 4),
}

# 展示顺序（GUI 的周期下拉框按这个排）
PERIOD_ORDER = ["5m", "15m", "30m", "1h", "2h", "4h"]

PERIOD_LABEL = {
    "5m": "5 分钟",
    "15m": "15 分钟",
    "30m": "30 分钟",
    "1h": "1 小时",
    "2h": "2 小时",
    "4h": "4 小时",
}


def tq_root() -> str:
    return os.path.join(os.path.dirname(data_dir("cn").rstrip("/\\")), "data_tq")


def main_path(symbol: str, exchange: str, period: str) -> str:
    src = PERIODS.get(period, (period, 1))[0]
    return os.path.join(tq_root(), "main", src, f"{symbol}_{exchange}.csv")


def switch_path(symbol: str, exchange: str) -> str:
    return os.path.join(tq_root(), "switch", f"{symbol}_{exchange}.csv")


def has_period(symbol: str, exchange: str, period: str) -> bool:
    """仓库里有没有这个品种的这个周期。查 DuckDB，不再看 CSV。"""
    if tqstore.exists() and tqstore.bar_count(symbol, exchange, PERIODS.get(period, (period, 1))[0]):
        return True
    return os.path.exists(main_path(symbol, exchange, period))


def available_periods(symbol: str, exchange: str) -> list[str]:
    return [p for p in PERIOD_ORDER if has_period(symbol, exchange, p)]


def trading_days(ts: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """K 线时间戳 → 交易日。夜盘（20:00 之后）算到下一个有日盘的日期。

    与 `download_tq.trading_days` 是同一套规则，两处必须一致：下载时用它对齐复权
    因子，读取时用它给 resample 分组。写错会让夜盘 bar 归到前一个交易日，
    15m/2h 这类合并周期在每天开盘那一段就会拼错。
    """
    s = pd.Series(pd.to_datetime(ts))
    day_session = pd.DatetimeIndex(sorted(set(s[s.dt.hour < 20].dt.normalize())))
    dates = s.dt.normalize().to_numpy()
    out = pd.Series(dates)
    night = (s.dt.hour >= 20).to_numpy()
    if night.any() and len(day_session):
        pos = day_session.values.searchsorted(dates[night], side="right")
        nxt = np.where(
            pos < len(day_session),
            day_session.values[np.clip(pos, 0, len(day_session) - 1)],
            dates[night],
        )
        out.iloc[np.flatnonzero(night)] = nxt
    return out


def load_main(
    symbol: str,
    exchange: str,
    period: str = "1h",
    adjust: str = "qfq",
    start=None,
    end=None,
) -> pd.DataFrame:
    """读主连行情，返回 index=datetime 的 DataFrame。

    `adjust` 与 `futures.load_local` 保持同一套语义：
      - `qfq`：前复权，最后一根等于真实盘面价（默认，界面显示用）；
      - `hfq`：后复权，落盘的就是这个口径，历史固定不变；
      - `none`：不复权，保留换月跳空。

    数据从 DuckDB 读（`tqstore`）。早期版本按品种存 CSV，这里保留 CSV 兜底，
    方便还没迁移的老数据继续能用；两条路径返回的列完全一致。

    列：open/high/low/close/volume/open_oi/close_oi/adj_factor/trading_day。
    """
    src, n = PERIODS.get(period, (period, 1))
    df = tqstore.load_bars(symbol, exchange, src, start, end) if tqstore.exists() else pd.DataFrame()

    if df.empty:
        path = main_path(symbol, exchange, period)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"没有 {symbol}_{exchange} 的 {period} 数据。"
                "先跑 `python download_tq.py --main 5m,1h --all`。"
            )
        df = pd.read_csv(path, parse_dates=["datetime"]).rename(columns={"datetime": "dt"})
        df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0
        if "trading_day" not in df.columns:
            df["trading_day"] = trading_days(df.index).to_numpy()

    mode = (adjust or "none").lower()
    f = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0).to_numpy(float)
    if mode == "qfq":
        last = f[-1] if len(f) and f[-1] > 0 else 1.0
        f = f / last
    elif mode == "none":
        f = np.ones(len(f))
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].to_numpy(float) * f

    if n > 1:
        out = merge_bars(out, n)
    return out


def merge_bars(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """把同一交易日内的连续 n 根合成一根。

    - open 取第一根，close 取最后一根，high/low 取极值，量额持仓累加；
    - datetime 用**最后一根**的时间（bar 结束时刻），与天勤原生 bar 的口径一致；
    - 一天内不足 n 根的尾巴保留成一根（不丢数据）—— 期货日盘常有 3 根 1h
      而夜盘只有 2 根，按整日切会把尾巴切掉。
    """
    if n <= 1 or df.empty:
        return df
    d = df.copy()
    d["_g"] = d.groupby("trading_day").cumcount() // n
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "open_oi": "last",
        "close_oi": "last",
        "adj_factor": "last",
        "trading_day": "first",
    }
    agg = {k: v for k, v in agg.items() if k in d.columns}
    out = d.groupby(["trading_day", "_g"], sort=True).agg(agg)
    last_dt = d.groupby(["trading_day", "_g"], sort=True)["_g"].size()
    idx = d.groupby(["trading_day", "_g"], sort=True).apply(
        lambda g: g.index[-1], include_groups=False
    )
    out.index = pd.DatetimeIndex(idx.to_numpy())
    out.index.name = "datetime"
    _ = last_dt
    return out


def summary() -> dict:
    """统计天勤数据的覆盖情况，给界面和自检用。"""
    st = tqstore.stats()
    out = {"root": tq_root(), "exists": bool(st.get("exists")), "db": st.get("path"),
           "size_mb": st.get("size_mb", 0), "periods": {}, "switch_files": 0}
    if not out["exists"]:
        return out
    for p, info in (st.get("periods") or {}).items():
        out["periods"][p] = {
            "files": info.get("symbols", 0),
            "rows": info.get("rows", 0),
            "first": info.get("first", ""),
            "last": info.get("last", ""),
        }
    out["switch_files"] = (st.get("switch") or {}).get("symbols", 0)
    return out
