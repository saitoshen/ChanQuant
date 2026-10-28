"""合约规格：合约乘数、保证金率。

回测和实盘都要回答「一手值多少钱、占多少保证金」，否则「50 万资金」没法折算成
手数。两个数都不写死在代码里：

- **乘数**从主力连续的成交额 / 成交量反推：`data_futures/*.csv` 里本来就有
  `volume`（手）和 `amount`（万元），成交额 ÷（手数 × 价格）就是每手多少单位。
  实测螺纹 10、铜 5、黄金 1000、股指 300、国债 10000，和交易所公布的一致。
- **保证金率**没有本地数据，按品种类别给默认值（商品 10%、股指 12%、国债 3%）。
  实盘有 `quote.margin` 时以交易所为准，这里只是兜底。

本模块只依赖 pandas 和标准库，不联网。
"""

from __future__ import annotations

import os

import pandas as pd

from .paths import data_dir

# 国债期货的保证金率明显低于商品：T/TF/TS/TL 大约 2%~5%，折 20~50 倍杠杆
_BOND = ("T", "TF", "TS", "TL")
_INDEX = ("IF", "IH", "IC", "IM")

_mult_cache: dict[str, float] = {}


def multiplier(symbol: str, exchange: str, root: str | None = None) -> float:
    """合约乘数（每手多少单位）。查不到时返回 1.0。

    取最近 500 根的中位数：单根会被停板、结算价、成交额单位误差影响，中位数
    不受个别异常值干扰。成交额单位是**万元**（Tushare `fut_daily` 口径）。
    """
    key = f"{symbol}_{exchange}"
    if key in _mult_cache:
        return _mult_cache[key]
    root = root or data_dir("cn")
    path = os.path.join(root, f"{key}.csv")
    if not os.path.exists(path):
        return 1.0
    try:
        df = pd.read_csv(path, usecols=["close", "volume", "amount"])
    except Exception:  # noqa: BLE001
        return 1.0
    df = df.dropna()
    df = df[(df["volume"] > 0) & (df["close"] > 0) & (df["amount"] > 0)]
    if df.empty:
        return 1.0
    ratio = (df["amount"] * 10000.0 / (df["volume"] * df["close"])).tail(500)
    val = float(ratio.median())
    if not (0.1 < val < 1e6):
        return 1.0
    # 乘数都是整数（10 / 5 / 300 / 10000），反推值在 0.1% 内浮动，四舍五入到整数
    val = float(round(val))
    _mult_cache[key] = val
    return val


def margin_rate(symbol: str) -> float:
    """默认保证金率。杠杆 = 1 / 保证金率。"""
    s = (symbol or "").upper()
    if s in _BOND:
        return 0.03            # 国债：约 33 倍
    if s in _INDEX:
        return 0.12            # 股指：约 8 倍
    return 0.10                # 商品：约 10 倍


def per_lot_margin(symbol: str, exchange: str, price: float, mult: float = 0.0) -> float:
    """每手保证金 = 价格 × 乘数 × 保证金率。"""
    m = float(mult or 0) or multiplier(symbol, exchange)
    if price <= 0 or m <= 0:
        return 0.0
    return float(price) * m * margin_rate(symbol)


def clear_cache() -> None:
    _mult_cache.clear()
