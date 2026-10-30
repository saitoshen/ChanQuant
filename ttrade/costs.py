"""交易成本档位：乐观 / 中性 / 压力。

国内期货的成本不再是一个「几个 bp」的拍脑袋数，而是拆成两块：

    单边成本 = 手续费 × 倍数 + 滑点跳数 × 最小变动价位

手续费走交易所口径（按成交额比例，或者按手固定，见 `symbols_info`），滑点按
**跳**算 —— 螺纹一跳 1 元、铜一跳 10 元，同样「1 跳」对应的钱差 10 倍，写成
一个 bp 永远对不齐。这也是为什么以前那套 `MARKET_COST["cn"]` 在铜上偏乐观、
在螺纹上偏悲观。

三档默认值（都能在「数据管理」页里改）：

    乐观  1.0 倍手续费 + 1.0 跳滑点   挂单能成交、盘口厚
    中性  1.0 倍手续费 + 1.5 跳滑点   默认
    压力  1.0 倍手续费 + 3.0 跳滑点   追单、流动性差、隔夜跳空

默认取「乐观」= 1 倍交易所手续费 + 1 跳滑点，和需求里写的默认口径一致。

美股不变：还是单边 5bp 手续费 + 5bp 滑点（`replay.MARKET_COST["us"]`）。
"""

from __future__ import annotations

import json
import os

from . import symbols_info
from .paths import user_dir

FILE = "costs.json"

# 顺序就是界面上的顺序
PROFILES: dict[str, dict] = {
    "optimistic": {"label": "乐观", "fee_mult": 1.0, "slip_ticks": 1.0},
    "neutral": {"label": "中性", "fee_mult": 1.0, "slip_ticks": 1.5},
    "stress": {"label": "压力", "fee_mult": 1.0, "slip_ticks": 3.0},
}

DEFAULT_PROFILE = "optimistic"

# 美股：百分比口径，和以前完全一致
US_FEE = 0.0005
US_SLIP = 0.0005


def _path() -> str:
    return os.path.join(user_dir(), FILE)


def load() -> dict:
    """读设置。返回 `{"profile": 当前档位, "profiles": {key: {...}}}`。"""
    d: dict = {}
    try:
        with open(_path(), encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            d = raw
    except Exception:  # noqa: BLE001
        pass
    out = {"profile": d.get("profile") or DEFAULT_PROFILE,
           "profiles": {k: dict(v) for k, v in PROFILES.items()}}
    for k, v in (d.get("profiles") or {}).items():
        if k in out["profiles"] and isinstance(v, dict):
            for f in ("fee_mult", "slip_ticks"):
                try:
                    out["profiles"][k][f] = float(v[f])
                except (KeyError, TypeError, ValueError):
                    continue
    if out["profile"] not in out["profiles"]:
        out["profile"] = DEFAULT_PROFILE
    return out


def save(settings: dict) -> None:
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def current() -> str:
    return load()["profile"]


def set_current(key: str) -> None:
    s = load()
    if key in s["profiles"]:
        s["profile"] = key
        save(s)


def profile(key: str | None = None) -> dict:
    """取一档的参数。传 None 用当前选中的那档。"""
    s = load()
    k = key or s["profile"]
    p = s["profiles"].get(k) or s["profiles"][DEFAULT_PROFILE]
    return {"key": k, **p}


def apply(bt, symbol: str, exchange: str, profile_key: str | None = None,
          market: str = "cn"):
    """把成本灌进 `BacktestConfig`。返回同一个对象，方便链式调用。

    美股走百分比口径，和以前完全一致；期货走「每手手续费 / 成交额比例 + 跳数
    滑点」，具体哪一种是查 `symbols_info` 得到的。
    """
    if market != "cn":
        bt.fee_rate, bt.slippage = US_FEE, US_SLIP
        bt.fee_per_lot, bt.tick_size, bt.slippage_ticks = 0.0, 0.0, 0.0
        return bt
    p = profile(profile_key)
    si = symbols_info.info(symbol, exchange)
    bt.fee_per_lot = 0.0
    bt.fee_rate = 0.0
    if str(si.get("fee_mode")) == "lot":
        bt.fee_per_lot = float(si.get("fee_value") or 0.0) * p["fee_mult"]
    else:
        bt.fee_rate = float(si.get("fee_value") or 0.0) * p["fee_mult"]
    bt.tick_size = float(si.get("tick") or 0.0)
    bt.slippage_ticks = float(p["slip_ticks"])
    # 没有 tick（新品种、反推失败）时退回一个保守的百分比滑点，否则成本会
    # 悄悄变成 0，回测结果好得不真实
    bt.slippage = 0.0 if bt.tick_size > 0 else 0.0002
    return bt


def describe(symbol: str, exchange: str, profile_key: str | None = None) -> str:
    """一句话说明这一档在这个品种上到底收多少，给界面上的提示用。"""
    p = profile(profile_key)
    si = symbols_info.info(symbol, exchange)
    tick = float(si.get("tick") or 0.0)
    if str(si.get("fee_mode")) == "lot":
        fee = f"{float(si.get('fee_value') or 0) * p['fee_mult']:g} 元/手"
    else:
        fee = f"成交额 × {float(si.get('fee_value') or 0) * p['fee_mult'] * 1e4:.2f}‱"
    slip = (f"{p['slip_ticks']:g} 跳 × {tick:g} = {p['slip_ticks'] * tick:g}"
            if tick > 0 else f"{p['slip_ticks']:g} 跳（tick 未知，按 2‱ 兜底）")
    return f"{p['label']}：手续费 {fee} + 滑点 {slip}"
