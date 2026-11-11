"""交易成本档位：乐观 / 中性 / 压力。

国内期货的成本不再是一个「几个 bp」的拍脑袋数，而是拆成两块：

    单边成本 = 手续费 × 倍数 + 滑点跳数 × 最小变动价位

手续费走交易所口径（按成交额比例，或者按手固定，见 `symbols_info`），滑点按
**跳**算 —— 螺纹一跳 1 元、铜一跳 10 元，同样「1 跳」对应的钱差 10 倍，写成
一个 bp 永远对不齐。这也是为什么以前那套 `MARKET_COST["cn"]` 在铜上偏乐观、
在螺纹上偏悲观。

五档默认值（都能在「数据管理」页里改）：

    挂单  1.0 倍手续费 + 0.1 跳滑点   限价挂单成交；网格、配对套利用这一档
    乐观  1.0 倍手续费 + 0.5 跳滑点   默认；挂单能成交、盘口厚
    中性  1.0 倍手续费 + 1.0 跳滑点   正常追价
    压力  1.0 倍手续费 + 2.0 跳滑点   追单、流动性差、隔夜跳空
    按 bp 1.0 倍手续费 + 3‱ 滑点     不看 tick，直接按价格比例

「按 bp」是给「不想被 tick 牵着走」的场合准备的：一篮子品种里有的跳 1 元、
有的跳 50 元，按跳收会让便宜品种的成本被放大、贵品种被缩小；按 bp 就是统一
的「成交额 × 万几」。代价是离实盘远一点 —— 真实滑点基本是「跳」的整数倍。

**滑点为什么还要乘复权因子。** 回测跑的是前复权价（`qfq`，最后一根等于真实
盘面价），历史价格被整体缩放过：`P_adj = P_raw × f`，`f = adj_factor / 最后一根
的 adj_factor`。价格变了，但**一跳是真实世界里的固定金额**（螺纹永远是 1 元/吨），
换算到复权价上就变成 `tick × f`。不换算的话，五年前那段的滑点会按 0.67 倍收，
越往前越便宜 —— 而 5 分钟策略的成本敏感度恰恰最高，这个偏差会直接把结论带偏。

比例型成本（手续费率、百分比滑点）不用换算：它们是「占名义额的百分比」，
分子分母同时被 f 缩放，自己就抵消了。所以这里只调按跳的滑点。

美股不变：还是单边 5bp 手续费 + 5bp 滑点（`replay.MARKET_COST["us"]`）。
"""

from __future__ import annotations

import json
import os

from . import symbols_info
from .paths import user_dir

FILE = "costs.json"

# 默认值改过一次：v2.4.8 起 1.0/1.5/3.0 → 0.5/1.0/2.0。存了老值又没动过的配置
# 要跟着换，否则「默认值」只是写在这里给人看的一段文字。用 rev 判：用户手动
# 改过的（rev 已是最新）一律尊重，不再覆盖
REV = 2

# 顺序就是界面上的顺序。
#
# `mode` 决定滑点怎么算：
#   tick —— 滑点 = 跳数 × 最小变动价位（品种之间差十倍，最贴近实盘）
#   bp   —— 滑点 = 价格 × N‱（不看 tick，一篮子品种用同一个假设）
#
# 两种口径各留一份数值，用哪个由 `mode` 说了算 —— 切档位时另一份不会丢，
# 用户调过「压力 2 跳」，切到 bp 档再切回来还是 2 跳。
PROFILES: dict[str, dict] = {
    "optimistic": {"label": "乐观", "mode": "tick", "fee_mult": 1.0,
                   "slip_ticks": 0.5, "slip_bp": 0.0},
    # 挂单 / 被动成交：网格、配对套利这类策略每一档都是**限价挂单**，成交价就是
    # 你挂的那个价，不是「穿过去再追回来」。按 1 跳收等于假设每笔都要追一次价，
    # 会把这类薄利策略直接收到负期望（实测螺纹：0.5 跳 → −0.17%，0.1 跳 → +0.09%）。
    "maker": {"label": "挂单", "mode": "tick", "fee_mult": 1.0,
              "slip_ticks": 0.1, "slip_bp": 0.0},
    "neutral": {"label": "中性", "mode": "tick", "fee_mult": 1.0,
                "slip_ticks": 1.0, "slip_bp": 0.0},
    "stress": {"label": "压力", "mode": "tick", "fee_mult": 1.0,
               "slip_ticks": 2.0, "slip_bp": 0.0},
    "bp": {"label": "按 bp", "mode": "bp", "fee_mult": 1.0,
           "slip_ticks": 0.0, "slip_bp": 3.0},
}

DEFAULT_PROFILE = "optimistic"

# 美股：百分比口径，和以前完全一致
US_FEE = 0.0005
US_SLIP = 0.0005


def _path() -> str:
    return os.path.join(user_dir(), FILE)


def load() -> dict:
    """读设置。返回 `{"profile": 当前档位, "profiles": {key: {...}}}`。

    老版本存下来的文件没有 `rev`，里面的三档值就是当时的默认值 —— 一律按新
    默认值重来（用户没动过，没什么可保留的）。有 `rev` 且等于当前版本的才认。
    """
    d: dict = {}
    try:
        with open(_path(), encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            d = raw
    except Exception:  # noqa: BLE001
        pass
    out = {"profile": d.get("profile") or DEFAULT_PROFILE,
           "rev": REV,
           "profiles": {k: dict(v) for k, v in PROFILES.items()}}
    fresh = d.get("rev") == REV
    for k, v in (d.get("profiles") or {}).items():
        if fresh and k in out["profiles"] and isinstance(v, dict):
            for f in ("fee_mult", "slip_ticks", "slip_bp"):
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
    # 只落盘「用户能改的那三个数」。`label` / `mode` 是程序里的口径，抄进文件
    # 只会变成一份会和代码漂移的副本（改个档位名，老文件里还留着旧名字）
    settings = {
        "profile": settings.get("profile") or DEFAULT_PROFILE,
        "rev": REV,
        "profiles": {
            k: {f: float((v or {}).get(f) or 0.0)
                for f in ("fee_mult", "slip_ticks", "slip_bp")}
            for k, v in (settings.get("profiles") or {}).items()
            if k in PROFILES
        },
    }
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


def short_text(p: dict) -> str:
    """一行说明这一档的成本口径（不含具体品种）。下拉框、提示条共用一处，
    免得「按 bp」那种档位在某个界面被写成「0 跳滑点」。"""
    fee = f"{float(p.get('fee_mult') or 0):g} 倍手续费"
    if str(p.get("mode") or "tick") == "bp":
        return f"{fee} + {float(p.get('slip_bp') or 0):g}‱ 滑点"
    return f"{fee} + {float(p.get('slip_ticks') or 0):g} 跳滑点"


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
    if str(p.get("mode") or "tick") == "bp":
        # 按 bp：直接给百分比滑点，不看 tick。跳数口径那两列清零，免得
        # `slip_amount` 里 `tick_size > 0` 那条分支又被走到
        bt.tick_size = 0.0
        bt.slippage_ticks = 0.0
        bt.slippage = float(p.get("slip_bp") or 0.0) / 1e4
        return bt
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
    if str(p.get("mode") or "tick") == "bp":
        slip = f"价格 × {float(p.get('slip_bp') or 0):g}‱（不看 tick）"
    else:
        slip = (f"{p['slip_ticks']:g} 跳 × {tick:g} = {p['slip_ticks'] * tick:g} 元"
                if tick > 0 else f"{p['slip_ticks']:g} 跳（tick 未知，按 2‱ 兜底）")
        if tick > 0:
            # 前复权价上要按当时的因子缩放，不说清楚用户会以为「明明设了 1 跳，
            # 五年前那段怎么只收了 0.67 跳」
            slip += "（回测里按当日复权因子缩放）"
    return f"{p['label']}：手续费 {fee} + 滑点 {slip}"
