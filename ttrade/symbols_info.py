"""品种基本信息：最小跳动、合约乘数、保证金率、手续费、交易时段。

这几项以前散在三处：`specs.multiplier()` 从成交额反推乘数、`specs.margin_rate()`
按品种类别拍一个保证金率、`replay.MARKET_COST` 给**整个市场**一个 bp 成本。
前两个够用，第三个太粗 —— 螺纹一手 10 吨、铜一手 5 吨，同样「1 个 bp」收的钱
差一个数量级；而真实的交易成本是「交易所手续费 + 几个 tick 的滑点」，跟名义额
没有固定比例关系。

所以这里立一张表：**每个品种一行**，用户可以在「数据管理」页里改。没改的吃
默认值（按交易所的常见口径给），改过的存 `symbols.json` —— 分开存是为了升级
时不会把用户改过的数冲掉，也不会把新版本补进来的默认值顶掉。

查询入口是 `info()`，**永远返回完整字典、不抛异常**：回测不该因为一个品种没
登记就跑不起来，宁可退回一个保守的默认值。
"""

from __future__ import annotations

import json
import math
import os
from decimal import Decimal

from . import specs
from .paths import user_dir

FILE = "symbols.json"

# 交易所代号 → 中文。列表里按这个分组显示。
EXCHANGES = {
    "SHFE": "上期所",
    "DCE": "大商所",
    "CZCE": "郑商所",
    "CFFEX": "中金所",
    "INE": "能源中心",
    "GFEX": "广期所",
}

# 最小变动价位（元 / 每单位）。只收常见的那些，查不到的按成交价反推
# （`infer_tick`）—— 反推比瞎猜一个数可靠得多，见那个函数的说明。
TICKS = {
    # 上期所
    "CU": 10.0, "AL": 5.0, "ZN": 5.0, "PB": 5.0, "NI": 10.0, "SN": 10.0,
    "SS": 5.0, "AU": 0.02, "AG": 1.0, "RB": 1.0, "WR": 1.0, "HC": 1.0,
    "BU": 1.0, "RU": 5.0, "SP": 2.0, "FU": 1.0, "LU": 1.0, "AO": 1.0,
    "BR": 5.0, "AD": 5.0, "OP": 1.0, "SH": 1.0,
    # 能源中心
    "SC": 0.1, "BC": 10.0, "NR": 5.0, "EC": 0.1,
    # 大商所
    "A": 1.0, "B": 1.0, "C": 1.0, "CS": 1.0, "M": 1.0, "Y": 2.0, "P": 2.0,
    "I": 0.5, "J": 0.5, "JM": 0.5, "JD": 1.0, "L": 1.0, "V": 1.0,
    "PP": 1.0, "EG": 1.0, "EB": 1.0, "PG": 1.0, "FB": 0.5, "BB": 0.5,
    "LH": 5.0, "LG": 0.5, "RR": 1.0,
    # 郑商所
    "CF": 5.0, "CY": 5.0, "SR": 1.0, "TA": 2.0, "MA": 1.0, "FG": 1.0,
    "SA": 1.0, "UR": 1.0, "SF": 2.0, "SM": 2.0, "ZC": 0.2, "AP": 1.0,
    "CJ": 5.0, "PK": 2.0, "PX": 2.0, "PR": 2.0, "PF": 2.0, "OI": 1.0,
    "RM": 1.0, "RS": 1.0, "WH": 1.0, "PM": 1.0, "RI": 1.0, "JR": 1.0,
    "LR": 1.0, "ER": 1.0, "RO": 1.0, "ME": 1.0, "TC": 0.2, "WS": 1.0,
    "WT": 1.0,
    # 中金所
    "IF": 0.2, "IH": 0.2, "IC": 0.2, "IM": 0.2,
    "T": 0.005, "TF": 0.005, "TS": 0.002, "TL": 0.01,
    # 广期所
    "SI": 5.0, "LC": 50.0, "PS": 5.0, "PT": 0.05, "PD": 0.05,
}

# 交易时段。按交易所 + 品种给默认，用户可在页面上改。
# 格式：`夜盘,日盘`，逗号分隔的若干段；没夜盘的就只写日盘。
S_DAY = "09:00-10:15,10:30-11:30,13:30-15:00"
S_NIGHT23 = "21:00-23:00," + S_DAY
S_NIGHT01 = "21:00-01:00," + S_DAY
S_NIGHT0230 = "21:00-02:30," + S_DAY
S_INDEX = "09:30-11:30,13:00-15:00"
S_BOND = "09:30-11:30,13:00-15:15"

# 贵金属夜盘到 02:30；有色到 01:00；黑色 / 农产品 23:00 收。
_NIGHT0230 = {"AU", "AG", "SC"}
_NIGHT01 = {"CU", "AL", "ZN", "PB", "NI", "SN", "SS", "BC", "NR", "LU", "BR"}
_NO_NIGHT = {
    "JD", "JR", "LR", "PM", "RI", "RS", "WH", "SF", "SM", "AP", "CJ",
    "PK", "UR", "SA", "PF", "SH", "PX", "PR", "IF", "IH", "IC", "IM",
    "T", "TF", "TS", "TL",
}

# 手续费。交易所口径两种：按成交额比例（rate）、按手固定（lot，元/手）。
# 这里只给一个**量级正确**的默认值，真实费率随交易所调整，用户可以改。
_FEE_BY_EXCH = {
    "SHFE": ("rate", 0.0001),
    "INE": ("rate", 0.0001),
    "DCE": ("rate", 0.0001),
    "GFEX": ("rate", 0.0001),
    "CZCE": ("lot", 3.0),          # 郑商所多数品种按手收
    "CFFEX": ("lot", 3.0),
}
_FEE_OVERRIDE = {
    # 股指按成交额万分之 0.23 收，比商品低一个数量级
    "IF": ("rate", 0.000023), "IH": ("rate", 0.000023),
    "IC": ("rate", 0.000023), "IM": ("rate", 0.000023),
    # 这几个商品按手收，用比例算会离谱
    "AU": ("lot", 10.0), "AG": ("lot", 5.0), "SC": ("lot", 20.0),
}

_DEFAULT = {
    "tick": 1.0,
    "mult": 0.0,          # 0 = 不覆盖，交给 specs.multiplier 从成交额反推
    "margin": 0.0,        # 0 = 不覆盖，交给 specs.margin_rate
    "fee_mode": "rate",   # rate：按成交额比例；lot：按手固定
    "fee_value": 0.0001,
    "session": S_DAY,
}


def session_of(symbol: str, exchange: str) -> str:
    s = (symbol or "").upper()
    ex = (exchange or "").upper()
    if ex == "CFFEX":
        return S_INDEX if s in {"IF", "IH", "IC", "IM"} else S_BOND
    if s in _NIGHT0230:
        return S_NIGHT0230
    if s in _NIGHT01:
        return S_NIGHT01
    if s in _NO_NIGHT:
        return S_DAY
    return S_NIGHT23


def fee_of(symbol: str, exchange: str) -> tuple[str, float]:
    s = (symbol or "").upper()
    if s in _FEE_OVERRIDE:
        return _FEE_OVERRIDE[s]
    return _FEE_BY_EXCH.get((exchange or "").upper(), ("rate", 0.0001))


def defaults(symbol: str, exchange: str) -> dict:
    """一个品种的默认信息。不查用户覆盖。"""
    s = (symbol or "").upper()
    mode, val = fee_of(s, exchange)
    out = dict(_DEFAULT)
    out["tick"] = TICKS.get(s, 0.0)
    out["fee_mode"] = mode
    out["fee_value"] = val
    out["session"] = session_of(s, exchange)
    out["mult"] = specs.multiplier(s, exchange) if s else 0.0
    out["margin"] = specs.margin_rate(s) if s else 0.0
    return out


# ------------------------------------------------------------------ 用户覆盖


def _path() -> str:
    return os.path.join(user_dir(), FILE)


_OVERRIDES: dict | None = None


def load_overrides() -> dict:
    """用户覆盖。进程内缓存一份 —— 回测每算一次都要问一遍，每次读盘没必要。"""
    global _OVERRIDES
    if _OVERRIDES is not None:
        return _OVERRIDES
    try:
        with open(_path(), encoding="utf-8") as fh:
            d = json.load(fh)
        _OVERRIDES = d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        _OVERRIDES = {}
    return _OVERRIDES


def save_overrides(d: dict) -> None:
    global _OVERRIDES
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _OVERRIDES = d


def info(symbol: str, exchange: str) -> dict:
    """品种信息 = 默认值 + 用户覆盖。永远返回完整字典。"""
    key = f"{symbol}_{exchange}" if exchange else symbol
    out = defaults(symbol, exchange)
    for k, v in (load_overrides().get(key) or {}).items():
        if k in out:
            try:
                out[k] = float(v) if k in ("tick", "mult", "margin", "fee_value") else v
            except (TypeError, ValueError):
                continue
    return out


def set_override(symbol: str, exchange: str, **fields) -> None:
    key = f"{symbol}_{exchange}" if exchange else symbol
    d = load_overrides()
    cur = d.get(key) or {}
    for k, v in fields.items():
        if k in _DEFAULT:
            cur[k] = v
    d[key] = cur
    save_overrides(d)


def reset_override(symbol: str, exchange: str) -> None:
    key = f"{symbol}_{exchange}" if exchange else symbol
    d = load_overrides()
    if d.pop(key, None) is not None:
        save_overrides(d)


def infer_tick(prices) -> float:
    """从成交价反推最小变动价位。

    期货报价永远落在 tick 的整数倍上，所以相邻两个不同价格的差也是 tick 的
    整数倍 —— 取一堆差值求「最大公约数」就是 tick。比维护一张全品种表可靠：
    新品种上市、交易所改 tick，这里自动跟上。

    用 Decimal 做整除：浮点求 gcd 会把 0.02 算成 0.019999999。样本不够或者
    价格全一样时返回 0（调用方退回默认值）。
    """
    try:
        vals = sorted({round(float(p), 6) for p in prices if p == p and float(p) > 0})
    except Exception:  # noqa: BLE001
        return 0.0
    if len(vals) < 3:
        return 0.0
    diffs = [b - a for a, b in zip(vals, vals[1:]) if b > a]
    if not diffs:
        return 0.0
    diffs = sorted(diffs)[:200]
    g = Decimal(str(round(diffs[0], 6)))
    for d in diffs[1:]:
        g = _gcd(g, Decimal(str(round(d, 6))))
        if g <= Decimal("0.000001"):
            return 0.0
    val = float(g)
    # 只认「像 tick 的数」：太小的说明样本噪声大（不同合约混在一起），不采信
    if not (0 < val <= 1000):
        return 0.0
    if not math.isclose(val, round(val, 4), rel_tol=1e-6):
        return 0.0
    return round(val, 4)


def _gcd(a: Decimal, b: Decimal) -> Decimal:
    while b:
        a, b = b, a % b
    return a
