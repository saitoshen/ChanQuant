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
import re
from datetime import datetime
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
# 🔴 AO（氧化铝）/ AD（铸造铝合金）也是有色，夜盘 21:00-01:00 —— 2026-11-17
# 用近 100 天真实 5m 逐根验时段表时才发现它们被漏在这儿之外（末根 00:55）。
_NIGHT01 = {"CU", "AL", "ZN", "PB", "NI", "SN", "SS", "BC", "NR", "LU", "BR",
            "AO", "AD"}
# 🔴 SA/PF/PR/PX/SH（郑商所纯碱、短纤、瓶片、对二甲苯、烧碱）**有**夜盘
# 21:00-23:00，以前被错列在这张「无夜盘」表里 —— 后果是收盘后照样认为在交易，
# 而且白天开盘前也把 21:00 后的 bar 当成非法。同样是逐根验表查出来的。
_NO_NIGHT = {
    "JD", "JR", "LR", "PM", "RI", "RS", "WH", "SF", "SM", "AP", "CJ",
    "PK", "UR", "IF", "IH", "IC", "IM",
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
    # 这两个字段是**来源标记**，不是数值：`"user"` = 在「数据管理 → 品种信息」
    # 里手工填的；`"live"` = 点「从天勤刷新」时写进来的。空 = 吃 `specs` 的默认值。
    #
    # 🔴 为什么要记来源：这张表现在同时决定**实盘和回测**的手数，改一个数
    # 两边仓位就一起变。刷新会整表覆盖，所以被覆盖的手填值必须在界面上点名
    # （见 `symsync.apply()` 的 `overrode`）—— 不能让人第二天发现手数变了
    # 却查不出是谁改的。
    "mult_src": "",
    "margin_src": "",
}

# 只允许写"数值"的字段：`*_src` 是字符串，不能被 `float()` 强转
_NUMERIC = ("tick", "mult", "margin", "fee_value")


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
# 覆盖表的版本号。任何写盘都 +1，`session_ranges` 用它判定缓存是否还新鲜。
# 为什么不用 `_OVERRIDES` 的 id / 内容比较：行情链路每秒问好几次，比较整张表
# 比直接解析还贵；整数比对是常数时间。
_VER: int = 0


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
    global _OVERRIDES, _VER
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _OVERRIDES = d
    # 用户改过「交易时段」⇒ 下面那些解析缓存必须整片作废。用版本号而不是
    # `clear()`：`session_ranges` 拿版本号比对，命中就是一次整数比较。
    _VER += 1


def info(symbol: str, exchange: str) -> dict:
    """品种信息 = 默认值 + 用户覆盖。永远返回完整字典。"""
    key = f"{symbol}_{exchange}" if exchange else symbol
    out = defaults(symbol, exchange)
    for k, v in (load_overrides().get(key) or {}).items():
        if k in out:
            try:
                out[k] = float(v) if k in _NUMERIC else v
            except (TypeError, ValueError):
                continue
    return out


# 常见的三字母缩写 → 天勤全称。交易所自己发的行情里两种写法都有（`rb2601.SHF`
# 这种是外部数据源的写法），不折一下下游 `specs.multiplier(sym, ex)` 会落空，
# 手续费跟着退化成「算不出来」。
_EX_ALIAS = {"SHF": "SHFE", "CZC": "CZCE", "CFX": "CFFEX",
             "GFX": "GFEX", "ZCE": "CZCE"}


def contract_product(contract: str) -> tuple[str, str]:
    """`SHFE.rb2701` → `("RB", "SHFE")`；`rb2601.SHF` / 裸 `rb2601` 也认。

    只给**显示**用（查 tick 决定价格显示几位小数），认不出来返回 `("", "")`，
    调用方自己退回「2 位小数」。不要拿它做合约校验 —— 真正的归一在
    `ttrade.tqstore` / `broker_api` 那两处。

    交易所那半段会折成天勤写法：`rb2601.SHF` 给出 `("RB", "SHFE")` 而不是
    `("RB", "SHF")` —— 下游 `specs.multiplier(sym, ex)` 只认全称，`SHF` 进去
    查不到乘数，手续费会跟着退化成「算不出来」。
    """
    s = str(contract or "").strip()
    if not s:
        return "", ""
    ex = ""
    if "." in s:
        head, _, tail = s.partition(".")
        if head.isalpha() and len(head) <= 6 and any(c.isdigit() for c in tail):
            ex, s = head.upper(), tail            # SHFE.rb2701
        elif tail.isalpha():
            ex, s = tail.upper(), head            # rb2601.SHF
    return ("".join(c for c in s if c.isalpha()).upper(),
            _EX_ALIAS.get(ex, ex))


def tick_of(symbol: str, exchange: str = "") -> float:
    """最小变动价位。品种代码或 `交易所.合约` 都认；查不到返回 0。

    0 表示「不知道」—— 显示层要据此退回 2 位小数，而不是当成「一跳 0 元」。
    """
    sym, ex = str(symbol or ""), str(exchange or "")
    if not ex and "." in sym:
        sym, ex = contract_product(sym)
    try:
        return float(info(sym, ex).get("tick") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def tick_decimals(tick: float) -> int:
    """最小变动价位要几位小数：10 → 0、0.5 → 1、0.2 → 1、0.01 → 2、0.005 → 3。

    用「反复 `round` 看能不能对上」而不是 `log10`：0.2 / 0.05 这种不是 10 的
    整数次幂，`log10` 算出来会少一位或负一位，价格就显示成 `8240.0` 或者把
    0.05 抹成 0.1。`tick <= 0`（不知道）按 2 位，和以前的硬编码保持一致。
    """
    if not (tick > 0 and tick == tick):
        return 2
    for d in range(0, 5):
        if abs(round(tick, d) - tick) < 1e-9:
            return d
    return 4


def price_text(v, symbol: str = "", exchange: str = "", *, tick: float = 0.0,
               signed: bool = False, dash: str = "—") -> str:
    """按**最小变动价位**把价格显示成该有的位数，带千分位。

    为什么显示层要收位数：铜 `90,040.00`、国债 `118.150`、棉花 `5,902.00`
    —— 全是 `:.2f` 硬编码出来的「假精度」，看着像行情在小数位上来回跳。
    位数本来就该由 tick 决定（用户 2026-11-20 提的）。

    🔴 **不许为了好看抹掉真实价位**：值在 tick 的位数下表示不出来（比如均价
    5942.8 配一个跳 1 的品种）就多留几位，最多 4 位。宁可多显示，也不能把
    真实的 0.5 舍成整数 —— 那是把用户的钱显示错了。

    `tick` 传 0 时按 `symbol` / `exchange` 查；两者都查不到就 2 位。
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    if f != f:
        return dash
    t = float(tick or 0) or tick_of(symbol, exchange)
    d = tick_decimals(t)
    while d < 4 and abs(f - round(f, d)) > 1e-9:
        d += 1
    sign = "+" if (signed and f > 0) else ""
    return f"{sign}{f:,.{d}f}"


def set_override(symbol: str, exchange: str, manual: bool = True,
                 **fields) -> None:
    """写覆盖。`manual=True`（默认）表示**用户在界面上填的**，会给相应字段
    打上 `*_src="user"` 标记；`manual=False` 是「从天勤刷新」写进来的，
    标记成 `"live"`。

    `src` 只用来在界面上区分「这个数是人手填的」还是「从天勤刷下来的」。
    **它不阻止任何写入**：刷新是用户主动点的，就该按真实值覆盖；实盘运行
    时也不会偷偷改这张表。

    🔴 这张表是**唯一**的手数来源（实盘和回测共用，见 `strategy_runner._terms`），
    所以「谁在什么时候改了它」必须可查 —— 实盘跑着跑着表被后台改了，
    用户就没法解释「昨天 4 手今天 1 手」。
    """
    key = f"{symbol}_{exchange}" if exchange else symbol
    d = load_overrides()
    cur = d.get(key) or {}
    src = "user" if manual else "live"
    for k, v in fields.items():
        if k in _DEFAULT:
            cur[k] = v
            if k in ("mult", "margin"):
                cur[f"{k}_src"] = src
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


# ---------------------------------------------------------------- 交易时段门禁
#
# 「这个品种现在开不开市」是**一条判据**：实盘行情（天勤 / CTP / 融航）和策略
# 引擎都问它，不许各处自己写一遍 `9 <= hour <= 15`（#5 同一语义只许一个判据）。
# 收盘之后还在更新行情、还在下单，就是这条判据缺失的直接后果。

_SESSION_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")
_PARSE_CACHE: dict = {}       # 时段字符串 → ((起, 止), …)，键就是字符串本身
_RANGE_CACHE: dict = {}       # (品种, 交易所) → (版本号, ((起, 止), …))
# 这张时段表只对**期货**有意义（默认值是按六个期货交易所给的）
_FUTURES_EX = {"SHFE", "DCE", "CZCE", "CFFEX", "INE", "GFEX"}


def parse_session(raw: str) -> tuple:
    """`"21:00-23:00,09:00-10:15"` → `((1260, 1380), (540, 615))`（0 点起分钟数）。

    **止 < 起 = 跨零点**（21:00-01:00 / 21:00-02:30）。

    解析不出来就跳过那一段，**不抛异常**：这是用户可编辑的文本框，写错了也只该
    退化成「少挡一段」，不该让行情线程崩掉。
    """
    raw = str(raw or "")
    hit = _PARSE_CACHE.get(raw)
    if hit is not None:
        return hit
    out = []
    for part in raw.split(","):
        m = _SESSION_RE.match(part)
        if not m:
            continue
        h1, m1, h2, m2 = (int(g) for g in m.groups())
        a, b = h1 * 60 + m1, h2 * 60 + m2
        if not (0 <= a < 1440 and 0 <= b <= 1440) or a == b:
            continue
        out.append((a, b))
    _PARSE_CACHE[raw] = tuple(out)
    return _PARSE_CACHE[raw]


def session_ranges(symbol: str, exchange: str) -> tuple:
    """品种 → 解析好的时段表。取**用户改过的**那张（`info()["session"]`）。

    只有**期货**走这张表：A 股 / 美股 / 交易所为空的一律返回空元组 =
    「不限制」。理由很直接 —— `session_of()` 那套默认表是按期货交易所给的，
    拿它去管 600519 会得出「21:00 有夜盘」这种结论，比不限制更糟。

    按版本号缓存：行情链路每秒问好几次，而 `info()` 里要算乘数和保证金率，
    不缓存撑不住。
    """
    key = ((symbol or "").upper(), (exchange or "").upper())
    if key[1] not in _FUTURES_EX:
        return ()
    # 具体合约（rb2601 / IM2612）退到品种：交易时段是按**品种**配的
    if any(ch.isdigit() for ch in key[0]):
        key = ("".join(ch for ch in key[0] if ch.isalpha()) or key[0], key[1])
    hit = _RANGE_CACHE.get(key)
    if hit is not None and hit[0] == _VER:
        return hit[1]
    segs = parse_session(info(key[0], key[1]).get("session"))
    if not segs:
        segs = parse_session(session_of(key[0], key[1]))
    _RANGE_CACHE[key] = (_VER, segs)
    return segs


def in_session(symbol: str, exchange: str, ts=None) -> bool:
    """此刻（或 `ts` 那一刻）这个品种是否在交易时段内。

    `ts` 缺省 = 现在；可以是 `datetime` / `pd.Timestamp`（datetime 子类）/
    `"2026-11-17 14:35[:00]"`。

    ## 星期规则（夜盘按**起算那天**归）

    - **日盘段**只在周一~周五成立；
    - **夜盘段 21:00 之后那截**在周日~周五成立 —— 周五夜盘真实存在（一直跑到
      周六凌晨），周日夜盘属于下周一，**周六 21:00 没有盘**；
    - **跨零点之后那截**（00:00-02:30）跟着「前一天」走：周六凌晨有（周五夜盘
      的延续），周日凌晨没有。

    🔴 **不判法定节假日**：这张表只有时段、没有日历。节假日里的工作日会被判成
    「开着」—— 但那时交易所本来就不推行情，挡不挡结果一样；真按日历挡得接一份
    交易日历，是另一件事。这一版只保证**收盘之后一定判成关**，那是这次要修的。

    🔴 判据用 **`a <= m < b`**（bar 的时刻是**起点**）：11:25 那根 5m 覆盖
    11:25~11:30，判开；11:30 没有那根 bar，收盘时刻本身判关。
    """
    segs = session_ranges(symbol, exchange)
    if not segs:
        return True                  # 解析不出时段 ⇒ 不限制（宁可漏挡，别误挡）
    if ts is None:
        ts = datetime.now()
    elif isinstance(ts, str):
        # 认几种常见写法就算（日志、自检、状态栏都会传字符串）
        raw = ts.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
            try:
                ts = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        else:
            ts = datetime.now()      # 认不出就当现在，别让门禁把行情整条挡掉
    m = ts.hour * 60 + ts.minute
    wd = ts.weekday()                # 周一=0 … 周日=6
    for a, b in segs:
        night = a >= 1200            # 20:00 起算夜盘段（这套表里夜盘都是 21:00 起）
        if a < b:                    # 不跨零点
            if a <= m < b and ((wd != 5) if night else (wd <= 4)):
                return True
        else:                        # 跨零点：前半截看今天，后半截看昨天
            if m >= a and wd != 5:
                return True
            if m < b and (wd - 1) % 7 != 5:
                return True
    return False
