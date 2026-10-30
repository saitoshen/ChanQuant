"""天勤（TqSdk）与 Tushare 两套合约命名之间的转换。

两套命名在四个地方不一致，任何一处搞错都会静默查不到数据：

| 维度 | Tushare | 天勤 |
|---|---|---|
| 品种大小写 | 全大写 `CU` | 上期/大商/能源/广期小写 `cu`，郑商/中金大写 `TA` `IF` |
| 月份位数 | 全 4 位 `2601` | 郑商所 3 位 `601`，其余 4 位 |
| 交易所后缀 | `SHF/ZCE/CFX/GFE/DCE/INE` | `SHFE/CZCE/CFFEX/GFEX/DCE/INE` |
| 合约形态 | `CU2601.SHF` | `SHFE.cu2601` |

主连（888）在天勤是 `KQ.m@SHFE.cu`，Tushare 侧 ChanQuant 用的是「品种_交易所」的文件名。

**为什么郑商所的 3 位月份必须带日期才能解析**：天勤的 `CZCE.TA609` 里 `6` 只是年份
个位，2016 年 9 月和 2026 年 9 月都叫 `TA609`。所以 `to_ts()` 强制要求传参考日期，
按「离参考日期最近的那个十年」消歧 —— 不传日期就抛错，不允许猜。

反过来 `to_tq()` 是确定的（4 位 `YYMM` 丢掉年份十位即可），不需要日期。
"""

from __future__ import annotations

import os
import re

# ChanQuant 内部交易所名 → 天勤交易所名。两边用的是同一套缩写，保留映射是为了
# 以后加交易所（比如将来可能有的雄安）时只改一处。
TQ_EXCHANGE = {
    "SHFE": "SHFE",
    "DCE": "DCE",
    "CZCE": "CZCE",
    "CFFEX": "CFFEX",
    "INE": "INE",
    "GFEX": "GFEX",
}

# 天勤品种代码大小写规则：郑商所和中金所的品种代码是大写，其余交易所小写。
# 这个规则来自天勤的合约命名约定，不是随便定的 —— `SHFE.CU2601` 是不存在的，
# 只有 `SHFE.cu2601`。
UPPER_PRODUCT_EXCHANGES = {"CZCE", "CFFEX"}

# 月份用 3 位的交易所（年份个位 + 两位月份）
SHORT_MONTH_EXCHANGES = {"CZCE"}

# 本地数据里个别品种代码和天勤不一致，逐个列出来，不做通用的「剥掉数字尾巴」——
# 剥数字会误伤以后真的带数字的品种，而这种差异本来就该是一份显式清单。
# TL0 是中金所 30 年期国债期货，Tushare 侧的本地文件叫 `TL0_CFFEX.csv`，
# 天勤叫 `KQ.m@CFFEX.TL`。
PRODUCT_ALIAS = {"TL0": "TL"}

_TS_SUFFIX = {
    "SHFE": "SHF",
    "DCE": "DCE",
    "CZCE": "ZCE",
    "CFFEX": "CFX",
    "INE": "INE",
    "GFEX": "GFE",
}
_TS_SUFFIX_REV = {v: k for k, v in _TS_SUFFIX.items()}

_CODE_RE = re.compile(r"^([A-Za-z]+)(\d{3,4})$")


def tq_product(variety: str, exchange: str) -> str:
    """Tushare 品种代码 → 天勤品种代码（只处理大小写与已知别名）。"""
    v = str(variety).strip()
    v = PRODUCT_ALIAS.get(v.upper(), v)
    return v.upper() if exchange in UPPER_PRODUCT_EXCHANGES else v.lower()


def ts_variety(tq_prod: str) -> str:
    """天勤品种代码 → Tushare 品种代码（一律大写）。"""
    return str(tq_prod).strip().upper()


def main_symbol(variety: str, exchange: str) -> str:
    """主连（888）合约：`KQ.m@SHFE.rb`。主连只能看行情，不能下单。"""
    return f"KQ.m@{TQ_EXCHANGE.get(exchange, exchange)}.{tq_product(variety, exchange)}"


def index_symbol(variety: str, exchange: str) -> str:
    """指数（加权）合约：`KQ.i@SHFE.rb`。"""
    return f"KQ.i@{TQ_EXCHANGE.get(exchange, exchange)}.{tq_product(variety, exchange)}"


def to_tq(ts_code: str) -> str:
    """Tushare 合约 → 天勤合约。`CU2601.SHF` → `SHFE.cu2601`；`TA2706.ZCE` → `CZCE.TA706`。

    只做格式转换，不校验合约是否存在。
    """
    s = str(ts_code).strip()
    if "." not in s:
        raise ValueError(f"不是合法的 ts_code（缺交易所后缀）：{ts_code}")
    code, suffix = s.rsplit(".", 1)
    exchange = _TS_SUFFIX_REV.get(suffix.upper(), suffix.upper())
    m = _CODE_RE.match(code)
    if not m:
        raise ValueError(f"无法解析合约代码：{ts_code}")
    variety, digits = m.group(1), m.group(2)
    if len(digits) == 4 and exchange in SHORT_MONTH_EXCHANGES:
        # YYMM → YMM（丢掉年份十位）：2706 → 706
        digits = digits[1:]
    elif len(digits) == 3 and exchange not in SHORT_MONTH_EXCHANGES:
        raise ValueError(f"{exchange} 的合约月份应为 4 位：{ts_code}")
    return f"{TQ_EXCHANGE.get(exchange, exchange)}.{tq_product(variety, exchange)}{digits}"


def to_ts(tq_code: str, ref_date) -> str:
    """天勤合约 → Tushare 合约。`SHFE.cu2601` → `CU2601.SHF`。

    `ref_date` 用来给郑商所的 3 位月份消歧，可以是 date / datetime / 字符串，
    只要 `pandas.Timestamp(ref_date)` 能解析。**不能省略** —— `TA609` 在 2016 和
    2026 年是同一个字符串，猜错十年不会报错，只会悄悄读到另一年的合约。
    """
    s = str(tq_code).strip()
    if "." not in s:
        raise ValueError(f"不是合法的天勤合约：{tq_code}")
    exchange, code = s.rsplit(".", 1)
    m = _CODE_RE.match(code)
    if not m:
        raise ValueError(f"无法解析天勤合约代码：{tq_code}")
    variety, digits = m.group(1), m.group(2)
    suffix = _TS_SUFFIX.get(exchange, exchange)

    if len(digits) == 3:
        if ref_date is None:
            raise ValueError(
                f"{tq_code} 的月份只有 3 位，必须给参考日期才能确定年份"
                "（TA609 既可能是 2016 年 9 月，也可能是 2026 年 9 月）"
            )
        year = _resolve_short_year(int(digits[0]), ref_date)
        month = int(digits[1:])
        digits = f"{year % 100:02d}{month:02d}"
    return f"{variety.upper()}{digits}.{suffix}"


def _resolve_short_year(year_digit: int, ref_date) -> int:
    """把「年份个位」还原成完整年份，取离参考日期最近的那个。

    期货合约挂牌一般提前 1~2 年，所以候选年份是 ref_year 附近的那两个十年。
    取绝对差最小的那个：ref=2026、digit=6 → 2026（差 0）而不是 2016（差 10）。
    """
    import pandas as pd

    ref_year = pd.Timestamp(ref_date).year
    base = ref_year // 10 * 10
    candidates = [base - 10 + year_digit, base + year_digit, base + 10 + year_digit]
    return min(candidates, key=lambda y: abs(y - ref_year))


def month_digits(tq_code: str) -> int:
    """合约的到期年月键（年*12+月），用于排序 / 判断「换月是否向前」。

    3 位的郑商所合约无法单独确定年份，这里统一按 2000 年代处理 —— 只用于
    **同一年内**的相对比较，跨十年的比较必须走 `to_ts()` 拿完整年份。
    """
    code = str(tq_code).rsplit(".", 1)[-1]
    m = _CODE_RE.search(code)
    if not m:
        return 0
    d = m.group(2)
    if len(d) == 3:
        return (2000 + int(d[0])) * 12 + int(d[1:])
    return (2000 + int(d[:2])) * 12 + int(d[2:])


def read_account(prefix: str = "") -> tuple[str, str]:
    """读天勤账户，返回 (用户名, 密码)。

    优先级：环境变量 → 项目根 `tq_account.txt` → 用户配置目录同名文件。
    文件格式是两行（用户名、密码），允许 `#` 注释和空行。

    `prefix` 用于多账户场景（比如实时行情用 `RT_` 前缀的环境变量）。
    """
    from .paths import bundle_root, user_dir

    env_user = os.environ.get(f"{prefix}TQ_USER", "").strip()
    env_pass = os.environ.get(f"{prefix}TQ_PASS", "").strip()
    if env_user and env_pass:
        return env_user, env_pass

    for path in (
        os.path.join(bundle_root(), "tq_account.txt"),
        os.path.join(user_dir(), "tq_account.txt"),
    ):
        if not os.path.isfile(path):
            continue
        lines = [
            ln.strip()
            for ln in open(path, encoding="utf-8").read().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if len(lines) >= 2:
            return lines[0], lines[1]
    raise RuntimeError(
        "找不到天勤账户。请设置环境变量 TQ_USER / TQ_PASS，"
        "或在项目根目录建 tq_account.txt（两行：用户名、密码，已在 .gitignore 里）。"
    )
