"""天勤实时行情与交易。

一个 `TqApi` 同时承担三件事：实时行情、账户、下单。它们共享同一条连接 ——
分开建会重复登录、重复订阅，而且账户对象必须和下单在同一个 api 上才一致。

**线程模型**：`wait_update()` 会阻塞，必须放在独立线程里跑。所有 `get_*` 拿到的
都是「活引用」，随 `wait_update()` 就地更新；不要在循环里重复 `get_*`（每次
调用都会新建订阅对象，天勤会按符号去重但语义上仍然是浪费）。所以这里的做法是
`start()` 里一次性建好引用，`wait()` 推进，`snapshot()` 拷出普通 dict 给界面。

🔴 **2026-11-24 起，推进 `wait_update` 的那条线程在 `LiveHub` 自己身上**
（`_pump`），不再是调用方。因为一个账户的会话现在由交易页和组合页**共用**
（`ttrade/hubreg.py`）：两个调用方各推一次 = 两条线程驱动同一个 `TqApi`，
tqsdk 明确不支持，会把内部订阅状态写坏。所以：

- `wait()` 只等「泵推来下一拍」，不再自己调 `wait_update()`；
- 所有碰 `TqApi` 的方法都进 `_lock`（`@_serial` 装饰器 / `_api_call()`）；
- 组合页要的 `TqApi` 同名方法（`get_position` / `get_order` / `insert_order` /
  `cancel_order`）在文件末尾原样转出去，所以 `PortfolioRunner` 一行都不用改。

**时间戳**：K 线 `datetime` 是 **UTC 纳秒**，而 `DataDownloader` 写出的 CSV 是
北京时间，两者差 8 小时且不报错。统一走 `tqfetch.local_time()` 转北京时间，
与 DuckDB 里已落盘的历史对齐。

**复权**：实时段的 `adj_factor` 一律 1.0。原因是调用方先把本地补到「上一根完整
bar」，实时段全部落在当前主力合约的时间区间内，而后复权因子的最新段恒为 1
（见 `tqfetch.adj_factor_series`）。如果实时期间天勤换了主力，`underlying` 会变，
这时应该重新走一遍补历史 —— `underlying_changed` 会告诉界面。
"""

from __future__ import annotations

import functools
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field

import pandas as pd

from . import daylimit, hubreg, symbols_info
from .paths import user_dir
from .tqdata import PERIODS, daily_csv_path, resample, trading_days
from .tqfetch import PERIOD_SEC, local_time
from .tqsdk_symbols import contract_from_input, contract_key, is_stock, trade_symbol

# 实时 K 线向天勤要多少根。10000 是单次上限；5m 的 10000 根约 208 个交易日，
# 足够覆盖「界面开着放了一周」这种场景，再长的缺口应该走 update_main 补。
KLINE_LEN = 8000

ACCOUNT_KINDS = [
    ("kq", "快期模拟（天勤账户自带）"),
    ("sim", "本地模拟（TqSim，仅本机）"),
    ("account", "实盘（期货公司资金账号）"),
    ("kq_stock", "快期股票模拟（需专业版）"),
    ("sim_stock", "本地股票模拟（TqSimStock）"),
    ("ctp", "CTP 柜台（SimNow / 期货公司）"),
    ("rohon", "融航资管中台（CTP 兼容接口）"),
]

# 股票账户的两种。期货和股票在天勤里是**两套账户类**，不能混用一个连接：
# 期货账户下单 `SHFE.rb2601`，股票账户下单 `SSE.600000`，撮合规则也不同
# （T+1、涨跌停、100 股一手）。
STOCK_KINDS = ("kq_stock", "sim_stock")

# 柜台后端。**这是「走哪个连接」的唯一判据** —— 以前到处写 `is_ctp`，加了融航
# 之后就变成「非 ctp 一律当天勤」，融航账户会被丢进 `LiveHub` 报「缺少天勤账户」。
# 三个后端的连接对象、登录参数、回报口径都不一样，各自一条连接：
#   tq    天勤 TqApi（多种账户共用一条连接，多账户走 TqMultiAccount）
#   ctp   期货公司 CTP 柜台（openctp-ctp）
#   rohon 融航资管中台（CTP 兼容 dll，独立于系统那份 openctp-ctp）
BACKENDS = ("tq", "ctp", "rohon")
BACKEND_LABELS = {"tq": "天勤", "ctp": "CTP", "rohon": "融航"}


# ------------------------------------------------------------------ 账户配置


@dataclass
class AccountSpec:
    """一个交易账户的登录信息。

    - `kq`：快期模拟，用天勤账户登录即可交易，账户数据与快期客户端一致；
    - `sim`：`TqSim`，纯本地撮合，重启即清空，适合试接口；
    - `account`：实盘，需要期货公司的 broker_id + 资金账号 + 密码；
    - `kq_stock` / `sim_stock`：股票模拟，见 `STOCK_KINDS`；
    - `ctp`：期货公司 CTP 柜台，**不走天勤**。参数是前置地址 + BrokerID +
      投资者账号，见 `md_front` / `td_front` / `app_id` / `auth_code`；
    - `rohon`：融航资管中台，同样不走天勤。字段和 `ctp` **一一对应**（融航
      本身就是 CTP 协议的中台）：`user`/`password` 是融航用户名密码，
      `broker_id` 是经纪商代码，`td_front`/`md_front` 是交易/行情服务器，
      `app_id` 是产品名称，`auth_code` 是授权编码。不另开一套字段：交易页只有
      一张账户表，用户换个柜台不该去别的地方再填一遍。

    `user` / `password` 是天勤账户（TqAuth），前五种模式都要；`ctp` / `rohon`
    模式下它们就是柜台账号与密码。
    """

    name: str = "默认"
    kind: str = "kq"
    user: str = ""
    password: str = ""
    broker_id: str = ""
    account_id: str = ""
    account_password: str = ""
    # CTP / 融航专用。留在同一个结构里而不是另开一张表：交易页只有一个账户列表，
    # 用户不该为了换个柜台去另一个地方再配一遍
    ctp_preset: str = "simnow_7x24"
    md_front: str = ""
    td_front: str = ""
    app_id: str = ""
    auth_code: str = ""

    def build(self):
        """构造天勤的账户对象。实盘缺参数直接报错，不要静默退化成模拟。"""
        if self.backend != "tq":
            # 柜台账户没有天勤账户对象可构造。在这里挡掉而不是让它掉到下面
            # `TqKq()` 的兜底分支 —— 那会静默连上一个快期模拟户，用户看着
            # 「登录成功」，实际下的单全在模拟盘里
            raise ValueError(
                f"账户「{self.name}」是{BACKEND_LABELS[self.backend]}账户，"
                "不走天勤，不能构造天勤账户对象"
            )
        from tqsdk import TqAccount, TqKq, TqKqStock, TqSim, TqSimStock

        if self.kind == "account":
            missing = [
                k for k in ("broker_id", "account_id", "account_password")
                if not getattr(self, k)
            ]
            if missing:
                raise ValueError(f"实盘账户缺少参数：{', '.join(missing)}")
            return TqAccount(self.broker_id, self.account_id, self.account_password)
        if self.kind == "sim":
            return TqSim()
        if self.kind == "sim_stock":
            # 本地股票模拟：100 股一手、T+1、涨跌停都按交易所规则撮合，
            # 和 TqSim 的期货口径完全不同 —— 拿 TqSim 跑股票会算出负股数
            return TqSimStock()
        if self.kind == "kq_stock":
            return TqKqStock()
        return TqKq()

    @property
    def is_stock(self) -> bool:
        return self.kind in STOCK_KINDS

    @property
    def backend(self) -> str:
        """这条账户走哪个连接：`tq` / `ctp` / `rohon`。

        全项目**只在这里判一次**。散在各处写 `kind == "ctp"` 的后果是：加了
        融航之后，「不是 CTP 就当它走天勤」把融航账户塞进 `LiveHub`，报出来
        的是「缺少天勤账户或密码」，看着像用户填错了账号。
        """
        return self.kind if self.kind in ("ctp", "rohon") else "tq"

    @property
    def is_ctp(self) -> bool:
        """CTP 账户不走天勤，`LiveHub` 一构建就会报「缺天勤账户」。"""
        return self.backend == "ctp"

    @property
    def is_rohon(self) -> bool:
        return self.backend == "rohon"

    def label(self) -> str:
        kind = dict(ACCOUNT_KINDS).get(self.kind, self.kind)
        return f"{self.name} · {kind}"


def accounts_path() -> str:
    return os.path.join(user_dir(), "accounts.json")


def load_accounts() -> list[AccountSpec]:
    """读本地账户列表。文件不存在或损坏时返回空列表，不抛。"""
    path = accounts_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return []
    out = []
    for it in raw if isinstance(raw, list) else []:
        try:
            out.append(AccountSpec(**{k: v for k, v in it.items()
                                      if k in AccountSpec.__dataclass_fields__}))
        except TypeError:
            continue
    return out


def save_accounts(specs: list[AccountSpec]) -> None:
    """写本地账户列表。

    密码是明文落在用户目录（`%LOCALAPPDATA%\\ChanQuant\\accounts.json`）里的 ——
    和项目里 `tq_account.txt` 的现状一致。这是本机自用工具的取舍：要加密就得管
    一个主密码，用起来反而更麻烦。别把这个文件提交到仓库。
    """
    path = accounts_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(s) for s in specs], fh, ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ 纯函数


def bars_from_serial(serial, period: str = "") -> pd.DataFrame:
    """把 `get_kline_serial` 返回的活引用转成标准行情表（index = 北京时间）。

    启动时天勤会先用占位行把序列填满（`datetime=0`、价格 NaN），不剔掉的话
    图上会出现一根塌到 0 的 K 线，指标全被带偏。
    """
    if serial is None or not len(serial):
        return pd.DataFrame()
    d = serial
    dt = local_time(d["datetime"].to_numpy())
    out = pd.DataFrame(
        {
            "open": d["open"].to_numpy(dtype=float),
            "high": d["high"].to_numpy(dtype=float),
            "low": d["low"].to_numpy(dtype=float),
            "close": d["close"].to_numpy(dtype=float),
            "volume": d["volume"].to_numpy(dtype=float),
            "open_oi": d["open_oi"].to_numpy(dtype=float),
            "close_oi": d["close_oi"].to_numpy(dtype=float),
        },
        index=dt,
    )
    out.index.name = "datetime"
    out = out[out.index > pd.Timestamp("1971-01-01")]
    out = out.dropna(subset=["close"])
    return out[~out.index.duplicated(keep="last")]


#: 复权要跟着价格一起缩放的列。成交量 / 持仓量不是价格，不参与缩放。
_PRICE_COLS = ("open", "high", "low", "close")


def merge_live(local: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """把实时 K 线接到本地历史后面。

    只接**严格晚于**本地最后一根的实时 bar，因子用本地那份（本地已经带了正确的
    复权因子，实时段一律 1.0，混着用会在拼接处产生一次假的缩放）。

    **重叠段用实时的 OHLC 覆盖**（乘上本地那根的 `adj_factor`）。两个理由：

    1. 本地最后一根可能是**下载 / 补库时写的半根快照**，价格停在那一刻，而实时
       这根一直在动。不覆盖的话图上最后一根永远不动 —— 日线最明显，库里已经有
       当天那根，实时这根被整根丢掉，一天里价格怎么走图都不变。
    2. 更糟的是**库尾半根不在最后一行**：库里停在 10:30（半根），实时段从 10:35
       往后接，那根半根就卡在中间、`iloc[:-1]` 切不到它，指标（ATR、60 根动量、
       突破阈值）全都按半根的 close 算。2026-11-19 库里实测到这种 bar：cu 11-03
       21:55 的 close 比天勤差 70 元、成交量只有一半。**只覆盖最后一根是不够的**，
       所以按整段重叠覆盖。

    实时价是**原始价**、本地是「原始价 × 复权因子」，所以覆盖时必须乘回本地的
    因子，否则接缝处会跳一下。
    """
    if live is None or live.empty:
        return local if local is not None else pd.DataFrame()
    if local is None or local.empty:
        add = live.copy()
        add["adj_factor"] = 1.0
        add["trading_day"] = trading_days(add.index).to_numpy()
        return add

    # 复制一份再改：`local` 是 DataStore 缓存里那张表，就地改会污染下一次回测
    out = local.copy()
    # 用 searchsorted 定位重叠段，别对 26 万根的索引做 intersection
    lo = int(local.index.searchsorted(live.index[0]))
    hi = int(local.index.searchsorted(live.index[-1], side="right"))
    if hi > lo:
        ov = local.index[lo:hi].intersection(live.index)
        if len(ov):
            if "adj_factor" in out.columns:
                f = out.loc[ov, "adj_factor"].astype(float).to_numpy()
            else:
                f = 1.0
            for c in _PRICE_COLS:
                if c in out.columns and c in live.columns:
                    out.loc[ov, c] = live.loc[ov, c].to_numpy() * f
            for c in ("volume", "open_oi", "close_oi"):
                if c in out.columns and c in live.columns:
                    out.loc[ov, c] = live.loc[ov, c].to_numpy()

    add = live[live.index > local.index[-1]].copy()
    if add.empty:
        return out
    add["adj_factor"] = 1.0
    add["trading_day"] = trading_days(add.index).to_numpy()
    for c in ("open_oi", "close_oi"):
        if c not in add.columns:
            add[c] = 0.0
    # 列要取**本地那份的全部**，不能取交集。本地表里带着展期 / 期权那几十列
    # （`c1_close` / `iv_near` / `pcr` …），实时推送里没有这些 —— 取交集等于把
    # 整列从合并结果里删掉，`factors.available()` 一看「这因子一根有值的都没有」
    # 就把它踢出合成。用户看到的是「一开实时，19 个因子变成 12 个，信号全变了」，
    # 而且同一根 bar 内只刷价格不重算，标签在两种状态间反复横跳，像是策略自己
    # 在变。新 bar 缺的列留 NaN：历史段本来就有值，因子仍然算可用，只是最后一根
    # 的分数是 NaN（`mean(skipna=True)` 会跳过，不影响已走完的 bar）。
    add = add.reindex(columns=local.columns)
    merged = pd.concat([out, add]).sort_index()
    # 🔴 索引精度必须跟**库内那份**一致，不能跟着实时段走。
    # DuckDB 读出来是 `datetime64[us]`，天勤推上来是 `datetime64[ns]`；
    # `pd.concat` 取更精的那个 ⇒ 整帧变成 ns。而基本面 / 宏观因子要把外部
    # 序列（us）`merge_asof` 上来，两侧精度不一致会**直接抛 MergeError**，
    # 因子被静默剔除、少一票、信号翻向。
    # 2026-11-20 白银 / 铅 1d_tq「研究页空仓、实盘 −2 手」的另一半就是它
    # （前半是 attrs，见下面那段注释）。因子层也有兜底（`tsalign.merge_asof_ns`），
    # 两处都留：这里保证「实盘帧 = 库内帧」同一种类型，那边保证任何来源都不炸。
    try:
        if merged.index.dtype != local.index.dtype:
            merged.index = merged.index.astype(local.index.dtype)
    except (AttributeError, TypeError):
        pass
    # 🔴 `pd.concat` 会丢掉 `df.attrs`，而 `factors_ext._symbol()` 就是靠
    # `attrs["symbol"]` 认品种的。丢了以后 `_fund()` 整段返回 NaN ⇒
    # `war_chg_5` / `war_val` / `inv_*` / `profit_*` / `spot_basis` 全列没值 ⇒
    # `factor_matrix` → `filter_available(min_obs=60)` 把它们**静默剔除** ⇒
    # `mean(skipna=True)` 少一票，分数符号可能直接翻转。
    # 实测 AG 2026-11-20：带 attrs score=+0.4582 sig=0；丢 attrs
    # score=-0.5568 sig=-1（`war_chg_5` 有效值 2438 → 0）。
    merged.attrs = dict(local.attrs)
    return merged


# ------------------------------------------------------------------ 会话


# 委托 / 成交表最多显示多少条。一天下来「被拒的废单」能有几百上千条
# （拒单循环那种情况），全塞进 QTableWidget 之后，切到账户页第一帧要摆
# 几万个单元格，界面直接卡死。只留最近的，用户要看细节有委托号可查。
SNAP_ROWS = 200


_MULT: dict[str, float] = {}
_TIME: dict[int, str] = {}


def _stamp(ns) -> str:
    """天勤的纳秒时间戳 → 北京时间字符串。0 / 非法值返回空串。

    委托表和成交表都要这一列，而快照每 0.4 秒重建一次、一次最多 200 行 ×
    账户数 —— 逐个走 `pd.Timestamp` 太贵，按纳秒值缓存（同一批委托的纳秒
    永远不变，缓存命中率接近 100%）。
    """
    try:
        n = int(ns)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    hit = _TIME.get(n)
    if hit is not None:
        return hit
    try:
        t = pd.Timestamp(n, unit="ns", tz="UTC").tz_convert("Asia/Shanghai")
        s = t.strftime("%m-%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        s = ""
    if len(_TIME) > 4000:      # 一天的委托也就这个量级，超了直接清
        _TIME.clear()
    _TIME[n] = s
    return s


def _mult_of(contract: str) -> float:
    """合约乘数，按合约代码推品种再查本地日线。查不到返回 1.0。

    账户列表要算杠杆和成交金额，都是「手 → 钱」，缺了乘数就没法算。这里不
    订阅行情：账户页每 0.4 秒刷一次，为几十个合约建 quote 会把连接压垮，
    而乘数是静态的，读一次本地 CSV 就够了（`ttrade.specs` 内部也做了缓存）。
    """
    if not contract:
        return 1.0
    if contract in _MULT:
        return _MULT[contract]
    import re

    head, _, tail = contract.partition(".")
    ex, code = (head, tail) if tail else ("", head)
    prod = re.sub(r"\d+$", "", code).upper()
    try:
        from . import specs as _specs

        val = float(_specs.multiplier(prod, ex))
    except Exception:  # noqa: BLE001
        val = 1.0
    _MULT[contract] = val
    return val


#: 本地日线最后一根收盘价的缓存：{csv 路径: (mtime, close)}
_LAST_CLOSE: dict[str, tuple[float, float]] = {}


def _last_daily_close(symbol: str, exchange: str = "") -> float:
    """本地日线**最后一根**的收盘价，给平仓盈亏当兜底成本基准。

    用武之地只有一个（2026-11-23 用户：「平仓盈亏可以根据昨日结算价或者收盘
    价格来计算，如果获取不到历史成交的话」）：成交流拿不到、行情也拿不到时，
    总不能给一格「—」。

    从文件**尾部**倒着取最后一行 —— 这些 CSV 是十几年的日线（rb 有 4000+ 行），
    在 0.4 秒一刷的账户循环里整份 `read_csv` 会把界面拖死；按 mtime 缓存，
    一天只读一次。

    🔴 这是**品种指数**（`RB.SHF`）的收盘，不是具体月份合约的 —— 所以排在
    昨结算 / 昨收 / 今收**之后**，只在前三级都拿不到时才用。量级对得上，
    别当精确值。读不到返回 0。
    """
    vari, ex = symbols_info.contract_product(str(symbol or ""))
    if not vari:
        return 0.0
    ex = ex or str(exchange or "").upper()
    try:
        path = daily_csv_path(vari.lower(), ex)
        mtime = os.path.getmtime(path)
    except Exception:  # noqa: BLE001 - 没下过这个品种的日线，就当没有
        return 0.0
    hit = _LAST_CLOSE.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    px = 0.0
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            chunk = f.read().decode("utf-8", "ignore")
        # 表头有 BOM，但最后一行没有；列序 `ts_code,name,date,open,high,low,close,...`
        for line in reversed([x for x in chunk.splitlines() if x.strip()]):
            cols = line.split(",")
            if len(cols) >= 7:
                px = float(cols[6] or 0.0)
                break
    except Exception:  # noqa: BLE001
        px = 0.0
    _LAST_CLOSE[path] = (mtime, px)
    return px if px > 0 else 0.0


#: 泵线程一次 `wait_update` 的上限（秒）。
#:
#: 它同时是「别的线程调 `TqApi` 最多被挡多久」：`wait_update` 期间会话锁是
#: 被占着的（`TqApi` 不是线程安全的），所以这个数不能大。0.1 秒的代价是空闲
#: 时每秒醒 10 次 —— 有消息时 `wait_update` 会立刻返回，不受这个数影响。
PUMP_SEC = 0.1


def _serial(fn):
    """把这个方法的**整段**包进会话锁（`LiveHub._lock`）。

    🔴 为什么需要（2026-11-24 用户要求「天勤账户也共享一条会话」）：一条
    `TqApi` 被泵线程 `wait_update` 驱动着，其它线程再去碰 `get_position()` /
    `insert_order()` / `get_kline_serial()` 就是**跨线程用同一个 TqApi**，
    tqsdk 明确不支持（会写坏内部订阅状态，表现为「单子偶尔丢、偶尔重复」）。
    所以读写一律串行化，泵也只在这个锁里推 `wait_update`。

    用装饰器而不是在每个函数体里缩进一层：这几个方法里全是长注释和 early
    return，缩进一改 diff 就没法看了。
    """

    @functools.wraps(fn)
    def inner(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    return inner



def _qnum(q, name: str, default: float = 0.0) -> float:
    """行情对象上的数值字段，**NaN 一律当 0**。

    天勤新订阅的第一帧、停牌、收盘后都是 `NaN`；`float('nan')` 不抛异常而且是
    truthy —— 不归零就会一路漏到价格框里，0 价的限价单会被柜台按「价格不合法」
    拒掉。所有读行情数值的地方都走这一处，别各写一遍。
    """
    try:
        v = float(getattr(q, name, default))
    except (TypeError, ValueError):
        return default
    return default if v != v else v


def _quote_row(q, fallback_symbol: str = "") -> dict:
    """把一个行情对象摊成一行普通 dict（「报价」框一行 / 盘口一条）。"""
    return {
        "symbol": str(getattr(q, "symbol", "") or fallback_symbol),
        "instrument": str(getattr(q, "instrument_id", "") or ""),
        "last": _qnum(q, "last_price"),
        "bid": _qnum(q, "bid_price1"),
        "ask": _qnum(q, "ask_price1"),
        "bid_vol": _qnum(q, "bid_volume1"),
        "ask_vol": _qnum(q, "ask_volume1"),
        # 涨跌幅要拿昨结算比；pre_close 在天勤里就是昨结算，两个都取一下兜底
        "pre_close": _qnum(q, "pre_close") or _qnum(q, "pre_settlement"),
        "open": _qnum(q, "open"),
        "high": _qnum(q, "highest"),
        "low": _qnum(q, "lowest"),
        "volume": _qnum(q, "volume"),
        "tick": _qnum(q, "price_tick"),
        "datetime": str(getattr(q, "datetime", "") or ""),
        "source": "live",
    }


def _sum_accounts(rows: list[dict]) -> dict:
    """把多账户的资金合成一份，给「全选」时的顶部资金卡用。

    风险度不能用 sum：它是比值，得拿合计保证金除以合计权益重算。
    """
    out = {}
    for k in ("balance", "available", "margin", "float_profit", "close_profit",
              "commission", "turnover", "day_profit"):
        out[k] = float(sum(float(r.get(k) or 0.0) for r in rows))
    bal = out["balance"]
    out["risk_ratio"] = (out["margin"] / bal) if bal > 0 else 0.0
    out["accounts"] = len(rows)
    return out


# --------------------------------------------------- 成交的「平仓盈亏」估算
#
# 天勤的 `Trade` 和 CTP 的成交回报上**都没有任何盈亏字段**（只有价、量、
# 方向、开平），所以「平仓盈亏」这一列只能自己倒推。口径是
# `(成交价 − 该方向开仓均价) × 手数 × 乘数 × 方向`。两个柜台共用这一份实现，
# 不允许各写一遍（同一个语义在两处各写一遍是这个项目的第一类 bug）。

# 开仓均价的保鲜期。持仓被平光后 `Position` 整个消失、均价就没了，所以缓存
# 上一轮；但隔夜后价格量级全变，拿昨天的均价算今天的平仓会凭空造出盈亏。
_OPEN_AVG_TTL = 6 * 3600.0


def _is_close_offset(offset) -> bool:
    """这一笔是不是平仓单。

    offset 空着当开仓：柜台漏推 offset 时，宁可算作开仓，也不要给一笔来路
    不明的「平仓盈亏」——那个数会被当成真钱。
    """
    s = str(offset or "").upper()
    return ("CLOSE" in s) or s in ("1", "3")


def update_open_avg(cache: dict, positions, now: float = 0.0) -> dict:
    """把一轮持仓快照里「有仓那一侧」的开仓均价记进缓存。

    `cache` 形状 `{合约: [多头均价, 空头均价, 时间戳]}`。均价 <= 0 的那一侧
    **不覆盖**旧值 —— 算不出来时留着上一轮的，别把能用的抹成 0。
    """
    ts = now or time.time()
    for p in positions or ():
        sym = str(p.get("symbol") or "")
        if not sym:
            continue
        cur = cache.get(sym) or [0.0, 0.0, 0.0]
        lg = float(p.get("open_price_long") or 0.0)
        sh = float(p.get("open_price_short") or 0.0)
        if lg > 0:
            cur[0] = lg
        if sh > 0:
            cur[1] = sh
        cur[2] = ts
        cache[sym] = cur
    # 过期的留着只会误导，顺手清掉（一天的合约数不过百，清起来不贵）
    for sym in [k for k, v in cache.items() if ts - float(v[2] or 0.0) > _OPEN_AVG_TTL * 2]:
        cache.pop(sym, None)
    return cache


def open_avg_of(cache: dict, symbol: str, now: float = 0.0,
                ttl: float = _OPEN_AVG_TTL) -> tuple[float, float]:
    """取某合约的 (多头均价, 空头均价)，过期的当没有。"""
    row = cache.get(symbol) if cache else None
    if not row:
        return (0.0, 0.0)
    ts = now or time.time()
    if ts - float(row[2] or 0.0) > ttl:
        return (0.0, 0.0)
    return (float(row[0] or 0.0), float(row[1] or 0.0))


def est_close_pnl(trades, cache: dict, mult_of=None, now: float = 0.0,
                  ttl: float = _OPEN_AVG_TTL) -> list[dict]:
    """就地给成交行补 `close_profit`（平仓盈亏，估算值）。

    * 卖出平多：`+（成交价 − 多头均价）× 手数 × 乘数`
    * 买入平空：`+（空头均价 − 成交价）× 手数 × 乘数`
    * 开仓单：`0.0`
    * 该方向均价拿不到 / 方向缺失：**`None`**

    最后那条是重点：`None` 界面画「—」，`0.0` 画「0.00」。实盘上「平了但
    不知道赚多少」和「这一笔刚好打平」是两回事，混成一个数会被当成真盈亏。
    """
    fn = mult_of or _mult_of
    ts = now or time.time()
    for t in trades or ():
        t["close_profit"] = 0.0
        if not _is_close_offset(t.get("offset")):
            continue
        sym = str(t.get("symbol") or "")
        vol = int(t.get("volume") or 0)
        price = float(t.get("price") or 0.0)
        if not sym or vol <= 0 or price <= 0:
            t["close_profit"] = None
            continue
        lg, sh = open_avg_of(cache, sym, ts, ttl)
        d = str(t.get("direction") or "").upper()
        if d.startswith("SELL"):
            avg, sign = lg, 1.0            # 卖出平多：价越高越赚
        elif d.startswith("BUY"):
            avg, sign = sh, -1.0           # 买入平空：价越低越赚
        else:
            t["close_profit"] = None
            continue
        if avg <= 0:
            t["close_profit"] = None
            continue
        t["close_profit"] = sign * (price - avg) * vol * float(fn(sym) or 1.0)
    return trades


def _trade_time_key(t: dict) -> tuple:
    """成交行的排序键。🔴 不能用 `time` 字符串直接比大小。

    两家的时间字段根本不是一回事：

    * 天勤：`time` = `"MM-DD HH:MM:SS"`（`_stamp()` 出的），带日期，字典序就是
      时序，优先用纳秒值 `trade_date_time`；
    * CTP：`time` = `TradeTime` = **只有 `"HH:MM:SS"`**，而夜盘 21:00 开、次日
      凌晨 02:30 收 —— 按字典序 `"01:30:00"` 会排到 `"21:00:00"` **前面**，
      一跨零点 FIFO 的账立刻错位（开仓被当成平仓之后的）。

    所以没有纳秒值时按「距夜盘开盘多少秒」重排：21:00 → 0，23:59 → 10740，
    00:00 → 10800（+3h 让凌晨接在深夜之后），09:00 → 43200。
    """
    try:
        ns = float(t.get("trade_date_time") or 0.0)
    except (TypeError, ValueError):
        ns = 0.0
    tid = str(t.get("trade_id") or "")
    if ns > 0:
        return (0.0, ns, tid)
    s = str(t.get("time") or "")
    hh = mm = ss = 0
    try:
        p = s.split()[-1].split(":")
        hh, mm, ss = int(p[0]), int(p[1]), int(p[2])
    except (IndexError, ValueError):
        pass
    if hh >= 20:                       # 夜盘（含 20:55 集合竞价）
        base = (hh - 21) * 3600 + mm * 60 + ss
    elif hh < 8:                       # 夜盘跨过零点那一段
        base = 3 * 3600 + hh * 3600 + mm * 60 + ss
    else:                              # 日盘
        base = 12 * 3600 + (hh - 9) * 3600 + mm * 60 + ss
    return (1.0, float(base), tid)


def replay_close_pnl(trades, mult_of=None, now: float = 0.0,
                     pre_settle_of=None) -> int:
    """用**当日成交流**按时间 FIFO 倒推每笔平仓的盈亏，返回算出来的笔数。

    🔴 为什么需要（2026-11-23 用户：「这边重新登录后，平仓盈亏都归 0 了」）：
    成交回报里没有盈亏字段（见 `est_close_pnl`），平时靠「持仓的开仓均价」倒推。
    可程序一重启，`_open_cache` 是空的，而**已经平掉的仓**在持仓表里也找不到
    均价 ⇒ 那些成交的平仓盈亏全变成「—」。

    当日成交流本身是**查得回来的**（CTP 走 `ReqQryTrade`，天勤登录后就有），
    按时间顺序把开仓堆进 FIFO、平仓从对面取，就能还原「这一笔平的是哪一批开仓、
    成本多少」。同一天开的仓算出来是**精确值**，不是估的。

    **成交流拿不到时按成本基准算**（2026-11-23 用户第二次拍板：「平仓盈亏可以
    根据昨日结算价或者收盘价格来计算，如果获取不到历史成交的话」）。所以 FIFO
    凑不齐的手数按 `pre_settle_of(sym)` 给的成本基准折算 —— 结算单的逐日盯市
    口径就是 `（平仓价 − 昨结算价）× 手数 × 乘数`，所以这不是编数。

    优先级（越前越准，都是「这笔平仓」的成本）：
      1. 当日成交流 FIFO（同日开平）—— 精确；
      2. `est_close_pnl` 已经写好的估算值（持仓开仓均价，只在还有仓时才有）——
         **不覆盖**：那一份比昨结算更贴这笔交易；
      3. 成本基准（昨结算 → 昨收 → 今收 → 本地日线昨收，由调用方给）。

    调用顺序：`est_close_pnl(...)` 之后再调它 —— 它能算的覆盖估算值。
    """
    fn = mult_of or _mult_of
    book: dict[tuple, list] = {}          # (合约, 多头/空头) -> [[开仓价, 剩余手数], …]
    rows = sorted(trades or (), key=_trade_time_key)
    done = 0
    for t in rows:
        sym = str(t.get("symbol") or "")
        vol = int(t.get("volume") or 0)
        px = float(t.get("price") or 0.0)
        d = str(t.get("direction") or "").upper()
        if not sym or vol <= 0 or px <= 0:
            continue
        if d.startswith("BUY"):
            mine, opp = "long", "short"
        elif d.startswith("SELL"):
            mine, opp = "short", "long"
        else:
            continue
        off = str(t.get("offset") or "").upper()
        if not _is_close_offset(off):
            book.setdefault((sym, mine), []).append([px, vol])
            continue
        lots = book.get((sym, opp)) or []
        need, left, cost = vol, vol, 0.0
        while left > 0 and lots:
            lot = lots[0]
            take = min(left, lot[1])
            cost += lot[0] * take
            lot[1] -= take
            left -= take
            if lot[1] <= 0:
                lots.pop(0)
        if left > 0:
            # 成交流里缺这部分开仓记录（昨仓，或者整个成交流都查不到）。
            # ① `est_close_pnl` 已经给了估算值 ⇒ 留着，它比昨结算更贴这笔交易
            if t.get("close_profit") is not None:
                continue
            # ② 退回成本基准（昨结算 / 昨收 / 今收 / 本地日线昨收）
            ps = 0.0
            if pre_settle_of is not None:
                try:
                    ps = float(pre_settle_of(sym) or 0.0)
                except Exception:  # noqa: BLE001
                    ps = 0.0
            if ps <= 0:
                continue                 # 连基准都没有，宁可留「—」也不编
            cost += ps * left            # 凑到的那部分按 FIFO，缺的按基准
        # 卖出平多：价越高越赚；买入平空：反过来
        sign = 1.0 if mine == "short" else -1.0
        t["close_profit"] = sign * (px - cost / need) * need * float(fn(sym) or 1.0)
        done += 1
    return done


def est_commission(trades, mult_of=None, exchange_of=None,
                   profile_key: str | None = None) -> list[dict]:
    """给成交行补 `commission`（这一笔的手续费，元）。

    **天勤的成交回报自带 `commission`**，直接透传 —— 那是柜台算的，和结算单
    一致。CTP / 融航的 `OnRtnTrade` 里**没有**逐笔手续费（只有账户级的汇总），
    所以只能按品种的成本档案估算，标 `commission_est=True`，界面上加个「≈」。

    `mult_of` / `exchange_of` 是给「合约代码不带交易所」的柜台用的（CTP 给的是
    光秃秃的 `rb2601`）：乘数查不到会退回 1.0，那手续费就差一个数量级；交易所
    查不到会退回「万分之一」的通用费率，也不准。两个都让调用方把更准的递进来。

    🔴 估算的**只算手续费、不含滑点**（见 `costs.commission`）：滑点是机会成本，
    不是账单。混进去这一列就没法和结算单对账了。

    `commission=None` 表示「算不出来」（乘数没维护 / 价格缺失），界面画「—」；
    0 表示「这一笔真的不收费」，两个不能混。
    """
    from . import costs as _costs

    for t in trades or ():
        if t.get("commission") is not None:
            t["commission_est"] = False
            continue
        sym = str(t.get("symbol") or "")
        head, _, tail = sym.partition(".")
        ex, code = (head, tail) if tail else ("", sym)
        prod = "".join(c for c in code if c.isalpha()).upper()
        if not ex and exchange_of is not None:
            try:
                ex = str(exchange_of(sym) or "")
            except Exception:  # noqa: BLE001 - 查不到就退回通用费率
                ex = ""
        mult = 0.0
        if mult_of is not None:
            try:
                mult = float(mult_of(sym) or 0.0)
            except Exception:  # noqa: BLE001
                mult = 0.0
        fee = _costs.commission(prod, ex, t.get("price") or 0.0,
                                int(t.get("volume") or 0), profile_key,
                                mult=mult)
        t["commission"] = fee
        t["commission_est"] = fee is not None
    return trades


@dataclass
class LiveState:
    """一次快照，界面拿去直接刷控件。全是普通 dict/list，不持有天勤对象。

    `account` / `positions` / `orders` / `trades` 是**全部账户合并**后的结果
    （多账户时求和 / 拼接），`by_account` 里则是按账户分开的那一份：

        by_account = {"账户名": {"spec": {...}, "account": {...},
                                "positions": [...], "orders": [...], "trades": [...]}}

    合并那份给「还没选账户」的首屏用，分开那份给账户列表和双击切换用。
    两份都是拷贝，界面线程怎么折腾都碰不到天勤对象。
    """

    connected: bool = False
    error: str = ""
    key: str = ""
    period: str = ""
    underlying: str = ""
    quote: dict = field(default_factory=dict)
    # 「报价」框：盯着的每个合约一行（持仓合约 + 下单框里的 + 手动加的）
    quotes: list = field(default_factory=list)
    account: dict = field(default_factory=dict)
    positions: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    accounts: list = field(default_factory=list)        # 账户列表（一行一个）
    by_account: dict = field(default_factory=dict)      # 账户名 → 该账户的明细


def merge_states(parts, feed: str = "") -> LiveState:
    """把几个柜台各自的快照合成一份，给「同时登录多个柜台」用。

    `parts` 是 `[(后端, LiveState)]`，后端取 `BACKENDS` 里的值；顺序决定账户表
    的排列（天勤 → CTP → 融航）。

    **行情只取一路**：`feed` 指定的那个后端的 `quote` / `bar 周期` / 主力合约
    作为整份快照的行情。两个柜台的 tick 拼在一条 K 线上会来回跳（时间戳、口径、
    甚至合约都不是同一个），所以这里不做「取最新」这种聪明事，认准一个源。

    **账户全合并**：`accounts` 拼起来，`by_account` 合并（同名时加后端后缀，
    极少见 —— 账户表按名字去重，正常不会撞）。合计资金用 `_sum_accounts` 重算，
    风险度是比值不能加。

    `bar` 本身不在快照里（由 worker 直接发选定的那一路），这里只统一
    `quote` / `underlying` / `period` / `key`，免得界面上「K 线是一个柜台的、
    盘口是另一个柜台的」这种对不上的情况。
    """
    parts = [(name, st) for name, st in (parts or []) if st is not None]
    if not parts:
        return LiveState()
    src = next((st for name, st in parts if name == feed), None)
    if src is None:
        # 选的那路没连上（或者调用方没指定）：退到第一个报得动行情的
        src = next((st for name, st in parts if (st.quote or st.key)), parts[0][1])

    st = LiveState(
        connected=any(p.connected for _, p in parts),
        error="；".join(f"[{BACKEND_LABELS.get(n, n)}] {p.error}"
                       for n, p in parts if p.error),
        key=src.key,
        period=src.period,
        underlying=src.underlying,
        quote=dict(src.quote or {}),
        quotes=list(src.quotes or []),
    )

    acc_dicts: list[dict] = []
    for name, p in parts:
        st.accounts.extend(p.accounts or [])
        st.positions.extend(p.positions or [])
        st.orders.extend(p.orders or [])
        st.trades.extend(p.trades or [])
        for key, item in (p.by_account or {}).items():
            k = key
            if k in st.by_account:
                # 两个柜台出现同名账户：加上后端后缀，否则后一个会把前一个顶掉
                k = f"{key}（{BACKEND_LABELS.get(name, name)}）"
            st.by_account[k] = item
            acc_dicts.append((item or {}).get("account") or {})
    st.account = _sum_accounts(acc_dicts) if acc_dicts else dict(src.account or {})
    return st


class LiveHub:
    """一个天勤连接：行情订阅 + 账户 + 下单。

    生命周期：`start()` → 反复 `wait()` → `close()`。`wait()` 之外的调用都只是
    读活引用，不会阻塞。所有异常都写进 `last_error`，不让界面线程炸掉。

    🔴 **这个对象是进程内共享的**（2026-11-24 起）：交易页和组合页按账户名
    从 `ttrade/hubreg.py` 领到**同一个 `LiveHub`**，`start()` 幂等、`close()`
    带引用计数。唯一能碰 `TqApi` 的那条线程是 `_pump`，别的线程全程走 `_lock`
    —— 详见 `_serial` 和 `_api_call` 的注释。

    **多账户**：`specs` 给多个账户时走天勤的 `TqMultiAccount` —— 一条连接、
    一次登录，`insert_order(..., account=obj)` 指定落到哪个账户。但**共享那条路
    只发单账户**：`hubreg` 的键是「一个账户」，两个账户共享一条 TqApi 会让
    「登出其中一个」变成「两个一起断」。
    """

    def __init__(self, specs):
        # `specs` 为空 = **纯行情会话**：研究页的「启动实时」只要 K 线，不要账户。
        # 这条链路和下载脚本共用一份鉴权（`read_account()`），不构造任何账户对象。
        if specs is None:
            specs = []
        elif isinstance(specs, AccountSpec):
            specs = [specs]
        self.specs: list[AccountSpec] = list(specs)
        self.spec = self.specs[0] if self.specs else None
        self.api = None
        self.last_error = ""
        self.key = ""            # 品种_交易所
        self.period = ""
        self._n = 1              # 目标周期相对源周期的合并根数
        self.tq_symbol = ""      # KQ.m@SHFE.rb
        self.underlying = ""     # 当前真实主力合约
        self._sym = ""           # 订阅时传进来的品种 / 交易所（交易时段门禁用）
        self._ex = ""
        self._quote_cache: dict = {}   # 最后一笔「交易时段内」的行情快照
        self._klines = None
        self._quote = None
        self._accs: list[tuple[AccountSpec, object]] = []   # (配置, 天勤账户对象)
        self._order_book: dict = {}
        self._targets: dict = {}
        # 开仓均价缓存：平仓盈亏靠它倒推（见 `update_open_avg`）。持仓一平光
        # `Position` 就消失，不留一份上一轮的均价，那一列永远是「—」
        self._open_cache: dict = {}
        # 「报价」框盯的合约（`watch()` 设）。这里只存代码，真正的 quote 对象
        # 在 `_watch` 里，`state()` 时逐个取最新价
        self._watch_syms: list[str] = []
        self._watch: dict = {}
        # 交易所对日内报单 / 撤单次数有硬上限。组合页的引擎和这里共用同一份
        # 落盘计数（同一个账户、同一个进程只能有一份）。多账户各自计数。
        self.limit = daylimit.DayLimit(self.spec.label() if self.spec else "账户")

        # ---- 会话共享（2026-11-24）----
        # 🔴 一个天勤账户在整个系统只能有一条 `TqApi`。以前交易页一条、组合页
        # 一条，同一账户被登了两次：天勤按会话推持仓/委托/成交回报，两条会话
        # 互不相通 ⇒ 「一键对齐发出去的单在交易页看得到、监控器上不动」。
        #
        # `TqApi` 是单线程语义（`wait_update()` 只能在一条线程里调），所以共享
        # 的做法和 `MarketFeed` 一样：**一条泵线程独占 `wait_update`**，其它
        # 线程的每次调用都走 `self._lock`，`wait()` 退化成「等泵推来下一拍」。
        self._lock = threading.RLock()
        #: 和 `_lock` 同一把（`Condition` 会在等待期间释放它，泵才拿得到）
        self._cv = threading.Condition(self._lock)
        #: `wait_update` 拿到新数据的次数。`wait()` 靠它判断「这一拍有没有来」
        self._tick = 0
        self._start_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stopped = False

    # ---- 生命周期 ----

    def start(self) -> None:
        """建立连接并取好所有活引用。失败抛异常，由调用方决定怎么提示。

        **幂等**：别人（交易页 / 组合页）已经登过这条会话时立刻返回，不会二次
        登录。`hubreg.acquire()` 会把同一个账户的同一条连接发给两个使用者，
        后到的那一方必须能安全地再调一次 `start()` —— 不幂等的话它会把已经
        登录好的 `api` 换掉，先到那方手里的活引用当场变野指针。
        """
        from tqsdk import TqApi, TqAuth

        with self._start_lock:
            if self.api is not None:
                self._start_pump()
                return
            if not self.specs:
                # 纯行情会话。天勤允许不带账户建连接（只要 TqAuth），这样研究页的
                # 实时不依赖「账户透视和交易」页有没有登录，也不会多占一个资金账号
                # 的连接名额。
                from ttrade.tqsdk_symbols import read_account

                user, pwd = read_account()
                if not user or not pwd:
                    raise ValueError("没有配置天勤账户，无法连接实时行情")
                self.api = TqApi(auth=TqAuth(user, pwd), web_gui=False)
                self._start_pump()
                return
            alien = [s.name for s in self.specs if s.backend != "tq"]
            if alien:
                # 柜台账户（CTP / 融航）在这里必须被挡住。掉到下面去的话，报的是
                # 「缺少天勤账户或密码」，用户会以为自己账号密码填错了
                raise ValueError(
                    f"账户「{'、'.join(alien)}」是柜台账户，不走天勤，"
                    "不能放进天勤连接（应各自单开一条柜台连接）"
                )
            for s in self.specs:
                if not s.user or not s.password:
                    raise ValueError(f"账户「{s.name}」缺少天勤账户或密码")
            objs = [s.build() for s in self.specs]
            # 同一个资金账号选两次会被天勤拒绝（account_key 重复）。提前说清楚，
            # 别让用户对着一句英文异常猜。
            keys = [getattr(o, "_account_key", id(o)) for o in objs]
            if len(set(keys)) != len(keys):
                raise ValueError("同一天勤资金账号不能重复连接，请取消其中一个的勾选")
            if len(objs) == 1:
                self.api = TqApi(
                    objs[0],
                    auth=TqAuth(self.specs[0].user, self.specs[0].password),
                    web_gui=False)
            else:
                from tqsdk import TqMultiAccount

                self.api = TqApi(TqMultiAccount(objs),
                                 auth=TqAuth(self.specs[0].user, self.specs[0].password),
                                 web_gui=False)
            self._accs = list(zip(self.specs, objs))
            self._start_pump()

    # ---- 泵线程：唯一驱动 wait_update 的那条线程 ----

    def _start_pump(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopped = False
        self._thread = threading.Thread(
            target=self._pump, name="tq-pump", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        """不停地 `wait_update`。**`TqApi` 只在这一条线程里被推进。**

        别的方法只读活引用、或者把调用塞进 `_lock` 串行执行 —— tqsdk 明确说
        `wait_update()` 只能在一条线程里调，两条线程一起驱动会把内部订阅状态
        写坏（表现是「单子偶尔丢、偶尔重复」，而且不报错）。

        出异常就停：再转下去只会每 0.1 秒刷一次同样的错。`last_error` 留着，
        界面据此显示「行情中断」，`wait()` 会因为没新拍而超时返回 False。
        """
        while not self._stopped:
            api = self.api
            if api is None:
                break
            try:
                with self._lock:
                    got = bool(api.wait_update(deadline=time.time() + PUMP_SEC))
            except Exception as exc:  # noqa: BLE001 - 泵线程不能抛出去
                self.last_error = f"{type(exc).__name__}: {exc}"
                got = False
                with self._cv:
                    # 叫醒还在等的人，否则他们的 `wait()` 会一直等到超时
                    self._tick += 1
                    self._cv.notify_all()
                break
            if got:
                with self._cv:
                    self._tick += 1
                    self._cv.notify_all()

    def _api_call(self, name: str, *args, **kwargs):
        """在会话锁里调一次 `TqApi` 的方法（泵线程用的是同一个 `TqApi`）。"""
        api = self.api
        if api is None:
            raise RuntimeError("还没连接")
        with self._lock:
            return getattr(api, name)(*args, **kwargs)

    def close(self) -> None:
        """收掉这条会话。**只有最后一个使用者才真的关**（见 `ttrade/hubreg.py`）。

        交易页和组合页共用同一条天勤连接：一边登出就把 `TqApi` 关掉，另一边
        的持仓 / 委托当场全冻住，而界面上只是「数据不更新了」。
        """
        if not hubreg.release(self):
            return                   # 还有别人在用，谁都不许关
        hubreg.forget(self)          # 直接 close 的路径（断线收尾）也要摘干净
        with self._start_lock:
            self._stopped = True
            t, self._thread = self._thread, None
        api, self.api = self.api, None
        self._klines = self._quote = None
        self._accs = []
        self._targets.clear()
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)      # 泵最多卡在 `PUMP_SEC` 的 wait_update 里
        with self._cv:
            self._cv.notify_all()    # 叫醒还在 wait() 里的人
        if api is not None:
            try:
                api.close()
            except Exception:
                pass

    @property
    def alive(self) -> bool:
        return self.api is not None

    # ---- 订阅 ----

    @_serial
    def subscribe(self, symbol: str, exchange: str, period: str) -> None:
        """订阅一个品种的主连 K 线与实时行情。

        重复订阅同一组参数是天勤允许的（同一个 serial 返回同一对象），但这里
        仍然记下当前 key，避免界面在没换品种时反复调用。
        """
        if self.api is None:
            raise RuntimeError("还没连接")
        # 订**源周期**再本地合并：15m/30m 的本地历史是 5m×3/×6 合出来的，
        # 直接订原生 15m 时间边界对不上，拼起来就是碎的（2h/4h 同理用 1h）
        src, n = PERIODS.get(period, (period, 1))
        sec = PERIOD_SEC.get(src)
        if sec is None:
            raise ValueError(f"实时不支持 {period}（源周期 {src} 没有对应秒数）")
        key = contract_key(symbol, exchange)
        if key == self.key and period == self.period and self._klines is not None:
            return
        # 期货订主连（`KQ.m@SHFE.rb`），股票直接订代码本身（`SSE.600000`）——
        # 股票没有主力合约这一说，`main_symbol()` 拼出来的 `KQ.m@SSE.600000`
        # 是不存在的合约
        tq_sym = trade_symbol(symbol, exchange)
        self._quote = self.api.get_quote(tq_sym)
        self._klines = self.api.get_kline_serial(
            tq_sym, sec, data_length=min(10000, KLINE_LEN * max(1, n))
        )
        self._n = n
        self.key, self.period, self.tq_symbol = key, period, tq_sym
        self._sym, self._ex = symbol, exchange
        self._quote_cache = {}
        # 股票没有「主连 → 真实合约」这一步，订的就是能下单的那个代码
        self.underlying = tq_sym if is_stock(symbol, exchange) else ""

    @_serial
    def watch(self, symbols) -> None:
        """盯一列合约的盘口，给交易页的「报价」框用。

        和 `subscribe()` 的分工：`subscribe` 是「画 K 线的那一个品种」，一条会话
        只有一个；`watch` 是「随时要看最新价的十几个合约」，用 `get_quote()`
        就够（不建 kline serial，不多占带宽）。幂等：重复传同一批只是命中缓存。

        **必须在 worker 线程里调** —— `get_quote` 会动订阅状态。
        """
        if self.api is None:
            return
        want: list[str] = []
        for raw in symbols or ():
            sym = str(raw or "").strip()
            if not sym:
                continue
            if "." not in sym:
                sym, _ex = contract_from_input(sym)
            if sym and sym not in want:
                want.append(sym)
        self._watch_syms = want
        for sym in want:
            if sym in self._watch:
                continue
            try:
                self._watch[sym] = self.api.get_quote(sym)
            except Exception:  # noqa: BLE001 - 单个合约拼错不该废掉整列
                continue
        for sym in [s for s in self._watch if s not in want]:
            self._watch.pop(sym, None)

    def watched_quotes(self) -> dict:
        """给日内现算用的订阅报价表（`{天勤合约: 行情对象}`）。

        和 `state()["quotes"]` 的区别：那份是给界面显示用的普通 dict，没有
        `open_interest`；这里给的是天勤活引用，现算器直接读它需要的值。
        """
        with self._lock:
            return dict(self._watch)

    # ---- 推进 ----

    def wait(self, timeout: float = 0.8) -> bool:
        """等泵推来下一拍。返回 False 表示这段时间没有新数据（不是出错）。

        🔴 **这里不再自己调 `wait_update`**（2026-11-24 会话共享）：`TqApi` 只能
        被一条线程推进，现在那条线程是 `_pump`。调用方（交易页 worker /
        组合页 runner）只负责「按自己的节奏转一圈」，数据由泵送进来 ——
        两边共用一条会话时这是唯一安全的写法。

        没有泵（单测里的假 api、或者还没 `start()`）时退回老行为，直接推一次：
        自检和注入的假柜台不需要线程。

        泵**因为异常退出**时把 `last_error` 抛出去 —— 不抛的话界面只会看到
        「每 0.8 秒都超时」，账户表和 K 线就那么冻着，日志里一句错的都没有
        （以前 `wait_update` 抛异常是从这里冒出去的，调用方靠它报「连接中断」）。
        """
        if self.api is None:
            return False
        t = self._thread
        if t is None:
            try:
                with self._lock:
                    return bool(self.api.wait_update(deadline=time.time() + timeout))
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise
        if not t.is_alive():
            if self.last_error and not self._stopped:
                raise RuntimeError(self.last_error)
            return False
        n = self._tick
        end = time.time() + max(0.0, float(timeout))
        with self._cv:
            while self._tick == n:
                left = end - time.time()
                if left <= 0 or self._stopped:
                    return False
                # `Condition(self._lock)` 在等待期间把会话锁放掉 —— 泵才拿得到
                self._cv.wait(left)
            return True

    # ---- 读 ----

    def quote_of(self, symbol: str, timeout: float = 3.0) -> dict:
        """取**任意一个**合约的最新盘口，给下单面板「填最新价」用。

        和 `state()["quote"]` 的区别：那个是当前订阅的主连，只会有一个；这里按
        合约现取。下单面板的合约和图上看的品种经常不是一个（图上 rb、手上要下
        m2601），所以必须有这条路径。

        **必须在 worker 线程里调** —— `get_quote` 会动订阅状态，所以要进会话锁
        （泵线程也在用同一个 `TqApi`）。新订阅第一次通常拿不到
        数据，`last_price` 是 NaN —— 要等一两轮推送。所以这里带墙钟上限轮询，
        拿不到就返回空 dict，让界面说「还没拿到这个合约的行情」，而不是把 NaN
        当成 0 填进价格框（0 价的限价单会被柜台按「价格不合法」拒掉）。

        `symbol` 必须是天勤写法 `SHFE.rb2601`；主连 `KQ.m@` 也能取，但下单要用
        真实合约，所以界面上不该出现它。
        """
        if self.api is None:
            raise RuntimeError("还没连接")
        sym = str(symbol or "").strip()
        if not sym or "." not in sym:
            raise ValueError(f"要天勤写法的合约（如 SHFE.rb2601），收到 {symbol!r}")
        with self._lock:
            q = self.api.get_quote(sym)

        end = time.time() + max(0.5, float(timeout))
        while time.time() < end:
            if _qnum(q, "last_price"):
                break
            try:
                # 等泵推下一拍就够了，别在这里自己驱动 `wait_update`
                self.wait(0.5)
            except Exception:  # noqa: BLE001 - 抢不到推送就再转一圈，超时由墙钟兜
                break
        last = _qnum(q, "last_price")
        if not last:
            return {}
        return _quote_row(q, sym)

    @_serial
    def bars(self) -> pd.DataFrame:
        """实时 K 线。

        🔴 **收盘之后的 bar 不采信**：天勤的模拟账户（快期模拟 / TqSim）在非交易
        时段照样会在序列里继续长 bar，界面上看着像还在交易。这里从尾部往前砍掉
        落在时段外的那些 —— 只砍尾，因为 bar 是按时间排的，中间不会冒出一根
        「时段外」的（真冒出来说明这张表本身配错了，那是另一件事）。
        只砍尾还有个好：通常是 0 根，`while` 一次就退出，不拖慢刷新。
        """
        df = resample(bars_from_serial(self._klines, self.period), self._n)
        if df.empty or symbols_info.in_session(self._sym, self._ex):
            return df
        j = len(df) - 1
        while j >= 0 and not symbols_info.in_session(self._sym, self._ex, df.index[j]):
            j -= 1
        return df.iloc[: j + 1]

    @_serial
    def state(self) -> LiveState:
        """拷一份普通数据出来。界面线程只碰这个，不碰天勤对象。"""
        # 快照要先取：`underlying` 是在 `_quote_snapshot()` 里跟着主连 quote 更新的，
        # 先构造 LiveState 会拿到上一轮的值（订阅后的第一次必然为空），而自动下单
        # 正是靠它拿真实合约 —— 空一次就丢一次信号。
        quote = self._quote_snapshot()
        st = LiveState(
            connected=self.api is not None,
            error=self.last_error,
            key=self.key,
            period=self.period,
            underlying=self.underlying,
        )
        st.quote = quote
        st.quotes = self._watch_rows()
        self._fill_accounts(st)
        return st

    def _watch_rows(self) -> list[dict]:
        """「报价」框的数据：盯着的每个合约一行。

        还没拿到第一帧的合约给 `last=0`（界面画「—」），不能漏掉这一行 ——
        漏了的话用户加进去的合约会在表里「消失」，看着像没加成功。
        """
        out = []
        for sym in self._watch_syms:
            q = self._watch.get(sym)
            if q is None:
                out.append({"symbol": sym, "instrument": "", "last": 0.0})
                continue
            out.append(_quote_row(q, sym))
        return out

    def _fill_accounts(self, st: LiveState) -> None:
        """按账户拷一份明细，再合并出总量。

        `positions` / `orders` / `trades` 三个活引用每个账户各有一份，逐账户
        取出来加 `account` 字段，界面就不用再猜某一行属于谁。
        """
        rows, detail = [], {}
        for spec, obj in self._accs:
            try:
                pos = self._positions_snapshot(obj)
                ords = self._orders_snapshot(obj)
                trds = self._trades_snapshot(obj)
                acc = self._account_snapshot(obj)
                # 平仓盈亏只能倒推：先用这一轮持仓刷新开仓均价，再拿它估成交。
                # 顺序不能反 —— 刚平光的仓这一轮已经不在持仓里了，靠的正是缓存
                update_open_avg(self._open_cache, pos)
                est_close_pnl(trds, self._open_cache)
                # 再用当日成交流 FIFO 精确还原一遍，覆盖估算值（2026-11-23
                # 用户：重新登录后平仓盈亏全归 0）。重启后 `_open_cache` 是空的，
                # 已经平掉的仓连持仓表里都没有均价 ⇒ 上面那步只能给 `None`
                replay_close_pnl(trds, pre_settle_of=self._pre_settle)
                # 算完再截断给界面（`_trades_snapshot()` 故意不截断，见那里注释）
                trds = trds[:SNAP_ROWS]
                est_commission(trds, mult_of=_mult_of)
            except Exception as exc:  # noqa: BLE001
                # 单个账户查不出来不该拖垮整页：其余账户照样能看能下单
                self.last_error = f"{spec.name}：{type(exc).__name__}: {exc}"
                pos, ords, trds, acc = [], [], [], {}
            for r in pos:
                r["account"] = spec.name
            for r in ords:
                r["account"] = spec.name
            for r in trds:
                r["account"] = spec.name
            detail[spec.name] = {
                "spec": asdict(spec), "account": acc,
                "positions": pos, "orders": ords, "trades": trds,
            }
            rows.append(self._account_row(spec, acc, pos, trds))

        st.accounts = rows
        st.by_account = detail
        if not rows:
            return
        st.account = _sum_accounts([r for r in rows])
        st.positions = sorted(
            [r for d in detail.values() for r in d["positions"]], key=lambda r: r["symbol"]
        )
        st.orders = sorted(
            [r for d in detail.values() for r in d["orders"]],
            key=lambda r: (r["status"] != "ALIVE", -r["insert_date_time"]),
        )[:SNAP_ROWS]
        st.trades = sorted(
            [r for d in detail.values() for r in d["trades"]],
            key=lambda r: -r["trade_date_time"],
        )[:SNAP_ROWS]

    def _account_row(self, spec: AccountSpec, acc: dict, pos: list, trds: list) -> dict:
        """账户列表的一行：资金 + 杠杆 + 今日成交额。

        杠杆和成交额都要「手 → 钱」的换算，也就是合约乘数。乘数从本地日线的
        成交额反推（`ttrade.specs`），不额外请求行情 —— 账户页每 0.4 秒刷一次，
        在这里去订阅几十个合约的 quote 会把连接压垮。
        """
        mv_long = mv_short = 0.0
        for p in pos:
            mult = _mult_of(p["symbol"])
            # 市值 = 开仓成本×乘数 + 浮盈。开仓成本是「价×手」，乘上乘数才是钱；
            # 浮盈本身就是钱，直接加。持仓对象上没有 last_price，拿不到盘口价时
            # 这个式子仍然成立（它只用开仓价与浮盈，两个都是现成的）。
            cost_l = p.get("open_cost_long") or p["open_price_long"] * p["pos_long"]
            cost_s = p.get("open_cost_short") or p["open_price_short"] * p["pos_short"]
            mv_long += cost_l * mult + p["float_profit_long"]
            mv_short += cost_s * mult + p["float_profit_short"]
        amount = 0.0
        for t in trds:
            amount += t["price"] * t["volume"] * _mult_of(t["symbol"])
        bal = float(acc.get("balance") or 0.0)
        equity = bal if bal > 0 else 0.0
        return {
            "name": spec.name,
            "platform": dict(ACCOUNT_KINDS).get(spec.kind, spec.kind),
            "account_id": spec.account_id or spec.user or "—",
            "balance": bal,                                   # 最新权益
            "day_profit": bal - float(acc.get("pre_balance") or bal),   # 当日盈亏
            "net_leverage": (abs(mv_long - mv_short) / equity) if equity else 0.0,
            "gross_leverage": ((mv_long + mv_short) / equity) if equity else 0.0,
            "available": float(acc.get("available") or 0.0),
            "turnover": amount,                               # 成交金额
            "margin": float(acc.get("margin") or 0.0),
            "commission": float(acc.get("commission") or 0.0),
            "positions": len(pos),
            "orders": len(trds),
        }

    def _num(self, obj, name: str, default=0.0) -> float:
        v = getattr(obj, name, None)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _quote_snapshot(self) -> dict:
        q = self._quote
        if q is None:
            return {}
        # 主连的 underlying_symbol 就是当前真实主力；换月时会变。
        # A 股没有这一层：订的代码本身就能下单，`underlying` 在 subscribe 里
        # 已经填好了，这里不能拿 quote 的 underlying_symbol 去覆盖（股票行情里
        # 这个字段是空的，覆盖一次就再也回不来了）
        und = getattr(q, "underlying_symbol", "") or ""
        if und and und != self.underlying:
            self.underlying = und
        # 🔴 交易时段门禁：和 `ctp_live` 同一条判据。开市时刷新缓存，收市时把
        # 最后一笔时段内快照原样还回去 —— 否则收盘后「最新价」还在跳，
        # 用户没法判断这台机器现在到底还在不在交易。
        open_now = symbols_info.in_session(self._sym, self._ex)
        if not open_now and self._quote_cache:
            return dict(self._quote_cache)
        snap = {
            "symbol": getattr(q, "instrument_id", ""),
            "underlying": und,
            "name": getattr(q, "instrument_name", ""),
            "datetime": getattr(q, "datetime", ""),
            "last": self._num(q, "last_price"),
            "bid": self._num(q, "bid_price1"),
            "ask": self._num(q, "ask_price1"),
            "bid_vol": self._num(q, "bid_volume1"),
            "ask_vol": self._num(q, "ask_volume1"),
            "open": self._num(q, "open"),
            "high": self._num(q, "highest"),
            "low": self._num(q, "lowest"),
            "pre_close": self._num(q, "pre_close"),
            "volume": self._num(q, "volume"),
            "oi": self._num(q, "open_interest"),
            "price_tick": self._num(q, "price_tick"),
            "multiple": self._num(q, "volume_multiple"),
            "expired": bool(getattr(q, "expired", False)),
        }
        if open_now:
            self._quote_cache = snap
        return dict(snap)

    def _account_snapshot(self, obj=None) -> dict:
        """单个账户的资金。`obj` 是天勤账户对象（TqKq / TqAccount / TqSim）。

        多账户模式下 `api.get_account()` 必须带账户实例，而账户对象自己的
        `get_account()` 在单账户、多账户两种模式下都能用 —— 所以这里一律走
        账户对象，两条路径共用一套代码。
        """
        if obj is None:
            return {}
        a = obj.get_account()
        if a is None:
            return {}
        return {
            "balance": self._num(a, "balance"),            # 动态权益
            "available": self._num(a, "available"),        # 可用资金
            "margin": self._num(a, "margin"),              # 占用保证金
            "float_profit": self._num(a, "float_profit"),  # 浮动盈亏（持仓盈亏）
            "position_profit": self._num(a, "position_profit"),
            "close_profit": self._num(a, "close_profit"),  # 平仓盈亏
            "commission": self._num(a, "commission"),
            "risk_ratio": self._num(a, "risk_ratio"),
            "market_value": self._num(a, "market_value"),
            "pre_balance": self._num(a, "pre_balance"),    # 昨结算权益（算当日盈亏）
            "static_balance": self._num(a, "static_balance"),
        }

    def _positions_snapshot(self, obj) -> list[dict]:
        if obj is None:
            return []
        src = obj.get_position()
        if src is None:
            return []
        out = []
        items = src.items() if hasattr(src, "items") else []
        for _sym, p in items:
            pos = int(self._num(p, "pos"))
            if pos == 0 and not self._num(p, "pos_long") and not self._num(p, "pos_short"):
                continue
            out.append({
                # `Position` 上没有 `symbol` 属性 —— 合约代码是 `get_position()`
                # 返回的 dict 的 key。早先取 `p.symbol` 拿到空串，持仓表的「合约」
                # 列整列空白，而委托/成交用的是 `instrument_id` 所以正常。
                "symbol": getattr(p, "instrument_id", "") or _sym,
                "pos": pos,
                "pos_long": int(self._num(p, "pos_long")),
                "pos_short": int(self._num(p, "pos_short")),
                "pos_long_today": int(self._num(p, "pos_long_today")),
                "pos_short_today": int(self._num(p, "pos_short_today")),
                "open_price_long": self._num(p, "open_price_long"),
                "open_price_short": self._num(p, "open_price_short"),
                "open_cost_long": self._num(p, "open_cost_long"),
                "open_cost_short": self._num(p, "open_cost_short"),
                "float_profit": self._num(p, "float_profit"),
                "float_profit_long": self._num(p, "float_profit_long"),
                "float_profit_short": self._num(p, "float_profit_short"),
                "position_profit": self._num(p, "position_profit"),
                "margin": self._num(p, "margin"),
                # Position 上没有 last_price（那是 SecurityPosition 的字段），
                # 这里留着 0 只为了列名对齐，别拿它当盘口价用
                "last_price": self._num(p, "last_price"),
            })
        out.sort(key=lambda r: r["symbol"])
        return out

    def _orders_snapshot(self, obj) -> list[dict]:
        if obj is None:
            return []
        src = obj.get_order()
        if src is None:
            return []
        items = list(src.items()) if hasattr(src, "items") else []
        # **先挑后拼**：一天被拒的废单能有几千条，全拼一遍 dict 再截断，
        # 等于每 0.4 秒白烧一次 CPU，还要和主线程抢 GIL。先排序取前 N 条，
        # 只给这几条建 dict。
        items.sort(key=lambda kv: (getattr(kv[1], "status", "") != "ALIVE",
                                   -self._num(kv[1], "insert_date_time")))
        out = []
        for oid, o in items[:SNAP_ROWS]:
            out.append({
                "order_id": oid,
                "symbol": getattr(o, "instrument_id", "") or getattr(o, "symbol", ""),
                "direction": getattr(o, "direction", ""),
                "offset": getattr(o, "offset", ""),
                "volume_orign": int(self._num(o, "volume_orign")),
                "volume_left": int(self._num(o, "volume_left")),
                "limit_price": self._num(o, "limit_price"),
                "status": getattr(o, "status", ""),
                "last_msg": getattr(o, "last_msg", ""),
                "insert_date_time": self._num(o, "insert_date_time"),
                "time": _stamp(getattr(o, "insert_date_time", 0)),
            })
        return out

    def _trades_snapshot(self, obj) -> list[dict]:
        if obj is None:
            return []
        src = obj.get_trade()
        if src is None:
            return []
        items = list(src.items()) if hasattr(src, "items") else []
        items.sort(key=lambda kv: -self._num(kv[1], "trade_date_time"))
        out = []
        # 🔴 这里**不截断**（2026-11-23）：`replay_close_pnl` 要拿当日全部成交
        # 按时间 FIFO 倒推平仓盈亏，只留最近 200 笔的话，早上的开仓记录就不在
        # 里面了 —— 平仓那几笔一笔都算不出来。截断挪到 `_fill_accounts` 算完
        # 之后（那里截一次，界面看到的仍然是 `SNAP_ROWS` 行）。
        for tid, t in items:
            # 天勤的 Trade 带 `commission`（元，柜台算的）。老版本 / 别的源可能
            # 没有这个键 —— 用 `None` 表示「没有」，交给 `est_commission` 估；
            # 不能写 0，0 的意思是「这一笔真的不收费」
            try:
                fee = float(getattr(t, "commission"))
            except (AttributeError, TypeError, ValueError):
                fee = None
            out.append({
                "trade_id": tid,
                "order_id": getattr(t, "order_id", ""),
                "symbol": getattr(t, "instrument_id", "") or getattr(t, "symbol", ""),
                "direction": getattr(t, "direction", ""),
                "offset": getattr(t, "offset", ""),
                "price": self._num(t, "price"),
                "volume": int(self._num(t, "volume")),
                "commission": fee,
                "trade_date_time": self._num(t, "trade_date_time"),
                "time": _stamp(getattr(t, "trade_date_time", 0)),
            })
        return out

    def _pre_settle(self, symbol: str) -> float:
        """平仓盈亏的**成本基准**：昨结算 → 昨收 → 今收 → 本地日线昨收。

        `replay_close_pnl` 在「成交流凑不齐这笔平仓的开仓记录」时拿它当成本
        （2026-11-23 用户：「平仓盈亏可以根据昨日结算价或者收盘价格来计算，
        如果获取不到历史成交的话」）。

        🔴 顺序按**口径正确性**排：`pre_settlement`（昨结算，结算单的逐日盯市
        基准）→ `pre_close`（昨收）→ `close`（今收，收市后才有值，盘中是 NaN）。
        盯着的表里没有这个合约时退回主连 quote —— 同品种不同月份的量级一定对得
        上，比留一个「—」有用；再不行读本地日线的最后一根收盘。

        拿不到返回 0（调用方据此不写这一笔），**绝不拿 0 当成本**。
        """
        q = self._watch.get(symbol)
        if q is None:
            mq = self._quote
            if mq is not None:
                inst = str(getattr(mq, "instrument_id", "") or "")
                und = str(getattr(mq, "underlying_symbol", "") or "")
                if symbol in (inst, und):
                    q = mq
        if q is not None:
            for key in ("pre_settlement", "pre_close", "close"):
                try:
                    v = float(_qnum(q, key) or 0.0)
                except Exception:  # noqa: BLE001
                    v = 0.0
                if v > 0:            # NaN 在这里也是 False，正好跳过
                    return v
        return _last_daily_close(symbol)

    # ---- 写 ----
    def _acc_obj(self, name: str = ""):
        """按名字找天勤账户对象。名字为空时返回唯一的那个（单账户）。"""
        if not self._accs:
            raise RuntimeError("还没连接")
        if not name:
            if len(self._accs) == 1:
                return self._accs[0][1]
            raise RuntimeError("多账户下必须指定账户")
        for spec, obj in self._accs:
            if spec.name == name:
                return obj
        raise ValueError(f"没有名为「{name}」的账户")

    @_serial
    def place(self, symbol: str, direction: str, offset: str, volume: int,
              limit_price: float | None = None, account: str = "") -> str:
        """下单。返回客户端订单号。`account` 是多账户时的目标账户名。

        **A 股的数量单位是「股」不是「手」**：天勤股票账户的 `volume` 就是股数，
        100 股起、必须是 100 的整数倍。界面上那一栏的标题会跟着账户类型变，
        这里不替用户乘 100 —— 悄悄放大一百倍比报错危险得多。

        注意 `insert_order()` 只是**入队**，真正的报文在下一次 `wait_update()` 才发出去。
        所以调用方下完单必须继续跑 `wait()` 循环，否则单子永远不会到交易所。
        """
        if self.api is None:
            raise RuntimeError("还没连接")
        if volume <= 0:
            raise ValueError("手数必须大于 0")
        if not self.limit.order():
            raise RuntimeError(
                f"今日报单已达 {daylimit.DAY_ORDER_LIMIT} 次上限，"
                "再发会被交易所限制交易，明天再来"
            )
        direction, offset = self._stock_side(symbol, direction, offset)
        # A 股没有市价单（交易所只有限价委托），天勤收到 `limit_price` 缺省会
        # 按「市价」发，柜台直接拒。在这里说清楚，别让人对着「委托被拒」猜
        if is_stock(symbol) and not (limit_price is not None and limit_price > 0):
            raise ValueError("A 股只有限价委托：填一个价格，不能下市价单")
        kwargs = {}
        if limit_price is not None and limit_price > 0:
            kwargs["limit_price"] = float(limit_price)
        if len(self._accs) > 1:
            kwargs["account"] = self._acc_obj(account)
        order = self.api.insert_order(
            symbol, direction, offset=offset, volume=int(volume), **kwargs
        )
        self._order_book[order.order_id] = order
        return order.order_id

    @staticmethod
    def _stock_side(symbol: str, direction: str, offset: str) -> tuple[str, str]:
        """A 股的方向 / 开平：`BUY→OPEN`、`SELL→CLOSE`，别的一律不收。

        股票没有「开空 / 平空」，也没有平今（T+1 只能平昨仓）。天勤的股票账户
        收到 `SELL + CLOSETODAY` 会直接拒单，报错信息还是一句英文 —— 不如在
        这里挡掉，顺带把语义讲清楚。
        """
        if not is_stock(symbol):
            return direction, offset
        d = str(direction or "").upper()
        if d not in ("BUY", "SELL"):
            raise ValueError(f"A 股只能 BUY / SELL，收到 {direction}")
        if d == "BUY":
            return "BUY", "OPEN"
        return "SELL", "CLOSE"

    @_serial
    def cancel(self, order_id: str, account: str = "") -> None:
        """撤单。同样是入队，下一次 `wait_update()` 才发。

        多账户下委托号全局唯一，但天勤要求撤单也指明账户实例 —— 传错了会被
        静默忽略（单子一直挂着），所以这里一并带上。
        """
        if self.api is None:
            raise RuntimeError("还没连接")
        if not self.limit.cancel():
            raise RuntimeError(
                f"今日撤单已达 {daylimit.DAY_CANCEL_LIMIT} 次上限，"
                "再撤会被交易所限制交易"
            )
        if len(self._accs) > 1:
            self.api.cancel_order(order_id, account=self._acc_obj(account))
        else:
            self.api.cancel_order(order_id)

    @_serial
    def set_target(self, symbol: str, volume: int, account: str = "") -> None:
        """把某个合约的目标净仓设成 `volume`（正多负空，0 平掉）。

        用 `TargetPosTask` 而不是自己 `insert_order`：反手要「先平后开」，
        上期所还要求区分平今/平仓，各交易所规则不一样，手写必然踩坑。
        TargetPosTask 按目标净仓自己拆单、自己追价、自己挑开平标志。

        同一个合约不要和 `place()` 混用 —— 两套逻辑同时管一个合约会打架。
        """
        if self.api is None:
            raise RuntimeError("还没连接")
        obj = self._acc_obj(account) if len(self._accs) > 1 else None
        key = f"{account}|{symbol}" if obj is not None else symbol
        task = self._targets.get(key)
        if task is None:
            from tqsdk import TargetPosTask

            task = TargetPosTask(self.api, symbol, account=obj) if obj is not None \
                else TargetPosTask(self.api, symbol)
            self._targets[key] = task
        task.set_target_volume(int(volume))

    # ---- 交易：`PortfolioRunner` 要的那套 tqsdk 同名接口 ----
    #
    # 组合页（`ttrade/strategy_runner.py`）以前自己 `TqApi(TqAccount(...))` 登一条，
    # 于是同一个天勤账户在系统里挂了两条会话：交易页一条、组合页一条，持仓和
    # 委托回报各推各的 —— 用户看到的就是「一键对齐发出去的单，监控器上不动」。
    # 现在两边认领**同一个 `LiveHub`**（`ttrade/hubreg.py`），
    # 所以这里必须把 runner 用到的那几个 `TqApi` 方法原样转出去：
    # `get_position` / `get_order` / `insert_order` / `cancel_order`。
    #
    # 全部走 `_api_call()`（持会话锁）—— 泵线程正在同一个 `TqApi` 上跑
    # `wait_update`，跨线程裸调会把内部订阅状态写坏。

    def get_position(self, *args, **kwargs):
        """`TqApi.get_position()` 的同一份活引用（账户 → `Position` 的 dict）。"""
        return self._api_call("get_position", *args, **kwargs)

    def get_order(self, *args, **kwargs):
        return self._api_call("get_order", *args, **kwargs)

    def get_trade(self, *args, **kwargs):
        return self._api_call("get_trade", *args, **kwargs)

    def get_account(self, *args, **kwargs):
        return self._api_call("get_account", *args, **kwargs)

    def insert_order(self, **kwargs):
        """下单。参数与 `TqApi.insert_order(...)` 完全一致。

        🔴 时间戳字段仍由泵线程的 `wait_update` 发出去 —— 这条会话任何时候都有
        泵在推（交易页关掉也一样，泵属于 hub 不属于调用方），所以不用担心
        「下完单没人推」。
        """
        return self._api_call("insert_order", **kwargs)

    def cancel_order(self, order=None, **kwargs):
        return self._api_call("cancel_order", order, **kwargs)

    def positions_ready(self) -> bool:
        """持仓**真从柜台读回来过**没有（`PortfolioRunner._pos_live()` 的判据）。

        🔴 空字典有两种含义：「账户真的没仓」和「第一帧还没到」。天勤在收到
        账户数据之前 `Account.balance` 是 `nan`，正好拿来区分：
        `_num()` 会把 nan 洗成 0，那就分不清了，所以这里直接读原属性判 nan。

        少了这道判据，「一键对齐」会在登录后头几秒拿「账户是空的」去算差额，
        把整个理论仓位重新开一遍（2026-11-23 融航那次就是这么来的，只是那边
        走的是柜台侧的 `positions_loaded`）。
        """
        if self.api is None or not self._accs:
            return False
        for _spec, obj in self._accs:
            try:
                a = obj.get_account()
            except Exception:  # noqa: BLE001
                return False
            if a is None:
                return False
            v = getattr(a, "balance", None)
            try:
                if not math.isfinite(float(v)):
                    return False
            except (TypeError, ValueError):
                return False
        return True
