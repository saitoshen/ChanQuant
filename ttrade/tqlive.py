"""天勤实时行情与交易。

一个 `TqApi` 同时承担三件事：实时行情、账户、下单。它们共享同一条连接 ——
分开建会重复登录、重复订阅，而且账户对象必须和下单在同一个 api 上才一致。

**线程模型**：`wait_update()` 会阻塞，必须放在独立线程里跑。所有 `get_*` 拿到的
都是「活引用」，随 `wait_update()` 就地更新；不要在循环里重复 `get_*`（每次
调用都会新建订阅对象，天勤会按符号去重但语义上仍然是浪费）。所以这里的做法是
`start()` 里一次性建好引用，`wait()` 推进，`snapshot()` 拷出普通 dict 给界面。

**时间戳**：K 线 `datetime` 是 **UTC 纳秒**，而 `DataDownloader` 写出的 CSV 是
北京时间，两者差 8 小时且不报错。统一走 `tqfetch.local_time()` 转北京时间，
与 DuckDB 里已落盘的历史对齐。

**复权**：实时段的 `adj_factor` 一律 1.0。原因是调用方先把本地补到「上一根完整
bar」，实时段全部落在当前主力合约的时间区间内，而后复权因子的最新段恒为 1
（见 `tqfetch.adj_factor_series`）。如果实时期间天勤换了主力，`underlying` 会变，
这时应该重新走一遍补历史 —— `underlying_changed` 会告诉界面。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import pandas as pd

from . import daylimit
from .paths import user_dir
from .tqdata import trading_days
from .tqfetch import PERIOD_SEC, local_time
from .tqsdk_symbols import main_symbol

# 实时 K 线向天勤要多少根。10000 是单次上限；5m 的 10000 根约 208 个交易日，
# 足够覆盖「界面开着放了一周」这种场景，再长的缺口应该走 update_main 补。
KLINE_LEN = 8000

ACCOUNT_KINDS = [
    ("kq", "快期模拟（天勤账户自带）"),
    ("sim", "本地模拟（TqSim，仅本机）"),
    ("account", "实盘（期货公司资金账号）"),
]


# ------------------------------------------------------------------ 账户配置


@dataclass
class AccountSpec:
    """一个交易账户的登录信息。

    - `kq`：快期模拟，用天勤账户登录即可交易，账户数据与快期客户端一致；
    - `sim`：`TqSim`，纯本地撮合，重启即清空，适合试接口；
    - `account`：实盘，需要期货公司的 broker_id + 资金账号 + 密码。

    `user` / `password` 是天勤账户（TqAuth），三种模式都要。
    """

    name: str = "默认"
    kind: str = "kq"
    user: str = ""
    password: str = ""
    broker_id: str = ""
    account_id: str = ""
    account_password: str = ""

    def build(self):
        """构造天勤的账户对象。实盘缺参数直接报错，不要静默退化成模拟。"""
        from tqsdk import TqAccount, TqKq, TqSim

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
        return TqKq()

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

    密码是明文落在用户目录（`%LOCALAPPDATA%\\TrendDesk\\accounts.json`）里的 ——
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


def merge_live(local: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """把实时 K 线接到本地历史后面。

    只接**严格晚于**本地最后一根的实时 bar。本地最后一根可能是下到一半的，
    重叠的 bar 一律保留本地那份 —— 本地那份已经带了正确的复权因子，而实时段的
    因子是 1.0，混着用会在拼接处产生一次假的缩放。
    """
    if live is None or live.empty:
        return local if local is not None else pd.DataFrame()
    if local is None or local.empty:
        add = live.copy()
        add["adj_factor"] = 1.0
        add["trading_day"] = trading_days(add.index).to_numpy()
        return add

    add = live[live.index > local.index[-1]].copy()
    if add.empty:
        return local
    add["adj_factor"] = 1.0
    add["trading_day"] = trading_days(add.index).to_numpy()
    for c in ("open_oi", "close_oi"):
        if c not in add.columns:
            add[c] = 0.0
    cols = [c for c in local.columns if c in add.columns]
    return pd.concat([local[cols], add[cols]]).sort_index()


# ------------------------------------------------------------------ 会话


# 委托 / 成交表最多显示多少条。一天下来「被拒的废单」能有几百上千条
# （拒单循环那种情况），全塞进 QTableWidget 之后，切到账户页第一帧要摆
# 几万个单元格，界面直接卡死。只留最近的，用户要看细节有委托号可查。
SNAP_ROWS = 200


@dataclass
class LiveState:
    """一次快照，界面拿去直接刷控件。全是普通 dict/list，不持有天勤对象。"""

    connected: bool = False
    error: str = ""
    key: str = ""
    period: str = ""
    underlying: str = ""
    quote: dict = field(default_factory=dict)
    account: dict = field(default_factory=dict)
    positions: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    trades: list = field(default_factory=list)


class LiveHub:
    """一个天勤连接：行情订阅 + 账户 + 下单。

    生命周期：`start()` → 反复 `wait()` → `close()`。`wait()` 之外的调用都只是
    读活引用，不会阻塞。所有异常都写进 `last_error`，不让界面线程炸掉。
    """

    def __init__(self, spec: AccountSpec):
        self.spec = spec
        self.api = None
        self.last_error = ""
        self.key = ""            # 品种_交易所
        self.period = ""
        self.tq_symbol = ""      # KQ.m@SHFE.rb
        self.underlying = ""     # 当前真实主力合约
        self._klines = None
        self._quote = None
        self._account = None
        self._positions = None
        self._orders = None
        self._trades = None
        self._order_book: dict = {}
        self._targets: dict = {}
        # 交易所对日内报单 / 撤单次数有硬上限。组合页的引擎和这里共用同一份
        # 落盘计数（同一个账户、同一个进程只能有一份）
        self.limit = daylimit.DayLimit(spec.label())

    # ---- 生命周期 ----

    def start(self) -> None:
        """建立连接并取好所有活引用。失败抛异常，由调用方决定怎么提示。"""
        from tqsdk import TqApi, TqAuth

        if not self.spec.user or not self.spec.password:
            raise ValueError("缺少天勤账户或密码")
        account = self.spec.build()
        self.api = TqApi(
            account,
            auth=TqAuth(self.spec.user, self.spec.password),
            web_gui=False,
        )
        self._account = self.api.get_account()
        self._positions = self.api.get_position()
        self._orders = self.api.get_order()
        self._trades = self.api.get_trade()

    def close(self) -> None:
        api, self.api = self.api, None
        self._klines = self._quote = None
        self._account = self._positions = self._orders = self._trades = None
        self._targets.clear()
        if api is not None:
            try:
                api.close()
            except Exception:
                pass

    @property
    def alive(self) -> bool:
        return self.api is not None

    # ---- 订阅 ----

    def subscribe(self, symbol: str, exchange: str, period: str) -> None:
        """订阅一个品种的主连 K 线与实时行情。

        重复订阅同一组参数是天勤允许的（同一个 serial 返回同一对象），但这里
        仍然记下当前 key，避免界面在没换品种时反复调用。
        """
        if self.api is None:
            raise RuntimeError("还没连接")
        if period not in PERIOD_SEC:
            raise ValueError(f"实时只支持 {', '.join(PERIOD_SEC)}，收到 {period}")
        key = f"{symbol}_{exchange}"
        if key == self.key and period == self.period and self._klines is not None:
            return
        tq_sym = main_symbol(symbol, exchange)
        self._quote = self.api.get_quote(tq_sym)
        self._klines = self.api.get_kline_serial(
            tq_sym, PERIOD_SEC[period], data_length=KLINE_LEN
        )
        self.key, self.period, self.tq_symbol = key, period, tq_sym
        self.underlying = ""

    # ---- 推进 ----

    def wait(self, timeout: float = 0.8) -> bool:
        """推进一次。返回 False 表示这段时间没有新数据（不是出错）。

        超时设成 0.8 秒：太短会变成忙轮询白烧 CPU，太长会让界面的「停止」按钮
        反应迟钝。
        """
        if self.api is None:
            return False
        try:
            return bool(self.api.wait_update(deadline=time.time() + timeout))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise

    # ---- 读 ----

    def bars(self) -> pd.DataFrame:
        return bars_from_serial(self._klines, self.period)

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
        st.account = self._account_snapshot()
        st.positions = self._positions_snapshot()
        st.orders = self._orders_snapshot()
        st.trades = self._trades_snapshot()
        return st

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
        # 主连的 underlying_symbol 就是当前真实主力；换月时会变
        und = getattr(q, "underlying_symbol", "") or ""
        if und and und != self.underlying:
            self.underlying = und
        return {
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

    def _account_snapshot(self) -> dict:
        a = self._account
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
        }

    def _positions_snapshot(self) -> list[dict]:
        src = self._positions
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
                "float_profit": self._num(p, "float_profit"),
                "position_profit": self._num(p, "position_profit"),
                "margin": self._num(p, "margin"),
                "last_price": self._num(p, "last_price"),
            })
        out.sort(key=lambda r: r["symbol"])
        return out

    def _orders_snapshot(self) -> list[dict]:
        src = self._orders
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
            })
        return out

    def _trades_snapshot(self) -> list[dict]:
        src = self._trades
        if src is None:
            return []
        items = list(src.items()) if hasattr(src, "items") else []
        items.sort(key=lambda kv: -self._num(kv[1], "trade_date_time"))
        out = []
        for tid, t in items[:SNAP_ROWS]:
            out.append({
                "trade_id": tid,
                "order_id": getattr(t, "order_id", ""),
                "symbol": getattr(t, "instrument_id", "") or getattr(t, "symbol", ""),
                "direction": getattr(t, "direction", ""),
                "offset": getattr(t, "offset", ""),
                "price": self._num(t, "price"),
                "volume": int(self._num(t, "volume")),
                "trade_date_time": self._num(t, "trade_date_time"),
            })
        return out

    # ---- 写 ----

    def place(self, symbol: str, direction: str, offset: str, volume: int,
              limit_price: float | None = None) -> str:
        """下单。返回客户端订单号。

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
        kwargs = {}
        if limit_price is not None and limit_price > 0:
            kwargs["limit_price"] = float(limit_price)
        order = self.api.insert_order(
            symbol, direction, offset=offset, volume=int(volume), **kwargs
        )
        self._order_book[order.order_id] = order
        return order.order_id

    def cancel(self, order_id: str) -> None:
        """撤单。同样是入队，下一次 `wait_update()` 才发。"""
        if self.api is None:
            raise RuntimeError("还没连接")
        if not self.limit.cancel():
            raise RuntimeError(
                f"今日撤单已达 {daylimit.DAY_CANCEL_LIMIT} 次上限，"
                "再撤会被交易所限制交易"
            )
        self.api.cancel_order(order_id)

    def set_target(self, symbol: str, volume: int) -> None:
        """把某个合约的目标净仓设成 `volume`（正多负空，0 平掉）。

        用 `TargetPosTask` 而不是自己 `insert_order`：反手要「先平后开」，
        上期所还要求区分平今/平仓，各交易所规则不一样，手写必然踩坑。
        TargetPosTask 按目标净仓自己拆单、自己追价、自己挑开平标志。

        同一个合约不要和 `place()` 混用 —— 两套逻辑同时管一个合约会打架。
        """
        if self.api is None:
            raise RuntimeError("还没连接")
        task = self._targets.get(symbol)
        if task is None:
            from tqsdk import TargetPosTask

            task = TargetPosTask(self.api, symbol)
            self._targets[symbol] = task
        task.set_target_volume(int(volume))
