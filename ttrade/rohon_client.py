"""融航资管中台（vnpy_rohon）：行情 + 交易。

**已停用（2026-11-19）** —— 现在走融航官方的「CTP 兼容 dll」路线，见
`ttrade/ctp_channel.py` 和 `ttrade/ctp_live.RohonHub`：把融航给的
`thosttraderapi_se.dll` 配 openctp-ctp 6.7.2.8，用标准 CTP 接口直接登录，
不再依赖 `vnpy_rohon`（那包只出 cp310/cp313 轮子，本机 base 是 3.14 装不上）。

这个文件保留，是因为它是 vnpy_rohon 路线的完整实现（行情 + 交易 + 字段映射），
哪天拿到 cp314 的轮子可以直接接回来。**它不再被任何代码引入**。

**为什么不装 vnpy**：`vnpy_rohon` 的包入口（`vnpy_rohon/__init__.py`）会
`from .gateway import RohonGateway`，而那个 gateway 要 `import vnpy`；vnpy 4.5.0
把 `pyside6==6.8.2.1` 钉成**硬依赖**，本程序锁的是 6.11.3（见 requirements.txt
里那两个平台坑）—— 为了一个柜台去降 Qt，代价是整个界面。

好消息是 `vnpy_rohon.api` 只是一层 pybind11 封装（`vnrohontd` / `vnrohonmd`
两个 .pyd + 一张常量表 + 几个 dll），**不 import vnpy**。所以这里直接绕过包入口
取子模块，等于只用了 vnpy 生态里的「驱动层」，不用它的框架层。

**和 `ctp_client.py` 的关系**：融航本身就是 CTP 协议的中台，回调字段、方向/开平
编码、委托状态字符全都一样。所以这里刻意做成同一套接口
（`start/wait_ready/place/cancel/query_*/snapshot/last`），让 `ctp_live.CtpHub`
一套账户解析能吃两家柜台，不用把「持仓怎么合并、上期所今昨仓怎么分」再写一遍。

**一个必须记住的差异**：`vnpy_rohon.api` 的回调**传 dict**，不是结构体（openctp
那边是结构体 + `getattr`）。参数名也是小写 `on` 开头的 CTP 原名（`onRtnOrder`、
`onRspUserLogin`），和 vnpy gateway 里逐字一致 —— 这些名字不能改，pybind 的
trampoline 是按名字找 Python 方法的，改了就是「回调静默不触发」。
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types
from dataclasses import dataclass

from .paths import user_dir

ROHON_IMPORT_ERROR = ""


def _import_api():
    """取 `vnpy_rohon.api` 的 (MdApi, TdApi)。

    先按正常方式 import；失败（没装、或者装了但 vnpy 装不上）再走「绕过包入口」
    的退路 —— 理由见模块开头的说明。
    """
    try:
        from vnpy_rohon.api import MdApi, TdApi

        return MdApi, TdApi
    except Exception:  # noqa: BLE001 - 包入口会抛 ModuleNotFoundError('vnpy')
        pass
    import importlib.util

    spec = importlib.util.find_spec("vnpy_rohon")
    if spec is None:
        raise ImportError("没装 vnpy_rohon")
    # 只用 `spec is None` 判「没装」。别加 `not spec.submodule_search_locations`
    # —— 打包后（PyInstaller 把包收进 PYZ）find_spec 查得到但不一定给搜索路径，
    # 那样写会让**发布包**永远提示「没装 vnpy_rohon」，而源码环境一切正常。
    stub = types.ModuleType("vnpy_rohon")
    stub.__path__ = list(spec.submodule_search_locations or [])
    # 必须留在 sys.modules 里：撤掉的话，`from vnpy_rohon.api import ...` 会重新
    # 去执行那个带 `import vnpy` 的包入口，又回到原来的问题
    sys.modules.setdefault("vnpy_rohon", stub)
    from vnpy_rohon.api import MdApi, TdApi

    return MdApi, TdApi


try:
    _MdApi, _TdApi = _import_api()
    from vnpy_rohon.api import (
        THOST_FTDC_AF_Delete,
        THOST_FTDC_CC_Immediately,
        THOST_FTDC_D_Buy,
        THOST_FTDC_D_Sell,
        THOST_FTDC_FCC_NotForceClose,
        THOST_FTDC_HF_Speculation,
        THOST_FTDC_OPT_LimitPrice,
        THOST_FTDC_OF_Open,
        THOST_FTDC_OFEN_Close,
        THOST_FTDC_OFEN_CloseToday,
        THOST_FTDC_OFEN_CloseYesterday,
        THOST_FTDC_PD_Long,
        THOST_FTDC_PD_Short,
        THOST_FTDC_TC_GFD,
        THOST_FTDC_VC_AV,
    )

    # 方向 / 开平：我们自己的写法 ↔ 柜台编码。柜台那边是**字符串**（'0' / '1' /
    # '3'），不是整数 —— 和 openctp 的 CTP 一致，别按 int 去比
    DIR_VT2ROHON = {"BUY": THOST_FTDC_D_Buy, "SELL": THOST_FTDC_D_Sell}
    DIR_ROHON2VT = {THOST_FTDC_D_Buy: "BUY", THOST_FTDC_D_Sell: "SELL",
                    THOST_FTDC_PD_Long: "BUY", THOST_FTDC_PD_Short: "SELL"}
    OFFSET_VT2ROHON = {"OPEN": THOST_FTDC_OF_Open,
                       "CLOSE": THOST_FTDC_OFEN_Close,
                       "CLOSETODAY": THOST_FTDC_OFEN_CloseToday}
    OFFSET_ROHON2VT = {THOST_FTDC_OF_Open: "OPEN",
                       THOST_FTDC_OFEN_Close: "CLOSE",
                       THOST_FTDC_OFEN_CloseToday: "CLOSETODAY",
                       THOST_FTDC_OFEN_CloseYesterday: "CLOSE"}
    HAS_ROHON = True
except Exception as exc:  # noqa: BLE001
    HAS_ROHON = False
    ROHON_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    _MdApi = _TdApi = object
    # 没装也要能 import 这个模块（自检要照实报「没装」，不是整个程序起不来）；
    # 空表在用到的地方会先被 HAS_ROHON 挡住
    DIR_VT2ROHON = {}
    DIR_ROHON2VT = {}
    OFFSET_VT2ROHON = {}
    OFFSET_ROHON2VT = {}

# 市价单要的两个枚举。**故意不放进上面那个大 try**：万一某个版本的
# vnpy_rohon 没导出它们，整个模块会变成「没装」，融航就连不上了。CTP 协议里
# 这两个值固定是 `'1'`（AnyPrice）/ `'3'`（IOC），拿不到就退回字面量。
THOST_FTDC_OPT_AnyPrice = "1"
THOST_FTDC_TC_IOC = "3"
try:
    from vnpy_rohon.api import (
        THOST_FTDC_OPT_AnyPrice as _OPT_ANY,
        THOST_FTDC_TC_IOC as _TC_IOC,
    )

    THOST_FTDC_OPT_AnyPrice, THOST_FTDC_TC_IOC = _OPT_ANY, _TC_IOC
except Exception:  # noqa: BLE001
    pass          # 拿不到就用上面那两个字面量，和市场价单没关系

# 委托状态字符 → 人话。**和 CTP 是同一张表**（两者都是 CTP 协议的 OrderStatus），
# 直接借 ctp_client 那份。那张表曾经整体错位一格：界面把活单显示成「已撤单」，
# 用户以为单子没了。两份表必然只会改一份，所以不复制。
from .ctp_client import _STATUS as STATUS_TEXT  # noqa: E402

# 浮点极限值。融航（和 CTP 一样）用 DBL_MAX 表示「这一档/这个字段没有值」，
# 直接显示会变成 1.7976931348623157e+308
_MAXF = sys.float_info.max


def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    return str(v)


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if x == _MAXF or x >= 1e300 or x != x else x


def _err(e) -> str:
    """回调里的 error 是一个 dict（`{"ErrorID":..., "ErrorMsg":...}`）。"""
    if not e:
        return ""
    try:
        code = int(e.get("ErrorID") or 0)
    except (TypeError, ValueError):
        code = 0
    if not code:
        return ""
    return f"[{code}] {_s(e.get('ErrorMsg'))}"


class RohonError(RuntimeError):
    """连接 / 登录 / 下单失败。带融航原始错误码，便于查表。"""


@dataclass
class RohonConfig:
    """一条融航账户的登录参数。字段和 `AccountSpec` 一一对应。"""

    name: str = ""
    user_id: str = ""
    password: str = ""
    broker_id: str = ""
    td_front: str = ""
    md_front: str = ""
    app_id: str = ""
    auth_code: str = ""

    def label(self) -> str:
        return f"{self.name} · 融航"


def _addr(text: str) -> str:
    """前置地址。柜台只认 `tcp://host:port`，用户通常只填 host:port。"""
    s = str(text or "").strip()
    if s and not s.startswith("tcp://"):
        s = "tcp://" + s
    return s


def spec_to_config(spec) -> RohonConfig:
    return RohonConfig(
        name=getattr(spec, "name", "") or "融航",
        user_id=getattr(spec, "user", "") or "",
        password=getattr(spec, "password", "") or "",
        broker_id=getattr(spec, "broker_id", "") or "",
        td_front=_addr(getattr(spec, "td_front", "")),
        md_front=_addr(getattr(spec, "md_front", "")),
        app_id=getattr(spec, "app_id", "") or "",
        auth_code=getattr(spec, "auth_code", "") or "",
    )


def _flow_dir() -> str:
    """流文件目录。**必须存在且可写**，否则 init() 直接崩。"""
    path = os.path.join(user_dir(), "rohon_flow")
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------ 行情


class RohonMdClient:
    """行情：连接 → 登录 → 订阅 → 回调推 tick。

    用法和 `ctp_client.MdClient` 完全一致：

        md = RohonMdClient(cfg)
        md.start(["rb2601"])
        md.wait_ready(10)
        md.last("rb2601")
    """

    def __init__(self, cfg: RohonConfig) -> None:
        self.cfg = cfg
        self.api = None
        self.ready = threading.Event()
        self.last_error = ""
        self.ticks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._instruments: list[str] = []
        self._req = 0

    def start(self, instruments=()) -> None:
        if not HAS_ROHON:
            raise RohonError(f"没装 vnpy_rohon，用不了融航（{ROHON_IMPORT_ERROR}）")
        if not self.cfg.md_front:
            raise RohonError("融航行情服务器地址为空")
        self._instruments = list(instruments or ())
        api = _make_md(self)
        flow = os.path.join(_flow_dir(), f"md_{int(time.time())}")
        api.createFtdcMdApi(flow)
        api.registerFront(self.cfg.md_front)
        api.init()
        self.api = api

    def stop(self) -> None:
        api, self.api = self.api, None
        self.ready.clear()
        if api is not None:
            try:
                api.exit()
            except Exception:  # noqa: BLE001
                pass

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self.ready.wait(timeout)

    def last(self, instrument: str) -> dict:
        with self._lock:
            return dict(self.ticks.get(instrument) or {})

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self.ticks.items()}

    def subscribe(self, instruments) -> None:
        items = [str(x) for x in (instruments or []) if str(x)]
        if not items:
            return
        self._instruments = items
        if self.api is not None and self.ready.is_set():
            self._send_sub(items)

    def _send_sub(self, items) -> None:
        """逐个合约订阅。

        `subscribeMarketData` 的签名是**一个** `str`（不是 CTP 那种 list），
        传 list 会 TypeError。vnpy 的融航 gateway 也是一次一个。
        """
        for ins in items:
            try:
                self.api.subscribeMarketData(ins)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"订阅 {ins} 失败：{exc}"

    def _next_req(self) -> int:
        self._req += 1
        return self._req


def _make_md(outer: RohonMdClient):
    class _Md(_MdApi):
        def onFrontConnected(self):  # noqa: N802 - 名字由 pybind 按 CTP 原名查找
            try:
                outer.api.reqUserLogin(
                    {
                        "UserID": outer.cfg.user_id,
                        "Password": outer.cfg.password,
                        "BrokerID": outer.cfg.broker_id,
                    },
                    outer._next_req(),
                )
            except Exception as exc:  # noqa: BLE001
                outer.last_error = f"行情登录请求失败：{exc}"

        def onFrontDisconnected(self, reason):  # noqa: N802
            outer.last_error = f"融航行情前置断开（{reason}）"
            outer.ready.clear()

        def onRspUserLogin(self, data, error, reqid, last):  # noqa: N802
            err = _err(error)
            if err:
                outer.last_error = f"行情登录失败：{err}"
                return
            outer.ready.set()
            if outer._instruments:
                outer._send_sub(outer._instruments)

        def onRspError(self, error, reqid, last):  # noqa: N802
            err = _err(error)
            if err:
                outer.last_error = f"行情接口报错：{err}"

        def onRspSubMarketData(self, data, error, reqid, last):  # noqa: N802
            err = _err(error)
            if err:
                outer.last_error = f"订阅失败：{err}"

        def onRtnDepthMarketData(self, data):  # noqa: N802
            if not data:
                return
            ins = _s(data.get("InstrumentID"))
            if not ins or not _s(data.get("UpdateTime")):
                return
            # 日期口径：大商所 / 广期所的 ActionDay 在夜盘是错的，用本机日期
            day = _s(data.get("ActionDay"))
            if not day:
                day = time.strftime("%Y%m%d")
            with outer._lock:
                outer.ticks[ins] = {
                    "instrument": ins,
                    "datetime": f"{day} {_s(data.get('UpdateTime'))}",
                    "last": _f(data.get("LastPrice")),
                    "bid": _f(data.get("BidPrice1")),
                    "ask": _f(data.get("AskPrice1")),
                    "bid_vol": _f(data.get("BidVolume1")),
                    "ask_vol": _f(data.get("AskVolume1")),
                    "open": _f(data.get("OpenPrice")),
                    "high": _f(data.get("HighestPrice")),
                    "low": _f(data.get("LowestPrice")),
                    "pre_close": _f(data.get("PreClosePrice")),
                    "volume": _f(data.get("Volume")),
                    "oi": _f(data.get("OpenInterest")),
                }

    return _Md()


# ------------------------------------------------------------------ 交易


class RohonTraderClient:
    """交易：认证 → 登录 → 结算确认 → 下单 / 撤单 / 查持仓。

    对外接口和 `ctp_client.TraderClient` 一模一样（`start/wait_ready/place/
    cancel/query_*/snapshot`，以及 `snapshot()` 里那几张同样的 dict），
    这样 `ctp_live.CtpHub` 不用为融航写第二套账户解析。

    **下单前必须确认结算结果**，否则柜台拒单，理由只写「未确认结算」。
    """

    def __init__(self, cfg: RohonConfig) -> None:
        self.cfg = cfg
        self.api = None
        self.connected = threading.Event()
        self.logged_in = threading.Event()
        self.settled = threading.Event()
        self.last_error = ""
        self.orders: dict[str, dict] = {}
        self.trades: dict[str, dict] = {}
        self.positions: dict[str, dict] = {}
        # 有没有**完整**读回来过一轮持仓（`bIsLast` 那一包到了才算）。
        # 🔴 和 CTP 上那份同一个语义（见 `ctp_client` 里 2026-11-23 融航那次）：
        # 「空表」既可能是真没仓，也可能是柜台还没回 —— 后者不能拿去做对齐。
        self.positions_loaded = False
        # 🔴 持仓查询的「代」：柜台一秒一个查询，上一轮的回包可能迟到现在才到；
        # 不认 RequestID 的话，它的 `last` 会把新一轮当成自己的整份快照提交。
        # 2026-11-23 那些「交易页有 7 个合约、监控器全是 0」就是这一路丢的。
        # 语义和 `ctp_client.TraderClient` 完全一致：**开轮清一次、之后原地累加**，
        # 不再攒 buf 整份替换（替换版一旦被多余的 `last` 空包碰上就把表刷成空的）。
        self._pos_req = 0             # 当前这一轮查询的 RequestID
        self._pos_started = True      # True = 没有进行中的查询（下一轮要清一次）
        self._pos_rows = 0            # 最近一轮回包的行数，只用于诊断
        self.account: dict = {}
        self.instruments: dict[str, dict] = {}
        self.instruments_done = threading.Event()
        # 合约 → 交易所的兜底解析器（`CtpHub` 会塞 `variety_exchange` 进来）
        self.exchange_resolver = None
        self.front_id = 0
        self.session_id = 0
        self._lock = threading.Lock()
        self._req = 0
        self._order_ref = int(time.time()) % 100000

    # ---- 生命周期 ----

    def start(self) -> None:
        if not HAS_ROHON:
            raise RohonError(f"没装 vnpy_rohon，用不了融航（{ROHON_IMPORT_ERROR}）")
        if not self.cfg.td_front:
            raise RohonError("融航交易服务器地址为空")
        api = _make_td(self)
        flow = os.path.join(_flow_dir(), f"td_{int(time.time())}")
        api.createFtdcTraderApi(flow)
        # 0 = THOST_TERT_RESTART；vnpy 的融航 gateway 用的也是 0，跟它保持一致，
        # 换成 QUICK 只会多收一堆重复回报
        api.subscribePrivateTopic(0)
        api.subscribePublicTopic(0)
        api.registerFront(self.cfg.td_front)
        api.init()
        self.api = api

    def stop(self) -> None:
        api, self.api = self.api, None
        for ev in (self.connected, self.logged_in, self.settled):
            ev.clear()
        if api is not None:
            try:
                api.exit()
            except Exception:  # noqa: BLE001
                pass

    def wait_ready(self, timeout: float = 15.0) -> bool:
        """等到「能下单」为止：登录 + 结算确认都完成。"""
        return self.settled.wait(timeout)

    # ---- 下单 ----

    def place(self, instrument: str, direction: str, offset: str,
              price: float, volume: int, market: bool = False) -> str:
        """报单。返回本地 OrderRef。

        `market=True` 发**市价单**（`OrderPriceType=AnyPrice`，价格填 0）。

        🔴 2026-11-23 用户：「ctp 和融航的下单，现在所有交易所应该都支持市价单了
        … 上期所和能源所在 2026 年 7 月 6 日也开放支持市价单了」。以前这里硬编码
        「融航席位只收限价单」，界面上市价那个勾对柜台账户直接禁用。
        时间条件必须跟着换成 **IOC**：GFD + AnyPrice 会被交易所打回「该合约不支持
        此时间条件」（和 `ctp_client.place()` 同一条规则）。
        """
        if self.api is None or not self.logged_in.is_set():
            raise RohonError("还没登录")
        if not self.settled.is_set():
            raise RohonError("还没确认结算结果，柜台会拒单")
        d = DIR_VT2ROHON.get(str(direction).upper())
        if d is None:
            raise RohonError(f"方向只能是 BUY / SELL，收到 {direction}")
        off = OFFSET_VT2ROHON.get(str(offset).upper())
        if off is None:
            raise RohonError(f"开平只能是 OPEN / CLOSE / CLOSETODAY，收到 {offset}")
        if volume <= 0:
            raise RohonError("手数必须大于 0")
        # `price` 允许 None / NaN（天勤那边就这么表示市价），统一折成市价
        try:
            px = float(price)
        except (TypeError, ValueError):
            px = float("nan")
        market = bool(market) or not (px > 0)
        if not market and px <= 0:
            raise RohonError("限价单必须给一个大于 0 的价格")

        self._order_ref += 1
        ref = str(self._order_ref)
        req = {
            "InstrumentID": instrument,
            "ExchangeID": self._exchange_of(instrument),
            "LimitPrice": 0.0 if market else px,
            "VolumeTotalOriginal": int(volume),
            "OrderPriceType": (THOST_FTDC_OPT_AnyPrice if market
                               else THOST_FTDC_OPT_LimitPrice),
            "Direction": d,
            "CombOffsetFlag": off,
            # 投机单。套保 / 套利要在柜台提前备案
            "CombHedgeFlag": THOST_FTDC_HF_Speculation,
            "OrderRef": ref,
            "InvestorID": self.cfg.user_id,
            "UserID": self.cfg.user_id,
            "BrokerID": self.cfg.broker_id,
            "ContingentCondition": THOST_FTDC_CC_Immediately,
            "ForceCloseReason": THOST_FTDC_FCC_NotForceClose,
            "IsAutoSuspend": 0,
            "TimeCondition": THOST_FTDC_TC_IOC if market else THOST_FTDC_TC_GFD,
            "VolumeCondition": THOST_FTDC_VC_AV,
            "MinVolume": 1,
        }
        try:
            n = self.api.reqOrderInsert(req, self._next_req())
        except Exception as exc:  # noqa: BLE001 - pybind 参数错会抛
            raise RohonError(f"报单请求发送失败：{exc}") from exc
        if n:
            raise RohonError(f"报单请求发送失败，错误代码：{n}")
        return ref

    def cancel(self, instrument: str, order_sys_id: str = "",
               order_ref: str = "") -> None:
        """撤单。有 `OrderSysID` 用它（柜台单号），没有就用本地 `OrderRef`。"""
        if self.api is None:
            raise RohonError("还没连接")
        req = {
            "InstrumentID": instrument,
            "ExchangeID": self._exchange_of(instrument),
            "ActionFlag": THOST_FTDC_AF_Delete,
            "BrokerID": self.cfg.broker_id,
            "InvestorID": self.cfg.user_id,
        }
        if order_sys_id:
            req["OrderSysID"] = order_sys_id
        elif order_ref:
            req["OrderRef"] = order_ref
            req["FrontID"] = self.front_id
            req["SessionID"] = self.session_id
        else:
            raise RohonError("撤单缺少 OrderSysID 和 OrderRef")
        # 和 `place()` 一样要看返回值：非 0 = 请求没进发送队列，那笔单还在
        # 交易所挂着，而调用方会按「已撤」处理（2026-11-23 融航那四笔）
        try:
            n = self.api.reqOrderAction(req, self._next_req())
        except Exception as exc:  # noqa: BLE001 - pybind 参数错会抛
            raise RohonError(f"撤单请求发送失败：{exc}") from exc
        if n:
            raise RohonError(f"撤单没发出去，错误代码：{n}")

    # ---- 查询 ----

    def _qry_note(self, what: str, n) -> None:
        """`reqQry*` 的返回值翻成人话（和 `ctp_client._qry_note` 同一套判据）。

        融航中台也是 CTP 那套「每秒一个查询」，超了直接返回非 0 且**不回包**。
        不看返回值的话，「没发出去」和「账户真没仓」在界面上长得一模一样 ——
        2026-11-23 组合页那条连接就是连发资金 + 持仓，持仓被丢掉、`_pos` 全空。
        """
        try:
            code = int(n or 0)
        except (TypeError, ValueError):
            code = 0
        old = str(getattr(self, "last_error", "") or "")
        if code:
            why = {-1: "网络连接失败", -2: "未处理请求超上限",
                   -3: "每秒查询超限（柜台一秒只收一个）"}.get(code, "柜台拒绝")
            self.last_error = f"查{what}没发出去：{code} {why}"
        elif old.startswith(f"查{what}没发出去"):
            self.last_error = ""

    def query_account(self) -> None:
        if self.api is not None:
            # vnpy 的融航 gateway 也是传空 dict：账户信息从登录态取
            self._qry_note("资金", self.api.reqQryTradingAccount({}, self._next_req()))

    def query_positions(self) -> None:
        if self.api is None:
            return
        # 记下这一轮的 RequestID，回包只认它；同时把「开轮清表」的闸门放下，
        # 由这一轮的第一条回包负责清（不能在这里清：清早了界面会看到半份持仓）
        rid = self._next_req()
        self._pos_req = rid
        self._pos_started = False
        self._pos_rows = 0
        self._qry_note("持仓", self.api.reqQryInvestorPosition(
            {"BrokerID": self.cfg.broker_id, "InvestorID": self.cfg.user_id},
            rid,
        ))

    def query_instruments(self, instrument: str = "") -> None:
        """查合约表。传 `instrument` 只查那一个（秒回）。

        和 CTP 一样**不查全量**：融航中台下挂的是资管户，全量合约流一旦开始，
        这个会话的报单和查资金全都排在它后面，界面表现就是「点下单没反应」。
        """
        if self.api is None:
            return
        req = {"InstrumentID": instrument} if instrument else {}
        if not instrument:
            self.instruments_done.clear()
        self.api.reqQryInstrument(req, self._next_req())

    # ---- 读 ----

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "connected": self.logged_in.is_set(),
                "settled": self.settled.is_set(),
                "error": self.last_error,
                "account": dict(self.account),
                "positions": {k: dict(v) for k, v in self.positions.items()},
                # 「持仓真读回来了吗」—— 空表不等于没仓（`BrokerApi.positions_ready`）
                "positions_loaded": self.positions_loaded,
                # 最近一轮持仓查询**回了几行**。0 行 + `loaded=True` 只有两种可能：
                # 真没仓，或者柜台回了个空包 —— 诊断行里带上它才分得清
                "pos_rows": self._pos_rows,
                "orders": {k: dict(v) for k, v in self.orders.items()},
                "trades": {k: dict(v) for k, v in self.trades.items()},
            }

    # ---- 内部 ----

    def _next_req(self) -> int:
        self._req += 1
        return self._req

    def _exchange_of(self, instrument: str) -> str:
        """合约 → 交易所。按「查表 → 外部解析器 → 品种前缀猜」三级兜底。

        撤单必须带上正确交易所，否则柜台返回「交易所不存在」。和 CTP 那份是同
        一个办法，但那边的实现绑在 `TraderClient` 上，这里照抄会变成两份判据 ——
        所以直接复用 `ctp_live.variety_exchange`（它本来就是纯品种前缀表）。
        """
        ins = str(instrument or "")
        info = self.instruments.get(ins) or {}
        ex = _s(info.get("exchange"))
        if ex:
            return ex
        if self.exchange_resolver is not None:
            try:
                ex = _s(self.exchange_resolver(ins))
            except Exception:  # noqa: BLE001
                ex = ""
            if ex:
                return ex
        from .ctp_live import variety_exchange

        return variety_exchange("".join(ch for ch in ins if ch.isalpha()))


def _make_td(outer: RohonTraderClient):
    """把回调接成 `RohonTraderClient` 的属性更新。"""

    class _Td(_TdApi):
        # ---- 连接 / 认证 / 登录 ----

        def onFrontConnected(self):  # noqa: N802
            outer.connected.set()
            try:
                if outer.cfg.auth_code:
                    outer.api.reqAuthenticate(
                        {
                            "UserID": outer.cfg.user_id,
                            "BrokerID": outer.cfg.broker_id,
                            "AuthCode": outer.cfg.auth_code,
                            "AppID": outer.cfg.app_id,
                        },
                        outer._next_req(),
                    )
                else:
                    _login(outer)
            except Exception as exc:  # noqa: BLE001
                outer.last_error = f"认证请求失败：{exc}"

        def onFrontDisconnected(self, reason):  # noqa: N802
            outer.last_error = f"融航交易前置断开（{reason}）"
            outer.connected.clear()
            outer.logged_in.clear()
            outer.settled.clear()

        def onRspAuthenticate(self, data, error, reqid, last):  # noqa: N802
            err = _err(error)
            if err:
                outer.last_error = f"认证失败：{err}"
                return
            try:
                _login(outer)
            except Exception as exc:  # noqa: BLE001
                outer.last_error = f"登录请求失败：{exc}"

        def onRspUserLogin(self, data, error, reqid, last):  # noqa: N802
            err = _err(error)
            if err:
                outer.last_error = f"登录失败：{err}"
                return
            outer.front_id = int((data or {}).get("FrontID") or 0)
            outer.session_id = int((data or {}).get("SessionID") or 0)
            outer.logged_in.set()
            try:
                outer.api.reqSettlementInfoConfirm(
                    {"BrokerID": outer.cfg.broker_id,
                     "InvestorID": outer.cfg.user_id},
                    outer._next_req(),
                )
            except Exception as exc:  # noqa: BLE001
                outer.last_error = f"结算确认请求失败：{exc}"

        def onRspSettlementInfoConfirm(self, data, error, reqid, last):  # noqa: N802
            err = _err(error)
            if err:
                outer.last_error = f"结算确认失败：{err}"
                return
            outer.settled.set()

        # ---- 报单 / 成交 ----

        def onRspOrderInsert(self, data, error, reqid, last):  # noqa: N802
            d = data or {}
            ref = _s(d.get("OrderRef"))
            err = _err(error)
            with outer._lock:
                outer.orders[ref] = {
                    "order_ref": ref,
                    "order_sys_id": "",
                    "instrument": _s(d.get("InstrumentID")),
                    "direction": DIR_ROHON2VT.get(_s(d.get("Direction")), ""),
                    "offset": OFFSET_ROHON2VT.get(_s(d.get("CombOffsetFlag")), ""),
                    "price": _f(d.get("LimitPrice")),
                    "volume": int(_f(d.get("VolumeTotalOriginal"))),
                    "traded": 0,
                    "status": "被拒" if err else "已提交",
                    "msg": err,
                }
            if err:
                outer.last_error = f"下单被拒：{err}"

        def onRtnOrder(self, data):  # noqa: N802
            if not data:
                return
            status = _s(data.get("OrderStatus"))
            # 融航会把状态为「未知」的委托也推一遍。这不是真单子，落进委托表
            # 只会多一行永远不动的「未知」
            if status == "a":
                return
            sys_id = _s(data.get("OrderSysID"))
            ref = _s(data.get("OrderRef"))
            # FrontID / SessionID 是整数，0 也得退回本地缓存的那一份（`_s(0)`
            # 会得到 "0"，直接 `or` 判不出来）
            front = _s(data.get("FrontID") or outer.front_id)
            sess = _s(data.get("SessionID") or outer.session_id)
            key = sys_id or f"{front}_{sess}_{ref}"
            row = {
                "order_ref": ref,
                "order_sys_id": sys_id,
                "instrument": _s(data.get("InstrumentID")),
                "direction": DIR_ROHON2VT.get(_s(data.get("Direction")), ""),
                "offset": OFFSET_ROHON2VT.get(_s(data.get("CombOffsetFlag")), ""),
                "price": _f(data.get("LimitPrice")),
                "volume": int(_f(data.get("VolumeTotalOriginal"))),
                "traded": int(_f(data.get("VolumeTraded"))),
                "status": STATUS_TEXT.get(status, "未知"),
                "msg": _s(data.get("StatusMsg")),
            }
            with outer._lock:
                outer.orders[key] = row
                if ref and ref != key:
                    outer.orders[ref] = dict(row)

        def onRtnTrade(self, data):  # noqa: N802
            if not data:
                return
            tid = _s(data.get("TradeID")) or str(time.time())
            row = {
                "trade_id": tid,
                "order_ref": _s(data.get("OrderRef")),
                "instrument": _s(data.get("InstrumentID")),
                "direction": DIR_ROHON2VT.get(_s(data.get("Direction")), ""),
                "offset": OFFSET_ROHON2VT.get(_s(data.get("OffsetFlag")), ""),
                "price": _f(data.get("Price")),
                "volume": int(_f(data.get("Volume"))),
                "time": _s(data.get("TradeTime")),
            }
            with outer._lock:
                outer.trades[tid] = row

        def onRspOrderAction(self, data, error, reqid, last):  # noqa: N802
            err = _err(error)
            if err:
                outer.last_error = f"撤单失败：{err}"
                return
            ref = _s((data or {}).get("OrderRef"))
            with outer._lock:
                for o in outer.orders.values():
                    if o.get("order_ref") == ref:
                        o["status"] = "已撤单"

        # ---- 查询回报 ----

        def onRspQryTradingAccount(self, data, error, reqid, last):  # noqa: N802
            if not data or "AccountID" not in data:
                return
            with outer._lock:
                outer.account = {
                    "balance": _f(data.get("Balance")),
                    "available": _f(data.get("Available")),
                    "margin": _f(data.get("CurrMargin")),
                    "float_profit": _f(data.get("PositionProfit")),
                    "position_profit": _f(data.get("PositionProfit")),
                    "close_profit": _f(data.get("CloseProfit")),
                    "commission": _f(data.get("Commission")),
                    # 昨结算权益：当日盈亏 = 动态权益 − 昨结算。Balance 已含浮盈
                    "pre_balance": _f(data.get("PreBalance")),
                }

        def onRspQryInvestorPosition(self, data, error, reqid, last):  # noqa: N802
            """查持仓回包。**开轮清一次、之后原地累加**，和 `ctp_client` 同一套。

            🔴 三个坑，前两个是 2026-11-23 下午「交易页有 7 个合约、监控器全 0」
            的真因：

            1. **没有按 RequestID 归轮**。柜台一秒一个查询，上一轮的回包可能迟到；
               它的 `last` 一到，就把新一轮刚攒的东西当成自己的整份快照提交了。
            2. **`outer._pos_buf or {}` 会把表刷成空**。攒 buf、`last` 时整份替换
               的写法，一旦碰上多余的 `last`（有的中台会多发一个空终止包），
               `buf` 已经是 None ⇒ 提交一个空 dict ⇒ 持仓全变 0，还顺手把
               `positions_loaded` 置真 —— 于是连「还没读到」都不说了，界面直接画 0。
            3. `profit` / `margin` 早先用 `+=` 且从不重置（问第二遍就翻倍），
               平掉的仓也永不删。改成开轮清一次，两个问题一起没了。

            还有一条：上期所 / 能源中心的持仓回包**同一方向会有两行**（今仓、昨仓
            各一行），所以六个数一律 `+=`，不能赋值 —— 赋值的话后一行把前一行
            盖掉，持仓手数直接少一半。
            """
            err = _err(error)
            if err:
                outer.last_error = f"查持仓失败：{err}"
                return
            # 只认当前这一轮。`reqid` 为 0（个别中台不回填）时不拦，宁可多收
            try:
                rid = int(reqid or 0)
            except (TypeError, ValueError):
                rid = 0
            if rid and outer._pos_req and rid != outer._pos_req:
                return
            with outer._lock:
                if not outer._pos_started:
                    outer.positions.clear()
                    outer._pos_started = True
                if not data:
                    # 空仓时柜台只回一条空包（`last=True`、内容为空）。「查完了、
                    # 确实一条持仓都没有」是有效结论，必须置真，否则界面永远停在
                    # 「还没读到」，真有仓了也不敢说一致
                    if last:
                        outer.positions_loaded = True
                        outer._pos_rows = len(outer.positions)
                    return
                ins = _s(data.get("InstrumentID"))
                side = {"2": "long", "3": "short"}.get(
                    _s(data.get("PosiDirection")), "net")
                n = int(_f(data.get("Position")))
                yd = int(_f(data.get("YdPosition")))
                td = int(_f(data.get("TodayPosition"))) or max(0, n - yd)
                row = outer.positions.setdefault(
                    ins, {"instrument": ins, "long": 0, "short": 0,
                          "long_yd": 0, "short_yd": 0,
                          "long_today": 0, "short_today": 0,
                          "cost_long": 0.0, "cost_short": 0.0,
                          "price": 0.0, "profit": 0.0, "margin": 0.0})
                # 一律累加，不赋值：上期所同一方向会推今仓、昨仓两行
                if side == "long":
                    row["long"] += n
                    row["long_yd"] += yd
                    row["long_today"] += td
                elif side == "short":
                    row["short"] += n
                    row["short_yd"] += yd
                    row["short_today"] += td
                # OpenCost 是**总成本**（含乘数），不是单价。多空两边的成本必须
                # 分开累加：`_positions()` 要按 PosiDirection 各算各的均价，
                # 混成一个 `price` 时「成交平仓盈亏」必错（CTP 那边也是分开记的）
                oc = _f(data.get("OpenCost"))
                if side == "long":
                    row["cost_long"] += oc
                elif side == "short":
                    row["cost_short"] += oc
                row["price"] += oc
                row["profit"] += _f(data.get("PositionProfit"))
                row["margin"] += _f(data.get("UseMargin"))
                outer._pos_rows += 1
                if last:
                    outer.positions_loaded = True

        def onRspQryInstrument(self, data, error, reqid, last):  # noqa: N802
            if last:
                outer.instruments_done.set()
            if not data:
                return
            ins = _s(data.get("InstrumentID"))
            if not ins:
                return
            with outer._lock:
                outer.instruments[ins] = {
                    "instrument": ins,
                    "exchange": _s(data.get("ExchangeID")),
                    "tick": _f(data.get("PriceTick")),
                    "multiple": _f(data.get("VolumeMultiple")),
                }

    return _Td()


def _login(outer: RohonTraderClient) -> None:
    outer.api.reqUserLogin(
        {
            "UserID": outer.cfg.user_id,
            "Password": outer.cfg.password,
            "BrokerID": outer.cfg.broker_id,
            "AppID": outer.cfg.app_id,
        },
        outer._next_req(),
    )
