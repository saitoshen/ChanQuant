"""CTP 行情 + 交易客户端（基于 openctp-ctp 的 CTP 官方接口封装）。

## 为什么要有这个

天勤覆盖了绝大多数场景，但它有两个短板：一是**股票和期货不能共用一个连接**，
二是模拟环境只有快期那一套。CTP 是国内的行业标准 —— 期货公司柜台、SimNow
模拟环境、各家量化终端都是它，接进来之后「换个柜台就能跑」这件事才成立。

`openctp-ctp` 把 CTP 的 C++ 头文件包成了 Python 扩展（`thostmduserapi` /
`thosttraderapi`），**不需要自己编译 DLL**，也不用装 CTP 原生 SDK。

## 依赖是可选的

`openctp-ctp` 只有 Windows / Linux 的 x86-64 wheel，macOS 上没有。所以这个模块
**导入不成功也不该让整个程序起不来**：`HAS_CTP` 为假时，界面照常显示账户页，
只是连接按钮会提示「装 openctp-ctp 才能用 CTP」。CI 的 macOS 构建就是靠这条
活下来的 —— 把 `openctp-ctp` 写进 `requirements.txt` 会让 macOS 打包直接失败。

## 和天勤那套的差别（迁移时最容易踩的三点）

| | 天勤 | CTP |
|---|---|---|
| 回调 | 同步返回 / `wait_update()` | 全是 `OnXxx` 回调，在 CTP 自己的线程里 |
| 价格 | float | float，但**必须按最小变动价位对齐**，否则报错 0x1001 |
| 开平 | `offset="OPEN"/"CLOSE"` | 字符常量 `'0'开 / '1'平 / '3'平今`，且上期所要区分平今平昨 |

回调线程不是主线程 —— 所有共享状态都用 `threading.Lock` 保护，界面只读快照。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field

from ttrade import ctp_channel
from ttrade.ctp_channel import ChannelError

# 注意：模块级**不再**直接 import openctp-ctp。为什么见 `ttrade/ctp_channel.py`：
# 融航那套（6.7.2.8 + 融航 dll）和标准那套（6.7.11）不能同进程共存，
# 一旦在 import 时就钉死成标准通道，融航账户就没法用了。改成用到时才绑定。

mdapi = None          # 绑定后才有的模块；`_make_spi()` 里按名字取结构体
tdapi = None
HAS_CTP = False       # 绑定成功后才为真
CTP_IMPORT_ERROR = "尚未绑定 CTP 通道"
BOUND_CHANNEL = ""    # "" / "ctp" / "rohon"


def bind_channel(kind: str = "ctp") -> bool:
    """把本模块的 `mdapi` / `tdapi` 绑到某个通道。重复绑定同一通道是安全的。

    返回是否可用（不可用时 `CTP_IMPORT_ERROR` 里有人话）。换通道会抛
    `ctp_channel.ChannelError` —— 那是进程级限制，不是能用 try/except 绕过的。
    """
    global mdapi, tdapi, HAS_CTP, CTP_IMPORT_ERROR, BOUND_CHANNEL

    kind = (kind or "ctp").strip().lower()
    if BOUND_CHANNEL == kind and HAS_CTP:
        return True
    try:
        md, td = ctp_channel.load(kind)
    except Exception as exc:  # noqa: BLE001 - ImportError / DLL 加载失败 / 通道冲突
        # 通道冲突（已经绑了另一个）必须往上抛：那是调用方用错了，不能静默降级
        if isinstance(exc, ChannelError):
            raise CtpError(str(exc)) from exc
        mdapi = tdapi = None
        HAS_CTP = False
        CTP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        BOUND_CHANNEL = kind
        return False
    mdapi, tdapi = md, td
    HAS_CTP = True
    CTP_IMPORT_ERROR = ""
    BOUND_CHANNEL = kind
    return True


# ------------------------------------------------------------------ 配置

# SimNow 的公开模拟环境。地址在 2025-06-19 换过一次，老的
# `180.168.146.187:10130/10131` 已经停了（实测连不上，超时），现在都走
# `182.254.243.31`：
#
#   - 7x24 环境：交易 40001 / 行情 40011。**只在非交易时段对外服务**
#     （交易日 16:00 ~ 次日 09:00，非交易日 16:00 ~ 次日 12:00），
#     定位是给 CTP 接口开发联调用，撮合和结算都不完整。
#   - 第一套环境：交易 30001 / 行情 30011（还有 30002/30012、30003/30013
#     两组备用）。服务时间和实盘交易时段完全一致 —— **周末和收盘后连不上，
#     TCP 直接 RST**，别当成配置错了。
#
# 所以自测要挑时间：白天开盘连第一套，晚上和周末连 7x24。
PRESETS = {
    "simnow_7x24": {
        "label": "SimNow 7x24（非交易时段联调）",
        "md_front": "tcp://182.254.243.31:40011",
        "td_front": "tcp://182.254.243.31:40001",
        "broker_id": "9999",
        "app_id": "simnow_client_test",
        "auth_code": "0000000000000000",
    },
    "simnow_day": {
        "label": "SimNow 第一套（交易时段）",
        "md_front": "tcp://182.254.243.31:30011",
        "td_front": "tcp://182.254.243.31:30001",
        "broker_id": "9999",
        "app_id": "simnow_client_test",
        "auth_code": "0000000000000000",
    },
    "openctp": {
        "label": "openctp（TTS 兼容）",
        "md_front": "tcp://121.37.80.177:20004",
        "td_front": "tcp://121.37.80.177:20002",
        "broker_id": "9999",
        "app_id": "client_ctp",
        "auth_code": "0000000000000000",
    },
}


#: 结算确认失败后隔多久重发（秒）。收盘后的结算窗口里，柜台常回
#: 「[26] 结算结果未就绪」这类错，等一会儿再发就好 —— 绝不能因为一次失败
#: 就把整条会话判死（那样界面只会说「登录超时」，用户完全看不出是结算没确认）
SETTLE_RETRY_SEC = 3.0


@dataclass
class CtpConfig:
    """一个 CTP 柜台的连接参数。"""

    name: str = "SimNow"
    preset: str = "simnow_7x24"
    md_front: str = ""
    td_front: str = ""
    broker_id: str = "9999"
    user_id: str = ""
    password: str = ""
    app_id: str = "simnow_client_test"
    auth_code: str = "0000000000000000"
    #: 走哪套 CTP 运行时：`ctp` = 系统装的 openctp-ctp；`rohon` = 融航那套
    #: （vendored 6.7.2.8 + 融航 dll）。一个进程只能用一个，见 ctp_channel
    channel: str = "ctp"

    def fill_preset(self) -> "CtpConfig":
        """没填前置地址时按预设补齐。手填过的字段不覆盖。"""
        p = PRESETS.get(self.preset) or PRESETS["simnow_7x24"]
        if not self.md_front:
            self.md_front = p["md_front"]
        if not self.td_front:
            self.td_front = p["td_front"]
        if not self.app_id:
            self.app_id = p["app_id"]
        if not self.auth_code:
            self.auth_code = p["auth_code"]
        return self

    def label(self) -> str:
        return f"{self.name} · {PRESETS.get(self.preset, {}).get('label', self.preset)}"


def config_path() -> str:
    from ttrade.paths import user_dir

    return os.path.join(user_dir(), "ctp.json")


def load_configs() -> list[CtpConfig]:
    path = config_path()
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
            out.append(CtpConfig(**{k: v for k, v in it.items()
                                    if k in CtpConfig.__dataclass_fields__}))
        except TypeError:
            continue
    return out


def save_configs(items: list[CtpConfig]) -> None:
    path = config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(c) for c in items], fh, ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ 小工具

def _s(v) -> str:
    """CTP 回调里的字符串字段可能是 bytes（OrderRef 之类）。"""
    if v is None:
        return ""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return v.decode("gbk", "ignore")
    return str(v)


def _err(pRspInfo) -> str:
    if pRspInfo is None:
        return ""
    try:
        if pRspInfo.ErrorID == 0:
            return ""
        return f"[{pRspInfo.ErrorID}] {_s(pRspInfo.ErrorMsg)}"
    except Exception:  # noqa: BLE001
        return ""


#: 委托行情 / 状态等的映射**只写一份**。`OnRtnOrder`（增量回报）和
#: `OnRspQryOrder`（登录后补查当日全部委托）拿到的是同一种
#: `CThostFtdcOrderField`，字段映射写两遍必然只会改一份 —— 而这里改漏的
#: 后果是「委托框里一行的时间列写着拒单理由」那种事。
_ORDER_FIELDS = {
    "order_ref": "OrderRef",
    "order_sys_id": "OrderSysID",
    "instrument": "InstrumentID",
    "direction": "Direction",
    "offset": "CombOffsetFlag",
    # 报单时间：InsertTime 只有 HH:MM:SS，配 TradingDay(YYYYMMDD) 才能拼出
    # 跨日的 MM-DD HH:MM:SS；缺一个就只能显示半截
    "insert_time": "InsertTime",
    "trading_day": "TradingDay",
}

_TRADE_FIELDS = {
    "trade_id": "TradeID",
    "order_ref": "OrderRef",
    "instrument": "InstrumentID",
    "direction": "Direction",
    "offset": "OffsetFlag",
    "time": "TradeTime",
    "trading_day": "TradingDay",
}


def _order_row(pOrder) -> dict:
    """`CThostFtdcOrderField` → 内部委托行。增量和历史查询共用。"""
    row = {k: _s(getattr(pOrder, attr, "")) for k, attr in _ORDER_FIELDS.items()}
    row.update(
        price=float(getattr(pOrder, "LimitPrice", 0) or 0),
        volume=int(getattr(pOrder, "VolumeTotalOriginal", 0) or 0),
        traded=int(getattr(pOrder, "VolumeTraded", 0) or 0),
        status=_STATUS.get(_s(getattr(pOrder, "OrderStatus", "")), "未知"),
        msg=_s(getattr(pOrder, "StatusMsg", "")),
        # `'1'` = 市价（AnyPrice），其余都是限价。界面靠它把「价格」写「市价」
        # 而不是一个 `0`（市价单的 LimitPrice 本来就是 0）
        price_type=_s(getattr(pOrder, "OrderPriceType", "")),
    )
    return row


def _trade_row(pTrade) -> dict:
    """`CThostFtdcTradeField` → 内部成交行。增量和历史查询共用。"""
    row = {k: _s(getattr(pTrade, attr, "")) for k, attr in _TRADE_FIELDS.items()}
    row["trade_id"] = row["trade_id"] or str(time.time())
    row.update(
        price=float(getattr(pTrade, "Price", 0) or 0),
        volume=int(getattr(pTrade, "Volume", 0) or 0),
    )
    return row


class CtpError(RuntimeError):
    """连接 / 登录 / 下单失败。带 CTP 原始错误码，便于查表。"""


# ------------------------------------------------------------------ 行情

class MdClient:
    """行情：连接 → 登录 → 订阅 → 回调推 tick。

    用法：

        md = MdClient(cfg)
        md.start(["rb2601", "cu2601"])
        md.wait_ready(10)
        md.last("rb2601")   # {'last':..., 'bid':..., 'ask':...}
    """

    def __init__(self, cfg: CtpConfig) -> None:
        self.cfg = cfg.fill_preset()
        self.api = None
        self.ready = threading.Event()
        self.last_error = ""
        self.ticks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._spi = None

    # ---- 生命周期 ----

    def start(self, instruments=()) -> None:
        if not bind_channel(getattr(self.cfg, "channel", "ctp") or "ctp"):
            raise CtpError(f"用不了 CTP 行情（{CTP_IMPORT_ERROR}）")
        spi = self._make_spi()
        self._spi = spi
        flow = os.path.join(_flow_dir(), f"md_{int(time.time())}")
        api = mdapi.CThostFtdcMdApi.CreateFtdcMdApi(flow)
        api.RegisterFront(self.cfg.md_front)
        api.RegisterSpi(spi)
        api.Init()
        self.api = api
        self._instruments = list(instruments)

    def stop(self) -> None:
        api, self.api = self.api, None
        self.ready.clear()
        if api is not None:
            try:
                api.Release()
            except Exception:  # noqa: BLE001
                pass

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self.ready.wait(timeout)

    # ---- 读 ----

    def last(self, instrument: str) -> dict:
        with self._lock:
            return dict(self.ticks.get(instrument) or {})

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self.ticks.items()}

    # ---- SPI ----

    def _make_spi(self):
        outer = self

        class _Spi(mdapi.CThostFtdcMdSpi):
            def OnFrontConnected(self):  # noqa: N802 - CTP 规定的方法名
                try:
                    req = mdapi.CThostFtdcReqUserLoginField()
                    req.BrokerID = outer.cfg.broker_id
                    req.UserID = outer.cfg.user_id
                    req.Password = outer.cfg.password
                    outer.api.ReqUserLogin(req, 1)
                except Exception as exc:  # noqa: BLE001
                    outer.last_error = f"行情登录请求失败：{exc}"

            def OnFrontDisconnected(self, nReason):  # noqa: N802
                outer.last_error = f"行情前置断开（{nReason}）"
                outer.ready.clear()

            def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                err = _err(pRspInfo)
                if err:
                    outer.last_error = f"行情登录失败：{err}"
                    return
                outer.ready.set()
                if outer._instruments:
                    outer.subscribe(outer._instruments)

            def OnRspSubMarketData(self, pSpecificInstrument, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                err = _err(pRspInfo)
                if err:
                    outer.last_error = f"订阅失败：{err}"

            def OnRtnDepthMarketData(self, pDepthMarketData):  # noqa: N802
                d = pDepthMarketData
                if d is None:
                    return
                ins = _s(d.InstrumentID)
                with outer._lock:
                    outer.ticks[ins] = {
                        "instrument": ins,
                        "datetime": f"{_s(d.ActionDay)} {_s(d.UpdateTime)}",
                        "last": _f(d.LastPrice),
                        "bid": _f(d.BidPrice1),
                        "ask": _f(d.AskPrice1),
                        "bid_vol": _f(d.BidVolume1),
                        "ask_vol": _f(d.AskVolume1),
                        "open": _f(d.OpenPrice),
                        "high": _f(d.HighestPrice),
                        "low": _f(d.LowestPrice),
                        "pre_close": _f(d.PreClosePrice),
                        # 昨结算：平昨仓的平仓盈亏要按它算（逐日盯市口径，见
                        # `tqlive.replay_close_pnl`）。`PreClosePrice` 是昨收，
                        # 两者在涨跌停基准上也不是一回事，别混用
                        "pre_settle": _f(getattr(d, "PreSettlementPrice", 0)),
                        # 当日收盘价：只有收市后才有值，盘中是 0 —— 成交流拿不到
                        # 时它是「昨收」的次日替代，调用方自己判 0
                        "close": _f(getattr(d, "ClosePrice", 0)),
                        "volume": _f(d.Volume),
                        "oi": _f(d.OpenInterest),
                    }

        return _Spi()

    def subscribe(self, instruments) -> None:
        """订阅行情。

        第二个参数是**个数**，不是「0 表示全部」—— 传 0 时柜台照样回
        `OnRspSubMarketData` 且 `ErrorID == 0`，然后一条 tick 都不推。这个坑
        在 SimNow 上实测过：订阅应答全是「成功」，盘面一片空白，日志里没有
        任何异常，只能靠「10 秒没 tick」去反推。
        """
        if self.api is None or not instruments:
            return
        items = [i.encode("utf-8") for i in instruments]
        try:
            self.api.SubscribeMarketData(items, len(items))
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"订阅异常：{exc}"


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _flow_dir() -> str:
    """CTP 的流文件目录。必须存在且可写，否则 Init() 直接崩。"""
    from ttrade.paths import cache_dir

    d = os.path.join(cache_dir(), "ctp_flow")
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ 交易

# CTP 的开平标志是字符常量，这里统一成可读名
OFFSET_OPEN = "0"
OFFSET_CLOSE = "1"
OFFSET_CLOSE_TODAY = "3"
DIR_BUY = "0"
DIR_SELL = "1"


def _req_why(n) -> str:
    """`Req*` 系列返回值翻成人话。用于报单 / 撤单 / 查询三条路径。

    🔴 这几条以前**全都不看返回值**（2026-11-23 融航）：CTP 的 `Req*` 返回非 0
    表示请求压根没进发送队列 —— 之后一条回调都不会来。于是「发了个寂寞」和
    「账户真没仓 / 单子在交易所好好挂着」在界面上长得一模一样。
    """
    try:
        code = int(n or 0)
    except (TypeError, ValueError):
        return str(n)
    why = {-1: "网络连接失败", -2: "未处理请求超上限",
           -3: "每秒查询 / 报单超限（柜台一秒只收一个）"}.get(code, "柜台拒绝")
    return f"{code} {why}"


class TraderClient:
    """交易：认证 → 登录 → 结算确认 → 下单 / 撤单 / 查持仓。

    **下单前必须确认结算结果**（`ReqSettlementInfoConfirm`），否则柜台会拒单，
    而拒单的理由是一句「未确认结算」，第一次接 CTP 的人基本都栽在这。

    回调在 CTP 线程里跑，这里只往 dict 里写；界面每帧读一份快照。
    """

    def __init__(self, cfg: CtpConfig) -> None:
        self.cfg = cfg.fill_preset()
        self.api = None
        self.connected = threading.Event()      # 前置连上
        self.logged_in = threading.Event()      # 登录成功
        self.settled = threading.Event()        # 结算已确认（可以下单）
        self.last_error = ""
        # 结算确认的重试 / 交易日跟踪。见 `confirm_settlement()`
        self.trading_day = ""       # 登录回包里的交易日（YYYYMMDD）
        self._settle_at = 0.0
        self._settle_warn = 0.0
        self.orders: dict[str, dict] = {}
        self.trades: dict[str, dict] = {}
        self.positions: dict[str, dict] = {}
        # 持仓查询的「代」。CTP 的 OnRspQryInvestorPosition 是一条一条推的，
        # 一整轮才凑出完整快照，所以**不能**在发请求时就把 positions 清掉 ——
        # 界面会看到半份/空仓，策略那边更会以为账户没仓而重复开仓。
        # 这里只记「这一轮的第一条回调到了没有」：到了才清，之后整轮往里累加。
        self._pos_req = -1            # 当前这轮查询的 RequestID
        self._pos_started = True      # 本轮是否已经清过（True=没有进行中的查询）
        self.account: dict = {}
        self.instruments: dict[str, dict] = {}
        # 当日委托 / 成交的**历史**查到没有。柜台只在登录之后推
        # `OnRtnOrder` / `OnRtnTrade`，上一次进程（甚至同一进程里更早的一次
        # 登录）发生的单子不查就永远看不见，见 `query_orders()`。
        # 用「查完了」而不是「查过」：柜台忙时会回一条错误的空包，那时不能算完
        self.orders_loaded = False
        self.trades_loaded = False
        # 持仓查询有没有**成功回来过一整轮**（`bIsLast` 到了，空仓也算）。
        # 🔴 光看 `positions` 是否为空是分不清「账户真没仓」和「柜台没回」的，
        # 而这两件事对一键对齐是天壤之别（2026-11-23 融航：交易页 6 个合约有仓，
        # 组合页那条连接一条持仓回调都没来，监控器却报「一致」）
        self.positions_loaded = False
        #: 最近一次**整轮**持仓查询回完的时刻。`positions_loaded` 只说「读到过」，
        #: 不说明那份数据有多旧 —— 组合页一旦没人推这条连接，`positions` 会永远
        #: 停在连接那一刻，而界面照旧把旧数当现值画（2026-11-23 融航）。见
        #: `BrokerApi.positions_ready()` 的 `POS_STALE_SEC`
        self.pos_at = 0.0
        # 全量合约表查完没有。SimNow 上 `ReqQryInstrument` 会回 1.6 万条，
        # 要 80 秒左右才流完，而**流没结束时后续请求全部排队**：查资金、报单
        # 都一声不响地卡着。所以这个事件是给调用方判断「现在能不能安全下单」
        # 用的，别拿它当装饰。
        self.instruments_done = threading.Event()
        # 合约 → 交易所的兜底解析器（`CtpHub` 会塞 `variety_exchange` 进来）。
        # 有了它就不用为了撤单去拉 1.6 万条合约表。
        self.exchange_resolver = None
        self.front_id = 0
        self.session_id = 0
        self._lock = threading.Lock()
        # 🔴 一条会话现在会被交易页和组合页**同时**用（见 `ttrade/hubreg.py`）。
        # 这两个计数器是「读一下再写」两步，两个线程同时进来会拿到同一号：
        # - `_req` 重号：`onRspQryInvestorPosition` 靠 RequestID 归轮，重号会让
        #   别的查询的回包被当成本轮的持仓；
        # - `_order_ref` 重号：CTP 的本地单号重号，柜台会拒单或把回报挂到
        #   另一笔单上 —— 表现是「下单成功、成交记在别人头上」。
        self._req_lock = threading.Lock()
        self._ord_lock = threading.Lock()
        self._order_ref = int(time.time()) % 100000
        self._spi = None

    # ---- 生命周期 ----

    def start(self) -> None:
        if not bind_channel(getattr(self.cfg, "channel", "ctp") or "ctp"):
            raise CtpError(f"用不了 CTP 交易（{CTP_IMPORT_ERROR}）")
        spi = self._make_spi()
        self._spi = spi
        flow = os.path.join(_flow_dir(), f"td_{int(time.time())}")
        api = tdapi.CThostFtdcTraderApi.CreateFtdcTraderApi(flow)
        api.RegisterFront(self.cfg.td_front)
        api.RegisterSpi(spi)
        # 私有流用 QUICK：成交 / 委托回报尽快到，实盘下单后等一秒就能看到状态
        api.SubscribePrivateTopic(tdapi.THOST_TERT_QUICK)
        api.SubscribePublicTopic(tdapi.THOST_TERT_QUICK)
        api.Init()
        self.api = api

    def stop(self) -> None:
        api, self.api = self.api, None
        for ev in (self.connected, self.logged_in, self.settled):
            ev.clear()
        if api is not None:
            try:
                api.Release()
            except Exception:  # noqa: BLE001
                pass

    def wait_ready(self, timeout: float = 15.0) -> bool:
        """等到「能下单」为止：登录 + 结算确认都完成。"""
        return self.settled.wait(timeout)

    # ---- 结算确认 ----

    def confirm_settlement(self, force: bool = False) -> bool:
        """发一次「投资者结算结果确认」。真的发出去了才返回 True。

        🔴 **登录之后必须做这一步**，否则柜台把每一笔报单都打回
        `[39] 未确认结算结果`（这个错误码和不确认结算的那句提示，是第一次接
        CTP 的人最容易栽的地方）。所以调用链是固定的三段：

            OnFrontConnected → ReqAuthenticate
            OnRspAuthenticate → ReqUserLogin
            OnRspUserLogin    → ReqSettlementInfoConfirm   ← 就是这里
            OnRspSettlementInfoConfirm → settled.set()     ← 之后才允许下单

        `start()` 里 `wait_ready()` 等的是 `settled`，不是 `logged_in`：登录成功
        但结算没确认，报单照样全被拒。

        为什么还要留一个公开的重试入口（2026-11-23 补）：确认**按交易日**有效，
        两处会掉：① 断线重连（`OnFrontDisconnected` 清掉 `settled`）；
        ② 跨交易日（夜盘 21:00 起算下一交易日，柜台的结算标志会被重置）。
        这两种情况以前没有任何人重发，用户只能自己断开重连整条会话。
        `CtpHub.wait()` 和 `BrokerApi._refresh()` 会隔 `SETTLE_RETRY_SEC` 秒
        调一次这里，确认成功即自动哑掉。
        """
        if tdapi is None or self.api is None:
            return False
        if not self.logged_in.is_set():
            return False
        if self.settled.is_set() and not force:
            return False
        now = time.monotonic()
        if not force and now - self._settle_at < SETTLE_RETRY_SEC:
            return False
        self._settle_at = now
        try:
            req = tdapi.CThostFtdcSettlementInfoConfirmField()
            req.BrokerID = self.cfg.broker_id
            req.InvestorID = self.cfg.user_id
            self.api.ReqSettlementInfoConfirm(req, self._next_req())
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"结算确认请求失败：{exc}"
            return False
        return True

    def _note_trading_day(self, day: str) -> None:
        """柜台回包里的 `TradingDay` 变了 ⇒ 切日了，重新确认结算结果。

        🔴 为什么要有（2026-11-23）：结算确认**按交易日**有效，柜台切日时会把
        确认标志重置。一条从昨天下午一直连着的会话，跨到夜盘（21:00 起算下一
        交易日）之后如果不重确认，第一笔报单就被打回「[39] 未确认结算结果」，
        而界面上「已连接」一切正常 —— 只有下单失败，很难联想到结算。

        判定只看「和上一次不一样」，所以同一天里调多少次都只会触发一次重确认。
        日期为空（柜台没给）时不动：宁可漏判一次，也不能拿空串当「切日」天天
        重确认。
        """
        day = str(day or "").strip()
        if not day:
            return
        if not self.trading_day:
            # 第一次见到交易日：只记下来，不重确认（登录时刚确认过）。
            # 不能因为「基准还是空」就永远不记 —— 那样跨日探测永远醒不过来
            self.trading_day = day
            return
        if day == self.trading_day:
            return
        old, self.trading_day = self.trading_day, day
        self.settled.clear()
        self._settle_at = 0.0
        self.confirm_settlement(force=True)
        if not self.settled.is_set():
            # 重确认没成功（多数是柜台正在结算）。`CtpHub.wait()` 会一直重试到成功
            self.last_error = (f"交易日已切到 {day}（原 {old}），正在重新确认结算结果；"
                               "确认之前下单会被柜台打回")

    # ---- 下单 ----

    def place(self, instrument: str, direction: str, offset: str,
              price: float, volume: int, market: bool = False) -> str:
        """报单。返回 OrderRef（CTP 侧的本地单号）。

        `direction` 用 `BUY` / `SELL`，`offset` 用 `OPEN` / `CLOSE` / `CLOSETODAY`，
        和平台其他地方的写法一致，转换在这里做。

        `market=True` 发**市价单**（`OrderPriceType=AnyPrice`，价格字段填 0）。

        🔴 2026-11-23 用户：「ctp 和融航的下单，现在所有交易所应该都支持市价单了
        … 上期所和能源所在 2026 年 7 月 6 日也开放支持市价单了」。所以这里不再
        一刀切拒绝市价单 —— 以前是硬编码「柜台只收限价单」，市价那个勾在界面上
        直接被禁掉。
        市价单的时间条件必须用 **IOC**（立即成交、剩余撤销）：GFD + AnyPrice 会被
        交易所打回「该合约不支持此时间条件」（CTP 的惯例，vnpy 也是这么发的）。
        """
        if self.api is None or not self.logged_in.is_set():
            raise CtpError("还没登录")
        if not self.settled.is_set():
            raise CtpError("还没确认结算结果，柜台会拒单")
        d = {"BUY": DIR_BUY, "SELL": DIR_SELL}.get(str(direction).upper())
        if d is None:
            raise CtpError(f"方向只能是 BUY / SELL，收到 {direction}")
        off = {"OPEN": OFFSET_OPEN, "CLOSE": OFFSET_CLOSE,
               "CLOSETODAY": OFFSET_CLOSE_TODAY}.get(str(offset).upper())
        if off is None:
            raise CtpError(f"开平只能是 OPEN / CLOSE / CLOSETODAY，收到 {offset}")
        if volume <= 0:
            raise CtpError("手数必须大于 0")
        # `price` 允许是 None / NaN（天勤那边就是这么表示市价的），统一折成
        # 「市价」而不是在 `price > 0` 上抛 TypeError
        try:
            px = float(price)
        except (TypeError, ValueError):
            px = float("nan")
        market = bool(market) or not (px > 0)   # 0 / NaN / None 都当市价
        if not market and px <= 0:
            raise CtpError("限价单必须给一个大于 0 的价格")

        with self._ord_lock:
            self._order_ref += 1
            ref = str(self._order_ref)
        req = tdapi.CThostFtdcInputOrderField()
        req.BrokerID = self.cfg.broker_id
        req.InvestorID = self.cfg.user_id
        req.UserID = self.cfg.user_id
        req.InstrumentID = instrument
        # 交易所代码。CTP 前台多数情况下能自己按合约查出来，但**查不出来时
        # 报的是 `[148] 无效的 ExchangeID 字段`**，而且这个错跟合约名没关系，
        # 极难反查（2026-11-19 TL 策略踩了：合约被拼成 `tlcffex0`，柜台只说
        # ExchangeID 无效）。显式填上，填不出来就在本地抛，别发一笔注定被拒的单。
        ex = self._exchange_of(instrument)
        if not ex:
            raise CtpError(
                f"认不出 {instrument} 属于哪个交易所，不能报单"
                "（多半是合约代码少写了交易所，或者品种代码不在本地数据里）")
        req.ExchangeID = ex
        req.OrderRef = ref
        req.OrderPriceType = (tdapi.THOST_FTDC_OPT_AnyPrice if market
                              else tdapi.THOST_FTDC_OPT_LimitPrice)
        req.LimitPrice = 0.0 if market else px
        req.Direction = d
        req.CombOffsetFlag = {"0": tdapi.THOST_FTDC_OF_Open,
                              "1": tdapi.THOST_FTDC_OF_Close,
                              "3": tdapi.THOST_FTDC_OF_CloseToday}[off]
        # 投机单。套保 / 套利要提前在柜台备案，写死投机最省事也最不容易被拒
        req.CombHedgeFlag = tdapi.THOST_FTDC_HF_Speculation
        req.VolumeTotalOriginal = int(volume)
        # 当日有效。**市价单必须换 IOC**：GFD + AnyPrice 会被交易所打回
        # 「该合约不支持此时间条件」
        req.TimeCondition = (tdapi.THOST_FTDC_TC_IOC if market
                             else tdapi.THOST_FTDC_TC_GFD)
        req.VolumeCondition = tdapi.THOST_FTDC_VC_AV      # 任意数量成交
        req.MinVolume = 1
        req.ContingentCondition = tdapi.THOST_FTDC_CC_Immediately
        req.ForceCloseReason = tdapi.THOST_FTDC_FCC_NotForceClose
        req.IsAutoSuspend = 0
        req.UserForceClose = 0
        # 🔴 返回值必须看（2026-11-23 融航那四笔）：`ReqOrderInsert` 返回非 0
        # 意味着柜台**根本没受理**，之后不会有任何委托回报 —— 而调用方以为
        # 发出去了，在途列表里挂 30 秒「ALIVE」再撤掉，委托框里一笔都没有。
        n = self.api.ReqOrderInsert(req, self._next_req())
        if n:
            raise CtpError("报单没发出去：" + _req_why(n))
        return ref

    def cancel(self, instrument: str, order_sys_id: str = "", order_ref: str = "") -> None:
        """撤单。有 `OrderSysID` 就用它（柜台单号），没有就用本地 `OrderRef`。"""
        if self.api is None:
            raise CtpError("还没连接")
        req = tdapi.CThostFtdcInputOrderActionField()
        req.BrokerID = self.cfg.broker_id
        req.InvestorID = self.cfg.user_id
        req.UserID = self.cfg.user_id
        req.InstrumentID = instrument
        req.ExchangeID = self._exchange_of(instrument)
        req.ActionFlag = tdapi.THOST_FTDC_AF_Delete
        if order_sys_id:
            req.OrderSysID = order_sys_id
        else:
            req.OrderRef = order_ref
            req.FrontID = self.front_id
            req.SessionID = self.session_id
        # 同样要看返回值：非 0 表示撤单请求没有受理，那笔单**还在交易所挂着**。
        # 以前静默丢掉，界面却按「已撤」处理了。
        n = self.api.ReqOrderAction(req, self._next_req())
        if n:
            raise CtpError("撤单没发出去：" + _req_why(n))

    # ---- 查询 ----

    def _qry_note(self, what: str, n) -> None:
        """把 `ReqQry*` 的返回值翻成人话记进 `last_error`。

        🔴 为什么要看（2026-11-23 融航 SA701）：CTP 同一会话**每秒最多一个查询**，
        超了 `ReqQry*` 会返回非 0（`-3` 查询未就绪 / `-2` 未处理请求超上限），
        **不回包、也不进任何回调**。以前这里全都不看返回值，于是「发了个寂寞」
        在界面上长得跟「账户真没仓」一模一样 —— 组合页那条连接就是连发
        「资金 + 持仓」两个查询，资金被受理、持仓被流控丢掉，`_pos` 永远空。

        只在**没发出去**时写，成功就清掉（不然上一次的流控一直挂在那儿误导）。
        `last_error` 一律走 `getattr`：自检里常用 `__new__` 造精简桩，而这是
        0.4 秒一次的实盘主路径，不能因为桩少一个属性就抛异常。
        """
        try:
            code = int(n or 0)
        except (TypeError, ValueError):
            code = 0
        old = str(getattr(self, "last_error", "") or "")
        if code:
            self.last_error = f"查{what}没发出去：{_req_why(code)}"
        elif old.startswith(f"查{what}没发出去"):
            self.last_error = ""

    def query_account(self) -> None:
        """查资金。**要看返回值** —— 非 0 表示柜台根本没受理（见 `_qry_note`）。"""
        if self.api is None:
            return
        n = self.api.ReqQryTradingAccount(tdapi.CThostFtdcQryTradingAccountField(),
                                         self._next_req())
        self._qry_note("资金", n)

    def query_positions(self) -> None:
        """查持仓。🔴 **每轮必须重新攒一份**，见回调里的累加注释。

        之前这里只是发请求，回调里又用 `+=`，而 `positions` 从来不清 ——
        每查一次就把上一轮的浮盈 / 保证金再叠一遍。界面 2 秒刷一轮，跑一小时
        浮盈就放大成千上万倍（用户看到「多了几个 0」，而且收盘后还在涨）。
        """
        if self.api is None:
            return
        req = tdapi.CThostFtdcQryInvestorPositionField()
        req.BrokerID = self.cfg.broker_id
        req.InvestorID = self.cfg.user_id
        rid = self._next_req()
        with self._lock:
            self._pos_req = rid
            self._pos_started = False
        self._qry_note("持仓", self.api.ReqQryInvestorPosition(req, rid))

    def query_orders(self) -> None:
        """查**当日全部委托**（历史补全）。🔴 登录后必须查一次。

        为什么必须有这一条（2026-11-23 融航那次）：持仓是查出来的
        （`query_positions`），委托和成交却只靠 `OnRtnOrder` / `OnRtnTrade`
        推送 —— 而推送只覆盖**登录之后**发生的事。于是重启一次之后，账户表上
        明明写着「今空 5 手」，委托框和成交框却是空的，用户只能理解成「系统
        没从账户里读」。CTP 的 `ReqQryOrder` 本来就是为此存在的。

        副作用是好的：`_sweep_orders()`（启动时撤掉自己留下的挂单）终于能
        看见上一次进程留下的单子，不再是空跑一遍。
        """
        if self.api is None:
            return
        req = tdapi.CThostFtdcQryOrderField()
        req.BrokerID = self.cfg.broker_id
        req.InvestorID = self.cfg.user_id
        self._qry_note("委托", self.api.ReqQryOrder(req, self._next_req()))

    def query_trades(self) -> None:
        """查**当日全部成交**（历史补全）。理由同 `query_orders()`。

        `CThostFtdcQryTradeField` 的 `TradeID` / `InstrumentID` 留空 = 不限，
        柜台按投资者账号把当日成交一条条推回来（一条一个 `OnRspQryTrade`）。
        """
        if self.api is None:
            return
        req = tdapi.CThostFtdcQryTradeField()
        req.BrokerID = self.cfg.broker_id
        req.InvestorID = self.cfg.user_id
        self._qry_note("成交", self.api.ReqQryTrade(req, self._next_req()))

    def query_instruments(self, instrument: str = "") -> None:
        """查合约表。传 `instrument` 就只查那一个（秒回），不传就查全量。

        **全量查询在 SimNow 上是 1.6 万条、80 秒**，而且 CTP 一个会话是串行的：
        流没结束之前，后面排队的报单和查资金都不会有任何回应。界面启动时如果
        顺手查一次全量，用户点下单就会静默卡一分钟 —— 这个坑实测踩过。
        所以只在下单/撤单真的需要交易所时才按合约查一个。
        """
        if self.api is None:
            return
        req = tdapi.CThostFtdcQryInstrumentField()
        if instrument:
            req.InstrumentID = instrument
        else:
            self.instruments_done.clear()
        self.api.ReqQryInstrument(req, self._next_req())

    # ---- 读 ----

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "connected": self.logged_in.is_set(),
                "settled": self.settled.is_set(),
                "error": self.last_error,
                "account": dict(self.account),
                "positions": {k: dict(v) for k, v in self.positions.items()},
                "orders": {k: dict(v) for k, v in self.orders.items()},
                "trades": {k: dict(v) for k, v in self.trades.items()},
                # 「查完了没有」必须和表一起透出去：`positions` 为空分不清
                # 「账户真没仓」和「柜台一条回调都没来」，后者报「一致」就是撒谎
                "positions_loaded": self.positions_loaded,
                # 那一轮是什么时候回来的。「读到过」和「刚才读过」是两件事，
                # 界面要拿它判手上这份持仓有多旧（见 `POS_STALE_SEC`）
                "pos_at": self.pos_at,
                "orders_loaded": self.orders_loaded,
                "trades_loaded": self.trades_loaded,
            }

    # ---- 内部 ----

    def _next_req(self) -> int:
        # 加锁：交易页和组合页共用一条会话，两个线程同时进来会拿到同一个号
        # （RequestID 是「哪一轮持仓查询」的判据，重号会把别的回包认进来）
        with self._req_lock:
            self._req = getattr(self, "_req", 0) + 1
            return self._req

    def _exchange_of(self, instrument: str) -> str:
        """合约 → 交易所。按「查表 → 外部解析器 → 品种前缀猜」三级兜底。

        猜错会导致撤单报「合约不存在」。查表最准但要先拉全量（80 秒，见
        `query_instruments` 的注释），所以正常路径是外部解析器 —— `CtpHub`
        会把 `variety_exchange`（扫本地 `data_futures/*.csv` 得到）挂进来，
        比写死的前缀表更可靠，新品种上市也认得。
        """
        info = self.instruments.get(instrument)
        if info and info.get("exchange"):
            return info["exchange"]
        if self.exchange_resolver is not None:
            try:
                ex = self.exchange_resolver(instrument)
            except Exception:  # noqa: BLE001 - 解析器出问题不能连累撤单
                ex = ""
            if ex:
                return ex
        from ttrade.symbols_info import EXCHANGES  # noqa: F401 - 只为了保持口径一致

        letters = "".join(ch for ch in instrument if ch.isalpha())
        # 本地数据目录（`data_futures/*.csv` 的文件名）比下面几张写死的前缀表
        # 更权威，新品种上市也认得。写在前面，TL / TL0 这类就不会掉到默认分支
        try:
            from ttrade.ctp_live import variety_exchange

            ex = variety_exchange(letters)
            if ex:
                return ex
        except Exception:  # noqa: BLE001
            pass
        for ex, names in (("CZCE", ("TA", "MA", "FG", "SA", "SR", "CF", "RM", "OI")),):
            if letters.upper().startswith(names):
                return ex
        if letters.isupper() and len(letters) <= 2 and instrument[len(letters):].isdigit() \
                and len(instrument[len(letters):]) == 3:
            return "CZCE"        # 郑商所三位月份
        if letters.upper() in ("IF", "IH", "IC", "IM", "T", "TF", "TS", "TL"):
            return "CFFEX"
        if letters.lower() in ("sc", "lu", "nr", "bc", "ec"):
            return "INE"
        if letters.lower() in ("si", "lc", "ps", "pt", "pd"):
            return "GFEX"
        if letters.lower() in ("a", "b", "c", "cs", "eb", "eg", "i", "j", "jd", "jm",
                               "l", "m", "p", "pg", "pp", "rr", "v", "y", "lh", "lg"):
            return "DCE"
        # 认不出来就**返回空**，别再默认上期所 —— 猜成 SHFE 的后果是报单带着
        # 一个错误的 ExchangeID 发出去，柜台回 `[148] 无效的 ExchangeID`，
        # 而这句话完全不提「其实是合约名错了」，反查极难（2026-11-19 TL0）
        return ""

    def _make_spi(self):
        outer = self

        class _Spi(tdapi.CThostFtdcTraderSpi):
            # ---- 连接 / 登录 ----

            def OnFrontConnected(self):  # noqa: N802
                outer.connected.set()
                try:
                    req = tdapi.CThostFtdcReqAuthenticateField()
                    req.BrokerID = outer.cfg.broker_id
                    req.UserID = outer.cfg.user_id
                    req.AppID = outer.cfg.app_id
                    req.AuthCode = outer.cfg.auth_code
                    outer.api.ReqAuthenticate(req, outer._next_req())
                except Exception as exc:  # noqa: BLE001
                    outer.last_error = f"认证请求失败：{exc}"

            def OnFrontDisconnected(self, nReason):  # noqa: N802
                outer.last_error = f"交易前置断开（{nReason}）"
                outer.connected.clear()
                outer.logged_in.clear()
                outer.settled.clear()

            def OnRspAuthenticate(self, pRspAuthenticateField, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                err = _err(pRspInfo)
                if err:
                    outer.last_error = f"认证失败：{err}"
                    return
                try:
                    req = tdapi.CThostFtdcReqUserLoginField()
                    req.BrokerID = outer.cfg.broker_id
                    req.UserID = outer.cfg.user_id
                    req.Password = outer.cfg.password
                    outer.api.ReqUserLogin(req, outer._next_req())
                except Exception as exc:  # noqa: BLE001
                    outer.last_error = f"登录请求失败：{exc}"

            def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                err = _err(pRspInfo)
                if err:
                    outer.last_error = f"登录失败：{err}"
                    return
                outer.front_id = getattr(pRspUserLogin, "FrontID", 0) or 0
                outer.session_id = getattr(pRspUserLogin, "SessionID", 0) or 0
                # 登录回包里的交易日：跨日探测的基准（见 `OnRspQryInvestorPosition`）
                day = _s(getattr(pRspUserLogin, "TradingDay", ""))
                if day:
                    outer.trading_day = day
                outer.logged_in.set()
                # 重新登录（断线重连）之后结算确认要**再来一遍**：清掉旧标志，
                # 结论一律以柜台的确认回包为准
                outer.settled.clear()
                outer.confirm_settlement(force=True)

            def OnRspSettlementInfoConfirm(self, pSettlementInfoConfirm, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                err = _err(pRspInfo)
                if err:
                    # 结算窗口里回「[26] 结算结果未就绪」是常态，隔几秒重发就好
                    # （`confirm_settlement()` 的重试）。这句话只每 30 秒记一次，
                    # 否则盘中一个没确认成的会话会把 last_error 刷成滚动字幕
                    now = time.monotonic()
                    if now - outer._settle_warn > 30:
                        outer._settle_warn = now
                        outer.last_error = (
                            f"结算确认失败（{err}），{SETTLE_RETRY_SEC:.0f} 秒后自动重试"
                            "（确认之前下单会被柜台打回「未确认结算结果」）")
                    return
                # 确认回包里带交易日，顺手校准一次（跨日探测的基准）
                day = _s(getattr(pSettlementInfoConfirm, "TradingDay", ""))
                if day:
                    outer.trading_day = day
                outer.last_error = ""
                outer.settled.set()

            # ---- 报单 / 成交 ----

            def OnRspOrderInsert(self, pInputOrder, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                ref = _s(getattr(pInputOrder, "OrderRef", ""))
                err = _err(pRspInfo)
                with outer._lock:
                    outer.orders[ref] = {
                        "order_ref": ref,
                        "order_sys_id": "",
                        "instrument": _s(getattr(pInputOrder, "InstrumentID", "")),
                        "direction": _s(getattr(pInputOrder, "Direction", "")),
                        "offset": _s(getattr(pInputOrder, "CombOffsetFlag", "")),
                        "price": float(getattr(pInputOrder, "LimitPrice", 0) or 0),
                        "volume": int(getattr(pInputOrder, "VolumeTotalOriginal", 0) or 0),
                        "traded": 0,
                        "status": "被拒" if err else "已提交",
                        "msg": err,
                        # 柜台回报里没有时间字段，只能记本地墙钟。界面「时间」列
                        # 靠这两个键，缺了会退化成显示委托说明（见 _orders）
                        "insert_time": time.strftime("%H:%M:%S"),
                        "trading_day": time.strftime("%Y%m%d"),
                    }
                if err:
                    outer.last_error = f"下单被拒：{err}"

            def OnRtnOrder(self, pOrder):  # noqa: N802
                row = _order_row(pOrder)
                sys_id, ref = row["order_sys_id"], row["order_ref"]
                key = sys_id or ref
                with outer._lock:
                    outer.orders[key] = row
                    if ref and ref != key:
                        outer.orders[ref] = dict(row)

            def OnRtnTrade(self, pTrade):  # noqa: N802
                row = _trade_row(pTrade)
                with outer._lock:
                    outer.trades[row["trade_id"]] = row

            def OnRspOrderAction(self, pInputOrderAction, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                err = _err(pRspInfo)
                if err:
                    outer.last_error = f"撤单失败：{err}"
                    return
                ref = _s(getattr(pInputOrderAction, "OrderRef", ""))
                with outer._lock:
                    for o in outer.orders.values():
                        if o.get("order_ref") == ref:
                            o["status"] = "已撤单"

            def OnErrRtnOrderInsert(self, pInputOrder, pRspInfo):  # noqa: N802
                err = _err(pRspInfo)
                if err:
                    outer.last_error = f"报单错误：{err}"

            def OnErrRtnOrderAction(self, pOrderAction, pRspInfo):  # noqa: N802
                err = _err(pRspInfo)
                if err:
                    outer.last_error = f"撤单错误：{err}"

            # ---- 查询回报 ----

            def OnRspQryTradingAccount(self, pTradingAccount, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                if pTradingAccount is None:
                    return
                # 🔴 跨交易日要**重新确认结算结果**（夜盘 21:00 起就算下一交易日）。
                # 结算确认是**按交易日**有效的，柜台切日时会把确认标志重置；不重
                # 确认的话夜盘第一笔报单被打回「[39] 未确认结算结果」，而界面一切
                # 正常 —— 只有下单失败，极难联想到结算。
                # 为什么探针放在**资金**回包上：`TradingDay` 两边都有，但空仓账户
                # 的持仓查询只回一条空包（`pInvestorPosition` 是 None），探不到；
                # 资金查询每 2 秒一轮、永远有内容。
                outer._note_trading_day(_s(getattr(pTradingAccount, "TradingDay", "")))
                with outer._lock:
                    outer.account = {
                        "balance": _f(pTradingAccount.Balance),
                        "available": _f(pTradingAccount.Available),
                        "margin": _f(pTradingAccount.CurrMargin),
                        "float_profit": _f(pTradingAccount.PositionProfit),
                        "position_profit": _f(pTradingAccount.PositionProfit),
                        "close_profit": _f(pTradingAccount.CloseProfit),
                        "commission": _f(pTradingAccount.Commission),
                        # 昨结算权益：界面算「当日盈亏」要用（动态权益 − 昨结算）。
                        # CTP 的 Balance 含浮盈，不减去 PreBalance 会把这部分
                        # 重复算进当日盈亏里
                        "pre_balance": _f(getattr(pTradingAccount, "PreBalance", 0)),
                    }

            def OnRspQryInvestorPosition(self, pInvestorPosition, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                # 柜台不放行 / 查询被打回：**这是「不知道」，不是「账户没仓」**。
                # `positions_loaded` 保持 False，界面据此说「还没读到」，
                # 绝不拿一张空表去报「账户与策略一致」（2026-11-23 融航）。
                err = _err(pRspInfo)
                if err:
                    outer.last_error = f"查持仓失败：{err}"
                    return
                # 🔴 这一轮的第一条回调（哪怕内容为 None，空仓时就是这样）先把上一轮
                # 的清掉，然后整轮往里累加。CTP 一次查询会推**多条**：同一合约同一
                # 方向按今仓 / 昨仓 / 不同开仓日期拆成好几条，`Position` 只是那一条
                # 的量。以前用 `=` 赋值，后一条把前一条盖掉 ——
                #   · 今仓盖掉昨仓（或反过来）⇒ 持仓手数少算，界面上「好几笔成交
                #     都没出现在持仓里」，极端情况下一条 0 手的已平记录把整行清零；
                #   · 浮盈 / 保证金用的是 `+=`，却又从不重置 ⇒ 每查一轮叠一遍，
                #     跑一小时就放大成千上万倍（用户看到「多了几个 0」，收盘后还在涨）。
                with outer._lock:
                    if nRequestID == outer._pos_req and not outer._pos_started:
                        outer.positions.clear()
                        outer._pos_started = True
                if pInvestorPosition is None:
                    # 空仓时柜台只回一条空包（`bIsLast=True`、内容为 None）。
                    # 「查完了、确实一条持仓都没有」是有效结论，必须置真 ——
                    # 否则界面永远停在「还没读到」，真有仓了也不敢说一致
                    if bIsLast:
                        outer.positions_loaded = True
                        outer.pos_at = time.time()
                    return
                # 兜底：主探针在 `OnRspQryTradingAccount`（空仓时持仓查询只回空包，
                # 探不到日期）。这里再补一刀，多重保险且不会重复确认 ——
                # `_note_trading_day` 更新完 `trading_day` 之后，第二个调用方就相等了
                outer._note_trading_day(_s(getattr(pInvestorPosition, "TradingDay", "")))
                ins = _s(pInvestorPosition.InstrumentID)
                pos_dir = _s(getattr(pInvestorPosition, "PosiDirection", ""))
                side = {"2": "long", "3": "short"}.get(pos_dir, "net")
                with outer._lock:
                    row = outer.positions.setdefault(
                        ins, {"instrument": ins, "long": 0, "short": 0,
                              "long_yd": 0, "short_yd": 0,
                              "long_today": 0, "short_today": 0,
                              "price": 0.0, "profit": 0.0, "margin": 0.0,
                              # 多空分开记总成本：算「成交平仓盈亏」要知道**这一侧**
                              # 的开仓均价，混在一个 price 里多空都有仓时必错
                              "cost_long": 0.0, "cost_short": 0.0})
                    n = int(getattr(pInvestorPosition, "Position", 0) or 0)
                    yd = int(getattr(pInvestorPosition, "YdPosition", 0) or 0)
                    # 今仓单独记：上期所平今和平昨是两个指令，界面要能分开看
                    td = int(getattr(pInvestorPosition, "TodayPosition", 0) or max(0, n - yd))
                    # 一律累加，不赋值 —— 理由见上面那段。同一方向的第 N 条记录
                    # 是「又一个批次」，不是「把前面的改掉」。
                    if side == "long":
                        row["long"] += n
                        row["long_yd"] += yd
                        row["long_today"] += td
                    elif side == "short":
                        row["short"] += n
                        row["short_yd"] += yd
                        row["short_today"] += td
                    # OpenCost 是**总成本**（含乘数），不是单价
                    oc = _f(getattr(pInvestorPosition, "OpenCost", 0))
                    row["price"] += oc
                    if side == "long":
                        row["cost_long"] += oc
                    elif side == "short":
                        row["cost_short"] += oc
                    row["profit"] += _f(getattr(pInvestorPosition, "PositionProfit", 0))
                    row["margin"] += _f(getattr(pInvestorPosition, "UseMargin", 0))
                if bIsLast:
                    outer.positions_loaded = True
                    outer.pos_at = time.time()

            def OnRspQryOrder(self, pOrder, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                """补查回来的当日委托。

                字段映射和 `OnRtnOrder` **共用 `_order_row()`**：两份表必然只会
                改一份，改漏的那次就是「时间列里写着拒单理由」。

                和持仓不同，这里**不清空**再累加：`orders` 里已经有本进程刚发的
                单子（`OnRtnOrder` 推来的、`place()` 塞的乐观条目），清掉会让
                还在追的那几笔直接从界面消失。历史行和增量行按 `OrderSysID`
                （没有就用 `OrderRef`）对齐，天然去重。
                """
                err = _err(pRspInfo)
                if err:
                    # 柜台不放行就把原因留下，`orders_loaded` 保持 False，
                    # 由 `CtpHub._query_hist()` 决定还试不试
                    outer.last_error = f"查历史委托失败：{err}"
                    return
                if pOrder is not None:
                    row = _order_row(pOrder)
                    key = row["order_sys_id"] or row["order_ref"]
                    with outer._lock:
                        # 柜台单号为空（还挂在本地队列）时不要覆盖已知的行
                        if key:
                            outer.orders[key] = row
                            if row["order_ref"] and row["order_ref"] != key:
                                outer.orders.setdefault(row["order_ref"], dict(row))
                if bIsLast:
                    outer.orders_loaded = True

            def OnRspQryTrade(self, pTrade, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                """补查回来的当日成交。映射同 `OnRtnTrade`（`_trade_row()`）。"""
                err = _err(pRspInfo)
                if err:
                    outer.last_error = f"查历史成交失败：{err}"
                    return
                if pTrade is not None:
                    row = _trade_row(pTrade)
                    with outer._lock:
                        outer.trades[row["trade_id"]] = row
                if bIsLast:
                    outer.trades_loaded = True

            def OnRspQryInstrument(self, pInstrument, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                # bIsLast 一定要处理：全量查询 1.6 万条，不记「查完了」的话
                # 调用方没有任何办法知道什么时候可以安全下单
                if bIsLast:
                    outer.instruments_done.set()
                if pInstrument is None:
                    return
                ins = _s(pInstrument.InstrumentID)
                if not ins:
                    return
                with outer._lock:
                    outer.instruments[ins] = {
                        "instrument": ins,
                        "exchange": _s(pInstrument.ExchangeID),
                        "tick": _f(getattr(pInstrument, "PriceTick", 0)),
                        "multiple": _f(getattr(pInstrument, "VolumeMultiple", 0)),
                    }

        return _Spi()


# CTP 的 OrderStatus 字符 → 人话
_STATUS = {
    # 这张表必须严格照 CTP 的 OrderStatus 枚举写。**'3' 是「未成交还在队列中」，
    # 是活单**，不是「已撤」—— 之前这里整体错位了一格，界面上活单显示成「已撤单」、
    # 成交单显示成「未成交」，用户会以为单子没了。SimNow 实测才发现。
    "0": "全部成交",
    "1": "部分成交",
    "2": "部分成交(不在队列)",
    "3": "未成交",
    "4": "未成交(不在队列)",
    "5": "已撤单",
    "a": "未知",
    "b": "未触发",
    "c": "已触发",
}

# 还在队列里、可以撤的状态
ALIVE_STATUS = ("未成交", "部分成交", "未触发", "已触发")


# ------------------------------------------------------------------ 组合入口

class CtpSession:
    """一条 CTP 连接：行情 + 交易共用一份配置。

    界面只需要这两样东西：`md.last(instrument)` 拿最新价，`trader.place(...)`
    下单。分开两个对象是因为 CTP 本来就分了两个 API —— 只做行情（不登录交易）
    是常见用法，不该逼着人把资金账号填进来。
    """

    def __init__(self, cfg: CtpConfig) -> None:
        self.cfg = cfg
        self.md = MdClient(cfg)
        self.trader = TraderClient(cfg)
        self.running = False

    def start(self, instruments=(), with_trade: bool = True) -> None:
        self.md.start(instruments)
        if with_trade:
            self.trader.start()
        self.running = True

    def stop(self) -> None:
        self.md.stop()
        self.trader.stop()
        self.running = False

    def status(self) -> str:
        if not HAS_CTP:
            return f"CTP 通道不可用（{CTP_IMPORT_ERROR}）"
        if self.trader.last_error:
            return self.trader.last_error
        if self.md.last_error:
            return self.md.last_error
        if self.trader.settled.is_set():
            return "已连接（可下单）"
        if self.trader.logged_in.is_set():
            return "已登录，正在确认结算"
        if self.trader.connected.is_set():
            return "已连前置，正在认证"
        return "未连接"
