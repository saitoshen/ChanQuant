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

try:  # 可选依赖：装不上就降级，别让整个程序 import 失败
    from openctp_ctp import thostmduserapi as mdapi
    from openctp_ctp import thosttraderapi as tdapi

    HAS_CTP = True
    CTP_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - ImportError 之外还可能是 DLL 加载失败
    mdapi = tdapi = None
    HAS_CTP = False
    CTP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------------------ 配置

# SimNow 的公开模拟环境。7x24 那组随时能连（周末也能），交易时段那组只在
# 期货交易时间内可用 —— 自测时连 7x24，别拿它下真单（撮合是假的）。
PRESETS = {
    "simnow_7x24": {
        "label": "SimNow 7x24（随时可连）",
        "md_front": "tcp://180.168.146.187:10131",
        "td_front": "tcp://180.168.146.187:10130",
        "broker_id": "9999",
        "app_id": "simnow_client_test",
        "auth_code": "0000000000000000",
    },
    "simnow_day": {
        "label": "SimNow 交易时段",
        "md_front": "tcp://180.168.146.187:10111",
        "td_front": "tcp://180.168.146.187:10110",
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
        if not HAS_CTP:
            raise CtpError(f"没装 openctp-ctp，用不了 CTP（{CTP_IMPORT_ERROR}）")
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
                        "volume": _f(d.Volume),
                        "oi": _f(d.OpenInterest),
                    }

        return _Spi()

    def subscribe(self, instruments) -> None:
        if self.api is None or not instruments:
            return
        try:
            self.api.SubscribeMarketData([i.encode("utf-8") for i in instruments], 0)
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
        self.orders: dict[str, dict] = {}
        self.trades: dict[str, dict] = {}
        self.positions: dict[str, dict] = {}
        self.account: dict = {}
        self.instruments: dict[str, dict] = {}
        self.front_id = 0
        self.session_id = 0
        self._lock = threading.Lock()
        self._order_ref = int(time.time()) % 100000
        self._spi = None

    # ---- 生命周期 ----

    def start(self) -> None:
        if not HAS_CTP:
            raise CtpError(f"没装 openctp-ctp，用不了 CTP（{CTP_IMPORT_ERROR}）")
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

    # ---- 下单 ----

    def place(self, instrument: str, direction: str, offset: str,
              price: float, volume: int) -> str:
        """报单。返回 OrderRef（CTP 侧的本地单号）。

        `direction` 用 `BUY` / `SELL`，`offset` 用 `OPEN` / `CLOSE` / `CLOSETODAY`，
        和平台其他地方的写法一致，转换在这里做。
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
        if price <= 0:
            raise CtpError("CTP 只收限价单，价格必须大于 0")

        self._order_ref += 1
        ref = str(self._order_ref)
        req = tdapi.CThostFtdcInputOrderField()
        req.BrokerID = self.cfg.broker_id
        req.InvestorID = self.cfg.user_id
        req.UserID = self.cfg.user_id
        req.InstrumentID = instrument
        req.OrderRef = ref
        req.OrderPriceType = tdapi.THOST_FTDC_OPT_LimitPrice
        req.LimitPrice = float(price)
        req.Direction = d
        req.CombOffsetFlag = {"0": tdapi.THOST_FTDC_OF_Open,
                              "1": tdapi.THOST_FTDC_OF_Close,
                              "3": tdapi.THOST_FTDC_OF_CloseToday}[off]
        # 投机单。套保 / 套利要提前在柜台备案，写死投机最省事也最不容易被拒
        req.CombHedgeFlag = tdapi.THOST_FTDC_HF_Speculation
        req.VolumeTotalOriginal = int(volume)
        req.TimeCondition = tdapi.THOST_FTDC_TC_GFD      # 当日有效
        req.VolumeCondition = tdapi.THOST_FTDC_VC_AV      # 任意数量成交
        req.MinVolume = 1
        req.ContingentCondition = tdapi.THOST_FTDC_CC_Immediately
        req.ForceCloseReason = tdapi.THOST_FTDC_FCC_NotForceClose
        req.IsAutoSuspend = 0
        req.UserForceClose = 0
        self.api.ReqOrderInsert(req, self._next_req())
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
        self.api.ReqOrderAction(req, self._next_req())

    # ---- 查询 ----

    def query_account(self) -> None:
        if self.api is not None:
            self.api.ReqQryTradingAccount(tdapi.CThostFtdcQryTradingAccountField(), self._next_req())

    def query_positions(self) -> None:
        if self.api is None:
            return
        req = tdapi.CThostFtdcQryInvestorPositionField()
        req.BrokerID = self.cfg.broker_id
        req.InvestorID = self.cfg.user_id
        self.api.ReqQryInvestorPosition(req, self._next_req())

    def query_instruments(self) -> None:
        if self.api is not None:
            self.api.ReqQryInstrument(tdapi.CThostFtdcQryInstrumentField(), self._next_req())

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
            }

    # ---- 内部 ----

    def _next_req(self) -> int:
        self._req = getattr(self, "_req", 0) + 1
        return self._req

    def _exchange_of(self, instrument: str) -> str:
        """合约 → 交易所。有查询到的合约表就用表，没有就按品种前缀猜。

        猜错会导致撤单报「合约不存在」，所以只在查表失败时兜底 —— 表是登录后
        `query_instruments()` 拉回来的，实盘建议先查一次。
        """
        info = self.instruments.get(instrument)
        if info and info.get("exchange"):
            return info["exchange"]
        from ttrade.symbols_info import EXCHANGES  # noqa: F401 - 只为了保持口径一致

        letters = "".join(ch for ch in instrument if ch.isalpha())
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
        return "SHFE"

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
                outer.logged_in.set()
                try:
                    req = tdapi.CThostFtdcSettlementInfoConfirmField()
                    req.BrokerID = outer.cfg.broker_id
                    req.InvestorID = outer.cfg.user_id
                    outer.api.ReqSettlementInfoConfirm(req, outer._next_req())
                except Exception as exc:  # noqa: BLE001
                    outer.last_error = f"结算确认请求失败：{exc}"

            def OnRspSettlementInfoConfirm(self, pSettlementInfoConfirm, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                err = _err(pRspInfo)
                if err:
                    outer.last_error = f"结算确认失败：{err}"
                    return
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
                    }
                if err:
                    outer.last_error = f"下单被拒：{err}"

            def OnRtnOrder(self, pOrder):  # noqa: N802
                sys_id = _s(getattr(pOrder, "OrderSysID", ""))
                ref = _s(getattr(pOrder, "OrderRef", ""))
                key = sys_id or ref
                row = {
                    "order_ref": ref,
                    "order_sys_id": sys_id,
                    "instrument": _s(getattr(pOrder, "InstrumentID", "")),
                    "direction": _s(getattr(pOrder, "Direction", "")),
                    "offset": _s(getattr(pOrder, "CombOffsetFlag", "")),
                    "price": float(getattr(pOrder, "LimitPrice", 0) or 0),
                    "volume": int(getattr(pOrder, "VolumeTotalOriginal", 0) or 0),
                    "traded": int(getattr(pOrder, "VolumeTraded", 0) or 0),
                    "status": _STATUS.get(_s(getattr(pOrder, "OrderStatus", "")), "未知"),
                    "msg": _s(getattr(pOrder, "StatusMsg", "")),
                }
                with outer._lock:
                    outer.orders[key] = row
                    if ref and ref != key:
                        outer.orders[ref] = dict(row)

            def OnRtnTrade(self, pTrade):  # noqa: N802
                tid = _s(getattr(pTrade, "TradeID", "")) or str(time.time())
                row = {
                    "trade_id": tid,
                    "order_ref": _s(getattr(pTrade, "OrderRef", "")),
                    "instrument": _s(getattr(pTrade, "InstrumentID", "")),
                    "direction": _s(getattr(pTrade, "Direction", "")),
                    "offset": _s(getattr(pTrade, "OffsetFlag", "")),
                    "price": float(getattr(pTrade, "Price", 0) or 0),
                    "volume": int(getattr(pTrade, "Volume", 0) or 0),
                    "time": _s(getattr(pTrade, "TradeTime", "")),
                }
                with outer._lock:
                    outer.trades[tid] = row

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
                with outer._lock:
                    outer.account = {
                        "balance": _f(pTradingAccount.Balance),
                        "available": _f(pTradingAccount.Available),
                        "margin": _f(pTradingAccount.CurrMargin),
                        "float_profit": _f(pTradingAccount.PositionProfit),
                        "close_profit": _f(pTradingAccount.CloseProfit),
                        "commission": _f(pTradingAccount.Commission),
                    }

            def OnRspQryInvestorPosition(self, pInvestorPosition, pRspInfo, nRequestID, bIsLast):  # noqa: N802
                if pInvestorPosition is None:
                    return
                ins = _s(pInvestorPosition.InstrumentID)
                pos_dir = _s(getattr(pInvestorPosition, "PosiDirection", ""))
                side = {"2": "long", "3": "short"}.get(pos_dir, "net")
                with outer._lock:
                    row = outer.positions.setdefault(
                        ins, {"instrument": ins, "long": 0, "short": 0,
                              "long_yd": 0, "short_yd": 0, "price": 0.0,
                              "profit": 0.0, "margin": 0.0})
                    n = int(getattr(pInvestorPosition, "Position", 0) or 0)
                    yd = int(getattr(pInvestorPosition, "YdPosition", 0) or 0)
                    if side == "long":
                        row["long"] = n
                        row["long_yd"] = yd
                    elif side == "short":
                        row["short"] = n
                        row["short_yd"] = yd
                    row["price"] = _f(getattr(pInvestorPosition, "OpenCost", 0)) or row["price"]
                    row["profit"] += _f(getattr(pInvestorPosition, "PositionProfit", 0))
                    row["margin"] += _f(getattr(pInvestorPosition, "UseMargin", 0))

            def OnRspQryInstrument(self, pInstrument, pRspInfo, nRequestID, bIsLast):  # noqa: N802
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
    "0": "未成交",
    "1": "部分成交",
    "2": "全部成交",
    "3": "未成交(已撤)",
    "4": "已撤单",
    "5": "已撤单",
    "a": "未知",
    "b": "未成交(未触发)",
    "c": "已撤单(未触发)",
}


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
            return f"没装 openctp-ctp（{CTP_IMPORT_ERROR}）"
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
