"""把 CTP 柜台接到界面上的那一层。

`ttrade/ctp_client.py` 只管「怎么和柜台说话」，这一层负责把它包装成
**和 `LiveHub` 一样的形状**，这样 `gui/live.py` 的线程模型、`gui/trade_page.py`
的三张表都不用改一行。

对齐的接口：

    start() / close() / subscribe(sym, ex, period) / wait(t) /
    bars() -> DataFrame / state() -> LiveState / place(...) / cancel(...)

三处和天勤不一样，都是 CTP 本身的规矩，不是偷懒：

1. **没有主连，也没有 K 线推送**。CTP 只给 tick。这里的 `bars()` 用 tick 自己
   聚合成周期 bar（成交量按累计量差分），所以研究页的 K 线照样能跟着走。
   合约必须具体（`rb2601`），订阅时给品种（`RB`）会先查主力合约表，查不到就
   明说「请在合约框里填具体合约」。
2. **只收限价单**。`place()` 传 `limit_price=None` 直接报错，不去猜价格 ——
   猜错方向的价格在实盘上是真亏钱。
3. **结算确认之前不能下单**。`wait_ready()` 等的是「登录 + 结算确认」两个
   事件，不是只等登录。

CTP 是**可选依赖**：没装 `openctp-ctp` 时 `CtpHub.start()` 抛 `CtpError`，
界面照常显示、只是连不上。
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrade import symbols_info
from ttrade.ctp_client import (
    ALIVE_STATUS,
    HAS_CTP,
    CtpConfig,
    CtpError,
    MdClient,
    TraderClient,
)
from ttrade.tqlive import LiveState, _mult_of

# 大写合约代码的交易所。CZCE 的合约是 `TA601` 这种，郑商所历史上一直用大写；
# 中金所 `IF2601` 也是大写。其余（上期/大商/能源/广期）全小写。
UPPER_EXCHANGES = ("CZCE", "CFFEX")

# 目标周期 → 秒。CTP tick 自带 `UpdateTime`，按这个秒数对齐切 bar。
PERIOD_SEC = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "1d": 86400}


# ------------------------------------------------------------------ 合约代码


_EX_MAP: dict[str, str] | None = None


def variety_exchange(variety: str) -> str:
    """品种 → 交易所。扫本地 `data_futures/*.csv` 的文件名（`RB_SHFE.csv`）。

    不用写死一张表：新品种上市时只要数据文件在，这里就认得。扫一次缓存住，
    交易页每 0.4 秒刷一次状态，不能每次都 listdir。
    """
    global _EX_MAP
    if _EX_MAP is None:
        out: dict[str, str] = {}
        try:
            from ttrade.paths import data_dir

            for name in os.listdir(data_dir("cn")):
                if not name.endswith(".csv"):
                    continue
                stem = name[:-4]
                if "_" not in stem:
                    continue
                sym, _, ex = stem.partition("_")
                out[sym.upper()] = ex.upper()
        except Exception:  # noqa: BLE001
            out = {}
        _EX_MAP = out
    return _EX_MAP.get(str(variety or "").upper(), "")


def ctp_instrument(contract: str, exchange: str = "") -> str:
    """把平台里的合约写法收敛成 CTP 的 `InstrumentID`。

    认 `rb2601` / `RB2601` / `RB2601.SHF` / `SHFE.rb2601` / `RB` + `SHFE`。
    品种（没有数字）原样返回 —— 调用方要先解析成具体合约，CTP 不认品种。
    """
    s = str(contract or "").strip()
    if not s:
        return ""
    # `RB2601.SHF` / `rb2601.SHF` / `TA2611.ZCE`
    if "." in s:
        head, _, tail = s.partition(".")
        if tail.upper() in ("SHF", "DCE", "ZCE", "CZC", "CFE", "CFX",
                            "INE", "GFE", "SGE"):
            s = head
        elif head.upper() in ("SHFE", "DCE", "CZCE", "CFFEX", "INE", "GFEX"):
            s = tail
    letters = "".join(ch for ch in s if ch.isalpha())
    digits = "".join(ch for ch in s if ch.isdigit())
    ex = str(exchange or "").upper() or variety_exchange(letters)
    if not digits:
        return letters.upper() if ex in UPPER_EXCHANGES else letters.lower()
    # 郑商所的合约在 CTP 里是**三位**月份（`TA611`），Tushare 是四位（`TA2611`）。
    # 多带一位的合约代码在 CTP 上订阅不到、也下不了单，而报错只会说「合约不存在」
    if ex == "CZCE" and len(digits) == 4:
        digits = digits[1:]
    body = f"{letters}{digits}"
    return body.upper() if ex in UPPER_EXCHANGES else body.lower()


def main_contract(symbol: str, exchange: str = "") -> str:
    """当前主力合约，CTP 写法。查不到返回空串。

    数据源是 `data_futures_aux/<品种>_<交易所>.csv` 的最后一行的 `c1_code`
    （Tushare 口径 `RB2601.SHF`）。这是本地文件，不联网、不占行情配额。
    """
    sym = str(symbol or "").strip().upper()
    ex = str(exchange or "").strip().upper()
    if not sym or not ex:
        return ""
    try:
        from ttrade.auxdata import _default_dir

        path = os.path.join(_default_dir(), f"{sym}_{ex}.csv")
        if not os.path.exists(path):
            return ""
        df = pd.read_csv(path, usecols=["date", "c1_code"])
        if df.empty:
            return ""
        return ctp_instrument(str(df["c1_code"].iloc[-1]), ex)
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------ 配置装配


def spec_to_config(spec) -> CtpConfig:
    """`AccountSpec`（kind=ctp）→ `CtpConfig`。

    交易页那一个对话框里填的就是这几项，不用再去 `ctp.json` 挑一份 ——
    两个入口（脚本 / 界面）读同一份结构，但界面以账户表为准。
    """
    cfg = CtpConfig(
        name=getattr(spec, "name", "") or "CTP",
        preset=getattr(spec, "ctp_preset", "") or "simnow_7x24",
        md_front=getattr(spec, "md_front", "") or "",
        td_front=getattr(spec, "td_front", "") or "",
        broker_id=getattr(spec, "broker_id", "") or "9999",
        user_id=getattr(spec, "user", "") or "",
        password=getattr(spec, "password", "") or "",
        app_id=getattr(spec, "app_id", "") or "",
        auth_code=getattr(spec, "auth_code", "") or "",
    )
    return cfg.fill_preset()


# ------------------------------------------------------------------ 行情聚合


@dataclass
class _BarBuilder:
    """把 tick 攒成周期 bar。

    CTP 的 `Volume` 是**当日累计成交量**，不是本 tick 的量 —— 直接拿去当 bar 的
    成交量，画出来是一条单调上升的斜线。这里存上一次的累计值做差分。

    隔夜 / 跨小节时累计量会归零，差分出来是负数，按「本根 bar 的量」处理
    （负数丢掉，用当前累计值兜底）。
    """

    period: str = "1m"
    # 「交易时段」门禁要的品种 / 交易所。空 = 不限制（见 `push`）。
    symbol: str = ""
    exchange: str = ""
    _bars: dict = field(default_factory=dict)
    _last_vol: float = -1.0

    def _bucket(self, stamp: str) -> pd.Timestamp | None:
        """`20261101 14:35:12` → 对齐到周期起点的 Timestamp。

        不要按固定下标切 `HH:MM:SS` —— 冒号也占位，`s[11:13]` 拿到的是 `:3`。
        按空格和冒号分开取，顺便容忍带毫秒的 `14:35:12.500`。
        """
        s = str(stamp or "").strip()
        if not s:
            return None
        try:
            if " " in s:                          # YYYYMMDD HH:MM:SS
                day, _, clock = s.partition(" ")
                hh, mm, ss = (clock.split(".")[0].split(":") + ["0", "0", "0"])[:3]
                ts = pd.Timestamp(
                    f"{day[0:4]}-{day[4:6]}-{day[6:8]} {int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
                )
            elif ":" in s:                        # HH:MM:SS，没有日期就用今天
                hh, mm, ss = (s.split(".")[0].split(":") + ["0", "0", "0"])[:3]
                ts = pd.Timestamp(
                    f"{pd.Timestamp.now():%Y-%m-%d} {int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
                )
            else:
                return None
        except (ValueError, TypeError):
            return None
        sec = PERIOD_SEC.get(self.period, 60)
        if sec >= 86400:
            return ts.normalize()
        return ts.floor(f"{sec}s")

    def push(self, tick: dict) -> None:
        ts = self._bucket(tick.get("datetime", ""))
        if ts is None:
            return
        # 🔴 交易时段门禁：收盘之后柜台还会继续推最后一次快照（模拟环境尤其
        # 明显：`last` 停在那、`volume` 却还变），不挡的话 bar 会一直往前长，
        # 策略就跟着在非交易时段算出新信号、发新单。判据只有一条 ——
        # 品种信息表里的「交易时段」（`symbols_info.in_session`）。
        if not symbols_info.in_session(self.symbol, self.exchange,
                                       ts.to_pydatetime()):
            return
        price = float(tick.get("last") or 0.0)
        if price <= 0:
            return
        vol_total = float(tick.get("volume") or 0.0)
        dv = vol_total - self._last_vol if self._last_vol >= 0 else 0.0
        if dv < 0:                                # 跨日归零
            dv = vol_total
        self._last_vol = vol_total

        bar = self._bars.get(ts)
        if bar is None:
            self._bars[ts] = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": dv,
                "oi": float(tick.get("oi") or 0.0),
            }
            return
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["volume"] += dv
        bar["oi"] = float(tick.get("oi") or bar["oi"])

    def frame(self, limit: int = 400) -> pd.DataFrame:
        if not self._bars:
            return pd.DataFrame()
        items = sorted(self._bars.items())[-limit:]
        df = pd.DataFrame(
            [{"dt": ts, **vals} for ts, vals in items]
        ).set_index("dt")
        df.index.name = "dt"
        return df


# ------------------------------------------------------------------ Hub


class CtpHub:
    """一个 CTP 柜台连接：行情（可选）+ 交易 + 下单。

    和 `LiveHub` 的差别只有「怎么连」和「数据从哪来」，对外的形状完全一致 ——
    `gui/live.py` 的 worker 循环拿到的就是这两样东西。

    **融航走的是同一个类**（`RohonHub` 只覆盖下面四个工厂方法）。两家都是 CTP
    协议的柜台：持仓怎么合并、上期所今昨仓怎么分、委托状态怎么翻成人话，这些
    规则一模一样。为融航再写一份解析，等于把「平今和平昨是两个指令」这类坑再
    踩一遍。
    """

    # 界面上账户行显示的柜台名，也是报错时的主语
    PLATFORM = "CTP"
    KIND = "ctp"

    def __init__(self, specs) -> None:
        if specs is None:
            specs = []
        elif not isinstance(specs, (list, tuple)):
            specs = [specs]
        self.specs = list(specs)
        self.spec = self.specs[0] if self.specs else None
        self.trader: TraderClient | None = None
        self.md: MdClient | None = None
        self.last_error = ""
        self.key = ""
        self.period = ""
        self.instrument = ""            # CTP 合约，如 rb2601
        self.underlying = ""
        self._sym = ""                  # 订阅时传进来的品种（交易时段门禁用）
        self._ex = ""
        self._quote_cache: dict = {}    # 最后一个「交易时段内」的行情快照
        self._builder = _BarBuilder()
        self._last_query = 0.0
        self._query_turn = 1          # 0 = 该发持仓了，1 = 该发资金了
        self._lock = threading.Lock()

    # ---- 柜台适配（融航覆盖这四个） ----

    def check_deps(self) -> None:
        if not HAS_CTP:
            raise CtpError("没装 openctp-ctp，用不了 CTP（pip install openctp-ctp）")

    def make_config(self, spec):
        return spec_to_config(spec)

    def make_trader(self, cfg):
        return TraderClient(cfg)

    def make_md(self, cfg):
        return MdClient(cfg)

    def login_hint(self, cfg) -> str:
        """登录超时时补一句「为什么」。CTP 的 SimNow 有个服务时段问题。"""
        if "182.254.243.31" in str(getattr(cfg, "td_front", "")):
            return ("（SimNow 只在服务时段内可连：第一套 30001 跟实盘交易时段一致，"
                    "7x24 环境 40001 是 16:00 ~ 次日 09:00；周末和收盘后 TCP 会直接被拒）")
        return ""

    # ---- 生命周期 ----

    def start(self) -> None:
        self.check_deps()
        if not self.specs:
            raise CtpError(f"{self.PLATFORM} 会话至少要有一个账户")
        cfg = self.make_config(self.specs[0])
        if not cfg.user_id or not cfg.password:
            raise ValueError(f"账户「{cfg.name}」缺少投资者账号或密码")
        if not cfg.td_front:
            raise ValueError(f"{self.PLATFORM} 交易前置地址为空")

        trader = self.make_trader(cfg)
        trader.start()
        if not trader.wait_ready(15.0):
            err = trader.last_error or "登录超时（15 秒）"
            trader.stop()
            raise CtpError(err + self.login_hint(cfg))
        trader.query_account()
        trader.query_positions()
        # 撤单要交易所代码，但**不查全量合约表**：SimNow 上是 1.6 万条、80 秒，
        # 而 CTP 一个会话是串行的 —— 流没结束之前报单会一直排队，界面表现就是
        # 「点了下单没反应」。改成用本地品种→交易所映射兜底，具体合约的
        # 最小变动价位在订阅时按单个合约查（秒回）。
        trader.exchange_resolver = lambda ins: variety_exchange(
            "".join(ch for ch in str(ins) if ch.isalpha()))
        self.trader = trader

        # 行情是可选：只做交易不看盘也能用。连不上就只报一条日志，不影响下单
        if cfg.md_front:
            md = self.make_md(cfg)
            try:
                md.start()
                if md.wait_ready(10.0):
                    self.md = md
                else:
                    self.last_error = f"行情未就绪：{md.last_error}"
                    md.stop()
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"行情连接失败：{exc}"
                try:
                    md.stop()
                except Exception:  # noqa: BLE001
                    pass

    def close(self) -> None:
        md, self.md = self.md, None
        trader, self.trader = self.trader, None
        for cli in (md, trader):
            if cli is not None:
                try:
                    cli.stop()
                except Exception:  # noqa: BLE001
                    pass

    def subscribe(self, symbol: str, exchange: str, period: str) -> None:
        """订阅一个合约的 tick。`symbol` 可以是品种（自动查主力）或具体合约。"""
        self.period = str(period or "1m")
        self._sym, self._ex = symbol, exchange
        self._builder = _BarBuilder(period=self.period, symbol=symbol,
                                   exchange=exchange)
        self._quote_cache = {}
        ins = ctp_instrument(symbol, exchange)
        if not ins:
            raise ValueError("合约代码为空")
        # 没有数字 = 还是品种，去本地主力表查一个具体合约
        if not any(ch.isdigit() for ch in ins):
            found = main_contract(symbol, exchange)
            if not found:
                raise ValueError(
                    f"{self.PLATFORM} 没有主连，要用具体合约（如 rb2601）。"
                    f"查不到 {symbol} 的主力合约表，请在下单面板的「合约」里填具体合约"
                )
            ins = found
        self.instrument = ins
        self.underlying = ins
        self.key = f"{ins}_{exchange}" if exchange else ins
        if self.md is not None:
            self.md.subscribe([ins])
        # 只查这一个合约（最小变动价位 / 合约乘数 / 交易所），秒回。
        # 不查全量 —— 理由见 start() 里的注释。
        if self.trader is not None:
            try:
                self.trader.query_instruments(ins)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"查合约信息失败：{exc}"

    def wait(self, timeout: float = 0.8) -> bool:
        """推进一步：读 tick 攒 bar，顺手每 2 秒刷一次资金和持仓。

        资金/持仓是**主动查询**（CTP 没有推送账户变动）。这里有两个坑，都是
        实测踩出来的：

        1. **柜台限制「每秒最多一个查询请求」**。资金和持仓如果背靠背连着发，
           第二个会被静默丢掉 —— 表现是「资金一直在动，持仓永远是 0」。所以
           两个查询错开 1 秒轮流发。
        2. 查太勤会打满流量，2 秒一轮在实盘里够用。
        """
        time.sleep(max(0.05, float(timeout)))
        if self.md is not None and self.instrument:
            tick = self.md.last(self.instrument)
            if tick:
                self._builder.push(tick)
        now = time.monotonic()
        if self.trader is None:
            return True
        try:
            # 资金和持仓交替发，中间至少隔 1 秒 —— 见上面第 1 条
            if now - self._last_query > 2.0:
                self._last_query = now
                self.trader.query_account()
                self._query_turn = 0
            elif self._query_turn == 0 and now - self._last_query > 1.0:
                self._query_turn = 1
                self.trader.query_positions()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"查询失败：{exc}"
        return True

    def bars(self) -> pd.DataFrame:
        return self._builder.frame()

    def quote_of(self, symbol: str, timeout: float = 2.0) -> dict:
        """取任意一个合约的最新 tick，给下单面板「填最新价」用。

        和天勤那边同一个用途。CTP 没有「查一下报价」的接口，只能订阅然后等柜台
        推 —— 订阅是幂等的，已经订过的合约再订一次没有副作用。

        拿不到就返回空 dict（不在交易时段、合约代码不对、柜台没开行情都是这个
        结果）。**不返回 0**：0 价的限价单会被柜台按「价格不合法」拒掉，而界面上
        看着像是填好了。

        必须在 worker 线程里调（要 `sleep` 等推送）。
        """
        if self.md is None:
            raise CtpError(f"{self.PLATFORM} 这条会话没有行情连接，取不到最新价")
        ins = ctp_instrument(symbol)
        if not ins:
            raise CtpError("合约代码为空")
        self.md.subscribe([ins])
        end = time.time() + max(0.5, float(timeout))
        tick: dict = {}
        while time.time() < end:
            tick = self.md.last(ins) or {}
            if float(tick.get("last") or 0) > 0:
                break
            time.sleep(0.2)
        last = float(tick.get("last") or 0)
        if last <= 0:
            return {}
        return {
            "symbol": ins,
            "instrument": ins,
            "last": last,
            "bid": float(tick.get("bid") or 0),
            "ask": float(tick.get("ask") or 0),
            "tick": 0.0,
            "datetime": str(tick.get("datetime") or ""),
            "source": "live",
        }

    # ---- 读 ----

    def state(self) -> LiveState:
        st = LiveState(connected=bool(self.trader and self.trader.logged_in.is_set()),
                       error=self.last_error, key=self.key, period=self.period,
                       underlying=self.underlying)
        if self.trader is None:
            return st
        snap = self.trader.snapshot()
        account = self._account(snap.get("account") or {})
        positions = self._positions(snap.get("positions") or {}, account)
        orders = self._orders(snap.get("orders") or {})
        trades = self._trades(snap.get("trades") or {})
        name = self.spec.name if self.spec is not None else self.PLATFORM
        row = self._account_row(name, account, positions, trades)
        st.account = account
        st.positions = positions
        st.orders = orders
        st.trades = trades
        st.accounts = [row]
        st.by_account = {
            name: {"spec": {"name": name, "kind": self.KIND},
                   "account": account, "positions": positions,
                   "orders": orders, "trades": trades}
        }
        st.quote = self._quote()
        return st

    def _mult(self, instrument: str) -> float:
        """合约乘数。`_mult_of` 认的是 `SHFE.rb2601` 这种带交易所的写法，
        CTP 给的是光秃秃的 `rb2601` —— 直接传进去乘数会变成 1.0，杠杆和
        成交金额就全错一个数量级。所以先补上交易所。
        """
        ins = str(instrument or "")
        ex = ""
        if self.trader is not None:
            ex = str((self.trader.instruments.get(ins) or {}).get("exchange") or "")
        if not ex:
            ex = variety_exchange("".join(ch for ch in ins if ch.isalpha()))
        try:
            return float(_mult_of(f"{ex}.{ins}") if ex else _mult_of(ins)) or 1.0
        except Exception:  # noqa: BLE001
            return 1.0

    def _quote(self) -> dict:
        if self.md is None or not self.instrument:
            return {}
        # 🔴 交易时段门禁：收盘之后 CTP 还会推最后一次快照，`last` 不跳了但
        # `volume` / `oi` 还在动 —— 直接透传的话界面上的「实时行情」看着还在
        # 更新，用户以为还在交易。这里**冻结在最后一笔时段内快照**：
        # 开市时更新缓存，收市时把缓存原样还回去。
        if not symbols_info.in_session(self._sym, self._ex) and self._quote_cache:
            return dict(self._quote_cache)
        t = self.md.last(self.instrument) or {}
        info = {}
        if self.trader is not None:
            info = self.trader.instruments.get(self.instrument) or {}
        q = {
            "symbol": self.instrument,
            "underlying": self.instrument,
            "name": self.instrument,
            "datetime": t.get("datetime", ""),
            "last": float(t.get("last") or 0.0),
            "bid": float(t.get("bid") or 0.0),
            "ask": float(t.get("ask") or 0.0),
            "bid_vol": float(t.get("bid_vol") or 0.0),
            "ask_vol": float(t.get("ask_vol") or 0.0),
            "open": float(t.get("open") or 0.0),
            "high": float(t.get("high") or 0.0),
            "low": float(t.get("low") or 0.0),
            "pre_close": float(t.get("pre_close") or 0.0),
            "volume": float(t.get("volume") or 0.0),
            "oi": float(t.get("oi") or 0.0),
            "price_tick": float(info.get("tick") or 0.0),
            "multiple": float(info.get("multiple") or 0.0),
            "expired": False,
        }
        # 只在开市时刷新。上面那条「收市直接还缓存」靠的就是它 —— 收盘那一刻
        # 手上这份就是最后一笔真行情，之后柜台再推什么都不再采信。
        self._quote_cache = q
        return dict(q)

    def _account(self, acc: dict) -> dict:
        """CTP 资金 → 平台口径。

        `day_profit` 用「动态权益 − 昨结算」算，和天勤那边同一个定义；CTP 的
        `PreBalance` 是昨结算，`Balance` 是动态权益（含浮盈）。
        """
        bal = float(acc.get("balance") or 0.0)
        pre = float(acc.get("pre_balance") or 0.0)
        margin = float(acc.get("margin") or 0.0)
        return {
            "balance": bal,
            "available": float(acc.get("available") or 0.0),
            "margin": margin,
            "float_profit": float(acc.get("float_profit") or 0.0),
            "position_profit": float(acc.get("position_profit") or 0.0),
            "close_profit": float(acc.get("close_profit") or 0.0),
            "commission": float(acc.get("commission") or 0.0),
            "risk_ratio": (margin / bal) if bal > 0 else 0.0,
            "pre_balance": pre,
            "static_balance": pre,
            "market_value": 0.0,
        }

    def _positions(self, src: dict, account: dict) -> list[dict]:
        out = []
        for ins, p in src.items():
            long_q = int(p.get("long") or 0)
            short_q = int(p.get("short") or 0)
            if not long_q and not short_q:
                continue
            long_td = max(0, long_q - int(p.get("long_yd") or 0))
            short_td = max(0, short_q - int(p.get("short_yd") or 0))
            mult = self._mult(ins)
            cost = float(p.get("price") or 0.0)          # CTP 的 OpenCost 是总成本
            total = long_q + short_q
            avg = (cost / (total * mult)) if total and cost else 0.0
            out.append({
                "symbol": ins,
                "pos": long_q - short_q,
                "pos_long": long_q,
                "pos_short": short_q,
                "pos_long_today": long_td,
                "pos_short_today": short_td,
                "open_price_long": avg if long_q else 0.0,
                "open_price_short": avg if short_q else 0.0,
                "open_cost_long": cost if long_q else 0.0,
                "open_cost_short": cost if short_q else 0.0,
                "float_profit": float(p.get("profit") or 0.0),
                "float_profit_long": float(p.get("profit") or 0.0) if long_q else 0.0,
                "float_profit_short": float(p.get("profit") or 0.0) if short_q else 0.0,
                "position_profit": float(p.get("profit") or 0.0),
                "margin": float(p.get("margin") or 0.0),
                "last_price": 0.0,
            })
        out.sort(key=lambda r: r["symbol"])
        return out

    def _orders(self, src: dict) -> list[dict]:
        out, seen = [], set()
        for oid, o in src.items():
            # CTP 的委托在 orders 里同时以 OrderRef 和 OrderSysID 各存一份
            key = (o.get("order_sys_id") or o.get("order_ref") or oid,
                   o.get("instrument"), o.get("volume"), o.get("price"))
            if key in seen:
                continue
            seen.add(key)
            status = str(o.get("status") or "")
            vol = int(o.get("volume") or 0)
            traded = int(o.get("traded") or 0)
            out.append({
                "order_id": o.get("order_sys_id") or o.get("order_ref") or oid,
                "symbol": str(o.get("instrument") or ""),
                "direction": {"0": "BUY", "1": "SELL"}.get(str(o.get("direction")), str(o.get("direction") or "")),
                "offset": {"0": "OPEN", "1": "CLOSE", "3": "CLOSETODAY"}.get(
                    str(o.get("offset")), str(o.get("offset") or "")),
                "volume_orign": vol,
                "volume_left": max(0, vol - traded),
                "limit_price": float(o.get("price") or 0.0),
                "status": "ALIVE" if status in ALIVE_STATUS else status,
                # 拒单理由全在 StatusMsg 里（「50:已撤单报单被拒绝SHFE:价格跌破跌停板」），
                # 只给「已撤单」等于什么都没说 —— 用户没法判断是价格错了还是权限不够
                "last_msg": str(o.get("msg") or status),
                "insert_date_time": 0.0,
                "time": str(o.get("msg") or ""),
            })
        # 未成交的排前面：撤单要点的就是它们
        out.sort(key=lambda r: (r["status"] != "ALIVE", r["symbol"]))
        return out[:200]

    def _trades(self, src: dict) -> list[dict]:
        out = []
        for tid, t in src.items():
            out.append({
                "trade_id": str(t.get("trade_id") or tid),
                "order_id": str(t.get("order_ref") or ""),
                "symbol": str(t.get("instrument") or ""),
                "direction": {"0": "BUY", "1": "SELL"}.get(str(t.get("direction")), str(t.get("direction") or "")),
                "offset": {"0": "OPEN", "1": "CLOSE", "3": "CLOSETODAY"}.get(
                    str(t.get("offset")), str(t.get("offset") or "")),
                "price": float(t.get("price") or 0.0),
                "volume": int(t.get("volume") or 0),
                "trade_date_time": 0.0,
                "time": str(t.get("time") or ""),
            })
        out.sort(key=lambda r: r["time"], reverse=True)
        return out[:200]

    def _account_row(self, name: str, account: dict, positions: list[dict],
                     trades: list[dict]) -> dict:
        mv_long = mv_short = 0.0
        for p in positions:
            mult = self._mult(p["symbol"])
            mv_long += p["open_cost_long"] * mult + p["float_profit_long"]
            mv_short += p["open_cost_short"] * mult + p["float_profit_short"]
        bal = float(account.get("balance") or 0.0)
        amount = sum(t["price"] * t["volume"] * self._mult(t["symbol"])
                     for t in trades)
        return {
            "name": name,
            "platform": self.PLATFORM,
            "account_id": (self.spec.user if self.spec is not None else "") or "—",
            "balance": bal,
            "day_profit": bal - float(account.get("pre_balance") or bal),
            "net_leverage": (abs(mv_long - mv_short) / bal) if bal > 0 else 0.0,
            "gross_leverage": ((mv_long + mv_short) / bal) if bal > 0 else 0.0,
            "available": float(account.get("available") or 0.0),
            "turnover": amount,
            "margin": float(account.get("margin") or 0.0),
            "commission": float(account.get("commission") or 0.0),
            "positions": len(positions),
            "orders": len(trades),
        }

    # ---- 写 ----

    def place(self, symbol: str, direction: str, offset: str, volume: int,
              limit_price: float | None = None, account: str = "") -> str:
        if self.trader is None:
            raise CtpError("还没连接")
        ins = ctp_instrument(symbol)
        if not ins:
            raise CtpError("合约代码为空")
        if limit_price is None or float(limit_price) <= 0:
            raise CtpError(f"{self.PLATFORM} 只收限价单：请填价格，别用市价")
        # 价格按最小变动价位对齐，否则柜台拒单理由是「价格不是最小变动价位整数倍」
        tick = 0.0
        info = self.trader.instruments.get(ins)
        if info:
            tick = float(info.get("tick") or 0.0)
        price = float(limit_price)
        if tick > 0:
            price = round(round(price / tick) * tick, 8)
        return self.trader.place(ins, direction, offset, price, int(volume))

    def cancel(self, order_id: str, account: str = "") -> None:
        if self.trader is None:
            raise CtpError("还没连接")
        oid = str(order_id)
        snap = self.trader.snapshot()
        target = None
        for o in (snap.get("orders") or {}).values():
            if oid in (str(o.get("order_sys_id") or ""), str(o.get("order_ref") or "")):
                target = o
                break
        if target is None:
            raise CtpError(f"找不到委托 {oid}")
        self.trader.cancel(
            str(target.get("instrument") or ""),
            order_sys_id=str(target.get("order_sys_id") or ""),
            order_ref=str(target.get("order_ref") or ""),
        )

    def set_target(self, symbol: str, volume: int, account: str = "") -> None:
        raise CtpError(f"{self.PLATFORM} 没有目标仓接口，请直接下单")


# ------------------------------------------------------------------ 融航


class RohonHub(CtpHub):
    """融航资管中台。除了「怎么连」以外，全部复用 `CtpHub`。

    融航（Rohon）对外是 CTP 协议的中台：同样的前置地址、同样的报单字段、同样的
    委托状态字符。所以持仓合并、上期所今昨仓拆分、成交金额折算这些规则一条都
    不用重写 —— 只换底下的三个客户端。

    驱动层用 `vnpy_rohon.api`（两个 pybind 模块），**不用 vnpy 框架**：vnpy 4.5.0
    把 `pyside6==6.8.2.1` 钉成硬依赖，装上就会把本程序的 Qt 降级。详见
    `ttrade/rohon_client.py` 开头的说明。
    """

    PLATFORM = "融航"
    KIND = "rohon"

    def check_deps(self) -> None:
        from .rohon_client import HAS_ROHON, ROHON_IMPORT_ERROR

        if not HAS_ROHON:
            raise CtpError(
                f"没装 vnpy_rohon，用不了融航（{ROHON_IMPORT_ERROR}）"
                "。装法：pip install vnpy_rohon==6.7.2.5 --no-deps"
            )

    def make_config(self, spec):
        from .rohon_client import spec_to_config as _rohon_config

        return _rohon_config(spec)

    def make_trader(self, cfg):
        from .rohon_client import RohonTraderClient

        return RohonTraderClient(cfg)

    def make_md(self, cfg):
        from .rohon_client import RohonMdClient

        return RohonMdClient(cfg)

    def login_hint(self, cfg) -> str:
        """融航没有 SimNow 那种「按时段开放」的问题，补的是缺字段的提示。"""
        miss = [k for k in ("broker_id", "td_front")
                if not getattr(cfg, k, "")]
        if miss:
            return f"（还缺：{'、'.join(miss)}。融航要填经纪商代码和交易服务器）"
        return "（检查经纪商代码 / 交易服务器 / 产品名称 / 授权编码是否和资管后台一致）"
