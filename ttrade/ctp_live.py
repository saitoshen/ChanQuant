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

from ttrade import ctp_channel, hubreg, symbols_info
from ttrade import ctp_client as ctp_client_mod
from ttrade.ctp_client import (
    ALIVE_STATUS,
    CtpConfig,
    CtpError,
    MdClient,
    TraderClient,
)
from ttrade.tqlive import (LiveState, _last_daily_close, _mult_of, est_close_pnl,
                           est_commission, replay_close_pnl, update_open_avg)

# 大写合约代码的交易所。CZCE 的合约是 `TA601` 这种，郑商所历史上一直用大写；
# 中金所 `IF2601` 也是大写。其余（上期/大商/能源/广期）全小写。
UPPER_EXCHANGES = ("CZCE", "CFFEX")

# 目标周期 → 秒。CTP tick 自带 `UpdateTime`，按这个秒数对齐切 bar。
PERIOD_SEC = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "1d": 86400}

#: 登录后补查「当日委托 / 当日成交」最多试几拍。试不成就是柜台不放行这条
#: 查询（有的中台只开资金和持仓），要记一条原因而不是一直重试。见 `_query_hist()`
HIST_TRIES = 4


# ------------------------------------------------------------------ 合约代码


_EX_MAP: dict[str, str] | None = None


def _ctp_stamp(day: str, hms: str) -> str:
    """CTP 的 `TradingDay` + `InsertTime` → `MM-DD HH:MM:SS`。

    `InsertTime` 只有 `HH:MM:SS`，跨夜盘就得靠 `TradingDay`（`YYYYMMDD`）补
    日期；只有一个时给哪个算哪个，都没有返回空串。
    **绝不退化成返回委托说明** —— 那种「时间列里写着拒单理由」的样子，正是
    这一列原先的 bug。
    """
    d = "".join(ch for ch in str(day or "") if ch.isdigit())
    t = str(hms or "").strip()
    if len(d) >= 8 and t:
        return f"{d[4:6]}-{d[6:8]} {t}"
    return t


def _ctp_ts(day: str, hms: str) -> str:
    """排序用的时间键：`YYYYMMDDHH:MM:SS`，比字符串大小就等于比时间先后。"""
    d = "".join(ch for ch in str(day or "") if ch.isdigit())
    return f"{d}{str(hms or '').strip()}"


def variety_exchange(variety: str) -> str:
    """品种 → 交易所。扫本地 `data_futures/*.csv` 的文件名（`RB_SHFE.csv`）。

    不用写死一张表：新品种上市时只要数据文件在，这里就认得。扫一次缓存住，
    交易页每 0.4 秒刷一次状态，不能每次都 listdir。

    文件名有两种形状，都要认：

    - `RB_SHFE.csv` —— 品种（绝大多数）
    - `TL0_CFFEX.csv` —— 主力连续代号被当成品种名的那种（30 年期国债在数据源里
      叫 `TL0`）。文件已正名为 `TL_CFFEX.csv`，但老机器/老备份里可能还是旧名，
      这里顺手把尾部数字剥掉再登记一次，免得 `variety_exchange("TL")` 返回空、
      `ctp_instrument` 拼出小写 `tlcffex`，柜台回「[148] 无效的 ExchangeID」
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
                # rpartition：品种名里不带下划线，但将来带了也不会切错边
                sym, _, ex = stem.rpartition("_")
                sym, ex = sym.strip().upper(), ex.strip().upper()
                if not sym or not ex:
                    continue
                out[sym] = ex
                base = sym.rstrip("0123456789")
                if base and base != sym:
                    out.setdefault(base, ex)     # TL0_CFFEX → TL 也认得
        except Exception:  # noqa: BLE001
            out = {}
        _EX_MAP = out
    return _EX_MAP.get(str(variety or "").upper(), "")


#: 交易所短码 / 各种写法 → CTP 的 `ExchangeID`。天勤写 `SHF` / `CFX`，
#: 平台内部写 `SHFE` / `CFFEX`，柜台只认后者
_EX_SHORT = {
    "SHF": "SHFE", "SHFE": "SHFE",
    "DCE": "DCE",
    "ZCE": "CZCE", "CZC": "CZCE", "CZCE": "CZCE",
    "CFE": "CFFEX", "CFX": "CFFEX", "CFFEX": "CFFEX",
    "INE": "INE",
    "GFE": "GFEX", "GFEX": "GFEX",
    "SGE": "SGE",
}


def looks_contract(code: str) -> bool:
    """这段代码是**具体合约**还是品种（含主力连续）？

    `TL2612` / `rb2601` / `TA611` 是合约；`TL` / `TL0` / `RB0` 是品种。
    判据不能只看「有没有数字」：主力连续代号就是 `品种 + 0`（数据源里的 `TL0`），
    把它当合约会绕开「查主力」那一层，最后把 `TL0` 这种不存在的合约发到柜台
    （2026-11-19：TL 策略 [148] 无效 ExchangeID 的另一半原因）。
    """
    digits = "".join(ch for ch in str(code or "") if ch.isdigit())
    return bool(digits) and digits.strip("0") != ""


def _parse_contract(contract: str, exchange: str = "") -> tuple[str, str, str, str]:
    """合约 / 品种键 → `(代码, 字母, 数字, 交易所)`。

    剥掉两种前后缀，都见过：

    - 点号：`RB2601.SHF` / `SHFE.rb2601` / `CFFEX.TL2612`
    - 下划线：`RB_SHFE` / `TL0_CFFEX` —— 这是**平台内部策略里的品种键**，
      以前只认点号，于是 `TL0_CFFEX` 被当成一个合约名，字母串拼成
      `TLCFFEX`，交易所查不到、数字又只剩那个 `0`，最后发出 `tlcffex0`
      （2026-11-19 TL 策略 `[148] 无效的 ExchangeID` 就是这个）
    """
    s = str(contract or "").strip()
    hint = ""
    if s and "." in s:
        head, _, tail = s.partition(".")
        if tail.upper() in _EX_SHORT:
            hint, s = _EX_SHORT[tail.upper()], head
        elif head.upper() in _EX_SHORT:
            hint, s = _EX_SHORT[head.upper()], tail
    if s and "_" in s:
        head, _, tail = s.rpartition("_")
        if tail.upper() in _EX_SHORT:
            hint, s = hint or _EX_SHORT[tail.upper()], head
    letters = "".join(ch for ch in s if ch.isalpha())
    digits = "".join(ch for ch in s if ch.isdigit())
    ex = str(exchange or "").upper() or hint or variety_exchange(letters)
    if ex in _EX_SHORT:                     # 传进来的可能是 `CFX` / `SHF`
        ex = _EX_SHORT[ex]
    return s, letters, digits, ex


def exchange_of_contract(contract: str, exchange: str = "") -> str:
    """合约 / 品种键 → CTP 的交易所代码（`CFFEX.TL2612` / `TL0_CFFEX` / `TL`）。"""
    return _parse_contract(contract, exchange)[3]


def ctp_instrument(contract: str, exchange: str = "") -> str:
    """把平台里的合约写法收敛成 CTP 的 `InstrumentID`。

    认 `rb2601` / `RB2601` / `RB2601.SHF` / `SHFE.rb2601` / `RB` + `SHFE`，
    以及平台内部的**品种键** `RB_SHFE` / `TL0_CFFEX`（下划线分隔）。
    品种（没有数字，或只有主力连续那个 `0`）原样返回 —— 调用方要先解析成
    具体合约，CTP 不认品种。
    """
    s, letters, digits, ex = _parse_contract(contract, exchange)
    if not s:
        return ""
    if not looks_contract(s):
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


def split_today_his(p: dict, side: str, total: int) -> tuple[int, int]:
    """柜台持仓行 → `(昨仓, 今仓)`，**今仓优先**。`side` 是 `long` / `short`。

    `ctp_client` 的快照里同时有 `{side}_yd`（YdPosition 累加）和
    `{side}_today`（TodayPosition 累加）。以前两张表各取一套：

    - 持仓表（交易页）优先 `long_today` → 显示「今多 1」；
    - 引擎（`broker_api._fill_positions`）直接拿 `long_yd` 当昨仓
      → 认为「昨多 1、今多 0」，一键对齐于是发 CLOSE 去平一手今天才开的仓，
      被上期所整单打回 `[51] CTP:平昨仓位不足`（2026-11-19 用户报的 sn2612）。

    为什么必须以今仓为准：CTP 对上期所的持仓按「今/昨 + 开仓日」**分条推送**，
    每条 `Position` 只算自己那一段，`YdPosition` 在分条累加后不一定等于昨仓
    （两条都带 Position=1、YdPosition 各填自己那份时，his+today 会大于总仓）。
    `TodayPosition` 是柜台对「今天开的」直接计数，反而是稳的。

    所以：今仓取 `TodayPosition`，昨仓 = 总 − 今。总仓是硬约束。
    """
    total = max(0, int(total or 0))
    if not total:
        return 0, 0

    def _i(name: str) -> int:
        try:
            return int(p.get(name) or 0)
        except (TypeError, ValueError):
            return 0

    today = _i(f"{side}_today")
    if today <= 0:                      # 柜台没给今仓，退回「总 − 昨」
        today = max(0, total - _i(f"{side}_yd"))
    today = min(max(0, today), total)
    return total - today, today


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
    #: 这条通道有没有行情。融航也支持 —— 它的行情前置就是标准 CTP md 服务，
    #: 账户里填了 `md_front` 就会连；留空则整段行情逻辑跳过
    SUPPORTS_MD = True
    #: 登录失败后的冷却秒数（见 `start()`）。20 秒足够避开「连点启动全部」，
    #: 又不至于让用户觉得「刷新按钮坏了」。
    FAIL_COOLDOWN = 20.0

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
        # 最近一次报单 / 撤单的时刻。柜台（尤其融航这类中台代理）常把**所有**
        # 请求算进同一个「一秒一个」的额度里，查询和报单撞在一起时被丢的往往是
        # 报单 —— 2026-11-23 那四笔 TL2612 就是「api 返回 0、柜台一条回报都没有」。
        # 所以发完单之后主动让一拍，别让 1Hz 的查询轮转去挤它。
        self._last_order_ts = 0.0
        self._query_turn = 1          # 0 = 该发资金了，1 = 该发持仓了（登录后先要持仓）
        # 发过多少拍持仓查询。柜台一枪不放时，靠它和 `positions_loaded` 一起
        # 在 `[POS]` 诊断行里把「发了 N 拍、一条回调都没来」说清楚
        self._pos_try = 0
        # 登录后补查历史委托 / 成交用掉了多少拍，以及「放弃」时只抱怨一次
        self._hist_try = 0
        self._hist_warned = False
        self._hist_at = 0.0
        self._lock = threading.Lock()
        # 🔴 一条会话现在可能被**两个页面同时用**（交易页的 worker + 组合页，见
        # `ttrade/hubreg.py`）。`start()` 因此必须**幂等且串行**：后到的一方撞上
        # 正在登录的那一方时，要等它登完再直接返回，而不是再登一次 —— 二次登录
        # 会把先到那方手里的 `trader` 换掉，它的持仓和委托回报当场断，而且
        # 一声不响（旧对象还在，只是再也不会被回调）。
        self._start_lock = threading.Lock()
        self._started = False
        #: 上一次登录失败的时刻 / 原因。见 `start()` 里的冷却 —— 融航登录失败后
        #: 每点一次「启动全部 / 刷新」就来一轮 `Release()` + `Init()`，实测会直接
        #: 把进程打崩（2026-11-23 19:28，`vendor\rohon_ctp\openctp_ctp.libs\
        #: MSVCR100.dll` 访问违例 0xc0000005，连崩三次）
        self._fail_at = 0.0
        self._fail_msg = ""
        # 查询额度的闸门锁。`_last_query` 是「先读再写」两个动作，两个线程同时
        # 进来会**都**通过判断 ⇒ 同一毫秒发两笔查询 ⇒ 柜台按每秒一个的规矩丢掉
        # 后一笔，且不回包。这就是「两个页面各抢一份额度」的最后一环。
        self._q_lock = threading.Lock()
        # 开仓均价缓存：成交的「平仓盈亏」靠它倒推（见 `tqlive.est_close_pnl`）
        self._open_cache: dict = {}
        # 「报价」框盯的合约。只存代码，行情对象按需订阅
        self._watch_syms: list[str] = []
        # 每个盯盘合约「最后一笔交易时段内」的行情。收市后 CTP 还会推快照
        # （`volume` / `oi` 还在动），不冻结的话界面看着像还在交易
        self._watch_cache: dict[str, dict] = {}

    # ---- 柜台适配（融航覆盖这四个） ----

    def check_deps(self) -> None:
        # 绑一次比「查 import 成功没有」更实在：DLL 加载失败、通道被占用
        # 都会在这里就报出来，而不是等 start() 里半路炸
        if not ctp_client_mod.bind_channel("ctp"):
            raise CtpError(
                f"用不了 CTP（{ctp_client_mod.CTP_IMPORT_ERROR}）"
                "。装法：pip install openctp-ctp")

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
        """登录。**幂等**：已经登上的直接返回（一条会话两个页面共用，见 hubreg）。

        串行化用的是 `_start_lock` 而不是可重入锁：第二个调用方会在这里等到
        第一个登完（最长 15 秒）。宁可让它等，也不能让它再登一次。
        """
        with self._start_lock:
            if self._started and self._alive():
                return
            # 🔴 失败冷却：不要「点一下就来一轮 Release + Init」。
            # `_start_locked()` 一进来就把老客户端 `stop()`（= `api.Release()`），
            # 再建一条新的 `Init()`；登录又失败时那条新的也在半路被 `Release()`。
            # 融航这条 dll 在这种反复拆建下会直接段错误 —— 2026-11-23 19:28
            # 连崩三次（`vendor\rohon_ctp\openctp_ctp.libs\MSVCR100.dll`，
            # 0xc0000005），而当时融航正是 `[-1] 交易核心未初始化` 一直登不上。
            # 20 秒内再来问，就原样把上次的错抛出去，一次也不碰底层 API。
            wait = self.FAIL_COOLDOWN - (time.monotonic() - self._fail_at)
            if wait > 0 and self._fail_msg:
                raise CtpError(f"{self._fail_msg}"
                               f"（{wait:.0f} 秒后可再试；期间不会重复连柜台）")
            try:
                self._start_locked()
            except Exception as exc:
                self._fail_at = time.monotonic()
                self._fail_msg = str(exc)
                raise
            self._fail_at, self._fail_msg = 0.0, ""

    def _alive(self) -> bool:
        """这条会话是不是**还真的连着**。

        只认 `logged_in` 这个事件，不看 `_started`：连接掉过之后 `_started` 还是
        真（没人去清），拿它当判据的话「断线后重连」会直接短路返回一条死会话，
        界面显示已连接、数据和下单一样都没有。
        """
        t = self.trader
        if t is None:
            return False
        ev = getattr(t, "logged_in", None)
        return True if ev is None else bool(ev.is_set())

    def _start_locked(self) -> None:
        self.check_deps()
        if not self.specs:
            raise CtpError(f"{self.PLATFORM} 会话至少要有一个账户")
        cfg = self.make_config(self.specs[0])
        if not cfg.user_id or not cfg.password:
            raise ValueError(f"账户「{cfg.name}」缺少投资者账号或密码")
        if not cfg.td_front:
            raise ValueError(f"{self.PLATFORM} 交易前置地址为空")
        # 上一次登录留下的客户端先收掉：重连时会有残留，不收就是两条 TCP 会话
        # 抢同一个账户（柜台按后登的为准，先登的那条从此收不到任何回报）。
        # 放在参数校验**之后**：配置写错了就不该动一条还活着的老连接。
        for old in (self.md, self.trader):
            if old is not None:
                try:
                    old.stop()
                except Exception:  # noqa: BLE001
                    pass
        self.md = None
        self.trader = None

        trader = self.make_trader(cfg)
        trader.start()
        # 🔴 等的是「登录 + 结算确认」，不是「登录」。结算确认在收盘后的结算窗口里
        # 常回「[26] 结算结果未就绪」，等几秒再发就好 —— 所以这里**边等边重发**，
        # 而不是干等 15 秒然后把整条会话判死（那样界面只说「登录超时」，用户
        # 完全看不出是结算没确认）。
        end = time.monotonic() + 15.0
        while not trader.wait_ready(1.0):
            trader.confirm_settlement()
            if time.monotonic() >= end:
                err = trader.last_error or "登录超时（15 秒）"
                trader.stop()
                raise CtpError(err + self.login_hint(cfg))
        # 🔴 登录后的**第一拍只能发一个查询**（2026-11-23 融航 SA701）：
        # CTP 同一会话每秒最多一个查询请求，这里原来连着 `query_account()` +
        # `query_positions()`，柜台只受理第一个，第二个直接被流控丢掉 ——
        # 不报错、不进回调，`_pos` 永远是空字典。表现就是「资金有数、一条持仓
        # 都没有」，而界面上那看着就是「账户没仓」。
        #
        # 现在只抢发**持仓**（它最要紧，界面一进来就要），资金交给
        # `pump_queries()` 的下一拍（`_query_turn=0` 就是「该发资金了」）。
        trader.query_positions()
        self._query_turn = 0
        # 撤单要交易所代码，但**不查全量合约表**：SimNow 上是 1.6 万条、80 秒，
        # 而 CTP 一个会话是串行的 —— 流没结束之前报单会一直排队，界面表现就是
        # 「点了下单没反应」。改成用本地品种→交易所映射兜底，具体合约的
        # 最小变动价位在订阅时按单个合约查（秒回）。
        trader.exchange_resolver = lambda ins: variety_exchange(
            "".join(ch for ch in str(ins) if ch.isalpha()))
        self.trader = trader

        # 行情是可选：只做交易不看盘也能用。连不上就只报一条日志，不影响下单
        # （账户没填 `md_front`、或通道 `SUPPORTS_MD` 为假时整段跳过）
        if cfg.md_front and self.SUPPORTS_MD:
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
        # 打上「登上了」的记号，只在这一步之后 `start()` 才会短路。放在最后：
        # 半路抛异常时不能置真，否则重试会直接返回一个没有 trader 的会话
        self._started = True

    def close(self) -> None:
        # 🔴 先从注册表摘掉再关（见 hubreg.forget 的注释）。两个页面共用一个
        # 对象时，谁先收尾都不该在表里留一条「refs>0 但 trader 已空」的僵尸。
        hubreg.forget(self)
        self._started = False
        try:
            from ttrade import crashlog

            _nm = ",".join(str(getattr(s, "name", "")) for s in (self.specs or []))
            crashlog.breadcrumb(f"hub.close({_nm})")
        except Exception:      # noqa: BLE001 - 取证不能拦着收尾
            pass
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
        # 还是品种（含 `TL0` 这种主力连续代号）就去本地主力表查一个具体合约。
        # 判据用 `looks_contract`，不能只看「有没有数字」—— `TL0` 带一个 0
        if not looks_contract(ins):
            found = main_contract(ins, exchange_of_contract(symbol, exchange))
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
        """推进一步：读 tick 攒 bar，顺手把账户查询的额度用掉（见 `pump_queries`）。

        资金/持仓是**主动查询**（CTP 没有推送账户变动）。这里有个坑，实测踩出来的：
        **柜台限制「每秒最多一个查询请求」**。资金和持仓如果背靠背连着发，第二个
        会被静默丢掉 —— 表现是「资金一直在动，持仓永远是 0」。
        """
        time.sleep(max(0.05, float(timeout)))
        if self.md is not None and self.instrument:
            tick = self.md.last(self.instrument)
            if tick:
                self._builder.push(tick)
        self.pump_queries()
        return True

    def pump_queries(self, force: bool = False) -> str:
        """按「每秒最多一个查询」的额度轮转一拍，返回这一拍发了什么。

        🔴 这套轮转**必须只有一份**（2026-11-23 融航那次）：

        交易页走 `CtpHub.wait()`，组合页走 `BrokerApi._refresh()`。两边早先各写
        一套节流 —— 组合页那套在调用 `_query_hist()` 时用的是**另一个**闸门
        （`_hist_at`），于是同一毫秒里既发了「查历史委托」又发了「查持仓」，
        柜台按每秒一个的规矩只收下前一个，后面那笔被静默丢弃。表现就是：交易页
        持仓表好好的，组合页那条连接 `positions` 永远是空的，而 `mismatch()`
        拿空表去比，界面写「账户与策略理论持仓一致」—— 其实一条都没读到。

        顺序有讲究：
          1. 结算没确认 → 先给它（没它一笔都下不出去，自带 3 秒节流不会霸占额度）
          2. 登录后前几拍 → 补历史委托 / 成交（`_query_hist()`）
          3. 之后 → 资金 / 持仓交替
        `force=True` 放开 1 秒的闸门（自检和 `BrokerApi._refresh(force=True)` 用）。
        """
        t = self.trader
        if t is None:
            return ""
        # 🔴 闸门 + 派发必须**原子**。一条会话现在会被交易页的 worker 和组合页
        # 同时调（`ttrade/hubreg.py`）：`_last_query` 是「先读再写」两个动作，
        # 两个线程同时进来会**都**通过一秒的判断 ⇒ 同一毫秒发两笔 ⇒ 柜台丢后
        # 一笔且不回包。拿不到锁就放过这一拍 —— 0.5~1 秒后自然还有下一拍，
        # 但绝不能在这里阻塞（调用方可能是界面主线程）。
        if not self._q_lock.acquire(blocking=False):
            return ""
        try:
            return self._pump_locked(t, force)
        finally:
            self._q_lock.release()

    def _pump_locked(self, t, force: bool) -> str:
        now = time.monotonic()
        if not force and now - self._last_query <= 1.0:
            return ""
        # 刚发过报单 / 撤单：这一拍查询让开。丢一拍持仓只是晚一秒更新，丢一笔
        # 报单就是「用户以为下了、其实没下」（见 `_last_order_ts` 的注释）。
        # `force=True` 是用户手点刷新，可以不等。
        if not force and now - getattr(self, "_last_order_ts", 0.0) <= 1.0:
            return ""
        self._last_query = now
        try:
            # 🔴 `confirm_settlement` 只有 openctp 那条通道有（融航的客户端在
            # 登录回包里就自己确认了）。早先这里直接 `t.confirm_settlement()`，
            # 融航上每次轮转都抛 AttributeError 被下面接住 ⇒ 资金和持仓的查询
            # **一拍都没发出去**，账户页看着有数（那是登录时那一次查的），
            # 之后再也不更新。用 getattr 判一下能力，别让一个可选方法瘫痪整套轮转。
            confirm = getattr(t, "confirm_settlement", None)
            if confirm is not None and confirm():
                return "settle"
            if self._query_hist():
                return "hist"              # 这一拍发的是历史查询
            if self._query_turn == 0:
                self._query_turn = 1
                t.query_account()
                return "account"
            self._query_turn = 0
            t.query_positions()
            self._pos_try = getattr(self, "_pos_try", 0) + 1
            return "positions"
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"查询失败：{exc}"
            return ""
    def _query_hist(self) -> bool:
        """登录后补一次当日委托 / 成交。返回 True 表示这一拍被它用掉了。

        🔴 为什么必须补（2026-11-23 融航那次）：CTP 只在**登录之后**推
        `OnRtnOrder` / `OnRtnTrade`。持仓靠 `query_positions` 补齐了，委托和
        成交却是两张空表 —— 用户看到「今空 5 手、成交框里一笔都没有」，只能
        理解成「系统没从账户里读数据」。

        两个查询各占一拍（柜台限制每秒一个），`HIST_TRIES` 拍还没成就放弃并
        记一条原因：有的中台不放行 `ReqQryOrder`，一直重试等于每 2 秒白打一枪。
        """
        t = self.trader
        if t is None:
            return False
        # 融航那条客户端**没有**查历史委托 / 成交的接口（中台不提供），也没有
        # `orders_loaded` 这两个标志。早先这里直接取属性：融航上每拍都抛
        # AttributeError，`pump_queries()` 吞掉之后资金/持仓查询永远轮不到。
        # 没有能力就当「已经查完了」，直接放过，把额度留给资金和持仓。
        if not hasattr(t, "orders_loaded") or not hasattr(t, "query_orders"):
            return False
        if t.orders_loaded and getattr(t, "trades_loaded", True):
            return False
        # 放弃判断要在节流闸门**之前**：放弃时什么都不发，不需要排队等额度；
        # 放到闸门后面的话，调用方只要卡在 1 秒内连调，就永远走不到这句，
        # 那个「柜台不给查」的说法也就永远说不出来
        if self._hist_try >= HIST_TRIES:
            if not self._hist_warned:
                self._hist_warned = True
                self.last_error = (
                    f"柜台不给查历史委托/成交（试了 {HIST_TRIES} 次），"
                    "委托框和成交框只显示本次登录之后的单子")
            return False
        now = time.monotonic()
        # 自带闸门：`BrokerApi._refresh()` 是 0.5 秒一轮，直接调它会打得比
        # 柜台允许的（每秒一个查询）还密。放在这里，两个调用方都不用各管一套
        if now - self._hist_at < 1.0:
            return False
        self._hist_at = now
        self._hist_try += 1
        if not t.orders_loaded:
            t.query_orders()
        else:
            t.query_trades()
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

    def watch(self, symbols) -> None:
        """盯一列合约的 tick，给交易页的「报价」框用。

        和 `quote_of()` 的区别：那个是「取一次价、等有值再返回」，这个是「订上
        就一直读」。CTP 的 `SubscribeMarketData` 幂等，重复订没有副作用。

        **必须在 worker 线程里调**（会动行情连接的订阅表）。
        """
        if self.md is None:
            return
        want: list[str] = []
        for raw in symbols or ():
            ins = ctp_instrument(raw)
            if ins and ins not in want:
                want.append(ins)
        self._watch_syms = want
        if not want:
            return
        try:
            self.md.subscribe(want)
        except Exception:  # noqa: BLE001 - 订不上就让那几行留着空值，别炸整条会话
            pass

    def _ins_exchange(self, ins: str) -> str:
        """CTP 合约（`rb2601`，不带交易所）→ 交易所代码。"""
        ex = ""
        if self.trader is not None:
            ex = str((self.trader.instruments.get(ins) or {}).get("exchange") or "")
        if not ex:
            try:
                ex = variety_exchange("".join(ch for ch in ins if ch.isalpha()))
            except Exception:  # noqa: BLE001
                ex = ""
        return ex

    def _watch_rows(self) -> list[dict]:
        """「报价」框的数据：盯着的每个合约一行。没 tick 的给 last=0。

        🔴 和 `_quote()` 一样要过**交易时段门禁**。收市之后 CTP 还会继续推
        最后一次快照，`volume` / `oi` 照样在变 —— 不过滤的话「报价」框看着
        还在跳，用户以为系统在盘中乱动。不在时段就还回最后一笔时段内的快照，
        `source` 标成 `closed`，界面上能一眼看出是冻结的。
        """
        out = []
        for ins in self._watch_syms:
            t = (self.md.last(ins) if self.md is not None else None) or {}
            info = (self.trader.instruments.get(ins) if self.trader is not None else None) or {}
            row = {
                "symbol": ins,
                "instrument": ins,
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
                # 日内现算要算持仓占比 / PCR，报价框不显示这一列但必须带出去
                "open_interest": float(t.get("open_interest") or 0.0),
                "tick": float(info.get("tick") or 0.0),
                "datetime": str(t.get("datetime") or ""),
                "source": "live",
            }
            code = "".join(ch for ch in ins if ch.isalpha())
            ex = self._ins_exchange(ins)
            open_now = True
            if code and ex:
                try:
                    open_now = symbols_info.in_session(code, ex)
                except Exception:  # noqa: BLE001
                    open_now = True
            if open_now:
                self._watch_cache[ins] = dict(row)
            else:
                cached = self._watch_cache.get(ins)
                if cached:
                    row = dict(cached)
                    row["source"] = "closed"
            out.append(row)
        return out

    def watched_quotes(self) -> dict:
        """给日内现算用的订阅报价表，键统一成天勤写法（和 `watch` 的入参一致）。"""
        out: dict = {}
        for sym, row in zip(self._watch_syms, self._watch_rows()):
            if str(row.get("source") or "") == "live":
                out[str(sym)] = row
        return out

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
        # 顺序不能反：刚平光的仓这一轮已经不在持仓里，靠的正是缓存里上一轮
        # 的均价；所以先刷新缓存，再倒推成交的平仓盈亏
        update_open_avg(self._open_cache, positions)
        est_close_pnl(trades, self._open_cache, self._mult)
        # 再用**当日成交流**按时间 FIFO 精确还原一遍（2026-11-23 用户：重新登录
        # 后平仓盈亏全变 0/—）。重启后 `_open_cache` 是空的，已经平掉的仓在持仓
        # 表里也没有均价 ⇒ 上面那步只能给 `None`。成交流是查得回来的
        # （`_query_hist()` 里的 `ReqQryTrade`），FIFO 能算出精确值，覆盖估算值；
        # 平昨仓缺开仓记录的按**昨结算价**算（逐日盯市口径，和结算单一致）
        replay_close_pnl(trades, self._mult, pre_settle_of=self._pre_settle)
        # 算完再截断（`_trades()` 故意不截断，见那里的注释）
        trades = trades[:200]
        # 柜台回报里没有逐笔手续费，按品种成本档案估（标 est，界面上加「≈」）
        est_commission(trades, mult_of=self._mult, exchange_of=self._ex_of)
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
        st.quotes = self._watch_rows()
        return st

    def _ex_of(self, instrument: str) -> str:
        """合约 → 交易所。CTP 给的合约代码不带交易所，得自己补。

        优先问柜台（订阅过就有），没订阅过就扫本地数据文件名
        （见 `variety_exchange`）—— 手数、成交金额、手续费都要它，缺了就全错。
        """
        ins = str(instrument or "")
        if not ins:
            return ""
        head, _, tail = ins.partition(".")
        if tail:                      # 已经是 `SHFE.rb2601` 这种写法
            return head.upper()
        if self.trader is not None:
            ex = str((self.trader.instruments.get(ins) or {}).get("exchange") or "")
            if ex:
                return ex
        return variety_exchange("".join(ch for ch in ins if ch.isalpha()))

    def _pre_settle(self, instrument: str) -> float:
        """平仓盈亏的**成本基准**：昨结算价 → 昨收 → 当日收盘价 → 本地日线昨收。

        `replay_close_pnl` 在「成交流凑不齐这笔平仓的开仓记录」时拿它当成本
        （2026-11-23 用户：「平仓盈亏可以根据昨日结算价或者收盘价格来计算，
        如果获取不到历史成交的话」）。

        🔴 顺序按**口径正确性**排，不是按好拿排：
        1. `pre_settle`（CTP 的 `PreSettlementPrice`）—— 结算单的逐日盯市基准；
        2. `pre_close`（`PreClosePrice`）—— 昨收，用户说的「收盘价格」；
        3. `close`（`ClosePrice`）—— 当日收盘价，**只有收市后才有值**，盘中是 0；
        4. 本地日线最后一根的收盘 —— 行情没订上/柜台不推行情时的最后兜底。

        拿不到返回 0（调用方据此不写这一笔），**绝不拿 0 当成本**。
        """
        ins = ctp_instrument(instrument)
        if not ins:
            return 0.0
        for key in ("pre_settle", "pre_close", "close"):
            if self.md is None:
                break
            try:
                v = float((self.md.last(ins) or {}).get(key) or 0.0)
            except Exception:  # noqa: BLE001
                v = 0.0
            if v > 0:
                return v
        return _last_daily_close(ins, self._ex_of(ins))

    def _mult(self, instrument: str) -> float:
        """合约乘数。`_mult_of` 认的是 `SHFE.rb2601` 这种带交易所的写法，
        CTP 给的是光秃秃的 `rb2601` —— 直接传进去乘数会变成 1.0，杠杆和
        成交金额就全错一个数量级。所以先补上交易所。
        """
        ins = str(instrument or "")
        ex = self._ex_of(ins)
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
        """柜台持仓 → 交易页持仓表。

        🔴 「当日盈亏」口径（2026-11-26 用户明确）：**持仓 vs 昨日收盘价**，不是
        柜台逐日盯市那个 `PositionProfit`（它按**昨结算**算，两者在期货上差很多 ——
        al2701 截图里昨结算口径 -4,050，按昨收是另一回事）。算不出昨收 / 最新价
        时退回柜台值，不写 0、不写「—」。
        """
        out = []
        for ins, p in src.items():
            long_q = int(p.get("long") or 0)
            short_q = int(p.get("short") or 0)
            if not long_q and not short_q:
                continue
            # 今仓优先用柜台给的 `TodayPosition`（多条记录累加后的值），
            # 拿不到才用「总 − 昨」推。口径统一在 `split_today_his()` 里 ——
            # 引擎那边的 `broker_api._fill_positions` 用的是同一个函数，
            # 两张表必须一眼对得上，否则「平今平昨」两边各说各话。
            _hl, long_td = split_today_his(p, "long", long_q)
            _hs, short_td = split_today_his(p, "short", short_q)
            mult = self._mult(ins)
            cost = float(p.get("price") or 0.0)          # CTP 的 OpenCost 是总成本
            total = long_q + short_q
            avg = (cost / (total * mult)) if total and cost else 0.0
            # 多空分开的均价：两边都有仓时混用一个均值，算「成交平仓盈亏」必错
            # （`cost_long/cost_short` 是 `ctp_client` 按 PosiDirection 分开记的）
            cl = float(p.get("cost_long") or 0.0)
            cs = float(p.get("cost_short") or 0.0)
            avg_l = (cl / (long_q * mult)) if (long_q and cl > 0) else (avg if long_q else 0.0)
            avg_s = (cs / (short_q * mult)) if (short_q and cs > 0) else (avg if short_q else 0.0)
            # 当日盈亏 = (现价 − 昨收) × 净手 × 乘数 × 方向。昨收优先 CTP tick 的
            # `pre_close`（PreClosePrice），没订上的品种退回本地日线最后一根收盘。
            # 两边都拿不到就用柜台的，绝不拿 0 当昨收硬算出一个「今日大赚」。
            tick = (self.md.last(ins) if self.md is not None else None) or {}
            last = float(tick.get("last") or 0.0)
            pre = float(tick.get("pre_close") or 0.0)
            if not pre:
                pre = _last_daily_close(ins, self._ex_of(ins))
            day = (float(p.get("profit") or 0.0)
                   if (last <= 0 or pre <= 0 or mult <= 0)
                   else (last - pre) * (long_q - short_q) * mult)
            out.append({
                "symbol": ins,
                "pos": long_q - short_q,
                "pos_long": long_q,
                "pos_short": short_q,
                "pos_long_today": long_td,
                "pos_short_today": short_td,
                "open_price_long": avg_l,
                "open_price_short": avg_s,
                "open_cost_long": (cl or cost) if long_q else 0.0,
                "open_cost_short": (cs or cost) if short_q else 0.0,
                "float_profit": float(p.get("profit") or 0.0),
                "float_profit_long": float(p.get("profit") or 0.0) if long_q else 0.0,
                "float_profit_short": float(p.get("profit") or 0.0) if short_q else 0.0,
                "position_profit": day,
                "margin": float(p.get("margin") or 0.0),
                "last_price": 0.0,
            })
        out.sort(key=lambda r: r["symbol"])
        return out

    def _orders(self, src: dict) -> list[dict]:
        pairs, seen = [], set()
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
            day = str(o.get("trading_day") or "")
            hms = str(o.get("insert_time") or "")
            pairs.append((_ctp_ts(day, hms), {
                "order_id": o.get("order_sys_id") or o.get("order_ref") or oid,
                "symbol": str(o.get("instrument") or ""),
                "direction": {"0": "BUY", "1": "SELL"}.get(str(o.get("direction")), str(o.get("direction") or "")),
                "offset": {"0": "OPEN", "1": "CLOSE", "3": "CLOSETODAY",
                           "4": "CLOSEYESTERDAY"}.get(
                    str(o.get("offset")), str(o.get("offset") or "")),
                "volume_orign": vol,
                "volume_left": max(0, vol - traded),
                "limit_price": float(o.get("price") or 0.0),
                "status": "ALIVE" if status in ALIVE_STATUS else status,
                # 拒单理由全在 StatusMsg 里（「50:已撤单报单被拒绝SHFE:价格跌破跌停板」），
                # 只给「已撤单」等于什么都没说 —— 用户没法判断是价格错了还是权限不够
                "last_msg": str(o.get("msg") or status),
                # 「时间」列。**这一列曾经错成显示 `msg`**（委托说明），因为
                # `ctp_client` 那时根本没读 InsertTime。现在两个字段都落下来了：
                # InsertTime 只有 HH:MM:SS，要配 TradingDay 才拼得出跨夜的
                # 「MM-DD HH:MM:SS」
                "insert_date_time": 0.0,
                "time": _ctp_stamp(day, hms),
            }))
        # 未成交的排前面：撤单要点的就是它们；同类里时间新的在上。
        # 稳定排序 ⇒ 先按时间倒一遍，再按「是否活单」正一遍
        pairs.sort(key=lambda kv: kv[0], reverse=True)
        pairs.sort(key=lambda kv: kv[1]["status"] != "ALIVE")
        return [kv[1] for kv in pairs][:200]

    def _trades(self, src: dict) -> list[dict]:
        """成交行。

        🔴 **这里不截断**（2026-11-23）：`replay_close_pnl` 要拿当日**全部**
        成交流按时间 FIFO 倒推平仓盈亏，只留最近 200 笔的话，早上那几笔开仓
        就不在里面了 —— 平仓一笔都算不出来，界面上又是一片「—」。截断挪到
        `state()` 算完之后（那里截一次，界面看到的还是 200 行）。
        """
        out = []
        for tid, t in src.items():
            out.append({
                "trade_id": str(t.get("trade_id") or tid),
                "order_id": str(t.get("order_ref") or ""),
                "symbol": str(t.get("instrument") or ""),
                "direction": {"0": "BUY", "1": "SELL"}.get(str(t.get("direction")), str(t.get("direction") or "")),
                # 🔴 平昨必须映射成 CLOSE 系列（2026-11-23）：原来表里只有
                # 0/1/3，`4`（CloseYesterday）会原样透出去，`_is_close_offset()`
                # 认不出来 ⇒ 一笔平昨被当成开仓，FIFO 的账立刻错位
                "offset": {"0": "OPEN", "1": "CLOSE", "3": "CLOSETODAY",
                           "4": "CLOSEYESTERDAY"}.get(
                    str(t.get("offset")), str(t.get("offset") or "")),
                "price": float(t.get("price") or 0.0),
                "volume": int(t.get("volume") or 0),
                "trade_date_time": 0.0,
                "time": str(t.get("time") or ""),
            })
        out.sort(key=lambda r: r["time"], reverse=True)
        return out

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
        """下单。`limit_price` 给 None / 0 就发**市价单**（2026-11-23）。

        以前这里抛「只收限价单」，界面上市价那个勾对 CTP / 融航账户直接禁掉。
        🔴 用户现在的口径：「ctp 和融航的下单，现在所有交易所应该都支持市价单了，
        上期所和能源所在 2026 年 7 月 6 日也开放支持市价单了」。所以放开：
        市价单不需要按最小变动价位对齐（价格字段本来就是 0），也不做 tick 校验。
        """
        if self.trader is None:
            raise CtpError("还没连接")
        ins = ctp_instrument(symbol)
        if not ins:
            raise CtpError("合约代码为空")
        try:
            px = float(limit_price)
        except (TypeError, ValueError):
            px = float("nan")
        if not (px > 0):                 # None / 0 / NaN 一律按市价
            # 报单会让查询轮转让开一拍，见 `_last_order_ts`
            self._last_order_ts = time.monotonic()
            return self.trader.place(ins, direction, offset, 0.0, int(volume),
                                     market=True)
        # 价格按最小变动价位对齐，否则柜台拒单理由是「价格不是最小变动价位整数倍」
        tick = 0.0
        info = self.trader.instruments.get(ins)
        if info:
            tick = float(info.get("tick") or 0.0)
        price = px
        if tick > 0:
            price = round(round(price / tick) * tick, 8)
        self._last_order_ts = time.monotonic()
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
            # 🔴 这句以前是「找不到委托 12」，然后被 `_cancel()` 里那个 `except:
            # pass` 吞掉 —— 2026-11-23 融航四笔 TL2612 挂满 30 秒被撤，实际上
            # 一次都没撤成，日志里却什么都没留下。现在会明着说出来。
            raise CtpError(
                f"找不到委托 {oid}：柜台没有这笔单的任何回包 —— "
                "要么报单没进柜台，要么委托回报没回来")
        self._last_order_ts = time.monotonic()
        self.trader.cancel(
            str(target.get("instrument") or ""),
            order_sys_id=str(target.get("order_sys_id") or ""),
            order_ref=str(target.get("order_ref") or ""),
        )

    def set_target(self, symbol: str, volume: int, account: str = "") -> None:
        raise CtpError(f"{self.PLATFORM} 没有目标仓接口，请直接下单")


# ------------------------------------------------------------------ 融航


def _tcp_addr(text: str) -> str:
    """`210.22.96.58:28111` → `tcp://210.22.96.58:28111`。

    CTP 的 `RegisterFront` 只认带协议的地址，裸 `ip:port` 会一直连不上，
    而且**不报错**——`Init()` 之后就是静默超时，界面只会说「登录超时」。
    账户表里融航填的是裸地址，所以这里补一次。
    """
    v = str(text or "").strip()
    if not v:
        return ""
    if v.lower().startswith(("tcp://", "ssl://", "udp://")):
        return v
    return "tcp://" + v


class RohonHub(CtpHub):
    """融航资管中台。除了「怎么连」以外，全部复用 `CtpHub`。

    融航（Rohon）对外是 CTP 协议的中台：同样的前置地址、同样的报单字段、同样的
    委托状态字符。所以持仓合并、上期所今昨仓拆分、成交金额折算这些规则一条都
    不用重写 —— 只换底下的客户端。

    驱动层是融航官方给的 **CTP 兼容 dll**（替换 `thosttraderapi_se.dll`），
    配 openctp-ctp 6.7.2.8 用标准 CTP 接口登录。以前那条 `vnpy_rohon` 的路
    走不通：那个包只有 cp310/cp313 的轮子，本机 base 是 3.14 装不上。
    `ttrade/rohon_client.py` 还留着（旧路线的完整实现，备用），但**没人调它**。
    """

    PLATFORM = "融航"
    KIND = "rohon"
    #: 融航的替换包里只给了 `thosttraderapi_se.dll`，但它的**行情前置是标准
    #: CTP md 服务**：vendor 里那份 6.7.2 的 `thostmduserapi_se` 连
    #: `124.160.66.120:41253` 能连上、能登录（交易日 20261119）、能订阅，
    #: 不需要融航另给 md dll（2026-11-19 实测）。
    #:
    #: 早先记的「标准 md 连融航行情口 0xC0000005 崩溃」是**探测脚本的问题**：
    #: `api.RegisterSpi(Spi(api))` 传的是临时对象，CPython 当场回收，CTP 侧
    #: 拿到悬空指针就崩。SPI 必须被变量持有（正式代码里是 `self._spi`）。
    #:
    #: 账户里填了 `md_front` 才连行情，留空就退回「只交易、行情靠天勤」。
    SUPPORTS_MD = True

    def check_deps(self) -> None:
        """融航走 CTP 兼容 dll，**不再依赖 vnpy_rohon**。

        `vnpy_rohon` 只有 cp310/cp313 的轮子，本机 base 是 3.14 装不上；
        而融航的 dll 导出的就是标准 CTP 的 C 接口（`CreateFtdcTraderApi` 等），
        配 openctp-ctp 6.7.2.8 能直接登录（2026-11-19 实测）。
        """
        ok, why = ctp_channel.available("rohon")
        if not ok:
            raise CtpError(
                f"融航通道用不了：{why}。\n"
                "融航运行时在 vendor/rohon_ctp/（openctp-ctp 6.7.2.8 + 融航替换 dll），"
                "从仓库拉全就能用，不需要 pip install vnpy_rohon")

    def make_config(self, spec):
        """融航账户 → `CtpConfig(channel="rohon")`。

        **不调 `fill_preset()`**：那个会按 SimNow 的预设补齐空地址，一旦
        融航的交易前置漏填，就会被悄悄填成 `182.254.243.31:30001`，
        然后融航的 dll 连标准 CTP 前置直接段错误 —— 界面上只会看到
        「程序没了」，查不出原因。
        """
        return CtpConfig(
            name=getattr(spec, "name", "") or "融航",
            preset="",
            md_front=_tcp_addr(getattr(spec, "md_front", "")),   # 填了就自己收行情
            td_front=_tcp_addr(getattr(spec, "td_front", "")),
            broker_id=getattr(spec, "broker_id", "") or "",
            user_id=getattr(spec, "user", "") or "",
            password=getattr(spec, "password", "") or "",
            app_id=getattr(spec, "app_id", "") or "",
            auth_code=getattr(spec, "auth_code", "") or "",
            channel="rohon",
        )

    def make_trader(self, cfg):
        # 同一个 `TraderClient`：融航的 dll 就是 CTP 接口，解析规则一模一样
        return TraderClient(cfg)

    def make_md(self, cfg):
        # 融航的 md 前置是标准 CTP 行情服务，直接复用 `MdClient`。
        # 收不到行情时 `start()` 只记一条 `last_error`，不影响下单。
        return MdClient(cfg)

    def login_hint(self, cfg) -> str:
        """融航没有 SimNow 那种「按时段开放」的问题，补的是缺字段的提示。"""
        miss = [k for k in ("broker_id", "td_front")
                if not getattr(cfg, k, "")]
        if miss:
            return f"（还缺：{'、'.join(miss)}。融航要填经纪商代码和交易服务器）"
        return "（检查经纪商代码 / 交易服务器 / 产品名称 / 授权编码是否和资管后台一致）"
