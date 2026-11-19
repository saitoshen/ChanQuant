"""把 CTP / 融航柜台包成**天勤 `TqApi` 的形状**，让 `PortfolioRunner` 不用改一行。

## 为什么要有这一层

`PortfolioRunner`（策略交易和组合页）从 3.2 起就是照天勤那套接口写的：
`get_kline_serial` / `get_quote` / `get_position` / `get_order` / `insert_order` /
`cancel_order` / `wait_update`。这套东西对天勤是「活引用」—— 返回的 DataFrame 和
dict 会原地更新，引擎每轮 `wait_update()` 之后直接读缓存就行。

柜台（CTP / 融航）没有这套接口：它是「主动查询 + 回调写 dict」，
且**完全没有历史行情**。所以这里做三件事：

1. **实时行情**来自柜台（`CtpHub.md` 的 tick），用 `ctp_live._BarBuilder` 拼成
   bar。bar 的时间戳按**桶起点**对齐 —— 和本地 `data_tq` 的写法一致
   （实测本地 5m 首根是 `09:00`、末根 `14:55`，都是起点）。
2. **历史行情**来自本地天勤数据（`tqdata.load_main`，口径 `adjust="none"`，
   即不复权的原始主连）。柜台拿不到历史，天勤那条链路我们又不想为了看盘再登
   一个账号，所以直接把本地那份灌进「序列」，实时 bar 接在后面 ——
   引擎里 `merge_live(local_qfq, live)` 本来就是这么拼的，天然对得上。
3. **交易**走 `hub.place()` / `hub.cancel()`，仓位和委托从 `trader.snapshot()`
   翻成天勤的字段名（`pos_long` / `volume_long_today` …），状态翻成
   `ALIVE` / `FINISHED` / `DEAD`。

## 几个刻意的取舍

- **`get_position()` / `get_order()` 返回的 dict 是同一个对象**，每轮原地改。
  天勤就是这么用的（引擎在 `start()` 里拿一次就不管了），换成每轮新建，引擎
  手里那份永远停在启动那一刻 —— 表现是「仓位永远对不上，反复重发同一笔单」。
- **下单后立刻塞一条乐观委托**（`ALIVE` + 全量 `volume_left`）。CTP 的
  `OnRtnOrder` 回报要几十到几百毫秒才到，这中间引擎查不到这笔单会当成
  「没发出去」再发一次。乐观条目在回报到达后被真回报覆盖。
- **不抛异常**：`wait_update()` 里行情/查询出错只记 `last_error`。抛出去会
  让界面把整条会话判成「行情中断」拆掉，而 CTP 掉一下再连上是常事。
- **没有历史委托查询**：`TraderClient` 没实现 `ReqQryOrder`，所以上一次进程
  留下的挂单在这儿看不见。启动时的 `_sweep_orders()` 会因此空跑一遍 ——
  这一点在日志里说清楚，别让用户以为已经撤干净了。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ttrade.ctp_live import (_EX_SHORT, _BarBuilder, ctp_instrument, exchange_of_contract,
                             looks_contract, main_contract, split_today_his,
                             variety_exchange)
from ttrade.tqlive import BACKEND_LABELS

#: `get_kline_serial` 的第二个参数是秒，反过来要认出周期名
_REV_SEC = {
    60: "1m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    7200: "2h",
    14400: "4h",
}

#: 主连 → 具体合约，多久重新解析一次。查的是本地文件（切换表 / 衍生数据），
#: 换月一天最多一次，60 秒足够灵敏又不会每轮都去读盘
MAIN_TTL = 60.0

#: 柜台限制「每秒最多一个查询请求」，资金和持仓必须错开发
ACCOUNT_SEC = 2.0
POSITION_SEC = 1.0

#: 下单之后多久没有柜台回报就判这笔委托没报进去（见 `_expire_optimistic`）
OPTIMISTIC_TTL = 30.0

#: 本地文件名和天勤品种代码对不上的，逐个列出来。
#: 30 年期国债：数据源（Tushare）把品种代码写成 `TL0`（`0` 是连续合约标记），
#: 本地文件曾经叫 `TL0_CFFEX.csv`。已正名为 `TL`，这里留一条反向映射，
#: 免得还没改过来的老策略配置读不到历史（2026-11-19）
_LOCAL_ALIAS = {"TL0": "TL"}

#: CTP 的 `OrderStatus` 中文 → 天勤的三态。**只认这三种**，别的一律当活单：
#: 把「部分成交(不在队列)」当成死单，`_chase` 会记一次拒单，三次就停机。
_STATUS_MAP = {
    "全部成交": "FINISHED",
    "已撤单": "DEAD",
    "被拒": "DEAD",
}


def _split(symbol: str) -> tuple[str, str]:
    """天勤写法 / 平台品种键 → `(交易所, 品种或合约)`。

    `KQ.m@SHFE.rb` → `("SHFE", "rb")`；`SHFE.rb2701` → `("SHFE", "rb2701")`；
    **`TL0_CFFEX` → `("CFFEX", "TL0")`**。第三种是策略配置里的品种键，以前不认，
    整串被当合约名，最后发出 `tlcffex0` 这种不存在的代码 —— 柜台报
    「[148] 无效的 ExchangeID」（2026-11-19 TL 策略）。
    """
    s = str(symbol or "").strip()
    if "@" in s:
        s = s.split("@", 1)[1]
    if "." in s:
        head, _, tail = s.partition(".")
        # 两种方向都见过：天勤 `SHFE.rb2701`，Tushare / 报表 `TL2612.CFFEX`。
        # 只看第一段当前半段是不是交易所，`TL2612.CFFEX` 会把 `TL2612` 当交易所
        if head.upper() in _EX_SHORT:
            return _EX_SHORT[head.upper()], tail
        if tail.upper() in _EX_SHORT:
            return _EX_SHORT[tail.upper()], head
        return head.upper(), tail
    if "_" in s:
        head, _, tail = s.rpartition("_")
        if tail.upper() in _EX_SHORT:
            return _EX_SHORT[tail.upper()], head
    return "", s


def _canon(exchange: str, instrument: str) -> str:
    """柜台合约 → 引擎里用的写法：`("SHFE", "rb2701")` → `SHFE.rb2701`。

    和天勤的 `SHFE.rb2701` / `CZCE.TA701` / `DCE.i2701` 是同一个形状（郑商所
    本来就是三位月份），所以引擎里那些 `symbol.split(".")[0]` 判交易所的代码
    不用改一行。
    """
    return f"{exchange.upper()}.{instrument}" if exchange else str(instrument)


def _ns(index: pd.DatetimeIndex) -> np.ndarray:
    """北京时间（naive）→ **UTC 纳秒**，和天勤 `datetime` 列同一个口径。

    引擎里 `bars_from_serial()` 会把它转回来（`local_time`），`_market_alive()`
    直接拿它当 epoch 秒用。少这一步，夜盘 21:00 的 bar 会变成 13:00，
    和本地数据一根都对不上，而且不报错。
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return np.zeros(0, dtype="int64")
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Shanghai")
    # 🔴 必须显式转成 `datetime64[ns]`：pandas 2.x 起 `DatetimeIndex` 的单位可能
    # 是微秒（按值域自动推断），`asi8` / `astype("int64")` 给的是**本单位的整数**。
    # 直接拿去当天勤的纳秒用，整条序列会落到 1970-01-22，而且 `bars_from_serial`
    # 只会把它整段剔掉 —— 表现为「有序列、但一根 bar 都没有」
    return idx.tz_convert("UTC").tz_localize(None) \
        .to_numpy("datetime64[ns]").astype("int64")


class _Obj:
    """天勤返回的是 dict 子类（属性 / 下标都能取）。这里给柜台对象也补上。"""

    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self) -> str:                      # pragma: no cover - 调试用
        body = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"<{type(self).__name__} {body}>"


class BrokerApi:
    """一个柜台账户，装成 `TqApi` 的样子。"""

    def __init__(self, spec, feed: str = "", log=None, hub=None, history=None) -> None:
        """
        - `spec`：`AccountSpec`，`backend` 必须是 `ctp` / `rohon`
        - `feed`：实时行情源。默认跟账户走 —— 一个 CTP 账户只能从 CTP 拿行情，
          选别的没有连接可用（要跨柜台拿行情，得另开一条 `CtpHub`，见 `HubSet`）
        - `hub`：已建好的 `CtpHub` / `RohonHub`。给单测注入假柜台用
        - `history`：`(品种, 交易所, 周期) -> DataFrame`，默认读本地 `data_tq`
        """
        self.spec = spec
        self.feed = str(feed or getattr(spec, "backend", "ctp") or "ctp")
        self._log = log or (lambda _msg: None)
        self.hub = hub
        self._history = history or self._local_history

        self.last_error = ""
        self._serials: dict[tuple, pd.DataFrame] = {}
        self._builders: dict[tuple, _BarBuilder] = {}
        #: `(合约, 周期)` → 用它的序列键。一个合约的 bar 要接进**它自己那些**
        #: 序列，早期版本按周期匹配，结果 5m 的 bar 会同时写进所有品种的 5m 序列
        self._links: dict[tuple, list[tuple]] = {}
        self._last_ns: dict[tuple, int] = {}
        self._main: dict[str, str] = {}          # "SHFE.rb" → "rb2701"
        self._main_at: dict[str, float] = {}
        self._pos: dict[str, _Obj] = {}
        self._ord: dict[str, _Obj] = {}
        self._quote: dict[str, _Obj] = {}
        self._main_quote: dict[str, _Obj] = {}
        self._cur_main: dict[str, str] = {}
        self._tick_at: dict[str, str] = {}       # 合约 → 上一次 tick 的时间串
        self._subbed: set[str] = set()
        self._warned: set[str] = set()
        self._last_query = 0.0
        self._turn = 1
        self._closed = False

    # ---- 生命周期 ----

    def start(self) -> None:
        from ttrade.ctp_live import CtpHub, RohonHub

        if self.hub is None:
            if self.feed not in ("ctp", "rohon"):
                raise ValueError(
                    f"行情源「{self.feed}」不是柜台，"
                    "这条连接只能跑 CTP / 融航")
            self.hub = RohonHub([self.spec]) if self.feed == "rohon" else CtpHub([self.spec])
        self.hub.start()
        if getattr(self.hub, "last_error", ""):
            # 行情没连上也照样能交易（`CtpHub` 的行情是可选的），只是没有 K 线。
            # 说出来，别让用户对着不动的图猜
            self._log(f"行情源（{BACKEND_LABELS.get(self.feed, self.feed)}）："
                      f"{self.hub.last_error}")
        self._refresh(force=True)

    def close(self) -> None:
        self._closed = True
        hub, self.hub = self.hub, None
        if hub is not None:
            try:
                hub.close()
            except Exception:  # noqa: BLE001
                pass

    # ---- 行情 ----

    def get_kline_serial(self, symbol: str, duration_sec: int = 300,
                         data_length: int = 200) -> pd.DataFrame:
        """引擎要的「活序列」。

        本地天勤历史灌在前面（不复权原始价），之后每轮 `wait_update()` 把柜台
        拼出来的 bar 接在后面 —— 直接改这个 DataFrame 对象本身，因为引擎握的是
        引用（`self._klines[key]`），换成新对象它永远看不到。
        """
        period = _REV_SEC.get(int(duration_sec or 300), "5m")
        key = (str(symbol), period)
        hit = self._serials.get(key)
        if hit is not None:
            return hit
        ex, code = _split(symbol)
        df = self._seed(code, ex, period, int(data_length or 200))
        self._serials[key] = df
        if len(df):
            self._last_ns[key] = int(df["datetime"].iloc[-1])
        self._build(key, code, ex, period)
        return df

    def get_quote(self, symbol: str):
        """盘口。`symbol` 可以是主连（`KQ.m@SHFE.rb`）或具体合约（`SHFE.rb2701`）。

        主连必须**换到具体合约**再取盘口：柜台上没有主连，而且下单下的也是
        具体合约 —— 拿主连的价去下具体合约的单，换月前后会差出一整个跳空。

        主连返回的是**一个常驻对象**，每轮 `wait_update()` 原地刷新它的
        `underlying_symbol`。引擎只在第一次订阅时取一次 quote，之后一直读缓存
        （`_quotes[key].underlying_symbol`）来判换月 —— 每次新建对象的话，
        它手里那个永远停在启动那一刻，换月永远发现不了。
        """
        key = str(symbol or "")
        ex, code = _split(key)
        # `TL0` 也算品种（主力连续代号 = 品种 + 0），不能因为「有数字」就当合约
        if "@" in key or not looks_contract(ctp_instrument(code, ex)):
            obj = self._main_quote.get(key)
            if obj is None:
                obj = _Obj(symbol=code, underlying_symbol="", datetime="",
                           last_price=0.0, ask_price1=0.0, bid_price1=0.0,
                           ask_volume1=0.0, bid_volume1=0.0, price_tick=0.0,
                           volume=0.0, open_interest=0.0, expired=False)
                self._main_quote[key] = obj
            self._sync_main(key, obj)
            return obj
        ins = ctp_instrument(code, ex)
        if not ins:
            return None
        # 下单 / 定手数都是按**具体合约**取盘口（`st.underlying`），而序列订的是
        # 「当前主力」—— 换月前后、或者账户上还留着老合约的仓，这个合约可能压根
        # 没订过，`md.last()` 返回空，`_price_at` 拿到 NaN 就永远「等盘口」。
        self._ensure_sub(ins)
        return self._quote_of(ins, ex)

    def _ensure_sub(self, ins: str) -> None:
        if not ins or ins in self._subbed:
            return
        self._subbed.add(ins)
        try:
            if getattr(self.hub, "md", None) is not None:
                self.hub.md.subscribe([ins])
            if getattr(self.hub, "trader", None) is not None:
                self.hub.trader.query_instruments(ins)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"订阅 {ins} 失败：{exc}"
            self._subbed.discard(ins)

    def _sync_main(self, key: str, obj) -> None:
        """把主连对象刷成「当前主力合约」的盘口，顺手处理换月。"""
        ex, code = _split(key)
        ins = self._main_of(code, ex)
        if not ins:
            return
        src = self._quote_of(ins, ex)
        if src is None:
            return
        for name in ("underlying_symbol", "datetime", "last_price", "ask_price1",
                     "bid_price1", "ask_volume1", "bid_volume1", "price_tick",
                     "volume", "open_interest"):
            setattr(obj, name, getattr(src, name, getattr(obj, name, 0.0)))
        obj.symbol = ins
        if self._cur_main.get(key) == ins:
            return
        old, self._cur_main[key] = self._cur_main.get(key), ins
        if old:
            self._log(f"主力换月：{key} {old} → {ins}（老合约持仓由引擎按换月处理）")
        # 换月之后要订新合约的 tick，否则序列就停在换月那一刻
        for skey, _df in list(self._serials.items()):
            if skey[0] != key:
                continue
            self._build(skey, code, ex, skey[1])

    def _quote_of(self, ins: str, ex: str = ""):
        hub = self.hub
        if hub is None:
            return None
        tick = {}
        if getattr(hub, "md", None) is not None:
            tick = hub.md.last(ins) or {}
        info = {}
        if getattr(hub, "trader", None) is not None:
            info = hub.trader.instruments.get(ins) or {}
        exchange = self._exchange_of(ins, ex)
        tick_size = float(info.get("tick") or 0.0)
        if tick_size <= 0:
            tick_size = self._tick_from_table(ins, exchange)
        last = float(tick.get("last") or 0.0)
        ask = float(tick.get("ask") or 0.0)
        bid = float(tick.get("bid") or 0.0)
        if last != last:                     # CTP 用 DBL_MAX 表示「没有」→ NaN
            last = 0.0
        q = self._quote.get(ins)
        if q is None:
            q = _Obj(symbol=ins, underlying_symbol=_canon(exchange, ins),
                     datetime="", last_price=0.0, ask_price1=0.0, bid_price1=0.0,
                     ask_volume1=0.0, bid_volume1=0.0, price_tick=0.0,
                     volume=0.0, open_interest=0.0, expired=False)
            self._quote[ins] = q
        # 盘口偶尔会推到 NaN / 极大值（涨跌停、集合竞价），**不要**拿它盖掉上
        # 一个有效值：`_price_at` 会算出 NaN 价格，`_send` 直接判「等盘口」，
        # 行情一跳就永远发不出单
        if last > 0:
            q.last_price = last
        if ask > 0:
            q.ask_price1 = ask
        if bid > 0:
            q.bid_price1 = bid
        q.ask_volume1 = float(tick.get("ask_vol") or 0.0)
        q.bid_volume1 = float(tick.get("bid_vol") or 0.0)
        q.volume = float(tick.get("volume") or 0.0)
        q.open_interest = float(tick.get("oi") or 0.0)
        q.datetime = str(tick.get("datetime") or "")
        if tick_size > 0:
            q.price_tick = tick_size
        return q

    # ---- 账户 ----

    def get_position(self) -> dict:
        """返回**同一个 dict 对象**，内容每轮原地刷新（理由见模块开头）。"""
        return self._pos

    def get_order(self) -> dict:
        return self._ord

    # ---- 交易 ----

    def insert_order(self, **kw):
        direction = str(kw.get("direction") or "").upper()
        offset = str(kw.get("offset") or "").upper()
        symbol = kw.get("symbol")
        volume = int(kw.get("volume") or 0)
        price = kw.get("limit_price")
        if self.hub is None:
            raise RuntimeError("还没连接柜台")
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = 0.0
        if not price or price != price:
            # 天勤用 NaN 表示市价，柜台只有限价单。静默换成 0 会被柜台拒成
            # 「价格不合法」，用户看到的是「委托被拒」而不是「这个柜台不收市价」
            raise ValueError(
                f"{BACKEND_LABELS.get(self.feed, self.feed)}只收限价单，"
                "拿不到对手价时不能下单（检查行情是否连上）")
        if volume <= 0:
            raise ValueError("手数必须大于 0")
        ex, code = _split(symbol)
        ins = ctp_instrument(code, ex) or ctp_instrument(code)
        if not ins:
            raise ValueError(f"合约代码为空（{symbol}）")
        exchange = self._exchange_of(ins, ex)
        ref = str(self.hub.place(ins, direction, offset, volume, price) or "")
        row = _Obj(order_id=ref, order_ref=ref, order_sys_id="", symbol=ins,
                   instrument=ins, exchange=exchange, direction=direction,
                   offset=offset, volume_orign=volume, volume_left=volume,
                   limit_price=price, status="ALIVE", last_msg="", optimistic=True,
                   ts=time.time())
        if ref:
            self._ord[ref] = row
        return row

    def cancel_order(self, order) -> None:
        if self.hub is None:
            return
        oid = str(getattr(order, "order_id", "") or getattr(order, "order_ref", "") or "")
        if not oid:
            return
        self.hub.cancel(oid)

    # ---- 推进 ----

    def wait_update(self, deadline: float | None = None) -> bool:
        """推进一步：取 tick 拼 bar，顺手按节流刷持仓和委托。

        和天勤的差别：天勤是「有东西变才返回」，这里是「睡到 deadline」。
        引擎只把返回值当布尔用，但它按 `timeout` 传 deadline，照做才不会被
        主线程一轮一轮空转打满 CPU。
        """
        if deadline is None:
            deadline = time.time() + 0.5
        while not self._closed:
            self._pump()
            now = time.time()
            if now >= deadline:
                break
            time.sleep(min(0.15, max(0.01, deadline - now)))
        if not self._closed:
            for key, obj in list(self._main_quote.items()):
                try:
                    self._sync_main(key, obj)
                except Exception as exc:  # noqa: BLE001
                    self.last_error = f"刷新主连盘口失败：{exc}"
            self._refresh()
        return True

    # ---- 内部：行情 ----

    def _build(self, key: tuple, code: str, ex: str, period: str) -> None:
        """登记「这个序列要用哪个合约的 tick」。"""
        ins = self._main_of(code, ex)
        if not ins:
            self._warn_once(
                f"main:{code}:{ex}",
                f"查不到 {code}_{ex} 的主力合约，"
                f"{BACKEND_LABELS.get(self.feed, self.feed)}行情无法订阅"
                "（去「数据管理」补一次主力切换表 / 衍生数据）")
            return
        bkey = (ins, period)
        if bkey not in self._builders:
            self._builders[bkey] = _BarBuilder(period=period)
        keys = self._links.setdefault(bkey, [])
        if key not in keys:
            keys.append(key)
        self._ensure_sub(ins)

    def _pump(self) -> None:
        """取一次所有合约的最新 tick，推进每个 builder。

        一个合约可能被两个周期用（5m + 1h），tick 只取一次、分别 push。
        """
        hub = self.hub
        if hub is None or getattr(hub, "md", None) is None or not self._builders:
            return
        by_ins: dict[str, list[_BarBuilder]] = {}
        for (ins, _period), b in self._builders.items():
            by_ins.setdefault(ins, []).append(b)
        for ins, builders in by_ins.items():
            try:
                tick = hub.md.last(ins)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"取行情失败：{exc}"
                continue
            if not tick:
                continue
            stamp = str(tick.get("datetime") or "")
            if stamp and self._tick_at.get(ins) == stamp:
                continue            # 同一条 tick，别重复累加
            self._tick_at[ins] = stamp
            for b in builders:
                b.push(tick)
        self._extend()

    def _extend(self) -> None:
        """把 builder 里已经走完的 bar 接进**它对应的**那些序列。"""
        for (ins, period), b in self._builders.items():
            keys = self._links.get((ins, period)) or []
            if not keys:
                continue
            frame = b.frame(limit=100000)
            if frame is None or frame.empty:
                continue
            for key in keys:
                df = self._serials.get(key)
                if df is None:
                    continue
                self._append(key, df, frame)

    def _append(self, key: tuple, df: pd.DataFrame, frame: pd.DataFrame) -> None:
        """把 builder 的 bar 接在序列后面。**原地改**，见 `get_kline_serial`。

        两种情况都要处理：

        - **新 bar**：接在最后（`builder` 里那批 bar 每轮都完整返回，只接严格
          晚于序列最后一根的，否则同一根会被重复追加）；
        - **正在走的那根**：时间戳等于最后一根时要**就地更新 OHLC**。只看「新增」
          的话，一根 bar 只在它诞生的那一瞬间定格 —— K 线内触发（`_intrabar_tick`
          拿最新价跟触发价比）永远比的是那一刻的价，突破当根永远不触发。
        """
        if df is None or frame is None or frame.empty:
            return
        ns = _ns(frame.index)
        oi = (frame["oi"].to_numpy(float) if "oi" in frame.columns
              else np.zeros(len(frame)))
        cols = ["datetime", "open", "high", "low", "close", "volume",
                "open_oi", "close_oi"]
        last = self._last_ns.get(key)
        for i, ts in enumerate(ns):
            n = int(ts)
            vals = [n,
                    float(frame["open"].iloc[i]), float(frame["high"].iloc[i]),
                    float(frame["low"].iloc[i]), float(frame["close"].iloc[i]),
                    float(frame["volume"].iloc[i]), float(oi[i]), float(oi[i])]
            if last is not None and n == last and len(df):
                row = df.index[-1]
                for c, v in zip(cols, vals):
                    df.at[row, c] = v          # 逐列写，别用 `.iloc[-1] =` 整行换
            elif last is None or n > last:
                df.loc[len(df)] = vals
                last = n
        if last is not None:
            self._last_ns[key] = last

    def _seed(self, code: str, ex: str, period: str, n: int) -> pd.DataFrame:
        """本地天勤历史 → 序列。读不到就返回空表（引擎会判「没数据」不动仓）。"""
        cols = ["datetime", "open", "high", "low", "close", "volume",
                "open_oi", "close_oi"]
        try:
            local = self._history(code, ex, period)
        except Exception as exc:  # noqa: BLE001
            self._warn_once(
                f"hist:{code}:{ex}:{period}",
                f"读不到 {code}_{ex} 的本地 {period} 历史"
                f"（{type(exc).__name__}: {exc}）——"
                "柜台没有历史行情，策略会因为没有历史而无法开仓。"
                "去「数据管理」页补一次天勤主连 5m / 1h")
            return pd.DataFrame(columns=cols)
        if local is None or local.empty:
            return pd.DataFrame(columns=cols)
        local = local.tail(max(2, int(n))).copy()

        def _col(name: str, fill: bool = True) -> np.ndarray:
            s = local[name] if name in local.columns else pd.Series(
                np.nan if not fill else 0.0, index=local.index)
            v = pd.to_numeric(s, errors="coerce")
            return (v.fillna(0.0) if fill else v).to_numpy(float)

        out = pd.DataFrame({
            "datetime": _ns(local.index),
            "open": _col("open", False),
            "high": _col("high", False),
            "low": _col("low", False),
            "close": _col("close", False),
            "volume": _col("volume"),
            "open_oi": _col("open_oi"),
            "close_oi": _col("close_oi"),
        })
        out = out.dropna(subset=["close"]).reset_index(drop=True)
        self._check_stale(code, ex, period, local)
        return out

    def _check_stale(self, code: str, ex: str, period: str, local: pd.DataFrame) -> None:
        """本地历史停在几天前时，**说出来**。

        引擎的 `_signal_frame` 会把本地历史和实时段拼起来，中间断了一截的话
        只能退回实时原始价 —— 那一小段根本不够算因子，策略会静默地不动仓。
        用户看到的是「启动了，但从来没下过单」。
        """
        try:
            last = pd.Timestamp(local.index[-1])
        except Exception:  # noqa: BLE001
            return
        gap = (pd.Timestamp.now() - last).total_seconds()
        limit = _period_sec(period) * 4
        if gap > max(limit, 3 * 86400):
            self._warn_once(
                f"stale:{code}:{ex}:{period}",
                f"本地 {code}_{ex} 的 {period} 历史停在 {last:%Y-%m-%d %H:%M}，"
                f"实时段接不上（现在 {pd.Timestamp.now():%Y-%m-%d %H:%M}）。"
                "信号会算不准 —— 去「数据管理」页补一次天勤主连")

    def _local_history(self, code: str, ex: str, period: str) -> pd.DataFrame:
        from ttrade import tqdata

        names = [str(code).upper()]
        alias = _LOCAL_ALIAS.get(str(code).upper())
        if alias:
            names.append(alias)
        last_exc: Exception | None = None
        for name in names:
            try:
                return tqdata.load_main(name, str(ex).upper(), period, adjust="none")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise last_exc or FileNotFoundError(f"没有 {code}_{ex}")

    # ---- 内部：账户刷新 ----

    def _refresh(self, force: bool = False) -> None:
        hub = self.hub
        if hub is None or getattr(hub, "trader", None) is None:
            return
        now = time.monotonic()
        # 柜台限制「每秒最多一个查询请求」，资金和持仓错开一轮发（见 `CtpHub.wait`）
        if force or now - self._last_query > ACCOUNT_SEC:
            self._last_query = now
            self._turn = 0
            self._query("query_account")
        elif self._turn == 0 and now - self._last_query > POSITION_SEC:
            self._turn = 1
            self._query("query_positions")
        try:
            snap = hub.trader.snapshot()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"读柜台快照失败：{exc}"
            return
        self._fill_positions(snap.get("positions") or {})
        self._fill_orders(snap.get("orders") or {})
        self._expire_optimistic()

    def _query(self, name: str) -> None:
        try:
            getattr(self.hub.trader, name)()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{name} 失败：{exc}"

    def _fill_positions(self, src: dict) -> None:
        """柜台持仓 → 天勤字段名，**原地**写进 `self._pos`。

        天勤有**两套**字段描述同一件事（`pos_long_his` / `volume_long_his`），
        引擎两个都读、取最大（见 `PortfolioRunner._split_pos`）。这里两套都填，
        免得引擎在「今仓 / 昨仓」上判错 —— 上期所平今和平昨是两个指令，
        填错就是整单被交易所打回。
        """
        out: dict[str, _Obj] = {}
        for ins, p in src.items():
            long_q = int(p.get("long") or 0)
            short_q = int(p.get("short") or 0)
            if not long_q and not short_q:
                continue
            # 今仓**优先用柜台给的 `TodayPosition`**，昨仓由「总 − 今」反推，
            # 和 `ctp_live._positions()` 走同一个函数（口径只要分家就会出人命：
            # 持仓表显示「今多 1」、引擎却认为「昨多 1」，一键对齐发 CLOSE 去平
            # 今天才开的仓，被上期所整单打回 `[51] 平昨仓位不足`）。
            # 为什么以今仓为准、为什么 YdPosition 不可靠，见 `split_today_his()`。
            long_yd, long_td = split_today_his(p, "long", long_q)
            short_yd, short_td = split_today_his(p, "short", short_q)
            ex = self._exchange_of(ins)
            key = _canon(ex, ins)
            out[key] = _Obj(
                symbol=key, exchange=ex,
                pos_long=long_q, pos_short=short_q,
                volume_long=long_q, volume_short=short_q,
                pos_long_his=long_yd, pos_short_his=short_yd,
                pos_long_today=long_td,
                pos_short_today=short_td,
                volume_long_his=long_yd,
                volume_short_his=short_yd,
                volume_long_today=long_td,
                volume_short_today=short_td,
                open_price_long=float(p.get("price") or 0.0),
                open_price_short=float(p.get("price") or 0.0),
                float_profit=float(p.get("profit") or 0.0),
                margin=float(p.get("margin") or 0.0),
            )
        self._pos.clear()
        self._pos.update(out)

    def _fill_orders(self, src: dict) -> None:
        """柜台委托 → 天勤三态。已成交 / 已撤的条目留着不清 —— `_chase` 就是
        靠看到 `FINISHED` / `DEAD` 才把在途单销掉，提前删掉会变成「追单追不到」。
        """
        for _key, o in src.items():
            ref = str(o.get("order_ref") or "")
            oid = ref or str(o.get("order_sys_id") or _key)
            if not oid:
                continue
            status = _STATUS_MAP.get(str(o.get("status") or ""), "ALIVE")
            vol = int(o.get("volume") or 0)
            traded = int(o.get("traded") or 0)
            ins = str(o.get("instrument") or "")
            row = _Obj(
                order_id=oid,
                order_ref=ref,
                order_sys_id=str(o.get("order_sys_id") or ""),
                symbol=ins, instrument=ins,
                exchange=self._exchange_of(ins),
                direction={"0": "BUY", "1": "SELL"}.get(
                    str(o.get("direction")), str(o.get("direction") or "")),
                offset={"0": "OPEN", "1": "CLOSE", "3": "CLOSETODAY"}.get(
                    str(o.get("offset")), str(o.get("offset") or "")),
                volume_orign=vol,
                volume_left=max(0, vol - traded),
                limit_price=float(o.get("price") or 0.0),
                status=status,
                last_msg=str(o.get("msg") or ""),
                optimistic=False,
            )
            self._ord[oid] = row

    def _expire_optimistic(self) -> None:
        """柜台一直没回报的乐观委托，超时就判死。

        不判死会**卡住整条策略**：引擎看到 `ALIVE` 就不会重发，等到追单超时去撤，
        柜台上却没有这笔单（`cancel` 报「找不到委托」），状态永远停在「撤单中」，
        `st.pending` 永远不清空 ⇒ `_rebalance` 一直让路，看着像死了。

        判死之后引擎按「委托被拒」处理：重报，连续几次就停机并报出柜台的原始
        提示 —— 比无声卡住强。
        """
        now = time.time()
        for row in self._ord.values():
            if not getattr(row, "optimistic", False) or row.status != "ALIVE":
                continue
            if now - float(getattr(row, "ts", 0.0) or 0.0) < OPTIMISTIC_TTL:
                continue
            row.status = "DEAD"
            row.last_msg = (f"{BACKEND_LABELS.get(self.feed, self.feed)}柜台 "
                            f"{OPTIMISTIC_TTL:.0f} 秒没有任何回报，委托可能没报进去")

    # ---- 内部：合约解析 ----

    def _main_of(self, code: str, ex: str) -> str:
        """主连品种 → 当前主力合约（CTP 写法）。多级兜底，查不到返回空串。"""
        if not code:
            return ""
        # `TL2612` 是合约；`TL` / `TL0` 是品种 —— 判据见 `looks_contract`，
        # 不能只看「有没有数字」，否则 `TL0` 会被当成合约直接发到柜台
        if looks_contract(code):
            return ctp_instrument(code, ex)          # 本来就是具体合约
        key = f"{str(ex).upper()}.{str(code).lower()}"
        now = time.time()
        hit = self._main.get(key)
        if hit and now - self._main_at.get(key, 0.0) < MAIN_TTL:
            return hit
        ins = ""
        try:
            from ttrade import tqstore

            df = tqstore.load_switch(str(code).upper(), str(ex).upper())
            if df is not None and not df.empty:
                # 切换表最后一条的「新合约」就是现在的主力。数据来自天勤的
                # `KQ.m@` 底层合约，和本地主连序列是同一个口径 —— 用 Tushare
                # 那份 `c1_code` 兜底之前先看它
                ins = ctp_instrument(str(df["new_tq"].iloc[-1] or ""), ex)
        except Exception:  # noqa: BLE001
            ins = ""
        if not ins:
            ins = main_contract(code, ex)
        if ins:
            self._main[key] = ins
            self._main_at[key] = now
        elif hit:
            return hit               # 这一轮查不到就先用上一次的，别把序列断了
        return ins

    def _exchange_of(self, ins: str, ex: str = "") -> str:
        """合约 → 交易所。优先问柜台（查过合约信息就有），否则按品种名推。"""
        e = str(ex or "").upper()
        if e:
            return e
        trader = getattr(self.hub, "trader", None) if self.hub is not None else None
        if trader is not None:
            info = (getattr(trader, "instruments", {}) or {}).get(ins) or {}
            e = str(info.get("exchange") or "").upper()
            if e:
                return e
        return variety_exchange("".join(ch for ch in str(ins) if ch.isalpha()))

    def _tick_from_table(self, ins: str, exchange: str) -> float:
        """最小变动价位。柜台还没查过合约信息时，退回「品种信息」那张表 ——
        和回测同源（`symbols_info`），不至于让追单的跳数算成 0。"""
        try:
            from ttrade import symbols_info

            variety = "".join(ch for ch in str(ins) if ch.isalpha())
            return float(symbols_info.info(variety.upper(), exchange).get("tick") or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        self._log(msg)


def _period_sec(period: str) -> float:
    for sec, name in _REV_SEC.items():
        if name == period:
            return float(sec)
    return 300.0
