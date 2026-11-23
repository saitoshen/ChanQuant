"""把实时会话接到 Qt 上。

分工：

- `LiveWorker` 跑在**独立线程**里，独占一条连接：天勤 `LiveHub`（一条 TqApi）、
  CTP `CtpHub`、融航 `RohonHub`。
- `LiveController` 活在主线程，负责起停线程、把界面操作**排队**丢给 worker，
  再把 worker 的信号转出来。

**一条会话一个账户**（2026-11-17 改）。早先是「一条会话装所有勾选的账户」：
天勤走 `TqMultiAccount` 一次登录，多个账户共用一条 TqApi。那样省连接，但账户
列表里**没法单独登录 / 登出某一个账户** —— 而用户要的就是每行一个按钮。所以
改成按账户分会话：一个账户一个线程、一条连接。代价是 N 个天勤账户就是 N 条
TqApi（N 次登录、各订一份行情）；账户是「两三个」这个量级时无所谓。

为什么下单要排队而不是直接调：天勤的 `insert_order()` 只是入队，真正的报文在
下一次 `wait_update()` 发出；而 `wait_update()` 正在 worker 线程里阻塞着。跨线程
直接调会写坏 TqApi 内部的订阅状态，表现为「单子偶尔丢、偶尔重复」。CTP / 融航的
`ReqOrderInsert` 倒是线程安全，但三家走同一条队列才能保证**成交回报的顺序和
下单顺序一致**，所以柜台也排队。

**行情源**：多个柜台同时登录时，K 线和盘口只能来自一个柜台（两家的 tick 拼在
一条 K 线上会来回跳）。选哪一个由 `feed` 参数决定，界面上是「行情源」下拉框；
不指定就按 天勤 → CTP → 融航 的顺序挑第一个连上的。账户不受影响：所有柜台的
持仓、委托、成交照样合并显示。
"""

from __future__ import annotations

import queue
import time

from PySide6.QtCore import QObject, QThread, Signal

from ttrade import hubreg
from ttrade.tqlive import BACKENDS, BACKEND_LABELS, LiveHub, LiveState, merge_states


def group_specs(spec) -> dict:
    """把账户按柜台分组，返回 `{后端: [spec]}`，只保留非空的组。

    `spec` 可能是 None（纯行情会话）、单个 AccountSpec，或一组。判断走
    `AccountSpec.backend`（唯一判据），不在这里再写一遍 kind 映射。
    """
    if spec is None:
        items = []
    elif isinstance(spec, (list, tuple)):
        items = list(spec)
    else:
        items = [spec]
    out: dict[str, list] = {}
    for s in items:
        key = getattr(s, "backend", "tq") or "tq"
        out.setdefault(key, []).append(s)
    # 固定顺序：账户表和资金合计的排列要稳定，不然勾选一变行序就跳
    return {k: out[k] for k in BACKENDS if k in out}


def make_hub(backend: str, specs: list):
    """按柜台造一个 hub。三家的对外接口完全一致（start/subscribe/wait/bars/
    state/place/cancel/set_target），worker 循环不需要知道底下是谁。

    🔴 **CTP / 融航走进程内注册表**（`ttrade/hubreg.py`，2026-11-23）：同一个
    账户如果组合页已经在用，这里拿到的就是**同一条连接**，`hub.start()` 会
    直接短路返回。一个账户在柜台那边只能有一条会话 —— 两条会话的委托回报互不
    相通、查询额度各抢一份，界面表现就是「交易页有仓、监控器说不一致」。
    天勤不进注册表：`TqApi` 是单线程语义，两个页面共用一个对象会写坏订阅状态。
    """
    if backend == "ctp":
        from ttrade.ctp_live import CtpHub

        return hubreg.acquire("ctp", specs, lambda: CtpHub(specs))
    if backend == "rohon":
        from ttrade.ctp_live import RohonHub

        return hubreg.acquire("rohon", specs, lambda: RohonHub(specs))
    return LiveHub(specs)


class HubSet:
    """多条柜台连接的门面。

    对 worker 循环暴露的接口和单个 hub 一模一样，内部：
    - 行情（`bars` / `quote` / `key`）只取 `feed` 那一路；
    - 账户（`state`）合并所有柜台；
    - 下单 / 撤单按**账户名**路由到所属柜台。
    """

    def __init__(self, hubs: dict, feed: str = "") -> None:
        self.hubs = dict(hubs)          # 后端 → hub（断线时会被移掉）
        # 台账：收尾时要放手的**全部**连接。`hubs` 会因为断线被 pop，只按它
        # 收尾的话，被摘掉那条的引用永远不还（`hubreg` 里 refs 卡在 2），
        # 组合页那条会话再也关不掉。
        self._opened = list(self.hubs.values())
        self.feed = feed if feed in self.hubs else (
            next(iter(self.hubs), ""))
        self.period = ""
        self.key = ""
        self.underlying = ""

    # ---- 生命周期 ----

    @property
    def alive(self) -> bool:
        return bool(self.hubs)

    @property
    def last_error(self) -> str:
        hub = self.hubs.get(self.feed)
        if hub is not None:
            return getattr(hub, "last_error", "")
        return "；".join(f"[{BACKEND_LABELS.get(n, n)}] {getattr(h, 'last_error', '')}"
                        for n, h in self.hubs.items())

    def subscribe(self, symbol: str, exchange: str, period: str) -> list[str]:
        """每个柜台各订一次。返回订阅失败的柜台名（界面拿去写日志）。"""
        bad = []
        for name, hub in self.hubs.items():
            try:
                hub.subscribe(symbol, exchange, period)
            except Exception:  # noqa: BLE001 - 一个柜台订不到不该拖死整条会话
                bad.append(name)
        feed = self.hubs.get(self.feed)
        if feed is not None:
            self.key = getattr(feed, "key", "")
            self.period = getattr(feed, "period", period)
            self.underlying = getattr(feed, "underlying", "")
        return bad

    def bars(self):
        hub = self.hubs.get(self.feed)
        if hub is None:
            import pandas as pd

            return pd.DataFrame()
        return hub.bars()

    def state(self) -> LiveState:
        return merge_states([(n, h.state()) for n, h in self.hubs.items()],
                            self.feed)

    def close(self) -> None:
        """放手。**只有最后一个使用者才真的关连接**（见 `ttrade/hubreg.py`）。

        组合页可能正在用同一条 CTP 会话跑策略 —— 这里无脑 `hub.close()` 会把
        它的持仓和委托回报一起掐掉，而组合页那边看着只是「数据不更新了」。
        `release()` 还对天勤的 hub 返回 True（它不进注册表），行为不变。
        """
        for hub in self._opened:
            try:
                if hubreg.release(hub):
                    hub.close()
            except Exception:  # noqa: BLE001
                pass
        self._opened = []
        self.hubs.clear()

    # ---- 写：按账户路由 ----

    def _owner(self, account: str = ""):
        """账户名 → 它所在柜台的 hub。

        找不到（或没指定）时退回行情源那一路 —— 界面单选一个账户时 `account`
        一定在，自动交易那条链路可能不传，用它兜底比直接报错好。
        """
        name = str(account or "")
        if name:
            for hub in self.hubs.values():
                for spec in getattr(hub, "specs", []) or []:
                    if getattr(spec, "name", "") == name:
                        return hub
        hub = self.hubs.get(self.feed)
        if hub is None:
            raise RuntimeError("没有可用的柜台连接")
        return hub

    def place(self, symbol: str, direction: str, offset: str, volume: int,
              limit_price: float | None = None, account: str = "") -> str:
        return self._owner(account).place(symbol, direction, offset, volume,
                                         limit_price, account=account)

    def cancel(self, order_id: str, account: str = "") -> None:
        self._owner(account).cancel(order_id, account=account)

    def set_target(self, symbol: str, volume: int, account: str = "") -> None:
        self._owner(account).set_target(symbol, volume, account=account)

    def quote_of(self, symbol: str, account: str = "",
                 timeout: float = 3.0) -> dict:
        """按账户路由到它那条连接去取最新盘口（下单面板「填最新价」）。"""
        return self._owner(account).quote_of(symbol, timeout=timeout)

    def watch(self, symbols) -> None:
        """每个柜台各订一遍「报价」框要盯的合约。

        **全订而不是只订行情源**：报价框显示的是行情源的价格，但行情源掉线时
        会在剩下的柜台里换一路，事先只订一家的话换过去就全是「—」。订阅本身
        幂等，多订十几行合约的代价可以忽略。
        """
        for hub in self.hubs.values():
            try:
                hub.watch(symbols)
            except Exception:  # noqa: BLE001 - 一个柜台订不上不该拖死整条会话
                pass


class LiveWorker(QObject):
    """跑在实时线程里：（可选）补历史 → 连接各柜台 → 订阅 → 循环推进。

    `need_history` 只在要画 K 线时才需要（研究页要把实时推送拼到本地历史上）。
    纯看账户的页面传 False —— 补历史一个品种十年 5m 要几十秒到几分钟，而账户
    信息在 `hub.start()` 之后就有了，没必要让用户干等。
    """

    ready = Signal(str)          # 订阅就绪，参数是「品种_交易所」
    bars = Signal(object)        # 实时 K 线 DataFrame（含未走完的最后一根）
    state = Signal(object)       # LiveState
    failed = Signal(str)
    log = Signal(str)
    order_sent = Signal(str)     # 下单成功，参数是 order_id

    def __init__(self, spec, symbol: str, exchange: str, period: str,
                 need_history: bool = True, feed: str = "") -> None:
        super().__init__()
        # `spec` 可以是一个 AccountSpec，也可以是列表（多账户 / 多柜台）
        self.spec = spec
        self.symbol = symbol
        self.exchange = exchange
        self.period = period
        self.need_history = need_history
        self.feed = feed
        self._stop = False
        self._queue: queue.Queue = queue.Queue()

    # ---- 主线程投递的操作 ----

    def submit(self, fn) -> None:
        """把一个 `fn(hubset)` 丢给 worker 线程执行。"""
        self._queue.put(fn)

    def stop(self) -> None:
        self._stop = True

    # ---- 线程内 ----

    def run(self) -> None:
        groups = group_specs(self.spec)
        # 纯行情会话（specs 为空）：只有天勤那一路
        if not groups:
            groups = {"tq": []}

        if self.need_history and "tq" in groups:
            # 补历史走 DataDownloader，可能要几十秒；放在线程里，界面不冻。
            # 失败**不能**拖累连接：账户能登上去比历史补齐重要得多，而且
            # 天勤运维窗口（19:00-19:30）里 update_main 会一直等到 19:31，
            # 那种情况下抛出来连账户都看不到。
            try:
                from ttrade.tqfetch import update_main

                self.log.emit("正在补齐本地历史…")
                n = update_main(self.symbol, self.exchange, self.period)
                self.log.emit(f"本地历史已补齐（本次写入 {n:,} 行）")
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"补历史失败（{type(exc).__name__}: {exc}），继续连接")

        # ---- 连各柜台。一个失败不影响其它：这就是「同时登录」的意义 ----
        hubs: dict = {}
        errors: dict = {}
        for name, specs in groups.items():
            label = BACKEND_LABELS.get(name, name)
            hub = None
            try:
                hub = make_hub(name, specs)
                if hubreg.refs(hub) > 1:
                    # 组合页已经为这个账户连过了：拿的是同一条连接。直说，省得
                    # 用户以为这里又登了一次（账户表上的「已连接」也不会重复）
                    self.log.emit(f"{label} 复用组合页已连的会话，不再重复登录")
                else:
                    self.log.emit(f"正在连接{label}…")
                hub.start()          # 幂等：复用已有会话时立刻返回
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"{type(exc).__name__}: {exc}"
                self.log.emit(f"{label} 连接失败：{errors[name]}")
                if hub is not None:
                    # 借到手却没连上：把引用还回去，否则注册表里永远多一个
                    # 「refs 卡着不降」的僵尸，组合页那条会话再也关不掉
                    try:
                        if hubreg.release(hub):
                            hub.close()
                    except Exception:  # noqa: BLE001
                        pass
                continue
            hubs[name] = hub

        if not hubs:
            detail = "；".join(f"{BACKEND_LABELS.get(n, n)} {m}"
                              for n, m in errors.items())
            self.failed.emit(detail or "没有可连接的账户")
            return

        hs = HubSet(hubs, self.feed)
        bad = hs.subscribe(self.symbol, self.exchange, self.period)
        for name in bad:
            label = BACKEND_LABELS.get(name, name)
            self.log.emit(f"{label} 订阅 {self.symbol} 失败，行情不来自它")
        if hs.feed in bad or hs.feed not in hubs:
            # 选定的行情源订不上：说清楚，别让用户对着空图猜
            self.log.emit(
                f"行情源（{BACKEND_LABELS.get(hs.feed, hs.feed)}）没有订阅成功，"
                "K 线会是空的；账户和下单不受影响")
        self.log.emit(
            "已连接：" + "、".join(
                f"{BACKEND_LABELS.get(n, n)}（{len(getattr(h, 'specs', []) or [])} 户）"
                for n, h in hubs.items())
            + f"；行情源 {BACKEND_LABELS.get(hs.feed, hs.feed)}")
        self.ready.emit(f"{self.symbol}_{self.exchange}")

        last_bar = None
        last_state = 0.0
        try:
            while not self._stop:
                for name in list(hs.hubs):
                    hub = hs.hubs.get(name)
                    if hub is None:
                        continue
                    try:
                        hub.wait(max(0.15, 0.8 / max(1, len(hs.hubs))))
                    except Exception as exc:  # noqa: BLE001
                        label = BACKEND_LABELS.get(name, name)
                        if name == hs.feed:
                            # 行情源断了，图就画不下去了 —— 整条会话收尾
                            self.failed.emit(f"{label}行情连接中断：{exc}")
                            self._stop = True
                        else:
                            # 账户柜台断了先摘掉，剩下的继续跑（持仓还在别的户里）
                            self.log.emit(f"{label}连接中断：{exc}，已从本次会话摘除")
                            hs.hubs.pop(name, None)
                            if not hs.hubs:
                                self.failed.emit("所有柜台都已断开")
                                self._stop = True
                        break
                if self._stop:
                    break

                # 排队的界面操作：下单/撤单必须在这个线程里执行
                while True:
                    try:
                        fn = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        fn(hs)
                    except Exception as exc:  # noqa: BLE001
                        self.failed.emit(f"操作失败：{type(exc).__name__}: {exc}")

                df = hs.bars()
                sig = None if df is None or df.empty else (
                    df.index[-1], float(df["close"].iloc[-1]), len(df)
                )
                if sig != last_bar:
                    last_bar = sig
                    self.bars.emit(df)

                now = time.monotonic()
                if now - last_state > 0.4:
                    last_state = now
                    self.state.emit(hs.state())
        finally:
            hs.close()


class _Session:
    """一条会话：一个账户 + 一个线程 + 一个 worker + 它最近一次的账户快照。

    `state` 存最近一份快照：合并多会话时要用，而且某个账户断开之后，别的账户
    的快照还在（不能因为这一条没了就把整页清空）。
    """

    __slots__ = ("name", "spec", "backend", "worker", "thread",
                 "symbol", "period", "state")

    def __init__(self, name: str, spec, backend: str) -> None:
        self.name = name
        self.spec = spec
        self.backend = backend
        self.worker: LiveWorker | None = None
        self.thread: QThread | None = None
        self.symbol = ""
        self.period = ""
        self.state = None


# 纯行情会话（`spec=None`，研究页的「启动实时」）在注册表里用的键。用 `\x00`
# 打头是为了不可能撞上任何一个真账户名。
MARKET_KEY = "\x00market"


def _session_key(spec) -> str:
    """账户名就是会话键。`spec=None` 是纯行情会话。"""
    if spec is None:
        return MARKET_KEY
    return str(getattr(spec, "name", "") or MARKET_KEY)


def _as_list(spec) -> list:
    if spec is None:
        return [None]
    if isinstance(spec, (list, tuple)):
        return list(spec) or [None]
    return [spec]


class LiveController(QObject):
    """主线程这一侧：起停线程、转发信号、排队界面操作。

    内部是一张**会话注册表** `{账户名: _Session}`。`start()` 可以一次登录一批
    （「连接」按钮），也可以只登录一个（账户表每行的「登录」按钮），已经连着的
    账户一律不动 —— 这正是「逐账户独立登录 / 登出」的实现方式。
    """

    bars = Signal(object)
    state = Signal(object)
    log = Signal(str)
    failed = Signal(str)
    started = Signal(str)
    stopped = Signal()
    order_sent = Signal(str)
    quote_got = Signal(object)      # 下单面板「填最新价」的回执（异步）
    sessions = Signal(object)       # 当前连着哪些账户（set[str]）

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sessions: dict[str, _Session] = {}
        self.spec = None
        self.key = ""
        self.feed = ""
        # 这条会话实际用的周期。`_on_live_started` 里不能拿研究页当前选的周期
        # 去猜 —— 交易页连的时候可能按 5m 连，而研究页正停在日线上。
        self.period = ""
        # `started` 只在「第一个就绪的会话」上发一次：三条会话同一个品种，发三次
        # 会让上层把 K 线状态重置三遍（每遍都要重读一次本地历史）
        self._emitted_key = ""
        self._stop_notified = False

    # ---- 注册表 ----

    @property
    def running(self) -> bool:
        """有没有任何一条会话在跑。上层只关心「连没连上」时用这个。"""
        return bool(self._sessions)

    @property
    def running_names(self) -> set:
        """连着哪些**账户**（纯行情会话不算账户）。"""
        return {n for n in self._sessions if n != MARKET_KEY}

    def is_running(self, name: str) -> bool:
        return str(name or "") in self._sessions

    def _primary(self) -> _Session | None:
        """K 线与盘口的来源。多柜台时认 `feed` 指定的那家，否则取第一条。"""
        if self.feed:
            for s in self._sessions.values():
                if s.backend == self.feed:
                    return s
        return next(iter(self._sessions.values()), None)

    def _session_for(self, account: str = "") -> _Session | None:
        """账户名 → 它那条会话。没指定（自动交易链路不传）就退回主会话。"""
        name = str(account or "")
        if name and name in self._sessions:
            return self._sessions[name]
        if name:
            for s in self._sessions.values():
                if getattr(s.spec, "name", "") == name:
                    return s
        return self._primary()

    def _merged(self):
        parts = [(s.backend, s.state) for s in self._sessions.values()
                 if s.state is not None]
        return merge_states(parts, self.feed)

    def _emit_state(self) -> None:
        self.state.emit(self._merged())

    def _emit_sessions(self) -> None:
        self.sessions.emit(self.running_names)

    # ---- 起停 ----

    def start(self, spec, symbol: str, exchange: str, period: str,
              need_history: bool = True, feed: str = "") -> None:
        """登录账户。`spec` 可以是一个、也可以是一批；**已经在跑的账户不动**。

        `feed` 是行情源（`tq` / `ctp` / `rohon`），多柜台同时登录时才需要传；
        空表示按固定顺序挑第一个连上的。**只传一次就记住** —— 单点某个账户登录
        时界面不会传 feed，不能把它清掉。
        """
        if feed:
            self.feed = feed
        self.spec = spec
        self.period = period
        for s in _as_list(spec):
            name = _session_key(s)
            old = self._sessions.get(name)
            if old is not None:
                # 已经连着了：品种 / 周期没变就原样留着。重复 start 等于把用户
                # 刚点的那条会话掐掉再重开，盘口会闪一下、委托列表也会空一拍
                if old.symbol == symbol and old.period == period:
                    continue
                self.stop_one(name, wait=True)
            self._start_one(s, name, symbol, exchange, period, need_history)

    def _start_one(self, spec, name: str, symbol: str, exchange: str,
                   period: str, need_history: bool) -> None:
        sess = _Session(name, spec, (getattr(spec, "backend", "tq") or "tq"))
        sess.symbol, sess.period = symbol, period
        worker = LiveWorker(spec, symbol, exchange, period, need_history,
                            self.feed)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ready.connect(lambda key, n=name: self._on_ready(n, key))
        worker.bars.connect(lambda df, n=name: self._on_bars(n, df))
        worker.state.connect(lambda st, n=name: self._on_state(n, st))
        worker.log.connect(self.log)
        worker.failed.connect(lambda msg, n=name: self._on_failed(n, msg))
        worker.order_sent.connect(self.order_sent)
        # `run()` 有两种提前返回：连接失败、行情中断。两种都不会回到 while 循环，
        # 而线程的事件循环还开着 —— 不显式 quit 的话 `finished` 永不发，
        # `stopped` 也就永远不发，界面会一直以为还连着。
        worker.failed.connect(thread.quit)
        # worker 循环结束就退线程，再让 Qt 回收
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda n=name: self._on_thread_finished(n))
        sess.worker, sess.thread = worker, thread
        self._sessions[name] = sess
        self._stop_notified = False
        thread.start()
        self._emit_sessions()

    def stop(self, wait: bool = False) -> None:
        """断开**全部**账户。"""
        for name in list(self._sessions):
            self.stop_one(name, wait=wait)

    def stop_one(self, name: str, wait: bool = False) -> bool:
        """只断开这一个账户，其余会话照跑。返回是否真停了一条。"""
        sess = self._sessions.pop(str(name or ""), None)
        if sess is None or sess.worker is None:
            return False
        sess.worker.stop()
        if wait:
            # 最多等 5 秒：worker 里可能正卡在 wait_update(0.8) 或一次下载里
            sess.thread.quit()
            sess.thread.wait(5000)
        else:
            sess.thread.quit()
        # 这一路的账户快照要立刻扔掉，不然界面上它还挂在那儿（数字不再更新，
        # 看着像「连着的账户不动了」）
        self._emit_state()
        self._emit_sessions()
        return True

    # ---- worker 回传 ----

    def _on_ready(self, name: str, key: str) -> None:
        if name not in self._sessions:
            return
        self.key = key
        if key and key != self._emitted_key:
            self._emitted_key = key
            self.started.emit(key)
        self._emit_sessions()

    def _on_bars(self, name: str, df) -> None:
        """只转发**行情源**那一路的 K 线。

        两条会话各推一份同一品种的 bar，都往上画就是「一根 bar 反复重算」。
        """
        sess = self._primary()
        if sess is None or sess.name != name:
            return
        self.bars.emit(df)

    def _on_state(self, name: str, st) -> None:
        sess = self._sessions.get(name)
        if sess is None:
            return
        sess.state = st
        self._emit_state()

    def _on_failed(self, name: str, msg: str) -> None:
        # 只有一个账户时别加前缀（「天勤：还没连上柜台」读着别扭），多账户才加
        if len(self._sessions) > 1:
            msg = f"{name}：{msg}"
        self.failed.emit(msg)

    def _on_thread_finished(self, name: str) -> None:
        self._sessions.pop(name, None)
        self._emit_state()
        self._emit_sessions()
        if self._sessions or self._stop_notified:
            return
        self._stop_notified = True
        self._emitted_key = ""
        self.stopped.emit()

    # ---- 界面操作：全部排队给 worker ----

    def place(self, symbol: str, direction: str, offset: str, volume: int,
              limit_price: float | None, account: str = "") -> bool:
        sess = self._session_for(account)
        if sess is None or sess.worker is None:
            self.failed.emit("还没连上柜台，无法下单")
            return False

        def job(hub):
            oid = hub.place(symbol, direction, offset, volume, limit_price,
                            account=account)
            self.order_sent.emit(oid)

        sess.worker.submit(job)
        return True

    def cancel(self, order_id: str, account: str = "") -> bool:
        sess = self._session_for(account)
        if sess is None or sess.worker is None:
            self.failed.emit("还没连上柜台，无法撤单")
            return False

        def job(hub):
            hub.cancel(order_id, account=account)

        sess.worker.submit(job)
        return True

    def set_target(self, symbol: str, volume: int, account: str = "") -> bool:
        """把某个合约的目标净仓交给 TargetPosTask（自动交易用）。"""
        sess = self._session_for(account)
        if sess is None or sess.worker is None:
            return False

        def job(hub):
            hub.set_target(symbol, volume, account=account)

        sess.worker.submit(job)
        return True

    def quote_of(self, symbol: str, account: str = "",
                 timeout: float = 3.0) -> bool:
        """取某个合约的最新价。**异步**：结果从 `quote_got` 回来。

        取价要等行情推一帧（天勤新订阅第一次是 NaN、CTP 要等柜台推 tick），
        放在主线程会把界面卡住 2~3 秒，所以排进 worker 队列。
        """
        sess = self._session_for(account)
        if sess is None or sess.worker is None:
            self.failed.emit("还没连上柜台，取不到最新价")
            return False

        def job(hub):
            try:
                info = hub.quote_of(symbol, account=account, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - 取价失败不该炸掉整条会话
                self.failed.emit(f"取最新价失败：{type(exc).__name__}: {exc}")
                return
            self.quote_got.emit(dict(info or {}, symbol=symbol))

        sess.worker.submit(job)
        return True

    def watch(self, symbols) -> bool:
        """让「报价」框盯一批合约（`SHFE.rb2601` 这种天勤写法）。

        **所有会话都订**，不按账户路由：报价框显示的是行情源那一路的价格，而
        行情源掉线时会在剩下的柜台里换一家 —— 只订当前选中账户那一路的话，换
        过去就全是「—」。`merge_states` 已经保证只把行情源那份画出来。

        **异步**：订上之后下一次 `state()` 才有数据（天勤要等第一帧 tick）。
        调用方负责去重 —— 本函数每次都真的排队，0.4 秒一次会刷满柜台日志。
        """
        syms = [str(s) for s in (symbols or ()) if str(s or "").strip()]
        if not syms:
            return False

        def job(hub):
            try:
                hub.watch(syms)
            except Exception as exc:  # noqa: BLE001 - 订不上只是那几行空着
                self.failed.emit(f"报价订阅失败：{type(exc).__name__}: {exc}")

        sent = 0
        for sess in list(self._sessions.values()):
            if sess.worker is None:
                continue
            sess.worker.submit(job)
            sent += 1
        return sent > 0
