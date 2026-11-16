"""把实时会话接到 Qt 上。

分工：

- `LiveWorker` 跑在**独立线程**里，独占一条或多条连接：天勤 `LiveHub`（一条
  TqApi）、CTP `CtpHub`、融航 `RohonHub`。**同时登录多个柜台**时，每个柜台
  各一条连接，进程里同时存在 —— 这是「天勤和 CTP 一起登」的前提。
- `LiveController` 活在主线程，负责起停线程、把界面操作**排队**丢给 worker，
  再把 worker 的信号转出来。

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
    state/place/cancel/set_target），worker 循环不需要知道底下是谁。"""
    if backend == "ctp":
        from ttrade.ctp_live import CtpHub

        return CtpHub(specs)
    if backend == "rohon":
        from ttrade.ctp_live import RohonHub

        return RohonHub(specs)
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
        for hub in list(self.hubs.values()):
            try:
                hub.close()
            except Exception:  # noqa: BLE001
                pass
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
            try:
                hub = make_hub(name, specs)
                self.log.emit(f"正在连接{label}…")
                hub.start()
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"{type(exc).__name__}: {exc}"
                self.log.emit(f"{label} 连接失败：{errors[name]}")
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


class LiveController(QObject):
    """主线程这一侧：起停线程、转发信号、排队界面操作。"""

    bars = Signal(object)
    state = Signal(object)
    log = Signal(str)
    failed = Signal(str)
    started = Signal(str)
    stopped = Signal()
    order_sent = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: LiveWorker | None = None
        self.spec = None
        self.key = ""
        self.feed = ""
        # 这条会话实际用的周期。`_on_live_started` 里不能拿研究页当前选的周期
        # 去猜 —— 交易页连的时候可能按 5m 连，而研究页正停在日线上。
        self.period = ""

    @property
    def running(self) -> bool:
        return self._worker is not None

    def start(self, spec, symbol: str, exchange: str, period: str,
              need_history: bool = True, feed: str = "") -> None:
        """启动实时会话。已经在跑会先停掉 —— 同一时刻只留一条会话。

        `feed` 是行情源（`tq` / `ctp` / `rohon`），多柜台同时登录时才需要传；
        空表示按固定顺序挑第一个连上的。
        """
        self.stop(wait=True)
        self.spec = spec
        self.period = period
        self.feed = feed
        worker = LiveWorker(spec, symbol, exchange, period, need_history, feed)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ready.connect(self.started)
        worker.bars.connect(self.bars)
        worker.state.connect(self.state)
        worker.log.connect(self.log)
        worker.failed.connect(self.failed)
        worker.order_sent.connect(self.order_sent)
        # `run()` 有两种提前返回：连接失败、行情中断。两种都不会回到 while 循环，
        # 而线程的事件循环还开着 —— 不显式 quit 的话 `finished` 永不发，
        # `stopped` 也就永远不发，界面会一直以为还连着。
        worker.failed.connect(thread.quit)
        # worker 循环结束就退线程，再让 Qt 回收
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._worker, self._thread = worker, thread
        thread.start()

    def stop(self, wait: bool = False) -> None:
        worker, thread = self._worker, self._thread
        if worker is None or thread is None:
            return
        self._worker, self._thread = None, None
        worker.stop()
        if wait:
            # 最多等 5 秒：worker 里可能正卡在 wait_update(0.8) 或一次下载里
            thread.quit()
            thread.wait(5000)
        else:
            thread.quit()

    def _on_thread_finished(self) -> None:
        self.stopped.emit()

    # ---- 界面操作：全部排队给 worker ----

    def place(self, symbol: str, direction: str, offset: str, volume: int,
              limit_price: float | None, account: str = "") -> bool:
        if self._worker is None:
            self.failed.emit("还没连上柜台，无法下单")
            return False

        def job(hub):
            oid = hub.place(symbol, direction, offset, volume, limit_price,
                            account=account)
            self.order_sent.emit(oid)

        self._worker.submit(job)
        return True

    def cancel(self, order_id: str, account: str = "") -> bool:
        if self._worker is None:
            self.failed.emit("还没连上柜台，无法撤单")
            return False

        def job(hub):
            hub.cancel(order_id, account=account)

        self._worker.submit(job)
        return True

    def set_target(self, symbol: str, volume: int, account: str = "") -> bool:
        """把某个合约的目标净仓交给 TargetPosTask（自动交易用）。"""
        if self._worker is None:
            return False

        def job(hub):
            hub.set_target(symbol, volume, account=account)

        self._worker.submit(job)
        return True
