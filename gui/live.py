"""把天勤实时会话接到 Qt 上。

分工：

- `LiveWorker` 跑在**独立线程**里，独占一个 `LiveHub`（也就是一条 TqApi 连接）。
  TqApi 不是线程安全的，所有 `get_*` / `insert_order` / `cancel_order` 都必须在
  它自己的线程里调用。
- `LiveController` 活在主线程，负责起停线程、把界面操作**排队**丢给 worker，
  再把 worker 的信号转出来。

为什么下单要排队而不是直接调：`insert_order()` 只是入队，真正的报文在下一次
`wait_update()` 发出；而 `wait_update()` 正在 worker 线程里阻塞着。跨线程直接调
`insert_order()` 会写坏 TqApi 内部的订阅状态，表现为「单子偶尔丢、偶尔重复」。
所以界面只往队列里塞一个闭包，worker 每轮循环取出来执行。
"""

from __future__ import annotations

import queue
import time

from PySide6.QtCore import QObject, QThread, Signal

from ttrade.tqlive import LiveHub, LiveState


class LiveWorker(QObject):
    """跑在天勤线程里：（可选）补历史 → 订阅 → 循环推进。

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
                 need_history: bool = True) -> None:
        super().__init__()
        self.spec = spec
        self.symbol = symbol
        self.exchange = exchange
        self.period = period
        self.need_history = need_history
        self._stop = False
        self._queue: queue.Queue = queue.Queue()

    # ---- 主线程投递的操作 ----

    def submit(self, fn) -> None:
        """把一个 `fn(hub)` 丢给 worker 线程执行。"""
        self._queue.put(fn)

    def stop(self) -> None:
        self._stop = True

    # ---- 线程内 ----

    def run(self) -> None:
        hub = LiveHub(self.spec)
        try:
            if self.need_history:
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
            self.log.emit("正在连接天勤…")
            hub.start()
            hub.subscribe(self.symbol, self.exchange, self.period)
            self.ready.emit(f"{self.symbol}_{self.exchange}")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            hub.close()
            return

        last_bar = None
        last_state = 0.0
        try:
            while not self._stop:
                try:
                    hub.wait(0.8)
                except Exception as exc:  # noqa: BLE001
                    self.failed.emit(f"行情连接中断：{exc}")
                    break

                # 排队的界面操作：下单/撤单必须在这个线程里执行
                while True:
                    try:
                        fn = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        fn(hub)
                    except Exception as exc:  # noqa: BLE001
                        self.failed.emit(f"操作失败：{type(exc).__name__}: {exc}")

                df = hub.bars()
                sig = None if df.empty else (
                    df.index[-1], float(df["close"].iloc[-1]), len(df)
                )
                if sig != last_bar:
                    last_bar = sig
                    self.bars.emit(df)

                now = time.monotonic()
                if now - last_state > 0.4:
                    last_state = now
                    self.state.emit(hub.state())
        finally:
            hub.close()


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

    @property
    def running(self) -> bool:
        return self._worker is not None

    def start(self, spec, symbol: str, exchange: str, period: str,
              need_history: bool = True) -> None:
        """启动实时会话。已经在跑会先停掉 —— 同一时刻只留一条连接。"""
        self.stop(wait=True)
        self.spec = spec
        worker = LiveWorker(spec, symbol, exchange, period, need_history)
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
              limit_price: float | None) -> bool:
        if self._worker is None:
            self.failed.emit("还没连上天勤，无法下单")
            return False

        def job(hub):
            oid = hub.place(symbol, direction, offset, volume, limit_price)
            self.order_sent.emit(oid)

        self._worker.submit(job)
        return True

    def cancel(self, order_id: str) -> bool:
        if self._worker is None:
            self.failed.emit("还没连上天勤，无法撤单")
            return False

        def job(hub):
            hub.cancel(order_id)

        self._worker.submit(job)
        return True

    def set_target(self, symbol: str, volume: int) -> bool:
        """把某个合约的目标净仓交给 TargetPosTask（自动交易用）。"""
        if self._worker is None:
            return False

        def job(hub):
            hub.set_target(symbol, volume)

        self._worker.submit(job)
        return True
