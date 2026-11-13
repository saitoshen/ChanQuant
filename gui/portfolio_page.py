"""「策略交易和组合」页。

和「策略研究」「账户透视和交易」平级的第三个大页面。它管的是**跑起来之后的事**：
组合里有哪些策略、每个绑哪个账户、现在跑到哪一步、账户仓位和策略理论仓位对不对得上。
页面结构：

    顶部  默认账户 / 启动全部 / 停止全部 / 状态灯
    左侧  策略列表：增删、单独启停、从策略库添加、绑定交易账户、查看回测
    右侧  监控器：逐合约列出「策略理论 / 账户实际 / 差额」，一键对齐
    底部  追单设置：等几秒撤单重报、追几次、每次追几跳

## 为什么这里也有一套 K 线 / 绩效 / 因子

「策略研究」页里那三页是**调参时**看的：改一个滑杆，图跟着变。但策略存进组合
之后，用户想知道的是「我存下的这一条，历史表现到底怎么样」—— 那不该让他跑回
研究页把参数一格一格调回来。所以双击列表里的一行，就地打开同一套结果视图。

结果视图是 `gui/result_view.ResultView`，和研究页**同一份实现**：一处改动两处
生效，不会出现「这边修好了那边还是老图」。

## 组合报告为什么只有一页绩效

单选一条策略：K 线、绩效对比（对买入持有）、因子分析、交易记录、阶段分析。
多选：只给绩效、交易记录、阶段分析 —— 组合是跨品种、跨周期加权，没有单一
对照物，硬拉一个「等权买入持有」只会误导。

## 为什么按账户分多个 Runner

天勤的 `TqApi` 在构造时绑定资金账号，一个 api 只能对一个账户；同账号再开第二个
连接还会把第一个踢下线。所以「策略 A 跑在账户甲、策略 B 跑在账户乙」只能靠
多个 Runner —— 每个账户一条连接。这一页负责把组合按 `cfg.account` 分组，
一组一个 `RunnerWorker`。

`cfg.account` 为空表示「跟随默认账户」，也就是顶部那个下拉框。

## 为什么启停要经过队列

和 `gui/live.py` 同一个道理：`TqApi` 不是线程安全的，`insert_order` / `cancel_order`
必须在它自己的线程里调。界面线程只往队列里塞闭包，worker 每轮 `wait_update`
之后统一执行。

## 一键对齐为什么连未托管仓位一起平

用户对这个按钮的预期是「账户和策略列表完全一致」。只补托管合约的话，策略停掉后
留在账户里的仓位、手工开的仓会一直挂在那儿 —— 监控器上永远红着，而按钮点下去
毫无反应。所以 `align()` 默认把未托管合约也平到 0。

但平仓是真金白银，而且这些仓位往往是用户自己开的。所以点下去先弹确认框，把
「哪个合约、现在几手、变成几手」一条条列出来，确认了才发单。
`align(close_orphans=False)` 保留「只补托管」的老行为，给脚本用。

## 为什么按两条腿比对，不按净仓

策略的 `target` 是个净数（正多负空），落成目标只能是「只多」或「只空」；
账户那边多空可以**同时**非零，也就是锁仓。净仓把锁仓压成 0 ——
多 4 空 4 和「什么仓都没有」算出来一模一样。

2026-11-11 的实盘就是这么漏的：`ag2612` 多 4 空 4、`cu2612` 多 1 空 1、
`lc2701` 多 1 空 1，三笔合起来 47 万保证金占着，监控器报「一致」，
一键对齐一笔都不发。所以 `mismatch()` 比的是 `(多, 空)` 两条腿，
`align()` 也按腿发单：真要平两边时拆成「卖平多 + 买平空」两笔 ——
它们各自只动自己那一侧，同时发是安全的；换成「先平后开」就不行，
第二笔会把第一笔还没成交的量重复算进可平量，发出去就是超量平仓。

锁仓是**会自己长出来**的，不是脏数据：策略反手时，可平量按「自己账本上那部分」
算（`_closeable`），账户里属于别人的反向仓一概不碰。假设账户已有 4 手空、
某个 held=0 的策略要开 4 手多，`closeable=0`，`_legs` 的可平量被压成 0，
这 4 手只能发成 OPEN —— 账户就成了「多 4 空 4」。净额 0、保证金翻倍。
这类仓位只能靠一键对齐收掉，所以它必须被看见。

## 为什么持仓没同步时不让点

`mismatch()` 在 `_pos is None`（刚连上、持仓还没推过来）时一定返回空。
那是「还不知道」，不是「一致」。这时候点一键对齐，等于拿「账户是空的」
去算差额，会把整个理论仓位重新开一遍。所以快照里带 `pos_ready`，
界面分开提示、`align()` 直接拒。同理，没挂策略的账户根本没有连接，
它的持仓一律不在比对范围 —— 标签里会点名，不能让「一致」两个字蒙过去。

## 为什么策略列表里没有「实际仓位」

账户是**一个**，策略是**一群**。同一个品种完全可能被两条策略同时交易，账户里的
净仓是它们的和 —— 单看账户分不出哪几手是谁的。而且用户可能同时在别的平台上跑
自动交易，或者自己手工下单，这些仓也都混在同一个账户里。

所以「每条策略的实际仓位」这个东西根本不存在。早先那栏显示的是账户净仓，再拿它
和理论仓比对、不一致就自动发单补齐，结果是：

- 两条策略做同一个品种时互相把对方的仓当成自己的，一条要 -11、一条要 0，
  两边都认为自己差 11 手，来回对打；
- 用户手工开一笔仓，策略立刻把它当成自己的偏差平掉 —— 甚至开成反向仓。

现在策略只跟**自己记的仓位**比（`ttrade/positionbook.py`，按自己委托的成交累计、
落盘），账户里多出来的部分一概不碰。要对齐账户，只有这里的一键对齐一个入口。
"""

from __future__ import annotations

import os
import queue
import time

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.pickers import SymbolPicker
from gui.result_view import PERIOD_LABEL, ResultView, ViewCtx, set_tab_visible
from gui.workspace_bar import AllocDialog, WorkspaceBar
from ttrade import replay
from ttrade import cross as cross_engine
from ttrade import portfolio_plan as PL
from ttrade import strategy_store as SStore
from ttrade import tqlive
from ttrade import workspace_store as WS
from ttrade.tqlive import merge_live
from ttrade.backtest import combine, combine_trades
from ttrade.strategies import STRATEGIES, param_label
from ttrade.strategy_runner import PortfolioRunner
from ttrade.universe import cn_name

# 与 trade_page.py / app.py 同一套配色。不 import 那两处：app 会 import 本模块。
C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_UP = "#d92b2b"
C_DOWN = "#0f9d58"
C_OK = "#16a34a"
C_BAD = "#dc2626"
C_IDLE = "#94a3b8"


def _int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _pos_text(actual, r: dict) -> str:
    """账户持仓写成一行。锁仓要带两条腿，不然 `0` 看不出有仓。"""
    if r.get("locked"):
        return f"{actual}（多{r.get('a_long')} 空{r.get('a_short')}）"
    return f"{actual:+d} 手"


def _delta_text(r: dict) -> str:
    """监控器「差额」一栏的文字。

    不能只写净额：锁仓（多 4 空 4）净差额是 0，一栏 `+0` 会让人以为不用动。
    按两条腿分别算，写成「平多4　平空4」这样一眼能对上单子。
    """
    a_long, a_short = _int(r.get("a_long")), _int(r.get("a_short"))
    t_long, t_short = _int(r.get("t_long")), _int(r.get("t_short"))
    bits = []
    if a_long > t_long:
        bits.append(f"平多{a_long - t_long}")
    elif a_long < t_long:
        bits.append(f"开多{t_long - a_long}")
    if a_short > t_short:
        bits.append(f"平空{a_short - t_short}")
    elif a_short < t_short:
        bits.append(f"开空{t_short - a_short}")
    if not bits:
        return "0"
    return "　".join(bits)


# ------------------------------------------------------------------ 线程层


def run_replay_job(cfg, kind: str, payload, with_bh: bool = True):
    """跑一份**已经取好数**的回放。纯计算，可以在后台线程里跑。

    `kind` 是取数方式（`cross` / `multi` / `single`），`payload` 是对应的行情。
    取数留在主线程是有意的：`DataStore` 是主线程的缓存，后台线程同时读同一个
    DuckDB 连接不安全（研究页那条多品种也是先取数再进线程的）。
    """
    # payload 可能是 `(行情, 跨周期行情)` 这种两元组：别的周期的行情在主线程
    # 里已经读好了（`_prepare`），后台线程只算。没配跨周期因子时就是单份行情
    extra = None
    if isinstance(payload, tuple):
        payload, extra = payload
    if kind == "cross":
        # 两份都是 `{品种: {周期: 行情}}`，同一份就够 —— 截面腿和时序腿可以指到
        # 不同周期上，但读的都是这几个品种的那几档行情
        return replay.run_cross(cfg, payload, timing_extra=extra, factor_extra=extra)
    if kind == "multi":
        return replay.run_multi(cfg, payload, extras=extra)
    return replay.run(cfg, payload, with_bh=with_bh, extra=extra)


class ReplayWorker(QObject):
    """在后台跑「查看回测 / 组合报告」。

    多品种一次是 N 遍策略计算，机器学习还要各训几十折 —— 组合页那条 19 品种 /
    5m 十年的策略实测十几分钟。同步跑的话整个窗口「未响应」，连「停止」都点
    不动，用户只能强杀进程。

    进度报不了折（`replay` 那层没接回调），所以这里只报「第几条 / 共几条」：
    组合报告几条策略时至少知道还在动。
    """

    done = Signal(object)      # [(cfg, Replay), ...]
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, jobs: list) -> None:
        super().__init__()
        self.jobs = list(jobs)

    def run(self) -> None:
        out = []
        try:
            for i, (cfg, kind, payload) in enumerate(self.jobs, 1):
                self.progress.emit(i, len(self.jobs), cfg.title())
                out.append((cfg, run_replay_job(cfg, kind, payload)))
        except Exception as exc:  # noqa: BLE001 - 错误要回到界面上说清楚
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(out)


class MeasureWorker(QObject):
    """在后台量「这条策略实际压了多少钱」。

    为什么必须后台：一条 5m 十年的策略回测要几秒到几十秒，工作区里几十条，
    同步跑窗口会「未响应」。这和「组合报告」那条线程分开 —— 量占用只关心
    `res.position` 和行情，不需要出图。

    占用口径和 `ttrade/portfolio_plan.py` 的 `measure()` 是同一份实现，
    这里一行算法都不重写：两处各算一套的话，界面显示的占用和配权用的占用
    迟早会不一样。
    """

    progress = Signal(int, int, str)
    done = Signal(object)      # ([PL.Usage], [没量出来的说明])
    failed = Signal(str)

    def __init__(self, jobs: list) -> None:
        super().__init__()
        self.jobs = list(jobs)

    def run(self) -> None:
        out, bad = [], []
        try:
            for i, (cfg, kind, payload) in enumerate(self.jobs, 1):
                self.progress.emit(i, len(self.jobs), cfg.title())
                try:
                    rp = run_replay_job(cfg, kind, payload, with_bh=False)
                    if not rp.ok:
                        bad.append(f"{cfg.title()}（{rp.error[:40]}）")
                        continue
                    # 占用要「持仓 × 收盘」，所以得有一份和 `res.position` 同索引
                    # 的行情。单品种用回测真正吃进去的那份（`rp.raw`），
                    # 多品种 / 截面是逐品种拼出来的组合曲线，没有单一收盘价 ——
                    # 传 None，`measure()` 会退回「只按波动率配权」并留一句备注
                    frame = getattr(rp, "raw", None)
                    if frame is None and kind == "single":
                        frame = payload[0] if isinstance(payload, tuple) else payload
                    mk = replay.market_of(cfg)
                    bt = replay.make_bt(mk, replay.default_allow_short(cfg), cfg)
                    out.append(PL.measure(cfg, frame, rp.res,
                                          bt.multiplier, bt.margin_rate))
                except Exception as exc:           # noqa: BLE001
                    bad.append(f"{cfg.title()}（{type(exc).__name__}: {exc}）")
        except Exception as exc:                   # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit((out, bad))


class DataWorker(QObject):
    """在后台补一次行情：主力连续日线 + 展期 / 期权衍生数据。

    为什么不复用数据管理页那个 worker：那一页带一整套市场、周期、范围的下拉，
    这里要的只是「把这一页用到的数据补到最新」，参数写死近一年就够 —— 两个
    函数都只补本地缺的部分，多给日期不会重复请求。

    跑在独立线程里是因为一次可能要几分钟：Tushare 限速 + 期权那张表要逐日
    遍历。放在主线程里，整个窗口会「未响应」，用户以为死机。
    """

    log = Signal(str)
    done = Signal(bool, str)           # (成功?, 一句话)

    def __init__(self, range_key: str = "1y") -> None:
        super().__init__()
        self.range_key = range_key
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        from ttrade import updater

        cb = lambda d, t, m: self.log.emit(f"[{d}/{t}] {m}")   # noqa: E731
        try:
            res = updater.update_futures(
                None, self.range_key, progress=cb, cancel=lambda: self._cancel
            )
            self.log.emit(f"日线：{res.summary()}")
            aux = updater.update_aux(
                range_key=self.range_key, want_contracts=True, want_options=True,
                progress=cb, cancel=lambda: self._cancel,
            )
            n = int(getattr(aux, "days", 0) or 0)
            self.log.emit(f"衍生数据（展期 / 期权）：补了 {n} 个交易日" if n
                          else "衍生数据（展期 / 期权）：已是最新")
        except Exception as exc:       # noqa: BLE001
            self.done.emit(False, f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(True, "补数据完成")


class TrainWorker(QObject):
    """在后台训练机器学习模型，产出 .pkl 供实盘直接调用。

    为什么要单独一个 worker：一次训练是**每个品种 30~40 秒**（65 因子 ×
    26 万根 5m 的 48 折 walk-forward），19 个品种就是十几分钟。放主线程里
    整个窗口会「未响应」，用户以为死机。

    训练完的产物落在用户目录的 `ml_models/` 下（`ttrade/ml_artifact`）。
    实盘启动时会自动加载它 —— 走「只预测」快路径，每根 bar 几秒，
    不再重训。
    """

    log = Signal(str)
    done = Signal(bool, str)           # (成功?, 一句话)

    def __init__(self, cfg, symbols: list) -> None:
        super().__init__()
        self.cfg = cfg
        self.symbols = list(symbols)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        from ttrade import ml_artifact
        from ttrade import replay
        from ttrade import tqdata

        cfg = self.cfg
        fcfg = replay.factor_config(cfg.factor_cfg or {})
        keys = list(cfg.factor_keys or [])
        if len(keys) < 2:
            self.done.emit(False, "机器学习至少要 2 个因子")
            return

        # 合并训练：所有品种的样本一起训一个模型，产物按父 sid 存一份
        if getattr(fcfg, "pooled", False) and len(self.symbols) >= 2:
            self.log.emit(f"合并训练 {len(self.symbols)} 个品种…")
            try:
                frames = {}
                for sym, ex in self.symbols:
                    df = tqdata.load_main(sym, ex, cfg.period, adjust="qfq")
                    if df is not None and not df.empty:
                        df = replay.window(df, getattr(cfg, "range_years", 0) or 0)
                        frames[f"{sym}_{ex}"] = df
                if len(frames) < 2:
                    self.done.emit(False, "合并训练至少要 2 个品种有数据")
                    return
                art, fs = ml_artifact.train_fold_set_pooled(
                    frames, sid=cfg.sid, cfg=fcfg, keys=keys,
                    name=cfg.name, period=cfg.period, save=True,
                    progress=lambda d, t, m: self.log.emit(f"    [{d}/{t}] {m}"),
                    cancel=lambda: self._cancel,
                )
                self.done.emit(
                    True, f"合并训练完成：{len(fs.folds)} 折  "
                          f"{art.train_seconds:.1f}s  "
                          f"（{len(frames)} 品种一个模型）")
            except Exception as exc:  # noqa: BLE001
                self.done.emit(False, f"合并训练失败：{type(exc).__name__}: {exc}")
            return

        # 逐品种训练
        ok_n = 0
        todo = list(self.symbols)
        for i, (sym, ex) in enumerate(todo, 1):
            if self._cancel:
                self.done.emit(False, "已取消")
                return
            self.log.emit(f"[{i}/{len(todo)}] {sym}_{ex} 读历史…")
            try:
                df = tqdata.load_main(sym, ex, cfg.period, adjust="qfq")
                if df is None or df.empty:
                    self.log.emit(f"    {sym}_{ex}：没有数据，跳过")
                    continue
                df = replay.window(df, getattr(cfg, "range_years", 0) or 0)
                # 产物按「父 sid#品种_交易所」命名，和引擎里子槽位的 sid 一致
                sid = (f"{cfg.sid}#{sym}_{ex}"
                       if len(cfg.multi_symbols or []) >= 2 else cfg.sid)
                art, fs = ml_artifact.train_fold_set(
                    df, sid=sid, cfg=fcfg, keys=keys, name=cfg.name,
                    symbol=sym, exchange=ex, period=cfg.period, save=True,
                    progress=lambda d, t, m: self.log.emit(f"    {m}"),
                    cancel=lambda: self._cancel,
                )
                ok_n += 1
                self.log.emit(f"    ✓ {sym}_{ex}  {len(fs.folds)} 折  "
                              f"{art.train_seconds:.1f}s")
            except Exception as exc:       # noqa: BLE001
                self.log.emit(f"    ✗ {sym}_{ex}：{type(exc).__name__}: {exc}")
        if ok_n:
            self.done.emit(True, f"训练完成 {ok_n}/{len(todo)} 个品种，"
                                 f"实盘会直接用这些模型")
        else:
            self.done.emit(False, "一个都没训成")


class RunnerWorker(QObject):
    """跑在天勤线程里：一条连接 + 一组策略。"""

    snapshot = Signal(str, object)     # (账户名, 快照)
    log = Signal(str)
    failed = Signal(str)

    def __init__(self, key: str, spec, cfgs: list, chase) -> None:
        super().__init__()
        self.key = key
        self.spec = spec
        self.cfgs = cfgs
        self.chase = chase
        self._stop = False
        self._queue: queue.Queue = queue.Queue()

    def submit(self, fn) -> None:
        self._queue.put(fn)

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        runner = PortfolioRunner(self.spec, self.chase)
        for cfg in self.cfgs:
            runner.add(cfg)
        runner.on_log(self.log.emit)
        try:
            runner.start()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{self.key}：{type(exc).__name__}: {exc}")
            runner.close()
            return

        # 连接成功之后再打开之前存下来的「运行中」标记 —— 连不上时打开没有意义，
        # 而且 enable() 里会去撤单，那时候还没有 api 对象。
        for cfg in self.cfgs:
            if cfg.enabled:
                runner.enable(cfg.sid, True)

        last = 0.0
        try:
            while not self._stop:
                try:
                    runner.wait(0.8)
                except Exception as exc:  # noqa: BLE001
                    self.failed.emit(f"{self.key} 行情中断：{exc}")
                    break

                while True:
                    try:
                        fn = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        fn(runner)
                    except Exception as exc:  # noqa: BLE001
                        self.failed.emit(f"{self.key} 操作失败：{type(exc).__name__}: {exc}")

                runner.tick()

                now = time.monotonic()
                if now - last > 0.4:
                    last = now
                    self.snapshot.emit(self.key, runner.snapshot())
        finally:
            runner.close()


class PortfolioController(QObject):
    """主线程这一侧：按账户起停线程，转发快照，排队界面操作。"""

    snapshot = Signal(str, object)
    log = Signal(str)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._workers: dict[str, RunnerWorker] = {}
        self._threads: dict[str, QThread] = {}

    @property
    def running(self) -> bool:
        return bool(self._workers)

    def is_running(self, key: str) -> bool:
        """这个账户的连接在不在跑。单条启停要看账户粒度，不能只看整体。"""
        return key in self._workers

    def keys(self) -> list[str]:
        return list(self._workers)

    def start(self, spec, cfgs: list, chase) -> None:
        key = spec.name
        if key in self._workers:
            return
        worker = RunnerWorker(key, spec, cfgs, chase)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.snapshot.connect(self.snapshot)
        worker.log.connect(self.log)
        worker.failed.connect(self.failed)
        # 连接失败时 `run()` 直接 return，不会再进 while 循环。线程的事件循环还
        # 开着，不显式 quit 就永远不发 `finished` —— `_workers` 里会残留一条
        # 「正在跑」的记录，「启动全部」按钮的逻辑就跟着错。
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda k=key: self._on_done(k))
        self._workers[key] = worker
        self._threads[key] = thread
        thread.start()

    def stop(self, wait: bool = False) -> None:
        for key, worker in list(self._workers.items()):
            worker.stop()
            thread = self._threads.get(key)
            if thread is None:
                continue
            thread.quit()
            if wait:
                # worker 可能正卡在 wait_update(0.8)，最多等 5 秒
                thread.wait(5000)

    def _on_done(self, key: str) -> None:
        self._workers.pop(key, None)
        self._threads.pop(key, None)
        if not self._workers:
            self.stopped.emit()

    # ---- 操作：全部排队给对应账户的 worker ----

    def _submit(self, key: str, fn) -> bool:
        worker = self._workers.get(key)
        if worker is None:
            return False
        worker.submit(fn)
        return True

    def enable(self, key: str, sid: str, on: bool) -> bool:
        return self._submit(key, lambda r: r.enable(sid, on))

    def add(self, key: str, cfg) -> bool:
        return self._submit(key, lambda r: r.add(cfg))

    def add_or_update(self, key: str, cfg) -> bool:
        """让引擎把这条接上：已有就只更新配置，没有才建槽位。

        界面新加的策略只写了磁盘，`add()` 会重建槽位把正在跑的抹掉，
        `update()` 又对不存在的槽位直接返回 —— 两个都不能用来「补挂」。
        """
        return self._submit(key, lambda r: r.add_or_update(cfg))

    def remove(self, key: str, sid: str) -> bool:
        return self._submit(key, lambda r: r.remove(sid))

    def update_cfg(self, key: str, cfg) -> bool:
        return self._submit(key, lambda r: r.update(cfg))

    def align(self, key: str) -> bool:
        return self._submit(key, lambda r: r.align())

    def req_bars(self, key: str, sids: list) -> bool:
        """让引擎在下一份快照里带上这些策略的实时 K 线。"""
        return self._submit(key, lambda r: r.want_bars(sids))

    def set_chase(self, chase) -> None:
        for key in list(self._workers):
            self._submit(key, lambda r, c=chase: setattr(r, "chase", c))


# ------------------------------------------------------------------ 对话框


class LibraryDialog(QDialog):
    """从策略库挑一条。两处用：加进组合、在研究页加载参数。"""

    def __init__(self, items: list, parent=None, title: str = "从策略库添加",
                 hint: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)

        self.lst = QListWidget()
        for cfg in items:
            # 内置模板和「我的」混在一张表里，得看得出哪条是只读的模板
            tag = "内置" if SStore.is_template(cfg.sid) else "我的"
            it = QListWidgetItem(f"［{tag}］{cfg.title()}　{cfg.describe()}")
            it.setData(Qt.UserRole, cfg.sid)
            self.lst.addItem(it)
        if items:
            self.lst.setCurrentRow(0)
        self.lst.itemDoubleClicked.connect(lambda _i: self.accept())
        lay.addWidget(self.lst, 1)

        tip = QLabel(hint or
                     "加入组合会复制一份独立配置，之后在组合里改手数、改账户都不影响库里的模板。")
        tip.setObjectName("Hint")
        tip.setWordWrap(True)
        lay.addWidget(tip)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def sid(self) -> str:
        it = self.lst.currentItem()
        return "" if it is None else str(it.data(Qt.UserRole))


RANGE_CHOICES = [
    ("全部历史", 0.0), ("近 1 年", 1.0), ("近 2 年", 2.0),
    ("近 3 年", 3.0), ("近 5 年", 5.0), ("近 10 年", 10.0),
]


def _hint(text: str) -> QLabel:
    """表单里的一行小字说明。用 QLabel 而不是 groupbox：QSS 里没有它的样式。"""
    lbl = QLabel(text)
    lbl.setObjectName("Hint")
    lbl.setWordWrap(True)
    return lbl


def _split_key(key: str, market: str = "") -> tuple[str, str]:
    """把 `DataStore` 的键拆回配置里的 (代号, 交易所)。

    和研究页 `_current_config` 同一个口径，**不能直接用 `replay.split_key`**：
    那边为了 `key()` 能原样拼回来，A 股把代号拆成 `600000`；而配置里的 `symbol`
    一直是**带前缀的整键**（`SSE.600000`，`cn_name` / 天勤合约都要这个写法）。
    """
    k = str(key or "")
    if market == "cn_stock":
        return k, (k.split(".", 1)[0] if "." in k else "")
    if market == "cn":
        sym, _, ex = k.rpartition("_")
        return (sym or k), ex
    return k, ""


def _sym_text(cfg) -> str:
    """策略表「标的」列。

    多品种 / 截面只写「N 个品种」：`key()` 是清单里的第一个，写出来会让人以为
    这条策略只跑那一个（存盘里明明有 78 个）。
    """
    n = len(cfg.symbols or [])
    if n >= 2 or (cfg.family == "cross" and n):
        return f"{n} 个品种 {cfg.period}"
    return f"{cfg.key()} {cfg.period}"


def _sym_tip(cfg) -> str:
    """「标的」列的悬停说明：多品种把清单摊开前几个。"""
    syms = list(cfg.symbols or [])
    if len(syms) < 2 and cfg.family != "cross":
        return cfg.key()
    head = "、".join(syms[:8]) + ("…" if len(syms) > 8 else "")
    kind = ("截面品种池" if cfg.family == "cross"
            else "多品种等权：每个品种各跑一遍再平均")
    return f"{kind}，共 {len(syms)} 个\n{head}"


class EditDialog(QDialog):
    """一条组合策略的全部可调项。

    分两段，因为影响的范围不一样：

    - **执行**（手数 / 做空 / 账户）只决定实盘怎么下单；
    - **策略参数**（快线、慢线…）实盘和回测同时生效 —— 改了它，下一根 K 线的
      信号就按新参数算；
    - **回测区间**只决定「双击这条策略看哪一段历史」，实盘信号永远只用最近的
      K 线，和这个值无关。

    **交易品种**这一栏是后加的：一条策略可以同时跑一批品种（等权组合）。以前
    品种只能从「策略研究」页带过来，进了组合就再也改不了 —— 存盘里明明记着
    78 个品种，这里显示的还是第一个，想加一个只能删掉重加。
    """

    def __init__(self, cfg, accounts: list[str], parent=None,
                 universe: list[dict] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置参数")
        self.setMinimumWidth(430)
        form = QFormLayout(self)

        self.ed_name = QLineEdit(cfg.name)
        self.ed_name.setPlaceholderText(cfg.title())
        form.addRow("名称", self.ed_name)

        # ---- 交易品种 ----
        # `pick` 为 None 表示这一条不动品种：截面族（品种池在研究页管）、
        # 以及调用方没递品种清单（老调用、脚本）的时候
        self.pick: SymbolPicker | None = None
        self.lbl_pick: QLabel | None = None
        self._mk = cfg.market
        if cfg.family == "cross":
            form.addRow(_hint("截面策略的品种池在「策略研究」页勾选，这里不单独设"))
        elif universe:
            self.pick = SymbolPicker("交易品种")
            self.pick.set_items(list(universe), keep=cfg.symbols)
            # 存盘里那个品种可能已经不在本地库里（换了机器、只下了一部分数据）。
            # `set_selected` 只认清单里有的键，所以这种情况下会是「一个都没勾」，
            # 而 `apply_to` 见空就不写 —— 不会因为开一次设置框把品种抹掉
            self.pick.set_selected(cfg.symbols)
            form.addRow("交易品种", self.pick)
            self.lbl_pick = _hint("")
            form.addRow("", self.lbl_pick)
            self.pick.changed.connect(self._sync_pick)
            self._sync_pick()

        # ---- 本金与仓位分配 ----
        # 默认不再是「1 手」：1 手螺纹和 1 手铜差 40 倍名义价值，同一条策略在两个
        # 品种上跑出来的风险完全不是一回事。默认按「本金 × 比例 ÷ 每手保证金」铺满。
        # 本金单独一行、不跟着口径隐藏：四种口径下它都决定净值曲线的起点。
        self.cb_size = QComboBox()
        for text, val in (("按资金分配", "capital"),
                          ("按止损金额", "risk"),
                          ("固定手数", "fixed")):
            self.cb_size.addItem(text, val)
        i = self.cb_size.findData(cfg.sizing or "capital")
        self.cb_size.setCurrentIndex(max(i, 0))
        self.cb_size.setToolTip(
            "按资金：手数 = 本金 × 资金比例 ÷ 每手保证金。\n"
            "杠杆由品种自己的保证金比例决定（商品、股指约 10 倍，国债 30~50 倍）。\n"
            "按止损金额：手数 = 本金 × 止损比例 ÷（止损倍数 × ATR × 合约乘数）。\n"
            "固定手数：老的「几手」口径。"
        )
        form.addRow("仓位分配", self.cb_size)

        self.sp_principal = QDoubleSpinBox()
        self.sp_principal.setRange(10_000, 1_000_000_000)
        self.sp_principal.setDecimals(0)
        self.sp_principal.setSingleStep(500_000)
        self.sp_principal.setSuffix(" 元")
        self.sp_principal.setValue(float(cfg.principal or 1_000_000))
        self.sp_principal.setToolTip(
            "这条策略的账户规模，默认 100 万。\n"
            "资金曲线从它起步，下面两个比例都按它折算。"
        )
        form.addRow("本金", self.sp_principal)

        self.sp_capital = QDoubleSpinBox()
        self.sp_capital.setRange(0.1, 500.0)
        self.sp_capital.setDecimals(1)
        self.sp_capital.setSingleStep(5.0)
        self.sp_capital.setSuffix(" %")
        self.sp_capital.setValue(float(cfg.capital_pct or 30.0))
        self.sp_capital.setToolTip(
            "拿本金的百分之多少开仓，默认 30%。手数按每手保证金折算，不按「几手」拍。"
        )
        form.addRow("资金比例", self.sp_capital)

        self.sp_vol = QSpinBox()
        self.sp_vol.setRange(1, 9999)
        self.sp_vol.setValue(max(1, _int(cfg.volume) or 1))
        form.addRow("固定手数", self.sp_vol)

        self.sp_risk = QDoubleSpinBox()
        self.sp_risk.setRange(0.01, 100.0)
        self.sp_risk.setDecimals(2)
        self.sp_risk.setSingleStep(0.5)
        self.sp_risk.setSuffix(" %")
        self.sp_risk.setValue(float(cfg.risk_pct or 1.0))
        self.sp_risk.setToolTip(
            "打止损时最多亏本金的百分之多少，默认 1%，手数由止损距离反推。"
        )
        form.addRow("止损比例", self.sp_risk)

        self.sp_atr = QSpinBox()
        self.sp_atr.setRange(2, 200)
        self.sp_atr.setValue(max(2, _int(cfg.stop_atr_n) or 14))
        form.addRow("止损 ATR 周期", self.sp_atr)

        self.sp_stop = QDoubleSpinBox()
        self.sp_stop.setRange(0.1, 50.0)
        self.sp_stop.setSingleStep(0.5)
        self.sp_stop.setDecimals(1)
        self.sp_stop.setValue(float(cfg.stop_mult or 5.0))
        form.addRow("止损 ATR 倍数", self.sp_stop)

        self.cb_size.currentIndexChanged.connect(self._sync_size)
        self._sync_size()

        self.chk_short = QCheckBox("允许做空")
        self.chk_short.setChecked(cfg.note == "short")
        form.addRow("", self.chk_short)

        # 和研究页那对开关同一语义：只做空要先允许做空，两个都勾等于不做，
        # 所以关掉「允许做空」时「禁止做多」直接灰掉并清空
        self.chk_no_long = QCheckBox("禁止做多（只做空）")
        self.chk_no_long.setChecked(bool(getattr(cfg, "short_only", False)))
        self.chk_no_long.setEnabled(self.chk_short.isChecked())
        self.chk_short.toggled.connect(self._sync_dir)
        self.chk_no_long.toggled.connect(self._sync_dir)
        form.addRow("", self.chk_no_long)

        # 反向交易只在规则族上有意义：因子策略的方向由因子本身的正负决定，
        # 要翻方向是在研究页把因子反过来，不是在这里
        self.chk_reverse = None
        if cfg.family == "rule":
            self.chk_reverse = QCheckBox("反向交易（多 ↔ 空）")
            self.chk_reverse.setChecked(bool(getattr(cfg, "rule_reverse", False)))
            self.chk_reverse.setToolTip(
                "信号整体取反：原先做多的地方做空、原先做空的地方做多，空仓不变。\n"
                "和「策略研究」页那个开关是同一个字段，回测和实盘同时生效。"
            )
            form.addRow("", self.chk_reverse)

        # 多账户：勾几个就在几个账户上各跑一份。**一个都不勾 = 跟随默认账户**，
        # 和老配置（只有 `account` 一个字段）的语义完全一致
        self.lst_acct = QListWidget()
        self.lst_acct.setMaximumHeight(104)
        self.lst_acct.setToolTip(
            "勾几个就在几个账户上同时跑这一条策略。\n"
            "一个都不勾 = 跟随顶部的默认账户。\n\n"
            "每个账户各算各的手数（各自的本金、各自的保证金），策略参数只有一份 ——\n"
            "所以不用为每个账户复制一遍策略。")
        chosen = list(cfg.accounts or ([cfg.account] if cfg.account else []))
        for a in accounts:
            it = QListWidgetItem(a)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if a in chosen else Qt.Unchecked)
            self.lst_acct.addItem(it)
        form.addRow("交易账户", self.lst_acct)
        if not accounts:
            form.addRow("", _hint("还没有交易账户，先到「账户透视和交易」页添加"))

        # ---- 策略参数 ----
        # 小数参数（vol_ratio 0.4、stop_mult 2.5）必须用 QDoubleSpinBox：
        # QSpinBox 的 setRange(int(lo), int(hi)) 会把 (0.1, 1.5) 夹成 (0, 1)，
        # 0.4 读出来是 0 —— 用户什么都没改，策略的波幅门槛就变成 0 了
        self.sp_params: dict[str, QWidget] = {}
        spec = STRATEGIES.get(cfg.rule_key) if cfg.family == "rule" else None
        if spec is not None and spec.params:
            form.addRow(_hint(f"策略参数 · {spec.label}（实盘与回测同时生效）"))
            for name, default in spec.params.items():
                lo, hi, step = spec.param_meta.get(name, (1, 1000, 1))
                cur = cfg.rule_params.get(name, default)
                if isinstance(default, float):
                    sp = QDoubleSpinBox()
                    sp.setDecimals(2)
                    sp.setRange(float(lo), float(hi))
                    sp.setSingleStep(float(step))
                else:
                    sp = QSpinBox()
                    sp.setRange(int(lo), int(hi))
                    sp.setSingleStep(max(1, int(step)))
                try:
                    sp.setValue(float(cur) if isinstance(default, float) else int(cur))
                except (TypeError, ValueError):
                    sp.setValue(default)
                self.sp_params[name] = sp
                form.addRow(param_label(spec, name), sp)
        elif cfg.family == "factor":
            form.addRow(_hint(
                "因子策略的参数（窗口 / 阈值 / 因子）请在「策略研究」页调好，"
                "再存入策略库、重新加入组合"
            ))

        # ---- 回测区间 ----
        self.cb_range = QComboBox()
        for text, years in RANGE_CHOICES:
            self.cb_range.addItem(text, years)
        i = self.cb_range.findData(float(cfg.range_years or 0))
        self.cb_range.setCurrentIndex(max(i, 0))
        self.cb_range.setToolTip(
            "双击这条策略时回测和看图的范围。\n"
            "实盘信号只用最近的 K 线，不受这里影响。"
        )
        form.addRow("回测区间", self.cb_range)

        # 多品种写「N 个品种」：`key()` 只是清单里的第一个，写出来会让人以为
        # 这条策略只跑那一个
        n_sym = len(cfg.symbols or [])
        lab = QLabel((f"{n_sym} 个品种" if n_sym >= 2 and cfg.family != "cross"
                      else cfg.key()) + f" {cfg.period}｜{cfg.strategy_label()}")
        lab.setObjectName("Hint")
        form.addRow("策略", lab)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def _sync_dir(self, _checked: bool = False) -> None:
        """方向两个开关联动，和研究页同一套规则（见 `MainWindow._sync_dir_flags`）。"""
        allow = self.chk_short.isChecked()
        self.chk_no_long.setEnabled(allow)
        if not allow and self.chk_no_long.isChecked():
            self.chk_no_long.blockSignals(True)
            self.chk_no_long.setChecked(False)
            self.chk_no_long.blockSignals(False)

    def _sync_size(self) -> None:
        """只显示当前口径用得上的字段，免得一堆数字看不出哪个生效。

        本金不参与这个开关：四种口径下它都决定净值曲线的起点。
        """
        mode = self.cb_size.currentData()
        form = self.layout()
        if form is None:
            return
        for w, on in ((self.sp_capital, mode == "capital"),
                      (self.sp_vol, mode == "fixed"),
                      (self.sp_risk, mode == "risk"),
                      (self.sp_atr, mode == "risk"),
                      (self.sp_stop, mode == "risk")):
            try:
                form.setRowVisible(w, on)
            except (TypeError, AttributeError):   # 老 Qt 没有这个重载
                pass

    def _sync_pick(self) -> None:
        """勾选变了就改下面那行小字：一眼看出现在是单品种还是多品种。"""
        if self.pick is None or self.lbl_pick is None:
            return
        picked = [k for k in self.pick.selected() if k]
        n = len(picked)
        if n >= 2:
            self.lbl_pick.setText(
                f"多品种：{n} 个，每个品种各跑一遍再等权平均。"
                "实盘会拆成 n 个槽位，每个品种分到 1/n 的本金"
            )
        elif n == 1:
            self.lbl_pick.setText(f"单品种：{picked[0]}")
        else:
            self.lbl_pick.setText("一个都没勾 —— 保存时品种保持不变")

    def apply_to(self, cfg) -> None:
        cfg.name = self.ed_name.text().strip()
        cfg.sizing = self.cb_size.currentData() or "capital"
        # 界面收百分比，配置里存本金 + 百分比（`capital` / `risk_amount` 是
        # property，按本金折算出来，不用也不能直接赋）
        cfg.principal = float(self.sp_principal.value())
        cfg.capital_pct = float(self.sp_capital.value())
        cfg.risk_pct = float(self.sp_risk.value())
        cfg.stop_atr_n = int(self.sp_atr.value())
        cfg.stop_mult = float(self.sp_stop.value())
        cfg.volume = self.sp_vol.value()
        cfg.note = "short" if self.chk_short.isChecked() else ""
        cfg.short_only = bool(self.chk_no_long.isChecked() and self.chk_short.isChecked())
        picked = [self.lst_acct.item(i).text()
                  for i in range(self.lst_acct.count())
                  if self.lst_acct.item(i).checkState() == Qt.Checked]
        cfg.accounts = picked
        cfg.account = picked[0] if picked else ""
        if self.sp_params:
            cfg.rule_params = {k: sp.value() for k, sp in self.sp_params.items()}
        if self.chk_reverse is not None:
            cfg.rule_reverse = self.chk_reverse.isChecked()
        cfg.range_years = float(self.cb_range.currentData() or 0.0)

        if self.pick is not None:
            picked = [k for k in self.pick.selected() if k]
            if picked:
                # `key()` 是拼出来的，反向拆要按市场走：A 股的键自带交易所前缀
                # （`SSE.600000`），一律按 `_` 拆会拆出一个不存在的交易所
                sym, ex = _split_key(picked[0], self._mk)
                cfg.symbol, cfg.exchange = sym, ex
                # 只勾了一个就退回单品种：留着一份单元素清单的话，`symbols`
                # 会一直返回它，之后改了 `symbol` 也不生效
                cfg.multi_symbols = list(picked) if len(picked) >= 2 else []


# ------------------------------------------------------------------ 页面


class PortfolioPage(QWidget):
    """策略交易和组合。"""

    # 补完数据要让主窗口把 DataStore 的缓存丢掉，否则这一页还在用补之前那份
    data_updated = Signal()

    def __init__(self, loader=None, parent=None, universe=None) -> None:
        super().__init__(parent)
        # loader(market, key, period) -> DataFrame | None。主窗口把它的 DataStore
        # 递进来 —— 这一页自己不该再扫一遍数据目录（那是几秒钟的事）。
        self._loader = loader
        # universe(market) -> list[dict]。改品种要用：设置参数里那个勾选面板得知道
        # 这个市场有哪些品种。和 loader 一样由主窗口递进来，这一页不自己扫盘
        self._universe = universe
        # 策略池：`portfolio.json` 里那一份，参数只有这一处。工作区只记
        # 「谁在里面、分到多少本金」，见 `ttrade/workspace_store.py`
        self._pool = SStore.load_portfolio()
        self._wss = WS.load_workspaces(self._pool)
        self._ws = self._active_ws()
        self._items = []
        # 实测占用：sid -> PL.Usage。「量一下占用」之后才有，切工作区不清空 ——
        # 同一条策略换个工作区，它**历史回测里**平均压多少钱是不变的
        self._usages: dict = {}
        self._measure_done = 0
        self._measure_bad: list[str] = []
        # 量占用的后台线程（它和「查看回测」是两条独立的路，不共线程）
        self._ms_thread: QThread | None = None
        self._ms_worker = None
        self._accounts = tqlive.load_accounts()
        self._chase = SStore.load_chase()
        self._snaps: dict[str, dict] = {}
        # sid -> 快照里的 slot（单账户时的主来源）
        self._slot_view: dict[str, dict] = {}
        # sid -> {账户: slot}。多账户策略在每个账户上各有一个槽位，`_slot_view`
        # 只会留下最后写进来的那个 —— 想显示「这条策略在每个账户上分别多少手」
        # 就得留着这份。单账户时两个结构内容一致
        self._slot_multi: dict[str, dict] = {}
        self._render_ts = 0.0                     # 上次重画两张表的时刻
        self._mon_sig = None                      # 监控表上次画的内容
        # 单策略视图的实时 K 线。`_live_base` 是本地库里的历史，实时段每 5 秒
        # 拼一次 —— 引擎订的是实时行情，界面却只画本地库，打开一看停在几天前，
        # 用户会以为策略没在跟数据。`_live_busy` 防重入：请求发出去了、快照
        # 还没回来时再发一次，只会排一串没用的活。
        self._live_cfg = None
        self._live_base = None
        self._live_sid = ""
        self._live_busy = False
        self._live_req = 0.0
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(5000)
        self._live_timer.timeout.connect(self._poll_live)
        # 补数据的后台线程。同时只允许一个 —— 两个 update_aux 并发写同一批
        # CSV 会互相覆盖
        self._data_thread: QThread | None = None
        self._data_worker = None
        # 训练模型的后台线程。同样只允许一个 —— 两个训练会往同一批产物
        # 文件里写，写一半被另一个覆盖就坏掉了
        self._train_thread: QThread | None = None
        self._train_worker = None
        # 「查看回测 / 组合报告」的后台线程。多品种一次是 N 遍策略计算，
        # 机器学习还要各训几十折，同步跑窗口会「未响应」
        self._view_thread: QThread | None = None
        self._view_worker = None
        self._view_mode = ""          # single / combo，回来时决定怎么出图
        self._view_missing: list[str] = []
        self._view_payload = None     # 单品种：原始行情，出图时当实时段的底
        self._views: dict[str, tuple[QTabWidget, ResultView]] = {}
        self.ctl = PortfolioController(self)
        self._build()
        self._load_chase_widgets()
        self._chase_hint(self._chase)
        self.reload()

        self.ctl.snapshot.connect(self._on_snapshot)
        self.ctl.log.connect(self._on_log)
        self.ctl.failed.connect(self._on_failed)
        self.ctl.stopped.connect(self._on_stopped)

    # ---------------------------------------------------------- 构建

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 两页：管理界面 / 结果视图。用 QStackedWidget 而不是弹窗 —— 弹窗里塞
        # 五个带 matplotlib 的页签，窗口一小图表就糊成一片，还得再调一次尺寸。
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_manage())
        self.stack.addWidget(self._build_view_page())
        root.addWidget(self.stack, 1)

    def _build_manage(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        root.addWidget(self._build_top())

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_strategy_card())
        split.addWidget(self._build_monitor_card())
        split.setSizes([760, 460])
        root.addWidget(split, 1)

        root.addWidget(self._build_chase_card())
        return page

    def _build_view_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        bar = QWidget()
        bar.setObjectName("Card")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(12, 6, 12, 6)
        bl.setSpacing(10)
        self.btn_back = QPushButton("← 返回策略列表")
        self.btn_back.setObjectName("Ghost")
        self.btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        bl.addWidget(self.btn_back)

        self.lbl_view = QLabel("")
        self.lbl_view.setStyleSheet("font-size: 13px; font-weight: 600; color: #0f172a;")
        bl.addWidget(self.lbl_view, 1)
        lay.addWidget(bar)

        # 单策略视图和组合视图各建一次、常驻不销毁：销毁要停 QTimer、要防
        # 已排队的重绘打到已删的画布上，切回来还得重建五个页签。留着更省事，
        # 代价只是多几 MB 的 Figure。
        self._view_stack = QStackedWidget()
        lay.addWidget(self._view_stack, 1)
        return page

    def _ensure_view(self, kind: str) -> tuple[QTabWidget, ResultView]:
        hit = self._views.get(kind)
        if hit is not None:
            return hit
        tabs = QTabWidget()
        view = ResultView(kind, self)
        for title, w in view.pages():
            tabs.addTab(w, title)
        view.status.connect(self.lbl_view.setText)
        self._view_stack.addWidget(tabs)
        self._views[kind] = (tabs, view)
        return tabs, view

    def _show_view(self, kind: str, title: str) -> None:
        tabs, _view = self._ensure_view(kind)
        self._view_stack.setCurrentWidget(tabs)
        self.lbl_view.setText(title)
        self.stack.setCurrentIndex(1)

    def _build_top(self) -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(8)

        lay.addWidget(QLabel("默认账户"))
        self.cb_acct = QComboBox()
        self.cb_acct.setMinimumWidth(200)
        self.cb_acct.setToolTip("组合里没单独指定账户的策略，都用这个账户")
        lay.addWidget(self.cb_acct)

        self.btn_start = QPushButton("启动全部")
        self.btn_start.clicked.connect(self._start_all)
        lay.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止全部")
        self.btn_stop.setObjectName("Ghost")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_all)
        lay.addWidget(self.btn_stop)

        self.lbl_dot = QLabel("●")
        self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
        lay.addWidget(self.lbl_dot)
        self.lbl_conn = QLabel("未启动")
        self.lbl_conn.setObjectName("Hint")
        lay.addWidget(self.lbl_conn)
        # 交易所对日内报单 / 撤单次数有硬上限，超了会被限制交易。这一格让人
        # 随时看得到还剩多少，而不是等停手了才去翻日志
        self.lbl_limit = QLabel("")
        self.lbl_limit.setObjectName("Hint")
        self.lbl_limit.setToolTip(
            "交易所对每个账户的日内报单 / 撤单笔数有硬上限，"
            "本程序达到上限时自动停手"
        )
        lay.addWidget(self.lbl_limit)

        lay.addStretch(1)
        self.lbl_msg = QLabel("")
        self.lbl_msg.setObjectName("Hint")
        lay.addWidget(self.lbl_msg)
        return card

    def _card(self, title: str) -> tuple[QWidget, QVBoxLayout, QHBoxLayout]:
        card = QWidget()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        head = QHBoxLayout()
        t = QLabel(title)
        t.setObjectName("SectionTitle")
        head.addWidget(t)
        head.addStretch(1)
        lay.addLayout(head)
        return card, lay, head

    def _table(self, headers: list[str], stretch: int = 0) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setShowGrid(False)
        h = t.horizontalHeader()
        for i in range(len(headers)):
            h.setSectionResizeMode(
                i, QHeaderView.Stretch if i == stretch else QHeaderView.ResizeToContents
            )
        return t

    def _build_strategy_card(self) -> QWidget:
        card, lay, head = self._card("策略列表")

        self.btn_view = QPushButton("查看回测")
        self.btn_view.setToolTip("打开这一条策略的 K 线与信号 / 绩效 / 因子分析 / 交易记录 / 阶段分析")
        self.btn_view.clicked.connect(self._open_single)
        head.addWidget(self.btn_view)

        self.btn_combo = QPushButton("组合报告")
        self.btn_combo.setObjectName("Ghost")
        self.btn_combo.setToolTip("按住 Ctrl 选中多条策略，看它们等权组合起来的资金曲线")
        self.btn_combo.setEnabled(False)
        self.btn_combo.clicked.connect(self._open_combo)
        head.addWidget(self.btn_combo)

        btn_lib = QPushButton("从策略库添加")
        btn_lib.setObjectName("Ghost")
        btn_lib.clicked.connect(self._add_from_library)
        head.addWidget(btn_lib)

        # 这一页的策略要用到展期 / 期权因子时，本地快照过期了信号就是死的。
        # 数据管理页也能补，但那要切页、还要把市场周期对一遍 —— 就地补一次更顺
        self.btn_data = QPushButton("补数据")
        self.btn_data.setObjectName("Ghost")
        self.btn_data.setToolTip(
            "补期货主力日线 + 展期 / 期权衍生数据（近一年，只补本地缺的部分）。\n"
            "衍生数据是期权 IV、期限结构那些因子的来源；几天没开程序，它们就停在几天前。"
        )
        self.btn_data.clicked.connect(self._supplement_data)
        head.addWidget(self.btn_data)

        btn_edit = QPushButton("设置参数")
        btn_edit.setObjectName("Ghost")
        btn_edit.setToolTip("改本金与仓位分配（资金比例 / 止损比例 / 固定手数）、账户、"
                            "策略参数（快线慢线…）和回测区间")
        btn_edit.clicked.connect(self._edit_selected)
        head.addWidget(btn_edit)

        # 机器学习策略：训练一次、存成 .pkl，实盘就直接用它算信号（不重训）。
        # 没有产物也能跑，只是每根 bar 要把整段历史重走一遍（几十秒），
        # 5 分钟周期追不上 —— 所以这一条对机器学习策略才有用
        self.btn_train = QPushButton("训练模型")
        self.btn_train.setObjectName("Ghost")
        self.btn_train.setToolTip(
            "离线训练这条机器学习策略，把模型存成 .pkl。\n"
            "训完实盘就直接用这些模型算信号（只预测，不重训）—— "
            "实测 65 因子 × 26 万根 5m 的策略，重训要 34 秒/品种，"
            "用产物只要 3 秒。\n"
            "多品种策略每个品种训一份。")
        self.btn_train.clicked.connect(self._train_selected)
        head.addWidget(self.btn_train)

        btn_toggle = QPushButton("启动 / 停止")
        btn_toggle.setObjectName("Ghost")
        btn_toggle.setToolTip(
            "只启停选中的这一条。\n"
            "账户的连接还没起来时，会**只**把它拉起来（不等「启动全部」）；\n"
            "判断依据是它此刻在不在跑，不是存盘的开关键。")
        btn_toggle.clicked.connect(self._toggle_selected)
        head.addWidget(btn_toggle)

        btn_del = QPushButton("删除")
        btn_del.setObjectName("Ghost")
        btn_del.setToolTip("只把它从**当前工作区**拿掉。别的分页还在用它时，策略池里留着")
        btn_del.clicked.connect(self._del_selected)
        head.addWidget(btn_del)

        # 工作区分页 + 资金分配。放在策略表**上面**：先选是哪个组合，再看组合里
        # 有哪几条 —— 顺序反过来的话，切一次分页就要在上面和下面之间来回找
        self.ws_bar = WorkspaceBar()
        self.ws_bar.switched.connect(self._ws_switch)
        self.ws_bar.create.connect(self._ws_create)
        self.ws_bar.rename.connect(self._ws_rename)
        self.ws_bar.remove.connect(self._ws_remove)
        self.ws_bar.measure.connect(self._measure_usages)
        self.ws_bar.preview.connect(self._alloc_preview)
        self.ws_bar.total_changed.connect(self._ws_total)
        self.ws_bar.mode_changed.connect(self._ws_mode)
        self.ws_bar.basis_changed.connect(self._ws_basis)
        lay.addWidget(self.ws_bar)

        self.tbl_strat = self._table(
            ["名称", "标的", "策略", "仓位", "账户", "理论", "状态",
             "平均开仓占用", "峰值占用", "利用率"]
        )
        # 多选是为了组合报告。双击原来是「启动 / 停止」，但那个动作有同名按钮，
        # 而「看一眼这条策略的历史表现」没有别的入口 —— 双击给后者更值。
        self.tbl_strat.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl_strat.doubleClicked.connect(lambda _i: self._open_single())
        self.tbl_strat.itemSelectionChanged.connect(self._sync_buttons)
        lay.addWidget(self.tbl_strat, 1)

        hint = QLabel(
            "双击一行 = 打开它的 K 线与绩效。按住 Ctrl 多选后点「组合报告」，"
            "看这几条策略等权组合的资金曲线。\n"
            "「启动 / 停止」只作用于选中的那一条：账户还没连上时会把它的账户单独连上。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        return card

    def _sync_buttons(self) -> None:
        """选中几条就改一次按钮：只有一条时「组合报告」没有意义。"""
        cfgs = self._selected_cfgs()
        n = len(cfgs)
        self.btn_combo.setEnabled(n >= 2)
        self.btn_combo.setText(f"组合报告（{n}）" if n >= 2 else "组合报告")
        self.btn_view.setEnabled(n >= 1)
        # 「训练模型」只对机器学习策略有意义。规则策略没有模型可训，
        # 按钮亮着只会让人以为点一下有用
        is_ml = bool(cfgs) and (cfgs[0].family or "") == "factor" \
            and (cfgs[0].factor_mode or "") == "ml"
        self.btn_train.setEnabled(is_ml and n >= 1)
        self.btn_train.setVisible(is_ml or n == 0)
        if is_ml:
            n_sym = len(cfgs[0].multi_symbols or [])
            # 已经训过的品种数标在按钮上 —— 用户一眼能看出「还差几个没训」，
            # 不用点进去才发现
            done = self._trained_count(cfgs[0])
            if done and n_sym >= 2:
                self.btn_train.setText(f"训练模型（{done}/{n_sym} 已训）")
            elif n_sym >= 2:
                self.btn_train.setText(f"训练模型（{n_sym} 个品种）")
            else:
                self.btn_train.setText("训练模型")

    @staticmethod
    def _trained_count(cfg) -> int:
        """这条策略里已经有训练产物的品种数。"""
        try:
            from ttrade import ml_artifact as MLA

            syms = list(cfg.multi_symbols or [])
            if len(syms) < 2:
                syms = [cfg.key()]
            n = 0
            for k in syms:
                sym, ex = _split_key(k, cfg.market)
                sid = (f"{cfg.sid}#{sym}_{ex}" if len(cfg.multi_symbols or []) >= 2
                       else cfg.sid)
                if os.path.exists(MLA.artifact_path(sid)):
                    n += 1
            return n
        except Exception:  # noqa: BLE001
            return 0

    def _build_monitor_card(self) -> QWidget:
        card, lay, head = self._card("监控器")

        btn_align = QPushButton("一键对齐")
        btn_align.setToolTip("让账户持仓与策略列表完全一致：托管合约补到理论持仓，\n"
                             "没有策略在管的仓位平掉（会先弹确认框）")
        btn_align.clicked.connect(self._align)
        head.addWidget(btn_align)

        # 多账户时同一个合约会在两张连接里各出现一次，必须带上账户才分得清
        self.tbl_mon = self._table(
            ["账户", "合约", "策略理论", "账户实际", "差额", "说明"], stretch=5
        )
        lay.addWidget(self.tbl_mon, 1)

        self.lbl_mon = QLabel("一致")
        self.lbl_mon.setObjectName("Hint")
        self.lbl_mon.setWordWrap(True)
        lay.addWidget(self.lbl_mon)
        return card

    def _build_chase_card(self) -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        t = QLabel("追单设置")
        t.setObjectName("SectionTitle")
        lay.addWidget(t)

        self.chk_chase = QCheckBox("启用")
        self.chk_chase.setToolTip("关掉之后策略发出的委托就一直挂着，不撤不追")
        lay.addWidget(self.chk_chase)

        lay.addWidget(QLabel("等"))
        self.sp_wait = QSpinBox()
        self.sp_wait.setRange(1, 600)
        self.sp_wait.setSuffix(" 秒")
        self.sp_wait.setMinimumWidth(78)
        self.sp_wait.setToolTip("委托挂多久没成交就撤掉重报")
        lay.addWidget(self.sp_wait)

        lay.addWidget(QLabel("未成交则撤单重报，最多追"))
        self.sp_retry = QSpinBox()
        self.sp_retry.setRange(0, 50)
        self.sp_retry.setSuffix(" 次")
        self.sp_retry.setMinimumWidth(78)
        lay.addWidget(self.sp_retry)

        lay.addWidget(QLabel("每次追"))
        self.sp_ticks = QSpinBox()
        self.sp_ticks.setRange(0, 200)
        self.sp_ticks.setSuffix(" 跳")
        self.sp_ticks.setMinimumWidth(78)
        self.sp_ticks.setToolTip("按盘口对价，每次往成交方向多让几跳")
        lay.addWidget(self.sp_ticks)

        self.chk_market = QCheckBox("最后一次用市价")
        lay.addWidget(self.chk_market)

        lay.addStretch(1)
        self.lbl_chase = QLabel("")
        self.lbl_chase.setObjectName("Hint")
        lay.addWidget(self.lbl_chase)

        for w in (self.chk_chase, self.chk_market):
            w.toggled.connect(self._save_chase)
        for w in (self.sp_wait, self.sp_retry, self.sp_ticks):
            w.valueChanged.connect(self._save_chase)
        return card

    # ---------------------------------------------------------- 数据

    def reload(self) -> None:
        """从磁盘重读组合与账户。切回本页时调，能看到「策略研究」里刚加的那条。"""
        self._pool = SStore.load_portfolio()
        self._wss = WS.load_workspaces(self._pool)
        self._ws = self._active_ws()
        self._sync_ws()
        self._accounts = tqlive.load_accounts()
        self._reload_accounts()
        self._fill_strategies()
        self._render()

    # ---------------------------------------------------------- 工作区

    def _active_ws(self):
        """当前分页对应的那个工作区。文件里记的那个 id 找不到就退回第一个。"""
        if not self._wss:
            self._wss = [WS.new_workspace(WS.DEFAULT_NAME)]
        cur = WS.active_id()
        for w in self._wss:
            if w.id and w.id == cur:
                return w
        return self._wss[0]

    def _ws_index(self) -> int:
        for i, w in enumerate(self._wss):
            if w is self._ws:
                return i
        return 0

    def _sync_ws(self) -> None:
        """把「当前工作区」摊成 `_items`，并刷新工具条。

        `_items` 是**解析出来的副本**：`principal` 用工作区给的那份，不是策略池
        里那份。后面所有渲染 / 回放 / 启动读的都是它，所以「这个工作区分了多少钱」
        全页只有一个来源。
        """
        if self._ws is None:
            self._ws = self._active_ws()
        cfgs, missing = WS.resolve(self._ws, self._pool)
        # 本工作区是否参与运行 = 策略自己的开关 ∧ 成员的开关
        for c in cfgs:
            m = self._ws.find(str(getattr(c, "sid", "") or ""))
            if m is not None:
                c.enabled = bool(c.enabled and m.enabled)
        self._items = cfgs
        if missing:
            self._on_log(f"有 {len(missing)} 条策略已不在策略池里，本工作区跳过")
        bar = getattr(self, "ws_bar", None)
        if bar is not None:
            bar.set_workspaces([w.name or "未命名" for w in self._wss], self._ws_index())
            bar.set_state(self._ws.live_total(self._pool), self._ws.mode, self._ws.basis)

    def _save_ws(self) -> None:
        WS.save_workspaces(self._wss, self._ws.id if self._ws else "")

    def _ws_switch(self, i: int) -> None:
        if not (0 <= i < len(self._wss)):
            return
        if self._wss[i] is self._ws:
            return
        self._ws = self._wss[i]
        # 只记「停在哪个分页」，不重写整个文件 —— 二十几条本金为了一个 id
        # 整份重写一遍，写一半崩了就是配置全丢
        WS.set_active(self._ws.id)
        self._sync_ws()
        self._fill_strategies()
        self._render()
        # 引擎不跟着切：运行是**按账户**合并所有工作区的，切页签只是换个视角。
        # 这里提醒一句，免得用户以为「切走了就等于停了」
        if self.ctl.running:
            self._on_log(f"已切到「{self._ws.name}」。运行中的策略不受影响 —— "
                         "启动是跨工作区按账户合并的")

    def _ws_create(self) -> None:
        name, ok = QInputDialog.getText(self, "新建工作区", "名字", text="新组合")
        if not ok:
            return
        w = WS.new_workspace((name or "").strip() or "新组合",
                             total=float(self._ws.live_total(self._pool)) if self._ws else 0.0)
        self._wss.append(w)
        self._ws = w
        self._save_ws()
        self._sync_ws()
        self._fill_strategies()
        self._render()
        self._on_log(f"已新建工作区「{w.name}」。切换分页后用「从策略库添加」往里加策略")

    def _ws_rename(self) -> None:
        if self._ws is None:
            return
        name, ok = QInputDialog.getText(self, "工作区改名", "名字",
                                        text=self._ws.name or "")
        if not ok or not (name or "").strip():
            return
        self._ws.name = name.strip()
        self._save_ws()
        self._sync_ws()

    def _ws_remove(self) -> None:
        if self._ws is None:
            return
        if len(self._wss) <= 1:
            self._on_failed("至少留一个工作区")
            return
        n = len(self._ws.members)
        msg = (f"删除工作区「{self._ws.name}」？\n"
               f"里面有 {n} 条策略的本金分配。\n"
               "策略本身还在策略池里：如果别的分页也在用它，那边不受影响；"
               "没有任何分页再用它时才会从池子里清掉。")
        if QMessageBox.question(self, "删除工作区", msg) != QMessageBox.Yes:
            return
        dead = self._ws
        self._wss = [w for w in self._wss if w is not dead]
        self._ws = self._wss[0]
        # 被删工作区独占的策略，从池子里一并清掉（没人用了）
        used = set()
        for w in self._wss:
            used.update(w.sids())
        for sid in dead.sids():
            if sid and sid not in used:
                SStore.remove_from_portfolio(sid)
        self._save_ws()
        self.reload()
        self._on_log(f"已删除工作区「{dead.name}」")

    def _ws_total(self, v: float) -> None:
        if self._ws is None:
            return
        self._ws.total = float(v or 0)
        self._save_ws()
        self._ws_hint()

    def _ws_mode(self, mode: str) -> None:
        if self._ws is None:
            return
        self._ws.mode = str(mode or "equal")
        self._save_ws()
        self._ws_hint()

    def _ws_basis(self, basis: str) -> None:
        if self._ws is None:
            return
        self._ws.basis = str(basis or "principal")
        self._save_ws()
        self._ws_hint()

    def _ws_hint(self) -> None:
        """工具条右边那句提示：本金合计 / 已量几条 / 校准口径是否可用。"""
        bar = getattr(self, "ws_bar", None)
        if bar is None or self._ws is None:
            return
        tot = self._ws.live_total(self._pool)
        n = len(self._items)
        prin = sum(float(getattr(c, "principal", 0) or 0) for c in self._items)
        bits = [f"{n} 条", f"本金合计 {prin / 10000:,.1f} 万"]
        if tot > 0 and prin > 0:
            bits.append(f"= 总资金的 {prin / tot:.2f} 倍")
        if self._usages:
            bits.append(f"已量 {len(self._usages)} 条占用")
        bad = str(self._ws.basis) != "principal" and not self._usages
        if bad:
            bits.append("⚠ 没量占用，校准会退回按分配额")
        if self._measure_bad:
            bits.append(f"⚠ {len(self._measure_bad)} 条没量出来")
        bar.hint("　·　".join(bits), bad=bad)

    # ---------------------------------------------------------- 资金分配

    def _measure_usages(self) -> None:
        """把当前工作区的策略逐条跑一遍历史回测，量出实际占用。

        和「组合报告」共用 `_prepare` / `run_replay_job` —— 同一份行情、同一套
        成本档，量出来的占用才是这条策略真实会压的钱，而不是另算一套。

        取数在主线程（`DataStore` 是主线程的缓存），计算放后台线程：几十条策略
        各自一遍回测，同步跑窗口会「未响应」。
        """
        if getattr(self, "_ms_thread", None) is not None:
            self._on_failed("上一次「量占用」还在跑，等它结束")
            return
        cfgs = list(self._items)
        if not cfgs:
            self._on_failed("这个工作区还没有策略")
            return
        jobs, bad = [], []
        for cfg in cfgs:
            try:
                kind, payload, why = self._prepare(cfg)
            except Exception as exc:               # noqa: BLE001
                kind, payload, why = "single", None, f"{type(exc).__name__}: {exc}"
            if payload is None:
                bad.append(f"{cfg.title()}（{why}）")
                continue
            jobs.append((cfg, kind, payload))
        if not jobs:
            self._on_failed("一条都取不到行情：" + "；".join(bad[:3]))
            return
        self._measure_bad = bad
        self._on_log(f"开始量占用：{len(jobs)} 条"
                     + (f"（{len(bad)} 条取不到数据，跳过）" if bad else ""))
        if hasattr(self, "ws_bar"):
            self.ws_bar.btn_measure.setEnabled(False)
        self._ms_thread = QThread(self)
        self._ms_worker = MeasureWorker(jobs)
        self._ms_worker.moveToThread(self._ms_thread)
        self._ms_thread.started.connect(self._ms_worker.run)
        self._ms_worker.progress.connect(self._on_measure_progress)
        self._ms_worker.done.connect(self._on_measure_done)
        self._ms_worker.failed.connect(self._on_measure_failed)
        self._ms_thread.start()

    def _on_measure_progress(self, i: int, n: int, label: str) -> None:
        self.lbl_msg.setText(f"量占用 {i}/{n}：{label}")

    def _on_measure_failed(self, msg: str) -> None:
        self._close_measure()
        self._on_failed(f"量占用失败：{msg}")

    def _close_measure(self) -> None:
        th, wk = getattr(self, "_ms_thread", None), getattr(self, "_ms_worker", None)
        self._ms_thread = self._ms_worker = None
        if hasattr(self, "ws_bar"):
            self.ws_bar.btn_measure.setEnabled(True)
        if wk is not None:
            wk.deleteLater()
        if th is not None:
            th.quit()
            th.wait(3000)
            th.deleteLater()
        self.lbl_msg.setText("")

    def _on_measure_done(self, payload) -> None:
        usages, bad = payload
        self._close_measure()
        self._usages.update({u.sid: u for u in usages})
        self._measure_bad = list(bad)
        empty = [u.name for u in usages if u.avg_margin <= 0]
        msg = f"量完 {len(usages)} 条占用"
        if empty:
            msg += f"（其中 {len(empty)} 条一笔都没开过，量不出占用）"
        if bad:
            msg += f"；{len(bad)} 条没量出来：{'；'.join(bad[:2])}"
        self._on_log(msg)
        self._fill_strategies()
        self._render()
        self._ws_hint()

    def _alloc_preview(self) -> None:
        """按当前分配方式算一遍，弹预览。点「应用分配」才写回工作区。"""
        if self._ws is None:
            return
        shares = WS.allocate_workspace(self._ws, self._pool, self._usages or None)
        if not shares:
            self._on_failed("这个工作区还没有策略，或者总资金是 0")
            return
        dlg = AllocDialog(shares, self._ws.live_total(self._pool),
                          basis=self._ws.basis, measured=bool(self._usages), parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        n = WS.apply_shares(self._ws, dlg.shares)
        # 手动口径下把这次的金额记下来，下次打开还是同一套比例
        if str(self._ws.mode) == "manual":
            self._ws.manual = {s.sid: float(s.new_principal) for s in dlg.shares
                               if s.new_principal > 0}
        self._save_ws()
        self._sync_ws()
        self._fill_strategies()
        self._render()
        after = sum(float(getattr(c, "principal", 0) or 0) for c in self._items)
        tot = self._ws.live_total(self._pool)
        self._on_log(
            f"已应用分配：{n} 条策略本金已更新，合计 {after / 10000:,.1f} 万"
            + (f"（总资金的 {after / tot:.2f} 倍）" if tot > 0 else "")
            + "。正在跑的连接不会自动改用新本金，需停止后重新启动")

    def _reload_accounts(self) -> None:
        cur = self.cb_acct.currentData()
        self.cb_acct.blockSignals(True)
        self.cb_acct.clear()
        for s in self._accounts:
            self.cb_acct.addItem(s.label(), s.name)
        if cur:
            idx = self.cb_acct.findData(cur)
            if idx >= 0:
                self.cb_acct.setCurrentIndex(idx)
        self.cb_acct.blockSignals(False)

    def _spec(self, name: str):
        for s in self._accounts:
            if s.name == name:
                return s
        return None

    def _account_of(self, cfg) -> str:
        """这条策略实际用哪个账户。空 = 跟随默认账户。"""
        return cfg.account or (self.cb_acct.currentData() or "")

    def _accounts_of(self, cfg) -> list[str]:
        """这条策略要跑的全部账户（多账户在这里展开）。"""
        return cfg.account_list(self.cb_acct.currentData() or "")

    def _save_cfg(self, cfg) -> None:
        """把一条策略写回磁盘。**参数和本金分两处写**。

        `principal` 是**工作区**的东西，不是策略的：同一条策略在「实盘组合」里
        分 50 万、在「新策略试跑」里分 5 万，两个都对。所以：

        - 参数（规则 / 因子 / 品种 / 仓位口径 / 账户 / 开关）→ `portfolio.json`
        - 本金 → 当前工作区的成员（`workspaces.json`）

        只写参数那份时，要把 `principal` 换回池子里那条的原值，否则会被当前
        工作区的钱覆盖 —— 换个分页看，默认本金就变了。
        """
        sid = str(getattr(cfg, "sid", "") or "")
        ws_p = float(getattr(cfg, "principal", 0) or 0)
        pool_p = 0.0
        for c in self._pool:
            if str(getattr(c, "sid", "") or "") == sid:
                pool_p = float(getattr(c, "principal", 0) or 0)
                break
        save = cfg.clone() if hasattr(cfg, "clone") else cfg
        save.sid = sid                      # clone 会换 sid，写回要认原 sid
        save.principal = pool_p or ws_p
        SStore.update_portfolio(save)
        if self._ws is not None and sid:
            self._ws.set_principal(sid, ws_p)
            m = self._ws.find(sid)
            if m is not None:
                m.enabled = bool(getattr(cfg, "enabled", True))
            self._save_ws()
        self._pool = SStore.load_portfolio()

    def _universe_of(self, cfg) -> list[dict]:
        """这条策略能挑的品种清单。

        拿不到（没接数据源、市场还没扫完）就给空表 —— 设置框里的品种面板随之
        不出现，品种保持原样。这里**不能**抛：开一个设置框把整页带崩是最糟的
        """
        if self._universe is None:
            return []
        try:
            return list(self._universe(cfg.market) or [])
        except Exception:  # noqa: BLE001
            return []

    # ---------------------------------------------------------- 渲染

    def _fill_strategies(self) -> None:
        """重建策略表。选中行按 sid 找回，不然每次刷新都会跳回第一行。"""
        keep = set(self._selected_sids())
        t = self.tbl_strat
        t.blockSignals(True)
        try:
            t.setRowCount(len(self._items))
            for i, cfg in enumerate(self._items):
                accs = cfg.account_list(self.cb_acct.currentData() or "")
                acc_txt = ("默认" if not accs else
                           (accs[0] if len(accs) == 1 else
                            f"{accs[0]} 等 {len(accs)} 个"))
                vals = [
                    cfg.name.strip() or cfg.title(),
                    _sym_text(cfg),
                    cfg.strategy_label(),
                    cfg.size_text(),
                    acc_txt,
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setData(Qt.UserRole, cfg.sid)
                    if j == 3:
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    t.setItem(i, j, item)
                it = t.item(i, 1)
                if it is not None:
                    it.setToolTip(_sym_tip(cfg))
                if len(accs) > 1:
                    it4 = t.item(i, 4)
                    if it4 is not None:
                        it4.setToolTip(
                            "同一条策略挂在这些账户上：\n　" + "\n　".join(accs)
                            + "\n每个账户各算各的手数（各自的本金和保证金），"
                              "策略参数只有一份")
                for j in (5, 6, 7, 8, 9):
                    t.setItem(i, j, QTableWidgetItem("—"))
            if keep:
                for i in range(t.rowCount()):
                    it = t.item(i, 0)
                    if it is not None and it.data(Qt.UserRole) in keep:
                        t.selectRow(i)
        finally:
            t.blockSignals(False)
        self._sync_buttons()
        self._ws_hint()

    def _selected_sids(self) -> list[str]:
        model = self.tbl_strat.selectionModel()
        if model is None:
            return []
        out = []
        for i in sorted({i.row() for i in model.selectedRows()}):
            it = self.tbl_strat.item(i, 0)
            sid = "" if it is None else str(it.data(Qt.UserRole) or "")
            if sid:
                out.append(sid)
        return out

    def _selected_cfgs(self) -> list:
        """当前选中的所有配置，按表格顺序。组合报告要这个。"""
        sids = self._selected_sids()
        out = []
        for sid in sids:
            for cfg in self._items:
                if cfg.sid == sid:
                    out.append(cfg)
                    break
        return out

    # ---------------------------------------------------------- 结果视图

    def _load(self, cfg):
        """取这个配置的行情。返回 (df, 出错原因)。"""
        if self._loader is None:
            return None, "没有接上行情数据源"
        mk = replay.market_of(cfg)
        key = cfg.key()
        try:
            df = self._loader(mk, key, cfg.period or "1d")
        except Exception as exc:  # noqa: BLE001
            return None, f"{key} 读取失败：{exc}"
        if df is None or df.empty:
            return None, f"{key} 没有 {cfg.period} 的行情数据"
        return df, ""

    def _load_frames(self, cfg, need: int = 0) -> tuple[dict, str]:
        """一批品种的行情。返回 (frames, 出错原因)。

        截面策略要 ≥ `MIN_CROSS` 个；规则 / 因子的**多品种**要 ≥ 2 个。两条路
        取数一样，只是下限不同 —— 以前这里写死截面，于是多品种策略在组合页
        只会按 `cfg.key()` 取一个品种，回测出来的曲线和列表上写的不一样。
        """
        keys = list(getattr(cfg, "symbols", None) or [])
        need = int(need or (cross_engine.MIN_CROSS
                            if getattr(cfg, "family", "") == "cross" else 2))
        if len(keys) < need:
            return {}, (f"只记了 {len(keys)} 个品种，"
                        + ("截面策略" if getattr(cfg, "family", "") == "cross"
                           else "多品种策略") + f"至少要 {need} 个")
        if self._loader is None:
            return {}, "没有接上行情数据源"
        mk = replay.market_of(cfg)
        frames: dict = {}
        missing: list[str] = []
        for key in keys:
            try:
                df = self._loader(mk, key, cfg.period or "1d")
            except Exception as exc:  # noqa: BLE001
                missing.append(f"{key}（{exc}）")
                continue
            if df is None or df.empty:
                missing.append(key)
                continue
            frames[key] = df
        if len(frames) < need:
            return {}, ("取不到足够的行情：" + "、".join(missing)
                        if missing else "行情为空")
        return frames, ""

    def _ctx(self, cfg, rp, period: str = "") -> ViewCtx:
        return ViewCtx(
            symbol=cfg.symbol,
            name=cn_name(cfg.symbol) or "",
            exchange=cfg.exchange,
            period=period or cfg.period,
            market=replay.market_of(cfg),
            label=rp.label,
            legend=("▲紫＝开多　▼紫＝平多" if rp.long_only
                    else "▲紫＝开多　▼紫＝平多　▼琥珀＝开空　▲琥珀＝平空"),
            price_note="价格（后复权）" if not cfg.exchange else "价格（前复权·主力连续）",
            # 配置里存了回测区间就只画这一段。`replay.run` 已经按区间切过数据，
            # 前面多留的 250 根预热段在这里被裁掉 —— 指标是全区间算的，图上只显示
            # 用户选的那几年，两边看到的是同一段行情
            range_years=(float(cfg.range_years) or None),
            rule_key=cfg.rule_key,
            overlay=True,
            long_only=rp.long_only,
            bars_per_year=rp.bars_per_year,
            factor_cfg=rp.factor_cfg,
            filter_active=rp.filter_active,
            selected_keys=rp.keys,
            outcome=rp.outcome,
        )

    def _open_single(self) -> None:
        cfgs = self._selected_cfgs()
        if not cfgs:
            self._on_failed("先选中一条策略")
            return
        if len(cfgs) > 1:
            self._on_log(f"选中了 {len(cfgs)} 条，先显示第一条；要看整体请点「组合报告」")
        self._start_view(cfgs[:1], "single")

    def _open_combo(self) -> None:
        cfgs = self._selected_cfgs()
        if len(cfgs) < 2:
            self._on_failed("按住 Ctrl 至少选两条策略，才谈得上组合")
            return
        self._start_view(cfgs, "combo")

    # ---- 回放：取数在主线程，计算在后台线程 ----
    #
    # 多品种 / 截面回放一次是 N 个品种的策略计算，机器学习还要各训几十折 ——
    # 19 个品种 / 5m 十年实测十几分钟。以前整段同步跑，窗口「未响应」，连
    # 「停止」都点不动，用户只能强杀进程。
    #
    # 取数留在主线程：`DataStore` 是主线程的缓存，后台线程同时读同一个 DuckDB
    # 连接不安全（研究页那条多品种也是先取数再进线程的）。
    #
    # `show_config` / `show_combo` 保留同步版本（脚本、自检、Agent 用得上），
    # 和按钮那条路共用 `_prepare` / `run_replay_job` / `_present_*`，不会跑出
    # 两套口径。

    def _prepare(self, cfg) -> tuple[str, object, str]:
        """主线程取数。返回 (kind, payload, 出错原因)。"""
        is_cross = getattr(cfg, "family", "") == "cross"
        keys = list(getattr(cfg, "symbols", None) or [])
        if is_cross:
            frames, why = self._load_frames(cfg)
            if not frames:
                return ("cross", None, why)
            return ("cross", (frames, self._extra_periods_multi(cfg, frames)), why)
        if len(keys) >= 2:
            frames, why = self._load_frames(cfg, need=2)
            if not frames:
                return ("multi", None, why)
            return ("multi", (frames, self._extra_periods_multi(cfg, frames)), why)
        df, why = self._load(cfg)
        if df is None:
            return ("single", None, why)
        return ("single", (df, self._extra_periods(cfg)), why)

    def _extra_periods(self, cfg) -> dict:
        """跨周期因子要用的别的周期行情（主线程取数）。

        取数走的是同一个 `self._loader` —— 组合页和主窗口的数据源是同一个，
        没必要各写一套。没配跨周期因子时返回空 dict，一次多余的读都不做。
        """
        fc = replay.factor_config(getattr(cfg, "factor_cfg", {}) or {})
        # 不能拿 `fc.factor_periods` 当闸门：周期编在因子实例 id 里
        # （`mom_60@1d_tq`），那个映射在新配置里是空的
        if self._loader is None:
            return {}
        from ttrade import factor_strategy as fstrat

        mk = replay.market_of(cfg)
        key = cfg.key()
        base = getattr(cfg, "period", "") or "1d"
        return fstrat.period_frames(
            fc, list(getattr(cfg, "factor_keys", []) or []), base,
            lambda p: self._loader(mk, key, p),
            extra_keys=list(getattr(fc, "trigger_keys", None) or []),
        )

    def _extra_periods_multi(self, cfg, frames: dict) -> dict:
        """多品种 / 截面：**逐品种**一份跨周期行情 `{品种: {周期: 行情}}`。

        每个品种各自去读：有的品种有 5 分钟、有的只有日线，共用一份等于
        按「第一个品种能读到什么」决定其余品种怎么算。
        """
        fc = replay.factor_config(getattr(cfg, "factor_cfg", {}) or {})
        if self._loader is None:
            return {}
        from ttrade import factor_strategy as fstrat

        mk = replay.market_of(cfg)
        base = getattr(cfg, "period", "") or "1d"
        keys = list(getattr(cfg, "factor_keys", []) or [])
        # 截面策略的因子在 `factor_keys`，时序腿的在 `timing_factor_keys` ——
        # 两条腿都可能被指到别的周期上
        tkeys = list(getattr(cfg, "timing_factor_keys", []) or [])
        out: dict[str, dict] = {}
        for k in frames:
            ex = fstrat.period_frames(
                fc, keys + tkeys, base,
                lambda p, _k=k: self._loader(mk, _k, p),
                extra_keys=list(getattr(fc, "trigger_keys", None) or []),
            )
            if ex:
                out[k] = ex
        return out

    def _view_hint(self, jobs: list, mode: str) -> str:
        """点下去立刻显示的那句话。用户得知道「在跑、跑什么、要多久」。"""
        if mode == "combo":
            return (f"正在回测 {len(jobs)} 条策略并合成组合…"
                    f"（后台运行，界面可以继续用）")
        cfg, kind, payload = jobs[0]
        # 多品种 / 截面的 payload 可能是 `(行情, 跨周期行情)`，这里只关心条数
        if isinstance(payload, tuple):
            payload = payload[0]
        if kind == "cross":
            return f"正在回测截面策略（{len(payload)} 个品种）…（后台运行）"
        if kind == "multi":
            fc = getattr(cfg, "factor_cfg", {}) or {}
            ml = ((getattr(cfg, "family", "") or "") == "factor"
                  and (getattr(cfg, "factor_mode", "") or "") == "ml")
            if ml:
                return (f"正在回测 {len(payload)} 个品种"
                        + ("（合并训练：一个模型训完再各自回测）"
                           if fc.get("pooled") else "（每个品种各训一遍）")
                        + "…机器学习要训几十折，可能要一两分钟"
                          "（后台运行，可以接着看别的）")
            return f"正在回测 {len(payload)} 个品种…（后台运行）"
        return f"正在回测 {cfg.title()}…（后台运行）"

    def _start_view(self, cfgs: list, mode: str) -> None:
        if self._view_thread is not None:
            self._on_log("上一次回测还在跑，等它算完再来")
            return
        jobs, missing = [], []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for i, cfg in enumerate(cfgs, 1):
                if len(cfgs) > 1:
                    # 取数本身也要几十秒（19 个品种 × 5m 十年），一条条报出来，
                    # 不然点完「组合报告」还是什么都没发生
                    self._on_log(f"读取行情 {i}/{len(cfgs)}：{cfg.title()}")
                    QApplication.processEvents()
                kind, payload, why = self._prepare(cfg)
                if payload is None:
                    missing.append(f"{cfg.title()}：{why}")
                    continue
                jobs.append((cfg, kind, payload))
        finally:
            QApplication.restoreOverrideCursor()
        if not jobs:
            self._on_failed("回测失败：" + "；".join(missing or ["没有可用的行情"]))
            return

        self._view_mode, self._view_missing = mode, missing
        # 单品种那条路出图时要用这份原始行情当「实时段往上拼」的底
        self._view_payload = jobs[0][2] if mode == "single" else None
        self.btn_view.setEnabled(False)
        self.btn_combo.setEnabled(False)
        self._on_log(self._view_hint(jobs, mode))
        QApplication.processEvents()

        thread = QThread(self)
        worker = ReplayWorker(jobs)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_view_progress)
        worker.done.connect(self._on_view_done)
        worker.failed.connect(self._on_view_failed)
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        self._view_thread, self._view_worker = thread, worker
        thread.start()

    def _on_view_progress(self, done: int, total: int, label: str) -> None:
        if total > 1:
            self._on_log(f"正在回测 {done}/{total}：{label}…")

    def _finish_view(self) -> None:
        self._view_thread = None
        self._view_worker = None
        self._view_mode = ""
        self._view_missing = []
        self._view_payload = None
        self._sync_buttons()

    def _on_view_done(self, pairs) -> None:
        mode, missing = self._view_mode, list(self._view_missing)
        raw = self._view_payload
        self._finish_view()
        try:
            if mode == "combo":
                self._present_combo(pairs, missing)
                return
            cfg, rp = pairs[0]
            if not rp.ok:
                self._on_failed(f"回测失败：{rp.error}")
                return
            # 单品种要留着这份原始行情做「实时段往上拼」的底（见 `_present_single`）
            self._present_single(cfg, rp, "", raw=raw)
        except Exception as exc:  # noqa: BLE001 - 出图炸了也要说清楚
            self._on_failed(f"出图失败：{type(exc).__name__}: {exc}")

    def _on_view_failed(self, msg: str) -> None:
        self._finish_view()
        self._on_failed(f"回测失败：{msg}")

    def show_config(self, cfg, note: str = "") -> str:
        """回放一份配置并展示（同步）。返回一段纯文本结论（失败时是原因）。

        按钮那条路走 `_start_view`（后台线程）；这个同步版本留给脚本 / 自检 /
        需要「跑完拿字符串」的调用方。失败也要返回原因，不能只弹个红字。
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            kind, payload, why = self._prepare(cfg)
        finally:
            QApplication.restoreOverrideCursor()
        if payload is None:
            return f"回测失败：{why}"
        self._on_log(self._view_hint([(cfg, kind, payload)], "single"))
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rp = run_replay_job(cfg, kind, payload)
        finally:
            QApplication.restoreOverrideCursor()
        if not rp.ok:
            return f"回测失败：{rp.error}"
        return self._present_single(cfg, rp, note, raw=payload)

    def _present_single(self, cfg, rp, note: str = "", raw=None) -> str:
        """把一份回放结果画出来。返回一段纯文本结论。"""
        is_cross = getattr(cfg, "family", "") == "cross"
        keys = list(getattr(cfg, "symbols", None) or [])
        is_multi = (not is_cross) and len(keys) >= 2
        period = PERIOD_LABEL.get(cfg.period, cfg.period)
        if is_cross or is_multi:
            # 多品种 / 截面都没有「某一条策略的 K 线」：仓位是一篮子，画 K 线会
            # 误导。走组合视图那几页（绩效 / 交易记录 / 阶段）
            self._stop_live()
            tabs, view = self._ensure_view("portfolio")
            set_tab_visible(tabs, "截面因子分析", is_cross)
            ctx = ViewCtx(
                symbol="", name=f"{len(keys)} 个品种",
                exchange="", period=cfg.period, market=replay.market_of(cfg),
                label=rp.label, price_note="净值",
                range_years=float(cfg.range_years or 0) or None,
                long_only=rp.long_only, bars_per_year=rp.bars_per_year,
                overlay=False, selected_keys=[],
                cross=getattr(rp, "cross", None) if is_cross else None,
            )
            view.show_result(
                None, None, rp.res, None, ctx,
                trades=[(t.symbol or rp.label, t) for t in rp.res.trades],
            )
            self._show_view("portfolio", f"{cfg.title()}｜{period}"
                                         + (f"｜{note}" if note else ""))
            if is_multi:
                self._on_log(f"回测完成：{len(keys)} 个品种 · {rp.label}")
            return replay.summary(rp, cfg.title(), period)

        _tabs, view = self._ensure_view("single")
        view.show_result(rp.df, rp.raw, rp.res, rp.bh, self._ctx(cfg, rp))
        # 记下本地历史这份「底」，实时段每 5 秒往上拼一次。存 df 而不是重读库：
        # 库里那份要解码几万行，5 秒重读一次等于白烧
        #
        # 🔴 `raw` 可能是 `_prepare` 的原始 payload，也就是 `(行情, 跨周期行情)`
        # 这种**两元组**（配了跨周期因子时才有第二份，但为了形状统一，没配也是
        # 两元组）。直接把它当 DataFrame 存下去，`merge_live` 里 `local.empty`
        # 就会抛 `AttributeError: 'tuple' object has no attribute 'empty'` ——
        # 界面顶部那行「实时 K 线刷新失败」红字，5 秒刷一次，图永远不动。
        # 拆包只有 `run_replay_job` 做了，这里和 `show_config` 都漏了。
        self._live_cfg, self._live_base, self._live_sid = (
            cfg, raw[0] if isinstance(raw, tuple) else raw, cfg.sid)
        self._live_busy = False
        self._live_timer.start()
        # cfg.title() 已经含「品种 · 策略」，别再拼一遍 key 和策略名
        self._show_view("single", f"{cfg.title()}｜{period}"
                                 + (f"｜{note}" if note else ""))
        return replay.summary(rp, cfg.key(), period)

    def show_combo(self, cfgs: list) -> str:
        """多策略等权组合报告（同步）。返回结论文本（失败时是原因）。"""
        reps: list = []
        missing: list[str] = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for cfg in cfgs:
                kind, payload, why = self._prepare(cfg)
                if payload is None:
                    missing.append(f"{cfg.title()}：{why}")
                    continue
                rp = run_replay_job(cfg, kind, payload, with_bh=False)
                if not rp.ok:
                    missing.append(f"{cfg.title()}：{rp.error}")
                    continue
                reps.append((cfg, rp))
        finally:
            QApplication.restoreOverrideCursor()
        return self._present_combo(reps, missing)

    def _present_combo(self, reps: list, missing: list) -> str:
        """把几条策略的等权组合画出来。`reps` 是 [(cfg, Replay), ...]。"""
        missing = list(missing or [])
        # 后台那条路把失败的也一起递回来了（失败原因要写进结论），这里筛掉
        ok = []
        for cfg, rp in reps:
            if rp.ok:
                ok.append((cfg, rp))
            else:
                missing.append(f"{cfg.title()}：{rp.error}")
        reps = ok

        if not reps:
            return "组合报告失败：" + "；".join(missing or ["没有可用的策略"])
        if len(reps) < 2:
            # 一条都算不出来时「组合」没有意义，直接说清楚少了谁
            return ("组合报告需要至少 2 条能算出结果的策略。"
                    + ("　" + "；".join(missing) if missing else ""))

        # 周期一致就照抄，混着就留空：留空时交易记录会带上时分（分钟级策略
        # 的交易时间差几十分钟是有意义的），持仓长度也改用「根（约 N 天）」
        periods = {c.period for c, _r in reps}
        period = periods.pop() if len(periods) == 1 else ""
        res = combine([rp.res for _c, rp in reps], label="组合")
        # 这里的 bars_per_year 是给 `hold_text` 换算「平均持仓 N 根」用的，取的是
        # **策略自己的** bar 频率（交易记录里的 bars 就是它）。组合曲线本身落在
        # 日频上，那个频率在 `combine` 内部算指标时用，别混用
        bpy = reps[0][1].bars_per_year
        # 交易记录按开仓时间混排：分组拼接的话，表格看起来像「先跑完 A 再跑 B」
        trades = combine_trades([(c.title(), rp.res) for c, rp in reps])
        ctx = ViewCtx(
            symbol="", name="", exchange="", period=period,
            market=replay.market_of(reps[0][0]), label="组合",
            price_note="净值", long_only=False, bars_per_year=bpy,
            selected_keys=[], overlay=False,
        )
        # 组合是几条策略的净值合成，没有「这一条策略的实时 K 线」可言
        self._stop_live()
        tabs, view = self._ensure_view("portfolio")
        set_tab_visible(tabs, "截面因子分析", False)
        view.show_result(None, None, res, None, ctx, trades=trades)
        names = "、".join(c.title() for c, _r in reps)
        self._show_view("portfolio", f"组合报告｜{len(reps)} 条策略等权｜{names}")

        head = f"组合（{len(reps)} 条策略）"
        rp = replay.Replay(res=res, bh=None, period=period,
                           bars_per_year=bpy)
        text = replay.summary(rp, head, "等权")
        if not period:
            # 混了日线和分钟线时「平均持仓 40 根」没法换算成天数：一根 30 分钟
            # 和一根日线不是同一个单位。与其给个错的天数，不如直说
            text += "\n注意：组合里混了不同周期，平均持仓的根数不可比"
        if missing:
            text += "\n未能加入组合：" + "；".join(missing)
        return text

    # 引擎 0.4 秒推一次快照。每次都把两张表全量重画，合约一多主线程就被占满，
    # 界面上看着就是「一切到这页就卡」。1 秒一次足够看清，也不耽误看数。
    RENDER_GAP = 1.0

    def _on_snapshot(self, key: str, snap: dict) -> None:
        self._snaps[key] = snap
        for slot in snap.get("slots", []):
            sid = str(slot.get("sid") or "")
            self._slot_view[sid] = slot
            if sid:
                self._slot_multi.setdefault(sid, {})[key] = slot
        bars = snap.get("bars") or {}
        if bars:
            self._apply_live(bars)
        if time.monotonic() - self._render_ts < self.RENDER_GAP:
            return
        self._render()

    def _stop_live(self) -> None:
        self._live_timer.stop()
        self._live_cfg = self._live_base = None
        self._live_sid = ""
        self._live_busy = False

    def _poll_live(self) -> None:
        """5 秒一趟：问引擎要一次当前这条策略的实时 K 线。

        引擎 0.4 秒推一次快照，但那里面不带 K 线（3000 行，每份都带太贵），
        这里按需点名要。找不到在跑的账户就安静等着 —— 策略没启动时也有图看，
        只是它停在库里最后一根。
        """
        # 不在结果页就别问了：切回列表还每 5 秒转一遍 3000 行，纯浪费
        if self.stack.currentIndex() != 1:
            return
        if self._live_cfg is None or not self.ctl.running:
            return
        # 请求发出去了却一直没回来（账户线程在重连之类），别把自己锁死
        if self._live_busy and time.monotonic() - self._live_req < 10:
            return
        sid = self._live_sid
        for key, snap in self._snaps.items():
            if any(s.get("sid") == sid for s in (snap.get("slots") or [])):
                self._live_busy = True
                self._live_req = time.monotonic()
                self.ctl.req_bars(key, [sid])
                return

    def _apply_live(self, bars: dict) -> None:
        """把实时段拼到本地历史后面，重跑一遍再画。"""
        self._live_busy = False
        cfg = self._live_cfg
        live = bars.get(self._live_sid)
        if cfg is None or live is None or live.empty:
            return
        try:
            df = merge_live(self._live_base, live)
            if df is None or df.empty:
                return
            rp = replay.run(cfg, df, extra=self._extra_periods(cfg))
        except Exception as exc:  # noqa: BLE001
            self._on_failed(f"实时 K 线刷新失败：{type(exc).__name__}: {exc}")
            return
        if not rp.ok:
            return
        _tabs, view = self._ensure_view("single")
        # **只重画 K 线**。走 `show_result` 会把绩效、阶段表、交易记录全部重建，
        # 三万点的净值曲线一次几百毫秒；用户切到「交易记录」页签看一眼，也在
        # 每 5 秒被拖住一次，Windows 直接标成「未响应」
        view.update_kline(rp.df)

    def _render_diag(self, t, i: int, cfg) -> None:
        """三列资金占用：平均开仓占用 / 峰值保证金占用 / 利用率。

        **平均 = 每笔交易开仓时压的钱**，取均值。不是逐根 bar 的均值 ——
        逐根会把空仓时间算进分母（日内策略一天只有两三根有仓），算出来只有
        本金的 1~2%，量的是「摊到每根 bar 用多少」而不是「出手一次用多少」。
        要跟「组合给它分了多少钱」比，必须用后者。

        峰值是逐根最大值（持仓期间保证金跟结算价走），和平均不是同一口径。

        没量过写「—」不写 0：0 会被读成「这条策略不用钱」。
        """
        u = self._usages.get(str(cfg.sid))
        if u is None:
            for j in (7, 8, 9):
                self._set_cell(t, i, j, "—", C_MUTED)
            it = t.item(i, 7)
            if it is not None:
                it.setToolTip("还没量。点上面的「量一下占用」跑一遍历史回测")
            return
        if u.avg_margin <= 0:
            for j in (7, 8, 9):
                self._set_cell(t, i, j, "—", C_MUTED)
            it = t.item(i, 7)
            if it is not None:
                it.setToolTip(
                    "量不出来：" + (u.note or "回测里一笔都没开过，没有占用可算"))
            return
        # 金额一律用「万」：26 条策略的占用从几千到几十万，写元位没人读得出量级
        self._set_cell(t, i, 7, f"{u.avg_margin / 10000:,.2f} 万", C_TEXT)
        self._set_cell(t, i, 8, f"{u.peak_margin / 10000:,.2f} 万", C_TEXT)
        self._set_cell(t, i, 9, f"{u.util * 100:.2f}%",
                       C_BAD if u.util > 0.9 else C_TEXT)
        it = t.item(i, 7)
        if it is not None:
            it.setToolTip(
                f"每笔开仓时平均压 {u.avg_margin / 10000:,.2f} 万"
                f"（名义敞口 {u.avg_notional / 10000:,.2f} 万 × 保证金率）\n"
                "口径：|手数| × 开仓价 × 乘数 × 保证金率，**逐笔取均值**\n"
                "（不是逐根 bar 的均值 —— 那个会被空仓时间稀释成 1~2%）\n"
                f"本金 {u.principal / 10000:,.1f} 万，"
                f"开仓预算 {u.capital / 10000:,.1f} 万，"
                f"共 {u.bars} 笔")
        it = t.item(i, 8)
        if it is not None:
            it.setToolTip(
                f"持仓期间压过的最高占用 {u.peak_margin / 10000:,.2f} 万"
                f"（逐根最大值，含持仓浮动；名义峰值 "
                f"{u.peak_notional / 10000:,.2f} 万）"
                + (f"\n= 本金的 {u.peak_margin / u.principal * 100:.1f}%"
                   if u.principal > 0 else ""))
        it = t.item(i, 9)
        if it is not None:
            it.setToolTip(
                "平均开仓占用 ÷ 本金 = **出手时用了分配额的几成**。\n"
                "`仓位` 那列写的 capital_pct% 就是这个数的上限：\n"
                "30% 仓位 + 利用率 28% → 它每次都几乎用满；\n"
                "利用率只有 8% 说明大部分开仓是被止损/风险口径卡住了手数。")

    def _render(self) -> None:
        self._render_ts = time.monotonic()
        t = self.tbl_strat
        for i, cfg in enumerate(self._items):
            slots = self._slots_of(cfg)
            # 多账户时「这条策略的目标仓」不是唯一的一个数（每个账户一个），
            # 塞其中一个进这一列就是错的 —— 交给右下监控器逐合约列出
            many = len(slots) > 1
            slot = {} if many else next(iter(slots.values()), {})
            if i >= t.rowCount():
                break
            self._render_diag(t, i, cfg)
            theo = _int(slot.get("target"))
            multi = _int(slot.get("multi"))
            # 仓位列写「口径 → 实际算出来的手数」。手数依赖价格和每手保证金，
            # 存盘时算不出来，只有引擎跑起来才知道 —— 所以两段拼在一起显示。
            lots = _int(slot.get("lots"))
            size = cfg.size_text() + (f" → {lots} 手" if lots > 0 else "")
            self._set_cell(t, i, 3, size, C_TEXT)
            it3 = t.item(i, 3)
            if it3 is not None:
                if multi:
                    it3.setToolTip(
                        f"{multi} 个品种各按自己的保证金算手数，逐合约见右下监控器。\n"
                        f"资金按品种数均分：每个品种 {cfg.principal / multi / 10000:g} 万"
                    )
                else:
                    it3.setToolTip(
                        f"每手保证金按品种自己的比例算（商品/股指约 10 倍，国债 30~50 倍）。"
                        f"\n当前价下满仓 {lots} 手" if lots > 0 else
                        "引擎还没算出实际手数（需要行情）"
                    )
            # 「理论」只对**正在运行**的策略显示。停机（报单到顶 / 出错）不会清零
            # target，继续显示 -11 会让人以为它还在管这 -11 手；而监控器按 running
            # 汇总，那边是 0 /「未托管」—— 同一页两处就打架了。停机时显示「—」，
            # 把残值放进 tooltip。
            if many:
                # 多账户：每个账户一个槽位，目标仓不唯一
                self._set_cell(t, i, 5, "—", C_MUTED)
                it5 = t.item(i, 5)
                if it5 is not None:
                    bits = [f"{k or '默认'}：目标 {_int(v.get('target')):+d} 手"
                            + ("" if v.get("running") else "（未运行）")
                            for k, v in slots.items()]
                    it5.setToolTip(
                        f"这条策略挂在 {len(slots)} 个账户上，各算各的手数：\n　"
                        + "\n　".join(bits)
                        + "\n逐合约的持仓在右下监控器里")
            elif slot.get("running") and multi:
                # 多品种没有「一个」目标仓：19 个品种各有一个，加起来是 19 个
                # 不同合约的手数之和，没有任何意义。逐合约的去监控器看
                self._set_cell(t, i, 5, "—", C_MUTED)
                it5 = t.item(i, 5)
                if it5 is not None:
                    it5.setToolTip(
                        f"{multi} 个品种各自的目标仓在右下监控器里逐合约列出"
                    )
            elif slot.get("running"):
                self._set_cell(t, i, 5, str(theo), C_TEXT)
                it5 = t.item(i, 5)
                if it5 is not None:
                    it5.setToolTip(f"策略当前目标净仓 {theo:+d} 手")
            else:
                self._set_cell(t, i, 5, "—", C_MUTED)
                it5 = t.item(i, 5)
                if it5 is not None:
                    it5.setToolTip(
                        (f"策略未运行。停机前最后目标 {theo:+d} 手，不再参与托管；"
                         "实际持仓在右下监控器里记为「未托管」。")
                        if slot else "还没启动"
                    )

            # 这里**不再有「实际」列**。账户净仓是所有策略 + 手工单 + 别的平台的
            # 和，单条策略的「实际」根本不存在，拿账户去比只会显示成一片红。
            # 要核对账户，看右下监控器（策略理论 / 账户实际 / 差额），对齐也只在
            # 那里的一键对齐里做。

            # 状态列只放短标签，「买了 2 手 @ 3046」这种细节放 tooltip ——
            # 塞进表格会把列挤成一团，横向滚动条一起来就没人看了。
            if not slot:
                if cfg.enabled and self.ctl.is_running(self._account_of(cfg)):
                    # 账户连接在跑、但引擎还没推来这一条的快照。两种可能：刚点完
                    # 启动正在连（几秒），或者从别处加进来后一直没推给引擎。
                    # 只说「未接入」会让人以为按钮没生效，去连点第二下
                    text, color = "接入中", C_MUTED
                    tip = ("账户连接在跑，这条的首份快照还没到。\n"
                           "刚点过启动的话等几秒；一直这样就是没接上，"
                           "再点一次「启动 / 停止」")
                else:
                    text, color, tip = "未连接", C_IDLE, "还没启动"
            elif slot.get("error") and "暂不支持实盘" in str(slot["error"]):
                # 截面这一类「引擎结构上跑不了」的，不该显示成红字「错误」——
                # 那不是故障，是这一条本来就只能回测。原因留在 tooltip 里
                text, color, tip = "仅回测", C_MUTED, slot["error"]
            elif slot.get("error"):
                text, color, tip = "错误", C_BAD, slot["error"]
            elif slot.get("ml_wait"):
                # 机器学习还没出信号。这段时间**不动仓**，显示成「运行中」会让人
                # 以为它在按 0 信号平仓。
                #
                # 三种等待长得一样，等待时间差 10 倍，必须分开说：
                #   retrain —— 没有可用产物，退回后台重训（几十秒到十几分钟）
                #   其他   —— 算不出来的**真实原因**（永远等不到）
                #   空     —— 还在算（快路径，几秒）
                why = str(slot.get("ml_why") or "")
                if why == "retrain":
                    text, color = "重训中", C_MUTED
                    tip = ("没有可用的训练产物，正在后台重新训练模型。\n"
                           "这一步很慢（每个品种几十秒，19 个品种十几分钟），"
                           "算出来之前不动仓。\n"
                           "建议先点「训练模型」把模型训好，再启动实盘。")
                elif why:
                    text, color = "信号异常", C_BAD
                    tip = (f"机器学习信号算不出来：{why}\n"
                           "算出来之前不动仓。\n"
                           "细节看日志：%LOCALAPPDATA%\\ChanQuant\\logs\\live.log")
                else:
                    text, color = "计算中", C_MUTED
                    tip = ("机器学习正在算信号（用训练好的模型，只预测不重训），"
                           "算出来之前不动仓。\n品种多 / 历史长时第一次要等几十秒")
            elif slot.get("running"):
                text, color = "运行中", C_OK
                tip = slot.get("action") or ""
            elif cfg.enabled:
                text, color = "已停止", C_MUTED
                tip = "启动开关是开的，但这一轮没跑起来 —— 看看顶部连接状态和日志"
            else:
                text, color, tip = "未启用", C_IDLE, "点「启动 / 停止」把它打开"
            self._set_cell(t, i, 6, text, color)
            it = t.item(i, 6)
            if it is not None and it.toolTip() != tip:
                it.setToolTip(tip)

        rows = []
        for key, snap in self._snaps.items():
            for r in snap.get("mismatch") or []:
                rows.append((key, r))
        rows.sort(key=lambda kr: (not kr[1].get("managed"), kr[0], kr[1]["symbol"]))
        m = self.tbl_mon
        # 内容没变就不重建。重建一次要给每格新建 QTableWidgetItem，
        # 几十个合约不一致的时候这活儿一秒干一次，主线程就废了
        sig = tuple((k, r["symbol"], r["target"], r["actual"], r["delta"],
                     bool(r.get("managed")), int(r.get("a_long") or 0),
                     int(r.get("a_short") or 0)) for k, r in rows)
        if sig != self._mon_sig:
            self._mon_sig = sig
            m.setRowCount(len(rows))
            for i, (key, r) in enumerate(rows):
                managed = bool(r.get("managed"))
                locked = bool(r.get("locked"))
                have = str(r["actual"])
                if locked:
                    # 净 0 但两条腿都有仓。只写 0 等于什么都没说
                    have = (f"{r['actual']}（多{r['a_long']} 空{r['a_short']}）")
                note = "策略持仓" if managed else "未托管（策略已停或手工开仓）"
                if locked:
                    note += " · 锁仓，两条腿都要平"
                vals = [
                    key, r["symbol"], str(r["target"]), have,
                    _delta_text(r), note,
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    if j in (2, 3, 4):
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if j == 4:
                        item.setForeground(QColor(C_BAD if managed else C_MUTED))
                    if j == 5 and not managed:
                        item.setForeground(QColor(C_MUTED))
                    m.setItem(i, j, item)

        limits = [v.get("limits") for v in self._snaps.values() if v.get("limits")]
        if limits:
            self.lbl_limit.setText(" · ".join(limits))
            used = any(v.get("limit_used") for v in self._snaps.values())
            self.lbl_limit.setStyleSheet(
                f"color: {C_BAD if used else C_MUTED}; font-size: 11px;")
        else:
            self.lbl_limit.setText("")

        aligning = sum(int(v.get("aligning") or 0) for v in self._snaps.values())
        # 跑起来的账户（有策略绑在它上面）才有连接、才看得到持仓。剩下的账户
        # 无论有没有仓，这里都是**瞎的** —— 不能笼统说「一致」
        idle = [s.name for s in self._accounts if s.name not in self._snaps]
        idle_txt = ""
        if idle and (self.ctl.running or self._snaps):
            idle_txt = ("\n" + "、".join(idle)
                        + " 没有策略在跑，它们的持仓不在比对范围（先给该账户挂一条策略）。")
        # `_pos` 还没拉到时 `mismatch()` 必定为空。那不是一致，是还不知道
        blind = [k for k, v in self._snaps.items() if not v.get("pos_ready", True)]
        if not self.ctl.running:
            self.lbl_mon.setText("未启动。策略跑起来之后这里才会比对。")
            self.lbl_mon.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        elif blind:
            self.lbl_mon.setText(
                "、".join(blind) + " 的账户持仓还没同步（刚连上），这会儿的比对结果不作数，"
                "稍等几秒再点「一键对齐」。"
            )
            self.lbl_mon.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        elif aligning:
            self.lbl_mon.setText(
                f"对齐执行中：{aligning} 笔委托在途，等它们成交或撤掉。"
                "这期间策略不调仓，也不用再点。"
            )
            self.lbl_mon.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        elif not rows:
            self.lbl_mon.setText("账户仓位与策略理论持仓一致。" + idle_txt)
            self.lbl_mon.setStyleSheet(f"color: {C_OK}; font-size: 11px;")
        else:
            n_managed = sum(1 for _k, r in rows if r.get("managed"))
            n_orphan = len(rows) - n_managed
            n_lock = sum(1 for _k, r in rows if r.get("locked"))
            bits = []
            if n_managed:
                bits.append(f"{n_managed} 个合约与策略理论持仓不一致")
            if n_orphan:
                bits.append(f"{n_orphan} 个合约没有策略在管")
            if n_lock:
                bits.append(f"{n_lock} 个是锁仓（多空双开、净额 0，只按净额看是看不见的）")
            self.lbl_mon.setText(
                "，".join(bits) + "。「一键对齐」会把账户摆成和策略列表一致，\n"
                "未托管的仓位也会被平掉（会先列出来让你确认）。" + idle_txt
            )
            self.lbl_mon.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")

    @staticmethod
    def _set_cell(t: QTableWidget, row: int, col: int, text: str, color: str) -> None:
        item = t.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            if col in (5, 6):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            t.setItem(row, col, item)
        if item.text() != text:
            item.setText(text)
        item.setForeground(QColor(color))

    # ---------------------------------------------------------- 起停

    def _slots_of(self, cfg) -> dict:
        """这条策略在各账户上的槽位（键 = 账户名）。

        多账户策略在每个账户上各有一个槽位（各算各的手数），所以「这条策略的
        目标仓」不是唯一的一个数。单账户时这里只有一个元素，行为和以前一样。
        """
        by = self._slot_multi.get(str(getattr(cfg, "sid", "") or "")) or {}
        if by:
            return by
        one = self._slot_view.get(str(getattr(cfg, "sid", "") or ""))
        return {"": one} if one else {}

    def _live_plan(self) -> tuple[dict, list]:
        """**所有工作区**里启用的成员 → `{账户: [配置]}`。

        运行不按工作区分，按**账户**合并：一个账户一条天勤连接，跑该账户下所有
        工作区里启用的策略。两个工作区用同一个账户时不会开两条连接去抢同一个仓。

        同一条策略在两个工作区里都被分到同一个账户、本金却不一样时，一个槽位
        没法同时满足两个数 —— 取先遇到的那个，并回一句告警（不静默）。
        """
        plan: dict[str, list] = {}
        seen: dict[tuple[str, str], float] = {}
        warn: list[str] = []
        default = self.cb_acct.currentData() or ""
        for w in self._wss:
            cfgs, _missing = WS.resolve(w, self._pool)
            for c in cfgs:
                m = w.find(str(getattr(c, "sid", "") or ""))
                if m is not None and not m.enabled:
                    continue
                if not getattr(c, "enabled", False):
                    continue
                p = float(getattr(c, "principal", 0) or 0)
                for acct in c.account_list(default):
                    k = (acct, str(c.sid))
                    if k in seen:
                        if abs(seen[k] - p) > 1.0:
                            warn.append(
                                f"「{c.title()}」在账户 {acct} 上被多个工作区分了"
                                f"不同的本金（{seen[k] / 10000:g} 万 / {p / 10000:g} 万），"
                                "取先遇到的那个")
                        continue
                    seen[k] = p
                    plan.setdefault(acct, []).append(c)
        return plan, warn

    def _start_all(self) -> None:
        if not self._pool:
            self._on_failed("还没有策略，先在某个工作区里「从策略库添加」")
            return
        if not self._accounts:
            self._on_failed("还没有交易账户，请先到「账户透视和交易」页添加")
            return

        groups, warn = self._live_plan()
        for msg in warn:
            self._on_log(f"注意：{msg}")
        if not groups:
            self._on_failed("所有工作区里都没有启用的策略")
            return
        for name in list(groups):
            if not name:
                self._on_failed("有策略没有可用账户（既没绑账户，也没有默认账户）")
                return
            if self._spec(name) is None:
                self._on_failed(f"账户「{name}」已不存在，请重新绑定")
                return

        # 允许在「已经单独启动过某一条」的情况下再点：把其余补齐，不逼用户先停止全部
        newly = 0
        for name, cfgs in groups.items():
            if self.ctl.is_running(name):
                for cfg in cfgs:
                    if str(cfg.sid) not in self._slot_view:
                        self.ctl.add_or_update(name, cfg)
                    self.ctl.enable(name, cfg.sid, True)
                continue
            self.ctl.start(self._spec(name), cfgs, self._chase)
            newly += 1
        if newly:
            self._on_log(f"启动 {newly} 个账户的连接，共 "
                         f"{sum(len(v) for v in groups.values())} 条策略")
        else:
            self._on_log("连接本来就在跑，已把其余启用的策略补进来")
        self._refresh_conn_label()

    def _stop_all(self) -> None:
        if not self.ctl.running:
            return
        self.ctl.stop()
        # 这里不重画：线程还没真退，`ctl.running` 仍是 True，重画会走到
        # 「仓位一致」那个分支，看着像是刚对齐过。等 `_on_stopped` 再刷。
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.lbl_conn.setText("正在停止…")
        self.lbl_conn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
        self._on_log("正在停止…")

    def _on_stopped(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
        self.lbl_conn.setText("已停止")
        self.lbl_conn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        self._slot_view.clear()
        self._slot_multi.clear()
        self._snaps.clear()
        self._render()

    def _single_or_warn(self) -> object | None:
        """这几个动作一次只能作用在一条策略上。多选时先劝一句，别闷头改第一条。"""
        cfgs = self._selected_cfgs()
        if not cfgs:
            self._on_failed("先选中一条策略")
            return None
        if len(cfgs) > 1:
            self._on_failed(f"选中了 {len(cfgs)} 条。这个动作一次只作用于一条，先只选一条")
            return None
        return cfgs[0]

    @staticmethod
    def _want_on(slot: dict) -> bool:
        """「启动 / 停止」此刻该往哪个方向走 —— 看引擎里在不在跑。

        不能看存盘的 `cfg.enabled`：停止全部只断连接、不清这个开关，于是
        「开关是开的但什么都没跑」是最常见的一态，按 `not enabled` 取反第一下
        点出来就是「关」（提示「标记为停止，启动后生效」），要连点两下。
        """
        return not bool((slot or {}).get("running"))

    def _toggle_selected(self) -> None:
        cfg = self._single_or_warn()
        if cfg is None:
            return
        accts = self._accounts_of(cfg)
        if not accts:
            self._on_failed("这条策略没有可用账户（既没绑账户，也没有默认账户）")
            return
        slots = self._slots_of(cfg)
        # 🔴 判据是「引擎里真实在不在跑」，不是存盘的 `enabled` 开关。
        # 多账户时只要有一个账户在跑，这一下就是「停」——全部停下来才叫停
        live = [a for a in accts if (slots.get(a) or {}).get("running")]
        on = not live
        sid = str(cfg.sid)

        if on:
            cfg.enabled = True
            self._save_cfg(cfg)
            for key in accts:
                if not self.ctl.is_running(key):
                    # 这个账户的连接还没起来 —— **只把这一个策略拉起来**，不是「启动全部」
                    spec = self._spec(key)
                    if spec is None:
                        cfg.enabled = False
                        self._save_cfg(cfg)
                        self._on_failed(f"账户「{key}」已不存在，请重新绑定")
                        return
                    self.ctl.start(spec, [cfg], self._chase)
                    self._on_log(f"已单独启动「{cfg.title()}」（账户 {key}）")
                    self._warn_solo_start(key, cfg)
                    continue
                # 连接在跑，但引擎可能根本不认识这一条：从「策略研究」页勾「加入
                # 策略交易」、组合页「导入策略」、或 ChanAgent 送过来的策略，加进
                # 组合时只写了磁盘，没有推给正在跑的连接。这时 `enable()` 会一路走到
                # `PortfolioRunner.enable()` 的 `self._slots.get(sid) is None` 然后
                # **静默 return** —— 点了没反应，也不报错。
                #
                # 不能无条件 `add()`：它重建槽位，正在跑的那条会被抹掉 running 和挂单。
                if sid not in (self._slot_multi.get(sid) or {}):
                    self.ctl.add_or_update(key, cfg)
                if not self.ctl.enable(key, sid, True):
                    self._on_failed(f"账户「{key}」的连接没了，请再点一次")
                    return
                self._on_log(f"「{cfg.title()}」已启动（账户 {key}）")
        else:
            cfg.enabled = False
            self._save_cfg(cfg)
            for key in (live or accts):
                if not self.ctl.enable(key, sid, False):
                    self._on_failed(f"账户「{key}」没在运行，没有可停的策略")
                    return
                self._on_log(f"「{cfg.title()}」已停止（账户 {key}）"
                             + ("，这个账户上已经没有策略在跑，连接还留着"
                                if self._no_enabled_left(key) else ""))
        self._refresh_conn_label()
        self._fill_strategies()
        self._render()

    def _no_enabled_left(self, key: str) -> bool:
        """这个账户上还有没有别的已启用策略（跨所有工作区）。

        `_live_plan` 里已经只有启用的，所以直接看清单空不空 —— 刚停掉的那条
        此刻已经从池子里读成 `enabled=False` 了。
        """
        plan, _warn = self._live_plan()
        return not (plan.get(key) or [])

    def _warn_solo_start(self, key: str, cfg) -> None:
        """单独只启动一条时提醒一下：其它策略没进引擎，一键对齐会把它们的仓平掉。"""
        plan, _warn = self._live_plan()
        others = [c for c in plan.get(key, [])
                  if str(getattr(c, "sid", "")) != str(getattr(cfg, "sid", ""))]
        if not others:
            return
        self._on_log(
            f"注意：账户「{key}」上还有 {len(others)} 条已启用的策略没进引擎，"
            "它们的持仓会被「一键对齐」当成无主仓位。要全部跑起来点「启动全部」。")

    def _refresh_conn_label(self) -> None:
        """按真实连接状态重刷顶部状态灯 —— 单条启停之后也要跟着变。"""
        keys = self.ctl.keys()
        if not keys:
            self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
            self.lbl_conn.setText("已停止")
            self.lbl_conn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return
        n = sum(1 for _a, cs in self._live_plan()[0].items() if _a in keys
                for c in cs)
        self.lbl_dot.setStyleSheet(f"color: {C_OK}; font-size: 15px;")
        self.lbl_conn.setText(f"运行中 · {len(keys)} 个账户 / {n} 条策略")
        self.lbl_conn.setStyleSheet(f"color: {C_OK}; font-size: 11px;")
        # 只单独启动过一条时，「启动全部」还要能用 —— 它会把剩下的补齐
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(True)

    # ---------------------------------------------------------- 增删改

    def _add_from_library(self) -> None:
        # 内置模板也算「库里的东西」：新用户的库是空的，只给「我的」等于
        # 让他在这一步无路可走
        lib = SStore.library_view()
        if not lib:
            self._on_failed("策略库是空的。到「策略研究」页点「存入策略库」")
            return
        dlg = LibraryDialog(lib, self)
        if dlg.exec() != QDialog.Accepted:
            return
        sid = dlg.sid()
        src = next((c for c in lib if c.sid == sid), None)
        if src is None:
            return
        item = SStore.add_to_portfolio(src)
        item.account = self.cb_acct.currentData() or ""
        SStore.update_portfolio(item)
        # 加进**当前工作区**，本金按工作区的等分给（总资金 ÷ 人数），不照抄
        # 模板里那份 —— 模板的本金是「单独跑这条策略时」的规模，和工作区无关
        total = self._ws.live_total(self._pool) if self._ws else 0.0
        n = len(self._ws.members) + 1 if self._ws else 1
        self._ws.add(item.sid, (total / n) if total > 0 else item.principal)
        self._save_ws()
        self.reload()
        # 连接在跑就让引擎先接上（默认不开仓，`enabled` 仍是 False）。不推的话
        # 状态列显示「未连接」，用户点「启动 / 停止」还得靠 `_toggle_selected`
        # 补挂一次 —— 能补上，但少一个来回总是好的。
        if self.ctl.running:
            for acct in self._accounts_of(item):
                self.ctl.add_or_update(acct, item)
        self._on_log(f"已加入工作区「{self._ws.name}」：{item.title()}")

    def _edit_selected(self) -> None:
        cfg = self._single_or_warn()
        if cfg is None:
            return
        dlg = EditDialog(cfg, [s.name for s in self._accounts], self,
                         universe=self._universe_of(cfg))
        if dlg.exec() != QDialog.Accepted:
            return
        old_acct = self._account_of(cfg)
        old_syms = list(cfg.symbols or [])
        dlg.apply_to(cfg)
        self._save_cfg(cfg)
        self._fill_strategies()
        self._render()          # 重建表格会把理论/实际重置成「—」，立刻补回来
        if self.ctl.running and self._account_of(cfg) != old_acct:
            self._on_failed("账户已改。运行中的连接不会自动迁移，请停止后重新启动")
        else:
            if self.ctl.running:
                # 手数 / 参数 / 品种改了都要让引擎立刻知道：`update` 会按新的
                # 品种清单重建槽位（多品种就是 N 个子槽位），不然要等重启
                self.ctl.update_cfg(self._account_of(cfg), cfg)
                if list(cfg.symbols or []) != old_syms:
                    self._on_log(f"{cfg.title()}：品种清单已改，引擎正在重建订阅")
            n = len(cfg.symbols or [])
            self._on_log(f"{cfg.title()}：{cfg.size_text()}"
                         + (f"｜{n} 个品种等权" if n >= 2 and cfg.family != "cross"
                            else ""))

    def accept(self, cfg) -> str:
        """从「ChanAgent」页送来的策略：加进组合，正在跑就直接挂上去。

        返回一句给人看的结果，调用方拿去显示。**不自动开新连接** —— 连接是用户
        在「启动全部」时明确点出来的，一个对话不该顺手把账户连上。
        """
        item = SStore.add_to_portfolio(cfg)
        item.account = self.cb_acct.currentData() or ""
        item.enabled = True
        SStore.update_portfolio(item)
        if self._ws is not None:
            total = self._ws.live_total(self._pool)
            n = len(self._ws.members) + 1
            self._ws.add(item.sid, (total / n) if total > 0 else item.principal)
            self._save_ws()
        self.reload()
        if not self.ctl.running:
            msg = f"已加入组合：{item.title()}。点「启动全部」开始交易"
        else:
            key = self._account_of(item)
            if self.ctl.add(key, item) and self.ctl.enable(key, item.sid, True):
                msg = f"已加入组合并启动：{item.title()}"
            else:
                msg = f"已加入组合，但账户「{key}」的连接不在运行，启动后才会交易"
        self._on_log(msg)
        return msg

    def _del_selected(self) -> None:
        cfg = self._single_or_warn()
        if cfg is None:
            return
        others = [w.name for w in self._wss
                  if w is not self._ws and str(cfg.sid) in w.sids()]
        msg = f"把「{cfg.title()}」从工作区「{self._ws.name}」里拿掉？\n策略库里的模板不受影响。"
        if others:
            msg += ("\n\n它还在这 %d 个工作区里：%s\n那些分页不受影响。"
                    % (len(others), "、".join(others)))
        else:
            msg += "\n没有别的分页用它，策略池里那份也一并清掉。"
        if QMessageBox.question(self, "删除策略", msg) != QMessageBox.Yes:
            return
        for acct in self._accounts_of(cfg):
            if self.ctl.running:
                self.ctl.remove(acct, cfg.sid)
        self._ws.drop(cfg.sid)
        if not others:
            SStore.remove_from_portfolio(cfg.sid)
        self._save_ws()
        self.reload()
        self._on_log(f"已从「{self._ws.name}」移除：{cfg.title()}")

    # ---------------------------------------------------------- 对齐

    def _align(self) -> None:
        """把账户仓位摆成和策略列表完全一致。**会平掉没有策略在管的仓位**。"""
        if not self.ctl.running:
            self._on_failed("还没启动，无法对齐")
            return
        if any(int(snap.get("aligning") or 0) for snap in self._snaps.values()):
            # 上一轮还在路上。仓位没变不等于没对齐 —— 单子还在排队 / 追价。
            # 这时候再发一轮就是「平了又开、开了又平」。
            self._on_failed("上一轮对齐还在执行（等成交或撤单），稍后再点")
            return
        blind = [k for k, v in self._snaps.items() if not v.get("pos_ready", True)]
        if blind:
            self._on_failed("、".join(blind) + " 的账户持仓还没同步，等几秒再点"
                            "（现在对齐会拿「账户是空的」去算差额，把仓位重开一遍）")
            return
        rows = [r for snap in self._snaps.values() for r in (snap.get("mismatch") or [])]
        if not rows:
            self._on_log("账户仓位与策略理论持仓已经一致，不用对齐")
            return

        managed = [r for r in rows if r.get("managed")]
        orphans = [r for r in rows if not r.get("managed")]

        # 平别人的仓是真金白银，不能点一下就干。把「动哪些、动多少」摆出来，
        # 用户点确认才发单
        lines = []
        if managed:
            lines.append(f"按策略理论持仓调整 {len(managed)} 个合约：")
            for r in managed[:10]:
                lines.append(f"　{r['symbol']}　{_pos_text(r['actual'], r)} → "
                             f"{r['target']}（{_delta_text(r)}）")
            if len(managed) > 10:
                lines.append(f"　…… 还有 {len(managed) - 10} 个")
        if orphans:
            lines.append("")
            lines.append(f"平掉 {len(orphans)} 个没有策略在管的合约：")
            for r in orphans[:10]:
                lines.append(f"　{r['symbol']}　持有 {_pos_text(r['actual'], r)} "
                             f"→ 0（{_delta_text(r)}）")
            if len(orphans) > 10:
                lines.append(f"　…… 还有 {len(orphans) - 10} 个")
            lines.append("")
            lines.append("这些仓位多半是策略停掉之后留下的，或者手工开的。"
                         "平掉之后无法撤销。")
        n_lock = sum(1 for r in rows if r.get("locked"))
        if n_lock:
            lines.append("")
            lines.append(f"其中 {n_lock} 个是锁仓（多空双开）：净额是 0，但两条腿都占着"
                         "保证金，会拆成「卖平多 + 买平空」两笔一起发。")

        ans = QMessageBox.question(
            self, "一键对齐", "\n".join(lines) + "\n\n继续下发委托？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return

        n = 0
        for key, snap in self._snaps.items():
            if snap.get("mismatch"):
                self.ctl.align(key)
                n += len(snap["mismatch"])
        self._on_log(f"已下发对齐委托，覆盖 {n} 个合约；剩下交给追单，成交前不用再点")

    # ---------------------------------------------------------- 追单设置

    def _load_chase_widgets(self) -> None:
        c = self._chase
        for w in (self.chk_chase, self.chk_market):
            w.blockSignals(True)
        for w in (self.sp_wait, self.sp_retry, self.sp_ticks):
            w.blockSignals(True)
        self.chk_chase.setChecked(c.enabled)
        self.chk_market.setChecked(c.use_market)
        self.sp_wait.setValue(max(1, _int(c.wait_sec) or 5))
        self.sp_retry.setValue(max(0, _int(c.retries)))
        self.sp_ticks.setValue(max(0, _int(c.ticks)))
        for w in (self.chk_chase, self.chk_market,
                  self.sp_wait, self.sp_retry, self.sp_ticks):
            w.blockSignals(False)

    def _save_chase(self, *_a) -> None:
        cfg = SStore.ChaseConfig(
            enabled=self.chk_chase.isChecked(),
            wait_sec=self.sp_wait.value(),
            retries=self.sp_retry.value(),
            ticks=self.sp_ticks.value(),
            use_market=self.chk_market.isChecked(),
        )
        self._chase = cfg
        SStore.save_chase(cfg)
        self.ctl.set_chase(cfg)
        self._chase_hint(cfg)

    def _chase_hint(self, cfg) -> None:
        """把设置翻成一句人话。数字好填，但「到底会怎么追」不好脑补。"""
        if not cfg.enabled:
            text = "已关闭：委托不撤不追"
        elif cfg.retries <= 0:
            text = f"等 {cfg.wait_sec} 秒不成交就撤单，不重报"
        else:
            tail = "，最后一次用市价" if cfg.use_market else ""
            text = (
                f"等 {cfg.wait_sec} 秒不成交就撤单重报，"
                f"最多 {cfg.retries} 次，每次让 {cfg.ticks} 跳{tail}"
            )
        self.lbl_chase.setText(text)

    # ---------------------------------------------------------- 消息

    def _on_log(self, msg: str) -> None:
        self.lbl_msg.setText(msg)
        self.lbl_msg.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")

    # ---------------------------------------------------------- 补数据
    def _supplement_data(self) -> None:
        """就地补一次行情，进度写在这一页的日志条上。"""
        if self._data_thread is not None:
            self._on_log("补数据还在跑，等它跑完")
            return
        if QMessageBox.question(
                self, "补数据",
                "补期货主力日线 + 展期 / 期权衍生数据（近一年）。\n"
                "只补本地缺的部分，可能要几十秒到几分钟。继续？",
        ) != QMessageBox.Yes:
            return
        self.btn_data.setEnabled(False)
        self._on_log("开始补数据…")
        thread = QThread(self)
        worker = DataWorker("1y")
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._on_log)
        worker.done.connect(self._on_data_done)
        self._data_thread, self._data_worker = thread, worker
        thread.start()

    def _on_data_done(self, ok: bool, msg: str) -> None:
        self._on_log(("✓ " if ok else "✗ ") + msg)
        thread, worker = self._data_thread, self._data_worker
        self._data_thread, self._data_worker = None, None
        if thread is not None:
            thread.quit()
            thread.wait(3000)
        self.btn_data.setEnabled(True)
        if ok:
            # 主窗口要把 DataStore 缓存丢掉 —— 否则这一页还在画补之前那份
            self.data_updated.emit()

    def _train_selected(self) -> None:
        """离线训练选中的机器学习策略，把模型存成 .pkl。

        实盘启动时会自动加载这些产物 —— 走「只预测」快路径，每根 bar 几秒，
        不再重训。没有产物也能跑，只是慢到追不上 bar 周期。
        """
        if self._train_thread is not None:
            self._on_log("训练还在跑，等它跑完")
            return
        cfgs = self._selected_cfgs()
        if not cfgs:
            self._on_log("先在下面选中一条策略")
            return
        cfg = cfgs[0]
        if (cfg.family or "") != "factor" or (cfg.factor_mode or "") != "ml":
            self._on_log("只有机器学习策略需要训练模型（当前这条不是）")
            return

        # 品种清单：多品种策略训全部，单品种策略就训它自己
        syms = []
        if len(cfg.multi_symbols or []) >= 2:
            for k in cfg.multi_symbols:
                sym, ex = _split_key(k, cfg.market)
                syms.append((sym, ex))
        else:
            syms.append((cfg.symbol, cfg.exchange))

        n = len(syms)
        est = n * 40
        if QMessageBox.question(
                self, "训练模型",
                f"训练「{cfg.name or cfg.sid}」的机器学习模型。\n\n"
                f"品种 {n} 个，{len(cfg.factor_keys or [])} 个因子。\n"
                f"每个品种大约 30~40 秒，全部大概 {est // 60} 分 {est % 60} 秒。\n"
                f"训完存成 .pkl，实盘启动时直接用（只预测，不重训）。\n\n继续？",
        ) != QMessageBox.Yes:
            return

        self.btn_train.setEnabled(False)
        self._on_log(f"开始训练 {n} 个品种…")
        thread = QThread(self)
        worker = TrainWorker(cfg, syms)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._on_log)
        worker.done.connect(self._on_train_done)
        self._train_thread, self._train_worker = thread, worker
        thread.start()

    def _on_train_done(self, ok: bool, msg: str) -> None:
        self._on_log(("✓ " if ok else "✗ ") + msg)
        thread, worker = self._train_thread, self._train_worker
        self._train_thread, self._train_worker = None, None
        if thread is not None:
            thread.quit()
            thread.wait(3000)
        self.btn_train.setEnabled(True)

    def _on_failed(self, msg: str) -> None:
        self.lbl_msg.setText(msg)
        self.lbl_msg.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")

    # ---------------------------------------------------------- 生命周期

    def shutdown(self) -> None:
        """窗口关闭时调，等线程真的退出再返回，避免解释器退出时崩在 TqApi 上。"""
        self.ctl.stop(wait=True)
        if self._data_thread is not None:
            # 补数据是 HTTP + 写 CSV，杀掉线程会留半截文件。让它在后台跑完
            # 不现实（用户已经关窗了），所以只等一小会儿，超时就放弃 ——
            # 下次补的时候增量会把它接上
            self._data_thread.quit()
            self._data_thread.wait(3000)
        if self._train_thread is not None:
            # 训练同理：杀掉会留半截 .pkl。不过训练产物是**先写临时文件再改名**
            # 的，半截的那个不会顶掉好的那份
            self._train_thread.quit()
            self._train_thread.wait(3000)
        if self._view_thread is not None:
            # 回放是纯计算，杀掉不留垃圾。但线程还活着时窗口析构，Qt 会报
            # "QThread: Destroyed while thread is still running" 然后进程崩在
            # 退出码上 —— 和 v2.6.3 那次自检翻车同一个坑
            self._view_thread.quit()
            self._view_thread.wait(3000)
        if getattr(self, "_ms_thread", None) is not None:
            # 量占用同样是纯计算，同一条理由
            self._ms_thread.quit()
            self._ms_thread.wait(3000)
