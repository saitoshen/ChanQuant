"""数据管理页：两条补齐通道，各管各的周期。

- **Tushare**：日线。美股是 20 年后复权，期货是主力连续 + 复权因子；期货还能
  顺带补真实合约（期限结构）和期权（隐含波动率）。
- **天勤 TqSdk**：分钟线 1h / 5m。只对国内期货有意义 —— 美股没有这条通道，
  所以当前市场是美股时整块禁用。

两条路都是「先看本地缺哪段、只补缺口」，重复点是安全的；区别只在于数据源、
限速和耗时。

## 为什么周期是入口而不是数据源的附庸

用户脑子里想的是「我要 5 分钟数据」，不是「我要用天勤」。所以周期放在最上面：
拖到 1h/5m 就把数据源切到天勤，拖回日线再切回 Tushare。两个源各有主场（日线
Tushare 有 20 年和复权因子，分钟线只有天勤有），所以两个方向都切。

只做默认值、不做强制：选完日线仍可以手动把源改成天勤 —— Tushare 额度用光时
那条路还得走，锁死就把唯一的备选也堵上了。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ttrade import updater

SRC_TUSHARE = "tushare"
SRC_TQ = "tq"

TQ_PERIODS = [("1h", "1 小时"), ("5m", "5 分钟"), ("1d", "日线")]
TQ_YEARS = [(1, "近 1 年"), (3, "近 3 年"), (5, "近 5 年"), (10, "近 10 年")]


# ------------------------------------------------------------------ 线程


class UpdateWorker(QObject):
    """在后台线程里跑 Tushare 更新，避免界面假死。

    下载是纯网络 IO，几十个标的顺序请求，放在主线程会让窗口完全卡住
    （连「取消」都点不动）。这里用 QThread + 信号把进度送回主线程。
    """

    progress = Signal(int, int, str)   # done, total, message
    done = Signal(object)              # UpdateResult
    failed = Signal(str)

    def __init__(self, market: str, data_dir: str, range_key: str,
                 source: str = "auto", min_interval: float = 60.0,
                 aux_contracts: bool = False, aux_options: bool = False):
        super().__init__()
        self.market = market
        self.data_dir = data_dir
        self.range_key = range_key
        self.source = source
        self.min_interval = min_interval
        self.aux_contracts = aux_contracts
        self.aux_options = aux_options
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        cb = lambda d, t, m: self.progress.emit(d, t, m)  # noqa: E731
        try:
            if self.market == "us":
                res = updater.update_us(
                    self.data_dir, self.range_key, progress=cb,
                    cancel=lambda: self._cancel,
                    source=self.source, min_interval=self.min_interval,
                )
            else:
                res = updater.update_futures(
                    self.data_dir, self.range_key, progress=cb,
                    cancel=lambda: self._cancel,
                )
                if self.aux_contracts or self.aux_options:
                    # 衍生数据放在主力连续之后跑：即使它失败，主行情也已经落盘了
                    try:
                        res.aux = updater.update_aux(
                            range_key=self.range_key,
                            want_contracts=self.aux_contracts,
                            want_options=self.aux_options,
                            progress=cb,
                            cancel=lambda: self._cancel,
                        )
                    except Exception as exc:  # noqa: BLE001
                        res.aux = None
                        res.aux_error = str(exc)[:200]
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(res)


class TqWorker(QObject):
    """在后台线程里跑天勤补分钟线。

    一次会话补全部品种：每个品种单独 `TqApi()` 等于登录 78 次，光是握手就要
    十几分钟。断线不在这里重试 —— 增量是按库里的最后时间算的，用户再点一次
    就是断点续传，比在这里猜重试策略可靠。
    """

    progress = Signal(int, int, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, period: str, years: int,
                 pairs: list[tuple[str, str]] | None = None) -> None:
        super().__init__()
        self.period = period
        self.years = years
        self.pairs = pairs
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from ttrade import tqfetch as TQ
            from ttrade import tqstore as S

            pairs = self.pairs or list(S.switch_symbols())
            if not pairs:
                self.failed.emit("本地还没有品种池，先用 Tushare 补一遍日线")
                return

            self.progress.emit(0, 0, "连接天勤…")
            sess = TQ.Session()
            try:
                api = sess.get()
                self.progress.emit(0, 0, "检查主力切换表…")
                # 补切换表要给每个品种请求几十次合约日线，78 个品种能跑半小时。
                # 不回调的话进度条会一直停在「检查主力切换表…」，看着像卡死了
                n_switch = TQ.ensure_switch(
                    api, pairs,
                    on_row=lambda key, n: self.progress.emit(
                        0, 0, f"切换表 {key}（{n} 次换月）"
                    ),
                )
                if self._cancel:
                    self.done.emit("已取消（切换表已更新，行情未补）")
                    return

                done = 0
                total = len(pairs)
                period_label = dict(TQ_PERIODS).get(self.period, self.period)

                def on_row(key: str, rows: int, first) -> None:
                    nonlocal done
                    done += 1
                    tail = f"+{rows} 行（{first:%Y-%m-%d} 起）" if rows else "无新数据"
                    self.progress.emit(done, total, f"{period_label} {done}/{total}　{key}　{tail}")

                got = TQ.write_main_many(
                    api, pairs, self.period, self.years,
                    on_row=on_row, cancel=lambda: self._cancel,
                )
            finally:
                sess.close()

            moved, touched = TQ.refresh_all(pairs)
            rows = sum(int(v or 0) for v in got.values())
            head = "已取消" if self._cancel else "完成"
            self.done.emit(
                f"天勤 {period_label} {head}：{len(pairs)} 个品种，本次写入 {rows:,} 行；"
                f"新补切换表 {n_switch} 个，校正换月日 {moved} 处，刷新复权 {touched} 个品种周期"
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ 页面


class DataPage(QWidget):
    """数据管理。与「策略研究」等四个页面平级。"""

    data_changed = Signal(str)     # 市场键：补完之后主窗口要重读标的池

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.market = "us"
        self.market_label = "美股"
        self.market_dir = ""
        self._upd_worker: UpdateWorker | None = None
        self._upd_thread: QThread | None = None
        self._tq_worker: TqWorker | None = None
        self._tq_thread: QThread | None = None
        self._build()

    # ---- 外部 ----

    def set_market(self, market: str, label: str, data_dir: str) -> None:
        """主窗口切市场时调。天勤只对期货有意义，美股下整块禁用。"""
        self.market = market
        self.market_label = label
        self.market_dir = data_dir
        self.lbl_market.setText(f"当前市场：{label}")
        self._sync_controls()

    def busy(self) -> bool:
        return self._upd_thread is not None or self._tq_thread is not None

    # ---- 构建 ----

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionTitle")
        return lbl

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(10)

        head = QLabel("数据管理")
        head.setObjectName("AppTitle")
        outer.addWidget(head)
        self.lbl_market = QLabel("当前市场：美股")
        self.lbl_market.setObjectName("AppSub")
        outer.addWidget(self.lbl_market)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_form(), 0)
        body.addWidget(self._build_log(), 1)
        outer.addLayout(body, 1)

    def _build_form(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        card.setFixedWidth(320)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(7)

        lay.addWidget(self._section("数据源"))
        self.cb_source = QComboBox()
        self.cb_source.addItem("Tushare · 日线", SRC_TUSHARE)
        self.cb_source.addItem("天勤 TqSdk · 分钟线", SRC_TQ)
        self.cb_source.currentIndexChanged.connect(self._sync_controls)
        lay.addWidget(self.cb_source)

        # 标题单独存一份：控件藏起来时标题必须跟着藏，否则会留下一行
        # 「周期」下面空着 —— 看着像加载失败
        self.lbl_sec_period = self._section("周期")
        lay.addWidget(self.lbl_sec_period)
        self.cb_period = QComboBox()
        for key, label in TQ_PERIODS:
            self.cb_period.addItem(label, key)
        self.cb_period.setToolTip(
            "日线走 Tushare（20 年、含复权因子）；1 小时 / 5 分钟走天勤（10 年）。"
        )
        self.cb_period.currentIndexChanged.connect(self._on_period_changed)
        lay.addWidget(self.cb_period)

        # Tushare 通道叫「区间」（近 1 个月…），天勤通道叫「年限」（近 1 年…），
        # 同一个位置放一个标题，写「区间 / 年限」两边都不算错
        lay.addWidget(self._section("区间 / 年限"))
        self.cb_upd_range = QComboBox()
        for key, label, _days in updater.RANGES:
            self.cb_upd_range.addItem(label, key)
        self.cb_upd_range.setToolTip(
            "只补这一段，不重下 20 年。\n"
            "「全部 20 年」会对每个标的整段重下，耗时较长。"
        )
        lay.addWidget(self.cb_upd_range)

        self.cb_tq_years = QComboBox()
        for years, label in TQ_YEARS:
            self.cb_tq_years.addItem(label, years)
        self.cb_tq_years.setCurrentIndex(3)
        self.cb_tq_years.setToolTip(
            "天勤 5 分钟线最早约 2016 年，10 年基本就是全部历史。\n"
            "库里已有的部分不会重下。"
        )
        lay.addWidget(self.cb_tq_years)

        self.cb_upd_source = QComboBox()
        for key, label in updater.SOURCE_LABELS.items():
            self.cb_upd_source.addItem(label, key)
        self.cb_upd_source.setToolTip(
            "仅对美股生效。东财给的是后复权价，与本地历史同口径；\n"
            "Tushare us_daily 是未复权且额度极紧，只作兜底。"
        )
        lay.addWidget(self.cb_upd_source)

        # 期货只有一个数据源，用一行静态说明顶替下拉框。
        # 下拉框里那些「东方财富」「自动」选项对期货毫无意义，
        # 摆在那里只会让人以为可以选，所以直接不显示。
        self.lbl_upd_src = QLabel("数据源：Tushare（日线与复权因子同步更新）")
        self.lbl_upd_src.setObjectName("Hint")
        self.lbl_upd_src.setWordWrap(True)
        self.lbl_upd_src.setVisible(False)
        lay.addWidget(self.lbl_upd_src)

        self.cb_upd_rate = QComboBox()
        for sec, label in [(60.0, "Tushare 限速：60 秒/次"),
                           (30.0, "Tushare 限速：30 秒/次"),
                           (10.0, "Tushare 限速：10 秒/次"),
                           (0.0, "Tushare 限速：不限")]:
            self.cb_upd_rate.addItem(label, sec)
        self.cb_upd_rate.setToolTip("只约束 Tushare 美股接口；东财与期货不受影响")
        lay.addWidget(self.cb_upd_rate)

        # 合约 / 期权衍生数据。这两项要按「每个交易日一次全市场横截面」取数，
        # 比主力连续的增量更新慢一个量级，所以做成显式勾选，默认不跟。
        self.lbl_upd_aux = QLabel("衍生数据（展期 / 期权因子用）")
        self.lbl_upd_aux.setObjectName("Hint")
        lay.addWidget(self.lbl_upd_aux)

        self.chk_upd_contracts = QCheckBox("一并更新真实合约（期限结构）")
        self.chk_upd_contracts.setToolTip(
            "下载全部真实合约的日线，生成近/次近/第三近合约的期限结构。\n"
            "展期收益率、展期年化、展期曲率、展期变化依赖它。\n"
            "每个交易日一次全市场请求，比主力连续慢很多（20 年全量约 40 分钟）。"
        )
        lay.addWidget(self.chk_upd_contracts)

        self.chk_upd_options = QCheckBox("一并更新期权（隐含波动率）")
        self.chk_upd_options.setToolTip(
            "下载商品期权日线，反解平值 IV、偏度、期限结构与看跌看涨持仓比。\n"
            "只保留聚合结果（原始期权明细约 1.5GB，默认不落盘）。\n"
            "商品期权 2017 年才上市，更早的日期会自动跳过。"
        )
        lay.addWidget(self.chk_upd_options)

        self.lbl_tq_hint = QLabel(
            "天勤需要行情账号：项目根目录或用户目录下的 tq_account.txt（两行：用户名、密码）。\n"
            "19:00–19:30 是天勤运维窗口，这段时间会自动等待。"
        )
        self.lbl_tq_hint.setObjectName("Hint")
        self.lbl_tq_hint.setWordWrap(True)
        self.lbl_tq_hint.setVisible(False)
        lay.addWidget(self.lbl_tq_hint)

        lay.addSpacing(4)
        self.btn_update = QPushButton("开始补充")
        self.btn_update.clicked.connect(self._on_clicked)
        lay.addWidget(self.btn_update)

        self.prog_update = QProgressBar()
        self.prog_update.setTextVisible(False)
        self.prog_update.setFixedHeight(6)
        self.prog_update.setVisible(False)
        lay.addWidget(self.prog_update)

        self.lbl_upd = QLabel("")
        self.lbl_upd.setObjectName("Hint")
        self.lbl_upd.setWordWrap(True)
        self.lbl_upd.setVisible(False)
        lay.addWidget(self.lbl_upd)

        lay.addStretch(1)
        self._sync_controls()
        return card

    def _build_log(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        lay.addWidget(self._section("日志"))
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size: 11px;"
            "background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;"
        )
        lay.addWidget(self.txt_log, 1)
        return card

    # ---- 联动 ----

    def _source(self) -> str:
        return self.cb_source.currentData() or SRC_TUSHARE

    def _on_period_changed(self) -> None:
        """周期是入口：分钟级 → 天勤，日线 → Tushare。

        双向都切，因为这是两个源各自的主场（日线 Tushare 有 20 年和复权因子，
        分钟线只有天勤有）。用户仍可以选完日线再手动把源改成天勤 —— 那条路
        Tushare 额度用光时还得走，所以这里只做默认值，不做强制。
        """
        want = SRC_TQ if self.cb_period.currentData() in ("1h", "5m") else SRC_TUSHARE
        i = self.cb_source.findData(want)
        if i >= 0 and self.cb_source.currentIndex() != i:
            self.cb_source.setCurrentIndex(i)
            return
        self._sync_controls()

    def _sync_controls(self, *_args) -> None:
        """按数据源 + 市场切换可见性。"""
        if not hasattr(self, "cb_upd_range"):
            return
        tq = self._source() == SRC_TQ
        is_us = self.market == "us"

        # 天勤通道只有国内期货。美股下选到它就切回去，不把下拉项置灰 ——
        # 置灰要靠 `QComboBox.model().item()`，那是对 QStandardItemModel 的
        # 实现细节下手，换掉默认模型就崩
        if tq and is_us:
            self._log("天勤通道只有国内期货，当前是美股，已切回 Tushare。")
            i = self.cb_source.findData(SRC_TUSHARE)
            if i >= 0:
                self.cb_source.blockSignals(True)
                self.cb_source.setCurrentIndex(i)
                self.cb_source.blockSignals(False)
            tq = False

        # 周期是入口，期货下始终显示 —— 从「日线」拖到「5 分钟」就应该自动
        # 切到天勤。美股没有分钟线，整块藏起来并锁回日线。
        self.lbl_sec_period.setVisible(not is_us)
        self.cb_period.setVisible(not is_us)
        if is_us:
            i = self.cb_period.findData("1d")
            if i >= 0 and self.cb_period.currentIndex() != i:
                self.cb_period.blockSignals(True)
                self.cb_period.setCurrentIndex(i)
                self.cb_period.blockSignals(False)

        self.cb_upd_range.setVisible(not tq)
        self.cb_tq_years.setVisible(tq)
        self.cb_upd_source.setVisible(not tq and is_us)
        self.cb_upd_rate.setVisible(not tq and is_us)
        self.lbl_upd_src.setVisible(not tq and not is_us)
        for w in (self.lbl_upd_aux, self.chk_upd_contracts, self.chk_upd_options):
            w.setVisible(not tq and not is_us)
        self.lbl_tq_hint.setVisible(tq)

        if self.busy():
            return
        self.btn_update.setEnabled(True)

    # ---- 执行 ----

    def _log(self, text: str) -> None:
        self.txt_log.appendPlainText(text)

    def _on_clicked(self) -> None:
        if self.busy():
            self._cancel()
            return
        if self._source() == SRC_TQ:
            self._start_tq()
        else:
            self._start_tushare()

    def _cancel(self) -> None:
        if self._upd_worker is not None:
            self._upd_worker.cancel()
        if self._tq_worker is not None:
            self._tq_worker.cancel()
        self.btn_update.setEnabled(False)
        self.btn_update.setText("取消中")
        self.lbl_upd.setText("正在取消，等当前这一批跑完…")
        self._log("… 已请求取消")

    def _start_tushare(self) -> None:
        mk = self.market
        key = self.cb_upd_range.currentData()
        # 期货没有第二个数据源，直接钉死 Tushare
        source = (
            self.cb_upd_source.currentData() if mk == "us" else updater.SOURCE_TUSHARE
        )
        rate = float(self.cb_upd_rate.currentData() or 0.0)
        aux_c = bool(mk == "cn" and self.chk_upd_contracts.isChecked())
        aux_o = bool(mk == "cn" and self.chk_upd_options.isChecked())

        self._upd_worker = UpdateWorker(
            mk, self.market_dir, key, source, rate,
            aux_contracts=aux_c, aux_options=aux_o,
        )
        self._upd_thread = QThread(self)
        self._upd_worker.moveToThread(self._upd_thread)
        self._upd_thread.started.connect(self._upd_worker.run)
        self._upd_worker.progress.connect(self._on_progress)
        self._upd_worker.done.connect(self._on_update_done)
        self._upd_worker.failed.connect(self._on_update_failed)
        # `started` 连普通函数，不显式 quit 就永远不发 finished（同 Agent 页的坑）
        self._upd_worker.done.connect(self._upd_thread.quit)
        self._upd_worker.failed.connect(self._upd_thread.quit)
        self._upd_thread.finished.connect(self._on_thread_done)

        self.prog_update.setRange(0, 0)      # 先转圈，等第一个进度再定总量
        self.prog_update.setVisible(True)
        self.lbl_upd.setVisible(True)
        self.lbl_upd.setText(f"准备更新{self.market_label}…")
        self._log(f"—— Tushare 更新{self.market_label}（{updater.RANGE_LABELS.get(key, key)}）")
        self.btn_update.setText("取消")
        self.btn_update.setEnabled(True)
        self._upd_thread.start()

    def _start_tq(self) -> None:
        period = self.cb_period.currentData() or "1h"
        years = int(self.cb_tq_years.currentData() or 10)
        self._tq_worker = TqWorker(period, years)
        self._tq_thread = QThread(self)
        self._tq_worker.moveToThread(self._tq_thread)
        self._tq_thread.started.connect(self._tq_worker.run)
        self._tq_worker.progress.connect(self._on_progress)
        self._tq_worker.done.connect(self._on_tq_done)
        self._tq_worker.failed.connect(self._on_update_failed)
        self._tq_worker.done.connect(self._tq_thread.quit)
        self._tq_worker.failed.connect(self._tq_thread.quit)
        self._tq_thread.finished.connect(self._on_thread_done)

        label = dict(TQ_PERIODS).get(period, period)
        self.prog_update.setRange(0, 0)
        self.prog_update.setVisible(True)
        self.lbl_upd.setVisible(True)
        self.lbl_upd.setText(f"准备从天勤补 {label}…")
        self._log(f"—— 天勤补齐 {label}（回溯 {years} 年）")
        self.btn_update.setText("取消")
        self.btn_update.setEnabled(True)
        self._tq_thread.start()

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.prog_update.setRange(0, total)
            self.prog_update.setValue(done)
        self.lbl_upd.setText(msg)
        # 每个品种一行太吵，只在有进展时写日志；调用方给的消息里已经带了计数
        if done > 0 or total == 0:
            self._log(msg)

    def _on_update_done(self, res) -> None:
        self.lbl_upd.setVisible(True)
        text = res.summary()
        aux = getattr(res, "aux", None)
        if aux is not None:
            text += f"\n衍生数据：{aux.summary()}"
        err = getattr(res, "aux_error", "")
        if err:
            text += f"\n衍生数据更新失败：{err}"
        self.lbl_upd.setText(text)
        self._log(text)

        self.data_changed.emit(res.market)

        problems = list(res.errors) + list(res.notes)
        if problems or err:
            more = f"\n… 另有 {len(problems) - 6} 条" if len(problems) > 6 else ""
            QMessageBox.information(
                self, "数据更新",
                text + ("\n\n" + "\n".join(problems[:6]) + more if problems else ""),
            )

    def _on_tq_done(self, text: str) -> None:
        self.lbl_upd.setVisible(True)
        self.lbl_upd.setText(text)
        self._log(text)
        self.data_changed.emit(self.market)

    def _on_update_failed(self, message: str) -> None:
        self.lbl_upd.setVisible(True)
        self.lbl_upd.setText("更新失败")
        self._log(f"！失败：{message}")
        QMessageBox.warning(self, "数据更新失败", message)

    def _on_thread_done(self) -> None:
        for attr in ("_upd_thread", "_tq_thread"):
            t = getattr(self, attr)
            if t is not None:
                t.deleteLater()
                setattr(self, attr, None)
        for attr in ("_upd_worker", "_tq_worker"):
            w = getattr(self, attr)
            if w is not None:
                w.deleteLater()
                setattr(self, attr, None)
        self.btn_update.setText("开始补充")
        self.btn_update.setEnabled(True)
        self.prog_update.setVisible(False)

    def shutdown(self) -> None:
        """关窗口时收尾：让后台线程别拖着进程。"""
        if self._upd_worker is not None:
            self._upd_worker.cancel()
        if self._tq_worker is not None:
            self._tq_worker.cancel()
        for attr in ("_upd_thread", "_tq_thread"):
            t = getattr(self, attr)
            if t is not None:
                t.quit()
                t.wait(3000)
