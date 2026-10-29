"""数据管理页：两条补齐通道，各管各的周期。

- **Tushare**：日线。美股是 20 年后复权，期货是主力连续 + 复权因子；期货还能
  顺带补真实合约（期限结构）和期权（隐含波动率）。
- **天勤 TqSdk**：日线 / 1 小时 / 5 分钟。只对国内期货有意义 —— 美股没有这条
  通道，所以当前市场是美股时整块禁用。可以顺带下真实合约，并用它们构建展期
  因子要的期限结构（`data_tq/curve/`）。

两条路都是「先看本地缺哪段、只补缺口」，重复点是安全的；区别只在于数据源、
限速和耗时。

## 市场由本页自己选

以前这一页跟着「策略研究」页的市场走。补数据其实是独立动作 —— 常见的是
「研究页看着期货，顺手把美股日线补一下」，跟着走反而碍事。现在页顶有单选，
点过之后主窗口再切市场也不会把它改回去（`_user_picked`）。

## 周期不再兼作数据源开关

老版本里「日线」等于 Tushare、「分钟」等于天勤，所以周期下拉能顺带切源。
天勤也能下日线之后，一个「日线」对应两个源，再让周期去猜就会把用户想下的
那份悄悄换掉 —— 所以周期只对天勤那一路生效，源由下拉明确选。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ttrade import updater

SRC_TUSHARE = "tushare"
SRC_TQ = "tq"

TQ_PERIODS = [("1d", "日线"), ("1h", "1 小时"), ("5m", "5 分钟")]

# 天勤的回溯区间。用**天数**而不是年数：用户要的是「近 1 天 / 近 1 周」，年表达
# 不了。20 年那档等于「全部」—— 天勤日线能给到二十几年，5m 只有十年出头，
# 取不到的更早部分不报错，只是历史更短。
TQ_RANGES = [
    (1, "近 1 天"),
    (7, "近 1 周"),
    (30, "近 1 个月"),
    (92, "近 3 个月"),
    (183, "近半年"),
    (7300, "近 20 年"),
]


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
    """在后台线程里跑天勤补行情。

    一次会话补全部品种：每个品种单独 `TqApi()` 等于登录 78 次，光是握手就要
    十几分钟。断线不在这里重试 —— 增量是按库里的最后时间算的，用户再点一次
    就是断点续传，比在这里猜重试策略可靠。

    三件事按顺序做：主连 → 真实合约 → 期限结构（展期因子要用的那份）。后两步
    可以关掉：真实合约十年 5m 是 8.4 GB，只想跑主连的人没必要下。
    """

    progress = Signal(int, int, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, period: str, days: int,
                 pairs: list[tuple[str, str]] | None = None,
                 contracts: bool = True) -> None:
        super().__init__()
        self.period = period
        self.days = int(days or 7300)
        self.pairs = pairs
        self.contracts = contracts
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from ttrade import tqcurve
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
                    api, pairs, self.period, 0,
                    on_row=on_row, cancel=lambda: self._cancel, days=self.days,
                )

                # ---- 真实合约 + 期限结构
                n_contracts = 0
                bad: list[str] = []
                dead = 0
                if self.contracts and not self._cancel:
                    for i, (sym, ex) in enumerate(pairs, 1):
                        if self._cancel:
                            break
                        self.progress.emit(i, len(pairs),
                                           f"真实合约 {i}/{len(pairs)}　{sym}_{ex}…")
                        try:
                            r = TQ.write_contracts(api, sym, ex, self.period, 0,
                                                   days=self.days,
                                                   cancel=lambda: self._cancel)
                        except Exception as exc:  # noqa: BLE001
                            # 一个品种的合约失败不该把整轮打断 —— 主连已经落库了
                            bad.append(f"{sym}_{ex}：{type(exc).__name__}")
                            self.progress.emit(i, len(pairs),
                                               f"真实合约 {sym}_{ex} 失败：{str(exc)[:60]}")
                            continue
                        if r.get("cancelled"):
                            break     # 用户点了取消，后面的品种一个都别再起
                        # **不抛异常不等于全下到了**：卡死保护会把没下完的合约记成
                        # 「等待超时」放进 failed，然后正常返回；「查合约失败」更是
                        # 直接 early return。这里必须看返回值，否则用户看到「真实
                        # 合约 78 个品种」以为齐了，展期因子却是断的（CLI 那边一直
                        # 是看 failed 的）。
                        if r.get("status") in ("ok", "无数据"):
                            n_contracts += 1
                        else:
                            note = str(r.get("status") or "")[:40]
                            bad.append(f"{sym}_{ex}：{note}")
                            self.progress.emit(i, len(pairs),
                                               f"真实合约 {sym}_{ex} 未下全：{note}")
                        # 连着几个品种一行都没下到，多半是连接已经死了：重建会话再
                        # 继续。不重建的话剩下每个品种都要白等一轮看门狗（3 分钟）。
                        if r.get("rows"):
                            dead = 0
                        else:
                            dead += 1
                            if dead >= 3:
                                self.progress.emit(i, len(pairs), "连接似乎已断，重建会话…")
                                sess.reset()
                                api = sess.get()
                                dead = 0
            finally:
                sess.close()

            moved, touched = TQ.refresh_all(pairs)

            n_curve = 0
            if self.contracts:
                self.progress.emit(0, 0, "构建期限结构（展期因子）…")
                built = tqcurve.build_all(pairs, period=self.period)
                n_curve = sum(1 for v in built.values() if v)

            rows = sum(int(v or 0) for v in got.values())
            head = "已取消" if self._cancel else "完成"
            extra = ""
            if self.contracts:
                extra = (f"；真实合约 {n_contracts} 个品种，"
                         f"期限结构 {n_curve} 个品种")
                if bad:
                    shown = "、".join(bad[:4]) + ("…" if len(bad) > 4 else "")
                    extra += f"；{len(bad)} 个品种有合约没下全（{shown}），再点一次会续传"
            self.done.emit(
                f"天勤 {period_label} {head}：{len(pairs)} 个品种，本次写入 {rows:,} 行{extra}；"
                f"新补切换表 {n_switch} 个，校正换月日 {moved} 处，刷新复权 {touched} 个品种周期"
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ 页面


class DataPage(QWidget):
    """数据管理。与「策略研究」等四个页面平级。"""

    data_changed = Signal(str)     # 市场键：补完之后主窗口要重读标的池

    def __init__(self, markets: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        # 市场由**本页自己的单选**决定，不再跟着「策略研究」页走：补数据是独立
        # 动作，经常是「研究页看着期货、顺手把美股日线补一下」。`markets` 由主窗口
        # 传进来（label / dir），避免这一页反过来 import gui.app 造成循环。
        self._markets = dict(markets or {})
        self.market = "us"
        self.market_label = "美股"
        self.market_dir = ""
        self._user_picked = False
        self._upd_worker: UpdateWorker | None = None
        self._upd_thread: QThread | None = None
        self._tq_worker: TqWorker | None = None
        self._tq_thread: QThread | None = None
        self._build()

    # ---- 外部 ----

    def set_market(self, market: str, label: str, data_dir: str) -> None:
        """主窗口切市场时给的**默认值**。

        用户在本页点过单选之后就不再覆盖（`_user_picked`）—— 否则在数据页选了
        「美股」、回研究页点一下期货，这里的选项会被悄悄改回去。
        """
        if self._user_picked:
            return
        self._apply_market(market, label, data_dir)
        btn = self._market_btns.get(market)
        if btn is not None:
            btn.setChecked(True)

    def _apply_market(self, market: str, label: str, data_dir: str) -> None:
        self.market = market
        self.market_label = label
        self.market_dir = data_dir
        self.lbl_market.setText(f"当前市场：{label}")
        self._sync_controls()

    def _on_market_radio(self, market: str) -> None:
        info = self._markets.get(market) or {}
        self._user_picked = True
        self._apply_market(market, info.get("label", market), info.get("dir", ""))
        self._log(f"—— 市场切到{info.get('label', market)}")

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

        # 市场单选：本页自己选，和「策略研究」页解耦。补数据是独立动作，
        # 不该因为研究页停在期货就把美股日线补不了。
        row = QHBoxLayout()
        row.setSpacing(14)
        self._market_btns: dict[str, QRadioButton] = {}
        self._market_group = QButtonGroup(self)
        for key, label in (("us", "美股"), ("cn", "国内期货")):
            btn = QRadioButton(label)
            btn.setChecked(key == self.market)
            # 用 clicked 而不是 toggled：`setChecked()` 是程序调用，也会发 toggled，
            # 那样一构造就把 `_user_picked` 置真，主窗口给的默认市场再也设不进来
            btn.clicked.connect(lambda _=False, k=key: self._on_market_radio(k))
            self._market_group.addButton(btn)
            self._market_btns[key] = btn
            row.addWidget(btn)
        row.addStretch(1)
        outer.addLayout(row)

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
        self.cb_source.addItem("天勤 TqSdk · 日线 / 分钟线", SRC_TQ)
        self.cb_source.currentIndexChanged.connect(self._sync_controls)
        lay.addWidget(self.cb_source)

        # 周期只属于天勤通道：Tushare 那边只有日线，没有可选的。
        # 标题单独存一份：控件藏起来时标题必须跟着藏，否则会留下一行
        # 「周期」下面空着 —— 看着像加载失败
        self.lbl_sec_period = self._section("周期")
        lay.addWidget(self.lbl_sec_period)
        self.cb_period = QComboBox()
        for key, label in TQ_PERIODS:
            self.cb_period.addItem(label, key)
        self.cb_period.setToolTip(
            "天勤原生只有日线 / 1 小时 / 5 分钟三种 bar，15m/30m/2h/4h 是本地合并的，\n"
            "不用单独下。日线能追到二十几年，5 分钟最早约 2016 年。"
        )
        lay.addWidget(self.cb_period)

        # Tushare 通道叫「区间」，天勤通道叫「回溯区间」，同一个位置放一个标题
        lay.addWidget(self._section("区间 / 回溯"))
        self.cb_upd_range = QComboBox()
        for key, label, _days in updater.RANGES:
            self.cb_upd_range.addItem(label, key)
        self.cb_upd_range.setToolTip(
            "只补这一段，不重下 20 年。\n"
            "「全部 20 年」会对每个标的整段重下，耗时较长。"
        )
        lay.addWidget(self.cb_upd_range)

        self.cb_tq_range = QComboBox()
        for days, label in TQ_RANGES:
            self.cb_tq_range.addItem(label, days)
        self.cb_tq_range.setCurrentIndex(len(TQ_RANGES) - 1)
        self.cb_tq_range.setToolTip(
            "只补这一段，库里已有的部分不会重下。\n"
            "「近 20 年」对 5 分钟等于全部历史（天勤 5m 最早约 2016 年）。"
        )
        lay.addWidget(self.cb_tq_range)

        # 天勤的真实合约 + 期限结构。默认勾上：展期那十几个因子全靠它，
        # 不勾的话界面上那些因子永远是「无数据」，很容易被当成 bug。
        self.chk_tq_contracts = QCheckBox("一并下载真实合约，并构建展期因子")
        self.chk_tq_contracts.setChecked(True)
        self.chk_tq_contracts.setToolTip(
            "下载每个品种的全部真实合约，再用它们算近/次近/第三近的期限结构，\n"
            "写到 data_tq/curve/，展期收益率/年化/曲率/变化都读这份。\n\n"
            "体积：日线全品种约 300 MB；5 分钟十年约 8.4 GB，第一次会很慢。\n"
            "已经下过的合约会按最后时间增量跳过，重跑不会从头再来。"
        )
        lay.addWidget(self.chk_tq_contracts)

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

        # 周期 / 回溯区间 / 真实合约都属于天勤那一路；Tushare 只有日线，
        # 只显示它自己的「区间」。
        self.lbl_sec_period.setVisible(tq)
        self.cb_period.setVisible(tq)
        self.cb_upd_range.setVisible(not tq)
        self.cb_tq_range.setVisible(tq)
        self.chk_tq_contracts.setVisible(tq)
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
        days = int(self.cb_tq_range.currentData() or 7300)
        contracts = self.chk_tq_contracts.isChecked()
        self._tq_worker = TqWorker(period, days, contracts=contracts)
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
        span = self.cb_tq_range.currentText()
        self.prog_update.setRange(0, 0)
        self.prog_update.setVisible(True)
        self.lbl_upd.setVisible(True)
        self.lbl_upd.setText(f"准备从天勤补 {label}…")
        self._log(
            f"—— 天勤补齐 {label}（{span}）"
            + ("，含真实合约与展期因子" if contracts else "，只补主连")
        )
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
        problems = list(res.errors) + list(res.notes)
        aux = getattr(res, "aux", None)
        if aux is not None:
            text += f"\n衍生数据：{aux.summary()}"
            # 衍生数据是逐日 try/except 的，失败只记在它自己的 errors 里。不并进
            # problems 的话，用户只能从「N 天失败」四个字里猜是哪几天 —— 而那几天
            # 恰好就是期限结构断掉的地方。
            problems += list(getattr(aux, "errors", []) or [])
        err = getattr(res, "aux_error", "")
        if err:
            text += f"\n衍生数据更新失败：{err}"
        self.lbl_upd.setText(text)
        self._log(text)

        self.data_changed.emit(res.market)

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
        # 有品种没下全就弹一下：跑几小时的下载用户未必盯着 label 看，而缺的那几个
        # 合约正是展期因子断掉的地方
        if "没下全" in text:
            QMessageBox.information(self, "天勤补齐", text)

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
