"""「交易Agent」页。

和「策略研究」「账户透视和交易」「策略交易和组合」并列的第四个大页面。

    ┌─────────────┬──────────────────────────┬──────────────┐
    │ 因子库       │                          │              │
    │ 策略库       │      对话（大）           │  详情面板     │
    │ 研报库       │                          │  公式 / 代码  │
    ├─────────────┤  ──────────────────────  │  逻辑 / 图例  │
    │ 模型状态     │  [输入框]  [发送] [停止]  │              │
    └─────────────┴──────────────────────────┴──────────────┘

四个库共用同一个详情面板：左边选中什么，右边就显示什么。对话是常驻的，切换库不会
把聊天记录弄丢 —— 用户经常是「看到一半想问一句」。

## 为什么模型调用要单开线程

一次回答可能流式吐十几秒。放在主线程里，整个界面会僵住，连「停止」都点不动。
`ChatWorker` 跑在 QThread 里，每收到一段就 emit 一个信号，主线程只管往文档里追加。

## 为什么动作必须过确认框

模型说「买 10 手」和真的下单之间必须有一个人。确认框列出「买什么、多少手、为什么」，
用户勾「下次不提醒」才会记住不再问 —— 这个开关在设置里能重新打开。
关掉之后动作直接执行，所以它只适合已经反复确认过流程的人。
"""

from __future__ import annotations

import os
import time

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ttrade import agent as A
from ttrade import library_docs as L
from ttrade import strategy_store as SStore
from ttrade.paths import user_dir

C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_OK = "#16a34a"
C_BAD = "#dc2626"
C_IDLE = "#94a3b8"
C_USER = "#1d4ed8"


def reports_dir() -> str:
    path = os.path.join(user_dir(), "reports")
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------ 线程


class ChatWorker(QObject):
    """跑在 QThread 里：一次问答。"""

    delta = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, cfg: A.AgentConfig, messages: list) -> None:
        super().__init__()
        self.cfg = cfg
        self.messages = messages
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        client = A.LLMClient(self.cfg)
        try:
            text = client.chat(
                self.messages,
                on_delta=self.delta.emit,
                should_stop=lambda: self._stop,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.done.emit(text)


# ------------------------------------------------------------------ 对话框


class ConfirmDialog(QDialog):
    """执行模型给的动作之前确认一次。"""

    def __init__(self, action: A.Action, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("确认执行")
        self.setMinimumWidth(440)
        lay = QVBoxLayout(self)

        head = QLabel(action.describe())
        head.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {C_TEXT};")
        lay.addWidget(head)

        p = action.payload
        rows = []
        if p.get("reason"):
            rows.append(("理由", str(p["reason"])))
        if p.get("symbol"):
            rows.append(("品种", f"{p['symbol']} {p.get('exchange', '')}".strip()))
        if p.get("period"):
            rows.append(("周期", str(p["period"])))
        if p.get("strategy"):
            params = p.get("params") or {}
            rows.append(("策略", f"{p['strategy']} {params if params else ''}".strip()))
        if p.get("volume"):
            rows.append(("手数", str(p["volume"])))
        if rows:
            form = QFormLayout()
            form.setContentsMargins(0, 6, 0, 6)
            for k, v in rows:
                lab = QLabel(v)
                lab.setWordWrap(True)
                form.addRow(QLabel(k), lab)
            lay.addLayout(form)

        warn = QLabel("确认后立即执行，没有撤销。")
        warn.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")
        lay.addWidget(warn)

        self.chk_silent = QCheckBox("下次不提醒（可在设置里重新打开）")
        lay.addWidget(self.chk_silent)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("执行")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def silent(self) -> bool:
        return self.chk_silent.isChecked()


class SettingsDialog(QDialog):
    """模型配置。"""

    def __init__(self, cfg: A.AgentConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("模型设置")
        self.setMinimumWidth(520)
        self.cfg = cfg
        self._result: A.AgentConfig | None = None

        form = QFormLayout(self)

        self.cb_provider = QComboBox()
        for p in A.PROVIDERS:
            self.cb_provider.addItem(p.label, p.key)
        idx = self.cb_provider.findData(cfg.provider)
        self.cb_provider.setCurrentIndex(max(idx, 0))
        form.addRow("供应商", self.cb_provider)

        self.ed_base = QLineEdit(cfg.base_url)
        self.ed_base.setPlaceholderText("https://api.deepseek.com/v1")
        form.addRow("接口地址", self.ed_base)

        self.cb_model = QComboBox()
        self.cb_model.setEditable(True)
        self.cb_model.setCurrentText(cfg.model)
        form.addRow("模型", self.cb_model)

        self.ed_key = QLineEdit(cfg.api_key)
        self.ed_key.setEchoMode(QLineEdit.Password)
        self.ed_key.setPlaceholderText("sk-...")
        form.addRow("API Key", self.ed_key)

        self.sp_temp = QDoubleSpinBox()
        self.sp_temp.setRange(0.0, 2.0)
        self.sp_temp.setSingleStep(0.1)
        self.sp_temp.setValue(cfg.temperature)
        form.addRow("温度", self.sp_temp)

        self.sp_tokens = QSpinBox()
        self.sp_tokens.setRange(256, 32000)
        self.sp_tokens.setSingleStep(256)
        self.sp_tokens.setValue(cfg.max_tokens)
        form.addRow("最大输出 token", self.sp_tokens)

        self.chk_confirm = QCheckBox("下单前弹确认框")
        self.chk_confirm.setChecked(cfg.confirm_trade)
        form.addRow("", self.chk_confirm)

        note = QLabel(
            "Key 明文存在本机用户目录，不会进版本库。\n"
            "推理型模型（deepseek-reasoner、o 系列）会自动省掉温度参数。"
        )
        note.setObjectName("Hint")
        note.setWordWrap(True)
        form.addRow(note)

        row = QHBoxLayout()
        self.btn_test = QPushButton("测试连接")
        self.btn_test.setObjectName("Ghost")
        self.btn_test.clicked.connect(self._test)
        row.addWidget(self.btn_test)
        self.lbl_test = QLabel("")
        self.lbl_test.setObjectName("Hint")
        row.addWidget(self.lbl_test, 1)
        form.addRow(row)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

        self.cb_provider.currentIndexChanged.connect(self._on_provider)

    def _on_provider(self) -> None:
        """换供应商时把地址和模型候选换掉，但**不动已经填好的 Key**。"""
        p = A.provider(self.cb_provider.currentData())
        if p.key == "custom":
            self.cb_model.clear()
            return
        self.ed_base.setText(p.base_url)
        cur = self.cb_model.currentText()
        self.cb_model.blockSignals(True)
        self.cb_model.clear()
        self.cb_model.addItems(p.models)
        self.cb_model.setCurrentText(cur if cur in p.models else (p.models[0] if p.models else ""))
        self.cb_model.blockSignals(False)

    def current(self) -> A.AgentConfig:
        return A.AgentConfig(
            provider=self.cb_provider.currentData(),
            base_url=self.ed_base.text().strip(),
            model=self.cb_model.currentText().strip(),
            api_key=self.ed_key.text().strip(),
            temperature=self.sp_temp.value(),
            max_tokens=self.sp_tokens.value(),
            timeout=self.cfg.timeout,
            confirm_trade=self.chk_confirm.isChecked(),
        )

    def _test(self) -> None:
        cfg = self.current()
        ok, why = cfg.ready()
        if not ok:
            self.lbl_test.setText(why)
            self.lbl_test.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")
            return
        self.lbl_test.setText("测试中…")
        self.lbl_test.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        self.btn_test.setEnabled(False)
        # 界面线程里同步等一次响应（最多 30 秒）。测试按钮是低频操作，
        # 为它再开一个线程不值当，但要把按钮禁掉防止连点。
        try:
            msg = A.LLMClient(cfg).ping()
            self.lbl_test.setText(f"连接正常：{msg}")
            self.lbl_test.setStyleSheet(f"color: {C_OK}; font-size: 11px;")
        except Exception as exc:  # noqa: BLE001
            self.lbl_test.setText(str(exc)[:160])
            self.lbl_test.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")
        finally:
            self.btn_test.setEnabled(True)


# ------------------------------------------------------------------ 页面


class AgentPage(QWidget):
    """交易 Agent。"""

    study_requested = Signal(object)      # SStore.StrategyConfig
    trade_requested = Signal(object)      # A.Action
    portfolio_requested = Signal(object)  # SStore.StrategyConfig

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.cfg = A.load_config()
        self._history: list[dict] = []
        self._worker: ChatWorker | None = None
        self._thread: QThread | None = None
        self._busy = False
        self._pending_report = ""     # 正在生成的研报品种，回完自动落盘
        self._stream_buf = ""
        self._build()
        self._refresh_status()
        self._fill_factors()
        self._fill_strategies()
        self._fill_reports()
        self._render_welcome()

    # ---------------------------------------------------------- 构建

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_left())
        split.addWidget(self._build_chat())
        split.addWidget(self._build_detail())
        split.setSizes([300, 620, 420])
        root.addWidget(split, 1)

    def _build_left(self) -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_factor_tab(), "因子库")
        self.tabs.addTab(self._build_strategy_tab(), "策略库")
        self.tabs.addTab(self._build_report_tab(), "研报库")
        self.tabs.currentChanged.connect(self._on_tab)
        lay.addWidget(self.tabs, 1)

        foot = QHBoxLayout()
        self.lbl_model = QLabel("")
        self.lbl_model.setObjectName("Hint")
        self.lbl_model.setWordWrap(True)
        foot.addWidget(self.lbl_model, 1)
        btn_cfg = QPushButton("设置")
        btn_cfg.setObjectName("Ghost")
        btn_cfg.clicked.connect(self._open_settings)
        foot.addWidget(btn_cfg)
        lay.addLayout(foot)
        return card

    def _build_factor_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(4)

        self.lst_factor = QListWidget()
        self.lst_factor.currentItemChanged.connect(self._show_factor)
        lay.addWidget(self.lst_factor, 1)

        btn = QPushButton("让模型写一个新因子")
        btn.setObjectName("Ghost")
        btn.clicked.connect(self._ask_new_factor)
        lay.addWidget(btn)
        return w

    def _build_strategy_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(4)

        self.lst_strat = QListWidget()
        self.lst_strat.currentItemChanged.connect(self._show_strategy)
        lay.addWidget(self.lst_strat, 1)

        row = QHBoxLayout()
        btn_bt = QPushButton("回测")
        btn_bt.setObjectName("Ghost")
        btn_bt.clicked.connect(self._backtest_selected)
        row.addWidget(btn_bt)
        btn_add = QPushButton("加入组合")
        btn_add.setObjectName("Ghost")
        btn_add.clicked.connect(self._portfolio_selected)
        row.addWidget(btn_add)
        lay.addLayout(row)

        btn_new = QPushButton("让模型生成一个策略")
        btn_new.setObjectName("Ghost")
        btn_new.clicked.connect(self._ask_new_strategy)
        lay.addWidget(btn_new)
        return w

    def _build_report_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(4)

        self.lst_report = QListWidget()
        self.lst_report.currentItemChanged.connect(self._show_report)
        lay.addWidget(self.lst_report, 1)

        row = QHBoxLayout()
        self.cb_report_sym = QComboBox()
        self.cb_report_sym.setEditable(True)
        self.cb_report_sym.setMinimumWidth(90)
        for it in _symbol_choices():
            self.cb_report_sym.addItem(it)
        row.addWidget(self.cb_report_sym, 1)
        btn = QPushButton("生成")
        btn.setObjectName("Ghost")
        btn.clicked.connect(self._make_report)
        row.addWidget(btn)
        lay.addLayout(row)

        note = QLabel("还没接基本面与互联网数据，现在只能基于价格、持仓、展期和模型自身知识写。")
        note.setObjectName("Hint")
        note.setWordWrap(True)
        lay.addWidget(note)
        return w

    def _build_chat(self) -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        head = QHBoxLayout()
        t = QLabel("对话")
        t.setObjectName("SectionTitle")
        head.addWidget(t)
        head.addStretch(1)
        btn_clear = QPushButton("清空")
        btn_clear.setObjectName("Ghost")
        btn_clear.clicked.connect(self._clear_chat)
        head.addWidget(btn_clear)
        lay.addLayout(head)

        self.log = QTextBrowser()
        self.log.setOpenExternalLinks(True)
        lay.addWidget(self.log, 1)

        self.ed_input = QPlainTextEdit()
        self.ed_input.setPlaceholderText(
            "比如：螺纹现在适合做多吗？用双均线帮我回测一下，1 手。"
            "（Ctrl+Enter 发送）"
        )
        self.ed_input.setFixedHeight(84)
        lay.addWidget(self.ed_input)

        row = QHBoxLayout()
        self.lbl_state = QLabel("")
        self.lbl_state.setObjectName("Hint")
        row.addWidget(self.lbl_state, 1)
        self.btn_send = QPushButton("发送")
        self.btn_send.setMinimumWidth(88)
        self.btn_send.clicked.connect(self._send)
        row.addWidget(self.btn_send)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("Ghost")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_stop)
        lay.addLayout(row)
        return card

    def _build_detail(self) -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        t = QLabel("详情")
        t.setObjectName("SectionTitle")
        lay.addWidget(t)
        self.detail = QTextBrowser()
        lay.addWidget(self.detail, 1)
        return card

    # ---------------------------------------------------------- 列表填充

    def _fill_factors(self) -> None:
        self.lst_factor.clear()
        for d in L.all_factor_docs():
            it = QListWidgetItem(f"{d.label}　·　{d.category}")
            it.setData(Qt.UserRole, d.key)
            it.setToolTip(d.desc)
            self.lst_factor.addItem(it)

    def _fill_strategies(self) -> None:
        self.lst_strat.clear()
        for d in L.all_strategy_docs():
            it = QListWidgetItem(f"{d.label}　·　内置")
            it.setData(Qt.UserRole, ("rule", d.key))
            it.setToolTip(d.desc)
            self.lst_strat.addItem(it)
        for cfg in SStore.load_library():
            it = QListWidgetItem(f"{cfg.title()}　·　我的")
            it.setData(Qt.UserRole, ("cfg", cfg.sid))
            it.setToolTip(cfg.describe())
            self.lst_strat.addItem(it)

    def _fill_reports(self) -> None:
        self.lst_report.clear()
        folder = reports_dir()
        files = sorted(
            (f for f in os.listdir(folder) if f.endswith(".md")),
            key=lambda f: os.path.getmtime(os.path.join(folder, f)),
            reverse=True,
        )
        for f in files:
            it = QListWidgetItem(f[:-3])
            it.setData(Qt.UserRole, os.path.join(folder, f))
            self.lst_report.addItem(it)

    def _refresh_status(self) -> None:
        ok, why = self.cfg.ready()
        p = A.provider(self.cfg.provider)
        if ok:
            self.lbl_model.setText(f"{p.label} · {self.cfg.model}")
            self.lbl_model.setStyleSheet(f"color: {C_OK}; font-size: 11px;")
        else:
            self.lbl_model.setText(f"未配置：{why}")
            self.lbl_model.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")

    # ---------------------------------------------------------- 详情渲染

    def _on_tab(self, idx: int) -> None:
        if idx == 0:
            self._show_factor(self.lst_factor.currentItem(), None)
        elif idx == 1:
            self._show_strategy(self.lst_strat.currentItem(), None)
        else:
            self._show_report(self.lst_report.currentItem(), None)

    def _show_factor(self, cur, _prev) -> None:
        if cur is None:
            return
        d = L.factor_doc(str(cur.data(Qt.UserRole)))
        if d is None:
            return
        html = [
            f"<h3 style='color:{C_TEXT};margin:0 0 4px 0'>{d.label}</h3>",
            f"<p style='color:{C_MUTED};margin:0 0 10px 0'>{d.category}　·　"
            f"键名 <code>{d.key}</code></p>",
            "<p style='font-weight:700;margin:10px 0 2px 0'>公式</p>",
            f"<p style='font-family:Consolas,monospace;background:#f6f8fa;"
            f"padding:8px;border-radius:4px'>{_esc(d.formula)}</p>",
            "<p style='font-weight:700;margin:10px 0 2px 0'>怎么读</p>",
            f"<p style='line-height:150%'>{_esc(d.explain)}</p>",
            f"<p style='color:{C_MUTED}'>历史倾向：{d.direction}。"
            "这只是统计倾向，回测里方向由滚动 IC 决定，不是交易信号。</p>",
            "<p style='font-weight:700;margin:10px 0 2px 0'>实现代码</p>",
            _code(d.code),
        ]
        self.detail.setHtml("".join(html))

    def _show_strategy(self, cur, _prev) -> None:
        if cur is None:
            return
        kind, key = cur.data(Qt.UserRole)
        if kind == "rule":
            d = L.strategy_doc(key)
            if d is None:
                return
            params = "，".join(f"{k}={v}" for k, v in d.params.items())
            html = [
                f"<h3 style='color:{C_TEXT};margin:0 0 4px 0'>{d.label}</h3>",
                f"<p style='color:{C_MUTED};margin:0 0 10px 0'>内置规则策略　·　"
                f"键名 <code>{d.key}</code></p>",
                _block("逻辑", d.logic),
                _block("优势", d.edge),
                _block("适应情况", d.fits),
                _block("风险", d.risk),
                _block("参数", f"默认 {params}。<br>{d.params_help}"),
                _block("实现代码", "", raw=True),
                _code(d.code),
            ]
            self.detail.setHtml("".join(html))
            return

        cfg = next((c for c in SStore.load_library() if c.sid == key), None)
        if cfg is None:
            self.detail.setHtml("<p>这条策略已经不在库里了。</p>")
            return
        html = [
            f"<h3 style='color:{C_TEXT};margin:0 0 4px 0'>{cfg.title()}</h3>",
            f"<p style='color:{C_MUTED}'>{cfg.describe()}</p>",
            _block("标的", f"{cfg.key()}　{cfg.period}"),
            _block("策略", cfg.strategy_label()),
            _block("参数", cfg.params_text() or "—"),
            _block("创建", cfg.created or "—"),
        ]
        if cfg.family == "rule":
            d = L.strategy_doc(cfg.rule_key)
            if d is not None:
                html += [_block("逻辑", d.logic), _block("优势", d.edge),
                         _block("适应情况", d.fits), _block("风险", d.risk)]
        else:
            html.append(_block("说明", "因子策略暂不支持实时交易，只能回测。"))
        self.detail.setHtml("".join(html))

    def _show_report(self, cur, _prev) -> None:
        if cur is None:
            self.detail.setHtml("<p style='color:#6b7280'>选中一份研报看内容。</p>")
            return
        path = str(cur.data(Qt.UserRole))
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            self.detail.setHtml(f"<p>读不出来：{_esc(str(exc))}</p>")
            return
        self.detail.setHtml(f"<pre style='white-space:pre-wrap;font-size:12px'>"
                            f"{_esc(text)}</pre>")

    # ---------------------------------------------------------- 对话

    def _render_welcome(self) -> None:
        ok, why = self.cfg.ready()
        tip = (
            "先在左下角「设置」里填好接口地址、模型和 API Key。"
            if not ok else
            "可以开始问了。比如：「螺纹现在适合做多吗？」「帮我用唐奇安回测铁矿」。"
        )
        self.log.setHtml(
            f"<p style='color:{C_MUTED};line-height:160%'>"
            f"我是这个平台里的交易助手，能查数据、解释因子、跑回测、按你的话下单。<br>"
            f"{tip}</p>"
        )

    def _clear_chat(self) -> None:
        self._history.clear()
        self._render_welcome()

    def _append(self, html: str) -> None:
        self.log.moveCursor(QTextCursor.End)
        self.log.insertHtml(html)
        self.log.moveCursor(QTextCursor.End)

    def _send(self) -> None:
        if self._busy:
            return
        text = self.ed_input.toPlainText().strip()
        if not text:
            return
        ok, why = self.cfg.ready()
        if not ok:
            self._set_state(why, C_BAD)
            self._open_settings()
            return

        self.ed_input.clear()
        self._append(
            f"<p style='margin:12px 0 2px 0'><b style='color:{C_USER}'>你</b></p>"
            f"<p style='margin:0 0 8px 0;line-height:160%'>{_esc(text)}</p>"
        )
        self._append(f"<p style='margin:12px 0 2px 0'><b>助手</b></p>")

        self._history.append({"role": "user", "content": text})
        msgs = A.build_messages(A.trim_history(self._history[:-1]), text)
        self._stream_buf = ""
        self._start_worker(msgs)

    def _start_worker(self, msgs: list) -> None:
        self._set_busy(True)
        worker = ChatWorker(self.cfg, msgs)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.delta.connect(self._on_delta)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_done)
        self._worker, self._thread = worker, thread
        thread.start()

    def _on_delta(self, piece: str) -> None:
        self._stream_buf += piece
        # 直接往文档尾部追加纯文本，比每次重设整篇 HTML 快得多 ——
        # 流式回答能来上百个片段，重设整篇会把滚动位置也弄丢。
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(piece)
        self.log.moveCursor(QTextCursor.End)

    def _on_done(self, text: str) -> None:
        final = text or self._stream_buf
        self._append("<p></p>")
        if not final.strip():
            self._append(f"<p style='color:{C_BAD}'>模型没有返回内容。</p>")
        self._history.append({"role": "assistant", "content": final})

        action = A.parse_action(final)
        if action.kind and not action.ok:
            self._append(
                f"<p style='color:{C_BAD};font-size:11px'>动作块有问题：{_esc(action.error)}</p>"
            )
        elif action.ok:
            self._handle_action(action)

        if self._pending_report:
            self._save_report(self._pending_report, final)
            self._pending_report = ""

    def _on_failed(self, msg: str) -> None:
        self._append(f"<p style='color:{C_BAD}'>出错了：{_esc(msg)}</p>")
        self._set_state(msg[:80], C_BAD)
        self._pending_report = ""

    def _on_thread_done(self) -> None:
        self._worker = None
        self._thread = None
        self._set_busy(False)
        # 只在「正在生成」时改状态：出错和已提交的回执不能在这里被覆盖掉
        if self.lbl_state.text() == "正在生成…":
            self._set_state("就绪", C_MUTED)

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._set_state("正在停止…", C_MUTED)

    def _set_busy(self, on: bool) -> None:
        self._busy = on
        self.btn_send.setEnabled(not on)
        self.btn_stop.setEnabled(on)
        if on:
            self._set_state("正在生成…", C_MUTED)

    def _set_state(self, text: str, color: str = C_MUTED) -> None:
        self.lbl_state.setText(text)
        self.lbl_state.setStyleSheet(f"color: {color}; font-size: 11px;")

    # ---------------------------------------------------------- 动作

    def _handle_action(self, action: A.Action) -> None:
        # 只有真下单才拦。回测是只读操作，为它弹一个「确认后立即执行」的框
        # 只会训练用户无脑点确认。
        if self.cfg.confirm_trade and action.kind == "trade":
            dlg = ConfirmDialog(action, self)
            if dlg.exec() != QDialog.Accepted:
                self._append(f"<p style='color:{C_MUTED};font-size:11px'>已取消。</p>")
                return
            if dlg.silent():
                self.cfg.confirm_trade = False
                A.save_config(self.cfg)
                self._append(
                    f"<p style='color:{C_MUTED};font-size:11px'>"
                    "以后不再询问，可在设置里重新打开。</p>"
                )
        self._append(
            f"<p style='color:{C_OK};font-size:11px'>已提交：{_esc(action.describe())}</p>"
        )
        self.trade_requested.emit(action)

    # ---------------------------------------------------------- 策略 / 研报动作

    def _selected_strategy(self):
        cur = self.lst_strat.currentItem()
        if cur is None:
            return None
        kind, key = cur.data(Qt.UserRole)
        if kind == "cfg":
            return next((c for c in SStore.load_library() if c.sid == key), None)
        d = L.strategy_doc(key)
        if d is None:
            return None
        return SStore.StrategyConfig(
            symbol="", exchange="", period="1d",
            rule_key=d.key, rule_params=dict(d.params), family="rule",
        )

    def _backtest_selected(self) -> None:
        cfg = self._selected_strategy()
        if cfg is None:
            self._set_state("先选中一条策略", C_BAD)
            return
        self.study_requested.emit(cfg)

    def _portfolio_selected(self) -> None:
        cfg = self._selected_strategy()
        if cfg is None:
            self._set_state("先选中一条策略", C_BAD)
            return
        if not cfg.symbol:
            self._set_state("这条是内置模板，还没有选品种。先到「策略研究」里配好再存进库", C_BAD)
            return
        self.portfolio_requested.emit(cfg)

    def _ask_new_factor(self) -> None:
        self.tabs.setCurrentIndex(0)
        self._prefill(
            "我想做一个新的因子：<在这里写思路，比如「近 20 日持仓量增加且价格上涨的强度」>。\n"
            "请按平台格式给出：因子名、公式、pandas 实现（函数签名 f_xxx(df) -> Series，"
            "只能用 df 的 open/high/low/close/volume/amount/open_interest 列）、"
            "以及它适合什么行情。"
        )

    def _ask_new_strategy(self) -> None:
        self.tabs.setCurrentIndex(1)
        self._prefill(
            "我想做一个新的规则策略：<在这里写思路>。\n"
            "请给出：入场条件、离场条件、参数与默认值、适合的品种和行情、"
            "以及用平台已有的 47 个因子能不能近似实现。"
        )

    def _prefill(self, text: str) -> None:
        self.ed_input.setPlainText(text)
        self.ed_input.setFocus()

    def _make_report(self) -> None:
        sym = self.cb_report_sym.currentText().strip()
        if not sym:
            self._set_state("先选一个品种", C_BAD)
            return
        self._pending_report = f"{sym}_{time.strftime('%Y%m%d_%H%M')}"
        self.ed_input.setPlainText(
            f"写一份 {sym} 的品种研报，面向做趋势跟踪的交易者。要求：\n"
            f"1. 先说明你只能基于平台已有的价格、持仓、展期数据和你的知识，"
            f"没有基本面数据库，涉及具体库存/产量/进口数据时明确说「未接入」；\n"
            f"2. 结构：供需与季节性（定性）、期限结构含义、波动特征、"
            f"适合的因子与策略、当前需要注意的风险；\n"
            f"3. 不要编造具体数字。"
        )
        self._send()

    def _save_report(self, name: str, text: str) -> None:
        path = os.path.join(reports_dir(), f"{name}.md")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            self._set_state(f"研报保存失败：{exc}", C_BAD)
            return
        self._fill_reports()
        self._set_state(f"研报已保存：{name}.md", C_OK)

    # ---------------------------------------------------------- 设置

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.cfg = dlg.current()
        A.save_config(self.cfg)
        self._refresh_status()
        ok, why = self.cfg.ready()
        self._set_state("设置已保存" if ok else why, C_OK if ok else C_BAD)

    # ---------------------------------------------------------- 外部通知

    def reload(self) -> None:
        """重读策略库与研报目录。切回本页时调，能看到刚存进库的策略。"""
        self._fill_strategies()
        self._fill_reports()

    def notify(self, msg: str, bad: bool = False) -> None:
        """别处（主窗口）执行完动作后的回执。写进对话流，用户能翻回去看。"""
        color = C_BAD if bad else C_OK
        self._set_state(msg, color)
        self._append(f"<p style='color:{color};font-size:11px'>{_esc(msg)}</p>")

    # ---------------------------------------------------------- 生命周期

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)


# ------------------------------------------------------------------ 小工具


def _esc(s: str) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _code(text: str) -> str:
    if not text.strip():
        return "<p style='color:#6b7280'>（打包后取不到源码）</p>"
    return (
        "<pre style='font-family:Consolas,monospace;font-size:11px;"
        "background:#f6f8fa;padding:8px;border-radius:4px;"
        f"white-space:pre-wrap'>{_esc(text)}</pre>"
    )


def _block(title: str, body: str, raw: bool = False) -> str:
    if raw or not body:
        return f"<p style='font-weight:700;margin:10px 0 2px 0'>{title}</p>"
    return (
        f"<p style='font-weight:700;margin:10px 0 2px 0'>{title}</p>"
        f"<p style='line-height:150%;margin:0'>{body}</p>"
    )


# 研报品种候选。**不查 DuckDB**：建页面时去开库，赶上别的进程在写就会卡住，
# 而这里只是个下拉框的候选，不值得为它冒这个风险。
REPORT_SYMBOLS = [
    "RB", "HC", "I", "J", "JM", "CU", "AL", "ZN", "NI", "AU", "AG",
    "SC", "FU", "MA", "TA", "EG", "PP", "L", "V", "SR", "CF", "M", "Y", "P", "OI",
]


def _symbol_choices() -> list[str]:
    return list(REPORT_SYMBOLS)
