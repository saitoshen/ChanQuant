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

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
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
from ttrade import agent_memory as Mem
from ttrade import factors as F
from ttrade import library_docs as L
from ttrade import strategy_store as SStore
from ttrade.strategies import STRATEGIES
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
        self._client: A.LLMClient | None = None
        # 自己攒一份：被中途掐断时 `chat()` 会抛异常、拿不到返回值，但已经吐出来
        # 的那部分得留住，否则「停止」会把答到一半的内容整段丢掉
        self._buf: list[str] = []

    def stop(self) -> None:
        self._stop = True
        # 只置标志不够：`chat()` 每两行才检查一次，模型思考时可能几十秒不推字节，
        # 线程就卡在 socket 读上。关掉 response 让读立刻抛错。
        if self._client is not None:
            self._client.abort()

    def _on_piece(self, piece: str) -> None:
        self._buf.append(piece)
        self.delta.emit(piece)

    def run(self) -> None:
        self._client = A.LLMClient(self.cfg)
        try:
            text = self._client.chat(
                self.messages,
                on_delta=self._on_piece,
                should_stop=lambda: self._stop,
            )
        except Exception as exc:  # noqa: BLE001
            if self._stop:
                # 主动停止时中断读必然抛异常，这不是错误 —— 把已收到的半截当结果
                self.done.emit("".join(self._buf))
            else:
                self.failed.emit(str(exc))
            return
        self.done.emit(text)


class MemoWorker(QObject):
    """跑在 QThread 里：把一段旧对话压成记忆。

    单独一个类而不是复用 `ChatWorker`：这个不流式、不显示、不需要「停止」，
    复用只会让 `ChatWorker` 多出一堆 `if quiet` 分支。
    """

    done = Signal(str)
    failed = Signal(str)

    def __init__(self, cfg: A.AgentConfig, convo: list) -> None:
        super().__init__()
        self.cfg = cfg
        self.convo = convo

    def run(self) -> None:
        try:
            text = A.LLMClient(self.cfg).chat(A.memory_messages(self.convo))
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
        if action.kind == "factor":
            # 注册因子要给人看代码。只写「注册一个因子」等于让人盲签 —— 这段代码
            # 会在本机执行，用户至少该有机会扫一眼。
            rows = [
                ("代号", p.get("key", "")),
                ("类别", p.get("category", "")),
                ("公式", p.get("formula", "") or "（模型没给公式）"),
                ("说明", p.get("desc", "")),
            ]
            form = QFormLayout()
            form.setContentsMargins(0, 6, 0, 6)
            for k, v in rows:
                if not v:
                    continue
                lab = QLabel(str(v))
                lab.setWordWrap(True)
                form.addRow(QLabel(k), lab)
            lay.addLayout(form)

            lay.addWidget(QLabel("实现代码"))
            code = QPlainTextEdit(p.get("code", ""))
            code.setReadOnly(True)
            code.setMinimumHeight(240)
            code.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
            lay.addWidget(code)

            warn = QLabel("确认后写入因子库并立刻可用；代码有问题会注册失败并给出原因。")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")
            lay.addWidget(warn)

            # 注册因子没有「下次不提醒」：每次的代码都不一样，盲签没有意义
            self.chk_silent = QCheckBox()
            self.chk_silent.hide()
            box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            box.button(QDialogButtonBox.Ok).setText("注册")
            box.button(QDialogButtonBox.Cancel).setText("取消")
            box.accepted.connect(self.accept)
            box.rejected.connect(self.reject)
            lay.addWidget(box)
            return

        if action.kind == "strategy":
            # 和注册因子一样：这段代码会在本机跑，而且会拿真金白银的数据做回测，
            # 所以代码必须摊开给人看。
            rows = [
                ("代号", p.get("key", "")),
                ("说明", p.get("desc", "")),
                ("默认参数", "　".join(f"{k}={v}" for k, v in (p.get("params") or {}).items())
                 or "（无参数）"),
            ]
            form = QFormLayout()
            form.setContentsMargins(0, 6, 0, 6)
            for k, v in rows:
                if not v:
                    continue
                lab = QLabel(str(v))
                lab.setWordWrap(True)
                form.addRow(QLabel(k), lab)
            lay.addLayout(form)

            lay.addWidget(QLabel("实现代码"))
            code = QPlainTextEdit(p.get("code", ""))
            code.setReadOnly(True)
            code.setMinimumHeight(260)
            code.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
            lay.addWidget(code)

            warn = QLabel("确认后写入策略库，之后可以在「策略研究」页选它回测或实盘；"
                          "代码跑不通会注册失败并给出原因。")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")
            lay.addWidget(warn)

            # 策略也没有「下次不提醒」：每次的代码都不一样
            self.chk_silent = QCheckBox()
            self.chk_silent.hide()
            box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            box.button(QDialogButtonBox.Ok).setText("注册")
            box.button(QDialogButtonBox.Cancel).setText("取消")
            box.accepted.connect(self.accept)
            box.rejected.connect(self.reject)
            lay.addWidget(box)
            return

        if action.kind == "fs":
            # 写入是整文件覆盖，看不见内容就是盲签。这里连「原来有没有这个文件」
            # 都要标出来 —— 覆盖已有文件和新建一个文件，风险差着一个量级。
            from ttrade import agent_fs as FS

            path = p.get("path", "")
            try:
                old = FS.exists(path)
            except Exception:  # noqa: BLE001
                old = False
            rows = [
                ("路径", path),
                ("动作", "覆盖已有文件" if old else "新建文件"),
                ("大小", f"{len(p.get('content') or '')} 字符"),
            ]
            form = QFormLayout()
            form.setContentsMargins(0, 6, 0, 6)
            for k, v in rows:
                lab = QLabel(str(v))
                lab.setWordWrap(True)
                form.addRow(QLabel(k), lab)
            lay.addLayout(form)

            lay.addWidget(QLabel("写入内容（整文件覆盖）"))
            body = QPlainTextEdit(p.get("content") or "")
            body.setReadOnly(True)
            body.setMinimumHeight(280)
            body.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
            lay.addWidget(body)

            warn = QLabel(
                "确认后整个文件被替换，没有撤销。改的是程序自己的代码时，"
                "重启之后平台可能起不来。"
            )
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")
            lay.addWidget(warn)

            self.chk_silent = QCheckBox()
            self.chk_silent.hide()
            box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            box.button(QDialogButtonBox.Ok).setText("写入")
            box.button(QDialogButtonBox.Cancel).setText("取消")
            box.accepted.connect(self.accept)
            box.rejected.connect(self.reject)
            lay.addWidget(box)
            return

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


class ManualDialog(QDialog):
    """平台手册预览。

    整份约四万字，用只读文本框摊开 —— 用户要核对「模型到底看到了什么」时，
    翻这里比翻源码快。另存为 Markdown 是给「想把手册喂给别的模型」用的。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from ttrade import factor_manual as M

        self.setWindowTitle("平台手册 · 数据与因子")
        self.resize(900, 720)
        self.text = M.manual()

        lay = QVBoxLayout(self)
        head = QLabel(
            "「交易Agent」查资料时看到的就是这份内容：行情从哪来、平台注入了哪些"
            "原语、每个因子的完整实现。它自己也会用 doc 动作按需取其中一段。"
        )
        head.setWordWrap(True)
        head.setObjectName("Hint")
        lay.addWidget(head)

        box = QPlainTextEdit(self.text)
        box.setReadOnly(True)
        box.setStyleSheet(
            "font-family: Consolas, 'Microsoft YaHei', monospace; font-size: 12px;"
        )
        lay.addWidget(box, 1)

        row = QHBoxLayout()
        self.lbl_info = QLabel(f"{len(self.text)} 字符")
        self.lbl_info.setObjectName("Hint")
        row.addWidget(self.lbl_info, 1)
        btn_save = QPushButton("另存为 Markdown")
        btn_save.setObjectName("Ghost")
        btn_save.clicked.connect(self._save)
        row.addWidget(btn_save)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        row.addWidget(btn_close)
        lay.addLayout(row)

    def _save(self) -> None:
        from ttrade import factor_manual as M

        path, _ = QFileDialog.getSaveFileName(
            self, "保存平台手册",
            os.path.join(user_dir(), "platform_manual.md"),
            "Markdown (*.md)",
        )
        if not path:
            return
        ok, msg = M.save(path)
        self.lbl_info.setText(msg)
        self.lbl_info.setStyleSheet(f"color: {C_OK if ok else C_BAD}; font-size: 11px;")


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

        # 权限分级。默认那档能做的事都写在提示词里（下单 / 回测 / 注册因子和策略），
        # 这一档是额外放开磁盘 —— 等于把平台自己的代码交给模型，所以要拦一道警告。
        self.chk_fs = QCheckBox("允许 Agent 读写 TrendDesk 目录（升级权限）")
        self.chk_fs.setChecked(cfg.allow_fs)
        self.chk_fs.setToolTip(
            "关着：Agent 只能下单、回测、注册因子和策略。\n"
            "打开：多一个文件动作，可读写 TrendDesk 根目录下的任意文件。"
        )
        self.chk_fs.toggled.connect(self._on_fs_toggled)
        form.addRow("", self.chk_fs)

        self.lbl_mem = QLabel(f"{Mem.count()} 条")
        self.lbl_mem.setObjectName("Hint")
        btn_mem = QPushButton("清空记忆")
        btn_mem.setObjectName("Ghost")
        btn_mem.clicked.connect(self._clear_memory)
        mem_row = QHBoxLayout()
        mem_row.addWidget(self.lbl_mem, 1)
        mem_row.addWidget(btn_mem)
        form.addRow("长期记忆", mem_row)

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
            allow_fs=self.chk_fs.isChecked(),
        )

    def _on_fs_toggled(self, on: bool) -> None:
        """打开升级权限前必须警告一次。关掉不拦 —— 降权永远安全。"""
        if not on:
            return
        ok = QMessageBox.warning(
            self, "升级权限",
            "打开之后，Agent 可以读写 TrendDesk 目录下的任何文件 —— 包括程序"
            "自己的代码、配置和历史数据。\n\n"
            "模型改错了代码，平台可能起不来；写错了数据文件，已经补好的行情"
            "可能被覆盖。每次写入仍会弹一次确认框，读文件不会。\n\n"
            "确定要打开吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            # 拒绝了就把勾去掉。`blockSignals` 防递归：`setChecked(False)` 会再
            # 发一次 toggled，虽然那里直接 return，少绕一圈更不容易看错
            self.chk_fs.blockSignals(True)
            self.chk_fs.setChecked(False)
            self.chk_fs.blockSignals(False)

    def _clear_memory(self) -> None:
        """清空长期记忆。对话记录不动 —— 那是两回事。"""
        n = Mem.count()
        if n == 0:
            return
        ok = QMessageBox.question(
            self, "清空长期记忆",
            f"删掉 {n} 条记忆？当前对话记录不受影响，只是下次开窗口不记得这些了。",
        )
        if ok != QMessageBox.Yes:
            return
        Mem.clear()
        self.lbl_mem.setText("0 条")

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
    factor_added = Signal(str)            # 新注册的因子 key，主窗口据此刷新下拉
    strategy_added = Signal(str)          # 新注册的策略 key，同上
    factor_removed = Signal(str)          # 删掉的因子 key，主窗口重建下拉
    strategy_removed = Signal(str)        # 删掉的策略 key，同上

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.cfg = A.load_config()
        self._history: list[dict] = []
        self._worker: ChatWorker | None = None
        self._thread: QThread | None = None
        self._busy = False
        self._pending_report = ""     # 正在生成的研报品种，回完自动落盘
        self._stream_buf = ""
        self._memo_worker: MemoWorker | None = None
        self._memo_thread: QThread | None = None
        self._memo_warned = False     # 压缩失败只提示一次，别每轮都刷
        self._tool_turns = 0            # 本轮提问已经连续自动跑了几次工具动作（查手册 / 读写文件）
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

        foot = QVBoxLayout()
        foot.setSpacing(4)
        self.lbl_model = QLabel("")
        self.lbl_model.setObjectName("Hint")
        self.lbl_model.setWordWrap(True)
        foot.addWidget(self.lbl_model)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_manual = QPushButton("手册")
        self.btn_manual.setObjectName("Ghost")
        self.btn_manual.setToolTip("平台手册：数据链路、注入的原语、每个因子的完整实现")
        self.btn_manual.clicked.connect(self._open_manual)
        row.addWidget(self.btn_manual)
        btn_cfg = QPushButton("设置")
        btn_cfg.setObjectName("Ghost")
        btn_cfg.clicked.connect(self._open_settings)
        row.addWidget(btn_cfg)
        foot.addLayout(row)
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

        # 删除只对自定义因子开放：内置因子写在 `factors.py` 里，不在 JSON 里，
        # 「删掉」下次启动又会回来，让人白删一次比不让删更糟。
        self.btn_del_factor = QPushButton("删除选中因子")
        self.btn_del_factor.setObjectName("Ghost")
        self.btn_del_factor.setEnabled(False)
        self.btn_del_factor.setToolTip("只能删自定义因子；内置因子改代码才会消失。")
        self.btn_del_factor.clicked.connect(self._delete_factor)
        lay.addWidget(self.btn_del_factor)
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

        # 同理，内置规则策略删不掉；「我的」那些是策略库模板，也允许删
        self.btn_del_strat = QPushButton("删除选中策略")
        self.btn_del_strat.setObjectName("Ghost")
        self.btn_del_strat.setEnabled(False)
        self.btn_del_strat.setToolTip("删自定义策略或策略库模板；内置策略改代码才会消失。")
        self.btn_del_strat.clicked.connect(self._delete_strategy)
        lay.addWidget(self.btn_del_strat)
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
            tag = "自定义" if _is_user_strategy(d.key) else "内置"
            it = QListWidgetItem(f"{d.label}　·　{tag}")
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
            # 升级权限是个危险开关，状态要一直看得见，不能只在设置里藏着
            extra = "｜升级权限" if self.cfg.allow_fs else ""
            self.lbl_model.setText(f"{p.label} · {self.cfg.model}{extra}")
            self.lbl_model.setStyleSheet(
                f"color: {C_BAD if self.cfg.allow_fs else C_OK}; font-size: 11px;"
            )
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
        # 列表重建时 cur 会变成 None，按钮得跟着灰掉，否则会出现「空选中也能点删除」
        self.btn_del_factor.setEnabled(
            cur is not None and _is_user_factor(str(cur.data(Qt.UserRole)))
        )
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
        kind, key = cur.data(Qt.UserRole) if cur is not None else ("", "")
        # 内置规则策略删不掉；自定义策略和「我的」模板都可以删
        self.btn_del_strat.setEnabled(
            cur is not None and (kind == "cfg" or (kind == "rule" and _is_user_strategy(key)))
        )
        if cur is None:
            return
        if kind == "rule":
            d = L.strategy_doc(key)
            if d is None:
                return
            params = "，".join(f"{k}={v}" for k, v in d.params.items())
            # 自定义策略没有「优势 / 适应情况 / 风险」这几段（那是给内置策略写死的
            # 说明），空标题一片白反而像加载失败，所以空的一律不渲染
            kind = "自定义策略" if _is_user_strategy(d.key) else "内置规则策略"
            html = [
                f"<h3 style='color:{C_TEXT};margin:0 0 4px 0'>{d.label}</h3>",
                f"<p style='color:{C_MUTED};margin:0 0 10px 0'>{kind}　·　"
                f"键名 <code>{d.key}</code></p>",
            ]
            for title, body in (
                ("逻辑", d.logic), ("优势", d.edge),
                ("适应情况", d.fits), ("风险", d.risk),
            ):
                if body:
                    html.append(_block(title, body))
            html += [
                _block("参数", f"默认 {params}。" + (f"<br>{d.params_help}" if d.params_help else "")),
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
        n_mem = Mem.count()
        mem_line = (
            f"<br>还记着以前聊过的 {n_mem} 条要点（左下角「设置」里可以清空）。"
            if n_mem else ""
        )
        self.log.setHtml(
            f"<p style='color:{C_MUTED};line-height:160%'>"
            f"我是这个平台里的交易助手，能查数据、解释因子、跑回测、按你的话下单。<br>"
            f"{tip}{mem_line}</p>"
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
        self._tool_turns = 0
        msgs = A.build_messages(
            A.trim_history(self._history[:-1]), text, memory=Mem.brief(),
            allow_fs=self.cfg.allow_fs,
        )
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
        # **必须显式 quit**。`started` 连的是一个普通函数，它返回后线程的事件循环
        # 还在跑，`finished` 永远不会发 —— 表现就是「发送按钮点一次就灰掉，停止也
        # 停不下来」，因为 `_on_thread_done` 从来没被调用过。
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
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

        self._maybe_compress()

    def _on_failed(self, msg: str) -> None:
        self._append(f"<p style='color:{C_BAD}'>出错了：{_esc(msg)}</p>")
        self._set_state(msg[:80], C_BAD)
        self._pending_report = ""

    def _on_thread_done(self) -> None:
        # 旧线程收尾时不能动状态：工具动作会连着起好几轮，上一轮的 `finished`
        # 可能比新一轮晚到，一律清空的话新线程就没人管了（按钮提前变灰）。
        if self.sender() is not None and self.sender() is not self._thread:
            return
        self._worker = None
        self._thread = None
        self._set_busy(False)
        # 只在「进行中」的状态上改：出错和「已提交」的回执不能在这里被覆盖掉。
        # 「正在停止…」也要收尾 —— 漏了它，点过停止的状态栏就永远停在那句话上。
        if self.lbl_state.text() in ("正在生成…", "正在停止…"):
            self._set_state("就绪", C_MUTED)

    def _stop(self) -> None:
        if self._worker is None:
            return
        self._worker.stop()
        self._set_state("正在停止…", C_MUTED)

    # ---------------------------------------------------------- 长期记忆

    # 超过 MEMO_AT 条就把最老的一段压成记忆，最近 MEMO_KEEP 条原样留着。
    # 16 条 = 8 轮：再短的话压缩一次只省下两三轮，请求数白花；再长的话
    # `trim_history` 的 40 条上限会先把中间那段切掉。
    MEMO_AT = 16
    MEMO_KEEP = 8

    def _maybe_compress(self) -> None:
        """一轮答完之后看一眼：历史够长了就把老的那截压成记忆。

        压缩本身是一次额外的模型请求，所以放后台线程，不能卡住界面 ——
        用户下一句可能立刻就来。
        """
        if self._memo_thread is not None or not self.cfg.ready():
            return
        # 工具动作还在自动续跑：这段历史下一秒就要用，压掉再删会让模型丢掉
        # 刚查到的内容。等这轮结束、用户下一句再压
        if self._tool_turns:
            return
        n = len(self._history) - self.MEMO_KEEP
        if n < self.MEMO_AT - self.MEMO_KEEP:
            return
        chunk = self._history[:n]
        chars = sum(len(m.get("content") or "") for m in chunk)
        # 全是「嗯」「好」就再攒攒；但历史快到硬上限时必须压，否则 `trim_history`
        # 会直接把这截丢掉，什么也留不下
        if chars < 400 and len(self._history) < A.MAX_HISTORY - 2:
            return

        worker = MemoWorker(self.cfg, chunk)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(lambda text, c=chunk: self._on_memo_done(text, c))
        worker.failed.connect(self._on_memo_failed)
        # 同 `ChatWorker`：`started` 连普通函数，不显式 quit 就永远不发 finished
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_memo_thread_done)
        self._memo_worker, self._memo_thread = worker, thread
        thread.start()

    def _on_memo_done(self, text: str, chunk: list) -> None:
        # 用户中途点了「清空」：这段对话他不要了，那也别往记忆里塞
        if not chunk or self._history[:len(chunk)] != chunk:
            return
        text = (text or "").strip()
        # 模型判断这段没什么可记的就只回「无」。那也照样丢 —— 留着占上下文，
        # 而它已经明确说了不值得记
        saved = 0
        if text and text not in ("无", "无。", "無"):
            saved = Mem.append([Mem.MemoryItem(
                text=text, kind="summary", turns=len(chunk),
            )])

        # 上面已经确认过这段还是原样，直接切掉
        del self._history[:len(chunk)]

        if saved:
            self._append(
                f"<p style='color:{C_MUTED};font-size:11px'>已把前 {len(chunk)} 条对话"
                f"压成记忆（共 {Mem.count()} 条），下次开窗口还在。</p>"
            )

    def _on_memo_failed(self, msg: str) -> None:
        # 只提示一次：模型没配好时每轮都失败，刷屏比不提示更烦
        if self._memo_warned:
            return
        self._memo_warned = True
        self._append(
            f"<p style='color:{C_MUTED};font-size:11px'>记忆压缩没成功"
            f"（{_esc(msg[:80])}），这段对话先留着，下次再试。</p>"
        )

    def _on_memo_thread_done(self) -> None:
        self._memo_worker = None
        self._memo_thread = None

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
        # 工具动作自己管确认框（文件写入必须弹、读和查手册不用），所以先走掉
        if action.kind == "fs":
            self._run_fs_action(action)
            return
        if action.kind == "doc":
            self._run_doc_action(action)
            return

        # 只有「会改本机状态」的动作才拦：下单、注册因子。回测是只读操作，为它
        # 弹一个「确认后立即执行」的框只会训练用户无脑点确认。
        #
        # 注册因子**不受** `confirm_trade` 开关控制 —— 那个开关管的是下单，而这里
        # 要执行的是模型写的 Python 代码，每次都不一样，盲签没有意义。
        need_confirm = action.kind in ("factor", "strategy") or (
            self.cfg.confirm_trade and action.kind == "trade"
        )
        if need_confirm:
            dlg = ConfirmDialog(action, self)
            if dlg.exec() != QDialog.Accepted:
                self._append(f"<p style='color:{C_MUTED};font-size:11px'>已取消。</p>")
                return
            if action.kind == "trade" and dlg.silent():
                self.cfg.confirm_trade = False
                A.save_config(self.cfg)
                self._append(
                    f"<p style='color:{C_MUTED};font-size:11px'>"
                    "以后不再询问，可在设置里重新打开。</p>"
                )

        if action.kind == "factor":
            self._register_factor(action.payload)
            return
        if action.kind == "strategy":
            self._register_strategy(action.payload)
            return

        self._append(
            f"<p style='color:{C_OK};font-size:11px'>已提交：{_esc(action.describe())}</p>"
        )
        self.trade_requested.emit(action)

    # ---------------------------------------------------------- 工具动作

    # 一次提问里最多连续自动跑几轮工具动作。模型拿到结果后可能再查、再读、再写，
    # 不封顶的话它俩能来回刷到把 token 烧光。
    MAX_TOOL_TURNS = 4

    def _run_doc_action(self, action: A.Action) -> None:
        """查平台手册。只读本地文本，不碰磁盘、不弹框。"""
        from ttrade import factor_manual as M

        topic = action.payload.get("topic", "")
        title, body = M.resolve(topic)
        self._append(
            f"<p style='color:{C_MUTED};font-size:11px'>已查阅平台手册：{_esc(title)}"
            f"（{len(body)} 字符）</p>"
        )
        self._set_state(f"已查手册：{title}", C_OK)
        self._after_tool(f"平台手册「{title}」的内容：\n\n{body}")

    def _run_fs_action(self, action: A.Action) -> None:
        """执行 `fs` 动作：读和列目录直接做，写必须弹确认框。"""
        from ttrade import agent_fs as FS

        p = action.payload
        op = p.get("op", "")
        path = p.get("path", "")

        # 解析层不管权限 —— `parse_action` 不该知道界面状态。在这里拦，
        # 并且说清楚为什么，否则用户只看到一句「已取消」，不知道是权限没开。
        if not self.cfg.allow_fs:
            self._append(
                f"<p style='color:{C_BAD}'>模型想{_esc(action.describe())}，"
                "但升级权限没打开。到左下角「设置」里勾上「允许 Agent 读写 "
                "TrendDesk 目录」才能用。</p>"
            )
            self._set_state("升级权限未开启", C_BAD)
            return

        if op == "list":
            text, err = FS.list_dir(path)
            if err:
                self._after_tool(f"列出 {path} 失败：{err}")
                return
            self._append(
                f"<p style='color:{C_MUTED};font-size:11px'>已列出 {_esc(path)}</p>"
                f"<pre style='font-size:11px'>{_esc(text)}</pre>"
            )
            self._after_tool(f"目录 {path} 的内容：\n{text}")
            return

        if op == "read":
            text, err = FS.read_text(path)
            if err:
                self._after_tool(f"读取 {path} 失败：{err}")
                return
            # 日志里只贴前 4000 字符：整篇贴进去 QTextBrowser 会卡，而用户
            # 真要核对时看文件本身更快。喂给模型的仍是全文。
            shown = text if len(text) <= 4000 else text[:4000] + "\n……（对话里只显示前 4000 字符）"
            self._append(
                f"<p style='color:{C_MUTED};font-size:11px'>已读取 {_esc(path)}"
                f"（{len(text)} 字符）</p>"
                f"<pre style='white-space:pre-wrap;font-family:Consolas,monospace;"
                f"font-size:11px'>{_esc(shown)}</pre>"
            )
            self._after_tool(f"文件 {path} 的全文：\n```\n{text}\n```")
            return

        # 写：内容摊开给人看。读可以不问，写不行 —— 一次覆盖就是几十行没了。
        content = p.get("content", "")
        dlg = ConfirmDialog(action, self)
        if dlg.exec() != QDialog.Accepted:
            self._append(f"<p style='color:{C_MUTED};font-size:11px'>已取消写入。</p>")
            self._after_tool(f"用户拒绝写入 {path}，不要重试同一个改动，先问清楚要改什么。")
            return
        ok, msg = FS.write_text(path, content)
        color = C_OK if ok else C_BAD
        self._append(f"<p style='color:{color}'>{_esc(msg)}</p>")
        self._set_state(msg[:80], color)
        self._after_tool(("写入成功：" if ok else "写入失败：") + msg)

    def _after_tool(self, result: str) -> None:
        """把执行结果喂回模型，让它接着干。

        标成 `user` 角色是协议限制：OpenAI 兼容端点里没有 `tool` 角色，塞进
        `assistant` 又会让模型以为那是自己说过的话。
        """
        note = "【平台执行结果】\n" + result + "\n\n需要继续就再给一个动作块，不需要就用人话总结。"
        self._history.append({"role": "user", "content": note})
        if self._tool_turns >= self.MAX_TOOL_TURNS:
            self._append(
                f"<p style='color:{C_MUTED};font-size:11px'>工具动作已连续执行 "
                f"{self._tool_turns} 轮，先停在这里。要继续就再发一句。</p>"
            )
            return
        self._tool_turns += 1
        msgs = A.build_messages(
            A.trim_history(self._history[:-1]), note,
            # 沿用当前开关：关着的时候不能因为「续跑」就悄悄把文件能力塞回去
            memory=Mem.brief(), allow_fs=self.cfg.allow_fs,
        )
        self._append("<p style='margin:12px 0 2px 0'><b>助手</b></p>")
        self._stream_buf = ""
        # 延到下一个事件循环再起线程：这里还在 `_on_done` 里，旧线程的
        # `finished` 可能比新线程先到，直接起会让 `_on_thread_done` 把新线程
        # 的状态抹掉（发送按钮变灰、停止键失灵）
        QTimer.singleShot(0, lambda m=msgs: self._start_worker(m))

    def _register_factor(self, p: dict) -> None:
        """把模型写的因子落地：试算 → 写盘 → 注册进 FACTORS → 刷新左边的库。"""
        from ttrade import userfactors as UF

        f = UF.UserFactor(
            key=p.get("key", ""), label=p.get("label", ""),
            category=p.get("category", "自定义"), desc=p.get("desc", ""),
            formula=p.get("formula", ""), hint_dir=int(p.get("hint_dir", 1)),
            func=p.get("func", ""), code=p.get("code", ""),
        )
        ok, msg = UF.register(f)
        if not ok:
            self._append(f"<p style='color:{C_BAD}'>注册失败：{_esc(msg)}</p>")
            self._set_state("因子注册失败", C_BAD)
            return

        self._append(
            f"<p style='color:{C_OK}'>已注册因子「{_esc(f.label)}」（{_esc(f.key)}）。"
            f"{_esc(msg)}</p>"
            f"<p style='color:{C_MUTED};font-size:11px'>"
            "在「策略研究」页的因子列表里能选到它。定义存在 "
            f"<code>{_esc(UF.path())}</code>，不想要就在左边选中它、点「删除选中因子」。</p>"
        )
        self._set_state(f"已注册因子 {f.key}", C_OK)
        self.reload()
        self.factor_added.emit(f.key)

    def _register_strategy(self, p: dict) -> None:
        """把模型写的策略落地：试算 → 写盘 → 注册进 STRATEGIES → 刷新左边的库。"""
        from ttrade import userstrategies as US

        s = US.UserStrategy(
            key=p.get("key", ""), label=p.get("label", ""), desc=p.get("desc", ""),
            params=dict(p.get("params") or {}),
            param_meta={k: list(v) for k, v in (p.get("param_meta") or {}).items()
                        if isinstance(v, (list, tuple)) and len(v) == 3},
            func=p.get("func", ""), code=p.get("code", ""),
        )
        ok, msg = US.register(s)
        if not ok:
            self._append(f"<p style='color:{C_BAD}'>注册失败：{_esc(msg)}</p>")
            self._set_state("策略注册失败", C_BAD)
            return

        self._append(
            f"<p style='color:{C_OK}'>已注册策略「{_esc(s.label)}」（{_esc(s.key)}）。"
            f"{_esc(msg)}</p>"
            f"<p style='color:{C_MUTED};font-size:11px'>"
            "在「策略研究」页的规则策略下拉里能选到它，参数可以拖滑杆调。定义存在 "
            f"<code>{_esc(US.path())}</code>，不想要就在左边选中它、点「删除选中策略」。</p>"
        )
        self._set_state(f"已注册策略 {s.key}", C_OK)
        self.reload()
        self.strategy_added.emit(s.key)

    # ---------------------------------------------------------- 删除自定义

    def _delete_factor(self) -> None:
        """删掉选中的自定义因子。"""
        from ttrade import userfactors as UF

        cur = self.lst_factor.currentItem()
        if cur is None:
            return
        key = str(cur.data(Qt.UserRole))
        spec = F.FACTORS.get(key)
        if spec is None or not _is_user_factor(key):
            return
        n = _refs_of_factor(key)
        warn = (
            f"\n\n有 {n} 份配置（策略库 / 交易组合）在用这个因子，删掉之后它们"
            "算不出来，得重新选因子。"
            if n else ""
        )
        ok = QMessageBox.question(
            self, "删除因子",
            f"删掉自定义因子「{spec.label}」（{key}）？{warn}",
        )
        if ok != QMessageBox.Yes:
            return
        if not UF.remove(key):
            self._append(f"<p style='color:{C_BAD}'>没删掉「{_esc(key)}」，库里找不到这一条。</p>")
            return

        self._append(
            f"<p style='color:{C_OK}'>已删除因子「{_esc(spec.label)}」。</p>"
            f"<p style='color:{C_MUTED};font-size:11px'>"
            "内置因子不受影响。已经存进策略库的配置不会自动改，重新跑之前记得换因子。</p>"
        )
        self._set_state(f"已删除因子 {key}", C_OK)
        self.reload()
        self.factor_removed.emit(key)

    def _delete_strategy(self) -> None:
        """删掉选中的自定义策略，或者策略库里「我的」那条模板。"""
        from ttrade import userstrategies as US

        cur = self.lst_strat.currentItem()
        if cur is None:
            return
        kind, key = cur.data(Qt.UserRole)
        if kind == "cfg":
            self._delete_library_item(str(key))
            return
        spec = STRATEGIES.get(key)
        if spec is None or not _is_user_strategy(key):
            return
        n = _refs_of_strategy(key)
        warn = (
            f"\n\n有 {n} 份配置（策略库 / 交易组合）在用这个策略，删掉之后它们"
            "跑不起来，得重新选策略。"
            if n else ""
        )
        ok = QMessageBox.question(
            self, "删除策略",
            f"删掉自定义策略「{spec.label}」（{key}）？{warn}",
        )
        if ok != QMessageBox.Yes:
            return
        if not US.remove(key):
            self._append(f"<p style='color:{C_BAD}'>没删掉「{_esc(key)}」，库里找不到这一条。</p>")
            return
        self._append(f"<p style='color:{C_OK}'>已删除策略「{_esc(spec.label)}」。</p>")
        self._set_state(f"已删除策略 {key}", C_OK)
        self.reload()
        self.strategy_removed.emit(key)

    def _delete_library_item(self, sid: str) -> None:
        """删策略库里的模板。**组合里跑着的实例不动** —— 那是两份独立的数据。"""
        cfg = next((c for c in SStore.load_library() if c.sid == sid), None)
        if cfg is None:
            return
        ok = QMessageBox.question(
            self, "删除策略模板",
            f"从策略库里删掉「{cfg.title()}」？\n\n"
            "「策略交易和组合」里已经加进去的实例不受影响，删的只是这个模板。",
        )
        if ok != QMessageBox.Yes:
            return
        if not SStore.remove_from_library(sid):
            self._append(f"<p style='color:{C_BAD}'>没删掉，库里找不到这条。</p>")
            return
        self._append(f"<p style='color:{C_OK}'>已从策略库删掉「{_esc(cfg.title())}」。</p>")
        self._set_state("已删除策略模板", C_OK)
        self.reload()

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
            "先判断平台已有的 5 套规则策略（双均线、唐奇安、ATR 通道、MACD、海龟）"
            "调参数能不能覆盖；覆盖不了就写成 strategy 动作块直接注册进策略库。\n"
            "另外说明：入场条件、离场条件、参数与默认值、适合的品种和行情。"
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

    def _open_manual(self) -> None:
        ManualDialog(self).exec()

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
        """重读因子库、策略库与研报目录。切回本页时调，能看到刚进库的东西。"""
        self._fill_factors()
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
        # 压缩线程没有 abort 手段（`MemoWorker` 不流式、拿不到 response），
        # 只能等它自己跑完这一轮；关窗口时最多等 2 秒，超时就算了
        if self._memo_thread is not None:
            self._memo_thread.quit()
            self._memo_thread.wait(2000)


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


def _is_user_strategy(key: str) -> bool:
    """内置策略的 `StrategySpec.code` 是空的，用户注册的把源码挂在上面。

    用这个而不是去读 `user_strategies.json`：列表每次切换都要渲染，为一个
    标记位读一次盘不值当，而且读盘失败会把自定义策略显示成「内置」。
    """
    spec = STRATEGIES.get(key)
    return bool(spec and spec.code)


def _is_user_factor(key: str) -> bool:
    """同 `_is_user_strategy`：自定义因子的源码挂在 `FactorSpec.code` 上。"""
    spec = F.FACTORS.get(key)
    return bool(spec and spec.code)


def _refs_of_factor(key: str) -> int:
    """有多少份配置在用这个因子。删之前得让用户知道会牵动什么。"""
    n = 0
    for cfg in SStore.load_portfolio() + SStore.load_library():
        if cfg.family == "factor" and key in (cfg.factor_keys or []):
            n += 1
    return n


def _refs_of_strategy(key: str) -> int:
    n = 0
    for cfg in SStore.load_portfolio() + SStore.load_library():
        if cfg.family == "rule" and cfg.rule_key == key:
            n += 1
    return n


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
