"""ChanAgent（Chan Quant Agent AI）页。

和「策略研究」「账户透视和交易」「策略交易和组合」并列的第四个大页面。

    ┌─────────────┬──────────────────────────┬──────────────┐
    │ 因子库       │                          │  详情面板     │
    │ 策略库       │      对话（大）           │  公式 / 代码  │
    │ 研报库       │                          │  逻辑 / 图例  │
    ├─────────────┤  ──────────────────────  ├──────────────┤
    │ 模型状态     │  [附件] [输入框]          │  运行输出     │
    │             │  [发送] [停止]            │  回测数字     │
    └─────────────┴──────────────────────────┴──────────────┘

四个库共用同一个详情面板：左边选中什么，右边就显示什么。对话是常驻的，切换库不会
把聊天记录弄丢 —— 用户经常是「看到一半想问一句」。

## 为什么右下角要有「运行输出」

回测的数字原来只出现在「策略研究」页的绩效卡上。模型跑完回测后只能凭记忆问
「结果是多少」，而用户手上根本没有一处可以复制粘贴的地方 —— 他得切到另一个
页面、从四张卡片里逐个抄。现在研究页算完就把结论推到这里，模型自己发起的那次
还会直接喂回给模型，它不用再问。

## 为什么文档走文本、图片走多模态

见 `ttrade.attachments`。一句话：图片没有文本等价物，只能让模型自己看；文档
本来就是文字，拼进消息最省事也最兼容。

## 为什么模型调用要单开线程

一次回答可能流式吐十几秒。放在主线程里，整个界面会僵住，连「停止」都点不动。
`ChatWorker` 跑在 QThread 里，每收到一段就 emit 一个信号，主线程只管往文档里追加。

## 为什么动作必须过确认框

模型说「买 10 手」和真的下单之间必须有一个人。确认框列出「买什么、多少手、为什么」，
用户勾「下次不提醒」才会记住不再问 —— 这个开关在设置里能重新打开。
关掉之后动作直接执行，所以它只适合已经反复确认过流程的人。
"""

from __future__ import annotations

import base64
import os
import re
import time

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QImage,
    QPixmap,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui import chat_render as CR
from ttrade import agent as A
from ttrade import agent_memory as Mem
from ttrade import attachments as ATT
from ttrade import factors as F
from ttrade import library_docs as L
from ttrade import session_files as SF
from ttrade import skills as S
from ttrade import strategy_store as SStore
from ttrade import strategy_templates as TPL
from ttrade.store import AnyStore
from ttrade.strategies import STRATEGIES
from ttrade.paths import user_dir

# 字号。这一页是拿自然语言做量化研究的地方，一屏要读很多字 —— 11px 的
# 「已提交」「已取消」看完要凑近屏幕，而正文比全局默认的 12px 还小。
# 统一从这里取，改一处整页跟着变。
FS_BODY = 15      # 对话正文、详情正文、输入框
FS_SMALL = 12     # 次要说明（已提交、已取消、工具回执）
FS_TINY = 11      # 附件条角标、缩略图下面的小字
FS_HEAD = 17      # 详情卡片标题
FS_CODE = 13      # 等宽代码
FS_UI = 14        # 按钮、下拉、列表

# 页面级样式。**只管 Agent 页** —— 别的页有各自的密度，一起放大会挤爆。
# 注意：Qt 里 widget 自己的 `setStyleSheet` 优先级高于这里，所以下面那些
# 写死在 QLabel 上的字号也要跟着改，否则会被覆盖回去。
AGENT_QSS = f"""
QWidget {{ font-size: {FS_UI}px; }}
QLabel#SectionTitle {{ font-size: 13px; }}
QLabel#Hint {{ font-size: {FS_SMALL}px; }}
QTextBrowser {{ font-size: {FS_BODY}px; }}
QPlainTextEdit {{ font-size: {FS_BODY}px; }}
QListWidget {{ font-size: {FS_UI}px; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ font-size: {FS_UI}px; }}
QPushButton {{ font-size: {FS_UI}px; }}
QTabBar::tab {{ font-size: {FS_UI}px; }}
"""

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


# ------------------------------------------------------------------ 输入框

_TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)
_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<br\s*/?>", re.I)


def html_table_to_md(html: str) -> str:
    """从剪贴板的 HTML 里抠表格，转成 markdown。

    用户从网页、Excel、Word 里框选表格复制时，剪贴板里除了纯文本还有一份
    HTML —— 纯文本那版是拿 Tab 隔开的，粘进来模型读得一塌糊涂（列对不齐、
    空单元格丢失）。这里直接读 `<table>`，转成 markdown 表格。

    只认第一张表：一次复制多张表的情况极少，而全拼进来会很长。
    """
    m = _TABLE_RE.search(html or "")
    if not m:
        return ""

    rows = []
    for tr in _ROW_RE.findall(m.group(0)):
        cells = []
        for cell in _CELL_RE.findall(tr):
            text = _BR_RE.sub(" ", cell)
            text = _TAG_RE.sub("", text).replace("\xa0", " ")
            text = re.sub(r"\s+", " ", text).strip()
            cells.append(text.replace("|", "\\|"))
        if any(cells):
            rows.append(cells)
    if len(rows) < 2:
        return ""

    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out) + "\n"


class ChatInput(QPlainTextEdit):
    """能直接接住截图、文件、网页表格的输入框。

    三件事分开走：

    - **截图 / 图片**：`mime.hasImage()` → 交给页面加进附件条。不落临时文件，
      直接编码进附件，因为 Qt 富文本认 `data:` URI，回显时能内联显示。
    - **文件**：`mime.hasUrls()` → 路径交给页面，走 `attachments.load()`。
    - **网页表格**：`mime.hasHtml()` 且里面有 `<table>` → 转 markdown 插进文本。

    剩下的（普通文字）交回父类，别抢走默认的粘贴行为。
    """

    files_dropped = Signal(list)      # [str 路径] 或 [QImage]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    # ---------------------------------------------------------- 粘贴

    def canInsertFromMimeData(self, src) -> bool:
        if src.hasImage() or src.hasUrls():
            return True
        if src.hasHtml() and _TABLE_RE.search(src.html() or ""):
            return True
        return super().canInsertFromMimeData(src)

    def insertFromMimeData(self, src) -> None:
        img = self._image_of(src)
        if img is not None:
            self.files_dropped.emit([img])
            return

        paths = self._paths_of(src)
        if paths:
            self.files_dropped.emit(paths)
            return

        if src.hasHtml():
            md = html_table_to_md(src.html())
            if md:
                self.insertPlainText(md)
                return

        super().insertFromMimeData(src)

    # ---------------------------------------------------------- 拖放

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasImage() or e.mimeData().hasUrls():
            e.acceptProposedAction()
            return
        super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasImage() or e.mimeData().hasUrls():
            e.acceptProposedAction()
            return
        super().dragMoveEvent(e)

    def dropEvent(self, e) -> None:
        img = self._image_of(e.mimeData())
        if img is not None:
            self.files_dropped.emit([img])
            e.acceptProposedAction()
            return

        paths = self._paths_of(e.mimeData())
        if paths:
            self.files_dropped.emit(paths)
            e.acceptProposedAction()
            return
        super().dropEvent(e)

    # ---------------------------------------------------------- 小工具

    @staticmethod
    def _image_of(md):
        """剪贴板里的图。`imageData()` 可能给 QImage 也可能给 QPixmap，都接住。"""
        if not md.hasImage():
            return None
        data = md.imageData()
        if isinstance(data, QImage) and not data.isNull():
            return data
        if isinstance(data, QPixmap) and not data.isNull():
            return data.toImage()
        return None

    @staticmethod
    def _paths_of(md) -> list[str]:
        if not md.hasUrls():
            return []
        out = []
        for u in md.urls():
            if not u.isLocalFile():
                continue              # 从浏览器拖一个链接过来，不是文件
            p = u.toLocalFile()
            if p and os.path.isfile(p):
                out.append(p)
        return out


# ------------------------------------------------------------------ 线程


class ProfileWorker(QObject):
    """跑在 QThread 里：整库因子体检。

    只做**计算**，不碰 Qt 的富文本 / 绘图对象 —— `QTextDocument` 和 `QPdfWriter`
    不是线程安全的，渲染 PDF 那一步必须回主线程（实测 0.3 秒，不会卡）。
    分析本身十几秒，正好放进线程。
    """

    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, years: int = 10) -> None:
        super().__init__()
        self.years = years

    def run(self) -> None:
        try:
            from ttrade import factor_profile as FP

            res = FP.build(years=self.years,
                           progress=lambda d, t, m: self.progress.emit(d, t, m))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(res)


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


# 确认框里 payload 字段的中文名。没列到的照抄字段名 —— 宁可露出 `file_id`
# 这种半生不熟的词，也不要把一个字段藏起来让人盲签。
FIELD_CN = {
    "reason": "理由", "symbol": "品种", "period": "周期", "strategy": "策略",
    "params": "参数", "volume": "手数", "side": "方向", "offset": "开平",
    "price": "价格", "path": "路径", "url": "网址", "title": "标题",
    "name": "名称", "sheet": "工作表", "sheets": "工作表", "content": "正文",
    "code": "代码", "file_id": "附件", "op": "操作", "factor": "因子",
    "rows": "行数", "cols": "列", "desc": "说明", "key": "代号",
    "category": "类别", "formula": "公式", "mode": "方式", "rule": "规则",
    "markdown": "正文",
}
# 先按这个顺序排，没列到的接在后面
FIELD_ORDER = ["reason", "symbol", "side", "offset", "volume", "price", "period",
               "strategy", "params", "path", "url", "op", "name", "title",
               "sheet", "sheets", "rows", "cols", "file_id", "factor", "rule"]
# 超过这个长度的值不给 QLabel（会撑爆对话框），单独给一个只读文本框
LONG_FIELD = 400


def _payload_rows(p: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """把动作参数摊成确认框里的 `(字段, 值)` 行，外加要单独给文本框的长内容。

    顺序固定、全字段可见。确认框的唯一职责就是「让人在点确定之前看清要发生什么」，
    所以宁可多列一行，也不要为了好看把字段藏掉。
    """
    side_cn = {"BUY": "买入", "SELL": "卖出"}
    try:
        from ttrade.skills.trade import OFFSET_CN
    except Exception:                # noqa: BLE001
        OFFSET_CN = {}

    keys = [k for k in FIELD_ORDER if k in p] + [k for k in p if k not in FIELD_ORDER]
    rows: list[tuple[str, str]] = []
    longs: list[tuple[str, str]] = []
    for k in keys:
        v = p.get(k)
        if v is None or v == "" or v == [] or v == {}:
            continue
        if k == "exchange":
            continue                 # 已经并进「品种」那一行
        if k == "period" and p.get("mode") == "manual" and not p.get("strategy"):
            continue                 # 手动单不看周期，写出来只是噪音
        if k == "symbol":
            text = f"{v} {p.get('exchange') or ''}".strip()
        elif k == "side":
            text = side_cn.get(str(v), str(v))
        elif k == "offset":
            text = OFFSET_CN.get(str(v), str(v))
        elif k == "mode":
            text = {"manual": "手动", "strategy": "按策略"}.get(str(v), str(v))
        elif k == "strategy":
            text = _strategy_text(str(v), p.get("params") or {})
        elif isinstance(v, bool):
            text = "是" if v else "否"
        elif isinstance(v, dict):
            text = "　".join(f"{a}={b}" for a, b in v.items()) or "（空）"
        elif isinstance(v, (list, tuple)):
            text = "、".join(str(x) for x in v)
        else:
            text = str(v)
        label = FIELD_CN.get(k, k)
        (longs if len(text) > LONG_FIELD else rows).append((label, text))
    return rows, longs


class ConfirmDialog(QDialog):
    """执行模型给的动作之前确认一次。"""

    def __init__(self, action: A.Action, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("确认执行")
        self.setMinimumWidth(440)
        lay = QVBoxLayout(self)

        head = QLabel(action.describe())
        head.setStyleSheet(f"font-size: {FS_HEAD}px; font-weight: 700; color: {C_TEXT};")
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
            code.setStyleSheet(f"font-family: Consolas, monospace; font-size: {FS_CODE}px;")
            lay.addWidget(code)

            warn = QLabel("确认后写入因子库并立刻可用；代码有问题会注册失败并给出原因。")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {C_BAD}; font-size: {FS_SMALL}px;")
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
            code.setStyleSheet(f"font-family: Consolas, monospace; font-size: {FS_CODE}px;")
            lay.addWidget(code)

            warn = QLabel("确认后写入策略库，之后可以在「策略研究」页选它回测或实盘；"
                          "代码跑不通会注册失败并给出原因。")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {C_BAD}; font-size: {FS_SMALL}px;")
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
            body.setStyleSheet(f"font-family: Consolas, monospace; font-size: {FS_CODE}px;")
            lay.addWidget(body)

            warn = QLabel(
                "确认后整个文件被替换，没有撤销。改的是程序自己的代码时，"
                "重启之后平台可能起不来。"
            )
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {C_BAD}; font-size: {FS_SMALL}px;")
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

        # 其余动作（下单 / 导出 / 出网 / 写表格）统一按 payload 摊开。
        # 以前这里只认「理由/品种/周期/策略/手数」五个字段，是给下单写的；
        # 后来 `write_pdf` 之类的写动作也走这条路，弹框里只剩一句「确认后立即执行」，
        # 连写到哪个路径都看不见 —— 等于让人盲签。改成把 payload 全摊开：
        # 字段名有中文就用中文，没有就照抄；太长的（正文/代码）单独给一个只读框。
        rows, longs = _payload_rows(p)
        if rows:
            form = QFormLayout()
            form.setContentsMargins(0, 6, 0, 6)
            for k, v in rows:
                lab = QLabel(v)
                lab.setWordWrap(True)
                form.addRow(QLabel(k), lab)
            lay.addLayout(form)

        for title, text in longs:
            lay.addWidget(QLabel(f"{title}（共 {len(text)} 字符）"))
            body = QPlainTextEdit(text)
            body.setReadOnly(True)
            body.setMinimumHeight(200)
            body.setStyleSheet(f"font-family: Consolas, monospace; font-size: {FS_CODE}px;")
            lay.addWidget(body)

        warn = QLabel("确认后立即执行，没有撤销。")
        warn.setStyleSheet(f"color: {C_BAD}; font-size: {FS_SMALL}px;")
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
            "「ChanAgent」查资料时看到的就是这份内容：行情从哪来、平台注入了哪些"
            "原语、每个因子的完整实现。它自己也会用 doc 动作按需取其中一段。"
        )
        head.setWordWrap(True)
        head.setObjectName("Hint")
        lay.addWidget(head)

        box = QPlainTextEdit(self.text)
        box.setReadOnly(True)
        box.setStyleSheet(
            f"font-family: Consolas, 'Microsoft YaHei', monospace; font-size: {FS_CODE}px;"
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
        self.lbl_info.setStyleSheet(f"color: {C_OK if ok else C_BAD}; font-size: {FS_SMALL}px;")


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
        self.chk_fs = QCheckBox("允许 Agent 读写 ChanQuant 目录（升级权限）")
        self.chk_fs.setChecked(cfg.allow_fs)
        self.chk_fs.setToolTip(
            "关着：Agent 只能下单、回测、注册因子和策略。\n"
            "打开：多一个文件动作，可读写 ChanQuant 根目录下的任意文件。"
        )
        self.chk_fs.toggled.connect(self._on_fs_toggled)
        form.addRow("", self.chk_fs)

        # 能力包开关。关掉的包，提示词里就不列它的工具 —— 模型不会写出一个
        # 必然被拒的动作块，用户也不会看到「没有权限」这种没头没尾的提示。
        # `fs` 不在这里：它由上面那个「升级权限」单独管，语义不一样。
        self._group_boxes: dict[str, QCheckBox] = {}
        grp = QGroupBox("能力包")
        glay = QVBoxLayout(grp)
        glay.setContentsMargins(8, 6, 8, 6)
        glay.setSpacing(1)
        for key in S.GROUP_ORDER:
            if key == "fs" or not S.group_tools(key):
                continue
            meta = S.GROUPS.get(key) or {}
            cb = QCheckBox(f"{meta.get('label', key)}　{meta.get('desc', '')}")
            cb.setChecked(key in cfg.enabled_groups)
            self._group_boxes[key] = cb
            glay.addWidget(cb)
        form.addRow(grp)

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
            enabled_groups=[k for k, cb in self._group_boxes.items() if cb.isChecked()],
        )

    def _on_fs_toggled(self, on: bool) -> None:
        """打开升级权限前必须警告一次。关掉不拦 —— 降权永远安全。"""
        if not on:
            return
        ok = QMessageBox.warning(
            self, "升级权限",
            "打开之后，Agent 可以读写 ChanQuant 目录下的任何文件 —— 包括程序"
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
            self.lbl_test.setStyleSheet(f"color: {C_BAD}; font-size: {FS_SMALL}px;")
            return
        self.lbl_test.setText("测试中…")
        self.lbl_test.setStyleSheet(f"color: {C_MUTED}; font-size: {FS_SMALL}px;")
        self.btn_test.setEnabled(False)
        # 界面线程里同步等一次响应（最多 30 秒）。测试按钮是低频操作，
        # 为它再开一个线程不值当，但要把按钮禁掉防止连点。
        try:
            msg = A.LLMClient(cfg).ping()
            self.lbl_test.setText(f"连接正常：{msg}")
            self.lbl_test.setStyleSheet(f"color: {C_OK}; font-size: {FS_SMALL}px;")
        except Exception as exc:  # noqa: BLE001
            self.lbl_test.setText(str(exc)[:160])
            self.lbl_test.setStyleSheet(f"color: {C_BAD}; font-size: {FS_SMALL}px;")
        finally:
            self.btn_test.setEnabled(True)


# ------------------------------------------------------------------ 会话资源

# 这几类走「资源化」：解析一次存进 `SessionFiles`，消息里只放摘要，
# 模型要更多就调 `read_excel` / `parse_pdf` 按 file_id 再取。
# 其余（md / txt / json / 图片）照旧整段内联 —— 它们本来就是一篇短文，
# 再包一层资源反而多一次往返。
SESSION_SUFFIXES = SF.TABLE_SUFFIXES | {".pdf", ".docx"}


class SessAtt:
    """附件条上的一个「会话资源」。

    把 `SessionFile` 包成 `ATT.Attachment` 的样子，附件条、`_att_chip`、
    `ATT.describe()` 都不用改 —— 它们只认 `name` / `kind` / `note` /
    `size_text()`。`kind` 统一报 `"text"`，这样它会被算进「几份文档」，
    而不会走进 `ATT.content()` 的文本内联分支（那条分支要 `.text` 字段）。
    """

    kind = "text"

    def __init__(self, item: "SF.SessionFile") -> None:
        self.item = item
        self.name = item.name
        self.path = item.path
        self.note = item.note
        self.file_id = item.file_id

    @property
    def ok(self) -> bool:
        return self.item.ok

    def size_text(self) -> str:
        if self.item.kind == "table":
            df = self.item.frame()
            return f"{len(df)} 行 × {len(df.columns)} 列"
        return f"{len(self.item.text)} 字符"


# ------------------------------------------------------------------ 页面


class AgentPage(QWidget):
    """ChanAgent。"""

    study_requested = Signal(object)      # SStore.StrategyConfig
    trade_requested = Signal(object)      # A.Action
    portfolio_requested = Signal(object)  # SStore.StrategyConfig
    factor_added = Signal(str)            # 新注册的因子 key，主窗口据此刷新下拉
    strategy_added = Signal(str)          # 新注册的策略 key，同上
    factor_removed = Signal(str)          # 删掉的因子 key，主窗口重建下拉
    strategy_removed = Signal(str)        # 删掉的策略 key，同上

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(AGENT_QSS)
        self.cfg = A.load_config()
        self._history: list[dict] = []
        self._worker: ChatWorker | None = None
        self._thread: QThread | None = None
        self._busy = False
        self._pending_report = ""     # 正在生成的研报品种，回完自动落盘
        self._stream_buf = ""
        # 当前这条回答在文档里的起始位置（流式收尾时从这里整段重排成富文本）
        self._msg_start: int | None = None
        self._memo_worker: MemoWorker | None = None
        self._memo_thread: QThread | None = None
        self._memo_warned = False     # 压缩失败只提示一次，别每轮都刷
        self._tool_turns = 0            # 本轮提问已经连续自动跑了几次工具动作（查手册 / 读写文件）
        self._atts: list = []          # 待发送的附件（`ATT.Attachment` 或 `SessAtt`）
        self.session = SF.SessionFiles()        # 会话里的附件资源（表格 / PDF / Word）
        self.store = AnyStore({})               # 主窗口把两个市场的 DataStore 递进来
        self._out: list[str] = []      # 运行输出（整段重绘，见 `push_output`）
        self._out_merge = False        # 上一条输出是否允许被覆盖（拖滑杆时别刷屏）
        self._last_output = ""         # 最近一次输出原文，给「发给 Agent」用
        self._expect_bt = False        # 下一次回测结果要自动喂回模型
        self._build()
        self._refresh_status()
        self._fill_factors()
        self._fill_strategies()
        self._fill_reports()
        self._render_welcome()

    def attach(self, stores) -> None:
        """主窗口把 DataStore 递进来 —— 查行情、列品种都要用它。

        传 dict（`{"cn": ..., "us": ...}`）而不是单个 store：Agent 页不跟市场走，
        用户可能在研究页看美股、在这里问螺纹，所以要能同时看到两边。
        """
        self.store = AnyStore(stores)

    def _ctx(self) -> "S.SkillContext":
        """工具执行上下文。三个东西：附件资源、行情库、开关配置。"""
        return S.SkillContext(session=self.session, store=self.store, cfg=self.cfg)

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

        # 因子手册 / 审核 / 深度分析：前两个是「问模型」，最后一个是「出一份 PDF」。
        # 审核和分析都是把一段结构化提示词塞进对话框再发出去 —— 用户能看见自己
        # 问了什么，也能接着追问，比弹一个黑盒窗口有用。
        row = QHBoxLayout()
        row.setSpacing(4)
        self.btn_audit = QPushButton("因子审核")
        self.btn_audit.setObjectName("Ghost")
        self.btn_audit.setToolTip(
            "让模型对选中因子做 PIT 审核：有没有用到未来数据、衍生数据是否过旧、"
            "夜盘 bar 的交易日归属是否合理、时段边界有没有缝"
        )
        self.btn_audit.setEnabled(False)
        self.btn_audit.clicked.connect(self._audit_factor)
        row.addWidget(self.btn_audit)

        self.btn_deep = QPushButton("因子深度分析")
        self.btn_deep.setObjectName("Ghost")
        self.btn_deep.setToolTip(
            "让模型对选中因子做深度分析：快因子还是慢因子、适合什么品种和周期、"
            "和谁搭组合、稳健性如何。有实测数字会一并交给它"
        )
        self.btn_deep.setEnabled(False)
        self.btn_deep.clicked.connect(self._deep_factor)
        row.addWidget(self.btn_deep)
        lay.addLayout(row)

        self.btn_handbook = QPushButton("因子手册（PDF）")
        self.btn_handbook.setObjectName("Ghost")
        self.btn_handbook.setToolTip(
            "对整库因子跑一遍体检，生成一份 PDF：快慢、品种、周期、组合、稳健性。\n"
            "约十几秒，跑完自动打开。"
        )
        self.btn_handbook.clicked.connect(self._make_handbook)
        lay.addWidget(self.btn_handbook)

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

        # 附件条：只在有附件时出现，免得平时白占一行
        self.att_box = QWidget()
        self.att_lay = QHBoxLayout(self.att_box)
        self.att_lay.setContentsMargins(0, 0, 0, 0)
        self.att_lay.setSpacing(4)
        self.att_box.setVisible(False)
        lay.addWidget(self.att_box)

        self.ed_input = ChatInput()
        self.ed_input.setPlaceholderText(
            "比如：螺纹现在适合做多吗？用双均线帮我回测一下，1 手。"
            "（Ctrl+Enter 发送；截图可以直接 Ctrl+V 粘，文件可以直接拖进来）"
        )
        self.ed_input.setFixedHeight(84)
        self.ed_input.files_dropped.connect(self._on_dropped)
        lay.addWidget(self.ed_input)

        row = QHBoxLayout()
        self.btn_attach = QPushButton("附件")
        self.btn_attach.setObjectName("Ghost")
        self.btn_attach.setToolTip(
            "添加图片（截图、行情图）或文档（md/txt/csv/xlsx/pdf/docx）。\n"
            "也可以直接把截图粘进输入框、把文件拖进来。\n"
            "图片需要模型支持视觉（GPT-4o、GLM-4V、Qwen-VL 等），纯文本模型会报错。"
        )
        self.btn_attach.clicked.connect(self._attach_files)
        row.addWidget(self.btn_attach)
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

        # 上下分：详情在上，运行输出在下。输出单独一块而不是塞进对话流 ——
        # 回测结论是「看数字」的，混在聊天记录里翻起来太费劲，而且模型每轮
        # 都要重新问一遍「结果是多少」。
        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(6)
        t = QLabel("详情")
        t.setObjectName("SectionTitle")
        tl.addWidget(t)
        self.detail = QTextBrowser()
        tl.addWidget(self.detail, 1)
        split.addWidget(top)
        split.addWidget(self._build_output())
        split.setSizes([420, 240])
        lay.addWidget(split, 1)
        return card

    def _build_output(self) -> QWidget:
        """回测 / 计算结果的输出框。

        以前这些数字只出现在「策略研究」页的绩效卡上，Agent 说「把结果贴给我」
        时用户根本没地方找。这里既给人看，也能一键发回给模型。
        """
        card = QWidget()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        head = QHBoxLayout()
        t = QLabel("运行输出")
        t.setObjectName("SectionTitle")
        head.addWidget(t)
        self.lbl_out = QLabel("")
        self.lbl_out.setObjectName("Hint")
        head.addWidget(self.lbl_out, 1)
        self.btn_out_feed = QPushButton("发给 Agent")
        self.btn_out_feed.setObjectName("Ghost")
        self.btn_out_feed.setToolTip("把最近一次输出作为消息发给模型，它就不用再问你结果是什么了")
        self.btn_out_feed.setEnabled(False)
        self.btn_out_feed.clicked.connect(self._feed_last_output)
        head.addWidget(self.btn_out_feed)
        btn_clear = QPushButton("清空")
        btn_clear.setObjectName("Ghost")
        btn_clear.clicked.connect(self._clear_output)
        head.addWidget(btn_clear)
        lay.addLayout(head)

        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setPlaceholderText(
            "回测跑完，这里会出现收益、回撤、夏普、交易笔数。\n"
            "Agent 让你回测时，结果会自动发回给它。"
        )
        self.out.setStyleSheet("font-family: Consolas, 'Microsoft YaHei', monospace;"
                               f"font-size: {FS_SMALL}px;")
        lay.addWidget(self.out, 1)
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
        """策略库列表：内置模板按分组排前面，规则策略和「我的」跟在后面。

        分组标题行是 `NoItemFlags` 的：看得见、选不中。没有它，34 条模板和一堆
        规则策略挤成一列，用户看不出「这几条是一组的、那几条是另一回事」——
        而这批模板的意义恰恰是「一组一组对比着看」。

        「我的」那一组里，原来的 `· 我的` 尾巴换成了组标题：名字本身就是用户起的，
        后面挂个后缀，他在搜索框里照着抄就永远对不上（`norm_name` 得专门剥这截）。
        """
        self.lst_strat.clear()
        for gkey, glabel, tkeys in TPL.groups_with_items():
            self._add_group_row(glabel, TPL.GROUP_DESC.get(gkey, ""))
            for k in tkeys:
                cfg = TPL.to_config(k)
                if cfg is None:
                    continue
                it = QListWidgetItem(f"　{cfg.title()}")
                it.setData(Qt.UserRole, ("tpl", k))
                it.setToolTip(f"{TPL.summary(k)}\n\n{TPL.teach(k)}")
                self.lst_strat.addItem(it)

        self._add_group_row("规则策略", "不用因子，纯规则。内置的改代码才会消失，"
                                       "自定义的可以删。")
        for d in L.all_strategy_docs():
            tag = "自定义" if _is_user_strategy(d.key) else "内置"
            it = QListWidgetItem(f"　{d.label}　·　{tag}")
            it.setData(Qt.UserRole, ("rule", d.key))
            it.setToolTip(d.desc)
            self.lst_strat.addItem(it)

        mine = SStore.load_library()
        self._add_group_row(
            "我的策略库",
            "自己存下来的配置。内置模板载入研究页、改完参数点「存入策略库」，"
            "副本就落在这里。",
        )
        if not mine:
            self._add_group_row(
                "（还是空的。选一条内置模板，点下面的「回测」载入研究页改）",
                "载入只是填界面，不会改模板；改完想留住就点「存入策略库」。",
                hint=True,
            )
        for cfg in mine:
            it = QListWidgetItem(f"　{cfg.title()}")
            it.setData(Qt.UserRole, ("cfg", cfg.sid))
            it.setToolTip(cfg.describe())
            self.lst_strat.addItem(it)

    def _add_group_row(self, label: str, tip: str = "", hint: bool = False) -> None:
        """列表里的分组标题。选不中，只是分隔和说明。"""
        it = QListWidgetItem(f"　{label}" if hint else f"— {label} —")
        it.setFlags(Qt.NoItemFlags)
        it.setData(Qt.UserRole, ("head", label))
        if tip:
            it.setToolTip(tip)
        font = it.font()
        font.setBold(not hint)
        it.setFont(font)
        if not hint:
            it.setForeground(QColor(C_MUTED))
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
            # 关掉的能力包也报一声：否则用户会以为模型变笨了，而不是「这个包我关了」
            off = [S.GROUPS[k]["label"] for k in S.GROUPS
                   if k != "fs" and S.group_tools(k)
                   and k not in self.cfg.enabled_groups]
            if off:
                extra += "｜已关：" + "、".join(off)
            self.lbl_model.setText(f"{p.label} · {self.cfg.model}{extra}")
            self.lbl_model.setStyleSheet(
                f"color: {C_BAD if self.cfg.allow_fs else C_OK}; font-size: {FS_SMALL}px;"
            )
        else:
            self.lbl_model.setText(f"未配置：{why}")
            self.lbl_model.setStyleSheet(f"color: {C_BAD}; font-size: {FS_SMALL}px;")

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
        # 审核 / 深度分析只要求「选中了某个因子」，自定义和内置都能问
        for b in (getattr(self, "btn_audit", None), getattr(self, "btn_deep", None)):
            if b is not None:
                b.setEnabled(cur is not None)
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

    def _show_template(self, key: str) -> None:
        """内置模板的详情：先说「这条在演示什么」，再说配方。

        顺序是刻意的。模板的价值不在「能跑」，在于它示范了一个手法；把
        「演示什么」摆在参数前面，用户才知道该看哪儿 —— 否则 24 个因子名
        列在那儿，看完只记住「因子很多」。
        """
        cfg = TPL.to_config(key)
        if cfg is None:
            self.detail.setHtml("<p>这条内置模板找不到了（版本升级时改过名？）。</p>")
            return
        group = TPL.GROUP_LABEL.get(TPL.group_of(key), "内置模板")
        html = [
            f"<h3 style='color:{C_TEXT};margin:0 0 4px 0'>{cfg.title()}</h3>",
            # 摘要（几个因子、方向、取反数）不写在这里：下面的「因子组合」块
            # 说得更细，两处都写就会撞出「单因子 · 动量 · 单因子 · 方向自适应」
            f"<p style='color:{C_MUTED};margin:0 0 10px 0'>内置模板 · {group}</p>",
        ]
        teach = TPL.teach(key)
        if teach:
            html.append(_block("这条在演示什么", _esc(teach)))
        html += [
            _block("标的", f"{cfg.key()}　{cfg.period}"),
            _block("因子组合", _factor_mix(cfg)),
            _block("参数", _esc(TPL.knobs(cfg))),
            _block("仓位", f"{cfg.size_text()}　（{_sizing_cn(cfg.sizing)}）"),
        ]
        gdesc = TPL.GROUP_DESC.get(TPL.group_of(key), "")
        if gdesc:
            html.append(_block("这一组在讲什么", _esc(gdesc)))
        html.append(
            f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px;margin-top:10px'>"
            "模板是只读的。点下面的「回测」把它载入研究页，改完参数再"
            "「存入策略库」，副本会出现在「我的策略库」里 —— 模板本身不会变。</p>"
        )
        self.detail.setHtml("".join(html))

    def _show_strategy(self, cur, _prev) -> None:
        kind, key = cur.data(Qt.UserRole) if cur is not None else ("", "")
        # 内置模板和内置规则策略都删不掉；自定义策略和「我的」模板可以删
        self.btn_del_strat.setEnabled(
            cur is not None and (kind == "cfg" or (kind == "rule" and _is_user_strategy(key)))
        )
        if cur is None or kind == "head":
            return
        if kind == "tpl":
            self._show_template(key)
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
        elif cfg.family == "cross":
            # 截面策略的「标的」不是一个品种，是一篮子。`cfg.key()` 只写了第一
            # 个，照着看会以为它只交易那一个
            syms = list(cfg.cross_symbols or [])
            html.append(_block(
                "品种池", f"{len(syms)} 个：{'、'.join(syms)}" if syms else "—"))
            html.append(_block(
                "说明", "截面策略目前只做回测：要同时挂一篮子品种、按权重整体调仓，"
                        "而实盘引擎一次只对一个合约。先在「策略研究」里把参数调好，"
                        "存进策略库，双击就能看回测。"))
        else:
            # 这句以前写的是「因子策略暂不支持实时交易，只能回测」—— 早就不是
            # 这样了：单因子 / 多因子在「策略交易和组合」里能直接挂实时，只有
            # 机器学习模式因为每根 K 线要重训而只做回测。说明写错会让人白放弃
            # 一条能跑的链路。
            mode = {"single": "单因子", "multi": "多因子", "ml": "机器学习"}.get(
                cfg.factor_mode, cfg.factor_mode or "因子"
            )
            if cfg.factor_mode == "ml":
                body = ("机器学习多因子只能回测：每根 K 线都要重训，实时跟不上。"
                        "要做实盘请把因子模式改成单因子或多因子。")
            else:
                body = (f"{mode}因子策略支持实时交易 —— 在「策略交易和组合」里"
                        "把它挂上「启动交易」即可，信号按每根走完的 K 线重算。"
                        "因子过滤（规则策略 + 因子否决）同样可以实盘。")
            html.append(_block("说明", body))
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
        self.detail.setHtml(f"<pre style='white-space:pre-wrap;font-size:{FS_CODE}px'>"
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
            f"我是这个平台里的 ChanAgent，能查数据、解释因子、跑回测、按你的话下单。<br>"
            f"{tip}{mem_line}</p>"
        )

    def _clear_chat(self) -> None:
        self._history.clear()
        # 附件资源跟对话一起作废：模型手里那些 file_id 一旦清空对话就指不到了，
        # 留着只会让下一轮的 `read_excel(f1)` 读到上一轮的表
        self.session.clear()
        self._atts = []
        self._refresh_atts()
        self._msg_start = None       # setHtml 会把文档整个换掉，旧位置作废
        self._render_welcome()

    def _append(self, html: str) -> None:
        """往对话区尾部追加一段 HTML。

        `insertHtml` 在「光标停在最后一段的末尾」时，会把片段里的**第一个块并进
        当前段落** —— 于是「你」「助手」会粘在上一句后面，`## 结论` 这种标题也
        缩成行内，整段对话糊成一片。所以插入前先看看当前块里有没有字，有就断开。
        """
        cur = self.log.textCursor()
        cur.movePosition(QTextCursor.End)
        if cur.block().text().strip():
            cur.insertBlock()
        self.log.setTextCursor(cur)
        self.log.insertHtml(html)
        self.log.moveCursor(QTextCursor.End)

    def _send(self) -> None:
        if self._busy:
            return
        text = self.ed_input.toPlainText().strip()
        if not text and not self._atts:
            return
        ok, why = self.cfg.ready()
        if not ok:
            # 配置没就绪时**不能**清空输入框和附件 —— 弹完设置回来还得重打一遍
            self._set_state(why, C_BAD)
            self._open_settings()
            return

        atts, self._atts = self._atts, []
        self._refresh_atts()
        self.ed_input.clear()
        if not text:
            # 只丢了个附件、一句话没说。给一句占位，别让模型收到一条空消息
            text = "看一下这个附件。"
        self._append(
            f"<p style='margin:12px 0 2px 0'><b style='color:{C_USER}'>你</b></p>"
            f"<p style='margin:0 0 8px 0;line-height:160%'>{_esc(text)}</p>"
        )
        if atts:
            self._append(
                f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px;margin:0 0 8px 0'>"
                f"{_esc(ATT.describe(atts))}</p>"
            )
        self._append(f"<p style='margin:12px 0 2px 0'><b>助手</b></p>")

        # 有图就是多模态 parts，没图就是纯字符串 —— 老网关只吃字符串。
        # 表格/PDF/Word 不在正文里展开，只附一段摘要（见 `_compose`）
        body = self._compose(text, atts)
        self._history.append({"role": "user", "content": body})
        self._tool_turns = 0
        # 新问题把上一轮的「等回测结果」作废：上一轮的回测要是没跑起来（异常提前
        # 返回），标记会一直挂着，下一次用户自己拖参数就会莫名其妙触发一轮对话
        self._expect_bt = False
        msgs = A.build_messages(
            A.trim_history(self._history[:-1]), body, memory=Mem.brief(),
            allow_fs=self.cfg.allow_fs, enabled=self.cfg.enabled_groups,
        )
        self._stream_buf = ""
        self._msg_start = None
        self._start_worker(msgs)

    # ---------------------------------------------------------- 附件

    def _attach_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "添加附件", "", ATT.file_filter()
        )
        if paths:
            self._add_paths(paths)

    def _on_dropped(self, items: list) -> None:
        """输入框粘/拖进来的东西。里面可能混着路径和 QImage。"""
        paths = [x for x in items if isinstance(x, str)]
        imgs = [x for x in items if not isinstance(x, str)]
        if paths:
            self._add_paths(paths)
        if imgs:
            self._add_images(imgs)

    def _compose(self, text: str, atts: list) -> str | list:
        """把「用户这句话 + 附件」拼成 `content`。

        两类附件分开处理：图片和短文照旧内联（模型直接看 / 直接读），
        表格和 PDF、Word 只放一段摘要，正文留给 `read_excel` / `parse_pdf` 现取。
        摘要放在文字部件的末尾而不是单独发一条消息 —— 单独发会让模型以为
        那是用户说的另一句话。
        """
        inline = [a for a in atts if not isinstance(a, SessAtt)]
        body = ATT.content(text, inline)
        summary = self.session.summary()
        if not summary:
            return body
        if isinstance(body, list):
            out = []
            for part in body:
                if isinstance(part, dict) and part.get("type") == "text":
                    out.append({"type": "text", "text": part["text"] + "\n\n" + summary})
                else:
                    out.append(part)
            return out
        return f"{body}\n\n{summary}".strip()

    def _add_paths(self, paths: list[str]) -> None:
        bad = []
        for p in paths:
            if any(a.path == p for a in self._atts):
                continue            # 同一个文件加两次没意义，还白占体积
            ext = os.path.splitext(p)[1].lower()
            if ext in SESSION_SUFFIXES:
                item = self.session.add_path(p)
                if item is not None:
                    self._atts.append(SessAtt(item))
                else:
                    bad.append(f"{os.path.basename(p)}：读不了（表格请存成 xlsx/csv，"
                               "PDF 要是文字版而不是扫描件）")
                continue
            a = ATT.load(p)
            if a.ok:
                self._atts.append(a)
            else:
                bad.append(f"{a.name}：{a.note}")
        self._finish_add(bad)

    def _add_images(self, imgs: list) -> None:
        bad = []
        for img in imgs:
            n = sum(1 for a in self._atts if a.kind == "image") + 1
            a = ATT.from_qimage(img, f"截图 {n}.png")
            if a.ok:
                self._atts.append(a)
            else:
                bad.append(f"截图：{a.note}")
        self._finish_add(bad)

    def _finish_add(self, bad: list[str]) -> None:
        self._refresh_atts()
        if bad:
            # 读不了的文件不进附件条，否则发出去模型也只看到一行「打不开」
            QMessageBox.warning(self, "这些加不上", "\n".join(bad))
        if self._atts:
            self._set_state(f"已添加 {len(self._atts)} 个附件", C_MUTED)

    def _refresh_atts(self) -> None:
        while self.att_lay.count():
            it = self.att_lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for i, a in enumerate(self._atts):
            self.att_lay.addWidget(self._att_chip(i, a))
        self.att_lay.addStretch(1)
        self.att_box.setVisible(bool(self._atts))

    def _att_chip(self, idx: int, a: "ATT.Attachment") -> QWidget:
        """一个附件小卡片：图片给缩略图，文档给类型角标。

        光写文件名的话，用户同时丢两张截图根本分不清哪张是哪张 —— 缩略图
        比任何文字都直接。
        """
        chip = QFrame()
        chip.setObjectName("AttChip")
        chip.setStyleSheet(
            "#AttChip{border:1px solid #e2e8f0;border-radius:4px;background:#f8fafc}"
        )
        lay = QHBoxLayout(chip)
        lay.setContentsMargins(4, 3, 4, 3)
        lay.setSpacing(6)

        thumb = QLabel()
        thumb.setFixedSize(34, 34)
        thumb.setAlignment(Qt.AlignCenter)
        pix = self._thumb(a)
        if pix is not None:
            thumb.setPixmap(pix)
        else:
            thumb.setText("PDF" if a.name.lower().endswith(".pdf")
                          else ("DOC" if a.name.lower().endswith((".docx", ".doc")) else "TXT"))
            thumb.setStyleSheet(
                f"color:{C_MUTED};font-size:{FS_TINY}px;font-weight:700;"
                "background:#eef2f7;border-radius:3px"
            )
        lay.addWidget(thumb)

        text = QWidget()
        col = QVBoxLayout(text)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        name = QLabel(a.name if len(a.name) <= 16 else a.name[:14] + "…")
        name.setStyleSheet(f"color:{C_TEXT};font-size:{FS_SMALL}px")
        size = QLabel(a.size_text())
        size.setStyleSheet(f"color:{C_MUTED};font-size:{FS_TINY}px")
        col.addWidget(name)
        col.addWidget(size)
        text.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        lay.addWidget(text)

        tip = a.note or ("图片会缩到 1600px 以内" if a.kind == "image"
                         else "文字会整段发给模型")
        chip.setToolTip(f"{a.name}\n{a.size_text()}｜{tip}")

        close = QPushButton("✕")
        close.setObjectName("Ghost")
        close.setFixedSize(20, 20)
        close.setToolTip("移除")
        close.clicked.connect(lambda _=False, k=idx: self._drop_att(k))
        lay.addWidget(close)
        return chip

    @staticmethod
    def _thumb(a: "ATT.Attachment"):
        """从附件的 base64 直接拼缩略图。不落临时文件。"""
        if a.kind != "image" or not a.b64:
            return None
        try:
            data = base64.b64decode(a.b64)
        except Exception:  # noqa: BLE001
            return None
        pm = QPixmap()
        if not pm.loadFromData(data):
            return None
        return pm.scaled(34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _drop_att(self, idx: int) -> None:
        if 0 <= idx < len(self._atts):
            a = self._atts.pop(idx)
            # 会话资源要一起删：只从附件条上拿掉的话，模型还能拿 file_id 读到它
            if isinstance(a, SessAtt):
                self.session.remove(a.file_id)
        self._refresh_atts()

    # ---------------------------------------------------------- 运行输出

    def push_output(self, text: str, coalesce: bool = False) -> None:
        """往「运行输出」里写一段。`coalesce=True` 表示这次可能连着来一串
        （拖滑杆每 220ms 重算一次），用新的覆盖上一条，别刷屏。

        整段重绘而不是增量插入：输出最多六十条，重绘比维护「删最后一段」的
        游标逻辑简单得多，也不会因为换行把块数算错。
        """
        text = (text or "").strip()
        if not text:
            return
        entry = f"[{time.strftime('%H:%M:%S')}] {text}"
        if self._out and text == self._last_output:
            # 同一份结果算了两遍（灌配置时的中间态 + 收尾那一次），别记两条
            self._out[-1] = entry
        elif coalesce and self._out and self._out_merge:
            self._out[-1] = entry
        else:
            self._out.append(entry)
        self._out_merge = coalesce
        del self._out[:-60]

        self.out.setPlainText("\n\n".join(self._out))
        sb = self.out.verticalScrollBar()
        sb.setValue(sb.maximum())
        self._last_output = text
        self.btn_out_feed.setEnabled(not self._busy)
        self.lbl_out.setText(f"{len(self._out)} 条")

        # 模型自己发起的回测：结果直接喂回去，它就不用再问「输出是什么」
        if self._expect_bt:
            self._expect_bt = False
            if self.cfg.ready():
                self._after_tool(
                    "回测跑完了，下面是平台算出来的完整结果：\n\n" + text +
                    "\n\n直接基于这些数字回答，不要让用户再贴一遍。"
                )

    def begin_backtest_feedback(self) -> None:
        """标记「下一次回测结果要喂回模型」。

        只有模型自己发起的回测才这样。用户手动点「回测」、拖参数时也走同一条
        计算链路，那些结果不该自动触发一轮对话 —— 白烧 token，还打断人。
        """
        self._expect_bt = True

    def _feed_last_output(self) -> None:
        if not self._last_output or self._busy:
            return
        if not self.cfg.ready():
            self._set_state("先配置模型", C_BAD)
            return
        self._append(
            f"<p style='margin:12px 0 2px 0'><b style='color:{C_USER}'>你</b></p>"
            f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px;margin:0 0 8px 0'>"
            "（把运行输出发给 Agent）</p>"
        )
        self._after_tool(
            "用户把「运行输出」面板里最近一次的结果发给你了：\n\n" + self._last_output
        )

    def _clear_output(self) -> None:
        self._out.clear()
        self._out_merge = False
        self._last_output = ""
        self.out.clear()
        self.btn_out_feed.setEnabled(False)
        self.lbl_out.setText("")

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
        if self._msg_start is None:
            # 记下这条回答的起点，收尾时从这里往后整段抹掉重排。
            # 必须在这里取而不是 `_send` 里：`_send` 之后可能还有别的插入
            # （附件说明、工具回执），起点取早了会把别人的内容一起抹了。
            self._msg_start = self.log.textCursor().position()
        self._stream_buf += piece
        # 流式期间仍走纯文本 —— 每次重排整条回答在长回复上会明显卡顿，
        # 而且滚动位置会被反复重置。收尾那一下再统一换 HTML。
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(piece)
        self.log.moveCursor(QTextCursor.End)

    def _render_reply(self, text: str) -> None:
        """把流式期间插入的纯文本换成渲染后的富文本。

        `markdown` 的 `**加粗**`、`| 表格 |`、三反引号代码块原来都是原文照贴 ——
        因为这里以前只 insertPlainText，收尾也不重排。
        """
        start = self._msg_start
        self._msg_start = None
        html = CR.render_text(text)
        if not html:
            html = "<p></p>"

        doc = self.log.document()
        if start is None or start < 0 or start > doc.characterCount():
            # 起点丢了（比如中途被 setHtml 过），退化成追加，至少内容不丢
            self._append(html)
            return

        cur = self.log.textCursor()
        cur.setPosition(start)
        cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        if cur.block().text().strip():
            # 光标停在「助手」那一行的末尾，直接 insertHtml 会把标题并进那一行
            cur.insertBlock()
        self.log.setTextCursor(cur)
        self.log.insertHtml(html)
        self.log.moveCursor(QTextCursor.End)

    def _on_done(self, text: str) -> None:
        final = text or self._stream_buf
        if final.strip():
            self._render_reply(final)
        else:
            self._render_reply("")
            self._append(f"<p style='color:{C_BAD}'>模型没有返回内容。</p>")
        self._append("<p></p>")
        self._history.append({"role": "assistant", "content": final})

        action = A.parse_action(final)
        if action.kind and not action.ok:
            self._append(
                f"<p style='color:{C_BAD};font-size:{FS_SMALL}px'>动作块有问题：{_esc(action.error)}</p>"
            )
        elif action.ok:
            self._handle_action(action)

        if self._pending_report:
            self._save_report(self._pending_report, final)
            self._pending_report = ""

        self._maybe_compress()

    def _on_failed(self, msg: str) -> None:
        # 断了也把已经收到的半截渲染出来 —— 纯文本留着比整段消失强，
        # 至少用户能看见模型说到哪儿了
        if self._msg_start is not None and self._stream_buf.strip():
            self._render_reply(self._stream_buf)
        self._msg_start = None
        self._append(
            f'<div style="border-left:5px solid {C_BAD};background-color:#fdeaea;'
            f'padding:8px 10px;margin:6px 0 8px 0">'
            f'<p style="margin:0;color:{C_BAD};font-weight:700">出错了</p>'
            f'<p style="margin:0;color:{C_TEXT}">{_esc(msg)}</p></div>'
        )
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
        # 带图消息的 content 是 parts 列表，`len()` 数出来是部件数不是字数
        chars = sum(len(A.text_of(m.get("content"))) for m in chunk)
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
                f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>已把前 {len(chunk)} 条对话"
                f"压成记忆（共 {Mem.count()} 条），下次开窗口还在。</p>"
            )

    def _on_memo_failed(self, msg: str) -> None:
        # 只提示一次：模型没配好时每轮都失败，刷屏比不提示更烦
        if self._memo_warned:
            return
        self._memo_warned = True
        self._append(
            f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>记忆压缩没成功"
            f"（{_esc(msg[:80])}），这段对话先留着，下次再试。</p>"
        )

    def _on_memo_thread_done(self) -> None:
        self._memo_worker = None
        self._memo_thread = None

    def _set_busy(self, on: bool) -> None:
        self._busy = on
        self.btn_send.setEnabled(not on)
        self.btn_stop.setEnabled(on)
        # 正在生成时不能把输出塞回去：`_after_tool` 会另起一个线程，两个 worker
        # 抢同一个 `self._worker`，先结束的那个会把新的状态抹掉
        self.btn_out_feed.setEnabled(not on and bool(self._last_output))
        if on:
            self._set_state("正在生成…", C_MUTED)

    def _set_state(self, text: str, color: str = C_MUTED) -> None:
        self.lbl_state.setText(text)
        self.lbl_state.setStyleSheet(f"color: {color}; font-size: {FS_SMALL}px;")

    # ---------------------------------------------------------- 动作

    def _handle_action(self, action: A.Action) -> None:
        """执行一个动作块。

        三层判断，顺序不能换：

        1. **能力包开没开** —— 关着的包提示词里不列，但模型可能从历史里翻出来
           一个旧动作块，所以还是要拦一道，并且说清去哪儿打开
        2. **要不要确认** —— `read` 不问，`write` 问，`trade` 问（且受
           `confirm_trade` 控制）。`fs` 自己管：读不问、写必问
        3. **谁来执行** —— 有 `handler` 的在 `ttrade` 里跑，其余按 `ui` 找界面方法
        """
        tool = S.get(action.kind)
        if tool is None:
            self._append(
                f"<p style='color:{C_BAD}'>平台没有「{_esc(action.kind)}」这个动作。</p>")
            return

        ctx = self._ctx()
        if not ctx.allow(tool.group):
            self._deny(tool)
            return

        if tool.risk != S.RISK_READ and not tool.self_confirm:
            # `confirm_trade` 只管下单：注册因子/策略要执行的是模型现写的 Python
            # 代码，每次都不一样，盲签没有意义，所以不受那个开关影响
            if tool.risk != S.RISK_TRADE or self.cfg.confirm_trade:
                dlg = ConfirmDialog(action, self)
                if dlg.exec() != QDialog.Accepted:
                    self._append(
                        f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>已取消。</p>")
                    return
                if tool.risk == S.RISK_TRADE and dlg.silent():
                    self.cfg.confirm_trade = False
                    A.save_config(self.cfg)
                    self._append(
                        f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>"
                        "以后不再询问，可在设置里重新打开。</p>"
                    )

        if tool.handler is not None:
            self._run_tool(tool, action, ctx)
            return

        fn = getattr(self, "_run_" + tool.ui, None) if tool.ui else None
        if fn is None:
            self._append(
                f"<p style='color:{C_BAD}'>「{_esc(tool.label)}」没有对应的界面处理。</p>")
            return
        fn(action)

    def _deny(self, tool: "S.Tool") -> None:
        """能力包关着时怎么告诉用户。要指名去哪打开 —— 只说「没有权限」等于没说。"""
        if tool.group == "fs":
            msg = ("模型想读写平台文件，但「升级权限」没打开。到左下角「设置」里勾上"
                   "「允许 Agent 读写 ChanQuant 目录」才能用。")
        else:
            label = S.GROUPS.get(tool.group, {}).get("label", tool.group)
            msg = (f"「{label}」这个能力包现在是关着的，所以没法执行「{tool.label}」。"
                   "到左下角「设置」→「能力包」里打开它。")
        self._append(f"<p style='color:{C_BAD}'>{_esc(msg)}</p>")
        self._set_state("能力包未开启", C_BAD)

    # ---------------------------------------------------------- 工具动作

    # 一次提问里最多连续自动跑几轮工具动作。模型拿到结果后可能再查、再读、再写，
    # 不封顶的话它俩能来回刷到把 token 烧光。
    MAX_TOOL_TURNS = 4

    def _run_tool(self, tool: "S.Tool", action: A.Action, ctx) -> None:
        """在 `ttrade` 里跑一个纯逻辑工具，把结果渲染出来再回灌模型。

        工具返回的是 Block，不是 HTML —— 渲染这一步留在界面层，`ttrade` 那边
        不需要知道 `QTextBrowser` 长什么样。
        """
        res = S.dispatch(ctx, action.kind, action.payload)
        if res.blocks:
            self._append(CR.render(res.blocks))
        if not res.ok:
            self._append(f"<p style='color:{C_BAD}'>{_esc(res.error)}</p>")
            self._set_state(f"{tool.label}失败", C_BAD)
        else:
            self._set_state((res.summary or f"{tool.label}完成")[:80], C_OK)
        # 出错也要回灌：模型得看到「列名不对」「文件读不了」才能改对，
        # 只写在界面上等于让它接着按错的假设往下聊
        self._after_tool(res.feed_text())

    def _run_signal(self, action: A.Action) -> None:
        """交给主窗口执行的动作：回测、下单。"""
        self._append(
            f"<p style='color:{C_OK};font-size:{FS_SMALL}px'>已提交：{_esc(action.describe())}</p>"
        )
        self.trade_requested.emit(action)

    def _run_factor(self, action: A.Action) -> None:
        self._register_factor(action.payload)

    def _run_strategy(self, action: A.Action) -> None:
        self._register_strategy(action.payload)

    def _run_fs(self, action: A.Action) -> None:
        """执行 `fs` 动作：读和列目录直接做，写必须弹确认框。

        权限判断在 `_handle_action` 里已经做过（`_deny`），这里只管执行。
        下面那个 `allow_fs` 检查留着当兜底 —— 万一以后有别的路径直接调到这儿。
        """
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
                "ChanQuant 目录」才能用。</p>"
            )
            self._set_state("升级权限未开启", C_BAD)
            return

        if op == "list":
            text, err = FS.list_dir(path)
            if err:
                self._after_tool(f"列出 {path} 失败：{err}")
                return
            self._append(
                f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>已列出 {_esc(path)}</p>"
                f"<pre style='font-size:{FS_CODE}px'>{_esc(text)}</pre>"
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
                f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>已读取 {_esc(path)}"
                f"（{len(text)} 字符）</p>"
                f"<pre style='white-space:pre-wrap;font-family:Consolas,monospace;"
                f"font-size:{FS_CODE}px'>{_esc(shown)}</pre>"
            )
            self._after_tool(f"文件 {path} 的全文：\n```\n{text}\n```")
            return

        # 写：内容摊开给人看。读可以不问，写不行 —— 一次覆盖就是几十行没了。
        content = p.get("content", "")
        dlg = ConfirmDialog(action, self)
        if dlg.exec() != QDialog.Accepted:
            self._append(f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>已取消写入。</p>")
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
                f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>工具动作已连续执行 "
                f"{self._tool_turns} 轮，先停在这里。要继续就再发一句。</p>"
            )
            return
        self._tool_turns += 1
        msgs = A.build_messages(
            A.trim_history(self._history[:-1]), note,
            # 沿用当前开关：关着的时候不能因为「续跑」就悄悄把文件能力塞回去
            memory=Mem.brief(), allow_fs=self.cfg.allow_fs,
            enabled=self.cfg.enabled_groups,
        )
        self._append("<p style='margin:12px 0 2px 0'><b>助手</b></p>")
        self._stream_buf = ""
        self._msg_start = None
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
            # 报错必须回到模型手里。以前只写在界面上，模型以为自己成功了，
            # 下一轮接着按错的假设聊 —— 用户看到的就是「它一直问我结果」。
            #
            # 契约那一节（705 字符）直接随报错贴上去，不指望它自己想起来查：
            # 一次失败只值一轮重试，让它多发一次 doc 请求去换同样的文字不划算。
            # 原语源码五千多字，那个按需查。
            from ttrade import factor_manual as M

            self._after_tool(
                f"因子「{f.key}」注册失败，平台的原始报错是：\n{msg}\n\n"
                f"{M.CONTRACT}\n\n"
                "改好之后重新给一个 factor 动作块。要是怀疑原语用法不对，先查：\n"
                '{"action": "doc", "topic": "primitives"}  原语的真实源码\n'
                '{"action": "doc", "topic": "factor:mom_20"}  照抄一个能跑的写法'
            )
            return

        self._append(
            f"<p style='color:{C_OK}'>已注册因子「{_esc(f.label)}」（{_esc(f.key)}）。"
            f"{_esc(msg)}</p>"
            f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>"
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
            self._after_tool(
                f"策略「{s.key}」注册失败，平台的原始报错是：\n{msg}\n\n"
                "改之前先查手册：策略能用的原语在\n"
                '{"action": "doc", "topic": "primitives"}  的「策略里额外能用的」一节，'
                "返回值约定见 \"contract\"。改好后重新给一个 strategy 动作块。"
            )
            return

        self._append(
            f"<p style='color:{C_OK}'>已注册策略「{_esc(s.label)}」（{_esc(s.key)}）。"
            f"{_esc(msg)}</p>"
            f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>"
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
            f"<p style='color:{C_MUTED};font-size:{FS_SMALL}px'>"
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
        if kind != "rule":
            # 内置模板 / 分组标题：按钮本来就是灰的，这里再兜一层
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
        if kind == "tpl":
            # 内置模板本身就是一份完整配置（有品种、有周期、有参数），
            # 直接拿去回测 / 加入组合即可，不用再问用户选什么品种
            return TPL.to_config(key)
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

    # ---------------------------------------------------------- 因子手册 / 审核

    HANDBOOK_NAME = "因子深度分析手册.pdf"
    PROFILE_NAME = "因子分析.json"

    def _handbook_path(self) -> str:
        d = os.path.join(user_dir(), "handbook")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, self.HANDBOOK_NAME)

    def _profile_path(self) -> str:
        return os.path.join(os.path.dirname(self._handbook_path()), self.PROFILE_NAME)

    def _load_profile(self) -> dict | None:
        """读上次体检的缓存。审核 / 深度分析要拿实测数字喂给模型。"""
        import json

        try:
            with open(self._profile_path(), encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else None
        except Exception:  # noqa: BLE001
            return None

    def _selected_factor_key(self) -> str:
        it = self.lst_factor.currentItem()
        return str(it.data(Qt.UserRole)) if it is not None else ""

    def _make_handbook(self) -> None:
        """整库体检 → PDF → 打开。已经有一份就再跑一次（数据会变）。"""
        if getattr(self, "_prof_thread", None) is not None:
            self._set_state("正在生成手册，稍等…", C_MUTED)
            return
        self.btn_handbook.setEnabled(False)
        self._set_state("正在跑因子体检（约十几秒）…", C_MUTED)
        self._prof_worker = ProfileWorker(years=10)
        self._prof_thread = QThread(self)
        self._prof_worker.moveToThread(self._prof_thread)
        self._prof_thread.started.connect(self._prof_worker.run)
        self._prof_worker.progress.connect(
            lambda d, t, m: self._set_state(f"因子体检 {d}/{t} · {m}", C_MUTED)
        )
        self._prof_worker.done.connect(self._on_profile_done)
        self._prof_worker.failed.connect(self._on_profile_failed)
        self._prof_worker.done.connect(self._prof_thread.quit)
        self._prof_worker.failed.connect(self._prof_thread.quit)
        self._prof_thread.finished.connect(self._prof_thread_done)
        self._prof_thread.start()

    def _on_profile_done(self, res) -> None:
        import json

        try:
            from ttrade import factor_pdf as PDF

            with open(self._profile_path(), "w", encoding="utf-8") as f:
                json.dump(res.to_dict(), f, ensure_ascii=False)
            path = PDF.write_pdf(res, self._handbook_path())
        except Exception as exc:  # noqa: BLE001
            self._set_state(f"手册生成失败：{type(exc).__name__}: {exc}", C_BAD)
            return
        self._set_state(f"手册已生成（{len(res.factors)} 个因子）", C_OK)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _on_profile_failed(self, msg: str) -> None:
        self._set_state(f"因子体检失败：{msg}", C_BAD)

    def _prof_thread_done(self) -> None:
        for attr in ("_prof_thread", "_prof_worker"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.deleteLater()
                setattr(self, attr, None)
        self.btn_handbook.setEnabled(True)

    def _factor_facts(self, key: str) -> str:
        """把这个因子的实测数字拼成几行，喂给模型当事实依据。

        没有缓存（还没生成过手册）就返回空串 —— 让模型自己去读代码，别编数字。
        """
        prof = self._load_profile()
        if not prof:
            return ""
        one = next((f for f in prof.get("factors") or [] if f.get("key") == key), None)
        if not one:
            return ""

        def _n(v, nd=4):
            try:
                x = float(v)
            except (TypeError, ValueError):
                return "—"
            return f"{x:.{nd}f}" if x == x else "—"

        if one.get("no_daily"):
            return (f"实测：日线数据上整列为空（日内 / 时段类因子），只在 5m / 1h 上有值。"
                    f"体检样本 {prof.get('meta', {}).get('span', ['', ''])[0]} ~ "
                    f"{prof.get('meta', {}).get('span', ['', ''])[1]}。")
        lines = [
            f"实测（{len(prof.get('meta', {}).get('symbols') or [])} 个品种横截面，"
            f"{prof.get('meta', {}).get('span', ['', ''])[0]} ~ "
            f"{prof.get('meta', {}).get('span', ['', ''])[1]}）：",
            f"- 横截面 IC 均值 {_n(one.get('ic_mean'))}，ICIR {_n(one.get('icir'), 3)}，"
            f"Newey-West t {_n(one.get('t'), 2)}，方向胜率 {_n(one.get('pos'), 2)}",
            (f"- IC 衰减半衰期 {_n(one.get('half_life'), 1)} 个交易日 → 判定为「{one.get('speed')}」，"
             f"建议周期 {one.get('period')}"
             if one.get("speed") != "信号弱" else
             f"- IC 太小（{_n(one.get('ic_mean'))}），不做快慢判断；建议先看它在哪个品种上有值"),
            f"- 换手 {_n(one.get('turnover'), 3)}，稳健分 {one.get('robust', 0):.0f}/100，"
            f"分年一致 {_n(one.get('year_ratio'), 2)}",
            f"- 最灵品种：{'、'.join(one.get('best_symbols') or []) or '—'}",
        ]
        mates = one.get("mates") or []
        if mates:
            lines.append("- 低相关搭子：" + "、".join(
                f"{m['key']}（相关 {m['corr']:+.2f}）" for m in mates))
        return "\n".join(lines)

    def _audit_factor(self) -> None:
        key = self._selected_factor_key()
        if not key:
            self._set_state("先在左边选一个因子", C_BAD)
            return
        spec = None
        try:
            from ttrade.factors import FACTORS

            spec = FACTORS.get(key)
        except Exception:  # noqa: BLE001
            pass
        head = f"`{key}`" + (f"（{spec.label}，类别 {spec.category}）" if spec else "")
        self._prefill(
            f"请对因子 {head} 做一次**PIT 审核**。用 "
            f'`{{"action":"doc","topic":"factor:{key}"}}` 取它的完整实现，逐条回答：\n'
            "1. **未来数据**：实现里有没有用到 T 时刻之后才知道的信息？重点查：居中滚动窗口、"
            "负向 shift、全样本分位数 / 全样本均值方差、把当日收盘价用在当日盘中、"
            "用未来的持仓量或成交额。\n"
            "2. **数据陈旧**：它依赖的展期 / IV 是日频快照，分钟线上只能用 D-1 的值。"
            "这个因子有没有在盘中拿当日还没落地的快照？滞后几个交易日算正常、几个算断档？\n"
            "3. **夜盘归属**：夜盘 21:00 的 bar 属于**下一个交易日**。这个因子在同一交易日的"
            "夜盘段和日盘段，读到的应该是同一份快照 —— 它按交易日对齐还是按自然日？\n"
            "4. **时段边界**：如果它读的是「当日」某个统计量，在 09:00 第一根 bar 上有没有值？"
            "会不会出现开盘半小时整列 NaN？\n"
            "5. **过旧**：因子值长期不变（比如只有换月才变）时，回测和实盘的行为是否一致？\n\n"
            "结论请给三档之一：**通过 / 有条件通过 / 不通过**，并写明具体是哪一行代码、"
            "怎么改。不确定的地方直接说「不确定」，不要猜。"
        )
        self._send()

    def _deep_factor(self) -> None:
        key = self._selected_factor_key()
        if not key:
            self._set_state("先在左边选一个因子", C_BAD)
            return
        spec = None
        try:
            from ttrade.factors import FACTORS

            spec = FACTORS.get(key)
        except Exception:  # noqa: BLE001
            pass
        head = f"`{key}`" + (f"（{spec.label}，类别 {spec.category}）" if spec else "")
        facts = self._factor_facts(key)
        self._prefill(
            f"请对因子 {head} 做一次**深度分析**。用 "
            f'`{{"action":"doc","topic":"factor:{key}"}}` 取完整实现。\n'
            + (facts + "\n" if facts else "")
            + "请围绕下面五点展开，能用上面的实测数字就用，没有数字就说明是推断：\n"
            "1. **快慢**：信号能活几天？半衰期多长？据此它该放在 5m、1h 还是日线？"
            "如果放在分钟线上，交易成本会不会把收益吃光？\n"
            "2. **品种**：哪些品种上更有效、为什么（产业集中度、库存周期、流动性、"
            "是否有对应期权）？哪些品种上应该直接关掉？\n"
            "3. **周期**：不同周期下这个因子的含义会变吗（比如 5m 上是反转、日线上是趋势）？\n"
            "4. **组合**：它适合和哪类因子搭？为什么是互补而不是重复？给出 2~3 个具体建议。\n"
            "5. **稳健性**：它最可能在什么情况下失效？参数敏感吗？样本外要注意什么？\n\n"
            "最后给一句结论：这个因子值不值得进组合，进的话权重给多少、和谁一起用。"
        )
        self._send()

    def _ask_new_strategy(self) -> None:
        self.tabs.setCurrentIndex(1)
        self._prefill(
            "我想做一个新的规则策略：<在这里写思路>。\n"
            "先判断平台已有的规则策略（双均线、唐奇安、ATR 通道、MACD、海龟、"
            "箱体+波幅突破·迷你、IM 波幅突破·时段版）"
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
        self._append(f"<p style='color:{color};font-size:{FS_SMALL}px'>{_esc(msg)}</p>")

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
        f"<pre style='font-family:Consolas,monospace;font-size:{FS_CODE}px;"
        "background:#f6f8fa;padding:8px;border-radius:4px;"
        f"white-space:pre-wrap'>{_esc(text)}</pre>"
    )


def _strategy_text(name, params: dict | None = None) -> str:
    """确认框里「策略」那一行。

    动作块里现在可能是策略库模板的 `sid`（`a62cd55e1389`）—— 一串十六进制对
    用户没有意义，看到它根本没法判断要点的是哪条策略。能对上模板就显示模板名
    和它锁定的品种/周期，对不上就原样显示。
    """
    raw = str(name or "").strip()
    if not raw:
        return ""
    try:
        cfg = SStore.find_library(raw)
    except Exception:  # noqa: BLE001
        cfg = None
    if cfg is not None:
        return f"{cfg.title()}（策略库模板 · {cfg.key()} {cfg.period}）"
    tail = f" {params}" if params else ""
    return (raw + tail).strip()


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


_SIZING_CN = {
    "capital": "按资金分配", "risk": "按止损金额",
    "fixed": "固定手数", "unit": "本金全额",
}


def _sizing_cn(mode: str) -> str:
    return _SIZING_CN.get(str(mode or "capital").lower(), str(mode or "按资金分配"))


def _factor_mix(cfg) -> str:
    """一条配置的因子组合：「n 个因子，其中 k 个取反」+ 逐个因子名。

    取反的因子标红。这是内置模板最该被看见的一层 —— 同一份因子集，反着用
    几个，结论就完全不同；不标出来，用户只会看到一长串因子名。
    """
    if cfg.family == "rule":
        spec = STRATEGIES.get(cfg.rule_key)
        return _esc(f"不用因子：{spec.label if spec else (cfg.rule_key or '—')}")
    keys = list(cfg.factor_keys or [])
    if not keys:
        return "—"
    rev = {k for k, v in ((cfg.factor_cfg or {}).get("reverse") or {}).items() if v}
    parts = []
    for k in keys:
        d = L.factor_doc(k)
        name = _esc(d.label if d else k)
        if k in rev:
            name += f"<span style='color:{C_BAD}'>（取反）</span>"
        parts.append(name)
    n_rev = len(rev & set(keys))
    head = f"{len(keys)} 个因子" + (f"，其中 {n_rev} 个取反" if n_rev else "")
    return f"<span style='color:{C_MUTED}'>{head}</span><br>" + "、".join(parts)


# 研报品种候选。**不查 DuckDB**：建页面时去开库，赶上别的进程在写就会卡住，
# 而这里只是个下拉框的候选，不值得为它冒这个风险。
REPORT_SYMBOLS = [
    "RB", "HC", "I", "J", "JM", "CU", "AL", "ZN", "NI", "AU", "AG",
    "SC", "FU", "MA", "TA", "EG", "PP", "L", "V", "SR", "CF", "M", "Y", "P", "OI",
]


def _symbol_choices() -> list[str]:
    return list(REPORT_SYMBOLS)
