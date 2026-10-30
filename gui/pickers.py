"""多品种勾选器。

「策略研究」的标的原本是一个下拉，一次只能看一个品种。截面策略和多品种回测
都要「一批品种」，用下拉加 Ctrl 多选很难点，所以做成一个按钮 + 勾选面板：

- 按钮上直接写「已选 N 个」，不用点开就知道现在是不是多品种模式；
- 面板按交易所分组，每组一个「全选」，顶部还有「全选 / 清空」；
- **点一个不会关** —— 面板是 `Qt.Popup` 的独立窗口，不是 `QMenu`。`QMenu` 里
  点任何 action 都会关菜单，勾 20 个品种就得点开 20 次；早先那版就是栽在这里。

勾选过程中**不重算**：面板关掉那一刻才发一次 `changed`。一次截面回测几秒钟，
勾 6 个品种重算 6 遍等于让人干等，而且面板还遮着结果，看不到。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QToolButton, QVBoxLayout, QWidget,
)

from ttrade import symbols_info

PANEL_WIDTH = 340
PANEL_MAX_H = 520
COLUMNS = 2


class _Panel(QWidget):
    """弹出面板。点面板外面自动关（`Qt.Popup` 自带的行为）。"""

    changed = Signal()

    def __init__(self, owner: "SymbolPicker") -> None:
        super().__init__(owner.window(), Qt.Popup)
        self._owner = owner
        self._items: list[dict] = []
        self._checked: set[str] = set()
        self._boxes: dict[str, QCheckBox] = {}
        self._dirty = False

        self.setObjectName("PickPanel")
        self.setStyleSheet(
            "#PickPanel { background:#ffffff; border:1px solid #d0d7de;"
            " border-radius:8px; }"
            "QLabel#PickHead { color:#6b7280; font-size:11px; }"
            "QPushButton#PickAll { border:none; color:#2563eb; font-size:11px;"
            " padding:0 4px; background:transparent; }"
            "QPushButton#PickAll:hover { text-decoration:underline; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(6)
        self.lbl = QLabel("选择品种")
        self.lbl.setObjectName("PickHead")
        head.addWidget(self.lbl)
        head.addStretch(1)
        btn_all = QPushButton("全选")
        btn_all.setObjectName("PickAll")
        btn_all.setCursor(Qt.PointingHandCursor)
        btn_all.clicked.connect(lambda: self._set_all(True))
        head.addWidget(btn_all)
        btn_clear = QPushButton("清空")
        btn_clear.setObjectName("PickAll")
        btn_clear.setCursor(Qt.PointingHandCursor)
        btn_clear.clicked.connect(lambda: self._set_all(False))
        head.addWidget(btn_clear)
        root.addLayout(head)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setMaximumHeight(PANEL_MAX_H)
        root.addWidget(self.scroll)

        self._body = QWidget()
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(0, 0, 0, 0)
        self._body_lay.setSpacing(4)
        self.scroll.setWidget(self._body)

        tip = QLabel("勾选多个品种：同一套策略在每个品种上各跑一遍再等权平均；"
                     "截面策略至少 3 个")
        tip.setObjectName("PickHead")
        tip.setWordWrap(True)
        root.addWidget(tip)

    # ---- 数据 ----

    def set_items(self, items: list[dict], keep: set[str] | None = None) -> None:
        self._items = list(items or [])
        avail = {it.get("key") for it in self._items}
        self._checked = {k for k in (self._checked if keep is None else keep) if k in avail}
        self._rebuild()

    def set_checked(self, keys) -> None:
        """外部灌值（恢复配置）。**不发 changed**。"""
        avail = {it.get("key") for it in self._items}
        self._checked = {k for k in (keys or []) if k in avail}
        for k, box in self._boxes.items():
            box.blockSignals(True)
            box.setChecked(k in self._checked)
            box.blockSignals(False)
        self._dirty = False
        self._owner._refresh_text(len(self._checked))

    # ---- 构建 ----

    def _rebuild(self) -> None:
        while self._body_lay.count():
            item = self._body_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())
        self._boxes = {}

        if not self._items:
            empty = QLabel("（这个市场没有品种）")
            empty.setObjectName("PickHead")
            self._body_lay.addWidget(empty)
            self._owner._refresh_text(0)
            return

        groups: dict[str, list[dict]] = {}
        for it in self._items:
            groups.setdefault(it.get("exchange") or "", []).append(it)
        order = [k for k in symbols_info.EXCHANGES if k in groups]
        order += [k for k in groups if k not in symbols_info.EXCHANGES]

        for ex in order:
            rows = groups[ex]
            label = symbols_info.EXCHANGES.get(ex, ex) or "其他"
            self._body_lay.addWidget(self._group_head(label, rows))

            holder = QWidget()
            grid = QGridLayout(holder)
            grid.setContentsMargins(8, 0, 0, 4)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(2)
            for i, it in enumerate(rows):
                box = QCheckBox(it["label"])
                box.setChecked(it["key"] in self._checked)
                box.toggled.connect(
                    lambda on, k=it["key"]: self._on_box(k, on)
                )
                self._boxes[it["key"]] = box
                grid.addWidget(box, i // COLUMNS, i % COLUMNS)
            self._body_lay.addWidget(holder)

        self._owner._refresh_text(len(self._checked))

    def _group_head(self, label: str, rows: list[dict]) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(6)
        t = QLabel(f"{label}（{len(rows)}）")
        t.setObjectName("PickHead")
        lay.addWidget(t)
        lay.addStretch(1)
        btn = QPushButton("全选")
        btn.setObjectName("PickAll")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda _c=False, rs=rows: self._toggle_group(rs))
        lay.addWidget(btn)
        return w

    def _clear_layout(self, lay) -> None:
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    # ---- 交互 ----

    def _on_box(self, key: str, on: bool) -> None:
        if on:
            self._checked.add(key)
        else:
            self._checked.discard(key)
        self._dirty = True
        self._owner._refresh_text(len(self._checked))

    def _toggle_group(self, rows: list[dict]) -> None:
        """组内全选 / 全不选（已经全选过就反选）。"""
        keys = [it["key"] for it in rows]
        on = not all(k in self._checked for k in keys)
        for k in keys:
            box = self._boxes.get(k)
            if box is None:
                continue
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)
            if on:
                self._checked.add(k)
            else:
                self._checked.discard(k)
        self._dirty = True
        self._owner._refresh_text(len(self._checked))

    def _set_all(self, on: bool) -> None:
        for k, box in self._boxes.items():
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)
        self._checked = {it["key"] for it in self._items} if on else set()
        self._dirty = True
        self._owner._refresh_text(len(self._checked))

    def selected(self) -> list[str]:
        """按清单顺序返回勾选的品种键。"""
        return [it["key"] for it in self._items if it["key"] in self._checked]

    def hideEvent(self, event) -> None:
        # 关面板才重算一次：勾选过程中重算既慢又看不到结果。
        # 主窗口正在关 / 最小化时也会走到这里 —— 那种时候不该再起一次回测
        if self._dirty and self._owner.isVisible():
            self._dirty = False
            self.changed.emit()
        super().hideEvent(event)


class SymbolPicker(QToolButton):
    """按钮 + 勾选面板。`changed` 在面板关闭且勾选有变动时发一次。"""

    changed = Signal()

    def __init__(self, title: str = "多品种", parent=None) -> None:
        super().__init__(parent)
        self._title = title
        # 不用 `InstantPopup`：那个模式会去弹 `menu()`，而这里没有 menu，
        # 点击事件会被吞掉。直接接 `clicked` 自己弹面板
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setObjectName("LiveBtn")      # 和研究页其它开关同一套样式
        self._panel = _Panel(self)
        self._panel.changed.connect(self.changed)
        self.clicked.connect(self._toggle)
        self._refresh_text(0)

    # ---- 数据 ----

    def set_items(self, items: list[dict], keep: list[str] | None = None) -> None:
        """重建面板。`keep` 里还在池子里的勾选保留，其余丢掉。"""
        self._panel.set_items(items, None if keep is None else set(keep))

    def selected(self) -> list[str]:
        return self._panel.selected()

    def set_selected(self, keys) -> None:
        """恢复配置用。不发 `changed`（调用方自己决定要不要重算）。"""
        self._panel.set_checked(keys)

    def clear(self) -> None:
        self._panel.set_checked([])

    # ---- 弹出 ----

    def _toggle(self) -> None:
        if self._panel.isVisible():
            self._panel.hide()
            return
        self._panel.lbl.setText(f"{self._title}（{len(self._panel.selected())} 已选）")
        self._panel.adjustSize()
        w = max(PANEL_WIDTH, self.width())
        self._panel.resize(w, min(self._panel.sizeHint().height(), PANEL_MAX_H + 80))
        pos = self.mapToGlobal(QPoint(0, self.height() + 2))
        screen = QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            if pos.x() + w > geo.right():
                pos.setX(max(geo.left(), geo.right() - w))
            if pos.y() + self._panel.height() > geo.bottom():
                pos.setY(max(geo.top(), self.mapToGlobal(QPoint(0, 0)).y()
                             - self._panel.height() - 2))
        self._panel.move(pos)
        self._panel.show()

    def _refresh_text(self, n: int) -> None:
        self.setText(f"{self._title}：已选 {n} 个" if n else f"{self._title}…")
        sel = self._panel.selected() if self._panel is not None else []
        self.setToolTip(
            ("已选：" + "、".join(sel)) if sel else
            "勾选多个品种后，研究页按**等权组合**回测：每个品种单独跑同一套策略，\n"
            "再把净值曲线等权平均。截面策略要求至少 3 个品种。"
        )
