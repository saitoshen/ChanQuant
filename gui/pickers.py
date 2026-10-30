"""多品种勾选器。

「策略研究」的标的原本是一个下拉，一次只能看一个品种。截面策略和多品种回测
都要「一批品种」，用下拉加 Ctrl 多选很难点，所以做成一个按钮 + 勾选菜单：

- 按钮上直接写「已选 N 个」，不用点开就知道现在是不是多品种模式；
- 菜单按交易所分组，组头不可点，和标的列表的分组一致；
- 顶部两个动作：全选当前组 / 清空。

**不复用 `QComboBox` 的 checkable 模型**：那需要自己接管 popup 的关闭时机
（点一下勾选就关掉了，勾 20 个要点 20 次）。`QMenu` 的 checkable action 默认
就不关菜单，正好是要的行为。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QMenu, QToolButton

from ttrade import symbols_info


class SymbolPicker(QToolButton):
    changed = Signal()

    def __init__(self, title: str = "多品种", parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._items: list[dict] = []
        self._checked: list[str] = []
        self.setPopupMode(QToolButton.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setObjectName("LiveBtn")      # 和研究页其它开关同一套样式
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self._refresh_text()

    # ---- 数据 ----

    def set_items(self, items: list[dict], keep: list[str] | None = None) -> None:
        """重建菜单。`keep` 里还在池子里的勾选保留，其余丢掉。"""
        self._items = list(items or [])
        avail = {it.get("key") for it in self._items}
        keep = [k for k in (keep if keep is not None else self._checked) if k in avail]
        self._checked = keep
        self._build()

    def selected(self) -> list[str]:
        """按清单顺序返回勾选的品种键。"""
        want = set(self._checked)
        return [it["key"] for it in self._items if it.get("key") in want]

    def set_selected(self, keys) -> None:
        avail = {it.get("key") for it in self._items}
        self._checked = [k for k in (keys or []) if k in avail]
        self._build()

    def clear(self) -> None:
        self._checked = []
        self._refresh_text()
        self._rebuild_soon()

    # ---- 菜单 ----

    def _build(self) -> None:
        self._menu.clear()
        if not self._items:
            act = self._menu.addAction("（这个市场没有品种）")
            act.setEnabled(False)
            self._refresh_text()
            return

        head = self._menu.addAction("清空")
        head.triggered.connect(self.clear)

        groups: dict[str, list[dict]] = {}
        for it in self._items:
            groups.setdefault(it.get("exchange") or "", []).append(it)
        order = [k for k in symbols_info.EXCHANGES if k in groups]
        order += [k for k in groups if k not in symbols_info.EXCHANGES]

        for ex in order:
            rows = groups[ex]
            label = symbols_info.EXCHANGES.get(ex, ex)
            sec = self._menu.addAction(f"—— {label}（{len(rows)}）——")
            sec.setEnabled(False)
            all_act = self._menu.addAction(f"　全选{label}")
            all_act.triggered.connect(lambda _c=False, rs=rows: self._toggle_group(rs, True))
            for it in rows:
                act = self._menu.addAction("　" + it["label"])
                act.setCheckable(True)
                act.setChecked(it["key"] in self._checked)
                act.setData(it["key"])
                act.toggled.connect(
                    lambda on, k=it["key"]: self._toggle_one(k, on)
                )
        self._refresh_text()

    def _toggle_group(self, rows: list[dict], on: bool) -> None:
        have = set(self._checked)
        for it in rows:
            if on:
                have.add(it["key"])
            else:
                have.discard(it["key"])
        # 保持清单顺序，勾选顺序不影响回测结果
        self._checked = [it["key"] for it in self._items if it["key"] in have]
        self._refresh_text()
        self._rebuild_soon()
        self.changed.emit()

    def _rebuild_soon(self) -> None:
        """延后一拍重建菜单。

        重建会 `clear()` 掉正在发信号的那个 action —— 直接在它的槽里删自己，
        Qt 会崩。等这一轮事件走完再重建。
        """
        QTimer.singleShot(0, self._build)

    def _toggle_one(self, key: str, on: bool) -> None:
        have = set(self._checked)
        if on:
            have.add(key)
        else:
            have.discard(key)
        self._checked = [it["key"] for it in self._items if it["key"] in have]
        self._refresh_text()
        self.changed.emit()

    def _refresh_text(self) -> None:
        n = len(self._checked)
        if n:
            self.setText(f"{self._title}：已选 {n} 个")
        else:
            self.setText(f"{self._title}…")
        self.setToolTip(
            "勾选多个品种后，研究页按**等权组合**回测：每个品种单独跑同一套策略，\n"
            "再把净值曲线等权平均。截面策略要求至少 3 个品种。"
            if not n else
            "已选：" + "、".join(self._checked)
        )
