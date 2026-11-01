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
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QToolButton, QVBoxLayout, QWidget,
)

import pandas as pd

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


class ContractPicker(QWidget):
    """品种下面那一层：主连 / 具体合约 / 期权合约。

    为什么单独一层而不是往品种下拉里塞：**品种和合约不是一回事**。

    - 品种（`RB_SHFE`）是主连、复权、连续的 —— 回测、因子、截面都建立在它上面
    - 合约（`RB2410` / `DCE.m2405-C-3000`）有到期日、不复权、不连续，
      数量还大得多（期货几万个、期权二十几万个）

    所以这一层默认是「主连」，选成具体合约之后**只影响看图**：回测仍然要回主连，
    因为一条会断掉的曲线算不出年化、也算不了换月。这一点在界面上会直说。

    三档：

        主连        —— 什么都不变，老行为
        具体合约    —— 期货真实合约，按月份挑
        期权        —— 期权链，按月份 + 行权价 + 认购认沽挑
    """

    changed = Signal()

    MODE_MAIN = "main"
    MODE_CONTRACT = "contract"
    MODE_OPTION = "option"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._symbol = ""
        self._exchange = ""
        self._period = "1d"
        # 期权库里 variety 的大小写**跟着合约代码走**（`IO` 大写、`br` 小写），
        # 而品种下拉里的代码是大写（`RB_SHFE`）。所以不能拿 `self._symbol.lower()`
        # 去查，得用 list_months 里实际存在的那个写法
        self._variety = ""
        self._loading = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.cb_mode = QComboBox()
        self.cb_mode.addItem("主连（连续，可回测）", self.MODE_MAIN)
        self.cb_mode.addItem("具体合约", self.MODE_CONTRACT)
        self.cb_mode.addItem("期权合约", self.MODE_OPTION)
        self.cb_mode.setToolTip(
            "主连是拼出来的连续序列，换月处按天勤的切换表复权 —— 回测、因子、\n"
            "截面都基于它。\n"
            "具体合约 / 期权合约只用来**看图**：合约有到期日，曲线是断的，\n"
            "算年化和最大回撤会失真。"
        )
        self.cb_mode.currentIndexChanged.connect(self._on_mode)
        lay.addWidget(self.cb_mode)

        # ---- 月份（合约 / 期权共用一行）
        self.row_month = QWidget()
        rm = QHBoxLayout(self.row_month)
        rm.setContentsMargins(0, 0, 0, 0)
        rm.setSpacing(4)
        self.lbl_month = QLabel("月份")
        self.lbl_month.setObjectName("Hint")
        self.cb_month = QComboBox()
        self.cb_month.setMinimumWidth(140)
        self.cb_month.currentIndexChanged.connect(self._on_month)
        rm.addWidget(self.lbl_month)
        rm.addWidget(self.cb_month, 1)
        lay.addWidget(self.row_month)

        # ---- 行权价 + 认购认沽（只有期权才显示）
        self.row_strike = QWidget()
        rs = QHBoxLayout(self.row_strike)
        rs.setContentsMargins(0, 0, 0, 0)
        rs.setSpacing(4)
        self.cb_strike = QComboBox()
        self.cb_strike.setMinimumWidth(90)
        self.cb_strike.currentIndexChanged.connect(self._emit)
        self.cb_cp = QComboBox()
        self.cb_cp.addItem("认购 C", "C")
        self.cb_cp.addItem("认沽 P", "P")
        # 换认购/认沽要**重填行权价**，不能只发 changed：行权价下拉里存的是
        # 合约代码，C 的清单里没有 P 的合约 —— 只发信号的话标签写着「认沽 P」，
        # 底下选的还是那个 C 合约，图上画的和标签对不上
        self.cb_cp.currentIndexChanged.connect(self._on_cp)
        rs.addWidget(self.cb_strike, 1)
        rs.addWidget(self.cb_cp)
        lay.addWidget(self.row_strike)

        self.lbl_note = QLabel("")
        self.lbl_note.setObjectName("Hint")
        self.lbl_note.setWordWrap(True)
        lay.addWidget(self.lbl_note)

        self.row_month.setVisible(False)
        self.row_strike.setVisible(False)

    # ---- 外部接口 ----

    def set_symbol(self, symbol: str, exchange: str, period: str = "1d") -> None:
        """换品种 / 换周期：重填合约清单。**回到主连**，不保留上次选的那个合约 ——
        换品种还停在「rb2410」上是没有意义的，那个合约根本不存在。
        """
        self._symbol = str(symbol or "")
        self._exchange = str(exchange or "")
        self._period = str(period or "1d")
        self._loading = True
        try:
            self.cb_mode.setCurrentIndex(0)
        finally:
            self._loading = False
        self._reload()

    def mode(self) -> str:
        return self.cb_mode.currentData() or self.MODE_MAIN

    def contract(self) -> str:
        """选中的合约代码，主连模式返回空串。"""
        m = self.mode()
        if m == self.MODE_CONTRACT:
            return self.cb_month.currentData() or ""
        if m == self.MODE_OPTION:
            return self.cb_strike.currentData() or ""
        return ""

    def note(self) -> str:
        return self.lbl_note.text()

    def select_option(self, contract: str) -> bool:
        """外部选中一个期权合约：把三档下拉都对齐到它，然后发 `changed`。

        给「T 型报价」窗双击用 —— 那张表里点了一档，上面这排控件就该跟着
        跳到同一个合约，否则图上画的是 `rb2701-C-3000`、控件里写着「主连」，
        两边对不上。

        返回是否成功。**失败时什么都不改**：改一半（比如模式切了、行权价没切）
        比不改更让人困惑，宿主那边直接走「自己画图」那条路。
        """
        from ttrade import optstore as OS

        code = str(contract or "")
        if not code:
            return False
        meta = OS.load_meta()
        if meta.empty:
            return False
        hit = meta[meta["contract"].astype(str) == code]
        if hit.empty:
            return False
        r = hit.iloc[0]
        und = str(r["underlying"])
        exp = pd.Timestamp(r["expiry"])
        strike = float(r["strike"])

        self._loading = True
        # 整段操作要**一次**发信号，而不是每个中间态发一次：从「主连」切到
        # 「rb2701-C-3000」要经过「切模式 → 重填月份 → 重填行权价 → 选中」四步，
        # 中间那几步的 `contract()` 是空的或者还是默认 ATM。宿主每收到一次就
        # 重画一遍图，等于白跑四五次回测。`blockSignals` 连自定义信号一起挡，
        # 所以最后手动 `emit` 一次。
        self.blockSignals(True)
        try:
            self.cb_mode.setCurrentIndex(2)          # MODE_OPTION
            self.row_month.setVisible(True)
            self.row_strike.setVisible(True)
            self.lbl_month.setText("期权链")
            self._fill_months()
            # 按「标的 + 到期日」找链，不按显示文本 —— 文本里有档数和「已到期」
            # 标记，换个措辞就匹配不上了
            i_month = -1
            for i in range(self.cb_month.count()):
                if str(self.cb_month.itemData(i)) != und:
                    continue
                tag = str(self.cb_month.itemData(i, Qt.UserRole + 1) or "")
                if tag and tag[:10] == exp.strftime("%Y-%m-%d"):
                    i_month = i
                    break
            if i_month < 0:
                return False
            self.cb_month.setCurrentIndex(i_month)
            self._fill_strikes()
            i_k = -1
            for i in range(self.cb_strike.count()):
                if str(self.cb_strike.itemData(i)) == code:
                    i_k = i
                    break
            if i_k < 0:
                # 认购/认沽不对：`_fill_strikes` 只填当前那一侧
                want = "P" if str(r["opt_class"]).upper().startswith("P") else "C"
                self.cb_cp.setCurrentIndex(1 if want == "P" else 0)
                self._fill_strikes()
                for i in range(self.cb_strike.count()):
                    if str(self.cb_strike.itemData(i)) == code:
                        i_k = i
                        break
            if i_k < 0:
                return False
            self.cb_strike.setCurrentIndex(i_k)
        finally:
            self._loading = False
            self.blockSignals(False)
        self.changed.emit()
        return True

    # ---- 内部 ----

    def _emit(self, *_a) -> None:
        if not self._loading:
            self.changed.emit()

    def _on_mode(self, *_a) -> None:
        if self._loading:
            # 外部正在批量改状态（`set_symbol` / `select_option`），它们自己
            # 收尾时会 `_reload()` 一次。这里再跑一遍就是重复取数 + 多发一次
            # `changed`，宿主那边等于白跑一次回测
            return
        m = self.mode()
        self.row_month.setVisible(m != self.MODE_MAIN)
        self.row_strike.setVisible(m == self.MODE_OPTION)
        self.lbl_month.setText("期权链" if m == self.MODE_OPTION else "月份")
        self._reload()

    def _on_month(self, *_a) -> None:
        # 重填清单时 `addItem` 会一路触发 `currentIndexChanged`，不拦住的话
        # 24 个月份就是 24 次 `list_strikes`（每次都要开库 + 扫 20 万合约）
        if self._loading:
            return
        if self.mode() == self.MODE_OPTION:
            self._fill_strikes()
        self._emit()

    def _on_cp(self, *_a) -> None:
        if self._loading:
            return
        if self.mode() == self.MODE_OPTION:
            self._fill_strikes()
        self._emit()

    def _reload(self) -> None:
        m = self.mode()
        self._loading = True
        try:
            if m == self.MODE_CONTRACT:
                self._fill_futures()
            elif m == self.MODE_OPTION:
                self._fill_months()
                self._fill_strikes()
            else:
                self.cb_month.clear()
                self.cb_strike.clear()
                self.lbl_note.setText("")
        finally:
            self._loading = False
        self._emit()

    def _fill_futures(self) -> None:
        from ttrade import tqstore as S

        self.cb_month.clear()
        self.lbl_note.setText("")
        if not self._symbol:
            return
        df = S.list_contracts(self._symbol, self._exchange, self._period)
        if df.empty:
            self.lbl_note.setText(
                f"本地还没有 {self._symbol} 的真实合约 {self._period} 数据 —— "
                "去「数据管理 → 天勤」勾上「一并下载真实合约」"
            )
            return
        today = pd.Timestamp.today().normalize()
        for _, r in df.sort_values("contract", ascending=False).iterrows():
            c = str(r["contract"])
            expired = pd.Timestamp(r["last"]) < today - pd.Timedelta(days=3)
            tag = "已到期" if expired else "在交易"
            self.cb_month.addItem(
                f"{c}　{r['first']:%Y-%m} → {r['last']:%Y-%m-%d}（{tag}，{int(r['rows'])} 根）",
                c,
            )
        self.lbl_note.setText(
            f"共 {len(df)} 个合约。**选合约只影响看图**：回测要回主连，"
            "一条会断的曲线算不出年化和最大回撤。"
        )

    def _fill_months(self) -> None:
        from ttrade import optstore as OS

        self.cb_month.clear()
        self._variety = ""
        if not self._symbol:
            return
        df = OS.list_months()
        if df is None:
            self.lbl_note.setText(
                "期权库正在被下载占用（DuckDB 是单写单读），下完再挑 —— "
                "进度看命令行那条 download_options.py"
            )
            return
        if df.empty:
            self.lbl_note.setText(
                "本地还没有期权库 —— 去「数据管理 → 天勤」勾上「一并更新期权」，"
                "或者命令行跑 download_options.py"
            )
            return
        mine = df[df["variety"].astype(str).str.lower() == self._symbol.lower()]
        fallback = mine.empty
        if fallback:
            # **IO / MO / HO 没有对应的期货品种**：品种池里只有 IF/IH/IC/IM
            # 四个股指期货，股指期权是挂在指数上的。这时候不能只回一句
            # 「没有」，否则这几个品种的期权永远点不到 —— 直接列出全库的链，
            # 标签里带上品种名让用户自己挑
            mine = df
        # 用库里实际那个写法（`IO` / `br`），后面查行权价要原样带回去
        self._variety = str(mine["variety"].iloc[0]) if not fallback else ""
        today = pd.Timestamp.today().normalize()
        pick = -1
        order = mine.sort_values(["variety", "expiry", "underlying"]) if fallback \
            else mine.sort_values(["expiry", "underlying"])
        for _, r in order.iterrows():
            exp = pd.Timestamp(r["expiry"])
            v = str(r["variety"])
            head = f"{v} · " if fallback else ""
            self.cb_month.addItem(
                f"{head}{exp:%Y-%m-%d} 到期 · {r['underlying']}"
                f"（{int(r['n'])} 个合约）",
                str(r["underlying"]),
            )
            i = self.cb_month.count() - 1
            self.cb_month.setItemData(i, exp.strftime("%Y-%m-%d"), Qt.UserRole + 1)
            self.cb_month.setItemData(i, v, Qt.UserRole + 2)
            # 默认落在**第一个还没到期**的月份：已经到期的那些点开就是空图，
            # 而最近月的链通常只有几根 K 线，也不是想看的
            if pick < 0 and exp >= today:
                pick = self.cb_month.count() - 1
        if pick >= 0:
            self.cb_month.setCurrentIndex(pick)
        if fallback:
            names = "、".join(sorted(df["variety"].unique())[:8])
            self.lbl_note.setText(
                f"期权库里没有 {self._symbol} 这个品种，下面列的是全库的链"
                f"（{names} …）—— 挑一条就行"
            )
        else:
            self.lbl_note.setText("")

    def _fill_strikes(self) -> None:
        from ttrade import optstore as OS

        # 重填期间挡信号：`clear()` 和自动落到平值那两下都会触发
        # `currentIndexChanged`，不挡的话一次换月要跑两遍取数
        self.cb_strike.blockSignals(True)
        try:
            self._fill_strikes_impl()
        finally:
            self.cb_strike.blockSignals(False)

    def _fill_strikes_impl(self) -> None:
        from ttrade import optstore as OS

        self.cb_strike.clear()
        und = self.cb_month.currentData()
        exp = self.cb_month.currentData(Qt.UserRole + 1)
        v = self.cb_month.currentData(Qt.UserRole + 2) or self._variety
        if not und or not v:
            return
        df = OS.list_strikes(str(v), str(und), exp or None)
        if df.empty:
            self.lbl_note.setText(f"{und} 这条链上一个合约都没有")
            return
        keep_cp = self.cb_cp.currentData() or "C"
        rows = df[df["cp"].astype(str).str.upper().str.startswith(keep_cp)]
        if rows.empty:
            rows = df
        # 平值：标的最新收盘价最接近的那个行权价。不自动落在这里，用户每换
        # 一个月都要从头滚到尾找 ATM，而 ATM 恰恰是最常看的。
        # **只在有成交的合约里挑**：虚值深的那几个天勤给了合约代码但一根
        # K 线都没有，默认落到它们身上就是「打开一片空白」
        spot = OS.underlying_close(str(und), "1d")
        traded = rows[rows["rows"].fillna(0) > 0]
        atm = None
        if spot and not traded.empty:
            atm = float(traded["strike"].iloc[
                (traded["strike"] - spot).abs().argsort().iloc[0]])
        for _, r in rows.iterrows():
            k = float(r["strike"])
            mark = "　← 平值" if atm is not None and abs(k - atm) < 1e-9 else ""
            dead = "　无成交" if int(r["rows"] or 0) == 0 else ""
            self.cb_strike.addItem(f"{k:g}{mark}{dead}", str(r["contract"]))
        if atm is not None:
            for i in range(self.cb_strike.count()):
                if "平值" in self.cb_strike.itemText(i):
                    self.cb_strike.setCurrentIndex(i)
                    break
            self.lbl_note.setText(
                f"标的 {spot:g}，平值 {atm:g}"
                + (f"，到期 {exp}" if exp else "")
                + "。换认购/认沽会重排行权价，一次看一个合约。"
            )
        elif spot:
            self.lbl_note.setText(
                f"标的 {spot:g}，但这条链上没有成交过的合约（天勤只有合约壳）。"
            )
        else:
            self.lbl_note.setText(
                f"标的 {und} 还没有收盘价（期权标的行情没下到），行权价按升序排。"
            )
