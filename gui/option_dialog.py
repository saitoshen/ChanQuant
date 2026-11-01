"""期权 T 型报价窗：一条链上所有行权价，认购在左、认沽在右。

为什么单独开一个窗而不是塞进研究页的合约下拉：**报价是横截面，下拉是纵截面**。
「螺纹 2701 这 16 档行权价，平值附近 IV 是多少、两边各偏多少」这个问题，
下拉里一次只能看到一个合约，来回切十几下才拼得出来。T 型表一眼看完。

双击任意一格 → 把那一档的合约送回研究页画 K 线。左侧点走认购，右侧点走认沽
（按列号分，不是按行 —— 同一行上下两个合约，点哪边就是哪边）。

IV 是**现场反解**的，不是读库里那张按品种聚合的摘要表：摘要表只有平值 / 偏度 /
曲率三个数，画不出微笑。口径见 `ttrade/optchain.py` 的模块说明。
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ttrade import optchain as OC

_HINT = "font-size: 11px; color: #64748b;"
_UP = QColor("#dc2626")     # 涨：红
_DOWN = QColor("#16a34a")   # 跌：绿
_ATM = QColor("#fef3c7")    # 平值行底色

# 列定义：标题 / 宽度 / 属于哪一侧（c / p / 空＝中间的行权价）
_COLS = [
    ("认购IV", 74, "c"),
    ("认购量", 74, "c"),
    ("认购持仓", 84, "c"),
    ("认购最新", 84, "c"),
    ("认购涨跌", 84, "c"),
    ("行权价", 88, ""),
    ("认沽涨跌", 84, "p"),
    ("认沽最新", 84, "p"),
    ("认沽持仓", 84, "p"),
    ("认沽量", 74, "p"),
    ("认沽IV", 74, "p"),
]


# 股指期权挂在**指数**上，不挂在期货品种上：品种池里是 IF/IH/IC/IM 四个股指
# 期货，期权库里是 IO/HO/MO。不做这个映射的话，在看 IF 的时候点开报价窗，
# 会落到全库第一个品种（AP 苹果）上，得手动翻十几下才找得到
_INDEX_MAP = {"IF": "IO", "IH": "HO", "IM": "MO"}


def _num(v, digits: int = 2, pct: bool = False) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:                      # NaN
        return "—"
    if pct:
        return f"{f * 100:+.1f}%"
    if digits == 0:
        return f"{f:,.0f}"
    return f"{f:,.{digits}f}"


class OptionQuoteDialog(QDialog):
    """T 型报价。`picked` 带出合约代码，宿主决定是画图还是别的。"""

    picked = Signal(str)

    def __init__(self, parent=None, variety: str = "", underlying: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("期权 T 型报价")
        self.resize(1000, 620)
        self._snap: dict = {}
        self._loading = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("品种"))
        self.cb_var = QComboBox()
        self.cb_var.setMinimumWidth(90)
        self.cb_var.currentIndexChanged.connect(self._on_variety)
        row.addWidget(self.cb_var)

        row.addWidget(QLabel("期权链"))
        self.cb_chain = QComboBox()
        self.cb_chain.setMinimumWidth(320)
        self.cb_chain.currentIndexChanged.connect(self._reload)
        row.addWidget(self.cb_chain, 1)

        self.btn_reload = QPushButton("刷新")
        self.btn_reload.setObjectName("Ghost")
        self.btn_reload.clicked.connect(self._reload)
        row.addWidget(self.btn_reload)
        lay.addLayout(row)

        self.lbl_head = QLabel("")
        self.lbl_head.setStyleSheet(_HINT)
        self.lbl_head.setWordWrap(True)
        lay.addWidget(self.lbl_head)

        self.tbl = QTableWidget(0, len(_COLS))
        self.tbl.setHorizontalHeaderLabels([c[0] for c in _COLS])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.doubleClicked.connect(self._on_pick)
        hh = self.tbl.horizontalHeader()
        for i, (_t, w, _s) in enumerate(_COLS):
            self.tbl.setColumnWidth(i, w)
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(QHeaderView.Fixed)
        lay.addWidget(self.tbl, 1)

        self.lbl_note = QLabel("")
        self.lbl_note.setStyleSheet(_HINT)
        self.lbl_note.setWordWrap(True)
        lay.addWidget(self.lbl_note)

        close = QHBoxLayout()
        close.addStretch(1)
        btn = QPushButton("关闭")
        btn.setObjectName("Ghost")
        btn.clicked.connect(self.reject)
        close.addWidget(btn)
        lay.addLayout(close)

        self._fill_varieties(variety, underlying)

    # ---- 填数据

    def _fill_varieties(self, variety: str, underlying: str) -> None:
        self._loading = True
        try:
            self.cb_var.clear()
            for v in OC.varieties():
                self.cb_var.addItem(v, v)
            i = self.cb_var.findData(variety) if variety else -1
            if i < 0 and variety:
                # 品种下拉里是大写（`RB`），期权库里是小写（`rb`）—— 找不到
                # 就按大小写不敏感再试一次
                for k in range(self.cb_var.count()):
                    if str(self.cb_var.itemData(k)).lower() == variety.lower():
                        i = k
                        break
            if i < 0 and variety.upper() in _INDEX_MAP:
                i = self.cb_var.findData(_INDEX_MAP[variety.upper()])
            self.cb_var.setCurrentIndex(max(0, i))
        finally:
            self._loading = False
        self._fill_chains(underlying)

    def _on_variety(self, *_a) -> None:
        if self._loading:
            return
        self._fill_chains("")

    def _fill_chains(self, prefer: str = "") -> None:
        variety = self.cb_var.currentData() or ""
        self._loading = True
        try:
            self.cb_chain.clear()
            df = OC.chains(variety)
            if df is None:
                self.lbl_head.setText("期权库正在被下载占用（DuckDB 单写单读），下完再刷新")
                return
            if df.empty:
                self.lbl_head.setText("本地没有这个品种的期权数据")
                return
            today = pd.Timestamp.today().normalize()
            pick = -1
            for _, r in df.sort_values(["expiry", "underlying"]).iterrows():
                exp = pd.Timestamp(r["expiry"])
                alive = exp >= today
                self.cb_chain.addItem(
                    f"{exp:%Y-%m-%d} 到期 · {r['underlying']}（{int(r['n'])} 档）"
                    + ("" if alive else " · 已到期"),
                    (str(r["underlying"]), exp),
                )
                i = self.cb_chain.count() - 1
                if prefer and str(r["underlying"]) == prefer and pick < 0:
                    pick = i
                if pick < 0 and alive:
                    pick = i
            self.cb_chain.setCurrentIndex(max(0, pick))
        finally:
            self._loading = False
        self._reload()

    def _reload(self, *_a) -> None:
        if self._loading:
            return
        variety = self.cb_var.currentData() or ""
        sel = self.cb_chain.currentData()
        if not variety or not sel:
            return
        underlying, expiry = sel
        snap = OC.snapshot(variety, underlying, expiry)
        self._snap = snap
        if snap["error"]:
            self.lbl_head.setText(snap["error"])
            self.tbl.setRowCount(0)
            self.lbl_note.setText("")
            return
        head = (f"{variety} · {underlying} · {pd.Timestamp(expiry):%Y-%m-%d} 到期"
                f"　剩余 {snap['dte']} 天")
        if snap["spot"]:
            head += f"　标的 {snap['spot']:,.2f}"
        if snap["atm"]:
            head += f"　平值 {snap['atm']:,.0f}"
        head += f"　数据日 {pd.Timestamp(snap['day']):%Y-%m-%d}"
        self.lbl_head.setText(head)
        self._fill_table(snap)
        self.lbl_note.setText(
            snap["note"] or
            "IV 为 Black-76 现场反解（自然日 / 365）。双击左侧单元格看认购、"
            "右侧看认沽；「—」= 深度实值贴边界或当天没成交。"
        )

    def _fill_table(self, snap: dict) -> None:
        rows = snap["rows"]
        self.tbl.setRowCount(len(rows))
        atm = snap.get("atm")
        bold = QFont()
        bold.setBold(True)
        for r in range(len(rows)):
            rec = rows.iloc[r]
            strike = float(rec["strike"])
            is_atm = atm is not None and abs(strike - float(atm)) < 1e-6
            for c, (_title, _w, side) in enumerate(_COLS):
                if side == "":
                    txt = f"{strike:,.0f}"
                elif c == 0 or c == 10:                  # IV
                    txt = _num(rec.get(f"{side}_iv"), 3)
                elif c == 1 or c == 9:                   # 量
                    txt = _num(rec.get(f"{side}_vol"), 0)
                elif c == 2 or c == 8:                   # 持仓
                    txt = _num(rec.get(f"{side}_oi"), 0)
                elif c == 3 or c == 7:                   # 最新
                    txt = _num(rec.get(f"{side}_last"), 1)
                else:                                     # 涨跌
                    txt = _num(rec.get(f"{side}_chg"), 1)
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if side == "":
                    it.setTextAlignment(Qt.AlignCenter)
                    it.setFont(bold)
                elif c in (4, 6):                         # 涨跌列染色
                    v = rec.get(f"{side}_chg")
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = float("nan")
                    if v == v and v > 0:
                        it.setForeground(_UP)
                    elif v == v and v < 0:
                        it.setForeground(_DOWN)
                if is_atm:
                    it.setBackground(_ATM)
                self.tbl.setItem(r, c, it)

    # ---- 选合约

    def _on_pick(self, idx) -> None:
        if idx is None or self._snap.get("rows") is None:
            return
        rows = self._snap["rows"]
        if rows.empty or idx.row() >= len(rows):
            return
        side = _COLS[idx.column()][2] if idx.column() < len(_COLS) else ""
        if not side:
            # 点在中间「行权价」那列：默认给认购，总比什么都不做强
            side = "c"
        code = self._contract_at(idx.row(), side)
        if code:
            self.picked.emit(code)
            self.accept()

    def _contract_at(self, row: int, side: str) -> str:
        """按行权价反查合约代码。表里没存代码列，从 meta 里查一次。"""
        from ttrade import optstore as OS

        variety = self.cb_var.currentData() or ""
        sel = self.cb_chain.currentData()
        if not sel:
            return ""
        underlying, expiry = sel
        strike = float(self._snap["rows"].iloc[row]["strike"])
        df = OS.list_strikes(variety, underlying, expiry)
        if df.empty:
            return ""
        cp = "C" if side == "c" else "P"
        hit = df[(df["strike"].astype(float).round(6) == round(strike, 6))
                 & (df["cp"].astype(str).str.upper().str[:1] == cp)]
        return str(hit["contract"].iloc[0]) if len(hit) else ""
