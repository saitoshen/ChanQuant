"""把 Block / markdown 变成 PDF 和 xlsx。

## 为什么和聊天共用渲染器

导出走的是 `QTextDocument` + `QPdfWriter`（`ttrade/factor_pdf.py` 已经在用这一套），
而聊天区用的也是 `QTextDocument`。既然引擎是同一个，就不该有两份 HTML 生成器 ——
所以这里直接用 `ttrade/render.py`，聊天里看到的表格、指标卡、告警块，导出后长得
一样。改了样式两边一起变，不会出现「界面上是新的、导出的还是旧版」。

## 不引 reportlab / weasyprint

reportlab 要自己排版（每个表格手算列宽、自己断页），weasyprint 要 GTK 一整套
（Windows 上装不上）。`QTextDocument` 已经把断页、表格、字体都做了，而且 PySide6
是已经打了包的依赖，多加 0 字节体积。
"""

from __future__ import annotations

import os

from ttrade import blocks as B
from ttrade import render as R

# A4 纸边距（毫米）。上下留白比左右多一点，打印出来才不挤
PAGE_MARGIN = (16, 14, 16, 14)


def _style() -> str:
    return (
        "font-family:'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif;"
        "font-size:10pt;color:#1f2937;"
    )


def document_html(blocks: list[dict], title: str = "") -> str:
    """Block 列表 → 一份完整的 HTML 文档（带标题、封面行）。"""
    body = R.render(blocks or [])
    head = ""
    if title.strip():
        head = (
            f"<p style='font-size:19pt;font-weight:700;margin:0 0 2px 0'>{R.inline(title)}</p>"
            f"<p style='font-size:9pt;color:#6b7280;margin:0 0 14px 0'>"
            f"ChanQuant · 由交易助手生成</p>"
        )
    return (
        "<html><head><meta charset='utf-8'></head>"
        f"<body style=\"{_style()}\">{head}{body}</body></html>"
    )


def markdown_html(text: str, title: str = "") -> str:
    """markdown 一步到完整 HTML 文档。

    正文第一块已经是标题（`#` / `##`）时**不再加封面标题** —— 模型写报告习惯
    自己起个 `# 螺纹周报`，再叠一个 `title` 就是同一行字印两遍。
    """
    blocks = R.parse(text or "")
    first = blocks[0] if blocks else None
    if first is not None and first.get("type") == "head" and first.get("level", 9) <= 2:
        title = ""
    return document_html(blocks, title)


def html_to_pdf(html: str, path: str, title: str = "") -> str:
    """写 PDF。**必须在有 QGuiApplication 的进程里调用**（Qt 的字体引擎要它）。"""
    from PySide6.QtCore import QMarginsF, QSizeF
    from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    w = QPdfWriter(path)
    w.setPageSize(QPageSize(QPageSize.A4))
    w.setResolution(144)
    w.setPageMargins(QMarginsF(*PAGE_MARGIN), QPageLayout.Millimeter)
    w.setTitle(title or "ChanQuant 导出")
    w.setCreator("ChanQuant")

    doc = QTextDocument()
    doc.setHtml(html)
    # 页面尺寸要按**设备像素**给：QPdfWriter 的 width/height 已经算上了 DPI 与页边距
    doc.setPageSize(QSizeF(w.width(), w.height()))
    doc.print_(w)
    return path


def write_pdf(path: str, markdown: str = "", title: str = "",
              blocks: list[dict] | None = None) -> str:
    """markdown 或 Block → PDF。两样给一样就行。"""
    if blocks is not None:
        html = document_html(blocks, title)
    else:
        html = markdown_html(markdown, title)
    return html_to_pdf(html, path, title)


# ------------------------------------------------------------------ xlsx


def write_xlsx(path: str, sheets: list[dict]) -> str:
    """写一个多 sheet 的 xlsx。

    `sheets` 每项 `{"sheet": "名字", "columns": [...], "rows": [[...]]}`。
    只做「表头加粗 + 冻结首行 + 按内容粗估列宽」这三件事 —— 导出的表是给人
    接着用的，不是给人欣赏的；再加样式就得引一套样式配置，收益不成比例。
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)

    for i, spec in enumerate(sheets or []):
        name = str(spec.get("sheet") or f"Sheet{i + 1}")[:31]  # Excel 上限 31 字
        ws = wb.create_sheet(name)
        cols = [str(c) for c in (spec.get("columns") or [])]
        rows = spec.get("rows") or []
        if cols:
            ws.append(cols)
            for cell in ws[1]:
                cell.font = Font(bold=True)
            ws.freeze_panes = "A2"
        for r in rows:
            ws.append([_cell(v) for v in r])

        # 列宽：按「中文 2、ASCII 1」估，夹在 8~42 之间。太窄看不清，太宽要横向拖
        widths = [_disp(c) for c in cols]
        for r in rows[:200]:
            for j, v in enumerate(r):
                if j < len(widths):
                    widths[j] = max(widths[j], _disp(v))
        for j, wd in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = min(42, max(8, wd + 2))

    if not wb.sheetnames:
        wb.create_sheet("Sheet1")
    wb.save(path)
    return path


def _disp(v) -> int:
    s = "" if v is None else str(v)
    return sum(2 if ord(ch) > 127 else 1 for ch in s)


def _cell(v):
    """Excel 里 `1/0`、`NaN`、超长字符串都要收敛一下，否则 openpyxl 会告警或写坏。"""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    if isinstance(v, (int, float, bool)):
        return v
    s = str(v)
    return s[:32000]


# ------------------------------------------------------------------ 导出目录


def export_dir() -> str:
    """导出的文件放哪。跟「研报库」同一个目录，用户不用记两个地方。"""
    from ttrade.paths import user_dir

    path = os.path.join(user_dir(), "export")
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(name: str, default: str = "导出") -> str:
    """把模型给的文件名收敛成能落盘的名字。

    模型经常给 `回测结果 2026/10/31.pdf` 这种带斜杠的，直接拼路径会写到别的
    目录去 —— 这里把所有路径分隔符和 Windows 保留字符都换掉。
    """
    base = os.path.basename(str(name or "").replace("\\", "/")).strip()
    bad = '<>:"/\\|?*'
    base = "".join("_" if ch in bad or ord(ch) < 32 else ch for ch in base).strip(" .")
    if not base:
        base = default
    return base[:80]


def export_path(name: str, suffix: str, default: str = "导出") -> str:
    """`export_dir()` 下一个安全的、带正确后缀的完整路径。重名就加序号。"""
    base = safe_name(name, default)
    stem, ext = os.path.splitext(base)
    if ext.lower() != suffix.lower():
        stem = base
    path = os.path.join(export_dir(), stem + suffix)
    n = 2
    while os.path.exists(path):
        path = os.path.join(export_dir(), f"{stem} ({n}){suffix}")
        n += 1
    return path
