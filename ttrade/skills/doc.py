"""读写文档：Excel / PDF / Word 进，PDF / xlsx 出。

## 为什么是「按 file_id 取」而不是把内容塞进消息

用户在输入框里丢一份 200 行 × 15 列的因子表，如果整段塞进 user 消息，六万字里
一半是重复的日期和空格，列还对不齐，而且**读完没有第二次机会** —— 模型想问
「第 2 个 sheet 呢」，只能让用户重发。

现在文件解析一次存进 `SessionFiles`，消息里只放一段摘要（sheet 名、行列数、
列名、前 5 行），模型觉得不够就调 `read_excel(file_id="f1", rows=100)` 再取。
摘要和取数的分工见 `ttrade/session_files.py`。

## 出文件落到哪

`ttrade/docgen.py` 的 `export_dir()`（用户目录下的 `export/`）。不往 ChanQuant
安装目录写 —— 打包版可能装在只读位置，而且用户找不着。
"""

from __future__ import annotations

import os
import re

from ttrade import blocks as B
from ttrade import docgen as DG
from ttrade.skills.base import (
    RISK_READ,
    RISK_WRITE,
    SkillResult,
    Tool,
    register,
)
from ttrade.session_files import slice_frame

MAX_ROWS = 200
MAX_COLS = 40
# 一次最多往对话里贴这么多字符。pdf 一份几十页，全贴进去会把上下文挤爆
MAX_TEXT_SHOWN = 6000


def _session(ctx):
    sess = getattr(ctx, "session", None)
    if sess is None:
        raise RuntimeError("这个会话里还没有附件缓存，先让用户把文件拖进输入框")
    return sess


def _file(ctx, file_id: str):
    sess = _session(ctx)
    item = sess.get(file_id)
    if item is None:
        have = "、".join(f"{f.file_id}={f.name}" for f in sess.all()) or "（空）"
        raise RuntimeError(f"没有 {file_id} 这个附件（现有：{have}）")
    if not item.ok:
        raise RuntimeError(f"附件 {file_id}（{item.name}）读不了：{item.note}")
    return item


def _fmt(v) -> str:
    """表格单元格 → 字符串。NaN 一律显示破折号，不要露出 `nan`。"""
    if v is None:
        return ""
    try:
        if v != v:                   # NaN
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:.4f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d") if v.hour == 0 and v.minute == 0 \
            else v.strftime("%Y-%m-%d %H:%M")
    return str(v)


# ------------------------------------------------------------------ read_excel

def _read_excel(ctx, p: dict) -> SkillResult:
    item = _file(ctx, p["file_id"])
    if item.kind != "table":
        raise RuntimeError(
            f"{item.name} 不是表格，用 parse_pdf / read_docx 读它"
        )
    df = item.frame(p.get("sheet") or None)
    if df is None or df.empty:
        names = "、".join(item.sheets) or "（没有 sheet）"
        raise RuntimeError(f"sheet「{p.get('sheet') or ''}」是空的（这份文件里有：{names}）")

    cols = [c for c in (p.get("cols") or []) if c in df.columns] or None
    if p.get("cols") and cols is None:
        raise RuntimeError(
            f"这些列不存在：{p['cols']}。可用的列是：{[str(c) for c in df.columns][:30]}"
        )
    part = slice_frame(df, rows=min(int(p.get("rows") or 20), MAX_ROWS),
                       cols=cols, start=int(p.get("start") or 0))
    show = part.iloc[:, :MAX_COLS]

    rows = [[_fmt(v) for v in r] for r in show.itertuples(index=False)]
    total = len(df)
    start = max(0, int(p.get("start") or 0))
    end = start + len(show)

    note = f"第 {start + 1}–{end} 行，共 {total} 行 × {len(df.columns)} 列"
    if end < total:
        note += f'｜继续取：read_excel(file_id="{item.file_id}", start={end})'
    blocks = [
        B.block_metrics(f"{item.name} · {item.main_sheet() if not p.get('sheet') else p['sheet']}", [
            ("行数", f"{total:,}", "flat"),
            ("列数", str(len(df.columns)), "flat"),
            ("本次", f"{len(show)} 行", "flat"),
            ("sheet", str(len(item.sheets)), "flat"),
        ]),
        B.block_table([str(c) for c in show.columns], rows, note=note),
    ]
    summary = (f"{item.name} 的 {note}；列：{'、'.join(str(c) for c in df.columns)[:400]}")
    return SkillResult(True, summary=f"读取 {item.name}：{note}", blocks=blocks,
                       feed=summary)


# ------------------------------------------------------------------ parse_pdf

_PAGE_SEP = re.compile(r"^—— 第 (\d+) 页 ——$", re.M)


def _pdf_pages(text: str) -> list[tuple[int, str]]:
    """把 `read_text_file()` 拼出来的正文拆回一页一页。"""
    marks = list(_PAGE_SEP.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((int(m.group(1)), text[m.end():end].strip()))
    return out


def _parse_pages(spec: str, total: int) -> list[int]:
    """`"1-5"` / `"3"` / `"2,5,9"` → 页码列表。空串表示全部（但会被截断）。"""
    spec = str(spec or "").strip()
    if not spec:
        return list(range(1, total + 1))
    want: list[int] = []
    for chunk in spec.replace("，", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.fullmatch(r"(\d+)\s*[-~]\s*(\d+)", chunk)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            want.extend(range(min(a, b), max(a, b) + 1))
        elif chunk.isdigit():
            want.append(int(chunk))
    return [n for n in dict.fromkeys(want) if 1 <= n <= total]


def _parse_pdf(ctx, p: dict) -> SkillResult:
    item = _file(ctx, p["file_id"])
    if item.kind != "text":
        raise RuntimeError(f"{item.name} 不是文本类文件，表格用 read_excel 读")
    if not item.name.lower().endswith(".pdf"):
        raise RuntimeError(f"{item.name} 不是 PDF")

    pages = _pdf_pages(item.text)
    if not pages:
        raise RuntimeError(f"{item.name} 里没抽出文字（可能是扫描件）")

    want = _parse_pages(p.get("pages", ""), len(pages))
    if not want:
        raise RuntimeError(f"页码超出范围，这份 PDF 一共 {len(pages)} 页有文字")

    got = [(n, body) for n, body in pages if n in want]
    text = "\n\n".join(f"—— 第 {n} 页 ——\n{b}" for n, b in got)
    clipped = len(text) > MAX_TEXT_SHOWN
    shown = text[:MAX_TEXT_SHOWN] if clipped else text

    note = f"{len(got)} 页 / 共 {len(pages)} 页有文字"
    blocks = [
        B.block_metrics(item.name, [
            ("页数", str(len(pages)), "flat"),
            ("本次", f"{len(got)} 页", "flat"),
            ("字数", f"{len(text):,}", "flat"),
        ]),
        B.block_text(shown),
    ]
    if clipped:
        blocks.append(B.block_warning(
            "只贴了前 6000 字符", ["要看后面的，用 pages=\"7-9\" 分段取"], "warn"))
    summary = f"{item.name} {note}：\n{text}"
    return SkillResult(True, summary=f"读取 {item.name}：{note}", blocks=blocks,
                       feed=summary)


# ------------------------------------------------------------------ read_docx

def _read_docx(ctx, p: dict) -> SkillResult:
    item = _file(ctx, p["file_id"])
    if item.kind != "text":
        raise RuntimeError(f"{item.name} 不是文本类文件，表格用 read_excel 读")

    text = item.text
    clipped = len(text) > MAX_TEXT_SHOWN
    shown = text[:MAX_TEXT_SHOWN] if clipped else text
    blocks = [
        B.block_metrics(item.name, [
            ("字数", f"{len(text):,}", "flat"),
            ("段落", str(text.count(chr(10)) + 1), "flat"),
        ]),
        B.block_text(shown),
    ]
    if clipped:
        blocks.append(B.block_warning(
            "只贴了前 6000 字符", ["要全文就让用户直接问其中一段"], "warn"))
    return SkillResult(True, summary=f"读取 {item.name}（{len(text)} 字符）",
                       blocks=blocks, feed=f"{item.name} 的正文：\n{text}")


# ------------------------------------------------------------------ write_xlsx

def _write_xlsx(ctx, p: dict) -> SkillResult:
    sheets = p.get("sheets") or []
    if not sheets:
        raise RuntimeError("sheets 是空的，至少给一个表")
    path = DG.export_path(p.get("name") or "导出.xlsx", ".xlsx", "导出")
    DG.write_xlsx(path, sheets)

    rows = sum(len(s.get("rows") or []) for s in sheets)
    blocks = [
        B.block_metrics("已导出 Excel", [
            ("文件", os.path.basename(path), "flat"),
            ("sheet", str(len(sheets)), "flat"),
            ("数据行", f"{rows:,}", "flat"),
        ]),
        B.block_link("打开所在文件夹", "file:///" + os.path.dirname(path).replace("\\", "/")),
    ]
    return SkillResult(True, summary=f"已导出 {os.path.basename(path)}（{len(sheets)} 个 sheet，{rows} 行）",
                       blocks=blocks, feed=f"已导出到 {path}")


# ------------------------------------------------------------------ write_pdf

def _write_pdf(ctx, p: dict) -> SkillResult:
    md = str(p.get("markdown") or "").strip()
    if not md:
        raise RuntimeError("markdown 是空的，没东西可导出")
    title = str(p.get("title") or "").strip() or "ChanQuant 导出"
    path = DG.export_path(p.get("name") or f"{title}.pdf", ".pdf", "导出")
    DG.write_pdf(path, markdown=md, title=title)

    size = os.path.getsize(path) / 1024
    blocks = [
        B.block_metrics("已导出 PDF", [
            ("文件", os.path.basename(path), "flat"),
            ("大小", f"{size:.0f} KB", "flat"),
            ("字数", f"{len(md):,}", "flat"),
        ]),
        B.block_link("打开所在文件夹", "file:///" + os.path.dirname(path).replace("\\", "/")),
    ]
    return SkillResult(True, summary=f"已导出 {os.path.basename(path)}（{size:.0f} KB）",
                       blocks=blocks, feed=f"已导出到 {path}")


# ------------------------------------------------------------------ 注册

register(Tool(
    name="read_excel",
    group="doc",
    label="读表格",
    description="按 file_id 读附件的表格内容（可分 sheet、分页、选列）",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["file_id"],
        "properties": {
            "file_id": {"type": "str"},
            "sheet": {"type": "str", "default": ""},
            "rows": {"type": "int", "default": 20, "minimum": 1, "maximum": MAX_ROWS},
            "start": {"type": "int", "default": 0, "minimum": 0},
            "cols": {"type": "list"},
        },
    },
    handler=_read_excel,
    prompt=(
        "**读表格**：`read_excel(file_id=\"f1\", sheet=\"因子\", rows=50)`。"
        "附件摘要里已经给了前 5 行和前 8 列，**摘要不够时**才调它取更多。"
        "要接着往后读就用 `start=`；只关心几列就给 `cols=[\"日期\",\"动量\"]`。"
    ),
    max_turns=3,
))

register(Tool(
    name="parse_pdf",
    group="doc",
    label="读 PDF",
    description="按 file_id 读附件的 PDF 正文，可指定页码",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["file_id"],
        "properties": {
            "file_id": {"type": "str"},
            "pages": {"type": "str", "default": ""},
        },
    },
    handler=_parse_pdf,
    prompt=(
        "**读 PDF**：`parse_pdf(file_id=\"f1\", pages=\"1-5\")`。"
        "不填 pages 就是全文，但一次最多贴 6000 字符，所以长文档要分段取。"
        "扫描件（图片型 PDF）抽不出文字，会明确报错 —— 那种情况让用户发截图。"
    ),
    max_turns=3,
))

register(Tool(
    name="read_docx",
    group="doc",
    label="读 Word",
    description="按 file_id 读附件的 Word 正文（只取文字，不保留表格行列）",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["file_id"],
        "properties": {"file_id": {"type": "str"}},
    },
    handler=_read_docx,
    prompt="**读 Word**：`read_docx(file_id=\"f1\")`。表格会变成按行排列的文字，行列关系会丢。",
    max_turns=2,
))

register(Tool(
    name="write_xlsx",
    group="doc",
    label="导出 Excel",
    description="把一组表格导出成 xlsx，落到用户目录的 export/ 下",
    risk=RISK_WRITE,
    inputSchema={
        "type": "object",
        "required": ["sheets"],
        "properties": {
            "name": {"type": "str", "default": ""},
            "sheets": {"type": "list"},
        },
    },
    handler=_write_xlsx,
    prompt=(
        "**导出 Excel**：`write_xlsx`。`sheets` 是一个数组，每项 "
        "`{\"sheet\":\"因子\",\"columns\":[\"日期\",\"动量\"],\"rows\":[[\"2026-01-05\",0.31]]}`。"
        "用户说「导出来 / 存成 Excel / 给我一份表」时用它。**不要**把整张表也贴回对话里，"
        "表格块本身已经在屏幕上显示了。"
    ),
))

register(Tool(
    name="write_pdf",
    group="doc",
    label="导出 PDF",
    description="把一段 markdown 导出成 PDF 报告，落到用户目录的 export/ 下",
    risk=RISK_WRITE,
    inputSchema={
        "type": "object",
        "required": ["markdown"],
        "properties": {
            "name": {"type": "str", "default": ""},
            "title": {"type": "str", "default": ""},
            "markdown": {"type": "str"},
        },
    },
    handler=_write_pdf,
    prompt=(
        "**导出 PDF**：`write_pdf`。`markdown` 就是报告的正文，支持标题、表格、"
        "列表、代码块 —— 和你平时回答的写法一样。用户说「出一份报告 / 导成 PDF」时用它。"
        "报告本身要**先在对话里说清楚**，再调这个动作导出，不要只给一个文件链接。"
    ),
))
