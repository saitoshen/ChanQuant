"""把模型回复的 markdown 变成 Qt 能吃的富文本。

## 为什么不用 markdown 库

`markdown-it-py` / `mistune` 产出的是给浏览器看的 HTML —— `<h2>`、嵌套 `<p>`、
`class="codehilite"`。而 `QTextDocument` 只认一小撮 CSS，`class` 会被整个忽略，
拿到之后还是得遍历一遍重写成行内样式。多一层依赖做同样的活，不如手写两百行。

## Qt 富文本的能力边界（逐像素实测过，别凭印象改）

| 样式 | 支持 | 备注 |
|---|---|---|
| `background` / `background-color` | ✅ | 卡片底色靠它 |
| `table border="1"` | ✅ | 默认 `#c0c0c0` 灰线 |
| `<td width="30%">` | ✅ | **列宽必须写在每个 td/th 上**，只写 th 不生效 |
| `pre` 的 `background-color` | ✅ | 代码块底色 |
| `<img src="data:image/png;base64,...">` | ✅ | 截图能内联，不用落临时文件 |
| `border-left` | ⚠️ **只在 td 上生效** | 写在 `div` / `p` 上会被直接忽略（`toHtml()` 里都不出现）。所以色条要用「窄 td + 内容 td」拼一张 1×2 表格 |
| `border-radius` | ❌ | 不支持，方角。0px 和 12px 渲染完全一致 |

所以卡片能做得挺像样，唯一妥协是方角。

## Block 协议

工具返回和模型回复共用一套结构，渲染器只认 Block，不关心它从哪来。以后要是把
`QTextBrowser` 换成 `QScrollArea` + 控件列表（为了圆角、悬停、卡片内按钮），
只换 `render()`，解析和工具都不用动。

```python
{"type": "text",    "text": "支持行内 markdown"}
{"type": "head",    "level": 1..4, "text": "..."}
{"type": "list",    "items": ["...", "..."]}
{"type": "quote",   "text": "..."}
{"type": "hr"}
{"type": "code",    "lang": "", "text": "..."}
{"type": "table",   "columns": [...], "rows": [[...]], "note": "", "align": ["l", "r"]}
{"type": "metrics", "title": "", "items": [["名", "值", "good|bad|flat"]], "link": ["文字", "url"]}
{"type": "warning", "title": "", "lines": ["..."], "level": "bad|warn"}
{"type": "image",   "b64": "...", "width": 260, "caption": ""}
{"type": "link",    "text": "...", "url": "..."}
```
"""

from __future__ import annotations

import re
from html import escape as _escape

# ---------------------------------------------------------------- 配色

C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_GOOD = "#166534"
C_BAD = "#b91c1c"
C_LINE = "#d1d5db"
C_HEAD_BG = "#eef2f7"      # 表头 / 行内码底
C_CODE_BG = "#eef1f5"
C_WARN_BG = "#fdeaea"
C_NOTE_BG = "#f6f7f9"

MONO = "Consolas, 'Courier New', monospace"
BODY_PT = "13px"


def _esc(s: str) -> str:
    """转义成 HTML 安全的文本。`quote=False` 是为了别把中文引号也搅进去。"""
    return _escape(str(s), quote=False)


# ---------------------------------------------------------------- 行内

_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_LINK = re.compile(r"\[([^\]\n]+)\]\(((?:https?|file|app)://[^)\s]+)\)")


def inline(s: str) -> str:
    """行内 markdown → HTML。**先转义再替换**，这样用户内容里的 `<` 不会变成标签。

    只认 `*` 不认 `_`：因子代号里全是下划线（`mom_20`、`f_vol_price_corr`），
    把 `_` 当斜体标记会把半个表格搅成斜体。
    """
    s = _esc(s)
    s = _CODE_SPAN.sub(
        lambda m: f'<span style="font-family:{MONO};background-color:{C_HEAD_BG}">'
                  f'{m.group(1)}</span>', s)
    s = _BOLD.sub(r"<b>\1</b>", s)
    s = _ITALIC.sub(r"<i>\1</i>", s)
    s = _LINK.sub(r'<a href="\2">\1</a>', s)
    return s


# ---------------------------------------------------------------- 解析

_SEP_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_HEADING = re.compile(r"^(#{1,4})\s+(.+)$")
_HR = re.compile(r"^\s*([-*_])\s*(?:\1\s*){2,}$")
_BULLET = re.compile(r"^\s*([-*+]|\d+[.)])\s+(.*)$")


def _split_cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _looks_numeric(cell: str) -> bool:
    t = cell.replace(",", "").replace(" ", "").rstrip("%")
    if not t or t in ("—", "-", "–"):
        return False
    try:
        float(t)
    except ValueError:
        return False
    return True


def _guess_align(columns: list[str], rows: list[list[str]]) -> list[str]:
    """逐列猜对齐：整列都是数字就右对齐。

    量化表格里数字右对齐是基本体面 —— 左对齐时 `0.031` 和 `-0.128` 的小数点
    对不上，扫一眼比不出来。
    """
    out = []
    for j in range(len(columns)):
        vals = [r[j] for r in rows if j < len(r) and str(r[j]).strip()]
        if vals and all(_looks_numeric(str(v)) for v in vals):
            out.append("r")
        else:
            out.append("l")
    return out


def _display_len(s) -> int:
    """按「终端宽度」算长度：中文占 2，ASCII 占 1。

    不这么算的话「参数」和 `mom_20` 会被当成一样长，列宽就分不匀。
    """
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(s))


def _col_widths(columns: list[str], rows: list[list[str]],
                min_pct: int = 7) -> list[int]:
    """按每列最长内容分宽度（百分比）。

    Qt 的自动列宽会把短列撑得很宽 —— 「参数 | 值 | 说明」三列会变成
    三等分，值那一列右边空一大片。自己算一遍按内容分配，看着才像表格。
    """
    n = len(columns)
    if not n:
        return []
    lens = []
    for j in range(n):
        widest = _display_len(columns[j])
        for r in rows:
            if j < len(r):
                widest = max(widest, _display_len(r[j]))
        lens.append(max(widest, 4))

    total = float(sum(lens))
    pcts = [max(min_pct, int(round(x * 100 / total))) for x in lens]

    # 补齐到 100：差多少都加到最长的那列上（它最能吸收，也最不容易挤坏短列）
    diff = 100 - sum(pcts)
    if diff:
        k = pcts.index(max(pcts))
        pcts[k] = max(min_pct, pcts[k] + diff)
    return pcts


def _read_table(lines: list[str], i: int) -> tuple[list[str], list[list[str]], int]:
    """从 `lines[i]` 读一张 markdown 表格。返回 (列名, 行, 下一行下标)。"""
    columns = _split_cells(lines[i])
    i += 2                                   # 跳过表头和分隔行
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = _split_cells(lines[i])
        # 列数对不上的（模型偶尔少写一个竖线）补空，别让渲染错位
        if len(cells) < len(columns):
            cells += [""] * (len(columns) - len(cells))
        rows.append(cells[:len(columns)])
        i += 1
    return columns, rows, i


def parse(text: str) -> list[dict]:
    """markdown → Block 列表。"""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[dict] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append({"type": "text", "text": "\n".join(buf)})
            buf.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()

        # 代码块（围栏）
        if s.startswith("```"):
            flush()
            lang = s[3:].strip()
            i += 1
            code = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1                            # 吃掉收尾围栏（没有也不卡住）
            out.append({"type": "code", "lang": lang, "text": "\n".join(code)})
            continue

        # 表格：本行有竖线、下一行是分隔行
        if "|" in s and i + 1 < n and _SEP_ROW.match(lines[i + 1]) and "-" in lines[i + 1]:
            flush()
            columns, rows, i = _read_table(lines, i)
            if columns:
                out.append({"type": "table", "columns": columns, "rows": rows,
                            "align": _guess_align(columns, rows)})
            continue

        # 引用
        if s.startswith(">"):
            flush()
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out.append({"type": "quote", "text": "\n".join(quote)})
            continue

        # 标题
        m = _HEADING.match(s)
        if m:
            flush()
            out.append({"type": "head", "level": len(m.group(1)), "text": m.group(2)})
            i += 1
            continue

        # 分隔线
        if _HR.match(line):
            flush()
            out.append({"type": "hr"})
            i += 1
            continue

        # 列表（连续的条目算一个块）
        if _BULLET.match(line):
            flush()
            items = []
            while i < n:
                mm = _BULLET.match(lines[i])
                if not mm:
                    break
                items.append(mm.group(2))
                i += 1
            out.append({"type": "list", "items": items})
            continue

        if not s:
            flush()
            i += 1
            continue

        buf.append(line.rstrip())
        i += 1

    flush()
    return out


# ---------------------------------------------------------------- 渲染

def _wrap(html: str, style: str) -> str:
    return f'<div style="{style}">{html}</div>'


def _card(inner: str, bar: str = "", bg: str = C_NOTE_BG,
          pad: str = "8px 10px", margin: str = "6px 0 10px 0") -> str:
    """带左侧色条的卡片。

    色条必须用**窄的 td** 拼，不能写 `border-left` —— 实测 Qt 只在表格单元格上
    认单边边框，写在 `div` / `p` 上连 `toHtml()` 里都不会出现，等于没写。
    """
    bar_cell = f'<td bgcolor="{bar}" width="5"></td>' if bar else ""
    return (
        f'<div style="margin:{margin}">'
        f'<table cellpadding="0" cellspacing="0" width="100%"><tr>'
        f'{bar_cell}<td bgcolor="{bg}" style="padding:{pad}">{inner}</td>'
        f'</tr></table></div>'
    )


def _render_table(b: dict) -> str:
    columns = b.get("columns") or []
    rows = b.get("rows") or []
    align = b.get("align") or ["l"] * len(columns)
    pcts = _col_widths(columns, rows)

    head = "".join(
        f'<th width="{pcts[j]}%" align="{"right" if align[j] == "r" else "left"}" '
        f'style="font-weight:700;color:{C_TEXT}">{inline(c)}</th>'
        for j, c in enumerate(columns)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td width="{pcts[j]}%" align="{"right" if align[j] == "r" else "left"}" '
            f'style="color:{C_TEXT}">{inline(row[j]) if j < len(row) else ""}</td>'
            for j in range(len(columns))
        )
        body.append(f"<tr>{cells}</tr>")

    html = (
        f'<table border="1" cellpadding="5" cellspacing="0" width="100%">'
        f'<tr bgcolor="{C_HEAD_BG}">{head}</tr>{"".join(body)}</table>'
    )
    note = b.get("note") or ""
    if note:
        html += (f'<p style="margin:3px 0 8px 0;color:{C_MUTED};font-size:11px">'
                 f'{inline(note)}</p>')
    return _wrap(html, "margin:6px 0 10px 0")


def _render_metrics(b: dict) -> str:
    """回测结果卡片：左名右值，值按好坏染色，底部一个链接。"""
    items = b.get("items") or []
    rows = []
    for it in items:
        name = str(it[0]) if len(it) > 0 else ""
        value = str(it[1]) if len(it) > 1 else ""
        flag = str(it[2]) if len(it) > 2 else "flat"
        color = {"good": C_GOOD, "bad": C_BAD}.get(flag, C_TEXT)
        rows.append(
            f'<tr><td width="62%" style="color:{C_MUTED}">{inline(name)}</td>'
            f'<td width="38%" align="right" style="color:{color};font-weight:700">'
            f'{inline(value)}</td></tr>'
        )
    title = b.get("title") or "结果"
    html = (
        f'<table cellpadding="0" cellspacing="0" width="100%"><tr>'
        f'<td bgcolor="{C_HEAD_BG}" style="padding:6px 10px">'
        f'<span style="font-weight:700;color:{C_TEXT}">{inline(title)}</span></td></tr></table>'
        f'<table border="1" cellpadding="5" cellspacing="0" width="100%">{"".join(rows)}</table>'
    )
    link = b.get("link")
    if link and len(link) >= 2:
        html += (f'<p style="margin:4px 0 8px 0"><a href="{_esc(link[1])}">'
                 f'{inline(link[0])}</a></p>')
    return _wrap(html, "margin:6px 0 10px 0")


def _render_warning(b: dict) -> str:
    """风控 / 错误块：左侧粗色条 + 浅底。"""
    level = b.get("level") or "bad"
    bar = C_BAD if level == "bad" else "#b45309"
    bg = C_WARN_BG if level == "bad" else "#fef6e7"
    title = b.get("title") or ""
    parts = []
    if title:
        parts.append(f'<p style="margin:0 0 4px 0;font-weight:700;color:{bar}">'
                     f'{inline(title)}</p>')
    for ln in b.get("lines") or []:
        parts.append(f'<p style="margin:0;color:{C_TEXT}">{inline(ln)}</p>')
    return _card("".join(parts) or "&nbsp;", bar=bar, bg=bg)


def _render_code(b: dict) -> str:
    code = b.get("text") or ""
    lang = b.get("lang") or ""
    head = (f'<p style="margin:0;color:{C_MUTED};font-size:11px">{_esc(lang)}</p>'
            if lang else "")
    return _wrap(
        head + f'<pre style="font-family:{MONO};background-color:{C_CODE_BG};'
               f'font-size:12px;margin:0">{_esc(code)}</pre>',
        "margin:6px 0 10px 0",
    )


def _render_image(b: dict) -> str:
    b64 = b.get("b64") or ""
    if not b64:
        return ""
    mime = b.get("mime") or "image/jpeg"
    width = int(b.get("width") or 260)
    caption = b.get("caption") or ""
    html = f'<img src="data:{mime};base64,{b64}" width="{width}">'
    if caption:
        html += (f'<p style="margin:2px 0 0 0;color:{C_MUTED};font-size:11px">'
                 f'{inline(caption)}</p>')
    return _wrap(html, "margin:6px 0 10px 0")


def _render_text(b: dict) -> str:
    """段落。块内的软换行变 `<br>`，段与段之间靠块边界。"""
    text = b.get("text") or ""
    lines = [inline(x) for x in text.split("\n")]
    return f'<p style="margin:2px 0 6px 0;line-height:160%">{"<br>".join(lines)}</p>'


def _render_head(b: dict) -> str:
    level = int(b.get("level") or 3)
    size = {1: "16px", 2: "15px", 3: "14px", 4: "13px"}.get(level, "13px")
    return (f'<p style="margin:10px 0 4px 0;font-size:{size};font-weight:700;'
            f'color:{C_TEXT}">{inline(b.get("text") or "")}</p>')


def _render_list(b: dict) -> str:
    items = b.get("items") or []
    lis = "".join(
        f'<li style="line-height:160%">{inline(x)}</li>' for x in items
    )
    return f'<ul style="margin:2px 0 8px 0">{lis}</ul>'


def _render_quote(b: dict) -> str:
    text = "<br>".join(inline(x) for x in (b.get("text") or "").split("\n"))
    return _card(
        f'<p style="margin:0;color:{C_TEXT};line-height:160%">{text}</p>',
        bar=C_LINE, bg=C_NOTE_BG, pad="6px 10px", margin="6px 0 8px 0",
    )


def _render_link(b: dict) -> str:
    url = _esc(b.get("url") or "")
    text = inline(b.get("text") or url)
    return f'<p style="margin:2px 0 6px 0"><a href="{url}">{text}</a></p>'


_RENDERERS = {
    "text": _render_text,
    "head": _render_head,
    "list": _render_list,
    "quote": _render_quote,
    "hr": lambda b: f'<hr style="color:{C_LINE}">',
    "code": _render_code,
    "table": _render_table,
    "metrics": _render_metrics,
    "warning": _render_warning,
    "image": _render_image,
    "link": _render_link,
}


def render(blocks: list[dict]) -> str:
    """Block 列表 → Qt 富文本 HTML。认不出的类型直接跳过（别整页空白）。"""
    out = []
    for b in blocks or []:
        fn = _RENDERERS.get(str(b.get("type") or ""))
        if fn is not None:
            out.append(fn(b))
    return "".join(out)


def render_text(text: str) -> str:
    """markdown 一步到 HTML。"""
    return render(parse(text))


# ---------------------------------------------------------------- 给工具用的构造器

def block_text(text: str) -> dict:
    return {"type": "text", "text": text}


def block_table(columns, rows, note: str = "", align=None) -> dict:
    """表格块。`align` 不给就自动按「整列是数字则右对齐」猜。"""
    columns = [str(c) for c in columns]
    rows = [[("" if v is None else str(v)) for v in r] for r in rows]
    return {"type": "table", "columns": columns, "rows": rows, "note": note,
            "align": align or _guess_align(columns, rows)}


def block_code(code: str, lang: str = "") -> dict:
    return {"type": "code", "lang": lang, "text": code}


def block_metrics(title: str, items, link=None) -> dict:
    """`items` 里每项是 `(名, 值)` 或 `(名, 值, "good"|"bad"|"flat")`。"""
    return {"type": "metrics", "title": title,
            "items": [list(it) for it in items], "link": link}


def block_warning(title: str, lines, level: str = "bad") -> dict:
    return {"type": "warning", "title": title, "lines": list(lines), "level": level}


def block_image(b64: str, width: int = 260, caption: str = "", mime: str = "image/jpeg") -> dict:
    return {"type": "image", "b64": b64, "width": width, "caption": caption, "mime": mime}


def block_link(text: str, url: str) -> dict:
    return {"type": "link", "text": text, "url": url}
