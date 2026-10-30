"""对话里的结构化块。

工具返回、模型回复、文档导出三处共用同一套结构 —— 渲染层（`gui/chat_render.py`）
只认 Block，不关心它从哪来；导出层（`ttrade/docgen.py`）也认 Block，所以
「聊出来的表格」和「导出的 PDF 表格」永远是同一份数据。

放这里而不是 `gui/` 下的原因：`ttrade` 不能 import `gui`（打包和依赖方向都会乱）。
构造器是纯函数，不依赖 Qt。

块类型：

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
{"type": "image",   "b64": "...", "width": 260, "caption": "", "mime": "image/jpeg"}
{"type": "link",    "text": "...", "url": "..."}
```
"""

from __future__ import annotations


def _looks_numeric(cell) -> bool:
    t = str(cell).replace(",", "").replace(" ", "").rstrip("%")
    if not t or t in ("—", "-", "–"):
        return False
    try:
        float(t)
    except ValueError:
        return False
    return True


def guess_align(columns, rows) -> list[str]:
    """逐列猜对齐：整列都是数字就右对齐。

    量化表格里数字右对齐是基本体面 —— 左对齐时 `0.031` 和 `-0.128` 的小数点
    对不上，扫一眼比不出来。
    """
    out = []
    for j in range(len(columns)):
        vals = [r[j] for r in rows if j < len(r) and str(r[j]).strip()]
        out.append("r" if vals and all(_looks_numeric(v) for v in vals) else "l")
    return out


def block_text(text: str) -> dict:
    return {"type": "text", "text": text}


def block_head(text: str, level: int = 3) -> dict:
    return {"type": "head", "level": level, "text": text}


def block_list(items) -> dict:
    return {"type": "list", "items": [str(x) for x in items]}


def block_table(columns, rows, note: str = "", align=None) -> dict:
    """表格块。`align` 不给就按「整列是数字则右对齐」自动猜。"""
    columns = [str(c) for c in columns]
    rows = [[("" if v is None else str(v)) for v in r] for r in rows]
    return {"type": "table", "columns": columns, "rows": rows, "note": note,
            "align": align or guess_align(columns, rows)}


def block_code(code: str, lang: str = "") -> dict:
    return {"type": "code", "lang": lang, "text": code}


def block_metrics(title: str, items, link=None) -> dict:
    """`items` 里每项是 `(名, 值)` 或 `(名, 值, "good"|"bad"|"flat")`。"""
    return {"type": "metrics", "title": title,
            "items": [list(it) for it in items], "link": link}


def block_warning(title: str, lines, level: str = "bad") -> dict:
    return {"type": "warning", "title": title, "lines": list(lines), "level": level}


def block_image(b64: str, width: int = 260, caption: str = "",
                mime: str = "image/jpeg") -> dict:
    return {"type": "image", "b64": b64, "width": width, "caption": caption, "mime": mime}


def block_link(text: str, url: str) -> dict:
    return {"type": "link", "text": text, "url": url}


def block_quote(text: str) -> dict:
    return {"type": "quote", "text": text}
