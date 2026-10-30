"""会话里的「附件资源」。

## 为什么不再把文件整段塞进消息

以前 Excel 是这样处理的：整份读成纯文本，最多六万字，拼进 user 消息。后果有两个：

- 一份 200 行 × 15 列的因子表，六万字里有一半是重复的日期和空格，列还对不齐
- **读完没有第二次机会**。模型想问「第 2 个 sheet 呢」，只能让用户重发

现在改成「会话资源 + 按需取」：文件解析一次，存进 `SessionFiles`，分配一个
`file_id`（`f1`、`f2`…），消息里只放一段**摘要**（sheet 名、行列数、列名、前 5 行，
约 300 字符）。模型觉得不够就调 `read_excel(file_id="f1", rows=100)` 再取。

## 为什么只资源化表格

xlsx / csv 是唯一「整段塞进去会明显吃亏」的类型 —— 它们又长又结构化。md / txt /
pdf / docx 本来就是一篇文章，整段给模型是对的（而且 `attachments` 那条链路已经
在用了）。所以这里只接管表格，其余保持原样，不做没收益的改动。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pandas as pd

# 摘要里最多放几列、几行。再多就不叫摘要了，跟整段塞进去没区别
PREVIEW_COLS = 8
PREVIEW_ROWS = 5
# 列名最多列这么多，其余用「…」带过
MAX_COLS_LISTED = 30

TABLE_SUFFIXES = {".xlsx", ".xlsm", ".xls", ".csv", ".tsv"}
TEXT_SUFFIXES = {
    ".md", ".markdown", ".txt", ".json", ".py", ".log", ".ini", ".cfg",
    ".toml", ".yml", ".yaml", ".html", ".htm", ".xml",
    ".pdf", ".docx",
}


@dataclass
class SessionFile:
    file_id: str
    name: str
    kind: str = "table"                 # table / text
    path: str = ""
    sheets: dict[str, pd.DataFrame] = field(default_factory=dict)
    text: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.sheets) or bool(self.text)

    def main_sheet(self) -> str:
        """主表名。优先叫「因子」「数据」「Sheet1」，否则取第一个。"""
        if not self.sheets:
            return ""
        for want in ("因子", "数据", "Sheet1", "sheet1"):
            if want in self.sheets:
                return want
        return next(iter(self.sheets))

    def frame(self, sheet: str | int | None = None) -> pd.DataFrame:
        if not self.sheets:
            return pd.DataFrame()
        if sheet in (None, ""):
            return self.sheets[self.main_sheet()]
        if isinstance(sheet, int):
            keys = list(self.sheets)
            return self.sheets[keys[sheet]] if 0 <= sheet < len(keys) else pd.DataFrame()
        return self.sheets.get(str(sheet), pd.DataFrame())

    def summary(self) -> str:
        """给模型看的一段话。**这段直接进上下文**，所以字数要克制。"""
        head = f"【附件 {self.file_id}：{self.name}"
        if not self.ok:
            return f"{head}】读不了：{self.note or '格式不支持'}"

        if self.kind == "text":
            preview = self.text[:600].replace("\n", " ")
            more = f"（共 {len(self.text)} 字符，下面只是开头）" if len(self.text) > 600 else ""
            return f"{head}，文本{more}】\n{preview}"

        names = list(self.sheets)
        main = self.main_sheet()
        df = self.sheets[main]
        cols = [str(c) for c in df.columns]
        listed = "、".join(cols[:MAX_COLS_LISTED])
        if len(cols) > MAX_COLS_LISTED:
            listed += f" …（共 {len(cols)} 列）"

        lines = [
            f"{head}，表格｜{len(names)} 个 sheet：{'、'.join(names[:6])}】",
            f"主表「{main}」{len(df)} 行 × {len(cols)} 列",
            f"列：{listed}",
        ]
        show = df.head(PREVIEW_ROWS).iloc[:, :PREVIEW_COLS]
        if not show.empty:
            head_row = "| " + " | ".join(str(c) for c in show.columns) + " |"
            sep = "|" + "---|" * len(show.columns)
            body = [
                "| " + " | ".join(_cell(v) for v in row) + " |"
                for row in show.itertuples(index=False)
            ]
            lines.append("前几行：\n" + "\n".join([head_row, sep] + body))
        if len(df) > PREVIEW_ROWS or len(cols) > PREVIEW_COLS:
            lines.append(
                f'要更多就调 read_excel(file_id="{self.file_id}", '
                f'sheet="{main}", rows=50)'
            )
        return "\n".join(lines)


class SessionFiles:
    """一次对话里所有附件资源。跟对话一起清空。"""

    def __init__(self) -> None:
        self._items: dict[str, SessionFile] = {}
        self._seq = 0

    def clear(self) -> None:
        self._items.clear()
        self._seq = 0

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> list[SessionFile]:
        return list(self._items.values())

    def get(self, file_id: str) -> SessionFile | None:
        return self._items.get(str(file_id or "").strip())

    def remove(self, file_id: str) -> bool:
        """删掉一个资源。序号**不回退** —— 回退会让下一份新文件复用刚删掉的
        `f1`，而模型手上可能还留着上一轮那个 `f1` 的引用。"""
        return self._items.pop(str(file_id or "").strip(), None) is not None

    def _next_id(self) -> str:
        self._seq += 1
        return f"f{self._seq}"

    # ---------------------------------------------------------- 加入

    def add_path(self, path: str) -> SessionFile | None:
        """加一个文件。**不抛异常** —— 读不了就返回 None，调用方退回旧链路。"""
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)

        if ext in TABLE_SUFFIXES:
            item = SessionFile(self._next_id(), name, "table", path)
            try:
                item.sheets = read_tables(path)
            except Exception as exc:  # noqa: BLE001
                item.note = f"{type(exc).__name__}: {exc}"
                return None
            if not item.sheets:
                return None
            self._items[item.file_id] = item
            return item

        if ext in TEXT_SUFFIXES:
            item = SessionFile(self._next_id(), name, "text", path)
            try:
                item.text, item.note = read_text_file(path)
            except Exception as exc:  # noqa: BLE001
                item.note = f"{type(exc).__name__}: {exc}"
                return None
            if not item.text.strip():
                return None
            self._items[item.file_id] = item
            return item

        return None

    def add_frame(self, df: pd.DataFrame, name: str) -> SessionFile:
        """直接塞一个 DataFrame（测试、或者工具之间传递）。"""
        item = SessionFile(self._next_id(), name, "table")
        item.sheets = {"Sheet1": df}
        self._items[item.file_id] = item
        return item

    def summary(self) -> str:
        """所有附件摘要拼起来。没有附件就是空串。"""
        return "\n\n".join(f.summary() for f in self._items.values())


# ------------------------------------------------------------------ 读文本
# 一份文本最多留这么多字符。十万字已经远超任何一次对话能用的量，再留只是占内存
MAX_TEXT = 200_000


def read_text_file(path: str) -> tuple[str, str]:
    """读一个文本类文件，返回 `(正文, 备注)`。

    pdf / docx 单独走：它们不是「按编码读字节」能解决的，硬读会得到一堆
    `PK\\x03\\x04` 或者压缩流。其余按 utf-8-sig → utf-8 → gbk 依次试
    （中文 Windows 导出的 txt 十有八九是 gbk）。

    读不了就抛异常，由 `add_path` 兜住 —— 这里不返回半截垃圾。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)

    last: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, encoding=enc) as fh:
                text = fh.read()
            break
        except UnicodeDecodeError as exc:
            last = exc
    else:
        raise last or RuntimeError("编码没认出来")
    if len(text) > MAX_TEXT:
        return text[:MAX_TEXT], f"只取了前 {MAX_TEXT} 字符"
    return text, ""


def _read_pdf(path: str) -> tuple[str, str]:
    """按页抽文字。扫描版 PDF 抽出来是空的 —— 那种要靠 OCR，这里不装。

    返回的正文带「—— 第 N 页 ——」分隔，模型问「第 3 页说了什么」时能对上。
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        try:
            body = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001 - 单页坏了不该让整份读不了
            body = ""
        if body:
            pages.append(f"—— 第 {i} 页 ——\n{body}")
    text = "\n\n".join(pages)
    note = ""
    if not text:
        note = "没抽出文字，可能是扫描件（图片型 PDF）"
    elif len(reader.pages) and len(text) < 20 * len(reader.pages):
        note = "文字很少，可能是扫描件"
    if len(text) > MAX_TEXT:
        return text[:MAX_TEXT], "只取了前 200000 字符"
    return text, note


_W_PARA = re.compile(r"</w:p>")
_W_TEXT = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _read_docx(path: str) -> tuple[str, str]:
    """从 docx 里抠正文。**不引 python-docx** —— 它要 lxml，打包体积翻倍，
    而我们要的只是 `word/document.xml` 里的 `<w:t>` 文本。表格里的文字也在
    同一个 xml 里，所以表格内容不会丢（只是没有行列结构）。
    """
    import zipfile

    with zipfile.ZipFile(path) as zf:
        try:
            xml = zf.read("word/document.xml").decode("utf-8", "replace")
        except KeyError as exc:
            raise RuntimeError("这个 docx 里没有 word/document.xml") from exc

    paras = []
    for chunk in _W_PARA.split(xml):
        parts = _W_TEXT.findall(chunk)
        if not parts:
            continue
        line = _TAG.sub("", "".join(parts)).strip()
        if line:
            paras.append(line)
    text = "\n".join(paras)
    note = "" if text else "没抽出文字（可能是空文档或老 .doc 格式）"
    if len(text) > MAX_TEXT:
        return text[:MAX_TEXT], "只取了前 200000 字符"
    return text, note


# ------------------------------------------------------------------ 读表

def read_tables(path: str) -> dict[str, pd.DataFrame]:
    """读一个表格文件，返回 {sheet 名: DataFrame}。

    csv 也走这里，包成单 sheet —— 调用方不用分情况。csv 的编码按
    utf-8-sig → utf-8 → gbk 依次试：中文 Windows 导出的十有八九是 gbk。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        last = None
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                return {"Sheet1": df}
            except UnicodeDecodeError as exc:
                last = exc
            except Exception:
                raise
        raise last or RuntimeError("编码没认出来")

    # xlsx / xlsm / xls。`sheet_name=None` 读全部 sheet
    return pd.read_excel(path, sheet_name=None)


def slice_frame(df: pd.DataFrame, rows: int = 20, cols: list[str] | None = None,
                start: int = 0) -> pd.DataFrame:
    """按行段 / 列名取一块。`cols` 认不出来就原样返回全部列（别报错）。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df
    if cols:
        keep = [c for c in cols if c in df.columns]
        if keep:
            out = df[keep]
    start = max(0, int(start or 0))
    rows = max(1, int(rows or 20))
    return out.iloc[start:start + rows]


def _cell(v) -> str:
    """摘要里一个单元格的显示文本。

    三件小事但都影响可读性：NaN 显示成空（不是 `nan`）、零点整的时间戳去掉
    `00:00:00`、浮点别带一串尾数。摘要每次提问都要发，能短一个字就短一个字。
    """
    if v is None:
        return ""
    try:
        if v != v:                       # NaN
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d") if (v.hour or v.minute or v.second) == 0 \
            else v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)
