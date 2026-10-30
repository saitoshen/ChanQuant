"""对话附件：把用户丢进来的图片和文档变成模型收得下的东西。

## 为什么文档走文本、图片走多模态

图片没有「文本等价物」——把 PNG 读成字节贴给模型只是噪音。OpenAI 兼容协议里
图片走 `image_url` 的 data URI，模型自己看。

文档反过来：md/csv/docx 里本来就是一段段文字，转成纯文本拼进 user 消息最省事，
而且所有模型都吃得下（包括不支持视觉的那些）。代价是排版和表格丢了 —— 真在意
表格的，截图发过来。

## 为什么图片要先缩小

1600px、JPEG 85 之后一张手机截图约 200KB，base64 后 270KB。不缩的话一张 4K
截图能到 8MB，base64 再膨胀三分之一，请求直接 413，而各家网关的报错通常只说
「请求体过大」，看不出是图片干的。

## 为什么 docx 自己解、pdf 不自己解

docx 就是一个 zip，`word/document.xml` 里全是 `<w:p>` 段落和 `<w:t>` 文本，
三十行就能抠干净，不用为它装 python-docx。

pdf 不行：正文被压缩流包着，字体还可能是 CID 编码，抠出来是一堆乱码 —— 与其
给一段看着像文字、其实对不上的东西，不如直说读不了。装了 pypdf 就走 pypdf。
"""

from __future__ import annotations

import base64
import io
import os
import re
import zipfile
from dataclasses import dataclass
from html import unescape

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
TEXT_SUFFIXES = {
    ".md", ".markdown", ".txt", ".csv", ".tsv", ".json", ".py", ".log",
    ".ini", ".cfg", ".toml", ".yml", ".yaml", ".html", ".htm", ".xml",
}

# 长边上限。再大对模型识别没有增益，只是让请求变慢
MAX_IMAGE_SIDE = 1600
# 单份文档最多喂这么多字。六万字约等于三万字中文，够一份完整需求文档；
# 再多会把上下文挤满，而模型也读不完
MAX_TEXT_CHARS = 60000


@dataclass
class Attachment:
    name: str
    path: str
    kind: str = "text"       # text / image / bad
    mime: str = ""
    b64: str = ""
    text: str = ""
    note: str = ""           # 缩放、截断、失败原因，给界面显示

    @property
    def ok(self) -> bool:
        return self.kind in ("text", "image")

    def size_text(self) -> str:
        if self.kind == "image":
            n = len(self.b64) * 3 // 4
            return f"{n / 1024:.0f} KB"
        return f"{len(self.text)} 字符"


def file_filter() -> str:
    """给 `QFileDialog` 用的过滤器。"""
    img = " ".join(f"*{s}" for s in sorted(IMAGE_SUFFIXES))
    doc = " ".join(f"*{s}" for s in sorted(TEXT_SUFFIXES | {".docx", ".pdf"}))
    return f"图片 ({img});;文档 ({doc});;所有文件 (*.*)"


# ------------------------------------------------------------------ 读取


def load(path: str) -> Attachment:
    """读一个文件。**不抛异常** —— 失败也返回一个 `ok=False` 的附件，
    界面照常显示它，让人知道是哪个文件没进来、为什么。"""
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    if not os.path.exists(path):
        return Attachment(name, path, "bad", note="文件不见了")
    if ext in IMAGE_SUFFIXES:
        return _load_image(path)
    if ext in TEXT_SUFFIXES:
        return _load_text(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext == ".pdf":
        return _load_pdf(path)
    return Attachment(
        name, path, "bad",
        note=f"不支持 {ext or '这种'} 文件。文字类请存成 md/txt/csv，"
             "表格或排版重要的直接截图",
    )


def _load_image(path: str) -> Attachment:
    name = os.path.basename(path)
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 - 没装 Pillow 也别把整个功能打死
        return Attachment(name, path, "bad", note="没装 Pillow，读不了图片")

    try:
        im = Image.open(path)
        im.load()
    except Exception as exc:  # noqa: BLE001
        return Attachment(name, path, "bad", note=f"打不开：{exc}")

    try:
        im.seek(0)          # gif 取第一帧，不然后面 save 会拿最后一帧
    except Exception:  # noqa: BLE001
        pass
    return _pack_image(im, name, path)


def from_qimage(qimg, name: str = "截图.png") -> Attachment:
    """把剪贴板 / 拖放拿到的一张 QImage 变成附件。

    走内存不落盘：`QPixmap` → PNG 字节 → Pillow → 和文件图走同一条缩放/转码路径，
    这样尺寸上限、透明底铺白、JPEG 85 三个规矩只需要维护一份。

    也不依赖 Qt —— 传进来只要是个有 `save` 的对象就行，方便单测直接喂 Pillow 图。
    """
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return Attachment(name, "", "bad", note="没装 Pillow，读不了图片")

    try:
        if isinstance(qimg, Image.Image):
            im = qimg.copy()
        else:
            from PySide6.QtCore import QBuffer, QByteArray

            buf = QBuffer()
            buf.open(QBuffer.ReadWrite)
            if not qimg.save(buf, "PNG"):
                return Attachment(name, "", "bad", note="这张图存不下来")
            im = Image.open(io.BytesIO(bytes(buf.data())))
            im.load()
            buf.close()
    except Exception as exc:  # noqa: BLE001
        return Attachment(name, "", "bad", note=f"读不了这张图：{exc}")

    return _pack_image(im, name, "")


def _pack_image(im, name: str, path: str) -> Attachment:
    """缩到长边 1600、透明底铺白、存成 JPEG。文件图和剪贴板图共用这一段。"""
    from PIL import Image

    note = ""
    w, h = im.size
    if max(w, h) > MAX_IMAGE_SIDE:
        im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        note = f"{w}×{h} 缩到 {im.width}×{im.height}"

    # 透明底转 JPEG 会变黑，先铺白。看着是小细节，但深色 UI 截图转完一片黑，
    # 模型只会说「图看不清」
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")

    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85, optimize=True)
    data = buf.getvalue()
    return Attachment(
        name, path, "image", mime="image/jpeg",
        b64=base64.b64encode(data).decode("ascii"), note=note,
    )


def _read_bytes_text(path: str) -> tuple[str, str]:
    """按 utf-8-sig → utf-8 → gbk 依次试。返回 (文本, 备注)。

    中文 Windows 上导出的 csv 十有八九是 GBK，直接按 utf-8 读会抛
    UnicodeDecodeError；用 `errors="replace"` 硬读则满屏问号，模型读出来的
    数字是错的还不自知 —— 所以宁可多试两种编码。
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(enc), ""
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace"), "编码没认出来，按 UTF-8 硬解，可能有乱码"


def _clip(text: str) -> tuple[str, str]:
    if len(text) <= MAX_TEXT_CHARS:
        return text, ""
    return text[:MAX_TEXT_CHARS], f"太长，只取前 {MAX_TEXT_CHARS} 字符"


def _load_text(path: str) -> Attachment:
    name = os.path.basename(path)
    try:
        text, note = _read_bytes_text(path)
    except OSError as exc:
        return Attachment(name, path, "bad", note=f"读不了：{exc}")
    text, cut = _clip(text)
    return Attachment(name, path, "text", text=text,
                      note="；".join(x for x in (note, cut) if x))


def _load_docx(path: str) -> Attachment:
    name = os.path.basename(path)
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        return Attachment(name, path, "bad", note=f"打不开 docx：{exc}")

    # 段落和单元格要留分隔，否则整篇糊成一行，"第 3 条：止损 2%" 会粘到上一句
    xml = xml.replace("</w:p>", "\n").replace("</w:tc>", "\t").replace("</w:tr>", "\n")
    text = unescape(re.sub(r"<[^>]+>", "", xml))
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return Attachment(name, path, "bad", note="这份 docx 里没有文字（可能全是图片）")
    text, cut = _clip(text)
    return Attachment(name, path, "text", text=text, note=cut)


def _load_pdf(path: str) -> Attachment:
    name = os.path.basename(path)
    try:
        from pypdf import PdfReader
    except Exception:  # noqa: BLE001
        return Attachment(
            name, path, "bad",
            note="读 pdf 要装 pypdf（pip install pypdf）；或者把关键几页截图发我",
        )
    try:
        reader = PdfReader(path)
        pages = [(p.extract_text() or "") for p in reader.pages]
    except Exception as exc:  # noqa: BLE001
        return Attachment(name, path, "bad", note=f"pdf 解析失败：{exc}")
    text = "\n\n".join(p for p in pages if p.strip()).strip()
    if not text:
        return Attachment(name, path, "bad",
                          note="这份 pdf 抠不出文字（多半是扫描件），截图发我")
    text, cut = _clip(text)
    return Attachment(name, path, "text", text=text, note=cut)


# ------------------------------------------------------------------ 拼消息


def text_block(atts: list[Attachment]) -> str:
    """附件里**文字**的那部分，拼成一段给模型看的文本。

    图片不在这里展开，只留一行说明 —— 正文由 `image_parts()` 单独发，
    这段文字是给模型对齐「第几张图是什么」的。
    """
    blocks = []
    for i, a in enumerate(atts, 1):
        if not a.ok:
            continue
        if a.kind == "image":
            extra = f"，{a.note}" if a.note else ""
            blocks.append(f"【附件 {i}：{a.name}（图片，{a.size_text()}{extra}）"
                          "——图像本身在下面单独传入】")
        else:
            extra = f"，{a.note}" if a.note else ""
            blocks.append(f"【附件 {i}：{a.name}（{a.size_text()}{extra}）】\n{a.text}")
    return "\n\n".join(blocks)


def image_parts(atts: list[Attachment]) -> list[dict]:
    return [
        {"type": "image_url",
         "image_url": {"url": f"data:{a.mime or 'image/jpeg'};base64,{a.b64}"}}
        for a in atts if a.ok and a.kind == "image"
    ]


def content(text: str, atts: list[Attachment]) -> str | list:
    """拼成 OpenAI 兼容的 `content`。

    没有图就是纯字符串 —— 绝大多数模型对「只有一个 text 部件的 parts 列表」
    也认，但没必要冒这个险；老一点的网关只吃字符串。
    """
    good = [a for a in atts if a.ok]
    if not good:
        return text
    full = (text + "\n\n" + text_block(good)).strip()
    imgs = image_parts(good)
    if not imgs:
        return full
    return [{"type": "text", "text": full}] + imgs


def describe(atts: list[Attachment]) -> str:
    """一行摘要，给对话区显示。"""
    names = "、".join(a.name for a in atts)
    n_img = sum(1 for a in atts if a.kind == "image")
    n_doc = sum(1 for a in atts if a.kind == "text")
    bits = []
    if n_img:
        bits.append(f"{n_img} 张图")
    if n_doc:
        bits.append(f"{n_doc} 份文档")
    tail = f"（{'，'.join(bits)}）" if bits else ""
    return f"附件：{names}{tail}"
