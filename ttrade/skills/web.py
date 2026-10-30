"""抓网页正文（`web_fetch`）。

## 只做 fetch，不做 search

搜索要接第三方搜索 API（Serper / Bing / Tavily），每个都要 key、都要钱，而且
返回的是一堆链接 —— 模型还是得自己挑。真需要「找资料」时，用户直接给链接更快，
也更可信。所以这里只做「给我一个 URL，我把正文取回来」。

## 出网是危险动作，四道闸

1. **只允许 http/https**。`file://`、`ftp://`、`data:` 一律拒 —— 那些能读本机文件
2. **解析后检查 IP**。`http://127.0.0.1:8080`、`http://169.254.169.254`（云厂商的
   元数据服务）、`http://10.0.0.1` 这些都不许访问。只检查字符串是不够的：
   `http://2130706433/` 和 `http://0x7f000001/` 都是 127.0.0.1
3. **逐跳检查重定向**。`requests` 自动跟随重定向时会绕过上面那道检查 ——
   公网域名 302 到内网地址是标准打法。所以关掉自动跟随，自己一跳一跳地查
4. **限体积**。`stream=True` + 边读边数，超过 5MB 就停。没有这道闸，一个
   几 G 的日志文件能把内存吃光

## 抓回来的是纯文本

`<script>` / `<style>` / `<nav>` 整块丢掉，剩下的标签剥掉、实体反转义、空行压掉。
不转 markdown：网页的结构化信息（表格）转 markdown 要写一套容错很差的解析器，
而模型读纯文本 + 段间空行已经够用了。
"""

from __future__ import annotations

import ipaddress
import re
import socket
from html import unescape
from urllib.parse import urlparse

from ttrade import blocks as B
from ttrade.skills.base import (
    RISK_WRITE,
    SkillResult,
    Tool,
    register,
)

# 一次最多取这么多字节。新闻/研报页面 200KB 已经很长了，5MB 是给 PDF 之外
# 的极端情况留的余量
MAX_BYTES = 5 * 1024 * 1024
# 交给模型/贴进对话的字符数。再多模型也读不完，还会把上下文挤满
MAX_CHARS = 12000
TIMEOUT = (6, 20)
MAX_REDIRECTS = 3
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) TrendDesk/1.7 "
      "(+local research tool)")

OK_TYPES = ("text/html", "text/plain", "application/xhtml", "application/json",
            "text/xml", "application/xml")


class Blocked(Exception):
    """这个地址不许访问。消息直接给人看。"""


# ------------------------------------------------------------------ 地址检查

def _bad_ip(ip: str) -> str:
    """这个 IP 为什么不能访问。空串表示可以。"""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return f"{ip} 不是合法地址"
    if addr.is_loopback:
        return f"{ip} 是本机地址"
    if addr.is_private:
        return f"{ip} 是内网地址"
    if addr.is_link_local:
        return f"{ip} 是链路本地地址（云服务器的元数据接口就在这一段）"
    if addr.is_reserved or addr.is_multicast or addr.is_unspecified:
        return f"{ip} 是保留地址"
    return ""


def check_url(url: str) -> str:
    """能不能访问这个 URL。返回空串表示可以，否则返回原因。

    域名要**解析之后再查**：只查字符串的话 `http://2130706433/`、
    `http://0x7f000001/`、`http://127.1/` 都能绕过 —— 它们都是 127.0.0.1。
    """
    try:
        u = urlparse(url)
    except ValueError as exc:
        return f"地址不合法：{exc}"
    if u.scheme not in ("http", "https"):
        return f"只支持 http / https，收到 {u.scheme or '（空）'}"
    host = (u.hostname or "").strip().lower().rstrip(".")
    if not host:
        return "地址里没有域名"
    if host in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        return f"{host} 是本机/局域网名字"
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return f"域名解析不了：{host}（{exc.strerror or exc}）"
    for info in infos:
        why = _bad_ip(info[4][0])
        if why:
            return why
    return ""


# ------------------------------------------------------------------ 取回

def _fetch(url: str) -> tuple[str, str, str]:
    """取一个 URL，返回 `(最终地址, content-type, 正文)`。跳转自己跟。"""
    import requests

    cur = url
    for _ in range(MAX_REDIRECTS + 1):
        why = check_url(cur)
        if why:
            raise Blocked(why)
        try:
            r = requests.get(
                cur, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"},
                timeout=TIMEOUT, stream=True, allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"连不上 {cur}：{exc}") from exc

        with r:
            if r.status_code in (301, 302, 303, 307, 308):
                nxt = r.headers.get("Location") or ""
                if not nxt:
                    raise RuntimeError(f"{cur} 返回 {r.status_code} 但没给跳转地址")
                # 相对地址要按当前 URL 补全，否则下一跳一定解析失败
                cur = requests.compat.urljoin(cur, nxt)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}：{cur}")
            ctype = (r.headers.get("Content-Type") or "").lower()
            if not any(t in ctype for t in OK_TYPES):
                raise RuntimeError(
                    f"这个地址返回的是 {ctype or '未知类型'}，不是网页正文。"
                    "要读 PDF 请先下载再当附件传进来"
                )
            chunks, total = [], 0
            for chunk in r.iter_content(65536):
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_BYTES:
                    raise RuntimeError(
                        f"页面超过 {MAX_BYTES // 1024 // 1024}MB，先别抓了")
            raw = b"".join(chunks)
        enc = r.encoding
        if not enc or enc.lower() == "iso-8859-1":
            # requests 对没有 charset 的页面默认给 iso-8859-1，中文站会全成乱码。
            # 先按 UTF-8 试，不合法再退回它的默认值
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode(enc or "utf-8", "replace")
        else:
            text = raw.decode(enc, "replace")
        return cur, ctype, text
    raise RuntimeError(f"跳转超过 {MAX_REDIRECTS} 次，停在这里：{cur}")


# ------------------------------------------------------------------ HTML → 文本

_DROP = re.compile(
    r"<(script|style|noscript|svg|iframe|head)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_BLOCK_TAG = re.compile(
    r"</?(p|div|br|li|tr|h[1-6]|section|article|header|footer|blockquote|pre)\b[^>]*>",
    re.I)
_TAG = re.compile(r"<[^>]+>")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_SPACES = re.compile(r"[ \t\u00a0]+")
_BLANKS = re.compile(r"\n{3,}")


def html_to_text(html: str) -> tuple[str, str]:
    """网页 → `(标题, 正文)`。够用的程度，不追求还原排版。"""
    title = ""
    m = _TITLE.search(html)
    if m:
        title = _SPACES.sub(" ", unescape(_TAG.sub("", m.group(1)))).strip()

    body = _COMMENT.sub(" ", html)
    body = _DROP.sub(" ", body)
    # 块级标签换成换行：不换的话整页会糊成一整段，模型分不清哪儿是标题
    body = _BLOCK_TAG.sub("\n", body)
    body = _TAG.sub(" ", body)
    body = unescape(body)
    body = _SPACES.sub(" ", body)
    lines = [ln.strip() for ln in body.split("\n")]
    body = "\n".join(ln for ln in lines if ln)
    body = _BLANKS.sub("\n\n", body)
    return title, body.strip()


# ------------------------------------------------------------------ 工具

def _web_fetch(ctx, p: dict) -> SkillResult:
    url = str(p["url"]).strip()
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url

    final, ctype, raw = _fetch(url)
    if "html" in ctype or "xml" in ctype:
        title, text = html_to_text(raw)
    else:
        title, text = "", raw.strip()

    if not text:
        raise RuntimeError("这个页面取回来是空的（可能要登录，或者正文是 JS 渲染的）")

    clipped = len(text) > MAX_CHARS
    shown = text[:MAX_CHARS] if clipped else text

    blocks = [
        B.block_metrics(title or "网页正文", [
            ("字数", f"{len(text):,}", "flat"),
            ("来源", urlparse(final).netloc, "flat"),
        ]),
        B.block_link(title or final, final),
        B.block_text(shown),
    ]
    if clipped:
        blocks.append(B.block_warning(
            "只取了前 12000 字符", ["要后面的内容，让用户把关键段落复制过来"], "warn"))

    feed = f"网页「{title or final}」（{final}）的正文：\n\n{text}"
    return SkillResult(True, summary=f"已抓取 {urlparse(final).netloc}（{len(text)} 字符）",
                       blocks=blocks, feed=feed)


register(Tool(
    name="web_fetch",
    group="web",
    label="抓网页",
    description="取一个网页的正文（只读，不执行 JS）。**会出网**",
    risk=RISK_WRITE,
    inputSchema={
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"type": "str"},
        },
    },
    handler=_web_fetch,
    prompt=(
        "**抓网页**：`web_fetch(url=\"https://...\")`。用户给了链接、让你看那篇文章/"
        "那个公告时用它。\n"
        "- 只能读公网 http/https 页面，本机和内网地址会被拒\n"
        "- 不执行 JS，所以要登录或者前端渲染的页面取回来是空的\n"
        "- PDF 链接抓不了，让用户下载后当附件传进来\n"
        "- 一次最多 12000 字符，长文只能看到开头\n"
        "抓回来的内容要**给出处**（域名），别把网页上的说法当成你自己验证过的结论。"
    ),
    max_turns=2,
))
