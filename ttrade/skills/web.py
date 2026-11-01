"""抓网页正文（`web_fetch`）+ 搜网页（`web_search`）。

## 为什么搜索走「必应 RSS」而不是搜索 API

Serper / Bing / Tavily 都要 key、都要钱。必应的 `&format=rss` 是它自己公开的
订阅输出，返回的是**直链**（不是 `bing.com/ck/a?...` 那种跳转壳），不需要
解析 base64，也不需要为每条结果再发一次请求。缺点是条数少（十来条）、
排序不如正式 API，但对「找一篇能引用的文章」够用。

第二顺位是 DuckDuckGo 的 html 端点（`html.duckduckgo.com`），它的跳转壳里
`uddg=` 参数就是原文 URL，一次解码即可。两家都取不到才报错。

## 出网是危险动作，四道闸

1. **只允许 http/https**。`file://`、`ftp://`、`data:` 一律拒 —— 那些能读本机文件
2. **解析后检查 IP**。`http://127.0.0.1:8080`、`http://169.254.169.254`（云厂商的
   元数据服务）、`http://10.0.0.1` 这些都不许访问。只检查字符串是不够的：
   `http://2130706433/` 和 `http://0x7f000001/` 都是 127.0.0.1
3. **逐跳检查重定向**。`requests` 自动跟随重定向时会绕过上面那道检查 ——
   公网域名 302 到内网地址是标准打法。所以关掉自动跟随，自己一跳一跳地查
4. **限体积**。`stream=True` + 边读边数，超过 5MB 就停。没有这道闸，一个
   几 G 的日志文件能把内存吃光

搜索走同一套闸：搜索引擎本身是公网域名，结果里的 URL 只**列出来**不访问，
模型要读哪条再单独 `web_fetch`（那时才逐跳校验）。

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
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

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
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) ChanQuant/1.7 "
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
    interrupt="cancel",
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


# ------------------------------------------------------------------ 搜索
# 一次最多返回这么多条。再多模型也读不完，而且第 10 条之后基本是垃圾
MAX_RESULTS = 20
DEFAULT_RESULTS = 8
SEARCH_TIMEOUT = (6, 25)


class SearchHit:
    __slots__ = ("title", "url", "snippet")

    def __init__(self, title: str, url: str, snippet: str = "") -> None:
        self.title = title.strip()
        self.url = url.strip()
        self.snippet = _SPACES.sub(" ", snippet or "").strip()


def _get(url: str, timeout=SEARCH_TIMEOUT) -> str:
    """搜索引擎自己的页面：走和 `_fetch` 同一套地址检查，但不限 content-type。"""
    import requests

    why = check_url(url)
    if why:
        raise Blocked(why)
    try:
        r = requests.get(url, headers={"User-Agent": UA,
                                       "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
                         timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"连不上 {urlparse(url).netloc}：{exc}") from exc
    if r.status_code >= 400:
        raise RuntimeError(f"{urlparse(url).netloc} 返回 HTTP {r.status_code}")
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        try:
            return r.content.decode("utf-8")
        except UnicodeDecodeError:
            return r.content.decode(r.encoding or "utf-8", "replace")
    return r.text


_ITEM = re.compile(r"<item>(.*?)</item>", re.S | re.I)
_TAG_IN_ITEM = re.compile(r"<(title|link|description)>(.*?)</\1>", re.S | re.I)


def _strip_tags(s: str) -> str:
    return unescape(_TAG.sub("", s or "")).strip()


def _search_bing(q: str, n: int) -> list[SearchHit]:
    """必应 RSS。直链、无跳转壳，是几家解析成本最低的。"""
    xml = _get(f"https://www.bing.com/search?q={quote_plus(q)}&format=rss&count={n}")
    out: list[SearchHit] = []
    for body in _ITEM.findall(xml):
        got = {k.lower(): v for k, v in _TAG_IN_ITEM.findall(body)}
        link = _strip_tags(got.get("link", ""))
        if not link.startswith(("http://", "https://")):
            continue
        title = _strip_tags(got.get("title", "")) or link
        out.append(SearchHit(title, link, _strip_tags(got.get("description", ""))))
        if len(out) >= n:
            break
    return out


_DDG_A = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_DDG_S = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.S)


def _search_ddg(q: str, n: int) -> list[SearchHit]:
    """DuckDuckGo html 端点。跳转壳里 `uddg=` 就是原文 URL。"""
    html = _get(f"https://html.duckduckgo.com/html/?q={quote_plus(q)}")
    links = _DDG_A.findall(html)
    snips = _DDG_S.findall(html)
    out: list[SearchHit] = []
    for i, (href, title) in enumerate(links):
        href = unescape(href)
        if "uddg=" in href:
            qs = parse_qs(urlparse(href).query)
            href = unquote((qs.get("uddg") or [""])[0]) or href
        if not href.startswith(("http://", "https://")):
            continue
        snippet = _strip_tags(snips[i]) if i < len(snips) else ""
        out.append(SearchHit(_strip_tags(title), href, snippet))
        if len(out) >= n:
            break
    return out


def _tokens(q: str) -> list[str]:
    """查询里的「实词」。中文按空白切之后整段保留（不再分词 —— 分错了更糟），
    英文按空白切。单字丢掉：它们几乎不带来区分度。"""
    return [t for t in re.split(r"[\s+，,、]+", q) if len(t) >= 2]


def _relevance(hits: list[SearchHit], q: str) -> float:
    """结果里命中查询词的比例。0 = 一条都没沾边。

    必应的中文分词对多词查询很不稳（实测「螺纹钢 库存 季节性」会退化成只搜
    「螺纹」，返回一堆机械螺纹的百科页）。拿这个分数做**重试判据**：
    一条都不沾边就换一种写法再搜一次，而不是把垃圾结果原样交给模型。
    """
    toks = _tokens(q)
    if not toks:
        return 1.0
    blob = " ".join(f"{h.title} {h.snippet}" for h in hits)
    hit = sum(1 for t in toks if t in blob)
    return hit / len(toks)


def _variants(q: str) -> list[str]:
    """同一个查询的几种写法。去重、保序。"""
    out = [q]
    toks = _tokens(q)
    if len(toks) >= 2:
        out.append("".join(toks))          # 去掉空格连成一个词
        out.append(" ".join(toks[-2:]))    # 只留后两个词（通常是品类 + 主题）
    seen, uniq = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def search(query: str, n: int = DEFAULT_RESULTS) -> tuple[str, list[SearchHit]]:
    """按顺序试几家搜索引擎，返回 `(用的哪家, 结果)`。

    先必应再 DuckDuckGo：必应对中文站点的覆盖更好，国内也能直连；
    DDG 在必应被限流时兜底。

    **带重试**：必应经常把多词查询截成第一个词，返回一堆不相关的结果。
    所以每家用几种写法各搜一次，按「命中查询词的比例」挑最好的一组 ——
    只要有一组沾边就用它，全都 0 命中才退而求其次用第一条非空结果。
    """
    errors = []
    best: tuple[float, str, list[SearchHit]] | None = None
    for name, fn in (("必应", _search_bing), ("DuckDuckGo", _search_ddg)):
        for v in _variants(query):
            try:
                hits = fn(v, n)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}：{exc}")
                break                      # 这家整体不通，换下一家
            if not hits:
                errors.append(f"{name}「{v}」：0 条")
                continue
            score = _relevance(hits, query)
            if best is None or score > best[0]:
                best = (score, name, hits)
            if score >= 0.6:
                return best[1], best[2]
    if best is not None:
        return best[1], best[2]
    raise RuntimeError("搜索失败 —— " + "；".join(errors[:6]))


def _web_search(ctx, p: dict) -> SkillResult:
    q = str(p.get("query") or "").strip()
    if not q:
        raise RuntimeError("query 不能为空")
    site = str(p.get("site") or "").strip()
    if site:
        # 只保留域名本身，防止模型把一整条 URL 塞进来
        site = urlparse(site if "//" in site else "//" + site).hostname or ""
        if site:
            q = f"{q} site:{site}"
    n = int(p.get("n") or DEFAULT_RESULTS)
    n = max(1, min(n, MAX_RESULTS))

    engine, hits = search(q, n)
    score = _relevance(hits, q)

    rows = [(h.title[:60] or "（无标题）", h.url) for h in hits]
    blocks = [
        B.block_metrics("搜索结果", [
            ("关键词", q[:40], "flat"),
            ("来源", engine, "flat"),
            ("条数", str(len(hits)), "flat"),
            ("命中", f"{score:.0%}", "flat"),
        ]),
        B.block_table(["标题", "链接"], rows) if rows else B.block_text("（没有结果）"),
    ]
    if score < 0.5:
        blocks.append(B.block_warning(
            "结果可能不相关",
            ["搜索引擎把查询截断了 —— 换更短更具体的词，或者用 site= 限定站点",
             "也可以试试英文关键词"],
            "warn"))
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. {h.title}\n   {h.url}"
                     + (f"\n   {h.snippet}" if h.snippet else ""))
    feed = (f"在「{engine}」搜「{q}」，{len(hits)} 条结果"
            f"（命中率 {score:.0%}）：\n\n" + "\n".join(lines)
            + "\n\n要看哪一条的正文，用 web_fetch 打开对应链接。"
            "搜索结果里的说法**没经过核实**，引用前先读原文。"
            + ("\n注意：这批结果和关键词的匹配度很低，大概率是搜索引擎截断了查询，"
               "别硬用，换关键词重搜。" if score < 0.5 else ""))
    return SkillResult(True, summary=f"{engine} 搜到 {len(hits)} 条（命中 {score:.0%}）",
                       blocks=blocks, feed=feed)


register(Tool(
    name="web_search",
    group="web",
    interrupt="cancel",
    label="搜网页",
    description="用搜索引擎找资料，返回标题 / 链接 / 摘要。**会出网**",
    risk=RISK_WRITE,
    inputSchema={
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "str"},
            "n": {"type": "int", "default": DEFAULT_RESULTS,
                  "minimum": 1, "maximum": MAX_RESULTS},
            "site": {"type": "str", "default": ""},
        },
    },
    handler=_web_search,
    prompt=(
        "**搜网页**：`web_search(query=\"螺纹钢 社会库存 季节性\")`。\n"
        "- 只返回标题、链接、摘要，**要正文还得 `web_fetch` 打开那一条**\n"
        "- `site=\"mysteel.com\"` 可以限定在某个站点里搜\n"
        "- 摘要经常是断章取义的，写进研报之前必须打开原文确认\n"
        "- 搜不到就换关键词（同义词、英文、加年份），别反复搜同一句"
    ),
    max_turns=2,
))
