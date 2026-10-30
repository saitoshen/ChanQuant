"""技能包（Agent Skills 协议）：磁盘上的 `SKILL.md` 目录，按需加载。

## 和 `ttrade/skills/` 的区别

`ttrade/skills/` 是**工具注册表** —— 每个 `register(Tool(...))` 是一条可调用
动作，Python 代码，随版本发布。

这里是**技能** —— 一个目录、一份 `SKILL.md`、一段工作流程说明。它不执行任何
东西，只在模型需要时把「这件事该怎么做」讲清楚。区别在于：

| | 工具（`ttrade/skills/`） | 技能（这里） |
|---|---|---|
| 是什么 | 一次可调用动作 | 一套做法 / 一段知识 |
| 载体 | Python 模块 | 磁盘上的 `SKILL.md` |
| 谁写 | 开发者 | 开发者**和用户**都能写 |
| 何时进上下文 | 目录一行一条 | 只进索引，正文按需读 |
| 能不能改 | 要改代码 | 改文本文件就行 |

## 为什么要把长说明搬出去

原来 `ttrade/agent.py` 的 `SYSTEM_PROMPT` 五千多字符，同时承担了四件事：
系统规则、工具说明、因子写法、策略写法。这正是 HARNESS.md 列的头号反模式
（把工具实现逻辑写进 System Rules）—— 改一句因子说明要动系统提示词，加一个
工具要同时改两处，两边一旦不同步，模型就按过时的说明书干活。

搬出来之后：基础提示词只留**规则**和**红线**，具体做法在技能里按需取。

## 目录

```
ttrade/skillpacks/<name>/SKILL.md     内置，随版本发布，只读
<user_dir>/skills/<name>/SKILL.md     用户自己加的，可改可删
```

用户目录优先：同名的用户技能覆盖内置的。

## frontmatter

```markdown
---
name: factor-authoring
description: 把主观想法写成可回测的因子代码
when_to_use: 用户描述一个平台没有的因子、要写代码注册时
group: factor
risk: write
allowed-tools: factor, factor_check, doc
version: 1
---
```

解析器只认 `key: value` 一行一条，**不引 PyYAML** —— 为这七八个字段装一个
C 扩展不值（和 `skills/base.py` 里不引 `jsonschema` 是同一个理由）。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from ttrade.paths import user_dir

BUILTIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skillpacks")
USER_SUBDIR = "skills"
FILE_NAME = "SKILL.md"

# 索引预算。技能的**目录**进上下文，正文不进 —— 这是「渐进披露」的全部意义。
# 一行大约 120 字符，1500 字符能放十几条，够用且不会喧宾夺主。
INDEX_BUDGET = 1500
# 单条描述的硬上限。索引只用来「发现」，写太细是浪费首轮 token。
MAX_DESC_CHARS = 120
# 正文上限。一份技能写两万字说明它该拆成两份。
MAX_BODY_CHARS = 16000

_NAME_OK = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass
class SkillPack:
    """一份技能。`body` 是正文，**不默认注入**，要 `read()` 才取。"""

    name: str
    description: str = ""
    when_to_use: str = ""
    group: str = ""              # 关联的能力包，用来做权限提示
    risk: str = "read"
    allowed_tools: list[str] = field(default_factory=list)
    version: str = ""
    source: str = "builtin"      # builtin / user
    base_dir: str = ""
    body: str = ""

    @property
    def path(self) -> str:
        return os.path.join(self.base_dir, FILE_NAME)

    def line(self) -> str:
        """索引里的一行。`when_to_use` 比 `description` 更决定「该不该用」。"""
        desc = self.description
        if self.when_to_use:
            desc = f"{desc} —— {self.when_to_use}" if desc else self.when_to_use
        desc = " ".join(desc.split())
        if len(desc) > MAX_DESC_CHARS:
            desc = desc[: MAX_DESC_CHARS - 1] + "…"
        return f"- `{self.name}`：{desc}"


# ------------------------------------------------------------------ frontmatter

def _parse(text: str) -> tuple[dict, str]:
    """拆出 frontmatter 和正文。没有 frontmatter 就整篇当正文。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    head = text[3:end].strip("\n")
    body = text[end + 4:].lstrip("\n")

    meta: dict[str, str] = {}
    for ln in head.split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("#") or ":" not in ln:
            continue
        key, _, value = ln.partition(":")
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key:
            meta[key] = value
    return meta, body


def _split_list(value: str) -> list[str]:
    """`a, b, c` / `[a, b]` / `a b` 都认。"""
    value = (value or "").strip().strip("[]")
    if not value:
        return []
    parts = re.split(r"[,，\s]+", value)
    return [p for p in (x.strip().strip('"').strip("'") for x in parts) if p]


def _load_one(base_dir: str, source: str) -> SkillPack | None:
    p = os.path.join(base_dir, FILE_NAME)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return None

    meta, body = _parse(raw)
    name = (meta.get("name") or os.path.basename(base_dir)).strip().lower()
    if not _NAME_OK.match(name):
        return None
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + f"\n\n…（正文超过 {MAX_BODY_CHARS} 字符已截断）"
    return SkillPack(
        name=name,
        description=(meta.get("description") or "").strip(),
        when_to_use=(meta.get("when_to_use") or meta.get("when-to-use") or "").strip(),
        group=(meta.get("group") or "").strip(),
        risk=(meta.get("risk") or "read").strip().lower(),
        allowed_tools=_split_list(meta.get("allowed-tools") or meta.get("allowed_tools") or ""),
        version=str(meta.get("version") or ""),
        source=source,
        base_dir=base_dir,
        body=body,
    )


def _scan(root: str, source: str) -> list[SkillPack]:
    out: list[SkillPack] = []
    if not os.path.isdir(root):
        return out
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for fn in names:
        d = os.path.join(root, fn)
        if not os.path.isdir(d):
            continue
        pack = _load_one(d, source)
        if pack is not None:
            out.append(pack)
    return out


# ------------------------------------------------------------------ 缓存

_cache: list[SkillPack] = []
_sig: tuple = ()


def user_root() -> str:
    """用户技能目录。不存在就建出来 —— 让用户知道「扔这儿就能加技能」。"""
    path = os.path.join(user_dir(), USER_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _signature() -> tuple:
    """目录签名。任何一份 SKILL.md 增删改都会变，变了就重扫。

    这样用户在文件管理器里加一个技能目录，下一条消息就能用上，不用重启。
    """
    parts: list[tuple] = []
    for root in (BUILTIN_DIR, user_root()):
        if not os.path.isdir(root):
            parts.append((root, 0, 0))
            continue
        try:
            names = sorted(os.listdir(root))
        except OSError:
            parts.append((root, 0, 0))
            continue
        for fn in names:
            p = os.path.join(root, fn, FILE_NAME)
            try:
                st = os.stat(p)
            except OSError:
                continue
            parts.append((p, int(st.st_mtime), st.st_size))
    return tuple(parts)


def load_all() -> list[SkillPack]:
    """全部技能。用户目录的同名技能覆盖内置的。"""
    global _cache, _sig
    sig = _signature()
    if sig == _sig and _cache:
        return _cache

    merged: dict[str, SkillPack] = {}
    for pack in _scan(BUILTIN_DIR, "builtin"):
        merged[pack.name] = pack
    for pack in _scan(user_root(), "user"):
        merged[pack.name] = pack          # 用户优先
    _cache = sorted(merged.values(), key=lambda p: p.name)
    _sig = sig
    return _cache


def reload() -> None:
    global _sig
    _sig = ()


def get(name: str) -> SkillPack | None:
    key = str(name or "").strip().lower()
    for pack in load_all():
        if pack.name == key:
            return pack
    return None


def names() -> list[str]:
    return [p.name for p in load_all()]


# ------------------------------------------------------------------ 提示词

def index_text(budget: int = INDEX_BUDGET) -> str:
    """索引：一行一条，只讲「有这个技能、什么时候用」。

    超预算时从后往前砍 —— 内置技能按 `name` 排序，排序靠前的通常是主线
    （因子、策略、回测），靠后的是补充说明。砍掉的会明说砍了几条，
    免得模型以为「就这么几个技能」。
    """
    packs = load_all()
    if not packs:
        return ""

    head = ("## 技能（只在需要时读正文）\n"
            "下面是平台内置和用户自己加的技能。**只有名字和用途** —— "
            "真要用时先 `read_skill` 取正文，不要凭名字猜怎么做。")
    lines: list[str] = []
    used = len(head)
    dropped = 0
    for pack in packs:
        line = pack.line()
        if used + len(line) + 1 > budget and lines:
            dropped = len(packs) - len(lines)
            break
        lines.append(line)
        used += len(line) + 1

    out = head + "\n\n" + "\n".join(lines)
    if dropped:
        out += f"\n\n（另有 {dropped} 个技能因篇幅未列出，可用 `read_skill` 按名字直接取）"
    return out


def read(name: str) -> tuple[str, str]:
    """取一份技能的正文。返回 `(正文, 错误)`。"""
    pack = get(name)
    if pack is None:
        near = [n for n in names() if name and (name[:3] in n or n[:3] in name)]
        hint = f"；是不是想用 {'、'.join(near[:3])}" if near else ""
        return "", f"没有名为「{name}」的技能{hint}。现有：{'、'.join(names())}"
    body = pack.body.strip()
    if not body:
        return "", f"技能「{pack.name}」的正文是空的（{pack.path}）"
    return body, ""


def catalog() -> str:
    """给人看的清单，给设置页 / 详情面板用。"""
    rows = []
    for p in load_all():
        src = "内置" if p.source == "builtin" else "我的"
        rows.append(f"[{src}] {p.name}：{p.description or p.when_to_use}")
    return "\n".join(rows)
