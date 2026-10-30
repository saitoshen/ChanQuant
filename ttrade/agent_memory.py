"""Agent 的长期记忆：把聊过的内容压缩成几条要点，下次开窗口还在。

## 为什么不能只靠 `trim_history`

`trim_history()` 到 40 条就把最早的一刀切掉 —— 用户昨天确认过的「螺纹只做多、
1 手、止损 2 倍 ATR」今天就没了，模型只能重新问一遍。剪掉之前先把这段压成要点，
存到用户目录，下次请求时塞进 system prompt，才叫「记得住」。

## 压什么、不压什么

压：结论、数字（回测收益、参数、阈值）、已做的决定（注册了哪个因子/策略）、
用户偏好（标的、周期、手数、风险口径）。
不压：客套话、模型的推理过程、被否决的中间方案、动作块里的代码。

## 为什么是「追加」而不是「覆盖」

一条记忆代表一段时间里确认过的事实。后面聊出新结论时，旧结论往往是**当时**成立
而不是错的（比如「先跑 2016 起」）。覆盖会把这层时间顺序抹掉，模型再被问到就会
拿最新一条当唯一真相。所以只追加、去重、封顶。

## 分四类，不混着记

Claude Code 的记忆系统有一条很硬的原则：**只记「不能从当前状态推导出来」的
东西**。代码模式、架构、git 历史都能 grep 出来，所以不该进记忆。

量化场景同理：因子数值、回测结果、K 线形态**都能现算**，记下来只会占地方，
而且会过期。真正该记的是四类：

| 类型 | 记什么 | 例子 |
|---|---|---|
| `user` | 交易风格、风险偏好、常用品种、仓位习惯 | 偏好日线、单笔风险不超 1%、不做夜盘 |
| `feedback` | 用户纠正过你的、或者明确认可过的做法 | 别拿 z=1.0 直接上实盘，先看 IC 衰减 |
| `project` | 某个策略/因子实验的结论 | AG 30m 动量 2024 后失效 |
| `reference` | 数据口径 | 手续费双边万三、滑点一跳、换月用持仓量 |

## 为什么是「追加 + 定期删」而不是「覆盖」

一条记忆代表一段时间里确认过的事实。后面聊出新结论时，旧结论往往是**当时**
成立而不是错的（比如「先跑 2016 起」）。覆盖会把这层时间顺序抹掉，模型再被
问到就会拿最新一条当唯一真相。

但只追加不删，三个月后就全是过期条目。所以压缩时**顺带问一句「哪条已经
过时了」** —— 这就是 Claude Code 的 autoDream 里 Prune 那一步。删是保守的：
只删被明确推翻的，不为了「整理」而删。

## 上限

120 条 / 注入 4000 字。超了从最旧的丢 —— 记忆是「最近还关心什么」的近似，
不是档案库。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

from ttrade.paths import user_dir

FILE_NAME = "agent_memory.json"
MAX_ITEMS = 120
MAX_CHARS = 4000          # 注入 prompt 的总长度上限
_SPACE = re.compile(r"\s+")

# 四类。顺序就是 `brief()` 里的呈现顺序 —— 用户偏好排最前，因为最常被用到
MEMORY_TYPES = ("user", "feedback", "project", "reference")
TYPE_LABEL = {
    "user": "用户偏好",
    "feedback": "纠正与反馈",
    "project": "项目结论",
    "reference": "口径与参考",
    "": "其他",
}
_TYPE_ORDER = {t: i for i, t in enumerate(MEMORY_TYPES + ("",))}


@dataclass
class MemoryItem:
    text: str = ""
    kind: str = "summary"     # summary（压缩出来的一段）/ fact（单条事实）
    type: str = ""            # user / feedback / project / reference；空 = 老条目
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))
    turns: int = 0            # 这段记忆压缩自多少条消息，排查用

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryItem":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def path() -> str:
    return os.path.join(user_dir(), FILE_NAME)


def load() -> list[MemoryItem]:
    """读盘。坏文件当成空 —— 记忆丢了不影响交易，起不来才影响。"""
    p = path()
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[MemoryItem] = []
    for d in items:
        if not isinstance(d, dict):
            continue
        m = MemoryItem.from_dict(d)
        if m.text.strip():
            out.append(m)
    return out


def save(items: list[MemoryItem]) -> None:
    """原子写：先写 .tmp 再 replace。

    覆盖之前把旧的留一份 `.bak` —— 记忆是**没法重新生成**的东西（它来自
    已经删掉的对话），一次误写就永久丢了。留一份上版内容，代价是几百字节。
    """
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p):
        try:
            with open(p, "rb") as src:
                old = src.read()
            if old.strip():
                with open(p + ".bak", "wb") as dst:
                    dst.write(old)
        except OSError:
            pass
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "items": [i.to_dict() for i in items]},
                  fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def _norm(text: str) -> str:
    """按行规整：压掉多余空白、去掉模型自己加的列表符号。

    模型压缩出来天然带「- 」前缀，`brief()` 还会再加一次，不去掉就会变成
    「- - 螺纹只做多」，看着像坏了。
    """
    lines: list[str] = []
    for ln in (text or "").splitlines():
        ln = _SPACE.sub(" ", ln).strip().lstrip("-*·").strip()
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def append(new: list[MemoryItem]) -> int:
    """追加若干条，去重、封顶。返回实际新增的条数。"""
    items = load()
    seen = {_norm(i.text) for i in items}
    added = 0
    for m in new:
        t = _norm(m.text)
        if not t or t in seen:
            continue
        m.text = t
        items.append(m)
        seen.add(t)
        added += 1
    if added:
        save(items[-MAX_ITEMS:])
    return added


# 模型输出的两段：`新增：` 下面是 `[type] 内容`，`删除：` 下面是编号
_ADD = re.compile(
    r"^\s*[-*·]?\s*[\[【]\s*(user|feedback|project|reference|其他)\s*[\]】]\s*(.+)$",
    re.I)
_NUM = re.compile(r"^\s*[-*·]?\s*(\d{1,3})\s*$")
_ADD_HEAD = {"新增", "新增记忆", "新增条目", "要加的"}
_DROP_HEAD = {"删除", "删除记忆", "删除条目", "要删的"}
_EMPTY = {"无", "无。", "無", "无。"}


def parse_changes(text: str) -> tuple[list[MemoryItem], list[int]]:
    """把模型的两段输出拆成「要加的条目」和「要删的编号」。

    模型不一定严格照格式写，所以宽松解析。实在认不出来就**退回老办法**：
    把每一行当成一条无类型记忆 —— 宁可分类不准，也不要因为格式没对上就
    整段丢掉。
    """
    adds: list[MemoryItem] = []
    drops: list[int] = []
    bare: list[str] = []
    section = ""

    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        head = ln.rstrip("：:").strip()
        if head in _ADD_HEAD:
            section = "add"
            continue
        if head in _DROP_HEAD:
            section = "drop"
            continue
        if section == "drop":
            m = _NUM.match(ln)
            if m:
                drops.append(int(m.group(1)))
            continue
        m = _ADD.match(ln)
        if m:
            typ = m.group(1).lower()
            typ = "" if typ == "其他" else typ
            body = m.group(2).strip()
            if body and body not in _EMPTY:
                adds.append(MemoryItem(text=body, type=typ))
            continue
        clean = ln.lstrip("-*· ").strip()
        if clean and clean not in _EMPTY and not clean.startswith("#"):
            bare.append(clean)

    if not adds and bare:
        adds = [MemoryItem(text=x) for x in bare]
    return adds, drops


def apply_changes(text: str, turns: int = 0) -> tuple[int, int]:
    """按模型输出的「新增 + 删除」更新记忆。返回 `(新增条数, 删除条数)`。

    删除是保守的：编号越界直接忽略，不会因为模型数错就删掉别的。
    """
    adds, drops = parse_changes(text)
    if not adds and not drops:
        return 0, 0

    items = load()
    removed = 0
    if drops:
        kill = {i for i in drops if 0 <= i < len(items)}
        if kill:
            items = [m for i, m in enumerate(items) if i not in kill]
            removed = len(kill)

    seen = {_norm(i.text) for i in items}
    added = 0
    for m in adds:
        t = _norm(m.text)
        if not t or t in seen:
            continue
        m.text = t
        m.turns = turns
        items.append(m)
        seen.add(t)
        added += 1

    if added or removed:
        save(items[-MAX_ITEMS:])
    return added, removed


def numbered(limit: int = 40) -> list[tuple[int, MemoryItem]]:
    """带编号的记忆清单，给压缩请求用（模型要按编号说删哪条）。"""
    items = load()
    return list(enumerate(items))[-limit:]


def clear() -> None:
    save([])


def count() -> int:
    return len(load())


def render(items: list[MemoryItem], limit: int = MAX_CHARS) -> str:
    """把一组记忆渲染成**按四类分组**的文本。

    分组不是装饰：模型看到「用户偏好」和「项目结论」是两个区块，就不会把
    「AG 动量失效」当成用户的偏好来用。老条目没有类型，落到「其他」。

    从最新的往前取，超长就截断 —— 记忆是「最近还关心什么」的近似。

    和 `brief()` 分开是为了可测：测试要能拿一组假数据渲染，而不碰用户的
    真实记忆文件。
    """
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    used = 0
    full = False

    for m in reversed(items):
        typ = m.type if m.type in MEMORY_TYPES else ""
        for ln in m.text.split("\n"):
            ln = ln.strip()
            if not ln or ln in seen:
                continue
            line = f"- {ln}"
            if used + len(line) + 1 > limit and used > 0:
                full = True
                break
            groups.setdefault(typ, []).append(line)
            seen.add(ln)
            used += len(line) + 1
        if full:
            break

    parts: list[str] = []
    for typ in MEMORY_TYPES + ("",):
        rows = groups.get(typ)
        if not rows:
            continue
        # 没分类的老条目不加小标题 —— 全都没分类时通篇都是「**其他**」，
        # 纯噪声，还会让注入文本多出一层结构
        head = f"**{TYPE_LABEL[typ]}**\n" if typ else ""
        parts.append(head + "\n".join(rows))
    return "\n\n".join(parts)


def brief(limit: int = MAX_CHARS) -> str:
    """拼成给 system prompt 的一段。见 `render()`。"""
    return render(load(), limit)
