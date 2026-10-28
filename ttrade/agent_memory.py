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

## 上限

60 条 / 注入 4000 字。超了从最旧的丢 —— 记忆是「最近还关心什么」的近似，不是
档案库。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

from ttrade.paths import user_dir

FILE_NAME = "agent_memory.json"
MAX_ITEMS = 60
MAX_CHARS = 4000          # 注入 prompt 的总长度上限
_SPACE = re.compile(r"\s+")


@dataclass
class MemoryItem:
    text: str = ""
    kind: str = "summary"     # summary（压缩出来的一段）/ fact（单条事实）
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
    """原子写：先写 .tmp 再 replace。"""
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
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


def clear() -> None:
    save([])


def count() -> int:
    return len(load())


def brief(limit: int = MAX_CHARS) -> str:
    """拼成给 system prompt 的一段。从最新的往前取，超长就截断。

    一条记忆里可能有几行，按行摊平后每行一个 `- `：一行一条才好被模型当成
    「一条事实」，整段塞成一坨它会把几条混着用。
    """
    lines: list[str] = []
    used = 0
    seen: set[str] = set()
    for m in reversed(load()):
        for ln in m.text.split("\n"):
            # 同一件事被压缩进两条记忆很常见（后一次聊到了同一个结论）。按行去重，
            # 保留最新的那条 —— 不然 prompt 里会反复出现同一句
            if ln in seen:
                continue
            line = f"- {ln}"
            if used + len(line) > limit and lines:
                return "\n".join(reversed(lines))
            lines.append(line)
            seen.add(ln)
            used += len(line) + 1
    return "\n".join(reversed(lines))
