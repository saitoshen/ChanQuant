"""工具结果的预算：超长的落盘，只回首尾预览。

## 为什么必须做这件事

量化场景天然产出大结果：一个 68 因子 × 5000 根的矩阵，序列化出来几 MB；
一次全市场扫描的表格几百 KB。这些整段塞进对话有三个后果 ——

1. 模型的上下文直接爆掉，后面几轮全在「忘事」
2. token 白烧，而且烧在它根本读不完的东西上
3. 界面渲染几万字符会卡

Claude Code 的做法是给每个工具一个 `maxResultSizeChars`，超了就把结果存到
磁盘、只给模型一段预览和文件路径。**关键是「不丢」** —— 预览不够时可以回头
读全文，而不是让模型凭半截数据猜。

## 这里怎么做的

- `clip()` 把超长文本落盘，返回「首尾预览 + 编号」
- 预览里带 `read_output` 的调用提示，模型可以分段把全文读回来
- 落盘目录在用户目录下（`agent_out/`），不进包、不进 git，7 天后自动清

## 为什么首尾各留一段

工具结果的信息密度不平均：开头通常是表头 / 结论，结尾通常是汇总行 / 报错。
只留开头会丢掉「最后那行说跑失败」这种最要紧的信息。
"""

from __future__ import annotations

import os
import re
import time

from ttrade.paths import user_dir

# 落盘目录名（在 user_dir() 下）
DIR_NAME = "agent_out"
# 超过这个天数就清掉。这些是中间产物，不是用户的资料
KEEP_DAYS = 7
# 预览里尾部占的比例。头部给得多，因为表头和信息主体都在前面
TAIL_RATIO = 0.25

_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def out_dir() -> str:
    path = os.path.join(user_dir(), DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _sweep() -> None:
    """清掉过期文件。失败就算了 —— 清理不该让一次工具调用出错。"""
    cutoff = time.time() - KEEP_DAYS * 86400
    try:
        for fn in os.listdir(out_dir()):
            p = os.path.join(out_dir(), fn)
            try:
                if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                    os.remove(p)
            except OSError:
                continue
    except OSError:
        pass


def spill(text: str, tag: str = "out") -> str:
    """把全文写盘，返回编号（文件名去掉扩展名）。写不进去返回空串。"""
    _sweep()
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", str(tag or "out"))[:32] or "out"
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}"
    try:
        with open(os.path.join(out_dir(), name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        return ""
    return name


def clip(text: str, limit: int, tag: str = "out") -> str:
    """按上限裁剪。`limit < 0` 表示不裁（工具自己已经限过长度）。

    返回的是**回灌给模型的文本**，不是给人看的。所以裁剪说明里要写清
    「怎么把剩下的拿回来」，否则模型只会回一句「结果被截断了」。
    """
    text = text or ""
    if limit < 0 or len(text) <= limit:
        return text

    head_n = int(limit * (1 - TAIL_RATIO))
    tail_n = limit - head_n
    head, tail = text[:head_n], text[-tail_n:]
    omitted = len(text) - head_n - tail_n

    name = spill(text, tag)
    if not name:
        # 落盘失败（磁盘满 / 权限）。宁可给一份截断的预览，也不能让整轮失败
        return (f"{head}\n\n…（中间省略 {omitted:,} 字符；"
                f"完整内容落盘失败，请缩小查询范围重试）…\n\n{tail}")

    return (
        f"{head}\n\n"
        f"…（中间省略 {omitted:,} 字符，全文 {len(text):,} 字符）…\n\n"
        f"{tail}\n\n"
        f"—— 完整结果已存为 `{name}`。要接着看就用动作块 "
        f'`{{"action": "read_output", "id": "{name}", "start": {head_n}}}` '
        f"分段读，别一次拉全文。"
    )


def read_chunk(name: str, start: int = 0, limit: int = 6000) -> tuple[str, int, str]:
    """分段读回落盘的结果。返回 `(这一段的文本, 全文长度, 错误)`。"""
    name = str(name or "").strip()
    if not name or not _ID.match(name):
        return "", 0, f"编号不合法：{name!r}"
    p = os.path.join(out_dir(), name + ".txt")
    if not os.path.isfile(p):
        return "", 0, f"找不到编号 {name}（可能已过期清理，7 天）"
    try:
        with open(p, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return "", 0, f"读不出来：{exc}"

    total = len(text)
    start = max(0, min(int(start or 0), total))
    limit = max(200, min(int(limit or 6000), 20000))
    chunk = text[start:start + limit]
    end = start + len(chunk)
    if end < total:
        chunk += (f"\n\n…（还有 {total - end:,} 字符。"
                  f'下一段用 {{"action": "read_output", "id": "{name}", '
                  f'"start": {end}}}）…')
    return chunk, total, ""


def list_outputs() -> list[tuple[str, int, str]]:
    """列出还留着的落盘结果，给界面用。`(编号, 字节数, 时间)`。"""
    _sweep()
    out: list[tuple[str, int, str]] = []
    try:
        for fn in sorted(os.listdir(out_dir()), reverse=True):
            if not fn.endswith(".txt"):
                continue
            p = os.path.join(out_dir(), fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out.append((fn[:-4], st.st_size,
                        time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime))))
    except OSError:
        pass
    return out
