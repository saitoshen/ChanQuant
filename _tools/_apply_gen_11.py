"""把 `_tools/_gen_11.txt` 追加进 `ttrade/strategy_templates_lib.py` 的 `RAW` 末尾。

幂等：已经插过的键跳过，重复跑不会插两遍。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ttrade", "strategy_templates_lib.py")
GEN = os.path.join(ROOT, "_tools", "_gen_11.txt")

text = open(SRC, encoding="utf-8").read()
gen = open(GEN, encoding="utf-8").read()

if not text.rstrip().endswith("]"):
    raise SystemExit("源码不是以 `]` 结尾，先看一眼再动")

keys = [ln.split("key=", 1)[1].split(",", 1)[0].strip().strip("'\"")
        for ln in gen.splitlines() if ln.lstrip().startswith("dict(key=")]
dup = [k for k in keys if f"key='{k}'" in text]
if dup:
    raise SystemExit("这些键已经在源码里了，别插第二遍：" + "、".join(dup))
print(f"待追加 {len(keys)} 条：" + "、".join(keys))

i = text.rstrip().rfind("]")
head, tail = text[:i].rstrip("\n"), text[i:]
new = head + "\n" + gen.rstrip("\n") + "\n" + tail
with open(SRC, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(new)
print(f"写入完成：{SRC}（{len(text)} → {len(new)} 字符）")
