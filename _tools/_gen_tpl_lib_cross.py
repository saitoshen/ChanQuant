"""把「新写的」截面策略从策略库收编为内置模板（追加到 strategy_templates_lib.py）。

复用 _gen_tpl_lib_11.py 的生成 / 校验逻辑；只处理 mine_cross 组里尚未内置的
12 条（有色 / 黑色 / 油脂 / 化工 / 全品种 / 股指期货 等），key 从 tpl_lib_cross_04 起。

**先生成到 _gen_cross.txt 并做回放等价校验，人工核对后再追加源码** —— 不直接写死。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttrade import strategy_store as S, strategy_templates_lib as L
from _gen_tpl_lib_11 import _ALWAYS, _BASE, _DROP, _teach, build, render, verify  # noqa: E402

# 库里名字（按创建顺序）；key 从 cross_04 开始自动编号
NAMES = [
    "有色截面对冲",
    "黑色截面对冲",
    "油脂截面对冲",
    "化工截面对冲",
    "全品种截面对冲",
    "股指期货截面对冲",
    "股指期货截面对冲2",
    "油脂截面对冲2",
    "油脂截面对冲3 Xgboost",
    "全品种截面对冲2",
    "全品种截面对冲3",
    "全品种截面对冲4",
]
GROUP = "mine_cross"
START = 4


def main() -> int:
    lib = {c.name: c for c in S.load_library()}
    builtin = {v["name"] for v in L.RAW}
    blocks, fails = [], []
    for i, name in enumerate(NAMES):
        key = f"tpl_lib_cross_{START + i:02d}"
        if name in builtin:
            print(f"[SKIP] {key:18s} {name}（已内置）")
            continue
        if name not in lib:
            fails.append(f"库里找不到 {name}")
            continue
        src = lib[name]
        t = build(name, key, GROUP)
        bad = verify(t, src)
        print(f"[{'OK ' if not bad else 'BAD'}] {key:18s} {name:28s} "
              f"字段 {len(t) - 4:2d}  teach={t['teach'][:60]}")
        for b in bad:
            print("      ! " + b)
        if bad:
            fails.append(f"{name}: {len(bad)} 处不等")
        blocks.append(render(t))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gen_cross.txt")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(blocks))
    print(f"\n生成 {len(blocks)} 条 -> {out}，失败 {len(fails)}")
    for f in fails:
        print("  FAIL " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
