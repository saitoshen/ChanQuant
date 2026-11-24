"""清掉「单独跑某个自检用例」在真实策略库里留下的残留条目。

背景：`_tools/_run_selftest_case.py` 是把某个用例从 `_run()` 里摘出来单独执行，
**跳过了全量自检末尾的「恢复策略库」那一步**。`_strategy_save_action` 会真的
往库里存两条「自检用·动作存库 2h / 多品种」，单独跑完就留在用户库里了 ——
下一次全量自检会因为「重名」直接红掉（2026-11-24 实测）。

这里直接改 JSON，只删**名字带「自检用·动作存库」且 created 是今天**的条目，
其它一个字节不动（不走 `save_library`，避免被消毒逻辑顺手改写别的字段）。
删之前先落一份 `.bak_selftest_residue_<时间>`。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

PREFIX = "自检用·动作存库"
TODAY = time.strftime("%Y-%m-%d")


def lib_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "ChanQuant", "strategy_lib.json")


def main() -> int:
    p = lib_path()
    items = json.load(open(p, encoding="utf-8"))
    keep, drop = [], []
    for d in items:
        name = str(d.get("name") or "")
        created = str(d.get("created") or "")
        if name.startswith(PREFIX) and created.startswith(TODAY):
            drop.append((d.get("sid"), name, created))
        else:
            keep.append(d)
    if not drop:
        print(f"没有要删的（{len(items)} 条）")
        return 0
    if len(items) - len(drop) < 700:
        print(f"🔴 删完只剩 {len(items) - len(drop)} 条，像是选错文件了，中止")
        return 1
    bak = f"{p}.bak_selftest_residue_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(p, bak)
    print(f"备份 → {bak}")
    for sid, name, created in drop:
        print(f"删除 {sid}  {name}  ({created})")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(keep, fh, ensure_ascii=False, indent=1)   # 与 _save() 同格式
    os.replace(tmp, p)
    print(f"{len(items)} → {len(keep)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
