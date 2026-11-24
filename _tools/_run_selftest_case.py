# -*- coding: utf-8 -*-
"""只跑**一个**自检用例（把嵌套函数从 `ttrade/selftest.py` 里抽出来单独执行）。

## 为什么需要它

`ttrade/selftest.py` 的全量入口只有 `launcher.py --selftest`，而那条路子会连
柜台 dll —— **界面开着的时候不能跑**（柜台线程撞 dll，`exit=139`）。可改完代码
总得有个办法先确认「新写的那个用例本身跑得通」（NameError / 拼错 / 断言写反），
而不是等到收盘后关掉界面才发现。

这个脚本用 `ast` 把 `_run()` 里指定的那个嵌套 `def` 摘出来，单独 exec 一遍。
它**不是**验收：跨用例的耦合、全量顺序、界面相关的检查都不在这里。

用法：
    python _tools/_run_selftest_case.py 天勤也一个账户一条会话
    python _tools/_run_selftest_case.py _tq_one_session      # 也可以给函数名
"""

from __future__ import annotations

import ast
import io
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    want = argv[1] if len(argv) > 1 else ""
    if not want:
        print("用法：python _tools/_run_selftest_case.py <用例名或函数名>")
        return 2
    src = io.open(ROOT / "ttrade" / "selftest.py", encoding="utf-8").read()
    tree = ast.parse(src)

    run_fn = next((n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "_run"), None)
    if run_fn is None:
        print("找不到 `_run()`")
        return 2

    picked = []
    for node in ast.walk(run_fn):
        if node is run_fn:
            continue
        # 只认 `_run()` 里直接定义的抓 bug 用例，不认里面嵌套定义的小工具函数
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("_"):
            continue
        if node.name == want:
            picked.append((node.name, node))
            continue
        # 用例名：同一段里那条 `check("…", _func)` 的字符串
        seg = ast.get_source_segment(src, node) or ""
        if want and want in seg:
            picked.append((node.name, node))

    if not picked:
        names = sorted({n.name for n in ast.walk(run_fn)
                        if isinstance(n, ast.FunctionDef) and n.name.startswith("_")
                        and n is not run_fn})
        print(f"没找到用例 {want!r}。可用的函数名（前 40 个）：")
        print("  " + "  ".join(names[:40]))
        return 2

    import ttrade.selftest as ST          # noqa: E402 - 只为拿模块级 globals

    bad = 0
    for name, node in picked:
        mod = ast.Module(body=[node], type_ignores=[])
        code = compile(mod, str(ROOT / "ttrade" / "selftest.py"), "exec")
        # 用真模块的 globals 当底：用例里会用到 `time` / `os` / `json` 这些，
        # 只塞 `__name__` 的话会在「跑得通」的代码上报 NameError（假红）
        env: dict = dict(vars(ST))
        exec(code, env)                      # noqa: S102 - 自检脚本，输入是仓库自己的文件
        fn = env[name]
        try:
            msg = fn()
            print(f"通过  {name}: {msg}")
        except Exception:                    # noqa: BLE001
            bad += 1
            print(f"失败  {name}")
            traceback.print_exc()
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
