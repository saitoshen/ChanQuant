"""验证 `selftest.log` 是**边跑边落盘**的（不是跑完才写）。

为什么要单独验：段错误时进程直接死，原来那份「跑完才 `open(...,'w')` 写一次」
的日志根本不存在 —— CI 上表现为「日志文件不存在（自检没跑完就没机会落盘）」，
arm64 那两轮（v3.9.6 / v3.9.15）就是这么白跑的。

做法：把 `log_dir()` 指到临时目录，跑真实的 `_run()`，但让**第 2 项**抛
`KeyboardInterrupt`（`check()` 只 catch `Exception`，它会穿透出来）——
进程在「第 1 项已记、第 2 项未记」的那一刻中止，此时读 `selftest.log`：
有 `[ OK ]` / `[FAIL]` 行 = 增量落盘成立；空文件或不存在 = 没生效。

界面开着也能跑：`_run()` 的前两项只判目录，不碰 Qt / 柜台。
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="cq_logprobe_")
    from ttrade import paths

    paths.log_dir = lambda: tmp        # noqa: SLF001 - 故意指到临时目录

    seen: list[str] = []

    class _Boom(KeyboardInterrupt):
        pass

    # 头部 `_emit` 会先调两次 data_dir（us / cn），第 3 次调用落在
    # `check("美股数据目录", ...)` 上 —— 那一次炸出去
    def fake_data_dir(market: str = "cn") -> str:
        seen.append(market)
        if len(seen) > 2:
            raise _Boom("故意断在第 2 项")
        return tmp

    paths.data_dir = fake_data_dir      # noqa: SLF001

    from ttrade import selftest

    try:
        selftest._run()                 # noqa: SLF001
        print("没被中断 —— 注入点失效，用例作废")
        return 1
    except _Boom:
        print(f"已断在第 2 项（data_dir 被调 {len(seen)} 次：{seen}）")

    log = os.path.join(tmp, "selftest.log")
    if not os.path.isfile(log):
        print(f"FAIL  日志不存在：{log}")
        return 1
    text = open(log, encoding="utf-8").read()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    print(f"日志存在：{len(text)} 字符 / {len(lines)} 行")
    for ln in lines:
        print("   | " + ln[:110])
    hits = [ln for ln in lines if ln.startswith(("[ OK ]", "[FAIL]"))]
    if not text.strip():
        print("FAIL  文件是空的 —— 说明还是「跑完才写」")
        return 1
    if not hits:
        print("FAIL  头部写了，但检查项一行都没落盘")
        return 1
    print(f"PASS  中断前已落盘 {len(hits)} 行检查结果 —— 崩溃时日志会留下来")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
