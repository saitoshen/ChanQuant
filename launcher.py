"""桌面程序的统一入口，源码运行和打包运行都从这里进。

    python launcher.py                    # 源码运行（等价于 python gui/app.py）
    python launcher.py --selftest         # 不开窗口，只做环境自检
    ChanQuant.exe --selftest              # 验证打包后的程序是否真的能用

单独放一个 launcher 而不是直接拿 `gui/app.py` 当入口，是因为 PyInstaller 会把
入口脚本当成顶层脚本执行，此时 `__name__ == "__main__"`、`sys.path` 的初始值
都和 `import gui.app` 不一样，容易在相对导入和路径推导上出问题。入口保持极薄，
业务代码全部留在 `gui.app` 里，源码运行和打包运行的行为就完全一致。
"""

from __future__ import annotations

import os
import sys

# 源码运行时把项目根加入 sys.path（打包后模块已收进 PYZ，不需要也不该改）
if not getattr(sys, "frozen", False):
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

if "--selftest" in sys.argv:
    from ttrade.selftest import run as run_selftest

    _code = run_selftest()
    # 自检跑完必须**硬退出**，不能让解释器慢慢 finalize 之后自然结束。
    #
    # 自检会把真实主窗口搭起来（QApplication + 几百个 QWidget），还要画图、开
    # duckdb 连接。CPython 退出时按 gc 顺序逐个析构这些对象，而 PySide6 与原生库
    # 的析构顺序不受我们控制 —— 实测**打包版**在这里 fail-fast：
    # Windows 退出码 0xC0000409，而 `selftest.log` 里 81 项全绿、一行 [FAIL] 都没有。
    # CI 只看退出码，于是这一轮被判定为「自检失败」，日志里却什么都看不到；
    # v3.7.4 的构建就是这么白跑一轮的（Release 被整个跳过）。
    #
    # 日志在 `run()` 返回前就已经 `with` 落盘并关闭文件、退出码也拿到了，之后没有
    # 任何非做不可的清理，OS 级退出最干净。
    try:
        for _stream in (sys.stdout, sys.stderr):
            if _stream is not None:
                _stream.flush()
    except Exception:  # noqa: BLE001 - 窗口程序没有控制台，flush 失败无所谓
        pass
    os._exit(_code)

from gui.app import _log_fatal, main


def run() -> int:
    # 崩溃取证要在**任何重活之前**装好（见 ttrade/crashlog.py）：
    # 每 10 秒把所有 Python 线程的栈顶写进 logs/heartbeat.log，崩了打开它就知道
    # 出事那一刻每个线程停在哪一行。2026-11-23/24 那串 `0xc0000005` 光靠事件日志
    # 只能拿到「哪个 dll 的哪个偏移」，完全看不出是查持仓、报单回调还是 Release。
    #
    # 🔴 放在 `run()` 里而不是模块顶层：`--selftest` 那条路在上面 `os._exit` 掉了，
    # 不会走到这里。自检如果也装，`os._exit` 跳过 atexit ⇒ `run.marker` 留在盘上
    # ⇒ 下次开界面会误报「上次是异常退出」。
    from ttrade import crashlog

    crashlog.install("gui")
    try:
        return main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - 无控制台时只能落盘记录
        _log_fatal(exc)
        raise


if __name__ == "__main__":
    sys.exit(run())
