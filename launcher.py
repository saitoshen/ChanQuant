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

    sys.exit(run_selftest())

from gui.app import _log_fatal, main


def run() -> int:
    try:
        return main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - 无控制台时只能落盘记录
        _log_fatal(exc)
        raise


if __name__ == "__main__":
    sys.exit(run())
