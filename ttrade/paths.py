"""统一的路径解析：同一份代码既能在源码目录里跑，也能在 PyInstaller 打出的包里跑。

两种运行形态下路径语义完全不同：

- **源码运行**：所有文件都在项目目录里，数据可读可写。
- **打包运行**（`sys.frozen = True`）：代码被打进 PYZ 归档，`data/`、`data_futures/`
  这些资源被复制到 `sys._MEIPASS`。在 Windows 的目录模式下 `_MEIPASS` 就是 exe 所在目录；
  在 macOS 的 `.app` 里它是 `Contents/Frameworks`，**签名后是只读的**。

因此这里把路径分成两类，不要混用：

- `bundle_root()` / `data_dir()` —— **只读**资源，打包后可能不可写
- `user_dir()` / `cache_dir()` / `log_dir()` —— **可写**目录，放在用户家目录下

之前踩过的坑：`MPLCONFIGDIR` 和 `error.log` 都指向项目目录，源码运行没问题，
但打包成 `.app` 后写入只读卷会静默失败，所以统一改走 `cache_dir()` / `log_dir()`。
"""

from __future__ import annotations

import os
import sys

APP_NAME = "TrendDesk"


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打出的包里。"""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> str:
    """只读资源根目录。

    - 打包后：`sys._MEIPASS`（单文件模式是临时解包目录，目录模式是 exe 旁边）
    - 源码运行：项目根目录
    """
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return meipass
        # 理论上不会走到这里；退化为可执行文件所在目录
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir(market: str = "us") -> str:
    """行情数据目录。market: 'us' 美股 / 'cn' 国内期货。"""
    sub = "data" if market == "us" else "data_futures"
    return os.path.join(bundle_root(), sub)


def user_dir() -> str:
    """可写目录，用于日志、缓存、用户自己的数据。"""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share"
        )
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def cache_dir(name: str = "mpl") -> str:
    """缓存目录。matplotlib 的字体缓存必须落在可写位置。"""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        path = os.path.join(base, APP_NAME, "cache", name)
    elif sys.platform == "darwin":
        path = os.path.join(os.path.expanduser("~"), "Library", "Caches", APP_NAME, name)
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        path = os.path.join(base, APP_NAME, name)
    os.makedirs(path, exist_ok=True)
    return path


def log_dir() -> str:
    """日志目录，打包后 `pythonw`/`.app` 没有控制台，异常只能写这里。"""
    path = os.path.join(user_dir(), "logs")
    os.makedirs(path, exist_ok=True)
    return path


def has_data(market: str = "us") -> bool:
    """数据目录里是否有可用的 CSV。"""
    d = data_dir(market)
    if not os.path.isdir(d):
        return False
    return any(fn.endswith(".csv") and not fn.startswith("_") for fn in os.listdir(d))
