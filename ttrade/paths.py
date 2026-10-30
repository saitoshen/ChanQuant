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
import shutil
import sys

APP_NAME = "ChanQuant"
# 旧名字。用户目录要能从它迁过来 —— 账户、策略、因子、对话记忆全在里面，
# 直接换个 APP_NAME 等于把它们全丢了
LEGACY_APP_NAME = "TrendDesk"


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
    """行情数据目录。market: 'us' 美股 / 'cn' 国内期货。

    另有三个「衍生数据」目录，都由 Tushare 明细聚合而来：

    - `contracts`：全部**真实合约**日线（每品种一个文件），体积大（约 130MB），
      不进 git、不打进包，靠 `python download_contracts.py` 生成；
    - `aux`：从真实合约/期权聚合出的**期限结构与波动率摘要**，每品种一个文件，
      体积小（约 30MB），随包分发 —— 展期因子和期权因子只依赖它；
    - `options_raw`：期权原始日线（可选，`--keep-raw` 才生成，1GB 量级）。
    """
    sub = {
        "us": "data",
        "cn": "data_futures",
        "contracts": "data_contracts",
        "aux": "data_futures_aux",
        "options_raw": "data_options_raw",
    }.get(market, market)
    return os.path.join(bundle_root(), sub)


def _base_dir() -> str:
    """用户数据目录的父目录（各平台的惯例位置）。"""
    if sys.platform == "win32":
        return os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    return os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share"
    )


_migrated = False


def _migrate_legacy() -> None:
    """把 `TrendDesk` 用户目录整体搬到 `ChanQuant`，只搬一次。

    项目改名前叫 TrendDesk，那会儿账户、策略、因子、对话记忆、日限额全在
    `%LOCALAPPDATA%\\TrendDesk` 下。改名时如果只是换个 `APP_NAME`，用户升上来会
    看到一个「全新」的程序 —— 账户没了、自己写的因子没了、连聊过的记忆都没了。

    所以第一次运行就整目录搬过来（同盘 rename，不拷数据，几毫秒）。搬不动
    （旧目录里有别的进程占着的文件、跨盘、权限不够）就退化成拷贝，再不行就
    留个 `.migrate_failed` 标记让用户自己搬 —— 但**绝不**覆盖新目录里已有的东西。
    """
    global _migrated
    if _migrated:
        return
    _migrated = True

    base = _base_dir()
    new = os.path.join(base, APP_NAME)
    old = os.path.join(base, LEGACY_APP_NAME)
    if not os.path.isdir(old) or os.path.abspath(old) == os.path.abspath(new):
        return

    if not os.path.exists(new):
        try:
            shutil.move(old, new)
            return
        except OSError:
            pass                          # 退化成拷贝

    if os.path.exists(os.path.join(new, ".migrated")):
        return                            # 已经迁过，旧目录是用户自己留的副本
    try:
        for name in os.listdir(old):
            src, dst = os.path.join(old, name), os.path.join(new, name)
            if os.path.exists(dst):
                continue                  # 新目录优先，不覆盖
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    except OSError as exc:  # noqa: BLE001
        print(f"[ChanQuant] 旧配置目录搬迁失败，请手动把 {old} 里的内容移到 {new}：{exc}")
    else:
        try:
            with open(os.path.join(new, ".migrated"), "w", encoding="utf-8") as fh:
                fh.write(f"migrated from {old}\n")
        except OSError:
            pass


def user_dir() -> str:
    """可写目录，用于日志、缓存、用户自己的数据。"""
    _migrate_legacy()
    path = os.path.join(_base_dir(), APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def cache_dir(name: str = "mpl") -> str:
    """缓存目录。matplotlib 的字体缓存必须落在可写位置。

    Windows 下挂在 `user_dir()` 里面，**必须**先走一次 `user_dir()` —— 否则
    缓存目录先被建出来，`_migrate_legacy()` 就会以为新目录已存在而放弃整目录搬迁。
    """
    if sys.platform == "win32":
        path = os.path.join(user_dir(), "cache", name)
    elif sys.platform == "darwin":
        path = os.path.join(os.path.expanduser("~"), "Library", "Caches", APP_NAME, name)
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
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
