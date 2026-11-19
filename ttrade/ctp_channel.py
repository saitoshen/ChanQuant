"""CTP API 通道：标准 CTP 与融航，二选一，**一个进程只绑一个**。

## 为什么要有这个模块

融航资管中台提供了一套「CTP 兼容」的替换 dll（`thosttraderapi_se.dll`，
约 218 KB，配 `RohonBaseV64.dll` + VC2010 运行时）。把它塞进 openctp-ctp 的
`openctp_ctp.libs`，就能用现成的 CTP 接口连融航，不需要 `vnpy_rohon`。

但这条路有三条**实测**出来的硬约束（2026-11-19，本机）：

1. **pyd 版本必须和 dll 的 API 版本对齐**。
   `openctp-ctp 6.7.11` 的 pyd 配融航 6.7.2 的 dll 会
   `DLL load failed — 找不到指定的程序`（ERROR_PROC_NOT_FOUND），
   因为 6.7.11 要的导出符号融航没有。所以融航这份固定用 **6.7.2.8**。

2. **融航 dll 只能连融航中台**。
   拿它去连 SimNow 的 `tcp://182.254.243.31:30001` 会直接
   `0xC0000005`（访问违例）崩掉，连一个回调都进不去。所以**绝不能
   全局替换 site-packages 里的 dll** —— 那会把普通 CTP 账户一连就带崩，
   整个界面连同所有在跑的策略一起死。

3. **两套库不能同进程共存**。
   同进程先建 6.7.11 的 API、再建 6.7.2.8 的 API，无论并发还是顺序，
   都 exit 139。因此这里做**进程级互斥**：一旦绑定了一个通道，
   再要另一个就直接抛 `ChannelError`，而不是让 CTP 在某个回调里崩掉。

所以：普通 CTP 账户和融航账户要么分进程跑，要么同一时间只连一边。

## 融航其实有行情

替换包里只给了 trader 的 dll，但**融航的行情前置就是标准 CTP md 服务**：
vendor 里那份 6.7.2 的 `thostmduserapi_se` 连 `124.160.66.120:41253`
能连上、能登录（交易日 20261119）、能订阅（2026-11-19 实测）。所以不用
向融航另要 md dll，也不需要天勤兜行情。

此前记的「标准 md 连融航行情口 `0xC0000005` 崩溃」是**探测脚本自己的错**：
`api.RegisterSpi(Spi(api))` 传的是临时对象，CPython 当场回收，CTP 侧拿到
悬空指针就崩。SPI 必须被变量持有 —— 正式代码里是 `MdClient._spi`。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading

#: 通道名 → 说明。界面和日志里用它讲话
CHANNELS = {
    "ctp": "标准 CTP（openctp-ctp，跟随系统装的版本）",
    "rohon": "融航资管中台（CTP 兼容 dll，固定 openctp-ctp 6.7.2.8）",
}

_LOCK = threading.Lock()
_BOUND = ""          # "" / "ctp" / "rohon"
_MODULES: dict[str, tuple] = {}


class ChannelError(RuntimeError):
    """通道不可用，或者想在一个进程里用两个通道。"""


def vendor_dir() -> str:
    """融航运行时的目录（源码运行与 PyInstaller 打包两种情况）。"""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        packed = os.path.join(base, "vendor", "rohon_ctp")
        if os.path.isdir(packed):
            return packed
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "vendor", "rohon_ctp")


def bound() -> str:
    """当前进程绑定的通道（空串 = 还没绑）。"""
    return _BOUND


def available(kind: str) -> tuple[bool, str]:
    """能不能用这个通道。返回 (可用, 不能用时的原因)。只看文件，不加载 dll。"""
    if kind == "ctp":
        try:
            import importlib.util as _u

            if _u.find_spec("openctp_ctp") is None:
                return False, "没装 openctp-ctp（pip install openctp-ctp）"
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
    if kind == "rohon":
        pkg = os.path.join(vendor_dir(), "openctp_ctp", "__init__.py")
        if not os.path.isfile(pkg):
            return False, f"没有融航 CTP 运行时（缺 {pkg}）"
        libs = os.path.join(vendor_dir(), "openctp_ctp.libs")
        if not os.path.isdir(libs):
            return False, f"没有融航 DLL 目录（缺 {libs}）"
        # ABI 必须对得上。vendor 里的 .pyd 是按某个 Python 大版本编的
        # （cp314 依赖 python314.dll），跨版本只会得到一句
        # `DLL load failed while importing _thosttraderapi`，看不出所以然
        cur = f"cp{sys.version_info.major}{sys.version_info.minor}"
        try:
            with open(os.path.join(vendor_dir(), "_abi.txt"), encoding="utf-8") as fh:
                want = fh.read().strip()
        except OSError:
            want = ""
        if want and want != cur:
            return False, (
                f"融航运行时是按 {want} 编的，当前解释器是 {cur}，"
                f"直接加载会报 DLL load failed。\n"
                f"换一份 {cur} 的 openctp-ctp wheel 解到 vendor/rohon_ctp/，"
                f"或者用 {want} 的解释器启动")
        return True, ""
    return False, f"未知通道 {kind!r}"


def _load_vendored():
    """加载 vendored 的 openctp-ctp 6.7.2.8（顶层名与系统包不同，互不覆盖）。"""
    d = os.path.join(vendor_dir(), "openctp_ctp")
    name = "rohon_openctp_ctp"
    if name in sys.modules:
        m = sys.modules[name]
        return m.mdapi, m.tdapi
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(d, "__init__.py"), submodule_search_locations=[d])
    if spec is None or spec.loader is None:
        raise ChannelError(f"加载不了融航运行时：{d}")
    m = importlib.util.module_from_spec(spec)
    # 必须先放进 sys.modules：包内是相对导入（`from . import thosttraderapi`），
    # 不放进去会在 exec 时报 "attempted relative import with no known parent"
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return m.mdapi, m.tdapi


def load(kind: str = "ctp"):
    """绑定并返回 `(mdapi, tdapi)`。同一个进程只能绑一次。"""
    global _BOUND
    kind = (kind or "ctp").strip().lower()
    with _LOCK:
        if _BOUND:
            if _BOUND == kind:
                return _MODULES[kind]
            raise ChannelError(
                f"本进程已经绑定「{CHANNELS.get(_BOUND, _BOUND)}」，"
                f"不能再加载「{CHANNELS.get(kind, kind)}」。\n"
                "两套 CTP 库（6.7.11 与融航 6.7.2）在同一进程里会段错误崩溃，"
                "这是柜台 dll 层面的限制，绕不过去。\n"
                "办法：把融航账户和普通 CTP 账户分到两个进程里跑 —— "
                "同一个界面里请只连同一类账户。"
            )
        if kind == "ctp":
            from openctp_ctp import thostmduserapi as mdapi
            from openctp_ctp import thosttraderapi as tdapi
        elif kind == "rohon":
            mdapi, tdapi = _load_vendored()
        else:
            raise ChannelError(f"未知通道 {kind!r}（只认 ctp / rohon）")
        _MODULES[kind] = (mdapi, tdapi)
        _BOUND = kind
        return mdapi, tdapi


def reset_for_test() -> None:
    """把绑定状态清掉。**只给自测用** —— dll 已经进进程了，真实场景清不掉。"""
    global _BOUND
    with _LOCK:
        _BOUND = ""
        _MODULES.clear()
