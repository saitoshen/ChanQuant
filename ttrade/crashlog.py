"""崩溃取证：**心跳栈快照** + 可选 `faulthandler` + 「上次是不是崩的」。

## 为什么要有这个模块

2026-11-23 / 24 那串崩溃，事件日志只给了这么点信息：

    错误模块名称: thosttraderapi_se-e8dc311f….dll
    异常代码: 0xc0000005   偏移: 0x25d9e0

「哪个 dll 的哪个偏移」能说明是原生访问违例，但**说不出出事时 Python 在干什么**
—— 是主循环在查持仓，还是 CTP 回调线程正把回报塞进 dict，还是 `Release()` 在收
会话。而这三种要修的地方完全不同。`live.log` 也帮不上：它只在「目标仓变了」
「对不上」这类业务事件上写字，崩溃前寂静十几分钟是常态（20:32 / 21:38 那两次
都是最后一行业务日志之后二十多分钟才死）。

所以这里做的是**一直在录、崩了能看到最后几秒**：每 `HEARTBEAT_SEC` 秒把所有
Python 线程的栈顶几帧写进 `logs/heartbeat.log`，崩溃后打开它就是「出事那一刻
每个线程停在哪一行」。

## 三件事

1. **心跳**（永远开）：线程栈快照，环形保留最后 `~MAX_LINES` 行。纯 Python
   读 `sys._current_frames()`，不改任何行为，不会把「本来能恢复的段错误变成
   致命错误」。
2. **`faulthandler`**（默认关，`CHANQUANT_FAULTHANDLER=1` 开启）：能把 C 层
   崩溃时的完整 Python 栈写下来，但它在 Windows 上装的是**第一现场**处理器
   —— Qt / 原生库那种「先探一个地址、失败了再恢复」的小段错误会被它当成致命
   错误。自检那次 macOS x64 从 exit 0 变 139 就是这么来的，所以默认不开。
3. **「上次是异常退出」判定**：正常退出会删掉 `logs/run.marker`；启动时它还在
   ⇒ 上一轮是被杀 / 崩溃的。界面日志里直接说一句，省得用户以为是「自己关的」。

## 怎么用

    from ttrade import crashlog
    crashlog.install("gui")        # 越早越好（QApplication 之前）

`install()` 幂等，重复调用只返回第一次的结论字符串。
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time

#: 每多少秒录一次
HEARTBEAT_SEC = 10.0
#: 每个线程录几帧（栈顶在最上面）
FRAMES = 5
#: 文件超过这个行数就开始裁
MAX_LINES = 1400
#: 裁完留多少行
KEEP_LINES = 700

_LOCK = threading.RLock()        # `_beat` 持锁时会进 `_trim`，必须可重入
_INSTALLED = False
_FH = None                  # 心跳文件句柄
_LINES = 0                  # 已经写了多少行
_THREAD = None
_STOP = threading.Event()
#: 上次异常退出的提示（空串 = 正常）
LAST_EXIT = ""


def _log(name: str) -> str:
    from ttrade.paths import log_dir
    d = log_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def _frame_line(frame) -> str:
    """一帧压成 `文件:行 函数`。只留文件名，不留整条绝对路径（太长）。"""
    code = frame.f_code
    fn = os.path.basename(code.co_filename or "?")
    return f"{fn}:{frame.f_lineno} {code.co_name}"


def snapshot(reason: str = "") -> list[str]:
    """所有线程的栈顶快照。**不加锁、不分配大对象**，可在任何时候调。"""
    out = [""]
    out.append(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {reason} "
               f"rss={_rss_mb():.0f}MB ===")
    me = threading.current_thread()
    frames = sys._current_frames()
    for th in threading.enumerate():
        if th is me:
            continue
        fr = frames.get(th.ident)
        tag = "D" if th.daemon else " "
        if fr is None:
            out.append(f"[{tag}] {th.name}: <无 Python 帧>")
            continue
        lines = [f"[{tag}] {th.name}:"]
        n = 0
        while fr is not None and n < FRAMES:
            lines.append(f"      {_frame_line(fr)}")
            fr = fr.f_back
            n += 1
        out.extend(lines)
    return out


def _rss_mb() -> float:
    """当前进程常驻内存（MB）。`psutil` 没装时退回 Windows API。

    崩溃前那一轮 RSS 很有用：泄露型问题和「内存耗光后原生分配失败」能一眼分开。
    """
    try:
        import psutil                                  # noqa: PLC0415
        return psutil.Process().memory_info().rss / 1048576.0
    except Exception:                                  # noqa: BLE001
        pass
    try:
        import ctypes                                  # noqa: PLC0415
        from ctypes import wintypes                    # noqa: PLC0415

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(
                h, ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / 1048576.0
    except Exception:                                  # noqa: BLE001
        pass
    return 0.0


def _trim() -> None:
    """环形裁剪。只在超过 `MAX_LINES` 时做，约两分钟一次。

    🔴 Windows 上**不能**「开着句柄把临时文件 `os.replace` 过去」—— 自己的
    句柄把目标文件锁住了，`os.replace` 直接 `PermissionError`。原来那版把它
    `except OSError: pass` 吞了，于是裁剪一次都没生效，`heartbeat.log` 会一直
    涨（实测跑 2 秒就 290 行，一天下来 40 万行）。正确做法是先关句柄、改完
    再以追加方式重开。
    """
    global _LINES, _FH
    if _FH is None:
        return
    path = getattr(_FH, "name", "")
    if not path:
        return
    try:
        _FH.close()
    except Exception:                                    # noqa: BLE001
        pass
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            rows = fh.readlines()
    except OSError:
        rows = []
    keep = rows[-KEEP_LINES:]
    # 从第一条快照头开始，别把上一个块切一半
    for i, row in enumerate(keep):
        if row.startswith("=== "):
            keep = keep[i:]
            break
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(keep)
        _LINES = len(keep)
    except OSError:
        pass
    try:
        _FH = open(path, "a", encoding="utf-8", buffering=1)
    except OSError:
        _FH = None


def _beat() -> None:
    global _LINES
    while not _STOP.wait(HEARTBEAT_SEC):
        try:
            rows = snapshot()
            with _LOCK:
                if _FH is None or _FH.closed:
                    return
                _FH.write("\n".join(rows) + "\n")
                _FH.flush()
                _LINES += len(rows)
                if _LINES > MAX_LINES:
                    _trim()
        except Exception:                             # noqa: BLE001
            # 取证工具自己不能把主程序带崩
            continue


def install(tag: str = "gui") -> str:
    """装好取证。返回一句人话（上次是不是异常退出）。幂等。"""
    global _INSTALLED, _FH, _LINES, _THREAD, LAST_EXIT
    with _LOCK:
        if _INSTALLED:
            return LAST_EXIT
        _INSTALLED = True

    marker = _log("run.marker")
    try:
        if os.path.isfile(marker):
            with open(marker, "r", encoding="utf-8", errors="replace") as fh:
                prev = fh.read().strip().splitlines()
            LAST_EXIT = ("上一次是**异常退出**（没走正常关闭流程）。"
                         f"出事时刻的线程栈见 logs/heartbeat.log"
                         + (f"；上一轮启动于 {prev[0]}" if prev else ""))
            try:
                os.replace(marker, marker + ".bak")
            except OSError:
                pass
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} "
                     f"tag={tag} py={sys.version.split()[0]}\n")
    except OSError:
        LAST_EXIT = LAST_EXIT or ""

    # 可选：faulthandler（默认关，见模块说明）
    if str(os.environ.get("CHANQUANT_FAULTHANDLER", "")).strip() not in ("", "0"):
        try:
            import faulthandler                          # noqa: PLC0415
            fh = open(_log("fault.log"), "a", encoding="utf-8", buffering=1)
            fh.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 "
                     f"pid={os.getpid()}\n")
            faulthandler.enable(fh, all_threads=True)
        except Exception:                                # noqa: BLE001
            pass

    try:
        path = _log("heartbeat.log")
        n = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                n = sum(1 for _ in fh)
        except OSError:
            n = 0
        _FH = open(path, "a", encoding="utf-8", buffering=1)
        rows = snapshot(f"install({tag})")
        _FH.write("\n".join(rows) + "\n")
        _LINES = n + len(rows)
        _THREAD = threading.Thread(target=_beat, name="heartbeat",
                                   daemon=True)
        _THREAD.start()
    except OSError:
        _FH = None

    atexit.register(_shutdown)
    return LAST_EXIT


def _shutdown() -> None:
    """正常退出：停心跳、删 marker。marker 没了，下一轮就不会误报「崩过」。"""
    _STOP.set()
    mark_clean_exit("正常退出")


def mark_clean_exit(reason: str = "正常退出") -> None:
    """记一行「正常退出」并删掉 marker。

    `atexit` 走这条路；`os._exit()` 那条路（关窗硬退出，见
    `gui.app.MainWindow.closeEvent`）**必须自己调一次** —— `os._exit` 不看
    `atexit`，不调的话下一轮开界面又会误报「上次是异常退出」。
    """
    global _FH
    try:
        if _FH is not None and not _FH.closed:
            with _LOCK:
                _FH.write("\n".join(snapshot(reason)) + "\n")
                _FH.flush()
                _FH.close()
    except Exception:                                    # noqa: BLE001
        pass
    try:
        p = _log("run.marker")
        if os.path.isfile(p):
            os.remove(p)
    except OSError:
        pass


def breadcrumb(text: str) -> None:
    """往心跳文件里插一行面包屑，用来定位「走到哪一步死的」。

    心跳每 10 秒才一拍，关窗这种几百毫秒的事根本拍不到；崩在关窗过程中时，
    日志上只能看到「最后一拍还在正常跑，然后没了」。插几行面包屑就一目了然。
    """
    try:
        if _FH is not None and not _FH.closed:
            with _LOCK:
                _FH.write(f"--- {time.strftime('%H:%M:%S')} {text}\n")
    except Exception:                                    # noqa: BLE001
        pass


# ------------------------------------------------------------------ 自测用

def _reset_for_test() -> None:
    """把状态清干净，**只给自检用**。"""
    global _INSTALLED, _FH, _LINES, _THREAD, LAST_EXIT
    _STOP.set()
    try:
        if _FH is not None and not _FH.closed:
            _FH.close()
    except Exception:                                    # noqa: BLE001
        pass
    _INSTALLED, _FH, _THREAD, _LINES = False, None, None, 0
    LAST_EXIT = ""
    _STOP.clear()


if __name__ == "__main__":                               # 手工看一眼
    print(install("probe"))
    for _ in range(2):
        time.sleep(1)
        print("\n".join(snapshot("probe")))
    _shutdown()
    print("ok")
