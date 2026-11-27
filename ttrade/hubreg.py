"""一个账户一条会话 —— 柜台连接的进程内注册表。

## 为什么要有这个文件（2026-11-23 融航那次）

组合页（`BrokerApi`）和交易页（`gui.live.LiveWorker`）原先各自
`CtpHub([spec]).start()` 一次。同一个融航账户因此在柜台那边挂着**两条 CTP
会话**，症状恰好是用户描述的那种「两块功能交互脱节」：

1. **委托回报按会话推送**。交易页「一键对齐」发出去的单，回报只进交易页那条
   会话的 `trader.orders`；组合页那条的 `_ord` 一直是空的 —— 用户看到的就是
   「对齐的委托在账户透视里查不到」。
2. **查询额度各抢一份**。柜台「每秒最多一个查询」是**按会话**算的，两套闸门
   各管各的，撞在同一毫秒时被丢的那笔静默消失（`-3` 都不一定回给你）。
3. **持仓是两份快照**。一边刷新快、一边刷新慢，界面上就是「交易页有 7 个
   合约的持仓，监控器全是不一致」。

合并不是「让组合页去读交易页的界面数据」，而是**两条链路认领同一个
`CtpHub` 对象**：同一个 `trader`、同一个 `md`、同一份 1Hz 查询轮转。

## 用法

    hub = hubreg.acquire("rohon", [spec], lambda: RohonHub([spec]))
    hub.start()                  # 幂等：后到的那一方立刻返回，不会二次登录
    ...
    if hubreg.release(hub):      # True = 我是最后一个使用者，该 close() 了
        hub.close()

`acquire()` 认 **天勤 / CTP / 融航** 三家。

天勤原来**不进**这张表，理由是「一条 `TqApi` 的订阅状态和 `wait_update()` 都是
单线程语义，两个页面共用一个对象只会把内部状态写坏」。那个结论没错，但结论
应该是「所以得先让它有一条自己的线程」，而不是「所以让它登两次」。2026-11-24
用户要求统一：

> 天勤的账户也要和 CTP 和融航账户一样，同一个账户在整个系统共享同一个会话，
> 这样一键对齐和账户持仓和委托成交的会话才能实时响应。

两条会话的后果在天勤这边更严重 —— 天勤是**按会话推送**持仓 / 委托 / 成交的，
两个页面的会话互不相通，「交易页下的单」在组合页那条会话里根本不会出现。
落地做法见 `ttrade/tqlive.py` 的 `LiveHub`：**一条泵线程独占 `wait_update`**，
其它线程的每次调用都进 `_lock`，`wait()` 退化成「等泵推来下一拍」。

## 键为什么带账号和前置

`(柜台, 账户名, 投资者账号, 交易前置, 账户类型)`。只按账户名是不够的：用户在账户表里把
一个账户从 SimNow 改成融航，名字没变、`kind` 没变（都是 ctp 系），只按名字
会拿到**上一条已经关掉的会话**。带上账号和前置之后，改配置就是换会话，符合
直觉。账户类型（`kq` / `sim` / `account`）是天勤那一列加进来的：同一个天勤账号
既能挂模拟也能挂实盘，两者必须是两条会话。
"""

from __future__ import annotations

import threading

#: 三家的**账户**会话都共享。天勤见文件开头那段（2026-11-24 起）
SHARED = ("tq", "ctp", "rohon")

_LOCK = threading.RLock()
#: (柜台, 账户名, 账号, 交易前置) → {"hub": 连接, "refs": 引用数}
_REG: dict[tuple, dict] = {}


def hub_key(kind: str, spec) -> tuple:
    """一条会话的身份。空账户（纯行情会话）也会得到一个键，但不会进表。"""
    return (
        str(kind or ""),
        str(getattr(spec, "name", "") or ""),
        str(getattr(spec, "user", "") or ""),
        str(getattr(spec, "td_front", "") or ""),
        # 天勤的「同一个天勤账号」可以挂不同的账户对象：`kq` 快期模拟 / `sim`
        # 本地模拟 / `account` 实盘。它们是**不同的会话**，只按账号+密码做键
        # 会让「实盘」拿到「模拟」那条连接 —— 下的单全在模拟盘里，界面上还
        # 一切正常。柜台那边这个字段是空串，不影响原来的键。
        str(getattr(spec, "kind", "") or ""),
    )


def _first(specs):
    if specs is None:
        return None
    if isinstance(specs, (list, tuple)):
        return specs[0] if specs else None
    return specs


def _reusable(hub) -> bool:
    """这条在册会话能不能借给别人。

    默认能。hub 可以自己实现 `reusable()` 否决复用 —— CTP/融航用它在
    「登录失败、冷却已过、从没真正登上过」时作废僵尸会话（2026-11-26：
    失败几次之后界面一直报「复用组合页已连的会话」却永远连不上，见
    `CtpHub.reusable()`）。没有这个方法的 hub（假柜台、天勤）行为不变。
    """
    probe = getattr(hub, "reusable", None)
    return True if not callable(probe) else bool(probe())


def acquire(kind: str, specs, factory):
    """借一条会话。表里已有可复用的就 **加一次引用** 直接返回，没有才造。

    注意返回值可能是**别人已经 `start()` 过**的连接 —— 所以 `CtpHub.start()`
    必须是幂等的（现在是：`_start_lock` 串行 + `_started` 短路）。不幂等的话
    后到的一方会把已经登录好的 `trader` 换掉，前一方手里的引用就成了野指针。

    🔴 会话自报不可复用时（`reusable()` 为假）**丢弃重建**：借到一条登录失败
    的僵尸会话，只会让 `start()` 撞失败冷却、界面误报「复用已连会话」，而且
    下一次 `release()` 又把它留在表里 —— 从此再也连不上。老持有者放手时
    `release()` 对已摘掉的会话返回 True，它自己 close，不会误伤别人。
    """
    spec = _first(specs)
    if spec is None or str(kind or "") not in SHARED:
        return factory()
    key = hub_key(kind, spec)
    with _LOCK:
        slot = _REG.get(key)
        if slot is not None and _reusable(slot["hub"]):
            slot["refs"] += 1
            return slot["hub"]
        if slot is not None:
            _REG.pop(key, None)
        hub = factory()
        _REG[key] = {"hub": hub, "refs": 1}
        return hub


def release(hub) -> bool:
    """放手。返回 True = 该 `close()` 了。

    - 还有别人在用 → False（谁都不许关，否则另一边的持仓/回报当场断）；
    - 最后一个 → True，并从表里摘掉；
    - **根本不在表里**（天勤的 hub、单测注入的假柜台）→ True，照旧由调用方关。
    """
    if hub is None:
        return False
    with _LOCK:
        for key, slot in list(_REG.items()):
            if slot["hub"] is not hub:
                continue
            slot["refs"] -= 1
            if slot["refs"] > 0:
                return False
            _REG.pop(key, None)
            return True
    return True


def forget(hub) -> None:
    """把一条**已经被关掉**的会话从表里摘掉（`CtpHub.close()` 里调）。

    没有这一步的话，先 `close()` 后 `release()` 会让表里留一个
    「refs > 0 但 trader 已经是 None」的僵尸，下一次 `acquire()` 拿到它之后
    所有查询都静默不发（`pump_queries()` 第一句就是 `if self.trader is None`）。
    """
    if hub is None:
        return
    with _LOCK:
        for key, slot in list(_REG.items()):
            if slot["hub"] is hub:
                _REG.pop(key, None)


def peek(kind: str, spec):
    """看一眼这个账户现在有没有活着的会话（不改引用数）。"""
    with _LOCK:
        slot = _REG.get(hub_key(kind, spec))
        return slot["hub"] if slot else None


def refs(hub) -> int:
    with _LOCK:
        for slot in _REG.values():
            if slot["hub"] is hub:
                return int(slot["refs"])
    return 0


def describe() -> str:
    """诊断用：`融航/融航户A refs=2` 这样的串，一行一条。"""
    with _LOCK:
        items = [(k, v) for k, v in _REG.items()]
    return "；".join(
        f"{k[0]}/{k[1] or k[2] or '?'} refs={v['refs']}" for k, v in items)


def size() -> int:
    with _LOCK:
        return len(_REG)


def clear() -> None:
    """自检和进程退出时清表。**不关连接** —— 关不关由持引用的一方决定。"""
    with _LOCK:
        _REG.clear()
