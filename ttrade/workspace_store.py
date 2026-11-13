"""组合工作区：一组策略 + 一份资金分配方案 = 组合页上的一个分页。

## 为什么不把成员直接塞进工作区

成员只存 `sid` 和**这个工作区分给它的本金**，策略本身的参数（品种、周期、
规则、成本档）还是 `portfolio.json` 那一份。这样：

- 同一条策略可以同时出现在两个工作区里，各拿不同的钱 —— 这正是「工作区」的
  意思：「实盘组合」和「新策略试跑」共用一条策略，但一个给 50 万、一个给 5 万
- 改策略参数只需要改一处，不会出现「同名两份、参数不一致」
- 删策略时工作区里那一条变成 `missing`，界面照实说，而不是整页打不开

## 运行

分页只是**视图**。运行的时候按**账户**合并：一个账户一条天勤连接，跑该账户
下所有工作区里 enabled 的策略，每条带自己的 `principal`。所以多个工作区是
真的同时在跑，但不会为了两个工作区开两条连接去抢同一个账户的仓。

## 兼容

第一次启动（`workspaces.json` 不存在）时，把现有 `portfolio.json` 整体搬进
一个叫「默认组合」的工作区，`principal` 取各自原值。**老文件不删** —— 回滚
到旧版本还能用。
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field

from . import portfolio_plan as PL
from . import strategy_store as SStore

FILE = SStore.WS_FILE
DEFAULT_NAME = "默认组合"


@dataclass
class Member:
    """工作区里的一个成员：哪条策略、分到多少本金。"""

    sid: str = ""
    principal: float = 0.0
    enabled: bool = True        # 本工作区里是否参与运行（策略自己也有 enabled）

    def to_dict(self) -> dict:
        return {"sid": self.sid, "principal": float(self.principal),
                "enabled": bool(self.enabled)}

    @classmethod
    def from_dict(cls, d: dict) -> "Member":
        d = d or {}
        return cls(sid=str(d.get("sid") or ""),
                   principal=float(d.get("principal") or 0.0),
                   enabled=bool(d.get("enabled", True)))


@dataclass
class Workspace:
    """一个组合工作区。"""

    id: str = ""
    name: str = ""
    # 总资金。0 = 用成员本金之和（老组合迁移过来的就是这个状态）
    total: float = 0.0
    mode: str = "equal"          # equal / risk / manual
    basis: str = "principal"     # principal / usage / peak
    # 校准倍数记在哪个参数上：split / principal / pct（见 PL.KNOBS）
    knob: str = "split"
    manual: dict = field(default_factory=dict)     # sid -> 金额（mode=manual）
    members: list = field(default_factory=list)    # [Member]
    account: str = ""            # 建议的账户（空 = 不指定）
    note: str = ""
    created: str = ""

    def sids(self) -> list[str]:
        return [m.sid for m in self.members if m.sid]

    def find(self, sid: str) -> Member | None:
        for m in self.members:
            if m.sid == sid:
                return m
        return None

    def principal_of(self, sid: str) -> float:
        m = self.find(sid)
        return float(m.principal) if m else 0.0

    def set_principal(self, sid: str, value: float) -> None:
        m = self.find(sid)
        if m is None:
            self.members.append(Member(sid=sid, principal=float(value)))
        else:
            m.principal = float(value)

    def add(self, sid: str, principal: float = 0.0) -> Member:
        m = self.find(sid)
        if m is None:
            m = Member(sid=sid, principal=float(principal))
            self.members.append(m)
        return m

    def drop(self, sid: str) -> bool:
        n = len(self.members)
        self.members = [m for m in self.members if m.sid != sid]
        return len(self.members) != n

    def live_total(self, pool: list) -> float:
        """总资金：没设就退回成员本金之和。"""
        if float(self.total or 0) > 0:
            return float(self.total)
        by = {str(getattr(c, "sid", "") or ""): c for c in pool}
        tot = 0.0
        for m in self.members:
            p = float(m.principal or 0)
            if p <= 0:
                c = by.get(m.sid)
                p = float(getattr(c, "principal", 0) or 0) if c is not None else 0.0
            tot += p
        return tot

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "total": float(self.total),
            "mode": self.mode, "basis": self.basis, "knob": self.knob,
            "manual": {k: float(v) for k, v in (self.manual or {}).items()},
            "members": [m.to_dict() for m in self.members],
            "account": self.account, "note": self.note, "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Workspace":
        d = dict(d or {})
        w = cls(
            id=str(d.get("id") or ""),
            name=str(d.get("name") or ""),
            total=float(d.get("total") or 0.0),
            mode=str(d.get("mode") or "equal"),
            basis=str(d.get("basis") or "principal"),
            knob=str(d.get("knob") or "split"),
            manual=dict(d.get("manual") or {}),
            account=str(d.get("account") or ""),
            note=str(d.get("note") or ""),
            created=str(d.get("created") or ""),
        )
        w.members = [Member.from_dict(x) for x in (d.get("members") or [])]
        if not w.id:
            w.id = uuid.uuid4().hex[:12]
        return w


# ---------------------------------------------------------------- 存取
def _raw() -> dict:
    try:
        d = SStore._load(FILE)          # noqa: SLF001 - 同一个包里的私有工具
    except Exception:                   # noqa: BLE001
        return {}
    return d if isinstance(d, dict) else {}


def load_workspaces(pool: list | None = None) -> list[Workspace]:
    """读工作区；文件不存在就从 `portfolio.json` 迁移一份。"""
    raw = _raw()
    items = [Workspace.from_dict(d) for d in (raw.get("items") or [])]
    items = [w for w in items if w.members]
    if items:
        return items
    return [_migrate(pool)]


def _writable() -> bool:
    """永远可写 —— 隔离靠 `strategy_store.set_store_dir()`，不靠开关。

    早先这里用一个 `CQ_NO_SAVE` 环境变量把写操作变成 no-op，用来防自检改用户
    数据。那个办法不行：自检里有「写完读回来」的用例，一刀切成 no-op 会把它们
    一起打红；而且调用点十几处，漏一个就是静默的数据事故。现在改成整目录重定向
    （自检写临时目录，读的是真实数据的快照），这里不需要再挡。
    """
    return True


def save_workspaces(items: list[Workspace], active: str = "") -> None:
    if not _writable():
        return
    for w in items:
        if not w.id:
            w.id = uuid.uuid4().hex[:12]
        if not w.created:
            w.created = time.strftime("%Y-%m-%d %H:%M")
    # 记住当前分页：下次打开还停在同一个工作区
    cur = active or active_id() or (items[0].id if items else "")
    SStore._save(FILE, {"active": cur, "items": [w.to_dict() for w in items]})  # noqa: SLF001


def active_id() -> str:
    return str(_raw().get("active") or "")


def set_active(wid: str) -> None:
    """只记「停在哪个分页」。

    切页签不该重写整个文件：`items` 里有二十几条本金，为了记一个 id 把整份
    重写一遍，写一半崩了就是配置全丢。这里只改 `active` 一个字段。
    """
    if not _writable():
        return
    raw = _raw()
    raw["active"] = str(wid or "")
    SStore._save(FILE, raw)             # noqa: SLF001


def _migrate(pool: list | None = None) -> Workspace:
    """把老的 `portfolio.json` 整体变成一个工作区。

    `principal` 取各自原值：迁移不该顺便改人的仓位。老文件留着不删 ——
    回滚到旧版本还能打开。
    """
    if pool is None:
        pool = SStore.load_portfolio()
    w = Workspace(id=uuid.uuid4().hex[:12], name=DEFAULT_NAME,
                  total=0.0, mode="equal", basis="principal",
                  created=time.strftime("%Y-%m-%d %H:%M"),
                  note="由原组合迁移")
    for c in pool:
        w.add(str(getattr(c, "sid", "") or ""), float(getattr(c, "principal", 0) or 0))
    return w


def new_workspace(name: str = "", total: float = 0.0) -> Workspace:
    return Workspace(id=uuid.uuid4().hex[:12], name=name or "新组合",
                     total=float(total), mode="equal", basis="principal",
                     created=time.strftime("%Y-%m-%d %H:%M"))


# ---------------------------------------------------------------- 解析
def resolve(ws: Workspace, pool: list) -> tuple[list, list[str]]:
    """工作区 → 可跑的配置列表（`principal` 已按工作区的分配改写）。

    返回 `(配置列表, 找不到的 sid)`。**不改原对象、不落盘** —— 分配方案要等
    用户点「应用」才写进 `workspaces.json`。
    """
    by = {str(getattr(c, "sid", "") or ""): c for c in pool}
    out, missing = [], []
    for m in ws.members:
        c = by.get(m.sid)
        if c is None:
            missing.append(m.sid)
            continue
        n = c.clone() if hasattr(c, "clone") else c
        # clone 会换 sid，工作区认的还是原 sid
        n.sid = c.sid
        if m.principal and float(m.principal) > 0:
            n.principal = float(m.principal)
        out.append(n)
    return out, missing


def allocate_workspace(ws: Workspace, pool: list,
                       usages: dict | None = None) -> list:
    """按工作区的分配方式算出每条策略该拿多少（返回 `PL.Share` 列表）。

    `usages`：`{sid: PL.Usage}`，来自回测实测。没给时 `basis` 会退回
    `principal`（没有实测占用就没法按占用校准）。
    """
    cfgs, _ = resolve(ws, pool)
    total = ws.live_total(pool)
    if not cfgs:
        return []
    us = []
    for c in cfgs:
        u = (usages or {}).get(str(c.sid))
        if u is None:
            u = PL.Usage(sid=str(c.sid), name=str(getattr(c, "title", lambda: c.sid)()),
                         sizing=str(getattr(c, "sizing", "") or "capital"),
                         principal=float(getattr(c, "principal", 0) or 0),
                         capital=float(getattr(c, "capital", 0) or 0),
                         capital_pct=float(getattr(c, "capital_pct", 0) or 0),
                         note="没有回测实测")
        us.append(u)
    basis = ws.basis if (basis_ok(ws.basis) and usages) else "principal"
    return PL.allocate(us, total, ws.mode,
                       manual=ws.manual or None, basis=basis,
                       knob=str(getattr(ws, "knob", "") or "split"))


def basis_ok(basis: str) -> bool:
    return str(basis or "").lower() in PL.BASES


def apply_shares(ws: Workspace, shares: list) -> int:
    """把分配结果写回工作区成员（只改 `Member.principal`，返回改了几条）。"""
    n = 0
    for s in shares:
        if s.new_principal <= 0:
            continue
        ws.set_principal(s.sid, s.new_principal)
        n += 1
    return n


def summarize_shares(shares: list):
    """给界面用的表。"""
    return PL.summarize(shares)
