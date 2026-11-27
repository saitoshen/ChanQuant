"""可携带文件：自定义策略 `.cqs`、自定义因子 `.cqf`、工作区 `.cqw`、账户 `.cqa`。

四种文件都是带版本号的 JSON，扩展名只用来区分内容类型 —— 换一台机器导入时
**不需要联网、不需要原项目目录**，因为自定义策略 / 因子的源码就在包里。

`.cqa` 是账户列表（含明文密码），只做同名合并，不碰本机账户定时。

## 为什么工作区要连策略一起打包

工作区本身只存 `sid` + 本金（`workspace_store` 的模块注释）。单独拷一份
`workspaces.json` 到另一台机器，那台机器没有 `portfolio.json` 里对应的策略，
打开就是一条条 `missing`。所以 `.cqw` 里放的是**闭包**：

    工作区 → 成员 sid 对应的策略配置 → 配置引用的自定义规则策略 / 自定义因子

导入顺序固定为 因子 → 规则策略 → 库模板 → 策略池 → 工作区，反向就会在
解析配置时找不到依赖。策略池**保留原 sid**写回 —— `add_to_portfolio()` 会
clone 出新 sid，用它导入等于让工作区所有成员指空。

## 冲突口径

- 内置因子 / 内置规则策略 / 内置模板的名字一律不覆盖：包里同名条目跳过并写明原因。
- 自定义条目同 key / 同 sid 默认覆盖（这是「导入新版本」的正常语义）；
  `overwrite=False` 时跳过。
- 工作区同 id 覆盖，不同 id 追加。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from ttrade import factors as F
from ttrade import strategies as S
from ttrade import strategy_store as SStore
from ttrade import userfactors as UF
from ttrade import userstrategies as US
from ttrade import workspace_store as WS

FMT_STRATEGY = "chanquant-strategies"
FMT_FACTOR = "chanquant-factors"
FMT_WORKSPACE = "chanquant-workspace"
FMT_ACCOUNT = "chanquant-accounts"
VERSION = 1

LABELS = {
    FMT_STRATEGY: "策略包（.cqs）",
    FMT_FACTOR: "因子包（.cqf）",
    FMT_WORKSPACE: "工作区包（.cqw）",
    FMT_ACCOUNT: "账户包（.cqa）",
}


class BadPackage(ValueError):
    """文件不是本平台的可携带包，或者结构坏了。"""


@dataclass
class Report:
    """导入结果。UI 直接 `summary()`，自检逐项看 `skipped` / `errors`。"""

    ok: int = 0
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def skip(self, name: str, why: str) -> None:
        self.skipped.append(f"{name}：{why}")

    def error(self, name: str, why: str) -> None:
        self.errors.append(f"{name}：{why}")

    def summary(self) -> str:
        bits = [f"成功 {self.ok} 条"]
        if self.skipped:
            bits.append(f"跳过 {len(self.skipped)} 条")
        if self.errors:
            bits.append(f"失败 {len(self.errors)} 条")
        detail = []
        for s in self.skipped[:4]:
            detail.append(f"跳过 {s}")
        for s in self.errors[:4]:
            detail.append(f"失败 {s}")
        more = len(self.skipped) + len(self.errors) - 8
        if more > 0:
            detail.append(f"……另有 {more} 条")
        return "，".join(bits) + ("\n" + "\n".join(detail) if detail else "")


# ---------------------------------------------------------------- 文件读写

def _wrap(fmt: str, **payload) -> dict:
    return {
        "format": fmt,
        "version": VERSION,
        "app": "ChanQuant",
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        **payload,
    }


def _write(path: str, payload: dict) -> None:
    p = os.path.abspath(str(path or ""))
    if not p:
        raise BadPackage("没有指定保存路径")
    parent = os.path.dirname(p)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def _read(path: str, fmt: str) -> dict:
    p = os.path.abspath(str(path or ""))
    if not p or not os.path.exists(p):
        raise BadPackage(f"文件不存在：{p}")
    # 一份正常包最多几百 KB；几十 MB 的手改文件读进来只会拖死界面
    if os.path.getsize(p) > 50 * 1024 * 1024:
        raise BadPackage("文件超过 50 MB，不像可携带包")
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:
        raise BadPackage(f"读不了这个 JSON：{exc}") from exc
    if not isinstance(d, dict) or d.get("format") != fmt:
        raise BadPackage(f"不是{LABELS.get(fmt, '可携带')}（format 不对）")
    if int(d.get("version") or 0) != VERSION:
        raise BadPackage(f"包版本 {d.get('version')} 不支持，本程序认 v{VERSION}")
    return d


def _items(raw, key: str) -> list[dict]:
    """只收 dict 条目。一条坏的不该让整包导入失败。"""
    out = raw.get(key) or []
    if not isinstance(out, list):
        return []
    return [x for x in out if isinstance(x, dict)]


# ---------------------------------------------------------------- 策略 .cqs

def strategy_items() -> list[tuple[str, object]]:
    """策略库里所有「用户自己的」可导出条目，供导出选择框列出来。

    `rule` 是自定义规则策略（AI 写的代码），`cfg` 是「我的策略库」里的配置。
    内置模板 / 内置规则策略不在里面 —— 它们写在源码里，导出一份反而让别的
    机器多出一份永远删不掉的假内置。
    """
    out: list[tuple[str, object]] = [("rule", s) for s in US.load_all()]
    out.extend(("cfg", c) for c in SStore.load_library())
    return out


def item_label(kind: str, item) -> str:
    if kind == "rule":
        return f"规则策略 · {getattr(item, 'label', '')}（{getattr(item, 'key', '')}）"
    if kind == "cfg":
        title = item.title() if hasattr(item, "title") else str(getattr(item, "name", ""))
        return f"策略库配置 · {title}"
    return str(item)


def export_strategies(items: list[tuple[str, object]], path: str) -> int:
    """导出选中的自定义策略。返回写进去的策略条数（不含随包依赖）。

    策略库配置可能引用自定义因子。只导配置不导因子，换台机器打开就是一条
    「策略在、因子没了」的空壳 —— 所以把引用的自定义因子一并塞进包里。
    规则策略的代码自带，不需要额外依赖。
    """
    rules, cfgs = [], []
    for kind, obj in items or []:
        if kind == "rule":
            rules.append(obj.to_dict())
        elif kind == "cfg":
            cfgs.append(obj.to_dict())
        else:
            raise BadPackage(f"不认识的策略类型：{kind}")

    fkeys: set[str] = set()
    for raw in cfgs:
        try:
            cfg = SStore.StrategyConfig.from_dict(raw)
        except Exception:  # noqa: BLE001 - 坏条目由导入端报错，这里只管尽力带依赖
            continue
        _r, got = _refs_of(cfg)
        fkeys |= got
    have = {f.key: f for f in UF.load_all()}
    factors = [have[k].to_dict() for k in sorted(fkeys) if k in have]

    _write(path, _wrap(FMT_STRATEGY, rules=rules, library=cfgs, factors=factors))
    return len(rules) + len(cfgs)


def _import_rules(raws: list[dict], rep: Report, overwrite: bool) -> None:
    user_keys = {s.key for s in US.load_all()}
    builtins = {k for k in S.STRATEGIES if k not in user_keys}
    for raw in raws:
        try:
            s = US.UserStrategy.from_dict(raw)
        except Exception as exc:  # noqa: BLE001
            rep.error("规则策略（坏条目）", str(exc))
            continue
        name = s.label or s.key or "未命名策略"
        if not s.key:
            rep.error(name, "没有代号")
            continue
        if s.key in builtins:
            rep.skip(name, "代号与内置策略冲突，不覆盖内置")
            continue
        if s.key in user_keys and not overwrite:
            rep.skip(name, "本地已有同代号，本次选择不覆盖")
            continue
        ok, msg = US.register(s, allow_overwrite=True)
        if ok:
            rep.ok += 1
        else:
            rep.error(name, msg)


def _import_library(raws: list[dict], rep: Report, overwrite: bool) -> None:
    have = {c.sid: c for c in SStore.load_library()}
    builtin_names = SStore.builtin_names()
    for raw in raws:
        try:
            cfg = SStore.StrategyConfig.from_dict(raw)
        except Exception as exc:  # noqa: BLE001
            rep.error("策略库配置（坏条目）", str(exc))
            continue
        name = cfg.title()
        old = have.get(cfg.sid)
        if old is not None and not overwrite:
            rep.skip(name, "本地已有同 sid，本次选择不覆盖")
            continue
        # 内置模板名不许重：界面把两组放一起，同名之后谁也指代不清
        if (old is None and SStore.norm_name(name) in builtin_names
                and not SStore.is_template(cfg.sid)):
            rep.skip(name, "名字与内置模板冲突，换个名字再导")
            continue
        try:
            SStore.add_to_library(cfg, overwrite=True)
        except (SStore.NameConflict, ValueError) as exc:
            rep.error(name, str(exc))
            continue
        have[cfg.sid] = cfg
        rep.ok += 1


def import_strategies(path: str, *, overwrite: bool = True) -> Report:
    d = _read(path, FMT_STRATEGY)
    rep = Report()
    # 因子先落地：配置里的 factor_keys 指向它们，顺序反了配置就是空壳
    _import_factors(_items(d, "factors"), rep, overwrite)
    _import_rules(_items(d, "rules"), rep, overwrite)
    _import_library(_items(d, "library"), rep, overwrite)
    return rep


# ---------------------------------------------------------------- 因子 .cqf

def factor_items() -> list:
    return UF.load_all()


def export_factors(items: list, path: str) -> int:
    rows = [i.to_dict() for i in (items or [])]
    _write(path, _wrap(FMT_FACTOR, factors=rows))
    return len(rows)


def _import_factors(raws: list[dict], rep: Report, overwrite: bool) -> None:
    have = {f.key for f in UF.load_all()}
    builtins = {k for k in F.FACTORS if k not in have}
    for raw in raws:
        try:
            f = UF.UserFactor.from_dict(raw)
        except Exception as exc:  # noqa: BLE001
            rep.error("因子（坏条目）", str(exc))
            continue
        name = f.label or f.key or "未命名因子"
        if not f.key:
            rep.error(name, "没有代号")
            continue
        if f.key in builtins:
            rep.skip(name, "代号与内置因子冲突，不覆盖内置")
            continue
        if f.key in have and not overwrite:
            rep.skip(name, "本地已有同代号，本次选择不覆盖")
            continue
        ok, msg = UF.register(f, allow_overwrite=True)
        if ok:
            have.add(f.key)
            rep.ok += 1
        else:
            rep.error(name, msg)


def import_factors(path: str, *, overwrite: bool = True) -> Report:
    d = _read(path, FMT_FACTOR)
    rep = Report()
    _import_factors(_items(d, "factors"), rep, overwrite)
    return rep


# ---------------------------------------------------------------- 工作区 .cqw

def _refs_of(cfg) -> tuple[set[str], set[str]]:
    """配置引用的自定义规则策略 key / 自定义因子 key。"""
    rules, factors = set(), set()
    for k in (getattr(cfg, "rule_key", ""), getattr(cfg, "timing_rule_key", "")):
        if k:
            rules.add(str(k))
    for key in ("factor_keys", "cross_pool_keys", "timing_factor_keys"):
        for f in (getattr(cfg, key, None) or []):
            if isinstance(f, str) and f:
                factors.add(f)
            elif isinstance(f, (list, tuple)) and f and isinstance(f[0], str):
                factors.add(f[0])
    return rules, factors


def export_workspaces(workspaces: list, path: str) -> tuple[int, list[str]]:
    """导出工作区及其依赖。返回 `(工作区数, 找不到配置的 sid)`。"""
    sids = set()
    for w in (workspaces or []):
        sids.update(w.sids())
    pool = SStore.load_portfolio()
    lib = SStore.load_library()
    chosen_pool = [c for c in pool if c.sid in sids]
    chosen_lib = [c for c in lib if c.sid in sids]
    missing = sorted(s for s in sids if not any(c.sid == s for c in chosen_pool + chosen_lib))

    rule_keys: set[str] = set()
    factor_keys: set[str] = set()
    for c in chosen_pool + chosen_lib:
        r, f = _refs_of(c)
        rule_keys |= r
        factor_keys |= f
    user_rules = {s.key: s for s in US.load_all()}
    user_factors = {f.key: f for f in UF.load_all()}

    _write(path, _wrap(
        FMT_WORKSPACE,
        workspaces=[w.to_dict() for w in (workspaces or [])],
        portfolio=[c.to_dict() for c in chosen_pool],
        library=[c.to_dict() for c in chosen_lib],
        rules=[user_rules[k].to_dict() for k in sorted(rule_keys) if k in user_rules],
        factors=[user_factors[k].to_dict() for k in sorted(factor_keys) if k in user_factors],
        missing=missing,
    ))
    return len(workspaces or []), missing


def _import_configs(raws: list[dict], rep: Report, overwrite: bool, *, kind: str) -> None:
    """按 sid 合并进策略池 / 策略库。**sid 原样保留**，否则工作区全部指空。"""
    if kind == "portfolio":
        have = {c.sid: c for c in SStore.load_portfolio()}
    else:
        have = {c.sid: c for c in SStore.load_library()}
    changed = False
    for raw in raws:
        try:
            cfg = SStore.StrategyConfig.from_dict(raw)
        except Exception as exc:  # noqa: BLE001
            rep.error("策略配置（坏条目）", str(exc))
            continue
        if not cfg.sid:
            rep.error("策略配置", "没有 sid")
            continue
        old = have.get(cfg.sid)
        if old is not None and not overwrite:
            rep.skip(cfg.title(), "本地已有同 sid，本次选择不覆盖")
            continue
        have[cfg.sid] = cfg
        changed = True
        rep.ok += 1
    if changed:
        items = list(have.values())
        if kind == "portfolio":
            SStore.save_portfolio(items)
        else:
            SStore.save_library(items)


def _import_workspaces(raws: list[dict], rep: Report, overwrite: bool) -> None:
    # 不用 `load_workspaces()`：文件为空时它会从 portfolio 迁移出一份「默认组合」，
    # 导入完就平白多出一个用户没导过的分页。
    raw = WS._raw()  # noqa: SLF001 - 同一个包里的存储工具
    cur = [WS.Workspace.from_dict(x) for x in _items(raw, "items")]
    cur = [w for w in cur if w.members]
    by_id = {w.id: i for i, w in enumerate(cur)}
    active = str(raw.get("active") or "")
    changed = False
    for x in raws:
        try:
            w = WS.Workspace.from_dict(x)
        except Exception as exc:  # noqa: BLE001
            rep.error("工作区（坏条目）", str(exc))
            continue
        if not w.members:
            rep.skip(w.name or "未命名", "没有成员，跳过")
            continue
        old = by_id.get(w.id)
        if old is not None:
            if not overwrite:
                rep.skip(w.name, "本地已有同 id，本次选择不覆盖")
                continue
            cur[old] = w
        else:
            by_id[w.id] = len(cur)
            cur.append(w)
        active = w.id
        changed = True
        rep.ok += 1
    if changed:
        WS.save_workspaces(cur, active)


def import_workspaces(path: str, *, overwrite: bool = True) -> Report:
    d = _read(path, FMT_WORKSPACE)
    rep = Report()
    missing = [str(x) for x in (d.get("missing") or []) if isinstance(x, str)]
    if missing:
        rep.skipped.append(f"包里原本就找不到配置的 sid：{', '.join(missing[:6])}"
                           + (f" 等 {len(missing)} 条" if len(missing) > 6 else ""))
    # 顺序是有依赖的：因子 / 规则策略先落地，配置里的引用才对得上
    _import_factors(_items(d, "factors"), rep, overwrite)
    _import_rules(_items(d, "rules"), rep, overwrite)
    _import_configs(_items(d, "library"), rep, overwrite, kind="library")
    _import_configs(_items(d, "portfolio"), rep, overwrite, kind="portfolio")
    _import_workspaces(_items(d, "workspaces"), rep, overwrite)
    return rep


# ---------------------------------------------------------------- 账户 .cqa


def export_accounts(items: list, path: str) -> int:
    """把账户列表存成 `.cqa`（按账户名去重，空名不导）。

    字段和 `%LOCALAPPDATA%\\ChanQuant\\accounts.json` 里的 `AccountSpec` 完全同构：
    少一个字段（比如 CTP 的 `td_front`）换台机器就连不上，所以整条 asdict 原样
    带走。**密码是明文的**，界面在保存前会先提醒，这里不替用户做「脱敏」——
    脱敏后的包导入进去连不上，用户会以为功能坏了。
    """
    from dataclasses import asdict

    from ttrade import tqlive as TQ

    fields = set(TQ.AccountSpec.__dataclass_fields__)
    rows: list[dict] = []
    seen: set[str] = set()
    for spec in items or []:
        raw = asdict(spec) if hasattr(spec, "__dataclass_fields__") else dict(spec)
        one = {k: str(raw.get(k) or "") for k in fields}
        name = one.get("name", "").strip()
        if not name or name in seen:
            continue
        one["name"] = name
        seen.add(name)
        rows.append(one)
    _write(path, _wrap(FMT_ACCOUNT, accounts=rows))
    return len(rows)


def import_accounts(path: str, *, overwrite: bool = True) -> Report:
    """导入 `.cqa`：按账户名合并，同名覆盖，新名追加在后面。

    只动账户列表，不碰账户定时（`account_timer.json`）—— 定时是本机排班，
    拷到另一台机器上按点自动登出别人的账户只会添乱。
    """
    from ttrade import tqlive as TQ

    d = _read(path, FMT_ACCOUNT)
    rep = Report()
    current = TQ.load_accounts()
    order = [s.name for s in current]
    by_name = {s.name: s for s in current}
    for raw in _items(d, "accounts"):
        try:
            spec = TQ.AccountSpec(**{
                k: str(raw.get(k) or "") for k in TQ.AccountSpec.__dataclass_fields__
            })
        except Exception as exc:  # noqa: BLE001 - 一条坏的不该挡住整包
            rep.error("账户（坏条目）", str(exc))
            continue
        spec.name = spec.name.strip()
        if not spec.name:
            rep.error("未命名账户", "没有名称")
            continue
        if spec.name in by_name and not overwrite:
            rep.skip(spec.name, "本地已有同名账户，本次选择不覆盖")
            continue
        if spec.name in by_name:
            by_name[spec.name] = spec
        else:
            order.append(spec.name)
            by_name[spec.name] = spec
        rep.ok += 1
    if rep.ok:
        TQ.save_accounts([by_name[n] for n in order])
    return rep
