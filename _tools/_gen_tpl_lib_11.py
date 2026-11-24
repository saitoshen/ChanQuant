"""把 11 条策略库配置生成为 `strategy_templates_lib.py` 的 `RAW` 条目。

**只读**策略库，生成文本到 `_tools/_gen_11.txt`，不直接改源码。

生成规则（与现有 94 条一致）：
- 只写与 `StrategyConfig` 默认值不同的字段，另加一组「总是写」的字段
  （`_ALWAYS`：这些是模板的骨架，缺了读起来要跳来跳去）
- 丢掉 `sid` / `created` / `enabled` / `account` / `accounts` / `note`
- `short_only` 原样保留（「基本面空」那条**本来就是只做空**，不是手滑）
- 生成后做**回放等价**校验：把生成的条目按 `to_config()` 的方式还原成
  `StrategyConfig`，逐字段比对原始配置
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttrade import strategy_store as S
from ttrade.strategy_templates import _SCFG_DEFAULTS

# (库里名字, 模板键, 分组)
PICKS = [
    ("NR-1h反转+收盘位置", "tpl_lib_factor_70", "mine_factor"),
    ("L日线", "tpl_lib_factor_71", "mine_factor"),
    ("ag短周期策略5m", "tpl_lib_factor_72", "mine_factor"),
    ("ru短周期策略5m", "tpl_lib_factor_73", "mine_factor"),
    ("cu短周期策略5m", "tpl_lib_factor_74", "mine_factor"),
    ("sc短周期策略5m", "tpl_lib_factor_75", "mine_factor"),
    ("sbbox多品种波段1h", "tpl_lib_rule_25", "mine_rule"),
    ("sbbox多品种波段1h 基本面空", "tpl_lib_rule_26", "mine_rule"),
    ("cu美元人民币展期动量 30m", "tpl_lib_factor_76", "mine_factor"),
    ("HL-point截面对冲", "tpl_lib_cross_02", "mine_cross"),
    ("HL-Point · 商品十分位多空（1h）", "tpl_lib_cross_03", "mine_cross"),
]

# 这些字段无论是否等于默认值都写出来：模板读完要能一眼看出「跑什么、什么家族、
# 因子怎么配」，靠 diff 拼出来的字典得回头翻 `_SCFG_DEFAULTS` 才知道缺了啥
_ALWAYS = (
    "cross_pool_keys", "cross_pool_reverse", "cross_symbols", "cross_tier_units",
    "exchange", "factor_cfg", "factor_keys", "factor_reverse", "multi_symbols",
    "symbol", "rule_key", "rule_params",
    "timing_factor_cfg", "timing_factor_keys", "timing_factor_reverse",
    "timing_rule_params",
)

# 运行时字段，不进模板
_DROP = ("sid", "created", "enabled", "account", "accounts", "note")

_BASE = S.StrategyConfig().to_dict()


def _num(x) -> str:
    """1.0 → '1.0'、1.2 → '1.2'、70 → '70'：小数量别被 `:g` 去掉小数位，
    现有 94 条的 teach 全是 `入场 z 1.0` 这个写法。"""
    s = f"{float(x):g}"
    return s + ".0" if ("." not in s and "e" not in s) else s


def _symbol_exchange(d: dict) -> str:
    s, e = str(d.get("symbol") or ""), str(d.get("exchange") or "")
    return f"{s}_{e}" if s and e else ""


def _teach(name: str, d: dict) -> str:
    """一行配方摘要，照现有 94 条的写法生成。"""
    fam = d.get("family") or "factor"
    if fam == "rule":
        from ttrade.strategies import STRATEGIES

        key = d.get("rule_key") or ""
        spec = STRATEGIES.get(key)
        label = spec.label if spec else key
        params = d.get("rule_params") or {}
        ps = ", ".join(f"{k}={v}" for k, v in params.items())
        head = f"规则策略：{label}（{ps}）" if ps else f"规则策略：{label}"
        if len(d.get("symbols") or d.get("multi_symbols") or []) > 1:
            head += f" · {len(d.get('multi_symbols') or [])} 个品种"
        if d.get("short_only"):
            head += " · 只做空"
        return head

    fc = dict(d.get("factor_cfg") or {})
    n = len(d.get("factor_keys") or [])
    if fam == "cross":
        reb = d.get("cross_rebalance") or 5
        unit = {"bar": "根", "day": "天", "month": "月"}.get(
            d.get("cross_rebalance_unit") or "bar", "根")
        bits = [f"截面多空 · {d.get('cross_mode') or 'hedge'}",
                f"{len(d.get('cross_symbols') or [])} 个品种",
                f"{reb}{unit}调仓",
                f"多 {d.get('cross_top_pct', 10):g}%/空 {d.get('cross_bot_pct', 10):g}%"]
        w = d.get("cross_weight") or "risk"
        wcn = {"equal": "等权", "score": "按分数", "risk": "风险平价", "custom": "自定义"}.get(w, w)
        scope = d.get("cross_weight_scope") or "pooled"
        bits.append(wcn + ("（腿内各自）" if scope == "legs" else "（多空合并）"))
        if (d.get("cross_label_mode") or "volret") != "volret":
            bits.append("标签 " + str(d.get("cross_label_mode")))
        if (d.get("cross_step1_period") or ""):
            bits.append("筛池周期 " + str(d.get("cross_step1_period")))
        if (d.get("cross_pool_mode") or "off") != "off":
            bits.append(f"筛池 {len(d.get('cross_pool_keys') or [])} 因子"
                        f"·前 {d.get('cross_pool_pct', 60):g}%")
        txt = " · ".join(bits)
    else:
        mode = {"single": "single", "multi": "multi", "ml": "ml"}.get(
            d.get("factor_mode") or "", "multi")
        txt = (f"{mode} 因子 · {n} 个 · 窗口 {fc.get('window', 252)}"
               f" · 入场 z {_num(fc.get('entry_z', 1.0))}"
               f" · {fc.get('orient') or 'auto'}")
        rev = [k for k, v in (fc.get("reverse") or {}).items() if v]
        if rev:
            txt += f" · {len(rev)} 个取反"

    # 🔴 品种池和名字对不上时必须写出来：库里有 5 条因子策略的 `multi_symbols`
    # 只有**一个**成员、且和 `symbol` 不是同一个品种（克隆上一条配置留下的），
    # 而 `StrategyConfig.symbols` 优先认 `multi_symbols` ⇒ 名字叫 ag、实跑 EG。
    # 收编原样保留（改了就不是「他跑过的那条」），但摘要里得说清楚。
    sy = [s for s in (d.get("multi_symbols") or []) if s]
    if len(sy) == 1 and sy[0] != _symbol_exchange(d):
        txt += f" · ⚠ 实跑 {sy[0]}（与名字不符）"
    if d.get("short_only"):
        txt += " · 只做空"
    return txt


def build(name: str, key: str, group: str) -> dict:
    c = S.load_library_by_name(name) if hasattr(S, "load_library_by_name") else None
    if c is None:
        idx = {x.name: x for x in S.load_library()}
        c = idx[name]
    d = c.to_dict()
    d["symbols"] = c.symbols          # 只给 `_teach` 用，不落进模板
    out: dict = {"key": key, "name": name, "group": group, "teach": _teach(name, d)}

    for k, v in d.items():
        if k in _DROP or k in ("name", "symbols"):
            continue
        if k in _ALWAYS or v != _BASE.get(k):
            out[k] = v
    return out


def render(t: dict) -> str:
    head = (f"    dict(key={t['key']!r}, name={t['name']!r}, "
            f"group={t['group']!r}, teach={t['teach']!r},")
    body = [f"         {k}={t[k]!r}," for k in sorted(t) if k not in
            ("key", "name", "group", "teach")]
    return head + "\n" + "\n".join(body) + "),\n"


def _norm_fcfg(x) -> dict:
    """把一份可能很稀疏的 `factor_cfg` 字典补成 `FactorConfig` 的完整字段，
    再逐项比。稀疏字典补齐后的值和「默认值」是一回事，直接比字典是假红。"""
    import dataclasses

    from ttrade.factor_strategy import FactorConfig

    full = dataclasses.asdict(FactorConfig())
    full.update(x or {})
    full["reverse"] = {k: True for k, v in (full.get("reverse") or {}).items() if v}
    return full


def verify(t: dict, src) -> list:
    """按 `to_config()` 的路子还原一遍，逐字段和原始配置比对。

    `factor_cfg` 要按**解析后的对象**比，不能比字典：模板里存的是稀疏字典
    （规则策略甚至就是 `{}`），`to_config()` 会拿 `_FCFG_DEFAULTS` 补齐，
    补出来的字段值和 `FactorConfig()` 的默认值一模一样 —— 比字典必然报假红。
    """
    d = {k: v for k, v in t.items()
         if k not in ("key", "name", "group", "teach")}
    d["sid"] = t["key"]
    d["name"] = t["name"]
    for k, v in _SCFG_DEFAULTS.items():
        d.setdefault(k, v)
    if (t.get("family") or "factor") != "rule":
        from ttrade.strategy_templates import _FCFG_DEFAULTS

        fcfg = dict(_FCFG_DEFAULTS)
        fcfg.update(t.get("factor_cfg") or {})
        fcfg["reverse"] = {k: True for k, v in (fcfg.get("reverse") or {}).items() if v}
        d["factor_cfg"] = fcfg
    got = S.StrategyConfig.from_dict(d)
    bad = []
    for k in sorted(set(vars(got)) | set(vars(src))):
        if k in ("sid", "name", "created", "enabled", "note", "short_only",
                 "account", "accounts"):
            continue
        a, b = getattr(got, k, None), getattr(src, k, None)
        if k in ("factor_cfg", "timing_factor_cfg"):
            a, b = _norm_fcfg(a), _norm_fcfg(b)
        if a != b:
            bad.append(f"{k}: 还原={a!r} 原={b!r}")
    return bad


def main() -> int:
    lib = {c.name: c for c in S.load_library()}
    blocks, fails = [], []
    for name, key, group in PICKS:
        if name not in lib:
            fails.append(f"库里找不到 {name}")
            continue
        src = lib[name]
        t = build(name, key, group)
        bad = verify(t, src)
        print(f"[{'OK ' if not bad else 'BAD'}] {key:18s} {name:32s} "
              f"字段 {len(t) - 4:2d}  长度 {len(render(t))}")
        for b in bad:
            print("      ! " + b)
        if bad:
            fails.append(f"{name}: {len(bad)} 处不等")
        blocks.append(render(t))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_gen_11.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(blocks))
    print(f"\n生成 {len(blocks)} 条，失败 {len(fails)}")
    for f in fails:
        print("  FAIL " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
