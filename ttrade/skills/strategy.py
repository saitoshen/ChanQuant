"""策略库：查清单、看参数定义。

注册策略和注册因子同理 —— 写盘 + 试算 + 刷新下拉，还要弹确认框给人看代码，
留在界面层的 `strategy` 动作里。这里只做只读查询。
"""

from __future__ import annotations

from ttrade import blocks as B
from ttrade.skills.base import (
    RISK_READ,
    RISK_WRITE,
    SkillResult,
    Tool,
    register,
)


def _strategies():
    from ttrade.strategies import STRATEGIES

    return STRATEGIES


def _param_text(spec) -> str:
    from ttrade.strategies import PARAM_LABELS

    bits = []
    for k, v in (spec.params or {}).items():
        label = PARAM_LABELS.get(k, k)
        bits.append(f"{label}={v}")
    return " ".join(bits) or "—"


def _strategy_list(ctx, p: dict) -> SkillResult:
    strategies = _strategies()
    q = str(p.get("query") or "").strip().lower()

    rows = []
    for k, spec in strategies.items():
        if q and q not in k.lower() and q not in spec.label:
            continue
        rows.append([k, spec.label, "自定义" if spec.code else "内置",
                     _param_text(spec), spec.desc[:36]])
    if not rows:
        raise RuntimeError(f"没有匹配「{q}」的策略")

    note = f"共 {len(rows)} 个"
    blocks = [
        B.block_table(["代号", "名称", "来源", "默认参数", "说明"], rows, note=note),
    ]
    feed = (f"注册的规则策略共 {len(rows)} 个：" +
            "；".join(f"{r[0]}={r[1]}（{r[3]}）" for r in rows))
    return SkillResult(True, summary=f"注册的规则策略共 {len(rows)} 个",
                       blocks=blocks, feed=feed)


def _strategy_info(ctx, p: dict) -> SkillResult:
    strategies = _strategies()
    key = str(p["strategy"] or "").strip().lower()
    spec = strategies.get(key)
    if spec is None:
        near = [k for k in strategies if key in k][:8]
        raise RuntimeError(f"没有名为 {key} 的策略"
                           + (f"（相近的有：{'、'.join(near)}）" if near else ""))

    from ttrade.strategies import PARAM_LABELS

    meta_rows = []
    for k, v in (spec.params or {}).items():
        lo, hi, step = (spec.param_meta.get(k) or ("", "", ""))[:3]
        meta_rows.append([PARAM_LABELS.get(k, k), k, str(v), f"{lo} ~ {hi}", str(step)])

    blocks = [
        B.block_metrics(spec.label, [
            ("代号", spec.key, "flat"),
            ("来源", "自定义" if spec.code else "内置", "flat"),
            ("可做空", "是" if spec.bidirectional else "否", "flat"),
            ("参数", str(len(spec.params or {})), "flat"),
        ]),
        B.block_text(spec.desc or "（没有说明）"),
    ]
    if meta_rows:
        blocks.append(B.block_table(["参数", "代号", "默认", "范围", "步长"], meta_rows))
    if spec.code:
        blocks.append(B.block_code(spec.code, "python"))

    feed = f"{spec.label}（{spec.key}）：{spec.desc}\n默认参数：{_param_text(spec)}"
    if spec.code:
        feed += f"\n实现：\n```python\n{spec.code}\n```"
    return SkillResult(True, summary=f"{spec.label}（{spec.key}）", blocks=blocks, feed=feed)


register(Tool(
    name="strategy_list",
    group="strategy",
    label="查策略库",
    description="列出注册的规则策略及其默认参数",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "properties": {"query": {"type": "str", "default": ""}},
    },
    handler=_strategy_list,
    prompt=(
        "**查策略库**：`strategy_list()`。注意这只列**注册的规则策略代号**；"
        "用户存好的整份模板（带 sid）在每次提问附的那张表里，不在这里。"
        "要按策略回测时，代号从这里抄，不要编。"
    ),
))

register(Tool(
    name="strategy_info",
    group="strategy",
    label="看策略定义",
    description="看一个策略的说明、参数范围和实现代码",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["strategy"],
        "properties": {"strategy": {"type": "str"}},
    },
    handler=_strategy_info,
    prompt=(
        "**看策略定义**：`strategy_info(strategy=\"donchian\")`。"
        "用户问「这个策略怎么做的 / 参数范围多少」，或者你要照着改一个现有策略时用它。"
    ),
))


# ------------------------------------------------------------------ 注册新策略

def prepare_strategy(p: dict) -> tuple[dict, str]:
    """注册策略的动作块。只校验「能不能落地」，不评判策略好坏 ——
    好坏由回测数字说话，不由这里拦。"""
    key = str(p.get("key") or "").strip().lower()
    label = str(p.get("label") or "").strip()
    code = str(p.get("code") or "")
    if not key:
        return p, "缺少策略代号 key"
    if not label:
        return p, "缺少显示名 label"
    if not code.strip():
        return p, "缺少实现代码 code"

    func = str(p.get("func") or f"sig_{key}").strip()
    if f"def {func}" not in code:
        return p, f"code 里要定义 def {func}(df, ...)"

    params = p.get("params") or {}
    if not isinstance(params, dict):
        return p, "params 要是一个对象"
    for k, v in params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return p, f"参数 {k} 的值 {v!r} 不是数字"

    meta = p.get("param_meta") or {}
    if not isinstance(meta, dict):
        meta = {}

    p["key"] = key
    p["label"] = label
    p["func"] = func
    p["code"] = code
    p["desc"] = str(p.get("desc") or "").strip()
    p["params"] = params
    p["param_meta"] = meta
    return p, ""


def describe_strategy(p: dict) -> str:
    return f"注册策略「{p.get('label', '')}」（{len(p.get('params') or {})} 个参数）"


register(Tool(
    name="strategy",
    group="strategy",
    label="注册策略",
    description="把一段 Python 写成新规则策略，注册进策略库。会弹确认框",
    risk=RISK_WRITE,
    inputSchema={
        "type": "object",
        "required": ["key", "label", "code"],
        "properties": {
            "key": {"type": "str"},
            "label": {"type": "str"},
            "desc": {"type": "str", "default": ""},
            "params": {"type": "dict", "default": {}},
            "param_meta": {"type": "dict", "default": {}},
            "func": {"type": "str", "default": ""},
            "code": {"type": "str"},
        },
    },
    prepare=prepare_strategy,
    ui="strategy",
    describe=describe_strategy,
))


# ---------------------------------------------------------------- 存进策略库
#
# `strategy` 那个动作注册的是**规则策略的 Python 代码**（进 `STRATEGIES` 表），
# 和「策略库」里那些带 sid 的整份配置是两件事。以前没有动作能存后者 —— 于是
# 「扫完一个板块，把结论落成条目」这一步只能靠人点界面，Agent 做到一半就断了。
#
# 这里补上。**工具自己不跑回测** —— 跑一遍可能要几十秒，塞在确认框前面等于让人
# 干等；「先跑再存」写在 `sector-factor-scan` 技能包里，由模型按顺序调用。

PERIODS_OK = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1d_tq",
              "2d", "1w", "2d_tq", "1w_tq")


def prepare_strategy_save(p: dict) -> tuple[dict, str]:
    name = str(p.get("name") or "").strip()
    if not name:
        return p, "缺少 name（策略库里显示的名字）"
    p["name"] = name

    period = str(p.get("period") or "1d_tq").strip()
    if period not in PERIODS_OK:
        return p, "period 只能是 %s 之一" % " / ".join(PERIODS_OK)
    p["period"] = period

    keys = p.get("factor_keys") or []
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        return p, "factor_keys 要是字符串数组"
    fam = str(p.get("family") or "factor").strip()
    if fam == "factor" and not keys:
        return p, "family=factor 时必须给 factor_keys"

    syms = p.get("symbols") or []
    if syms and not isinstance(syms, list):
        return p, "symbols 要是数组（写成 MA_CZCE 这种）"
    syms = [str(s).strip().upper() for s in syms if str(s).strip()]
    for s in syms:
        if "_" not in s:
            return p, "多品种要写成 品种_交易所，比如 MA_CZCE（%r 少了交易所）" % s
    p["symbols"] = syms

    if not syms:
        sym = str(p.get("symbol") or "").strip().upper()
        ex = str(p.get("exchange") or "").strip().upper()
        if not sym or not ex:
            return p, "单品种要给 symbol + exchange；多品种给 symbols"
        p["symbol"], p["exchange"] = sym, ex

    cfg = p.get("factor_cfg") or {}
    if not isinstance(cfg, dict):
        return p, "factor_cfg 要是一个对象（window / horizon / entry_z …）"
    p["factor_cfg"] = cfg
    return p, ""


def describe_strategy_save(p: dict) -> str:
    who = ("%d 个品种" % len(p.get("symbols") or [])) if p.get("symbols") else \
        "%s_%s" % (p.get("symbol", ""), p.get("exchange", ""))
    return "存进策略库「%s」（%s · %s）" % (p.get("name", ""), who, p.get("period", ""))


def _strategy_save(ctx, p: dict) -> SkillResult:
    from ttrade.strategy_store import NameConflict, StrategyConfig, add_to_library

    cfg = StrategyConfig(
        name=p["name"],
        symbol=p.get("symbol", ""), exchange=p.get("exchange", ""),
        period=p.get("period", "1d_tq"),
        multi_symbols=list(p.get("symbols") or []),
        family=p.get("family", "factor"),
        factor_mode=p.get("factor_mode", "multi"),
        factor_keys=list(p.get("factor_keys") or []),
        factor_cfg=dict(p.get("factor_cfg") or {}),
        sizing=p.get("sizing", "budget"),
        principal=float(p.get("principal") or 1_000_000),
        capital_pct=float(p.get("capital_pct") if p.get("capital_pct") is not None else 50.0),
        note=p.get("note", "short"),
        cost_profile=p.get("cost_profile", ""),
    )
    try:
        got = add_to_library(cfg, overwrite=bool(p.get("overwrite")))
    except NameConflict as exc:
        # 内置撞名 / 同名已存在 —— 原话回给模型，让它自己决定改名还是 overwrite
        return SkillResult(False, error=str(exc), feed="存不进去：%s" % exc)

    feed = ("已存进策略库：%s（sid=%s，%s · %s，%s 个因子）。\n"
            "**别直接说「策略做好了」** —— 双击跑一遍，把笔数 / 卡玛 / 回撤报给用户，"
            "数字和扫描时不一样就直说不一样。"
            % (got.name, got.sid, got.period,
               ("%d 品种" % len(got.multi_symbols)) if got.multi_symbols
               else "%s_%s" % (got.symbol, got.exchange),
               len(got.factor_keys)))
    return SkillResult(True, summary="已存入策略库：%s" % got.name, feed=feed)


register(Tool(
    name="strategy_save",
    group="strategy",
    label="存入策略库",
    description="把一份完整的因子/规则配置存进策略库（带 sid，界面左侧能搜到）",
    risk=RISK_WRITE,
    inputSchema={
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "str"},
            "symbol": {"type": "str", "default": ""},
            "exchange": {"type": "str", "default": ""},
            "symbols": {"type": "list", "default": []},
            "period": {"type": "str", "default": "1d_tq"},
            "family": {"type": "str", "default": "factor"},
            "factor_mode": {"type": "str", "default": "multi"},
            "factor_keys": {"type": "list", "default": []},
            "factor_cfg": {"type": "dict", "default": {}},
            "sizing": {"type": "str", "default": "budget"},
            "principal": {"type": "float", "default": 1000000},
            "capital_pct": {"type": "float", "default": 50.0},
            "note": {"type": "str", "default": "short"},
            "cost_profile": {"type": "str", "default": ""},
            "overwrite": {"type": "bool", "default": False},
        },
    },
    prepare=prepare_strategy_save,
    handler=_strategy_save,
    describe=describe_strategy_save,
    prompt=(
        "**存进策略库**：`strategy_save(name=\"化工·甲醇(MA) 2h 低分多｜仓单分位+展期占比\", "
        "symbol=\"MA\", exchange=\"CZCE\", period=\"2h\", factor_keys=[\"war_pct\", "
        "\"roll_main_share\"], factor_cfg={\"window\": 983, \"horizon\": 20, \"entry_z\": 1.0})`。\n"
        "和 `strategy` 的区别：`strategy` 注册的是**规则策略的 Python 代码**；"
        "`strategy_save` 存的是**整份配置**（带 sid，左侧策略库列表里能搜到、能双击跑）。"
        "给板块出一批策略时用这个。\n"
        "`symbols` 给多品种时写 `MA_CZCE` 这种带交易所的键，别只写 `MA`。"
        "同名已存在会报错，确认要覆盖再传 `overwrite=True`。"
    ),
))
