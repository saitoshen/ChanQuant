"""因子库：查清单、看定义、试算校验。

## 为什么注册不在这里

注册（写盘 + 试算 + 塞进 `FACTORS` + 刷新界面下拉）动的是本机状态，还要弹确认框
给人看代码 —— 那件事留在界面层的 `factor` 动作里。这里只做**只读**的事：
「库里有啥」「这个因子怎么算」「换个品种还能不能算出来」。

分开的另一个理由：查询类动作可以连着调好几次（先看清单、再看一个、再试算），
而注册每轮只该做一次。
"""

from __future__ import annotations

import pandas as pd

from ttrade import blocks as B
from ttrade.skills.base import (
    RISK_READ,
    RISK_WRITE,
    SkillResult,
    Tool,
    register,
)

# 清单一次最多列这么多行。68 个内置 + 用户自建，全列出来是三百多行
MAX_LISTED = 120


def _factors():
    from ttrade.factors import FACTORS

    return FACTORS


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def _match(key: str, label: str, q: str) -> bool:
    q = _norm(q)
    return not q or q in key.lower() or q in label


# ------------------------------------------------------------------ factor_list

def _factor_list(ctx, p: dict) -> SkillResult:
    factors = _factors()
    cat = str(p.get("category") or "").strip()
    q = str(p.get("query") or "").strip()

    rows, cats = [], {}
    for k, spec in factors.items():
        cats[spec.category] = cats.get(spec.category, 0) + 1
        if cat and cat not in spec.category:
            continue
        if not _match(k, spec.label, q):
            continue
        rows.append([k, spec.label, spec.category,
                     {"1": "偏涨", "-1": "偏跌", "0": "无"}.get(str(spec.hint_dir), "无"),
                     spec.desc[:40]])

    if not rows:
        hint = f"（类别有：{'、'.join(cats)}）" if cats else ""
        raise RuntimeError(f"没找到匹配的因子{hint}")

    shown = rows[:MAX_LISTED]
    note = f"共 {len(rows)} 个" + (f"｜只列前 {MAX_LISTED} 个" if len(rows) > MAX_LISTED else "")
    blocks = [
        B.block_metrics("因子库", [
            ("总数", str(len(factors)), "flat"),
            ("类别", str(len(cats)), "flat"),
            ("本次", f"{len(shown)} 个", "flat"),
        ]),
        B.block_table(["代号", "名称", "类别", "历史倾向", "说明"], shown, note=note),
    ]
    feed = (f"因子库共 {len(factors)} 个（{len(cats)} 类）。"
            + "；".join(f"{r[0]}={r[1]}（{r[2]}）" for r in rows[:40])
            + (f"… 还有 {len(rows) - 40} 个" if len(rows) > 40 else ""))
    return SkillResult(True, summary=f"因子库共 {len(factors)} 个，本次列出 {len(shown)} 个",
                       blocks=blocks, feed=feed)


# ------------------------------------------------------------------ factor_info

def _factor_info(ctx, p: dict) -> SkillResult:
    from ttrade import factor_manual as M

    factors = _factors()
    key = _norm(p["factor"])
    spec = factors.get(key)
    if spec is None:
        near = [k for k in factors if key in k][:8]
        raise RuntimeError(f"没有名为 {key} 的因子"
                           + (f"（相近的有：{'、'.join(near)}）" if near else ""))

    blocks = [
        B.block_metrics(spec.label, [
            ("代号", spec.key, "flat"),
            ("类别", spec.category, "flat"),
            ("历史倾向", {"1": "偏涨", "-1": "偏跌", "0": "无"}.get(str(spec.hint_dir), "无"), "flat"),
        ]),
        B.block_text(spec.desc or "（没有说明）"),
    ]
    if spec.formula:
        blocks.append(B.block_text(f"**公式**：{spec.formula}"))
    if spec.code:
        blocks.append(B.block_code(spec.code, "python"))
    else:
        # 内置因子的源码在手册里，这里只取一段，别把整份手册拉进对话
        try:
            _title, body = M.resolve(f"factor:{spec.key}")
            blocks.append(B.block_code(body[:4000], "python"))
        except Exception:  # noqa: BLE001 - 手册里没有就只给说明
            pass

    feed = f"{spec.label}（{spec.key}，{spec.category}）：{spec.desc}"
    if spec.formula:
        feed += f"\n公式：{spec.formula}"
    if spec.code:
        feed += f"\n实现：\n```python\n{spec.code}\n```"
    return SkillResult(True, summary=f"{spec.label}（{spec.key}）", blocks=blocks, feed=feed)


# ------------------------------------------------------------------ factor_check

def _factor_check(ctx, p: dict) -> SkillResult:
    """在指定品种/周期上试算，报「能不能算、有多少有效值、最新值」。

    这是写新因子时最有用的一步：注册前先试算，比注册失败再看报错快一轮。
    """
    from ttrade.skills.data import _bars

    factors = _factors()
    key = _norm(p["factor"])
    spec = factors.get(key)
    if spec is None:
        raise RuntimeError(f"没有名为 {key} 的因子")

    df = _bars(ctx, p["symbol"], p.get("period", "1d"))
    try:
        series = spec.func(df)
    except Exception as exc:  # noqa: BLE001 - 试算失败就是要报给人看
        raise RuntimeError(f"试算 {key} 报错：{type(exc).__name__}: {exc}") from exc

    if series is None:
        raise RuntimeError(f"{key} 返回了 None，契约要求返回等长 Series")
    if not isinstance(series, pd.Series):
        raise RuntimeError(f"{key} 返回的是 {type(series).__name__}，契约要求 Series")
    if len(series) != len(df):
        raise RuntimeError(
            f"长度不一致：行情 {len(df)} 行，因子 {len(series)} 行。"
            "不要 dropna() / reset_index()，预热期的 NaN 要留着"
        )

    valid = series.dropna()
    n = len(valid)
    if not n:
        raise RuntimeError(
            f"{key} 在 {p['symbol']} 的 {p.get('period', '1d')} 上整列为空 —— "
            "多半是这类因子只在分钟线上有效（日内 / 时段类）"
        )

    last = float(valid.iloc[-1])
    pct = float((valid <= last).mean())
    blocks = [
        B.block_metrics(f"试算 {spec.label}", [
            ("品种", str(p["symbol"]), "flat"),
            ("周期", str(p.get("period", "1d")), "flat"),
            ("有效", f"{n} / {len(df)}", "good" if n > len(df) * 0.5 else "bad"),
            ("最新", f"{last:.4f}", "flat"),
            ("分位", f"{pct * 100:.0f}%", "flat"),
        ]),
    ]
    summary = (f"{key} 在 {p['symbol']} {p.get('period', '1d')} 上可算："
               f"{n}/{len(df)} 个有效值，最新 {last:.4f}，历史分位 {pct * 100:.0f}%")
    return SkillResult(True, summary=summary, blocks=blocks, feed=summary)


# ------------------------------------------------------------------ 注册

register(Tool(
    name="factor_list",
    group="factor",
    label="查因子库",
    description="列出因子库里的因子（可按类别或关键词过滤）",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "properties": {
            "category": {"type": "str", "default": ""},
            "query": {"type": "str", "default": ""},
        },
    },
    handler=_factor_list,
    prompt=(
        "**查因子库**：`factor_list(category=\"趋势\")` 或 `factor_list(query=\"动量\")`。"
        "用户问「有没有 XX 因子 / 库里都有什么」时用它，别凭印象报因子名。"
    ),
))

register(Tool(
    name="factor_info",
    group="factor",
    label="看因子定义",
    description="看一个因子的说明、公式和实现代码",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["factor"],
        "properties": {"factor": {"type": "str"}},
    },
    handler=_factor_info,
    prompt=(
        "**看因子定义**：`factor_info(factor=\"mom_20\")`。"
        "用户问「这个因子怎么算的」，或者你要照着改一个现有因子时，先用它看实现。"
    ),
))

register(Tool(
    name="factor_check",
    group="factor",
    label="试算因子",
    description="在某个品种/周期上试算一个因子，报有效值和最新值",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["factor", "symbol"],
        "properties": {
            "factor": {"type": "str"},
            "symbol": {"type": "str"},
            "period": {"type": "str", "default": "1d"},
        },
    },
    handler=_factor_check,
    prompt=(
        "**试算因子**：`factor_check(factor=\"mom_20\", symbol=\"RB_SHFE\")`。"
        "写新因子前先用它验证契约（等长、PIT、不留 NaN 尾巴），比注册失败再看报错快一轮。"
    ),
))


# ------------------------------------------------------------------ 注册新因子

def prepare_factor(p: dict) -> tuple[dict, str]:
    """注册因子的动作块。字段校验只做「能不能落地」，不评判因子好坏。

    真正的试算在界面层的 `_register_factor` 里 —— 那里要写盘、要 exec、要把
    报错原样回灌给模型，都不是纯解析该干的事。
    """
    key = str(p.get("key") or "").strip().lower()
    label = str(p.get("label") or "").strip()
    code = str(p.get("code") or "")
    if not key:
        return p, "缺少因子代号 key"
    if not label:
        return p, "缺少显示名 label"
    if not code.strip():
        return p, "缺少实现代码 code"

    func = str(p.get("func") or f"f_{key}").strip()
    if f"def {func}" not in code:
        return p, f"code 里要定义 def {func}(df)"

    p["key"] = key
    p["label"] = label
    p["func"] = func
    p["code"] = code
    p["category"] = str(p.get("category") or "自定义").strip()
    p["desc"] = str(p.get("desc") or "").strip()
    p["formula"] = str(p.get("formula") or "").strip()
    try:
        hint = int(p.get("hint_dir", 1))
    except (TypeError, ValueError):
        hint = 1
    p["hint_dir"] = hint if hint in (-1, 0, 1) else 1
    return p, ""


def describe_factor(p: dict) -> str:
    return f"注册因子「{p.get('label', '')}」（{p.get('category', '')}）"


register(Tool(
    name="factor",
    group="factor",
    label="注册因子",
    description="把一段 Python 写成新因子，注册进因子库。会弹确认框",
    risk=RISK_WRITE,
    inputSchema={
        "type": "object",
        "required": ["key", "label", "code"],
        "properties": {
            "key": {"type": "str"},
            "label": {"type": "str"},
            "category": {"type": "str", "default": "自定义"},
            "desc": {"type": "str", "default": ""},
            "formula": {"type": "str", "default": ""},
            "hint_dir": {"type": "int", "default": 1},
            "func": {"type": "str", "default": ""},
            "code": {"type": "str"},
        },
    },
    prepare=prepare_factor,
    ui="factor",
    describe=describe_factor,
))
