"""期权因子研判：把 T 型报价背后的信息翻译成研报能用的结论。

## 为什么不直接给 T 型报价

T 型报价是给交易员看具体档位价格的，信息密度对研报来说太低——研报需要的不是
「行权价 4000 的认购收 52.5」，而是「市场在这个品种上押注的方向和波动率预期」。

期权库里已经算好了**按品种聚合的 IV 摘要**（`option_iv` 表），每品种每天一行：
平值 IV（近月 / 次近月）、偏度、曲率、看跌看涨持仓比、参与拟合的报价点数。
这五个数字背后是市场对标的的看法，本模块把它们翻译成研报能直接写进去的文字。

## 五个因子各代表什么

| 因子 | 含义 | 研报怎么用 |
|------|------|-----------|
| iv_near | 近月平值隐含波动率 | 波动率水平：市场预期标的未来波动多大 |
| iv_next | 次近月平值 IV | 期限结构：近贵还是远贵 → 近期事件 vs 长期不确定性 |
| skew | IV 微笑在平值处的斜率 | 方向倾向：负偏度 = 虚值 put 更贵 = 市场在防跌 |
| pcr | 看跌持仓量 / 看涨持仓量 | 方向倾向：>1 偏空对冲，<1 偏多布局 |
| curv | IV 微笑在平值处的二阶导 | 尾部风险：曲率大 = 两翼定价偏离平值多 |

## 口径

所有数字来自 `option_iv` 表，口径在 `ttrade/options.py` 头注释里写明：
Black-76 反解、虚值、结算价、二次微笑拟合、±20% moneyness、15~180 天。
研报引用时写「近月平值 IV XX%（Black-76 反解，虚值，±20% moneyness 拟合）」即可。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ttrade import blocks as B
from ttrade import optstore as OS
from ttrade.skills.base import (
    RISK_READ,
    SkillResult,
    Tool,
    register,
)

# IV 分位计算的最低样本量。期权 2019 年才有数据（天勤口径），不到 100 个
# 观测值算分位没意义
_MIN_IV_OBS = 60

# PCR / 偏度的判定阈值。这些不是拍脑袋——是基于实盘数据的分布特征：
# - 商品期权 PCR 中位数约 1.0，>1.3 才算明显偏空（铜 / 铝对冲盘常年 >1）
# - 股指期权 PCR 常年 <1（卖出认沽是主流策略），<0.6 才算明显偏多
# - 偏度绝对值 >0.15 在商品期权里已经是不弱的信号（分布标准差约 0.12）
_PCR_HIGH = 1.3       # 偏空对冲
_PCR_LOW = 0.7        # 偏多布局
_SKEW_STRONG = 0.15   # 方向信号明显
_SKEW_MODERATE = 0.08 # 有一定方向信号


def _split_symbol(symbol: str) -> str:
    """`RB_SHFE` → `RB`，`rb` → `rb`。只取品种代码，交易所后缀丢掉。"""
    s = str(symbol or "").strip()
    sym, _, ex = s.rpartition("_")
    if ex and ex.isalpha() and ex.isupper():
        return sym
    return s


def _match_variety(symbol: str, varieties: list[str]) -> str | None:
    """大小写不敏感匹配品种代码 → 期权库里的写法。

    库里商品期权是小写（`rb` / `m` / `cu`），股指期权是大写（`IO` / `MO` / `HO`），
    研报代号是 `RB_SHFE` / `M_DCE`。按 lower() 匹配就能对上。
    """
    sym_lower = symbol.lower()
    for v in varieties:
        if str(v).lower() == sym_lower:
            return str(v)
    return None


def _iv_percentile(series: pd.Series, current: float) -> float:
    """当前 IV 在自身历史上的分位（0~1）。"""
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) < _MIN_IV_OBS:
        return float("nan")
    return float((valid <= current).mean())


def _interpret_skew(skew: float) -> tuple[str, str]:
    """偏度 → (方向标签, 一句话解释)。

    偏度 = IV 微笑在平值处的一阶导 dIV/dlnK。
    - 负值：虚值 put 比 call 贵 → 市场在防跌 → 偏空
    - 正值：虚值 call 比 put 贵 → 市场在追涨 → 偏多
    - 绝对值越大信号越强
    """
    abs_s = abs(skew)
    if abs_s < _SKEW_MODERATE:
        return "中性", "偏度接近零，微笑曲线基本对称，市场无明显的方向押注"
    if skew < -_SKEW_STRONG:
        return "偏空", f"偏度 {skew:+.2f}，虚值 put 显著贵于 call，市场在为下跌付费"
    if skew < -_SKEW_MODERATE:
        return "略偏空", f"偏度 {skew:+.2f}，虚值 put 略贵，有一定防跌情绪"
    if skew > _SKEW_STRONG:
        return "偏多", f"偏度 {skew:+.2f}，虚值 call 显著贵于 put，市场在为上涨付费"
    return "略偏多", f"偏度 {skew:+.2f}，虚值 call 略贵，有一定追涨情绪"


def _interpret_pcr(pcr: float) -> tuple[str, str]:
    """PCR → (方向标签, 一句话解释)。

    PCR = put 持仓量 / call 持仓量。
    - >1：put 持仓多于 call → 偏空对冲为主
    - <1：call 持仓多于 put → 偏多布局为主
    - 注意：股指期权卖出认沽是主流收入策略，PCR 常年 <1，不一定代表偏多
    """
    if not np.isfinite(pcr):
        return "—", "持仓量数据缺失"
    if pcr > _PCR_HIGH:
        return "偏空", f"PCR {pcr:.2f}，put 持仓显著高于 call，对冲盘主导"
    if pcr > 1.0:
        return "略偏空", f"PCR {pcr:.2f}，put 持仓略多，有一定对冲需求"
    if pcr < _PCR_LOW:
        return "偏多", f"PCR {pcr:.2f}，call 持仓显著高于 put，多头布局为主（注意：股指期权卖认沽是常见策略，PCR 偏低不一定代表看多）"
    if pcr < 1.0:
        return "略偏多", f"PCR {pcr:.2f}，call 持仓略多"
    return "中性", f"PCR {pcr:.2f}，多空持仓基本均衡"


def _interpret_term(iv_near: float, iv_next: float,
                    dte_near: int, dte_next: int) -> tuple[str, str]:
    """期限结构 → (标签, 解释)。

    近月 IV > 远月 IV = 近期不确定性更高（事件驱动、临近到期效应）
    近月 IV < 远月 IV = 长期不确定性更高（结构性风险溢价）
    """
    if not (np.isfinite(iv_near) and np.isfinite(iv_next)):
        return "—", "期限结构数据不完整"
    spread = iv_near - iv_next
    # 用 spread / iv_near 做归一化，10% 以上才算明显
    ratio = abs(spread) / iv_near if iv_near > 0 else 0
    if ratio < 0.05:
        return "平坦", f"近月 IV {iv_near:.1%} 与远月 {iv_next:.1%} 接近，期限结构平坦"
    if spread > 0:
        return "近高远低", (f"近月（{dte_near} 天）IV {iv_near:.1%} 高于远月"
                          f"（{dte_next} 天）{iv_next:.1%}，近期事件或临近到期推升波动率")
    return "近低远高", (f"近月（{dte_near} 天）IV {iv_near:.1%} 低于远月"
                      f"（{dte_next} 天）{iv_next:.1%}，长期风险溢价更高")


def _interpret_iv_level(iv: float, pct: float) -> tuple[str, str]:
    """IV 水平 → (标签, 解释)。"""
    if not np.isfinite(iv):
        return "—", "IV 数据缺失"
    parts = [f"近月平值 IV {iv:.1%}"]
    if np.isfinite(pct):
        level = "高位" if pct > 0.8 else ("低位" if pct < 0.2 else "中位")
        parts.append(f"历史分位 {pct:.0%}（{level}）")
    else:
        level = "—"
        parts.append("历史样本不足，算不出分位")
    return level, "，".join(parts)


def _interpret_curv(curv: float) -> str:
    """曲率 → 一句话。曲率大 = 两翼定价偏离平值多 = 尾部风险定价高。"""
    if not np.isfinite(curv):
        return "曲率数据缺失"
    if curv > 10:
        return f"曲率 {curv:.1f}，微笑两翼明显上翘，尾部风险定价偏高"
    if curv > 5:
        return f"曲率 {curv:.1f}，微笑有一定弧度"
    return f"曲率 {curv:.1f}，微笑曲线较平"


def build_brief(symbol: str) -> dict:
    """构建期权因子研判。

    返回::

        {
            "blocks": [...],   # 渲染块（指标卡 + 表格 + 文字结论）
            "feed": str,       # 回灌给模型的文本
            "has_data": bool,  # 是否有期权数据
        }

    无期权库或无该品种数据时 ``has_data=False``，blocks/feed 为空 ——
    调用方自行决定是报错还是跳过。
    """
    out: dict = {"blocks": [], "feed": "", "has_data": False}

    if not OS.exists():
        return out

    sym = _split_symbol(symbol)
    if not sym:
        return out

    # 全量 IV 数据（需要历史序列算分位）
    iv_df = OS.load_iv(period="1d")
    if iv_df.empty:
        return out

    varieties = sorted(iv_df["variety"].astype(str).unique())
    v = _match_variety(sym, varieties)
    if v is None:
        return out

    sub = iv_df[iv_df["variety"].astype(str) == v].sort_values("dt")
    if sub.empty:
        return out

    last = sub.iloc[-1]
    day = pd.Timestamp(last["dt"])
    iv_near = float(last["iv_near"]) if pd.notna(last["iv_near"]) else float("nan")
    iv_next = float(last["iv_next"]) if pd.notna(last["iv_next"]) else float("nan")
    skew = float(last["skew"]) if pd.notna(last["skew"]) else float("nan")
    curv = float(last["curv"]) if pd.notna(last["curv"]) else float("nan")
    pcr = float(last["pcr"]) if pd.notna(last["pcr"]) else float("nan")
    dte_near = int(last["dte_near"]) if pd.notna(last["dte_near"]) else 0
    dte_next = int(last["dte_next"]) if pd.notna(last["dte_next"]) else 0
    n_opt = int(last["n_opt"]) if pd.notna(last["n_opt"]) else 0

    # IV 历史分位
    iv_pct = _iv_percentile(sub["iv_near"], iv_near)

    # ---- 各因子解读
    iv_label, iv_text = _interpret_iv_level(iv_near, iv_pct)
    term_label, term_text = _interpret_term(iv_near, iv_next, dte_near, dte_next)
    skew_label, skew_text = _interpret_skew(skew)
    pcr_label, pcr_text = _interpret_pcr(pcr)
    curv_text = _interpret_curv(curv)

    # ---- 综合方向倾向：偏度和 PCR 交叉验证
    # 两个因子同向才算强信号；矛盾就标「分歧」
    skew_bearish = skew < -_SKEW_MODERATE
    skew_bullish = skew > _SKEW_MODERATE
    pcr_bearish = pcr > 1.0 and np.isfinite(pcr)
    pcr_bullish = pcr < 1.0 and np.isfinite(pcr)

    if skew_bearish and pcr_bearish:
        bias = "偏空"
        bias_detail = "偏度和 PCR 均指向偏空：虚值 put 较贵且 put 持仓偏高"
    elif skew_bullish and pcr_bullish:
        bias = "偏多"
        bias_detail = "偏度和 PCR 均指向偏多：虚值 call 较贵且 call 持仓偏高"
    elif skew_bearish and pcr_bullish:
        bias = "分歧"
        bias_detail = "偏度指向偏空（put 较贵）但 PCR 指向偏多（call 持仓多）——可能是卖出认沽的收入策略主导"
    elif skew_bullish and pcr_bearish:
        bias = "分歧"
        bias_detail = "偏度指向偏多（call 较贵）但 PCR 指向偏空（put 持仓多）——可能是保护性 put 对冲"
    else:
        bias = "中性"
        bias_detail = "偏度和 PCR 均无明显的方向信号"

    # ---- 波动率状态
    if np.isfinite(iv_pct):
        if iv_pct > 0.8:
            vol_state = "高位"
        elif iv_pct < 0.2:
            vol_state = "低位"
        elif iv_pct > 0.6:
            vol_state = "偏高"
        elif iv_pct < 0.4:
            vol_state = "偏低"
        else:
            vol_state = "中位"
    else:
        vol_state = "—"

    # ---- 指标卡
    def _pct(x):
        if not np.isfinite(x):
            return "—"
        return f"{x:.1%}"

    def _f(x, nd=2):
        if not np.isfinite(x):
            return "—"
        return f"{x:.{nd}f}"

    blocks: list[dict] = []
    blocks.append(B.block_metrics(
        f"{symbol} 期权因子（{day:%Y-%m-%d}）",
        [
            ("近月 IV", _pct(iv_near), "flat"),
            ("远月 IV", _pct(iv_next), "flat"),
            ("IV 分位", _pct(iv_pct) if np.isfinite(iv_pct) else "—",
             "good" if iv_pct > 0.8 else ("bad" if iv_pct < 0.2 else "flat")),
            ("偏度", _f(skew, 3), "flat"),
            ("曲率", _f(curv, 1), "flat"),
            ("PCR", _f(pcr, 2), "flat"),
            ("近月 DTE", str(dte_near), "flat"),
            ("报价点数", str(n_opt), "flat"),
        ],
    ))

    # ---- 研判结论表
    rows = [
        ["波动率水平", vol_state, iv_text],
        ["期限结构", term_label, term_text],
        ["偏度信号", skew_label, skew_text],
        ["PCR 信号", pcr_label, pcr_text],
        ["尾部风险", "—", curv_text],
        ["方向倾向", bias, bias_detail],
    ]
    blocks.append(B.block_table(
        ["维度", "判定", "解读"],
        rows,
        note=(f"数据口径：Black-76 反解，虚值，±20% moneyness 二次拟合，"
              f"15~180 天，结算价。{v} 近月剩余 {dte_near} 天 / 远月 {dte_next} 天。"
              f"参与拟合 {n_opt} 个虚值报价点。"),
    ))

    # ---- feed（回灌给模型的文本）
    feed_parts = [
        f"【期权因子·{symbol}】数据日期 {day:%Y-%m-%d}。",
        f"波动率：{iv_text}。期限结构：{term_text}。",
        f"方向：偏度 {skew_label}（{skew_text}）；PCR {pcr_label}（{pcr_text}）。",
        f"综合方向倾向「{bias}」——{bias_detail}。",
        f"尾部风险：{curv_text}。",
    ]
    feed = " ".join(feed_parts)

    out["blocks"] = blocks
    out["feed"] = feed
    out["has_data"] = True
    return out


# ------------------------------------------------------------------ 工具注册

def _option_brief(ctx, p: dict) -> SkillResult:
    symbol = str(p.get("symbol") or "").strip()
    if not symbol:
        raise RuntimeError("symbol 不能为空，比如 RB_SHFE")

    brief = build_brief(symbol)
    if not brief["has_data"]:
        from ttrade import optstore as OS
        if not OS.exists():
            raise RuntimeError(
                "本地还没有期权库 —— 去「数据管理 → 天勤」勾上"
                "「一并更新期权」，或命令行跑 download_options.py"
            )
        # 有库但没这个品种
        iv_df = OS.load_iv(period="1d")
        have = "、".join(sorted(iv_df["variety"].astype(str).unique())[:12])
        raise RuntimeError(
            f"期权库里没有 {symbol} 的 IV 数据（有的是：{have} …）"
        )

    return SkillResult(
        True,
        summary=f"{symbol} 期权因子研判（波动率 + 方向倾向）",
        blocks=brief["blocks"],
        feed=brief["feed"],
    )


register(Tool(
    name="option_brief",
    group="data",
    label="期权因子研判",
    description="从期权 IV / 偏度 / PCR 提取品种方向倾向和波动率状态，供研报使用",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["symbol"],
        "properties": {"symbol": {"type": "str"}},
    },
    handler=_option_brief,
    prompt=(
        "**期权因子研判**：`option_brief(symbol=\"RB_SHFE\")`。\n"
        "从期权库的 IV 摘要里提取五个维度的信号：\n"
        "- **波动率水平**：近月平值 IV 及其历史分位（高 / 低 / 中位）\n"
        "- **期限结构**：近月 vs 远月 IV（近高远低 = 近期事件驱动）\n"
        "- **偏度信号**：微笑曲线斜率，负 = 防跌，正 = 追涨\n"
        "- **PCR 信号**：put/call 持仓比，>1 偏空，<1 偏多\n"
        "- **方向倾向**：偏度和 PCR 交叉验证后的综合判断\n"
        "写研报时先调它拿到结论和数字，再用自己的话展开。\n"
        "数据口径：Black-76 反解，虚值，±20% moneyness 二次拟合，15~180 天，结算价。"
    ),
    max_turns=3,
    search_hint="option IV skew pcr volatility",
))
