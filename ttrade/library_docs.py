"""因子库与策略库的说明文档。

「交易Agent」页要把每个因子的**数学公式、实现代码、图例解释**摆出来，策略库要
把**逻辑、优势、适应情况、代码**摆出来。这个模块只放文字，不放任何计算。

## 为什么公式和代码要放在一起

文档和实现各写一份，早晚会对不上 —— 改了 `factors.py` 里的窗口，文档还写着旧数字，
而看文档的人不会去核对代码。所以：

- **代码**一律 `inspect.getsource()` 从函数对象上取，永远是当前实现；
- **公式**在这里手写（数学式子没法从代码里反推），但每一条都在冒烟测试里跟因子
  注册表对齐过：注册表里有而这里没有的因子会被判失败，反过来也一样。

## 为什么公式用纯文本

界面是 Qt 的 `QLabel` / `QTextBrowser`，没有 MathJax。用 Unicode 上下标和
`P_t / P_{t−n}` 这种写法，不装字体也能读，复制到大模型对话里也不变形。

## 方向提示不是交易信号

`hint_dir` 只说明「历史上这类因子高分倾向涨还是跌」，真实方向一律由滚动 IC 决定
（见 `factors.py` 的注释）。文档里也照这个口径写，不给人「照着做」的错觉。
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field

from ttrade.factors import FACTORS
from ttrade.strategies import STRATEGIES


# ------------------------------------------------------------------ 因子公式

# 每条：formula（数学式子）+ explain（怎么读、什么时候失效）
FACTOR_DOCS: dict[str, dict] = {
    # ---- 趋势 ----
    "mom_20": {
        "formula": "Mom(n) = P_t / P_{t−n} − 1，n = 20",
        "explain": "20 根 K 线的涨跌幅。趋势因子最朴素的形态：涨得多的继续涨。"
                   "周期越短噪声越大，20 根在日线上约一个月，属于短周期趋势。",
    },
    "mom_60": {
        "formula": "Mom(n) = P_t / P_{t−n} − 1，n = 60",
        "explain": "三个月动量，商品期货上最经典的一档。学术界说的「商品动量」"
                   "通常就是这个窗口。",
    },
    "mom_120": {
        "formula": "Mom(n) = P_t / P_{t−n} − 1，n = 120",
        "explain": "半年动量。比 60 根更平滑，代价是对反转的响应慢一两个月。",
    },
    "mom_250": {
        "formula": "Mom(n) = P_t / P_{t−n} − 1，n = 250",
        "explain": "年度动量。最慢也最稳，一年里只有一两次方向切换；"
                   "换月频繁的品种上要确认复权口径正确，否则这条因子会被换月跳空污染。",
    },
    "ma_gap_20": {
        "formula": "Gap(n) = P_t / MA_n(t) − 1，MA_n = n 根收盘均值",
        "explain": "收盘相对均线的偏离度。正值=在均线上方。和动量高度相关，"
                   "但因为是「相对均线」，在单边行情里不会像动量那样随价格水平漂移。",
    },
    "ma_gap_60": {
        "formula": "Gap(n) = P_t / MA_n(t) − 1，n = 60",
        "explain": "季度均线的偏离度，比 20 根更慢，适合过滤短期噪声。",
    },
    "ma_slope_20": {
        "formula": "Slope = MA_20(t) / MA_20(t−10) − 1",
        "explain": "均线自身的斜率。均线已经滤过一遍价格，所以斜率比价格动量平滑；"
                   "代价是转折点滞后。",
    },
    "breakout_pos": {
        "formula": "Pos = (P_t − LL_55) / (HH_55 − LL_55)",
        "explain": "0 = 贴着 55 根最低，1 = 贴着最高。唐奇安突破的连续化版本："
                   "突破策略只取 0/1 两个状态，这条因子把整条通道都变成信息。",
    },
    "efficiency": {
        "formula": "ER = |P_t − P_{t−20}| / Σ|ΔP|（20 根）",
        "explain": "净位移 ÷ 总路程。接近 1 说明这段走势很「直」（趋势清晰），"
                   "接近 0 说明来回震荡。常用来给动量因子加权：ER 高的时段动量更可信。",
    },
    # ---- 反转 ----
    "rev_5": {
        "formula": "Rev(n) = −(P_t / P_{t−n} − 1)，n = 5",
        "explain": "短期反转：取 5 根涨跌幅的负数。和动量方向相反，"
                   "在震荡市里有效、单边市里是灾难。",
    },
    "rev_20": {
        "formula": "Rev(n) = −(P_t / P_{t−n} − 1)，n = 20",
        "explain": "月度反转。商品上的反转效应比股票弱，多数品种 20 根尺度上仍是动量占优，"
                   "所以这条更适合拿来做对照而不是单独用。",
    },
    "rsi_14": {
        "formula": "RSI = 100 − 100 / (1 + RS)，RS = EMA(Δ⁺, α=1/14) / EMA(Δ⁻, α=1/14)",
        "explain": "0~100 的相对强弱。>70 常被当超买、<30 超卖，但**别直接比阈值**："
                   "不同品种的 RSI 分布差很多，进模型前要标准化（平台已经做了）。",
    },
    "bb_pos_20": {
        "formula": "BB = (P_t − MA_20) / (2·σ_20)",
        "explain": "布林带位置，±1 是上下轨。用 2σ 归一化，跨品种可比；"
                   "突破 ±1 在趋势市里是延续信号而不是反转信号。",
    },
    "stoch_14": {
        "formula": "%K = (P_t − LL_14) / (HH_14 − LL_14)",
        "explain": "收盘在近 14 根高低区间中的位置。和 breakout_pos 同构，只是窗口更短、"
                   "更敏感。",
    },
    # ---- 波动 ----
    "vol_20": {
        "formula": "σ_20 = std(r_t, 20) × √252，r_t = P_t/P_{t−1} − 1",
        "explain": "年化已实现波动率。低波动因子：低波动的品种/时段风险调整后收益更好。"
                   "注意年化系数按日线写死 252，分钟级数据上这个数会失真（见回测的年化口径）。",
    },
    "vol_ratio": {
        "formula": "VR = σ_10 / σ_60",
        "explain": "短长波动比。>1 说明波动在放大，常出现在趋势启动或恐慌下跌时；"
                   "<1 说明市场安静下来。",
    },
    "atr_pct": {
        "formula": "ATR_14 / P_t，ATR = Wilder 平滑的 TR",
        "explain": "ATR 占价格的比例，把「一天波动多少钱」换算成百分比，"
                   "这样螺纹和黄金能放在一起比。仓位管理里常用它定手数。",
    },
    "downside_ratio": {
        "formula": "DR = σ(r | r<0, 20) / σ(r, 20)",
        "explain": "下行波动占总波动的比例，是偏度的粗略代理。>1 说明跌的时候更剧烈，"
                   "商品上很常见（跌得快、涨得慢）。",
    },
    "maxdd_60": {
        "formula": "DD = P_t / HH_60 − 1",
        "explain": "当前价距 60 根高点的距离，永远是负数或 0。越接近 0 说明在创新高。",
    },
    # ---- 量能 ----
    "vol_surge": {
        "formula": "VS = V_t / MA_20(V) − 1",
        "explain": "放量倍数。放量突破比缩量突破可信；但换月当天成交量会跳，"
                   "换月附近这条因子不可用。",
    },
    "pv_corr": {
        "formula": "ρ = corr(r_t, ΔV_t/V_{t−1})，20 根",
        "explain": "价量相关性。正相关=价涨量增（健康），负相关=背离。"
                   "分钟级数据上噪声很大。",
    },
    "obv_slope": {
        "formula": "OBV_t = Σ sign(ΔP)·V，因子 = ΔOBV_20 / ΣV_20",
        "explain": "能量潮的 20 根变化除以同期总量，得到 -1~1 的「量能累积速度」。"
                   "除以总量是为了跨品种可比。",
    },
    "illiq": {
        "formula": "ILLIQ = mean_20(|r_t| / Amount_t) × 10⁹",
        "explain": "Amihud 非流动性：单位成交额能推动多少涨跌。值大=流动性差。"
                   "需要成交额字段，没有就返回 NaN（不是 0）。",
    },
    "oi_change": {
        "formula": "ΔOI = OI_t / OI_{t−20} − 1",
        "explain": "持仓量 20 根变化率，期货特有。增仓上涨比减仓上涨更可信，"
                   "因为前者是新资金进场。",
    },
    # ---- 形态 ----
    "hl_range": {
        "formula": "HL = mean_20((H_t − L_t) / C_t)",
        "explain": "平均振幅占收盘价的比例。和 ATR 类似但用的是当日真实高低，"
                   "不含跳空。",
    },
    "close_pos": {
        "formula": "CP = mean_10((C_t − L_t) / (H_t − L_t))",
        "explain": "收盘在当日振幅中的平均位置。>0.5 说明收盘偏强（尾盘买盘），"
                   "是日内动量的日线代理。",
    },
    "gap": {
        "formula": "Gap = mean_10(O_t / C_{t−1} − 1)",
        "explain": "平均跳空幅度。中国商品有夜盘，隔夜跳空比股票小，但节假日前后会很大。",
    },
    "macd_hist": {
        "formula": "DIF = EMA_12 − EMA_26，DEA = EMA_9(DIF)，因子 = (DIF − DEA)/C",
        "explain": "MACD 柱除以收盘价，把趋势强度归一化成跨品种可比的量。"
                   "除以价格这一步很关键，否则高价品种（黄金）永远排在前面。",
    },
    # ---- 展期 ----
    "roll_yield": {
        "formula": "RY = P₁ / P₂ − 1（P₁ 近月、P₂ 次近月）",
        "explain": "展期收益率。现货溢价（backwardation）时为正：近月比远月贵，"
                   "多头移仓能赚价差；期货溢价（contango）时为负，持有多头要付移仓成本。"
                   "商品期货上最经典的一类因子。",
    },
    "roll_annual": {
        "formula": "RY_ann = RY × 365 / (T₂ − T₁)",
        "explain": "把价差按两个合约的到期间隔折算成年化。不年化就没法横向比较："
                   "同样是 1% 的价差，隔 15 天和隔 150 天完全不是一回事。",
    },
    "roll_curv": {
        "formula": "k₁ = ln(P₁/P₂)/(T₂−T₁)，k₂ = ln(P₂/P₃)/(T₃−T₂)，因子 = (k₂ − k₁) × 365",
        "explain": "期限结构曲率。用「斜率之差」而不是三点二阶差分，"
                   "因为三个合约的到期间隔通常不等距，直接差分会被间隔差异污染。"
                   "负值=近端更陡（近月紧张）。",
    },
    "roll_change": {
        "formula": "ΔRY_ann = RY_ann(t) − RY_ann(t−20)",
        "explain": "年化展期收益率的 20 根变化，反映曲线陡峭度的动量。"
                   "用年化值做差是为了减少换月日的台阶。",
    },
    "near_oi_share": {
        "formula": "Share = OI₁ / (OI₁ + OI₂)",
        "explain": "近月持仓占比。接近 1 说明资金挤在近月（现货紧张或交割博弈），"
                   "偏低说明主力已经移到远月。只在原口径提供 —— OI3/OI6 的三腿本来就是"
                   "「持仓量前 3/前 6」，再算占比恒在 0.5~0.7 之间，没有信息量。",
    },
    "roll_yield_o3": {
        "formula": "RY = P₁ / P₂ − 1，三腿取自**持仓量前 3** 的合约",
        "explain": "筛活跃口径的展期收益率。主力月份跳跃的品种（螺纹只做 1/5/10）上，"
                   "纯按到期日取近月常常落在冷门月，筛一遍持仓量能滤掉空壳合约。",
    },
    "roll_annual_o3": {
        "formula": "RY_ann = RY × 365 / (T₂ − T₁)，OI3 口径",
        "explain": "OI3 口径的年化展期收益率。",
    },
    "roll_curv_o3": {
        "formula": "因子 = (k₂ − k₁) × 365，OI3 口径",
        "explain": "OI3 口径的期限结构曲率。三腿都落在活跃月上，报价噪声比原口径小。",
    },
    "roll_change_o3": {
        "formula": "ΔRY_ann = RY_ann(t) − RY_ann(t−20)，OI3 口径",
        "explain": "OI3 口径的展期变化。",
    },
    "roll_yield_o6": {
        "formula": "RY = P₁ / P₂ − 1，三腿取自**持仓量前 6** 的合约",
        "explain": "候选池比 OI3 宽，第三腿更可能落在真正活跃的远月上；"
                   "代价是换腿更频繁。流动性前 40 的品种上这一档通常更强。",
    },
    "roll_annual_o6": {
        "formula": "RY_ann = RY × 365 / (T₂ − T₁)，OI6 口径",
        "explain": "OI6 口径的年化展期收益率。",
    },
    "roll_curv_o6": {
        "formula": "因子 = (k₂ − k₁) × 365，OI6 口径",
        "explain": "OI6 口径的期限结构曲率。",
    },
    "roll_change_o6": {
        "formula": "ΔRY_ann = RY_ann(t) − RY_ann(t−20)，OI6 口径",
        "explain": "OI6 口径的展期变化。",
    },
    # ---- 期权 ----
    "iv_atm": {
        "formula": "IV_atm = f(K = F)，f 为当日近月期权微笑的二次拟合",
        "explain": "平值隐含波动率。不是单点插值：整条微笑做二次拟合后在 K=F 处取值，"
                   "这样单个报价异常不会把值带跑。IV 高说明市场为未来波动付了高价。",
    },
    "iv_term": {
        "formula": "Term = (IV₂ − IV₁) / (T₂ − T₁) × 30",
        "explain": "IV 期限结构斜率，单位是「每月多少 IV 点」。正值=远月更贵（正常），"
                   "负值=近月恐慌，是压力期最典型的信号之一。",
    },
    "iv_skew": {
        "formula": "Skew = dIV / dlnK ｜_{K=F}",
        "explain": "偏度。正值=虚值看涨比虚值看跌贵（市场怕涨）；负值=虚值看跌更贵"
                   "（市场怕跌）。商品期权多数时间负偏，因为下跌常伴随波动率跳升。",
    },
    "iv_rv": {
        "formula": "VRP = IV_atm − σ_20",
        "explain": "方差风险溢价。期权通常比事后实现的波动更贵，所以长期偏正；"
                   "被压到 0 附近甚至转负，说明期权便宜或市场已进入高波动。两个量都年化，可直接相减。",
    },
    "iv_change": {
        "formula": "ΔIV = IV_atm(t) − IV_atm(t−20)",
        "explain": "IV 的 20 根变化，波动率预期的动量。",
    },
    "pcr": {
        "formula": "PCR = ln(PutOI / CallOI)",
        "explain": "看跌/看涨持仓比，取对数。取对数的原因：这个比值天然是乘性的"
                   "（0.5 与 2.0 是同一件事的两个方向），取完才左右对称。>0 偏空。",
    },
}


# ------------------------------------------------------------------ 策略说明

STRATEGY_DOCS: dict[str, dict] = {
    "ma_cross": {
        "logic": "快均线上穿慢均线做多，下穿做空，始终在市、对称反手。"
                 "信号在收盘确定，次日开盘成交。",
        "edge": "趋势跟踪的地基。参数只有两个，过拟合空间小；"
                "任何一段像样的单边行情都能抓到，且不会因为「觉得贵了」提前下车。",
        "fits": "中长周期、流动性好的品种：螺纹、铁矿、铜、豆粕。"
                "震荡市里会被反复止损，这是趋势策略的固有成本，不是 bug。",
        "risk": "横盘时连续小亏；参数（尤其 slow）敏感，换个周期最优值就变。",
        "params": "fast 快线根数，slow 慢线根数。fast 越接近 slow 交易越频繁。",
    },
    "donchian": {
        "logic": "收盘突破前 N 根最高价做多，跌破前 N 根最低价做空；"
                 "反向突破 M 根通道时离场。",
        "edge": "海龟法则的核心。只在「创新高/新低」时动手，天然过滤掉大部分噪声；"
                "离场用更短的通道，保证趋势结束能及时退出。",
        "fits": "趋势性强、假突破少的品种。上市时间短的品种要小心："
                "N=55 意味着前 55 根没有信号。",
        "risk": "假突破（突破后立刻回落）会造成连续止损；震荡市里最难受。",
        "params": "entry 入场通道长度，exit 离场通道长度。exit 必须明显小于 entry。",
    },
    "atr_channel": {
        "logic": "以均线为中轴，上下各 k 倍 ATR 构成通道；收盘突破上轨做多、"
                 "跌破下轨做空，回到均线离场。",
        "edge": "通道宽度随波动率自适应 —— 波动大的时候门槛自动放宽，"
                "不会像固定百分比通道那样在高波动期被反复打脸。",
        "fits": "波动率变化大的品种（原油、镍、纯碱）。低波动品种上通道太窄，交易过频。",
        "risk": "波动率突然放大时通道急速变宽，可能错过启动点。",
        "params": "n 均线根数，k ATR 倍数。k 越小越激进。",
    },
    "macd": {
        "logic": "零轴上方金叉做多、零轴下方死叉做空；反向交叉或柱线回零轴离场。",
        "edge": "双均线差值的平滑版，比单纯均线交叉少一些假信号；"
                "「零轴」这个条件把信号限制在趋势方向上。",
        "fits": "中长周期趋势品种。分钟级上 MACD 参数要重新调，默认 12/26/9 是日线口径。",
        "risk": "参数多一个（signal_n），过拟合空间比双均线大；"
                "震荡市里零轴附近的交叉几乎全是噪声。",
        "params": "fast/slow 两条 EMA 的跨度，signal_n DEA 的平滑跨度。",
    },
    "turtle": {
        "logic": "唐奇安通道双向突破入场，持仓后用 2.5×ATR 的移动止损跟随，"
                 "止损被打到就平仓，同时兼顾反向通道离场。",
        "edge": "把「入场」和「风险控制」分开：入场靠突破，离场靠 ATR。"
                "单笔风险可控，长期看靠少数大盈利覆盖多次小亏。",
        "fits": "波动大、趋势持续久的品种。ATR 倍数需要按品种调 —— "
                "2.5 对螺纹合适，对铜可能太紧。",
        "risk": "震荡市里止损被反复触发，交易成本高；ATR 用错周期会让止损过松或过紧。",
        "params": "entry/exit 通道长度，atr_n ATR 根数，atr_mult 止损的 ATR 倍数。",
    },
}


# ------------------------------------------------------------------ 文档对象


@dataclass
class FactorDoc:
    key: str
    label: str
    category: str
    desc: str
    formula: str
    explain: str
    code: str
    hint_dir: int = 1

    @property
    def direction(self) -> str:
        """历史倾向，**不是**交易信号。回测里方向由滚动 IC 决定。"""
        return {1: "高分偏涨", -1: "高分偏跌", 0: "无明显方向"}.get(self.hint_dir, "")

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "category": self.category,
            "desc": self.desc, "formula": self.formula, "explain": self.explain,
            "code": self.code, "hint_dir": self.hint_dir,
        }


@dataclass
class StrategyDoc:
    key: str
    label: str
    desc: str
    logic: str
    edge: str
    fits: str
    risk: str
    params_help: str
    params: dict = field(default_factory=dict)
    param_meta: dict = field(default_factory=dict)
    code: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "desc": self.desc,
            "logic": self.logic, "edge": self.edge, "fits": self.fits,
            "risk": self.risk, "params_help": self.params_help,
            "params": dict(self.params), "code": self.code,
        }


def _source(func) -> str:
    """取函数源码。取不到就返回空串 —— 打包成 exe 之后可能没有 .py 文件。"""
    try:
        return textwrap.dedent(inspect.getsource(func)).rstrip()
    except (OSError, TypeError):
        return ""


# ------------------------------------------------------------------ 取用


def factor_doc(key: str) -> FactorDoc | None:
    spec = FACTORS.get(key)
    if spec is None:
        return None
    doc = FACTOR_DOCS.get(key, {})
    return FactorDoc(
        key=key,
        label=spec.label,
        category=spec.category,
        desc=spec.desc,
        formula=doc.get("formula", "（未收录公式）"),
        explain=doc.get("explain", spec.desc),
        code=_source(spec.func),
        hint_dir=spec.hint_dir,
    )


def all_factor_docs() -> list[FactorDoc]:
    return [d for k in FACTORS if (d := factor_doc(k)) is not None]


def strategy_doc(key: str) -> StrategyDoc | None:
    spec = STRATEGIES.get(key)
    if spec is None:
        return None
    doc = STRATEGY_DOCS.get(key, {})
    return StrategyDoc(
        key=key,
        label=spec.label,
        desc=spec.desc,
        logic=doc.get("logic", spec.desc),
        edge=doc.get("edge", ""),
        fits=doc.get("fits", ""),
        risk=doc.get("risk", ""),
        params_help=doc.get("params", ""),
        params=dict(spec.params),
        param_meta=dict(spec.param_meta),
        code=_source(spec.func),
    )


def all_strategy_docs() -> list[StrategyDoc]:
    return [d for k in STRATEGIES if (d := strategy_doc(k)) is not None]


# ------------------------------------------------------------------ 给大模型的清单


def factor_brief(keys: list[str] | None = None) -> str:
    """因子清单，一行一个。塞进大模型 prompt 用，不要带源码（太长）。"""
    items = [factor_doc(k) for k in (keys or list(FACTORS))]
    lines = []
    for d in items:
        if d is None:
            continue
        lines.append(f"- {d.key}｜{d.label}｜{d.category}｜{d.formula}｜{d.desc}")
    return "\n".join(lines)


def strategy_brief() -> str:
    lines = []
    for d in all_strategy_docs():
        p = "，".join(f"{k}={v}" for k, v in d.params.items())
        lines.append(f"- {d.key}｜{d.label}｜默认参数 {p}｜{d.desc}")
    return "\n".join(lines)
