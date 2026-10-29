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
    # 主力/次主力：两端都按持仓量选，不是「按到期日排三腿」
    "roll_main_oi": {
        "formula": "RY_main = P_main / P_next − 1，P_main = 持仓最大的腿、"
                   "P_next = 其**之后**持仓最大的腿；再取 5 日均值",
        "explain": "近端和远端都由持仓量决定，所以近端一定是最活跃的合约、远端一定比它远。"
                   "中国商品期货的主力月份是跳跃的（螺纹只做 1/5/10），按到期日排出来的"
                   "「近月」经常是个空壳合约，持仓口径不会 —— 这是它和 roll_yield 的根本区别。"
                   "5 日均值是为了抹掉换腿当天的台阶。正值 = backwardation。",
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

    "roll_intraday": {
        "formula": "Factor = [c1/c2 − 1]_close − [c1/c2 − 1]_首根 bar",
        "explain": "日内价差漂移：同一对近远月合约，收盘时的价差减去当天开盘时的"
                   "价差。正的表示近月当天相对远月走强，是「谁在被买」的即时读数。"
                   "实测只在次日有微弱 IC（约 0.011），5 日以上归零，当辅助项用。",
    },

    # ---- 日内（需要 5m / 1h 行情，日线上整列为空）----
    "vwap_gap": {
        "formula": "VWAP_t = Σ(P_typ·V) / ΣV，P_typ = (H+L+C)/3（当日累计）"
                   "；Factor = C_t / VWAP_t − 1",
        "explain": "现价相对当日成交量加权均价的偏离。VWAP 是当天所有成交的平均成本，"
                   "站在它上面说明买方愿意追价。库里没有成交额，用典型价代替成交价，"
                   "误差在日内高低点被反复测试的日子会略大。",
    },
    "pv_pos": {
        "formula": "Pos = Σ(V·(C−L)/(H−L)) / ΣV（当日累计）− 0.5",
        "explain": "量价位置：把成交量当权重，看当天的量堆在高价区还是低价区。"
                   ">0 = 放量在高位（有承接），<0 = 放量在低位（在派发）。"
                   "和单纯收盘位置的区别在于「量」——缩量收高和放量收高不是一回事。",
    },
    "range_break": {
        "formula": "Factor = (C_t − O_day) / (max H − min L)（当日累计）",
        "explain": "收盘相对开盘的位移占当日已实现振幅的比例，取值 −1~+1。"
                   "+1 表示收在当日最高、−1 表示收在最低。趋势日顶到两端，震荡日贴 0。",
    },
    "intraday_mom": {
        "formula": "Factor = C_t / O_day − 1",
        "explain": "日内动量，把隔夜跳空剔掉。日线的 mom 把隔夜和日内揉在一起，"
                   "隔夜多半是外盘驱动、日内才是本国资金的态度，分开看才能判断"
                   "「是外盘带的还是自己走的」。",
    },
    "overnight_gap": {
        "formula": "Gap = O_day / C_{day−1} − 1",
        "explain": "隔夜跳空。国内夜盘 21:00 就开盘，所以这个缺口实际是上一交易日"
                   "15:00 到当晚 21:00 之间外盘与消息的影响。连续同向跳空往往意味着"
                   "有资金在抢跑。",
    },
    "close_drive": {
        "formula": "Factor = (C_close / O_14:00 − 1) / ((H_upto − L_upto)/C)",
        "explain": "尾盘驱动：最后一个小时的收益占当日振幅的比例。收盘前是资金定调的"
                   "时候，尾盘拉升且占比高说明有人愿意持仓过夜，反之是尾盘出货。"
                   "分母用「截至当前 bar」的振幅（cummax/cummin），不是当日全天高低点。"
                   "14:00 之前整列为空 —— 尾盘还没发生，这个量不存在。",
    },

    # ---- 时段 ----
    "trend_window": {
        "formula": "Factor = C(10:15) / O(09:45) − 1",
        "explain": "活跃时段收益。开盘头半小时是集合竞价的余波、噪音大；09:45 之后"
                   "隔夜挂单消化完，日内资金开始表态，这段方向对全天的延续性最好。"
                   "需要 5m 数据，1h 切不出 30 分钟窗口（返回空）。"
                   "**窗口走完之前整列为空**：09:05 那根 bar 不该知道 10:15 的收盘价，否则就是拿当天后半段去预测当天，IC 会虚高。",
    },
    "quiet_window": {
        "formula": "Factor = −(C(14:00) / O(13:30) − 1)",
        "explain": "非活跃时段反转。午后开盘这段是一天里最清淡的窗口，单边推动多半"
                   "是流动性不足造成的过度反应，取负号让「高分=预期反转」。同样需要 5m。"
                   "**窗口走完之前整列为空**：09:05 那根 bar 不该知道 10:15 的收盘价，否则就是拿当天后半段去预测当天，IC 会虚高。",
    },
    "first_hour": {
        "formula": "Factor = C(10:00) / O(09:00) − 1",
        "explain": "开盘首小时收益，隔夜信息的消化窗口。1h 上退化为 09:00 那根 bar"
                   "自己的涨跌，仍可用。**窗口走完之前整列为空**：09:05 那根 bar 不该知道 10:15 的收盘价，否则就是拿当天后半段去预测当天，IC 会虚高。",
    },
    "night_ret": {
        "formula": "Factor = C(23:00) / O(21:00) − 1",
        "explain": "夜盘收益。夜盘由外盘驱动，和日盘是两拨资金；没有夜盘的品种"
                   "（如部分农产品）返回空。**窗口走完之前整列为空**：09:05 那根 bar 不该知道 10:15 的收盘价，否则就是拿当天后半段去预测当天，IC 会虚高。"
                   "（夜盘窗口在交易日里排在最前面，23:00 之后到收盘都有值）",
    },
    "session_spread": {
        "formula": "Factor = Ret(21:00–23:00) − Ret(09:00–15:00)",
        "explain": "日夜盘强弱差。夜盘强、日盘弱说明外盘在带而国内在抛，"
                   "次日容易低开回补；反过来是内盘独立走强。"
                   "要等日盘 15:00 走完才出值，所以 5m 上只有收盘那一根非空 —— "
                   "它本来就是日频概念。",
    },

    # ---- 季节性 ----
    "season_avg": {
        "formula": "S_m(Y) = mean{ R_{y,m} : y < Y }，R 为当月累计收益，需 ≥3 年",
        "explain": "平均月度曲线在当月的值，即「这个品种这个月历史上平均涨多少」。"
                   "严格只用往年（不含当年），所以每年 1 月会重新估一次。",
    },
    "season_ir": {
        "formula": "Factor = mean{ R_{y,m} : y < Y } / std{ R_{y,m} : y < Y }",
        "explain": "季节性的稳定性。同样的 +2% 均值，有的是年年如此，有的是某一年"
                   "暴涨拉起来的；除以标准差才能分开。只有稳定重现的才配叫季节性。",
    },
    "season_rank": {
        "formula": "Factor = rank_12( S_m ) / 12 − 0.5",
        "explain": "当月历史均值在 12 个月中的分位。跨品种可比：铜的 3 月是 +1.2%、"
                   "豆粕是 +3%，绝对值没法比，但在各自 12 个月里的排位可比。",
    },
    "season_gap": {
        "formula": "Factor = MTD − S_m · (已过交易日 / 21)",
        "explain": "本月至今收益减去「历史同月到此刻该走的部分」。超额部分要么是"
                   "基本面新信息，要么是季节性透支。按进度折算，不拿整月均值比半个月。",
    },
    "season_cal": {
        "formula": "Factor ∈ {+1, −1, 0}：按品种产业日历查当月窗口",
        "explain": "基本面季节窗口：+1 旺季（如螺纹金三银四、白糖 6-8 月消费旺季）、"
                   "−1 淡季（如豆粕 9-11 月北美收割压力）。日历写死在代码里，来自"
                   "产业常识而非价格拟合；表里没有的品种返回空。",
    },
    "season_cal_mom": {
        "formula": "Factor = Calendar(m) · (P_t / P_{t−20} − 1)",
        "explain": "季节窗口当门控的动量：旺季做多趋势、淡季做空趋势、窗口外为 0。"
                   "动量单用的问题是趋势在淡季也照样延续，用日历当条件能砍掉一半"
                   "无效信号。",
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
    "sbbox_mini": {
        "logic": "两个条件同时成立才入场：收盘突破 240 根箱体，且相对**当日开盘**的偏离"
                 "不小于 0.4 倍参考波幅。离场两条：反向走 5×ATR（止损），或反向突破"
                 "720 根箱体（趋势结束）。信号收盘定，下一根开盘成交。",
        "edge": "只看突破的话，5 分钟上窄幅横盘里「高一个点」也算突破，一天能开十几次；"
                "加一条「偏离够大」就把这些滤掉了。箱体管方向、波幅管力度，两把尺子"
                "各管一头，比单用其中一个稳定。",
        "fits": "5 分钟周期、日内波动有明显放大的品种（碳酸锂、白银、纯碱）。"
                "1 分钟上噪声太多，日线上 240 根箱体要一年才成形，都不合适。",
        "risk": "震荡市里两个条件都容易满足（假突破 + 大波幅），会连续小亏；"
                "vol_ratio 调太小等于退回单条件，调太大一天都开不出一笔。",
        "params": "entry/exit_n 入场与离场箱体根数，vol_ratio 波幅确认倍数，"
                  "range_n 参考波幅取几根，atr_n/stop_mult 止损口径。",
    },
    "im_orb": {
        "logic": "只在 09:40-10:20 开仓：价格碰到「当日开盘 ± 0.4 × 前 N 日全天振幅」"
                 "就顺方向进，3×ATR 止损，次日 09:35 平。日线数据上自动退化为"
                 "次日收盘平。",
        "edge": "时段是硬约束不是优化出来的：开盘头十分钟价格还在找方向，而突破得越晚、"
                "离收盘越近，赚的越不够覆盖隔夜风险。振幅门槛用**前一日**的全天振幅，"
                "不用当天的（当天高低点收盘才知道，用了就是未来函数）。",
        "fits": "股指日内（IM/IC/IF），需要 5 分钟数据。其它品种也能跑，但「次日 09:35 平」"
                "是按股指的连续交易时段定的，夜盘品种上这个时间点没有意义。",
        "risk": "单边大趋势里 ORB 只吃到第一段；窄幅日会被振幅门槛挡在门外（这是好事）。"
                "参数里三个时间点（entry_start/entry_end/exit_min）都以分钟为单位，"
                "改成别的品种时要重算。",
        "params": "entry_start/entry_end 开仓时段（0 点起的分钟数），exit_min 平仓时刻，"
                  "vol_ratio 振幅倍数，range_n 参考几日振幅，atr_n/stop_mult 止损口径。",
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
    # 用户自定义因子把公式和源码存在 `FactorSpec` 上：它们不在 FACTOR_DOCS 里，
    # `inspect.getsource()` 也取不到（exec 出来的函数没有对应文件）。
    return FactorDoc(
        key=key,
        label=spec.label,
        category=spec.category,
        desc=spec.desc,
        formula=doc.get("formula") or spec.formula or "（未收录公式）",
        explain=doc.get("explain") or spec.desc,
        code=spec.code or _source(spec.func),
        hint_dir=spec.hint_dir,
    )


def all_factor_docs() -> list[FactorDoc]:
    return [d for k in FACTORS if (d := factor_doc(k)) is not None]


def strategy_doc(key: str) -> StrategyDoc | None:
    spec = STRATEGIES.get(key)
    if spec is None:
        return None
    doc = STRATEGY_DOCS.get(key, {})
    # 用户注册的策略不在 STRATEGY_DOCS 里，源码也取不到（exec 出来的函数没有
    # 源文件），公式和代码只能靠 `StrategySpec` 自己带着
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
        code=spec.code or _source(spec.func),
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
