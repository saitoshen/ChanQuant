"""多策略自动交易引擎。

一个 `TqApi` 管一组策略：每个策略订阅自己的主连 K 线，新 bar 一到就用**刚收盘
的那根**重算信号，目标净仓变了就按新 bar 的开盘价下单。和 `gui/app.py` 里那个
「单策略 + 手动绿灯」是两套东西 —— 那边是研究用，这边是组合运行。

## 成交时点：T 根收盘出信号，T+1 根开盘进场

回测引擎里是 `target = np.roll(sig, 1)`，即第 T 根收盘的信号在第 T+1 根开盘
成交。实时这边必须同口径，否则「回测赚钱、实盘亏钱」永远说不清是谁的问题。

天勤的 K 线序列最后一根是**正在走的活 bar**，它的 `datetime` 在新 bar 出现时
就变了，但 OHLC 要到收盘才定。拿它算信号有两个毛病：

- 开盘那一瞬 `close == open`，突破 / 偏离类条件会在开盘就成立、收盘又消失，
  实盘比回测早一根进场，进的还是「开盘时的假信号」；
- ATR 这类用 `high - low` 的量在开盘等于 0，止损线被算得极窄。

所以 `_refresh_signal` 只喂**已收盘**的 bar（`df.iloc[:-1]`）：此刻正是新 bar
刚开盘，用上一根收盘的信号在这一刻下单，就是回测里的 T+1 开盘成交。

## 为什么按账户分组，而不是一个 Runner 管所有

天勤的 `TqApi` 在构造时绑定账户，一个 api 只能对应一个资金账号；同一账号再开
第二个连接还会把第一个踢下线。所以「每个策略绑不同账户」只能靠**多个 Runner**，
每个 Runner 一个 api。`PortfolioRunner` 就是「一个账户 + 一组策略」。

## 为什么不用 TargetPosTask

`TargetPosTask` 适合「按目标净仓调仓」，反手时自己拆平/开、自己处理平今平仓，
但它按固定节奏重报，遇到流动性差的合约经常挂一整天，而且不给上层「等几秒、
追几次」的把手。这里改成自己 `insert_order` + 追单状态机，代价是要自己判断
平今/平仓 —— 逻辑见 `_offset_for()`。

## 追单为什么要有 cancelling 这一态

撤单和重报之间必须等旧委托真的死掉。直接「撤完立刻重报」会在旧单还有效的窗口里
重复成交，一次信号变两倍仓位。所以状态机是 `live → cancelling → 重报`。
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import pandas as pd

from ttrade import factor_strategy as fstrat
from ttrade import ml_artifact
from ttrade import replay
from ttrade import (barclock, daylimit, orderbook, positionbook, topup, tqdata,
                    tqstore)
from ttrade import strategy_store as SStore
from ttrade import specs, symbols_info
from ttrade.market_feed import MarketFeed
from ttrade.backtest import size_lots_intrabar, size_plan
from ttrade.strategies import STRATEGIES, next_levels, run_strategy
from ttrade.tqfetch import PERIOD_SEC
from ttrade.tqlive import (BACKEND_LABELS, AccountSpec, bars_from_serial,
                           merge_live)
from ttrade.tqsdk_symbols import is_stock, trade_symbol

KLINE_LEN = 3000          # 每个策略回看多少根。日线 3000 根够跑 MA200，5m 够跑一周多
MIN_BARS = 30             # 少于这么多根不算信号
# 因子策略要多得多的历史：窗口 252 加上 z 分数，少于这个数算出来的分数全是 NaN
FACTOR_MIN_BARS = 300

# 多品种策略在引擎里被拆成「一个品种一个槽位」，槽位 id 用 `父 id#品种键`。
# 界面上仍然是一条策略一行 —— 用户管的是「这条策略」，不是 19 个各自为政的
# 槽位；拆开只是为了每个品种有自己的订阅、自己的仓位账本、自己的手数。
GROUP_SEP = "#"


def require_live_account(spec: AccountSpec) -> None:
    """启动前的账户体检：**在最前面**（补数据、连柜台之前）把配错的挡掉。

    `_ensure_aux()` 会先补期货日线和展期/期权快照（几分钟），用户等完才看到一句
    「缺少天勤账户或密码」，既不知道错在哪，也不知道该改什么。所以缺参数、柜台
    不支持这类**一眼就能判的**问题都挪到这里。

    三种后端都能跑：

    - `tq`：天勤 TqApi，历史 + 实时 + 下单全走天勤；
    - `ctp` / `rohon`：历史走本地天勤数据（柜台没有历史行情），实时行情和下单
      走柜台（见 `ttrade/broker_api.py`）。

    唯一跑不了的是「柜台 + A 股」：CTP / 融航 只做期货，没有股票代码可报。
    """
    be = getattr(spec, "backend", "tq")
    who = str(getattr(spec, "name", "") or "")
    if not str(getattr(spec, "user", "") or "") or not str(getattr(spec, "password", "") or ""):
        label = "天勤账户" if be == "tq" else "柜台账号"
        raise ValueError(f"账户「{who}」缺少{label}或密码")
    if be == "tq":
        return
    if getattr(spec, "is_stock", False):
        label = BACKEND_LABELS.get(be, str(be))
        raise ValueError(
            f"账户「{who}」是{label}账户。{label}只做期货，没有 A 股可报 —— "
            "股票策略请用天勤账户跑。"
        )


# 「没有训练产物」这件事多久之内不再重试 / 不再刷日志。产物是用户手动训的，
# 训完之后要么重启、要么下一次重试就会加载上 —— 每次 tick 都去试一遍 + 刷一行
# 日志，日志会被冲爆。
ML_PRED_RETRY = 60.0

# 实时订阅能直接用的周期。`PERIOD_SEC` 是下载链路的口径，只有 5m/1h/1d；
# 天勤的 `get_kline_serial` 本身支持 15m / 30m / 2h / 4h，实时段不用像本地历史
# 那样由 5m/1h 合并出来。
RT_SEC = barclock.RT_SEC          # 与 `barclock` 共用一份，免得两处漂

# 分平今 / 平昨的交易所：上期所、能源中心。其余（大商所 / 郑商所 / 中金所）
# 一律用 CLOSE，给它们发 CLOSETODAY 会被交易所整单打回。
SPLIT_OFFSET_EX = ("SHFE", "INE")

# 被交易所打回之后的冷却。`tick()` 跟着行情推送跑，一秒能来十几次；没有冷却
# 的话，一笔注定失败的委托一秒能重报十几遍 —— 日志刷屏，主线程直接卡死。
REJECT_COOL = 5.0
MAX_REJECTS = 3

# 非交易时段挂起时写在策略行上的那句话。判据见 `_open_now()`。
# 非交易时段的那句话。**只暂停下单，信号与理论仓位照算** —— 用户 2026-11-17：
# 「不能不显示，只是停止信号与下单，不是停止显示信号和理论持仓数量」
CLOSED_NOTE = "非交易时段：暂停下单（理论仓位照算）"
#: 没接账户时的占位说明。和 `CLOSED_NOTE` 分开：一个是「现在不能下单」，一个是
#: 「没有账户可下单」—— 前者等开盘会自己消失，后者要人去接账户
NO_ACCOUNT_NOTE = "未接账户：只算信号、不下单"
#: 没接账户时台账 / 委托账本用的键。**只此一处**：和界面那个
#: `gui.portfolio_page.NO_ACCOUNT_KEY` 不是一个东西（那边带括号，是分组名）
BOOK_NO_ACCOUNT = "未接账户"

#: 补衍生数据（展期 / 期权）的全局闸。DuckDB 单写者，几个 runner 一起补会互卡
_AUX_LOCK = threading.Lock()

#: 一次 `tick()` 里最多让几个槽位真正重算信号。重启 / 换行情源之后所有槽位
#: 都在等第一根 bar，一次性算完实测要 15 秒以上，工作线程被占住 —— 界面上的
#: 表格和曲线整段时间不动，就是用户说的「卡住」。分摊到几轮 tick（约 8 次/秒），
#: 首屏信号几秒内陆续出来，界面一直可交互。
SIG_BUDGET = 2

# 刚记完账（本策略成交）之后，账户持仓明细要过几百毫秒才跟上。这段时间里
# 账本比账户多是一时的，不能当成「被手工平了」去收窄 —— 见 `_trim`。
BOOK_GRACE = 3.0

# 「一键对齐」两次点击之间至少隔这么久。连点两下不该下发两轮。
ALIGN_GAP = 5.0

# 对齐单最多挂多久。超时就撤掉、清空在途列表，让策略恢复调仓 ——
# 否则一个挂在盘口外的限价单能让整个账户永远停在「对齐执行中」。
ALIGN_TIMEOUT = 30.0

# 报单发出去之后，柜台多久没给**任何**委托回报就该说话了。
#
# CTP 的 `OnRtnOrder` 是毫秒级回的，8 秒没有就是异常。为什么要单独一条：
# 账本里那行委托是**下单瞬间按「乐观」写的**（`status="ALIVE"` 是我们填的，
# 不是柜台说的），8 秒没被回报覆盖，就说明柜台根本没有这笔单 —— 而界面上
# 它长得跟「挂着等成交」一模一样。2026-11-23 融航就是：四笔 TL2612 发出去、
# 每笔挂满 30 秒「ALIVE」被撤，委托框里一笔都没有，用户只能问「到底发了没」。
#
# 🔴 这个值**必须比 `ALIGN_TIMEOUT` 小**，否则「柜台没回报」的判词还没写出来，
# 对齐那边已经先撤单并打印「ALIVE」了（30 秒撞 30 秒，实测永远看不到真相）。
ORD_ACK_SEC = 8.0

# 补历史没成功（刚订阅、序列还没填充）时隔多久再试。`tick()` 跟着行情推送跑，
# 一秒十几次；每次都把 3000 行转一遍 DataFrame 纯属白烧。
TOP_RETRY = 2.0

# 仓位比对多久重算一次。比对要把所有策略的理论净仓和整个账户的持仓对一遍，
# 账户里合约一多就是几百次属性读取；而快照是 0.4 秒推一次的 —— 每推一次都
# 重算，等于为了一个「秒级才变一次」的数字空转。5 秒一次够看，也够用户点
# 对齐时拿到的新鲜度（`align()` 会强制重算，见 `mismatch(fresh=True)`）。
MISMATCH_GAP = 5.0

# 持仓 / 委托引用多久重挂一次。见 `PortfolioRunner._reanchor()`：账户接口承诺
# 返回「活引用」，实盘里真的断过一次，而断了不报错。
POS_ANCHOR_SEC = 2.0

# 同一条不一致连续存在多久才往 `live.log` 写一行 `[MM]`。界面每 5 秒算一次，
# 写进去的必须是「查了三分钟还没动」的那种，否则日志全是噪音。
MM_LOG_SEC = 120.0

# 账户挂着、可持仓一直没读回来，多久抱怨一次 `[POS]`。见 `_pos_note()`：
# 这时候界面所有「一致」都是假的，必须把柜台的原始情况写进日志才查得动。
POS_LOG_SEC = 30.0


def default_margin_rate(symbol: str) -> float:
    """品种的默认保证金率。杠杆 = 1 / 保证金率。

    只在「数据管理 → 品种信息」里那一格是空/0 时兜底 —— 正常路径一律读表
    （见 `_terms`）。**同一个实现放在 `specs.margin_rate`**，回测也用它，
    两边兜底率必须一样，否则表里漏填一个数就又是两套手数。
    """
    return specs.margin_rate(symbol)


def bt_cfg(cfg):
    """实盘策略配置 → **回测那一份** `BacktestConfig`（手数的唯一入口）。

    `replay.make_bt()` 已经把这八件事映射好了：`sizing` / `capital`（←本金×比例）
    / `risk_amount` / `stop_mult` / `stop_atr_n` / `volume` / 本金 / 合约乘数与
    保证金率（读「品种信息」表，和 `_terms` 同一张）。实盘再自己写一遍就是第二
    个口径 —— 2026-11-19 那次「回测 4 手、实盘 1 手」就是这么来的。

    `market` 用 `replay.market_of()` 而不是 `cfg.market`：前者只看 `exchange`
    （空则看品种池），自检里拿 `SimpleNamespace` 造配置时也能判对；后者是
    `StrategyConfig` 上的属性，替身对象没有。

    每次现造，不缓存：`make_bt` 只是一串字典取值 + 一次 `symbols_info.info`
    （覆盖表进程内已缓存），比 `_terms` 原来那次查询不多什么。
    """
    return replay.make_bt(replay.market_of(cfg),
                          replay.resolve_allow_short(cfg), cfg)


def atr(df: pd.DataFrame, n: int = 14) -> float:
    """真实波幅均值（最后一根）。给「按止损金额定仓」反推手数用。

    和 `sbbox_mini` 里的算法一致：`max(h-l, |h-c[-1]|, |l-c[-1]|)` 的 N 根均值。
    """
    n = max(1, int(n or 14))
    if df is None or len(df) < 2:
        return 0.0
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    v = tr.rolling(n, min_periods=n).mean()
    val = float(v.iloc[-1]) if len(v) else 0.0
    return val if math.isfinite(val) and val > 0 else 0.0


# ------------------------------------------------------------------ 状态

#: 理论建仓价的三个来源，值越大越可信（见 `SlotState.theo_src`）。
#: `order` —— 策略实际下发的限价（对手价，已含滑点）；
#: `ib`    —— K 线内触发价（`ib_entry_long/short`）；
#: `sig`   —— 没有委托可查时按信号序列倒推的「下一根开盘 + 成本」。
THEO_SIG, THEO_IB, THEO_ORDER = "sig", "ib", "order"
_THEO_RANK = {THEO_SIG: 1, THEO_IB: 2, THEO_ORDER: 3}


def _as_float_seq(v, n: int):
    """把一列（Series / ndarray / None）统一成 `n` 个 float 的 list。

    长度对不上就返回 `None` —— **绝不按位置硬塞**：差一根就会把止损距离错配到
    另一根 K 线上，手数看着「有值」其实全是错的，比直接不给还糟。
    """
    if v is None:
        return [math.nan] * n
    try:
        s = pd.to_numeric(pd.Series(v), errors="coerce")
        a = s.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return None
    if len(a) != n:
        return None
    return [float(x) for x in a]


def ib_open_trade(df: pd.DataFrame, bt, e_long=None, e_short=None,
                  x_long=None, x_short=None, stop_d=None):
    """K 线内那一族：**照回测那台状态机**走一遍，返回最近一笔还没平的开仓。

    返回 `(开仓那根的下标, 不含滑点的成交基准价, 手数)`；走完整段还空仓 ⇒ `None`。

    🔴 为什么必须是「同一台状态机」而不是「从信号段起点倒推」：K 线内那一族的
    `signal` 列**不是持仓状态**。实测（7 品种 / 1h / 真实配置）——

    | 品种 | `signal` 段起点 | 回测真正开仓 |
    |---|---|---|
    | AG | 09-24 00:00 | **09-23 21:00**（比段起点还早） |
    | RB | 10-22 09:00 | **09-09 22:00**（早六周）|
    | C  | 11-10 14:00 | **11-10 09:00** |

    用户 2026-11-23 的原话：「signal 段起点如果是 10-22，那确实会很多错误的」。
    段起点只说明「从这根起连着一串同向信号」，和「哪根真正碰到触发价成交」
    没有关系 —— 中间可能止损过、可能离场过、可能反手过。

    🔴 「和回测一致」的另一半是**起点一致**：这份 `df` 必须就是回测喂进去的
    那一份（同样经过 `replay.window(..., range_years)`）。调用方
    `_signal_frame()` 已经保证了这一点，`replay.window()` 也改成按整日锚定
    起点，免得回测末根 15:00、实盘末根盘中而错开一根。

    这里刻意**不建 Trade / 不算净值**：只要状态和手数。实测 2.6 万根 1h 约
    二三十毫秒，配合调用方的「新 bar 才算」缓存，一小时一次，可以忽略。
    """
    n = len(df)
    if n < 2:
        return None
    try:
        o = [float(x) for x in df["open"].to_numpy(dtype=float)]
        hi = [float(x) for x in (df["high"] if "high" in df.columns
                                 else df["open"]).to_numpy(dtype=float)]
        lo = [float(x) for x in (df["low"] if "low" in df.columns
                                 else df["open"]).to_numpy(dtype=float)]
        scale = ([float(x) for x in pd.to_numeric(df["price_scale"], errors="coerce")
                  .fillna(1.0).to_numpy(dtype=float)]
                 if "price_scale" in df.columns else None)
    except (KeyError, TypeError, ValueError):
        return None
    up = _as_float_seq(e_long, n)
    dn = _as_float_seq(e_short, n)
    xl = _as_float_seq(x_long, n)
    xs = _as_float_seq(x_short, n)
    sd = _as_float_seq(stop_d, n)
    if up is None or dn is None or xl is None or xs is None or sd is None:
        return None
    # 「只做多 / 只做空」的方向裁剪和 `run_intrabar_backtest` 同一处、同一写法
    if getattr(bt, "long_only", False):
        dn = [math.nan] * n
    if getattr(bt, "short_only", False):
        up = [math.nan] * n

    mult = float(getattr(bt, "multiplier", 0) or 1.0)
    cash = float(getattr(bt, "initial_cash", 0) or 0.0)
    reentry = bool(getattr(bt, "intrabar_reentry_samebar", False))
    pos = 0
    lots = 0.0
    entry_px = 0.0
    entry_raw = 0.0
    entry_bar = -1
    stop_px = math.nan
    closed_i = -1

    def _slip(px: float, i: int) -> float:
        try:
            return float(bt.slip_amount(px, scale[i] if scale is not None else 1.0))
        except Exception:                # noqa: BLE001 - 成本算不出就当 0
            return 0.0

    def _fee(q: float, px: float) -> float:
        try:
            return float(bt.fee_amount(abs(q), mult, px))
        except Exception:                # noqa: BLE001
            return 0.0

    for i in range(n):
        oi, hii, loi = o[i], hi[i], lo[i]
        # 进场当根不出场（和引擎同一个 `i > entry_bar` 守卫，见
        # `BacktestConfig.intrabar_samebar_stop` 上面那段）
        live = pos != 0 and i > entry_bar
        # ① 上一根收盘给的出场：本根开盘执行
        if live and i > 0:
            want = (xl[i - 1] > 0.5) if pos == 1 else (xs[i - 1] > 0.5)
            if want:
                sp = _slip(oi, i)
                fill = oi - sp if pos == 1 else oi + sp
                cash += lots * mult * (fill - entry_px) - _fee(lots, fill)
                pos, lots, entry_px = 0, 0.0, 0.0
                stop_px, closed_i = math.nan, i
        # ② 已有仓位的止损（开盘已穿过就按开盘价，更差）
        if live and math.isfinite(stop_px):
            if pos == 1 and loi <= stop_px:
                raw = min(oi, stop_px)
                sp = _slip(raw, i)
                fill = raw - sp
                cash += lots * mult * (fill - entry_px) - _fee(lots, fill)
                pos, lots, entry_px = 0, 0.0, 0.0
                stop_px, closed_i = math.nan, i
            elif pos == -1 and hii >= stop_px:
                raw = max(oi, stop_px)
                sp = _slip(raw, i)
                fill = raw + sp
                cash += lots * mult * (fill - entry_px) - _fee(lots, fill)
                pos, lots, entry_px = 0, 0.0, 0.0
                stop_px, closed_i = math.nan, i
        # ③ 空仓才看本根有没有触发入场；本根刚平过仓就不再开新仓
        if pos == 0 and (reentry or closed_i != i):
            ready = math.isfinite(sd[i]) and sd[i] > 0
            long_ok = ready and math.isfinite(up[i]) and hii >= up[i]
            short_ok = ready and math.isfinite(dn[i]) and loi <= dn[i]
            if long_ok and short_ok:
                long_ok = short_ok = False     # 两头都打穿，分不出先后
            if long_ok or short_ok:
                d = 1 if long_ok else -1
                raw = max(oi, up[i]) if d > 0 else min(oi, dn[i])
                sp = _slip(raw, i)
                fill = raw + sp if d > 0 else raw - sp
                q = float(size_lots_intrabar(bt, fill, cash, sd[i], mult))
                if q > 0:
                    cash -= _fee(q, fill)
                    lots = q if d > 0 else -q
                    pos = d
                    entry_px = float(fill)
                    entry_raw = float(raw)
                    entry_bar = i
                    # 止损绑**实际成交价**，和引擎一致
                    stop_px = fill - sd[i] if d > 0 else fill + sd[i]

    if pos == 0 or entry_bar < 0:
        return None
    return (int(entry_bar), float(entry_raw), int(round(abs(lots))))



@dataclass
class PendingOrder:
    """一笔正在追的委托。"""

    order_id: str
    symbol: str
    direction: str
    offset: str
    volume: int
    price: float
    ts: float
    retries: int = 0
    phase: str = "live"        # live / cancelling
    # 平今平昨被柜台打回后，是否已经换过一次（CLOSE ↔ CLOSETODAY）。
    # 只翻一次：两个都说没仓，就是真的没仓，交给计数 + 停机，别来回弹。
    flipped: bool = False


@dataclass
class SlotState:
    """一个策略槽位的运行状态。界面只读这个，不碰天勤对象。"""

    cfg: SStore.StrategyConfig
    running: bool = False
    signal: int = 0
    lots: int = 1              # 一次满仓对应几手（按资金 / 止损金额算出来）
    #: **下单目标**（手）。普通策略它就等于「理论净仓」= signal × lots；
    #: K 线内那一族（`sbbox_ib` 等）只有盘中碰到触发价才会变 —— 它们的理论
    #: 净仓另算（`_theo_qty`，= signal × lots，与回测同一口径），因为「策略
    #: 认为该持多少」和「现在该不该按这个价发单」是两件事。
    target: int = 0
    # **这条策略自己**在各合约上的净仓（手），按自己委托的成交累计，落盘在
    # `positionbook`。不是账户净仓 —— 账户是所有策略 + 手工单 + 别的平台的和，
    # 拿它当自己的仓，同品种两条策略就会互相把对方的仓当成自己的。
    held: dict = field(default_factory=dict)
    #: 每个合约的**持仓均价**（开仓方向上的加权成本），和 `held` 同一批键。
    #: 只用来算「持仓盈亏」，不参与任何下单判断 —— 算错了顶多是那一列数字不准，
    #: 不会变成多下一手。落盘在 `positionbook` 的 `_costs` 段。
    cost: dict = field(default_factory=dict)
    #: 本交易日**已经落袋**的平仓盈亏（元）。当日盈亏 = 这个 + 持仓盈亏。
    #: 不落盘：重启后从 0 起算，那时界面上就只剩持仓那一半 —— 宁可少算，
    #: 也不能把上一天的落袋盈亏算进今天。换交易日自动清零（`daylimit.trading_day`）。
    real_day: float = 0.0
    real_day_of: str = ""
    #: 理论建仓价 —— 「持仓盈亏 / 当日盈亏」两列**只读这三个数 + 行情**，
    #: 既不读台账（`held`/`cost`/`real_day`），也不读账户成交。
    #: 用户 2026-11-20 明确：这两列是「策略理论上」的数，跟实际下单无关。
    #: `theo_sig` 是这两格对应的信号方向，方向变了就重算。
    #:
    #: 🔴 `theo_px` **由策略自己写**（用户 2026-11-20 第二次纠正：
    #: 「理论开仓价格，应该是从实盘策略中直接获取，不然如果 k 线中开仓，又算不对了」）。
    #: 显示层不许另立一套「下一根开盘 + 成本」的规则去猜 K 线内那一族的价格 ——
    #: 触发价是盘中算出来的，推出不来。谁写、按什么优先级覆盖，看 `theo_src`：
    #:
    #: | `theo_src` | 谁写的 | 含义 | 含成本吗 |
    #: |---|---|---|---|
    #: | `order` | `_send()` | 这一笔**实际下发**的限价（对手价）| 已含滑点，不再叠 |
    #: | `ib`    | `_intrabar_tick()` | K 线内**触发价**（`ib_entry_long/short`）| 叠一层每边成本 |
    #: | `sig`   | `_note_theo_entry()` | 没有委托可查时的**推断**（下一根开盘 + 成本）| 叠一层每边成本 |
    #:
    #: 优先级 `order` > `ib` > `sig`，同级不覆盖（同向加仓不许改建仓价）。
    #:
    #: **不落盘**：重启后内存里是空的，`_note_theo_entry` 会按同一套信号倒推
    #: 重新算出来（`sig` 档）。故意不落盘 —— 落盘的错值（比如库被锁时按降级帧
    #: 算出来的那个）会一直挂着自愈不了，这正是 2026-11-20 那次的教训。
    theo_px: float = 0.0
    theo_day: str = ""
    theo_sig: int = 0
    #: 这个 `theo_px` 是哪一路给的，见上表。空串 = 还没记过。
    theo_src: str = ""
    #: 这一笔的**开仓手数**（正数，方向看 `signal`）—— 与 `theo_px` 配对的另一半。
    #:
    #: 🔴 2026-11-23 用户：「理论列需要按照**开仓的手数**来算，不能根据当前 K 线
    #: 来算，实际上应该就从开仓那边获取，而不是自己擅自计算」。
    #:
    #: 以前 `_theo_qty` 用的是 `st.lots`，而 `st.lots` **每根 bar 都按当根 ATR
    #: 重算**。risk 口径下 `手数 = 止损额 ÷ (ATR × 倍数 × 乘数)`，ATR 从 12 掉到
    #: 4.7，同一笔仓的手数就从 39 手漂到 106 手 —— 界面上「理论」列每根都在变，
    #: 用户拿它和回测交易记录里那一笔（39 手）对账，永远对不上。
    #:
    #: 回测口径是：**开仓那一根算一次，之后固定到平仓**（`size_lots(cfg, 开盘价,
    #: 现金, 开仓那根的 ATR)`）。这里的 `theo_lots` 就是那个数，由
    #: `_note_theo_entry()` 交给 `ib_open_trade()` / `_entry_lots_at()` 算出来，
    #: 一波之内不再重算；方向翻了 / 配置改了才重算。
    #:
    #: 🔴 K 线内那一族（`sbbox_ib` / `im_orb`）**不能拿 `signal` 段起点推**：
    #: 那一列不是持仓状态，段起点和真正的开仓那根毫无关系（RB 1h 实测差六周）。
    #: 现在整段交给 `ib_open_trade()` —— 照 `run_intrabar_backtest` 那台状态机
    #: 走一遍，取「最近一笔还没平」的开仓。
    #:
    #: 0 = 还没算出来（历史不够、机器学习那族在等信号），此时退回 `lots` 兜一下。
    theo_lots: int = 0
    #: `theo_lots` 是给哪个方向算的（+1/-1）。方向一变就作废重算。
    theo_lots_sig: int = 0
    #: `theo_lots` 的来源：`sig` = 按信号波段起点那根 K 线推的；`ib` = K 线内
    #: 真的触发了、用**那一刻实际下单的手数**刷新的（更可信，同级不覆盖）。
    theo_lots_src: str = ""
    book_ts: float = 0.0       # 最近一次账本变动的时刻，见 `_trim` 的宽限期
    # 本进程是否已经完成**加载时的首轮同步**。加载（启动 / 重新启用）后的第一轮
    # 只把账本摆成「账户里现成的仓」并记下来，**一手都不发** —— 用户 2026-11-17：
    # 「加一个策略加载的时候不要自动去根据台账去做交易」。
    # 之后才按 `target - held` 调仓；换月仍是唯一无条件自动的动作。
    based: bool = False
    # 加载（启动 / 重新启用）时那一轮的目标仓。**目标没变就一手都不发** ——
    # 这是「加载时不要自动根据台账交易」的判据本身：加载后账本和账户差多少
    # 都不管，等信号真的把目标改了才按台账调仓。换月不走这条（无条件自动）。
    based_target: int = 0
    underlying: str = ""       # 当前真实主力合约
    last_bar: object = None
    # 这一轮算信号用的是哪份行情：「本地 N 根 + 实时 M 根」/「退回实时原始价」/
    # 「K 线缺 N 个交易日」。目标仓一变就带着它落进 live.log —— 用户问
    # 「实盘开了仓、回测却是空仓，是不是信号在闪」时，光有 target 一条数字
    # 什么都证明不了，必须知道**当时喂进去的是什么**（2026-11-19）。
    frame_note: str = ""
    #: 这一轮的行情帧**是不是可信**（`_signal_frame()` 给）。
    #: `False` = 降级帧（没有本地前复权历史，拿实时原始价顶的；或者库没补上）。
    #: 那种帧上的建仓价 / 手数和回测是两码事 —— 2026-11-23 白银基本面分位组合
    #: 就这么把「回测 11-18 已平仓、应该是 0」显示成 −2。所以降级帧只用来**算
    #: 信号是否照常跑**，**不许写理论值**（`_note_theo_entry` 开头就挡掉）。
    frame_ok: bool = True
    error: str = ""
    last_action: str = ""
    pending: list = field(default_factory=list)
    retired: list = field(default_factory=list)   # 换月后要清仓的老合约
    rejects: int = 0           # 连续被交易所打回的次数
    reject_ts: float = 0.0     # 最近一次被打回的时间，用于冷却
    reject_msg: str = ""       # 最近一次被打回的原因
    # 机器学习实盘：信号在后台算，算出来之前**不许动仓**（见 `_MlSignals`）。
    # 这里为 False 时 `tick()` 跳过调仓 —— 拿「还没算出来」当信号 0 会立刻把
    # 上次留下的仓平掉，重启一次就白交一遍手续费
    ml_ready: bool = True
    ml_age: float = 0.0        # 手上这份信号是多久以前算出来的（秒）
    # 机器学习为什么还没出信号。三种状态必须分开，界面上长得一样、
    # 等待时间差 10 倍：
    #   ""      —— 还在算（快路径，几秒）
    #   "retrain" —— 没有可用产物，退回后台重训（几十秒到十几分钟）
    #   其他文本 —— 算不出来的**真实原因**（永远等不到）
    ml_why: str = ""
    # 这次提交走的是快路径（有训练产物、只预测）还是慢路径（后台重训）。
    # 慢路径没出结果时说「重训中」，快路径说「计算中」—— 两者等待时间差 10 倍
    ml_fast: bool = False

    # ---- K 线内进出场（`StrategySpec.intrabar` 的规则策略用） ----
    # 这根 bar 的触发价算好没有。**每根 bar 只算一次**：算一次要把整个
    # `_signal_frame` 跑一遍策略，几百毫秒，放在每个 tick 里做就是自残
    ib_armed: bool = False
    # 这根 bar 已经按触发价进过一次场。防止「成交回报没到 → 又开始比价 →
    # 再发一遍」这种重复下单；新的 bar 会把它清掉
    ib_fired: bool = False
    ib_entry_long: float = float("nan")    # 本根做多触发价
    ib_entry_short: float = float("nan")   # 本根做空触发价
    ib_stop_dist: float = float("nan")     # 止损距离（价格单位，= 倍数 × ATR）
    # 这一笔**实际挂的**止损价：绑在成交时的触发价上，不跟着 bar 走。
    # 重启会丢，`_ib_arm` 会按标记价补挂一条并写日志
    ib_stop: float = float("nan")
    # 已经决定平仓、但成交还没回来（或者回报慢）。下一根 bar 的 `_ib_arm`
    # 看到 `st.held` 还是老仓时**不能**把它重新捡回来：那等于把刚下的平仓
    # 单撤掉，止损白打了。成交回报到、`held` 归零之后自动清掉
    ib_exit_pending: bool = False

    def title(self) -> str:
        return self.cfg.title()


# ------------------------------------------------------------------ 引擎


def hole_days(df: pd.DataFrame) -> list:
    """这段 K 线里**缺掉的交易日**（拿全库日历比）。

    为什么必须有这条：本地库落后几天时，`merge_live` 会把实时段直接贴在旧的
    最后一根后面 —— 中间那几天凭空消失，索引看着还是连的。指标跨着这个断口算
    （60 根动量、上一交易日波幅、ATR），出来的信号和回测**不一样**，而且完全
    没有痕迹。2026-11-17 的现场就是它：IF 的 5m 库停在 11-10，柜台账户的历史
    也只有从库里种进去的 300 根（同样停在 11-10），实时从当天接上；同一个策略
    在两个账户上一个多 1 手、一个 0 手，而回测按完整数据算出来是持仓的。

    不能按自然日判：春节、国庆一周不开市，按自然日会把长假当成缺口天天误报。
    日历取 `tqstore.calendar()`（全库日线出现过的交易日）。

    返回缺掉的交易日（空表 = 这段是完整的）。日历读不到时也返回空表 ——
    「不知道」不能当成「缺了」，那会把所有策略一起停掉。
    🔴 判空只能用 `len(df)`：`DataFrame.empty` 在**没有列**时也是 True（pandas 的
    规定是「任一轴长度为 0」），拿它判空会把只有索引的帧当成没数据，缺口静默漏掉。
    """
    if df is None or len(df) == 0:
        return []
    from ttrade import tqstore

    cal = tqstore.calendar()
    if not cal:
        return []
    have = {pd.Timestamp(x).date() for x in tqdata.trading_days(df.index)}
    if not have:
        return []
    # 上限卡到日历最后一根：当天的日线还没落库，别把「今天」当成缺口
    hi = min(max(have), max(cal))
    return sorted(d for d in cal if min(have) <= d <= hi and d not in have)


# `expected_last_bar` 的实现在 `ttrade/barclock.py`：补库、合并、算信号三处
# 必须用同一条「这根 bar 到几点算走完」的口径，各写各的一定会漂。这里只做
# 转发，保持 `strategy_runner.expected_last_bar` 这个老入口还能用。
expected_last_bar = barclock.expected_last_bar


class LiveLog:
    """实盘诊断日志，落盘到 `logs/live.log`。

    **为什么要有这个：** 界面上只显示一行「训练中」，出问题时用户没法判断
    到底是「还在算」、「产物没找到」、「配置对不上退回重训」还是「模型废了」。
    这些原因在 GUI 的日志控件里（内存，关窗就没了），而且用户不会去看。

    落盘的日志能回答三个问题：
    - 走的是快路径（只预测）还是慢路径（后台重训）？慢路径 19 个品种要十几分钟，
      界面上看和「卡住」一模一样；
    - 产物为什么用不了（配置对不上？文件不在？）；
    - 信号算不出来时，后台线程记下的**真实原因**是什么。

    写文件必须**不抛异常** —— 日志失败绝不能把实盘带崩。
    """

    def __init__(self, path: str | None = None) -> None:
        self._lock = threading.Lock()
        self._path = path
        if self._path is None:
            try:
                from ttrade import paths

                self._path = os.path.join(paths.log_dir(), "live.log")
            except Exception:  # noqa: BLE001
                self._path = None

    def write(self, tag: str, msg: str) -> None:
        if not self._path:
            return
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{tag}] {msg}\n"
        try:
            with self._lock:
                # 只留最近 512KB —— 实盘一天几万行，不能把硬盘写满
                if os.path.exists(self._path) and os.path.getsize(self._path) > 512 * 1024:
                    try:
                        with open(self._path, "r", encoding="utf-8") as fh:
                            lines = fh.readlines()[-2000:]
                        with open(self._path, "w", encoding="utf-8") as fh:
                            fh.writelines(lines)
                    except Exception:  # noqa: BLE001
                        pass
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line)
        except Exception:  # noqa: BLE001
            pass


class MlSignals:
    """机器学习实盘的信号缓存：后台算，tick 只读。

    为什么不能像规则策略那样在 tick 里算：机器学习每根 bar 都要把整段历史重走
    一遍（65 个因子 × 十几万根，实测一个品种 30~40 秒）。放进 tick 循环等于把
    整条连接卡住 —— 一个账户的策略共享同一个 api，卡住就是全部停摆。

    所以挪到一条单独的线程：

    - `submit()` 只更新「这个品种最新该算哪份行情」，同一品种排队里最多一份，
      新的直接覆盖旧的（算到一半又来了新 bar，算旧那份没有意义）；
    - `signal()` 给 tick 线程读，还没算出来时返回 None，调用方**不许动仓**；
    - `info()` 报出这份信号有多旧，界面上要能看出来「它其实落后了几根」。

    算的是**和回测同一个函数**（`run_factor_strategy`）、同一段历史（`_signal_frame`
    已经按 `range_years` 切过）。同源同参是硬要求：两边各写一份，「回测赚钱、
    实盘亏钱」就永远说不清是谁的问题。代价是慢，所以用「算不过来就用上一份」
    兜底，而不是换个近似算法。
    """

    def __init__(self, log=None, diag=None) -> None:
        self._log = log
        self._diag = diag            # LiveLog（落盘诊断日志），可为 None
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}     # key -> {df, keys, cfg, predictor}
        self._out: dict[str, dict] = {}      # key -> {signal, ts, bars, cost}
        self._err: dict[str, str] = {}       # key -> 最近一次失败的原因
        self._q: queue.Queue = queue.Queue()
        self._queued: set[str] = set()
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True, name="ml-live")
        self._t.start()

    # ---- 给 tick 线程 ----

    def submit(self, key: str, df, cfg, keys: list, predictor=None,
               extra: dict | None = None) -> None:
        """提交一次信号计算。

        `predictor` 是 `ml_artifact.LivePredictor` —— 有它就走「只预测」快路径
        （加载好的模型 + 续算状态，**不训练**）；没有就退回「后台重训」慢路径
        （每根 bar 把整段历史重走一遍 walk-forward）。

        两条路径算的是**同一个函数**的两条路，信号口径一致（见 `ml_artifact`）。
        """
        if df is None or len(df) < FACTOR_MIN_BARS:
            return
        with self._lock:
            self._jobs[key] = {"df": df, "cfg": cfg, "keys": list(keys),
                               "predictor": predictor, "extra": dict(extra or {})}
            if key in self._queued:
                return            # 已经在排队，直接换掉内容就行
            self._queued.add(key)
        self._q.put(key)

    def signal(self, key: str):
        """(信号, 年龄秒)。还没算出来时返回 (None, 0)。"""
        with self._lock:
            hit = self._out.get(key)
        if not hit:
            return None, 0.0
        return int(hit["signal"]), max(0.0, time.time() - hit["ts"])

    def info(self, key: str) -> dict:
        with self._lock:
            return dict(self._out.get(key) or {})

    def error(self, key: str) -> str:
        """最近一次「算不出信号」的原因。

        界面上不能一直显示「机器学习训练中」—— 那是在说「还在算」，但实际
        可能是**永远算不出来**（模型训废了、历史预测不够、因子全缺）。
        两种情况长得一样，用户只能干等。
        """
        with self._lock:
            return str(self._err.get(key) or "")

    def pending(self) -> int:
        return self._q.qsize()

    def stop(self) -> None:
        self._stop = True
        try:
            self._q.put_nowait("")          # 叫醒阻塞在 get() 上的线程
        except Exception:  # noqa: BLE001
            pass

    # ---- 后台线程 ----

    def _loop(self) -> None:
        while not self._stop:
            key = self._q.get()
            if self._stop:
                break
            if not key:
                continue
            with self._lock:
                self._queued.discard(key)
                job = self._jobs.get(key)
            if job is None:
                continue
            t0 = time.time()
            try:
                predictor = job.get("predictor")
                if predictor is not None:
                    # ---- 快路径：加载好的模型，只预测，不训练 ----
                    r = predictor.predict(job["df"])
                    if not r.get("ok"):
                        # 算不出来**不是**信号 0：不写缓存，调用方继续用上一份
                        # 信号（或者保持不动仓）。拿「算不出来」当 0 会把上次
                        # 留下的仓平掉，重启一次就白交一遍手续费
                        why = str(r.get("error") or "算不出信号")
                        with self._lock:
                            self._err[key] = why
                        if self._log is not None:
                            self._log(f"机器学习信号暂不可用（{key}）：{why}")
                        if self._diag is not None:
                            self._diagw("ML", f"{key} 快路径算不出来：{why}")
                        continue
                    sig = int(r["signal"])
                    if self._diag is not None:
                        self._diagw(
                            "ML", f"{key} 快路径 ok signal={sig} "
                                  f"score={r.get('score'):.4f} "
                                  f"耗时={time.time() - t0:.2f}s bar={len(job['df'])}")
                else:
                    # ---- 慢路径：后台重训（没有训练产物时的兜底）----
                    if self._diag is not None:
                        self._diagw("ML", f"{key} 走慢路径（后台重训），"
                                              f"bar={len(job['df'])} —— 这条要几十秒")
                    _fc = replay.factor_config(job["cfg"].factor_cfg or {})
                    out = fstrat.run_factor_strategy(
                        job["df"], "ml", job["keys"], _fc,
                        extra=job.get("extra") or {},
                        base_period=getattr(job["cfg"], "period", "") or "",
                    )
                    sig = out.df["signal"].iloc[-1]
                    sig = 0 if pd.isna(sig) else int(sig)
                    if self._diag is not None:
                        self._diagw("ML", f"{key} 慢路径完成 signal={sig} "
                                              f"耗时={time.time() - t0:.1f}s")
            except Exception as exc:  # noqa: BLE001
                why = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._err[key] = why
                if self._log is not None:
                    self._log(f"机器学习实时信号计算失败（{key}）：{why}")
                if self._diag is not None:
                    self._diagw("ML", f"{key} 抛异常：{why}")
                continue
            with self._lock:
                self._err.pop(key, None)
                self._out[key] = {"signal": sig, "ts": time.time(),
                                  "bars": int(len(job["df"])),
                                  "cost": time.time() - t0,
                                  "fast": predictor is not None}


class PortfolioRunner:
    """行情 + 一组策略（+ 一个**可选**的账户）。

    纯 Python，不依赖 Qt —— 界面层用 QThread 包一层（见 `gui/portfolio_page.py`）。

    **行情和账户是两件事**（2026-11-20 重构）。行情走全局共享的
    `MarketFeed`（天勤 / CTP / 融航 三选一），账户只负责下单和查持仓：

    - 没接账户也能跑：信号照算、理论仓位照显示、台账照记，**一笔不发**。
      实盘账户日终要登出，策略不能跟着断 —— 7×24 靠的就是这一条。
    - 账户中途登录 / 登出不重启策略：走 `attach_account()` / `detach_account()`，
      槽位、K 线缓存、台账全保留。
    - 以前 `self.api` 一个对象兼着行情和交易，柜台账户（`simnow_7x24`）拿不到
      中金所行情时，策略「算不出信号」和「真的看空」在界面上长得一样。
    """

    def __init__(self, spec: AccountSpec | None = None,
                 chase: SStore.ChaseConfig | None = None,
                 monitor: SStore.MonitorConfig | None = None) -> None:
        # 账户可以为空：`spec` 表达的是「绑定了哪个户」，`api` 表达的是「现在
        # 连没连上」。两者分开之后，账户登出不会把策略一起带走。
        self.spec = spec
        self.chase = chase or SStore.load_chase()
        #: 监控器（一键对齐）多久重读一次账户。走 property，**改完立刻生效**
        self._monitor = monitor or SStore.load_monitor()
        # 监控器（一键对齐）多久重读一次账户。见 `MonitorConfig` / `wait()`
        self._acct_sec = max(1.0, min(60.0, float(
            getattr(self._monitor, "refresh_sec", 5) or 5)))
        # 上一次真的重读账户的时刻（单调钟）。0 = 还没读过，第一次立刻读
        self._acct_at = 0.0
        #: 交易连接（账户）。None = 没绑定 / 已登出 —— **策略照跑**。
        self.api = None
        #: 行情连接（全局共享，跨账户跨策略一份）。`start()` 里**先起它**。
        self.feed = None
        #: 上次用的行情源编号。切源后缓存的 serial 全失效，靠它发现并清缓存
        self._feed_gen = -1
        self.last_error = ""
        self._slots: dict[str, SlotState] = {}
        # 多品种：父 sid -> 子 sid 列表，子 sid -> 父 sid（反查），父 sid -> 父配置。
        # 一条 19 品种的策略在这里是 19 个槽位 + 1 条映射，界面上仍然是一行
        self._groups: dict[str, list[str]] = {}
        self._child_of: dict[str, str] = {}
        self._parents: dict[str, SStore.StrategyConfig] = {}
        # 机器学习实盘的信号缓存（后台线程，见 `MlSignals`）
        self._diag = LiveLog()
        self._ml = MlSignals(log=self._log, diag=self._diag)
        # 机器学习「只预测」快路径的预测器缓存：槽位键 -> ("ok", 预测器) 或
        # ("miss", None, 试过的时刻)。见 `_predictor_for`
        self._ml_pred: dict[str, tuple] = {}
        self._ml_miss: dict[str, str] = {}
        # 上面那张缓存是**跨线程**读的（信号在 worker 线程算，快路径缓存也可能
        # 在别处被填），要加锁。用 RLock 而不是 Lock：`_predictor_for` 里
        # 可能间接调到别处也拿这把锁的地方，普通 Lock 会死锁。
        self._lock = threading.RLock()
        self._klines: dict[str, object] = {}
        self._quotes: dict[str, object] = {}
        self._pos = None
        self._orders = None
        self._log_fn = None
        # 交易所对日内报单 / 撤单次数有硬上限，超了会限制交易。按账户按交易日数，
        # 落盘 —— 重启程序不清零，见 ttrade/daylimit.py
        self.limit = daylimit.DayLimit(
            self.spec.label() if self.spec is not None else BOOK_NO_ACCOUNT)
        self._halted = ""
        # 用户**主动**在界面上登出了账户（≠ 没接上 / 接入失败）。界面「刷新」
        # 会把「没接上」的账户接回来，但绝不能把用户明确登出的又偷偷登回去 ——
        # 靠这个标记分开。见 `detach_account()` / `snapshot()["user_detached"]`
        self._detached_by_user = False
        # 「一键对齐」发出的单放在这个槽位里。它只当在途单的容器，不参与信号：
        # 对齐是**一次性动作**，发完就交给追单，成交 / 撤完就结束。没有它的话，
        # 对齐的单没人记，`_rebalance` 下一 tick 看到仓位还没变又发一遍 ——
        # 一秒十几次 tick，一笔对齐能变成十几笔委托。
        self._align_slot = SlotState(cfg=SStore.StrategyConfig(
            sid="__align__", name="一键对齐", period="1d"))
        self._align_ts = 0.0
        # 这一轮对齐真正发出去过的合约。超时撤单时按它判断「账本要不要回摆」，
        # 也方便把「点了 10 个只发了 5 个」和「发了但没成交」分开看。
        self._align_touched: set[str] = set()
        # 已经为哪几笔委托写过「柜台没回委托回报」。每单只写一次，见 `_ack_note()`
        self._ack_warned: set[str] = set()
        # 上一次算出来的比对结果和它的时间戳。见 `MISMATCH_GAP`。
        self._mm_cache: list[dict] = []
        self._mm_ts = 0.0
        # 每个不一致合约「第一次看到它不一致」的时刻，以及已经抱怨过没有。
        # 只用来写 `[MM]` 诊断行，见 `MM_LOG_SEC`
        self._mm_since: dict[str, float] = {}
        self._mm_logged: set[str] = set()
        # 「账户持仓一直没读回来」这件事开始于什么时候，以及上次抱怨的时刻。
        # 见 `_pos_note()`
        self._pos_blind_at = 0.0
        self._pos_warned = 0.0
        # 上一次重挂持仓引用的时刻。见 `POS_ANCHOR_SEC` / `_reanchor()`
        self._pos_ts = 0.0
        # 界面想看的实时 K 线。放这里而不是让界面直接读 `_klines`：那是天勤的
        # 活引用，只有 worker 线程能动，跨线程读会撞上它正在被写的时刻。
        self._want: list[str] = []
        # 已经补过历史的 (品种, 交易所, 源周期)。一台机器开着几十条策略时，
        # 同一个品种会被好几条共享，补一次就够了 —— 每补一次都要写一次库
        self._topped: set[tuple[str, str, str]] = set()
        # (品种, 交易所, 源周期) -> 上次尝试的时刻。见 `TOP_RETRY`
        self._top_try: dict[tuple[str, str, str], float] = {}
        # 上次失败的原因。同样的原因只打一条日志，不然每 2 秒刷一行
        self._top_err: dict[tuple[str, str, str], str] = {}
        # 算信号用的前复权基准，(品种_交易所@周期) -> qfq 历史。实盘必须和回测
        # 同一份复权口径（见 `_signal_frame`）。主连换月时失效重载。
        self._qfq_base: dict[str, pd.DataFrame] = {}
        # 读 `_qfq_base` 那一刻库文件的 mtime。**库变了就整份重读** —— 用户
        # 「数据管理」补完历史，实盘不该还要重启一次才用上新数据；而补库往往
        # 只是覆盖了库尾的几根（行数不变），靠「库尾比手上这份新」是判不出来的
        self._qfq_stamp: float = 0.0
        self._qfq_stamp_at: float = 0.0
        # K 线缺交易日的提示（键同 `_qfq_base`），同样的原因只记一次
        self._gap_note: dict[str, str] = {}
        # 第一次跑持仓账本（升级上来的老用户，或者本机第一次跑）。这种时候账户里
        # 可能已经有仓了，账本从 0 开始会把目标再开一遍 —— 见 `_adopt_if_needed`。
        self._fresh_book = not positionbook.exists()
        self._adopted: set[str] = set()
        self._aux_warned: set[str] = set()
        # 因子被剔除只提示一次（按 sid 去重），理由见 `_note_dropped`
        self._factor_warned: set[str] = set()
        # 已经落过盘的槽位错误：(sid, 错误原文)。见 `snapshot`
        self._err_logged: set[tuple] = set()
        # 每槽位「上一次记过的状态说明」。同一条说明只记一次 —— 每 tick 都刷会把
        # 日志淹掉，而这类说明（行情没就绪 / 历史不够）恰恰要让人一眼看到
        self._notes: dict[str, str] = {}

    # ---- 生命周期 ----

    def on_log(self, fn) -> None:
        self._log_fn = fn

    def _log(self, msg: str) -> None:
        if self._log_fn is not None:
            try:
                self._log_fn(msg)
            except Exception:  # noqa: BLE001
                pass

    def _diagw(self, tag: str, msg: str) -> None:
        """落盘诊断日志，**带账户标签**。

        🔴 为什么必须带（2026-11-23 那次）：一个组合可以同时连好几个账户，
        每个账户一个 `PortfolioRunner`、各写各的 —— 但落的是**同一个**
        `live.log`，两个线程的行会交错。那天排查「一键对齐只发了几笔」，日志里
        `ALIGN` 行和 `ORD` 行错开一位，同一个合约还出现两遍（两个账户各一遍），
        完全分不清哪条属于谁，白猜了半小时。带上账户名之后一眼就能按账户切开。
        """
        diag = getattr(self, "_diag", None)
        if diag is None:
            return
        lab = ""
        try:
            lab = self.spec.label() if self.spec is not None else ""
        except Exception:  # noqa: BLE001 - 标签取不到就别带，日志不能因此丢
            lab = ""
        diag.write(tag, f"[{lab}] {msg}" if lab else msg)

    def _note_once(self, st: SlotState, msg: str) -> None:
        """同一条状态说明只记一次，并挂到 `st.last_action` 上让界面看得到。

        2026-11-20 那次「盘中出了信号、重载后消失」里，`_refresh_signal` 和
        `_signal` 各有一处**静默 return** —— 行情订不上 / 历史不够时界面显示的是
        上一次算出的目标，和「策略真的看空」长得一模一样，日志里一条线索都没有。
        """
        if self._notes.get(st.cfg.sid) == msg:
            return
        self._notes[st.cfg.sid] = msg
        st.last_action = msg
        self._log(f"{st.title()}：{msg}")

    # 用到这几类因子的策略，启动前要把衍生快照补到最新
    _AUX_PREFIXES = ("roll_", "iv_", "near_oi", "pcr")

    def _uses_aux(self) -> bool:
        for st in self._slots.values():
            # 「规则 + 因子过滤」的因子同样要衍生列，不能只看 factor 族
            uses = st.cfg.family == "factor" or (
                st.cfg.family == "rule" and bool(st.cfg.factor_mode or ""))
            if uses:
                for k in (st.cfg.factor_keys or []):
                    if k.startswith(self._AUX_PREFIXES):
                        return True
        return False

    def _aux_recent(self) -> bool:
        """快照都够新吗？只要有一个品种落后就返回 False。

        没有这道检查的话，每天第一次启动都要跑一遍 `update_futures` ——
        94 个品种逐个请求，实测 125 秒，哪怕一根新数据都没有。判据用自然日差
        而不是交易日历：差 <= 3 天覆盖周末，节假日会多跑一次（无害）。
        """
        try:
            from datetime import date

            from ttrade.auxdata import _default_dir, load_curve

            aux_dir = _default_dir()
            for st in self._slots.values():
                c = load_curve(aux_dir, st.cfg.symbol, st.cfg.exchange)
                if c.empty:
                    continue
                gap = (date.today() - pd.Timestamp(c.index[-1]).date()).days
                if gap > 3:
                    return False
            return True
        except Exception:  # noqa: BLE001
            return False                # 判不了就按「落后」处理，宁可多补一次

    def _ensure_aux(self) -> None:
        """启动前把展期 / IV 的日频快照补到最新。

        这两个因子在分钟线上只能用「上一个交易日」的收盘快照 —— 回测和实盘
        同一个口径，盘中实时算反而是另一种量。但快照本身得有人去下：用户几天
        没开程序，它就停在几天前，实盘拿着上周的期限结构算信号，有值但是死的。

        **只在用到这些因子时才补**：没勾展期的策略不该为它白等几十秒。
        补失败不拦启动 —— Tushare 挂了也得能交易，退回本地快照就是。
        """
        if not self._uses_aux():
            return
        if self._aux_recent():
            return                      # 已经是最新的，别每天白跑两分钟
        try:
            from ttrade import updater

            self._log("检查期货日线与展期 / 期权数据…")
            # **先补主力连续日线**：`update_aux` 拿 data_futures 的日期当交易日历，
            # 日线停在几天前的话它会认为「衍生数据已是最新」，缺的其实是源头。
            # 窗口给足一年：两个更新函数都只补本地没有的部分，多给的日期不会
            # 重复请求，只是让「很久没开程序」的缺口一次补完。
            f = updater.update_futures(
                None, "1y",
                progress=lambda d, t, m: self._log(f"  日线 [{d}/{t}] {m}"),
            )
            self._log(f"日线：{f.summary()}")
            res = updater.update_aux(
                None, range_key="1y", want_contracts=True, want_options=True,
                progress=lambda d, t, m: self._log(f"  补衍生数据 [{d}/{t}] {m}"),
            )
            n = int(getattr(res, "days", 0) or 0)
            self._log(f"衍生数据已补 {n} 个交易日" if n else "衍生数据已是最新")
            if getattr(res, "failed", 0):
                self._log(f"  {res.failed} 天失败，下次启动会再试")
        except Exception as exc:  # noqa: BLE001
            self._log(f"衍生数据补充失败（{type(exc).__name__}: {exc}），先用本地快照")

    # ---- 监控设置（一键对齐的刷新周期） ----

    @property
    def monitor(self) -> "SStore.MonitorConfig":
        """界面「监控设置」里的那份配置（`refresh_sec` = 账户重读周期）。"""
        return self._monitor

    @monitor.setter
    def monitor(self, cfg) -> None:
        """改完**立刻生效**，不用重启策略。

        界面那边是 `setattr(runner, "monitor", cfg)`（见 `PortfolioController.
        set_chase` 的同一套写法），所以节流用的 `_acct_sec` 必须在这里一起换 ——
        只换 `_monitor` 的话用户改完看不到任何变化，会以为按钮坏了。
        """
        self._monitor = cfg
        try:
            self._acct_sec = max(1.0, min(60.0, float(
                getattr(cfg, "refresh_sec", 5) or 5)))
        except (TypeError, ValueError):
            self._acct_sec = 5.0

    def start(self) -> None:
        """起行情 → 起账户（可选）。**行情失败不该挡住策略**。

        顺序不能反：以前先连账户再补数据、再算信号，账户连不上整个 runner 就
        没有 `api`，`wait()` / `tick()` 直接 return —— 策略变成账户的附属品。
        """
        # 行情是全局共享的那一条，程序打开就连上（`MarketFeed.get()` 首次即连）
        if self.feed is None:
            self.feed = MarketFeed.get()
            self._feed_gen = self.feed.generation
            if not self.feed.ready:
                self._log(f"行情源未连上（{self.feed.error or '原因不明'}）："
                          "先按最后一根已完成的 bar 算信号，不进新仓")
        # 衍生数据（展期 / 期权）是**策略的输入**，跟有没有账户无关：不接账户
        # 也要算因子。但补齐很慢 —— 2026-11-20 实测 `_ensure_aux()` **15.1 秒**，
        # 放在 `start()` 里同步跑，「启动全部」时界面会僵在那儿。
        # 改后台：策略先用本地快照算信号，补齐了下一根 bar 自然用新值。
        threading.Thread(target=self._aux_bg, daemon=True,
                         name="aux-warm").start()
        # 账户是**最后一步**：绑了才连，连不上只记一条，策略照跑
        if self.spec is not None:
            try:
                self._connect_account()
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._log(f"账户未接入：{self.last_error}（策略继续运行，不下单）")
                self._diagw("ACC", f"账户未接入：{self.last_error}"
                            "（策略继续运行，不下单；界面点「刷新」会再试一次）")
        else:
            self._log("未绑定账户：只算信号、不报单（可在交易页随时接入）")
            self._diagw("ACC", "未绑定账户：只算信号、不报单")

    def _aux_bg(self) -> None:
        """后台补衍生数据。多个 runner 同时起时只让一个去补，其余直接跳过。

        `_ensure_aux()` 实测 15.1 秒（要读 9 个品种的展期 / 期权快照并补缺失
        交易日）。几个账户一起启动就并发写同一个库，DuckDB 是单写者，会互相
        卡住 —— 所以用一把**非阻塞**的锁，抢到就补，抢不到说明别人正在补。
        """
        if not _AUX_LOCK.acquire(blocking=False):
            return
        try:
            self._ensure_aux()
        except Exception as exc:  # noqa: BLE001
            self._log(f"衍生数据检查失败（{type(exc).__name__}: {exc}），先用本地快照")
        finally:
            _AUX_LOCK.release()

    def _connect_account(self) -> None:
        """连账户。**只做交易那部分** —— 行情早已由 `self.feed` 提供。"""
        # 先体检账户：配错的（缺账号密码、柜台+A股）到下面连的时候才炸，而那时候
        # `_ensure_aux()` 已经补了几分钟数据。理由见 `require_live_account()`
        require_live_account(self.spec)
        backend = getattr(self.spec, "backend", "tq")

        if backend == "tq":
            from tqsdk import TqApi, TqAuth

            self.api = TqApi(
                self.spec.build(),
                auth=TqAuth(self.spec.user, self.spec.password),
                web_gui=False,
            )
        else:
            self.api = self._broker_api(backend)
        self._pos = self.api.get_position()
        self._orders = self.api.get_order()
        if backend == "tq":
            self._log(f"账户已连接：{self.spec.label()}")
        else:
            self._log(f"账户已连接：{self.spec.label()}"
                      f"（行情走 {self._feed_label()}，历史行情读本地天勤数据）")
        # 🔴 账户接上 / 断开是**最该留痕**的一件事，但 `_log()` 只进界面那块
        # 文本框，关窗就没了 —— 2026-11-23 用户报「登出再登录后监控器还说未登录」，
        # live.log 里一条接/脱记录都没有，只能靠猜。所以这里同时落一行 `[ACC]`。
        self._diagw("ACC", "账户已连接："
                    + (self.spec.label() if self.spec is not None else "（未命名）"))
        self._detached_by_user = False
        self._sweep_orders()

    def _feed_label(self) -> str:
        try:
            return BACKEND_LABELS.get(self.feed.source, self.feed.source)
        except Exception:  # noqa: BLE001
            return "行情源"

    # ---- 账户的接 / 脱（策略不中断） ----

    @property
    def account_ready(self) -> bool:
        return self.api is not None and self._pos is not None

    def attach_account(self, spec) -> bool:
        """接入 / 切换账户。**策略不停、K 线缓存不清、台账不动。**

        返回是否连上；连不上时策略继续跑（只是不发单），界面上账户状态显示未连。
        """
        if self.api is not None:
            self.detach_account()
        old = self.spec
        self.spec = spec
        self.limit = daylimit.DayLimit(spec.label() if spec is not None
                                       else "未接账户")
        try:
            self._connect_account()
            return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._log(f"接入账户失败：{self.last_error}（策略继续运行，不下单）")
            self._diagw("ACC", f"接入账户失败：{self.last_error}（策略继续运行，不下单）")
            self.api = None
            self._pos = self._orders = None
            if old is not None:
                self.spec = old
            return False

    def detach_account(self) -> None:
        """登出账户。**策略继续运行**，信号 / 理论仓位 / 台账全部保留。

        在途单先撤：账户登出之后撤单接口就没了，那几笔会一直挂在交易所。
        """
        api, self.api = self.api, None
        # 用户主动登出 ≠ 没接上。界面「刷新」会自动把**没接上**的账户接回来，
        # 但绝不能把用户明确登出的账户又偷偷登回去 —— 所以这个标记必须有
        self._detached_by_user = True
        self._pos = self._orders = None
        for st in list(self._slots.values()) + [self._align_slot]:
            try:
                self._cancel_all(st)
            except Exception:  # noqa: BLE001
                st.pending.clear()
        if api is not None:
            try:
                api.close()
            except Exception:  # noqa: BLE001
                pass
            self._log("账户已登出：策略继续运行（只算信号，不下单）")
            self._diagw("ACC", "账户已登出（用户主动）：策略继续运行，只算信号")

    def _broker_api(self, backend: str):
        """柜台账户的执行层：实时行情 + 下单走柜台，历史走本地天勤数据。"""
        from ttrade.broker_api import BrokerApi

        api = BrokerApi(self.spec, feed=backend, log=self._log)
        api.start()
        return api

    def _sweep_orders(self) -> None:
        """撤掉上一次运行留下、还挂着的委托。

        程序被强杀（或直接关机）时 `close()` 跑不到，限价单会一直挂在交易所。
        下次打开账户，用户什么都没点，委托却在成交 —— 那其实是上一次留下的，
        看着就像「不点按钮它自己在下单」。

        只撤**本程序自己下过**的单（order_id 记在 `live_orders.json`），
        快期客户端里手点的委托不在名单里，不会被误伤。
        """
        ids = orderbook.load(self._book_key)
        if not ids:
            return
        if getattr(self.spec, "backend", "tq") != "tq":
            # 柜台的历史委托是**登录后异步补查**的（`CtpHub._query_hist()`，
            # 每秒一拍、和资金/持仓轮着来），跑到这里时往往还没查回来 ——
            # 下面那圈会一笔都撤不到。这不是「柜台不支持」，是「还没轮到」，
            # 提示要照着实际情况写，别让用户以为只能手工撤。
            self._log(
                "柜台的当日委托正在补查（登录后约几秒）。上一次运行遗留的挂单"
                "若仍然挂着，稍后在「账户透视和交易」页对着委托列表撤掉，"
                "或者重启一次让程序自己撤")
        n = 0
        for oid in ids:
            o = self._orders.get(oid) if self._orders is not None else None
            if o is None or getattr(o, "status", "") != "ALIVE":
                continue
            try:
                self.api.cancel_order(o)
                n += 1
            except Exception:  # noqa: BLE001
                pass
        orderbook.clear(self._book_key)
        if n:
            self._log(f"撤掉上次遗留的 {n} 笔委托（程序上次没有正常退出）")

    def close(self) -> None:
        # 退之前把在途单撤掉。程序都退出了还留着限价单，等于「没人管的仓位」：
        # 下次打开账户，用户没点任何按钮，委托却照样成交 —— 那其实就是上一次
        # 留下的单。强杀时这一段跑不到，靠下次启动的 `_sweep_orders()` 兜底。
        if self.api is not None:
            for st in list(self._slots.values()) + [self._align_slot]:
                try:
                    self._cancel_all(st)
                except Exception:  # noqa: BLE001
                    st.pending.clear()
        api, self.api = self.api, None
        self._klines.clear()
        self._quotes.clear()
        self._pos = self._orders = None
        # **行情源不关**：它是全局共享的一条连接（`MarketFeed` 单例），关掉
        # 正在跑的另一组策略 / 账户透视页会一起断
        for st in self._slots.values():
            st.running = False
            st.pending.clear()
            st.retired.clear()
        self._align_slot.pending.clear()
        # 机器学习那条后台线程也要收掉，否则关掉账户之后它还在算，白烧一个核
        try:
            self._ml.stop()
        except Exception:  # noqa: BLE001
            pass
        if api is not None:
            try:
                api.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def alive(self) -> bool:
        """runner 在不在跑。判据是**行情**，不是账户 —— 账户登出策略仍在运行。"""
        return self.feed is not None

    # ---- 槽位管理 ----

    def children_of(self, cfg: SStore.StrategyConfig) -> list:
        """把一条多品种配置拆成「一个品种一份」。

        单品种 / 截面族返回空表（前者本来就不用拆，后者是真的不支持 —— 截面要
        一篮子同时按权重调仓，不是 N 个独立信号）。
        """
        keys = [k for k in (getattr(cfg, "symbols", None) or []) if k]
        if getattr(cfg, "family", "") == "cross" or len(keys) < 2:
            return []
        n = len(keys)
        out = []
        for k in keys:
            sym, ex = replay.split_key(k, cfg.market)
            one = cfg.clone(symbol=sym, exchange=ex,
                            multi_symbols=[], cross_symbols=[])
            one.sid = f"{cfg.sid}{GROUP_SEP}{k}"
            # 资金按品种数均分。组合回测里每个品种各占一份账户、再把净值等权
            # 平均，换算到一个真实账户上就是「每个品种只分到 1/N 的钱」。
            # 不拆的话 19 个品种各按 30% 满仓，实际敞口是配置的 19 倍
            # 口径收在 `per_symbol_principal()` 里，回测那边读同一个函数，
            # 免得实盘除以 N、回测不除，同一份配置两边差 N 倍
            one.principal = cfg.per_symbol_principal(n)
            out.append(one)
        return out

    @property
    def _book_key(self) -> str:
        """台账 / 委托账本的键。

        **没接账户也要有台账**：目标仓和小节流都记在这份账本里，不写的话策略
        一停就忘了自己算到哪，界面上的「理论仓位」也会跟着变 0。
        """
        return self.spec.label() if self.spec is not None else BOOK_NO_ACCOUNT

    def _add_one(self, cfg: SStore.StrategyConfig) -> SlotState:
        st = SlotState(cfg=cfg)
        # 账本跟着「账户 + 策略 id」走，重启后接着记。不读回来的话，目标 -11
        # 会从头再开一遍，账户直接变 -22。
        book = positionbook.load(self._book_key)
        st.held = dict(book.get(cfg.sid, {}))
        st.cost = dict(positionbook.load_costs(self._book_key).get(cfg.sid, {}))
        # 仓位还在、均价丢了（老版本升级上来）时，按当前价补一个 —— 不补的话
        # 那一列会一直是 0 盈亏，看着像「刚开仓」。宁可写一个近似值并在日志里
        # 说清楚，也不要让用户对着一个不动的 0 猜
        self._seed_cost(st)
        self._slots[cfg.sid] = st
        return st

    def add(self, cfg: SStore.StrategyConfig) -> SlotState:
        kids = self.children_of(cfg)
        if not kids:
            return self._add_one(cfg)
        self._groups[cfg.sid] = [k.sid for k in kids]
        self._parents[cfg.sid] = cfg
        first = None
        for k in kids:
            self._child_of[k.sid] = cfg.sid
            st = self._add_one(k)
            first = first or st
        self._log(f"{cfg.title()}：{len(kids)} 个品种各自建一个槽位"
                  f"（每个分到 {kids[0].principal / 10000:g} 万）")
        return first

    def add_or_update(self, cfg: SStore.StrategyConfig) -> None:
        """引擎已经认识这条就只更新配置，不认识才新建槽位。

        「界面新加了一条策略，让正在跑的连接把它接上」只能走这里 —— 另外两个
        方法在这个场景下都是错的：

        - `add()` 无条件重建 `SlotState`（`_add_one` 里直接 `self._slots[sid] = st`），
          对已经在跑的槽位等于把 running / 挂单 / 目标仓全抹掉；
        - `update()` 反过来，槽位不存在时直接 return，什么都不做。

        2026-11-11 用户报的「新加的策略点启动/停止没反应」就是后者：从「策略研究」
        页勾「加入策略交易」或组合页「导入策略」加进来的只写了磁盘，没有推给引擎，
        于是 `enable()` 一路走到 `self._slots.get(sid) is None` 然后**静默 return**，
        界面既不报错也没变化。
        """
        if cfg.sid in self._slots or cfg.sid in self._groups:
            self.update(cfg)
            return
        self.add(cfg)

    def _remove_one(self, sid: str) -> None:
        st = self._slots.pop(sid, None)
        if st is not None:
            self._cancel_all(st)
            # 策略删了，账本也别留着 —— 同 id 再建一条会拿到别人的仓位
            positionbook.forget(self._book_key, sid)

    def remove(self, sid: str) -> None:
        for k in self._groups.pop(sid, []) or []:
            self._child_of.pop(k, None)
            self._remove_one(k)
        self._parents.pop(sid, None)
        self._remove_one(sid)

    def _save_book(self, st: SlotState) -> None:
        st.book_ts = time.time()
        positionbook.save(self._book_key, st.cfg.sid, st.held, cost=st.cost)

    def get(self, sid: str) -> SlotState | None:
        return self._slots.get(sid)

    def slots(self) -> list[SlotState]:
        return list(self._slots.values())

    def group_of(self, sid: str) -> str:
        """子槽位属于哪条多品种策略。单品种返回空串。"""
        return self._child_of.get(sid, "")

    def _update_one(self, cfg: SStore.StrategyConfig) -> None:
        st = self._slots.get(cfg.sid)
        if st is None:
            return
        old = st.cfg
        st.cfg = cfg
        # 🔴 **只有「定手数的输入」变了才重算这一笔的手数**。改个名字、改个快线
        # 参数也会走到这里；跟着清锁存的话，`_theo_qty` 会退回「当前这根」的
        # `st.lots`（可能已经漂到另一头），`target` 一变 `_rebalance` 就真发单
        # —— 改个名字把仓平掉一半，比原来的 bug 更糟。
        def _size_key(c):
            g = lambda n, d=0: getattr(c, n, d) or d       # noqa: E731
            return (str(g("sizing", "capital")).lower(),
                    float(g("capital_pct")), float(g("risk_pct")),
                    float(g("principal")), int(g("volume")),
                    float(g("stop_mult")), int(g("stop_atr_n")))

        sizing_changed = _size_key(old) != _size_key(cfg)
        # 手数变了目标仓位跟着变，立刻重算 —— 不然要等下一根 bar 才调整。
        # 停着的时候也算：`_rebalance` 只对 running 的槽位生效，重新启动时
        # 若刚好没出新 bar（`_refresh_signal` 会早退），靠的就是这个值。
        # `lots` 是上一根 bar 按价格算出来的手数；改分配资金后要等下一次
        # `_refresh_signal` 才会重算，所以下面还要清 last_bar。
        # 固定手数模式例外：手数就写在配置里，立刻生效，不用等行情。
        if (cfg.sizing or "capital").lower() == "fixed":
            st.lots = max(1, int(cfg.volume or 1))
            # 固定手数写在配置里，用户改完就该立刻生效 —— 顺手把锁存的开仓手数
            # 换成新值，不然要等下一次「方向变了」才轮到它
            if sizing_changed and int(getattr(st, "signal", 0) or 0):
                st.theo_lots = int(st.lots)
                st.theo_lots_sig = 1 if int(st.signal) > 0 else -1
                st.theo_lots_src = THEO_SIG
        elif sizing_changed:
            # 资金 / 止损额 / 分仓口径改了 ⇒ 这一笔的手数要按新配置重算一次
            # （用户预期的「改完资金仓位跟着变」）。只清「算过」的标记，**不动
            # `st.target`** —— 清了目标仓，`_rebalance` 会在下一根 `_refresh_signal`
            # 之前把它当成「该平仓」，一笔冤枉单就出去了。
            st.theo_lots = 0
        # 🔴 K 线内那一族**不按 signal 反推目标仓**：它的 target 只能由盘中触发价
        # 给（`_ib_arm` / `_intrabar_tick`）。`signal` 从 2026-11-23 起保留真值
        # （给「理论持仓」列用），这里照旧套公式就成了「改一次手数 → 拿一根已经
        # 收盘的 bar 去追一个过去的价位」，等于凭空发一笔单。手数改了等下一根
        # `_ib_arm` 自然会重算（它按止损距离定手数）。
        if not self._ib_on(st):
            st.target = self._theo_lots_qty(st)
        # 策略参数（快线 / 慢线）也可能一起改了。不清掉 last_bar 的话，
        # `_refresh_signal` 认为「这根 bar 算过了」直接早退，新参数要等下一根
        # K 线才生效 —— 用户会以为改了没用。
        st.last_bar = None

    def update(self, cfg: SStore.StrategyConfig) -> None:
        """换掉一条策略的配置。

        界面改了手数 / 名称之后必须走这里：`SlotState.cfg` 是启动时那一份的引用，
        组合文件里改完不会自动同步过来，否则「改完手数没反应」得等重启才生效。
        """
        kids = self._groups.get(cfg.sid)
        if not kids:
            self._update_one(cfg)
            return
        # 品种清单整个换过：槽位对不上，只能重建。重建期间要保持「原来在跑」，
        # 否则用户在设置里加一个品种，整条策略就悄悄停了
        want = [f"{cfg.sid}{GROUP_SEP}{k}" for k in (cfg.symbols or [])]
        if want != kids:
            running = any(self._slots[k].running for k in kids if k in self._slots)
            for k in kids:
                self._child_of.pop(k, None)
                self._remove_one(k)
            self._groups.pop(cfg.sid, None)
            self.add(cfg)
            if running:
                self.enable(cfg.sid, True)
            return
        self._parents[cfg.sid] = cfg
        for child in self.children_of(cfg):
            self._update_one(child)

    def enable(self, sid: str, on: bool) -> None:
        kids = self._groups.get(sid)
        if kids:
            # 一条多品种策略 = 全部子槽位一起启停。单个品种起不来（没数据、
            # 报单到顶）不该拖住其它品种 —— 所以逐个开，失败的原因留在各自
            # 槽位上，父那一行会汇总显示
            for k in kids:
                self.enable(k, on)
            return
        st = self._slots.get(sid)
        if st is None:
            return
        if on and st.cfg.family == "cross":
            # 截面策略要**同时**持有一篮子品种、按权重整体调仓，而实盘引擎一个
            # 槽位只挂一个合约（多品种策略是拆成多个独立槽位、各自按固定
            # 比例下单，没有「按权重整体调仓」这回事）。硬跑会变成「只按第一个
            # 品种下单」，那是最危险的一种错误 —— 看着在跑，实际仓位完全不是
            # 回测里那条曲线
            st.error = ("截面策略暂不支持实盘：实盘引擎每个槽位只挂一个合约，"
                        "截面要一篮子同时按权重调仓。可以先用组合页看回测")
            return
        if on and not st.cfg.rule_key and st.cfg.family == "rule":
            st.error = "这条策略没有选规则"
            return
        # 「规则 + 因子过滤」：过滤条件是策略的一半，缺了它跑出来的就是另一条策略
        if (on and st.cfg.family == "rule" and (st.cfg.factor_mode or "")
                and not st.cfg.factor_keys):
            st.error = "这条策略开了因子过滤，但一个过滤因子都没选"
            return
        # 机器学习过滤要走 walk-forward 重训，每来一根新 bar 就得重算一遍
        # （分钟级上是几十秒），实盘来不及。宁可拦下来，也不能默默按**不过滤**
        # 的规则信号下单 —— 那等于换了一条策略在跑
        if (on and st.cfg.family == "rule"
                and (st.cfg.factor_mode or "") == "ml"):
            st.error = ("「机器学习」过滤暂不支持实盘：每根 bar 都要重训，"
                        "来不及下单。把过滤方式换成单因子 / 多因子组合再开")
            return
        if on and st.cfg.family == "factor" and not st.cfg.factor_keys:
            st.error = "这条因子策略没有选中因子"
            return
        # 2 日 / 周线由日线合成，不进实时。周线按自然周分组本来是稳的，但盘中
        # 那根周线还没收；2 日线更硬伤 —— 它按「连续两根日线成对」合成，每来
        # 一根新日线，配对就整体错位一格，实时段和历史段算出来的不是同一根 K 线。
        if on and tqdata.is_multiday(st.cfg.period):
            st.error = (f"「{tqdata.PERIOD_LABEL.get(st.cfg.period, st.cfg.period)}」"
                        "由日线合成，不支持实盘；要实盘请把周期切到「日线」")
            return
        # 门禁按**源周期**判断。`1d_tq`（天勤日线）在 `RT_SEC` 里没有这个键 ——
        # 它的源周期是 `1d`，`_subscribe()` 也是这么映射的。按原键判断会让日线
        # 策略一启动就报「实时不支持 1d_tq」，永远起不来。
        src = tqdata.PERIODS.get(st.cfg.period, (st.cfg.period, 1))[0]
        if on and src not in RT_SEC:
            ok = [p for p in tqdata.PERIOD_ORDER
                  if tqdata.PERIODS.get(p, (p, 1))[0] in RT_SEC]
            st.error = f"实时不支持 {st.cfg.period} 周期（可用：{'/'.join(ok)}）"
            return
        if on and self.limit.order_left() <= 0:
            st.error = (f"今日报单已达 {daylimit.DAY_ORDER_LIMIT} 次上限，"
                        "再发会被交易所限制交易，明天再开")
            st.running = False
            self._log(f"{st.title()}：{st.error}")
            return
        was = bool(st.running)
        st.running = bool(on)
        if on and not was:
            # 「启动」等于**重新加载一条策略**：目标基准作废，下一轮重新记。
            # 不这么做的话，停机前那个目标还在，重启后第一轮就按老目标调仓了
            st.based = False
        st.error = ""
        st.rejects = 0
        st.reject_msg = ""
        st.last_action = "启动" if on else "停止"
        if on and self._is_ml(st.cfg):
            # 机器学习要先把模型训出来才有信号。这之前**不许动仓** —— 拿
            # 「还没算出来」当信号 0，会把上次留下的仓平掉，重启一次白交一遍手续费
            st.ml_ready = False
            st.last_action = "启动（机器学习训练中）"
        if not on:
            self._cancel_all(st)
        self._log(f"{st.title()}：{'启动' if on else '停止'}")

    # ---- 推进 ----

    def wait(self, timeout: float = 0.8) -> bool:
        """驱动一次：行情推一步，**账户按周期也推一步**。

        行情走全局 `MarketFeed`（跨账户跨策略一份），账户走 `self.api` ——
        两者是两件事，缺一条就少一半。🔴 2026-11-23 用户报「一键对齐发了单、
        监控器还是不动」，真根因就在这：这里以前只推行情，柜台的
        `BrokerApi.wait_update()` 除了 `start()` 里那一次再没被调过，持仓 / 委托
        / 资金全冻在「连上账户那一刻」（`[MM]` 那句「已经不一致 122 秒」＝连接
        至今），对齐单的柜台回报其实早到了，只是没人去读 `snapshot()`。

        账户这一步按 `_acct_sec`（默认 5 秒，「监控设置」可改）节流，不是每轮都推：
        柜台「每秒一个查询」和报单共用同一个名额，推太密只会和报单抢。
        """
        if self.feed is None:
            return False
        # 切了行情源：手里那些 serial 全是上一条连接的活引用，已经不会再更新了。
        # 不清掉的话策略会一直读一份冻住的 K 线，看着「没信号」而不报任何错。
        gen = self.feed.generation
        if gen != self._feed_gen:
            self._feed_gen = gen
            self._klines.clear()
            self._quotes.clear()
            self._log(f"行情源已切换：{self._feed_label()}")
        # 账户：只对柜台（`BrokerApi`）推。天勤那条 `TqApi` 自己有后台线程在
        # 收数据，`get_position()` 就是活引用，不需要（也不能）在这里再
        # `wait_update()` —— 那会把同一个 TqApi 从两个线程驱动。
        #
        # 🔴 按 `_acct_sec` 节流（默认 5 秒，界面「监控设置」可改）。为什么不是
        # 每个循环都推：柜台限制「同一会话每秒一个查询，和报单共用」，
        # 太快只会和报单抢名额；而监控比对用几秒前的持仓完全够。
        # 点「刷新」走 `refresh_now()`，不受这里限制。
        refresh = getattr(self.api, "refresh_account", None)
        if refresh is not None:
            sec = float(getattr(self, "_acct_sec", 0.0) or 0.0)
            now = time.monotonic()
            if now - getattr(self, "_acct_at", 0.0) >= sec:
                self._acct_at = now
                try:
                    refresh()
                except Exception as exc:  # noqa: BLE001 - 查询失败不能拖累调仓
                    self.last_error = f"{type(exc).__name__}: {exc}"
        try:
            return bool(self.feed.wait(timeout))
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise

    def tick(self) -> None:
        """每轮 wait_update 之后调一次：刷信号、调仓、追单。"""
        if self.feed is None:
            return
        # 没接账户：信号、理论仓位、台账全照常算，只是**一笔不发**。以前这里
        # 直接 return，策略跟着账户一起停 —— 实盘日终登出就等于策略断线
        can_trade = self.api is not None
        if can_trade:
            # 每 2 秒把持仓 / 委托的引用重挂一次。**不影响 `can_trade`**：
            # 重挂拿不到东西时（账户刚断）沿用上一次那份，别把策略一起停了
            self._reanchor()
        done = 0
        for st in list(self._slots.values()):
            if not st.running:
                # 停机**不等于**不补数据。补库只依赖那条订阅，和策略在不在跑无关；
                # 而库里停在几天前，回测、详情页、下次启动读到的全是旧数据。
                # 2026-11-17 的现场就是这样：IF 在 11:45 被拒单停机，之后补库
                # 也一起停了，5m 库一直卡在 11-10。
                try:
                    if self._skey(st.cfg) in self._klines:
                        self._top_up(st)
                except Exception:  # noqa: BLE001
                    pass
                continue
            if done >= SIG_BUDGET:
                # 这一轮配额用完了。重启 / 换源之后每个槽位都在等第一根 bar，
                # 一口气算完实测 15 秒以上，工作线程占住、界面几秒不动一下 ——
                # 重算分摊到后面几轮，首屏信号照样几秒内出来。
                # 但**订阅照投**（非阻塞、几乎不花时间）：不投的话排在后面的
                # 品种要等好几轮才开始回填
                try:
                    self._subscribe(st)
                except Exception:  # noqa: BLE001
                    pass
                continue
            try:
                before = st.last_bar
                # 订阅**不受**时段限制（在 `_refresh_signal` 里先订再判）：
                # 策略可能是在午休、收盘之后启动的，那一刻也得把 K 线订上，
                # 否则开盘时手上没有底料
                self._refresh_signal(st)
                if st.last_bar != before:
                    done += 1          # 这一条真的重算了信号，记一笔配额
                if not can_trade:
                    # 没接账户：目标已经算好、台账已经记了，只是**不下单也不追单**。
                    # 在途单直接丢掉 —— 账户都登出了，那些单不可能在本地落地
                    st.pending.clear()
                    if not st.last_action:
                        st.last_action = NO_ACCOUNT_NOTE
                    continue
                if not self._open_now(st.cfg):
                    # 非交易时段：**只跳过下单链路**（K 线内不触发、不调仓、
                    # 不追单 —— 收盘后重报只会被柜台打回，攒够 MAX_REJECTS
                    # 策略就自己停机）。信号与理论仓位已经在 `_refresh_signal`
                    # 里算好了，所以这里**不许盖掉它写的那句话**。
                    if not st.last_action:
                        st.last_action = CLOSED_NOTE
                    continue
                if not st.ml_ready:
                    # 机器学习还在算第一批信号：仓位保持原样，只把在途单追完
                    self._chase(st)
                    continue
                # K 线内触发：只用最新价跟「这一根的触发价」比，不需要新 bar。
                # 突破策略的真实成交就发生在这一刻，等收盘再看等于白送一段行情
                self._intrabar_tick(st)
                self._rebalance(st)
                self._chase(st)
            except Exception as exc:  # noqa: BLE001
                st.error = f"{type(exc).__name__}: {exc}"
                st.running = False
                self._log(f"{st.title()} 出错停机：{st.error}")

        # 对齐发出的单也要追、也要撤。它不属于任何策略，单独走一遍。
        if self._align_slot.pending:
            oldest = min(o.ts for o in self._align_slot.pending)
            timeout = self._align_timeout()
            if time.time() - oldest > timeout:
                # 挂太久了。不撤的话在途列表永远不清空，`_rebalance` 就永远让路，
                # 整个账户看着像卡死。撤掉、放开，让策略接着跑。
                # 撤之前先把每笔的状态打出来：事后想知道「是压根没成交、还是
                # 成交了一部分」只能靠这一行（界面上的委托表只显示最近几十条）。
                self._diagw("ALIGN", f"对齐超时撤单（{timeout:.0f} 秒），在途明细："
                            + "、".join(self._pending_detail(self._align_slot)))
                self._cancel_all(self._align_slot)
                # 🔴 账本必须跟着撤回来：`_sync_after_align` 是发单时就摆的，
                # 撤单意味着那些仓位根本没到位。
                moved = self._resync_books()
                self._align_touched = set()
                if moved:
                    self._diagw("ALIGN", "对齐没落地，账本按账户实际回摆："
                                + "、".join(moved[:8])
                                + ("…" if len(moved) > 8 else ""))
                self._log(
                    f"对齐委托 {timeout:.0f} 秒未成交，已撤单，策略恢复调仓"
                    + ("（账本已按账户实际回摆）" if moved else "")
                )
            else:
                try:
                    self._chase(self._align_slot)
                except Exception as exc:  # noqa: BLE001
                    self._log(f"对齐追单出错：{type(exc).__name__}: {exc}")

    # ---- 信号 ----

    @staticmethod
    def _skey(cfg) -> str:
        """订阅槽的键。**必须带周期**。

        `cfg.key()` 只是「品种_交易所」，同一个品种的 5m 和 1h 是**两条不同的
        序列**，共用一槽会让后订阅的策略直接拿到前一条 —— 1h 策略实际按 5m
        信号下单，补库还会把 5m 的 bar 写进 1h 表（图上最后一截变成五分钟一根）。
        """
        return f"{cfg.key()}@{cfg.period}"

    def _predictor_for(self, st: SlotState):
        """这个槽位能不能用「只预测」快路径。

        有**训练产物**（`.pkl`）就用它：加载一次，之后每根新 bar 只是
        「重算因子 + predict + 状态机续算」，实测 2.94 秒/品种（65 因子 ×
        26 万根 5m），而不是重训的 34 秒。

        没有产物就返回 None —— 调用方退回「后台重训」慢路径（`MlSignals`）。
        慢路径慢，但**信号口径一致**，不会因为「有没有产物」给出两种策略。

        产物和当前配置对不上（改了因子 / 阈值 / 模型类型）也退回慢路径，
        并在日志里说明差在哪 —— 拿着旧模型跑新配置，出来的信号没有意义。
        """
        if not self._is_ml(st.cfg):
            return None
        # 缓存键用 **sid**，不是 `_skey()`（品种_交易所@周期）。
        # `_skey` 是**订阅**的键 —— 同一个品种同一周期本来就能被多条策略共用
        # 一份 K 线。但**训练产物是按 sid 存的**：两条策略跑同一个品种同一
        # 周期，因子集 / 参数不同，sid 不同，产物也不同。共用 `_skey` 的话，
        # 先来的那条把「没有产物」缓存下来，后来的那条 60 秒内直接被判死
        # —— 明明产物在，引擎却说没有。
        key = st.cfg.sid
        with self._lock:
            hit = self._ml_pred.get(key)
            if hit is not None:
                if hit[0] == "ok":
                    return hit[1]
                if hit[0] == "miss" and time.time() - hit[2] < ML_PRED_RETRY:
                    return None       # 刚试过，别反复刷日志
        try:
            cfg = replay.factor_config(st.cfg.factor_cfg or {})
            keys = list(st.cfg.factor_keys or [])
            pred, why = ml_artifact.load_for_live(st.cfg.sid, cfg, keys, log=self._log)
        except Exception as exc:  # noqa: BLE001
            pred, why = None, f"{type(exc).__name__}: {exc}"
        if pred is None:
            with self._lock:
                self._ml_pred[key] = ("miss", None, time.time())
            if why and why not in self._ml_miss.get(key, ""):
                self._ml_miss[key] = why
                self._log(f"{st.title()}：没有可用的训练产物，"
                          f"退回后台重训（{why}）")
            if self._diag is not None:
                self._diagw("ML", f"{st.title()}（sid={key}）退回后台重训："
                                       f"{why or '（原因不明）'}")
            return None
        with self._lock:
            self._ml_pred[key] = ("ok", pred, time.time())
        self._log(f"{st.title()}：用训练产物算信号（只预测，不重训）")
        if self._diag is not None:
            self._diagw("ML", f"{st.title()}（sid={key}）用训练产物，"
                                   f"走快路径（只预测）")
        return pred

    def _subscribe(self, st: SlotState) -> None:
        key = self._skey(st.cfg)
        tq = trade_symbol(st.cfg.symbol, st.cfg.exchange)
        # 订**源周期**再本地合并（15m/30m 走 5m，2h/4h 走 1h），和本地历史的
        # 合并口径一致；直接订原生 15m 时间边界对不上，拼起来就是碎的
        src, n = tqdata.PERIODS.get(st.cfg.period, (st.cfg.period, 1))
        # 行情走全局源（`self.feed`），不走账户 —— 账户只负责下单。
        # 柜台账户（`simnow_7x24`）没有中金所行情，以前从这里订下去就是一条
        # 永远不更新的空序列，策略「算不出信号」和「真的看空」长得一模一样
        if self.feed is None:
            return
        length = min(10000, KLINE_LEN * max(1, n))
        # 🔴 **投递后立刻返回，不等回填**。天勤首次订阅一个品种要把 3000~4500 根
        # 历史下回来，实测 **0.5~2.3 秒/个**；一个账户 27 个「品种@周期」串着等
        # 就是 30~60 秒，那段时间引擎卡在第一次 tick 里、界面不刷新 ——
        # 2026-11-20 用户报的「启动 / 普通界面都卡住」就是这个。
        # 回填由行情线程自己完成，这里下一 tick 再取，期间显示「行情未就绪」。
        dur = RT_SEC[src]
        # 🔴 **取的时候必须带上同一个周期秒数**：行情源的 K 线缓存按
        # `(合约, 周期)` 建键（同一品种的日线和 1h 是两条序列，见
        # `market_feed.__init__`）。这里原来只传合约，等于去取「0 秒」那一档 ——
        # 2026-11-23 棕榈油就是这样拿到别人订的 1h 序列、日线策略按小时线算信号。
        self.feed.request_kline(tq, dur, length)
        self.feed.request_quote(tq)
        kl = self.feed.peek_kline(tq, dur)
        if kl is not None:
            # 只缓存拿得到的：没回填时 `_klines` 不写空值，否则上面那句
            # `if key in self._klines` 会以为「已经订好了」，再也不重试
            self._klines[key] = kl
        q = self.feed.peek_quote(tq)
        if q is not None:
            self._quotes[key] = q
        self._top_up(st)

    def _top_up(self, st: SlotState) -> None:
        """把引擎手上这份实时 K 线写回本地库，库里就不再停在几天前。

        落的是**源周期**（5m / 1h）：15m / 30m / 2h / 4h 在库里本来就是由它们
        合并出来的，单独存一份读的时候根本对不上。`_subscribe` 现在一律订源
        周期，所以这里直接复用那份序列，不再为补库多订一条。

        失败不抛：补数据是锦上添花，不该把正在跑的策略打死。刚订阅时序列还
        没填充是常态（`ok=False`），下一 tick 再来。
        """
        if self.feed is None:
            return
        if is_stock(st.cfg.symbol, st.cfg.exchange):
            # 补库那条路写的是 `data_futures/*.csv`（期货 CSV 目录），A 股在
            # `stock.duckdb` 里，写错地方会凭空造出一堆假 CSV。股票的本地历史
            # 由 `download_stock_tq.py` / `download_stock_ts.py` 负责
            return
        src = topup.source_of(st.cfg.period)
        if src not in topup.SOURCE_PERIODS:
            return
        slot = (st.cfg.symbol, st.cfg.exchange, src)
        if slot in self._topped:
            return
        now = time.time()
        if now - self._top_try.get(slot, 0.0) < TOP_RETRY:
            return
        self._top_try[slot] = now
        serial = self._klines.get(self._skey(st.cfg))
        if serial is None or not len(serial):
            return
        r = topup.save(serial, st.cfg.symbol, st.cfg.exchange, src)
        if r.get("gap"):
            # 接缝有洞：库缺了几天，而手上这份序列也没有那几天（柜台账户的序列
            # 只有 300 根，就是从库里种进去的）。追加帮不上忙，得靠全量下载补。
            msg = (f"本地历史与实时段之间缺一截（{r['gap']}），"
                   "补库补不上，请在「数据管理」补该品种历史")
            if self._top_err.get(slot) != msg:
                self._top_err[slot] = msg
                self._log(f"{st.title()}：{msg}")
        if not r["ok"]:
            # 序列还没填充，或者库被别的连接锁着（下载器在跑、主线程正在读）。
            # 两种都值得过 2 秒再试 —— 一次写库失败就把这个品种永久放弃，
            # 会变成「启动那一刻刚好在读数据，之后就再也不补了」。
            why = r["error"] or "序列还没填充"
            if self._top_err.get(slot) != why:
                self._top_err[slot] = why
                self._log(f"{st.title()}：本地历史稍后再补（{why}）")
            return
        self._topped.add(slot)
        self._top_err.pop(slot, None)
        # 这次是我们自己写的库：把 stamp 记成「已读」，别让
        # `_refresh_db_stamp` 把 27 个品种的前复权历史全部重读一遍 ——
        # 自身写库写的是实时段，`merge_live` 合并时本来就会覆盖重叠那几根
        self._note_own_write()
        if r["added"]:
            self._log(f"{st.title()}：本地历史补到最新 +{r['added']} 根"
                      f"（{src}，到 {r['last']:%Y-%m-%d %H:%M}）")

    def _refresh_signal(self, st: SlotState) -> None:
        self._subscribe(st)
        # 🔴 非交易时段**照常算信号，只是不下单**。老写法在这里直接 return，
        # 结果面板上的理论仓位一直是 0，用户以为「策略没仓 / 台账没了」
        # （2026-11-17 两次误判都是它）。收盘后模拟账户还在长的那根 bar 会被
        # 下面 `iloc[:-1]` 切掉，算的是最后一根真 bar；下单链路在 `tick()` 里
        # 整段跳过，`_send` 还有最后一道闸 —— 一笔也发不出去。
        open_now = self._open_now(st.cfg)
        key = self._skey(st.cfg)
        kl = self._klines.get(key)
        if kl is None or not len(kl):
            # 行情没订上（行情源没连、或者还没回填）。以前这里直接 return，
            # 界面继续显示上一次算出的目标 —— 「信号消失」在现场就是这么发生的
            src = self._feed_label()
            self._note_once(
                st, f"行情未就绪（{src}）：算不出信号，理论仓位保持上次值"
                    f"{'，原因：' + self.feed.error if self.feed is not None and self.feed.error else ''}")
            return
        q = self._quotes.get(key)
        und = getattr(q, "underlying_symbol", "") if q is not None else ""
        if und and und != st.underlying:
            if st.underlying and st.underlying not in st.retired:
                # 主连换月了。老合约上的仓位不会自己消失 —— 不记下来，它就变成
                # 一笔没人管的持仓，只能靠人肉发现。
                st.retired.append(st.underlying)
                self._log(f"{st.title()}：换月 {st.underlying} → {und}，老合约待清仓")
            st.underlying = und
            # 换月带来新的换月跳空，前复权基准要按新数据重算
            self._qfq_base.pop(key, None)
        # 必须 `.iloc[-1]`：天勤的 kline 是 DataFrame，`kl["datetime"]` 是带
        # RangeIndex 的 Series，`series[-1]` 走的是**按标签**取值，直接抛
        # `KeyError: -1`。这一行曾经让每条策略在第一个 tick 就停机。
        last = int(kl["datetime"].iloc[-1])
        if last == st.last_bar:
            # 同一根 bar 里不重算行情。**但机器学习的信号是后台线程算的** ——
            # 算完那一刻往往还没来新 bar（收盘之后更是永远等不到下一根），
            # 如果这里直接返回，缓存里算好的信号就没人来读，界面会一直停在
            # 「计算中」，直到下一个交易时段。所以同一根 bar 里也留一条缝：
            # 只读缓存，不重算行情、不重新提交。
            if self._is_ml(st.cfg) and not st.ml_ready:
                self._pick_ml_signal(st)
            return
        st.last_bar = last

        df = self._signal_frame(st, kl)
        if not st.frame_ok:
            # 🔴 降级帧（没有本地前复权历史 / 库没补上 / 库里有断口）：**这一轮
            # 不算信号、不动目标仓、不写理论值**，等下一轮重读成功再恢复。
            #
            # 为什么不再「将就用原始价」：用户 2026-11-23 报「白银基本面分位组合
            # 回测最后一笔 11-18 就平了、持仓应该是 0，界面理论还显示 −2」——
            # 那条策略当时跑的就是「没有本地历史，用实时原始价 2643 根」。原始价
            # 带着换月跳空，算出来的 signal 和回测压根不是两回事。同一类事故
            # 2026-11-20 也发生过（铅按 15,740 记建仓，回测是 15,545）。
            #
            # `target` 保持上一次的值是**故意**的：写成 0 会让 `_rebalance` 把仓
            # 全平掉，一次读库失败就清仓比不交易糟糕得多。
            self._note_once(
                st, f"{st.frame_note} ⇒ 本轮暂停算信号（理论列与下单都不动，"
                    "等本地历史读回来）")
            return
        # 最后一根是正在走的活 bar，**K 线内策略要用它**（「当日开盘价」在一天
        # 的第一根上就是它自己）。下面那行 `iloc[:-1]` 只是给「收盘定信号」的
        # 老策略用的副本，`full` 留给 `_ib_arm`
        full = df
        # 🔴 只有**还在走**的最后一根才切。原来无条件 `iloc[:-1]`，收盘之后那条
        # 已经走完了还照样切掉 —— 日线策略最惨：11-19 收盘后实时序列里已经有
        # 11-19 那根，切掉就还在用 11-18 的信号，而回测用的是 11-19（2026-11-19
        # 17:53 的 live.log 实测：末根=2026-11-18 00:00，当天是 11-19）。这就是
        # 「盘中出信号、更新数据/重载之后又没了」的一条直接来源。
        live_tail = bool(len(df)) and barclock.tail_is_live(
            df.index[-1], st.cfg.symbol, st.cfg.exchange, st.cfg.period)
        if len(df) > 1 and live_tail:
            df = df.iloc[:-1]
        if len(df) < MIN_BARS:
            # 静默 return 是最糟的一种：界面上只看到「理论 0 手」，谁也想不到
            # 是因为手上只有几根 bar。柜台账户尤其容易撞上 —— 它没有回填历史的
            # 能力，序列全靠本地库种进去的 300 根。
            if len(df):
                note = f"K 线只有 {len(df)} 根（要 {MIN_BARS} 根），先不算信号"
                if st.last_action != note:
                    st.last_action = note
                    self._log(f"{st.title()}：{note}")
            return
        if self._is_ml(st.cfg):
            # 机器学习：整段历史重走一遍要几十秒，交给后台线程，这一 tick 只读缓存
            # 有训练产物时走「只预测」快路径（几秒），没有才退回后台重训（几十秒）
            #
            # 手数先在这里算好：它和信号无关，只跟价格 / ATR 有关；而信号从后台
            # 回来时**往往没有新 bar**（收盘之后更是永远等不到），那一刻现算手数
            # 既没有 df 也没有意义
            st.lots = self._lots(st, self._mark_price(st, df),
                                 atr(df, st.cfg.stop_atr_n))
            pred = self._predictor_for(st)
            st.ml_fast = pred is not None
            _fc = replay.factor_config(st.cfg.factor_cfg or {})
            self._ml.submit(self._skey(st.cfg), self._with_aux(st, df), st.cfg,
                            list(st.cfg.factor_keys or []), predictor=pred,
                            extra=self._period_extra(st, _fc))
            if not self._pick_ml_signal(st):
                return
        else:
            st.signal = self._signal(st, df)
        if st.cfg.note != "short":
            st.signal = max(0, st.signal)     # 不允许做空时把空头信号裁掉
        elif getattr(st.cfg, "short_only", False):
            st.signal = min(0, st.signal)     # 禁止做多：+1 砍成空仓，只留空头
        st.lots = self._lots(st, self._mark_price(st, df), atr(df, st.cfg.stop_atr_n))
        if self._ib_on(st):
            # K 线内那一族**不用 signal 定下单目标** —— 入场 / 止损由盘中触发价
            # 决定（见 `_intrabar_tick`）。这根 bar 在这里只做三件事：
            # ① 算「理论持仓」；② 重算本根的触发价；③ 执行上一根收盘留下的
            #    「反向突破离场」。
            #
            # 🔴 ① 是 2026-11-23 补的，之前这里是 `st.signal = 0`。
            # `signal` 被清 0 之后，「理论」列和两个盈亏列（都按 `signal × lots`
            # 或 `target` 算）对 K 线内策略**永远是 +0**：用户看到「sbbox 多品种
            # 波段1h 的 34 个品种理论持仓全是 0」，可回测里明明有 29 个品种持仓。
            # 那 29 个正好等于 `run_strategy("sbbox_ib")` 最后一根上非零信号的
            # 品种数 —— 状态机说得清清楚楚，是这一行把它擦了。
            #
            # 理论仓和下单目标是**两件事**：理论仓是「策略口径该持多少」（回测
            # 同一个数，与有没有触发、账户有没有成交无关，见 `_theo_qty`）；
            # 下单目标只能由触发价给（`_ib_arm`），拿 signal 直接下单就等于
            # 用「这根已经收盘的 bar」去追一个已经过去的价位。
            st.signal = self._signal(st, df)
            if st.cfg.note != "short":
                st.signal = max(0, st.signal)
            elif getattr(st.cfg, "short_only", False):
                st.signal = min(0, st.signal)
            if not open_now:
                # K 线内那一族的触发价要盘中价格才算得出来，收盘后重算只会
                # 把最后一根的价位当成「现在」。保留上一轮的目标就行
                # （理论仓已经算好了，界面看得到）。
                return
            self._ib_arm(st, full)
            return
        prev = int(st.target)
        # 🔴 用 `_theo_qty()` 而不是 `signal × st.lots`：`st.lots` 是**这一根**按
        # 现在的 ATR 算的（每根都变），拿它当目标仓，同一笔持仓会被 ATR 的漂移
        # 一路加仓 / 减仓（`_rebalance` 只比 `target − held`，它会当成信号变了）。
        # 回测里手数是开仓那一根定死的，实盘必须一样。
        st.target = self._theo_lots_qty(st)
        self._sync_theo_entry(st, self._mark_price(st, df))
        if int(st.target) != prev:
            # 目标一变就**落一次盘**（不只是写进界面那份内存日志）。
            # 用户问「实盘开了仓、回测却空仓，是不是信号在闪」时，磁盘上必须
            # 能回答：哪一根 bar、signal 几、手数几、当时喂进去的是哪份行情。
            # 这三样缺一样，就只能靠猜（2026-11-19 就卡在这儿）。
            bar = df.index[-1] if len(df) else ""
            self._note_target(st, prev, bar)
        if not open_now:
            # 让状态栏说清楚「不是没算，是收盘了不下单」。**0 也要写出来** ——
            # 空着或者只写「暂停下单」时，用户分不清「算了但是空仓」和
            # 「压根没算」（2026-11-17 就为这个反复排查了三轮）。
            st.last_action = f"非交易时段：暂停下单，理论 {st.target:+d} 手"

    def _pick_ml_signal(self, st: SlotState) -> bool:
        """从后台缓存取一次机器学习信号；取到就落信号 / 手数，返回 True。

        为什么单独抽出来：信号在后台线程算，**算完的时刻和新 bar 到达的时刻
        无关**。只在「来了新 bar」时读缓存的话，收盘前最后一次计算就永远读不到，
        界面会一直停在「计算中」，用户以为程序卡住了。

        「还在算」和「永远算不出来」必须分开说 —— 以前一律显示「训练中」，
        用户会一直等一个永远不会来的信号。
        """
        key = self._skey(st.cfg)
        sig, age = self._ml.signal(key)
        st.ml_age = age
        if sig is None:
            why = self._ml.error(key)
            if why:
                st.ml_why = why
                st.last_action = f"机器学习信号算不出来：{why}"
            elif not st.ml_fast:
                # 没有可用产物 → 走的是**后台重训**慢路径。这个比快路径
                # 慢一个数量级（19 个品种要十几分钟），界面必须说清楚，
                # 否则用户以为程序卡死了。
                st.ml_why = "retrain"
                st.last_action = ("机器学习训练中（没有可用的训练产物，"
                                  "后台重新训练，较慢），暂不动仓")
            else:
                st.ml_why = ""
                st.last_action = "机器学习信号计算中，暂不动仓"
            st.ml_ready = False
            return False
        st.ml_ready = True
        st.ml_why = ""
        st.signal = int(sig)
        if st.cfg.note != "short":
            st.signal = max(0, st.signal)
        prev = int(st.target)
        # 机器学习那一族没有信号序列可倒推，手数只能按「方向刚变的那一刻」锁存
        # （见 `_latch_theo_lots`），之后整波持有期间不再重算 —— 和其余口径一致
        self._latch_theo_lots(st)
        st.target = self._theo_lots_qty(st)
        # 机器学习那条路没有信号序列，理论建仓价只能按「目标刚变的那一刻」记
        self._sync_theo_entry(st)
        if int(st.target) != prev:
            # 机器学习信号是后台算的，这里手上没有 df —— 末根留空，
            # 但要留下 ml_age，能反推「这份信号是多久以前出的」
            self._note_target(st, prev, f"ML {st.ml_age:.0f}s 前算出")
        return True

    # ---- K 线内进出场（`StrategySpec.intrabar`） ----

    @staticmethod
    def _ib_on(st: SlotState) -> bool:
        """这条策略走不走「K 线内触发」。

        只认**规则策略**：因子 / 机器学习那两族的信号是整段历史打分出来的
        一个数，没有「触发价」这个概念，硬套只会给它们加一层假精确。
        """
        if st.cfg.family != "rule" or not st.cfg.rule_key:
            return False
        # 「规则 + 因子过滤」走的是收盘定信号那条路（`run_rule_with_filter`
        # 已经把 K 线内那几列扔了）。这里也必须退回去，否则会出现「回测按
        # 过滤后的信号、实盘按触发价」两套口径
        if getattr(st.cfg, "factor_mode", ""):
            return False
        spec = STRATEGIES.get(st.cfg.rule_key)
        return bool(spec is not None and getattr(spec, "intrabar", False))

    def _ib_arm(self, st: SlotState, full: pd.DataFrame) -> None:
        """新 bar 到点：算出**这一根**的触发价，并执行上一根留下的出场。

        和回测（`run_intrabar_backtest`）同一条规则、同一个策略函数，唯一
        差别是这里没有「未来这根 K 线的收盘」，所以触发价必须靠 `shift(1)` /
        「上一根 ATR」这些**已经定型**的量算出来（由 `next_levels` 保证）。
        """
        try:
            lv = next_levels(full, st.cfg.rule_key, st.cfg.rule_params,
                             reverse=bool(getattr(st.cfg, "rule_reverse", False)))
        except Exception as exc:  # noqa: BLE001
            st.ib_armed = False
            st.last_action = f"算不出 K 线内触发价：{type(exc).__name__}: {exc}"
            self._log(f"{st.title()}：{st.last_action}")
            return
        if not lv:
            # 算不出触发价（历史不够 / 策略没给这几列）：**保持原仓**。
            # 这时候 target 留在 0 就会被 `_rebalance` 当成「该平仓」，
            # 一次缺数就把用户的仓平掉，比不交易糟糕得多
            st.ib_armed = False
            st.target = int(st.held.get(st.underlying, 0)) if st.underlying else 0
            return

        st.ib_entry_long = lv["entry_long"]
        st.ib_entry_short = lv["entry_short"]
        st.ib_stop_dist = lv["stop_dist"]
        # 「按止损金额定仓」：手数直接用**这一根的止损距离**算，
        # 而不是 `stop_mult × ATR(stop_atr_n)` 反推 —— 后者要求
        # `cfg.stop_mult` / `cfg.stop_atr_n` 和策略自己的参数一字不差，
        # 改一个滑杆没同步另一个，仓位就按另一个风险在走，还不报错。
        # 放在这里算：止损距离只有策略自己知道（它是 `atr_n × stop_mult`）
        if (st.cfg.sizing or "").lower() == "risk":
            d = float(lv["stop_dist"])
            m = self._volume_multiple(st)
            if d == d and d > 0 and m > 0:
                # 与回测同一个函数（`size_lots_intrabar`）；原来这里自己写了一遍
                # `max(1, lots_round(risk_amount/(d×m)))`，是第三份副本
                st.lots = int(size_lots_intrabar(bt_cfg(st.cfg), 0.0, 0.0, d, m))
        st.ib_fired = False
        st.ib_armed = True

        held = int(st.held.get(st.underlying, 0)) if st.underlying else 0
        if held == 0:
            st.ib_exit_pending = False     # 平掉了，欠的那笔账清掉

        # ① 反向突破离场：上一根**收盘**决定，这一根**开盘**执行
        if held and ((held > 0 and lv["exit_long"]) or (held < 0 and lv["exit_short"])):
            st.target = 0
            st.ib_armed = False          # 这根不再看入场，等下一根重来
            st.ib_exit_pending = True
            st.ib_stop = float("nan")
            st.last_action = "反向突破离场（本根开盘）"
            self._log(f"{st.title()}：{st.underlying} 反向突破离场，本根开盘平")
            return

        # ② 有仓：继续拿着，等盘中的止损
        if held:
            if st.ib_exit_pending:
                # 上一根已经决定平了，单子还没成交（或者回报没到）。这时候把
                # `target` 写回 `held`，`_rebalance` 会认为目标仓没变 ⇒ 平仓单
                # 不会被撤，但下一轮又按「持有」算 —— 只要中间哪次重算把
                # target 改回去，这笔平仓就丢了。索性一直按平仓走到底
                st.target = 0
                st.ib_armed = False
                return
            st.target = held
            if not (st.ib_stop == st.ib_stop and st.ib_stop > 0):
                # 没有止损价 = 程序重启过（`ib_stop` 只存在内存里）。按当前
                # 标记价补挂一条，宁可比原来宽松，也不能裸奔到下一根
                d = st.ib_stop_dist
                mark = self._mark_price(st, full)
                if d == d and d > 0 and mark > 0:
                    st.ib_stop = mark - d if held > 0 else mark + d
                    self._log(f"{st.title()}：{st.underlying} 持仓没有止损价"
                              f"（多半是刚重启），按标记价 {mark:.3f} 补挂 "
                              f"{st.ib_stop:.3f}")
            return

        # ③ 空仓：目标就是空，等盘中触发价
        st.target = 0

    def _latch_to_order(self, st: SlotState, direction: int) -> None:
        """K 线内真的触发了：理论手数刷新成**那一刻实际下单的那一份**。

        这是最可信的来源（`theo_lots_src = "ib"`），和 `theo_px` 的 `order` 档
        同一个道理 —— 用户 2026-11-23：「实际上应该就从开仓那边获取」。触发价
        是盘中算的，那一刻的止损距离和收盘那根不一样，手数也就不一样；拿收盘
        那根的推断值顶着，就会又出现「理论 6 手、实盘开了 7 手」。
        """
        n = int(getattr(st, "lots", 0) or 0)
        if n <= 0:
            return
        st.theo_lots = n
        st.theo_lots_sig = int(direction)
        st.theo_lots_src = THEO_IB

    def _intrabar_tick(self, st: SlotState) -> None:
        """每个 tick 拿最新价比一次触发价 / 止损价，越价就把目标仓改掉。

        **只改 `st.target`，不发单**：真正的下单、追单、在途管理还是走
        `_rebalance` / `_chase` 那一套。这样「盘中触发」和「收盘调仓」共用
        同一条下单链路，不会出现两套互不知情的委托在互相打。
        """
        if not st.ib_armed or not st.underlying:
            return
        q = self._contract_quote(st.underlying)
        price = float(getattr(q, "last_price", 0) or 0) if q is not None else 0.0
        if not (price > 0 and math.isfinite(price)):
            return                    # 没行情（收盘 / 断流）：这一轮什么也不做

        held = int(st.held.get(st.underlying, 0))

        if held:
            sp = st.ib_stop
            if sp == sp and sp > 0:
                hit = price <= sp if held > 0 else price >= sp
                if hit:
                    st.target = 0
                    st.ib_armed = False      # 这根 bar 不再入场
                    st.ib_exit_pending = True
                    st.ib_stop = float("nan")
                    st.last_action = f"K 线内止损：现价 {price:.3f} 越过 {sp:.3f}"
                    self._log(f"{st.title()}：K 线内止损 {st.underlying} "
                              f"现价 {price:.3f}，止损 {sp:.3f}")
            return

        if st.ib_fired or st.lots <= 0:
            return

        d = st.ib_stop_dist
        el, es = st.ib_entry_long, st.ib_entry_short
        if el == el and price >= el and not getattr(st.cfg, "short_only", False):
            # 禁止做多时多头触发价作废：价格冲上去了也不开多
            st.target = st.lots
            st.ib_fired = True
            st.ib_exit_pending = False     # 重新持仓了，上一笔的平仓账已清
            # 理论手数跟着「这一笔实际下的手数」走，见 `_latch_to_order`
            self._latch_to_order(st, 1)
            # 止损绑**触发价**（那一刻的下单价），不是此刻的最新价 ——
            # 最新价可能已经冲过去一段，用它会悄悄把止损挪得更松
            st.ib_stop = el - d if (d == d and d > 0) else float("nan")
            # 理论建仓价：**策略自己的触发价**（`ib` 档）。紧接着的 `_send()`
            # 会用真实对手价把它升成 `order` 档 —— 显示层不许自己编一个价
            self._note_ib_entry(st, float(el), 1)
            st.last_action = f"K 线内入场：现价 {price:.3f} 越过 {el:.3f}"
            self._log(f"{st.title()}：K 线内做多 {st.underlying} 触发价 {el:.3f}"
                      f"，现价 {price:.3f}，止损 {st.ib_stop:.3f}")
        elif (es == es and price <= es
              and replay.resolve_allow_short(st.cfg)):
            # 不允许做空时忽略空头触发价。判据走 `replay.resolve_allow_short()`，
            # 不要看 `note`：`note` 只是备注（曾被当批次标签写），照它判会让
            # 回测能双向、实盘只做多（#145）
            st.target = -st.lots
            st.ib_fired = True
            st.ib_exit_pending = False
            self._latch_to_order(st, -1)
            st.ib_stop = es + d if (d == d and d > 0) else float("nan")
            self._note_ib_entry(st, float(es), -1)
            st.last_action = f"K 线内入场：现价 {price:.3f} 跌破 {es:.3f}"
            self._log(f"{st.title()}：K 线内做空 {st.underlying} 触发价 {es:.3f}"
                      f"，现价 {price:.3f}，止损 {st.ib_stop:.3f}")

    def _hole_days(self, df: pd.DataFrame) -> list:
        """这段 K 线里缺掉的交易日。判据见模块级 `hole_days()`。"""
        return hole_days(df)

    def _note_own_write(self) -> None:
        """记下「库刚刚被我写过」，避免把自己写的那一次当成「用户补了数据」。

        两个后果都要避免：① 白白重读 27 个品种的前复权历史（几十秒卡顿）；
        ② 日志里冒出一句「本地行情库有更新」，用户会去查是谁补的。
        """
        try:
            from ttrade import tqstore

            self._qfq_stamp = os.path.getmtime(tqstore.db_path())
            self._qfq_stamp_at = time.time()
        except Exception:                # noqa: BLE001
            pass

    def _refresh_db_stamp(self) -> None:
        """库文件变了就把整份前复权缓存作废。

        每 2 秒最多 stat 一次：`_signal_frame` 只在**出现新 bar** 时调用（一根
        5m 一次），但 19 个品种加起来仍会反复问；stat 本身很便宜，攒一下更稳。
        """
        now = time.time()
        if now - self._qfq_stamp_at < 2.0:
            return
        self._qfq_stamp_at = now
        try:
            from ttrade import tqstore

            stamp = os.path.getmtime(tqstore.db_path())
        except Exception:                # noqa: BLE001
            return
        if stamp == self._qfq_stamp:
            return
        first = self._qfq_stamp == 0.0
        self._qfq_stamp = stamp
        if first:
            return                       # 第一次：缓存本来就是空的
        if self._qfq_base:
            self._qfq_base.clear()
            self._log("本地行情库有更新，已重读前复权历史（不用重启程序）")

    def _signal_frame(self, st: SlotState, kl) -> pd.DataFrame:
        """算信号用的行情：**本地前复权历史 + 实时 bar**。

        回测（详情页）走 `tqdata.load_main(adjust="qfq")`，实盘必须同一份口径 ——
        直接拿天勤原始价算，换月那一下的复权差会把 60 根动量推到 `entry_z` 两侧：
        回测空仓、实盘反向持仓，两边永远对不上。

        `merge_live` 只接**严格晚于**本地最后一根的实时 bar，重叠的那几根用实时
        OHLC 覆盖（半根自愈，见那个函数的注释）。换月时调用方会清掉缓存，让新的
        换月跳空也按前复权抹平。

        库文件变了（用户补完历史 / 定时补库跑过）就**整份重读**：`_qfq_base` 是
        启动时读的一份快照，不重读的话补了数据也白补 —— 用户看到的是「补完了
        信号还是老样子，得重启程序」。判断用 mtime 而不是「库尾更新」，因为补库
        经常只是**覆盖**库尾几根，行数和最后一根的时间都没变。
        """
        self._refresh_db_stamp()
        live = bars_from_serial(kl, st.cfg.period)
        n = tqdata.PERIODS.get(st.cfg.period, (st.cfg.period, 1))[1]
        if n > 1:
            live = tqdata.resample(live, n)
        if live.empty:
            return live
        key = self._skey(st.cfg)

        def _degrade(note: str, out=None, *, retry: bool = True):
            """降级帧：打上标记，并按节流丢掉 `last_bar` 让下一轮重读。

            🔴 为什么必须丢 `last_bar`：`_refresh_signal()` 是**先**把
            `st.last_bar = last` 写掉、**再**调这里的。只要这一次拿到的是降级帧
            （库被锁 / 补库没跟上 / 刚启动还没读回来），「同一根 bar 不重算」那道
            闸门就会把这条策略**永久**钉在降级帧上 —— 2026-11-23 19:14 的白银
            基本面分位组合就是这样：库里明明有 2445 根日线，它却一直用「实时原始
            价 2643 根」算信号，理论列显示 −2，而回测最后一笔 11-18 就平了（该 0）。

            节流 5 秒：真·没有历史时不能每个 tick 都去读一遍几十万行。
            """
            st.frame_ok = False
            st.frame_note = note
            if retry:
                now = time.time()
                seen = getattr(self, "_frame_retry", None)
                if seen is None:
                    seen = self._frame_retry = {}
                if now - seen.get(key, 0.0) > 5.0:
                    seen[key] = now
                    st.last_bar = None       # 逼下一轮重读，别把坏帧钉死
            return live if out is None else out

        base = self._qfq_base.get(key)
        if base is None:
            try:
                base = tqdata.load_main(
                    st.cfg.symbol, st.cfg.exchange, st.cfg.period, adjust="qfq"
                )
            except Exception as exc:  # noqa: BLE001
                # 读不到前复权历史也别把策略打死：退回实时原始价，至少还出得了信号
                self._log(
                    f"{st.title()}：读前复权历史失败，暂用实时原始价"
                    f"（{type(exc).__name__}: {exc}）"
                )
                base = pd.DataFrame()
            if base is not None and not base.empty:
                self._qfq_base[key] = base
            # 🔴 读失败**不能**把一张空表写进缓存：写进去以后 `_refresh_db_stamp`
            # 只在库文件 mtime 变化时才清，于是「锁一解除 / 库一修好」也不会重试
            # —— 这条策略会一直跑在「实时原始价」降级帧上，而原始价带着换月跳空，
            # 算出来的建仓价、信号全是另一个东西（2026-11-20 16:49 铅就是这么被
            # 记成 11-13 建仓的）。空表只当「这次没读到」，下一次 tick 再读。
        if base is None or base.empty:
            return _degrade(f"没有本地历史，用实时原始价 {len(live)} 根")
        if live.index[0] > base.index[-1]:
            # 本地历史太旧，和实时段接不上（中间断了一截）。硬拼会在断口造一次
            # 假跳空，比用原始价更糟 —— 退回实时原始价，并记一条日志
            self._log(f"{st.title()}：本地历史与实时段接不上，暂用实时原始价")
            return _degrade(f"本地历史与实时段接不上（本地止于 "
                            f"{base.index[-1]:%m-%d %H:%M}），用实时原始价 "
                            f"{len(live)} 根")
        # **和回测同源同长**：`replay.run` 先按 `range_years` 切一刀（还多留
        # `WARMUP_BARS` 根预热），这里切同一刀。详情页画多少根、从哪根开始算，
        # 实盘就用同样那些根 —— 所见即所得，不再截 KLINE_LEN。
        merged = merge_live(base, live)
        miss = self._hole_days(merged)
        if miss:
            # 🔴 先用库里的最新进度自愈一次，再决定要不要报缺：`_qfq_base` 是
            # **进程启动时**读的那一份副本，库补好了它也不会自己变 —— 2026-11-17
            # 用户上午补完 5m 历史，界面上的「理论」还全是 0，就是因为这里一直
            # 拿着 11-10 的旧 base 判缺、每 tick 直接 return（连信号都不算）。
            # 判据用「库里最后一根」：只有它比手上这份新才重读，不会每 tick 重读。
            try:
                db_last = tqstore.last_dt(st.cfg.symbol, st.cfg.exchange, st.cfg.period)
            except Exception:  # noqa: BLE001
                db_last = None
            if db_last is not None and db_last > base.index[-1]:
                self._qfq_base.pop(key, None)
                self._log(f"{st.title()}：本地库已补到 {db_last:%m-%d %H:%M}，"
                          "重读前复权历史（不用重启）")
                # 🔴 这里原来只 `return pd.DataFrame()`，`st.last_bar` 早在调用前
                # 就写好了 ⇒ 下一轮「同一根不重算」直接返回 ⇒ 这句「不用重启」
                # 其实永远不会生效（2026-11-23 与白银那条一起查出来的）。
                st.last_bar = None
                return pd.DataFrame()
            # 🔴 缺交易日就**不算信号**：跨着断口算出来的指标是错的，拿它下单就是
            # 「回测持仓、实盘空仓」这种对不上账。保持原仓，把原因写出来。
            note = (f"K 线缺 {len(miss)} 个交易日（{miss[0]:%m-%d}~{miss[-1]:%m-%d}），"
                    "实时段直接接在旧数据后面 —— 先补历史再算信号")
            if self._gap_note.get(key) != note:
                self._gap_note[key] = note
                st.last_action = note
                self._log(f"{st.title()}：{note}")
            return _degrade(note, pd.DataFrame())
        self._gap_note.pop(key, None)
        # 🔴 帧尾整体落后：`hole_days` 查不出来（内部没断口，实时段最后一根就是
        # 合并后最后一根），但这一帧比「此刻本该到的 bar」旧了 —— 说明本地库没补上、
        # 实时序列也还没回补完。在这上面算出的 signal 和回测是两码事，不能拿去下单。
        exp = expected_last_bar(st.cfg.symbol, st.cfg.exchange, st.cfg.period)
        step = pd.Timedelta(seconds=barclock.rt_sec(st.cfg.period))
        if exp is not None and merged.index[-1] < exp - step:
            note = (f"行情帧停在 {merged.index[-1]:%m-%d %H:%M}，"
                    f"按交易时段此刻该到 {exp:%m-%d %H:%M}（本地库没补上/实时未回补完），"
                    "本根不算信号")
            if self._gap_note.get(key) != note:
                self._gap_note[key] = note
                st.last_action = note
                self._log(f"{st.title()}：{note}")
            return _degrade(note, pd.DataFrame())
        merged = replay.window(merged, getattr(st.cfg, "range_years", 0) or 0)
        if merged is None or len(merged) < 2:
            return _degrade(f"本地历史切片后不足 2 根，用实时原始价 {len(live)} 根")
        st.frame_ok = True
        st.frame_note = (f"前复权本地 {len(base)} 根 + 实时 {len(live)} 根"
                         f"（合并后 {len(merged)} 根，起点 "
                         f"{merged.index[0]:%Y-%m-%d %H:%M}）")
        return merged

    def _note_target(self, st: SlotState, prev: int, bar) -> None:
        """目标仓一变就写一行 `[TGT]` 到 `live.log`。

        只进界面日志是不够的：那份在内存里，关窗就没了。而「实盘这根 bar 为什么
        开仓」恰恰是事后才问的 —— 到时候账户、持仓、委托全变了，只剩日志能说明
        当时 signal / lots / 行情来源各是什么。
        """
        msg = (f"{st.title()}：目标 {prev:+d} → {int(st.target):+d} 手"
               f"（signal={st.signal:+d} lots={st.lots} 末根={bar}"
               + (f"；{st.frame_note}" if st.frame_note else "") + "）")
        if self._diag is not None:
            self._diagw("TGT", msg)

    @staticmethod
    def _is_ml(cfg) -> bool:
        """这条策略是不是机器学习。ML 实盘的信号走后台线程（见 `MlSignals`）。"""
        return (getattr(cfg, "family", "") == "factor"
                and (getattr(cfg, "factor_mode", "") or "single") == "ml")

    def _signal(self, st: SlotState, df: pd.DataFrame) -> int:
        """按当前配置算出最后一根 bar 上的目标仓位（-1 / 0 / 1）。

        规则策略和因子策略走两条不同的路。因子策略用的是和研究页同一套
        `run_factor_strategy`：同一个因子、同一组阈值，回测里是几手、实盘就是
        几手 —— 两边各写一份的话，「回测赚钱、实盘亏钱」永远说不清是策略不行
        还是实现不一致。

        **「规则 + 因子过滤」走 `run_rule_with_filter`**：以前这里只跑规则、
        完全不看 `factor_mode`，于是回测里被否决掉的那些开仓，实盘照开 ——
        账户里真会多出那些仓位，而且两条曲线从第一天起就不是同一条。

        **机器学习不走这里**（实盘）：`_refresh_signal` 会把 ML 交给后台线程，
        这里只留一条同步的慢路径给离线测试 / 脚本用。

        `orient="auto"` 在实时段有个天然缺陷：方向是按全样本 IC 的符号定的，
        样本一变就可能翻。用户要是没显式指定方向，日志里会看到它翻。
        """
        mode = st.cfg.factor_mode or ""
        # 要算因子的三种情况：因子策略本身、规则 + 因子过滤。衍生列（展期 /
        # 期权）得先挂上，不然那几个因子整列 NaN，过滤会把所有信号都否掉
        needs_factor = st.cfg.family == "factor" or (
            st.cfg.family == "rule" and bool(mode))
        if needs_factor:
            if len(df) < FACTOR_MIN_BARS:
                # 不许静默返回 0：面板上的「理论 0 手」会被读成「策略看空」，
                # 而真相是「历史还不够，没算出来」
                self._note_once(
                    st, f"数据不足 {len(df)}/{FACTOR_MIN_BARS} 根："
                        "因子全是 NaN，暂按空仓处理（等历史补齐）")
                return 0
            df = self._with_aux(st, df)

        if st.cfg.family == "factor":
            fc = replay.factor_config(st.cfg.factor_cfg or {})
            out = fstrat.run_factor_strategy(
                df, mode or "single", list(st.cfg.factor_keys or []),
                fc, extra=self._period_extra(st, fc), base_period=st.cfg.period,
            )
            self._note_dropped(st, list(st.cfg.factor_keys or []), out)
            self._note_theo_entry(st, df, out.df["signal"])
            sig = out.df["signal"].iloc[-1]
            return 0 if pd.isna(sig) else int(sig)

        rule = run_strategy(df, st.cfg.rule_key, st.cfg.rule_params,
                            reverse=bool(getattr(st.cfg, "rule_reverse", False)))
        if not needs_factor:
            # K 线内那一族要带上策略自己给的止损距离、触发价和出场标志：
            # `risk` 口径的手数必须用**触发那一根**的止损距离算（`stop_mult × ATR`
            # 反推要求两个配置和策略参数一字不差），开仓那一根也要按回测那台
            # 状态机找（进出场都会改变「当前这笔是哪根开的」），不能假设
            # 「信号下一根就成交」，也不能只看 signal 列
            self._note_theo_entry(
                st, df, rule["signal"],
                rule["stop_dist"] if "stop_dist" in rule.columns else None,
                rule["entry_long"] if "entry_long" in rule.columns else None,
                rule["entry_short"] if "entry_short" in rule.columns else None,
                rule["exit_long"] if "exit_long" in rule.columns else None,
                rule["exit_short"] if "exit_short" in rule.columns else None)
            sig = rule["signal"].iloc[-1]
            return 0 if pd.isna(sig) else int(sig)

        # 规则 + 因子过滤：规则信号是「原料」，过滤只做减法（否决 / 延后 / 离场）
        fc = replay.factor_config(st.cfg.factor_cfg or {})
        out = fstrat.run_rule_with_filter(
            df, rule["signal"].to_numpy(int), mode or "single",
            list(st.cfg.factor_keys or []), fc,
            extra=self._period_extra(st, fc), base_period=st.cfg.period,
        )
        self._note_dropped(st, list(st.cfg.factor_keys or []), out)
        self._note_theo_entry(st, df, out.df["signal"])
        sig = out.df["signal"].iloc[-1]
        return 0 if pd.isna(sig) else int(sig)

    def _period_extra(self, st: SlotState, fc) -> dict:
        """跨周期因子要用的别的周期行情：`{周期: DataFrame}`。

        和回测同一套取数（`tqdata.load_main`，前复权）—— 实盘的日线因子和
        回测的日线因子必须是同一份数，否则「回测赚钱、实盘亏钱」又要从头查。
        没配跨周期因子时返回空 dict，一次多余的读都不做。
        """
        # 闸门不能是 `fc.factor_periods`：周期编在因子实例 id 里
        # （`mom_60@1d_tq`），新配置下那个映射是空的
        keys = list(st.cfg.factor_keys or [])
        sym, ex = st.cfg.symbol, st.cfg.exchange
        return fstrat.period_frames(
            fc, keys, st.cfg.period,
            lambda p: self._aux_frame(
                tqdata.load_main(sym, ex, p, adjust="qfq"), sym, ex, p),
            extra_keys=list(getattr(fc, "trigger_keys", None) or []),
        )

    @staticmethod
    def _aux_frame(df: pd.DataFrame, sym: str, ex: str,
                   period: str = "") -> pd.DataFrame:
        """给跨周期帧补上衍生列（展期 / 期权 / 基本面）。

        `tqdata.load_main()` 只给 OHLCV + 持仓量。研究页那边 `DataStore.get()`
        加载时会走 `auxdata.attach_for_period` 把期限结构和期权列
        并进来；实盘这条链路自己拼 df，漏了这一步 —— `roll_main_oi`、`iv_*`、
        `pcr` 这些因子在跨周期帧上整列 NaN，`factor_matrix` 把它们**静默剔除**，
        于是实盘跑的是「少几个因子」的另一条策略，界面上一个字都不提示。
        实测 IM 上 `roll_main_oi@1d_tq`：研究页 78% 有值，实盘 0%。

        日线用 `attach`（那根 K 线的收盘和期限结构是同一天同一时刻算的，谁都
        不领先谁）；分钟用 `attach_intraday`（同一天上午的 bar 不可能知道收盘后
        才有的近远月价差，只能用上一交易日的快照）。
        """
        if df is None or df.empty:
            return df
        try:
            # 口径只有一份：`attach_for_period`（日线同日 / 分钟隔一日）。
            # 以前这里是第三份实现（按 `bar_period(df)` 猜），`_with_aux` 又是
            # 第四份（不分周期一律隔一日）—— 同一个「日线该用哪天的期限结构」
            # 有四种说法，漂掉是必然的。没传周期时退回按帧形态判。
            from ttrade.auxdata import attach_for_period, bar_period

            return attach_for_period(df, sym, ex, period or bar_period(df))
        except Exception:      # noqa: BLE001 - 衍生数据缺了不该让策略停摆
            return df

    def _note_dropped(self, st: SlotState, keys: list, out) -> None:
        """有因子被剔除时说一次 —— 静默剔除是最难查的一类问题。

        `factor_matrix` 把「这个标的、这条链路上算不出值」的因子剔掉是合理的
        （美股没有持仓量），但**实盘这条链路缺列**也会走到同一条路径：策略照
        跑、仓照开，只是过滤条件比回测少几票，曲线上看不出异常，等到实盘和
        回测对不上才发现（本次就是 `roll_main_oi@1d_tq` 在跨周期帧上整列 NaN）。
        """
        fmat = getattr(getattr(out, "pack", None), "fmat", None)
        if fmat is None:
            return
        kept = set(getattr(fmat, "columns", []))
        gone = [str(k) for k in keys if str(k) not in kept]
        sid = str(getattr(st.cfg, "sid", "") or "")
        # 名字只是给人看的，**不许在这条路上抛异常**：日志函数一炸，整轮信号都
        # 没了。自检里借的是个没有 `title()` 的壳，实盘配置也可能没有 sid
        try:
            name = str(st.title() or "")
        except Exception:                    # noqa: BLE001
            name = ""
        name = name or sid or str(getattr(st, "symbol", "") or "槽位")
        # 自检里只借 `_signal` 一条路，不会走完整 __init__
        warned = getattr(self, "_factor_warned", None)
        if warned is None:
            warned = self._factor_warned = set()
        # 🔴 **「实际参与合成几个」这条一定要记**（哪怕一个都没被剔除）：
        # 回测和实盘对不上时，第一个要问的就是「两边喂进去的列一样吗」。
        # 只记「谁被剔了」不够 —— 情况反过来（实盘比回测**多**几票）时它是
        # 一条线索都没有的，2026-11-23 集运欧线就是这么查了半天：回测空仓、
        # 实盘 +1 手，而日志里既没有剔除也没有报错。
        key = sid or name
        if key not in warned:
            warned.add(key)
            self._log(f"{name}：{len(kept)}/{len(keys)} 个因子参与合成"
                      + (f"，缺 {'、'.join(gone[:4])}" if gone else "")
                      + "（回测若与这里不同，就是两条链路喂进去的列不一样）")

    def _with_aux(self, st: SlotState, df: pd.DataFrame) -> pd.DataFrame:
        """给实时序列挂上期限结构 / 期权列。

        天勤的 K 线只有 OHLC 和持仓量，展期、期权那十几个因子读的 `c1_close`、
        `iv_near` 之类全在衍生表里。研究页是 `DataStore` 加载时挂的，实时这条
        链路自己拼 df，不挂就是整列 NaN —— 因子分数算不出来，策略永远空仓。

        🔴 **口径必须跟研究页一模一样，所以只能走 `attach_for_period`**：
        日线同日、分钟隔一个交易日。这里原来是不分周期一律 `attach_intraday`，
        于是 `1d_tq` 策略回测吃当天的期限结构、实盘吃昨天的，展期因子差一天，
        分数就分叉 —— 2026-11-20 棕榈油「回测空仓、实盘还持 2 手空单」就是它
        （score −1.2259 vs −1.3808，正好跨过 `exit_z`）。

        🔴 **失败不许静默**（2026-11-23 补）：原来 `except Exception: return df`
        一个字都不写，于是「挂不上衍生列」和「这个品种本来就没有衍生数据」在看
        日志时长得一模一样。实测代价：集运欧线多因子日内 15 个因子里 7 个读衍生
        列，挂不上只剩 8 个参与合成 ⇒ 末根 signal `+1`；挂上 12 个 ⇒ `0`。
        回测（`DataStore` 挂过）显示空仓、实盘显示 +1 手，而日志里没有任何线索。
        现在失败会写一行原因（每槽位每周期只报一次），下次一眼能看到。
        """
        try:
            from ttrade.auxdata import attach_for_period

            out = attach_for_period(df, st.cfg.symbol, st.cfg.exchange,
                                    st.cfg.period)
        except Exception as exc:  # noqa: BLE001 - 挂不上也要能算信号（只是因子少几票）
            self._note_once(
                st, f"衍生列挂不上（{type(exc).__name__}: {exc}）⇒ 展期 / 期权那几个"
                    "因子本轮没有值，信号会比回测少几票 —— 回测是挂过衍生列的")
            return df
        if len(out.columns) <= len(df.columns):
            # 不报错但也没挂上：品种没有衍生文件，或文件里没有重叠的交易日。
            # 前者正常（纯价格因子照样能跑），后者是数据没更新 —— 说清楚是哪种
            self._note_once(
                st, "衍生列没挂上任何一列（没有该品种的期限结构 / 期权文件，"
                    "或文件与行情没有重叠日期）—— 展期、期权类因子本轮无值")
        self._check_aux_fresh(st, out)
        return out

    def _check_aux_fresh(self, st: SlotState, df: pd.DataFrame) -> None:
        """衍生快照落后太多就提示一次。

        展期 / 期权在分钟线上只能用「上一个交易日」的收盘快照，实盘也一样 ——
        这是 PIT 的要求，不是缺陷（盘中不可能知道今天收盘后的期限结构）。
        但如果用户几天没更新衍生数据，快照就停在几天前：因子有值、但那个值是
        上周的期限结构，看着正常其实是死的。

        **要看快照文件本身的末日，不能看 df 里最后一个非空值**：
        `attach_intraday` 用的是 `ffill`，缺的交易日会沿用最后一条快照，
        df 里永远「有值到今天」，判据就永远不会触发。

        正常滞后 1 个交易日（今天用昨天的），>= 2 才算缺。
        """
        key = self._skey(st.cfg)
        if key in self._aux_warned:
            return
        try:
            from ttrade.auxdata import _default_dir, load_curve, load_opt

            aux_dir = _default_dir()
            last = None
            for part in (load_curve(aux_dir, st.cfg.symbol, st.cfg.exchange),
                         load_opt(aux_dir, st.cfg.symbol, st.cfg.exchange)):
                if not part.empty:
                    d = pd.Timestamp(part.index[-1])
                    last = d if last is None or d < last else last
            if last is None:
                return                      # 没有衍生数据，因子本来就是空的
            td = (pd.to_datetime(df["trading_day"]) if "trading_day" in df.columns
                  else pd.Series(pd.DatetimeIndex(df.index).normalize(), index=df.index))
            days = pd.DatetimeIndex(sorted(set(td.dropna())))
            lag = int((days > last).sum())
            if lag >= 2:
                self._aux_warned.add(key)
                self._log(
                    f"{st.title()}：衍生数据只到 {str(last)[:10]}，"
                    f"落后 {lag} 个交易日 —— 展期 / 期权因子用的是旧快照，"
                    f"去「数据管理」页补一次"
                )
        except Exception:  # noqa: BLE001
            pass

    # ---- 手数 ----

    def _quote_obj(self, symbol: str):
        """按合约取盘口。**统一走全局行情源**，源没连上才退回账户那条连接。

        为什么不是「账户有就用账户的」：一条策略的盘口只该有一个来源。行情源和
        账户各取一个价，会出现「按 A 源定价、报文发到 B 柜台」，追单时两边来回
        跳。源断了才退化到账户，因为柜台自己也有 md，能撑一会儿。
        """
        if not symbol:
            return None
        src = self.feed if (self.feed is not None and self.feed.ready) else self.api
        if src is None:
            return None
        try:
            return src.get_quote(symbol)
        except Exception:  # noqa: BLE001
            return None

    def _contract_quote(self, symbol: str):
        """真实合约的行情对象。拿不到就返回 None —— 定仓失败不该让策略停机。"""
        return self._quote_obj(symbol)

    def _mark_price(self, st: SlotState, df: pd.DataFrame) -> float:
        """算手数用的价格。优先真实合约的最新价，没有就用最后一根收盘。"""
        q = self._contract_quote(st.underlying)
        p = float(getattr(q, "last_price", 0) or 0) if q is not None else 0.0
        if p > 0 and math.isfinite(p):
            return p
        try:
            return float(df["close"].iloc[-1])
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- 品种价格 / 当日盈亏 / 持仓盈亏（界面上那三列） ----

    def _last_px(self, symbol: str) -> float:
        """合约最新价。取不到返回 0（调用方一律按「没有价」处理，不当成 0 元）。"""
        if not symbol:
            return 0.0
        q = self._contract_quote(symbol)
        try:
            p = float(getattr(q, "last_price", 0) or 0) if q is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
        return p if p > 0 and math.isfinite(p) else 0.0

    def _pre_px(self, symbol: str) -> float:
        """昨结算价。天勤叫 `pre_close`（它就是昨结算），CTP / 融航叫
        `pre_settlement` / `pre_close` —— 都试一遍，取到就用。"""
        if not symbol:
            return 0.0
        q = self._contract_quote(symbol)
        if q is None:
            return 0.0
        for name in ("pre_close", "pre_settlement"):
            try:
                v = float(getattr(q, name, 0) or 0)
            except (TypeError, ValueError):
                continue
            if v > 0 and math.isfinite(v):
                return v
        return 0.0

    def _seed_cost(self, st: SlotState) -> None:
        """账本上有仓、却没有开仓均价时，按当前价补一个并落盘。

        什么时候会缺：老版本升级上来（那时还没有均价这个概念）、或者账本被
        手工改过。不补的话那几列永远是 0 盈亏 —— 看着像刚开仓，用户无从判断。
        """
        miss = [s for s, n in st.held.items() if int(n or 0) and not st.cost.get(s)]
        if not miss:
            return
        got = []
        for sym in miss:
            px = self._last_px(sym)
            if px > 0:
                st.cost[sym] = px
                got.append(sym)
        if not got:
            return
        self._save_book(st)
        self._log(f"{st.title()}：{len(got)} 个合约没有开仓均价记录，"
                  "已按当前价补上 —— 这些品种的持仓盈亏从 0 起算")

    def _ib_open_trade(self, st: SlotState, df: pd.DataFrame,
                       e_long=None, e_short=None, x_long=None, x_short=None,
                       stop_dist=None):
        """K 线内那一族：拿到「最近一笔还没平的开仓」＝ (哪根, 成交基准价, 手数)。

        真正的活儿在模块级 `ib_open_trade()`（照 `run_intrabar_backtest` 那台
        状态机走一遍）。这里只负责两件事：① 按策略配置造出**和回测同一份**
        `BacktestConfig`（`replay.make_bt`，研究页 / 实盘共用这一个出口，免得
        两边乘数、保证金率、多空方向各写一遍）；② 同一帧只算一次。

        缓存键是 `(策略, 根数, 末根时间)`：新 bar 才重算。不加缓存的话，2.6 万
        根 1h 每次算信号都要扫一遍 —— 25 个品种每小时同一秒一起到点，界面会
        卡一下。
        """
        if len(df) < 2:
            return None
        try:
            bt = bt_cfg(st.cfg)
        except Exception:                    # noqa: BLE001 - 造不出配置就当没仓
            return None
        cache = getattr(self, "_ibt_cache", None)
        if cache is None:
            cache = self._ibt_cache = {}
        key = (st.cfg.sid, len(df), df.index[-1])
        if key in cache:
            return cache[key]
        got = ib_open_trade(df, bt, e_long, e_short, x_long, x_short, stop_dist)
        if len(cache) > 400:
            cache.clear()
        cache[key] = got
        return got

    def _entry_lots_at(self, st: SlotState, df: pd.DataFrame, j: int,
                       px: float, stop_dist=None) -> int:
        """第 `j` 根（**开仓那一根**）按当时的价 / 止损距离算出来的手数。

        和回测 `size_lots(cfg, 开仓价, 现金, 开仓那根的 ATR)` / `size_lots_intrabar`
        是同一件事 —— 回测在开仓那一根算一次就再也不动了，这里也必须同一根、
        同一套输入，否则「回测 39 手、界面理论 106 手」这种账永远对不上。

        `stop_dist`：K 线内那一族的止损距离序列（策略自己给的）。`risk` 口径
        **必须**用它，不能用 `stop_mult × ATR` 反推 —— 后者要求两个配置和策略
        参数一字不差（同 `_ib_arm` 里那段注释）。
        """
        if not (px == px and px > 0):
            return 0
        cfg = st.cfg
        if (cfg.sizing or "").lower() == "risk" and stop_dist is not None:
            try:
                # 🔴 传进来的常常是 **Series**（`run_strategy` 的那一列），拿整数
                # 下标取会给到「标签索引」→ KeyError → 悄悄退回 ATR 那条路，
                # 手数就和回测差一大截（RB 1h 实测 61 vs 77）。必须按位置取。
                d = float(stop_dist.iloc[j] if hasattr(stop_dist, "iloc")
                          else stop_dist[j])
            except (IndexError, KeyError, TypeError, ValueError):
                d = float("nan")
            m = self._volume_multiple(st)
            if d == d and d > 0 and m > 0:
                # 和回测 K 线内那一族**同一个函数**（`size_lots_intrabar` 的 risk
                # 分支就是 `max(1, lots_round(risk_amount / (stop_dist × 乘数)))`）
                return int(size_lots_intrabar(bt_cfg(cfg), float(px), 0.0, d, m))
        try:
            cut = df.iloc[: j + 1]
            a = atr(cut, cfg.stop_atr_n)
        except Exception:                    # noqa: BLE001 - 算不出就退回本根
            return 0
        n, _why = self._size(st, float(px), a)
        return int(n)

    def _note_theo_entry(self, st: SlotState, df: pd.DataFrame,
                         series, stop_dist=None,
                         e_long=None, e_short=None,
                         x_long=None, x_short=None) -> None:
        """兜底：从信号序列倒推「理论建仓价 / 建仓交易日 / **开仓手数**」。

        🔴 **价格这一半不是主口径**（用户 2026-11-20）：「理论开仓价格，应该是从
        实盘策略中直接获取，不然如果 k 线中开仓，又算不对了」。价格的主口径是
        策略自己写下来的那个价 —— `_send()` 记的**下单价**、`_intrabar_tick()`
        记的**触发价**，见 `SlotState.theo_src`。

        🔴 **手数这一半反过来，只有这一条口径**（用户 2026-11-23）：「理论列需要
        按照开仓的手数来算，不能根据当前 K 线来算，实际上应该就从开仓那边获取，
        而不是自己擅自计算」。所以这里倒推到这波信号的起点 `k`、取成交的那一根
        `j = k+1`，用**那根 K 线的价格和 ATR** 算手数，写进 `st.theo_lots`；
        整波持有期间不再重算（回测就是这么做的，见 `SlotState.theo_lots`）。
        唯一的例外是 K 线内那一族真的触发了 —— 那一刻会按**实际下单的手数**
        刷新（`theo_lots_src = "ib"`），这里不覆盖它。

        这里只在两种情况下补价格：① 策略还没发出过委托（没行情、非交易时段、
        刚启动第一根 bar 还没走完）；② 机器学习那条路没有信号序列可推（走
        `_sync_theo_entry`）。写进去的记为 `sig` 档，**覆盖不了** `order`/`ib`
        —— 否则真成交价会被「下一根开盘」这种推断冲掉。

        `theo_day` 存的是**交易日**（夜盘 20 点后归下一个），不是自然日：拿自然
        日跟 `daylimit.trading_day()` 比，夜盘那段永远不相等，「今天开的仓」会被
        当成隔夜仓，两个数又对不上了。
        """
        try:
            n = len(series)
        except TypeError:
            n = 0
        if not n or len(df) != n:
            return
        # 🔴 **降级帧一律不写理论值**（用户 2026-11-23：白银基本面分位组合回测
        # 最后一笔 11-18 就平了、理论该是 0，界面却显示 −2）。原因就是那一轮
        # 行情是「没有本地历史，用实时原始价 2643 根」（见 `_signal_frame`）——
        # 原始价带换月跳空，算出来的信号和回测压根不是一回事。信号照算、单照下
        # （那是交易链路的事），但「理论」这一列只认**和回测同源**的前复权帧；
        # 帧一降级就冻住上一次的值，宁可不更新，也不给一个看着像真的错数字。
        if not bool(getattr(st, "frame_ok", True)):
            return
        last = series.iloc[-1]
        sig = 0 if pd.isna(last) else int(last)
        if sig == 0:
            st.theo_px, st.theo_day, st.theo_sig, st.theo_src = 0.0, "", 0, ""
            st.theo_lots, st.theo_lots_sig, st.theo_lots_src = 0, 0, ""
            return

        # ---- 开仓那一根：按**回测那台状态机**找，不再拿「信号段起点」当它 ----
        # 顺序很重要：手数**必须在下面那个 `rank >= order` 早退之前算**。早退是
        # 「价格已经有更可信的来源了，别再改价」，和手数是两件事 —— 放在早退
        # 后面，一旦这一波发过委托，`theo_lots` 就永远停在上一波的值上。
        #
        # 🔴 2026-11-23 用户两次点出这里的病根：
        #   「signal 段起点如果是 10-22，那确实会很多错误的」
        #   「实盘引擎的开始时间要和回测的一致」
        # `signal` 列**不是持仓状态**：AG 的段起点（09-24 00:00）比回测真正的
        # 开仓（09-23 21:00）还晚，RB 的段起点（10-22 09:00）比回测开仓
        # （09-09 22:00）晚六周 —— 拿它推手数当然对不上。现在整段交给
        # `ib_open_trade()`：同一份帧、同一个起点、同一台状态机。
        arr = series.to_numpy()
        k = n - 1
        while k > 0 and (int(arr[k - 1]) if not pd.isna(arr[k - 1]) else 0) == sig:
            k -= 1                       # 信号段起点（只在状态机给不出时兜底用）
        filled = k + 1 <= n - 1
        got = 0
        j = min(k + 1, n - 1)
        try:
            ref = float(df["open" if filled else "close"].iloc[j])
        except (KeyError, IndexError, TypeError, ValueError):
            ref = 0.0
        if e_long is not None or e_short is not None:
            # K 线内那一族：开仓不是「信号的下一根」，而是状态机说**真正成交过**的
            # 那根 —— 那一根定了手数，整笔持有期间不再变
            hit = self._ib_open_trade(st, df, e_long, e_short,
                                      x_long, x_short, stop_dist)
            if hit is not None:
                j, ref, got = hit
        # 🔴 价格**不折回原始价**（和 `theo_px` 那里相反），因为要跟回测对齐：
        # 回测就是用这份（前复权）帧算 `size_lots(cfg, open[i], ...)` 的。前后复权
        # 只差一个常数因子，`risk` 口径（ATR 也在同一空间）逐位相等；`capital`
        # 口径下回测本来也是按前复权价算的，折回来反而两边又分叉。而**末根的
        # adj_factor 恒为 1**，所以真正开仓那一刻，这里的数和实盘按实时报价算出
        # 来的手数是同一个。
        # 🔴 只有「换了方向」或「还没算过」才重算手数。同一波里每根都重算，就等于
        # 又拿当前 K 线算了一遍 —— 那正是用户 11-23 报的 bug。
        # 🔴 一律 `getattr` 取：自检里常有 `__new__` 造的精简桩（`_St`），
        # 少一个属性就 AttributeError，整条策略直接判「错误」
        prev_lots = int(getattr(st, "theo_lots", 0) or 0)
        prev_sig = int(getattr(st, "theo_lots_sig", 0) or 0)
        prev_src = str(getattr(st, "theo_lots_src", "") or "")
        if got > 0:
            # 状态机报得出「有一笔没平的开仓」⇒ 它**就是**回测口径，每根都按它写。
            # 置的同一个输入不会变（同一笔持仓算出来恒等），所以不会漂。
            # 🔴 它可以盖掉 `ib` 档：`ib` 档记的是「盘中触发那一刻 `st.lots` 是多少」，
            # 而 `st.lots` 是按**当根** ATR 重算的 —— 正是用户说不许拿来当显示值的
            # 那个数（2026-11-23：AG 理论 −1、回测 −2，其余 6 个品种都对上）。
            if int(got) != prev_lots or prev_sig != sig or prev_src != THEO_SIG:
                st.theo_lots = int(got)
                st.theo_lots_sig = int(sig)
                st.theo_lots_src = THEO_SIG
                # 只在**变了**的时候落一行，方便事后核「这一笔到底怎么来的」
                # （自检里的精简桩没有 `_diagw`，所以 getattr 兜一下）
                diag = getattr(self, "_diagw", None)
                if diag is not None:
                    diag("LOTS", f"{st.title()}：开仓手数 {prev_lots} → {int(got)}"
                                 f"（回测口径，开仓根 {df.index[j]:%m-%d %H:%M}）")
        elif prev_src != THEO_IB:
            # 状态机说现在空仓：才退回「信号下一根」估一个（总比 0 手强）。
            # `ib` 档（盘中真触发过、这一根还没收盘）不许盖 —— 那是最接近实时
            # 真相的一档，收盘后状态机会自己追上并把来源降回 `sig`。
            got = self._entry_lots_at(st, df, j, ref, stop_dist)
            if got > 0 and (int(got) != prev_lots or prev_sig != sig
                            or prev_src != THEO_SIG):
                st.theo_lots = int(got)
                st.theo_lots_sig = int(sig)
                st.theo_lots_src = THEO_SIG

        # 同一波已经拿到策略自己的价（`order`/`ib`）⇒ 推断出的开盘价原地作废，
        # 不许覆盖。覆盖了就会出现「持仓盈亏按对手价、当日盈亏按开盘价」两套。
        if (int(getattr(st, "theo_sig", 0) or 0) == sig
                and self._theo_rank(st) >= _THEO_RANK[THEO_ORDER]):
            return
        # 🔴 **价格每根 bar 都重算**，不搞「方向没变就沿用上次」。
        # 那套缓存会锁死一个错值：2026-11-20 16:49 本地库被锁、`_signal_frame`
        # 退回「实时原始价 2642 根」，当时按降级帧记下的建仓价（15,740）就一直
        # 挂着 —— 等真正的库内前复权回来了也不再更新，于是持仓盈亏按 15,740、
        # 当日盈亏按昨结算 15,545，两个数永远对不上。重算一次只是往回扫几千个
        # int，新 bar 才触发一次，可以忽略。
        # 🔴 **成交价取下一根的「开盘」，不是收盘**（用户 2026-11-20 指出：
        # 「开仓价格是开盘价格+成本，收盘价格比这个低，应该有亏损，不是 0」）。
        # 信号是**收盘**才定下来的，回测里 `position(t) = signal(t-1)`，成交发生在
        # t 那根的**开盘**。以前取 close[k+1]，于是「今天开仓」的理论价等于今天的
        # 收盘价 —— 两列必然都是 0，看着像没仓。
        # 已经翻到最后一根（没有下一根）时，这笔仓**还没成交**：按这根收盘记，
        # 盈亏 0，避免拿窗口内不存在的开盘价（那是偷未来）。
        # 🔴 开仓价优先取 `ref`：K 线内那一族它是状态机算出来的**成交基准价**
        # （`max(开, 触发价)`，跳空时就是开盘价），和回测 `Trade.entry_price`
        # 的毛价口径一致；其余策略 `ref` 本来就是「下一根开盘」。
        col = "open" if filled else "close"
        px = float(ref) if (ref == ref and ref > 0) else 0.0
        if not px:
            try:
                px = float(df[col].iloc[j])
            except (KeyError, IndexError, TypeError, ValueError):
                try:
                    px = float(df["close"].iloc[j])
                except (KeyError, IndexError, TypeError, ValueError):
                    return
        # 🔴 帧是**前复权**价，`_last_px` / `_pre_px` 是**实时报价**（原始价）。
        # 换月前后 adj_factor ≠ 1，直接拿前复权价当建仓价，乘出来的盈亏是错的
        # 价格空间的差（AG 早年能差 30%）。统一折回原始价，和报价同一个口径。
        try:
            f = float(df["adj_factor"].iloc[j])
            if f > 0 and f == f:
                px /= f
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        if not (px == px and px > 0):
            return
        # 加**每边成本**（滑点 + 手续费，折成价格）：开仓价 = 开盘价 ± 成本。
        # 多头买贵、空头卖贱，所以按信号方向加号。
        cost = self._side_cost_px(st, px)
        if cost > 0:
            px += sig * cost
        try:
            day = tqdata.trading_days(df.index).iloc[j]
            theo_day = pd.Timestamp(day).strftime("%Y-%m-%d")
        except Exception:                # noqa: BLE001 - 日历读不出来就退回自然日
            theo_day = pd.Timestamp(df.index[j]).strftime("%Y-%m-%d")
        st.theo_px = px
        st.theo_day = theo_day
        st.theo_sig = sig
        st.theo_src = THEO_SIG

    # ---- 理论建仓价：由策略自己写（主口径） ----

    @staticmethod
    def _theo_rank(st: SlotState) -> int:
        """当前 `theo_px` 的可信程度，见 `SlotState.theo_src` 那张表。"""
        return _THEO_RANK.get(str(getattr(st, "theo_src", "") or ""), 0)

    def _put_theo(self, st: SlotState, px: float, sig: int, src: str,
                  day: str = "") -> bool:
        """把策略自己给出的建仓价写进去（`src` 见 `SlotState.theo_src`）。

        覆盖规则只有一条：**低档不许盖高档**。具体是

        - `order`（真实下单价）盖 `ib`（触发价）盖 `sig`（推断）；
        - 同档不覆盖 —— 同方向加仓会把「建仓价」刷成加仓价，持仓盈亏立刻假
          涨一截；反过来 `order` 一直在盖 `order` 也会让每次追单都把成本往上挪。
        - 方向变了（`sig` 和记着的那一波不同）**无条件覆盖**：这是新的一仓。

        返回是否真的写了。
        """
        px = float(px or 0)
        if not (px == px and px > 0) or not sig:
            return False
        cur_sig = int(getattr(st, "theo_sig", 0) or 0)
        rank = _THEO_RANK.get(src, 0)
        if cur_sig == sig and self._theo_rank(st) >= rank:
            return False
        st.theo_px = px
        st.theo_day = day or daylimit.trading_day()
        st.theo_sig = int(sig)
        st.theo_src = src
        return True

    def _note_order_entry(self, st: SlotState, delta: int, px: float) -> None:
        """`_send()` 出口：**策略真正下出这一笔**时记下它的价格（主口径）。

        为什么用下单价而不是成交回报：成交回报要到 `_chase` 下一轮才回来，
        而且**这张页面的两个数本来就是理论值**（用户 2026-11-20「和实际账户
        交易无关」）—— 拿不到回报时（挂单没成、被拒、非交易时段）也得有个数。
        下单价是策略在这一刻自己算出来的、真正报进交易所的价，已经是这一笔
        最真实的「策略给的价格」。

        成本**不再另加**：`_price_at()` 取的是**对手价**（买 ask1 / 卖 bid1），
        滑点已经含在价差里，再叠一层 `_side_cost_px` 就是双重计算。这也和账户
        的「开仓均价」同口径，用户两个数能直接对着看。

        只在**开仓 / 反手**时写（`delta` 方向 == 目标方向）。减仓不写 —— 减仓
        的低价不是建仓价。
        """
        tgt = int(getattr(st, "target", 0) or 0)
        if not tgt or not delta:
            return
        if (1 if tgt > 0 else -1) != (1 if delta > 0 else -1):
            return                      # 这一笔是平仓，不是建仓
        self._put_theo(st, px, 1 if tgt > 0 else -1, THEO_ORDER)

    def _note_ib_entry(self, st: SlotState, px: float, sig: int) -> None:
        """K 线内触发：用**策略算出来的触发价**记建仓价（`ib` 档）。

        非规则策略没有触发价这一说，`_ib_on()` 已经把这条路口关掉了。紧跟着的
        `_send()` 会用真实对手价把它升成 `order` 档；这里兜的是「这一轮还没
        发出去」的窗口（没盘口、刚开盘、被门禁挡住），那时候显示触发价总比
        显示「—」或者拿下一根开盘瞎猜准。

        触发价是**纯触发位**，不含滑点，所以这里把每边成本叠上去，和回测
        `fill = max(open, entry_long) + slip` 同一个意思。
        """
        cost = self._side_cost_px(st, px)
        if cost > 0:
            px = px + sig * cost
        self._put_theo(st, px, sig, THEO_IB)

    def _side_cost_px(self, st: SlotState, px: float) -> float:
        """**每手每一边**的成本，折成价格（元/吨、元/点）。

        为什么显示上也要算成本：用户 2026-11-20 的口径是「开仓价 = 开盘价 + 成本」。
        不算的话，理论开仓价正好等于开盘价，一跳的滑点就白送了，读起来像「白捡
        一个开盘价」。口径和回测**同一个成本模型**（`costs.apply` 那一套：手续费
        按手或按成交额 + 跳数滑点，或 bp 滑点），所以两边的「理论盈亏」能对得上。

        取不到品种信息 / 成本档位就返回 0：显示上少算一点，不该因为读不到表就
        把整列变成「—」。
        """
        try:
            from ttrade import costs as _costs

            p = _costs.profile(getattr(st.cfg, "cost_profile", "") or None)
            si = symbols_info.info(st.cfg.symbol, st.cfg.exchange)
            fee_mult = float(p.get("fee_mult") or 0.0)
            # 滑点（价格单位）
            if str(p.get("mode") or "tick") == "bp":
                slip = px * float(p.get("slip_bp") or 0.0) / 1e4
            else:
                tick = float(si.get("tick") or 0.0)
                slip = tick * float(p.get("slip_ticks") or 0.0)
                if tick <= 0:                    # 没有 tick 时和回测一样退回 bp
                    slip = px * 0.0002
            # 手续费（价格单位）
            fee_val = float(si.get("fee_value") or 0.0) * fee_mult
            if str(si.get("fee_mode")) == "lot":
                mult = float(si.get("mult") or 0.0) or self._volume_multiple(st)
                fee = fee_val / mult if mult > 0 else 0.0
            else:
                fee = fee_val * px
            out = slip + fee
            return out if (out == out and out > 0) else 0.0
        except Exception:                        # noqa: BLE001 - 只影响显示
            return 0.0

    def _latch_theo_lots(self, st: SlotState) -> None:
        """没有信号序列可倒推时（机器学习那一族）：方向一变就把这一刻的手数锁存。

        与 `_note_theo_entry` 的分工：那边能从信号波段的起点倒推出「开仓那一根」，
        更准；这里只知道「方向是在这会儿变的」，于是把当下这根的手数当成开仓
        手数。共同点是**锁存** —— 同一波持有期间绝不重算，否则 ATR 一漂，
        「理论」列就跟着漂（用户 2026-11-23 报的就是这个）。
        """
        sig = int(getattr(st, "signal", 0) or 0)
        sig = 1 if sig > 0 else (-1 if sig < 0 else 0)
        if not sig:
            st.theo_lots, st.theo_lots_sig, st.theo_lots_src = 0, 0, ""
            return
        if (sig != int(getattr(st, "theo_lots_sig", 0) or 0)
                or int(getattr(st, "theo_lots", 0) or 0) <= 0):
            st.theo_lots = max(1, int(st.lots or 1))
            st.theo_lots_sig = int(sig)
            st.theo_lots_src = THEO_SIG

    def _sync_theo_entry(self, st: SlotState, px: float = 0.0) -> None:
        """兜底：`_signal` 没记过理论建仓价时，用当前价补一个（`sig` 档）。

        `_note_theo_entry` 从**信号序列**倒推建仓点，规则 / 因子两条路都有；
        只有机器学习那条（信号是模型给的）没有序列可倒推，只能在「目标刚变
        方向」的那一刻按当时的价格记一笔。方向没变就不动，否则每 tick 都会
        把建仓价刷成最新价，两列永远显示 0。

        ML 的真实下单价由 `_send()` 记（`order` 档，比这里的标记价可信），
        这里只是「委托还没发出去」那一下的占位。
        """
        tgt = int(getattr(st, "target", 0) or 0)
        sig = 1 if tgt > 0 else (-1 if tgt < 0 else 0)
        if not sig:
            st.theo_px, st.theo_day, st.theo_sig, st.theo_src = 0.0, "", 0, ""
            return
        got = float(px or 0) or self._last_px(st.underlying)
        if not (got == got and got > 0):
            return
        self._put_theo(st, got, sig, THEO_SIG)

    def _theo_lots_qty(self, st: SlotState) -> int:
        """**方向 × 开仓手数**（锁存值）。K 线内那族的理论仓、普通策略的 target
        都从这里来。

        `theo_lots` 为 0（历史不够 / 还没算过）时退回 `lots` 兜一下 —— 退回的那个
        会随 ATR 漂，但总比显示 0 手强，而且只在启动第一根之前出现。
        """
        sig = int(getattr(st, "signal", 0) or 0)
        sig = 1 if sig > 0 else (-1 if sig < 0 else 0)
        if not sig:
            return 0
        n = int(getattr(st, "theo_lots", 0) or 0) or int(st.lots or 0)
        return sig * n

    def _theo_qty(self, st: SlotState) -> int:
        """策略口径的**理论持仓**（手，带符号）。对应回测里的 position 那一列。

        - 普通策略：就是 `target`（而 `target` 现在由 `_theo_lots_qty` 定，也就是
          「方向 × 开仓手数」，不再随当前 K 线的 ATR 漂）；
        - **K 线内那一族**（`sbbox_ib` / `strat_im_orb` 这类）：`target` 是「下单
          目标」——盘中没碰到触发价就是 0，所以理论仓必须单独按信号算。拿
          `target` 当理论仓，界面会一直说「策略空仓」，而回测在同一根 bar 上明明
          是持仓的：2026-11-23 用户报「sbbox多品种波段1h 的 34 个品种理论持仓全是
          0，回测里 29 个品种有仓」，这 29 个正好是 `run_strategy("sbbox_ib")`
          最后一根上非零信号的品种数。

        🔴 手数取的是**开仓那一根定的那个数**（`theo_lots`），不是当前这根 K 线
        按现在的 ATR 算出来的（用户 2026-11-23：「理论列需要按照开仓的手数来算，
        不能根据当前 k 线来算，实际上应该就从开仓那边获取，而不是自己擅自计算」）。
        以前用 `st.lots`（每根 bar 重算）时，risk 口径下 ATR 从 12 掉到 4.7，
        同一笔仓的理论手数就从 39 手漂到 106 手，和回测交易记录里那一笔永远
        对不上。现在 `theo_lots` 只在开仓那根算一次（K 线内那族由
        `ib_open_trade()` 按回测状态机给出开仓那根），持有期间不动；真的触发
        时会刷新成实际下单的手数。

        🔴 与账户、台账、成交一律无关（用户 2026-11-20 两次强调的口径）。
        也**不**能拿它去下单（K 线内那族的下单目标必须由触发价给，见
        `_refresh_signal` 里那一支的注释）。
        """
        if self._ib_on(st):
            return self._theo_lots_qty(st)
        return int(getattr(st, "target", 0) or 0)

    def _slot_pnl(self, st: SlotState) -> tuple[float, float, float, str]:
        """这条策略的「品种价格 / 当日盈亏 / 持仓盈亏」——**全是理论值**。

        🔴 口径（2026-11-20 用户两次明确）：
        「持仓盈亏和当日盈亏都是策略理论上的，和实际账户交易无关」
        「不需要和实际交易和台账的情况拿来算」

        所以两个数**只读三样东西**：理论仓位（`_theo_qty`：普通策略 = `target`
        = signal × lots，K 线内那一族 = signal × lots）、
        理论建仓价 `theo_px`（**策略自己写下来的**，见 `SlotState.theo_src`）、
        当前行情价。台账 `held` / 均价 `cost` / 已落袋 `real_day` **一律不参与**
        —— 以前拿它们算，于是「实际成交和理论不一致」时这两列会跟着实际仓走，
        看起来像策略赚了钱，其实赚的是手工单的钱。

        🔴 `theo_px` 的主口径是**策略的下单价 / K 线内触发价**，不是显示层自己
        按「下一根开盘 + 成本」推出来的（用户 2026-11-20：「理论开仓价格，应该
        是从实盘策略中直接获取，不然如果 k 线中开仓，又算不对了」）。推断只在
        策略还没发出过委托时兜底。

        - **品种价格** = 当前主力合约最新价（没仓也显示）。
        - **持仓盈亏** = (最新价 − 理论建仓价) × 理论手数 × 乘数。
        - **当日盈亏** = (最新价 − 当日基准) × 理论手数 × 乘数；**当日基准**：
          理论仓是**今天**建的 ⇒ 就是建仓价（所以当天开的仓两个数必然相等）；
          隔夜仓 ⇒ 昨结算（只算今天这一天的波动）。这正是回测里
          `position(t)=signal(t-1)` 那个假设下的「今天赚了多少」。

        以前当日盈亏 = 昨结算持仓 + 台账已落袋。问题：新建的理论仓也拿合约的
        昨结算当成本 ⇒ 今天开的仓会算出一个巨大的假当日盈亏（IC 11-20 实测
        +55,400，而持仓盈亏只有 +8,080，差 6.9 倍）。
        """
        price = self._last_px(st.underlying)
        qty = self._theo_qty(st)
        if not qty:
            # 理论空仓：没有仓就没有盈亏。不能写 None —— 用户会以为「算不出来」
            return price, 0.0, 0.0, ""
        mult = self._volume_multiple(st)
        if mult <= 0:
            return price, float("nan"), float("nan"), "乘数没维护，算不出盈亏"
        px = price if price > 0 else 0.0
        if px <= 0:
            return price, float("nan"), float("nan"), "没有行情价，算不出盈亏"
        entry = float(getattr(st, "theo_px", 0.0) or 0)
        # 方向对不上 = 手上那个价是**上一波**的（刚翻向、这一笔还没发出去）。
        # 拿它算会把上一波的建仓价按在新方向上，盈亏符号直接反掉
        if int(getattr(st, "theo_sig", 0) or 0) != (1 if qty > 0 else -1):
            entry = 0.0
        if entry <= 0:
            return (price, float("nan"), float("nan"),
                    "策略还没给出建仓价（这一笔的委托还没发出去）")
        pos = (px - entry) * qty * mult
        theo_day = str(getattr(st, "theo_day", "") or "")
        if theo_day and theo_day == daylimit.trading_day():
            pre = entry                  # 今天开的仓：当日盈亏 ≡ 持仓盈亏
        else:
            pre = self._pre_px(st.underlying) or entry
        day = (px - pre) * qty * mult
        note = ""
        if not theo_day:
            note = "理论建仓日未知，当日盈亏按建仓价算"
        return price, day, pos, note

    def _realized_today(self, st: SlotState) -> float:
        """本交易日已落袋的平仓盈亏。跨交易日自动清零（夜盘 20 点后算次日）。"""
        today = daylimit.trading_day()
        if st.real_day_of != today:
            st.real_day_of = today
            st.real_day = 0.0
        return st.real_day

    @staticmethod
    def _terms(cfg) -> tuple[float, float, str]:
        """乘数 / 保证金率 —— **实盘和回测只读「数据管理 → 品种信息」那一张表**。

        🔴 以前实盘走天勤合约上的 `quote.volume_multiple` / `quote.margin`，
        回测走这张表，两边天生对不上，而实盘那一侧还有一串不确定性：

        - 主连（`KQ.m@`）压根没有 `margin`，恒 0；
        - 模拟 / 快照环境不推这个字段，也恒 0；
        - 真合约的 `margin` 按**当日结算价**天天变，同一天里价格一动比例就漂；
        - 上面任一情况都掉进「每手保证金 = 0 → 保底 1 手」，**不报错**。

        实测同一条 T 策略一天之内手数可以在 1~4 手之间跳，而回测永远是 4~5 手 ——
        用户没法解释、也没法对账。

        现在两边同源：**表的维护只有一个入口**（「从天勤刷新」一次写进来，或者
        手填），下多少手事先就能算出来，历史回测也能复现。代价是保证金率不再
        跟交易所逐日变动 —— 所以表里该填**实盘真实冻结**的比例（含经纪商加收），
        而不是交易所裸标准，否则实盘会因为保证金不足被拒单。

        返回 `(乘数, 保证金率, 来源说明)`。表里没维护（0）才兜底，并写进说明里，
        免得「保底 1 手」再次变成静默失败。
        """
        sym = str(getattr(cfg, "symbol", "") or "")
        ex = str(getattr(cfg, "exchange", "") or "")
        si = symbols_info.info(sym, ex)
        try:
            mult = float(si.get("mult") or 0)
            rate = float(si.get("margin") or 0)
        except (TypeError, ValueError):
            mult, rate = 0.0, 0.0
        bits = []
        if mult <= 0:
            bits.append("乘数没维护")
        elif mult < 2:
            bits.append(f"乘数 {mult:g} 可疑（国内期货不会这么小），去品种信息页刷一下")
        if rate <= 0:
            rate = default_margin_rate(sym)
            bits.append(f"保证金率没维护，按 {rate:.1%} 兜底")
        src = "品种信息表" + ("｜" + "；".join(bits) if bits else "")
        return mult, rate, src

    def _volume_multiple(self, st: SlotState) -> float:
        """这个合约每手多少单位（合约乘数）。和回测同源（读品种信息表）。"""
        return self._terms(st.cfg)[0]

    def _lots(self, st: SlotState, price: float, atr_value: float) -> int:
        """这条策略满仓该下几手。

        - `capital`：手数 = 本金 × 资金比例 / 每手保证金。100 万本金取 30%，
          就是拿 30 万铺 10 倍杠杆的品种 = 300 万名义价值；1 手 10 万的合约正好
          30 手。国债保证金低（3% 左右），同样的钱自然铺出更多手 —— 这就是
          「杠杆按品种自己的保证金比例」的意思。
        - `risk`：手数 = 本金 × 止损比例 / (止损距离 × 合约乘数)。止损距离 =
          倍数 × ATR，和策略自己的止损口径一致，所以「打止损就亏 1% 本金」
          是真的 1%。
        - `budget`：「按分配资金定手数」，手数 = 分配资金 ÷ **名义价值**
          （价格 × 乘数），四舍五入且最少 1 手。网格策略用这个口径（股指一手
          名义就 80~120 万，按保证金算会算出 0 手），回测里一直是这么算的 ——
          实盘以前漏了这一支，会掉进 `capital` 分支，两边手数差一个保证金率
          的倒数。
        - `unit`：名义敞口 = 本金（`principal`，不是分配资金）。实盘只能整手，
          所以四舍五入取整并保底 1 手；回测里它可以是小数。
        - `fixed`：老行为，固定手数。

        🔴 手数一变就记一行**凭什么**：手数的四个输入（预算 / 乘数 / 价格 /
        保证金率）里，乘数和保证金率现在**和回测同源**（品种信息表），所以
        回测报 4 手、实盘下 1 手这种「对不上账」不该再出现；真出现了，日志这
        一行会直接指出是哪个数不同（价格另算，实盘是实时价、回测是收盘价）。
        """
        n, why = self._size(st, price, atr_value)
        if int(n) != int(getattr(st, "lots", 0) or 0):
            self._log(f"{st.title()}：{int(n)} 手 —— {why}")
        return int(n)

    def _size(self, st: SlotState, price: float, atr_value: float) -> tuple[int, str]:
        """算手数，并返回「这个数是怎么来的」（给 `_lots` 记日志用）。

        🔴 **公式只有一份**：`backtest.size_plan(..., integer=True)`。这里只做
        三件事 —— ① 把实盘配置映射成回测配置（`bt_cfg`）；② 传 `integer=True`
        （实盘只能整手，这是物理约束，不是另一个口径）；③ 把返回的中间量讲成
        人话给日志。

        以前这里和 `size_lots` 各写一份五分支实现，靠一条自检钉着。改一处忘
        另一处就分叉，而且只在某个口径、某个价位上分叉（比如 0.45 手那条线），
        界面上看不出来 —— 用户只能看到「回测 4 手、实盘 1 手」。
        """
        cfg = st.cfg
        mode = (cfg.sizing or "capital").lower()
        # `_terms` 只为了在日志里点名「乘数没维护 / 保证金率兜底」——数字不吃它。
        # 但**乘数是 0 这一条必须当闸门**：`replay.make_bt` 对缺失的乘数兜底成
        # 1.0（回测那边「别崩」够了），照它算就是 105,033 ÷ (110×1×3%) = 31,810 手。
        # 2026-11-23 自检当场抓到这一条：共用公式之后，安全网不能跟着丢。
        mult, _rate, src = self._terms(cfg)
        if mode != "fixed" and mult <= 0:
            return 1, f"乘数没维护（{src}）：先 1 手，不按兜底乘数放大手数"
        try:
            bt = bt_cfg(cfg)
        except Exception as exc:              # noqa: BLE001 - 造不出配置也要能下单
            return 1, f"造不出回测配置（{type(exc).__name__}: {exc}）：先 1 手"
        n, plan = size_plan(bt, float(price), 0.0, atr_value, integer=True)
        n = int(n)
        mode = str(plan.get("mode") or mode)
        mult = float(plan.get("mult") or 0.0)
        rate = float(plan.get("margin_rate") or 0.0)
        raw = float(plan.get("raw") or 0.0)
        per = float(plan.get("per") or 0.0)
        budget = float(plan.get("budget") or 0.0)

        if mode == "fixed":
            return n, f"固定手数 {n}"
        if mode == "risk":
            stop_px = float(cfg.stop_mult or 0) * float(atr_value or 0)
            if not plan.get("note"):
                return n, (f"risk 口径，止损额 {float(cfg.risk_amount or 0):,.0f} ÷ "
                           f"(止损距离 {stop_px:.4g} × 乘数 {mult:g}，{src})")
            return n, f"%s：先 {n} 手" % plan["note"]
        if mode == "budget":
            if plan.get("note"):
                return n, f"{plan['note']}：先 {n} 手"
            return n, (f"budget 口径，预算 {budget:,.0f} ÷ 名义 {per:,.0f}"
                       f"（乘数 {mult:g}，价 {price:.3f}）")
        if mode == "unit":
            if plan.get("note"):
                return n, f"{plan['note']}：先 {n} 手"
            why = (f"unit 口径，本金 {float(plan.get('principal') or 0):,.0f} ÷ "
                   f"名义 {per:,.0f}（乘数 {mult:g}，价 {price:.3f}）")
            if plan.get("floored"):
                why += (f"；本金只够 {raw:.2f} 手，实盘按 1 手兜底"
                        "（回测在这是 0~小数手）")
            return n, why

        # capital（默认）
        if plan.get("note"):
            return n, f"{plan['note']}（{src}）：先 {n} 手"
        why = (f"capital 口径，预算 {budget:,.0f} ÷ 每手保证金 {per:,.0f}"
               f"（{src}：乘数 {mult:g} × 价 {price:.3f} × 保证金率 {rate:.2%}）")
        if plan.get("floored"):
            # 回测在这是 0 手（四舍五入到 0，钱不够半手），实盘保底 1 手而不是
            # 不动 —— 但这一手会明显超出预算，必须说出来，否则就是「开了但不知道
            # 为什么被拒」。
            # 🔴 分界点是 0.5 不是 1：0.6 手四舍五入就是 1 手，两边一致，不该报警
            why += (f"；预算只够 {raw:.2f} 手（不足半手，回测在这里是 0 手），"
                    "实盘按 1 手兜底，可能因保证金不足被拒")
        elif per > 0 and n > raw:
            # 四舍五入会略微超出预算（4.77 手 → 5 手 = 多占 4.8%）。这是这个
            # 取整口径的必然代价，写清楚才不会让人以为占用算错了
            over = n * per / max(budget, 1e-9) - 1.0
            why += (f"；四舍五入进位，占用 {n * per:,.0f}，超预算 {over:.1%}")
        return n, why

    # ---- 调仓 ----

    @staticmethod
    def _bar_seconds(period: str) -> float:
        """一个周期多少秒。

        `PERIODS` 记的是 (源周期, 几倍) —— `15m` 是 `("5m", 3)`，所以秒数是
        「源周期自己的秒数 × 倍数」，不是「倍数 × 60」。按后者算会把 15m 当成
        3 分钟，行情活性判据提前 5 倍判死，收盘前的调仓就全被拦掉了。
        """
        unit, n = tqdata.PERIODS.get(period, (period, 1))
        digits = "".join(ch for ch in unit if ch.isdigit())
        num = float(digits) if digits else 1.0
        if unit.endswith("m"):
            return float(n) * num * 60.0
        if unit.endswith("h"):
            return float(n) * num * 3600.0
        if unit.endswith("w"):
            # 周线不在 `PERIODS` 里（走默认分支，unit 就是 "1w"），但万一哪天
            # 放开了实时，7 天不能算成 1 天 —— 活性判据会把「刚来一根」误判成
            # 「行情停了」，收盘前该调仓的全被拦掉
            return float(n) * num * 7 * 86400.0
        return float(n) * num * 86400.0

    def _market_alive(self, st: SlotState) -> bool:
        """最近一根 bar 是不是刚来的 —— 判断行情还在不在走。

        为什么需要：机器学习信号在后台线程算，算完那一刻 `ml_ready` 就翻成
        True，`_rebalance` 当场就会算出新目标仓。如果这一刻已经收盘（日盘
        15:00、夜盘 23:00 结束），单子发出去只会被打回，攒够 3 次策略就停机。
        「多久没来新 bar」是这里唯一拿得到、又不依赖交易所日历的活性判据。
        """
        try:
            bar_s = float(st.last_bar) / 1e9      # 天勤 datetime 是纳秒
        except (TypeError, ValueError):
            return False
        if bar_s <= 0:
            return False
        # 有的品种 bar 记开盘、有的记收盘：收盘记法下 age 是负的，同样算「没过期」
        age = time.time() - bar_s
        return age < self._bar_seconds(st.cfg.period) * 1.1

    def _open_now(self, cfg) -> bool:
        """这条策略此刻有没有品种在**交易时段**内 —— 信号与下单的唯一门禁。

        判据是品种信息表里的「交易时段」（`symbols_info.in_session`），
        **不是** `_market_alive()`。两者解决的是不同问题，别合并：

        - `_market_alive()`：**断流**检测（最近一根 bar 是不是刚来的）——
          ML 信号在后台线程算，算完那一刻行情可能已经断了；
        - `_open_now()`：**交易时段**（这个品种现在根本不该有行情）。

        🔴 为什么非加不可：天勤模拟账户在收盘之后照样推 bar、推最新价，
        CTP 也会把最后一次快照一直往外吐 —— `_market_alive()` 会一路判 True，
        策略就跟着在非交易时段触发、开仓、下单。这个门禁是「按交易所日历」
        的，不看行情本身活不活。

        板块级策略（没有 `symbol`、只有 `multi_symbols`）按「**任一**成员开市」
        算：各腿的交易时段本来就不同，只要有一腿在交易，这一轮就该跑。
        连品种都没有（配置还没填完）时返回 True —— 宁可让它按老样子跑，
        也不要因为查不到时段把一条正在跑的实盘策略静默停掉。
        """
        sym = str(getattr(cfg, "symbol", "") or "")
        ex = str(getattr(cfg, "exchange", "") or "")
        if sym:
            return symbols_info.in_session(sym, ex)
        keys = list(getattr(cfg, "multi_symbols", None) or [])
        if not keys:
            return True
        for k in keys:
            s, _, e = str(k).partition("_")
            if s and symbols_info.in_session(s, e or ex):
                return True
        return False

    def _net_pos(self, symbol: str) -> int:
        if self._pos is None:
            return 0
        p = self._pos.get(symbol)
        if p is None:
            return 0
        try:
            return int(p.pos_long) - int(p.pos_short)
        except (TypeError, ValueError):
            return 0

    def _others_book(self, st: SlotState, symbol: str) -> int:
        """别的策略在这个合约上的账本合计。

        用途只有一个：从账户净仓里把「别人的仓」剥掉，剩下的才可能算本策略的。
        账户 -11 手、别的策略账本合计 -11 手 ⇒ 账户里没有无主的仓，本策略的账本
        是 0 就是真的 0（纯建仓可以放行）；反过来差 -1 手，说明账户上那 1 手没人
        认领（多半是重启丢了账本），这时不能照着目标再开一笔。
        """
        total = 0
        for other in self._slots.values():
            if other is st:
                continue
            total += int(other.held.get(symbol, 0))
        return total

    def _side_pair(self, symbol: str) -> tuple[int, int]:
        """(多头手数, 空头手数)。**锁仓时两边都非零**。

        只用 `_net_pos` 看账户是不够的：多 4 空 4 的净仓是 0，和「什么仓都没有」
        长得一模一样。2026-11-11 的实盘就是这样 —— ag2612 多 4 空 4、cu2612 多 1
        空 1、lc2701 多 1 空 1，三笔共 47 万保证金占着，监控器却报「一致」，
        一键对齐点下去一笔都不发。要判断仓位对不对得上，必须看两条腿。
        """
        if self._pos is None:
            return 0, 0
        if symbol not in self._pos:
            return 0, 0
        return self._side_pos(symbol, "long"), self._side_pos(symbol, "short")

    def _pos_detail(self, symbol: str) -> str:
        """账户这个合约的「今 / 昨」明细，拼成一句挂在 `[ALIGN]` 日志后面。

        🔴 为什么值得多打这一行（2026-11-23）：那天监控器上 `lc2701` 写着
        「账户 +3」、交易页的持仓表却写着「净 -1 空1」，同一时刻两个界面给出
        两个数，而日志里只有一个净数 —— 分不清是 `_pos` 没跟上、还是今昨仓被
        算重复了。把 `多/空/今多/今空/昨多/昨空` 全打出来，下次一跑就能定位。
        取不到明细就返回空串：这只是诊断，不能因为它让对齐失败。
        """
        try:
            out = []
            for side, cn in (("long", "多"), ("short", "空")):
                got = self._split_pos(symbol, side)
                total = self._side_pos(symbol, side)
                if got is None:
                    out.append(f"{cn} {total}（今昨未就绪）")
                else:
                    out.append(f"{cn} {total}=昨{got[0]}+今{got[1]}")
            return " ｜ 明细 " + "，".join(out)
        except Exception:  # noqa: BLE001
            return ""

    def _pos_raw(self, symbol: str) -> str:
        """把账户接口里这个合约的**原始字段**打出来，给 `[MM]` 诊断用。

        监控器显示「账户 -2」而交易页持仓表里根本没有这个合约时，可选的成因
        只有两个：`_pos` 没跟上（活引用断了），或者读数口径错了（今昨仓叠了
        一遍）。只看净数永远分不清，所以把两条腿和今昨两个字段一起打。
        """
        try:
            if self._pos is None:
                return "_pos 还没拿到"
            p = self._pos.get(symbol)
        except Exception:  # noqa: BLE001
            return "读 _pos 失败"
        if p is None:
            return "不在 _pos 里（账户接口已经没有这个合约）"
        out = []
        for name in ("pos_long", "pos_short", "volume_long", "volume_short",
                     "pos_long_today", "pos_short_today",
                     "pos_long_his", "pos_short_his"):
            try:
                out.append(f"{name}={getattr(p, name, '-')}")
            except Exception:  # noqa: BLE001
                out.append(f"{name}=?")
        return " ".join(out)

    def _reanchor(self) -> None:
        """按账户接口重新拿一次持仓 / 委托的**引用**，并让比对缓存失效。

        🔴 为什么需要（2026-11-23）：`_pos` 只在 `_connect_account()` 里取一次，
        指望账户接口返回的是「活引用」（`broker_api` 和天勤 `Entity` 都承诺这
        一点）。实盘看到的却是：账户已经把 `T2612` 平掉了、交易页的持仓表里
        也没有它，组合页的监控器还写着「账户 -2、平空2」，一直不动 —— 活引用
        在某条链路上断了，而**断了不报错**，读到的永远是冻住的那一份。

        每 2 秒重挂一次是最便宜的兜底：`broker_api.get_position()` 返回的就是
        同一个 dict，天勤的 `get_position()` 只是重走一遍 `_data` 树，两者都
        不发网络请求、不占柜台查询额度。

        顺带把 `MISMATCH_GAP` 的缓存清掉：持仓刚变过，界面不该再等一个 5 秒
        周期才显示「已对齐」——用户看到的「刷新太慢」就是这个缓存。
        """
        now = time.time()
        if now - getattr(self, "_pos_ts", 0.0) < POS_ANCHOR_SEC:
            return
        self._pos_ts = now
        try:
            pos = self.api.get_position()
            if pos is not None and pos is not self._pos:
                self._pos = pos
            # 委托同理：天勤的 `get_order()` 也是活引用，断了就看不到在途单
            orders = self.api.get_order()
            if orders is not None and orders is not self._orders:
                self._orders = orders
        except Exception:  # noqa: BLE001
            return                       # 账户抖动：沿用上一份，别把策略带崩
        # 引用换过了就一定重算一次（重算本身很便宜，见 `MISMATCH_GAP`）
        self._mm_ts = 0.0

    def refresh_now(self) -> list[dict]:
        """界面「刷新」按钮：**立刻**重挂持仓引用 + 清掉比对缓存，返回当下的不一致。

        为什么要专门一个方法（2026-11-23 用户要求）：「一键对齐」旁边加个刷新，
        点一下就要看到最新的差额，而不是等 `MISMATCH_GAP=5` 秒的缓存过期、再等
        `_reanchor()` 的 `POS_ANCHOR_SEC=2` 秒节拍。

        三步每步都有用：
          - `_pos_ts = 0` → 下面 `_reanchor()` 这一拍**必定**重挂（否则 2 秒内的
            第二次点刷新会什么都不做，用户以为按钮坏了）；
          - `_reanchor()` → 账户接口的活引用断了也能续上（见那个方法的长注释）；
          - `_mm_ts = 0` → 下一次 `mismatch()` 不走缓存。

        🔴 还要**先把这一拍的查询抢出去**（`refresh_account(force=True)` 绕过
        每秒一个的闸门），再等一小会儿让回包落地 —— 否则「刷新」只是把手里
        那份旧数据重挂一遍，点完看着毫无变化（用户说的「点刷新也没用」）。

        返回的是**此刻**的不一致行；界面也会在 0.4 秒内收到一份新快照（worker 的
        推送节拍），两处内容一致。
        """
        pump = getattr(getattr(self, "api", None), "refresh_account", None)
        if pump is not None:
            try:
                pump(force=True)          # 这一拍一定要发出去
                time.sleep(0.35)          # 柜台回包几十到几百毫秒，等一轮再读
            except Exception:  # noqa: BLE001 - 账户抖动：沿用上一份
                pass
        self._pos_ts = 0.0
        self._reanchor()
        self._mm_ts = 0.0
        return self.mismatch(fresh=True)

    def _rebalance(self, st: SlotState) -> None:
        if not st.underlying:
            return
        if self._align_slot.pending:
            # 一键对齐的委托还在路上。它要达成的目标仓位和策略此刻算出来的
            # 未必一样（信号刚翻、手数刚变），两边一起动手就会「平了又开」。
            # 对齐是用户明确要求的动作，让它先落地。
            return
        if st.pending:
            return                       # 还有没追完的单，等它落地

        # 换月留下的老合约先平掉。一次只发一笔，等它落地再谈新合约 ——
        # 同时发两笔，账户上会出现一段「老合约还持着、新合约已开着」的双倍敞口。
        for sym in list(st.retired):
            have = self._trim(st, sym, int(st.held.get(sym, 0)))
            if have == 0:
                st.retired.remove(sym)
                if st.held.pop(sym, None) is not None:
                    self._save_book(st)
                continue
            if self._side_pos(sym, "long" if have > 0 else "short") < abs(have):
                # 账户明细还没跟上（刚成交）。这时候发出去，`_legs` 会把平不掉
                # 的部分当成开仓 —— 等下一轮，别抢
                st.last_action = "老合约持仓还没跟上，等下一轮"
                return
            self._send(st, sym, -have, closeable=abs(have))
            return

        # ---- 加载首轮：**不根据台账交易** ----
        # 用户 2026-11-17：「策略加载的时候不要自动去根据台账去做交易」。
        # 老行为是加载完立刻 `target - held` 把差额发出去 —— 用户刚发现仓位
        # 异常，单子已经在路上了。现在加载后的第一轮只把「这时的目标」记成基准，
        # 之后**目标没变就一手都不发**（账本和账户不一致交给「一键对齐」）；
        # 只有信号真的把目标改了，才按台账调仓。
        # 注意位置：换月段在上面，老合约平仓不受这条影响（换月仍无条件自动）。
        if not st.based:
            st.based = True
            st.based_target = int(st.target)
            st.last_action = (
                f"已加载：目标 {int(st.target):+d} 手、"
                f"账本 {sum(int(v) for v in st.held.values()):+d} 手，"
                "目标不变前不自动交易（要跟账户对齐请点一键对齐）")
            self._log(f"{st.title()}：{st.last_action}")
            return
        if int(st.target) == st.based_target:
            # 目标没变过 ⇒ **默认**这一轮不发单。这正是「加载即发单」被挡住的地方：
            # 重启后目标往往和停机前一样，而账本/账户差了十几手。
            #
            # 但「目标没变」不等于「什么都不该做」，两种情形必须放行（2026-11-19）：
            #
            # ① **减仓**。每天开盘执行的那笔平仓正好落在 `target == based_target`
            #    上 —— 信号出在昨天最后一根 bar，今天开盘执行（T+1），加载时算出来
            #    的 target 已经是 0，于是这笔平仓被这条闸永久挡住，账户里的仓一直
            #    挂着，只有点「一键对齐」才会平（用户 11-18、11-19 报的定时平仓）。
            #    减敞口造不出双倍仓位，交易所侧还按可平量校验，放行是安全的。
            #
            # ② **纯建仓**：账本 0、账户上也没有一分钱算得到这条策略头上。
            #    这时没有任何历史仓可以重复开，挡着只是把信号白白吞掉 ——
            #    用户看到的就是「目标 -1、账本 0，要我自己点一键对齐」。
            #
            # 真正要挡的是 11-17 说的那种：账本有仓（或账户上有「无主」的仓），
            # 重启后目标与停机前一样，一加载就把差额补出去。判据因此取「账户净仓
            # 减去其它策略账本」：不为 0 说明账户里有本策略说不清的仓，宁可不发。
            held_now = int(st.held.get(st.underlying, 0)) if st.underlying else 0
            if held_now and abs(int(st.target)) < abs(held_now):
                pass                                  # ① 减仓
            elif (self._pos is not None and not held_now and int(st.target)
                  and self._net_pos(st.underlying) == self._others_book(st, st.underlying)):
                pass                                  # ② 纯建仓
            else:
                return

        # 只跟**自己记的仓位**比。账户净仓是所有策略 + 手工单 + 别的平台的和，
        # 拿它当基准，同品种两条策略就会互相把对方的仓当成自己的 —— 一条想
        # 空 11 手、一条想空 0 手，账户 -11，两边都认为自己差 11 手，来回对打。
        self._adopt_if_needed(st)
        held = self._trim(st, st.underlying, int(st.held.get(st.underlying, 0)))
        delta = st.target - held
        if delta == 0:
            # 仓位对上了，之前的拒单记录可以清了
            st.rejects = 0
            st.reject_msg = ""
            # 基准跟着「已对齐的仓位」往前挪。不挪的话，加载时那个目标值在本进程
            # 里永远不可再交易：信号走一圈回到它，就会被当成「目标没变」挡住。
            if st.based_target != int(st.target):
                st.based_target = int(st.target)
            return
        if st.rejects >= MAX_REJECTS:
            return                       # `_chase` 已经停机，这里只是兜底
        if st.rejects and time.time() - st.reject_ts < REJECT_COOL:
            return                       # 刚被打回，冷一会儿再试
        if self._is_ml(st.cfg) and not self._market_alive(st):
            # 行情已经停了（收盘 / 断流）。机器学习信号是后台线程算的，算完那一刻
            # ml_ready 就翻成 True，这里当场就会算出新目标仓 —— 但收盘之后发出去
            # 的单只会被打回，攒够 3 次策略就停机了。等下一根 bar 再说。
            st.last_action = "行情已停（收盘或断流），等下一根 bar 再调仓"
            return
        if delta * held < 0:
            # 要平自己那部分仓。账户明细可能还没跟上（刚成交），这时候发出去，
            # `_legs` 会把平不掉的量当成开仓 —— 反手开一笔没道理的仓。
            need = min(abs(delta), abs(held))
            side = "long" if held > 0 else "short"
            if self._side_pos(st.underlying, side) < need:
                st.last_action = "账户持仓还没跟上，等下一轮"
                return
        self._send(st, st.underlying, delta, closeable=self._closeable(held, delta))

    @staticmethod
    def _closeable(held: int, delta: int) -> int:
        """这次调仓里，本策略手上能拿去平的手数。

        卖（delta<0）能平的是自己多的那部分，买（delta>0）能平的是自己空的那
        部分。方向相同就是纯加仓，没有可平的量。
        """
        if delta < 0:
            return max(0, int(held))
        if delta > 0:
            return max(0, -int(held))
        return 0

    def _trim(self, st: SlotState, sym: str, held: int) -> int:
        """账本比账户里实际能平的还多时，按账户收窄。

        只会发生在「有人手工平过、或者别的程序平过」之后：账户里已经没有这些手
        了，再照着账本去平会被交易所打回；更糟的是 `_legs` 会把平不掉的量当成
        开仓，反手开出一笔完全没道理的反向仓。所以这里**只收不放** —— 账户比
        账本多的时候不管，那些多半是别人的仓，本来就不该由这条策略处置。

        刚成交那几百毫秒账户明细还没跟上，`BOOK_GRACE` 之内不判，免得把正常
        在途的仓位误判成「被别人平了」。
        """
        if not held:
            return 0
        if time.time() - st.book_ts <= BOOK_GRACE:
            return held
        side = "long" if held > 0 else "short"
        have = self._side_pos(sym, side)
        if abs(held) <= have:
            return held
        fixed = (1 if held > 0 else -1) * have
        if fixed:
            st.held[sym] = fixed
        else:
            st.held.pop(sym, None)
        self._save_book(st)
        self._log(
            f"{st.title()}：{sym} 账本 {held:+d} 手，账户只有 {have} 手可平，"
            f"已按账户收窄到 {fixed:+d}（可能被手工或别的程序动过）"
        )
        return fixed

    def _adopt_if_needed(self, st: SlotState) -> None:
        """本机第一次启用持仓账本时，把账户里现有的仓位认给这条策略。

        升级上来的老用户账户里已经有仓了（老版本一直是「把账户对齐到目标」，
        所以账户 ≈ 目标）。账本从 0 开始的话，目标 -11 会在已有的 -11 之上
        再开 11 手，账户直接变 -22。所以**账本文件还不存在**这一次，按品种把
        账户净仓认下来。

        只认一次、只认一条：同品种还有别的策略在跑时不认（分不清是谁的）。
        文件一旦存在就再也不认 —— 之后账户里多出来的仓位是手工开的、别的平台
        开的，不属于任何策略，也不该被哪条策略拿去处置。
        """
        key = st.cfg.key()
        if not self._fresh_book or key in self._adopted:
            return
        self._adopted.add(key)
        sym = st.underlying
        if not sym or st.held.get(sym):
            return
        same = [s for s in self._slots.values() if s.cfg.key() == key and s.running]
        if len(same) > 1:
            self._log(f"{st.title()}：同品种有 {len(same)} 条策略在跑，"
                      "账户里已有的仓位分不清是谁的，不认领")
            return
        have = self._net_pos(sym)
        if not have:
            return
        st.held[sym] = have
        self._save_book(st)
        self._log(f"{st.title()}：认领账户里已有的 {sym} {have:+d} 手"
                  "（本机第一次启用持仓账本）")

    # ---- 日内额度 ----

    def _halt(self, reason: str) -> None:
        """额度用完就全部停机。继续重试只会把交易所的限制招来。"""
        if self._halted == reason:
            return
        self._halted = reason
        for st in self._slots.values():
            if st.running:
                st.running = False
                st.error = reason
        self._log(reason)

    def _order(self, **kw):
        """统一下单入口。额度不够返回 None，调用方按「没发出去」处理。

        计数在**发报文之前** +1：就算 `insert_order` 抛异常也认这一笔，因为报文
        可能已经到了交易所 —— 宁可少用一次额度，不要超。
        """
        if self.api is None:
            return None
        if not self.limit.order():
            self._halt(
                f"今日报单已达 {daylimit.DAY_ORDER_LIMIT} 次上限，已停止调仓"
                "（再发会被交易所限制交易）"
            )
            return None
        order = self.api.insert_order(**kw)
        # 记下来。程序下次启动时按这个名单把还挂着的撤掉
        orderbook.remember(self._book_key, self._order_id(order))
        return order

    def _cancel(self, order) -> None:
        """统一撤单入口。撤单次数也是交易所盯的一项。"""
        if self.api is None or order is None:
            return
        if not self.limit.cancel():
            self._halt(f"今日撤单已达 {daylimit.DAY_CANCEL_LIMIT} 次上限，已停止调仓")
            return
        try:
            self.api.cancel_order(order)
        except Exception as exc:  # noqa: BLE001
            # 🔴 撤单失败以前是**静默**的：那笔单还挂在交易所，界面却当它撤了，
            # 之后账户突然多出一个仓谁也说不清。现在必须留痕。
            self._log(f"撤单没成功：{exc}")
            self._diagw("ORD", f"撤单没成功：{exc}")

    def _side_pos(self, symbol: str, side: str) -> int:
        """单边总持仓（手）。`side` 是 long / short。"""
        p = self._pos.get(symbol) if self._pos is not None else None
        if p is None:
            return 0

        def _i(name: str) -> int:
            try:
                return int(getattr(p, name, 0) or 0)
            except (TypeError, ValueError):
                return 0

        return max(_i(f"pos_{side}"), _i(f"volume_{side}"))

    def _split_pos(self, symbol: str, side: str) -> tuple[int, int] | None:
        """(昨仓, 今仓)；明细还没跟上时返回 None。`side` 是 long / short。

        天勤有**两套**字段描述同一件事：模拟盘填 `pos_long_his` / `pos_long_today`，
        实盘 CTP 填 `volume_long_his` / `volume_long_today`。只读一套的话，另一套
        环境里今仓恒为 0 —— 明明全是今天开的仓，也会被判成「昨仓足够」而发 CLOSE，
        交易所直接打回「平昨手数超过持仓仓量」，追单再把这笔注定失败的单重报几次。

        两套加起来还小于总数，说明明细滞后（刚成交那几百毫秒经常这样）。
        **不能猜**：早先猜成昨仓，发出去必然被打回。返回 None 让调用方等下一轮 ——
        行情一秒推十几次，下一轮很快就到。
        """
        p = self._pos.get(symbol) if self._pos is not None else None
        if p is None:
            return None

        def _i(name: str) -> int:
            try:
                return int(getattr(p, name, 0) or 0)
            except (TypeError, ValueError):
                return 0

        his = max(_i(f"pos_{side}_his"), _i(f"volume_{side}_his"))
        today = max(_i(f"pos_{side}_today"), _i(f"volume_{side}_today"))
        total = self._side_pos(symbol, side)
        if his + today < total:
            return None
        # 总仓是硬约束。两套字段（天勤 pos_* / 柜台 volume_*）混着取 max，
        # 或者柜台把今昨分两条推、累加后各自都等于总仓，都会让 his+today > total。
        # 这时**以今仓为准**（TodayPosition 是柜台对「今天开的」直接计数，
        # YdPosition 在分条推送下并不可靠），剩下的才算昨仓。
        # 反过来做（信 YdPosition、把余量算今仓）正是 2026-11-19 那次
        # 「一键对齐平 sn 今仓却发 CLOSE，被 [51] 平昨仓位不足 打回」的成因。
        if today > total:
            today = total
        his = max(0, total - today)
        return his, today

    def _legs(self, symbol: str, delta: int,
              closeable: int | None = None) -> list[tuple[str, int]] | None:
        """把一次调仓拆成「平昨 → 平今 → 开仓」几笔。今昨明细没就绪时返回 None。

        交易所对 CLOSE / CLOSETODAY 的手数**分别**校验：CLOSE 超过昨仓会被整单
        打回。反手时最明显 —— 持 5 手多、目标 -3 手空，delta = -8，整笔当 CLOSE
        发出去就是「平 8 手多」，而实际只有 5 手，于是被拒、重报、再被拒。

        `closeable` 是**这条策略自己**能平的手数。不给就按账户持仓算 —— 一键
        对齐要的就是账户口径。给的话取两者的小值：账户里确实有、而且这部分是
        自己的，才拿去平；剩下的算开仓。

        **只有上期所（SHFE）和能源中心（INE）分平今平昨**，其余交易所（含大商所、
        郑商所、中金所）一律用 CLOSE。判交易所必须看 `symbol` 的前缀
        （`SHFE.rb2601` → `SHFE`）。早先写 `symbol.startswith("CFFEX")` 永远不成立，
        于是给大商所也发了 CLOSETODAY，整单被交易所打回。
        """
        vol = abs(int(delta))
        if vol <= 0:
            return []
        direction = "BUY" if delta > 0 else "SELL"
        side = "long" if direction == "SELL" else "short"
        ex = (symbol.split(".")[0] or "").upper()
        split = ex in SPLIT_OFFSET_EX

        his = today = 0
        have = self._side_pos(symbol, side)
        cap = None if closeable is None else max(0, int(closeable))
        if cap is not None:
            have = min(have, cap)
        if split and have > 0:
            got = self._split_pos(symbol, side)
            if got is None:
                return None              # 明细还没跟上，等下一轮，别瞎猜
            his, today = got
            # 🔴 这里必须**再夹一次** `cap`。`his + today` 是**账户口径**（所有策略
            # + 手工单），拿它覆盖 `have` 会把「自己只有 2 手」抬成「账户有 9 手」，
            # 于是平 5 手 —— 顺手把别的策略的仓平掉了。上期所/能源中心才走这条，
            # 其余交易所没有 `_split_pos`，所以这个洞只在 rb/au/ag/cu/ni/sc… 上出现。
            have = max(have, his + today)
            if cap is not None:
                have = min(have, cap)

        out: list[tuple[str, int]] = []
        if have > 0:
            closing = min(vol, have)
            if split:
                n_his = min(closing, his)
                if n_his > 0:
                    out.append(("CLOSE", n_his))
                if closing > n_his:
                    out.append(("CLOSETODAY", closing - n_his))
            else:
                out.append(("CLOSE", closing))
        rest = vol - min(vol, have)
        if rest > 0:
            out.append(("OPEN", rest))
        return out

    def _price_at(self, symbol: str, direction: str, ticks: int = 0) -> float:
        """按**合约**取对手价，往成交方向让 `ticks` 跳。盘口还没来就返回 NaN。

        不能拿订阅主连时缓存的那个 quote：要下单的是 `underlying`（真实主力），
        和 `KQ.m@...` 是两个 symbol，换月之后更是完全不同的一本盘口。
        `get_quote` 对同一合约是幂等的，重复调不会产生新订阅。
        """
        q = self._quote_obj(symbol)
        if q is None:
            return float("nan")
        tick = float(getattr(q, "price_tick", 0) or 0)
        if direction == "BUY":
            base = float(getattr(q, "ask_price1", 0) or 0) or float(getattr(q, "last_price", 0) or 0)
            return base + tick * ticks
        base = float(getattr(q, "bid_price1", 0) or 0) or float(getattr(q, "last_price", 0) or 0)
        return base - tick * ticks

    @staticmethod
    def _order_id(order) -> str:
        """从 `insert_order()` 的返回值里取订单号。

        tqsdk 3.x 的 `insert_order()` 返回的是 **Order 对象**（dict 子类），不是
        订单号字符串。早先直接把返回值当 id 存进 `PendingOrder`，下一轮
        `self._orders.get(o.order_id)` 就拿一个 dict 去查字典 ——
        `TypeError: unhashable type: 'Order'`，策略在第一个 tick 之后全部停机。
        """
        oid = getattr(order, "order_id", None)
        if oid is None and isinstance(order, dict):
            oid = order.get("order_id")
        return str(oid or "")

    def _send(self, st: SlotState, symbol: str, delta: int,
              closeable: int | None = None) -> int:
        """按差额下发委托，返回实际发出去的笔数。"""
        # 最后一道闸。`tick()` 已经按交易时段整槽跳过了，这里再挡一次是因为
        # `_send` 是唯一的发单出口 —— 以后谁加了别的调用方，也不会绕过门禁。
        if not self._open_now(st.cfg):
            st.last_action = CLOSED_NOTE
            return 0
        # 同一个出口还要按**这一个合约**再判一次。上面那句看的是 `st.cfg`
        # （策略有没有在跑的品种），而「一键对齐」的槽位没有 symbol，永远判成
        # 开市 —— 于是午休 / 收盘点对齐时，`_price_at` 要么拿不到价（静默不发），
        # 要么拿到**收盘前最后一次 tick** 的陈旧价去报单。两种都不对：前者让
        # 用户以为「平今平昨写错了」，后者直接把昨天的价报进今天的交易所。
        # 注意拆分方向：`symbol` 是「交易所.合约」（`SHFE.ni2612`），
        # 前半段才是交易所 —— 反了的话 `in_session` 解析不出时段会一律放行。
        ex, _, code = str(symbol or "").partition(".")
        if code and ex and not symbols_info.in_session(code, ex):
            st.last_action = "非交易时段：这个合约现在不能下单（开盘后再点对齐）"
            return 0
        # 账户侧再问一句能不能报单（柜台接口才有这个方法：登录 + 结算确认）。
        # 🔴 为什么要问而不是「发出去再说」：`insert_order` 抛异常会被 `tick()`
        # 接住 → 「这条策略出错停机」。跨交易日那几秒（柜台正在结算）会把所有
        # 策略一起停掉，用户看到一排「出错停机」，实际什么事都没有。
        gate = getattr(self.api, "ready_to_trade", None)
        if gate is not None:
            try:
                ok, why = gate()
            except Exception:  # noqa: BLE001 - 判据出错就放行，别挡住下单
                ok, why = True, ""
            if not ok:
                st.last_action = why
                return 0
        legs = self._legs(symbol, delta, closeable)
        if legs is None:
            st.last_action = "持仓今昨明细未就绪，等下一轮"
            return 0
        if not legs:
            return 0
        direction = "BUY" if delta > 0 else "SELL"
        price = self._price_at(symbol, direction)
        if not price or price != price:      # NaN / 0 都说明盘口还没来
            st.last_action = "等盘口（这个合约还没收到行情，可能是冷门合约或刚开盘）"
            return 0
        parts = []
        for offset, vol in legs:
            # 🔴 `insert_order` / `place` 现在会**在柜台没受理时报错**
            # （`ReqOrderInsert` 返回非 0 就抛，见 `ctp_client.place`）。以前
            # 谁都不看返回值，「发了个寂寞」和「发出去挂着了」长得一模一样。
            # 这里接住并落成「没发出去」，别让它冒到 `tick()` 那里变成
            # 「这条策略出错停机」—— 一次流控不该停掉一条实盘策略。
            try:
                order = self._order(
                    symbol=symbol, direction=direction, offset=offset,
                    volume=vol, limit_price=price,
                )
            except Exception as exc:  # noqa: BLE001
                st.last_action = f"报单没发出去：{exc}"
                self._log(f"{st.title()}：{symbol} {st.last_action}")
                break
            if order is None:
                st.last_action = "今日报单次数已用完，停手"
                break
            oid = self._order_id(order)
            st.pending.append(
                PendingOrder(order_id=oid, symbol=symbol, direction=direction,
                             offset=offset, volume=vol, price=price, ts=time.time())
            )
            parts.append(f"{offset} {vol}")
        if not parts:
            return 0
        # 理论建仓价**由策略自己写**：这一刻真正报进交易所的价（对手价）。
        # 组合页那两列只认这个数，显示层不再按「下一根开盘 + 成本」去猜 ——
        # 猜出来的价碰到 K 线内（`intrabar`）那一族必错，触发价是盘中算的。
        if st is not self._align_slot:
            self._note_order_entry(st, delta, price)
        st.last_action = (
            f"{'买' if direction == 'BUY' else '卖'} {' + '.join(parts)} 手 @ {price:.2f}"
        )
        self._log(f"{st.title()}：{st.last_action}")
        # 每一笔真发出去的委托都落盘。以前只有界面日志，事后复盘时「到底发的是
        # 平仓还是开仓、几手、什么价」全无据可查（2026-11-20 那次就只剩截图）。
        self._diagw("ORD", f"{st.title()}：{symbol} {st.last_action}")
        return len(parts)

    def _align_timeout(self) -> float:
        """对齐单最多挂多久（秒）。

        不能只写死 `ALIGN_TIMEOUT`：追单是「等 `wait_sec` 没成交就撤掉重报，
        最多 `retries` 次」。默认 5 秒 × 3 次 + 首轮 ≈ 20 秒，30 秒的窗口是够的
        —— 但只要用户把 `wait_sec` 调大（比如 10 秒），对齐就会在追单正做到一半
        时把整批撤掉，用户看到的是「点了好几次，仓位还是没对齐」（2026-11-23）。
        所以按追单参数放大：至少要覆盖「首轮 + 每次重报」各等一遍。
        """
        try:
            c = self.chase
            if not getattr(c, "enabled", False):
                return ALIGN_TIMEOUT
            need = float(c.wait_sec) * (int(c.retries) + 2)
        except Exception:  # noqa: BLE001
            return ALIGN_TIMEOUT
        return max(ALIGN_TIMEOUT, need)

    def _pending_detail(self, st: SlotState) -> list[str]:
        """在途单的「合约 / 开平 / 委x 剩y / 状态」，给撤单前的日志用。

        🔴 「ALIVE」**不许**用来描述我们自己填的那一行（2026-11-23 融航）：
        柜台一条委托回报都没回时，`BrokerApi` 里那行是下单瞬间按「乐观」写的，
        `status="ALIVE"` 是我们填的、不是柜台说的。日志写「剩2 ALIVE」等于告诉
        用户「单子在交易所好好挂着」，而实际可能是根本没报进去 —— 那天四笔
        TL2612 每笔都「ALIVE」30 秒然后被撤，委托框里一笔都没有。
        这类行统一写「无回报」，和「挂着等成交」分开。
        """
        out = []
        now = time.time()
        for o in list(st.pending):
            order = self._orders.get(o.order_id) if self._orders is not None else None
            status = getattr(order, "status", "") if order is not None else "查不到"
            left = getattr(order, "volume_left", "?") if order is not None else "?"
            if getattr(order, "optimistic", False):
                age = now - float(getattr(order, "ts", 0.0) or 0.0)
                status = f"柜台 {age:.0f} 秒没回委托"
            out.append(f"{o.symbol.split('.')[-1]} {o.offset} "
                       f"{o.volume}手 剩{left} {status}")
        return out

    def _ack_note(self, st: SlotState, o: PendingOrder) -> None:
        """报单发出去 `ORD_ACK_SEC` 秒柜台还没回委托回报，落一行日志（每单一次）。

        `optimistic` 只在柜台真回包时才会被 `BrokerApi._fill_orders()` 覆盖成
        False（它按 `OrderRef` 对 key，和下单那一刻写进去的那行**是同一个**），
        所以「一直是 True」确实等于「柜台从没回过这笔」。这是唯一能回答
        「到底发没发出去」的判据 —— 单纯看 `st.pending` 看不出来。
        """
        if not o.order_id:
            return
        order = self._orders.get(o.order_id) if self._orders is not None else None
        if order is None or not getattr(order, "optimistic", False):
            return
        warned = getattr(self, "_ack_warned", None)
        if warned is None:
            warned = self._ack_warned = set()
        if o.order_id in warned:
            return
        age = time.time() - float(getattr(order, "ts", 0.0) or 0.0)
        if age < ORD_ACK_SEC:
            return
        warned.add(o.order_id)
        msg = (f"{o.symbol} {o.offset} {o.volume} 手（OrderRef={o.order_id}）"
               f"已发出 {age:.0f} 秒，柜台一条委托回报都没有 —— 要么报单没进柜台，"
               f"要么回报没回来。去「账户透视和交易」页核一下这个合约有没有这笔委托")
        if st is self._align_slot:
            self._align_log(msg)
        else:
            self._log(f"{st.title()}：{msg}")
            self._diagw("ORD", f"{st.title()}：{msg}")

    def _cancel_all(self, st: SlotState) -> None:
        if self.api is None:
            st.pending.clear()
            return
        for o in st.pending:
            try:
                order = self._orders.get(o.order_id) if self._orders is not None else None
                if order is not None and getattr(order, "status", "") == "ALIVE":
                    self._cancel(order)
            except Exception:  # noqa: BLE001
                pass
        st.pending.clear()

    # ---- 追单 ----

    def _book(self, st: SlotState, o: PendingOrder, filled: int) -> None:
        """把一笔委托的成交记进**本策略自己**的账本。

        平仓和开仓都按方向加减：BUY 一律让净仓 +n，SELL 一律 -n。CLOSE 不是
        「加回去」—— 账本记的是净仓，卖 3 手平多，净仓本来就是 +5 → +2，和
        「开 3 手空」是同一次减法。偏移只影响交易所怎么校验，不影响净仓。

        「一键对齐」的委托也走 `_chase`，但它不属于任何策略，不能记进账本 ——
        记了就会出现一条 id 是 `__align__` 的幽灵策略持仓。
        """
        filled = int(filled or 0)
        if not filled or st is self._align_slot:
            return
        sign = 1 if o.direction == "BUY" else -1
        prev = int(st.held.get(o.symbol, 0))
        cur = prev + sign * filled
        self._book_cost(st, o, prev, sign * filled)
        if cur:
            st.held[o.symbol] = cur
        else:
            st.held.pop(o.symbol, None)
            st.cost.pop(o.symbol, None)
        self._save_book(st)

    def _book_cost(self, st: SlotState, o: PendingOrder, prev: int,
                   delta: int) -> None:
        """按成交更新「持仓均价」，顺手把平掉那部分的盈亏记进当日已实现。

        三种情况：

        - 新开 / 反手（`prev` 为 0，或成交后方向翻过来）：均价 = 这笔成交价。
          反手时**先结算旧仓**（已实现盈亏），再把新方向的成本设成成交价 ——
          拿旧仓的均价给新仓当成本，持仓盈亏会直接从几十个点的错数开始。
        - 同向加仓：按手数加权平均。
        - 减仓：均价不变（卖剩下的还是原来那批）。

        减仓（以及反手时平掉的那部分）按 `(成交价 − 均价) × 手数 × 乘数 × 方向`
        计入 `real_day`。这一项是「当日盈亏」里已经落袋的那一半，`_slot_pnl`
        会把持仓那一半再加上去。
        """
        sym = o.symbol
        price = float(o.price or 0)
        if price <= 0 or not delta:
            return
        old = float(st.cost.get(sym) or 0)
        new = prev + delta
        if prev and (prev > 0) != (delta > 0):
            # 平掉 min(|prev|, |delta|) 手
            qty = min(abs(prev), abs(delta))
            cost = old if old > 0 else price
            mult = self._volume_multiple(st)
            if mult > 0:
                pnl = (price - cost) * qty * mult * (1 if prev > 0 else -1)
                st.real_day = self._realized_today(st) + pnl
        if not new:
            st.cost.pop(sym, None)
        elif not prev or (prev > 0) != (new > 0):
            st.cost[sym] = price
        elif abs(new) > abs(prev):
            if old > 0:
                st.cost[sym] = (old * abs(prev) + price * abs(delta)) / abs(new)
            else:
                st.cost[sym] = price
        # 减仓同向：均价保持不动

    def _chase(self, st: SlotState) -> None:
        """追单状态机：live →（超时）cancelling →（确认死）重报 → live …

        两个坑，都是「看起来能跑、其实没生效」那种：

        - 撤单之后必须等 `status == "DEAD"` 才重报。旧单还有效的窗口里重报，
          一次信号会变成两倍仓位；
        - 判断「这笔结束了」只能看 `FINISHED`（成交）和**自己撤掉之后**的
          `DEAD`。把 DEAD 一律当成结束，重报那条分支永远走不到 —— 追单就退化
          成「只撤不追」，界面上「最多追 3 次」是个摆设。
        """
        if not st.pending or self._orders is None:
            return
        now = time.time()
        keep: list[PendingOrder] = []
        for o in st.pending:
            order = self._orders.get(o.order_id)
            status = getattr(order, "status", "") if order is not None else ""
            left = int(getattr(order, "volume_left", 0) or 0) if order is not None else 0

            # 「柜台到底有没有这笔单」只有 `optimistic` 能回答。必须在状态机
            # 之前问一句 —— 它卡在 ALIVE 上时，下面所有分支都表现得像「正常在途」
            self._ack_note(st, o)

            if status == "FINISHED":
                self._book(st, o, o.volume)
                st.last_action = "已成交"
                st.rejects = 0
                st.reject_msg = ""
                orderbook.forget(self._book_key, o.order_id)
                continue

            if status == "DEAD":
                # 撤单 / 拒单之前可能已经成交了一部分，先把这部分记进账本，
                # 再把 `o.volume` 减掉 —— 不减的话重报会按原手数再发一遍，
                # 成交之后就是「本该 5 手，实际 8 手」。
                filled = max(0, o.volume - left)
                if filled:
                    self._book(st, o, filled)
                    o.volume -= filled
                orderbook.forget(self._book_key, o.order_id)
                if o.phase != "cancelling":
                    # 不是我们撤的：拒单、涨跌停打回、资金不足……
                    # 只记一笔是不够的 —— `_rebalance` 下一轮照样再发，而 tick
                    # 跟着行情推送一秒能跑十几次，一个注定失败的单会把日志刷爆、
                    # 把主线程拖死。所以计数 + 冷却，连续几次直接停机。
                    why = getattr(order, "last_msg", "") or "未知原因"
                    # 平今平昨判错是**可以自己纠正**的：手数、方向、价格都没问题，
                    # 只是告诉交易所「平哪一边」错了。这里换一边重报一次，
                    # 不占「连续被拒」的次数 —— 否则一次判错就直接停机
                    # （2026-11-19 一键对齐平 sn 今仓，发成 CLOSE 被打回）。
                    if self._flip_offset(st, o, why):
                        keep.append(o)
                        continue
                    st.rejects += 1
                    st.reject_ts = now
                    st.reject_msg = why
                    st.last_action = f"委托被拒（第 {st.rejects} 次）：{why}"
                    self._log(f"{st.title()}：{st.last_action}")
                    if st.rejects >= MAX_REJECTS:
                        st.error = f"连续 {st.rejects} 次被拒，已停机：{why}"
                        st.running = False
                        self._log(f"{st.title()}：{st.error}")
                    continue
                if o.volume <= 0:
                    continue             # 已经全部成交，只是状态翻成了 DEAD
                if o.retries >= self.chase.retries:
                    st.last_action = f"追单 {o.retries} 次仍未成交，放弃"
                    self._log(f"{st.title()}：{st.last_action}")
                    orderbook.forget(self._book_key, o.order_id)
                    continue
                o.retries += 1
                o.phase = "live"
                o.ts = now
                price = self._price_at(o.symbol, o.direction, self.chase.ticks * o.retries)
                if self.chase.use_market and o.retries >= self.chase.retries:
                    price = float("nan")     # 天勤用 NaN 表示市价
                try:
                    again = self._order(
                        symbol=o.symbol, direction=o.direction, offset=o.offset,
                        volume=o.volume, limit_price=price,
                    )
                    if again is None:
                        st.last_action = "今日报单次数已用完，停手"
                        continue
                    o.order_id = self._order_id(again)
                    o.price = price
                    # 末次追单可能是市价（`price` 是 NaN），别把「@ nan」写进日志
                    px_txt = "市价" if price != price else f"{price:.2f}"
                    st.last_action = f"追单第 {o.retries} 次 @ {px_txt}"
                    self._log(f"{st.title()}：{st.last_action}")
                except Exception as exc:  # noqa: BLE001
                    st.last_action = f"追单失败：{exc}"
                    continue
                keep.append(o)
                continue

            # 已经撤了、还没确认死：什么都别做，等下一轮
            if o.phase == "cancelling":
                keep.append(o)
                continue

            # 成交吃满了但状态还没翻成 FINISHED
            if order is not None and left == 0:
                self._book(st, o, o.volume)
                continue

            # 还活着。关掉追单、或者本来就不追，就一直挂着等它自己成交 ——
            # 这里不能把单子从 pending 里丢掉：丢了 `_rebalance` 会以为没在途，
            # 立刻再发一笔，仓位直接翻倍
            if not self.chase.enabled or self.chase.retries <= 0:
                keep.append(o)
                continue
            if now - o.ts < self.chase.wait_sec:
                keep.append(o)
                continue
            if order is not None:
                self._cancel(order)
            o.phase = "cancelling"
            keep.append(o)
        st.pending = keep

    def _flip_offset(self, st: SlotState, o: PendingOrder, why: str) -> bool:
        """「平昨仓位不足」这类拒单：把 CLOSE / CLOSETODAY 换一边重报一次。

        上期所（SHFE）和能源中心（INE）把平今、平昨当成**两个指令**分别校验：
        报 CLOSE 而昨仓是 0，整单被 `[51] 平昨仓位不足` 打回；报 CLOSETODAY
        而今天没开过仓，同样打回。手数、方向、价格都是对的，错的只是
        「平哪一边」这一位 —— 换一边重报，成交了平掉的还是同一笔仓。

        只在拒单文案确实指向仓位不足时动手（资金不足、涨跌停之类换了没用），
        且 `o.flipped` 保证只翻一次：两边都说没仓就是真没仓，交给计数 + 停机。

        这是**兜底**，不是正路。正路是 `_split_pos` 拿到对的今昨明细，
        一开始就发对指令（2026-11-19 的根因在 `broker_api._fill_positions`
        把 YdPosition 当昨仓，与持仓表口径不一致）。
        """
        if o.flipped:
            return False
        if o.offset not in ("CLOSE", "CLOSETODAY"):
            return False
        ex = str(o.symbol or "").split(".")[0].upper()
        if ex not in SPLIT_OFFSET_EX:
            return False
        text = str(why or "")
        if not any(k in text for k in ("仓位不足", "持仓不足", "可平", "平昨", "平今")):
            return False
        price = self._price_at(o.symbol, o.direction)
        if not price or price != price:      # NaN：盘口还没来，等下一轮
            return False
        want = "CLOSETODAY" if o.offset == "CLOSE" else "CLOSE"
        try:
            again = self._order(
                symbol=o.symbol, direction=o.direction, offset=want,
                volume=o.volume, limit_price=price,
            )
        except Exception as exc:  # noqa: BLE001
            st.last_action = f"换平今平昨重报失败：{exc}"
            return False
        if again is None:
            return False                     # 额度用完，交给上层按「报单到顶」处理
        o.offset = want
        o.flipped = True
        o.order_id = self._order_id(again)
        o.price = price
        o.phase = "live"
        o.ts = time.time()
        st.last_action = (
            f"平今平昨对不上（{text}），改 {want} 重报 {o.volume} 手 @ {price:.2f}")
        self._log(f"{st.title()}：{st.last_action}")
        return True

    # ---- 监控器 ----

    def theoretical(self) -> dict[str, int]:
        """按合约汇总策略的理论净仓。同一个品种被多个策略持有时取和。"""
        out: dict[str, int] = {}
        for st in self._slots.values():
            if not st.running:
                continue
            # 换月留下的老合约目标是 0：它仍然归这条策略管，所以必须算进
            # `managed`，否则一键对齐会绕开它，仓位就永远挂在那儿。
            for sym in st.retired:
                out.setdefault(sym, 0)
            if st.underlying:
                out[st.underlying] = out.get(st.underlying, 0) + st.target
        return out

    def _pos_live(self) -> bool:
        """账户持仓到底**读到手里**没有 —— 「账户真没仓」≠「柜台一条都没回」。

        🔴 为什么要单独一个判据（2026-11-23 融航）：用户账户里明明有 6 个合约的
        持仓，组合页的监控器却写「账户仓位与策略理论持仓一致」。比对没算错，是
        这条连接**一条持仓回调都没收到** —— `_pos` 是个空字典，而 `mismatch()`
        的合约集合是「理论持仓 ∪ `_pos` 里有的合约」，两边都空，集合就是空的，
        于是「一个不一致都没有」。

        空字典本来有两种含义，早先被混成了一种：

          - 账户真没仓     → 可以报「一致」
          - 柜台没回 / 被拒 → 只能说「还没读到」；`align()` 更不能拿它去算差额
                            （会以为账户是空的，把整个理论仓位重新开一遍）

        判据放在账户接口那一侧（`BrokerApi.positions_ready()`），因为交易页和
        组合页是**同一个账户的两个会话**，一条读到了不代表另一条读到了。
        """
        if self._pos is None:
            return False
        ready = getattr(self.api, "positions_ready", None)
        if ready is None:
            # 天勤：`get_position()` 是活引用，登上去了就在，没有「还没到」这个状态
            return True
        try:
            return bool(ready())
        except Exception:  # noqa: BLE001
            return True                  # 判不了就别拦着用户

    def _pos_note(self) -> None:
        """持仓一直读不回来时往 `live.log` 写一行 `[POS]`，把柜台那边的情况带上。

        界面这时候只会写「还没同步」，用户没法判断是「再等几秒」还是要去找中台
        开通查询权限。日志里必须有：发过多少拍查询、`positions_loaded`、柜台最后
        一句错话、`_pos` 里几个合约 —— 否则下次还是只能靠猜（见 `POS_LOG_SEC`）。
        """
        now = time.time()
        # 没接账户（收盘后登出、只算信号）是**正常**状态，不是「读不到」——
        # 不区分的话，登出 30 秒后就会冒出一行「持仓没从柜台读到」，纯噪音
        if self.api is None or self._pos_live():
            self._pos_blind_at = 0.0
            self._pos_warned = 0.0
            return
        # 这两个字段同样要 `getattr` 兜：自检里常用 `__new__` 造精简桩，
        # 而 `snapshot()`（进而 `mismatch()`）是 0.4 秒一次的实盘主路径
        since = getattr(self, "_pos_blind_at", 0.0)
        if not since:
            self._pos_blind_at = since = now
        warned = getattr(self, "_pos_warned", 0.0)
        # 第一次等够 `POS_LOG_SEC` 才说话，之后每 4 倍时长再提一次（别刷屏）
        if now - since < POS_LOG_SEC or (warned and now - warned < POS_LOG_SEC * 4):
            return
        self._pos_warned = now
        hub = getattr(self.api, "hub", None)
        trader = getattr(hub, "trader", None)
        err = (getattr(self.api, "last_error", "")
               or getattr(hub, "last_error", "") or "（柜台没有报错）")
        # `_book_key` 是个 property（要读 `self.spec`），精简桩上可能取不到；
        # 诊断代码绝不能把 snapshot() 这条主路径带崩
        try:
            who = self._book_key
        except Exception:  # noqa: BLE001
            who = ""
        # 资金那边的占用保证金是**独立证据**：仓位一条都没读到、保证金却大于 0，
        # 那就不是「账户没仓」，而是「柜台没把持仓给我们」—— 这两种情况在
        # `_pos` 上长得一模一样，只有对一下资金才分得开
        acct = getattr(trader, "account", None) or {}
        try:
            margin = float(acct.get("margin") or 0.0)
        except Exception:  # noqa: BLE001
            margin = 0.0
        hint = ("（占用保证金 %.0f 却一条持仓都没读到 —— 这不是「没仓」，"
                "是柜台没把持仓给这条连接）" % margin) if margin > 0 else ""
        # 有一轮收到了半截 / 有几轮有开头没结尾（柜台丢包）。这两个数只在这里
        # 才有意义：`positions` 现在**只存收全的整轮**，所以「少一个合约」不可能
        # 再是半份表造成的，只可能是柜台这轮真没给（或查询被丢了）
        partial = "、正在收一轮（还没收全）" if getattr(trader, "_pos_stage", None) is not None else ""
        dropped = int(getattr(trader, "_pos_dropped", 0) or 0)
        self._diagw("POS", (
            f"{who} 的持仓已经 {now - since:.0f} 秒没从柜台读到："
            f"发过 {int(getattr(hub, '_pos_try', 0) or 0)} 拍持仓查询、"
            f"positions_loaded={bool(getattr(trader, 'positions_loaded', False))}、"
            f"最近一轮回包 {int(getattr(trader, '_pos_rows', -1) or 0)} 行、"
            f"_pos 里 {len(self._pos or {})} 个合约、"
            f"丢包 {dropped} 轮{partial}"
            f" ｜ 柜台最后一句：{err}{hint}"
            " ｜ 只要还在这个状态，界面上所有「一致」都不作数"
            "（不是比对算错，是一条持仓都没读到）"))

    def mismatch(self, fresh: bool = False) -> list[dict]:
        """理论持仓与实际持仓不一致的合约。

        `managed` 表示这个合约上有**正在运行**的策略。为 False 的多半是「策略停了
        但仓位还留着」，或者是用户手工开的仓。界面靠这个标记把两类分开显示 ——
        前者是策略自己没跟上，后者得先让人看清楚再动。

        **比对的是两条腿，不是净仓。** 策略理论仓位是个净数（`target`，正多负空），
        落成目标只能是「只多」或「只空」：`target>0 → (target, 0)`。账户那边
        多空可以同时非零（锁仓），净仓会把它压成 0 —— 早先只看净仓，于是
        「多 4 空 4」被判成「没有仓」，既不上报也平不掉。现在按
        `(多, 空) == (目标多, 目标空)` 判，锁仓一定会露出来。

        结果缓存 `MISMATCH_GAP` 秒：快照 0.4 秒推一次，每次都重算纯属空转。
        `fresh=True` 绕过缓存 —— 真正要下单（`align()`）时必须用当下的持仓，
        拿 5 秒前的旧数去补仓，补出来的量就是错的。

        🔴 **返回空表不等于「一致」**。这个函数只回答「按我知道的持仓，哪几个
        对不上」；「我知道的持仓」是否可信，由 `_pos_live()` 单独回答：

          - `_pos` 还没拿到（`None`）        → 空表是「不知道」
          - 柜台一条持仓回调都没回来（空字典）→ 空表还是「不知道」

        两种情况下 `keys` 都会是空的（理论持仓空 + `_pos` 空），一行都不产出。
        所以界面必须按 `pos_synced` 分开说，`align()` 也必须自己再拦一道 ——
        2026-11-23 融航就是拿这张空表报了「账户仓位与策略理论持仓一致」，
        而账户里实际有 6 个合约的仓。
        """
        now = time.time()
        if not fresh and now - self._mm_ts < MISMATCH_GAP:
            # 给副本。界面拿到的 list 会被到处传，别让它改到缓存里的 dict。
            return [dict(r) for r in self._mm_cache]
        theo = self.theoretical()
        keys = set(theo)
        if self._pos is not None:
            for sym in list(self._pos):
                if any(self._side_pair(sym)):
                    keys.add(sym)
        out = []
        for sym in sorted(keys):
            want = int(theo.get(sym, 0))
            have = self._net_pos(sym)
            a_long, a_short = self._side_pair(sym)
            # 目标落成两条腿。want 是净数，只能占一边
            t_long, t_short = (want, 0) if want > 0 else (0, -want)
            if (a_long, a_short) == (t_long, t_short):
                continue
            out.append({
                "symbol": sym, "target": want, "actual": have,
                "delta": want - have, "managed": sym in theo,
                "t_long": t_long, "t_short": t_short,
                "a_long": a_long, "a_short": a_short,
                # 多空双开。界面据此把「账户实际」写成 `0（多4 空4）`，
                # 否则一行 `0 / 0 / +0` 摆在那里，谁都不知道要动什么
                "locked": bool(a_long and a_short),
            })
        self._mm_cache = out
        self._mm_ts = now
        self._mm_note(out)
        return [dict(r) for r in out]

    def _mm_note(self, rows: list[dict]) -> None:
        """不一致**一直**存在时落一行 `[MM]`，把账户接口的原始读数记下来。

        🔴 为什么要（2026-11-23）：监控器一直写着「`T2612` 账户 -2、平空2」，
        而账户其实早就平了、交易页的持仓表里连这个合约都没有。日志里只有
        `[ALIGN]` 行，没有任何「此刻 `_pos` 里是什么」的记录 —— 到底是活引用
        断了（读到一份冻住的旧数），还是今昨仓被叠了一遍，只能猜。

        只在**连续** `MM_LOG_SEC` 秒都一样的时候写一次，合约恢复一致就把记录
        清掉：界面每 2 秒算一次，不设门槛会把 `live.log` 刷满。
        """
        now = time.time()
        # 这三个容器用 `getattr` 兜一下：单元测试里常用
        # `PortfolioRunner.__new__` 造一个只有几个字段的桩，直接 `self._mm_since`
        # 会抛 AttributeError —— 而 `mismatch()` 在实盘是 0.4 秒一次的路径，
        # 诊断代码把主流程带崩是最不能接受的一种崩法
        since_map = getattr(self, "_mm_since", None)
        if since_map is None:
            since_map = self._mm_since = {}
        logged = getattr(self, "_mm_logged", None)
        if logged is None:
            logged = self._mm_logged = set()
        live = set()
        for r in rows:
            sym = str(r.get("symbol") or "")
            if not sym:
                continue
            live.add(sym)
            since = since_map.setdefault(sym, now)
            if sym in logged or now - since < MM_LOG_SEC:
                continue
            logged.add(sym)
            self._diagw("MM", (
                f"{sym} 已经不一致 {now - since:.0f} 秒：目标 {int(r['target']):+d}"
                f"（多{int(r.get('t_long') or 0)} 空{int(r.get('t_short') or 0)}）"
                f" 账户 净{int(r['actual']):+d}"
                f"（多{int(r.get('a_long') or 0)} 空{int(r.get('a_short') or 0)}）"
                f" ｜ 账户接口原始读数：{self._pos_raw(sym)}"
                " ｜ 一直不动就是这个合约的持仓没读对（不是「对齐没生效」）"))
        for sym in list(since_map):
            if sym not in live:
                since_map.pop(sym, None)
                logged.discard(sym)

    def align(self, close_orphans: bool = True,
              rows: list[dict] | None = None) -> int:
        """把账户仓位调成和策略列表完全一致。返回下发的委托数。

        - 托管合约（有策略在跑）：补到理论净仓，多了减、少了加
        - 未托管合约（没有策略在管）：`close_orphans=True` 时平到 0
        - **锁仓（多空双开）**：两条腿各自减到目标，见 `_send_row`
        - `rows`：**确认框里给用户看过的那几行**。传了就只在这几个合约上动作，
          而且账户持仓必须还和框里写的一样。

        第二类是「策略停掉之后留下的仓」和「手工开的仓」。默认平掉是因为
        用户对这个按钮的预期就是「一键把账户摆成策略列表的样子」；但这是
        真金白银，调用方必须先确认过再调，别在启动流程里顺手带上。

        🔴 **为什么要有 `rows`**（2026-11-20 那次）：确认框是用界面缓存的那份
        `mismatch` 拼的，而这里又自己算了一遍 `mismatch(fresh=True)`。两份数据
        一旦不同步，用户确认的是「2 个合约 · 平空4 / 平多2」，实际发出去的却是
        另一批，还多了一个框里没出现过的合约。所以：

          - **合约集合**以确认框为准，框里没有的一笔都不发；
          - **持仓基数**也必须对得上，`多/空` 两条腿和框里不一样就跳过这个
            合约并写明原因 —— 换个基数去算，会把「平空4」算成「开空2」；
          - 量仍然按当下持仓差多少平多少（`_send_row` 只许平、不许反手开）。

        **一次点击只发一轮**。单子记进 `_align_slot.pending`，之后由 `_chase`
        负责追价 / 撤单 / 重报，追完为止；期间 `_rebalance` 全体让路。上一轮
        还没落地时再点直接忽略 —— 否则「仓位还没变 → 判定没对齐 → 再发一遍」
        会变成一个每秒十几次的死循环。
        """
        if self.api is None:
            self._align_log("对齐没有发出委托：账户没有接进来（只算信号不下单）")
            return 0
        if not self._pos_live():
            # 持仓还没同步就发单，等于拿着「账户是空的」去算差额 —— 会把
            # 整个理论仓位重新开一遍。宁可让用户再点一次。
            # 🔴 判据用 `_pos_live()` 而不是 `self._pos is None`：柜台一条持仓
            # 回调都没回来时，`_pos` 是个**空字典**（不是 None），拿它算差额
            # 和「账户真的空仓」得到的结果一模一样（2026-11-23 融航）
            self._align_log(
                "账户持仓还没从柜台读回来，这会儿对齐会把仓位重开一遍，等几秒再点")
            return 0
        if self._align_slot.pending:
            self._align_log("上一轮对齐还没落地，等它成交或撤掉再点")
            return 0
        now = time.time()
        if now - self._align_ts < ALIGN_GAP:
            self._align_log(f"对齐刚点过，{ALIGN_GAP:.0f} 秒内不重复下发")
            return 0
        self._align_ts = now
        self._align_slot.rejects = 0
        self._align_slot.reject_msg = ""
        self._align_slot.last_action = ""
        wanted = None
        if rows is not None:
            wanted = {str(r.get("symbol")): (int(r.get("a_long") or 0),
                                            int(r.get("a_short") or 0))
                      for r in rows}
        n = 0
        skipped, stale = [], []
        self._align_touched = set()
        for row in self.mismatch(fresh=True):
            sym = row["symbol"]
            pair = (int(row.get("a_long") or 0), int(row.get("a_short") or 0))
            if wanted is not None:
                if sym not in wanted:
                    # 确认框里没有它：多半是刚冒出来的未托管仓。绝不能顺手平掉
                    skipped.append(sym)
                    continue
                if pair != wanted[sym]:
                    stale.append(f"{sym}（确认时 {wanted[sym][0]}多/{wanted[sym][1]}空，"
                                 f"现在 {pair[0]}多/{pair[1]}空）")
                    continue
            if not row["managed"] and not close_orphans:
                continue
            self._align_log(
                f"对齐 {sym}：目标 {int(row['target']):+d} 手"
                f"（多{int(row.get('t_long') or 0)} 空{int(row.get('t_short') or 0)}）"
                f" 账户 净{int(row['actual']):+d}"
                f"（多{pair[0]} 空{pair[1]}） 差额 {int(row['delta']):+d}"
                f"{self._pos_detail(sym)}")
            sent = self._send_row(self._align_slot, row)
            if sent:
                self._align_touched.add(sym)
                # 对齐的目标就是 `sum(target)`，所以发完之后各策略的账本正好
                # 是「自己的 target」。不跟着摆的话：停掉的策略留着 -11 的账本，
                # 对齐把仓平了，再启动它目标 -11、账本 -11、差额 0 —— 它会以为
                # 自己还空着 11 手，实际上账户是平的。
                #
                # 🔴 这是**乐观**写法：发出去 ≠ 成交。超时撤单那一头必须把它
                # 摆回来（见 `_resync_books`），否则账本说「到位了」、账户却是
                # 原样，之后策略按账本发「平仓手数超过持仓量」（2026-11-23 那张
                # 截图里 T2612 / IC2612 的拒单就是这么来的）。
                self._sync_after_align(sym)
            else:
                # 这笔一笔都没发出去。原因（等盘口 / 明细未就绪 / 非交易时段 /
                # 可平量不够）写在 `last_action` 里，**必须逐个落日志** ——
                # 以前只在「整轮一笔都没发」时才 echo 一次，于是「监控器上 10 个
                # 合约、委托框里只有 5 笔」在日志里完全看不出另外几个为什么不动，
                # 用户只能反复点（2026-11-23 报的「发出的委托单很少」）。
                self._align_log(
                    f"对齐 {sym} 没发出去："
                    f"{self._align_slot.last_action or '没有需要动的腿'}")
            n += sent
        if skipped:
            self._align_log("对齐跳过（确认框里没有这几个合约）："
                            + "、".join(skipped[:8]) + ("…" if len(skipped) > 8 else ""))
        if stale:
            self._align_log("对齐跳过（持仓在点确认之后变了，换个基数会把平仓算成开仓）："
                            + "、".join(stale[:6]) + ("…" if len(stale) > 6 else "")
                            + "　—— 看清楚了再点一次")
        if n:
            self._align_log(f"对齐已下发 {n} 笔，剩下的交给追单")
        else:
            # 一笔都没发时**必须说话**。以前这里静默返回 0，用户点了按钮看不到
            # 任何反应，只能猜「是不是平今平昨写错了」—— 实际多半是收盘/午休、
            # 冷门合约没有盘口，或者本轮已经有单在途。
            self._align_log(
                f"对齐没有发出委托：{self._align_slot.last_action or '没有需要调整的仓位'}")
        return n

    def _align_log(self, msg: str) -> None:
        """对齐的每一步都要留痕 —— 只进界面日志等于没有。

        界面日志在内存里，关窗就没了；而「对齐到底发了什么、为什么没发」恰恰是
        事后才问的（2026-11-20 那次就只剩截图）。所以除界面外再写一行 `[ALIGN]`
        到 `live.log`。
        """
        self._log(msg)
        self._diagw("ALIGN", msg)

    def _send_row(self, st: SlotState, row: dict) -> int:
        """按「多边 / 空边」各发一笔，返回发出的笔数。

        净额口径表达不了锁仓：多 4 空 4、目标 0，`delta` 算出来是 0，
        `_send(st, sym, 0)` 一笔都不发 —— 这正是 2026-11-11 那三笔平不掉的原因。

        两条边都要减的时候拆成两笔：SELL 平多、BUY 平空。它们各自只动自己那一
        侧（`_send` 按方向反推可平量），互不占用额度，所以同时发是安全的；
        不像「先平后开」，那样第二笔会把第一笔还没成交的量重复算进可平量，
        发出去就是超量平仓、被交易所打回。
        """
        sym = row["symbol"]
        a_long = int(row.get("a_long") or 0)
        a_short = int(row.get("a_short") or 0)
        t_long = int(row.get("t_long") or 0)
        t_short = int(row.get("t_short") or 0)
        close_long = max(0, a_long - t_long)      # 要 SELL 平掉的多头
        close_short = max(0, a_short - t_short)   # 要 BUY 平掉的空头

        # 🔴 「本意是平」就必须在账户上查到那么多可平量，查不到**不许开反向仓**。
        #
        # 这是 2026-11-20 那次「一键对齐点确定后发出去的是反方向的开仓」的直接
        # 闸门。`_legs()` 在 `have == 0` 时会把整笔当 OPEN 发出去：想平 2 手多，
        # 账户那一刻读不到多头（刚成交、刚对齐过一轮），就变成「开 2 手空」——
        # 方向反了、仓位还多了一倍，而且界面上的委托看起来完全正常。
        # `_rebalance()` 早就有这道闸（见它里面 `delta * held < 0` 那段），
        # 一键对齐这条路径漏了。
        for side, want, cn in (("long", close_long, "多"),
                               ("short", close_short, "空")):
            if want and self._side_pos(sym, side) < want:
                st.last_action = (
                    f"{sym} 本意平{cn}{want} 手，账户只查到 "
                    f"{self._side_pos(sym, side)} 手（持仓快照还没跟上）"
                    "—— 不发反向开仓，等下一轮")
                self._align_log(f"{st.title()}：{st.last_action}")
                return 0

        if close_long and close_short:
            n = self._send(st, sym, -close_long, closeable=close_long)   # SELL 平多
            n += self._send(st, sym, close_short, closeable=close_short)  # BUY 平空
            if n:
                st.last_action = f"平锁仓：卖 {close_long} + 买 {close_short} 手"
            return n
        # `closeable` 一定要给：不给的话 `_legs` 按账户口径算可平量，而账户里
        # 还有别的策略的仓 —— 一笔「平自己 2 手」会变成「平账户 9 手」。
        delta = int(row["delta"])
        closed = min(close_long + close_short, abs(delta))
        return self._send(st, sym, delta, closeable=closed)

    def _sync_after_align(self, sym: str) -> None:
        """对齐发出去之后，把各策略的账本摆成「对齐完成时的样子」。

        运行中且就是这个合约的策略，账本取它自己的 `target`；其余情况（策略停着、
        或者这是换月留下的老合约）归 0 —— 对齐会把这些当未托管平掉。
        """
        for st in self._slots.values():
            if sym not in st.held:
                continue
            if st.running and sym == st.underlying and st.target:
                st.held[sym] = int(st.target)
            else:
                st.held.pop(sym, None)
            if sym in st.retired and sym not in st.held:
                st.retired.remove(sym)
            # 对齐之后「当前目标」就是新基准：否则加载时记的旧基准还在，
            # 目标没变 ⇒ 后面每一轮都被「目标不变不发单」挡住，策略再也不动
            st.based = True
            st.based_target = int(st.target or 0)
            self._save_book(st)

    def _resync_books(self) -> list[str]:
        """对齐**没落地**时，把账本按账户实际重新摆一遍。返回改动过的清单。

        🔴 为什么必须有这一步（2026-11-23）：`_sync_after_align()` 是**乐观**的
        —— 委托一发出就把 `held` 写成 `target`。可对齐单是限价单，盘中价格一跑
        就挂在那儿，`ALIGN_TIMEOUT` 一到整批撤掉，账户原地没动，而账本已经说
        「我到位了」。之后策略一翻信号，就按这个假账本发「平仓 2 手」——
        账户里那 2 手根本不存在，柜台当场打回「平仓手数超过持仓量」（那张截图
        里 T2612 / IC2612 的说明就是它），而且被拒一次就少一次报单额度。

        撤单之后账本 = 账户实际，谁都没到位，重新从真实起点算，这才是对的。
        只改 `held`，**不动 `based` / `based_target`**：对齐没成功就不该把
        当前目标设成新基准，否则「目标不变不发单」会把策略永久冻住。
        """
        changed: list[str] = []
        for st in list(self._slots.values()):
            for sym in list(st.held):
                have = int(self._net_pos(sym))
                old = int(st.held.get(sym) or 0)
                if old == have:
                    continue
                if have:
                    st.held[sym] = have
                else:
                    st.held.pop(sym, None)
                changed.append(f"{st.cfg.sid} {sym} {old:+d}→{have:+d}")
                self._save_book(st)
        return changed

    # ---- 给界面看的实时 K 线 ----

    def bars_of(self, sid: str) -> pd.DataFrame | None:
        """某个策略当前订阅着的 K 线，含正在走的那根。没有就返回 None。

        组合页原来只画本地库里的历史 —— 打开一看停在几天前，用户会以为策略
        没在跟行情。引擎这边本来就是实时的，缺的只是一个把它交出去的通道。
        """
        st = self._slots.get(sid)
        if st is None:
            return None
        kl = self._klines.get(self._skey(st.cfg))
        if kl is None or not len(kl):
            return None
        try:
            df = bars_from_serial(kl, st.cfg.period)
            n = tqdata.PERIODS.get(st.cfg.period, (st.cfg.period, 1))[1]
            return tqdata.resample(df, n)
        except Exception:  # noqa: BLE001
            return None

    def want_bars(self, sids: list[str]) -> None:
        """下一份快照里带上这些策略的实时 K 线。

        不做成「每份快照都带」：3000 行转 DataFrame 要几毫秒，0.4 秒一次、
        账户里还有好几个策略，等于白烧。界面看 K 线是 5 秒一次的事。
        """
        self._want = list(sids or [])

    # ---- 快照 ----

    def _group_slots(self, pnl: dict | None = None) -> list[dict]:
        """多品种策略在列表里那一行：把 N 个子槽位汇成一条。

        没有这一行，界面上就找不到这条策略（子槽位的 sid 是 `父#品种`）。

        目标仓这里写 `None`（界面显示「—」），**这是用户 11-17 15:43 明确要求的
        口径**：19 个品种各有一个目标，加起来没有金融含义（一手对应的价值差几十
        倍），逐合约的目标在右下监控器里列。这条策略「有没有仓」用 `action` 回答
        —— 写成「19 个品种：5 个有仓」，比一个 0 或一个「—」清楚。
        别再往这里塞 `tgt_long`/`tgt_short` 之类的合计（那套已经被回滚过）。

        **盈亏是例外，可以合计**：金额是可加的（每个品种已经按自己的手数 ×
        乘数折成元），加起来就是「这条多品种策略今天一共赚了多少」。价格不行
        —— 19 个品种没有一个共同的价，写 `None`，让界面显示「—」。
        """
        pnl = pnl or {}
        out = []
        for pid, kids in self._groups.items():
            rows = [self._slots[k] for k in kids if k in self._slots]
            if not rows:
                continue
            cfg = self._parents.get(pid) or rows[0].cfg
            n_pos = sum(1 for r in rows if r.target)
            n_err = sum(1 for r in rows if r.error)
            running = any(r.running for r in rows)
            err = next((r.error for r in rows if r.error), "")
            action = f"{len(rows)} 个品种：{n_pos} 个有仓"
            if n_err:
                action += f"，{n_err} 个出错"
            # 汇总子槽位的盈亏。**只要有一个算不出来就整体不写**：拿部分品种
            # 的和当合计，用户会以为「只有这几十块」，比显示「—」更容易误判
            day_sum = pos_sum = 0.0
            bad = []
            for r in rows:
                _px, d, p, note = pnl.get(r.cfg.sid, (0.0, 0.0, 0.0, ""))
                if d != d or p != p:
                    bad.append(r.cfg.symbol)
                    continue
                day_sum += float(d)
                pos_sum += float(p)
                if note:
                    bad.append(r.cfg.symbol)
            note = (f"{len(bad)} 个品种算不全（缺价 / 缺乘数）："
                    + "、".join(bad[:6]) + ("…" if len(bad) > 6 else "")) if bad else ""
            out.append({
                "sid": pid,
                "title": cfg.title(),
                "key": "",
                "period": cfg.period,
                "strategy": cfg.strategy_label(),
                "volume": cfg.volume,
                "sizing": cfg.sizing,
                "size_text": cfg.size_text(),
                "lots": 0,
                "running": running,
                "signal": 0,
                "target": None,          # 没有「一个」目标仓
                "underlying": "",
                # 多品种没有「一个价」；盈亏是合计（见上一段）
                "price": None,
                "day_pnl": float(day_sum),
                "pos_pnl": float(pos_sum),
                "pnl_note": note,
                # 一个都没跑起来才把错误顶到这一行；部分失败写进 action，
                # 免得 18/19 个品种正常、界面上却是红字「错误」
                "error": err if (n_err and not running) else "",
                "action": action,
                "retired": [],
                "group": "",
                "multi": len(rows),
                "ml_wait": any((not r.ml_ready and r.running) for r in rows),
                "ml_age": max((float(r.ml_age or 0) for r in rows), default=0.0),
                # 多品种时取「最严重」的那个原因：只要有一个品种永远算不出来，
                # 整条策略就等下去 —— 得让用户看到那个真实原因。
                "ml_why": next((r.ml_why for r in rows
                                if r.ml_why and r.ml_why != "retrain"), "") or (
                    "retrain" if any((not r.ml_ready and r.running
                                      and r.ml_why == "retrain") for r in rows) else ""),
            })
        return out

    def _log_errors(self) -> None:
        """把槽位错误**落一次盘**（GUI 那份日志只在内存里，关窗就没了）。

        表格「错误」那一格只有 tooltip 一句话，事后想查是启动校验拦下的、还是
        tick 里抛的，盘上没有任何记录 —— 只能说「不知道」。同一条错误只记一次，
        否则每 0.4 秒一份快照会刷满 `live.log`。
        """
        for st in self._slots.values():
            if not st.error:
                continue
            key = (st.cfg.sid, st.error)
            if key in self._err_logged:
                continue
            self._err_logged.add(key)
            self._diagw("LIVE", f"{st.title()}：{st.error}")

    def snapshot(self) -> dict:
        self._log_errors()
        self._pos_note()
        # 取一次就清掉。没清的话界面关掉 K 线之后，这份开销还每份快照都付
        want, self._want = self._want, []
        # 盈亏每个槽位只算一次：多品种的父行要按子槽位汇总，直接读这里的结果，
        # 否则 27 个品种就要取 54 遍盘口
        pnl = {sid: self._slot_pnl(st) for sid, st in self._slots.items()}
        slots = []
        for st in self._slots.values():
            px, day, pos, note = pnl[st.cfg.sid]
            slots.append({
                "sid": st.cfg.sid,
                "title": st.title(),
                "key": st.cfg.key(),
                "period": st.cfg.period,
                "strategy": st.cfg.strategy_label(),
                "volume": st.cfg.volume,
                "sizing": st.cfg.sizing,
                "size_text": st.cfg.size_text(),
                "lots": st.lots,
                "running": st.running,
                "signal": st.signal,
                "target": st.target,
                # 「理论持仓」——界面「理论」列和两个盈亏列按它算。
                # 大多数策略 `theo == target`；K 线内那一族不一样：`target` 是
                # 下单目标（没触发就是 0），`theo` 才是策略口径该持多少
                "theo": self._theo_qty(st),
                "underlying": st.underlying,
                # 品种价格 / 当日盈亏 / 持仓盈亏。NaN 一律转 None —— 界面拿
                # 「不知道」显示成「—」，不能显示成 0（0 会被读成「不赚不亏」）
                "price": px if px == px and px > 0 else None,
                "day_pnl": None if day != day else float(day),
                "pos_pnl": None if pos != pos else float(pos),
                "pnl_note": note,
                "error": st.error,
                "action": st.last_action,
                "retired": list(st.retired),
                "group": self._child_of.get(st.cfg.sid, ""),
                "ml_wait": bool(not st.ml_ready and st.running),
                "ml_age": float(st.ml_age or 0.0),
                "ml_why": str(st.ml_why or ""),
                # K 线内那一族：界面要能看到「这一根的触发价 / 止损挂在哪」，
                # 否则用户只知道「有仓」，不知道离被打掉还有多远
                "intrabar": bool(self._ib_on(st)),
                "ib_entry_long": (None if st.ib_entry_long != st.ib_entry_long
                                  else float(st.ib_entry_long)),
                "ib_entry_short": (None if st.ib_entry_short != st.ib_entry_short
                                   else float(st.ib_entry_short)),
                "ib_stop": None if st.ib_stop != st.ib_stop else float(st.ib_stop),
            })
        slots.extend(self._group_slots(pnl))
        return {
            "connected": self.alive,
            "error": self.last_error,
            "account": self._book_key,
            # 🔴 **账户登录**（≠ 行情）。`alive` 看的是 feed，账户登出了策略照样跑；
            # 一键对齐必须看这一个：没登账户时发单只是空转（`align()` 直接
            # 返回 0），界面得先把这些账户排除掉，别让用户以为点过了就平完了
            "account_ready": self.api is not None,
            # 「账户没有」有两种：用户自己登出了（要尊重），和它压根没接上
            # （启动时接入失败 / 账户在交易页重新登录过）。后者界面「刷新」
            # 应当真的去接一次，而不是把空引用再挂一遍装作刷新过了
            # `getattr` 兜一下：自检里常用 `__new__` 造精简桩，而 `snapshot()`
            # 是 0.4 秒一次的实盘主路径，不能因为桩少一个属性就抛
            "user_detached": bool(getattr(self, "_detached_by_user", False)),
            # 账户持仓是否已经拉到手。为 False 时 `mismatch()` 一定返回空 ——
            # 那不是「一致」，是「还不知道」。界面必须分开说，否则用户点一键对齐
            # 会毫无反应，还以为是按钮坏了
            "pos_ready": self._pos is not None,
            # 🔴 比 `pos_ready` 更严的一条：**持仓真的从柜台读回来过一整轮**。
            # `pos_ready` 只表示「账户接口挂上了」，它同时被界面用来判断「账户在
            # 线/该显示登出还是登录」，所以不能拿来当「比对结果作不作数」用。
            # 为 False 时 `mismatch()` 的空表是「不知道」，不是「一致」——
            # 2026-11-23 融航就是拿这个空表报了「账户与策略理论持仓一致」，
            # 而账户里实际有 6 个合约的仓。
            "pos_synced": self._pos_live(),
            # 读到了几个合约的持仓。界面不用它做判断，出问题时一眼看得出
            # 「读到了但比对说一致」和「一条都没读到」的区别
            "pos_count": len(self._pos) if self._pos is not None else 0,
            "slots": slots,
            "mismatch": self.mismatch(),
            # `want_bars()` 点名的策略才带，其余时候是空字典
            "bars": {sid: self.bars_of(sid) for sid in want} if want else {},
            "limits": self.limit.summary(),
            "limit_used": self.limit.exhausted(),
            "aligning": len(self._align_slot.pending),
            "align_action": self._align_slot.last_action,
        }
