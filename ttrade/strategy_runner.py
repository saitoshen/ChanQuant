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
from ttrade import daylimit, orderbook, positionbook, topup, tqdata
from ttrade import strategy_store as SStore
from ttrade.strategies import STRATEGIES, next_levels, run_strategy
from ttrade.tqfetch import PERIOD_SEC
from ttrade.tqlive import AccountSpec, bars_from_serial, merge_live
from ttrade.tqsdk_symbols import is_stock, trade_symbol

KLINE_LEN = 3000          # 每个策略回看多少根。日线 3000 根够跑 MA200，5m 够跑一周多
MIN_BARS = 30             # 少于这么多根不算信号
# 因子策略要多得多的历史：窗口 252 加上 z 分数，少于这个数算出来的分数全是 NaN
FACTOR_MIN_BARS = 300

# 多品种策略在引擎里被拆成「一个品种一个槽位」，槽位 id 用 `父 id#品种键`。
# 界面上仍然是一条策略一行 —— 用户管的是「这条策略」，不是 19 个各自为政的
# 槽位；拆开只是为了每个品种有自己的订阅、自己的仓位账本、自己的手数。
GROUP_SEP = "#"

# 「没有训练产物」这件事多久之内不再重试 / 不再刷日志。产物是用户手动训的，
# 训完之后要么重启、要么下一次重试就会加载上 —— 每次 tick 都去试一遍 + 刷一行
# 日志，日志会被冲爆。
ML_PRED_RETRY = 60.0

# 实时订阅能直接用的周期。`PERIOD_SEC` 是下载链路的口径，只有 5m/1h/1d；
# 天勤的 `get_kline_serial` 本身支持 15m / 30m / 2h / 4h，实时段不用像本地历史
# 那样由 5m/1h 合并出来。
RT_SEC = {**PERIOD_SEC, "15m": 900, "30m": 1800, "2h": 7200, "4h": 14400}

# 分平今 / 平昨的交易所：上期所、能源中心。其余（大商所 / 郑商所 / 中金所）
# 一律用 CLOSE，给它们发 CLOSETODAY 会被交易所整单打回。
SPLIT_OFFSET_EX = ("SHFE", "INE")

# 被交易所打回之后的冷却。`tick()` 跟着行情推送跑，一秒能来十几次；没有冷却
# 的话，一笔注定失败的委托一秒能重报十几遍 —— 日志刷屏，主线程直接卡死。
REJECT_COOL = 5.0
MAX_REJECTS = 3

# 刚记完账（本策略成交）之后，账户持仓明细要过几百毫秒才跟上。这段时间里
# 账本比账户多是一时的，不能当成「被手工平了」去收窄 —— 见 `_trim`。
BOOK_GRACE = 3.0

# 「一键对齐」两次点击之间至少隔这么久。连点两下不该下发两轮。
ALIGN_GAP = 5.0

# 对齐单最多挂多久。超时就撤掉、清空在途列表，让策略恢复调仓 ——
# 否则一个挂在盘口外的限价单能让整个账户永远停在「对齐执行中」。
ALIGN_TIMEOUT = 30.0

# 补历史没成功（刚订阅、序列还没填充）时隔多久再试。`tick()` 跟着行情推送跑，
# 一秒十几次；每次都把 3000 行转一遍 DataFrame 纯属白烧。
TOP_RETRY = 2.0

# 仓位比对多久重算一次。比对要把所有策略的理论净仓和整个账户的持仓对一遍，
# 账户里合约一多就是几百次属性读取；而快照是 0.4 秒推一次的 —— 每推一次都
# 重算，等于为了一个「秒级才变一次」的数字空转。5 秒一次够看，也够用户点
# 对齐时拿到的新鲜度（`align()` 会强制重算，见 `mismatch(fresh=True)`）。
MISMATCH_GAP = 5.0


def default_margin_rate(symbol: str) -> float:
    """品种的默认保证金率。杠杆 = 1 / 保证金率。

    只在拿不到交易所实时保证金时兜底（`quote.margin` 一般都有）。商品、股指
    约 10 倍，国债 30~50 倍 —— 国债波动小、保证金低，同样的钱能铺更多手，
    这正是不该用「固定几手」的理由。
    """
    s = (symbol or "").upper()
    if s in ("T", "TF", "TS", "TL"):
        return 0.03            # 国债期货：约 33 倍
    if s in ("IF", "IH", "IC", "IM"):
        return 0.12            # 股指期货：约 8 倍
    return 0.10                # 商品期货：约 10 倍


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


@dataclass
class SlotState:
    """一个策略槽位的运行状态。界面只读这个，不碰天勤对象。"""

    cfg: SStore.StrategyConfig
    running: bool = False
    signal: int = 0
    lots: int = 1              # 一次满仓对应几手（按资金 / 止损金额算出来）
    target: int = 0            # 策略理论净仓（手）
    # **这条策略自己**在各合约上的净仓（手），按自己委托的成交累计，落盘在
    # `positionbook`。不是账户净仓 —— 账户是所有策略 + 手工单 + 别的平台的和，
    # 拿它当自己的仓，同品种两条策略就会互相把对方的仓当成自己的。
    held: dict = field(default_factory=dict)
    book_ts: float = 0.0       # 最近一次账本变动的时刻，见 `_trim` 的宽限期
    underlying: str = ""       # 当前真实主力合约
    last_bar: object = None
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
                            self._diag.write("ML", f"{key} 快路径算不出来：{why}")
                        continue
                    sig = int(r["signal"])
                    if self._diag is not None:
                        self._diag.write(
                            "ML", f"{key} 快路径 ok signal={sig} "
                                  f"score={r.get('score'):.4f} "
                                  f"耗时={time.time() - t0:.2f}s bar={len(job['df'])}")
                else:
                    # ---- 慢路径：后台重训（没有训练产物时的兜底）----
                    if self._diag is not None:
                        self._diag.write("ML", f"{key} 走慢路径（后台重训），"
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
                        self._diag.write("ML", f"{key} 慢路径完成 signal={sig} "
                                              f"耗时={time.time() - t0:.1f}s")
            except Exception as exc:  # noqa: BLE001
                why = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._err[key] = why
                if self._log is not None:
                    self._log(f"机器学习实时信号计算失败（{key}）：{why}")
                if self._diag is not None:
                    self._diag.write("ML", f"{key} 抛异常：{why}")
                continue
            with self._lock:
                self._err.pop(key, None)
                self._out[key] = {"signal": sig, "ts": time.time(),
                                  "bars": int(len(job["df"])),
                                  "cost": time.time() - t0,
                                  "fast": predictor is not None}


class PortfolioRunner:
    """一个账户 + 一组策略。

    纯 Python，不依赖 Qt —— 界面层用 QThread 包一层（见 `gui/portfolio_page.py`）。
    """

    def __init__(self, spec: AccountSpec, chase: SStore.ChaseConfig | None = None) -> None:
        self.spec = spec
        self.chase = chase or SStore.load_chase()
        self.api = None
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
        self.limit = daylimit.DayLimit(self.spec.label())
        self._halted = ""
        # 「一键对齐」发出的单放在这个槽位里。它只当在途单的容器，不参与信号：
        # 对齐是**一次性动作**，发完就交给追单，成交 / 撤完就结束。没有它的话，
        # 对齐的单没人记，`_rebalance` 下一 tick 看到仓位还没变又发一遍 ——
        # 一秒十几次 tick，一笔对齐能变成十几笔委托。
        self._align_slot = SlotState(cfg=SStore.StrategyConfig(
            sid="__align__", name="一键对齐", period="1d"))
        self._align_ts = 0.0
        # 上一次算出来的比对结果和它的时间戳。见 `MISMATCH_GAP`。
        self._mm_cache: list[dict] = []
        self._mm_ts = 0.0
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
        # 第一次跑持仓账本（升级上来的老用户，或者本机第一次跑）。这种时候账户里
        # 可能已经有仓了，账本从 0 开始会把目标再开一遍 —— 见 `_adopt_if_needed`。
        self._fresh_book = not positionbook.exists()
        self._adopted: set[str] = set()
        self._aux_warned: set[str] = set()

    # ---- 生命周期 ----

    def on_log(self, fn) -> None:
        self._log_fn = fn

    def _log(self, msg: str) -> None:
        if self._log_fn is not None:
            try:
                self._log_fn(msg)
            except Exception:  # noqa: BLE001
                pass

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

    def start(self) -> None:
        from tqsdk import TqApi, TqAuth

        if not self.spec.user or not self.spec.password:
            raise ValueError("缺少天勤账户或密码")
        # 补数据放在连接之前：连上就要开始算信号了，那时候快照还是旧的
        self._ensure_aux()
        self.api = TqApi(
            self.spec.build(),
            auth=TqAuth(self.spec.user, self.spec.password),
            web_gui=False,
        )
        self._pos = self.api.get_position()
        self._orders = self.api.get_order()
        self._log(f"账户已连接：{self.spec.label()}")
        self._sweep_orders()

    def _sweep_orders(self) -> None:
        """撤掉上一次运行留下、还挂着的委托。

        程序被强杀（或直接关机）时 `close()` 跑不到，限价单会一直挂在交易所。
        下次打开账户，用户什么都没点，委托却在成交 —— 那其实是上一次留下的，
        看着就像「不点按钮它自己在下单」。

        只撤**本程序自己下过**的单（order_id 记在 `live_orders.json`），
        快期客户端里手点的委托不在名单里，不会被误伤。
        """
        ids = orderbook.load(self.spec.label())
        if not ids:
            return
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
        orderbook.clear(self.spec.label())
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
        return self.api is not None

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
            if (one.sizing or "capital").lower() != "fixed":
                one.principal = float(cfg.principal or 0) / n
            out.append(one)
        return out

    def _add_one(self, cfg: SStore.StrategyConfig) -> SlotState:
        st = SlotState(cfg=cfg)
        # 账本跟着「账户 + 策略 id」走，重启后接着记。不读回来的话，目标 -11
        # 会从头再开一遍，账户直接变 -22。
        st.held = dict(positionbook.load(self.spec.label()).get(cfg.sid, {}))
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

    def _remove_one(self, sid: str) -> None:
        st = self._slots.pop(sid, None)
        if st is not None:
            self._cancel_all(st)
            # 策略删了，账本也别留着 —— 同 id 再建一条会拿到别人的仓位
            positionbook.forget(self.spec.label(), sid)

    def remove(self, sid: str) -> None:
        for k in self._groups.pop(sid, []) or []:
            self._child_of.pop(k, None)
            self._remove_one(k)
        self._parents.pop(sid, None)
        self._remove_one(sid)

    def _save_book(self, st: SlotState) -> None:
        st.book_ts = time.time()
        positionbook.save(self.spec.label(), st.cfg.sid, st.held)

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
        st.cfg = cfg
        # 手数变了目标仓位跟着变，立刻重算 —— 不然要等下一根 bar 才调整。
        # 停着的时候也算：`_rebalance` 只对 running 的槽位生效，重新启动时
        # 若刚好没出新 bar（`_refresh_signal` 会早退），靠的就是这个值。
        # `lots` 是上一根 bar 按价格算出来的手数；改分配资金后要等下一次
        # `_refresh_signal` 才会重算，所以下面还要清 last_bar。
        # 固定手数模式例外：手数就写在配置里，立刻生效，不用等行情。
        if (cfg.sizing or "capital").lower() == "fixed":
            st.lots = max(1, int(cfg.volume or 1))
        st.target = st.signal * max(1, st.lots)
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
        st.running = bool(on)
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
        if self.api is None:
            return False
        try:
            return bool(self.api.wait_update(deadline=time.time() + timeout))
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise

    def tick(self) -> None:
        """每轮 wait_update 之后调一次：刷信号、调仓、追单。"""
        if self.api is None:
            return
        for st in list(self._slots.values()):
            if not st.running:
                continue
            try:
                self._refresh_signal(st)
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
            if time.time() - oldest > ALIGN_TIMEOUT:
                # 挂太久了。不撤的话在途列表永远不清空，`_rebalance` 就永远让路，
                # 整个账户看着像卡死。撤掉、放开，让策略接着跑。
                self._cancel_all(self._align_slot)
                self._log(
                    f"对齐委托 {ALIGN_TIMEOUT:.0f} 秒未成交，已撤单，策略恢复调仓"
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
                self._diag.write("ML", f"{st.title()}（sid={key}）退回后台重训："
                                       f"{why or '（原因不明）'}")
            return None
        with self._lock:
            self._ml_pred[key] = ("ok", pred, time.time())
        self._log(f"{st.title()}：用训练产物算信号（只预测，不重训）")
        if self._diag is not None:
            self._diag.write("ML", f"{st.title()}（sid={key}）用训练产物，"
                                   f"走快路径（只预测）")
        return pred

    def _subscribe(self, st: SlotState) -> None:
        key = self._skey(st.cfg)
        if key in self._klines:
            self._top_up(st)      # 第一次没订成源周期的话，这里接着试
            return
        tq = trade_symbol(st.cfg.symbol, st.cfg.exchange)
        # 订**源周期**再本地合并（15m/30m 走 5m，2h/4h 走 1h），和本地历史的
        # 合并口径一致；直接订原生 15m 时间边界对不上，拼起来就是碎的
        src, n = tqdata.PERIODS.get(st.cfg.period, (st.cfg.period, 1))
        self._klines[key] = self.api.get_kline_serial(
            tq, RT_SEC[src], data_length=min(10000, KLINE_LEN * max(1, n))
        )
        self._quotes[key] = self.api.get_quote(tq)
        self._top_up(st)

    def _top_up(self, st: SlotState) -> None:
        """把引擎手上这份实时 K 线写回本地库，库里就不再停在几天前。

        落的是**源周期**（5m / 1h）：15m / 30m / 2h / 4h 在库里本来就是由它们
        合并出来的，单独存一份读的时候根本对不上。`_subscribe` 现在一律订源
        周期，所以这里直接复用那份序列，不再为补库多订一条。

        失败不抛：补数据是锦上添花，不该把正在跑的策略打死。刚订阅时序列还
        没填充是常态（`ok=False`），下一 tick 再来。
        """
        if self.api is None:
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
        if r["added"]:
            self._log(f"{st.title()}：本地历史补到最新 +{r['added']} 根"
                      f"（{src}，到 {r['last']:%Y-%m-%d %H:%M}）")

    def _refresh_signal(self, st: SlotState) -> None:
        self._subscribe(st)
        key = self._skey(st.cfg)
        kl = self._klines.get(key)
        if kl is None or not len(kl):
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
        # 最后一根是正在走的活 bar，**K 线内策略要用它**（「当日开盘价」在一天
        # 的第一根上就是它自己）。下面那行 `iloc[:-1]` 只是给「收盘定信号」的
        # 老策略用的副本，`full` 留给 `_ib_arm`
        full = df
        if len(df) > 1:
            df = df.iloc[:-1]
        if len(df) < MIN_BARS:
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
        st.lots = self._lots(st, self._mark_price(st, df), atr(df, st.cfg.stop_atr_n))
        if self._ib_on(st):
            # K 线内那一族不用 `signal` 定目标仓 —— 入场 / 止损由盘中触发价
            # 决定（见 `_intrabar_tick`）。这根 bar 在这里只做两件事：
            # ① 重算本根的触发价；② 执行上一根收盘留下的「反向突破离场」
            st.signal = 0
            self._ib_arm(st, full)
            return
        st.target = st.signal * st.lots

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
        st.target = st.signal * st.lots
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
                st.lots = max(1, int(float(st.cfg.risk_amount or 0) / (d * m)))
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
        if el == el and price >= el:
            st.target = st.lots
            st.ib_fired = True
            st.ib_exit_pending = False     # 重新持仓了，上一笔的平仓账已清
            # 止损绑**触发价**（那一刻的下单价），不是此刻的最新价 ——
            # 最新价可能已经冲过去一段，用它会悄悄把止损挪得更松
            st.ib_stop = el - d if (d == d and d > 0) else float("nan")
            st.last_action = f"K 线内入场：现价 {price:.3f} 越过 {el:.3f}"
            self._log(f"{st.title()}：K 线内做多 {st.underlying} 触发价 {el:.3f}"
                      f"，现价 {price:.3f}，止损 {st.ib_stop:.3f}")
        elif (es == es and price <= es
              and st.cfg.note == "short"):    # 不允许做空时忽略空头触发价
            st.target = -st.lots
            st.ib_fired = True
            st.ib_exit_pending = False
            st.ib_stop = es + d if (d == d and d > 0) else float("nan")
            st.last_action = f"K 线内入场：现价 {price:.3f} 跌破 {es:.3f}"
            self._log(f"{st.title()}：K 线内做空 {st.underlying} 触发价 {es:.3f}"
                      f"，现价 {price:.3f}，止损 {st.ib_stop:.3f}")

    def _signal_frame(self, st: SlotState, kl) -> pd.DataFrame:
        """算信号用的行情：**本地前复权历史 + 实时 bar**。

        回测（详情页）走 `tqdata.load_main(adjust="qfq")`，实盘必须同一份口径 ——
        直接拿天勤原始价算，换月那一下的复权差会把 60 根动量推到 `entry_z` 两侧：
        回测空仓、实盘反向持仓，两边永远对不上。

        `merge_live` 只接**严格晚于**本地最后一根的实时 bar，重叠的保留本地那份
        （本地带正确复权因子，实时段因子是 1.0）。换月时调用方会清掉缓存，让新的
        换月跳空也按前复权抹平。
        """
        live = bars_from_serial(kl, st.cfg.period)
        n = tqdata.PERIODS.get(st.cfg.period, (st.cfg.period, 1))[1]
        if n > 1:
            live = tqdata.resample(live, n)
        if live.empty:
            return live
        key = self._skey(st.cfg)
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
            self._qfq_base[key] = base
        if base is None or base.empty:
            return live
        if live.index[0] > base.index[-1]:
            # 本地历史太旧，和实时段接不上（中间断了一截）。硬拼会在断口造一次
            # 假跳空，比用原始价更糟 —— 退回实时原始价，并记一条日志
            self._log(f"{st.title()}：本地历史与实时段接不上，暂用实时原始价")
            return live
        # **和回测同源同长**：`replay.run` 先按 `range_years` 切一刀（还多留
        # `WARMUP_BARS` 根预热），这里切同一刀。详情页画多少根、从哪根开始算，
        # 实盘就用同样那些根 —— 所见即所得，不再截 KLINE_LEN。
        merged = merge_live(base, live)
        merged = replay.window(merged, getattr(st.cfg, "range_years", 0) or 0)
        if merged is None or len(merged) < 2:
            return live
        return merged

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
                return 0                 # 历史不够，分数全是 NaN，先空仓等数据
            df = self._with_aux(st, df)

        if st.cfg.family == "factor":
            fc = replay.factor_config(st.cfg.factor_cfg or {})
            out = fstrat.run_factor_strategy(
                df, mode or "single", list(st.cfg.factor_keys or []),
                fc, extra=self._period_extra(st, fc), base_period=st.cfg.period,
            )
            sig = out.df["signal"].iloc[-1]
            return 0 if pd.isna(sig) else int(sig)

        rule = run_strategy(df, st.cfg.rule_key, st.cfg.rule_params,
                            reverse=bool(getattr(st.cfg, "rule_reverse", False)))
        if not needs_factor:
            sig = rule["signal"].iloc[-1]
            return 0 if pd.isna(sig) else int(sig)

        # 规则 + 因子过滤：规则信号是「原料」，过滤只做减法（否决 / 延后 / 离场）
        fc = replay.factor_config(st.cfg.factor_cfg or {})
        out = fstrat.run_rule_with_filter(
            df, rule["signal"].to_numpy(int), mode or "single",
            list(st.cfg.factor_keys or []), fc,
            extra=self._period_extra(st, fc), base_period=st.cfg.period,
        )
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
            lambda p: tqdata.load_main(sym, ex, p, adjust="qfq"),
            extra_keys=list(getattr(fc, "trigger_keys", None) or []),
        )

    def _with_aux(self, st: SlotState, df: pd.DataFrame) -> pd.DataFrame:
        """给实时序列挂上期限结构 / 期权列。

        天勤的 K 线只有 OHLC 和持仓量，展期、期权那十几个因子读的 `c1_close`、
        `iv_near` 之类全在衍生表里。研究页是 `DataStore` 加载时挂的，实时这条
        链路自己拼 df，不挂就是整列 NaN —— 因子分数算不出来，策略永远空仓。
        """
        try:
            from ttrade.auxdata import attach_intraday

            out = attach_intraday(df, st.cfg.symbol, st.cfg.exchange)
        except Exception:  # noqa: BLE001
            return df
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

    def _contract_quote(self, symbol: str):
        """真实合约的行情对象。拿不到就返回 None —— 定仓失败不该让策略停机。"""
        if self.api is None or not symbol:
            return None
        try:
            return self.api.get_quote(symbol)
        except Exception:  # noqa: BLE001
            return None

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

    @staticmethod
    def _margin_per_lot(q, price: float, mult: float, symbol: str) -> float:
        """每手保证金。优先用交易所给的 `quote.margin`，没有才按乘数 × 保证金率估。"""
        m = float(getattr(q, "margin", 0) or 0) if q is not None else 0.0
        if m > 0 and math.isfinite(m):
            return m
        if mult <= 0 or price <= 0:
            return 0.0
        return price * mult * default_margin_rate(symbol)

    def _volume_multiple(self, st: SlotState) -> float:
        """这个合约每手多少单位（合约乘数）。拿不到返回 0 —— 调用方自己兜底。"""
        q = self._contract_quote(st.underlying)
        try:
            return float(getattr(q, "volume_multiple", 0) or 0) if q is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _lots(self, st: SlotState, price: float, atr_value: float) -> int:
        """这条策略满仓该下几手。

        - `capital`：手数 = 本金 × 资金比例 / 每手保证金。100 万本金取 30%，
          就是拿 30 万铺 10 倍杠杆的品种 = 300 万名义价值；1 手 10 万的合约正好
          30 手。国债保证金低（3% 左右），同样的钱自然铺出更多手 —— 这就是
          「杠杆按品种自己的保证金比例」的意思。
        - `risk`：手数 = 本金 × 止损比例 / (止损距离 × 合约乘数)。止损距离 =
          倍数 × ATR，和策略自己的止损口径一致，所以「打止损就亏 1% 本金」
          是真的 1%。
        - `fixed`：老行为，固定手数。
        """
        cfg = st.cfg
        mode = (cfg.sizing or "capital").lower()
        if mode == "fixed":
            return max(1, int(cfg.volume or 1))

        q = self._contract_quote(st.underlying)
        mult = self._volume_multiple(st)

        if mode == "risk":
            stop_px = max(0.0, float(cfg.stop_mult or 0)) * float(atr_value or 0)
            if stop_px <= 0 or mult <= 0:
                return 1                     # 拿不到 ATR / 乘数：先 1 手，别裸奔
            per_lot = stop_px * mult
            return max(1, int(float(cfg.risk_amount or 0) / per_lot))

        per_margin = self._margin_per_lot(q, price, mult, cfg.symbol)
        if per_margin <= 0:
            return 1
        return max(1, int(float(cfg.capital or 0) / per_margin))

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
        orderbook.remember(self.spec.label(), self._order_id(order))
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
        except Exception:  # noqa: BLE001
            pass

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
        if his + today < self._side_pos(symbol, side):
            return None
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
        if closeable is not None:
            have = min(have, max(0, int(closeable)))
        if split and have > 0:
            got = self._split_pos(symbol, side)
            if got is None:
                return None              # 明细还没跟上，等下一轮，别瞎猜
            his, today = got
            have = max(have, his + today)

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
        try:
            q = self.api.get_quote(symbol)
        except Exception:  # noqa: BLE001
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
        legs = self._legs(symbol, delta, closeable)
        if legs is None:
            st.last_action = "持仓今昨明细未就绪，等下一轮"
            return 0
        if not legs:
            return 0
        direction = "BUY" if delta > 0 else "SELL"
        price = self._price_at(symbol, direction)
        if not price or price != price:      # NaN / 0 都说明盘口还没来
            st.last_action = "等盘口"
            return 0
        parts = []
        for offset, vol in legs:
            order = self._order(
                symbol=symbol, direction=direction, offset=offset,
                volume=vol, limit_price=price,
            )
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
        st.last_action = (
            f"{'买' if direction == 'BUY' else '卖'} {' + '.join(parts)} 手 @ {price:.2f}"
        )
        self._log(f"{st.title()}：{st.last_action}")
        return len(parts)

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
        cur = int(st.held.get(o.symbol, 0)) + sign * filled
        if cur:
            st.held[o.symbol] = cur
        else:
            st.held.pop(o.symbol, None)
        self._save_book(st)

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

            if status == "FINISHED":
                self._book(st, o, o.volume)
                st.last_action = "已成交"
                st.rejects = 0
                st.reject_msg = ""
                orderbook.forget(self.spec.label(), o.order_id)
                continue

            if status == "DEAD":
                # 撤单 / 拒单之前可能已经成交了一部分，先把这部分记进账本，
                # 再把 `o.volume` 减掉 —— 不减的话重报会按原手数再发一遍，
                # 成交之后就是「本该 5 手，实际 8 手」。
                filled = max(0, o.volume - left)
                if filled:
                    self._book(st, o, filled)
                    o.volume -= filled
                orderbook.forget(self.spec.label(), o.order_id)
                if o.phase != "cancelling":
                    # 不是我们撤的：拒单、涨跌停打回、资金不足……
                    # 只记一笔是不够的 —— `_rebalance` 下一轮照样再发，而 tick
                    # 跟着行情推送一秒能跑十几次，一个注定失败的单会把日志刷爆、
                    # 把主线程拖死。所以计数 + 冷却，连续几次直接停机。
                    why = getattr(order, "last_msg", "") or "未知原因"
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
                    orderbook.forget(self.spec.label(), o.order_id)
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
                    st.last_action = f"追单第 {o.retries} 次 @ {price:.2f}"
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

    def mismatch(self, fresh: bool = False) -> list[dict]:
        """理论持仓与实际持仓不一致的合约。

        `managed` 表示这个合约上有**正在运行**的策略。为 False 的多半是「策略停了
        但仓位还留着」，或者是用户手工开的仓。界面靠这个标记把两类分开显示 ——
        前者是策略自己没跟上，后者得先让人看清楚再动。

        结果缓存 `MISMATCH_GAP` 秒：快照 0.4 秒推一次，每次都重算纯属空转。
        `fresh=True` 绕过缓存 —— 真正要下单（`align()`）时必须用当下的持仓，
        拿 5 秒前的旧数去补仓，补出来的量就是错的。
        """
        now = time.time()
        if not fresh and now - self._mm_ts < MISMATCH_GAP:
            # 给副本。界面拿到的 list 会被到处传，别让它改到缓存里的 dict。
            return [dict(r) for r in self._mm_cache]
        theo = self.theoretical()
        keys = set(theo)
        if self._pos is not None:
            for sym, p in self._pos.items():
                if int(getattr(p, "pos_long", 0) or 0) or int(getattr(p, "pos_short", 0) or 0):
                    keys.add(sym)
        out = []
        for sym in sorted(keys):
            want = int(theo.get(sym, 0))
            have = self._net_pos(sym)
            if want != have:
                out.append({
                    "symbol": sym, "target": want, "actual": have,
                    "delta": want - have, "managed": sym in theo,
                })
        self._mm_cache = out
        self._mm_ts = now
        return [dict(r) for r in out]

    def align(self, close_orphans: bool = True) -> int:
        """把账户仓位调成和策略列表完全一致。返回下发的委托数。

        - 托管合约（有策略在跑）：补到理论净仓，多了减、少了加
        - 未托管合约（没有策略在管）：`close_orphans=True` 时平到 0

        第二类是「策略停掉之后留下的仓」和「手工开的仓」。默认平掉是因为
        用户对这个按钮的预期就是「一键把账户摆成策略列表的样子」；但这是
        真金白银，调用方必须先确认过再调，别在启动流程里顺手带上。

        **一次点击只发一轮**。单子记进 `_align_slot.pending`，之后由 `_chase`
        负责追价 / 撤单 / 重报，追完为止；期间 `_rebalance` 全体让路。上一轮
        还没落地时再点直接忽略 —— 否则「仓位还没变 → 判定没对齐 → 再发一遍」
        会变成一个每秒十几次的死循环。
        """
        if self.api is None:
            return 0
        if self._align_slot.pending:
            self._log("上一轮对齐还没落地，等它成交或撤掉再点")
            return 0
        now = time.time()
        if now - self._align_ts < ALIGN_GAP:
            self._log(f"对齐刚点过，{ALIGN_GAP:.0f} 秒内不重复下发")
            return 0
        self._align_ts = now
        self._align_slot.rejects = 0
        self._align_slot.reject_msg = ""
        n = 0
        for row in self.mismatch(fresh=True):
            if not row["managed"] and not close_orphans:
                continue
            sym = row["symbol"]
            delta = row["delta"]
            if delta == 0:
                continue
            sent = self._send(self._align_slot, sym, delta)
            if sent:
                # 对齐的目标就是 `sum(target)`，所以发完之后各策略的账本正好
                # 是「自己的 target」。不跟着摆的话：停掉的策略留着 -11 的账本，
                # 对齐把仓平了，再启动它目标 -11、账本 -11、差额 0 —— 它会以为
                # 自己还空着 11 手，实际上账户是平的。
                self._sync_after_align(sym)
            n += sent
        if n:
            self._log(f"对齐已下发 {n} 笔，剩下的交给追单")
        return n

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
            self._save_book(st)

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

    def _group_slots(self) -> list[dict]:
        """多品种策略在列表里那一行：把 N 个子槽位汇成一条。

        没有这一行，界面上就找不到这条策略（子槽位的 sid 是 `父#品种`）。
        目标仓不汇总 —— 19 个品种各有一个目标，加起来没有意义，界面显示「—」，
        逐合约的去监控器看。
        """
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

    def snapshot(self) -> dict:
        # 取一次就清掉。没清的话界面关掉 K 线之后，这份开销还每份快照都付
        want, self._want = self._want, []
        slots = [
            {
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
                "underlying": st.underlying,
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
            }
            for st in self._slots.values()
        ]
        slots.extend(self._group_slots())
        return {
            "connected": self.alive,
            "error": self.last_error,
            "account": self.spec.label(),
            "slots": slots,
            "mismatch": self.mismatch(),
            # `want_bars()` 点名的策略才带，其余时候是空字典
            "bars": {sid: self.bars_of(sid) for sid in want} if want else {},
            "limits": self.limit.summary(),
            "limit_used": self.limit.exhausted(),
            "aligning": len(self._align_slot.pending),
            "align_action": self._align_slot.last_action,
        }
