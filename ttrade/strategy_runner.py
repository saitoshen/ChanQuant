"""多策略自动交易引擎。

一个 `TqApi` 管一组策略：每个策略订阅自己的主连 K 线，新 bar 到了就重算信号，
目标净仓变了就下单。和 `gui/app.py` 里那个「单策略 + 手动绿灯」是两套东西 ——
那边是研究用，这边是组合运行。

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

import time
from dataclasses import dataclass, field

import pandas as pd

from ttrade import factor_strategy as fstrat
from ttrade import replay
from ttrade import strategy_store as SStore
from ttrade.strategies import run_strategy
from ttrade.tqfetch import PERIOD_SEC
from ttrade.tqlive import AccountSpec, bars_from_serial
from ttrade.tqsdk_symbols import main_symbol

KLINE_LEN = 3000          # 每个策略回看多少根。日线 3000 根够跑 MA200，5m 够跑一周多
MIN_BARS = 30             # 少于这么多根不算信号
# 因子策略要多得多的历史：窗口 252 加上 z 分数，少于这个数算出来的分数全是 NaN
FACTOR_MIN_BARS = 300

# 实时订阅能直接用的周期。`PERIOD_SEC` 是下载链路的口径，只有 5m/1h/1d；
# 天勤的 `get_kline_serial` 本身支持 15m / 30m / 2h / 4h，实时段不用像本地历史
# 那样由 5m/1h 合并出来。
RT_SEC = {**PERIOD_SEC, "15m": 900, "30m": 1800, "2h": 7200, "4h": 14400}


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
    target: int = 0            # 策略理论净仓（手）
    actual: int = 0            # 账户在该合约上的实际净仓
    underlying: str = ""       # 当前真实主力合约
    last_bar: object = None
    error: str = ""
    last_action: str = ""
    pending: list = field(default_factory=list)
    retired: list = field(default_factory=list)   # 换月后要清仓的老合约

    def title(self) -> str:
        return self.cfg.title()


# ------------------------------------------------------------------ 引擎


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
        self._klines: dict[str, object] = {}
        self._quotes: dict[str, object] = {}
        self._pos = None
        self._orders = None
        self._log_fn = None

    # ---- 生命周期 ----

    def on_log(self, fn) -> None:
        self._log_fn = fn

    def _log(self, msg: str) -> None:
        if self._log_fn is not None:
            try:
                self._log_fn(msg)
            except Exception:  # noqa: BLE001
                pass

    def start(self) -> None:
        from tqsdk import TqApi, TqAuth

        if not self.spec.user or not self.spec.password:
            raise ValueError("缺少天勤账户或密码")
        self.api = TqApi(
            self.spec.build(),
            auth=TqAuth(self.spec.user, self.spec.password),
            web_gui=False,
        )
        self._pos = self.api.get_position()
        self._orders = self.api.get_order()
        self._log(f"账户已连接：{self.spec.label()}")

    def close(self) -> None:
        api, self.api = self.api, None
        self._klines.clear()
        self._quotes.clear()
        self._pos = self._orders = None
        for st in self._slots.values():
            st.running = False
            st.pending.clear()
            st.retired.clear()
        if api is not None:
            try:
                api.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def alive(self) -> bool:
        return self.api is not None

    # ---- 槽位管理 ----

    def add(self, cfg: SStore.StrategyConfig) -> SlotState:
        st = SlotState(cfg=cfg)
        self._slots[cfg.sid] = st
        return st

    def remove(self, sid: str) -> None:
        st = self._slots.pop(sid, None)
        if st is not None:
            self._cancel_all(st)

    def get(self, sid: str) -> SlotState | None:
        return self._slots.get(sid)

    def slots(self) -> list[SlotState]:
        return list(self._slots.values())

    def update(self, cfg: SStore.StrategyConfig) -> None:
        """换掉一条策略的配置。

        界面改了手数 / 名称之后必须走这里：`SlotState.cfg` 是启动时那一份的引用，
        组合文件里改完不会自动同步过来，否则「改完手数没反应」得等重启才生效。
        """
        st = self._slots.get(cfg.sid)
        if st is None:
            return
        st.cfg = cfg
        # 手数变了目标仓位跟着变，立刻重算 —— 不然要等下一根 bar 才调整。
        # 停着的时候也算：`_rebalance` 只对 running 的槽位生效，重新启动时
        # 若刚好没出新 bar（`_refresh_signal` 会早退），靠的就是这个值。
        st.target = st.signal * max(1, int(cfg.volume or 1))
        # 策略参数（快线 / 慢线）也可能一起改了。不清掉 last_bar 的话，
        # `_refresh_signal` 认为「这根 bar 算过了」直接早退，新参数要等下一根
        # K 线才生效 —— 用户会以为改了没用。
        st.last_bar = None

    def enable(self, sid: str, on: bool) -> None:
        st = self._slots.get(sid)
        if st is None:
            return
        if on and not st.cfg.rule_key and st.cfg.family == "rule":
            st.error = "这条策略没有选规则"
            return
        if on and st.cfg.family == "factor" and (st.cfg.factor_mode or "single") == "ml":
            # 机器学习每根 K 线都要重训一遍（几百折），放进 tick 循环会把整个
            # 连接卡住 —— 一个账户的策略是共享同一个 api 的，卡住就是全都停摆
            st.error = "机器学习因子策略暂不支持实时交易（每根 K 线重训太慢），请改用单因子 / 多因子"
            return
        if on and st.cfg.family == "factor" and not st.cfg.factor_keys:
            st.error = "这条因子策略没有选中因子"
            return
        if on and st.cfg.period not in RT_SEC:
            st.error = f"实时不支持 {st.cfg.period} 周期（可用：{'/'.join(RT_SEC)}）"
            return
        st.running = bool(on)
        st.error = ""
        st.last_action = "启动" if on else "停止"
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
                self._rebalance(st)
                self._chase(st)
            except Exception as exc:  # noqa: BLE001
                st.error = f"{type(exc).__name__}: {exc}"
                st.running = False
                self._log(f"{st.title()} 出错停机：{st.error}")

    # ---- 信号 ----

    def _subscribe(self, st: SlotState) -> None:
        key = st.cfg.key()
        if key in self._klines:
            return
        tq = main_symbol(st.cfg.symbol, st.cfg.exchange)
        self._klines[key] = self.api.get_kline_serial(
            tq, RT_SEC[st.cfg.period], data_length=KLINE_LEN
        )
        self._quotes[key] = self.api.get_quote(tq)

    def _refresh_signal(self, st: SlotState) -> None:
        self._subscribe(st)
        key = st.cfg.key()
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
        # 必须 `.iloc[-1]`：天勤的 kline 是 DataFrame，`kl["datetime"]` 是带
        # RangeIndex 的 Series，`series[-1]` 走的是**按标签**取值，直接抛
        # `KeyError: -1`。这一行曾经让每条策略在第一个 tick 就停机。
        last = int(kl["datetime"].iloc[-1])
        if last == st.last_bar:
            return                       # 同一根 bar 里不重算
        st.last_bar = last

        df = bars_from_serial(kl, st.cfg.period)
        if len(df) < MIN_BARS:
            return
        st.signal = self._signal(st, df)
        if st.cfg.note != "short":
            st.signal = max(0, st.signal)     # 不允许做空时把空头信号裁掉
        st.target = st.signal * max(1, int(st.cfg.volume))

    def _signal(self, st: SlotState, df: pd.DataFrame) -> int:
        """按当前配置算出最后一根 bar 上的目标仓位（-1 / 0 / 1）。

        规则策略和因子策略走两条不同的路。因子策略用的是和研究页同一套
        `run_factor_strategy`：同一个因子、同一组阈值，回测里是几手、实盘就是
        几手 —— 两边各写一份的话，「回测赚钱、实盘亏钱」永远说不清是策略不行
        还是实现不一致。

        `orient="auto"` 在实时段有个天然缺陷：方向是按全样本 IC 的符号定的，
        样本一变就可能翻。用户要是没显式指定方向，日志里会看到它翻。
        """
        if st.cfg.family == "factor":
            if len(df) < FACTOR_MIN_BARS:
                return 0                 # 历史不够，分数全是 NaN，先空仓等数据
            out = fstrat.run_factor_strategy(
                df, st.cfg.factor_mode or "single", list(st.cfg.factor_keys or []),
                replay.factor_config(st.cfg.factor_cfg or {}),
            )
            sig = out.df["signal"].iloc[-1]
            return 0 if pd.isna(sig) else int(sig)

        out = run_strategy(df, st.cfg.rule_key, st.cfg.rule_params)
        sig = out["signal"].iloc[-1]
        return 0 if pd.isna(sig) else int(sig)

    # ---- 调仓 ----

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
        st.actual = self._net_pos(st.underlying)
        if st.pending:
            return                       # 还有没追完的单，等它落地

        # 换月留下的老合约先平掉。一次只发一笔，等它落地再谈新合约 ——
        # 同时发两笔，账户上会出现一段「老合约还持着、新合约已开着」的双倍敞口。
        for sym in list(st.retired):
            have = self._net_pos(sym)
            if have == 0:
                st.retired.remove(sym)
                continue
            self._send(st, sym, -have)
            return

        delta = st.target - st.actual
        if delta == 0:
            return
        self._send(st, st.underlying, delta)

    def _offset_for(self, symbol: str, direction: str, volume: int) -> str:
        """开仓 / 平今 / 平仓。

        上期所、大商所、郑商所区分平今；中金所不区分（一律 CLOSE）。
        判断依据是「要平的那一侧有多少今仓」：今仓够就 CLOSETODAY，否则 CLOSE。
        """
        if self._pos is None:
            return "OPEN"
        p = self._pos.get(symbol)
        if p is None:
            return "OPEN"
        # 卖出平多头，买入平空头
        if direction == "SELL":
            total = int(getattr(p, "pos_long", 0) or 0)
            today = int(getattr(p, "pos_long_today", 0) or 0)
        else:
            total = int(getattr(p, "pos_short", 0) or 0)
            today = int(getattr(p, "pos_short_today", 0) or 0)
        if total <= 0:
            return "OPEN"
        if symbol.startswith("CFFEX"):
            return "CLOSE"
        return "CLOSETODAY" if today >= volume else "CLOSE"

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

    def _send(self, st: SlotState, symbol: str, delta: int) -> None:
        direction = "BUY" if delta > 0 else "SELL"
        vol = abs(int(delta))
        offset = self._offset_for(symbol, direction, vol)
        price = self._price_at(symbol, direction)
        if not price or price != price:      # NaN / 0 都说明盘口还没来
            st.last_action = "等盘口"
            return
        oid = self._order_id(self.api.insert_order(
            symbol=symbol, direction=direction, offset=offset,
            volume=vol, limit_price=price,
        ))
        st.pending.append(
            PendingOrder(order_id=oid, symbol=symbol, direction=direction,
                         offset=offset, volume=vol, price=price, ts=time.time())
        )
        st.last_action = f"{'买' if direction == 'BUY' else '卖'}{offset} {vol} 手 @ {price:.2f}"
        self._log(f"{st.title()}：{st.last_action}")

    def _cancel_all(self, st: SlotState) -> None:
        if self.api is None:
            st.pending.clear()
            return
        for o in st.pending:
            try:
                order = self._orders.get(o.order_id) if self._orders is not None else None
                if order is not None and getattr(order, "status", "") == "ALIVE":
                    self.api.cancel_order(order)
            except Exception:  # noqa: BLE001
                pass
        st.pending.clear()

    # ---- 追单 ----

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
                st.last_action = "已成交"
                continue

            if status == "DEAD":
                if o.phase != "cancelling":
                    # 不是我们撤的：拒单、涨跌停打回、资金不足……记一笔就走
                    why = getattr(order, "last_msg", "") or "未知原因"
                    st.last_action = f"委托被拒：{why}"
                    self._log(f"{st.title()}：{st.last_action}")
                    continue
                if o.retries >= self.chase.retries:
                    st.last_action = f"追单 {o.retries} 次仍未成交，放弃"
                    self._log(f"{st.title()}：{st.last_action}")
                    continue
                o.retries += 1
                o.phase = "live"
                o.ts = now
                price = self._price_at(o.symbol, o.direction, self.chase.ticks * o.retries)
                if self.chase.use_market and o.retries >= self.chase.retries:
                    price = float("nan")     # 天勤用 NaN 表示市价
                try:
                    o.order_id = self._order_id(self.api.insert_order(
                        symbol=o.symbol, direction=o.direction, offset=o.offset,
                        volume=o.volume, limit_price=price,
                    ))
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
            try:
                if order is not None:
                    self.api.cancel_order(order)
                o.phase = "cancelling"
            except Exception:  # noqa: BLE001
                pass
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

    def mismatch(self) -> list[dict]:
        """理论持仓与实际持仓不一致的合约。

        `managed` 表示这个合约上有**正在运行**的策略。为 False 的多半是「策略停了
        但仓位还留着」，或者是用户手工开的仓。界面靠这个标记把两类分开显示 ——
        前者是策略自己没跟上，后者得先让人看清楚再动。
        """
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
        return out

    def align(self, close_orphans: bool = True) -> int:
        """把账户仓位调成和策略列表完全一致。返回下发的委托数。

        - 托管合约（有策略在跑）：补到理论净仓，多了减、少了加
        - 未托管合约（没有策略在管）：`close_orphans=True` 时平到 0

        第二类是「策略停掉之后留下的仓」和「手工开的仓」。默认平掉是因为
        用户对这个按钮的预期就是「一键把账户摆成策略列表的样子」；但这是
        真金白银，调用方必须先确认过再调，别在启动流程里顺手带上。
        """
        if self.api is None:
            return 0
        n = 0
        for row in self.mismatch():
            if not row["managed"] and not close_orphans:
                continue
            sym = row["symbol"]
            delta = row["delta"]
            if delta == 0:
                continue
            direction = "BUY" if delta > 0 else "SELL"
            vol = abs(delta)
            offset = self._offset_for(sym, direction, vol)
            price = self._price_at(sym, direction)
            if not price or price != price:
                continue
            self.api.insert_order(
                symbol=sym, direction=direction, offset=offset,
                volume=vol, limit_price=price,
            )
            n += 1
            self._log(f"对齐 {sym}：{direction} {offset} {vol} 手 @ {price:.2f}")
        return n

    # ---- 快照 ----

    def snapshot(self) -> dict:
        return {
            "connected": self.alive,
            "error": self.last_error,
            "account": self.spec.label(),
            "slots": [
                {
                    "sid": st.cfg.sid,
                    "title": st.title(),
                    "key": st.cfg.key(),
                    "period": st.cfg.period,
                    "strategy": st.cfg.strategy_label(),
                    "volume": st.cfg.volume,
                    "running": st.running,
                    "signal": st.signal,
                    "target": st.target,
                    "actual": st.actual,
                    "underlying": st.underlying,
                    "error": st.error,
                    "action": st.last_action,
                    "retired": list(st.retired),
                }
                for st in self._slots.values()
            ],
            "mismatch": self.mismatch(),
        }
