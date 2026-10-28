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

from ttrade import strategy_store as SStore
from ttrade.strategies import run_strategy
from ttrade.tqfetch import PERIOD_SEC
from ttrade.tqlive import AccountSpec, bars_from_serial
from ttrade.tqsdk_symbols import main_symbol

KLINE_LEN = 3000          # 每个策略回看多少根。日线 3000 根够跑 MA200，5m 够跑一周多
MIN_BARS = 30             # 少于这么多根不算信号

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

    def enable(self, sid: str, on: bool) -> None:
        st = self._slots.get(sid)
        if st is None:
            return
        if on and not st.cfg.rule_key and st.cfg.family == "rule":
            st.error = "这条策略没有选规则"
            return
        if on and st.cfg.family == "factor":
            st.error = "因子策略暂不支持实时交易，请用规则策略"
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
        out = run_strategy(df, st.cfg.rule_key, st.cfg.rule_params)
        sig = int(out["signal"].iloc[-1])
        if st.cfg.note != "short":
            sig = max(0, sig)            # 不允许做空时把空头信号裁掉
        st.signal = sig
        st.target = sig * max(1, int(st.cfg.volume))

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

    def _send(self, st: SlotState, symbol: str, delta: int) -> None:
        direction = "BUY" if delta > 0 else "SELL"
        vol = abs(int(delta))
        offset = self._offset_for(symbol, direction, vol)
        price = self._price_at(symbol, direction)
        if not price or price != price:      # NaN / 0 都说明盘口还没来
            st.last_action = "等盘口"
            return
        oid = self.api.insert_order(
            symbol=symbol, direction=direction, offset=offset,
            volume=vol, limit_price=price,
        )
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
        if not st.pending or self._orders is None:
            return
        now = time.time()
        keep: list[PendingOrder] = []
        for o in st.pending:
            order = self._orders.get(o.order_id)
            status = getattr(order, "status", "") if order is not None else ""
            left = int(getattr(order, "volume_left", 0) or 0) if order is not None else 0

            if status in ("FINISHED", "DEAD") or (order is not None and left == 0):
                if status == "FINISHED" and left == 0:
                    st.last_action = "已成交"
                continue                 # 这笔结束了，不再追

            if o.phase == "cancelling":
                # 等撤单真的生效再重报，否则旧单还有效时重报会重复成交
                if status == "DEAD":
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
                        o.order_id = self.api.insert_order(
                            symbol=o.symbol, direction=o.direction, offset=o.offset,
                            volume=o.volume, limit_price=price,
                        )
                        o.price = price
                        st.last_action = f"追单第 {o.retries} 次 @ {price:.2f}"
                        self._log(f"{st.title()}：{st.last_action}")
                    except Exception as exc:  # noqa: BLE001
                        st.last_action = f"追单失败：{exc}"
                        continue
                keep.append(o)
                continue

            if now - o.ts < self.chase.wait_sec:
                keep.append(o)
                continue

            # 超时：撤掉，下一轮确认死了再重报
            if self.chase.retries <= 0:
                continue                 # 不追单，挂着就好
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
        但仓位还留着」，或者是用户手工开的仓。这两种都不该被一键对齐扫掉 ——
        对齐只补托管合约，剩下的列出来让人自己处理。
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

    def align(self) -> int:
        """把账户仓位补到策略理论持仓。返回下发的委托数。

        只处理 `managed` 的合约：没有策略在跑的合约理论持仓恒为 0，把它们一起
        「对齐」就是把账户清空，那不是这个按钮的意思。
        """
        if self.api is None:
            return 0
        n = 0
        for row in self.mismatch():
            if not row["managed"]:
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
