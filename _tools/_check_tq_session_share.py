# -*- coding: utf-8 -*-
"""独立验算：天勤账户在整个系统只有一条会话（交易页 / 组合页共用同一个 LiveHub）。

纯 Python，不登录天勤（用假的 `TqApi` 桩），界面开着也能跑。
正式验收入口仍是 `launcher.py --selftest`。

验证的东西：

1. `hubreg` 认天勤了：同一个账户 `acquire()` 两次拿到**同一个对象**；
2. 键里带账户类型：同一个天勤账号的 `kq`（快期模拟）和 `account`（实盘）
   必须是两条会话（否则实盘会拿到模拟那条连接）；
3. `LiveHub` 自带泵线程：`wait_update()` **只**在 `tq-pump` 这条线程里被调，
   `wait()` 只等它的下一拍（以前是调用方自己推）；
4. `close()` 带引用计数：还有人用就不真关，最后一个才关；
5. 组合页要的那几个 `TqApi` 同名方法（`get_position` / `get_order` /
   `insert_order` / `cancel_order`）都转出去了；
6. `positions_ready()` 用 `Account.balance` 的 nan 区分「还没读到」和「真的是 0」。

用法：C:/ProgramData/anaconda3/python.exe _tools/_check_tq_session_share.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttrade import hubreg                                    # noqa: E402
from ttrade.tqlive import AccountSpec, LiveHub               # noqa: E402

OK, NG = "OK  ", "FAIL"
bad = 0


def chk(name: str, got, want) -> None:
    global bad
    good = got == want
    if not good:
        bad += 1
    print(f"{OK if good else NG}  {name}: {got!r}"
          + ("" if good else f"（应为 {want!r}）"))


class FakePos:
    def __init__(self, sym, long_, short_):
        self.instrument_id = sym
        self.pos_long, self.pos_short = long_, short_
        self.pos_long_today = self.pos_short_today = 0
        self.pos = long_ - short_
        self.open_price_long = self.open_price_short = 0.0
        self.float_profit = self.margin = 0.0


class FakeOrder:
    def __init__(self, oid):
        self.order_id = oid
        self.status = "ALIVE"


class FakeApi:
    """只实现 `LiveHub` 用到的那几个方法，并记下是谁在调。"""

    def __init__(self):
        self.threads = set()
        self.calls = []
        self.updates = 0
        self.orders = []
        self._pos = {"SHFE.cu2701": FakePos("SHFE.cu2701", 0, 8)}

    def wait_update(self, deadline=None):
        self.threads.add(threading.current_thread().name)
        time.sleep(0.01)                 # 模拟一次阻塞
        self.updates += 1
        return True

    def get_position(self, *a, **k):
        self.threads.add(threading.current_thread().name)
        return dict(self._pos)

    def get_order(self, *a, **k):
        self.threads.add(threading.current_thread().name)
        return {o.order_id: o for o in self.orders}

    def insert_order(self, **kw):
        self.threads.add(threading.current_thread().name)
        self.calls.append(("insert_order", dict(kw)))
        o = FakeOrder(f"o{len(self.orders) + 1}")
        self.orders.append(o)
        return o

    def cancel_order(self, order=None, **kw):
        self.threads.add(threading.current_thread().name)
        self.calls.append(("cancel_order", order))
        return True

    def close(self):
        self.calls.append(("close", None))


class FakeAccount:
    """天勤账户对象的桩：它自己带 `get_account()`（`positions_ready` 走这条）。"""

    def __init__(self, balance):
        self.balance = balance

    def get_account(self):
        return self


def main() -> int:
    hubreg.clear()
    spec_a = AccountSpec(name="快期A", kind="kq", user="u1", password="p1")
    spec_b = AccountSpec(name="快期B", kind="kq", user="u2", password="p2")
    spec_real = AccountSpec(name="快期A", kind="account", user="u1", password="p1")

    print("— 1. 注册表：同一个账户只发一条会话 —")
    chk("SHARED 认天勤", "tq" in hubreg.SHARED, True)
    made = []
    h1 = hubreg.acquire("tq", [spec_a], lambda: made.append(1) or LiveHub([spec_a]))
    h2 = hubreg.acquire("tq", [spec_a], lambda: made.append(1) or LiveHub([spec_a]))
    chk("两次 acquire 是同一个对象", h1 is h2, True)
    chk("只构造了一次", len(made), 1)
    chk("refs", hubreg.refs(h1), 2)
    h3 = hubreg.acquire("tq", [spec_b], lambda: LiveHub([spec_b]))
    chk("不同账户是不同会话", h3 is h1, False)

    print("\n— 2. 键里带账户类型（kq 与实盘不能串）—")
    h4 = hubreg.acquire("tq", [spec_real], lambda: LiveHub([spec_real]))
    chk("同名同账号、类型不同 ⇒ 另开一条", h4 is h1, False)
    chk("键里含 kind", hubreg.hub_key("tq", spec_real)[-1], "account")

    print("\n— 3. 泵线程独占 wait_update —")
    hub = h1
    api = FakeApi()
    hub.api = api
    hub._start_pump()
    got = hub.wait(1.0)
    chk("wait() 拿到一拍", got, True)
    time.sleep(0.1)
    chk("调 wait_update 的线程集合", api.threads, {"tq-pump"})
    chk("wait 自己没推（updates 由泵累加）", api.updates > 0, True)

    print("\n— 4. 组合页要的 tqsdk 同名接口 —")
    chk("get_position", sorted(hub.get_position()), ["SHFE.cu2701"])
    chk("get_order 空表", hub.get_order(), {})
    o = hub.insert_order(symbol="SHFE.cu2701", direction="BUY",
                         offset="CLOSE", volume=8, limit_price=70000.0)
    chk("insert_order 返回订单", bool(o and o.order_id), True)
    chk("下单参数原样传下去", api.calls[0][1]["volume"], 8)
    chk("cancel_order", hub.cancel_order(o), True)
    chk("这些调用都不在泵线程里", "tq-pump" not in {
        threading.current_thread().name}, True)

    print("\n— 5. positions_ready：空字典 ≠ 没仓 —")
    chk("balance 是 nan ⇒ 还没读到",
        (lambda h, o: (setattr(h, "_accs", [(spec_a, FakeAccount(float("nan")))]),
                       h.positions_ready())[1])(hub, None), False)
    hub._accs = [(spec_a, FakeAccount(1_000_000.0))]
    chk("balance 有值 ⇒ 读到了", hub.positions_ready(), True)

    print("\n— 7. 组合页的引擎直接吃这条会话（`_pos` = `hub.get_position()`）—")
    from ttrade import strategy_store as SStore
    from ttrade.strategy_runner import PortfolioRunner, SlotState

    hub2 = hubreg.acquire("tq", [spec_b], lambda: LiveHub([spec_b]))
    api2 = FakeApi()
    hub2.api = api2
    hub2._start_pump()
    hub2._accs = [(spec_b, FakeAccount(500_000.0))]

    r = PortfolioRunner.__new__(PortfolioRunner)
    r._log_fn = None
    r._diag = None
    r._mm_cache, r._mm_ts = [], 0.0
    r._align_ts = 0.0
    r.api = hub2
    r._save_book = lambda s: None
    r._adopt_if_needed = lambda s: None
    r._pos = hub2.get_position()          # tqsdk 形状的活引用
    r._orders = hub2.get_order()
    r._slots = {}
    st = SlotState(cfg=SStore.StrategyConfig(sid="t", name="cu", period="1h"))
    st.running, st.underlying, st.target = True, "SHFE.cu2701", -8
    r._slots["t"] = st
    r._align_slot = SlotState(cfg=SStore.StrategyConfig(
        sid="__align__", name="一键对齐", period="1d"))
    chk("_pos 里有这个合约", sorted(r._pos), ["SHFE.cu2701"])
    chk("_side_pair 认得出空 8 手", r._side_pair("SHFE.cu2701"), (0, 8))
    chk("_net_pos", r._net_pos("SHFE.cu2701"), -8)
    chk("positions_ready 走 LiveHub", r._pos_live(), True)
    chk("理论与账户一致 ⇒ 不报差",
        [x["symbol"] for x in r.mismatch(fresh=True)], [])

    print("\n— 8. close() 按引用计数 —")
    chk("refs 现在是 2", hubreg.refs(hub), 2)
    hub.close()                       # 交易页放手
    chk("还有人在用 ⇒ 不关", hub.api is not None, True)
    chk("refs 降到 1", hubreg.refs(hub), 1)
    hub.close()                       # 组合页放手
    chk("最后一个 ⇒ 真关", hub.api is None, True)
    chk("api.close() 调过", ("close", None) in api.calls, True)
    chk("表里已摘掉", hubreg.refs(hub), 0)

    h3.close()
    h4.close()
    hubreg.clear()

    print()
    print("全部通过" if not bad else f"{bad} 项不通过")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
