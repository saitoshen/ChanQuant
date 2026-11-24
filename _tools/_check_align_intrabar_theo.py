# -*- coding: utf-8 -*-
"""独立验算：K 线内那族（`sbbox_ib`）的理论仓能不能被监控器/一键对齐读出来。

纯 Python，不碰柜台 dll，界面开着也能跑。逻辑与
`ttrade/selftest.py::_align_intrabar_theo` 一致，只是为了「不开界面也能先确认」
单独拎出来跑一遍。正式验收入口仍是 `launcher.py --selftest`。

用法：C:/ProgramData/anaconda3/python.exe _tools/_check_align_intrabar_theo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttrade import strategy_store as SStore                    # noqa: E402
from ttrade.strategy_runner import (STRATEGIES, PortfolioRunner,   # noqa: E402
                                    SlotState)

OK, NG = "OK  ", "FAIL"


def _p(lo: int, sh: int):
    """造一个柜台持仓对象（和自检里那个同名桩一致）。"""
    class P:
        def __init__(self):
            self.pos_long, self.pos_short = lo, sh
            for s, v in (("long", lo), ("short", sh)):
                setattr(self, f"pos_{s}_his", v)
                setattr(self, f"pos_{s}_today", 0)
                setattr(self, f"volume_{s}_his", v)
                setattr(self, f"volume_{s}_today", 0)
    return P()


def main() -> int:
    bad = 0

    def chk(name: str, got, want) -> None:
        nonlocal bad
        good = got == want
        bad += 0 if good else 1
        print(f"{OK if good else NG}  {name}: {got!r}"
              + ("" if good else f"（应为 {want!r}）"))

    spec = STRATEGIES.get("sbbox_ib")
    print(f"前提：sbbox_ib intrabar = {getattr(spec, 'intrabar', None)}")
    if not getattr(spec, "intrabar", False):
        print("前提不成立，退出")
        return 2

    r = PortfolioRunner.__new__(PortfolioRunner)
    r._log_fn = None
    r._diag = None
    r._mm_cache, r._mm_ts = [], 0.0
    r._align_ts = 0.0
    r.api = object()
    r._save_book = lambda s: None
    # 加载台账那套（`_fresh_book` / `_adopted`）不在本脚本的验算范围里
    r._adopt_if_needed = lambda s: None
    st = SlotState(cfg=SStore.StrategyConfig(
        sid="t", name="sbbox多品种波段1h 基本面空", period="1h",
        family="rule", rule_key="sbbox_ib", note="short", short_only=True))
    st.running, st.underlying = True, "SHFE.cu2701"
    st.signal, st.theo_lots, st.lots = -1, 8, 8
    st.target = 0                                   # 这一轮从没触发入场
    r._slots = {"t": st}
    r._align_slot = SlotState(cfg=SStore.StrategyConfig(
        sid="__align__", name="一键对齐", period="1d"))
    r._pos = {}

    chk("_ib_on", r._ib_on(st), True)
    chk("_theo_qty", r._theo_qty(st), -8)
    chk("theoretical", r.theoretical().get("SHFE.cu2701"), -8)

    rows = r.mismatch(fresh=True)
    hit = [x for x in rows if x["symbol"] == "SHFE.cu2701"]
    chk("mismatch 行数", len(hit), 1)
    if hit:
        row = hit[0]
        chk("target/actual/delta",
            (row["target"], row["actual"], row["delta"]), (-8, 0, -8))
        chk("theo_only", row["theo_only"], True)

    sent = []
    r._send = lambda s, sym, delta, closeable=None: (
        sent.append((sym, delta)) or s.pending.append("x") or 1)
    n = r.align(close_orphans=True)
    chk("对齐实发", sent, [("SHFE.cu2701", -8)])
    chk("对齐笔数", n, 1)
    chk("账本 held", int(st.held.get("SHFE.cu2701") or 0), -8)

    # ④b 对齐后 `target` 必须跟着摆。只写 held 不写 target ⇒
    #     `_rebalance` 走「目标没变 + 减仓」那条 ⇒ 把刚补开的 8 手买平
    # 先把账户摆成「已成交」（成交回报到了），这样 `_rebalance` 里的
    # 「账户持仓还没跟上」那道闸不会把它挡掉 —— 要验的是它会不会反手平仓
    r._pos = {"SHFE.cu2701": _p(0, 8)}
    chk("对齐后 target", int(st.target or 0), -8)
    r._align_slot.pending.clear()
    sent.clear()
    r._rebalance(st)
    chk("对齐后立刻调仓的实发（必须一笔不发）", sent, [])

    # 反证：把 target 掰回 0（= 只写 held 的旧写法），上面那条必须变红
    keep = (st.target, st.based_target)
    st.target, st.based_target = 0, 0
    sent.clear()
    r._rebalance(st)
    chk("反证·只写 held 时会被自己平掉", sent, [("SHFE.cu2701", 8)])
    st.target, st.based_target = keep

    r._pos = {"SHFE.cu2701": _p(0, 8)}
    r._mm_ts = 0.0
    r._align_slot.pending.clear()
    chk("补上后仍报不一致", [x["symbol"] for x in r.mismatch(fresh=True)], [])

    # 反证一：真触发过（target 有值）不该再标「回测口径」
    st.target = -8
    r._mm_ts = 0.0
    r._pos = {}
    chk("反证·_theo_pending", sorted(r._theo_pending()), [])
    rows2 = r.mismatch(fresh=True)
    chk("反证·theo_only", rows2[0]["theo_only"] if rows2 else None, False)

    # 反证二：普通策略空仓时理论仓必须还是 0（别被这次改动带进来）
    st2 = SlotState(cfg=SStore.StrategyConfig(
        sid="n", name="普通空仓策略", period="1h", family="rule",
        rule_key="ma_cross"))
    st2.running, st2.underlying = True, "DCE.jd2701"
    st2.signal, st2.theo_lots, st2.lots, st2.target = -1, 9, 9, 0
    r._slots = {"n": st2}
    # 注意：`theoretical()` 会给每个在跑的策略建键（未持有时值为 0），
    # 所以这里断言「真值为 0」而不是「键不存在」—— 关键是别变成 -9
    chk("反证·普通策略理论仓", int(r.theoretical().get("DCE.jd2701") or 0), 0)
    r._mm_ts = 0.0
    chk("反证·普通策略不进监控器",
        [x["symbol"] for x in r.mismatch(fresh=True)], [])

    # 反证三：策略自己的委托还在飞时，对齐不许动这个合约（同一笔仓开两遍）
    r._slots = {"t": st}
    st.held.clear()
    st.held["SHFE.cu2701"] = -8
    st.target, st.based, st.based_target = -8, True, -8
    st.pending.append("order-1")
    r._pos = {}
    r._mm_ts = 0.0
    r._align_ts = 0.0
    r._align_slot.pending.clear()
    sent.clear()
    n2 = r.align(close_orphans=True)
    chk("在途单时对齐实发（必须一笔不发）", sent, [])
    chk("在途单时对齐笔数", n2, 0)
    st.pending.clear()

    print()
    print("全部通过" if not bad else f"{bad} 项不通过")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
