"""CTP 实盘联调：拿真账号把整条链路走一遍。

自检（`launcher.py --selftest`）验的是纯函数，**连不连得上柜台不影响它变绿**。
所以这里单独放一个真连的脚本，凭据只从环境变量读，仓库里不留任何账号。

用法：

    set CTP_USER=091435
    set CTP_PASSWORD=...
    set CTP_BROKER=9999
    set CTP_TD=tcp://182.254.243.31:40001
    set CTP_MD=tcp://182.254.243.31:40011
    python smoke_test_ctp.py [合约]

`CTP_TD` / `CTP_MD` 不填就用 SimNow 7x24 那组。合约默认 `rb2611`，要换就传参数。

**服务时段**：SimNow 第一套（30001/30011）跟实盘交易时段一致，收盘和周末 TCP
直接 RST；7x24 环境（40001/40011）是 16:00 ~ 次日 09:00。连不上先看时间。

这个脚本会**真的下单**（模拟盘）。买入 1 手立刻平掉，净持仓不变，只留一点点
手续费和价差损失。
"""

from __future__ import annotations

import os
import sys
import time

from ttrade.ctp_client import HAS_CTP
from ttrade.ctp_live import CtpHub
from ttrade.tqlive import AccountSpec

RESULTS: list[tuple[str, bool, str]] = []


def step(name: str, ok: bool, note: str = "") -> None:
    RESULTS.append((name, ok, note))
    print(f"[{' OK ' if ok else 'FAIL'}] {name}  {note}", flush=True)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def main() -> int:
    if not HAS_CTP:
        print("没装 openctp-ctp，跑不了（pip install openctp-ctp）")
        return 1
    user, pwd = env("CTP_USER"), env("CTP_PASSWORD")
    if not user or not pwd:
        print("请先设 CTP_USER / CTP_PASSWORD（可选 CTP_BROKER / CTP_TD / CTP_MD）")
        return 2

    spec = AccountSpec(
        name=f"CTP {user}",
        kind="ctp",
        user=user,
        password=pwd,
        broker_id=env("CTP_BROKER", "9999"),
        ctp_preset="simnow_7x24",
        td_front=env("CTP_TD", "tcp://182.254.243.31:40001"),
        md_front=env("CTP_MD", "tcp://182.254.243.31:40011"),
        app_id=env("CTP_APPID", "simnow_client_test"),
        auth_code=env("CTP_AUTHCODE", "0000000000000000"),
    )
    contract = (sys.argv[1] if len(sys.argv) > 1 else "rb2611").strip()

    hub = CtpHub([spec])
    t0 = time.time()
    try:
        hub.start()
    except Exception as exc:  # noqa: BLE001
        step("连接 + 登录 + 结算确认", False, str(exc))
        return 1
    step("连接 + 登录 + 结算确认", True, f"{time.time() - t0:.1f}s")

    # 资金和持仓错开发，等两轮（柜台限制每秒一个查询）
    for _ in range(6):
        hub.wait(1.0)
    st = hub.state()
    acc = st.account or {}
    step("查资金", bool(acc), f"权益 {acc.get('balance', 0):,.0f} / "
                              f"可用 {acc.get('available', 0):,.0f} / "
                              f"占用 {acc.get('margin', 0):,.0f}")
    step("查持仓", True, f"{len(st.positions)} 个合约" +
         ("：" + ", ".join(f"{p['symbol']}({p['pos']:+d})" for p in st.positions[:4])
          if st.positions else "（空仓）"))

    # ---- 行情 ----
    hub.subscribe(contract, "", "1m")
    tick = {}
    for _ in range(15):
        hub.wait(1.0)
        tick = hub.state().quote or {}
        if tick.get("last"):
            break
    step("订阅行情", bool(tick.get("last")),
         f"{contract} 最新 {tick.get('last')} 买一 {tick.get('bid')} 卖一 {tick.get('ask')}"
         if tick.get("last") else "15 秒没收到 tick（不在交易时段？）")

    bars = hub.bars()
    step("tick 聚合 bar", len(bars) > 0, f"{len(bars)} 根 1m")
    if not bars.empty and "volume" in bars:
        vols = bars["volume"].to_numpy()
        # 累计量没差分的话，后面每根都比前面大，是一条单调上升的斜线
        step("成交量已差分", not (len(vols) > 1 and all(
            vols[i] >= vols[i - 1] for i in range(1, len(vols))) and vols[-1] > 1000),
            f"逐根 {[round(float(v), 1) for v in vols[-3:]]}")

    # ---- 报单 → 成交 → 平今 ----
    ask, bid = float(tick.get("ask") or 0), float(tick.get("bid") or 0)
    if not ask:
        print("\n没有盘口，跳过报单验证")
        hub.close()
        return _summary()

    def pos_of() -> int:
        for p in hub.state().positions:
            if p["symbol"] == contract:
                return p["pos"]
        return 0

    before = pos_of()
    try:
        hub.place(contract, "BUY", "OPEN", 1, ask)
        step("报单（买开 @卖一）", True, f"@{ask}")
    except Exception as exc:  # noqa: BLE001
        step("报单（买开 @卖一）", False, str(exc))
        hub.close()
        return _summary()

    filled = False
    for _ in range(8):
        time.sleep(1)
        hub.wait(0.2)
        st = hub.state()
        rows = [o for o in st.orders if o["symbol"] == contract]
        if any(o["status"] == "全部成交" for o in rows):
            filled = True
            break
    step("成交回报", filled,
         next((o["status"] for o in rows), "-") if rows else "没有委托回报")

    # 平今：上期所今天开的仓必须用 CLOSETODAY，用 CLOSE 会被拒
    try:
        hub.place(contract, "SELL", "CLOSETODAY", 1, bid)
        step("平今（卖平 @买一）", True, f"@{bid}")
    except Exception as exc:  # noqa: BLE001
        step("平今（卖平 @买一）", False, str(exc))

    for _ in range(8):
        time.sleep(1)
        hub.wait(0.2)
        if pos_of() == before:
            break
    step("持仓回到开仓前", pos_of() == before, f"{before} → {pos_of()}")

    hub.close()
    return _summary()


def _summary() -> int:
    ok = sum(1 for _, o, _ in RESULTS if o)
    print(f"\n通过 {ok}/{len(RESULTS)} 项")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
