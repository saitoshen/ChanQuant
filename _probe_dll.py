# -*- coding: utf-8 -*-
"""探针：两套 CTP 库能不能在同一个进程里共存。分阶段跑，定位崩在哪一步。

阶段 A：只用系统那套（openctp-ctp）—— 基线，证明探针本身没问题
阶段 B：只用融航那套（vendor/rohon_ctp，改名后的 dll）
阶段 C：两套同时（各建 api、各 Init、各起线程）

用法：python _probe_dll.py A|B|C|all
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def say(*a):
    print(*a, flush=True)


def _flow(root: str, tag: str) -> str:
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    return d + os.sep          # CTP 要求目录存在 + 结尾带分隔符


def run(td, md, tag: str, root: str) -> object:
    """建一个 trader + 一个 md，各连 1.5 秒，然后 Release。返回 api 对象。"""
    class TSpi(td.CThostFtdcTraderSpi):
        def OnFrontConnected(self):        # noqa: N802
            say(f"  [{tag}] 交易前置连上")

        def OnFrontDisconnected(self, n):  # noqa: N802
            say(f"  [{tag}] 交易前置断开 code={n}")

    class MSpi(md.CThostFtdcMdSpi):
        def OnFrontConnected(self):        # noqa: N802
            say(f"  [{tag}] 行情前置连上")

        def OnFrontDisconnected(self, n):  # noqa: N802
            say(f"  [{tag}] 行情前置断开 code={n}")

    tspi, mspi = TSpi(), MSpi()      # SPI 必须被变量持有，否则 CTP 侧是悬空指针
    t = td.CThostFtdcTraderApi.CreateFtdcTraderApi(_flow(root, tag + "_td"))
    m = md.CThostFtdcMdApi.CreateFtdcMdApi(_flow(root, tag + "_md"))
    say(f"  [{tag}] 两个对象建好了")
    for api, spi in ((t, tspi), (m, mspi)):
        api.RegisterSpi(spi)
        api.RegisterFront("tcp://127.0.0.1:1")
        try:
            api.SubscribePrivateTopic(2)
            api.SubscribePublicTopic(2)
        except Exception:  # noqa: BLE001 - md 没有这两个
            pass
        api.Init()
    say(f"  [{tag}] Init 完成，跑 1.5 秒")
    time.sleep(1.5)
    for api in (t, m):
        api.Release()
    say(f"  [{tag}] Release 完成")
    return (t, m)


def main() -> None:
    which = (sys.argv[1] if len(sys.argv) > 1 else "all").upper()
    root = tempfile.mkdtemp(prefix="ctpflow_")
    say("flow 根目录:", root)

    from ttrade import ctp_channel as C

    say("---- 载入系统 openctp-ctp")
    from openctp_ctp import thosttraderapi as td_std
    from openctp_ctp import thostmduserapi as md_std
    say("  版本:", td_std.CThostFtdcTraderApi.GetApiVersion())

    say("---- 载入融航 vendored 运行时")
    md_rohon, td_rohon = C._load_vendored()
    say("  版本:", td_rohon.CThostFtdcTraderApi.GetApiVersion())
    say("---- 两套 dll 同时躺在进程里了，还没建 api")

    if which in ("A", "ALL"):
        say("[A] 只用标准那套")
        run(td_std, md_std, "std", root)
    if which in ("B", "ALL"):
        say("[B] 只用融航那套")
        run(td_rohon, md_rohon, "rohon", root)
    if which in ("C", "ALL"):
        say("[C] 两套同时各建一条")
        a = td_std.CThostFtdcTraderApi.CreateFtdcTraderApi(_flow(root, "c1_td"))
        b = td_rohon.CThostFtdcTraderApi.CreateFtdcTraderApi(_flow(root, "c2_td"))
        say("  两个交易 api 都建出来了，是同一个对象吗:", a is b)
        am = md_std.CThostFtdcMdApi.CreateFtdcMdApi(_flow(root, "c1_md"))
        bm = md_rohon.CThostFtdcMdApi.CreateFtdcMdApi(_flow(root, "c2_md"))
        say("  两个行情 api 都建出来了")
        for api in (a, b, am, bm):
            api.Release()
        say("  Release 完成 —— 建对象这一层没问题")

    if which in ("D", "ALL"):
        # 🔴 决定性一步：两条会话**同时 Init**，dll 内部线程同时在跑
        say("[D] 两套同时 Init（线程同时在跑）")

        class TS(td_std.CThostFtdcTraderSpi):
            def OnFrontConnected(self):        # noqa: N802
                say("  [std] 交易前置连上")

            def OnFrontDisconnected(self, n):  # noqa: N802
                say(f"  [std] 交易前置断开 code={n}")

        class TR(td_rohon.CThostFtdcTraderSpi):
            def OnFrontConnected(self):        # noqa: N802
                say("  [rohon] 交易前置连上")

            def OnFrontDisconnected(self, n):  # noqa: N802
                say(f"  [rohon] 交易前置断开 code={n}")

        class MS(md_std.CThostFtdcMdSpi):
            def OnFrontConnected(self):        # noqa: N802
                say("  [std-md] 行情前置连上")

            def OnFrontDisconnected(self, n):  # noqa: N802
                say(f"  [std-md] 行情前置断开 code={n}")

        class MR(md_rohon.CThostFtdcMdSpi):
            def OnFrontConnected(self):        # noqa: N802
                say("  [rohon-md] 行情前置连上")

            def OnFrontDisconnected(self, n):  # noqa: N802
                say(f"  [rohon-md] 行情前置断开 code={n}")

        spis = [TS(), TR(), MS(), MR()]      # 全部被变量持有，防悬空指针
        ta = td_std.CThostFtdcTraderApi.CreateFtdcTraderApi(_flow(root, "d1_td"))
        tb = td_rohon.CThostFtdcTraderApi.CreateFtdcTraderApi(_flow(root, "d2_td"))
        ma = md_std.CThostFtdcMdApi.CreateFtdcMdApi(_flow(root, "d1_md"))
        mb = md_rohon.CThostFtdcMdApi.CreateFtdcMdApi(_flow(root, "d2_md"))
        pairs = [(ta, spis[0]), (tb, spis[1]), (ma, spis[2]), (mb, spis[3])]
        for api, spi in pairs:
            api.RegisterSpi(spi)
            api.RegisterFront("tcp://127.0.0.1:1")
        for api, _spi in pairs:
            api.Init()
        say("  四条都 Init 了，跑 3 秒（两边线程同时在跑）")
        time.sleep(3.0)
        # 再各问一次资金：真正打到 dll 内部链路（连不上会同步回错误，不崩就对）
        for api, _spi in pairs:
            api.Release()
        say("  四条都 Release 完成 —— 全程没有段错误")

    say("PROBE_OK")


if __name__ == "__main__":
    main()
