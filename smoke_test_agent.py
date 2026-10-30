"""ChanAgent 的策略名解析：冒烟。

    python smoke_test_agent.py

为什么单独一个脚本
------------------
用户存进策略库的「碳酸锂动量60-5m」是一条**因子策略模板**（整份配置，sid 在
`strategy_lib.json` 里），不是 `STRATEGIES` 里的一条规则策略。Agent 从截图里只
认得中文名，把它写进动作块 `"strategy": "碳酸锂动量60-5m"` 之后，老代码拿
`key in STRATEGIES` 硬比，一律回「平台没有这个策略」—— 用户再怎么让模型重试都
没用，因为模型没有别的信息可用。

这里验三件事：
1. 名字 → sid 的归一化（`· 我的` 尾巴、全角空格、大小写都得认）
2. `STRATEGIES` 的代号与显示名都能解析成代号
3. 提示词里真的注入了当前清单（模型看得见才抄得对）
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ttrade import agent as A  # noqa: E402
from ttrade import strategy_store as SStore  # noqa: E402
from ttrade.strategies import STRATEGIES, resolve_key  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    if not cond:
        FAILS.append(what)


def main() -> int:
    # ---- 1. 归一化：界面上那截「· 我的」不是名字的一部分
    check(SStore.norm_name("碳酸锂动量60-5m") == SStore.norm_name("碳酸锂动量60-5m · 我的"),
          "「· 我的」尾巴没被剥掉")
    check(SStore.norm_name(" Ａg  动量 ") == SStore.norm_name("ag动量"), "空格/全角没归一")

    # ---- 2. 规则策略：代号、显示名、带归属尾巴的显示名
    for k, spec in STRATEGIES.items():
        check(resolve_key(k) == k, f"代号 {k} 解析失败")
        check(resolve_key(spec.label) == k, f"显示名 {spec.label} 解析失败")
        check(resolve_key(spec.label + " · 内置") == k, f"{spec.label} + 内置 解析失败")
    check(resolve_key("根本没有这个策略") == "", "不存在的名字不该解析出东西")
    check(resolve_key("") == "", "空名字不该解析出东西")

    # ---- 3. 策略库：名字、带尾巴的名字、sid 都要能找到同一条
    #     同名多条（「白银动量60-1h」存过三版）按约定取**最后存的**那条，
    #     所以只在名字唯一时断言 sid，重复名只断言「找得到同名的」
    lib = SStore.load_library()
    if not lib:
        print("策略库是空的，跳过模板匹配（不影响结论）")
    titles = [c.title() for c in lib]
    for c in lib:
        for name in (c.sid, c.title(), c.title() + " · 我的"):
            got = SStore.find_library(name)
            if name == c.sid or titles.count(c.title()) == 1:
                check(got is not None and got.sid == c.sid,
                      f"策略库模板「{name}」没找到（期望 {c.sid}）")
            else:
                check(got is not None and got.title() == c.title(),
                      f"策略库模板「{name}」没找到同名条目")
    check(SStore.find_library("根本没有这个模板") is None, "不存在的模板不该找到")

    # ---- 4. 提示词：清单必须真的进了消息，且带上每一条
    cat = A.strategy_catalog()
    for c in lib:
        check(c.sid in cat, f"清单里漏了模板 {c.title()}")
        check(c.title() in cat, f"清单里漏了模板名 {c.title()}")
    for k in STRATEGIES:
        check(f"`{k}`" in cat, f"清单里漏了代号 {k}")
    msgs = A.build_messages([], "你好", memory="", allow_fs=False)
    check(any(m["content"] == cat for m in msgs if m["role"] == "system"),
          "清单没进 messages")
    quiet = A.build_messages([], "你好", memory="", allow_fs=False, catalog=False)
    check(not any("## 当前平台上的策略" in str(m.get("content")) for m in quiet),
          "catalog=False 时不该注入清单")

    if FAILS:
        print(f"===== 失败 {len(FAILS)} 项 =====")
        for f in FAILS:
            print(" -", f)
        return 1
    print(f"===== 全过（策略库 {len(lib)} 条模板 / 规则策略 {len(STRATEGIES)} 条） =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
