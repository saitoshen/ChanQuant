"""matplotlib 的全局样式设置，GUI 与自检共用。

**跨平台字体是打包后最容易翻车的地方。** 源码开发在 Windows 上，
`font.sans-serif` 只写了 `Microsoft YaHei` / `SimHei`，在 macOS 上这两个字体都不存在，
matplotlib 会静默回退到 DejaVu Sans——中文全部渲染成方框，而且**不报错**，
只有 `UserWarning: Glyph xxxxx missing from font(s)` 这种容易被忽略的警告。

所以这里按平台优先级列出候选字体，从 Windows / macOS / Linux 常见中文字体里挑：

- Windows：Microsoft YaHei（微软雅黑）、SimHei（黑体）
- macOS：PingFang SC（苹方，系统默认中文字体）、Hiragino Sans GB、Heiti SC、STHeiti
- Linux：Noto Sans CJK SC、WenQuanYi Micro Hei

只把**实际装了的**字体写进 rcParams，找不到任何中文字体时发出警告而不是静默出方框。
"""

from __future__ import annotations

import warnings

CJK_CANDIDATES = [
    # Windows
    "Microsoft YaHei",
    "SimHei",
    "Microsoft JhengHei",
    # macOS
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "STHeiti",
    "Songti SC",
    # Linux / 通用
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]

_APPLIED = False


def available_cjk_fonts() -> list[str]:
    """返回当前系统实际可用的中文字体名（按上面的优先级排序）。"""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    return [name for name in CJK_CANDIDATES if name in installed]


def setup_matplotlib_fonts(verbose: bool = False) -> list[str]:
    """配置中文字体与负号显示。返回最终生效的字体列表。"""
    global _APPLIED

    import matplotlib.pyplot as plt

    found = available_cjk_fonts()
    chain = found + ["DejaVu Sans"]

    plt.rcParams["font.sans-serif"] = chain
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    if not found:
        warnings.warn(
            "未找到任何中文字体，图表中的中文会显示为方框。"
            f"请安装以下任一字体：{', '.join(CJK_CANDIDATES[:4])}",
            RuntimeWarning,
            stacklevel=2,
        )
    elif verbose:
        print(f"中文字体：{found[0]}（候选 {len(found)} 个）")

    _APPLIED = True
    return chain


def check_cjk_render(fig, path: str) -> None:
    """把 fig 存到 path，并断言没有缺字警告（自检用）。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(path)
    missing = [str(w.message) for w in caught if "missing from font" in str(w.message)]
    if missing:
        raise RuntimeError(
            f"图表中文渲染缺字（{len(missing)} 处），当前字体链不可用：{missing[0][:120]}"
        )
