"""生成应用图标（.ico / .icns / .png），打包和窗口都用它。

    python packaging/make_icon.py

## 设计思路：静定有界，以智助量化

图标里只有三样东西，多的一个都不要：

1. **禅圆（円相）** —— 一笔画成、**不闭合**的圆。禅画里的円相讲的是「不完美但
   完整」，那一小段缺口是「还在继续」。它同时是「有界」：量化的第一件事是知道
   边界在哪 —— 仓位有上限、亏损有底线、模型有适用区间。
2. **一根朱砂 K 线** —— 立在圆里。量化这件事本身的符号，也是「以智助量化」的那个
   「量」。只有一根：两根以上在 16px 下就糊成一团。
3. **墨底** —— 深墨渐变（不是纯黑），让纸白的笔画浮起来，像灯下看字。

配色是宣纸 + 墨 + 朱砂：纸白 `#EEE7D6` 的圆、朱砂 `#C63C2C` 的 K 线、墨色底。
朱砂只用在一处 —— 印章的规矩，点睛的地方只该有一个红点。

圆角 22.5%，和系统图标的圆角量级对齐。

## 笔画怎么画

禅圆的笔锋是「起笔重、收笔轻」，所以不是 `arc(width=...)` 能画出来的：沿圆周取
一圈点，按角度把笔画**宽度**从 `W_MAX` 收到 `W_MIN`，内外半径连成一个多边形再
填充。这样在 1024 上是一笔有粗细变化的墨，缩到 16px 也还是一圈有轻重感的圆。

## 只用 Pillow

不依赖外部素材，任何机器上 `python packaging/make_icon.py` 都能重建 —— 图标
丢了、要改配色、要出 1024 的大图，都不用找人要源文件。
"""

from __future__ import annotations

import math
import os
import sys

from PIL import Image, ImageDraw

# 英文版 Windows 控制台默认 cp1252，print 中文会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 墨底：上浅下深，纯黑会显脏
BG_TOP = (28, 33, 41)
BG_BOT = (10, 13, 17)
PAPER = (238, 231, 214)      # 纸白（禅圆）
VERMILION = (198, 60, 44)    # 朱砂（K 线）

# 禅圆：半径与笔锋
RING_R = 0.300               # 圆半径（占边长）
W_MAX = 0.106                # 起笔宽度
W_MIN = 0.042                # 收笔宽度
# 缺口开在右上（1~2 点钟）。禅画的円相就是「从下往上扫、收在右上」，
# 顺带这个开口正好是 ChanQuant 的 C
GAP_START = -24.0            # 缺口起点（角度，0° 在 3 点钟方向，顺时针为负＝偏上）
GAP_SPAN = 48.0              # 缺口跨度


def _gradient(size: int, top, bot) -> Image.Image:
    """竖向渐变。1px 宽画完再拉宽 —— 逐像素画 1024×1024 要几百毫秒。"""
    strip = Image.new("RGB", (1, size))
    px = strip.load()
    for y in range(size):
        t = y / max(1, size - 1)
        px[0, y] = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
    return strip.resize((size, size), Image.BILINEAR)


def _enso(img: Image.Image, s: int) -> None:
    """一笔禅圆：沿圆周按角度收细，内外半径连成多边形再填充。"""
    cx = cy = s / 2.0
    r = s * RING_R
    a0 = GAP_START + GAP_SPAN                 # 起笔
    a1 = GAP_START + 360.0                    # 收笔
    steps = max(180, s // 2)                  # 1024 上 512 段，边缘看不出折线
    outer: list[tuple[float, float]] = []
    inner: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        ang = math.radians(a0 + (a1 - a0) * t)
        # 起笔粗、收笔细；0.75 次幂让粗的一段留得更久，像毛笔压下去再提起来
        w = s * (W_MIN + (W_MAX - W_MIN) * (1.0 - t) ** 0.75) / 2.0
        ca, sa = math.cos(ang), math.sin(ang)
        outer.append((cx + (r + w) * ca, cy + (r + w) * sa))
        inner.append((cx + (r - w) * ca, cy + (r - w) * sa))

    layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(layer).polygon(outer + inner[::-1], fill=PAPER + (255,))
    img.alpha_composite(layer)


def _candle(img: Image.Image, s: int) -> None:
    """一根朱砂 K 线：上影 + 实体 + 下影，整体略偏右上，重心才不呆。"""
    cx = s * 0.503
    body_w = s * 0.082
    body_top, body_bot = s * 0.408, s * 0.618
    wick_w = s * 0.019

    d = ImageDraw.Draw(img)
    # 影线先画，实体后画 —— 实体盖住影线接缝，边界才干净
    d.rounded_rectangle(
        [cx - wick_w / 2, s * 0.330, cx + wick_w / 2, s * 0.692],
        radius=wick_w / 2, fill=VERMILION + (255,))
    d.rounded_rectangle(
        [cx - body_w / 2, body_top, cx + body_w / 2, body_bot],
        radius=body_w * 0.22, fill=VERMILION + (255,))


def draw_icon(size: int = 1024) -> Image.Image:
    """画一张 RGBA 图标。所有尺寸都从比例算，16px 和 1024px 长得一样。"""
    s = size
    img = _gradient(s, BG_TOP, BG_BOT).convert("RGBA")

    # 圆角底。22.5% 是系统图标（macOS / Windows 11）的圆角量级，看起来才像
    # 「装在系统里的应用」而不是一张贴图
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, s - 1, s - 1], radius=int(s * 0.225), fill=255)
    img.putalpha(mask)

    # 左上柔光：墨底不能是死的，得有一点灯下反光
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse(
        [-int(s * 0.28), -int(s * 0.40), int(s * 0.88), int(s * 0.48)],
        fill=(255, 255, 255, 16))
    glow.putalpha(Image.composite(
        glow.getchannel("A"), Image.new("L", (s, s), 0), mask))
    img = Image.alpha_composite(img, glow)

    _enso(img, s)
    _candle(img, s)
    return img


def main() -> int:
    base = draw_icon(1024)

    png_path = os.path.join(OUT_DIR, "icon.png")
    base.save(png_path)

    ico_path = os.path.join(OUT_DIR, "icon.ico")
    base.save(ico_path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                               (128, 128), (256, 256)])

    icns_path = os.path.join(OUT_DIR, "icon.icns")
    try:
        base.save(icns_path)  # Pillow 的 icns 写出器要求 1024 或 512
        made_icns = True
    except Exception as exc:  # pragma: no cover - 仅在某些 Pillow 版本失败
        print(f"  icns 生成失败（不影响 Windows 构建）: {exc}")
        made_icns = False

    print(f"图标已生成：{png_path}")
    print(f"           {ico_path}")
    if made_icns:
        print(f"           {icns_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
