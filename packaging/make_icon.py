"""生成应用图标（.ico / .icns / .png），打包和窗口都用它。

    python packaging/make_icon.py

## 设计思路：QuantBuddy

这个程序的用法是「用一句人话让它去查数据、跑回测」，所以图标不能做成 K 线图
—— 那是行情软件的脸，用户看见只会觉得「又一个看盘工具」。要传达的是**有个
搭子在帮你干量化的活**，于是主体是一个对话气泡（有人在那儿回你话），气泡里
三根逐级抬高的柱子（量化、在变好），最高那根的顶盖换成琥珀色（在动的那一根）。

配色用靛蓝→紫的竖向渐变：比深蓝年轻、比纯绿（金融软件的俗套）干净，圆角
22.5% 跟系统图标的圆角量级对齐。左上有一层很淡的柔光，避免整块色死板。

## 只用 Pillow

不依赖外部素材，任何机器上 `python packaging/make_icon.py` 都能重建 —— 图标
丢了、要改配色、要出 1024 的大图，都不用找人要源文件。
"""

from __future__ import annotations

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

# 底色渐变：靛蓝 → 紫
BG_TOP = (104, 96, 255)
BG_BOT = (124, 58, 237)
WHITE = (255, 255, 255)
INK = (49, 46, 129)        # 柱子最浅那一根
AMBER = (251, 191, 36)     # 顶盖 / 强调色


def _gradient(size: int, top, bot) -> Image.Image:
    """竖向渐变。1px 宽画完再拉宽 —— 逐像素画 1024×1024 要几百毫秒。"""
    strip = Image.new("RGB", (1, size))
    px = strip.load()
    for y in range(size):
        t = y / max(1, size - 1)
        px[0, y] = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
    return strip.resize((size, size), Image.BILINEAR)


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

    # 左上柔光
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse(
        [-int(s * 0.30), -int(s * 0.42), int(s * 0.90), int(s * 0.50)],
        fill=(255, 255, 255, 30))
    glow.putalpha(Image.composite(
        glow.getchannel("A"), Image.new("L", (s, s), 0), mask))
    img = Image.alpha_composite(img, glow)

    d = ImageDraw.Draw(img)

    # ---- 对话气泡 ----
    x0, y0 = int(s * 0.150), int(s * 0.215)
    x1, y1 = int(s * 0.850), int(s * 0.700)
    r = int(s * 0.165)
    # 尾巴先画、气泡后画：两块白色叠在一起，接缝处不会有半透明的缝
    d.polygon([
        (x0 + int(s * 0.115), y1 - r // 3),
        (x0 + int(s * 0.375), y1 - r // 3),
        (x0 + int(s * 0.085), y1 + int(s * 0.110)),
    ], fill=WHITE + (252,))
    d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=WHITE + (252,))

    # ---- 三根柱子 ----
    # 高度按气泡高度的比例给，这样改气泡大小不用重调柱子
    bw = int(s * 0.092)
    bh = y1 - y0
    base_y = y1 - int(bh * 0.24)
    heights = (0.28, 0.44, 0.60)
    xs = (0.28, 0.50, 0.72)
    for i, (cx, h) in enumerate(zip(xs, heights)):
        bx = x0 + int((x1 - x0) * cx)
        top = base_y - int(bh * h)
        # 从深靛到亮紫：右边那根最新，颜色也最跳
        col = tuple(int(INK[k] + (BG_BOT[k] - INK[k]) * (i / 2.0)) for k in range(3))
        d.rounded_rectangle([bx - bw // 2, top, bx + bw // 2, base_y],
                            radius=bw // 2, fill=col + (255,))

    # ---- 琥珀顶盖 ----
    # 画在最高那根的**圆头范围内**，不是浮在上面 —— 早一版把它放在柱子上方，
    # 结果超出气泡边界，32px 下像一颗掉在外面的糖
    bx = x0 + int((x1 - x0) * xs[2])
    top = base_y - int(bh * heights[2])
    rr = bw // 2
    d.ellipse([bx - rr, top, bx + rr, top + 2 * rr], fill=AMBER + (255,))

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
