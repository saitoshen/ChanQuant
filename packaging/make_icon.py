"""生成应用图标（.ico / .icns / .png），打包时用。

    python packaging/make_icon.py

设计：深蓝底 + 三根上涨的 K 线 + 一条均线，和程序的「趋势跟踪」主题对应。
只用 Pillow 绘制，不依赖外部素材，保证在任何机器上都能重建。
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

BG = (15, 23, 42)          # slate-900
UP = (52, 211, 153)        # emerald-400
DOWN = (248, 113, 113)     # red-400
LINE = (147, 197, 253)     # blue-300
GRID = (30, 41, 59)        # slate-800


def draw_icon(size: int = 1024) -> Image.Image:
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # 圆角底
    r = int(s * 0.22)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=r, fill=BG)

    # 背景网格（3 条横线，暗示坐标轴）
    for i in (1, 2, 3):
        y = int(s * (0.32 + i * 0.16))
        d.line([(int(s * 0.14), y), (int(s * 0.86), y)], fill=GRID, width=max(2, s // 200))

    # 三根 K 线：两根阳线一根阴线，整体向上
    bars = [
        # (中心 x 比例, 实体上, 实体下, 影线上, 影线下, 颜色)
        (0.28, 0.62, 0.76, 0.56, 0.82, DOWN),
        (0.50, 0.44, 0.60, 0.38, 0.66, UP),
        (0.72, 0.24, 0.42, 0.18, 0.50, UP),
    ]
    body_w = int(s * 0.105)
    for cx, top, bot, hi, lo, color in bars:
        x = int(s * cx)
        y0, y1 = int(s * top), int(s * bot)
        yh, yl = int(s * hi), int(s * lo)
        wick_w = max(3, s // 130)
        d.line([(x, yh), (x, yl)], fill=color, width=wick_w)
        d.rounded_rectangle(
            [x - body_w // 2, y0, x + body_w // 2, y1],
            radius=max(3, s // 100),
            fill=color,
        )

    # 均线：穿过三根 K 线的斜线
    pts = [(int(s * 0.18), int(s * 0.72)), (int(s * 0.50), int(s * 0.52)),
           (int(s * 0.82), int(s * 0.26))]
    d.line(pts, fill=LINE, width=max(4, s // 90), joint="curve")

    return img


def main() -> int:
    base = draw_icon(1024)

    png_path = os.path.join(OUT_DIR, "icon.png")
    base.save(png_path)

    ico_path = os.path.join(OUT_DIR, "icon.ico")
    base.save(ico_path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64),
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
