"""界面侧的转出层。

渲染器本体已经搬到 `ttrade/render.py` —— 导出 PDF 用的是同一个
`QTextDocument`，两处必须共用同一套 HTML，否则「聊天里看到的表」和
「导出的表」会慢慢分家。

这里只做转出，保持 `from gui import chat_render as CR` 的老用法不变。
"""

from __future__ import annotations

from ttrade.render import (  # noqa: F401
    BODY_PT,
    C_BAD,
    C_CODE_BG,
    C_GOOD,
    C_HEAD_BG,
    C_LINE,
    C_MUTED,
    C_NOTE_BG,
    C_TEXT,
    C_WARN_BG,
    MONO,
    inline,
    parse,
    render,
    render_text,
)
from ttrade.blocks import (  # noqa: F401
    block_code,
    block_head,
    block_image,
    block_link,
    block_list,
    block_metrics,
    block_quote,
    block_table,
    block_text,
    block_warning,
)
