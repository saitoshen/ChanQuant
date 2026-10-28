"""交易 Agent「升级权限」下的文件读写。

默认权限下 Agent 只能触发平台定义好的动作（下单、回测、注册因子、注册策略），
碰不到磁盘。打开升级权限后它多一个 `fs` 动作，可以读写 TrendDesk 根目录下的
任意文件 —— 用途是让它自己改代码、调配置、看日志。

四道闸：

1. 路径必须落在根目录内（`..`、绝对路径、盘符、软链接都要挡住）
2. 写之前界面一定弹确认框，把内容摊开给人看
3. 单次读写有大小上限
4. 写是整文件覆盖、原子替换（先写 `.tmp` 再 `os.replace`），不会留半截文件

**挡得住手滑，挡不住蓄意破坏** —— 模型真想搞坏点什么，这层拦不住。所以打开
这个开关的地方必须写清楚风险，而不是指望这里兜底。
"""

from __future__ import annotations

import os
import sys

# 单次读的字符上限。源码文件通常几千字符，20 万足够看一个完整模块；
# 再大就不是「看一眼」而是「搬文件」了，那不该走对话。
MAX_READ = 200_000
MAX_WRITE = 400_000
MAX_LIST = 400

# 列目录时跳过的：这些目录动辄上万个文件，列出来只会把上下文塞满
SKIP_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".idea", ".pytest_cache", ".mypy_cache", "build", "dist",
}


def root() -> str:
    """可写的「TrendDesk 根目录」。

    打包成单文件时 `bundle_root()` 指向临时解包目录（`sys._MEIPASS`），往那里
    写等于写进回收站 —— 所以冻结之后取 exe 所在目录。
    """
    from ttrade.paths import bundle_root, is_frozen

    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return bundle_root()


def _base() -> str:
    """真实路径形式，软链接已解开。所有比较都基于它。"""
    return os.path.realpath(root())


def resolve(rel: str) -> tuple[str, str]:
    """把动作里的相对路径落成绝对路径，并确认它没跑出根目录。

    返回 `(绝对路径, 错误)`。错误非空时绝对路径为空串。
    """
    raw = (rel or "").strip().replace("\\", "/")
    if not raw:
        return "", "没给路径"
    # 绝对路径一律不收：模型没有理由知道用户的盘符，写绝对路径多半是它在
    # 猜。只收相对路径，根目录就永远是那个根目录。
    if raw.startswith("/") or (len(raw) > 1 and raw[1] == ":"):
        return "", f"只能用相对路径（相对 TrendDesk 根目录），不收 {rel}"
    base = _base()
    full = os.path.realpath(os.path.join(base, raw))
    try:
        inside = os.path.commonpath([os.path.normcase(base), os.path.normcase(full)]) == \
            os.path.normcase(base)
    except ValueError:      # 不同盘符，commonpath 会抛
        inside = False
    if not inside:
        return "", f"路径跑出 TrendDesk 目录了：{rel}"
    return full, ""


def rel_of(full: str) -> str:
    """反过来：绝对路径 → 相对根目录的展示用路径。"""
    try:
        return os.path.relpath(full, _base()).replace("\\", "/")
    except ValueError:
        return full


def exists(rel: str) -> bool:
    full, err = resolve(rel)
    return not err and os.path.isfile(full)


def list_dir(rel: str = ".") -> tuple[str, str]:
    """列一层目录。返回 `(文本, 错误)`。"""
    full, err = resolve(rel or ".")
    if err:
        return "", err
    if not os.path.isdir(full):
        return "", f"不是目录：{rel or '.'}"
    rows = []
    try:
        names = sorted(os.listdir(full))
    except OSError as exc:
        return "", f"读不了这个目录：{exc}"
    for name in names:
        if name in SKIP_NAMES:
            continue
        p = os.path.join(full, name)
        if os.path.isdir(p):
            rows.append(f"{name}/")
        else:
            try:
                size = os.path.getsize(p)
            except OSError:
                size = 0
            rows.append(f"{name}  {size} B")
        if len(rows) >= MAX_LIST:
            rows.append(f"……（还有更多，只列了前 {MAX_LIST} 项）")
            break
    if not rows:
        return "", f"目录是空的：{rel or '.'}"
    return "\n".join(rows), ""


def read_text(rel: str) -> tuple[str, str]:
    """读文本文件。二进制直接拒绝 —— 把 parquet 塞进上下文只会烧 token。"""
    full, err = resolve(rel)
    if err:
        return "", err
    if not os.path.isfile(full):
        return "", f"没有这个文件：{rel}"
    size = os.path.getsize(full)
    try:
        with open(full, "rb") as fh:
            head = fh.read(8192)
            if b"\x00" in head:
                return "", f"这是二进制文件（{size} B），不读：{rel}"
            fh.seek(0)
            # 按 UTF-8 最坏情况 4 字节一字，多读一点再截断，省得刚好卡在
            # 多字节字符中间
            data = fh.read(MAX_READ * 4)
    except OSError as exc:
        return "", f"读不出来：{exc}"
    text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_READ:
        return text[:MAX_READ] + f"\n……（已截断，原文 {size} B）", ""
    return text, ""


def _match_eol(old: str, text: str) -> str:
    """按原文件的换行风格改写。

    这个仓库是 CRLF 的。模型给的 content 里只有 `\\n`，直接写回去会让一个
    CRLF 文件变成「一半 CRLF 一半 LF」，git diff 会整片变红，`Edit` 工具也会
    匹配不上。
    """
    body = text.replace("\r\n", "\n")
    if "\r\n" in old:
        return body.replace("\n", "\r\n")
    if "\r" in old and "\n" not in old:      # 老 Mac 风格，理论上遇不到
        return body.replace("\n", "\r")
    return body


def write_text(rel: str, text: str) -> tuple[bool, str]:
    """整文件覆盖。返回 `(成功, 说明)`。"""
    full, err = resolve(rel)
    if err:
        return False, err
    if os.path.isdir(full):
        return False, f"这是个目录，不能写：{rel}"
    text = text if isinstance(text, str) else str(text)
    if len(text) > MAX_WRITE:
        return False, f"内容太长（{len(text)} 字符，上限 {MAX_WRITE}）"

    existed = os.path.isfile(full)
    if existed:
        try:
            with open(full, "rb") as fh:
                old = fh.read().decode("utf-8", errors="replace")
            text = _match_eol(old, text)
        except OSError as exc:
            return False, f"读原文件失败：{exc}"

    parent = os.path.dirname(full)
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as exc:
        return False, f"建不了目录：{exc}"
    tmp = full + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, full)
    except OSError as exc:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False, f"写不进去：{exc}"
    return True, f"已{'覆盖' if existed else '新建'} {rel_of(full)}（{len(text)} 字符）"
