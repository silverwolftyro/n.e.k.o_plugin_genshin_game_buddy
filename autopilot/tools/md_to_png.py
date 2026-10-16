"""把一份「一页纸」Markdown 渲染成可直发的 PNG。

    python -m autopilot.tools.md_to_png <in.md> <out.png> [--image shot.png]

为什么需要它
----------
微信/飞书里发 .md 没人看，发本地图片路径又渲染不出来 —— 必须产出**真的 PNG**。

三个必须处理的坑（都实际踩过）
------------------------------
1. **中文**：Pillow 默认字体不含 CJK，必须显式加载 ``msyh.ttc``（微软雅黑）。
2. **emoji**：✅🌟❌ 这些**不在雅黑里**。用雅黑画会变成豆腐块 □□□。
   必须单独用 ``seguiemj.ttf``（Segoe UI Emoji，彩色字体）画，
   且要 ``embedded_color=True``，否则只画出黑白的空心轮廓。
   所以渲染时按字符把一行拆成若干段，中文段走雅黑、emoji 段走彩色字体。
3. **纵��只有一个真值**：所有绘制都基于 ``self.y``。一旦某个函数自己再
   推进一次 ``y``（或用局部 y 画字），相邻元素就会**叠在一起** ——
   第一版就是这样，表格行和标题全糊了。

排版策略
--------
* 深色底：聊天窗口里更跳，长时间看不刺眼；彩色 emoji 在深底上对比也更好。
* 表格用细线分行，**不用** markdown 管道符。
* 列宽按内容分配，**最后统一压缩**到内容宽度 —— 宁可缩短单元格，
  也不让列溢出（溢出的文字会盖到下一列上，肉眼根本读不出来）。
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---- 配色（深色底）------------------------------------------------------
BG = (17, 21, 28)
LINE = (48, 58, 72)
FG = (233, 238, 245)
DIM = (146, 158, 175)
ACCENT = (86, 204, 242)
GOOD = (92, 214, 140)
STAR = (250, 200, 80)
BAD = (245, 110, 110)
QUOTE = (205, 216, 230)

FONT_CJK = r"C:\Windows\Fonts\msyh.ttc"
FONT_CJK_BD = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_EMOJI = r"C:\Windows\Fonts\seguiemj.ttf"

#: 语义 emoji -> 着色（用作前置色块）
SEM = {"✅": GOOD, "🌟": STAR, "❌": BAD, "⚠": BAD, "🎮": ACCENT}

_TAG_RE = re.compile(r"<br\s*/?>|</?sub>|</?b>")


def clean(s: str) -> str:
    """去掉 HTML 标签（``<br>`` / ``<sub>``）—— 渲染器不处理它们。"""
    return _TAG_RE.sub("", s)


def is_emoji(ch: str) -> bool:
    """单个字符是不是 emoji。

    ⚠️ 用 ``unicodedata.category`` —— **没有** ``everything()`` 这个函数，
    第一版写成 ``unicodedata.everything(ch)`` 会抛 ``AttributeError``，
    被 except 吞掉后永远返回 False，于是所有 emoji 都掉进雅黑变豆腐块。
    """
    try:
        return unicodedata.category(ch) == "So"
    except (TypeError, ValueError):
        return False


def split_runs(text: str):
    """把一行拆成 ``[(片段, 是否emoji), ...]``。

    变体选择符 U+FE0F / ZWJ U+200D / 零宽空格跟着前面的 emoji 走，
    否则它们会被当成中文画出一串「不可见字符」把排版撑歪。
    """
    FOLLOW = ("️", "‍", "⁠", " ")
    runs, buf, cur = [], "", False
    for ch in text:
        if ch in FOLLOW and buf:
            buf += ch
            continue
        e = is_emoji(ch)
        if buf and e != cur:
            runs.append((buf, cur))
            buf = ""
        cur = e
        buf += ch
    if buf:
        runs.append((buf, cur))
    return runs


def semantic_of(ch: str):
    return SEM.get(ch)


class R:
    """渲染器。**``self.y`` 是唯一的纵向真值**，任何元素都不得自己再推。"""

    def __init__(self, width=1080, margin=48):
        self.W, self.M = width, margin
        self.CW = width - margin * 2
        self.y = margin
        self.img = Image.new("RGB", (width, 1600), BG)
        self.draw = ImageDraw.Draw(self.img)
        self._fonts: dict = {}

    # -- 字体缓存 ----------------------------------------------------
    def f(self, size, bold=False):
        k = (size, bold)
        if k not in self._fonts:
            self._fonts[k] = ImageFont.truetype(
                FONT_CJK_BD if bold else FONT_CJK, size)
        return self._fonts[k]

    def fe(self, size):
        k = ("e", size)
        if k not in self._fonts:
            self._fonts[k] = ImageFont.truetype(FONT_EMOJI, size)
        return self._fonts[k]

    # -- 画布 --------------------------------------------------------
    def need(self, bottom: int) -> None:
        if bottom <= self.img.height:
            return
        nh = max(self.img.height + 400, bottom + 200)
        new = Image.new("RGB", (self.W, nh), BG)
        new.paste(self.img, (0, 0))
        self.img = new
        self.draw = ImageDraw.Draw(new)

    # -- 文本 --------------------------------------------------------
    def draw_runs(self, x, y, s, font, fill=FG) -> float:
        """在**绝对 y** 处画一行，返回结束 x。emoji 与中文自动分字体。"""
        cur = x
        for frag, is_e in split_runs(s):
            if not frag:
                continue
            if is_e:
                ef = self.fe(int(font.size * 0.95))
                self.draw.text((cur, y), frag, font=ef, embedded_color=True)
                adv = self.draw.textlength(frag, font=ef)
            else:
                self.draw.text((cur, y), frag, font=font, fill=fill)
                adv = self.draw.textlength(frag, font=font)
            cur += float(adv)
        return cur

    def draw_md(self, x, y, s, size, color=FG, bold_all=False) -> float:
        """画一行并处理 ``**粗体**``（去星号、用粗体字渲染）。

        表格单元格也走这里 —— 第一版只在 ``para`` 里处理 ``**``，
        结果表格里满屏星号（``**只用眼睛，不读游戏内存**``）。
        """
        cur = x
        for tok in re.split(r"(\*\*[^*]+\*\*)", s):
            if not tok:
                continue
            bold = bold_all or tok.startswith("**")
            body = tok.strip("*") if tok.startswith("**") else tok
            cur = self.draw_runs(cur, y, body,
                                 self.f(size, bold), color)
        return cur

    def wrap(self, s, font, maxw):
        """按像素宽度折行。中文逐字，西文尽量在空格处断。"""
        out, line = [], ""
        for ch in s:
            if ch == "\n":
                out.append(line)
                line = ""
                continue
            if self.draw.textlength(line + ch, font=font) > maxw and line:
                if " " in line[-6:]:
                    cut = line.rfind(" ")
                    out.append(line[:cut])
                    line = line[cut + 1:] + ch
                else:
                    out.append(line)
                    line = ch
            else:
                line += ch
        if line:
            out.append(line)
        return out or [""]

    # -- 元素 --------------------------------------------------------
    def h1(self, s):
        self.need(self.y + 62)
        self.draw_runs(self.M, self.y, clean(s), self.f(29, True))
        self.y += 62

    def h2(self, s):
        self.need(self.y + 74)
        self.y += 24
        self.draw_runs(self.M, self.y, clean(s), self.f(25, True))
        self.draw.line([(self.M, self.y + 40), (self.M + 50, self.y + 40)],
                       fill=ACCENT, width=3)
        self.y += 52

    def para(self, s, size=20, color=FG, gap=10, indent=0):
        """段落。``**粗体**`` 去星号并用粗体字渲染。

        ⚠️ 折行必须在**已画到的 x 位置**上判断，不能只按 token 各自折 ——
           第一版按 token 折完就直接往当前行末尾画，于是「token 自己够宽」
           的那一段会冲出右边距被裁掉（实测丢了「不是没做完」半句）。
        """
        s = clean(s)
        lh = int(size * 1.62)
        right = self.M + self.CW
        x0 = self.M + indent
        tokens = re.split(r"(\*\*[^*]+\*\*)", s)
        line_y = self.y
        cur = x0
        for tok in tokens:
            if not tok:
                continue
            bold = tok.startswith("**")
            body = tok.strip("*") if bold else tok
            fnt = self.f(size, bold)
            for piece in self.wrap(body, fnt, self.CW - indent):
                w = self.draw.textlength(piece, font=fnt)
                # 放不下就换行 —— 判断的是**累计位置**，不是 token 自身宽度
                if cur > x0 and cur + w > right:
                    line_y += lh
                    cur = x0
                self.need(line_y + lh + 40)
                cur = self.draw_runs(cur, line_y, piece, fnt, color)
        self.y = line_y + lh + gap

    def note(self, s, size=16):
        s = clean(s)
        lh = int(size * 1.6)
        self.need(self.y + lh + 8)
        self.draw_runs(self.M, self.y, s, self.f(size), DIM)
        self.y += lh + 8

    def quote(self, lines, size=19):
        lh = int(size * 1.6)
        self.need(self.y + lh * len(lines) + 34)
        top = self.y
        for ln in lines:
            # 引用块里也可能有 **（文案草稿就是加粗的），必须同样处理
            plain = re.sub(r"\*\*", "", clean(ln))
            for piece in self.wrap(plain, self.f(size), self.CW - 34):
                self.need(top + lh * 2 + 34)
                self.draw_md(self.M + 30, self.y, piece, size, QUOTE)
                self.y += lh
        self.draw.line([(self.M + 10, top), (self.M + 10, self.y - 8)],
                       fill=ACCENT, width=3)
        self.y += 26

    def rule(self):
        self.need(self.y + 20)
        self.draw.line([(self.M, self.y), (self.W - self.M, self.y)],
                       fill=LINE, width=1)
        self.y += 22

    def image(self, path: Path, max_h=560):
        im = Image.open(path).convert("RGB")
        tw = self.CW
        th = int(im.height * tw / im.width)
        if th > max_h:
            th, tw = max_h, int(im.width * max_h / im.height)
        im = im.resize((tw, th), Image.LANCZOS)
        self.need(self.y + th + 30)
        self.img.paste(im, (self.M + (self.CW - tw) // 2, self.y))
        # 一圈细边，让截图和文字区有分隔
        self.draw.rectangle(
            [self.M + (self.CW - tw) // 2 - 1, self.y - 1,
             self.M + (self.CW + tw) // 2, self.y + th],
            outline=LINE, width=1)
        self.y += th + 28

    # -- 表格 --------------------------------------------------------
    def table(self, rows):
        if not rows:
            return
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        # 量宽度前先去掉 ** —— 星号也是字符，不去掉的话列宽会被高估，
        # 于是单元格被无谓地截短（第一版就出现了大片「…」）。
        rows = [[re.sub(r"\*\*", "", c) for c in r] for r in rows]
        fs = 19
        fh = self.f(fs, True)
        fr = self.f(fs)

        # 列宽：按各列最长内容分配，最后统一压到内容宽
        want = []
        for ci in range(ncol):
            w = 56
            for ri, row in enumerate(rows):
                fnt = fh if ri == 0 else fr
                w = max(w, self.draw.textlength(row[ci], font=fnt) + 26)
            want.append(w)
        total = sum(want)
        if total > self.CW:
            k = self.CW / total
            widths = [max(62, int(w * k)) for w in want]
            # 压缩后可能被截断，逐列再收一次，保证不溢出
            for _ in range(6):
                over = False
                for ci in range(ncol):
                    for ri, row in enumerate(rows):
                        fnt = fh if ri == 0 else fr
                        t = row[ci]
                        while t and self.draw.textlength(
                                t, font=fnt) > widths[ci] - 20:
                            t = t[:-1]
                        rows[ri][ci] = t + ("…" if t != row[ci] else "")
                        if self.draw.textlength(
                                rows[ri][ci], font=fnt) > widths[ci]:
                            over = True
                if not over:
                    break
        else:
            widths = [int(w) for w in want]

        lh, pad = 32, 11
        for ri, row in enumerate(rows):
            head = ri == 0
            self.need(self.y + lh + pad * 2)
            ty = self.y + pad
            x = self.M
            for ci in range(ncol):
                cell = row[ci]
                col = DIM if head else FG
                # ⚠️ 必须走 draw_md —— 单元格里的 **也要处理，
                #    否则会满屏星号（第一版就在这翻的车）。
                self.draw_md(x + 12, ty, cell, fs, col, bold_all=head)
                x += widths[ci]
            self.y += lh + pad
            self.draw.line([(self.M, self.y - pad // 2),
                            (self.W - self.M, self.y - pad // 2)],
                           fill=LINE, width=1)
        self.y += 8

    def save(self, path):
        self.need(self.y + self.M)
        self.img = self.img.crop((0, 0, self.W, self.y + self.M))
        self.img.save(path, "PNG", optimize=True)
        return self.img.width, self.img.height


# ==================================================================
# Markdown 解析（只认这份一页纸用到的语法）
# ==================================================================

def parse(md: str):
    lines = md.splitlines()
    blocks, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and \
                re.fullmatch(r"\|[\s:|-]+\|", lines[i + 1].strip()):
            rows = [[c.strip() for c in s.strip("|").split("|")]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in
                             lines[i].strip().strip("|").split("|")])
                i += 1
            blocks.append(("table", rows))
            continue
        if s.startswith("|"):
            i += 1
            continue
        if s.startswith("# "):
            blocks.append(("h1", s[2:].strip()))
        elif s.startswith("## "):
            blocks.append(("h2", s[3:].strip()))
        elif s.startswith("### "):
            blocks.append(("h2", s[4:].strip()))
        elif s.startswith("> "):
            q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(("quote", q))
            continue
        elif re.fullmatch(r"-{3,}|\*{3,}", s):
            blocks.append(("rule", None))
        elif s.startswith("- "):
            blocks.append(("note", "• " + s[2:].strip()))
        else:
            item = []
            while i < len(lines) and lines[i].strip() and \
                    not lines[i].strip().startswith(("#", "|", ">", "-")) and \
                    not re.fullmatch(r"-{3,}|\*{3,}", lines[i].strip()):
                item.append(lines[i].strip())
                i += 1
            blocks.append(("para", " ".join(item)))
            continue
        i += 1
    return blocks


def render(md_path: Path, png_path: Path, width: int,
           img_path: Path | None) -> tuple[int, int]:
    blocks = parse(md_path.read_text(encoding="utf-8"))
    r = R(width=width)
    hero = False
    footer_seen = False

    for kind, payload in blocks:
        if kind == "h1":
            r.h1(payload)
            if img_path and img_path.exists() and not hero:
                r.image(img_path)
                hero = True
        elif kind == "h2":
            r.h2(payload)
        elif kind == "para":
            # 「配图需手动贴」这类说明已经在图下方了，不再重复
            if payload.startswith("📎"):
                footer_seen = True
                continue
            r.para(payload)
        elif kind == "quote":
            r.quote(payload)
        elif kind == "table":
            r.table(payload)
        elif kind == "note":
            r.note(payload)
        elif kind == "rule":
            r.rule()

    if not hero and img_path and img_path.exists():
        r.image(img_path)
    if not footer_seen:
        r.rule()
        r.note("数据来自 48 次真实运行日志 ｜ 2026-10-12")
    return r.save(png_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown 一页纸 -> PNG")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--image", default="", help="插在大标题下方的图")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    img = Path(args.image) if args.image else None
    w, h = render(src, dst, args.width, img)
    kb = dst.stat().st_size / 1024
    print(f"OK  {dst}")
    print(f"    {w} x {h}   {kb:.0f} KB")
    if h > 3000:
        print(f"⚠ 高度 {h}px > 3000 —— 微信压缩后看不清，建议拆成 2 张")
    if kb > 500:
        print(f"⚠ {kb:.0f} KB 偏大 —— 微信会自动压缩，可先用 --width 900 重出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
