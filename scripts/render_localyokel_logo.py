"""Bake the exact word LocalYokel into the generated marks (no model text)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(r"J:\LLM\workspaces\FreeToken-vcruz")
ASSETS = ROOT / "assets"
SESSION = Path(
    r"C:\Users\owenm\.grok\sessions\J%3A%5CLLM\01a03e55-5786-7eb1-9191-e64bab9a863b\images"
)
FONT = Path(r"C:\Windows\Fonts\segoeuib.ttf")
AMBER = (255, 176, 55, 255)
TEAL = (64, 224, 208, 255)
WHITE = (236, 242, 248, 255)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size)


def _glow_text(
    base: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    glow: tuple[int, int, int, int],
    radius: int = 18,
) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.text(xy, text, font=font, fill=glow)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=radius))
    base.alpha_composite(overlay)
    ImageDraw.Draw(base).text(xy, text, font=font, fill=fill)


def _draw_word(base: Image.Image, x: int, y: int, size: int, tracking: int = 2) -> None:
    font = _font(size)
    local, yokel = "Local", "Yokel"
    _glow_text(base, (x, y), local, font, AMBER, (255, 150, 30, 160), radius=max(12, size // 18))
    lw = int(font.getlength(local)) + tracking
    _glow_text(
        base,
        (x + lw, y),
        yokel,
        font,
        TEAL,
        (40, 180, 170, 150),
        radius=max(12, size // 18),
    )


def lockup() -> Path:
    src = Image.open(SESSION / "3.jpg").convert("RGBA")
    w, h = src.size
    # scale to a github-friendly banner
    target_w = 1920
    scale = target_w / w
    src = src.resize((target_w, int(h * scale)), Image.Resampling.LANCZOS)
    canvas = src
    # word sits in the open right field
    size = 118
    font = _font(size)
    word_w = int(font.getlength("LocalYokel"))
    x = int(canvas.width * 0.46)
    if x + word_w + 60 > canvas.width:
        x = canvas.width - word_w - 80
    y = int(canvas.height * 0.42)
    _draw_word(canvas, x, y, size, tracking=4)
    out = ASSETS / "localyokel-lockup.png"
    canvas.convert("RGB").save(out, "PNG", optimize=True)
    return out


def square() -> Path:
    src = Image.open(SESSION / "4.jpg").convert("RGBA")
    w, h = src.size
    # the rounded app tile sits in the middle of the studio frame — inset crop
    inset = int(min(w, h) * 0.14)
    tile = src.crop((inset, inset, w - inset, h - inset)).resize((1024, 1024), Image.Resampling.LANCZOS)
    # fade the bottom of the tile so the word sits on the lantern, not a caption strip
    fade = Image.new("L", (1024, 1024), 0)
    fd = ImageDraw.Draw(fade)
    for i, a in enumerate(range(0, 200, 2)):
        y0 = 720 + i
        fd.rectangle((0, y0, 1024, 1024), fill=min(210, a))
    veil = Image.new("RGBA", (1024, 1024), (8, 10, 14, 255))
    veil.putalpha(fade)
    tile.alpha_composite(veil)
    font_size = 88
    font = _font(font_size)
    tw = int(font.getlength("LocalYokel"))
    x = (1024 - tw) // 2
    y = 1024 - 168
    _draw_word(tile, x, y, font_size, tracking=2)
    out = ASSETS / "localyokel-icon.png"
    tile.convert("RGB").save(out, "PNG", optimize=True)
    return out


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    a = lockup()
    b = square()
    print("wrote", a, a.stat().st_size)
    print("wrote", b, b.stat().st_size)
    print("WORD LocalYokel")


if __name__ == "__main__":
    main()
