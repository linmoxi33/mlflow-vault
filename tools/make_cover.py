"""Render the mlflow-vault social preview card (1280x640).

The card has to say one thing: a database-only backup restores every run and zero
artifacts, and nothing warns you. So the right half is a two-row comparison rather
than decoration — the numbers are the ones the drill actually produced.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "cover.png")

# warm editorial palette: espresso brown + amber gold, with a muted red/green pair
BG_TOP = (26, 18, 16)
BG_BOT = (44, 31, 26)
PANEL = (35, 25, 21)
PANEL_EDGE = (68, 50, 42)
AMBER = (232, 163, 61)
AMBER_DIM = (150, 104, 40)
CREAM = (243, 233, 221)
MUTED = (158, 136, 119)
FAINT = (110, 92, 79)
DANGER = (211, 96, 82)
SAFE = (122, 170, 114)

FONT_DIR = "C:/Windows/Fonts"
F_MONO = os.path.join(FONT_DIR, "consola.ttf")
F_MONO_BOLD = os.path.join(FONT_DIR, "consolab.ttf")
F_DISPLAY = os.path.join(FONT_DIR, "msyhbd.ttc")
F_BODY = os.path.join(FONT_DIR, "msyh.ttc")


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def vgradient(size, top, bottom) -> Image.Image:
    img = Image.new("RGB", (1, size[1]))
    d = ImageDraw.Draw(img)
    for y in range(size[1]):
        t = y / max(1, size[1] - 1)
        d.point((0, y), tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    return img.resize(size, Image.Resampling.BILINEAR)


def tracked(d, xy, text, f, fill, spacing=3.0):
    """Draw text with letter-spacing; returns the x cursor after the last glyph."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f) + spacing
    return x


def rounded(d, box, radius, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    img = vgradient((W, H), BG_TOP, BG_BOT).convert("RGBA")

    # Warm glow behind the panel, so the right side reads as lit rather than pasted.
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for r in range(460, 0, -12):
        alpha = int(15 * (1 - r / 460) ** 2)
        gd.ellipse([1010 - r, 250 - r, 1010 + r, 250 + r], fill=(232, 163, 61, alpha))
    img = Image.alpha_composite(img, glow)

    d = ImageDraw.Draw(img, "RGBA")

    # ---------------- left column ----------------
    x0 = 76
    tracked(d, (x0, 60), "MLFLOW  ·  INTEGRITY-FIRST BACKUP", font(F_MONO, 19), AMBER_DIM, 3.2)

    title_f = font(F_MONO_BOLD, 74)
    d.text((x0 - 4, 100), "mlflow-vault", font=title_f, fill=CREAM)

    d.line([x0, 214, x0 + 96, 214], fill=AMBER, width=3)

    hook_f = font(F_DISPLAY, 33)
    d.text((x0, 240), "备份只有 db，产物全丢。", font=hook_f, fill=CREAM)
    d.text((x0, 288), "MLflow 不报错，只返回空列表。", font=hook_f, fill=CREAM)

    sub_f = font(F_BODY, 21)
    d.text((x0, 348), "audit · backup · verify · restore · redact", font=sub_f, fill=MUTED)

    # command block
    cmds = [
        ("mlflow-vault audit  ./mlruns", "measure the risk"),
        ("mlflow-vault backup ./mlruns ./b", "db + artifacts + hashes"),
        ("mlflow-vault verify ./b", "exit 1 if anything is off"),
    ]
    cy = 402
    mono = font(F_MONO, 19)
    note = font(F_BODY, 17)
    for cmd, desc in cmds:
        d.text((x0, cy), "$", font=mono, fill=AMBER)
        d.text((x0 + 22, cy), cmd, font=mono, fill=(216, 203, 190))
        d.text((x0 + 22, cy + 24), desc, font=note, fill=FAINT)
        cy += 62

    d.text(
        (x0, 588),
        "stdlib only  ·  works when MLflow itself will not import",
        font=font(F_BODY, 18),
        fill=FAINT,
    )

    # ---------------- right panel ----------------
    px0, py0, px1, py1 = 704, 92, 1204, 508
    rounded(d, [px0, py0, px1, py1], 18, fill=PANEL, outline=PANEL_EDGE, width=1)

    lab = font(F_MONO, 16)
    tracked(d, (px0 + 28, py0 + 24), "SAME STORE, TWO RESTORES", lab, FAINT, 2.2)

    # --- row 1: naive copy ---
    r1y = py0 + 62
    d.text((px0 + 28, r1y), "$ cp mlflow.db  backup/", font=font(F_MONO, 19), fill=(198, 186, 174))
    d.text((px0 + 28, r1y + 34), "4 runs", font=font(F_DISPLAY, 22), fill=(200, 188, 176))
    d.text((px0 + 118, r1y + 34), "0 artifacts", font=font(F_DISPLAY, 22), fill=DANGER)
    # cross mark
    d.line([px0 + 26, r1y + 82, px0 + 38, r1y + 94], fill=DANGER, width=3)
    d.line([px0 + 38, r1y + 82, px0 + 26, r1y + 94], fill=DANGER, width=3)
    d.text((px0 + 48, r1y + 78), "no error raised", font=font(F_BODY, 17), fill=FAINT)

    d.line([px0 + 28, r1y + 116, px1 - 28, r1y + 116], fill=(52, 39, 33), width=1)

    # --- row 2: vault ---
    r2y = r1y + 138
    d.text((px0 + 28, r2y), "$ mlflow-vault backup", font=font(F_MONO, 19), fill=(198, 186, 174))
    d.text((px0 + 28, r2y + 34), "4 runs", font=font(F_DISPLAY, 22), fill=(200, 188, 176))
    d.text((px0 + 118, r2y + 34), "12 artifacts", font=font(F_DISPLAY, 22), fill=SAFE)
    # check mark
    d.line([px0 + 26, r2y + 88, px0 + 34, r2y + 96], fill=SAFE, width=3)
    d.line([px0 + 34, r2y + 96, px0 + 48, r2y + 80], fill=SAFE, width=3)
    d.text((px0 + 56, r2y + 78), "verify: 85 rows, dangling 0", font=font(F_BODY, 17), fill=FAINT)

    # --- footer strip ---
    fy = py1 - 42
    d.line([px0 + 28, fy - 18, px1 - 28, fy - 18], fill=(52, 39, 33), width=1)
    d.text((px0 + 28, fy), "sha256 per file", font=font(F_MONO, 15), fill=AMBER_DIM)
    d.text((px0 + 186, fy), "·", font=font(F_MONO, 15), fill=FAINT)
    d.text((px0 + 204, fy), "merkle root_hash", font=font(F_MONO, 15), fill=AMBER_DIM)
    d.text((px0 + 372, fy), "·", font=font(F_MONO, 15), fill=FAINT)
    d.text((px0 + 390, fy), "URI rewrite", font=font(F_MONO, 15), fill=AMBER_DIM)

    img.convert("RGB").save(OUT, "PNG", optimize=True)
    print(f"wrote {os.path.abspath(OUT)}  {os.path.getsize(OUT)} bytes")


if __name__ == "__main__":
    main()
