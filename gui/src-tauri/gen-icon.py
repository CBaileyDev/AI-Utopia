"""Generate a 1024x1024 source PNG for `tauri icon`.

Draws the AI Utopia teal "diamond" logo (rotated rounded square with "Ai"
wordmark) on the app's void-dark backdrop. One-shot helper: the repo has no
source logo PNG (the in-app diamond is pure CSS), and `tauri icon` requires a
1024x1024 PNG. Run once, then `npx tauri icon icons/source.png`.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 1024
BG = (2, 2, 4, 255)  # --void-ish #020204
TEAL = (0, 229, 204, 255)  # --accent-cyan #00E5CC
TEAL_DARK = (0, 184, 164, 255)
INK = (3, 17, 15, 255)  # diamond glyph ink

out_dir = Path(__file__).resolve().parent / "icons"
out_dir.mkdir(parents=True, exist_ok=True)

img = Image.new("RGBA", (SIZE, SIZE), BG)

# Subtle radial-ish glow behind the diamond (cheap manual vignette).
glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
cx = cy = SIZE // 2
for r in range(420, 0, -8):
    a = int(70 * (r / 420.0) * 0.6)
    gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 229, 204, max(0, 60 - a)))
img = Image.alpha_composite(img, glow)

# Rounded-square "diamond": draw upright as a rounded rect, then rotate 45deg.
card = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
cd = ImageDraw.Draw(card)
half = 300
box = [cx - half, cy - half, cx + half, cy + half]

# vertical gradient fill teal -> teal_dark, masked by a rounded rect
grad = Image.new("RGBA", (2 * half, 2 * half), (0, 0, 0, 0))
for y in range(2 * half):
    t = y / (2 * half)
    col = (
        int(TEAL[0] * (1 - t) + TEAL_DARK[0] * t),
        int(TEAL[1] * (1 - t) + TEAL_DARK[1] * t),
        int(TEAL[2] * (1 - t) + TEAL_DARK[2] * t),
        255,
    )
    for x in range(2 * half):
        grad.putpixel((x, y), col)

mask = Image.new("L", (2 * half, 2 * half), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, 2 * half - 1, 2 * half - 1], radius=120, fill=255)
card.paste(grad, (box[0], box[1]), mask)

# Wordmark "Ai" centered on the card (drawn upright, rotated with the card).
try:
    font = ImageFont.truetype("arialbd.ttf", 300)
except OSError:
    font = ImageFont.load_default()
glyph = Image.new("RGBA", (2 * half, 2 * half), (0, 0, 0, 0))
gdraw = ImageDraw.Draw(glyph)
text = "Ai"
tb = gdraw.textbbox((0, 0), text, font=font)
tw, th = tb[2] - tb[0], tb[3] - tb[1]
gdraw.text(((2 * half - tw) / 2 - tb[0], (2 * half - th) / 2 - tb[1]), text, font=font, fill=INK)
card.paste(glyph, (box[0], box[1]), glyph)

card = card.rotate(45, resample=Image.BICUBIC, expand=False)
img = Image.alpha_composite(img, card)

src = out_dir / "source.png"
img.save(src)
print(f"wrote {src} ({src.stat().st_size} bytes, {SIZE}x{SIZE})")
