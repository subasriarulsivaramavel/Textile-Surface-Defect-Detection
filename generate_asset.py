import cv2, numpy as np, random, os

# Use a real woven fabric sample as the base texture.
base_img = cv2.imread("samples/sample_2.png")
if base_img is None:
    base_img = cv2.imread("samples/clean.png")

h0, w0 = base_img.shape[:2]
tile_h, tile_w = 360, 1200
# Build a seamless-ish fabric canvas by tiling + slight offsets.
canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
for y in range(0, tile_h, h0):
    for x in range(0, tile_w, w0):
        yy = slice(y, min(y + h0, tile_h))
        xx = slice(x, min(x + w0, tile_w))
        patch = base_img[:yy.stop - y, :xx.stop - x]
        canvas[yy, xx] = patch

# Enhance color/saturation for an attractive look.
hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
canvas = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# Add a soft colorful light sweep (rose -> gold).
overlay = canvas.copy()
for i in range(tile_w):
    t = i / tile_w
    color = (int(255 * (1 - t) * 0.9 + 255 * t * 0.1),
             int(255 * 0.3 + 120 * (1 - t)),
             int(80 + 180 * t))
    cv2.line(overlay, (i, 0), (i, tile_h), color, 1)
canvas = cv2.addWeighted(canvas, 0.78, overlay, 0.22, 0)

# Subtle vignette.
mask = np.zeros((tile_h, tile_w), dtype=np.float32)
cv2.ellipse(mask, (tile_w // 2, tile_h // 2), (tile_w // 2, tile_h // 2), 0, 0, 360, 1.0, -1)
mask = cv2.GaussianBlur(mask, (0, 0), 60)
canvas = (canvas.astype(np.float32) * mask[:, :, None]).astype(np.uint8)

# Add a few attractive defects.
random.seed(7)
for _ in range(12):
    cx = random.randint(60, tile_w - 60)
    cy = random.randint(40, tile_h - 40)
    r = random.randint(8, 34)
    col = random.choice([(40, 40, 40), (120, 60, 90), (230, 230, 230)])
    cv2.circle(canvas, (cx, cy), r, col, -1)

os.makedirs("assets", exist_ok=True)
cv2.imwrite("assets/hero_fabric.png", canvas)
print("saved assets/hero_fabric.png")