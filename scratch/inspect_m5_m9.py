import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

img_path = "data_samples/native_images/ATTIOT_Bottom_jpg.rf.94cd89169043c7506cde7ced6de19680.jpg"
lbl_path = "data_samples/native_labels/ATTIOT_Bottom_jpg.rf.94cd89169043c7506cde7ced6de19680.txt"
img = cv2.imread(img_path)
H, W = img.shape[:2]

gt_caps = []
with open(lbl_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(parts[0])
        if cls_id == 0:
            xc, yc, w, h = map(float, parts[1:5])
            x1, y1 = int((xc - w/2) * W), int((yc - h/2) * H)
            x2, y2 = int((xc + w/2) * W), int((yc + h/2) * H)
            gt_caps.append((x1, y1, x2, y2))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gx1, gy1, gx2, gy2 = gt_caps[0]
proto_h = gray[gy1:gy2, gx1:gx2]

proto_v = None
for x1, y1, x2, y2 in gt_caps:
    if (y2 - y1) > (x2 - x1) * 1.5:
        proto_v = gray[y1:y2, x1:x2]
        break
if proto_v is None:
    proto_v = np.rot90(proto_h)

res_h = cv2.matchTemplate(gray, proto_h, cv2.TM_CCOEFF_NORMED)
res_v = cv2.matchTemplate(gray, proto_v, cv2.TM_CCOEFF_NORMED)

pad_h_y, pad_h_x = proto_h.shape[0] // 2, proto_h.shape[1] // 2
pad_v_y, pad_v_x = proto_v.shape[0] // 2, proto_v.shape[1] // 2

full_h = np.zeros((H, W), dtype=np.float32)
full_v = np.zeros((H, W), dtype=np.float32)
full_h[pad_h_y : pad_h_y + res_h.shape[0], pad_h_x : pad_h_x + res_h.shape[1]] = res_h
full_v[pad_v_y : pad_v_y + res_v.shape[0], pad_v_x : pad_v_x + res_v.shape[1]] = res_v
corr = np.maximum(full_h, full_v)

from scipy.ndimage import maximum_filter
local_max = maximum_filter(corr, size=15) == corr
peaks = []
for y, x in zip(*np.where(local_max & (corr >= 0.68))):
    peaks.append((x, y, corr[y, x]))

unmatched = []
for px, py, sc in peaks:
    found = False
    for x1, y1, x2, y2 in gt_caps:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if np.hypot(px - cx, py - cy) < 22:
            found = True
            break
    if not found:
        unmatched.append((px, py, sc))

# Save crops of M5, M6, M7, M8, M9
fig, axes = plt.subplots(1, len(unmatched[4:]), figsize=(4 * len(unmatched[4:]), 4))
if len(unmatched[4:]) == 1:
    axes = [axes]

for i, (px, py, sc) in enumerate(unmatched[4:]):
    idx = i + 5
    x1, y1 = max(0, px - 35), max(0, py - 35)
    x2, y2 = min(W, px + 35), min(H, py + 35)
    crop = img[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rel_x, rel_y = px - x1, py - y1
    cv2.drawMarker(crop_rgb, (rel_x, rel_y), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
    axes[i].imshow(crop_rgb)
    axes[i].set_title(f"M{idx} (Corr: {sc:.2f})\n({px}, {py})")
    axes[i].axis("off")

plt.tight_layout()
out_path = Path("/home/edgar/.gemini/antigravity-ide/brain/fa71b4ef-8789-4a6e-9038-7acf3fb2be58/missing_labels_m5_m9.png")
plt.savefig(out_path, dpi=150)
plt.close()
print("Saved M5-M9!")
