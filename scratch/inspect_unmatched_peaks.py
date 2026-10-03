import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load image and templates
img_path = "data_samples/native_images/ATTIOT_Bottom_jpg.rf.94cd89169043c7506cde7ced6de19680.jpg"
lbl_path = "data_samples/native_labels/ATTIOT_Bottom_jpg.rf.94cd89169043c7506cde7ced6de19680.txt"
img = cv2.imread(img_path)
H, W = img.shape[:2]

# Load GT capacitors
gt_caps = []
with open(lbl_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(parts[0])
        if cls_id == 0:  # Capacitor
            xc, yc, w, h = map(float, parts[1:5])
            x1, y1 = int((xc - w/2) * W), int((yc - h/2) * H)
            x2, y2 = int((xc + w/2) * W), int((yc + h/2) * H)
            gt_caps.append((x1, y1, x2, y2))

# Load prototype templates
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Using exemplary horizontal and vertical boxes from ATTIOT
gx1, gy1, gx2, gy2 = gt_caps[0]
proto_h = gray[gy1:gy2, gx1:gx2]

# Find a vertical one
proto_v = None
for x1, y1, x2, y2 in gt_caps:
    if (y2 - y1) > (x2 - x1) * 1.5:
        proto_v = gray[y1:y2, x1:x2]
        break
if proto_v is None:
    proto_v = np.rot90(proto_h)

# Match template
res_h = cv2.matchTemplate(gray, proto_h, cv2.TM_CCOEFF_NORMED)
res_v = cv2.matchTemplate(gray, proto_v, cv2.TM_CCOEFF_NORMED)

pad_h_y, pad_h_x = proto_h.shape[0] // 2, proto_h.shape[1] // 2
pad_v_y, pad_v_x = proto_v.shape[0] // 2, proto_v.shape[1] // 2

full_h = np.zeros((H, W), dtype=np.float32)
full_v = np.zeros((H, W), dtype=np.float32)
full_h[pad_h_y : pad_h_y + res_h.shape[0], pad_h_x : pad_h_x + res_h.shape[1]] = res_h
full_v[pad_v_y : pad_v_y + res_v.shape[0], pad_v_x : pad_v_x + res_v.shape[1]] = res_v
corr = np.maximum(full_h, full_v)

# Find peaks above 0.70
from scipy.ndimage import maximum_filter
local_max = maximum_filter(corr, size=15) == corr
peaks = []
for y, x in zip(*np.where(local_max & (corr >= 0.70))):
    peaks.append((x, y, corr[y, x]))

# Separate matched vs unmatched
matched = []
unmatched = []
for px, py, sc in peaks:
    is_gt = False
    for x1, y1, x2, y2 in gt_caps:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if np.hypot(px - cx, py - cy) < 20:
            is_gt = True
            break
    if is_gt:
        matched.append((px, py, sc))
    else:
        unmatched.append((px, py, sc))

print(f"Total peaks >= 0.70: {len(peaks)}")
print(f"Matched to GT: {len(matched)}")
print(f"UNMATCHED (Candidate Missing Labels): {len(unmatched)}")

# Save crops of unmatched peaks
fig, axes = plt.subplots(max(1, len(unmatched)), 1, figsize=(6, 3 * max(1, len(unmatched))))
if len(unmatched) == 1:
    axes = [axes]

for i, (px, py, sc) in enumerate(unmatched):
    crop = img[max(0, py - 30): min(H, py + 30), max(0, px - 30): min(W, px + 30)]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    axes[i].imshow(crop_rgb)
    axes[i].set_title(f"Unmatched Peak #{i+1} at ({px}, {py}), Corr={sc:.3f}")
    axes[i].axis("off")

plt.tight_layout()
out_dir = Path("/home/edgar/.gemini/antigravity-ide/brain/fa71b4ef-8789-4a6e-9038-7acf3fb2be58")
plt.savefig(out_dir / "unmatched_peaks_investigation.png", dpi=150)
plt.close()
print("Saved unmatched peaks visualization!")
