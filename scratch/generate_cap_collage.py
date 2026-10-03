import json
import cv2
import numpy as np
from pathlib import Path

out_dir = Path("results/audit_missing_labels")
with open(out_dir / "audit_missing_labels.json") as f:
    data = json.load(f)

cap_cands = [c for c in data["candidates"] if c["predicted_class"] == "Capacitor"]
cap_cands.sort(key=lambda x: x["confidence"] * (1.0 + x["correlation"]), reverse=True)

top_k = 16
top_cands = cap_cands[:top_k]
cols = 4
rows = (len(top_cands) + cols - 1) // cols
cell_w, cell_h = 320, 260
collage = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 30

for idx, c in enumerate(top_cands):
    r, col = idx // cols, idx % cols
    crop_path = out_dir / c["crop_rel_path"]
    if not crop_path.exists():
        continue
    crop = cv2.imread(str(crop_path))
    if crop is None:
        continue
    max_cw, max_ch = cell_w - 20, cell_h - 60
    scale = min(max_cw / crop.shape[1], max_ch / crop.shape[0])
    nw, nh = max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))
    crop_res = cv2.resize(crop, (nw, nh))

    y_off = r * cell_h + 10 + (max_ch - nh) // 2
    x_off = col * cell_w + 10 + (max_cw - nw) // 2
    collage[y_off:y_off+nh, x_off:x_off+nw] = crop_res

    banner_y = r * cell_h + cell_h - 35
    text1 = f"#{idx+1} Capacitor | Conf: {c['confidence']:.2f}"
    text2 = f"NCC: {c['correlation']:.2f} | {c['stem'][:18]}"
    cv2.putText(collage, text1, (col * cell_w + 12, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(collage, text2, (col * cell_w + 12, banner_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

out_img = Path("/home/edgar/.gemini/antigravity-ide/brain/fa71b4ef-8789-4a6e-9038-7acf3fb2be58/audit_capacitors_grid.png")
cv2.imwrite(str(out_img), collage)
print(f"Saved {out_img} successfully!")
