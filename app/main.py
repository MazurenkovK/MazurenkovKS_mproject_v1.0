# app/main.py
import os
import cv2
from  core.pipeline import PressurePipeline

IMG_DIR = "/Users/konstantinmazurenkov/MazurenkovKS_fcw/MazurenkovKS_fcw/app/images_test"

pipeline = PressurePipeline(
    gauge_model_path="/Users/konstantinmazurenkov/MazurenkovKS_fcw/MazurenkovKS_fcw/app/models/best_gauge.pt ",
    seg_model_path="/Users/konstantinmazurenkov/MazurenkovKS_fcw/MazurenkovKS_fcw/app/models/best_seg.pt",
    pressure_max=6
)

# Тестовая обработка 1 изображения
# img = cv2.imread("/Users/konstantinmazurenkov/pressure_base/project_base/test3.jpg")
# pressure = pipeline.process(img)
# print("PRESSURE:", pressure)

# допустимые расширения
EXTS = (".jpg", ".jpeg", ".png")

files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith(EXTS)
])

print(f"Found {len(files)} images\n")

for fname in files:
    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"[SKIP] {fname} — cannot read image")
        continue

    try:
        pressure = pipeline.process(img, visualize=True)
        print(f"[OK] {fname} → PRESSURE = {pressure:.2f}")
    except Exception as e:
        print(f"[ERR] {fname} → {e}")

