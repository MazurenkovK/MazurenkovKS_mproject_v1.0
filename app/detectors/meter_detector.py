# app/detectors/meter_detector.py
import cv2
from ultralytics import YOLO

class MeterDetector:
    """
    Отвечает ТОЛЬКО за:
    - поиск манометра на исходном изображении
    - обрезку ROI
    """

    def __init__(self, scale_factor=1.1):
        self.scale_factor = scale_factor
        self.model = YOLO("/Users/konstantinmazurenkov/pressure_base/project_base/yolov8n.pt")

    def _expand_bbox(self, x1, y1, x2, y2, w, h):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        bw = (x2 - x1) * self.scale_factor
        bh = (y2 - y1) * self.scale_factor

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(w, int(cx + bw / 2))
        y2 = min(h, int(cy + bh / 2))

        return x1, y1, x2, y2

    def detect_and_crop(self, image):
        """
        Возвращает:
        - cropped_img (BGR)
        - center_of_meter (x, y) в координатах cropped_img
        """
        h, w = image.shape[:2]

        results = self.model(image, conf=0.4, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None, None

        # берем самый уверенный bbox
        box = max(results.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        x1, y1, x2, y2 = self._expand_bbox(x1, y1, x2, y2, w, h)

        cropped = image[y1:y2, x1:x2]

        # центр манометра в cropped
        cx = (x2 - x1) // 2
        cy = (y2 - y1) // 2

        return cropped, (cx, cy)
