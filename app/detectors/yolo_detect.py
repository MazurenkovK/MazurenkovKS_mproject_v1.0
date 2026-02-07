# app/detectors/yolo_detect.py

from ultralytics import YOLO
import cv2


class GaugeYOLO:
    def __init__(self, model_path, conf=0.3, imgsz=(480, 480)):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names

    def boxes_intersect(self, b1, b2):
        """
        b = (x1, y1, x2, y2)
        """
        return not (
            b1[2] < b2[0] or
            b1[0] > b2[2] or
            b1[3] < b2[1] or
            b1[1] > b2[3]
        )


    def detect_scale(self, image):
        """
        Возвращает:
        minimum (x, y)
        maximum (x, y)
        + img_yolo
        """
        img_yolo = cv2.resize(image, self.imgsz)

        results = self.model(
            img_yolo,
            conf=self.conf,
            imgsz=self.imgsz[0],
            verbose=False
        )[0]

        mins = []
        maxs = []

        if results.boxes is not None:
            for box in results.boxes:
                label = self.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                item = {
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy)
                }

                if label == "minimum":
                    mins.append(item)
                elif label == "maximum":
                    maxs.append(item)

        # сортируем по confidence
        mins.sort(key=lambda x: x["conf"], reverse=True)
        maxs.sort(key=lambda x: x["conf"], reverse=True)

        # перебор пар (обычно 1–2 итерации)
        for m in mins:
            for M in maxs:
                if not self.boxes_intersect(m["bbox"], M["bbox"]):
                    return m["center"], M["center"], img_yolo

        # если все пересекаются — fallback
        if mins and maxs:
            return mins[0]["center"], maxs[0]["center"], img_yolo

        return None, None, img_yolo
